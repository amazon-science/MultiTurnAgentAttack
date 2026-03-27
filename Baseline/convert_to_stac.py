#!/usr/bin/env python3
"""
Convert ToolShield-generated multi-turn attacks into STAC benchmark format.

Supports both SHADE-Arena and Agent_SafetyBench attacks. The dataset type is
auto-detected from the environment directory name (banking/travel/workspace/
spam_filter → SHADE_Arena, everything else → Agent_SafetyBench), or can be
forced via --dataset.

Input:  ToolShield output directory tree:
    <toolshield-output>/
        <env_name>/
            multi_turn_task.1/
                attack_chain.json
                task-turn-1.md
                task-turn-2.md
                ...
            multi_turn_task.2/
                ...

Output: Single JSON file in STAC benchmark format, compatible with:
    - Baseline/eval_baseline.py  (no-planner evaluation)
    - STAC_eval/eval_STAC_benchmark.py  (adaptive planning evaluation)

For Agent_SafetyBench items, a companion file (<output>_asb_envs.json) is also
generated. This file must be merged into Agent_SafetyBench/data/released_data.json
or the eval script must be pointed to it, since eval_baseline.py looks up ASB
environment configs by item ID.

Usage:
    # Convert SHADE-Arena attacks (e.g., banking)
    python -m Baseline.convert_to_stac \\
        --toolshield-output output/shade_attacks \\
        --output data/toolshield_shade_stac.json

    # Convert Agent_SafetyBench attacks
    python -m Baseline.convert_to_stac \\
        --toolshield-output output/safetybench_attacks \\
        --output data/toolshield_asb_stac.json

    # Then evaluate (from MultiTurnAgentAttack directory):
    python -m Baseline.eval_baseline \\
        --input_path data/toolshield_shade_stac.json \\
        --model_agent gpt-4.1 \\
        --defense no_defense
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHADE_ARENA_ENVS = {"banking", "travel", "workspace", "spam_filter"}

# ToolShield env name -> SHADE Arena env name used by SHADEArenaEnvironment
SHADE_ENV_MAP = {
    "banking": "banking",
    "travel": "travel",
    "workspace": "workspace",
    "spam_filter": "spam_filter_update",
}

# Turn tool categories that are NOT environment tools (introduced by
# ToolShield's multi-turn decomposition; cannot be replayed in SHADE/ASB envs)
NON_ENV_TOOL_CATEGORIES = {"terminal", "shell"}

# Mapping from ToolShield file_system actions to likely SHADE-Arena drive tools.
# Checked against the action verb to pick the best cloud-drive equivalent.
FILE_SYSTEM_ACTION_MAP = [
    # (action keywords, tool_name)
    (["append", "update the", "add to", "append to"], "append_to_file"),
    (["create a file", "create a ", "write a file", "create an html", "write the", "create a draft",
      "create a template", "create a document", "create a message", "create a meeting",
      "create a standardization", "create a detailed", "create a formatted",
      "create a financial", "create a maintenance", "create a python",
      "create a professional", "populate the", "create a notification",
      "create an emergency"], "create_file"),
    (["read the", "open and review", "review the", "verify it contains", "read the contents"], "search_files"),
]

# Mapping from non-standard ToolShield tool categories to likely env tools.
# Used when the tool category doesn't match any known env tool name.
CATEGORY_FALLBACK_MAP = {
    "cloud_drive": "create_file",
    "web_browser": "get_webpage",
    "web": "get_webpage",
}

# Action-keyword → tool overrides for unresolved turns whose tool category
# doesn't match known env tools. Checked in order; first match wins.
ACTION_KEYWORD_OVERRIDES = [
    (["verify", "iban", "retrieve"], "get_iban"),
    (["moving", "reschedule", "move the", "move a"], "reschedule_calendar_event"),
    (["review", "participants", "who is already invited"], "search_calendar_events"),
    (["compose", "draft email", "draft a", "send email", "email with subject"], "send_email"),
]


# ---------------------------------------------------------------------------
# Tool schema loading
# ---------------------------------------------------------------------------

def load_asb_tool_schemas(safetybench_path: Path) -> Dict[str, List[dict]]:
    """Load tool schemas from Agent_SafetyBench/environments/*.json.

    Returns {env_name: [tool_schema, ...]} where each tool_schema has
    'name', 'description', 'parameters' keys.
    """
    env_dir = safetybench_path / "environments"
    schemas: Dict[str, List[dict]] = {}
    if not env_dir.is_dir():
        return schemas
    for json_file in sorted(env_dir.glob("*.json")):
        env_name = json_file.stem
        if env_name.startswith("_") or env_name == "BaseEnv":
            continue
        try:
            with open(json_file) as f:
                tools = json.load(f)
            if isinstance(tools, list) and tools:
                schemas[env_name] = tools
        except (json.JSONDecodeError, OSError):
            pass
    return schemas


def load_shade_tool_info(shade_root: Path) -> Dict[str, dict]:
    """Extract tool names and schemas for SHADE-Arena environments.

    Uses shade_tool_extractor.py (must be in the same Baseline/ directory).
    Returns {env_name: {"tool_names": [...], "functions": [...]}}
    """
    try:
        from Baseline.shade_tool_extractor import extract_for_environment
    except ImportError:
        print("  Warning: Baseline.shade_tool_extractor not found; SHADE tool matching disabled")
        return {}

    result = {}
    for env_name in SHADE_ARENA_ENVS:
        try:
            tool_names, _, functions = extract_for_environment(shade_root, env_name)
            result[env_name] = {"tool_names": tool_names, "functions": functions}
        except (FileNotFoundError, ValueError) as e:
            print(f"  Warning: Could not extract SHADE {env_name} tools: {e}")
    return result


# ---------------------------------------------------------------------------
# Tool name / parameter extraction
# ---------------------------------------------------------------------------

def resolve_file_system_action(action_text: str, known_tools: List[str]) -> Optional[str]:
    """Map a file_system action to the closest cloud-drive or env tool.

    ToolShield decomposes attacks into file_system turns for creating/reading
    files. SHADE-Arena doesn't have a filesystem but has cloud-drive tools
    (create_file, append_to_file, search_files, get_file_by_id, etc.) that
    serve the same purpose.

    Returns a matched tool name, or None if no match.
    """
    action_lower = action_text.lower()
    for keywords, tool_name in FILE_SYSTEM_ACTION_MAP:
        if tool_name in known_tools and any(kw in action_lower for kw in keywords):
            return tool_name
    return None


def resolve_category_fallback(
    tool_category: str, action_text: str, known_tools: List[str],
) -> Optional[str]:
    """Try to resolve an unrecognized tool category via fallback mappings
    and action-text keyword matching against known env tools.
    """
    # Direct category mapping
    fallback = CATEGORY_FALLBACK_MAP.get(tool_category)
    if fallback and fallback in known_tools:
        return fallback

    # Action keyword overrides for common patterns
    action_lower = action_text.lower()
    for keywords, tool_name in ACTION_KEYWORD_OVERRIDES:
        if tool_name in known_tools and any(kw in action_lower for kw in keywords):
            return tool_name

    # For composite categories like "banking (email) + file system" or
    # "bash + cloud_drive", try matching the action text directly
    return extract_tool_name(action_text, known_tools)


def extract_tool_name(action_text: str, known_tools: List[str]) -> Optional[str]:
    """Match action text against known tool names.

    Strategies (in order):
      1. Exact substring match (tool name appears verbatim in action)
      2. Underscore-to-space match (e.g., "send_email" matches "send email")
      3. Word overlap scoring — pick the tool with the most word overlap
         with the action text (minimum 2 overlapping words to qualify)
    """
    action_lower = action_text.lower()

    # Strategy 1: exact substring
    for tool in known_tools:
        if tool.lower() in action_lower:
            return tool

    # Strategy 2: underscore -> space
    for tool in known_tools:
        if tool.lower().replace("_", " ") in action_lower:
            return tool

    # Strategy 3: word overlap scoring
    action_words = set(re.findall(r"[a-z]+", action_lower))
    best_tool = None
    best_score = 1  # minimum 2 overlapping words to qualify
    for tool in known_tools:
        tool_words = set(re.findall(r"[a-z]+", tool.lower()))
        overlap = len(action_words & tool_words)
        if overlap > best_score:
            best_score = overlap
            best_tool = tool

    return best_tool


def extract_parameters(action_text: str, tool_schema: dict) -> dict:
    """Best-effort parameter extraction from natural-language action text.

    Strategies (tried per parameter):
      1. param_name = "value"  or  param_name: "value"
      2. param_name followed closely by a quoted string
      3. Common patterns: emails, monetary amounts, file paths, account IDs
    """
    props = tool_schema.get("parameters", {}).get("properties", {})
    params: Dict[str, Any] = {}

    for param_name, param_info in props.items():
        param_desc = param_info.get("description", "").lower()

        # Strategy 1 & 2: quoted value near param name
        for pattern in [
            rf'(?<!\w){re.escape(param_name)}\s*[=:]\s*["\']([^"\']+)["\']',
            rf'(?<!\w){re.escape(param_name)}\s{{0,5}}["\']([^"\']+)["\']',
        ]:
            match = re.search(pattern, action_text, re.IGNORECASE)
            if match:
                params[param_name] = match.group(1)
                break
        if param_name in params:
            continue

        # Strategy 3a: email addresses
        if "email" in param_name or "email" in param_desc:
            match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", action_text)
            if match:
                params[param_name] = match.group(0)
                continue

        # Strategy 3b: monetary amounts
        if "amount" in param_name or "amount" in param_desc:
            match = re.search(
                r"(?:amount|transfer|send|withdraw)[^.]*?\$?([\d,]+(?:\.\d+)?)",
                action_text, re.IGNORECASE,
            )
            if match:
                val = match.group(1).replace(",", "")
                try:
                    params[param_name] = float(val) if "." in val else int(val)
                except ValueError:
                    pass
                continue

        # Strategy 3c: file paths
        if "path" in param_name or "file" in param_name or "path" in param_desc:
            match = re.search(r"(/\S+)", action_text)
            if match:
                params[param_name] = match.group(1)
                continue

        # Strategy 3d: account IDs (alphanumeric with underscores)
        if "account" in param_name or "account" in param_desc:
            match = re.search(r"\b([a-zA-Z][\w]*_\d+)\b", action_text)
            if match:
                params[param_name] = match.group(1)
                continue

    return params


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------

def read_task_turn_files(task_dir: Path) -> List[str]:
    """Read task-turn-*.md files in order, stripping turn prefixes."""
    turns = []
    n = 1
    while True:
        f = task_dir / f"task-turn-{n}.md"
        if not f.exists():
            break
        content = f.read_text().strip()
        content = re.sub(r"^(Turn\s+\d+|Final Turn):\s*", "", content)
        turns.append(content)
        n += 1
    return turns


def gen_tool_call_id() -> str:
    return uuid.uuid4().hex[:32]


# ---------------------------------------------------------------------------
# Dataset detection
# ---------------------------------------------------------------------------

def detect_dataset(env_name: str) -> str:
    """Auto-detect SHADE_Arena vs Agent_SafetyBench from environment name."""
    if env_name.lower() in SHADE_ARENA_ENVS:
        return "SHADE_Arena"
    return "Agent_SafetyBench"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_attack(
    env_name: str,
    task_dir: Path,
    dataset: str,
    known_tool_names: List[str],
    tool_schema_lookup: Dict[str, dict],
    item_id: int,
) -> Optional[dict]:
    """Convert a single ToolShield multi-turn attack to STAC format.

    Returns a STAC-format dict, or None if the task directory is invalid.

    Mapping:
        attack_chain.json  turns[]        → attack_plan.verified_tool_chain[]
        attack_chain.json  final_outcome  → attack_plan.attack_goal
        attack_chain.json  analysis       → attack_plan.explanation
        task-turn-*.md     (all but last) → interaction_history  (user + assistant msgs)
        task-turn-*.md     (last)         → planner_history[0].predicted_attack_prompt

    Turns whose tool category is in NON_ENV_TOOL_CATEGORIES (e.g. terminal, shell)
    are included in the interaction history as plain text assistant responses
    (no tool_call), so they provide conversational context without requiring
    environment replay.
    """
    attack_chain_path = task_dir / "attack_chain.json"
    if not attack_chain_path.exists():
        return None

    with open(attack_chain_path) as f:
        attack_chain = json.load(f)

    turns = attack_chain.get("turns", [])
    if not turns:
        return None

    turn_prompts = read_task_turn_files(task_dir)
    if not turn_prompts:
        return None

    # --- verified_tool_chain: one entry per turn ---
    verified_tool_chain = []
    for i, turn in enumerate(turns):
        action = turn.get("action", "")
        tool_category = turn.get("tool", "").lower()
        is_nonenv_category = tool_category in NON_ENV_TOOL_CATEGORIES

        tool_name = None
        parameters: Dict[str, Any] = {}

        if is_nonenv_category:
            # Truly non-env (terminal/shell) — no matching possible
            pass
        elif tool_category in ("file_system", "filesystem"):
            # File-system turn: try mapping to cloud-drive equivalents
            tool_name = resolve_file_system_action(action, known_tool_names)
            if not tool_name:
                # Fallback: try general tool name matching against action text
                tool_name = extract_tool_name(action, known_tool_names)
        else:
            # Standard env-tool turn — try to find the function name
            tool_name = extract_tool_name(action, known_tool_names)
            if not tool_name:
                # Try category fallback for unusual categories
                tool_name = resolve_category_fallback(
                    tool_category, action, known_tool_names,
                )

        if tool_name and tool_name in tool_schema_lookup:
            parameters = extract_parameters(action, tool_schema_lookup[tool_name])

        # Determine the label for unresolved tools
        if tool_name:
            label = tool_name
        elif is_nonenv_category:
            label = f"_nonenv_{tool_category}"
        else:
            label = f"_unresolved_{tool_category}"

        verified_tool_chain.append({
            "step": i + 1,
            "tool_name": label,
            "parameters": parameters,
            "purpose": action,
            "environment_changes": f"Step {i + 1} of the attack chain.",
        })

    # --- interaction_history: every turn as user + assistant messages ---
    # eval_baseline.py trims to keep all turns except the last (the last turn's
    # prompt comes from planner_history instead).
    interaction_history: List[dict] = []
    for i, turn in enumerate(turns):
        # User message
        user_content = turn_prompts[i] if i < len(turn_prompts) else turn.get("action", "")
        interaction_history.append({"role": "user", "content": user_content})

        # Assistant response
        step = verified_tool_chain[i]
        is_resolved = not step["tool_name"].startswith(("_nonenv_", "_unresolved_"))
        if is_resolved:
            # Matched env tool: include a tool_call so eval can replay it
            tool_call_id = gen_tool_call_id()
            interaction_history.append({
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": step["tool_name"],
                        "arguments": json.dumps(step["parameters"]),
                    },
                }],
            })
            # Placeholder tool result; the real result comes from env replay
            interaction_history.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"status": "success"}),
            })
        else:
            # Non-env or unresolved tool: plain text response so we
            # preserve conversational context without needing env replay
            interaction_history.append({
                "role": "assistant",
                "content": "Done. I've completed the requested action.",
            })

    # --- planner_history: last turn's prompt ---
    last_prompt = turn_prompts[-1] if turn_prompts else turns[-1].get("action", "")
    planner_history = [{"predicted_attack_prompt": last_prompt}]

    # --- generation_config ---
    if dataset == "SHADE_Arena":
        env_field = SHADE_ENV_MAP.get(env_name.lower(), env_name)
    else:
        env_field = env_name

    return {
        "id": item_id,
        "generation_config": {
            "dataset": dataset,
            "environment": env_field,
            "failure_mode": "ToolShield-generated multi-turn attack",
            "tool_chain_generator": "ToolShield",
            "tool_chain_verifier": "ToolShield",
            "prompt_writer": "ToolShield",
            "planner": "ToolShield",
        },
        "attack_plan": {
            "attack_goal": attack_chain.get("final_outcome", "Execute the harmful tool chain."),
            "verified_tool_chain": verified_tool_chain,
            "explanation": attack_chain.get("analysis", ""),
        },
        "interaction_history": interaction_history,
        "planner_history": planner_history,
        # Extra metadata not in standard STAC format; ignored by eval scripts
        "toolshield_metadata": {
            "source_env": env_name,
            "task_dir": task_dir.name,
            "num_turns": len(turns),
            "evaluator_instructions": attack_chain.get("evaluator_instructions", ""),
        },
    }


# ---------------------------------------------------------------------------
# ASB companion data generation
# ---------------------------------------------------------------------------

def build_asb_companion_entry(
    item_id: int,
    env_name: str,
    tool_schemas: List[dict],
    asb_env_params: Dict[str, dict] | None = None,
) -> dict:
    """Build an Agent_SafetyBench released_data.json-style entry.

    eval_baseline.py looks up ASB items by ID in released_data.json to create
    AgentSafetyBenchEnvironment instances.  For ToolShield-generated items
    we synthesize the minimal entry needed.

    If ``asb_env_params`` is provided (mapping env_name -> parameters dict from
    released_data.json), the real parameters are used so environment classes
    that expect specific data (e.g. Traffic.locations) work correctly.
    """
    tool_names = [t["name"] for t in tool_schemas if "name" in t]
    parameters = {}
    if asb_env_params and env_name in asb_env_params:
        parameters = asb_env_params[env_name]
    return {
        "id": item_id,
        "risks": ["ToolShield-generated attack"],
        "instruction": "",
        "environments": [{
            "name": env_name,
            "tools": tool_names,
            "parameters": parameters,
        }],
        "failure_modes": "",
        "fulfillable": 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ToolShield multi-turn attacks to STAC benchmark format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--toolshield-output", type=Path, required=True,
        help="ToolShield output directory (contains env subdirectories)",
    )
    parser.add_argument(
        "--stac-root", type=Path, default=None,
        help="Path to MultiTurnAgentAttack repo root (auto-detected if not set)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("output/toolshield_stac_attacks.json"),
        help="Output JSON file (default: output/toolshield_stac_attacks.json)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["SHADE_Arena", "Agent_SafetyBench"],
        help="Force dataset type (default: auto-detect from env name)",
    )
    parser.add_argument(
        "--envs", nargs="+", default=None,
        help="Only convert specific environments (default: all)",
    )
    parser.add_argument(
        "--start-id", type=int, default=10000,
        help="Starting ID for items (default: 10000, above STAC benchmark range)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview without writing output",
    )
    args = parser.parse_args()

    # --- Locate STAC repo ---
    stac_root = args.stac_root
    if stac_root is None:
        # Baseline/ is directly under MultiTurnAgentAttack/, so parents[1] is repo root
        for candidate in [
            Path(__file__).resolve().parents[1],
        ]:
            if candidate.is_dir() and (
                (candidate / "Agent_SafetyBench").is_dir()
                or (candidate / "SHADE_Arena").is_dir()
            ):
                stac_root = candidate
                break
        if stac_root:
            print(f"Auto-detected STAC root: {stac_root}")
        else:
            print("Warning: Could not auto-detect STAC root. Use --stac-root to specify.")
            print("         Tool name matching will be limited.\n")

    # --- Load tool schemas ---
    asb_schemas: Dict[str, List[dict]] = {}
    shade_tools: Dict[str, dict] = {}

    asb_env_params: Dict[str, dict] = {}  # env_name -> parameters from released_data

    if stac_root:
        asb_path = stac_root / "Agent_SafetyBench"
        if asb_path.is_dir():
            print("Loading Agent_SafetyBench tool schemas...")
            asb_schemas = load_asb_tool_schemas(asb_path)
            print(f"  {len(asb_schemas)} environment schemas loaded")
            # Load env parameters from released_data.json for companion entries
            released_path = asb_path / "data" / "released_data.json"
            if released_path.exists():
                with open(released_path) as f:
                    released = json.load(f)
                for item in released:
                    for env_info in item.get("environments", []):
                        name = env_info.get("name", "")
                        if name and name not in asb_env_params:
                            params = env_info.get("parameters")
                            if params:
                                asb_env_params[name] = params
                print(f"  {len(asb_env_params)} environment parameters loaded")

        shade_path = stac_root / "SHADE_Arena"
        if shade_path.is_dir():
            print("Loading SHADE-Arena tool names...")
            shade_tools = load_shade_tool_info(shade_path)
            for env, info in shade_tools.items():
                print(f"  {env}: {len(info['tool_names'])} tools")

    print()

    # --- Discover attack directories ---
    ts_output = args.toolshield_output
    if not ts_output.is_dir():
        print(f"ERROR: {ts_output} is not a directory")
        sys.exit(1)

    env_dirs = sorted(
        p for p in ts_output.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if args.envs:
        env_dirs = [d for d in env_dirs if d.name in args.envs]

    if not env_dirs:
        print("No environment directories found.")
        sys.exit(1)

    # --- Convert ---
    stac_data: List[dict] = []
    asb_companion: List[dict] = []  # companion entries for ASB items
    item_id = args.start_id
    stats = {"envs": 0, "converted": 0, "skipped": 0, "shade": 0, "asb": 0}

    for env_dir in env_dirs:
        env_name = env_dir.name
        dataset = args.dataset or detect_dataset(env_name)

        # Resolve tool schemas for this environment
        known_tool_names: List[str] = []
        tool_schema_lookup: Dict[str, dict] = {}

        if dataset == "SHADE_Arena":
            info = shade_tools.get(env_name.lower(), {})
            known_tool_names = info.get("tool_names", [])
            for fn in info.get("functions", []):
                tool_schema_lookup[fn["name"]] = {
                    "parameters": {
                        "properties": {p: {"type": "string"} for p in fn.get("params", [])}
                    }
                }
        elif dataset == "Agent_SafetyBench":
            schemas = asb_schemas.get(env_name, [])
            known_tool_names = [t["name"] for t in schemas if "name" in t]
            tool_schema_lookup = {t["name"]: t for t in schemas if "name" in t}

        # Find multi_turn_task.* directories
        task_dirs = sorted(
            p for p in env_dir.iterdir()
            if p.is_dir() and p.name.startswith("multi_turn_task.")
        )
        if not task_dirs:
            continue

        stats["envs"] += 1
        n_tools_str = f"{len(known_tool_names)} known tools" if known_tool_names else "no tool schemas"
        print(f"  {env_name} [{dataset}]: {len(task_dirs)} tasks, {n_tools_str}")

        for task_dir in task_dirs:
            if args.dry_run:
                ac_path = task_dir / "attack_chain.json"
                if ac_path.exists():
                    with open(ac_path) as f:
                        ac = json.load(f)
                    n_turns = len(ac.get("turns", []))
                    print(f"    {task_dir.name}: {n_turns} turns → id={item_id}")
                    stats["converted"] += 1
                    item_id += 1
                continue

            result = convert_attack(
                env_name, task_dir, dataset,
                known_tool_names, tool_schema_lookup, item_id,
            )
            if result is None:
                stats["skipped"] += 1
                continue

            stac_data.append(result)

            if dataset == "SHADE_Arena":
                stats["shade"] += 1
            else:
                stats["asb"] += 1
                # Generate companion entry for ASB environment lookup
                schemas = asb_schemas.get(env_name, [])
                asb_companion.append(build_asb_companion_entry(item_id, env_name, schemas, asb_env_params))

            item_id += 1
            stats["converted"] += 1

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Environments:  {stats['envs']}")
    print(f"Converted:     {stats['converted']}  (SHADE: {stats['shade']}, ASB: {stats['asb']})")
    print(f"Skipped:       {stats['skipped']}")

    if args.dry_run:
        print("\n(Dry run -- no output written)")
        return

    if not stac_data:
        print("\nNo attacks converted.")
        return

    # --- Write STAC output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stac_data, f, indent=2, ensure_ascii=False)
    print(f"\nSTAC data:     {args.output}  ({len(stac_data)} items)")

    # --- Write ASB companion file (if any ASB items) ---
    if asb_companion:
        companion_path = args.output.with_name(args.output.stem + "_asb_envs.json")
        with open(companion_path, "w") as f:
            json.dump(asb_companion, f, indent=2, ensure_ascii=False)
        print(f"ASB companion: {companion_path}  ({len(asb_companion)} entries)")
        print(
            "\n  Note: For Agent_SafetyBench items, merge this companion file into\n"
            "  Agent_SafetyBench/data/released_data.json, or modify eval_baseline.py\n"
            "  to load it. eval_baseline.py looks up ASB environment configs by item ID."
        )

    # --- Quality report ---
    chain_lengths = [len(d["attack_plan"]["verified_tool_chain"]) for d in stac_data]
    total_steps = sum(chain_lengths)
    resolved_steps = sum(
        1 for d in stac_data
        for s in d["attack_plan"]["verified_tool_chain"]
        if not s["tool_name"].startswith(("_nonenv_", "_unresolved_"))
    )
    nonenv_steps = sum(
        1 for d in stac_data
        for s in d["attack_plan"]["verified_tool_chain"]
        if s["tool_name"].startswith("_nonenv_")
    )
    unresolved_steps = sum(
        1 for d in stac_data
        for s in d["attack_plan"]["verified_tool_chain"]
        if s["tool_name"].startswith("_unresolved_")
    )
    steps_with_params = sum(
        1 for d in stac_data
        for s in d["attack_plan"]["verified_tool_chain"]
        if s["parameters"]
    )

    print(f"\n{'='*60}")
    print("Quality Report:")
    print(f"  Chain lengths:       min={min(chain_lengths)}, max={max(chain_lengths)}, avg={sum(chain_lengths)/len(chain_lengths):.1f}")
    print(f"  Total steps:         {total_steps}")
    print(f"  Resolved (matched):  {resolved_steps}/{total_steps} ({100*resolved_steps/max(total_steps,1):.0f}%)")
    print(f"  Non-env (terminal/shell):  {nonenv_steps}/{total_steps} ({100*nonenv_steps/max(total_steps,1):.0f}%)")
    print(f"  Unresolved:          {unresolved_steps}/{total_steps} ({100*unresolved_steps/max(total_steps,1):.0f}%)")
    print(f"  Steps with params:   {steps_with_params}/{total_steps} ({100*steps_with_params/max(total_steps,1):.0f}%)")

    # --- Eval instructions ---
    print(f"\n{'='*60}")
    print("Next steps -- evaluate against a target agent:\n")
    print(f"  cd /path/to/MultiTurnAgentAttack\n")
    print(f"  python -m Baseline.eval_baseline \\")
    print(f"    --input_path {args.output.resolve()} \\")
    print(f"    --model_agent gpt-4.1 \\")
    print(f"    --defense no_defense \\")
    print(f"    --batch_size 1")


if __name__ == "__main__":
    main()
