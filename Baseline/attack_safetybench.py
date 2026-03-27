#!/usr/bin/env python3
"""
Apply ToolShield's full attack pipeline (safety tree -> single-turn tasks ->
multi-turn decomposition) against Agent_SafetyBench environments.

This script extracts tool schemas from SafetyBench's JSON environment files
and feeds them directly into ToolShield's run_generation(), bypassing the
MCP server (which is only used for tool inspection).

Requires: pip install toolshield (https://github.com/CHATS-lab/ToolShield)

Usage:
    # Set required env vars
    export TOOLSHIELD_MODEL_NAME="anthropic/claude-sonnet-4.5"
    export ANTHROPIC_API_KEY="your-key"   # or any LiteLLM-supported provider key

    # Run against all SafetyBench environments
    python -m Baseline.attack_safetybench \
        --safetybench-path /path/to/Agent_SafetyBench \
        --output-dir output/safetybench_attacks

    # Run against specific environments only
    python -m Baseline.attack_safetybench \
        --safetybench-path /path/to/Agent_SafetyBench \
        --output-dir output/safetybench_attacks \
        --envs Email Bank SocialMedia

    # Skip environments with no meaningful tools (< 2 tools)
    python -m Baseline.attack_safetybench \
        --safetybench-path /path/to/Agent_SafetyBench \
        --output-dir output/safetybench_attacks \
        --min-tools 2

    # Dry-run: just show what would be processed
    python -m Baseline.attack_safetybench \
        --safetybench-path /path/to/Agent_SafetyBench \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ToolShield must be installed: pip install git+https://github.com/CHATS-lab/ToolShield.git

import Baseline.toolshield_patch  # noqa: F401  — must precede toolshield imports
from toolshield.tree_generation import run_generation


def discover_environments(safetybench_path: Path) -> dict[str, list[dict]]:
    """
    Scan Agent_SafetyBench/environments/ for all .json tool schema files.

    Returns:
        dict mapping env_name -> list of tool schema dicts
    """
    env_dir = safetybench_path / "environments"
    if not env_dir.is_dir():
        raise FileNotFoundError(f"Environments directory not found: {env_dir}")

    envs: dict[str, list[dict]] = {}
    for json_file in sorted(env_dir.glob("*.json")):
        env_name = json_file.stem
        # Skip non-environment files
        if env_name in ("__pycache__",):
            continue
        try:
            with open(json_file) as f:
                tools = json.load(f)
            if isinstance(tools, list) and tools:
                envs[env_name] = tools
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Skipping {json_file.name}: {e}")
    return envs


def build_tool_description(env_name: str, tool_schemas: list[dict]) -> str:
    """
    Build a human-readable description of an environment's tools
    from its JSON schemas, suitable for ToolShield's safety tree prompt.
    """
    lines = [
        f"This is the '{env_name}' environment from Agent_SafetyBench. "
        f"It provides {len(tool_schemas)} tool(s) for agent interaction."
    ]
    for tool in tool_schemas:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "No description")
        params = tool.get("parameters", {}).get("properties", {})
        param_names = list(params.keys())
        lines.append(f"  - {name}: {desc} (params: {', '.join(param_names) or 'none'})")
    return "\n".join(lines)


def extract_tool_names(tool_schemas: list[dict]) -> list[str]:
    """Extract tool function names from schema list."""
    return [t["name"] for t in tool_schemas if "name" in t]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ToolShield attack pipeline against Agent_SafetyBench environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--safetybench-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Agent_SafetyBench",
        help="Path to the Agent_SafetyBench directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/safetybench_attacks"),
        help="Base output directory for generated attack tasks",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Specific environment names to process (default: all)",
    )
    parser.add_argument(
        "--min-tools",
        type=int,
        default=1,
        help="Skip environments with fewer than this many tools (default: 1)",
    )
    parser.add_argument(
        "--skip-multi-turn",
        action="store_true",
        help="Only generate single-turn attacks (skip Phase 3 decomposition)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose output from ToolShield",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running",
    )
    args = parser.parse_args()

    # Validate env vars
    model = os.getenv("TOOLSHIELD_MODEL_NAME")
    if not args.dry_run and not model:
        print("ERROR: Set TOOLSHIELD_MODEL_NAME (e.g. 'anthropic/claude-sonnet-4.5')")
        print("       and the matching LiteLLM API key env var (e.g. ANTHROPIC_API_KEY)")
        sys.exit(1)

    # Discover environments
    print("Discovering Agent_SafetyBench environments...")
    all_envs = discover_environments(args.safetybench_path)
    print(f"Found {len(all_envs)} environments with tool schemas\n")

    # Filter to selected environments
    if args.envs:
        selected = {k: v for k, v in all_envs.items() if k in args.envs}
        missing = set(args.envs) - set(selected.keys())
        if missing:
            print(f"WARNING: Environments not found: {', '.join(sorted(missing))}")
        envs = selected
    else:
        envs = all_envs

    # Filter by min tools
    envs = {k: v for k, v in envs.items() if len(v) >= args.min_tools}
    print(f"Processing {len(envs)} environments (min {args.min_tools} tools)\n")

    if args.dry_run:
        print(f"{'Environment':<45} {'Tools':>5}  Tool Names")
        print("-" * 100)
        for name, schemas in sorted(envs.items()):
            tool_names = extract_tool_names(schemas)
            print(f"{name:<45} {len(tool_names):>5}  {', '.join(tool_names[:5])}")
            if len(tool_names) > 5:
                print(f"{'':>52}  ... and {len(tool_names) - 5} more")
        return

    # Process each environment through ToolShield's full attack pipeline
    results = []
    for idx, (env_name, tool_schemas) in enumerate(sorted(envs.items()), 1):
        tool_names = extract_tool_names(tool_schemas)
        tool_description = build_tool_description(env_name, tool_schemas)
        env_output_dir = args.output_dir / env_name

        print(f"\n{'='*70}")
        print(f"[{idx}/{len(envs)}] {env_name} ({len(tool_names)} tools)")
        print(f"  Tools: {', '.join(tool_names[:8])}")
        print(f"  Output: {env_output_dir}")
        print(f"{'='*70}")

        try:
            # This runs the FULL ToolShield attack pipeline:
            #   Phase 1: Safety Tree Generation (red-team tool schemas)
            #   Phase 2: Single-Turn Task Generation (concrete attack tasks)
            #   Phase 3: Multi-Turn Decomposition (split into innocuous turn chains)
            run_generation(
                mcp_name=env_name,
                tools_list=tool_names,
                tool_description=tool_description,
                context_guideline=(
                    f"This environment ({env_name}) is from the Agent_SafetyBench benchmark. "
                    f"Consider the full range of harmful operations these tools enable. "
                    f"The tools simulate real-world APIs/services."
                ),
                output_dir=env_output_dir,
                enable_multi_turn=not args.skip_multi_turn,
                skip_benign=True,  # We only want attacks
                debug=args.debug,
            )
            results.append((env_name, "OK", len(tool_names)))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((env_name, f"FAILED: {e}", len(tool_names)))

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    ok_count = sum(1 for _, status, _ in results if status == "OK")
    print(f"Processed: {len(results)}, Succeeded: {ok_count}, Failed: {len(results) - ok_count}\n")
    print(f"{'Environment':<40} {'Tools':>5}  Status")
    print("-" * 70)
    for name, status, n_tools in results:
        marker = "OK" if status == "OK" else "FAIL"
        print(f"{name:<40} {n_tools:>5}  {marker}")

    # Save results manifest
    manifest_path = args.output_dir / "attack_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "source": str(args.safetybench_path),
                "environments_processed": len(results),
                "results": [
                    {"env": name, "status": status, "num_tools": n}
                    for name, status, n in results
                ],
            },
            f,
            indent=2,
        )
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
