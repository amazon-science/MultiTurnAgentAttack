#!/usr/bin/env python3
"""
Extract tool function names and docstrings from SHADE Arena's Python tool files.

Produces the same format ToolShield's tree_generation expects:
  - A list of tool function names
  - A description string per environment
  - A detailed tool capabilities string

Usage:
    python -m Baseline.shade_tool_extractor --shade-root /path/to/SHADE_Arena
    python -m Baseline.shade_tool_extractor --shade-root /path/to/SHADE_Arena --env banking
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Map SHADE Arena environment types to their tool modules
# ---------------------------------------------------------------------------

# Each environment uses a specific set of tool files
ENVIRONMENT_TOOL_FILES: Dict[str, List[str]] = {
    "banking": [
        "email_tools.py",
        "calendar_tools.py",
        "drive_tools.py",
        "financial_transfer.py",
        "communication_tools.py",
        "web_tools.py",
    ],
    "travel": [
        "email_tools.py",
        "calendar_tools.py",
        "drive_tools.py",
        "flight_tools.py",
        "hotel_tools.py",
        "restaurant_tools.py",
        "communication_tools.py",
        "travel_tools.py",
        "excel_tools.py",
        "web_tools.py",
    ],
    "workspace": [
        "email_tools.py",
        "calendar_tools.py",
        "drive_tools.py",
        "communication_tools.py",
        "linkedin_tools.py",
        "web_tools.py",
        "excel_tools.py",
    ],
    "spam_filter": [
        "spam_filter_tools.py",
        "email_tools.py",
        "communication_tools.py",
    ],
}


def extract_functions_from_file(filepath: Path) -> List[Dict[str, str]]:
    """Parse a Python file and extract public function names + docstrings."""
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        # Skip private/internal functions
        if node.name.startswith("_"):
            continue

        docstring = ast.get_docstring(node) or ""
        # Extract just the first line of the docstring for brevity
        first_line = docstring.split("\n")[0].strip() if docstring else ""

        # Extract parameter names (skip 'self' and annotated Depends params)
        params = []
        for arg in node.args.args:
            name = arg.arg
            # Skip injected dependencies (Annotated[..., Depends(...)])
            ann = arg.annotation
            if ann and _is_depends_annotation(ann):
                continue
            params.append(name)

        functions.append({
            "name": node.name,
            "short_doc": first_line,
            "full_doc": docstring,
            "params": params,
            "file": filepath.name,
        })

    return functions


def _is_depends_annotation(ann: ast.expr) -> bool:
    """Check if an annotation is Annotated[..., Depends(...)]."""
    if isinstance(ann, ast.Subscript):
        if isinstance(ann.value, ast.Name) and ann.value.id == "Annotated":
            return True
    return False


def extract_for_environment(
    shade_root: Path, env_name: str
) -> Tuple[List[str], str, List[Dict[str, str]]]:
    """
    Extract tools for a given SHADE Arena environment.

    Returns:
        (tool_names, description_string, detailed_functions)
    """
    tools_dir = shade_root / "tools"
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"Tools directory not found: {tools_dir}")

    tool_files = ENVIRONMENT_TOOL_FILES.get(env_name)
    if tool_files is None:
        available = ", ".join(ENVIRONMENT_TOOL_FILES.keys())
        raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")

    all_functions: List[Dict[str, str]] = []
    for tf in tool_files:
        fp = tools_dir / tf
        if not fp.exists():
            print(f"Warning: {fp} not found, skipping", file=sys.stderr)
            continue
        all_functions.extend(extract_functions_from_file(fp))

    tool_names = [f["name"] for f in all_functions]

    # Build a capabilities description string
    by_file: Dict[str, List[Dict]] = {}
    for f in all_functions:
        by_file.setdefault(f["file"], []).append(f)

    desc_parts = [f"SHADE Arena '{env_name}' environment tools:"]
    for filename, funcs in by_file.items():
        category = filename.replace("_tools.py", "").replace(".py", "").replace("_", " ").title()
        desc_parts.append(f"\n**{category}:**")
        for fn in funcs:
            doc = fn["short_doc"] or "No description"
            desc_parts.append(f"  - {fn['name']}: {doc}")

    description = "\n".join(desc_parts)

    return tool_names, description, all_functions


def build_tool_capabilities_string(all_functions: List[Dict[str, str]]) -> str:
    """Build a TOOL_CAPABILITIES-style string from extracted functions."""
    by_file: Dict[str, List[Dict]] = {}
    for f in all_functions:
        by_file.setdefault(f["file"], []).append(f)

    parts = []
    for filename, funcs in by_file.items():
        category = filename.replace("_tools.py", "").replace(".py", "").replace("_", " ").title()
        lines = [f"**{category}**"]
        for fn in funcs:
            doc = fn["short_doc"] or fn["name"]
            lines.append(f"- {fn['name']}: {doc}")
        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SHADE Arena tools for ToolShield")
    parser.add_argument(
        "--shade-root", type=Path,
        default=Path(__file__).resolve().parents[1] / "SHADE_Arena",
        help="Path to SHADE_Arena directory",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        choices=list(ENVIRONMENT_TOOL_FILES.keys()),
        help="Specific environment (default: all)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file (default: stdout)",
    )
    args = parser.parse_args()

    envs = [args.env] if args.env else list(ENVIRONMENT_TOOL_FILES.keys())

    results = {}
    for env_name in envs:
        tool_names, description, functions = extract_for_environment(args.shade_root, env_name)
        capabilities = build_tool_capabilities_string(functions)
        results[env_name] = {
            "tool_names": tool_names,
            "description": description,
            "capabilities": capabilities,
            "num_tools": len(tool_names),
            "functions": [
                {"name": f["name"], "short_doc": f["short_doc"], "params": f["params"]}
                for f in functions
            ],
        }
        print(f"{env_name}: {len(tool_names)} tools extracted", file=sys.stderr)

    output = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
