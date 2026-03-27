#!/usr/bin/env python3
"""
Config-driven wrapper for STAC benchmark evaluation.

Runs eval_STAC_benchmark with parameters from configs.yaml.
Supports sweeps over models and defenses.

Examples:
    # Run a named config
    python -m STAC_eval.run_eval --config gpt41

    # Run multiple configs
    python -m STAC_eval.run_eval --config gpt41 gpt41_reasoning

    # List all configs
    python -m STAC_eval.run_eval --list-configs

    # CLI args (no config)
    python -m STAC_eval.run_eval --model_agent gpt-4.1 --defense no_defense

    # CLI args override config values
    python -m STAC_eval.run_eval --config gpt41 --defense reasoning --batch_size 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "configs.yaml"

ALL_DEFENSES = ["no_defense", "failure_modes", "summarization", "reasoning", "spotlighting"]

# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

def load_configs() -> dict[str, Any]:
    """Load named configurations from configs.yaml.

    The ``_global`` key provides defaults inherited by all configs.
    The ``env`` sub-dict is merged key-by-key so per-config env vars
    override individual global env vars.
    """
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f) or {}

    global_cfg: dict[str, Any] = {}
    if "_global" in raw:
        global_cfg = raw.pop("_global")

    global_env = global_cfg.pop("env", {}) or {}

    for cfg in raw.values():
        for key, val in global_cfg.items():
            cfg.setdefault(key, val)
        merged_env = dict(global_env)
        merged_env.update(cfg.get("env", {}) or {})
        cfg["env"] = merged_env

    return raw


def list_configs() -> None:
    """Print all available configurations and exit."""
    configs = load_configs()
    if not configs:
        print("No configurations found in configs.yaml")
        sys.exit(0)

    print(f"\nAvailable configurations ({CONFIG_PATH.name}):\n")
    print(f"  {'Name':<25} Description")
    print(f"  {'-'*25} {'-'*50}")
    for name, cfg in configs.items():
        desc = cfg.get("description", "")
        print(f"  {name:<25} {desc}")
        for key in ("model_agent", "defense"):
            val = cfg.get(key)
            if isinstance(val, list):
                print(f"  {'':25} {key}={val}")
        print()
    print("Usage: python -m STAC_eval.run_eval --config <name>")


# -------------------------------------------------------------------------
# Env var management
# -------------------------------------------------------------------------

def set_env_vars(env_vars: dict[str, str]) -> dict[str, str | None]:
    """Set environment variables, returning old values for restoration."""
    old = {}
    for key, val in env_vars.items():
        if val is None or str(val).strip() == "":
            continue
        old[key] = os.environ.get(key)
        os.environ[key] = str(val)
        # Mask secrets in output
        display = str(val)[:8] + "..." if len(str(val)) > 12 else str(val)
        print(f"  export {key}={display}")
    return old


def restore_env_vars(old: dict[str, str | None]) -> None:
    """Restore environment variables to previous values."""
    for key, val in old.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


# -------------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------------

def run_eval(model_agent: str, defense: str, args: argparse.Namespace) -> None:
    """Run eval_STAC_benchmark for one (model_agent, defense) combo."""
    cmd = [
        sys.executable, "-m", "STAC_eval.eval_STAC_benchmark",
        "--model_agent", model_agent,
        "--model_planner", args.model_planner,
        "--model_judge", args.model_judge,
        "--defense", defense,
        "--batch_size", str(args.batch_size),
        "--max_n_turns", str(args.max_n_turns),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--region", args.region,
        "--input_path", args.input_path,
        "--output_dir", args.output_dir,
    ]

    print(f"\n{'=' * 60}")
    print(f"  agent={model_agent}  defense={defense}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: agent={model_agent} defense={defense} (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_config(args: argparse.Namespace) -> None:
    """Execute evaluation, sweeping over model_agent and defense lists."""
    models = args.model_agent if isinstance(args.model_agent, list) else [args.model_agent]
    defenses = args.defense if isinstance(args.defense, list) else [args.defense]

    total = len(models) * len(defenses)
    print(f"\nEvaluation plan: {total} run(s)")
    print(f"  Models:   {models}")
    print(f"  Defenses: {defenses}")
    print(f"  Input:    {args.input_path}")
    print(f"  Batch:    {args.batch_size}")

    for model in models:
        for defense in defenses:
            run_eval(model, defense, args)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

DEFAULTS = dict(
    input_path="data/STAC_benchmark_data.json",
    output_dir="data/Eval",
    model_planner="gpt-4.1",
    model_judge="gpt-4.1",
    model_agent="gpt-4.1",
    batch_size=512,
    defense="no_defense",
    max_n_turns=3,
    temperature=0.0,
    top_p=0.95,
    region="us-west-2",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven wrapper for STAC benchmark evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument("--config", nargs="+", default=None, metavar="NAME",
                        help="Named config(s) from configs.yaml")
    parser.add_argument("--list-configs", action="store_true",
                        help="List all available configurations and exit")

    # Evaluation options (override config values)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_planner", type=str, default=None)
    parser.add_argument("--model_judge", type=str, default=None)
    parser.add_argument("--model_agent", type=str, default=None)
    parser.add_argument("--defense", type=str, default=None, choices=ALL_DEFENSES)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_n_turns", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--region", type=str, default=None)

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    # ------------------------------------------------------------------
    # Config mode
    # ------------------------------------------------------------------
    if args.config:
        all_configs = load_configs()
        for config_name in args.config:
            if config_name not in all_configs:
                print(f"ERROR: Unknown config '{config_name}'. Use --list-configs to see available.")
                sys.exit(1)

            cfg = all_configs[config_name]
            print(f"\n{'#' * 60}")
            print(f"  Config: {config_name}")
            print(f"  {cfg.get('description', '')}")
            print(f"{'#' * 60}")

            # Build run_args from defaults -> config -> CLI overrides
            run_args = argparse.Namespace(**DEFAULTS)

            # Apply config values
            for key in DEFAULTS:
                if key in cfg:
                    setattr(run_args, key, cfg[key])

            # CLI overrides (only if explicitly provided)
            skip_keys = {"config", "list_configs"}
            for k, v in vars(args).items():
                if v is not None and k not in skip_keys:
                    setattr(run_args, k, v)

            # Set env vars
            old_env = {}
            if "env" in cfg and isinstance(cfg["env"], dict):
                print("\nSetting environment variables:")
                old_env = set_env_vars(cfg["env"])

            try:
                run_config(run_args)
            finally:
                if old_env:
                    restore_env_vars(old_env)

        print(f"\n{'#' * 60}")
        print(f"  All configs complete: {', '.join(args.config)}")
        print(f"{'#' * 60}")
        return

    # ------------------------------------------------------------------
    # Direct CLI mode (no --config)
    # ------------------------------------------------------------------
    run_args = argparse.Namespace(**{
        k: (getattr(args, k) if getattr(args, k) is not None else v)
        for k, v in DEFAULTS.items()
    })

    run_config(run_args)

    print(f"\n{'=' * 60}")
    print("  Evaluation complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
