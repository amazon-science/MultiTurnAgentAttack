#!/usr/bin/env python3
"""
One-command wrapper for the full Baseline attack-and-evaluate pipeline.

Runs up to three stages in sequence:
  1. generate  -- Generate multi-turn attacks with ToolShield
  2. convert   -- Convert ToolShield output to STAC benchmark format
  3. evaluate  -- Run attacks against an agent and score with LLM judge

Each stage can be run independently or as part of the full pipeline.
Use --config to load predefined configurations from configs.yaml.

Examples:
    # Run a named config
    python -m Baseline.run_baseline_pipeline --config shade_gpt41

    # Run multiple configs in sequence
    python -m Baseline.run_baseline_pipeline --config eval_shade_gpt41 eval_shade_claude

    # List all available configs
    python -m Baseline.run_baseline_pipeline --list-configs

    # Full pipeline via CLI args
    python -m Baseline.run_baseline_pipeline --dataset shade

    # Evaluate only (with pre-generated data)
    python -m Baseline.run_baseline_pipeline --steps evaluate \
        --input_path data/STAC_benchmark_data.json

    # CLI args override config values
    python -m Baseline.run_baseline_pipeline --config shade_gpt41 --defense reasoning
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

TOOLSHIELD_INSTALL_CMD = "pip install git+https://github.com/CHATS-lab/ToolShield.git"
LITELLM_INSTALL_CMD = "pip install litellm"

# Default paths
SHADE_ATTACK_DIR = "output/shade_attacks"
ASB_ATTACK_DIR = "output/safetybench_attacks"
SHADE_STAC_PATH = "data/toolshield_shade_stac.json"
ASB_STAC_PATH = "data/toolshield_asb_stac.json"

ALL_STEPS = ["generate", "convert", "evaluate"]
ALL_DEFENSES = ["no_defense", "failure_modes", "summarization", "reasoning", "spotlighting"]

# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

def load_configs() -> dict[str, Any]:
    """Load named configurations from configs.yaml.

    The special ``_global`` key is not a runnable config. All of its values
    are merged as defaults into every other config (per-config values win).
    The ``env`` sub-dict is merged separately so per-config env vars override
    individual global env vars rather than replacing the whole dict.
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
        # Merge non-env global defaults (per-config wins)
        for key, val in global_cfg.items():
            cfg.setdefault(key, val)
        # Merge env dicts (per-config env vars override global ones)
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
    print(f"  {'Name':<30} Description")
    print(f"  {'-'*30} {'-'*45}")
    for name, cfg in configs.items():
        desc = cfg.get("description", "")
        steps = cfg.get("steps", ALL_STEPS)
        dataset = cfg.get("dataset", "shade")
        print(f"  {name:<30} {desc}")
        print(f"  {'':30} steps={steps}  dataset={dataset}")
        # Show sweep dimensions if any
        for key in ("defense", "model_agent"):
            val = cfg.get(key)
            if isinstance(val, list):
                print(f"  {'':30} {key}={val}")
        print()
    print(f"Usage: python -m Baseline.run_baseline_pipeline --config <name>")


def apply_config(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply config values to args. CLI overrides are handled by the caller."""
    mapping = {
        "steps": "steps",
        "dataset": "dataset",
        "shade_env": "env",
        "asb_envs": "asb_envs",
        "input_path": "input_path",
        "model_agent": "model_agent",
        "model_judge": "model_judge",
        "defense": "defense",
        "batch_size": "batch_size",
        "region": "region",
        "profile": "profile",
        "debug": "debug",
    }
    for cfg_key, arg_key in mapping.items():
        if cfg_key in cfg:
            setattr(args, arg_key, cfg[cfg_key])


# -------------------------------------------------------------------------
# Env var management
# -------------------------------------------------------------------------

def set_env_vars(env_vars: dict[str, str]) -> dict[str, str | None]:
    """Set environment variables, returning old values for restoration."""
    old = {}
    for key, val in env_vars.items():
        old[key] = os.environ.get(key)
        os.environ[key] = str(val)
        print(f"  export {key}={val}")
    return old


def restore_env_vars(old: dict[str, str | None]) -> None:
    """Restore environment variables to previous values."""
    for key, val in old.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


# -------------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------------

def run_cmd(cmd: list[str], description: str) -> None:
    """Run a command, streaming output. Exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def step_generate(args: argparse.Namespace) -> None:
    """Stage 1: Generate attacks with ToolShield."""
    if args.dataset in ("shade", "both"):
        cmd = [sys.executable, "-m", "Baseline.generate_shade_attacks",
               "--output-dir", SHADE_ATTACK_DIR]
        if args.env:
            cmd += ["--env", args.env]
        if args.debug:
            cmd += ["--debug"]
        run_cmd(cmd, "Generate SHADE-Arena attacks")

    if args.dataset in ("asb", "both"):
        cmd = [sys.executable, "-m", "Baseline.attack_safetybench",
               "--output-dir", ASB_ATTACK_DIR]
        if args.asb_envs:
            cmd += ["--envs"] + args.asb_envs
        if args.debug:
            cmd += ["--debug"]
        run_cmd(cmd, "Generate Agent_SafetyBench attacks")


def step_convert(args: argparse.Namespace) -> None:
    """Stage 2: Convert ToolShield output to STAC format."""
    if args.dataset in ("shade", "both"):
        attack_dir = REPO_ROOT / SHADE_ATTACK_DIR
        if attack_dir.is_dir():
            cmd = [sys.executable, "-m", "Baseline.convert_to_stac",
                   "--toolshield-output", SHADE_ATTACK_DIR,
                   "--output", SHADE_STAC_PATH,
                   "--dataset", "SHADE_Arena"]
            run_cmd(cmd, "Convert SHADE-Arena attacks to STAC format")
        else:
            print(f"  Skipping SHADE convert: {attack_dir} not found")

    if args.dataset in ("asb", "both"):
        attack_dir = REPO_ROOT / ASB_ATTACK_DIR
        if attack_dir.is_dir():
            cmd = [sys.executable, "-m", "Baseline.convert_to_stac",
                   "--toolshield-output", ASB_ATTACK_DIR,
                   "--output", ASB_STAC_PATH,
                   "--dataset", "Agent_SafetyBench"]
            run_cmd(cmd, "Convert Agent_SafetyBench attacks to STAC format")
        else:
            print(f"  Skipping ASB convert: {attack_dir} not found")


def step_evaluate(args: argparse.Namespace, model_agent: str, defense: str) -> None:
    """Stage 3: Evaluate attacks against agent + LLM judge."""
    input_paths: list[str] = []
    if args.input_path:
        input_paths.append(args.input_path)
    else:
        if args.dataset in ("shade", "both"):
            p = REPO_ROOT / SHADE_STAC_PATH
            if p.exists():
                input_paths.append(SHADE_STAC_PATH)
            else:
                print(f"  Skipping SHADE eval: {p} not found")
        if args.dataset in ("asb", "both"):
            p = REPO_ROOT / ASB_STAC_PATH
            if p.exists():
                input_paths.append(ASB_STAC_PATH)
            else:
                print(f"  Skipping ASB eval: {p} not found")

    if not input_paths:
        print("  No input data found for evaluation. Run generate + convert first,")
        print("  or pass --input_path / input_path in config.")
        sys.exit(1)

    for input_path in input_paths:
        cmd = [sys.executable, "-m", "Baseline.eval_baseline",
               "--input_path", input_path,
               "--model_agent", model_agent,
               "--model_judge", args.model_judge,
               "--defense", defense,
               "--batch_size", str(args.batch_size)]
        if args.region:
            cmd += ["--region", args.region]
        if args.profile:
            cmd += ["--profile", args.profile]
        run_cmd(cmd, f"Evaluate: {input_path}  agent={model_agent}  defense={defense}")


# -------------------------------------------------------------------------
# Dependency checks
# -------------------------------------------------------------------------

def check_dependencies(steps: list[str]) -> None:
    """Verify required packages are installed before running the pipeline."""
    missing = []

    if "generate" in steps:
        try:
            import toolshield  # noqa: F401
        except ImportError:
            missing.append(("toolshield", TOOLSHIELD_INSTALL_CMD))

        try:
            import litellm  # noqa: F401
        except ImportError:
            missing.append(("litellm", LITELLM_INSTALL_CMD))

    if missing:
        print("\nERROR: Missing required dependencies for the 'generate' step:\n")
        for pkg, cmd in missing:
            print(f"  {pkg:<15}  ->  {cmd}")
        print("\nInstall them and re-run.")
        sys.exit(1)


# -------------------------------------------------------------------------
# Run one configuration
# -------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the pipeline for a single (possibly swept) configuration."""
    steps = args.steps
    check_dependencies(steps)

    # Expand sweep dimensions: defense and model_agent can be lists
    defenses = args.defense if isinstance(args.defense, list) else [args.defense]
    models = args.model_agent if isinstance(args.model_agent, list) else [args.model_agent]

    print(f"\nPipeline: {' -> '.join(steps)}")
    print(f"Dataset:  {args.dataset}")
    if "evaluate" in steps:
        print(f"Agents:   {models}")
        print(f"Defenses: {defenses}")

    # Generate and convert run once (not per sweep combination)
    if "generate" in steps:
        step_generate(args)
    if "convert" in steps:
        step_convert(args)

    # Evaluate runs for each (model, defense) combination
    if "evaluate" in steps:
        for model in models:
            for defense in defenses:
                step_evaluate(args, model, defense)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Baseline attack-and-evaluate pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Config
    parser.add_argument(
        "--config", nargs="+", default=None, metavar="NAME",
        help="Named config(s) from configs.yaml (run in sequence)",
    )
    parser.add_argument(
        "--list-configs", action="store_true",
        help="List all available configurations and exit",
    )

    # Pipeline control
    parser.add_argument(
        "--steps", nargs="+", default=None,
        choices=ALL_STEPS,
        help=f"Pipeline steps to run (default: all = {' '.join(ALL_STEPS)})",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=["shade", "asb", "both"],
        help="Which dataset to process (default: shade)",
    )

    # Generation options
    parser.add_argument(
        "--env", type=str, default=None,
        choices=["banking", "travel", "workspace", "spam_filter"],
        help="Specific SHADE environment (default: all)",
    )
    parser.add_argument(
        "--asb-envs", nargs="+", default=None,
        help="Specific Agent_SafetyBench environments (default: all)",
    )

    # Evaluation options
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--model_agent", type=str, default=None)
    parser.add_argument("--model_judge", type=str, default=None)
    parser.add_argument(
        "--defense", type=str, default=None,
        choices=ALL_DEFENSES,
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)

    # General
    parser.add_argument("--debug", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    # ------------------------------------------------------------------
    # Config mode: load named config(s) and run each
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

            # Start from defaults, apply config, then override with any CLI args
            run_args = argparse.Namespace(
                steps=ALL_STEPS,
                dataset="shade",
                env=None,
                asb_envs=None,
                input_path=None,
                model_agent="gpt-4.1",
                model_judge="gpt-4.1",
                defense="no_defense",
                batch_size=1,
                region=None,
                profile=None,
                debug=False,
            )
            apply_config(cfg, run_args)

            # CLI args override config (only if explicitly provided)
            # Skip bool flags (store_true defaults to False, not None)
            skip_keys = {"config", "list_configs", "debug"}
            cli_overrides = {k: v for k, v in vars(args).items()
                            if v is not None and k not in skip_keys}
            for k, v in cli_overrides.items():
                setattr(run_args, k, v)
            # debug: only override if user passed --debug on CLI
            if args.debug:
                run_args.debug = True

            # Set env vars from config
            old_env = {}
            if "env" in cfg and isinstance(cfg["env"], dict):
                print("\nSetting environment variables:")
                old_env = set_env_vars(cfg["env"])

            try:
                run_pipeline(run_args)
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
    run_args = argparse.Namespace(
        steps=args.steps or ALL_STEPS,
        dataset=args.dataset or "shade",
        env=args.env,
        asb_envs=args.asb_envs,
        input_path=args.input_path,
        model_agent=args.model_agent or "gpt-4.1",
        model_judge=args.model_judge or "gpt-4.1",
        defense=args.defense or "no_defense",
        batch_size=args.batch_size or 1,
        region=args.region,
        profile=args.profile,
        debug=args.debug,
    )

    run_pipeline(run_args)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
