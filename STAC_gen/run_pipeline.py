#!/usr/bin/env python3
"""
Config-driven wrapper for the full STAC generation pipeline (Steps 1-4).

Reads parameters from STAC_gen/configs.yaml and runs the 4 sequential steps.
Supports running individual steps or the full pipeline.

Examples:
    # Run full pipeline with a named config
    python -m STAC_gen.run_pipeline --config asb_gpt41

    # Run only steps 1 and 2
    python -m STAC_gen.run_pipeline --config asb_gpt41 --steps 1 2

    # List all configs
    python -m STAC_gen.run_pipeline --list-configs

    # CLI args (no config)
    python -m STAC_gen.run_pipeline --dataset Agent_SafetyBench --batch_size 512

    # CLI args override config values
    python -m STAC_gen.run_pipeline --config asb_gpt41 --batch_size 2 --steps 1
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

ALL_STEPS = [1, 2, 3, 4]

# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

def load_configs() -> dict[str, Any]:
    """Load named configurations from configs.yaml."""
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
    print()
    print("Usage: python -m STAC_gen.run_pipeline --config <name>")


# -------------------------------------------------------------------------
# Env var management
# -------------------------------------------------------------------------

def set_env_vars(env_vars: dict[str, str]) -> dict[str, str | None]:
    """Set environment variables, returning old values for restoration."""
    old = {}
    for key, val in env_vars.items():
        if val is None or str(val).strip() == "" or str(val).startswith("your-"):
            continue
        old[key] = os.environ.get(key)
        os.environ[key] = str(val)
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
# Path helpers
# -------------------------------------------------------------------------

def get_step1_output(cfg: argparse.Namespace) -> str:
    return f"output/{cfg.dataset}/tool_chain_attacks_{cfg.model_generator}.json"


def get_step2_output_dir(cfg: argparse.Namespace) -> str:
    return f"output/{cfg.dataset}/verification/tool_chain_attacks_{cfg.model_generator}_{cfg.model_verifier}"


def get_step2_output(cfg: argparse.Namespace) -> str:
    return f"{get_step2_output_dir(cfg)}/gen_res.json"


def get_step3_output_dir(cfg: argparse.Namespace) -> str:
    return get_step2_output_dir(cfg)


def get_step3_output(cfg: argparse.Namespace) -> str:
    return f"{get_step2_output_dir(cfg)}/Prompts/{cfg.model_prompt_writer}/gen_res.json"


def get_step4_input(cfg: argparse.Namespace) -> str:
    return get_step3_output(cfg)


# -------------------------------------------------------------------------
# Step runners
# -------------------------------------------------------------------------

def run_cmd(cmd: list[str], step_label: str) -> None:
    """Run a subprocess command and exit on failure."""
    print(f"\n{'=' * 60}")
    print(f"  {step_label}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\nFAILED: {step_label} (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_step1(cfg: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "STAC_gen.step_1_gen_tool_chains",
        "--dataset", cfg.dataset,
        "--model_name_or_path", cfg.model_generator,
        "--n_cases", str(cfg.n_cases),
        "--batch_size", str(cfg.batch_size),
    ]
    run_cmd(cmd, "Step 1: Generate tool chains")


def run_step2(cfg: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "STAC_gen.step_2_verify_tool_chain",
        "--dataset", cfg.dataset,
        "--model", cfg.model_verifier,
        "--input_path", get_step1_output(cfg),
        "--batch_size", str(cfg.batch_size),
        "--temperature", str(cfg.temperature),
        "--top_p", str(cfg.top_p),
        "--region", cfg.region,
    ]
    run_cmd(cmd, "Step 2: Verify tool chains")


def run_step3(cfg: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "STAC_gen.step_3_reverse_engineer_prompts",
        "--dataset", cfg.dataset,
        "--model", cfg.model_prompt_writer,
        "--output_dir", get_step3_output_dir(cfg),
        "--batch_size", str(cfg.batch_size),
        "--temperature", str(cfg.temperature),
        "--top_p", str(cfg.top_p),
        "--region", cfg.region,
    ]
    run_cmd(cmd, "Step 3: Reverse-engineer prompts")


def run_step4(cfg: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "STAC_gen.step_4_eval_adaptive_planning",
        "--benchmark", cfg.dataset,
        "--input_path", get_step4_input(cfg),
        "--model_planner", cfg.model_planner,
        "--model_judge", cfg.model_judge,
        "--model_agent", cfg.model_agent,
        "--defense", cfg.defense,
        "--batch_size", str(cfg.batch_size),
        "--max_n_turns", str(cfg.max_n_turns),
        "--temperature", str(cfg.temperature),
        "--top_p", str(cfg.top_p),
        "--region", cfg.region,
    ]
    run_cmd(cmd, "Step 4: Adaptive planning evaluation")


STEP_RUNNERS = {1: run_step1, 2: run_step2, 3: run_step3, 4: run_step4}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

DEFAULTS = dict(
    dataset="Agent_SafetyBench",
    model_generator="gpt-4.1",
    model_verifier="gpt-4.1",
    model_prompt_writer="gpt-4.1",
    model_planner="gpt-4.1",
    model_judge="gpt-4.1",
    model_agent="gpt-4.1",
    n_cases=120,
    batch_size=512,
    defense="no_defense",
    max_n_turns=3,
    temperature=0.6,
    top_p=0.95,
    region="us-west-2",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven wrapper for the STAC generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default=None, metavar="NAME",
                        help="Named config from configs.yaml")
    parser.add_argument("--list-configs", action="store_true",
                        help="List all available configurations and exit")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        choices=ALL_STEPS, metavar="N",
                        help="Which steps to run (default: all 4). E.g. --steps 1 2")

    parser.add_argument("--dataset", type=str, default=None,
                        choices=["Agent_SafetyBench", "SHADE_Arena"])
    parser.add_argument("--model_generator", type=str, default=None)
    parser.add_argument("--model_verifier", type=str, default=None)
    parser.add_argument("--model_prompt_writer", type=str, default=None)
    parser.add_argument("--model_planner", type=str, default=None)
    parser.add_argument("--model_judge", type=str, default=None)
    parser.add_argument("--model_agent", type=str, default=None)
    parser.add_argument("--n_cases", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--defense", type=str, default=None)
    parser.add_argument("--max_n_turns", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--region", type=str, default=None)

    args = parser.parse_args()

    if args.list_configs:
        list_configs()
        return

    steps = args.steps or ALL_STEPS

    # Build run config: defaults -> config file -> CLI overrides
    run_cfg = argparse.Namespace(**DEFAULTS)

    if args.config:
        all_configs = load_configs()
        if args.config not in all_configs:
            print(f"ERROR: Unknown config '{args.config}'. Use --list-configs to see available.")
            sys.exit(1)

        cfg = all_configs[args.config]
        print(f"\n{'#' * 60}")
        print(f"  Config: {args.config}")
        print(f"  {cfg.get('description', '')}")
        print(f"{'#' * 60}")

        for key in DEFAULTS:
            if key in cfg:
                setattr(run_cfg, key, cfg[key])

        # Set env vars
        if "env" in cfg and isinstance(cfg["env"], dict):
            print("\nSetting environment variables:")
            set_env_vars(cfg["env"])

    # CLI overrides
    skip_keys = {"config", "list_configs", "steps"}
    for k, v in vars(args).items():
        if v is not None and k not in skip_keys:
            setattr(run_cfg, k, v)

    # Print plan
    print(f"\nPipeline plan:")
    print(f"  Dataset:        {run_cfg.dataset}")
    print(f"  Steps:          {steps}")
    print(f"  Generator:      {run_cfg.model_generator}")
    print(f"  Verifier:       {run_cfg.model_verifier}")
    print(f"  Prompt writer:  {run_cfg.model_prompt_writer}")
    print(f"  Planner:        {run_cfg.model_planner}")
    print(f"  Judge:          {run_cfg.model_judge}")
    print(f"  Agent:          {run_cfg.model_agent}")
    print(f"  Batch size:     {run_cfg.batch_size}")
    print(f"  N cases:        {run_cfg.n_cases}")
    print(f"  Defense:        {run_cfg.defense}")

    # Run steps
    for step in sorted(steps):
        STEP_RUNNERS[step](run_cfg)

    print(f"\n{'#' * 60}")
    print(f"  Pipeline complete: steps {steps}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
