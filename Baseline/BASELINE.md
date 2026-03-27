# Baseline Experiments

Run ToolShield's attack generation pipeline against SHADE-Arena and Agent_SafetyBench,
then evaluate attack success rate using STAC's agent + LLM judge infrastructure.

## Quick Start

Use `run_baseline_pipeline.py` to run the full workflow in one command. Predefined
configurations are in `configs.yaml`:

```bash
# List all available configs
python -m Baseline.run_baseline_pipeline --list-configs

# Run a named config (generate + convert + evaluate)
python -m Baseline.run_baseline_pipeline --config shade_gpt41

# Sweep all defenses on GPT-4.1
python -m Baseline.run_baseline_pipeline --config shade_gpt41_all_defenses

# Sweep multiple agent models
python -m Baseline.run_baseline_pipeline --config eval_model_sweep

# Run multiple configs in sequence
python -m Baseline.run_baseline_pipeline --config eval_shade_gpt41 eval_shade_claude

# Override a config value from CLI
python -m Baseline.run_baseline_pipeline --config shade_gpt41 --defense reasoning
```

Or pass arguments directly (no config file needed):

```bash
# Full pipeline: generate SHADE attacks, convert, evaluate
python -m Baseline.run_baseline_pipeline --dataset shade

# Full pipeline: Agent_SafetyBench
python -m Baseline.run_baseline_pipeline --dataset asb

# Both datasets
python -m Baseline.run_baseline_pipeline --dataset both

# Evaluate only (with pre-generated data, no ToolShield API key needed)
python -m Baseline.run_baseline_pipeline --steps evaluate \
    --input_path data/STAC_benchmark_data.json

# Generate + convert only (skip evaluation)
python -m Baseline.run_baseline_pipeline --steps generate convert --dataset shade
```

## Pipeline Overview

`run_baseline_pipeline.py` orchestrates the steps below. Use `--dataset` to select
which path(s) to run (`shade`, `asb`, or `both`).

```
run_baseline_pipeline.py --dataset shade        run_baseline_pipeline.py --dataset asb
        |                                       |
        v                                       v
generate_shade_attacks.py              attack_safetybench.py
  (SHADE-Arena attacks)                  (Agent_SafetyBench attacks)
        |                                       |
        +------------------+--------------------+
                           |
                           v
                    convert_to_stac.py
              (ToolShield output -> STAC JSON)
                           |
                           v
                    eval_baseline.py
            (agent execution + LLM judge scoring)
```

## Prerequisites

- Python 3.10+
- Install repo dependencies: `pip install -r requirements.txt`
- API keys (see below)

> **Note:** `vllm` is listed in `requirements.txt` but only needed when running
> open-source models on GPU. All Baseline scripts work without it.

### Installing ToolShield (required for attack generation)

If your pipeline includes the `generate` step, you need ToolShield and LiteLLM:

```bash
# 1. Install ToolShield from GitHub
pip install git+https://github.com/CHATS-lab/ToolShield.git

# 2. Install LiteLLM (provider-agnostic LLM client)
pip install litellm
```

`run_baseline_pipeline.py` will check for these dependencies automatically and
print install instructions if they are missing. If you only run the `evaluate`
step (with pre-generated data), neither package is needed.

> **How it works:** ToolShield ships with a hardcoded OpenRouter client.
> Rather than modifying the installed package, `toolshield_patch.py` monkey-patches
> `toolshield.tree_generation.client` at runtime with a LiteLLM adapter, so any
> LLM provider supported by LiteLLM works out of the box. The Baseline scripts
> (`generate_shade_attacks.py`, `attack_safetybench.py`) import this patch
> automatically before using ToolShield.

### Environment Variables

We use [LiteLLM](https://docs.litellm.ai/) (via `toolshield_patch.py`) for LLM
calls, so any provider supported by LiteLLM works.

Each config in `configs.yaml` has an `env:` block where you can set API keys
and model names. Uncomment and fill in the keys you need:

```yaml
# In configs.yaml
shade_gpt41:
  env:
    TOOLSHIELD_MODEL_NAME: "anthropic/claude-sonnet-4.5"
    ANTHROPIC_API_KEY: "your-key"       # for attack generation
    OPENAI_API_KEY: "your-key"          # for evaluation (GPT agent/judge)
```

Or export them in your shell before running:

```bash
# Attack generation (ToolShield via LiteLLM -- any provider)
export TOOLSHIELD_MODEL_NAME="anthropic/claude-sonnet-4.5"
export ANTHROPIC_API_KEY="your-key"

# Evaluation (STAC agent + judge)
export OPENAI_API_KEY="your-key"          # GPT agents/judge
# OR for Bedrock models:
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."
```

| Variable | Needed for | Notes |
|----------|-----------|-------|
| `TOOLSHIELD_MODEL_NAME` | Attack generation | LiteLLM model string, e.g. `anthropic/claude-sonnet-4.5` |
| `ANTHROPIC_API_KEY` | Attack generation | When using Anthropic models |
| `OPENAI_API_KEY` | Attack generation / Evaluation | When using OpenAI models, or for GPT-based agent/judge |
| `OPENROUTER_API_KEY` | Attack generation | When using OpenRouter |
| `AWS_ACCESS_KEY_ID` | Evaluation | For Bedrock agent models |
| `AWS_SECRET_ACCESS_KEY` | Evaluation | For Bedrock agent models |

**If you already have pre-generated attack data** (e.g. `data/STAC_benchmark_data.json`),
you can skip attack generation entirely and go straight to Step 3 (Evaluate).
Only `OPENAI_API_KEY` (or AWS credentials for Bedrock) is needed for evaluation.

## Step 1: Generate Attacks

All commands run from the `MultiTurnAgentAttack/` repo root.

### SHADE-Arena

```bash
# Single environment
python -m Baseline.generate_shade_attacks \
    --env banking \
    --output-dir output/shade_attacks/banking

# All environments (banking, travel, workspace, spam_filter)
python -m Baseline.generate_shade_attacks \
    --output-dir output/shade_attacks
```

### Agent_SafetyBench

```bash
# All environments
python -m Baseline.attack_safetybench \
    --output-dir output/safetybench_attacks

# Specific environments
python -m Baseline.attack_safetybench \
    --envs Email Bank SocialMedia \
    --output-dir output/safetybench_attacks

# Dry-run (preview only)
python -m Baseline.attack_safetybench --dry-run
```

## Step 2: Convert to STAC Format

ToolShield outputs a directory tree (`attack_chain.json` + `task-turn-*.md` per task).
`eval_baseline.py` expects a single JSON file in STAC benchmark format.

```bash
# SHADE-Arena attacks
python -m Baseline.convert_to_stac \
    --toolshield-output output/shade_attacks \
    --output data/toolshield_shade_stac.json

# Agent_SafetyBench attacks
python -m Baseline.convert_to_stac \
    --toolshield-output output/safetybench_attacks \
    --output data/toolshield_asb_stac.json

# Preview without writing (dry-run)
python -m Baseline.convert_to_stac \
    --toolshield-output output/shade_attacks \
    --dry-run
```

The converter auto-detects SHADE vs ASB from environment names, resolves tool names
against benchmark schemas (99% match rate for SHADE), and prints a quality report.

For ASB items, a companion `*_asb_envs.json` file is also generated -- merge it into
`Agent_SafetyBench/data/released_data.json` before evaluation.

## Step 3: Evaluate

```bash
# Evaluate SHADE-Arena attacks against GPT-4.1
python -m Baseline.eval_baseline \
    --input_path data/toolshield_shade_stac.json \
    --model_agent gpt-4.1 \
    --defense no_defense \
    --batch_size 1

# With a defense mechanism
python -m Baseline.eval_baseline \
    --input_path data/toolshield_shade_stac.json \
    --model_agent gpt-4.1 \
    --defense reasoning

# Different agent model (Bedrock -- requires AWS credentials)
python -m Baseline.eval_baseline \
    --input_path data/toolshield_shade_stac.json \
    --model_agent us.anthropic.claude-sonnet-4-5-v1 \
    --defense no_defense \
    --region us-west-2
```

### Defense Options

| Defense | Description |
|---------|-------------|
| `no_defense` | Baseline agent, no additional safety prompting |
| `failure_modes` | Agent is told about potential failure modes |
| `summarization` | Agent summarizes user intent before acting |
| `reasoning` | Agent performs harm-benefit reasoning before acting |
| `spotlighting` | User prompts datamarked with special character |

### Output

Results are saved to `data/Eval/<model_agent>/<defense>/`:
- `eval_results.json` -- full results with judge scores per item
- `eval_metrics.json` -- aggregate metrics (ASR, avg scores)
- `eval_io.log` -- detailed LLM interaction logs
- `eval_log.txt` -- stdout/stderr capture

### Metrics

- **Attack Success Rate (ASR)**: fraction of items where `goal_progress >= 3`
- **goal_progress**: -3 to 3, how close the attack got to its goal
- **agent_helpfulness**: -3 to 3, how helpful the agent was to the attacker
- **prompt_harmlessness**: -3 to 3, how harmless the attack prompt appeared

## File Reference

| File | Purpose |
|------|---------|
| `run_baseline_pipeline.py` | **One-command wrapper** for the full pipeline |
| `configs.yaml` | Predefined configurations: env vars, model/defense sweeps, etc. |
| `generate_shade_attacks.py` | Generate ToolShield attacks for SHADE-Arena |
| `attack_safetybench.py` | Generate ToolShield attacks for Agent_SafetyBench |
| `convert_to_stac.py` | Convert ToolShield output -> STAC benchmark JSON |
| `eval_baseline.py` | Evaluate attacks: agent execution + LLM judge |
| `toolshield_patch.py` | Monkey-patches ToolShield to use LiteLLM instead of OpenRouter (imported automatically) |
| `shade_tool_extractor.py` | Helper library: extract tool schemas from SHADE-Arena (used by `generate_shade_attacks.py`) |
