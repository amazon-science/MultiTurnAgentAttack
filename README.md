# MultiTurnAgentAttack

This repository contains data and code for the STAC (Sequential Tool Attack Chaining) framework, which generates and evaluates multi-turn adversarial attacks against LLM agents in tool-use environments.

📄 **Paper**: [STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents](https://arxiv.org/pdf/2509.25624)

## Quick Start

### Just want to evaluate your model on our STAC Benchmark?

If you only want to evaluate your model on our pre-generated **STAC Benchmark (483 test cases)**, you can skip the full pipeline and directly run:

```bash
# Set up environment
conda env create -f environment.yml
conda activate STAC
export OPENAI_API_KEY="your-openai-api-key-here"  # For planner and judge models

# Run evaluation
python -m STAC_eval.eval_STAC_benchmark \
    --model_agent gpt-4.1 \
    --defense no_defense \
    --batch_size 512
```

**Input**: `data/STAC_benchmark_data.json` (483 test cases from both SHADE-Arena and Agent-SafetyBench)

**Output**: Evaluation results in `data/Eval/{model_planner}/{model_agent}/{defense}/gen_res.json`

The benchmark automatically handles both SHADE-Arena and Agent-SafetyBench test cases. Skip to the [Benchmark Evaluation](#benchmark-evaluation-stac_eval) section for more details.

---

## Full Setup

### Prerequisites

Clone the repository and set up the conda environment on a Linux machine:

```bash
# Clone the repository
cd MultiTurnAgentAttack-main

# Create and activate conda environment
conda env create -f environment.yml
conda activate STAC
```

### Alternative Setup (pip/venv)

If you prefer using pip instead of conda:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Before running any scripts, configure the required API keys:

#### OpenAI API (Required for most steps)
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

#### Hugging Face (Required for HF models)
```bash
export HF_TOKEN="your-hf-token-here"
```

#### AWS Bedrock API (Required for AWS models only)
```bash
export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-aws-secret-access-key"
export AWS_SESSION_TOKEN="your-aws-session-token"
```
**Note**: AWS session tokens expire every 12 hours and must be refreshed regularly.

---

## Repository Structure

```
MultiTurnAgentAttack-main/
├── STAC_gen/                          # Full STAC attack generation pipeline
│   ├── step_1_gen_tool_chains.py      # Generate tool chain attacks
│   ├── step_2_verify_tool_chains.py   # Verify generated attacks
│   ├── step_3_reverse_engineer_prompts.py  # Generate adversarial prompts
│   └── step_4_eval_adaptive_planning.py    # Evaluate with adaptive planning
│
├── STAC_eval/                         # Benchmark evaluation
│   └── eval_STAC_benchmark.py         # Evaluate models on STAC benchmark
│
├── data/
│   └── STAC_benchmark_data.json       # Pre-generated benchmark (483 cases)
│
├── Agent_SafetyBench/                 # Agent-SafetyBench environments
├── SHADE_Arena/                       # SHADE-Arena environments
│
├── src/                               # Core implementation
│   ├── Agents.py
│   ├── Environments.py
│   ├── LanguageModels.py
│   ├── STAC.py
│   └── utils.py
│
└── prompts/                           # System prompts for all components
```

---

## Overview

The repository provides two main usage modes:

### 1. **Benchmark Evaluation** (`STAC_eval/`)
Evaluate your model on our pre-generated STAC Benchmark containing 483 test cases across SHADE-Arena and Agent-SafetyBench environments. This is the recommended starting point for most users.

### 2. **Full Automated Attack Generation Pipeline** (`STAC_gen/`)
Generate new attacks from scratch using the complete 4-step STAC pipeline. This is useful for:
- Creating attacks for new environments
- Experimenting with different attack generation strategies
- Extending the benchmark with additional test cases

---

## Benchmark Evaluation (STAC_eval)

### Quick Evaluation

Evaluate your agent against the STAC Benchmark with a single command:

```bash
python -m STAC_eval.eval_STAC_benchmark \
    --model_agent gpt-4.1 \
    --defense no_defense \
    --batch_size 512
```

### Parameters

- `--input_path`: Path to benchmark data (default: `data/STAC_benchmark_data.json`)
- `--output_dir`: Output directory (default: `data/Eval`)
- `--model_planner`: Model for attack planning (default: `gpt-4.1`)
- `--model_judge`: Model for evaluation (default: `gpt-4.1`)
- `--model_agent`: **Your model to evaluate** (default: `gpt-4.1`)
- `--temperature`: Sampling temperature (default: `0.0`)
- `--top_p`: Nucleus sampling parameter (default: `0.95`)
- `--batch_size`: Batch size for evaluation (default: `1`)
- `--region`: AWS region for Bedrock API (default: `us-west-2`)
- `--max_n_turns`: Maximum conversation turns (default: `3`)
- `--defense`: Defense mechanism to evaluate (default: `no_defense`)

### Defense Mechanisms

Evaluate different defense strategies:

```bash
# No defense (baseline)
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense no_defense

# Spotlighting (datamarking user prompts)
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense spotlighting

# Failure mode awareness
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense failure_modes

# User intent summarization
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense summarization

# Harm-benefit reasoning
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --defense reasoning
```

### Evaluating Different Models

#### OpenAI Models
**Requirements**: `OPENAI_API_KEY`

```bash
# GPT-4.1
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1 --batch_size 512

# GPT-4.1-mini
python -m STAC_eval.eval_STAC_benchmark --model_agent gpt-4.1-mini --batch_size 512
```

#### AWS Bedrock Models
**Requirements**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`

```bash
# Llama 3.3 70B
python -m STAC_eval.eval_STAC_benchmark \
    --model_agent us.meta.llama3-3-70b-instruct-v1:0 \
    --batch_size 10

# Llama 3.1 405B
python -m STAC_eval.eval_STAC_benchmark \
    --model_agent meta.llama3-1-405b-instruct-v1:0 \
    --batch_size 10
```

#### Hugging Face Models
**Requirements**: GPU access (tested on H100s), `HF_TOKEN`

```bash
# Request GPU node
# e.g., on slurm: srun --exclusive --pty --partition=p4 --nodes=1 /bin/bash

# Evaluate model
python -m STAC_eval.eval_STAC_benchmark \
    --model_agent Qwen/Qwen3-32B \
    --temperature 0.6 \
    --batch_size 100

# Exit GPU session
exit
```

---

## Full STAC Generation Pipeline (STAC_gen)

Use this pipeline to generate new attacks from scratch. The pipeline consists of 4 sequential steps:

### Step 1: Generate Tool Chains

Generate adversarial tool chain attacks for different environments.

**Requirements**: `OPENAI_API_KEY`

```bash
python -m STAC_gen.step_1_gen_tool_chains \
    --dataset SHADE_Arena \
    --model_name_or_path gpt-4.1 \
    --n_cases 120 \
    --batch_size 32
```

**Parameters**:
- `--dataset`: Target dataset (`SHADE_Arena` or `Agent_SafetyBench`) - **required**
- `--model_name_or_path`: Model for attack generation (default: `gpt-4.1`)
- `--output_path`: Output path for generated attacks (optional, auto-generated if not provided)
- `--n_cases`: Number of cases to generate (default: 120 for Agent_SafetyBench, all for SHADE_Arena)
- `--batch_size`: Batch size for OpenAI Batch API. Use 'None' for synchronous API (default: `None`)

### Step 2: Verify Tool Chains

Verify that generated attacks execute correctly in their environments.

**Requirements**: `OPENAI_API_KEY`

```bash
python -m STAC_gen.step_2_verify_tool_chain \
    --dataset SHADE_Arena \
    --model gpt-4.1 \
    --batch_size 512
```

**Parameters**:
- `--dataset`: Target dataset (`SHADE_Arena` or `Agent_SafetyBench`) - **required**
- `--model`: Model for verification (default: `gpt-4.1`)
- `--input_path`: Path to Step 1 outputs (optional, auto-generated if not provided)
- `--batch_size`: Number of attacks to verify simultaneously (default: `512`)
- `--temperature`: Sampling temperature (default: `0.6`)
- `--top_p`: Nucleus sampling parameter (default: `0.95`)
- `--max_tokens`: Maximum tokens per verification (default: `8192`)
- `--env`: Filter for specific environment (optional)
- `--region`: AWS region for model access (default: `us-east-1`)

### Step 3: Reverse Engineer Prompts

Generate natural language prompts that lead to the verified tool chains.

**Requirements**: GPU cluster access (tested on H100s)

```bash
# Request GPU node (example using slurm)
srun --exclusive --pty --partition=p4 --nodes=1 /bin/bash

# Run reverse engineering
python -m STAC_gen.step_3_reverse_engineer_prompts \
    --dataset SHADE_Arena \
    --model Qwen/Qwen3-32B \
    --temperature 0.6 \
    --batch_size 512

# Exit GPU session
exit
```

**Parameters**:
- `--dataset`: Target dataset (`SHADE_Arena` or `Agent_SafetyBench`) - **required**
- `--model`: Model for prompt generation (default: `Qwen/Qwen3-32B`)
- `--output_dir`: Directory containing Step 2 verification results (optional, auto-generated if not provided)
- `--temperature`: Sampling temperature (default: `0.6`)
- `--top_p`: Nucleus sampling parameter (default: `0.95`)
- `--batch_size`: Number of attacks to process simultaneously (default: `1`)
- `--region`: AWS region for model access (default: `us-east-1`)

### Step 4: Evaluate with Adaptive Planning

Evaluate agents against generated attacks using adaptive planning.

**Requirements**: Depends on model being evaluated

```bash
python -m STAC_gen.step_4_eval_adaptive_planning \
    --benchmark SHADE_Arena \
    --model_agent gpt-4.1 \
    --defense no_defense \
    --batch_size 512
```

**Parameters**:
- `--benchmark`: Target benchmark (`SHADE_Arena` or `Agent_SafetyBench`) - **required**
- `--model_planner`: Model for attack planning (default: `gpt-4.1`)
- `--model_judge`: Model for evaluation (default: `gpt-4.1`)
- `--model_agent`: Model to evaluate (default: `gpt-4.1`)
- `--defense`: Defense mechanism (`no_defense`, `failure_modes`, `summarization`, `reasoning`, `spotlighting`) (default: `no_defense`)
- `--temperature`: Sampling temperature (default: `0.0`)
- `--top_p`: Nucleus sampling parameter (default: `0.95`)
- `--batch_size`: Evaluation batch size (default: `1`)
- `--max_n_turns`: Maximum conversation turns (default: `3`)
- `--input_path`: Path to Step 3 outputs (optional, auto-generated if not provided)
- `--output_dir`: Output directory (optional, auto-generated if not provided)
- `--region`: AWS region for Bedrock API (default: `us-west-2`)
- `--n_gpus_per_model`: Number of GPUs per model (default: `8`)

---

## Understanding the Benchmark Data

### STAC Benchmark Structure

The benchmark (`data/STAC_benchmark_data.json`) contains **483 test cases**:
- Each test case includes a verified tool chain attack
- Test cases span multiple environments from SHADE-Arena and Agent-SafetyBench
- Each item contains `generation_config.dataset` field indicating its source benchmark

### Data Fields

```json
{
  "id": "unique-identifier",
  "generation_config": {
    "dataset": "SHADE_Arena",  // or "Agent_SafetyBench"
    "environment": "workspace",
    ...
  },
  "attack_plan": {
    "attack_goal": "...",
    "explanation": "...",
    "verified_tool_chain": [...]
  },
  "interaction_history": [...],
  ...
}
```

---

## Defense Mechanisms Explained

- **`no_defense`**: Baseline agent without additional safety measures
- **`failure_modes`**: Agent is made aware of potential failure modes and risks
- **`summarization`**: Agent summarizes user intent before taking actions
- **`reasoning`**: Agent employs harm-benefit reasoning before executing tools
- **`spotlighting`**: User prompts are marked with special characters for disambiguation from system prompts

---

## Important Notes

1. **GPU Requirements**:
   - STAC_gen Steps 3 and 4 (HF models) require GPU access
   - STAC_eval with HF models requires GPU access

2. **AWS Token Expiration**:
   - AWS session tokens expire every 12 hours
   - You'll need to refresh credentials and restart for long-running jobs

3. **Batch Size Guidelines**:
   - Start with batch_size=2 to verify the script works
   - Increase based on your resources (e.g., number of GPUs)

4. **Sequential Execution** (STAC_gen only):
   - Pipeline steps must be run in order
   - Each step depends on outputs from previous steps

5. **Model-Specific Requirements**:
   - Check that you have access to the model before running
   - Different models require different API keys/tokens

---

## Troubleshooting

### Common Issues

**Q: "KeyError: 'OPENAI_API_KEY'"**
```bash
# Make sure to export (not just set) the variable
export OPENAI_API_KEY="sk-your-key-here"

# Verify it's set
python -c "import os; print('Key found:', 'OPENAI_API_KEY' in os.environ)"
```

**Q: "Model not found" for HuggingFace**
```bash
# Ensure you have access to the model on HuggingFace
# Set your token
export HF_TOKEN="your-token-here"
```

**Q: "AWS credentials expired"**
```bash
# Refresh your AWS credentials
export AWS_ACCESS_KEY_ID="new-key-id"
export AWS_SECRET_ACCESS_KEY="new-secret"
export AWS_SESSION_TOKEN="new-token"
```

---

## Requirements

See `requirements.txt` for the full list of dependencies.

---

## Citation

If you use this code or the STAC benchmark in your research, please cite our work:

```bibtex
@misc{li2025stacinnocenttoolsform,
      title={STAC: When Innocent Tools Form Dangerous Chains to Jailbreak LLM Agents}, 
      author={Jing-Jing Li and Jianfeng He and Chao Shang and Devang Kulshreshtha and Xun Xian and Yi Zhang and Hang Su and Sandesh Swamy and Yanjun Qi},
      year={2025},
      eprint={2509.25624},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2509.25624}, 
}
```

---

## Security

See [SECURITY](SECURITY.md) and [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

---

## License

Our framework is constructed using data/scenarios from the following sources:

- [Agent-SafetyBench](https://github.com/thu-coai/Agent-SafetyBench/) – released under the MIT License
- [SHADE-Arena](https://github.com/safety-research/SHADE-Arena) – released under the MIT License

Accordingly, we release our framework under the **CC BY-NC 4.0 License**.
