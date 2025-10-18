# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This codebase reproduces and extends the paper "How large language models encode theory-of-mind: a study on sparse parameter patterns" (see `full_paper.md`). The pipeline identifies ToM-sensitive parameters in LLMs by computing Fisher information via squared gradients on ToM vs. C4 datasets, then evaluates how perturbing these parameters affects Theory-of-Mind capabilities and perplexity.

## Pipeline Workflow

The evaluation pipeline consists of 4 sequential steps:

1. **`create_gradient.py`** - Compute squared gradients (Fisher approximation) and save as model checkpoints
2. **`chunk_gradient.py`** - Split gradient checkpoints into per-layer chunks for efficient masking
3. **`ToM_and_perplexity_evaluation.py`** - Build masked models across m values, evaluate ToM tasks and perplexity
4. **`summarize.py`** - Aggregate ToM results into summary CSV

## Supported Models

The codebase supports LLaMA-family architectures (including LLaMA, Vicuna, Qwen, DeepSeek, **Gemma**), Mistral, and OPT. Models are auto-detected by name, but all LLaMA-family models share the same layer structure:

- Layers: `model.layers`
- Attention: `self_attn.{q,k,v,o}_proj`
- MLP: `mlp.{gate,up,down}_proj`

**Note:** Gemma models (e.g., `google/gemma-2b-it`, `google/gemma-2b-pt`) use RoPE and have identical layer naming to LLaMA, so they work out of the box as of the latest code changes.

## Common Commands

### 1. Generate gradients
```bash
# ToM dataset (full sequence, supervise only last token)
python create_gradient.py \
  --model google/gemma-2b-it \
  --dataset tom \
  --data_path tom_training_data.json \
  --nsamples 100 \
  --seqlen 0 \
  --cache_dir ./cache \
  --out ./gradients/tom_grad

# C4 dataset (random 128-token windows)
python create_gradient.py \
  --model google/gemma-2b-it \
  --dataset c4 \
  --nsamples 100 \
  --seqlen 128 \
  --cache_dir ./cache \
  --out ./gradients/c4_grad
```

### 2. Chunk gradients
```bash
python chunk_gradient.py \
  --model ./gradients/tom_grad \
  --output_path ./chunks/tom \
  --cache_dir ./cache

python chunk_gradient.py \
  --model ./gradients/c4_grad \
  --output_path ./chunks/c4 \
  --cache_dir ./cache
```

### 3. Run evaluation sweep
```bash
python ToM_and_perplexity_evaluation.py \
  --model google/gemma-2b-it \
  --grad_tom_chunks ./chunks/tom \
  --grad_c4_chunks ./chunks/c4 \
  --tom_tasks ToM_tasks.py \
  --out_dir ./results \
  --cache_dir ./cache \
  --tensor_parallel_size 1 \
  --max_model_len 1024 \
  --batch_size 64 \
  --reps 5 \
  --m_start 0.0 --m_end 5e-5 --m_step 2e-6
```

### 4. Summarize results
```bash
python summarize.py \
  --root ./results/tom \
  --reps 5 \
  --out_csv ./results/tom_summary.csv
```

## Key Architecture Details

### Gradient Computation (`create_gradient.py`)
- Registers hooks to square gradients: `g → g²`
- For ToM dataset: supervises only the **last non-pad token** per sample
- For C4: random contiguous windows with full supervision
- Saves gradients by overwriting weights in model checkpoint

### Masking Logic (`ToM_and_perplexity_evaluation.py`)
The mask identifies parameters where:
- ToM gradient is in top-m percentile (`mask1`)
- C4 gradient is NOT in top-m percentile (`~mask2`)
- Final mask: `mask = mask1 & (~mask2)`

Perturbation replaces masked weights with the mean of non-masked weights (when `--scale 0.0`).

### ToM Tasks
Three task types (loaded from `ToM_tasks.py` or JSON with keys `s1`/`s2`/`s3`):
- **S1 (unexpected contents)**: e.g., chocolate box contains popcorn
- **S2 (unexpected transfer)**: e.g., keys moved while protagonist is away
- **S3 (irony)**: detect sarcastic/ironic remarks

Each task type includes false-belief and control variants (correct label, informed protagonist, etc.).

## Experiment Variations

When running experiments with different models or hyperparameters:
- Use consistent `--nsamples` and `--seqlen` across gradient runs for fair comparison
- The `--m_start/end/step` parameters define the sweep over sparsity levels (proportion of parameters)
- Increase `--reps` for more stable ToM accuracy estimates (default: 5)
- For larger models, adjust `--tensor_parallel_size` and `--max_model_len`

## Important Notes

- **Model context limits**: `create_gradient.py` respects `max_position_embeddings` and keeps the tail of long sequences to preserve the supervised last token
- **ToM evaluation uses vLLM**: ensure sufficient GPU memory for inference
- **Perplexity**: evaluated on WikiText-2 test set with 2048-token sequences
- The paper focuses on RoPE-based models; non-RoPE models (e.g., Jamba) show different sensitivity patterns
