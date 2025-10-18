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

## Storage Management

### What to Keep Permanently

**Gradient checkpoints** (~2GB each for 1B models, ~16GB for 8B models):
- These are computed ONCE per dataset and reused for all m values
- Computing gradients is the slowest step (30min-3hrs depending on nsamples)
- Store in: `./permanent_storage`

**Result CSVs** (<1MB total):
- Your actual experimental data
- Store in: `./permanent_storage/`

### What Can Be Temporary

**Chunked gradients** (same size as checkpoints):
- Generated quickly (<5min) from gradient checkpoints via `chunk_gradient.py`
- Can regenerate as needed
- Store in: `./tmp/chunks/` or delete after evaluation

**Masked models** (~2GB each for 1B models):
- One per m value tested
- Evaluate immediately then delete
- Use `tempfile.TemporaryDirectory()` or `./tmp/masked_models/`

### Storage Optimization

For a full experiment sweep (6 datasets × 6 m values):
- **Without optimization**: ~84GB (1B model) or ~672GB (8B model)
- **With optimization**: ~12GB permanent + temporary scratch space

**Recommended approach:**
```python
# Keep gradients permanent
GRAD_DIR = "./permanent/gradients"  

# Use temp for everything else
CHUNKS_DIR = "./tmp/chunks"
MASKED_DIR = "./tmp/masked_models"

# Evaluate and delete models immediately after
with tempfile.TemporaryDirectory() as tmp_dir:
    masked_path = apply_mask_and_save(..., out_dir=tmp_dir)
    results = evaluate(masked_path)
    save_results(results)  # To permanent location
    # tmp_dir auto-deleted here
```

## Supported Models

The codebase supports LLaMA-family architectures (including LLaMA, Vicuna, Qwen, DeepSeek, **Gemma**), Mistral, and OPT. Models are auto-detected by name, but all LLaMA-family models share the same layer structure:
- Layers: `model.layers`
- Attention: `self_attn.{q,k,v,o}_proj`
- MLP: `mlp.{gate,up,down}_proj`

**Note:** Gemma models (e.g., `google/gemma-2b-it`, `google/gemma-2b-pt`) use RoPE and have identical layer naming to LLaMA, so they work out of the box as of the latest code changes.

## Common Commands

### 1. Generate gradients (SLOW - compute once, keep permanently)

```bash
# ToM dataset (full sequence, supervise only last token)
python create_gradient.py \
  --model google/gemma-2b-it \
  --dataset tom \
  --data_path tom_training_data.json \
  --nsamples 100 \
  --seqlen 0 \
  --cache_dir ./cache \
  --out ./gradients/tom_grad  # PERMANENT storage

# C4 dataset (random 128-token windows, supervise all tokens)
python create_gradient.py \
  --model google/gemma-2b-it \
  --dataset c4 \
  --nsamples 100 \
  --seqlen 128 \
  --cache_dir ./cache \
  --out ./gradients/c4_grad  # PERMANENT storage
```

### 2. Chunk gradients (FAST - can regenerate)

```bash
python chunk_gradient.py \
  --model ./gradients/tom_grad \
  --output_path /tmp/chunks/tom \
  --cache_dir ./cache

python chunk_gradient.py \
  --model ./gradients/c4_grad \
  --output_path /tmp/chunks/c4 \
  --cache_dir ./cache
```

### 3. Run evaluation sweep (MEDIUM - reuses gradients)

```bash
python ToM_and_perplexity_evaluation.py \
  --model google/gemma-2b-it \
  --grad_tom_chunks /tmp/chunks/tom \
  --grad_c4_chunks /tmp/chunks/c4 \
  --tom_tasks ToM_tasks.py \
  --out_dir /tmp/eval \
  --cache_dir ./cache \
  --tensor_parallel_size 1 \
  --max_model_len 1024 \
  --batch_size 64 \
  --reps 5 \
  --m_start 0.0 --m_end 5e-5 --m_step 2e-6

# Copy only CSVs to permanent storage
cp /tmp/eval/*.csv ./results/
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
- For ToM dataset: supervises only the **last non-pad token** per sample (~100 tokens total for 100 samples)
- For C4: random contiguous windows with full supervision (~12,800 tokens total for 100 samples)
- **Important:** The paper does NOT normalize by number of supervised tokens, creating a ~128× gradient magnitude difference
- Saves gradients by overwriting weights in model checkpoint

### Masking Logic (`ToM_and_perplexity_evaluation.py`)

The mask identifies parameters where:
- ToM gradient is in top-m percentile (`mask1`)
- C4 gradient is NOT in top-m percentile (`~mask2`)
- Final mask: `mask = mask1 & (~mask2)`

Perturbation replaces masked weights with the mean of non-masked weights (when `--scale 0.0`).

**Note:** The m value is independent of gradient computation - you can try different m values by rerunning evaluation with the same gradient checkpoints.

### ToM Tasks

Three task types (loaded from `ToM_tasks.py` or JSON with keys `s1`/`s2`/`s3`):
- **S1 (unexpected contents)**: e.g., chocolate box contains popcorn
- **S2 (unexpected transfer)**: e.g., keys moved while protagonist is away
- **S3 (irony)**: detect sarcastic/ironic remarks

Each task type includes false-belief and control variants (correct label, informed protagonist, etc.).

## Methodological Considerations

### Token Supervision Imbalance

The original paper creates an imbalance in gradient signals:
- **C4**: All 128 tokens supervised per sample → ~12,800 gradient updates
- **ToM**: Only last token supervised per sample → ~100 gradient updates

This means C4 gradients have ~128× more signal, which may make the subtraction `mask1 & (~mask2)` overly aggressive.

**Potential improvements to test:**
1. **Per-token normalization**: Divide FIM by number of supervised tokens
2. **Equal token budgets**: Use equal total supervised tokens across datasets
3. **Contrastive pairs**: Replace C4 with ToM control conditions (correct label, informed protagonist) to isolate ToM-specific reasoning

### Training Data Diversity

The paper uses 100 training samples to compute gradients. Consider:
- **More samples** (1000+) for more stable FIM estimates
- **Diverse scenarios** across all task types (S1, S2, S3) rather than narrow distributions
- **Multiple prompt formats** to test robustness

## Experiment Variations

When running experiments with different models or hyperparameters:
- Use consistent `--nsamples` and `--seqlen` across gradient runs for fair comparison
- The `--m_start/end/step` parameters define the sweep over sparsity levels (proportion of parameters)
- Increase `--reps` for more stable ToM accuracy estimates (default: 5)
- For larger models, adjust `--tensor_parallel_size` and `--max_model_len`
- **Gradient checkpoints can be reused** across all m values - only need to compute once per dataset

## Important Notes

- **Model context limits**: `create_gradient.py` respects `max_position_embeddings` and keeps the tail of long sequences to preserve the supervised last token
- **ToM evaluation uses vLLM**: ensure sufficient GPU memory for inference
- **Perplexity**: evaluated on WikiText-2 test set with 2048-token sequences
- The paper focuses on RoPE-based models; non-RoPE models (e.g., Jamba) show different sensitivity patterns
- **Time estimates** (1B model): Gradient computation 30min-3hrs depending on nsamples; evaluation per m value ~40min; total experiment ~24-37hrs