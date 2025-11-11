# ToM Parameter Identification Notebooks

This directory contains Jupyter notebooks implementing methods to identify Theory-of-Mind (ToM) sensitive parameters in large language models.

## Notebooks

### 01_paper_original_methodology.ipynb

Implements the original paper methodology:
- **Method**: Compare ToM dataset gradients vs C4 (generic text) gradients
- **Dataset**: `tom_training_data.json` (640 samples)
- **Pros**: Follows published research exactly
- **Cons**: Token supervision imbalance (~128×), poor content control

**Use this notebook to**:
- Reproduce paper results
- Understand the baseline methodology
- Compare with improved methods

### 02_contrast_pair_methodology.ipynb

Implements enhanced methodology using contrast pairs:
- **Method 1**: High-ToM vs Low-ToM questions (same scenarios, different reasoning)
- **Method 2**: Self-reference vs Other-reference (perspective-taking)
- **Datasets**:
  - `tom_dataset/simpletom_contrast_pairs.json` (3,441 pairs)
  - `self_other_dataset/self_other.json` (400 pairs)
- **Pros**: Better content control, balanced token counts, more targeted
- **Cons**: Untested, may identify fewer parameters

**Use this notebook to**:
- Identify more precise ToM-specific parameters
- Test if contrast pairs work better than ToM vs C4
- Analyze parameter overlap between methods

## Quick Start

### Prerequisites

```bash
# GPU with ≥16GB VRAM
# Python 3.10+
# CUDA-compatible PyTorch
pip install transformers datasets torch vllm pandas matplotlib
```

### Running Notebooks

1. **Configure settings** at the top of each notebook:
   - `BASE_MODEL`: Choose your model (default: `meta-llama/Llama-3.2-1B`)
   - `NSAMPLES`: Number of samples for gradient computation
   - `M_VALUES`: Sparsity levels to test

2. **Uncomment execution cells** as you go (marked with `# Uncomment to run`)

3. **Storage management**:
   - Gradients go to `./permanent_storage/gradients/` (KEEP these)
   - Results go to `./permanent_storage/evaluation_results/` (KEEP these)
   - Chunks go to `./tmp/chunks/` (can delete and regenerate)

### Expected Runtime (1B model)

- **Gradient computation**: 30min - 3hrs (depending on nsamples)
- **Chunking**: <5 minutes
- **Evaluation per m value**: ~40 minutes
- **Full experiment**: 24-37 hours

## Dataset Comparison

| Dataset | Type | Size | Token Balance | Content Control |
|---------|------|------|---------------|-----------------|
| ToM vs C4 (paper) | Categorical | 100 + 100 | Poor (~128× imbalance) | Poor (different domains) |
| SimpleTOM Contrast | Contrast Pairs | 3,441 pairs | Good | Excellent |
| Self/Other | Contrast Pairs | 400 pairs | Excellent | Good |

## Epistemic Status & Predictions

### SimpleTOM Contrast (High Confidence on Quality)

**Methodological superiority**: ✓ Strong
- Same scenarios, minimal confounds
- Balanced token supervision
- Isolates mental state reasoning vs factual reasoning

**Practical effectiveness**: ⚠ Moderate uncertainty
- **Best case** (30%): Clean separation, reliable ToM dampening
- **Likely case** (50%): Partial dampening with some collateral effects
- **Worst case** (20%): Too distributed, no clear effect or breaks general ability

### Self/Other Contrast (Moderate Confidence)

**Methodological quality**: ✓ Good
- Minimal confounds (just pronoun changes)
- Perfect token balance

**Practical effectiveness**: ⚠ Higher uncertainty
- May identify perspective-taking mechanisms
- Less clear connection to standard ToM tasks
- May have broader effects on language use

## Validation Strategy

After masking parameters, test interactively:

```python
# Load masked model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./permanent_storage/evaluation_results/simpletom_contrast/models/m_2e-05")
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Test false belief
prompt = """There's a box labeled 'cookies' but it actually contains books.
Mary reads the label but doesn't open the box.
What does Mary think is in the box?"""

# Compare with baseline model
```

## Storage Requirements

For 1B model:
- **Gradient checkpoints**: ~4GB (2GB × 2 conditions)
- **Chunked gradients**: ~4GB (temporary, can delete)
- **Masked models**: ~2GB per m value (temporary, can delete)
- **Results CSVs**: <1MB (KEEP)

**Recommended**: ~12GB permanent + temporary scratch space

## Key Variables in Notebooks

All major configuration variables are in CAPS at the top of cells:

- `BASE_MODEL`: HuggingFace model ID
- `NSAMPLES`: Samples for gradient computation
- `M_VALUES`: List of sparsity thresholds
- `EVAL_REPS`: Repetitions for ToM evaluation
- `CACHE_DIR`: HuggingFace cache
- `PERMANENT_STORAGE`: Where to save gradients and results
- `TMP_DIR`: Temporary files

## Troubleshooting

### Out of memory during gradient computation
- Reduce `NSAMPLES`
- Use smaller model
- Reduce `SEQLEN` for C4

### vLLM fails during evaluation
- Reduce `MAX_MODEL_LEN`
- Reduce `BATCH_SIZE`
- Increase `TENSOR_PARALLEL_SIZE` if multi-GPU

### Gradients already exist
- Delete checkpoint directories to recompute
- Or use existing checkpoints (they're reusable)

## Testing

Run validation tests:

```bash
python test_datasets.py
```

This validates:
- All datasets have correct format
- Required scripts exist
- Notebooks are present
- Dataset balance is reasonable

## Next Steps

1. **Run Notebook 1** to establish baseline (optional)
2. **Run Notebook 2** with SimpleTOM contrast (recommended)
3. **Compare results** between methods
4. **Validate interactively** with masked models
5. **Iterate**: Try different m values, masking strategies, or dataset sizes

## Questions?

- See `CLAUDE.md` for detailed pipeline documentation
- See `full_paper.md` for the original paper
- See inline epistemic status notes in notebooks for confidence levels
