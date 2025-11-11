# Quick Start Guide

## TL;DR

Run the enhanced contrast pair methodology (recommended):

```bash
# 1. Open Jupyter
jupyter notebook 02_contrast_pair_methodology.ipynb

# 2. In notebook, set these variables:
BASE_MODEL = "meta-llama/Llama-3.2-1B"  # or your preferred model
NSAMPLES = 500  # More samples = better but slower
M_VALUES = [0.0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5]  # Sparsity levels

# 3. Run cells in order, uncommenting execution commands
# 4. Wait 24-37 hours for full pipeline
# 5. Check results in: ./permanent_storage/evaluation_results/simpletom_contrast/
```

## What This Does

1. **Computes gradients** on high-ToM vs low-ToM questions (same scenarios, different reasoning)
2. **Identifies parameters** that are sensitive to mental state reasoning specifically
3. **Perturbs these parameters** and measures ToM accuracy + perplexity
4. **Outputs**: Which parameters matter for ToM, how much dampening is possible

## Why Use Contrast Pairs Instead of Paper Method?

| Aspect | Paper (ToM vs C4) | Contrast Pairs (Recommended) |
|--------|-------------------|------------------------------|
| Content control | Poor (different domains) | Excellent (same scenarios) |
| Token balance | Very poor (~128× imbalance) | Good (balanced) |
| Specificity | Generic text vs ToM | Mental state vs factual |
| Predicted separation | Noisy | Cleaner |
| Predicted dampening | May work | Better chance |

## Recommended Workflow

### Phase 1: Test with Small Model (4-8 hours)
```python
BASE_MODEL = "meta-llama/Llama-3.2-1B"
NSAMPLES = 100  # Small sample for quick test
M_VALUES = [0.0, 2e-5, 5e-5]  # Just 3 m values
```

**Goal**: Verify pipeline works, get rough estimates

### Phase 2: Full Run on Target Model (24-37 hours)
```python
BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"  # Your actual model
NSAMPLES = 500  # Better gradient estimates
M_VALUES = [0.0, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 1e-4]  # Full sweep
```

**Goal**: Get publication-quality results

### Phase 3: Validate Interactively
```python
# Load masked model and test manually
model_path = "./permanent_storage/evaluation_results/simpletom_contrast/models/m_2e-05"
# Test false belief understanding in conversation
```

**Goal**: Confirm dampening works in practice

## Key Outputs

After running, check:

1. **ToM summary**: `./permanent_storage/evaluation_results/simpletom_contrast/tom_summary.csv`
   - Shows accuracy on each ToM task type vs m value

2. **Perplexity**: `./permanent_storage/evaluation_results/simpletom_contrast/perplexity_results.csv`
   - Shows WikiText-2 perplexity vs m value

3. **Visualizations**: `./permanent_storage/contrast_comparison.png`
   - Plots both metrics

4. **Masked models**: `./permanent_storage/evaluation_results/simpletom_contrast/models/m_*/`
   - Actual perturbed models you can load and interact with

## Expected Results

**If it works well**:
- ToM accuracy drops 20-40% at m=2e-5 to 5e-5
- Perplexity increases <10%
- Masked model fails false belief tasks interactively

**If it works partially**:
- ToM accuracy drops 10-25%
- Perplexity increases 5-20%
- Inconsistent dampening

**If it doesn't work**:
- No ToM effect OR massive perplexity increase
- Try higher m values or different masking strategy

## Epistemic Status: Will It Work?

**Best case** (30% probability): Clean dampening, minimal collateral damage
**Likely** (50% probability): Partial dampening with some side effects
**Worst case** (20% probability): No effect or breaks general ability

**Why uncertainty?**
- ToM may be too distributed across parameters
- Evaluation tasks may differ from training data distribution
- LLMs may have redundant ToM circuits

## Storage Cleanup

After experiments, you can delete:
```bash
rm -rf ./tmp/  # Chunked gradients and temp files (~4GB)
rm -rf ./permanent_storage/evaluation_results/*/models/  # Masked models (~2GB per m value)
```

**Keep**:
- `./permanent_storage/gradients/` - Expensive to recompute
- `./permanent_storage/evaluation_results/*/*.csv` - Your actual results

## Comparison with Paper Method

Want to run both and compare? Use Notebook 1 for paper method:

```bash
jupyter notebook 01_paper_original_methodology.ipynb
```

Then compare results between:
- `./permanent_storage/evaluation_results/paper_methodology/`
- `./permanent_storage/evaluation_results/simpletom_contrast/`

## Troubleshooting

**"CUDA out of memory"**
- Use smaller model
- Reduce `NSAMPLES`
- Reduce `BATCH_SIZE` or `MAX_MODEL_LEN`

**"Gradient checkpoint missing"**
- Make sure gradients computed successfully (Step 1)
- Check `./permanent_storage/gradients/`

**"Results look random"**
- May need more `NSAMPLES` (try 1000+)
- May need to try different m values
- May indicate ToM is too distributed

## Next Steps After Getting Results

1. **Analyze parameter overlap** (built into Notebook 2)
2. **Try self/other contrast** for comparison
3. **Test different masking strategies** (multiplicative scaling instead of mean)
4. **Combine multiple contrasts** (SimpleTOM + Self/Other)
5. **Try activation steering** if parameter perturbation doesn't work well

## Citation

If using these notebooks:
```bibtex
@article{original_paper,
  title={How large language models encode theory-of-mind: a study on sparse parameter patterns},
  author={[Original authors]},
  journal={[Journal]},
  year={2024}
}

% Note: Contrast pair methodology is an enhancement, not in original paper
```
