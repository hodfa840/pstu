# Not All Secrets Are Equal: Type-Aware Unlearning for Language Model Secret Removal

**Hoda Fakhar**

Per-Secret-Type Unlearning (PSTU) — a training-free framework that removes memorized secrets from LLMs by adapting unlearning intensity to the distinct resistance levels of different secret types.

> Submitted to ECML PKDD 2026.

---

## Key Results

| Model | Method | Memorized ↓ | Mean Exposure ↓ | PPL ↓ | ΔPPL |
|-------|--------|:-----------:|:---------------:|:-----:|:----:|
| Pythia-1.4B | Best Baseline (NPO) | **0** | 0.68 | 34.39 | +53.9% |
| Pythia-1.4B | **PSTU (Ours)** | **0** | **0.03** | **22.62** | **+1.3%** |
| Pythia-2.8B | Best Baseline (SimNPO) | **0** | 0.47 | 82.24 | +329.5% |
| Pythia-2.8B | **PSTU (Ours)** | **0** | **0.03** | **19.61** | **+2.4%** |
| Pythia-6.9B | All 5 Baselines | *Failed*† | — | — | — |
| Pythia-6.9B | **PSTU-Trim (Ours)** | **0** | **0.06** | **17.96** | **+3.2%** |
| Llama-3.1-8B | All 5 Baselines | *Failed*† | — | — | — |
| Llama-3.1-8B | **PSTU-Trim (Ours)** | **0** | **0.11** | **13.87** | **+14.4%** |

†Exhaustive Bayesian search (30 trials per method) found no configuration that simultaneously removes secrets and preserves utility.

## Method

PSTU operates in weight space without any gradient-based training:

1. **Task vector:** `τ = θ_infected − θ_clean`
2. **Per-type saliency:** compute gradient saliency separately for each secret type to identify which parameters encode which category of secret
3. **Adaptive scaling:** `α(θ_i) = α_base + α_boost · Σ_t w_t · Ĝ_t(θ_i)`
4. **Weight subtraction:** `θ' = θ − α ⊙ τ`

For 7B+ models, **PSTU-Trim** denoises the task vector by zeroing entries below a magnitude quantile before subtraction (reduces ΔPPL from +18% to +3.2% on Pythia-6.9B).

Runs in 5–17 seconds on a single A100.

## Repository Structure

```
├── notebooks/
│   └── pstu_demo.ipynb           # Demo notebook (Pythia-70M) with paper figures
├── pstu_code/
│   ├── pstu/                     # Core PSTU library
│   │   ├── method.py             # apply_pstu(), saliency, PSTU-Trim
│   │   ├── evaluation.py         # Carlini exposure, WikiText-2 PPL
│   │   └── hyperopt.py           # Two-phase Optuna search
│   ├── baselines/                # GradAscent, GradDiff, NPO, SimNPO, RMU
│   └── scripts/                  # CLI entry points
├── script/
│   └── pstu_comprehensive.py     # Full-scale PSTU pipeline with Optuna
├── data/
│   └── neurips/                  # 175 synthetic secrets, 25 types, 100 decoys each
└── results/                      # Experiment outputs
```

## Usage

```bash
# Interactive notebook (Pythia-70M demo, single GPU)
jupyter notebook notebooks/pstu_demo.ipynb

# Full-scale hyperparameter search (Pythia-1.4B, ~30 min on 1x A100)
python script/pstu_comprehensive.py --model pythia-1.4b --n-trials 500

# With PSTU-Trim for larger models
python script/pstu_comprehensive.py --model pythia-6.9b-gentle --n-trials 500 --trim
```

## Evaluation Metrics

- **Carlini Exposure:** `log₂(N) − log₂(rank + 1)`, where `rank` is how many decoys rank higher than the true secret. Exposure ≈ 0 means forgotten; exposure = log₂(N) means fully memorized.
- **WikiText-2 PPL:** perplexity on held-out text, measuring general language modeling quality post-unlearning.

## Citation

```bibtex
@inproceedings{fakhar2026pstu,
  title     = {Not All Secrets Are Equal: Type-Aware Unlearning for Language Model Secret Removal},
  author    = {Fakhar, Hoda},
  booktitle = {ECML PKDD},
  year      = {2026}
}
```
