# Not All Secrets Are Equal: Type-Aware Unlearning for Language Model Secret Removal

**Hoda Fakhar**

Per-Secret-Type Unlearning (PSTU) — a training-free framework that removes memorized secrets from LLMs by adapting unlearning intensity to the distinct resistance levels of different secret types.

> [arXiv preprint coming soon]  
> Paper source: [hodfa840/ECML](https://github.com/hodfa840/ECML)

---

## Key Results

### Secret Removal (Tables 1–2)

| Model | Method | Mem. ↓ | Exp. ↓ | PPL ↓ | ΔPPL |
|-------|--------|:------:|:------:|:-----:|:----:|
| Pythia-1.4B | GA | 20 | 0.72 | 79.06 | +253.9% |
| Pythia-1.4B | GD | **0** | 0.14 | 49.02 | +119.4% |
| Pythia-1.4B | NPO | **0** | 0.68 | 34.39 | +53.9% |
| Pythia-1.4B | SimNPO | **0** | 0.51 | 61.70 | +176.2% |
| Pythia-1.4B | RMU | 1 | 0.23 | 90.94 | +307.1% |
| Pythia-1.4B | **PSTU** | **0** | **0.03** | **22.62** | **+1.3%** |
| | | | | | |
| Pythia-6.9B | All 5 baselines | — | — | — | *Destroyed* |
| Pythia-6.9B | **PSTU-Trim** | **0** | **0.06** | **17.96** | **+3.2%** |
| | | | | | |
| Llama-3.1-8B | All 5 baselines | — | — | — | *Destroyed* |
| Llama-3.1-8B | **PSTU-Trim** | **0** | **0.11** | **13.87** | **+14.4%** |

### LUME Benchmark — OLMo (Table 3)

| Model | Method | Forget QA ↓ | ROUGE-L ↓ | PPL ↓ | ΔPPL |
|-------|--------|:-----------:|:---------:|:-----:|:----:|
| OLMo-1B | GA | 0% | 0.000 | >10³⁰ | *Destroyed* |
| OLMo-1B | NPO | 100% | 0.685 | 110.19 | *No effect* |
| OLMo-1B | **PSTU** | **0%** | **0.105** | **9.73** | **+0.9%** |
| | | | | | |
| OLMo-7B | GA | 0% | 0.000 | >10³⁴ | *Destroyed* |
| OLMo-7B | NPO | 94.6% | 0.727 | 2,109 | *No effect* |
| OLMo-7B | **PSTU-Trim** | **0%** | **0.088** | **8.27** | **−1.7%** |

PSTU is the only method that achieves **0/175 memorized** (or 0% forget QA) at every scale. No baseline finds a viable configuration on 7B+ models.

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
│   └── pstu_demo.ipynb           # Demo notebook with pre-rendered figures
├── pstu_code/
│   ├── pstu/                     # Core PSTU library
│   │   ├── method.py             # apply_pstu(), saliency, PSTU-Trim
│   │   ├── evaluation.py         # Carlini exposure, WikiText-2 PPL
│   │   └── hyperopt.py           # Two-phase Optuna search
│   ├── baselines/                # GradAscent, GradDiff, NPO, SimNPO, RMU
│   └── scripts/                  # CLI entry points
└── data/                         # Dataset on HF: Hodfa71/pstu-synthetic-secrets
```

## Usage

```bash
# Interactive notebook (Pythia-70M demo, single GPU)
jupyter notebook notebooks/pstu_demo.ipynb

# Full-scale hyperparameter search (Pythia-1.4B, ~30 min on 1x A100)
python pstu_code/scripts/run_pstu.py --model pythia-1.4b --n-trials 500

# With PSTU-Trim for larger models
python pstu_code/scripts/run_pstu.py --model pythia-6.9b-gentle --n-trials 500 --trim
```

## Dataset

The synthetic secrets benchmark (175 secrets, 25 types, 100 decoys each) is hosted on Hugging Face:

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("Hodfa71/pstu-synthetic-secrets", "secrets_train.jsonl", repo_type="dataset")
secrets = [json.loads(line) for line in open(path)]
```

All data is synthetically generated — no real credentials or PII.
See the [dataset card](https://huggingface.co/datasets/Hodfa71/pstu-synthetic-secrets) for details.

## Evaluation Metrics

- **Carlini Exposure:** `log₂(N) − log₂(rank + 1)`, where `rank` is how many decoys rank higher than the true secret. Exposure ≈ 0 means forgotten; exposure = log₂(N) means fully memorized.
- **WikiText-2 PPL:** perplexity on held-out text, measuring general language modeling quality post-unlearning.

## Citation

```bibtex
@article{fakhar2026pstu,
  title   = {Not All Secrets Are Equal: Type-Aware Unlearning for Language Model Secret Removal},
  author  = {Fakhar, Hoda},
  journal = {arXiv preprint},
  year    = {2026}
}
```
