---
layout: page
title: "insurance-credibility"
description: "Bühlmann-Straub group credibility and Bayesian individual policy experience rating for UK non-life insurance pricing."
permalink: /insurance-credibility/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-credibility"
pypi_package: "insurance-credibility"
---

Credibility models for UK non-life insurance pricing: Bühlmann-Straub group credibility for scheme and large account pricing, and Bayesian experience rating at individual policy level.

**[View on GitHub](https://github.com/burning-cost/insurance-credibility)** &middot; **[PyPI](https://pypi.org/project/insurance-credibility/)**

---

## The problem

Two problems that look similar but need different tools:

**Group credibility (schemes, large accounts):** A fleet scheme has 3 years of loss history. How much should you weight it against the market rate? Too much and you are pricing noise. Too little and you leave money on the table. The Bühlmann-Straub formula gives the optimal credibility weight — it depends on the scheme's own variance, the portfolio variance, and the amount of exposure observed.

**Individual policy experience rating:** A commercial motor policy has been with you for 5 years with no claims. How much is that worth relative to the a priori GLM rate? The answer depends on portfolio heterogeneity, exposure, and claim frequency. A static NCD table ignores all three.

Benchmark: 6.8% MAE improvement on thin schemes vs raw experience.

---

## Installation

```bash
pip install insurance-credibility
```

Optional deep attention model (requires PyTorch):

```bash
pip install "insurance-credibility[attention]"
```

---

## Quick start

Group-level credibility for scheme pricing:

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

df = pl.DataFrame({
    "scheme":    ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":      [2022, 2023, 2024, 2022, 2023, 2024, 2022, 2023, 2024],
    "loss_rate": [0.12, 0.09, 0.11, 0.25, 0.28, 0.22, 0.08, 0.07, 0.09],
    "exposure":  [120.0, 135.0, 140.0, 45.0, 50.0, 48.0, 300.0, 310.0, 320.0],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.z_)          # credibility factors per scheme (Z_i)
print(bs.k_)          # Bühlmann's k: noise-to-signal ratio
print(bs.premiums_)   # credibility-blended premium per scheme
```

Individual policy experience rating:

```python
from insurance_credibility import ClaimsHistory, StaticCredibilityModel

histories = [
    ClaimsHistory("POL001", periods=[1, 2, 3], claim_counts=[0, 1, 0],
                  exposures=[1.0, 1.0, 0.8], prior_premium=400.0),
    ClaimsHistory("POL002", periods=[1, 2, 3], claim_counts=[2, 1, 2],
                  exposures=[1.0, 1.0, 1.0], prior_premium=400.0),
]

model = StaticCredibilityModel()
model.fit(histories)

cf = model.predict(histories[0])
posterior_premium = histories[0].prior_premium * cf
```

---

## Key API reference

### `BuhlmannStraub`

Group credibility using the Bühlmann-Straub (1970) method of moments estimator.

```python
BuhlmannStraub()
```

- `.fit(df, group_col, period_col, loss_col, weight_col)` — fits structural parameters
- `.z_` — Polars DataFrame `[group, Z]`; Z_i = w_i / (w_i + k)
- `.k_` — float; Bühlmann's k = within-group variance / between-group variance
- `.mu_` — grand mean (portfolio rate)
- `.premiums_` — Polars DataFrame `[group, credibility_premium]`
- `.summary()` — prints a governance-ready summary table

### `HierarchicalBuhlmannStraub`

Hierarchical credibility for nested structures (scheme → book, sector → district → area). Implements Jewell (1975).

```python
from insurance_credibility import HierarchicalBuhlmannStraub

hbs = HierarchicalBuhlmannStraub(levels=["sector", "district", "area"])
hbs.fit(df, loss_col="loss_rate", weight_col="exposure")
hbs.premiums_    # credibility-blended premium at each level
```

### `ClaimsHistory`

Data container for an individual policy's experience history.

```python
ClaimsHistory(
    policy_id,      # str
    periods,        # list of period identifiers
    claim_counts,   # list of observed claim counts per period
    exposures,      # list of exposure weights per period
    prior_premium,  # float, a priori GLM pure premium
)
```

### Experience rating models

**`StaticCredibilityModel`** — Bühlmann-Straub at individual level. Standard static credibility factor from policy history.

**`DynamicPoissonGammaModel`** — State-space model with time-varying risk level. More accurate for long-tenure policies where risk may have evolved.

**`SurrogateModel`** — Importance sampling surrogate for computational efficiency. Approximates the posterior when claim history is long.

**`DeepAttentionModel`** — Transformer-based experience rating following Wüthrich (2024). Requires `[attention]` extra.

All models implement `.fit(histories)` and `.predict(history)` returning a credibility factor.

### Calibration

```python
from insurance_credibility import (
    balance_calibrate,      # calibrate to portfolio mean
    apply_calibration,      # apply calibration result to predictions
    balance_report,         # governance summary of calibration
    credibility_factor,     # compute Z_i given structural params
    posterior_premium,      # posterior mean premium given history
)
```

---

## Benchmark

| Method | MAE vs true rate |
|--------|-----------------|
| Raw experience | 0.051 |
| Bühlmann-Straub credibility | **0.047** |
| Improvement | 6.8% |

Tested on synthetic thin schemes (n < 100 exposure per scheme per year). The improvement grows as scheme size falls — credibility is most valuable for the smallest, noisiest segments.

---

## Related blog posts

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](https://burning-cost.github.io/2026/02/19/buhlmann-straub-credibility-in-python/)
- [When Credibility Meets CatBoost](https://burning-cost.github.io/2026/02/28/when-credibility-meets-catboost/)
- [Credibility Transformer: Deep Attention for Insurance Experience Rating](https://burning-cost.github.io/2026/03/11/credibility-transformer/)
