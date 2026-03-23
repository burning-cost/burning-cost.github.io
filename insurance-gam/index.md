---
layout: page
title: "insurance-gam"
description: "Interpretable GAM toolkit for insurance pricing. EBM, Actuarial Neural Additive Model, and Pairwise Interaction Networks. Shape functions per rating factor with exact Shapley values."
permalink: /insurance-gam/
---

Interpretable GAM toolkit for insurance pricing. Three modelling approaches — Explainable Boosting Machine, Actuarial Neural Additive Model, and Pairwise Interaction Networks — that sit between a GLM and a black-box gradient booster. All interpretable, all exposure-aware.

**[View on GitHub](https://github.com/burning-cost/insurance-gam)** &middot; **[PyPI](https://pypi.org/project/insurance-gam/)**

---

## The problem

GLMs have been the industry standard for decades. They are interpretable, well-understood, and regulators like them. But they leave predictive power on the table — particularly on non-linear effects and interactions.

Gradient boosters capture non-linearities but produce black-box output that pricing committees cannot challenge and PRA SS1/23 validation requires explanation of. There is a design choice sitting in the middle: GAMs and their variants give shape functions per rating factor — exactly the transparency of a GLM — while capturing the non-linearities that a GLM misses.

The question is not whether interpretability is worth a Gini loss. In insurance pricing, a model you cannot explain does not go live. The question is whether you are leaving money on the table with a GLM that could be a GAM.

---

## Installation

```bash
pip install "insurance-gam[ebm]"     # EBM (requires interpret)
pip install "insurance-gam[neural]"  # ANAM and PIN (requires torch)
pip install "insurance-gam[all]"     # everything
```

---

## Quick start

```python
import numpy as np
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

rng = np.random.default_rng(42)
n = 2000

df = pl.DataFrame({
    "driver_age":   rng.integers(17, 75, n).astype(float),
    "vehicle_age":  rng.integers(0, 15, n).astype(float),
    "ncd_years":    rng.integers(0, 9, n).astype(float),
    "annual_miles": rng.integers(3000, 20000, n).astype(float),
})
exposure = rng.uniform(0.3, 1.0, n)
y = rng.poisson(np.exp(-2.5 - 0.12 * df["ncd_years"].to_numpy()) * exposure)

model = InsuranceEBM(loss="poisson", interactions="3x")
model.fit(df[:1600], y[:1600], exposure=exposure[:1600])

rt = RelativitiesTable(model)
print(rt.table("ncd_years"))
# shape_value  relativity
# 0.0          1.000
# 3.0          0.694
# 9.0          0.340

print(rt.summary())
```

---

## Key API reference

### `insurance_gam.ebm` — Explainable Boosting Machine

Wraps interpretML's `ExplainableBoostingRegressor` with insurance-specific tooling.

**`InsuranceEBM`**

```python
from insurance_gam.ebm import InsuranceEBM

InsuranceEBM(
    loss="poisson",         # poisson | gamma | tweedie
    interactions="3x",      # number of pairwise interactions to model (or "all")
    max_bins=256,           # binning resolution
)
```

- `.fit(X, y, exposure=None)` — exposure-aware fit
- `.predict(X, exposure=None)` — predicted pure premium
- `.explain_global()` — interpretML global explanation object
- `.monotonicity_constrain(feature, direction)` — enforce post-fit monotonicity

**`RelativitiesTable`**

Extracts GLM-style multiplicative relativities from an EBM.

```python
from insurance_gam.ebm import RelativitiesTable

rt = RelativitiesTable(fitted_ebm)
rt.table(feature)          # Polars DataFrame: shape_value | relativity
rt.summary()               # summary of all features sorted by contribution
rt.plot(feature)           # shape function plot
rt.to_excel("factors.xlsx") # export factor table for pricing committee
```

**`GLMComparison`**

Side-by-side EBM vs GLM comparison on the same data.

```python
from insurance_gam.ebm import GLMComparison

comp = GLMComparison(ebm=fitted_ebm, glm=fitted_glm)
comp.gini_comparison(X_test, y_test, exposure_test)
comp.factor_comparison(feature="driver_age")
```

### `insurance_gam.anam` — Actuarial Neural Additive Model

Neural Additive Model with actuarial extensions: exposure offset, Poisson/Gamma loss, SHAP-based exact Shapley values.

```python
from insurance_gam.anam import ANAM

model = ANAM(
    hidden_units=(64, 32),   # per-feature sub-network architecture
    loss="poisson",
    dropout=0.1,
)
model.fit(X_train, y_train, exposure=exposure_train,
          X_val=X_val, y_val=y_val, exposure_val=exposure_val)
model.predict(X_test, exposure=exposure_test)
model.feature_effect(feature="driver_age")   # shape function as numpy array
model.shapley_values(X_test)                 # exact Shapley values
```

### `insurance_gam.pin` — Pairwise Interaction Networks

Extends NAM with pairwise interaction terms. Each interaction is modelled by a dedicated sub-network, keeping the model interpretable while capturing interactions.

```python
from insurance_gam.pin import PINModel

model = PINModel(
    main_hidden=(64, 32),
    interaction_hidden=(32, 16),
    interactions=[("driver_age", "vehicle_age"), ("driver_age", "ncd_years")],
    loss="poisson",
)
model.fit(X_train, y_train, exposure=exposure_train)
model.interaction_effect("driver_age", "vehicle_age")  # 2D interaction surface
```

---

## Choosing between EBM, ANAM, and PIN

| | EBM | ANAM | PIN |
|--|-----|------|-----|
| Extra dependency | interpret | torch | torch |
| Training speed | Fast | Moderate | Moderate |
| Interactions | Automatic detection | None | User-specified |
| Exact Shapley | Yes | Yes | Yes |
| Monotonicity constraint | Post-fit | During training | During training |
| Factor table export | Yes | Yes | Requires aggregation |
| Best for | Starting point; GLM comparison | Clean additive structure | Known interaction pairs |

If you are starting out, use EBM. It trains fast, produces exact relativities, and you can hand the factor table to a pricing committee with the same explanation you would give for a GLM.

---

## Benchmark

Tested on 5,000 synthetic UK motor policies (Poisson DGP with known non-linear NCD and driver age effects).

| Model | Gini coefficient | Factor RMSE |
|-------|-----------------|-------------|
| GLM | 0.312 | 0.089 |
| EBM | **0.341** | **0.061** |
| ANAM | 0.335 | 0.067 |

EBM recovers the true non-linear NCD effect with 32% lower RMSE than GLM while maintaining full interpretability.

---

## Related blog posts

- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](https://burning-cost.github.io/2026/03/14/insurance-gam-interpretable-nonlinearity/)
- [The Governance Bottleneck for EBM Adoption](https://burning-cost.github.io/2026/03/16/the-governance-bottleneck-for-ebm-adoption/)
