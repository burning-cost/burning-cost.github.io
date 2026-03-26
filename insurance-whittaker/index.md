---
layout: page
title: "insurance-whittaker"
description: "Whittaker-Henderson smoothing for insurance pricing tables. REML lambda selection, Bayesian credible intervals, 1D/2D/Poisson variants. Polars-native."
permalink: /insurance-whittaker/
---

Whittaker-Henderson smoothing for insurance experience rating tables. 1D, 2D, and Poisson variants with automatic REML lambda selection and Bayesian credible intervals. The tool every UK pricing actuary is doing in Excel is now properly packaged in Python.

**[View on GitHub](https://github.com/burning-cost/insurance-whittaker)** &middot; **[PyPI](https://pypi.org/project/insurance-whittaker/)**

---

## The problem

Every UK motor or home pricing actuary smooths experience rating tables. You collect claims data, bin by age or vehicle group, and end up with noisy cell-level observations. Age 47 looks cheaper than age 46 for no reason other than random variation. You need a smooth curve that respects the underlying data but is not held hostage by each cell's noise.

Whittaker-Henderson smoothing is the actuarial standard tool. It is a penalised least-squares method with a mathematically principled approach to selecting how smooth the curve should be. Every UK pricing team does this. Most do it in Excel or SAS.

The mathematical problem:

```
minimise: sum_i w_i (y_i - theta_i)^2 + lambda * ||D^q theta||^2
```

where `w_i` are exposure weights, `D^q` is the q-th order difference operator, and `lambda` controls the smoothness penalty. Lambda is selected by maximising the restricted marginal likelihood (REML), which has a unique, well-defined maximum and avoids the overfitting tendency of GCV.

Benchmark: 57.2% MSE reduction versus raw rates on a synthetic UK motor age curve.

---

## Installation

```bash
pip install insurance-whittaker
pip install "insurance-whittaker[plot]"   # with matplotlib plotting
```

---

## Quick start

1-D age curve smoothing:

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

ages = np.arange(17, 80)
loss_ratios = 0.6 + 0.2 * np.exp(-(ages - 25) ** 2 / 200)
exposures = np.random.exponential(100, len(ages))

wh = WhittakerHenderson1D(order=2)
result = wh.fit(ages, loss_ratios, weights=exposures)

print(result.smoothed)         # smoothed loss ratios
print(result.lambda_)          # selected lambda (REML)
print(result.ci_lower)         # 90% credible interval lower
print(result.ci_upper)         # 90% credible interval upper
```

2-D cross-table smoothing (age x vehicle group):

```python
from insurance_whittaker import WhittakerHenderson2D

wh2d = WhittakerHenderson2D(order1=2, order2=2)
result = wh2d.fit(
    age_index=ages,
    group_index=vehicle_groups,
    values=loss_rate_matrix,
    weights=exposure_matrix,
)
```

---

## Key API reference

### `WhittakerHenderson1D`

1-D smoother for age curves, NCD scales, vehicle group factors, bonus-malus scales.

```python
WhittakerHenderson1D(
    order=2,                # difference order (1=linear, 2=quadratic penalty)
    method="reml",          # lambda selection: reml | gcv | aic | bic
    alpha=0.10,             # credible interval level (90% CI)
)
```

- `.fit(x, y, weights=None)` — fits smoother; returns `WHResult1D`
- `result.smoothed` — smoothed values (numpy array)
- `result.lambda_` — selected lambda
- `result.effective_df` — effective degrees of freedom
- `result.ci_lower`, `result.ci_upper` — Bayesian credible interval bounds
- `result.plot()` — if `[plot]` extra installed

### `WhittakerHenderson2D`

2-D smoother for cross-tables (age x vehicle group, age x claim-free years).

```python
WhittakerHenderson2D(
    order1=2,       # difference order for first dimension
    order2=2,       # difference order for second dimension
    method="reml",  # lambda selection
)
```

- `.fit(age_index, group_index, values, weights=None)` — fits 2-D smoother; returns `WHResult2D`
- `result.smoothed` — 2-D array of smoothed values
- `result.lambda1_`, `result.lambda2_` — selected lambdas per dimension

### `WhittakerHendersonPoisson`

Poisson PIRLS smoother for smoothing claim frequencies directly from count data.

```python
from insurance_whittaker import WhittakerHendersonPoisson

wh_pois = WhittakerHendersonPoisson(order=2)
result = wh_pois.fit(
    x=ages,
    counts=claim_counts,
    exposure=policy_counts,
)
# Poisson IRLS with Whittaker penalty — accounts for integer claim counts
# result.smoothed: smoothed frequency rates
# result.lambda_: selected lambda
```

### Result types

```python
from insurance_whittaker import WHResult1D, WHResult2D, WHResultPoisson
```

All result types have `.smoothed`, `.lambda_`, `.ci_lower`, `.ci_upper` attributes. `WHResult1D` and `WHResultPoisson` additionally have `.effective_df` and `.hat_matrix` (for influence diagnostics).

---

## Benchmark

| Metric | Raw rates | Whittaker-Henderson | Improvement |
|--------|-----------|--------------------:|-------------|
| MSE (age curve) | 0.0042 | 0.0018 | 57.2% |

Tested on a synthetic UK motor age curve with realistic exposure per age band. Lambda selected by REML.

---

## Related blog posts

- [Your Rating Table Smoothing Is Wrong](https://burning-cost.github.io/2026/03/18/your-rating-table-smoothing-is-wrong/)
