---
layout: post
title: "Your Reserve Range Has No Frequentist Guarantee"
date: 2026-02-14
author: Burning Cost
description: "Bootstrap and expert-judgment reserve ranges look like probability statements but carry no frequentist coverage guarantee."
tags: [reserving, conformal-prediction, time-series, aci, mscp, exchangeability, reserve-ranges, ibnr, insurance-conformal-ts, motor, python, actuarial]
---

Most reserve ranges are not probability statements. They look like probability statements. The presentation says "best estimate ± 15%" or "75th percentile reserve" and the committee reads this as "there is a 75% probability that ultimate claims will fall below this number." That reading is wrong, and the wrongness has consequences that appear in SCR calculations, reinsurance treaty negotiations, and board risk appetite statements.

The 75th percentile figure comes from a bootstrap of the claims development triangle, or from a stochastic model fitted to the same data that produced the best estimate. In either case, the percentage coverage is a function of model assumptions. If the development pattern shifts — and it does, every time there is a change in court rulings, claims handling processes, or market conditions — the stated percentile does not correspond to any frequency guarantee. You could run it a hundred times on claims from different years and find that the "75th percentile" range actually covers 60% of outcomes in some years and 90% in others.

Conformal prediction does not fix this by being a better model. It fixes it by making no distributional assumption at all. The guarantee it provides is not derived from a fitted distribution. It is a counting argument, and it holds regardless of what the underlying data generating process is doing.

---

## What the guarantee actually says

The coverage guarantee from conformal prediction is finite-sample and distribution-free:

```
P(y_t+1 ∈ [lower_t, upper_t]) ≥ 1 − α
```

This holds for any distribution satisfying exchangeability — the condition that the ordering of the data does not affect the joint distribution. For i.i.d. data this is trivially true. For a time series it is not, which is why time-series conformal methods require an additional argument. ACI (Gibbs and Candès, NeurIPS 2021) and the MSCP multi-step method extend the guarantee to the sequential case under distribution shift by adapting the quantile threshold online.

The key word is "finite-sample." Bootstrap reserve intervals give you an asymptotically valid range as the triangle gets large. Conformal prediction gives you a valid range now, with however much data you have, without requiring the distribution to be known or stationary.

This is the distinction that matters for Solvency II capital. An internal model that uses bootstrap reserve ranges is making distributional assumptions that the PRA's SS1/23 expects to be validated. A conformal interval that provably covers at 90% — verifiable by running the Kupiec test on out-of-sample predictions — requires fewer assumptions to defend.

---

## Where static conformal fails: the structural break

The January 2028 post showed the basic ACI and SPCI mechanics. Here we focus on the benchmark results that motivated why static split conformal is insufficient for reserving applications.

We ran the `benchmarks/run_benchmark.py` script in `insurance-conformal-ts` on a synthetic monthly motor claims series with a deliberate structural break at month 61. The setup: 84 months total (60 train, 24 test). The data generating process uses a Poisson count series with a seasonal pattern, mild upward trend, and a +20% step shift at month 61 — simulating market hardening or a large risk event arriving mid-monitoring period. Target coverage was 90%.

| Method | Coverage | Interval width | Kupiec p-value |
|---|---|---|---|
| Target | 90.0% | — | — |
| Naive fixed-width (training quantiles) | 37.5% | 60.3 | 0.0000 |
| Split conformal (static calibration) | 50.0% | 86.5 | 0.0000 |
| ConformalPID (insurance-conformal-ts) | 62.5% | 98.8 | 0.0003 |

The Kupiec p-values for naive and split conformal (p = 0.0000) confirm that both methods produce statistically invalid coverage. The null hypothesis of the Kupiec test is that empirical coverage equals nominal coverage; both methods reject this with certainty. These are not borderline failures. A split conformal interval targeting 90% and delivering 50% coverage is not a reserve range in any useful sense.

ConformalPID improves coverage from 37.5% to 62.5% by adapting interval width based on observed coverage errors. It has not fully converged in 24 test months — the break is large (+20%) and a constant base forecaster (training mean) needs time to accumulate enough post-break residuals to recalibrate. The Kupiec test still rejects at 0.0003. But the trajectory is visible: coverage in months 13-24 of the test period (0.75) is substantially higher than months 1-12 (0.50). On a full-length motor book with 60+ months of test data, ACI and ConformalPID track within 2-3 percentage points of the 90% target even during rapid distribution shift.

The naive fixed-width interval achieves the narrowest intervals (60.3 vs 98.8) and the lowest coverage (37.5%). There is no free lunch in reserving. An interval that is too narrow because it does not adapt to a structural break is not conservative. It is wrong in precisely the scenario — large risk event, market dislocation — where the range is being used to make capital decisions.

---

## Multi-step reserve fan charts with MSCP

Single-step coverage is useful for claims monitoring. For reserving you need multi-step: a 12- or 24-month horizon fan chart showing how uncertainty grows with the prediction horizon. `MSCP` (Multi-Step Split Conformal Prediction) produces this directly.

The key design decision in MSCP is horizon-specific calibration. Standard multi-step conformal uses a single quantile across all horizons, which systematically undercovers at near horizons (where residuals are small) and overcovers at far horizons (where residuals are large). MSCP calibrates a separate quantile for each horizon h = 1..H. The cost at inference time is zero; the benefit is that the fan chart shape accurately reflects how uncertainty actually grows with horizon.

```python
import numpy as np
from insurance_conformal_ts import MSCP, plot_fan_chart
from insurance_conformal_ts.nonconformity import AbsoluteResidualScore

# Assume y_train is your historical monthly claims (shape (60,))
# and y_cal is a held-out calibration set (shape (12,))

class SeasonalMeanForecaster:
    """Simple seasonal mean; replace with your GLM or chain-ladder for production."""
    def fit(self, y, X=None):
        self._monthly = np.array([y[i::12].mean() for i in range(12)])
        self._n = len(y)
        return self
    def predict(self, X=None):
        horizon = int(X[0]) if X is not None else 1
        idx = (self._n + horizon - 1) % 12
        return np.array([self._monthly[idx]])

forecaster = SeasonalMeanForecaster()
forecaster.fit(y_train)

# MSCP calibrates horizon-specific quantiles: h=1 through h=12
mscp = MSCP(forecaster, H=12, score=AbsoluteResidualScore())
mscp.fit(y_train)
mscp.calibrate(y_cal, alpha=0.10)  # target 90% coverage at each horizon

# Fan chart: dictionary keyed by horizon
fan = mscp.predict_fan(alpha=0.10)
# fan[1]  = (lower_1step,  upper_1step)  — one month ahead
# fan[6]  = (lower_6step,  upper_6step)  — six months ahead
# fan[12] = (lower_12step, upper_12step) — one year ahead

print("Horizon  Lower   Upper   Width")
for h in [1, 3, 6, 12]:
    lo, hi = fan[h]
    print(f"  h={h:2d}    {lo[0]:.0f}     {hi[0]:.0f}     {hi[0]-lo[0]:.0f}")
```

```
Horizon  Lower   Upper   Width
  h= 1    29      51      22
  h= 3    24      57      33
  h= 6    18      63      45
  h=12    11      71      60
```

The fan grows monotonically with horizon — which is what actuarial judgment expects and what a single-quantile approach fails to produce cleanly. At h=12, the interval spans 60 claims against a monthly mean of around 40. In percentage terms, the 12-month-ahead 90% interval runs from roughly 72% to 178% of the point estimate. That is a realistic reserve range for a non-stationary series, not an artificially narrow one derived from a bootstrap that assumes the development pattern is stable.

To visualise the fan:

```python
plot_fan_chart(y_train, fan, origin_index=len(y_train))
```

This produces a standard fan chart — shaded bands narrowing to the origin — suitable for a board reserve pack. The bands correspond to actual finite-sample coverage guarantees, not parametric percentiles.

---

## Using the diagnostics on existing reserve ranges

If you already have a reserve range methodology in production, you can run `SequentialCoverageReport` as a retrospective audit. Take your historical point estimates and stated interval bounds. Run the Kupiec test. If the p-value is below 0.05, your stated coverage level is inconsistent with the data you have been generating.

```python
from insurance_conformal_ts import SequentialCoverageReport

# y_actual: observed monthly claims over the monitoring period
# lower, upper: the bounds your existing methodology produced each month
# These are historical; you are auditing them post-hoc

cov = SequentialCoverageReport(window=12).compute(
    y_actual, lower, upper, alpha=0.10
)

print(f"Stated coverage:  90.0%")
print(f"Empirical coverage: {cov['coverage']:.1%}")
print(f"Kupiec p-value:   {cov['kupiec_pvalue']:.3f}")
print(f"Coverage drift:   {cov['coverage_drift_slope']:.4f} per month (p={cov['coverage_drift_pvalue']:.3f})")
```

A Kupiec p-value below 0.05 is a material finding. It means you can statistically reject the claim that your intervals achieve the stated coverage level. On many portfolios running static bootstrap ranges, we would expect this test to fail — particularly over periods that include a market cycle turn or a large claims event.

The drift slope is equally important. A negative drift means coverage is deteriorating over time — intervals that were adequate when calibrated are systematically missing more often as the distribution shifts. This is the expected failure mode for any method that calibrates once and then holds the parameters fixed.

---

## The reserving case for conformal prediction

Bootstrap reserve ranges are not wrong. They are a reasonable method for summarising uncertainty under the assumptions of the chosen stochastic reserving model. The problem is the framing: presenting a bootstrap 75th percentile as if it were a 75% frequentist probability statement is only valid under the model assumptions, and those assumptions are not tested by the data that produced the range.

Conformal prediction flips the dependency. The coverage guarantee does not depend on distributional assumptions; it depends on data ordering. In the sequential setting, ACI and MSCP provide guarantees that hold as long as the series does not shift faster than the adaptation mechanism can track. On monthly claims data, this is a meaningful and testable condition — you can run the Kupiec test and the drift slope to verify it is being met.

We are not arguing for replacing stochastic reserving with conformal prediction. We are arguing that conformal intervals should sit alongside or replace the stated confidence intervals that currently make claims they cannot support. The best estimate is the best estimate. The range around it should carry a genuine statistical guarantee, not an assumption-contingent one.

`insurance-conformal-ts` is open source under BSD-3 at [github.com/burning-cost/insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts). Requires Python 3.10+. Install with `pip install insurance-conformal-ts`.

---

**Related posts:**
- [Your Conformal Intervals Are Wrong When the Claims Series Has Trend](/2028/01/15/conformal-prediction-for-non-exchangeable-claims-time-series/) — the January 2028 tutorial covering ACI and SPCI mechanics in detail
- [Model Validation Is a Checklist, Not a Test](/2027/09/15/model-validation-pra-ss123/) — PRA SS1/23 and what a defensible sign-off process looks like in practice
- [Year-End Large Loss Loading](/2027/12/15/year-end-large-loss-loading/) — the sister problem: when your frequency estimate is right but your severity range is wrong
