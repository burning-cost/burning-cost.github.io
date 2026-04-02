---
layout: post
title: "ModelMonitor: Calibration Testing That Actually Tells You What To Do"
date: 2026-04-02
categories: [model-monitoring, insurance-pricing]
tags: [model-monitoring, calibration, gini, drift, murphy-decomposition, insurance-monitoring, actuarial, arXiv-2510.04556, pra-ss123]
description: "insurance-monitoring v1.0.0 adds ModelMonitor with check_gmcb and check_lmcb — separate tests for global and local calibration drift, wired into a three-way REDEPLOY/RECALIBRATE/REFIT decision."
author: burning-cost
---

Most monitoring setups tell you when a model has gone wrong. Fewer tell you how wrong, and almost none tell you what to do about it. The distinction matters: recalibrating a model that needs a refit is governance theatre, and refitting a model that only needed a balance correction wastes six weeks of pricing actuary time.

[insurance-monitoring v1.0.0](https://pypi.org/project/insurance-monitoring/) ships `ModelMonitor` — an integrated class that runs three statistical tests in sequence and returns a typed decision: `REDEPLOY`, `RECALIBRATE`, or `REFIT`. The decision is based on the framework from Brauer, Menzel & Wüthrich (arXiv:2510.04556, published December 2025), which gives a mathematically clean answer to when each action is appropriate.

---

## The Murphy decomposition and why GMCB/LMCB matter

The framework rests on a deviance decomposition that dates to the Murphy score literature. For a response Y and predictions hat_mu, total deviance decomposes as:

```
S(Y, hat_mu) = UNC - DSC + MCB
```

Where UNC is the uncertainty baseline (deviance of the grand mean), DSC is discrimination — the improvement from the model's ranking ability — and MCB is miscalibration, the cost of pricing at wrong levels. The paper then splits MCB further:

```
MCB = GMCB + LMCB
```

GMCB (global mean calibration bias) captures a single global level shift — the model is consistently overpricing or underpricing by a multiplicative factor. This is fixable without a refit: apply a balance correction (a GLM step on the link scale) and redeploy.

LMCB (local mean calibration bias) captures structural miscalibration across rating cells. Even after applying the global balance correction, some cohorts are mispriced. This requires a model refit.

The distinction is not just conceptual. Under the Equality Act 2010 and FCA Consumer Duty, cohort-level mispricing is a compliance risk. A model with clean global calibration can still have LMCB drift — if, say, young drivers' claim rates have shifted following telematics adoption, while the rest of the book is stable. LMCB catches this; an overall MCB test misses it.

---

## check_gmcb and check_lmcb

The two new functions implement Algorithm 4a and 4b from the paper, each with a parametric bootstrap:

```python
from insurance_monitoring.calibration import check_gmcb, check_lmcb

# On your monitoring period data
gmcb = check_gmcb(y_new, y_hat, exposure,
                  distribution='poisson',
                  bootstrap_n=999,
                  significance_level=0.32)

print(gmcb.gmcb_score)     # observed GMCB value (>= 0)
print(gmcb.p_value)        # bootstrap p-value under H_0: GMCB = 0
print(gmcb.is_significant) # True if p_value < significance_level
print(gmcb.balance_factor) # global alpha = sum(w*y) / sum(w*y_hat)
```

The bootstrap is parametric: variance at each observation is estimated by isotonic regression of squared residuals on the predictions, then new samples are drawn from the appropriate EDF distribution. This is the procedure from Algorithm 1 of the paper, applied to GMCB and LMCB separately rather than to the total MCB.

Both functions support `distribution='poisson'`, `'gamma'`, `'tweedie'`, and `'normal'`, matching the EDF family your model was estimated under.

One default worth noting: `significance_level=0.32`. This is not a typo. The paper demonstrates (Figure 8) that the conventional 0.05 threshold has high Type II error for realistic insurance drift magnitudes — you miss genuine drift. The 0.32 level (one-sigma equivalent) means you flag earlier and review more often, which is appropriate when a false alarm costs a model review and a missed detection costs mis-pricing. If your pricing governance cycle already has a quarterly review cadence, you want the test to be sensitive.

---

## ModelMonitor: the integrated workflow

`ModelMonitor` wraps the Gini ranking drift test (Algorithm 3) and the GMCB/LMCB tests (Algorithm 4) into a single class:

```python
from insurance_monitoring import ModelMonitor
import numpy as np

# Fit on your holdout from the last training period
monitor = ModelMonitor(
    distribution='poisson',
    n_bootstrap=500,
    alpha_gini=0.32,
    alpha_global=0.32,
    alpha_local=0.32,
    random_state=0,
)
monitor.fit(y_ref, y_hat_ref, exposure_ref)

# Run on the new monitoring period
result = monitor.test(y_new, y_hat_new, exposure_new)

print(result.decision)         # 'REDEPLOY' | 'RECALIBRATE' | 'REFIT'
print(result.gini_p)           # p-value for Gini ranking drift
print(result.gmcb_p)           # p-value for global calibration drift
print(result.lmcb_p)           # p-value for local calibration drift
print(result.summary())        # governance paragraph for model validation report
```

The decision logic follows Section 3.1 of the paper precisely:

- If nothing is significant: **REDEPLOY**. The model is holding.
- If only GMCB is significant (Gini and LMCB are not): **RECALIBRATE**. Global level shift only; apply balance correction, no refit needed.
- If Gini or LMCB is significant: **REFIT**. Structural drift. Balance correction is not enough.

The `result.summary()` string produces a governance-ready paragraph with all p-values, z-statistics, and the decision reason — suitable for pasting into a PRA SS1/23 model validation report without further editing.

---

## The virtual vs real drift distinction

Before running any test, it is worth understanding what the tests can and cannot tell you.

Virtual drift is a shift in the covariate distribution — your portfolio composition changes, but the true pricing function has not. An influx of EVs changes the mix of vehicle types, but the claim rate model for each vehicle type is still correct. Virtual drift does not require model action. Both GMCB and LMCB can be nonzero under virtual drift even when the underlying model is correct, because the portfolio weighting changes.

Real concept drift is a shift in the true conditional distribution F_{Y|X} — the model has become wrong. This is what GMCB and LMCB are designed to detect.

The framework cannot automatically separate the two — that requires your knowledge of what changed in the portfolio. But the three-test sequence gives you a structured starting point: if only GMCB fires, suspect claims inflation or a systematic rate change rather than a structural model failure.

---

## A practical pitfall: time-split ETL data

The paper calls this out explicitly in Section 3.3 and we reproduced the effect in testing. If your claims ETL produces one row per claim rather than one row per policyholder with aggregated claim counts, the Gini drops by roughly 4% and deviance inflates by a factor of three. Both effects create spurious test signals.

`ModelMonitor.fit()` and `.test()` both emit a `UserWarning` when the median exposure is below 0.05 — a strong signal that data has not been pre-aggregated. Always aggregate to policyholder-year level before passing data to any monitoring call.

---

## Where this sits in the monitoring stack

`ModelMonitor` is the fourth distinct monitoring component in the library:

| Component | Version added | What it tests |
|---|---|---|
| `PricingDriftMonitor` | v0.10.0 | Portfolio composition shifts and premium adequacy |
| `CalibrationCUSUM` | v0.10.0 | Sequential detection of calibration drift with CUSUM control |
| `ConformedControlChart` | v0.11.0 | Conformal prediction intervals for individual predictions |
| `ModelMonitor` | v1.0.0 | Gini ranking drift + GMCB/LMCB calibration decision |

These address different timescales and different questions. `CalibrationCUSUM` is better for real-time sequential monitoring where you want earliest possible detection; `ModelMonitor` is better for structured periodic review with a formal governance decision.

The library is available on [PyPI](https://pypi.org/project/insurance-monitoring/) and the source is on [GitHub](https://github.com/insurance-monitoring/insurance-monitoring). The paper is at [arXiv:2510.04556](https://arxiv.org/abs/2510.04556).

---

## What we would improve

The current implementation runs all three bootstrap tests regardless of intermediate results. For production use with large datasets and governance reporting timelines, you may want to short-circuit: if the Gini test fires and you are already committed to a refit, there is no practical reason to wait for the LMCB bootstrap to complete. We may add an `early_stop` option in v1.1.0.

The other open question is drift mode: the paper recommends using only the most recent training year as the reference holdout when you expect sudden drift, and computing per-year Gini baselines for gradual drift. `ModelMonitor.fit()` takes whatever reference data you pass it — the right holdout strategy is yours to determine based on your knowledge of how your portfolio drifts. Documentation for this is sparse; we will expand it.

**Paper:** [arXiv:2510.04556](https://arxiv.org/abs/2510.04556) | **Library:** [insurance-monitoring on PyPI](https://pypi.org/project/insurance-monitoring/) | **GitHub:** [insurance-monitoring](https://github.com/insurance-monitoring/insurance-monitoring)
