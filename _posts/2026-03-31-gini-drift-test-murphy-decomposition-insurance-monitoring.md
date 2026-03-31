---
layout: post
title: "Gini Drift Tests and Murphy Decomposition: What Insurance Model Monitoring Was Missing"
seo_title: "Gini Hypothesis Test and Murphy Score Decomposition for Insurance Pricing Model Monitoring"
date: 2026-03-31
categories: [monitoring]
tags: [gini, model-monitoring, murphy-decomposition, calibration, discrimination, bootstrap, hypothesis-testing, Consumer-Duty, insurance-monitoring, PSI, CSI, recalibrate, refit, python, arXiv-2510-04556, Wüthrich, CUSUM, GLM, concept-drift]
description: "Brauer, Menzel & Wüthrich (arXiv:2510.04556) give us two things we have been missing: a formal hypothesis test for Gini drift and a Murphy score decomposition that tells you whether to recalibrate or refit. We explain both, show the Python API from insurance-monitoring, and explain why PSI alone does not answer the question."
author: burning-cost
math: true
---

Most UK pricing teams already track the Gini coefficient on their holdout set. They track it quarterly, compare it to the training value, and flag it if it falls materially. What "materially" means is usually left to the judgement of whoever is presenting at the monitoring committee that month.

That is not a monitoring framework. That is a vibe.

A paper published in October 2025 by Brauer, Menzel and Wüthrich — "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing" (arXiv:2510.04556) — replaces the vibe with three precise components: an asymptotic hypothesis test for Gini drift with a bootstrap variance estimator, a Murphy score decomposition that separates calibration loss from discrimination loss, and a decision tree that routes the output of both into a binary answer: recalibrate or refit.

We have implemented all three in [`insurance-monitoring`](/insurance-monitoring/). This post explains what they are, why they are better than what most teams have now, and what is still genuinely missing.

---

## Why PSI and CSI are not enough

Population Stability Index and Characteristic Stability Index are the monitoring industry's comfort blanket. They are ubiquitous, easy to compute, and answer exactly one question: has the distribution of the input features shifted?

That question matters, but it is not the question your pricing committee needs answered. What they need to know is:

1. Has the model's ability to rank risks degraded?
2. If so, is it a calibration problem (wrong price level) or a discrimination problem (wrong price ordering)?
3. What should we do about it?

PSI answers none of these. A large PSI tells you the population has shifted. It does not tell you whether the model is still correctly ranking the new population. You can have enormous PSI — a portfolio that has shifted from urban-heavy to rural-heavy over 18 months — while the model continues to rank risks as well as it ever did on both sub-populations. Conversely, you can have stable PSI and a quietly degrading Gini as the underlying frequency-severity relationship drifts.

The other standard metric — Actual/Expected ratio — has a different limitation. It tells you whether the model is calibrated at the portfolio level. A model with a portfolio A/E of 1.02 looks fine on that measure. It might be 0.75 in the £800–1,200 band and 1.35 in the £200–400 band, with the errors cancelling in the aggregate. PSI and A/E, used alone, miss this entirely.

The Gini coefficient is more powerful because it is rank-based: it measures how well the model separates high-risk from low-risk policies, independent of the absolute price level. But the raw Gini number, without a variance estimate and a hypothesis test, is not actionable. When the Gini drops from 0.45 to 0.43, you do not know whether that is sampling noise or real signal without knowing the standard error.

That is the first gap the paper fills.

---

## The asymptotic Gini test

The theoretical foundation is a CLT result (Theorem 1 in the paper). Under mild regularity conditions — finite first moments, continuous marginal distributions — the sample Gini is root-n consistent and asymptotically normal:

$$\sqrt{n}\left(\hat{G}_n - G\right) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

The variance $\sigma^2$ exists and is positive, but there is no closed-form expression for it. The paper establishes existence via Hadamard differentiability of the Gini functional. The only practical way to estimate $\sigma^2$ is with the bootstrap.

**Algorithm 2: non-parametric bootstrap variance**

For a monitoring dataset $\mathcal{T} = \{(y_i, \hat{\mu}_i, v_i)\}_{i=1}^n$ with $B$ bootstrap replicates:

```python
import numpy as np
from insurance_monitoring.discrimination import gini_coefficient

def bootstrap_gini_variance(actual, predicted, exposure=None, n_bootstrap=500, seed=42):
    rng = np.random.default_rng(seed)
    n = len(actual)
    boot_ginis = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g = gini_coefficient(actual[idx], predicted[idx],
                             exposure[idx] if exposure is not None else None)
        boot_ginis.append(g)
    boot_ginis = np.array(boot_ginis)
    return boot_ginis.mean(), boot_ginis.std(ddof=1)
```

For monitoring windows with $n > 20{,}000$, the paper recommends capping the bootstrap at a random subsample of 20,000 to bound the $O(n \log n)$ per-replicate cost. The variance estimate is negligibly affected: bootstrap standard errors converge by around $n = 5{,}000$.

**Algorithm 3: one-sample monitoring test**

The production-realistic design. At training time you store a single scalar: the training Gini. At monitoring time you have new observations but not the raw training data. The test compares the new Gini against the stored training value:

```python
from insurance_monitoring.discrimination import gini_drift_test_onesample

result = gini_drift_test_onesample(
    training_gini=0.452,       # scalar stored at model deployment
    monitor_actual=monitor_claims,
    monitor_predicted=monitor_mu_hat,
    monitor_exposure=monitor_exposure,
    n_bootstrap=500,
    alpha=0.32,                # one-sigma threshold (see below)
)

print(result.z_statistic)     # z = (G_new - G_old) / SE_boot
print(result.p_value)         # two-sided
print(result.significant)     # True if p < alpha
```

Under $H_0$ (no real concept drift):

$$z = \frac{\hat{G}_{\text{new}} - G_{\text{old}}}{\hat{\sigma}_{\text{boot}}} \sim \mathcal{N}(0, 1)$$

The paper also specifies a two-sample form for when both reference and monitoring datasets are available — relevant for A/B validation or when raw historical data is retained. Both are implemented in [`insurance-monitoring`](/insurance-monitoring/), in `gini_drift.py` (class interface) and `discrimination.py` (functional interface).

**On the alpha threshold**

The paper's recommendation here is notable, and we agree with it: do not use $\alpha = 0.05$ for routine monitoring. The asymmetric loss argument is straightforward. In a monitoring context, missing a genuine 3% Gini decline for six months is far more costly than investigating a false alarm once. The paper proposes the one-sigma rule — $\alpha = 0.32$ — which flags anything outside one standard deviation of the null.

Under $\alpha = 0.05$ you need a roughly 1.96-sigma shift before acting. Under $\alpha = 0.32$ you act on a 1-sigma shift. Given that real model decay tends to be gradual, the earlier signal is worth taking. We implement a two-threshold scheme: amber at $p < 0.32$ (investigate), red at $p < 0.10$ (escalate).

For small monitoring books — under 500 policies per period — the normal approximation deteriorates. At that scale, report bootstrap percentile intervals rather than p-values, and resist the temptation to run the test at all if $n < 200$.

---

## Murphy score decomposition: calibration vs discrimination

Knowing the Gini has drifted is useful. Knowing *why* it has drifted determines what you should do about it.

The Murphy decomposition (Section 3 of the paper, drawing on Murphy 1973 from meteorological forecasting) partitions the total deviance loss into three components. For the weighted normalised deviance over an exponential dispersion family (Poisson, Gamma, Tweedie, Normal):

$$S(Y, \hat{\mu}, V) = \text{UNC}(Y, V) - \text{DSC}(Y, \hat{\mu}, V) + \text{MCB}(Y, \hat{\mu}, V)$$

**UNC (Uncertainty)** is the deviance of the grand-mean model — the intercept-only benchmark. It is a property of the data, not the model. It measures how hard the prediction problem is.

**DSC (Discrimination)** measures how much better than grand-mean the model ranks risks, after stripping out calibration error. It is computed using the isotonic recalibration of $\hat{\mu}$:

$$\text{DSC} = S(Y, \bar{y}, V) - S(Y, \hat{\mu}_{\text{rc}}, V)$$

where $\hat{\mu}_{\text{rc}}$ is the isotonic regression of $Y$ on $\hat{\mu}$. A model with no ranking ability whatsoever has $\text{DSC} = 0$.

**MCB (Miscalibration)** is the excess deviance due to wrong price levels:

$$\text{MCB} = S(Y, \hat{\mu}, V) - S(Y, \hat{\mu}_{\text{rc}}, V)$$

A perfectly calibrated model has $\text{MCB} = 0$.

MCB further decomposes into two parts:

$$\text{MCB} = \text{GMCB} + \text{LMCB}$$

**GMCB (Global Miscalibration)** is the portion removable by multiplying all predictions by a scalar — the balance correction factor $\alpha = \sum V_i Y_i / \sum V_i \hat{\mu}_i$. This is cheap to fix: a single portfolio-level rate adjustment, no model rebuild required.

**LMCB (Local Miscalibration)** is the residual after balance correction. The model has the right aggregate level but the wrong shape of relativities. This requires either isotonic recalibration (a post-processing correction) or a full model refit.

```python
from insurance_monitoring.calibration._murphy import murphy_decomposition

result = murphy_decomposition(
    y=monitor_claims,
    y_hat=monitor_mu_hat,
    exposure=monitor_exposure,
    distribution="poisson",
)

print(result.uncertainty)        # UNC
print(result.discrimination)     # DSC
print(result.miscalibration)     # MCB = GMCB + LMCB
print(result.global_mcb)         # GMCB — fixable with rate change
print(result.local_mcb)          # LMCB — requires model work
print(result.verdict)            # "OK" / "RECALIBRATE" / "REFIT"
```

The verdict logic:

| Condition | Meaning | Response |
|---|---|---|
| MCB/UNC < 1% | Well-calibrated | No action on calibration |
| GMCB > LMCB | Level shift, shape intact | Recalibrate — multiply by $\alpha$ |
| LMCB ≥ GMCB | Wrong shape, not just level | Refit required |
| DSC/UNC < 1% | No discrimination remaining | Refit (model is essentially useless) |

This distinction between recalibrate and refit matters enormously in practice. A recalibration is a pricing decision — it can be executed by underwriting in days and documented as a rate change. A refit is a model rebuild — for a GLM-based personal lines model, that is a minimum of 8–12 weeks including data extraction, feature engineering, sign-off, and system deployment. Mistaking an LMCB problem for a GMCB problem costs you several months. Murphy decomposition is what stops you making that mistake.

A note on what DSC is not: DSC is deviance-based discrimination, and it is fundamentally different from the Gini coefficient. Gini is rank-based — it measures the Lorenz curve area and is invariant to any monotone transformation of the score. DSC depends on the absolute scale of predictions, not just their ordering. Both are needed: Gini detects rank-order degradation; DSC quantifies how much discrimination contributes to the overall deviance decomposition. A model that is monotonically recalibrated preserves Gini perfectly but changes DSC.

---

## The decision framework

In `MonitoringReport`, the two tests combine into a single recommendation:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=train_claims,
    reference_predicted=train_mu_hat,
    current_actual=monitor_claims,
    current_predicted=monitor_mu_hat,
    exposure=monitor_exposure,
    murphy_distribution="poisson",
    gini_bootstrap=True,
)

print(report.recommendation)
# NO_ACTION / RECALIBRATE / REFIT / INVESTIGATE / MONITOR_CLOSELY

print(report.results_["gini"])
# gini, delta, z_statistic, p_value, ci_lower, ci_upper, band

print(report.results_["murphy"])
# uncertainty, discrimination, miscalibration, global_mcb, local_mcb, verdict
```

The routing logic:

```
Murphy verdict = REFIT   OR   Gini band = red    -->  REFIT
Murphy verdict = RECAL   OR   A/E band = red     -->  RECALIBRATE
Gini = amber  AND  A/E = red                     -->  INVESTIGATE
A/E = amber   OR   Gini = amber                  -->  MONITOR_CLOSELY
None of the above                                -->  NO_ACTION
```

If `murphy_distribution=None`, the report falls back to the Gini/A/E heuristic without Murphy. For any book where you can assume a Poisson frequency model, you should pass the distribution.

---

## What insurance-monitoring already has — and what it does not

To be clear about implementation status. As of v0.9.2, the library already has PSI/CSI covariate shift monitoring, A/E ratio tests with exact Poisson confidence intervals, CUSUM and Page-Hinkley sequential tests, and PIT-based calibration monitoring. The Gini drift test and Murphy decomposition are also fully implemented, as described above.

What the paper describes that we have not yet built:

**A bootstrap p-value on LMCB.** The current verdict treats LMCB as a deterministic magnitude: if LMCB ≥ GMCB, it recommends REFIT. For large books this is fine. For small monitoring windows — under 2,000 policies — LMCB can be nonzero from sampling noise alone. A bootstrap test on whether LMCB is significantly greater than zero would prevent the framework from over-triggering REFIT recommendations on noisy data.

**The E-hat correction in the one-sample test.** The paper's Algorithm 3 uses the bootstrap mean of the monitoring sample (not the stored training Gini) as the null hypothesis centre. This matters when the portfolio mix has shifted: if the book has drifted from urban-heavy to rural-heavy, the expected Gini changes even if the model is perfect on both populations. Our current implementation uses the stored training Gini as the null centre, which is conservative — it will flag virtual drift as Gini drift in this scenario. The fix is a one-line substitution and is on the short list.

**An explicit virtual/real drift classification.** The paper defines three cases: PSI high with stable Gini (virtual drift — population has shifted, model still valid on its target), Gini degraded with stable PSI (real concept drift — the underlying loss relationship has changed), and both degraded (mixed — worst case). The current `MonitoringReport` results contain enough information to make this classification, but it is not yet surfaced as a named property.

---

## What this means for Consumer Duty

FCA Consumer Duty (PS22/9) requires firms to demonstrate good outcomes for customers. For pricing models, "good outcomes" includes the model remaining fit for purpose — which means demonstrating that discrimination and calibration are not materially degrading between sign-offs.

The FCA's multi-firm Consumer Duty review in TR24/2 (2024) explicitly found that firms needed stronger governance around ongoing model performance, with outcomes monitored at the segment level rather than just at the portfolio aggregate. "A/E is 1.02 across the book" does not satisfy this. A monitoring report showing that the Gini test is non-significant at $\alpha = 0.32$, that GMCB/UNC is 0.8% and LMCB/UNC is 0.3%, and that the recommendation is NO_ACTION — that is evidence of evidenced monitoring.

The ability to distinguish a GMCB problem (a rate change fixes it in days) from an LMCB problem (a model rebuild required) is also relevant to the Dear CEO letter framework for model risk governance, which expects firms to understand the nature of model failure, not just detect it.

---

## The paper

Brauer, F., Menzel, M. & Wüthrich, M.V. (2025). *Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing*. arXiv:2510.04556.

The Murphy decomposition they apply comes from: Murphy, A.H. (1973). *A New Vector Partition of the Probability Score*. Journal of Applied Meteorology 12: 595–600. The application to insurance scoring rules is more recent; the paper cites Ehm et al. (2016) for the theoretical foundation.

The implementation is in [`insurance-monitoring`](/insurance-monitoring/). The primary entry point for most users is `MonitoringReport`. The `GiniDriftTest` and `GiniDriftBootstrapTest` classes in `gini_drift.py` are available for teams that want the test in isolation without running the full monitoring suite.
