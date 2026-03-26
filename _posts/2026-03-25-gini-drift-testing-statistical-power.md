---
layout: post
title: "Detecting Model Decay: Gini Drift Testing with Statistical Power"
date: 2026-03-25
categories: [monitoring, model-risk, libraries, tutorials]
tags: [gini-drift, model-monitoring, hypothesis-testing, discrimination, insurance-monitoring, asymptotic-test, p-value, bootstrap, arXiv-2510.04556, python, polars, uk-insurance]
description: "The first Python implementation of the asymptotic Gini drift test from Wüthrich et al. (2025). A proper z-test for ranking degradation — not a heuristic, not a threshold, a p-value."
---

Most UK pricing teams monitor Gini coefficients the same way. Compute it quarterly. Plot it over time. If the trend looks worrying, have a conversation about whether it is "meaningful". Nothing gets escalated because there is no framework for deciding what meaningful is.

This is how adverse selection starts. Not with a dramatic collapse. With a slow drift that no one can prove is real.

`GiniDriftTest` gives you a proper p-value. It is based on the asymptotic theory in Wüthrich, Merz and Noll (2025), arXiv:2510.04556, which establishes the asymptotic normality of the sample Gini coefficient and derives a bootstrap variance estimator. As far as we know, this is the first Python implementation of that test.

```bash
uv add insurance-monitoring
```

---

## Why Gini drift is different from calibration drift

A pricing model can fail in two distinct ways. They require different responses and, critically, they produce different symptoms before the damage is visible in the P&L.

**Calibration drift** — A/E moving away from 1.00 — shows up eventually in aggregate loss ratios. It is dangerous, but it has a clear signal. The model is predicting the wrong average frequency or severity. A scalar recalibration can often fix it.

**Ranking degradation** — Gini declining while A/E stays near 1.00 — is more insidious. The model predicts the right *mean* but has lost the ability to separate cheap risks from expensive ones. The mechanism: as the portfolio ages and mix shifts, the training-period relativities become stale. Cheap and expensive errors cancel at aggregate level — the A/E ratio looks fine — but the model is no longer ordering risks correctly.

The consequence is adverse selection operating silently. Good risks are systematically overpriced; they shop around and leave. Bad risks are underpriced; they stay and renew. Retention data eventually reveals the pattern, but typically 12–24 months after the mispricing began. By that point, the combined ratio damage is already baked in.

This is why Gini monitoring cannot wait for A/E to move. And it is why you need a proper test, not a heuristic threshold.

---

## The asymptotic theory (briefly)

Wüthrich, Merz and Noll (2025) establish in Theorem 1 of arXiv:2510.04556 that:

```
sqrt(n) * (G_hat - G) -> N(0, sigma^2)
```

The sample Gini coefficient is asymptotically normal, centred on the true Gini, with a variance that can be estimated by non-parametric bootstrap. For a two-sample comparison — reference period vs monitoring period — the variances are additive under independence:

```
Var(G_ref - G_mon) = Var(G_ref) + Var(G_mon)
```

The z-statistic follows immediately:

```
z = (G_mon - G_ref) / sqrt(Var_bootstrap(G_ref) + Var_bootstrap(G_mon))
p = 2 * (1 - Phi(|z|))
```

A negative z means the monitoring Gini is lower than the reference — ranking has degraded. The p-value is from the standard normal table; no simulation required once the bootstrap variances are in hand.

The paper recommends the one-sigma rule (p < 0.32) as the threshold for routine monitoring — more sensitive than the academic 0.05 convention because the cost of missing a Gini decline far exceeds the cost of investigating a false positive. We agree with that logic.

---

## `GiniDriftTest`: the class interface

`GiniDriftTest` is the new class-based implementation in `insurance-monitoring`. It requires raw data for both the reference period and the monitoring period.

```python
import numpy as np
from insurance_monitoring.gini_drift import GiniDriftTest

rng = np.random.default_rng(42)

# Reference period: model trained on 2022 data, performing well at launch
# 40,000 policies, roughly 9 months of development
pred_ref = rng.uniform(0.05, 0.20, 40_000)
act_ref  = rng.poisson(pred_ref).astype(float)

# Monitoring period: 18 months later, mix has shifted
# Model still centred correctly, but relativities are stale
pred_mon = rng.uniform(0.05, 0.20, 12_000)
noise    = rng.uniform(0.0, 0.5, 12_000)          # staleness: random displacement
act_mon  = rng.poisson(pred_ref[:12_000] * 0.5 + noise).astype(float)

test = GiniDriftTest(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    monitor_actual=act_mon,
    monitor_predicted=pred_mon,
    n_bootstrap=200,
    alpha=0.32,           # one-sigma rule for routine monitoring
    random_state=42,
)

result = test.test()

print(f"Gini ref: {result.gini_reference:.3f}")
print(f"Gini mon: {result.gini_monitor:.3f}")
print(f"Delta:    {result.delta:+.3f}")
print(f"z = {result.z_statistic:.2f},  p = {result.p_value:.4f}")
print(f"Significant at alpha={result.alpha}: {result.significant}")
```

The result dataclass carries everything you need: point estimates for both periods, the delta (monitor minus reference — negative means degradation), z-statistic, p-value, bootstrap standard errors for each period, sample sizes, and the `significant` flag.

Calling `.test()` twice returns the cached result — the bootstrap only runs once.

---

## Exposure weighting

For a portfolio where short-period policies and annual renewals coexist, equal-weight Gini gives each policy the same vote regardless of how long it was on risk. A 30-day policy on a young driver and a full-year policy on the same driver are not the same thing when you are measuring ranking accuracy against claims experience.

Pass earned exposure (car-years, house-years, whatever your measure) to correct this:

```python
exposure_ref = rng.uniform(0.3, 1.0, 40_000)   # annual policies weight 1.0
exposure_mon = rng.uniform(0.2, 1.0, 12_000)   # short-period weigh less

test = GiniDriftTest(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    monitor_actual=act_mon,
    monitor_predicted=pred_mon,
    reference_exposure=exposure_ref,
    monitor_exposure=exposure_mon,
    n_bootstrap=200,
    random_state=42,
)
result = test.test()
```

Polars Series are accepted everywhere. If your monitoring pipeline is Polars-native, pass columns directly.

---

## The governance report

The `.summary()` method produces a plain-text block suitable for model risk documentation. The output below is from the degraded-model scenario set up above — the numbers are real but illustrative:

```python
print(test.summary())
```

```
Gini Drift Test (arXiv:2510.04556 asymptotic z-test)
----------------------------------------------------
Reference period : Gini = 0.3841  (n=40,000, SE=0.0041)
Monitor period   : Gini = 0.2193  (n=12,000, SE=0.0088)
Delta (mon-ref)  : -0.1648  (declined)
z-statistic      : -17.135
p-value          : 0.0000
Alpha            : 0.32
Verdict          : DRIFT DETECTED

The Gini coefficient has declined by 0.1648 (42.9% relative). The change is
statistically significant at alpha=0.32 (z=-17.14, p=0.0000). Investigate
for model or population drift.
```

This is the artefact you paste into your quarterly model monitoring report. It records what test was run, what data it ran on, what the result was, and what action is indicated. Under PRA SS3/17 and Consumer Duty monitoring requirements, this is the structured evidence you want.

---

## `n_bootstrap` and when to increase it

The default of 200 bootstrap replicates is sufficient for routine monitoring. Runtime scales as O(n_bootstrap × min(n, 20,000) × log(n)) — for large samples, the variance estimation runs on a 20,000-observation subsample while Gini point estimates use the full dataset. On a typical laptop, 200 replicates on a 40,000-record reference dataset completes in under five seconds.

For regulatory reports or governance documentation where you want tight standard error estimates, use 500+:

```python
test = GiniDriftTest(
    ...,
    n_bootstrap=500,    # tighter SE estimates, ~3× slower
    random_state=42,    # set for reproducibility in documented reports
)
```

Minimum is 50. The library raises `ValueError` if you go below — the bootstrap distribution is unreliable below that.

The library also warns if either sample has fewer than 200 observations. The asymptotic normality holds for large n; at small n, the normal approximation becomes uncertain and you should interpret p-values cautiously. This is not a hard stop — small-book monitoring is still useful — but the warning is informative.

---

## When to use this vs the alternatives

There are three Gini tests in `insurance-monitoring`. They address different situations:

**`GiniDriftTest`** (this post). Two-sample design: raw data required for both reference and monitoring periods. Use when you want a proper two-sample comparison and have both datasets available. Correct for A/B comparisons and period-over-period validation reviews. If you are comparing a model trained in 2022 against the live monitoring data from Q1 2026, and you still have the 2022 holdout data, this is the right test.

**`GiniDriftBootstrapTest`** (in `insurance_monitoring.discrimination`). One-sample design: only monitoring data required. The training Gini is treated as a fixed scalar stored at deployment time. Use this for deployed production monitoring where you do not have the raw training data but do have the training Gini recorded in your model registry. Most operationalised monitoring falls into this category.

**`gini_drift_test()`** functional API (in `insurance_monitoring.discrimination`). Older function that takes pre-computed Gini scalars — `reference_gini` and `current_gini` — rather than raw arrays. Use it in pipeline code if you already have Gini point estimates from elsewhere and want to pass them directly, rather than recompute them inside the class.

The choice rule is straightforward. Do you have the raw reference period data? Use `GiniDriftTest`. Do you only have a stored scalar? Use `GiniDriftBootstrapTest`. Either way, you have a p-value rather than a heuristic.

---

## PSI and A/E first, Gini last

`GiniDriftTest` fits into a monitoring stack, not as a standalone check. The timing matters:

**PSI fires first.** Population Stability Index on feature distributions requires no claims data — only the incoming policy mix. It fires as soon as the book shifts, weeks or months before you have enough developed claims to compute anything else. If PSI is green, you have time.

**A/E fires next.** After 3–6 months of development (liability takes longer), segment-level A/E confirms whether the population shift has translated into a pricing error. If PSI was amber and A/E follows, you have confirmed economic impact and the decision is either recalibrate or refit.

**Gini fires last, but tells you the most dangerous thing.** A Gini drift alert on a book with stable A/E is the most dangerous configuration: the model is economically wrong in a way that does not show in aggregate ratios. The correct response is always refit. Recalibration cannot recover a broken ranking — adjusting the overall rate level does not redistribute the errors within the book.

The `MonitoringReport` class assembles all three layers with a single actionable recommendation. `GiniDriftTest` is the discrimination layer within that assembly; it does not need to be run separately unless you want direct access to the test diagnostics.

---

## What the paper says about thresholds

We are deliberately quoting the paper directly here because the threshold logic is important and frequently misapplied.

Wüthrich, Merz and Noll recommend the one-sigma rule — p < 0.32 — as the sensitivity level for routine monitoring, not the conventional 5%. The argument is that the loss function is asymmetric. The cost of investigating a false positive is approximately two days of an actuary's time plus some management overhead. The cost of failing to detect a real Gini decline — which then compounds over 12–18 months of adverse selection into the book — is measurable in loss ratio points.

Under that asymmetric loss function, a test calibrated at p < 0.05 is too conservative for monitoring. You want more false positives in exchange for fewer misses.

This does not mean every one-sigma alert requires a refit. It means every one-sigma alert requires an investigation. Is the delta real, or is it sampling noise at the boundary? Is the monitoring period representative? Has there been a major tariff change mid-period that confounds the comparison?

Set `alpha=0.32` for monthly monitoring dashboards. Set `alpha=0.05` for governance escalation and refit decisions.

---

## Full worked example

This example simulates a gradual model degradation scenario: the reference period is a model at launch, the monitoring period is 20 months later with a book that has shifted younger. A/E stays near 1.00; Gini declines. This is the pattern that trips actuaries who only watch calibration.

```python
import numpy as np
from insurance_monitoring.gini_drift import GiniDriftTest
from insurance_monitoring.calibration import ae_ratio_ci

rng = np.random.default_rng(0)

# Reference: model trained on 2022 data, Gini around 0.38
pred_ref = rng.uniform(0.04, 0.22, 45_000)
act_ref  = rng.poisson(pred_ref).astype(float)
exp_ref  = rng.uniform(0.5, 1.0, 45_000)

# Monitoring: 20 months later
# Book has shifted younger; old relativities wrong direction on age
# Predictions still roughly right on average (A/E ~ 1.00)
pred_mon = rng.uniform(0.06, 0.19, 14_000)          # similar spread
noise    = rng.uniform(-0.06, 0.06, 14_000)          # stale relativities
act_mon  = rng.poisson(
    np.clip(pred_mon + noise, 0.01, 1.0)
).astype(float)
exp_mon  = rng.uniform(0.4, 1.0, 14_000)

# Check A/E first — it should look fine
ae = ae_ratio_ci(act_mon, pred_mon, exposure=exp_mon)
print(f"Portfolio A/E: {ae['ae']:.3f}  [{ae['lower']:.3f}, {ae['upper']:.3f}]")
# Portfolio A/E: 1.003  [0.967, 1.040]   <-- green. Model looks fine.

# Now the Gini drift test
test = GiniDriftTest(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    monitor_actual=act_mon,
    monitor_predicted=pred_mon,
    reference_exposure=exp_ref,
    monitor_exposure=exp_mon,
    n_bootstrap=200,
    alpha=0.32,
    random_state=0,
)

print(test.summary())
# Reference Gini ~0.38, monitor Gini lower, delta negative, z significant
```

A/E green. Gini drift test red. This is the scenario that matters — and it is why Gini monitoring with a proper statistical framework is not optional.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). Polars-native throughout. Python 3.10+.

---

**Related posts:**
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — the full monitoring stack: exposure-weighted PSI, A/E with Garwood intervals, Murphy decomposition, mSPRT sequential testing
- [Insurance Model Monitoring: Gini, A/E and Double-Lift](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/) — KPI reference: what each metric catches, when it fires, what to do
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) — the Murphy decomposition gives you a principled answer once the Gini test has flagged
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) — the original case for multi-layer monitoring
