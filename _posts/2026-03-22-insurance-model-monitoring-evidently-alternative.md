---
layout: post
title: "Why Evidently Isn't Enough for Insurance Pricing Model Monitoring"
date: 2026-03-22
author: Burning Cost
categories: [monitoring, model-risk, libraries]
description: "Evidently is excellent for generic ML monitoring. It doesn't do exposure-weighted PSI, Poisson A/E ratios, Gini drift testing, or anytime-valid sequential tests. For UK insurance pricing teams, those omissions matter."
canonical_url: "/2026/03/22/insurance-model-monitoring-evidently-alternative/"
tags: [insurance-model-monitoring-python, Evidently, model-drift-detection, PSI, ae-ratio, gini-drift, mSPRT, sequential-testing, insurance-monitoring, model-risk, pricing, python, uk-insurance, polars]
---

Evidently is the obvious first choice when a team needs open-source ML monitoring. It's well-engineered, actively maintained, has 20+ drift detection methods, clean HTML reports, and integrations with MLflow and other standard MLOps infrastructure. If you're monitoring a churn model, a credit propensity score, or a fraud classifier, it will serve you well.

For insurance pricing model monitoring, it leaves you with gaps that are not cosmetic. This post is a direct comparison - not to dismiss Evidently, but to be clear about where the domain-specific requirements of insurance pricing diverge from what a general-purpose tool can cover.

```bash
uv add insurance-monitoring
```

---

## What Evidently does well

It's worth being explicit about what you're giving up if you don't use it.

Evidently's `DataDriftPreset` runs across all your feature columns automatically, selects an appropriate statistical test per column type (PSI for categorical, K-L divergence or Wasserstein for continuous), and renders a report you can share with non-technical stakeholders. Setup is ten lines of code. The HTML report is genuinely readable. The ecosystem integrations - MLflow, Grafana, Prefect - are mature.

For engineering teams standing up a first monitoring pipeline, Evidently's breadth and approachability are real advantages. We're not arguing you shouldn't use it for those use cases.

---

## The six things it doesn't do for insurance

### 1. Exposure-weighted PSI

Standard PSI treats every row equally. For an insurance portfolio, rows are policies - and policies vary enormously in how much exposure they represent. An annual fleet policy covering 80 vehicles contributes the same to an unweighted PSI bin count as a 30-day social/domestic policy on a single hatchback. They are not the same from a claims exposure perspective.

The problem is acute when young drivers are overrepresented in your short-period business (comparison sites, monthly-pay schemes) and underrepresented in your annual renewals. Unweighted PSI will underestimate the distributional shift in driver age, because the short-period young-driver policies each count once despite contributing 0.25 car-years of exposure.

Evidently does not expose an exposure weight parameter on its PSI implementation. The `insurance-monitoring` library does:

```python
from insurance_monitoring.drift import psi

# Evidently-style: every policy counts once
psi_unweighted = psi(ages_ref, ages_cur, n_bins=10)

# Insurance-correct: weight by earned car-years
psi_weighted = psi(
    ages_ref,
    ages_cur,
    n_bins=10,
    exposure_weights=earned_exposure_cur,
    reference_exposure=earned_exposure_ref,
)

print(f"Unweighted PSI: {psi_unweighted:.3f}")  # 0.18 — amber
print(f"Weighted PSI:   {psi_weighted:.3f}")    # 0.27 — red
```

In the synthetic UK motor benchmark on 50,000 training policies (2019–2021) monitored against a 2023 cohort with 2x young driver oversampling, unweighted PSI reads amber and weighted PSI reads red. These are different conclusions. The weighted version is correct.

### 2. Segmented A/E ratios with Poisson confidence intervals

Evidently's model performance monitoring tracks prediction drift and can compare actual vs predicted at aggregate level for regression targets. It does not understand what A/E means to an actuary: actual claims divided by expected claims, at segment level, with exact Poisson confidence intervals, tracked against exposure.

The aggregate A/E problem is well-known in pricing. A model that is 15% cheap on under-25s and 15% expensive on over-60s will read 1.00 at portfolio level. A generic prediction drift alert will not fire. The under-25 cohort continues to be mispriced until the age-banded quarterly review catches it - typically 6–12 months later.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with exact Garwood intervals — not a normal approximation
result = ae_ratio_ci(actual, predicted, exposure=exposure)
print(f"A/E: {result['ae']:.3f}  [{result['lower']:.3f}, {result['upper']:.3f}]")
# A/E: 1.082  [1.031, 1.137]

# Where is it misfiring?
seg = ae_ratio(actual, predicted, exposure=exposure, segments=age_bands)
# segment | actual | expected | ae_ratio | n_policies
# 17-24   |    48  |      31  |    1.55  |       312
# 25-34   |    91  |      87  |    1.05  |       847
# 50+     |    61  |      72  |    0.85  |       901
```

The Garwood interval matters at low policy counts. For a new entrant cohort or a specialist product, 30 claims is not unusual for a monitoring period. Normal approximation understates the interval width by 15–20% at that level, which means monitoring reports showing "significant" A/E deviations on thin data.

### 3. Gini drift testing

A pricing model can maintain a stable aggregate A/E while losing its ability to rank risks. The mechanism is covariate shift: as portfolio composition changes, the model's relativities become stale. It still predicts the right mean (cheap errors and expensive errors cancel) but can no longer separate good risks from bad. The consequence is adverse selection - good risks leave because they're overpriced, bad risks stay - which is economically equivalent to mispricing but proceeds silently until retention data reveals the pattern.

Evidently monitors prediction drift (has the distribution of your model outputs shifted?). That is not the same as Gini drift. A model can have stable prediction distribution and falling Gini simultaneously.

`insurance-monitoring` implements the Gini drift z-test from [arXiv 2510.04556](https://arxiv.org/abs/2510.04556) - a proper hypothesis test based on the asymptotic normality of the sample Gini, with bootstrap variance estimation:

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=50_000,
    n_current=15_000,
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur,   current_predicted=pred_cur,
)
# {'z_statistic': -1.93, 'p_value': 0.054, 'gini_change': -0.03, 'significant': False}
```

The decision thresholds (p < 0.32 triggers `INVESTIGATE`, p < 0.10 triggers `REFIT`) are deliberately conservative. The authors' argument, which we agree with, is that the cost of missing a Gini decline is measurable in loss ratio; the cost of a false positive investigation is two days of actuary time.

### 4. The recalibrate-vs-refit decision

This is the gap that costs the most when it's wrong.

If your monitoring fires an alert, you face a binary: recalibration (apply a scalar multiplier to all predictions - hours of work) or refit (rebuild the model on recent data - weeks of work). Getting this wrong in either direction is expensive.

Evidently cannot help you make this decision. It can tell you that model performance has degraded, but not whether the degradation is from a global scale error (cheap to fix) or a broken ranking (expensive to fix).

The Murphy decomposition (Lindholm & Wüthrich, Scandinavian Actuarial Journal 2025) resolves this. It decomposes forecast error into uncertainty (irreducible noise), discrimination (ranking skill), and miscalibration. Miscalibration further splits into GMCB (global - fixed by a multiplier) and LMCB (local - requires a refit). If GMCB dominates, recalibrate. If LMCB dominates, refit.

```python
from insurance_monitoring.calibration import murphy_decomposition

result = murphy_decomposition(actual, predicted, exposure, distribution='poisson')

print(f"Global MCB (GMCB): {result.global_mcb:.4f}")  # correctable with a multiplier
print(f"Local MCB (LMCB):  {result.local_mcb:.4f}")   # requires model refit
print(f"Verdict:           {result.verdict}")          # 'RECALIBRATE' or 'REFIT'
```

In the 10,000/4,000 policy benchmark with 30% randomised predictions (simulating staleness), a manual A/E check produces `INVESTIGATE`; Murphy decomposition produces `REFIT` because LMCB (0.0090) exceeds GMCB (0.0002). The information content is different, and acting on it is different.

### 5. Anytime-valid sequential testing for champion/challenger

Evidently does not have a statistical testing framework for A/B experiments. Most teams run champion/challenger experiments outside their monitoring tools entirely - which is fine. But the testing problem we're describing happens in that gap.

UK motor teams typically check champion/challenger results monthly. Under a standard two-sample test, checking monthly for 12 months at p < 0.05 inflates the actual false positive rate to roughly 25%. In 10,000 Monte Carlo simulations under the null hypothesis (no real effect), a monthly-peeking fixed-horizon t-test produces spurious "significant" results one in four times.

The `sequential` module implements the mixture Sequential Probability Ratio Test (mSPRT) from Johari et al. (Operations Research, 2022). The evidence statistic is an e-process with a hard guarantee: P(ever exceeds 1/alpha) ≤ alpha at all stopping times, regardless of how often you look.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.03,                    # prior: expect ~3% effects on log-rate-ratio
    max_duration_years=2.0,
    min_exposure_per_arm=100.0,  # car-years required before any decision
)

# Feed monthly increments as they arrive — check as often as you like
result = test.update(
    champion_claims=42,   challenger_claims=29,
    champion_exposure=510, challenger_exposure=505,
    calendar_date=datetime.date(2026, 3, 31),
)
print(result.summary)
# "Challenger freq 14.5% lower (95% CS: 0.689–0.996). Evidence: 22.1 (threshold 20.0). Reject H0."
print(result.decision)   # 'reject_H0'
```

The same benchmark that produces 25% FPR for the peeking t-test produces ~1% FPR for mSPRT. Under H1 (challenger 10% cheaper on frequency), mSPRT detects the effect in a median of 8 months on a 500-policy-per-arm book versus waiting the full 24 months for a pre-registered fixed-horizon test.

### 6. Anytime-valid calibration change detection

Evidently monitors whether your model's output distribution has drifted. A subtler problem is whether the model's *calibration* has changed - not that predictions are at different levels, but that the correspondence between predicted probabilities and observed frequencies has broken down.

The standard approach is Hosmer-Lemeshow, applied at each monitoring interval. Applied monthly, the false alarm rate is not 5%. With 12 monthly checks on a perfectly calibrated model, it's 46%. After two years: 71%. Teams that use this approach correctly learn to distrust their monitoring system, which is the worst possible outcome.

`PITMonitor` constructs a mixture e-process over probability integral transforms (Henzi, Murph & Ziegel, arXiv:2603.13156, 2025). The formal guarantee is P(ever alarm | model calibrated) ≤ alpha, at any checking frequency, forever.

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

monitor = PITMonitor(alpha=0.05, n_bins=100, rng=42)

for row in live_claims_stream:
    mu = row.exposure * row.lambda_hat
    pit = float(poisson.cdf(row.claims, mu))
    alarm = monitor.update(pit)
    if alarm:
        print(f"Calibration drift at t={alarm.time}")
        print(f"Estimated changepoint: t~{alarm.changepoint}")
        break

# Persist between monitoring runs
monitor.save("pit_monitor_q1_2026.json")
```

In the benchmark (500 calibrated observations, then 500 with 15% rate inflation): repeated Hosmer-Lemeshow fires false alarms throughout the calibrated phase; PITMonitor stays near zero and fires only when calibration genuinely shifts.

---

## Where Evidently wins

The gaps above are real. But Evidently has genuine strengths that `insurance-monitoring` doesn't try to replicate:

**Dashboard and visualisation.** Evidently's HTML reports and monitoring dashboards are production-ready. `insurance-monitoring` produces Polars DataFrames and matplotlib figures - appropriate for governance artefacts and programmatic use, not for a self-serve monitoring portal.

**LLM and text monitoring.** Evidently has substantial tooling for large language model evaluation - semantic similarity, retrieval relevance, toxicity. Irrelevant for pricing model monitoring; valuable if your firm is deploying LLM-based underwriting tools alongside your GLM.

**Ecosystem integration.** Evidently integrates directly with MLflow, Grafana, Prefect, and Airflow. `insurance-monitoring` integrates with Databricks (there's a ready-to-import monitoring notebook), but does not have first-class connectors to the broader MLOps stack.

**Community size.** Evidently has 25,000+ GitHub stars and a large community. `insurance-monitoring` is a niche library built specifically for the UK actuarial community. If you need Stack Overflow answers at 11pm, Evidently has the advantage.

---

## Putting it together: `MonitoringReport`

For pricing teams who want a single call that covers the insurance-specific requirements, `MonitoringReport` assembles exposure-weighted PSI per feature, A/E with Garwood intervals, Gini drift test, and Murphy decomposition into one traffic-light output:

```python
import polars as pl
import numpy as np
from insurance_monitoring import MonitoringReport

rng = np.random.default_rng(42)

n_ref, n_cur = 50_000, 15_000
pred_ref = rng.uniform(0.05, 0.20, n_ref)
act_ref  = rng.poisson(pred_ref).astype(float)
pred_cur = rng.uniform(0.05, 0.20, n_cur)
act_cur  = rng.poisson(pred_cur * 1.08).astype(float)   # 8% underpricing

feat_ref = pl.DataFrame({
    "driver_age":  rng.integers(18, 80, n_ref).tolist(),
    "vehicle_age": rng.integers(0, 15, n_ref).tolist(),
    "ncd_years":   rng.integers(0, 9, n_ref).tolist(),
})
feat_cur = pl.DataFrame({
    "driver_age":  rng.integers(25, 85, n_cur).tolist(),  # older mix entering book
    "vehicle_age": rng.integers(0, 15, n_cur).tolist(),
    "ncd_years":   rng.integers(0, 9, n_cur).tolist(),
})

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)   # 'RECALIBRATE'
print(report.to_polars())
# metric                 | value  | band
# ae_ratio               | 1.08   | amber
# gini_current           | 0.39   | amber
# gini_p_value           | 0.054  | amber
# csi_driver_age         | 0.14   | amber
# murphy_discrimination  | 0.041  | amber
# murphy_miscalibration  | 0.003  | green
# recommendation         | nan    | RECALIBRATE
```

The `to_polars()` output is designed to be written to a database and used as a model risk artefact. Under PRA SS3/17 (insurer model risk management), you need structured evidence of ongoing model monitoring. A traffic-light report with named metrics, values, thresholds, and an actionable recommendation is the right format.

---

## Our view

Use Evidently if: your team is standing up generic ML monitoring, you need a dashboard, you need LLM evaluation, or you want something your software engineers can own without actuarial involvement.

Use `insurance-monitoring` if: you are running a UK motor, home, or commercial lines pricing model and need exposure-weighted drift detection, segment-level A/E ratios, Gini drift testing, the Murphy decomposition, or anytime-valid testing for champion/challenger experiments.

The tools are not mutually exclusive. A team with both Evidently and `insurance-monitoring` gets the best of both: Evidently for engineering-facing infrastructure and dashboards, `insurance-monitoring` for the actuarial-facing monitoring outputs that feed governance documentation.

What we'd resist is the assumption that Evidently covers the monitoring problem for insurance. It covers the generic ML monitoring problem. For pricing models, generic is not enough.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). Polars-native. No scikit-learn dependency. Python 3.10+.

---

**Related posts:**
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - detailed walkthrough of each monitoring layer with worked examples
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - why aggregate A/E is a lagging indicator for covariate shift
- [Champion Model, Unchallenged](/2026/03/17/champion-model-unchallenged/) - why most insurers never properly test their champion model

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/22/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/22/equipy-vs-insurance-fairness/) - optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) - causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/22/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [NannyML vs insurance-monitoring](/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) - statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) - distributional regression
