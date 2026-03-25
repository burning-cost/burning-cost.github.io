---
layout: post
title: "Alibi Detect vs insurance-monitoring: Drift Detection for Insurance Pricing Models"
date: 2026-03-22
author: Burning Cost
categories: [monitoring, model-risk, libraries, comparisons]
description: "Alibi Detect is a solid general-purpose drift detection library. It doesn't do exposure-weighted PSI, segmented A/E ratios, Gini discrimination drift, or PRA SS3/17 regulatory framing. For insurance pricing teams, those omissions are the whole problem."
tags: [alibi-detect-insurance, model-drift-insurance-pricing, alibi-detect-vs, insurance-model-monitoring-python, PSI, ae-ratio, gini-drift, mSPRT, sequential-testing, insurance-monitoring, model-risk, PRA-SS3-17, python, uk-insurance, polars]
---

If you search "model drift detection Python", Alibi Detect appears near the top. It deserves to. SeldonIO have built a serious library: Kolmogorov-Smirnov, Maximum Mean Discrepancy, Chi-squared, Least-Squares Density Difference - the standard statistical tests, cleanly implemented, with good documentation and around 2,200 GitHub stars at time of writing. For a general ML pipeline monitoring tabular models, text, or images, Alibi Detect is a reasonable default.

For insurance pricing model monitoring, it misses the point almost entirely.

This is the direct comparison - what each library does, where they diverge, and when to use which. If you are not monitoring insurance pricing models, stop here and use Alibi Detect. The remainder of this post is for the people who are.

---

## What Alibi Detect does well

Worth being explicit about this before the critique.

Alibi Detect's `TabularDrift` detector runs KS tests on continuous features and chi-squared on categorical, across all columns simultaneously, with a correction for multiple comparisons. The API is clean:

```python
from alibi_detect.cd import TabularDrift

detector = TabularDrift(x_ref, p_val=0.05)
result = detector.predict(x_current)
print(result["data"]["is_drift"])   # True/False
print(result["data"]["p_val"])      # per-feature p-values
```

That works. The statistical tests are correctly implemented. Alibi Detect also has `MMDDrift` (Maximum Mean Discrepancy, appropriate for high-dimensional features), `LSDDDrift` (Least-Squares Density Difference), and `ClassifierDrift` (uses a trained binary classifier to distinguish reference from current data - powerful for catching complex multivariate shift that marginal tests miss).

It integrates with Seldon Deploy, which matters if you're in a Seldon MLOps shop. The documentation is thorough. The library is actively maintained.

None of this is relevant to what insurance pricing teams actually need from a monitoring system.

---

## Problem 1: PSI without exposure weighting

Population Stability Index is the operational standard for insurance pricing drift monitoring. Every major UK insurer uses some variant of it. Alibi Detect does not implement PSI - it uses KS and chi-squared. This is not a criticism: KS and chi-squared are statistically sounder. But insurance teams inherited PSI thresholds from decades of regulatory practice. A PSI above 0.25 triggers a model review. That convention is embedded in governance frameworks, validation committees, and Lloyd's syndicate oversight processes.

More importantly: even if Alibi Detect's KS tests are statistically preferable, they still treat every row equally. For insurance, rows are policies - and policies vary enormously in how much exposure they represent.

A fleet policy covering 80 vehicles that ran for 12 months contributes one row to KS's test. A 30-day moped policy also contributes one row. They are not equivalent observations. The fleet policy represents roughly 80 car-years of claims exposure; the moped represents 0.08 car-years. Treating them identically means your drift detector has its signal contaminated by short-duration policies that barely had time to generate claims.

`insurance-monitoring` implements exposure-weighted PSI and CSI directly:

```python
from insurance_monitoring.drift import psi

# Alibi Detect / generic approach: every policy counts once
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

In the `insurance-monitoring` benchmark (50,000 training policies, 2019–2021 reference, 2023 current cohort with 2x young driver oversampling): unweighted PSI reads amber at 0.18, exposure-weighted PSI reads red at 0.27. These are different governance conclusions. The weighted version is correct - the young drivers entering via comparison sites are on short-term monthly policies and each count once regardless of their actual exposure contribution.

---

## Problem 2: no actuarial calibration metrics

Alibi Detect monitors statistical properties of feature distributions and model outputs. It does not know what an A/E ratio is.

A/E (actual claims divided by expected claims) is the primary operational metric for pricing model calibration. An aggregate A/E of 1.00 tells you your model is correctly priced on average. An aggregate A/E of 1.00 composed of 1.55 in the under-25 cohort and 0.85 in the over-50 cohort tells you your model is misfiring badly by segment and will generate adverse selection - good risks leave because they're overpriced, bad risks stay.

Alibi Detect's prediction drift test will not catch this. It can tell you that the distribution of your model outputs has shifted; it cannot tell you whether actual experience is tracking your predictions at segment level.

```python
from insurance_monitoring.calibration import ae_ratio, ae_ratio_ci

# Aggregate A/E with exact Garwood (Poisson) confidence intervals
result = ae_ratio_ci(actual, predicted, exposure=exposure)
print(f"A/E: {result['ae']:.3f}  [{result['lower']:.3f}, {result['upper']:.3f}]")
# A/E: 1.082  [1.031, 1.137]

# Segment-level decomposition
seg = ae_ratio(actual, predicted, exposure=exposure, segments=age_bands)
# segment | actual | expected | ae_ratio | n_policies
# 17-24   |    48  |      31  |    1.55  |       312
# 25-34   |    91  |      87  |    1.05  |       847
# 50+     |    61  |      72  |    0.85  |       901
```

The Garwood interval is not cosmetic. At 30 observed claims - a realistic monitoring window for a specialist product or a new entrant cohort - a normal approximation understates the confidence interval by 15–20%. Teams running monitoring with normal approximations will generate spurious "significant" A/E deviations on thin segments, train their actuaries to discount alerts, and miss real problems.

---

## Problem 3: no concept of Gini discrimination drift

This is the failure mode that destroys books silently.

A pricing model can maintain a stable aggregate A/E - it is correctly priced on average - while losing its ability to rank risks. The mechanism is covariate shift: as the portfolio composition changes, the model's relativities become stale. It still predicts the right mean but can no longer separate a good risk from a bad risk within any cohort. The consequence is adverse selection. Alibi Detect monitors prediction drift (has the distribution of your model outputs shifted?). That is not the same question.

`insurance-monitoring` implements the Gini drift z-test from arXiv 2510.04556 - a hypothesis test based on the asymptotic normality of the sample Gini, with bootstrap variance estimation:

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
# GiniDriftResult(z_statistic=-1.93, p_value=0.054, gini_change=-0.03, significant=False)
```

The default alpha is 0.32 - the one-sigma rule. The rationale, from arXiv 2510.04556, is that the cost of missing a Gini decline is measurable in loss ratio; the cost of a false positive investigation is two actuarial days. We agree with this. The governance output is a `GiniDriftBootstrapTest` with a `.plot()` method that produces a standard IFoA/PRA model validation deliverable - bootstrap histogram, CI shading, training Gini line, monitor Gini line, z and p annotation.

---

## Problem 4: no anytime-valid sequential testing

Alibi Detect has no statistical framework for champion/challenger experiments. Most teams handle this outside their monitoring tools, which is where the problem lives.

UK motor teams typically check champion/challenger results monthly. Under a standard two-sample test at p < 0.05, checking monthly for 12 months inflates the actual false positive rate to roughly 25%. In 10,000 Monte Carlo simulations under the null (no real effect), a monthly-peeking t-test produces spurious "significant" results one in four times. Teams that know this discount their own results. Teams that don't know this make bad deployment decisions.

`insurance-monitoring`'s `sequential` module implements the mixture Sequential Probability Ratio Test (mSPRT) from Johari et al. (Operations Research, 2022). The evidence statistic is an e-process with a hard guarantee: P(ever exceeds 1/alpha) ≤ alpha at all stopping times, regardless of how often you look:

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.03,                    # prior: expect ~3% effects on log-rate-ratio
    max_duration_years=2.0,
    min_exposure_per_arm=100.0,  # car-years before any decision
)

# Feed monthly increments — check whenever you like
result = test.update(
    champion_claims=42,   challenger_claims=29,
    champion_exposure=510, challenger_exposure=505,
    calendar_date=datetime.date(2026, 3, 31),
)
print(result.summary)
# "Challenger freq 14.5% lower (95% CS: 0.689–0.996). Evidence: 22.1 (threshold 20.0). Reject H0."
print(result.decision)   # 'reject_H0'
```

The same benchmark: 25% FPR for the monthly-peeking t-test, ~1% FPR for mSPRT. Under H1 (challenger 10% cheaper on frequency), mSPRT detects the effect in a median of 8 months on a 500-policy-per-arm book versus waiting the full 24 months for a pre-registered fixed-horizon test.

---

## Problem 5: no PIT-based calibration monitoring

Alibi Detect monitors distributional drift in features and predictions. It does not have a framework for testing whether the model's calibration - the correspondence between predicted rates and observed claims frequencies - has changed.

The naive approach is Hosmer-Lemeshow applied monthly. Applied monthly for 12 months on a perfectly calibrated model, the false alarm rate is not 5%. It is 46%. After 24 months: 71%. Teams learn to distrust their monitoring output, which is worse than having no monitoring.

`PITMonitor` in `insurance-monitoring` constructs a mixture e-process over probability integral transforms (Henzi, Murph & Ziegel, arXiv:2603.13156, 2025). The guarantee is P(ever alarm | model calibrated) ≤ alpha, at any checking frequency, for ever:

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

monitor.save("pit_monitor_q1_2026.json")
```

In the benchmark (500 calibrated observations, then 500 with 15% rate inflation): repeated Hosmer-Lemeshow fires false alarms throughout the calibrated phase; PITMonitor stays quiet and fires only when calibration genuinely shifts.

---

## Problem 6: no regulatory framing

Alibi Detect is a data science library. It produces drift statistics. It has no concept of PRA SS3/17, Consumer Duty, model risk management, or governance artefacts.

PRA SS3/17 - the supervisory statement on model risk management - requires UK insurers to demonstrate ongoing model monitoring with documented, structured evidence. "The KS test p-value for driver age was 0.03" is not that evidence. "A/E ratio 1.08 [1.031, 1.137], amber threshold exceeded, recommendation: recalibrate" is.

`insurance-monitoring`'s `MonitoringReport` produces a structured Polars DataFrame designed to be written to a database and cited in governance documentation:

```python
from insurance_monitoring import MonitoringReport

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

The Murphy decomposition (Lindholm & Wüthrich, Scandinavian Actuarial Journal 2025) is what drives the `RECALIBRATE` vs `REFIT` verdict. If global miscalibration dominates, apply a multiplier - hours of work. If local miscalibration dominates, refit - weeks of work. Getting this wrong in either direction is expensive. Alibi Detect cannot make this distinction.

---

## Where Alibi Detect wins

The gaps above are real. But Alibi Detect has genuine strengths `insurance-monitoring` does not try to replicate.

**Breadth of drift methods.** MMD and LSDD are theoretically superior to PSI for detecting complex multivariate drift. Alibi Detect's `ClassifierDrift` - train a binary classifier to distinguish reference from current data, use classification accuracy as the drift statistic - is an elegant method that catches non-linear distributional shifts that marginal column-by-column tests miss. `insurance-monitoring` has `InterpretableDriftDetector` (TRIPODD, Panda et al. 2025) for feature-interaction-aware drift attribution, but does not implement MMD or LSDD.

**Non-tabular data.** If you are building a claims image classifier, a fraud text model, or an NLP underwriting tool, Alibi Detect handles image and text drift detection with proper kernel methods. `insurance-monitoring` is tabular-only.

**Seldon Deploy integration.** If you are in a Seldon MLOps shop, Alibi Detect connects directly to Seldon Deploy for production monitoring pipelines. `insurance-monitoring` integrates with Databricks and produces Polars DataFrames - right for UK pricing teams, not for a Seldon deployment.

**Community.** Alibi Detect has ~2,200 GitHub stars and a maintainer team at SeldonIO. `insurance-monitoring` is a niche library for the UK actuarial community. At 11pm when your monitoring pipeline is broken, Alibi Detect has the Stack Overflow thread.

---

## The PSI example in one sentence

Alibi Detect's drift detection treats a £50,000 commercial vehicle policy the same as a £500 moped policy. `insurance-monitoring` does not. For insurance drift detection, that is the entire argument.

---

## When to use each

**Use Alibi Detect if:**
- You are monitoring any ML system that is not an insurance pricing model
- You need text or image drift detection
- You are in a Seldon Deploy shop and want integrated production monitoring
- You need MMD or LSDD for complex multivariate drift
- You want something your ML engineering team can own without actuarial context

**Use insurance-monitoring if:**
- You are running a UK motor, home, or commercial lines pricing model
- You need exposure-weighted PSI/CSI where a fleet policy counts proportionally more than a moped
- You need segmented A/E ratios with Garwood confidence intervals by age band, vehicle type, or geography
- You need Gini drift testing to detect ranking degradation before adverse selection bites
- You need anytime-valid sequential testing for champion/challenger experiments that you check monthly
- You need PITMonitor for calibration monitoring that does not false-alarm under repeated testing
- You need `MonitoringReport` producing PRA SS3/17-ready governance documentation

The tools are not alternatives for the same problem. Alibi Detect solves generic ML drift detection well. `insurance-monitoring` solves insurance pricing model monitoring specifically. A team with both gets Alibi Detect for engineering-facing infrastructure and `insurance-monitoring` for the actuarial outputs that feed governance documentation.

---

## Installation

```bash
pip install insurance-monitoring
```

Polars-native. No scikit-learn dependency. Python 3.10+. Version 0.7.1 at time of writing.

---

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).

---

**Related:**
- [Why Evidently Isn't Enough for Insurance Pricing Model Monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - the same comparison against Evidently, the more common alternative
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - detailed walkthrough of each monitoring layer
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - why aggregate A/E is a lagging indicator for covariate shift
- [Champion Model, Unchallenged](/2026/03/17/champion-model-unchallenged/) - why most insurers never properly test their champion model

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/22/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/22/equipy-vs-insurance-fairness/) - optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) - conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) - causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/22/dowhy-vs-insurance-causal-inference-insurance-pricing/) - causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) - model monitoring
- [NannyML vs insurance-monitoring](/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - drift detection
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) - distributional regression
