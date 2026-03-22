---
layout: post
title: "Fairlearn vs insurance-fairness: Why Generic ML Fairness Tools Miss What the FCA Cares About"
date: 2026-03-22
featured: true
author: Burning Cost
categories: [fairness, model-risk, libraries, regulation]
description: "Fairlearn is excellent for classification fairness. It was not built for insurance pricing, the Equality Act 2010, or the FCA's specific concern: proxy discrimination in a multiplicative rating model. Here is what the difference means in practice."
canonical_url: "https://burning-cost.github.io/2026/03/22/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/"
tags: [fairlearn, insurance-fairness, FCA, proxy-discrimination, EP25/2, Consumer-Duty, pricing, fairness, python, uk-insurance, Equality-Act, demographic-parity]
---

Fairlearn is the obvious starting point when a team is asked to audit an ML model for fairness. It's well-engineered, maintained by Microsoft, has clean scikit-learn compatibility, and implements the standard academic fairness criteria — demographic parity, equalised odds, equitable predictions — through both assessment and mitigation. If you're auditing a fraud classifier or a claims triage model, it is a reasonable first tool to reach for.

For insurance pricing models under FCA supervision, it answers the wrong question.

This post is a direct comparison. Not to dismiss Fairlearn — we recommend it for non-pricing use cases below — but to be precise about where the FCA's actual concern diverges from what a general-purpose fairness library was built to detect.

```bash
uv add insurance-fairness
```

---

## The FCA does not care about demographic parity

This is the central point, and it is worth stating plainly before going any further.

Demographic parity says: the average prediction (or premium) should be the same across protected groups. This is the default framing in most ML fairness literature, and it is the framing Fairlearn was designed to enforce.

Insurance pricing is legally and actuarially different. UK motor, home, and commercial insurers are explicitly permitted to charge different premiums to different risk groups, including groups that correlate with protected characteristics. An older driver does pay more than a 25-year-old. A high-theft-rate postcode does attract a higher premium. This is not discrimination — it is risk differentiation, and it is the basis on which the entire Lloyd's market operates. The Gender Directive (EU, 2012, implemented 2013) banned the use of gender as a direct rating factor, but it did not require premium equality between men and women; risk-based differentiation through other actuarially justified factors is still lawful.

What the FCA cares about — and what Fairlearn cannot detect — is **proxy discrimination**: a model that does not use a protected characteristic directly, but uses non-protected rating factors that are sufficiently correlated with it that the effect is the same as if it had. The postcode is the canonical example. Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, totalling £213m per year, driven entirely by postcodes that encode demographic information without any insurer explicitly modelling ethnicity.

This is the test FCA Exploratory Paper EP25/2 is designed to address. The question is not "is the average premium the same for men and women?" The question is "do your non-protected rating factors act as conduits for characteristics you cannot legally use?"

Those are different questions. They require different tools.

---

## What Fairlearn does

Fairlearn's assessment module computes fairness metrics for a fitted model: demographic parity ratio, equalised odds, true positive rate difference, false positive rate difference. These are the standard classification metrics, designed for models that output a binary prediction or a probability.

```python
from fairlearn.metrics import MetricFrame, demographic_parity_ratio, equalized_odds_ratio
import pandas as pd

mf = MetricFrame(
    metrics={
        "demographic_parity_ratio": demographic_parity_ratio,
        "equalized_odds_ratio":     equalized_odds_ratio,
    },
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_df,
)
print(mf.by_group)
```

The mitigation module applies reductions or postprocessing to bring a model closer to a chosen fairness criterion:

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

mitigator = ExponentiatedGradient(estimator=base_classifier, constraints=DemographicParity())
mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)
```

This is technically well-implemented. The `ExponentiatedGradient` and `ThresholdOptimizer` approaches are academically grounded. For a classification problem where you want to enforce that a model's positive prediction rate is similar across demographic groups, these tools work.

For a pricing GLM with a log-link Poisson frequency component and a Gamma severity component, they do not apply:

- There is no binary output to adjust thresholds on
- The distribution of premiums across groups is not the relevant compliance metric
- The FCA will not accept "we applied demographic parity mitigations to our pricing model" as regulatory evidence — because demographic parity is not the legal standard
- Mitigation before the fact does not produce the audit trail that pricing committees and FCA file reviews require

---

## What insurance-fairness does instead

The `insurance-fairness` library approaches the problem differently because it is solving a different problem: not "are outcomes equal across groups?" but "do your model inputs act as proxies for characteristics you are not permitted to use?"

### Proxy detection, not bias mitigation

The central output of `FairnessAudit.run()` is a proxy vulnerability assessment. It identifies which rating factors have statistically significant correlation with protected characteristics, using three complementary methods:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",   # rate, not total — the audit weights by exposure internally
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)

report = audit.run()
report.to_markdown("audit_q4_2024.md")   # FCA-ready report with regulatory mapping
```

The proxy detection layer uses:

- **Mutual information scores**: catches non-linear relationships between a rating factor and a protected characteristic. A postcode with a non-monotone correlation to ethnicity — higher in band SW1, lower in E1, higher again in M1 — will show up here when Spearman rank correlation misses it.
- **CatBoost proxy R-squared**: fits a gradient boosting model to predict the protected characteristic from the factor, using CatBoost's native categorical support. An R-squared above 0.10 is treated as a proxy concern. On synthetic UK motor data (50,000 policies) with a known postcode-ethnicity correlation, this method surfaces the issue in under 60 seconds.
- **SHAP proxy scores**: links the proxy correlation to actual price impact — the difference in predicted premium attributable to the proxy relationship. A factor that is a strong predictor of protected characteristics but has small SHAP values in the pricing model is a different concern than one that is both a proxy and a major pricing driver.

Fairlearn has no equivalent to any of these. It can tell you whether your model's outputs are demographically unequal. It cannot tell you whether a specific input factor is doing discriminatory work inside the model.

### Exposure-weighted throughout

Insurance portfolios are not rows. A three-month direct debit policy on a city centre vehicle contributes a third of the exposure of an annual policy on the same risk. Fairlearn treats observations as exchangeable. It does not have an exposure parameter.

`insurance-fairness` weights every metric by earned exposure:

```python
from insurance_fairness import calibration_by_group, demographic_parity_ratio

# Calibration by group — the metric most defensible under Equality Act Section 19
cal = calibration_by_group(
    df,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",    # no equivalent in Fairlearn
    n_deciles=10,
)
print(f"Max A/E disparity: {cal.max_disparity:.4f} [{cal.rag}]")

# Demographic parity ratio — log-space, because insurance models are multiplicative
dp = demographic_parity_ratio(df, "gender", "predicted_rate", "exposure")
print(f"Log-ratio: {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f})")
```

The demographic parity ratio is computed in log-space because insurance pricing is multiplicative. A pricing model outputs a rate, and relativity factors compound. The meaningful comparison is the ratio of average rates, not the difference in levels. Fairlearn's `demographic_parity_ratio` computes a ratio in probability space, which is appropriate for a classifier but not for a Poisson rate model.

### The double fairness problem Fairlearn cannot see

A subtler issue, documented in FCA TR24/2 (August 2024): action fairness (premium parity) and outcome fairness (loss ratio parity) can conflict, and firms that audit only at the point of quoting miss the Consumer Duty Outcome 4 obligation.

On a synthetic UK motor TPLI portfolio of 20,000 policies, minimising premium disparity (Delta_1) worsens loss ratio disparity (Delta_2) substantially. The two cannot both be zeroed without abandoning risk differentiation. The FCA does not require you to achieve both simultaneously — but it does expect you to have considered the trade-off and documented it.

`DoubleFairnessAudit` recovers the full Pareto front along the action/outcome trade-off, with quantified revenue impact at each operating point:

```python
import numpy as np
from insurance_fairness import DoubleFairnessAudit

# DoubleFairnessAudit takes arrays, not a DataFrame
X = df[feature_cols].to_numpy()
y_primary   = df["written_premium"].to_numpy()   # primary outcome: revenue
y_fairness  = df["claim_amount"].to_numpy()       # fairness outcome: loss
S           = df["gender"].to_numpy()             # protected group: binary 0/1
exposure    = df["exposure"].to_numpy()

double_audit = DoubleFairnessAudit(n_alphas=20)
double_audit.fit(X, y_primary, y_fairness, S, exposure=exposure)
result = double_audit.audit()   # returns DoubleFairnessResult with full Pareto front
print(result.summary())         # plain-text Pareto front table
print(double_audit.report())    # FCA-ready report mapping to PRIN 2A Outcome 4 and TR24/2
```

Fairlearn's mitigation tools optimise for a single fairness criterion at a time. They will not show you the trade-off surface, and they cannot tell you the revenue cost of each operating point. For a UK insurer documenting their Consumer Duty assessment, the Pareto front is the auditable evidence.

### Financial impact quantification

One concrete thing `insurance-fairness` produces that Fairlearn does not: a sterling estimate of the proxy discrimination cost.

The `ProxyVulnerabilityScore` module implements the Côté, Côté and Charpentier (2025) framework, computing per-policyholder proxy vulnerability as the difference between the unaware premium (no protected attribute in the model) and the awareness-corrected premium (marginalised over the protected characteristic distribution):

```python
from insurance_fairness import ProxyVulnerabilityScore

scorer = ProxyVulnerabilityScore(
    df=df,
    sensitive_col="gender",
    unaware_col="mu_unaware",
    aware_col="mu_aware",
    exposure_col="exposure",
)
result = scorer.compute()
result.summary()
# Proxy Vulnerability Summary
#   D = 0 (F): Mean PV: -1.84, % overcharged: 43.2%, TVaR_95 overcharge: £22.14
#   D = 1 (M): Mean PV: +1.83, % overcharged: 55.1%, TVaR_95 overcharge: £25.31
```

The Citizens Advice (2022) estimate of £213m per year in ethnicity-related overcharging is essentially this calculation applied at market scale. `ProxyVulnerabilityScore` runs it on your portfolio. Fairlearn cannot produce this number because it is not a classification metric — it requires the rate-model structure, exposure weighting, and the marginalisation over the protected attribute that only makes sense in a multiplicative pricing context.

---

## Side-by-side comparison

| | Fairlearn | insurance-fairness |
|---|---|---|
| Primary use case | Classification fairness | Pricing proxy discrimination detection |
| Regulatory context | General algorithmic fairness | FCA EP25/2, Consumer Duty, Equality Act 2010 s.19 |
| Core metric | Demographic parity, equalised odds | Proxy R-squared, mutual information, SHAP proxy scores |
| Exposure weighting | No | Throughout |
| Model structure | Any scikit-learn estimator | Poisson/Tweedie rate models |
| Mitigation tools | Yes — reductions, threshold adjustment | No — detection and reporting |
| Assumes protected characteristic is available | Yes | Tests whether non-protected features proxy for protected ones |
| Output | Metric tables | FCA-ready Markdown audit report with regulatory mapping |
| Financial impact quantification | No | ProxyVulnerabilityScore, parity cost in £ |
| Double fairness (action vs outcome) | No | DoubleFairnessAudit with full Pareto front |

The asymmetry that matters most is in the fourth-to-last row. Fairlearn assumes you have access to the protected characteristic at training time and wants to ensure your model's outputs are fair with respect to it. The FCA EP25/2 framework starts from the opposite position: your model probably does not use the protected characteristic directly (gender has been a prohibited direct rating factor since 2013), but your rating factors may carry it implicitly. The compliance test is whether you have checked for that implicitness — and Fairlearn's architecture is not designed to run that check.

---

## When to use Fairlearn

Use Fairlearn for:

- **Fraud detection**: if you want to ensure your fraud classifier has similar true positive rates across demographic groups, `MetricFrame` and `ExponentiatedGradient` are the right tools
- **Claims triage**: classification with genuine demographic parity or equalised odds requirements
- **Any binary or multiclass model** where you want to report on or constrain protected-group outcomes
- **Experiments where you have protected characteristics in your training data** and want to produce fairness-constrained model variants

Fairlearn is a good library. Its documentation is clear, its academic grounding is solid, and the `MetricFrame` visualisations are genuinely useful for internal review.

Use `insurance-fairness` for:

- Pricing model audits under FCA Consumer Duty (PS22/9) or Equality Act 2010 Section 19
- EP25/2 compliance evidence: detecting whether rating factors act as proxies for protected characteristics
- Exposure-weighted fairness metrics appropriate for Poisson/Tweedie rate models
- Producing audit reports with explicit regulatory mapping for pricing committees and FCA file reviews
- Quantifying the financial impact of proxy discrimination in sterling

---

## Our view

The ML fairness literature largely assumes that the problem is a model producing unequal outcomes across demographic groups, and that the solution is to constrain or adjust the model to reduce that inequality. Fairlearn implements this framework well.

Insurance pricing sits in a different legal and statistical context. Unequal premiums across demographic groups are not intrinsically problematic — they follow from risk differentiation, which is the point. What is prohibited, under Equality Act 2010 Section 19 and FCA Consumer Duty, is a specific mechanism: non-protected factors acting as conduits for protected characteristics in a way that produces discriminatory outcomes without the insurer intending or knowing it.

That is a detection problem, not a mitigation problem. And it requires tools built for the structure of insurance pricing models — multiplicative, rate-based, exposure-weighted — not tools built for classification.

Use Fairlearn when you are doing classification and want to enforce demographic parity. Use `insurance-fairness` when you are pricing and need to demonstrate to the FCA that your model does not use proxies.

---

`insurance-fairness` is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). Polars-native. Python 3.10+.

---

**Related posts:**
- [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/03/your-pricing-model-might-be-discriminating/) — the Lindholm-Richman-Tsanakas-Wüthrich framework, the Citizens Advice data in full, and what a defensible audit trail looks like
- [FCA Consumer Duty Fair Value Assessments: What Good Looks Like](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) — what TR24/2 actually requires vs what most insurers are doing

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [EquiPy vs insurance-fairness](/2026/03/22/equipy-vs-insurance-fairness/) — optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) — conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) — causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/22/dowhy-vs-insurance-causal-inference-insurance-pricing/) — causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) — model monitoring
- [NannyML vs insurance-monitoring](/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/) — drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) — statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) — distributional regression
