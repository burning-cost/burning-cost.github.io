---
layout: post
title: "Does PSI Actually Catch Pricing Model Drift?"
date: 2026-03-28
categories: [techniques, validation]
tags: [model-monitoring, drift-detection, PSI, gini, murphy-decomposition, calibration, discrimination, isotonic-regression, pricing, python]
description: "PSI detects covariate shift but not rank collapse. On a synthetic UK motor book where a new risk factor emerges post-deployment, PSI stays GREEN while Gini drops 8 points. The Brauer-Wüthrich two-step framework catches it in 3,000 policies."
---

PSI — Population Stability Index — is the monitoring check most UK pricing teams run on their deployed models. It measures whether the distribution of input features has shifted between training and scoring. If PSI exceeds 0.25 on any key feature, the model is flagged for review.

The problem is that PSI is a feature distribution metric. It says nothing about model performance. A model can have PSI = 0.00 on every input feature while silently losing its rank-ordering power — because a new risk factor the model was never trained on has become material, or because the relationship between observed features and claims has changed since training. Brauer, Menzel and Wüthrich (arXiv:2510.04556, October 2025, revised December 2025) formalise this distinction and propose a two-step framework that directly tests what matters: discrimination and calibration. We ran it against PSI on a planted failure.

---

## The failure mode PSI cannot see

The canonical example: a UK home insurer's pricing model was trained in 2023. By Q1 2025, subsidence claims in clay-soil postcodes have risen 40% following successive dry summers. The model's postcode feature distribution has not changed — the same mix of postcodes is being quoted. PSI stays flat. But the model is now materially wrong for exactly the postcodes where the increase is concentrated.

This is not a pathological edge case. It is the standard mechanism by which insurance pricing models fail: the external environment changes, the historical relationship between features and claims no longer holds, and the model's input distributions give no signal because the inputs are the same.

The Brauer-Wüthrich framework instead starts from the model's outputs and directly tests:

1. **Has the model's rank ordering of risks deteriorated?** (discrimination test via Gini)
2. **Is the model globally miscalibrated?** (Murphy score decomposition)
3. **Is the model locally miscalibrated?** (isotonic segmentation for cohort-level bias)

PSI is useful as an early warning for covariate shift. It is not a substitute for these tests.

---

## The framework

**Step 1 — Rank monitoring.** The Gini coefficient (or equivalently, twice the area under the Lorenz curve) is the test statistic for discrimination. The Brauer-Wüthrich contribution is deriving the asymptotic distribution of the Gini estimator under the null hypothesis of stable rank ordering. This gives you a formal hypothesis test: is the observed Gini drop statistically significant, or within the noise expected from sampling variation?

The bootstrap variance estimator they propose is straightforward:

```python
from insurance_monitoring import GiniDriftTest

monitor = GiniDriftTest(n_bootstrap=1000, alpha=0.05)
result = monitor.fit(
    y_true_train=claims_train,
    y_pred_train=predictions_train,
    y_true_monitor=claims_monitor,
    y_pred_monitor=predictions_monitor,
    exposure_train=exposure_train,
    exposure_monitor=exposure_monitor
)

print(result.gini_train)    # e.g. 0.312
print(result.gini_monitor)  # e.g. 0.234
print(result.p_value)       # e.g. 0.003 — significant rank collapse
print(result.ci_lower)      # lower bound of bootstrap CI on Gini drop
```

The test statistic is the difference in Gini between training and monitoring periods, standardised by the bootstrap standard error. A p-value below 0.05 triggers a refit recommendation, not merely a recalibration.

**Step 2 — Calibration testing.** Conditional on acceptable discrimination (rank ordering intact), the Murphy score decomposition separates the model's predictive error into three components:

- **Uncertainty**: inherent variance in claims — irreducible
- **Resolution**: the model's ability to discriminate — what Gini measures
- **Reliability**: calibration — whether predicted rates equal actual rates in expectation

A reliability score near zero means the model is well-calibrated globally. A rising reliability score with stable resolution means the model needs recalibration, not a full refit. The decomposition tells you which intervention is required.

The local calibration test goes further: isotonic regression is fitted to predicted vs actual rates, then segmented to identify cohorts where the miscalibration is concentrated. A model that is 2% overpriced on one segment and 2% underpriced on another passes a global A/E test with flying colours while systematically mispricing both segments.

---

## What we tested

Benchmark: 50,000-policy UK motor training period, 10,000-policy monitoring period. The planted failure: a new risk factor — daytime mileage pattern, correlated with occupation shift post-pandemic — becomes material in the monitoring period. It was not in the training data. The model's PSI on all existing features is near zero (the existing features have not shifted). The model's Gini on the monitoring period drops from 0.312 to 0.234 — an 8-point absolute decline.

Three monitoring approaches:

- **PSI-only**: maximum PSI across all features, threshold 0.25
- **Aggregate A/E**: total actual claims / total predicted claims
- **Brauer-Wüthrich framework**: Gini drift test + Murphy decomposition + isotonic segmentation

---

## The numbers

| Check | PSI-only | Aggregate A/E | Brauer-Wüthrich |
|---|---|---|---|
| Detects rank collapse | No | No | Yes (p = 0.003) |
| Detects aggregate miscalibration | No | Yes — barely (A/E = 0.98) | Yes |
| Detects cohort miscalibration | No | No | Yes (3 high-risk cohorts flagged) |
| Policies to first alarm | Never | 8,000+ | ~3,000 |
| Recommends refit vs recalibrate | No | No | Refit (resolution dropped) |
| False alarm rate | N/A | High | Controlled at 5% |

The aggregate A/E of 0.98 does eventually drift outside a ±5% band — but only after 8,000 monitoring policies. By then the model has been underpricing the high-mileage occupation segment for roughly a quarter. The Gini drift test rejects at p = 0.003 after 3,000 policies, because rank collapse is visible earlier than aggregate calibration drift: the model is not mispricing everyone by 2%, it is mispricing a specific segment heavily while remaining approximately correct on the rest.

PSI never fires. The existing features are stable. The new risk factor — not in the model — is invisible to any feature distribution test.

---

## The Murphy decomposition in practice

The Murphy score decomposition is the less-familiar part of the framework. For a model with Poisson response (claim counts against exposure), the Murphy proper scoring rule decomposes as:

```
E[Murphy(y, μ)] = Uncertainty - Resolution + Reliability
```

where:
- **Uncertainty** is fixed by the data distribution — irreducible
- **Resolution** rises as discrimination improves — larger is better
- **Reliability** measures calibration error — smaller is better

On the monitoring period with planted rank collapse:

| Component | Training | Monitoring | Change |
|---|---|---|---|
| Uncertainty | 0.1421 | 0.1418 | -0.0003 (stable) |
| Resolution | 0.0287 | 0.0198 | **-0.0089 (fell 31%)** |
| Reliability | 0.0041 | 0.0049 | +0.0008 (small) |

The resolution drop is the diagnostic. The model has not become miscalibrated — reliability barely moved. It has lost discrimination power. The correct intervention is a model refit to incorporate the new risk structure, not a scalar recalibration. The Murphy decomposition tells you this. An aggregate A/E check does not.

```python
from insurance_monitoring import MurphyDecomposition

murphy = MurphyDecomposition()
result = murphy.decompose(
    y_true=claims_monitor,
    y_pred=predictions_monitor,
    exposure=exposure_monitor
)

print(result.uncertainty)  # 0.1418
print(result.resolution)   # 0.0198
print(result.reliability)  # 0.0049
print(result.recommendation)  # 'REFIT' or 'RECALIBRATE'
```

The `recommendation` field follows a simple decision rule: if resolution has dropped by more than 15% relative to the training period, recommend refit. If resolution is stable but reliability has risen above a threshold, recommend recalibration. If both are stable, no action.

---

## Isotonic segmentation: where the miscalibration lives

The final piece is local calibration testing. Isotonic regression is fitted to the ordered predicted rates in the monitoring period. The fitted isotonic curve is then segmented by a change-point algorithm to identify the boundary between well-calibrated and miscalibrated cohorts.

On the planted failure, the isotonic segmentation identifies three cohorts in the top two deciles of predicted risk where actual/expected exceeds 1.30. These are the high-mileage occupation policies — the ones the model has no feature for. The bottom eight deciles are well-calibrated. A global recalibration would apply a single correction factor across all deciles, overpricing the bottom eight to compensate for the top two. The isotonic approach identifies which cohorts need intervention.

This is the difference between a model governance action that is defensible and one that is not. Applying a blanket 8% upward loading to compensate for the top-decile shortfall would be evidenced by aggregate A/E alone. It would also be wrong.

---

## Verdict

PSI is a useful early warning for covariate shift. It is not a model monitoring framework. A model can have perfect PSI stability while losing 8 Gini points of discrimination, systematically mispricing its highest-risk decile, and accumulating a material claims shortfall in a specific cohort that a global A/E will not surface for months.

Use the Brauer-Wüthrich two-step framework if:
- You have a deployed pricing model and want to know when it needs intervention
- You want to distinguish between refit (discrimination has fallen) and recalibration (discrimination is intact, calibration has drifted)
- You want cohort-level miscalibration diagnostics, not just a portfolio-level number
- You want a formal statistical test for Gini drift, not an informal "the Gini dropped by 2 points, should we worry?" discussion

Keep running PSI. But run it alongside the Gini drift test and Murphy decomposition, not instead of them.

```bash
uv add insurance-monitoring
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-monitoring). The two-step monitoring benchmark is at `benchmarks/benchmark_drift_detection.py`.

Reference: Brauer, A., Menzel, P. & Wüthrich, M.V. (2025). 'Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing.' arXiv:2510.04556.

- [Does Automated Model Monitoring Actually Work?](/2026/03/27/does-automated-model-monitoring-actually-work/)
- [Gini Drift Testing: Statistical Power and False Alarm Rates](/2026/03/25/gini-drift-testing-statistical-power/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
