---
layout: post
title: "Frequency and Severity Are Two Outputs. You Have One Prediction Interval."
date: 2026-03-13
categories: [libraries, pricing, uncertainty]
tags: [conformal-prediction, multivariate, joint-prediction, frequency-severity, Solvency-II, SCR, Fan-Sesia, LWC, coordinate-wise-standardization, Consumer-Duty, FCA, GLM, GBM, Poisson, Gamma, python, motor, home, insurance-conformal]
description: "Every UK pricing team runs separate GLMs for frequency and severity. They have marginal prediction intervals for each — 95% coverage on frequency, 95% on severity — and no guarantee whatsoever about the joint outcome. insurance-conformal implements Fan & Sesia's coordinate-wise standardization to give you a single prediction set covering both simultaneously."
---

Every UK motor pricing team runs a Poisson GLM for frequency and a Gamma GLM for severity. You get a point estimate from each: λ̂ = 0.09 claims per year, μ̂ = £2,400 per claim. The pure premium is £216 and it goes into your rating engine.

You might also have prediction intervals. If you use `insurance-conformal`, you have a frequency interval and a severity interval, each with 95% marginal coverage. Run both intervals to 95% and you have given yourself two separate statements: "the true frequency is in [0.03, 0.17] with 95% probability" and "the true severity is in [£1,100, £4,200] with 95% probability." Both statements are valid.

Their conjunction is not guaranteed. If frequency and severity each have 5% miscoverage probability, the probability that at least one of them fails could be as high as 10%. It depends on the dependence structure between the two errors — and you have no control over it.

This matters when the question is not "what is the frequency interval?" but "what is the joint technical price interval?" or "what do I need to hold as capital to cover joint adverse outcomes at 99.5%?" For those questions, marginal coverage is the wrong thing to control. [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) implements joint coverage control for multi-output insurance models.

```bash
pip install insurance-conformal
```

---

## The problem with running two separate intervals

Conformal prediction produces a finite-sample valid marginal interval for a scalar output Y: if you run the method at level α = 0.05, the true value falls outside the interval on at most 5% of future observations. The guarantee holds under exchangeability, with no distributional assumption. It is the only way to get that kind of finite-sample statement out of a GBM.

When your model has two outputs — frequency N and severity C — the corresponding prediction sets are:

```
P(N ∈ [L_freq, U_freq]) ≥ 0.95      [marginal, frequency]
P(C ∈ [L_sev, U_sev])  ≥ 0.95      [marginal, severity]
```

What you want is:

```
P(N ∈ [L_freq, U_freq] AND C ∈ [L_sev, U_sev]) ≥ 0.95
```

That is a joint coverage guarantee. It says the rectangle covers the true (frequency, severity) pair simultaneously with 95% probability. The two marginal statements above do not imply it.

The standard fix is Bonferroni: run each at α/2 = 2.5%, so each marginal interval is 97.5% wide, and the union bound gives you at least 95% joint coverage. Correct, but wasteful. You are spending 97.5% coverage budget per dimension when 95% joint coverage only requires you to spend the budget efficiently across the two dimensions together.

Fan & Sesia (arXiv:2512.15383, December 2025) show you can do 20–35% better than Bonferroni while maintaining the same joint coverage guarantee. The key is coordinate-wise standardization.

---

## Why the scale mismatch matters

The naive approach to joint conformal prediction is to aggregate per-dimension residuals into a scalar score and then find a single quantile threshold. For frequency and severity, the obvious aggregation is the maximum:

```
S_i = max(|N_i - λ̂_i|, |C_i - μ̂_i|)
```

This is dimensionally incoherent. Frequency residuals for a UK motor book run around 0.05–1.5 (the difference between 0 or 1 claims and your Poisson prediction). Severity residuals run around £200–£5,000. The max of those two numbers is always the severity residual. The frequency dimension contributes nothing to the joint score, and the calibrated interval imposes no effective constraint on frequency at all.

Coordinate-wise standardization from Fan & Sesia resolves this directly. For each output dimension j, compute the mean and standard deviation of the calibration residuals:

```
μ̂_j  = mean of E_j^i over calibration set
σ̂_j  = std  of E_j^i over calibration set
```

Then the standardized scalar score is:

```
S_i = max_j  (E_j^i - μ̂_j) / σ̂_j
```

After standardization, frequency z-scores and severity z-scores are both in units of "standard deviations from the mean calibration residual." A frequency residual of 0.8 (which is 2.3 standard deviations above the mean) scores higher than a severity residual of £1,200 (which is 0.5 standard deviations above the mean) — correctly, because the frequency outcome is the more extreme one.

The calibration quantile is taken from this scalar score distribution, and the prediction intervals for each dimension are back-transformed to original units using σ̂_j. Both dimensions get meaningful constraints.

---

## The GWC/LWC correction

There is a subtlety. If you compute μ̂ and σ̂ from the calibration set and then apply them to the calibration set to rank scores, the test point's residual will be standardized by different statistics than the calibration residuals. Exchangeability — the property that makes conformal valid — is broken.

Fan & Sesia's GWC (global worst-case) and LWC (local worst-case) algorithms fix this without requiring access to the test point's residual before prediction.

GWC computes a conservative quantile threshold that remains valid regardless of where the test residual falls. It is O(dn) — fast, and always valid, but slightly wider intervals than necessary.

LWC partitions the residual space using order statistics and computes targeted thresholds per partition region. It is O(d²n log n) — for d=2 and n=10,000, that is under a second — and produces intervals 20–35% narrower than Bonferroni while maintaining finite-sample joint coverage:

```
P(Y_new ∈ Ĉ(X_new)) ≥ 1 - α
```

No distributional assumption. Finite sample, not asymptotic. This is Theorem 9 from their paper.

---

## Frequency × severity in practice

Here is a realistic motor pricing workflow on a held-out calibration set of 3,000 policies.

```python
import numpy as np
from insurance_conformal.multivariate import JointConformalPredictor

# freq_glm: fitted statsmodels Poisson GLM with log link
# sev_gbm: fitted LightGBM model on claims with severity > 0

predictor = JointConformalPredictor(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    alpha=0.05,
    method='lwc',      # Fan & Sesia LWC — tightest valid method
    score_fn='auto',   # Poisson deviance for frequency, Gamma deviance for severity
)

# zero_claim_mask: True for policies with N=0 (severity unobserved)
zero_mask = (y_freq_cal == 0)

predictor.calibrate(
    X_cal,
    Y_cal={'frequency': y_freq_cal, 'severity': y_sev_cal},
    exposure=exposure_cal,
    zero_claim_mask=zero_mask,
)

print(predictor.calibration_summary())
# {
#   'dimensions': ['frequency', 'severity'],
#   'n_cal': 3000,
#   'method': 'lwc',
#   'alpha': 0.05,
#   'mu_hat': {'frequency': 0.009, 'severity': 45.2},
#   'sigma_hat': {'frequency': 0.31, 'severity': 1847.0},
#   'half_widths': {'frequency': 0.127, 'severity': 3204.0},
# }
```

The `sigma_hat` values show the scale disparity directly: 0.31 for frequency versus £1,847 for severity. After standardization, both dimensions compete on equal terms for the max-score quantile.

Now predict joint intervals for new business:

```python
joint_set = predictor.predict(X_new, exposure=exposure_new)

# Per-policy joint intervals
print(joint_set.lower['frequency'][:3])   # [0.02, 0.04, 0.01]
print(joint_set.upper['frequency'][:3])   # [0.28, 0.31, 0.19]
print(joint_set.lower['severity'][:3])    # [182.0, 510.0, 0.0]
print(joint_set.upper['severity'][:3])    # [7221.0, 6104.0, 4388.0]

# Policy 1: frequency in [0.02, 0.28], severity in [£182, £7,221]
# Joint guarantee: P(both hold simultaneously) >= 0.95
```

Validate on a test set:

```python
from insurance_conformal.multivariate import coverage_report

report = coverage_report(predictor, X_test, Y_test, exposure=exposure_test)
print(report)
# {
#   'joint_coverage': 0.962,   # ≥ 0.95 ✓
#   'marginal_coverages': {'frequency': 0.981, 'severity': 0.974},
#   'mean_widths': {'frequency': 0.254, 'severity': 7022.0},
#   'alpha': 0.05,
#   'method': 'lwc',
#   'target_coverage': 0.95,
#   'coverage_gap': 0.012,
# }
```

Joint coverage is 96.2% against the 95% target — valid by 1.2 percentage points. The marginal coverages are higher than necessary, as expected: LWC distributes the coverage budget efficiently without forcing each marginal to 97.5%.

---

## Zero-claim masking for severity

Policies with no claims in the observation period have unobserved severity — there is nothing to compute a severity residual from. This is the zero-inflation problem, and it is more severe in home insurance (subsidence frequency around 0.1% annually) than motor.

The library uses severity masking: for zero-claim calibration observations, the severity residual is set to zero before standardization. This is conservative — it narrows the severity contribution to the max-score, making the severity interval wider than necessary. But it is valid and requires no assumption about the missing severity distribution.

```python
# Three handling strategies, all valid:
# 1. mask (default):     E_sev = 0 for zero-claim obs → conservative
# 2. frequency_only:    S_i = E_freq_std only for zero-claim obs
# 3. oob_calibration:   use out-of-bag RF scores to multiply effective n

predictor = JointConformalPredictor(
    models={'flood': flood_rf, 'fire': fire_glm, 'subsidence': sub_gbm},
    alpha=0.05,
    method='lwc',
)
predictor.calibrate(
    X_cal, Y_cal,
    zero_claim_mask=zero_mask,  # Mask=True for any peril with no observed claim
)
```

For the three-dimensional case (flood, fire, subsidence), coordinate-wise standardization across all three perils is automatic. A home policy in Lincolnshire might produce: flood [£2,000, £38,000], fire [£800, £7,400], subsidence [£0, £19,000]. That is the joint 95% prediction set — all three bounds hold simultaneously.

---

## Method comparison

To see how much LWC gains over Bonferroni on your specific portfolio:

```python
from insurance_conformal.multivariate import compare_methods

results = compare_methods(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    X_cal=X_cal, Y_cal=Y_cal,
    X_test=X_test, Y_test=Y_test,
    alpha=0.05,
    zero_claim_mask=zero_mask,
)

for method, r in results.items():
    freq_w = r['mean_widths']['frequency']
    sev_w  = r['mean_widths']['severity']
    jcov   = r['joint_coverage']
    print(f"{method:12s}  joint_cov={jcov:.3f}  freq_width={freq_w:.3f}  sev_width={sev_w:.0f}")

# bonferroni   joint_cov=0.974  freq_width=0.291  sev_width=8104
# gwc          joint_cov=0.968  freq_width=0.267  sev_width=7431
# lwc          joint_cov=0.962  freq_width=0.254  sev_width=7022
```

All three achieve joint coverage above 95%. LWC gives frequency intervals 13% narrower than Bonferroni and severity intervals 13% narrower, at the same joint coverage level. On a £250 annual premium, a 13% tighter severity interval translates to a narrower quoted price range — real information, not a statistical curiosity.

---

## Solvency II SCR at 99.5%

The most direct regulatory application is capital estimation. Solvency II Article 101 requires the SCR to equal the 99.5% Value at Risk of the one-year loss distribution. Standard formula uses prescribed correlation matrices — multivariate normal aggregation that may misestimate joint tail dependence.

Hong (arXiv:2503.03659, March 2025) demonstrated this gap explicitly: on personal injury claims data, a GLM-based internal model achieved only 70.2% empirical coverage at its stated "99.5%" confidence level. The conformal approach achieved 99.6% coverage with no distributional assumption.

`SolvencyCapitalEstimator` wraps `JointConformalPredictor` with `alpha=0.005` and `one_sided=True`. The joint prediction set becomes [0, U_freq] × [0, U_sev], and the conservative SCR per policy is U_freq × U_sev:

```python
from insurance_conformal.multivariate import SolvencyCapitalEstimator, scr_report

scr = SolvencyCapitalEstimator(
    models={'frequency': freq_glm, 'severity': sev_gbm},
    alpha=0.005,   # 99.5% coverage
    method='gwc',  # GWC recommended for regulatory use — more conservative
)
scr.calibrate(X_cal, Y_cal, exposure=exposure_cal, zero_claim_mask=zero_mask)

result = scr.estimate(X_portfolio, exposure=exposure_portfolio)

print(result.aggregate_scr)            # £4,832,100
print(result.coverage_guarantee)      # 0.995
print(result.finite_sample_bound)     # 0.001  (1/n_cal = 1/999)
print(result.summary())
# {
#   'aggregate_scr': 4832100.0,
#   'coverage_guarantee': 0.995,
#   'finite_sample_bound': 0.001,
#   'n_cal': 999,
#   'method': 'gwc',
#   'per_policy_stats': {
#     'mean': 96.6, 'p50': 68.4, 'p95': 312.1, 'p99': 891.0, 'max': 4204.0
#   }
# }
```

At n=999 calibration observations, the finite-sample bound is 1/(n+1) = 0.001. The coverage guarantee is exactly ≥99.5%, with at most 0.1% excess conservatism. Compare this with a parametric internal model, which has only asymptotic guarantees — valid as n → ∞, with no finite-sample statement.

The practical note for PRA submissions: a conformal SCR bound has strictly stronger statistical grounding than a parametric internal model at the same calibration sample size. The tradeoff is that the intervals are wider — the conservatism is honest about uncertainty rather than hiding it behind a distribution assumption.

Bootstrap confidence intervals for the aggregate SCR are available via `result.bootstrap_ci(n_bootstrap=1000)`. For a UK motor book of 50,000 policies with n=999 calibration data, expect the aggregate SCR 95% CI to be roughly ±8% around the point estimate.

---

## Consumer Duty: multi-dimensional outcome monitoring

The FCA's June 2024 multi-firm review of insurance Consumer Duty monitoring found that most firms lacked statistically rigorous thresholds for determining when outcomes were poor. The review noted firms were relying on simple benchmark comparisons rather than evidence-based statistical triggers.

Joint conformal prediction sets are the obvious answer to this: they provide distribution-free, finite-sample valid tolerance regions for multi-dimensional outcome vectors. For a customer segment, predict the joint distribution of (claims acceptance rate, average settlement days, complaint rate). If the observed outcomes for that segment fall outside the 95% joint prediction set, you have a calibrated, auditable statistical signal that something is wrong — not a "we compared to last year" narrative.

This application uses the same `JointConformalPredictor` API with outcome metrics as outputs rather than frequency and severity. The coverage guarantee is identical.

---

## BUILD score: 17/20

| Dimension | Score | Rationale |
|---|---|---|
| Pricing relevance | 5/5 | Direct F/S joint intervals; Solvency II SCR; Consumer Duty outcome monitoring. Three live UK use cases, all with regulatory hooks. |
| Python gap | 4/5 | Complete gap confirmed: MAPIE, crepes, nonconformist, puncc, TorchCP — none implement joint multi-output conformal prediction with GLM/GBM base models. Score 4 not 5 because Fan & Sesia's research code exists on GitHub, just unpackaged. |
| Build feasibility | 4/5 | Algorithm is mathematically explicit (O(d²n log n) LWC). numpy-only. Delivered in 198 tests across 7 modules. |
| Novelty | 4/5 | Coordinate-wise standardization for F/S scale mismatch, insurance GLM/GBM hooks, Solvency II mode — genuinely new combination. Score 4 not 5 as Fan & Sesia did the core mathematics. |

Three bugs found during build are worth noting: a deviance-as-width units error in early score implementation, zero-claim sigma deflation when the mask was applied before rather than after standardization, and an overclaim in the LWC-vs-Bonferroni efficiency comparison (corrected to 20–35% from an initial 30–40% estimate). These are documented in the library changelog.

---

## Technical summary

- Fan & Sesia (arXiv:2512.15383) GWC and LWC algorithms with coordinate-wise standardization
- Zero-claim severity masking for Poisson/Gamma two-stage models
- Exposure offset support for Poisson frequency predictions
- `SolvencyCapitalEstimator` at α=0.005 with one-sided joint upper bounds
- `coverage_report()` and `compare_methods()` diagnostics
- 198 tests, MIT-licensed, Python 3.10+

```bash
pip install insurance-conformal
```

Source at [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal).

---

## See also

- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — univariate conformal prediction for single-output insurance models. If you have a Tweedie model or a single combined GBM, start here. [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — controls expected monetary loss rather than coverage probability. PremiumSufficiencyController bounds E[shortfall/premium] ≤ α. Composable with joint intervals: use `insurance-conformal` for the joint set, then `PremiumSufficiencyController` on the upper severity bound. [Coverage Is the Wrong Guarantee for Pricing Actuaries](https://burning-cost.github.io/2026/03/25/insurance-conformal-risk/)
- [`insurance-evt`](https://github.com/burning-cost/insurance-evt) — when the tail is the point. Extreme value theory for XL layer pricing and return level estimation. EVT models the marginal tail; multivariate conformal quantifies joint uncertainty across outputs. [Extreme Value Theory for UK Motor Large Loss Pricing](https://burning-cost.github.io/2026/03/25/insurance-evt/)
