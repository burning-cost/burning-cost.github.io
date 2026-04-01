---
layout: post
title: "Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims"
date: 2026-04-01
categories: [techniques, pricing]
tags: [conformal-prediction, MOPI, ShapeAdaptiveCP, conditional-coverage, MSCE, skewed-distributions, Tweedie, group-conditional, masked-Z, GDPR, FCA, insurance-conformal, arXiv-2603-23374, Bao-Zhang-Wang-Ren-Zou, minimax, Adam, CQR, Mondrian, python]
description: "Standard conformal prediction gives symmetric intervals calibrated on average. For right-skewed claim distributions, that average includes a lot of zero claims pulling thresholds down, and the intervals are wrong exactly where it matters most. ShapeAdaptiveCP in insurance-conformal v0.9.0 implements MOPI (Bao et al., arXiv:2603.23374), which calibrates per-group thresholds via minimax optimisation to achieve near-uniform conditional coverage across rating cells — with a critical feature for UK pricing: the conditioning variable can be masked at deployment."
math: true
author: burning-cost
---

Standard split conformal prediction gives you marginal coverage: over a long run of predictions, 90% of your intervals will contain the true value. For a motor frequency model fitted on a balanced population, this is workable. For a Tweedie severity model with a mass of zero claims and a heavy right tail — which is what UK property and motor claims actually look like — the marginal guarantee is nearly useless. It holds on average across a distribution dominated by small claims, while your intervals are systematically wrong for the large risks that drive your combined ratio.

This is not a subtle failure mode. It is structural. And `insurance-conformal` v0.9.0 ships a fix.

---

## The symmetric interval problem

Split conformal prediction calibrates a scalar threshold $\hat{q}$ on nonconformity scores over a held-out calibration set. For regression, the standard score is $s_i = |y_i - \hat\mu_i|$. The prediction interval is $[\hat\mu - \hat{q}, \hat\mu + \hat{q}]$ — symmetric around the point prediction.

Two things go wrong in insurance.

**The single threshold folds together heterogeneous risk groups.** Your calibration set contains young drivers, mature drivers, urban postcodes, rural postcodes, high-value vehicles, fleet risks. Each group has a different claim distribution. A single $\hat{q}$ satisfying 90% marginal coverage will overcoverage some groups and undercoverage others. If young drivers account for 8% of your calibration set but 30% of your large claims, the threshold calibrated on the pooled distribution will undercoverage young drivers and overcoverage the bulk of policyholders who rarely claim.

**The symmetric form is wrong for skewed distributions.** A Tweedie claim prediction $\hat\mu = £2,400$ is not the centre of a symmetric distribution. The true distribution is right-skewed: the lower bound cannot go below zero, the upper tail extends to £50k+ for large bodily injury claims. A symmetric interval of $[£-1,600, £6,400]$ truncates at zero, wastes coverage on the lower tail, and is too narrow at the top where the real exposure is.

Mondrian conformal prediction — the standard fix for the first problem — partitions your calibration set per group and calibrates a separate threshold in each cell. This works when groups have enough calibration data. With K=20 rating cells and $n_{\text{cal}}=1{,}000$, you have roughly 50 observations per cell. That is barely enough to estimate a 90th percentile quantile, and any sparse cell (young male drivers in rural postcodes) will produce wildly unstable thresholds.

---

## What MOPI does differently

Bao, Zhang, Wang, Ren and Zou (arXiv:2603.23374, March 2026) reframe calibration as a minimax optimisation problem. Instead of choosing one threshold, or K independent thresholds from K independent subsets, you solve:

$$\min_h \max_{f \in \mathcal{F}} \; \mathbb{E}\left[ f(Z)\left(\mathbf{1}\{Y \notin C(X; h)\} - \alpha\right) - f(Z)^2 \right]$$

The inner maximisation over weight functions $f$ finds the **worst-case subgroup**: the combination of miscoverage deviations across groups that is hardest to equate. The outer minimisation over thresholds $h$ forces per-group coverage to be as uniform as possible under that adversarial weighting.

For finite categorical groups (rating cells, age bands, vehicle classes), Lemma 3.1 of Bao et al. shows the inner max has a closed form:

$$\max_{f \in \mathcal{F}} = \sum_k \frac{\delta_k^2}{4 n_k}$$

where $\delta_k = \sum_{i \in \text{group } k} (\mathbf{1}\{y_i \notin C(x_i; h)\} - \alpha)$ is the miscoverage deviation for group $k$. This is exactly the mean-squared conditional coverage error (MSCE) — the right metric for fair prediction. The outer loop then optimises $h$ (a vector of K thresholds) to minimise this expression.

All K thresholds are optimised jointly on the full calibration set. Groups with sparse calibration data borrow strength from the structure of the objective rather than being estimated in isolation. This is why MOPI outperforms Mondrian when cells are thin.

The implementation uses Adam with 200 steps and a smoothed sigmoid surrogate for the indicator function (temperature 0.1 — tight enough to approximate the indicator well, stable enough to differentiate through). No external ML framework required: the whole algorithm is pure numpy.

---

## The masked Z feature

This is what makes MOPI directly relevant to UK pricing rather than just academically interesting.

Standard conditional calibration methods — CQR, Mondrian, and the conditional calibration (CC) framework of Gibbs et al. — require the conditioning variable $Z$ at prediction time. If $Z$ is age band or gender, you need those features when generating intervals at deployment.

For UK pricing, this is a problem. The FCA's rules on indirect discrimination, and GDPR constraints on using protected characteristics in automated decisions, mean you often want prediction intervals that do not explicitly condition on age or gender at the point of quote. But you still want the interval to reflect the actual risk composition of that rating cell.

MOPI decouples calibration from prediction. $Z$ enters only the calibration objective — forcing the K thresholds to achieve equitable group coverage — but the prediction function $C(X; h)$ depends only on $X$. At deployment, $Z$ is never seen.

In `ShapeAdaptiveCP`, this is implemented via `group_fn`: a function that maps test features $X$ to group labels without requiring $Z$ directly. You calibrate with age bands in $Z_{\text{cal}}$, and at deployment you supply `group_fn=lambda X: np.digitize(X[:, age_col], age_bins) - 1` — which derives the age band from age (a feature already in your rating model) without treating gender or protected characteristics as explicit inputs to the interval computation.

---

## API walkthrough: Tweedie motor model

```python
import numpy as np
from insurance_conformal import ShapeAdaptiveCP

# Fit a Tweedie GLM (or GBM) on training data
# glm.predict(X) -> E[Y|X], using Tweedie link
# np.sqrt(glm.predict(X)) -> proxy for scale (Var[Y|X] ~ mu^p for Tweedie)

# --- Calibration ---
# Z_cal: age bands [0, 1, 2, 3] from rating model (young -> mature)
# Present at calibration: used to equalise coverage across age groups

cp = ShapeAdaptiveCP(
    score_fn="standardised",  # s_i = |y_i - mu_i| / sigma_i
    alpha=0.1,                # target 90% conditional coverage
    mode="group",             # finite categorical Z
    n_iter=200,
    lr=0.01,
)
cp.calibrate(
    X_cal, y_cal, Z_cal,
    mu_cal=glm.predict(X_cal),
    sigma_cal=np.sqrt(glm.predict(X_cal)),
)

print(cp)
# ShapeAdaptiveCP(mode='group', alpha=0.1, score_fn='standardised',
#                 calibrated on 2000 obs, msce=0.0008)

# Per-group coverage on calibration set (diagnostic only — biased)
print(cp.coverage_by_group_)
# {0: 0.921, 1: 0.908, 2: 0.896, 3: 0.901}

# --- Prediction (masked Z) ---
# age_col: column index of age in X_test
# age_bins: the band breakpoints used at calibration (17, 25, 35, 50, 99)
lower, upper = cp.predict(
    X_test,
    mu_test=glm.predict(X_test),
    sigma_test=np.sqrt(glm.predict(X_test)),
    group_fn=lambda X: np.digitize(X[:, age_col], [25, 35, 50]),
)

# Honest evaluation on held-out test set
report = cp.coverage_report(y_test, lower, upper, Z_test)
# shape: (4, 5) — group, n_obs, coverage, target_coverage, miscoverage_deviation
msce_test = cp.msce(y_test, lower, upper, Z_test)
```

The `score_fn="standardised"` choice — $s_i = |y_i - \hat\mu_i| / \hat\sigma_i$ — is what makes the intervals asymmetric in practice. The Tweedie model produces predictions $\hat\mu$ and scale estimates $\hat\sigma = \sqrt{\hat\mu}$ that vary substantially across risk groups. The MOPI-calibrated threshold $h_k$ (learned per group) is then applied in units of standard deviations: the interval is $[\hat\mu - h_k \hat\sigma, \hat\mu + h_k \hat\sigma]$. For a high-risk young driver with $\hat\mu = £4{,}000$ and $\hat\sigma = £63$, the threshold may be wider than for a low-risk mature driver with $\hat\mu = £900$. The shape adapts not through asymmetric bounds directly, but through the group-specific scaling of the standardised score. This is the "shape-adaptive" mechanism: the interval width is driven by both the predicted scale and the group-specific calibration.

---

## Comparing to the existing methods in the library

The `insurance-conformal` library now has four approaches to interval calibration, and they address different problems:

| Method | Conditional on | Handles sparse groups? | Z masked at deployment? |
|---|---|---|---|
| Split conformal (CQR) | Nothing — marginal only | N/A | N/A |
| Mondrian | Exact group partition | No — independent per cell | Yes, if group_fn provided |
| CC / RLCP | Continuous or group Z | Partially — kernel smoothing | No — Z required at test time |
| `ShapeAdaptiveCP` (MOPI) | Group or continuous Z | Yes — joint calibration | Yes — by design |

The empirical result from Bao et al. (Table 1, 100 replications, 90% nominal coverage) for worst-case coverage across groups: MOPI 0.856, CC 0.840, RLCP 0.837, split conformal 0.813. MSCE: MOPI 0.037, CC 0.044, SCP 0.061. These are not marginal improvements — the gap between MOPI and split conformal on MSCE is 40%.

The gap between MOPI and Mondrian is harder to quote from the paper (Mondrian is not in their Table 1) but the argument is structural: Mondrian at K=20 groups needs $n_{\text{cal}}/K$ observations per cell for stable threshold estimation, while MOPI uses all $n_{\text{cal}}$ points jointly. For UK personal lines rating models with 15–30 meaningful age bands and 2,000 calibration claims, MOPI's joint calibration is the practical choice.

---

## RKHS mode for continuous Z

For continuous conditioning (age as a number rather than a band, or a composite risk score), `mode="rkhs"` uses a Gaussian RBF kernel to interpolate thresholds across the $Z$ space:

```python
cp_rkhs = ShapeAdaptiveCP(
    score_fn="standardised",
    alpha=0.1,
    mode="rkhs",
    kernel_bandwidth=0.5,  # in units of Z
    gamma=1e-3,            # Cholesky regularisation
)
cp_rkhs.calibrate(X_cal, y_cal, Z_cal, mu_cal=mu_cal, sigma_cal=sigma_cal)
lower, upper = cp_rkhs.predict(X_test, mu_test=mu_test, sigma_test=sigma_test,
                               Z_test=Z_test)
```

RKHS mode is $O(n^3)$ at calibration (one Cholesky factorisation of the $n \times n$ kernel matrix). With $n_{\text{cal}} = 2{,}000$ this is roughly 8M matrix entries and takes a few seconds on CPU — feasible for offline calibration, not for real-time quote paths. Use group mode for production and RKHS for research and validation.

---

## The diagnostic loop

The MSCE reported in `cp.msce_` after calibration is computed on the calibration set and is biased downward — the optimisation minimised it on that data. For honest evaluation:

```python
# Use a separate test set, never the calibration set
msce_honest = cp.msce(y_test, lower_test, upper_test, Z_test)
report = cp.coverage_report(y_test, lower_test, upper_test, Z_test)
```

`coverage_report()` returns a Polars DataFrame with one row per group: group label, observation count, empirical coverage, target coverage, and the deviation. A well-calibrated predictor should show deviations within roughly $\pm 0.02$ for groups with 100+ observations. Groups with coverage systematically below target (negative deviation) are the ones that would have generated FCA complaints under a marginal-only calibration.

---

## What this is not

MOPI solves conditional coverage, not interval efficiency. The intervals it produces are still symmetric in the standardised score space — the asymmetry comes from the heteroscedastic scale model, not from asymmetric bound construction. Conformalized quantile regression with asymmetric pinball scores (CQR) can produce genuinely asymmetric intervals, but CQR cannot condition on groups in the MOPI sense without falling back to Mondrian.

For Tweedie severity specifically, the right combination is: fit a Tweedie GLM or GBM to get $\hat\mu$ and $\hat\sigma$, use MOPI standardised scores for the conditional coverage guarantee, and accept that the asymmetry is driven by the heteroscedastic model rather than by asymmetric conformal bounds. For distributions where you have a good scale model, this is sufficient. For distributions where the residual after standardisation is itself heavily skewed — some large commercial risks, heavy-tailed marine cargo — a custom `score_fn` (callable, taking `y, mu, sigma`) lets you experiment with skewed nonconformity scores.

---

## Getting it

`ShapeAdaptiveCP` is exported from `insurance_conformal` in v0.9.0:

```python
from insurance_conformal import ShapeAdaptiveCP
```

Source: `src/insurance_conformal/mopi.py`. Tests: `tests/test_mopi.py` (40 tests). The full minimax derivation is in Bao et al. (arXiv:2603.23374); the algorithm docstring in `mopi.py` walks through the gradient computation step by step.

---

## The paper

Bao, Yajie, Chuchen Zhang, Zhaojun Wang, Haojie Ren, and Changliang Zou. "Shape-Adaptive Conditional Calibration for Conformal Prediction via Minimax Optimization." arXiv:2603.23374 [stat.ML]. March 24, 2026.

---

## Related posts

- [Two Ways to Control Risk in Automated Underwriting: Conditional vs Marginal](/techniques/underwriting/2026/04/01/selective-conformal-prediction-automated-underwriting-conditional-vs-marginal-risk/) — SelectiveConformalRC and SCoRE for STP triage
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/techniques/pricing/uncertainty/2026/03/13/insurance-conformal-risk/) — conformal risk control and the `insurance-conformal` library overview
- [Conformal Prediction for Insurance Pricing: Intervals, Risk Control, and the Practical Toolkit](/techniques/pricing/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — when conformal prediction works, when it does not, and what calibration data you actually need
- [Conditional Coverage in Conformal Prediction: Model Selection with CVI](/techniques/pricing/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — checking whether your conformal guarantees hold conditionally, not just marginally
