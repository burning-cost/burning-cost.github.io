---
layout: post
title: "Calibration Testing That Goes Beyond the Residual Plot"
date: 2026-03-09
categories: [techniques, libraries, pricing]
tags: [calibration, balance-property, murphy-decomposition, auto-calibration, glm, gbm, poisson, python, insurance-monitoring]
description: "The full calibration framework for insurance pricing: balance property test, auto-calibration by price cohort, and Murphy decomposition (UNC/DSC/MCB/GMCB/LMCB). The companion post Recalibrate or Refit focuses on using the Murphy split operationally."
---

There is a class of pricing model error that does not show up in the Gini coefficient. The model ranks risks correctly (low-risk policies get lower prices than high-risk ones), but the price levels are wrong. The aggregate predicted premium is 7% below actual claims cost. The GBM discriminates well but miscalibrates consistently, and nobody has put a test in the validation pipeline that would catch it.

This happens because most validation pipelines stop at discrimination. Gini, lift chart, double lift, maybe a decile plot. These all measure whether the model ranks risks correctly. None of them measure whether it prices them at the right level. A model can pass all of these tests and still produce a portfolio loss ratio 10 points worse than the actuarial assumption, because the prediction scale is off in a way that the ranking metrics do not penalise.

The balance property, the requirement that sum(predicted claims) = sum(actual claims) across the portfolio, is the minimal calibration requirement. It is also the one most commonly omitted from validation frameworks, partly because scikit-learn's calibration module handles binary classification only, and partly because the actuarial tools that do exist (R's `actuaRE`, for instance) are tightly coupled to their own model objects. There has been no Python library that took raw prediction arrays, handled exposure weighting, and ran the Poisson/Gamma/Tweedie deviance framework that insurance data requires.

[`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) is that library. Based on Lindholm & Wüthrich (SAJ 2025) and Brauer et al. (arXiv:2510.04556, 2025). Seven modules, 99 tests, v0.1.0 on PyPI.

```bash
uv add insurance-monitoring
```

---

## Three questions before sign-off

The library answers three ordered questions. Each is a stronger condition than the previous.

**1. Is the model globally unbiased?** Does sum(predicted) equal sum(actual), exposure-weighted? This is the balance property: the weakest and most commercially important calibration requirement.

**2. Is each price cohort self-financing?** Do the high-risk predictions actually correspond to high observed claims, at the right level? A model can be globally balanced but have systematic cross-subsidisation between segments. This is auto-calibration.

**3. Does the miscalibration come from levelling or structure?** If the model is miscalibrated, is it because the overall scale is wrong (cheap to fix: multiply all predictions by a constant), or because the model's shape is wrong (expensive to fix: needs a refit)? This is the Murphy decomposition.

The answers drive different remediation responses. Getting the question right saves model development time.

---

## The balance property test

```python
import numpy as np
from insurance_monitoring.calibration import check_balance

# Synthetic motor frequency data
rng = np.random.default_rng(42)
n = 50_000
exposure = rng.uniform(0.5, 2.0, n)
y_hat = rng.gamma(2, 0.05, n)          # predicted frequency
# Simulate a model that under-predicts by ~8%
y = rng.poisson(exposure * y_hat * 1.08) / exposure

result = check_balance(y, y_hat, exposure, distribution='poisson', seed=42)

print(f"Balance ratio: {result.balance_ratio:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
print(f"Balanced: {result.is_balanced}")
print(f"Observed total: {result.observed_total:,.0f} claims")
print(f"Predicted total: {result.predicted_total:,.0f} claims")
```

The balance ratio `alpha = sum(v_i * y_i) / sum(v_i * y_hat_i)` should be 1.0. Values above 1.0 mean the model is under-predicting — it is charging less than claims will cost. In this example alpha is approximately 1.08: every premium is 8% too cheap on average.

The p-value uses a Poisson z-test. Under H0, the observed claim count is approximately Normal(predicted, predicted), giving a z-statistic. This is the appropriate test for frequency data. The 95% confidence interval comes from 999 bootstrap resamples of the (y, y_hat, exposure) triplets — it preserves the joint dependence structure.

A model with a balance ratio of 1.08 and a CI of [1.06, 1.10] is not borderline. It is definitively charging the wrong premium. The CI tells you how certain that conclusion is.

---

## Auto-calibration: the stronger test

Global balance can be satisfied even when the model is systematically wrong within segments. Consider a model that under-prices high-risk policies by 15% and over-prices low-risk policies by 15%. The totals cancel, so the balance test passes. But high-risk cohorts are subsidised by low-risk ones — a pricing equity failure and a retention risk (low-risk customers will be shopped away by competitors who price them more accurately).

Auto-calibration requires E[Y | mu_hat(X)] = mu_hat(X) at every level of the prediction distribution, not just in aggregate. GLMs with canonical links satisfy this on training data by construction — the score equations enforce it. GBMs and neural networks do not. If you have switched from GLM to GBM without adding a recalibration step, you should run this test.

```python
from insurance_monitoring.calibration import check_auto_calibration

result = check_auto_calibration(
    y, y_hat, exposure,
    distribution='poisson',
    n_bins=10,
    method='bootstrap',   # Algorithm 1 from Brauer et al. (2025)
    seed=42,
)

print(f"p-value: {result.p_value:.4f}")
print(f"Calibrated: {result.is_calibrated}")
print(f"Worst bin deviation: {result.worst_bin_ratio:.3f}")  # e.g. 0.22 = worst cohort is 22% mis-priced
print(result.per_bin)
```

The `per_bin` output is a Polars DataFrame with one row per prediction decile: predicted mean, observed mean, ratio, exposure, and policy count. This is the reliability diagram in tabular form — the first thing a reviewing actuary will want to see.

The bootstrap test (default, recommended) implements Algorithm 1 of Brauer et al. (2025): simulate datasets under H0 (y ~ Poisson(y_hat)), compute the MCB statistic on each, and compare to the observed MCB. The p-value is the fraction of bootstrap MCB values that exceed the observed MCB. This is more powerful than the Hosmer-Lemeshow alternative and does not depend on bin count choices.

One calibration decision worth flagging: Brauer et al. (2025) recommend significance_level = 0.32 for routine monitoring, not 0.05. At alpha = 0.05, a one-standard-deviation deterioration in calibration has low detection probability. The economically important error is failing to detect real miscalibration, not triggering unnecessary reviews. At 0.32, there is roughly 50% power to detect a one-SD shift. Use 0.05 for initial model validation at launch; use 0.32 for quarterly monitoring.

```python
# For ongoing monitoring, use the Brauer et al. recommendation
monitoring_result = check_auto_calibration(
    y_new_quarter, y_hat_new_quarter, exposure_new_quarter,
    significance_level=0.32,
    seed=0,
)
```

---

## The Murphy decomposition: levelling error versus structural error

When the model fails calibration, the next question is what to do about it. The Murphy decomposition answers this.

Every deviance score decomposes as:

```
D(y, y_hat) = UNC − DSC + MCB
```

- **UNC** (Uncertainty): the baseline deviance from an intercept-only model (the grand mean). This is determined by the data, not the model. You cannot reduce UNC.
- **DSC** (Discrimination): how much better the model is than the grand mean, after correcting calibration. This is essentially the model's Gini expressed as a deviance reduction. High DSC means good ranking.
- **MCB** (Miscalibration): the excess deviance from wrong price levels, independent of ranking. A well-calibrated model has MCB close to zero.

MCB splits further:

- **GMCB** (Global MCB): the portion of miscalibration removable by multiplying all predictions by a single constant. Cheap to fix.
- **LMCB** (Local MCB): residual miscalibration after balance correction. Requires model refit or isotonic recalibration.

```python
from insurance_monitoring.calibration import murphy_decomposition

murphy = murphy_decomposition(y, y_hat, exposure, distribution='poisson')

print(f"Total deviance:    {murphy.total_deviance:.4f}")
print(f"Uncertainty (UNC): {murphy.uncertainty:.4f}")
print(f"Discrimination:    {murphy.discrimination:.4f}  ({murphy.discrimination_pct:.1f}% of total)")
print(f"Miscalibration:    {murphy.miscalibration:.4f}  ({murphy.miscalibration_pct:.1f}% of total)")
print(f"  Global MCB:      {murphy.global_mcb:.4f}  <- fixable by recalibration")
print(f"  Local MCB:       {murphy.local_mcb:.4f}  <- requires refit")
print(f"\nVerdict: {murphy.verdict}")
```

The verdict logic is:

- `MCB / UNC < 1%` and `DSC > 0`: **OK**: calibration is acceptable.
- `GMCB >= LMCB`: **RECALIBRATE**: the dominant error is a global scale shift. Multiply predictions by the balance ratio and redeploy. Model structure is sound.
- `LMCB > GMCB`: **REFIT**: the error is in the model's shape, not its level. A scalar correction will not fix it. The model needs rebuilding or isotonic recalibration on a large holdout sample.

This is the decision the decomposition is designed for. "RECALIBRATE" costs an afternoon. "REFIT" costs weeks and a governance cycle. Getting the diagnosis wrong in either direction is expensive. The post [Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question](/2026/05/14/recalibrate-or-refit/) goes deeper on the operational use of this split: what it looks like in practice when GMCB dominates versus LMCB, common team mistakes when using A/E alone, and the governance documentation pattern for PRA SS1/23.

---

## Rectification

Once you know the type of miscalibration, the library applies the correction.

**Multiplicative correction** (the RECALIBRATE case):

```python
from insurance_monitoring.calibration import rectify_balance

# Simple: multiply all predictions by sum(v*y) / sum(v*y_hat)
y_hat_corrected = rectify_balance(y_hat, y, exposure, method='multiplicative')
```

This is a single scalar multiplication. It restores global balance exactly and does not change the model's ranking or relative price structure. Run `check_balance` on the corrected predictions to confirm alpha = 1.0.

**Affine correction**: for when there is also a slope error:

```python
# Fits log(mu) = beta_0 + beta_1 * log(y_hat), minimising Poisson deviance
# Corrects both global level (beta_0) and scale responsiveness (beta_1)
y_hat_affine = rectify_balance(y_hat, y, exposure, method='affine', distribution='poisson')
```

Affine correction is the more general form from Lindholm & Wüthrich (2025). If beta_1 != 1, the model is not just offset; it is over- or under-responsive to risk factors. A beta_1 < 1 means the model's predictions are too compressed: high-risk policies are under-charged even after balance correction, and low-risk policies are over-charged. This is common in GBMs with low learning rates on gradient-boosted frequency models.

**Isotonic recalibration**: for the REFIT case when a full refit is not yet possible:

```python
from insurance_monitoring.calibration import isotonic_recalibrate

# Preserves ranking; corrects price levels empirically
# Only meaningful on holdout data — trivially achieves in-sample calibration
y_hat_recal = isotonic_recalibrate(y, y_hat, exposure)
```

Isotonic recalibration finds the order-preserving function that minimises the weighted distance to actuals. It is the most flexible correction, but it uses the holdout data to fit the correction — apply it to new data with a Wüthrich & Ziegel (SAJ 2024) complexity warning in mind. If the number of isotonic steps exceeds sqrt(n), the library raises a warning: the recalibration is fitting noise, not signal.

---

## Monitoring pipeline

The `CalibrationChecker` class wraps the three tests into a fit/check monitoring workflow:

```python
from insurance_monitoring.calibration import CalibrationChecker

checker = CalibrationChecker(
    distribution='poisson',
    alpha=0.32,        # Brauer et al. recommendation for monitoring
    n_bins=10,
    bootstrap_n=999,
)

# Fit on the launch holdout (establishes baseline)
checker.fit(y_holdout, y_hat_holdout, exposure_holdout, seed=0)

# Run each quarter
report = checker.check(y_q1, y_hat_q1, exposure_q1, seed=0)
print(report.verdict())    # 'OK' | 'MONITOR' | 'RECALIBRATE' | 'REFIT'
print(report.summary())    # human-readable paragraph
df = report.to_polars()    # one-row Polars DataFrame for logging
```

The `to_polars()` output is a one-row DataFrame with all diagnostic values: balance ratio, balance CI bounds, auto-calibration p-value, Murphy components, and the verdict string. Write it to a monitoring table in your data warehouse each quarter. When the verdict changes from OK to RECALIBRATE, you have a timestamp, a balance ratio, and the Murphy decomposition showing why.

---

## The GLM versus GBM distinction

This is worth being explicit about. GLMs with canonical links (Poisson with log link, Gamma with log link) satisfy the auto-calibration property on training data as a mathematical consequence of the score equations. If `sum(predicted) == sum(actual)` on training data, it is not a coincidence; it is a constraint the optimiser enforced.

GBMs do not have this property. A gradient-boosted Poisson model minimises Poisson deviance but does not enforce the calibration constraint at the prediction level. The holdout predictions will typically have a balance ratio different from 1.0, and the auto-calibration test will often reject even on well-developed models, because the GBM has fitted the residuals from an implicitly miscalibrated base rather than enforcing calibration at each step.

This is not a reason to avoid GBMs. It is a reason to add a recalibration step after any GBM training, and to check the result. The multiplicative balance correction is almost always sufficient for the GMCB component. Whether LMCB is significant enough to warrant concern depends on the Murphy decomposition.

A canonical GLM that passes into a GBM pipeline without calibration checking at the GBM stage will typically lose its calibration guarantee. The GLM was calibrated; the GBM trained on top of it may not be.

---

## What this replaces

The standard alternative is a residual plot: actual versus predicted, often in deciles, checked visually. This catches gross failures but is insensitive to modest systematic miscalibration. A 7% global balance error is unambiguous in the data; it is often invisible in a decile residual plot because the axis scale is set by the prediction range, not the residual range.

The Hosmer-Lemeshow test (already in `insurance-governance`) catches auto-calibration failures but without the MCB decomposition. It tells you the model is miscalibrated. It does not tell you whether the fix is a scalar correction or a model refit.

The Murphy decomposition is the piece that was missing from Python tooling for insurance. The academic framework has existed since Brier score decompositions were formalised; its application to the insurance Poisson-deviance setting is from Brauer et al. (arXiv:2510.04556, 2025), published October 2024. The library implements it precisely: UNC, DSC, MCB, GMCB, LMCB, with the GMCB/LMCB split via isotonic recalibration on the balance-corrected predictions.

---

## Install

```bash
uv add insurance-monitoring
```

Python 3.10+. Dependencies: numpy, scipy >= 1.12, polars, matplotlib. No scikit-learn required; isotonic regression uses scipy.stats.isotonic_regression (scipy >= 1.12) with a pure numpy PAVA fallback.

[`insurance-monitoring` on GitHub](https://github.com/burning-cost/insurance-monitoring) — MIT licence. 99 tests, 7 modules.

Pairs with [`insurance-governance`](https://github.com/burning-cost/insurance-governance) for the discrimination side: Gini, double lift, Lorenz curve, PRA SS1/23 report generation. The two libraries answer different questions. Validation tells you whether the model ranks risks correctly. Calibration tells you whether it prices them at the right level. You need both.

---

**References:**

1. Lindholm, M. and Wüthrich, M.V. (2025). "The Balance Property in Insurance Pricing." *Scandinavian Actuarial Journal*. DOI: 10.1080/03461238.2025.2552909
2. Brauer, A., Denuit, M., Krvavych, Y. et al. (2025). "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing." arXiv:2510.04556
3. Wüthrich, M.V. and Ziegel, J. (2024). "Isotonic Recalibration under a Low Signal-to-Noise Ratio." *Scandinavian Actuarial Journal*, 2024(3), 279–299.

---

**Related articles from Burning Cost:**
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)
- [Champion/Challenger Testing with ICOBS 6B.2.51R Compliance](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/)
