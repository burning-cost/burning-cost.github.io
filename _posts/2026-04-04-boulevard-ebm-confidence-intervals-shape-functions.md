---
layout: post
title: "Confidence Intervals for EBM Shape Functions — Boulevard Regularisation for Insurance Pricing"
date: 2026-04-04
categories: [techniques, model-governance]
tags: [EBM, GAM, interpretable-ml, uncertainty, confidence-intervals, insurance-gam, model-validation, PRA, pricing, python]
description: "Boulevard regularisation turns EBM shape functions into kernel ridge regression estimates, giving valid CLT-based confidence intervals. We show why this matters for model governance and how to use it with insurance-gam v0.5.1."
---

A rating factor that nobody can show is statistically distinguishable from zero is a governance liability. EBMs give you shape functions — smooth, interpretable curves showing how each feature contributes to the premium — but standard EBMs give you no uncertainty. You cannot tell from the fitted curve whether the age-25 dip is real structure or an artefact of three thin accident years. Boulevard regularisation fixes this.

Fang, Tan, Pipping-Gamon and Hooker (AISTATS 2026, arXiv:2601.18857) prove that a moving-average boosting variant causes EBM shape functions to converge to a kernel ridge regression (KRR) solution, and that KRR has a known asymptotic distribution. The result is analytical confidence intervals for individual shape functions — no bootstrap, no simulation, no resampling. We have implemented this in [`insurance-gam`](/insurance-gam/) as `BoulevardEBM` and `BoulevardInference`.

---

## Why actuarial teams need CIs on shape functions

Model governance for a deployed pricing model increasingly requires more than a Gini coefficient and a double-lift chart. PRA CP6/24 on insurance model risk expects evidence that rating factors have a defensible actuarial basis. The FCA's Consumer Duty (PS22/9) expects pricing to reflect genuine risk differentiation, not noise. "The model fitted it and the Gini went up" is not sufficient for either.

Shape function CIs answer three concrete governance questions:

**Factor validation.** At annual miles 3,000–5,000, the shape function says +0.08 log-premium. With a standard EBM you cannot say whether that is significant. With Boulevard, the 95% CI for that bin is [+0.03, +0.13] — excludes zero, the factor holds. At annual miles 12,000–14,000 the CI is [−0.02, +0.04] — does not exclude zero, this bin is flat. Document both in the sign-off pack.

**Thin data bands.** Young driver ages 17–19 always have sparse data on a UK motor book. The CI widens visibly there — exactly the communication pricing teams need when underwriting asks whether the young-driver loading is supported by the data or inferred from adjacent older bands.

**Model change reviews.** When you retrain on a new accident year, shape functions shift. Without CIs you cannot tell whether the shift is sampling noise or genuine trend. A shift outside the previous CI is evidence of real movement; within it is consistent with noise.

The `significance_table()` call does the first cut automatically — one row per feature, with a flag for whether any bin's CI excludes zero.

---

## What Boulevard regularisation does

Standard EBM boosting fits a stump to the residuals and adds it to the ensemble with a learning rate. The estimator converges, but to a convergence point with no simple analytical distribution. Bootstrapping is the only option for uncertainty quantification — InterpretML's bagging runs O(B × n × p × T) operations where B is the number of bags.

Boulevard replaces the additive update with a moving average:

```
f_b^(k) = ((b-1)/b) * f_{b-1}^(k) + (λ/b) * t̃_b^(k)
```

The 1/b harmonic decay causes successive stumps to contribute progressively less. The series converges to a KRR fixed point as boosting rounds grow. Fang et al. prove this via Robbins-Monro stochastic approximation theory, extending Zhou and Hooker (JMLR 2022). The KRR limit has a known asymptotic Gaussian distribution.

The computational win comes from EBM binning. Equal-frequency bins (at most 256 per feature) cause all kernel operations to factorise into m × m matrices. The influence norm for each bin — the diagonal of the hat matrix — is computed by inverting a single 256 × 256 matrix per feature. For n = 500,000 motor policies and 10 features: 10 independent 256 × 256 solves, a few milliseconds, independent of n. InterpretML's bagging at B = 100 bags costs roughly 50× more.

---

## The CLT result (Theorem 4.12)

Under Lipschitz GAM and stump geometry assumptions, Fang et al. show that each shape function estimate is asymptotically normal. The 95% CI for feature k at bin j is:

```
f̂^(k)(j)  ±  1.96 × σ̂ × sqrt(influence_norm(j) / n)
```

The influence norm is the diagonal element of the bin-space hat matrix, computed analytically from bin counts and λ. Bins with more data have smaller influence norms and narrower CIs. Thin age bands have large influence norms and wide CIs.

**The critical limitation:** Theorem 4.12 assumes homoscedastic Gaussian errors ε ~ N(0, σ²). Insurance frequency models use Poisson or Tweedie loss. The CIs from `BoulevardEBM` are not valid for those distributions. The defensible use for a frequency model is to fit on log-transformed pure premium — approximately Gaussian after transformation — or to use the CIs for exploratory factor analysis rather than formal statistical tests. We document this in the library; do not use these CIs for formal Poisson-inference claims.

---

## Code

```python
import numpy as np
import polars as pl
from insurance_gam.ebm import BoulevardEBM, BoulevardInference

# MSE loss on log(pure_premium) — the supported case
log_pp_train = np.log(pure_premium_train + 1e-6)

model = BoulevardEBM(
    n_estimators=1000,
    lambda_=1.0,       # tune by held-out MSE; 1.0–5.0 for large books
    max_bins=256,
    min_samples_leaf=20,
    random_state=42,
)
model.fit(X_train, log_pp_train)

inf = BoulevardInference(model)

# Shape function CI for driver age: bin_label | score | lower | upper | se
ci_df = inf.shape_ci("driver_age", alpha=0.05)

# Bins where CI excludes zero
print(ci_df.filter(
    (pl.col("lower") > 0) | (pl.col("upper") < 0)
))

# Feature-level significance: feature | max_absolute_score | ci_excludes_zero | max_se
print(inf.significance_table())

# Chart for governance pack — shape function + CI band + data density
ax = inf.plot_shape("driver_age", alpha=0.05, show_data_density=True)
ax.figure.savefig("driver_age_shape_ci.png", dpi=150)
```

The `plot_shape` output is the chart for your sign-off pack: shape function in blue with a shaded 95% CI band, grey density bars on the right axis. The CI widens where data are sparse. A governance committee member can read it without knowing what kernel ridge regression is.

---

## Limitations that actually matter

**Sigma inflation.** `sigma_hat_` is estimated from training residuals, which include KRR approximation error as well as noise. In practice this inflates sigma by roughly 2× on synthetic benchmarks, making the CIs conservative. A factor whose CI excludes zero under the conservative estimate is genuinely robust — this is the right direction to be wrong.

**Interaction terms get no CIs.** Pairwise interaction features — vehicle age × driver age being the structurally important one for UK motor — are not covered. The CI on the driver age main effect does not speak to the interaction contribution.

**Lambda needs tuning.** The paper gives no closed-form selector. Run held-out MSE across a log-space grid from 0.1 to 100; pick the minimum on 20% withheld data. Values of 1.0 to 5.0 work for books above 50,000 policies; push to 10.0 for thin books below 10,000. A `cv_lambda()` classmethod is planned for v0.5.2.

**Small books.** The asymptotic CLT assumes n large enough for a Berry-Esseen approximation. Below roughly 5,000 training observations, treat the CIs as indicative.

---

## The governance workflow

1. Fit `BoulevardEBM` on log-scale pure premium with chosen lambda.
2. Run `significance_table()` and include in the validation pack. Any feature where `ci_excludes_zero` is False is a candidate for removal or consolidation.
3. Run `plot_shape()` for the top three or four features by `max_absolute_score`. The widening CI bands in sparse age bands are self-explaining to a non-technical committee.
4. For thin segments — annual mileage below 3,000, drivers above 75 — show bin-level CIs from `shape_ci()` and document that those factor loads reflect limited data, not confident estimates.

This is the substance behind "statistically defensible rating factors". The shape function CI answers "how do you know that loading is real?"

---

## Verdict

Boulevard regularisation gives EBM pricing models something standard boosting cannot: confidence intervals on individual rating factors without bootstrapping. Asymptotically valid under Gaussian errors, conservative in practice, main effects only. For UK personal lines pricing with large books (>50,000 policies) and log-scale MSE fitting, they are usable in model governance documentation today.

The Poisson/Tweedie extension needs new theory. We are watching github.com/hetankevin/ebm-inference for updates.

```bash
uv add insurance-gam
```

Source and tests at [GitHub](https://github.com/burning-cost/insurance-gam). Boulevard implementation: `src/insurance_gam/ebm/_boulevard.py`. Benchmark notebook: `notebooks/boulevard_demo.py`.

Reference: Fang, R., Tan, Z., Pipping-Gamon, T. & Hooker, G. (2026). 'Statistical Inference for Explainable Boosting Machines.' AISTATS 2026. arXiv:2601.18857.

- [Does Monotonicity-Constrained EBM Actually Work for Insurance Pricing?](/2026/03/28/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/)
- [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/)
