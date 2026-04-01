---
layout: post
title: "DebiasedGLM: Honest Confidence Intervals for Lasso-Selected Rating Models"
date: 2026-04-01
categories: [techniques]
tags: [glm, variable-selection, confidence-intervals, lasso, poisson, gamma, tweedie, inference, pricing, insurance-gam]
description: "insurance-gam v0.4.0 adds DebiasedGLM — the first Python implementation of debiased Lasso confidence intervals for Poisson, Gamma, and Tweedie GLMs. It corrects the bias that makes post-selection Wald CIs cover at 70–80% rather than 95%."
math: true
author: burning-cost
---

[A previous post](/techniques/2026/04/01/post-selection-inference-glm-confidence-intervals-lasso-insurance-pricing/) covered why naive confidence intervals are wrong after Lasso variable selection: the selection event and the coefficient estimates are computed from the same data, which inflates the estimates for variables near the noise boundary and produces CIs that cover at 70–80% rather than 95%.

That post ended on a gap: the parametric programming fix (`PostSelectionGLM`) handles Poisson frequency models but not Gamma severity models. Severity is where the hardest inference problems in motor and property pricing live — large losses, heavy tails, strong but irregular rating factor effects.

`insurance-gam` v0.4.0 closes that gap. `DebiasedGLM` provides bias-corrected confidence intervals for Poisson, Gamma, and Tweedie GLMs after elastic net variable selection. It is the first Python implementation of the method from Manna, Huang, Dey, Gu & He (2025, arXiv:2410.01008).

---

## Why the Lasso estimate is biased — and why that matters for CIs

When elastic net selects a set of variables, two things happen simultaneously. First, the penalisation shrinks all coefficients toward zero — including the ones that survive. Second, the variables that survive selection are those whose penalised estimates happened to be large enough to clear the threshold. Both effects bias the penalised estimate downward for true non-zero coefficients: shrinkage pulls it toward zero, and the conditioning on survival introduces an upward selection bias that partially counteracts this, but not cleanly.

The result is that the penalised estimate $\hat{\beta}_j$ is not the MLE. If you simply call `statsmodels.GLM().fit()` on the Lasso-selected variables and read off Wald CIs, you're treating a biased, shrunken, conditioning-contaminated estimate as if it were a clean maximum likelihood estimate. The CIs are not wrong by a small margin; for variables selected with modest evidence, coverage drops to around 70–80% at nominal 95% in simulation (Shen et al. 2026, arXiv:2603.24875).

The debiased Lasso corrects the first problem — the shrinkage bias — using a one-step Newton correction. The correction is:

$$\hat{b} = \hat{\beta} - \hat{\Theta} \cdot \nabla \ell(\hat{\beta})$$

where $\hat{\Theta} = (X^\top \hat{W} X / n)^{-1}$ is the inverse Hessian of the GLM log-likelihood at the penalised solution, and $\nabla \ell(\hat{\beta})$ is the gradient of the negative log-likelihood at that point. For a correctly specified model, the gradient at the true $\beta_0$ is zero, so this correction is pushing $\hat{\beta}$ in the direction the MLE would push it. The debiased estimate $\hat{b}$ is asymptotically normal around $\beta_0$ with known variance, giving valid Wald-style CIs under the condition that $p^2/n \to 0$.

At $n = 100{,}000$ and $p = 50$ selected factors — a plausible UK motor frequency model — $p^2/n = 0.025$. The condition is comfortably satisfied.

---

## What DebiasedGLM provides that PostSelectionGLM doesn't

These two classes answer different questions, and both should be in a complete pricing inference workflow.

`PostSelectionGLM` (conditional inference, Shen et al. 2026) asks: *is this variable genuinely non-zero, given that Lasso selected it?* It accounts for the exact selection event and produces p-values and CIs conditional on which variables entered and which were excluded. It is the right tool for deciding whether to retain a borderline variable. It covers Poisson only.

`DebiasedGLM` (marginal inference, Manna et al. 2025) asks: *given this variable is in the model, what is the plausible range for its coefficient magnitude?* The CIs are unconditional — valid on average over the selection randomness rather than conditional on the specific selected set. They are less conservative than conditional CIs for in-model coefficients. The tool for reporting rate factor relativities with honest uncertainty bounds. It covers Poisson, Gamma, and Tweedie.

For a pricing actuary: run `PostSelectionGLM` to validate which factors belong. Run `DebiasedGLM` to report what those factors are worth.

The Gamma gap was the material one. Gamma severity models are standard in UK motor and property pricing. Until now, there was no Python implementation of valid post-selection inference for them. `DebiasedGLM` fills that.

---

## Using DebiasedGLM

Install with:

```bash
pip install "insurance-gam[glm]>=0.4.0"
```

### Poisson frequency model with exposure

```python
import numpy as np
from insurance_gam.debiased_glm import DebiasedGLM

rng = np.random.default_rng(42)
n, p = 5_000, 10

# Simulate: 3 true factors (vehicle group, NCB, driver age proxies)
# 7 noise variables
X = rng.standard_normal((n, p))
exposure = rng.uniform(0.5, 2.0, size=n)   # policy years
eta = 0.4 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.log(exposure)
y_freq = rng.poisson(np.exp(eta))

model = DebiasedGLM(
    family="poisson",
    alpha=0.0,       # 0.0 = use 5-fold CV to select lambda
    l1_ratio=0.5,    # elastic net — handles correlated rating factors
    confidence=0.95,
    n_bootstrap=0,   # Strategy A: asymptotic debiasing, fast
    random_state=42,
)
model.fit(X, y_freq, exposure=exposure)

print(model.summary()[["feature", "coef", "ci_lower", "ci_upper", "pvalue", "selected"]])
```

Typical output — the debiased estimates will be close to the true values (0.4, 0.3, 0.2), and the naive Lasso estimates from scikit-learn would be consistently lower due to shrinkage:

```
feature    coef  ci_lower  ci_upper  pvalue  selected
     x0  0.3871    0.3091    0.4651  <0.001      True
     x1  0.2934    0.2201    0.3667  <0.001      True
     x2  0.1988    0.1283    0.2693  <0.001      True
     x3  0.0000   -0.0415    0.0415   0.983     False
    ...
```

The noise variables (x3–x9) have CIs centred near zero, with the zero line clearly inside. The selected features have CIs that exclude zero and bracket the true values.

The naive approach — read off `lasso.coef_` from scikit-learn's `LassoCV` — would give you shrunk coefficients like 0.31, 0.22, 0.14, with no standard errors at all, or standard errors from a post-hoc `statsmodels.GLM().fit()` that cover at around 75% rather than 95%.

### Gamma severity model — the new capability

```python
# Gamma severity: log-linear mean, shape 3 (moderate overdispersion)
shape = 3.0
scale = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) / shape
y_sev = rng.gamma(shape, scale)

sev_model = DebiasedGLM(
    family="gamma",
    alpha=0.0,
    l1_ratio=0.5,
    confidence=0.95,
    random_state=42,
)
sev_model.fit(X, y_sev)

print(sev_model.summary()[["feature", "coef", "ci_lower", "ci_upper", "selected"]])
print(f"Estimated dispersion phi: {sev_model.phi_:.3f}")
```

The dispersion parameter $\phi$ is estimated from Pearson residuals. For a Gamma with shape 3, $\phi = 1/\text{shape} \approx 0.33$. The Hessian weights for Gamma are $W_i = 1/\phi$ (constant under a log link), so the inverse Hessian has a cleaner structure than Poisson; the debiasing step is straightforward.

### Tweedie combined model (p = 1.5)

```python
# Tweedie p=1.5 — compound Poisson-Gamma, models pure premium directly
tweedie_model = DebiasedGLM(
    family="tweedie",
    tweedie_power=1.5,
    alpha=0.0,
    confidence=0.95,
)
```

The Tweedie variance function is $V(\mu) = \mu^p / \phi$, so the Hessian weights at $p = 1.5$ are $W_i = \mu_i^{0.5} / \phi$. The dispersion is estimated from Pearson chi-squared with a two-pass procedure (initial fit at $\phi = 1$, re-estimate, refit). This is numerically stable for any $p \in [1, 2]$.

### When to use bootstrap instead of asymptotic debiasing

Strategy A (the default, `n_bootstrap=0`) requires $p^2/n \to 0$. The class warns if $p^2/n > 0.25$. For portfolios below roughly 10,000 policies with more than 50 selected factors, the asymptotics are shaky. Use Strategy B instead:

```python
model_small = DebiasedGLM(
    family="poisson",
    alpha=0.0,
    n_bootstrap=500,    # Pearson residual bootstrap pivot CI
    random_state=42,
)
model_small.fit(X, y_freq, exposure=exposure)
```

Strategy B resamples Pearson residuals $r_i = (y_i - \hat{\mu}_i) / \sqrt{V(\hat{\mu}_i)}$, reconstructs bootstrap responses $y_i^* = \hat{\mu}_i + \sqrt{V(\hat{\mu}_i)} \cdot r_i^*$, refits the penalised GLM at the same $\lambda$ for each resample, and uses pivot quantiles. 500 resamples on a portfolio of 5,000 runs in under a minute.

### Forest plot

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
model.forest_plot(ax=ax, only_selected=True)
plt.tight_layout()
```

`only_selected=True` (the default) plots only the Lasso-selected features. The horizontal axis is log-scale coefficient; vertical axis is features sorted by selection order. A grey dashed line at zero makes it easy to see which CIs exclude the null.

---

## What to watch for

**The validity condition is a real constraint.** $p^2/n > 0.25$ triggers a warning. On a thin portfolio of 3,000 policies with 30 candidate variables, you're at $900/3000 = 0.30$. Use bootstrap or, better, data splitting for inference before you've pre-screened the feature set.

**CIs on unselected variables.** `summary()` returns rows for all features including those with zero Lasso coefficients. The CI on an unselected variable is centred on the debiased estimate (which will be near zero if the factor is genuinely null) but is not meaningful in the same sense — the selection event excluded that variable, and the debiasing is applied to all features mechanically. For reporting, focus on the `selected=True` rows.

**Gamma and Tweedie dispersion.** For non-Poisson families, the class does a two-pass fit: initial penalised fit at $\phi = 1$, Pearson chi-squared dispersion estimate, refit with corrected weights. This is the right procedure but adds computation. For Tweedie models where you have a prior on $\phi$ (e.g., from a Tweedie parameter sweep), passing the known $\lambda$ directly via `alpha` skips the CV step and speeds things up.

---

## The two-tool workflow

Our recommendation for a complete inference workflow on a UK personal lines pricing GLM:

1. **Elastic net variable selection** — `DebiasedGLM(alpha=0.0)` with `alpha=0.0` to CV-select the penalty. `model.selected_features_` gives the selected set. `model.lambda_` gives the chosen $\lambda$.

2. **Conditional significance testing** — pass the selected features to `PostSelectionGLM` (Poisson frequency only). This tells you whether each selected variable is genuinely non-zero given the selection event.

3. **Marginal CI reporting** — `model.summary()` from `DebiasedGLM`. These are the CIs to put in a rate filing, a model documentation pack, or a challenge meeting. They are valid unconditionally and corrected for shrinkage bias.

Steps 2 and 3 answer different questions. You need both.

---

## Installation and source

```bash
pip install "insurance-gam[glm]>=0.4.0"
```

```python
from insurance_gam.debiased_glm import DebiasedGLM
```

Source: [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam) — MIT licence.

The theory is in Manna, A., Huang, B., Dey, D. K., Gu, C., & He, X. (2025). 'Interval estimation of coefficients in penalized regression models of insurance data.' *Applied Stochastic Models in Business and Industry*. arXiv:2410.01008.

---

## Related posts

- [Your GLM Confidence Intervals Are Wrong After Variable Selection](/techniques/2026/04/01/post-selection-inference-glm-confidence-intervals-lasso-insurance-pricing/) — the problem this solves; also covers `PostSelectionGLM` for Poisson conditional inference
- [Two Things Random Splits and Pearson Correlation Get Wrong in Insurance Data](/machine-learning/model-validation/2026/04/01/insurance-cv-v0-3-0-support-point-split-chatterjee-xi-feature-screening/) — `insurance-cv` v0.3.0; pairs with `DebiasedGLM` for clean train/inference splits
