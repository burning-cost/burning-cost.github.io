---
layout: post
title: "Your GLM Confidence Intervals Are Wrong After Lasso — Here Is the Bias-Corrected Fix"
date: 2026-04-01
categories: [techniques, pricing]
tags: [glm, lasso, elastic-net, variable-selection, confidence-intervals, post-selection-inference, bias-correction, tweedie, poisson, gamma, debiasing, residual-bootstrap, solvency-ii, consumer-duty, insurance-gam, arXiv-2410-01008, Manna, python]
description: "PenalizedGLMInference in insurance-gam v0.5.0 implements Manna et al. arXiv:2410.01008: bias-corrected confidence intervals for Poisson, Gamma, and Tweedie GLMs after Lasso or elastic-net variable selection. Standard post-selection Wald CIs cover at 70–80% at 95% nominal; this corrects them via asymptotic debiasing or residual bootstrap, with a forest_plot() for visual comparison."
math: true
author: burning-cost
---

There is a dirty secret in how most UK pricing teams validate their GLMs. The pipeline runs something like this: fit a Lasso or elastic-net to knock out the weak rating factors, then compute standard errors on the surviving coefficients using the standard GLM Hessian as if no selection had occurred. Report those standard errors as confidence intervals. Tell the model validation committee — or the FCA, if they ask — that each retained factor is statistically significant at the 95% level.

Those confidence intervals are wrong. Not slightly conservative. Wrong in the direction of false confidence: actual coverage is 70–80% at a nominal 95%, according to simulation results from Manna, Huang, Dey, Gu, and He in a paper published in *Applied Stochastic Models in Business and Industry* (arXiv:2410.01008, October 2024, v5 July 2025). The selection event and the coefficient estimates are computed from the same data. Variables near the noise boundary get selected *because* their estimated coefficients are inflated by chance. Computing CIs that ignore that inflation produces intervals that are too narrow and systematically off-centre.

`insurance-gam` v0.5.0 ships the fix.

---

## Why selection creates bias

A coefficient in a Lasso-fitted GLM is biased toward zero by construction — the $\ell_1$ penalty shrinks all coefficients, and sets some to exactly zero. When you subsequently compute a confidence interval using the standard Wald formula:

$$\hat\beta_j \pm z_{\alpha/2} \cdot \hat{\text{SE}}_j$$

you are treating $\hat\beta_j$ as if it were the unpenalised MLE. It is not. The shrunken estimate has a different distribution from the unpenalised estimate, and the Hessian-based standard error is computed at the penalised solution, not the true unpenalised optimum.

The problem is compounded by selection. Variables are retained precisely because their penalised estimates survived the $\ell_1$ threshold. That is a conditioning event: you are computing CIs conditional on the variable being selected, but using a distributional assumption that ignores that condition. The result is systematically short intervals with incorrect centres.

For UK motor and property pricing, this matters in three practical contexts:

**Model validation for Solvency II.** Internal Model approval requires demonstrating that each model component is statistically justified. CIs covering at 78% when the validation report says 95% is a validation failure if a regulator looks closely.

**FCA Consumer Duty.** If a rating factor is driving a price differential and someone challenges it, your answer to "is this factor statistically significant?" needs to be grounded in correct inference. Naive post-selection CIs make that answer unreliable.

**Rating factor decisions.** Borderline factors — those with estimated effects near the materiality threshold — are exactly where coverage errors bite hardest. The factor that looks marginally significant at 95% may not be significant at all under correct inference.

---

## What PenalizedGLMInference does

`PenalizedGLMInference` implements two strategies from Manna et al. for correcting this bias. Both strategies work on the same fitted object; you call `fit()` once and then choose your inference method.

### Strategy A: Asymptotic debiasing

The debiased estimate corrects for the gradient of the penalised log-likelihood at the estimated solution:

$$H = \frac{1}{n} X^T \text{diag}(W) X$$

where $W_i = \mu_i^{2-p} / \phi$ are the Tweedie Hessian weights (with $p$ the Tweedie power parameter and $\phi$ the dispersion). Then:

$$\Theta = H^{-1}$$

$$\hat{b}_{\text{debiased}} = \hat\beta - \Theta \cdot \nabla_\beta \ell_n(\hat\beta)$$

The gradient term $\nabla_\beta \ell_n(\hat\beta) = X^T (\hat\mu - y) / n$ measures how far the penalised solution is from the likelihood maximum. Subtracting $\Theta$ times this gradient approximately removes the shrinkage bias, giving an estimate that behaves like the unpenalised MLE asymptotically.

Standard errors follow:

$$\hat{\text{SE}}_j = \sqrt{\Theta_{jj} / n}$$

and the CI is $\hat{b}_{\text{debiased},j} \pm z_{\alpha/2} \cdot \hat{\text{SE}}_j$.

This is fast — one matrix inversion after fitting, no additional model runs. The validity condition is $p^2/n \to 0$, where $p$ is the number of features. The class warns when $p^2/n > 0.25$ because you are in a genuinely high-dimensional regime where the asymptotic approximation degrades.

### Strategy B: Residual bootstrap

The bootstrap strategy is slower but avoids the asymptotic condition on $p^2/n$. The procedure:

1. Compute Pearson residuals: $r_i = (y_i - \hat\mu_i) / \sqrt{V(\hat\mu_i)}$
2. For each bootstrap iteration: resample residuals, reconstruct $y^* = \hat\mu + \sqrt{V(\hat\mu)} \cdot r^*$, refit the Lasso at the original $\lambda$ on the bootstrap data
3. Construct a pivotal interval: $[2\hat\beta_j - Q_{1-\alpha/2}^*, \; 2\hat\beta_j - Q_{\alpha/2}^*]$

The pivotal construction — subtracting the bootstrap quantile from twice the estimate — corrects for bias in the bootstrap distribution itself. This is a standard device in bootstrap CI theory.

The default is `n_bootstrap=500`, which takes a few minutes for a GLM with $p = 50$ features and $n = 100{,}000$ observations. Parallelisation across bootstrap samples is not currently implemented — contributions welcome.

---

## API

```python
from insurance_gam import PenalizedGLMInference

pgi = PenalizedGLMInference(family="tweedie", power=1.5, alpha=0.01)
pgi.fit(X, y, exposure=exposure)

ci = pgi.confidence_intervals(alpha=0.05)       # Strategy A: asymptotic debiasing
ci_boot = pgi.bootstrap_ci(alpha=0.05, n_bootstrap=500)  # Strategy B: residual bootstrap

pgi.forest_plot(alpha=0.05, strategy="asymptotic")
```

`fit()` runs the penalised IRLS loop and stores the fitted object. Both `confidence_intervals()` and `bootstrap_ci()` operate on the same fitted result — no retraining. The `alpha` parameter in the constructor controls the Lasso penalty ($\ell_1$ mixing), not the CI coverage level; the CI coverage level is the `alpha` passed to `confidence_intervals()`.

The output DataFrames contain: `feature`, `coef_penalized` (the raw Lasso estimate), `coef` (the debiased estimate), `se`, `ci_lower`, `ci_upper`, `pvalue`, and `selected` (Boolean). For the bootstrap strategy, `se` and `pvalue` are `NaN` — the bootstrap gives interval bounds directly, not distributional quantities.

Supported families: `poisson` (dispersion fixed at 1), `gamma` (dispersion estimated by Pearson $\chi^2$), `tweedie` with $p \in [1, 2]$ (compound Poisson-Gamma range). All use log link. The `exposure` argument is handled as a log offset, which is the standard UK pricing convention.

---

## forest_plot()

The forest plot displays coefficient estimates with CI bars, sorted by effect magnitude, with separate panels for asymptotic and bootstrap strategies when both have been run. Variables where the CI excludes zero are visually distinct from those where it does not.

```python
pgi.forest_plot(alpha=0.05, strategy="asymptotic")
```

For a model validation pack going to a committee or regulator, this is the chart you want: clear visual evidence of which factors are robustly significant and which are borderline, with the correct bias-corrected CIs rather than the naive post-selection ones.

---

## Where this sits in insurance-gam's post-selection toolbox

`insurance-gam` v0.5.0 completes the post-selection inference module. Three classes now cover different questions:

| Class | Question answered | Paper |
|---|---|---|
| `PostSelectionGLM` | Is this coefficient truly non-zero? (hypothesis test) | Shen et al., arXiv:2603.24875 (polyhedral lemma) |
| `DebiasedGLM` | What is the debiased CI for this coefficient? (Poisson/Gamma) | Manna et al., arXiv:2410.01008 |
| `PenalizedGLMInference` | What is the debiased CI, with choice of strategy? (Poisson/Gamma/Tweedie) | Manna et al., arXiv:2410.01008 |

`PostSelectionGLM` and `PenalizedGLMInference` answer genuinely different questions. The polyhedral lemma test controls whether a variable's inclusion is itself a false discovery. The debiased CI, given that a variable has been selected, tells you the plausible range of its effect magnitude. Both are needed: one for selection decisions, one for effect estimation.

`DebiasedGLM` (v0.4.0) commits to a single inference strategy at construction. `PenalizedGLMInference` separates `fit()` from inference, which is more useful when you want to compare asymptotic and bootstrap strategies on the same fitted model without rerunning the IRLS loop.

---

## The high-dimensional warning

The asymptotic debiasing is valid when $p^2/n \to 0$. For a motor pricing model with $p = 40$ features and $n = 500{,}000$ policy years, $p^2/n = 1{,}600/500{,}000 = 0.003$ — well within the valid regime. For a model with $p = 200$ (e.g., a one-hot expanded postcode model) and $n = 50{,}000$, $p^2/n = 0.8$ — squarely in the warning zone. In that case, use Strategy B.

The warning threshold is $p^2/n > 0.25$, which is conservative but sensible. The class emits:

```
UserWarning: p^2/n = 0.31 > 0.25; asymptotic debiasing approximation may be unreliable.
Consider bootstrap_ci() instead.
```

---

## Getting it

`PenalizedGLMInference` is exported from `insurance_gam` in v0.5.0:

```python
from insurance_gam import PenalizedGLMInference
```

Source: `src/insurance_gam/penalized_glm_inference.py`. Tests: `tests/test_penalized_glm_inference.py` (42 tests). The paper is Manna et al. (arXiv:2410.01008), published in *Applied Stochastic Models in Business and Industry*, 2025.

---

## The paper

Manna, Alokesh, Zijian Huang, Dipak K. Dey, Yuwen Gu, and Robin He. "Interval Estimation of Coefficients in Penalized Regression Models of Insurance Data." *Applied Stochastic Models in Business and Industry*, 2025. arXiv:2410.01008. First submitted October 1, 2024. Revised July 15, 2025.

---

## Related posts

- [Your GLM Confidence Intervals Are Wrong After Variable Selection](/techniques/2026/04/01/post-selection-inference-glm-confidence-intervals-lasso-insurance-pricing/) — the motivation post: why naive post-selection CIs cover at 70–80%
- [DebiasedGLM: Honest Confidence Intervals for Lasso-Selected Rating Models](/techniques/2026/04/01/debiased-glm-inference/) — the v0.4.0 predecessor, covering Poisson and Gamma with a single-strategy API
- [GAM vs NAM for Insurance Pricing: A Decision Guide](/techniques/pricing/2026/03/31/gam-vs-nam-insurance-pricing-decision-guide/) — interpretability tradeoffs when moving beyond the standard GLM
- [GLM Insurance Python: UK Pricing Actuary Guide](/techniques/pricing/2026/03/22/glm-insurance-python-uk-pricing-actuary-guide/) — the foundational GLM reference for the library
