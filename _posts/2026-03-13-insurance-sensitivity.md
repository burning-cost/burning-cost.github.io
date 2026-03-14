---
layout: post
title: "Shapley Effects for Rating Factor Variance Decomposition"
date: 2026-03-13
categories: [libraries, pricing, sensitivity]
tags: [global-sensitivity-analysis, shapley-effects, variance-decomposition, rating-factors, sobol, insurance-sensitivity, python, GLM, GBM, correlated-inputs, FCA, consumer-duty]
description: "Which rating factor drives the most premium variance in your motor book? The honest answer requires Shapley effects — not SHAP values, not R², not sequential deviance. insurance-sensitivity implements the Song 2016 random permutation estimator with Rabitti-Tzougas 2025 CLH subsampling, exposure weighting, and support for correlated categoricals. Sobol indices included for completeness, but Shapley effects are what you should be reporting."
canonical_url: "https://burning-cost.github.io/2026/03/10/insurance-sensitivity/"
---

Every pricing team has had this argument. Someone from actuarial says driver age accounts for 40% of premium variability. Someone from pricing says vehicle group matters more. Both are working from GLM coefficients, p-values, and intuition. None of those answer the question being asked.

The question is: what fraction of the variance in fitted premiums is attributable to each rating factor, properly accounting for the correlations between factors? That is a global sensitivity analysis problem. The actuarially correct answer is Shapley effects.

[`insurance-sensitivity`](https://github.com/burning-cost/insurance-sensitivity) implements them. Song et al. (2016) random permutation estimator, Rabitti-Tzougas (2025) CLH subsampling for large datasets, exposure-weighted variance throughout, and a fitted-model interface so you hand it your existing GLM or GBM directly. 112 tests, MIT-licensed, on PyPI.

```bash
uv add insurance-sensitivity
```

---

## Why not SHAP values

SHAP values are local. For a specific policy, the SHAP value for driver age tells you how much that factor contributed to that policy's predicted premium, relative to the mean. Aggregating SHAP values across the portfolio — taking mean absolute SHAP, for instance — gives you a model-importance ranking, but it is not a variance decomposition. It does not tell you what fraction of `Var[premium]` is attributable to driver age.

More critically, SHAP local-to-global aggregation is heuristic. Mean |SHAP| is not a principled quantity under correlated inputs. If driver age and NCD are correlated (they are, strongly), the TreeSHAP conditional perturbation approach will split their contribution in a way that depends on the tree structure, not on the underlying variance attribution. The result can be individually plausible but jointly misleading.

Shapley effects are global from the start. They apply the cooperative game Shapley formula to the characteristic function `v(S) = Var[E[Y | X_S]]` — the variance explained when conditioning on a subset S of factors. The result: a set of non-negative values summing to `Var[Y]`, correctly attributed across factors even under strong correlations (Owen, 2014). This is the quantity that answers the management question.

The Biessy (ASTIN Bulletin, 2024) paper demonstrated this on a car insurance dataset, using Shapley effects to rank and select rating factors by their contribution to premium variance. The conclusion: p-values and standardised regression coefficients give systematically different rankings from Shapley effects under correlated inputs. The correlation between rankings was 0.67 — worse than most practitioners assume.

---

## The estimation problem

Computing Shapley effects exactly requires averaging over all `d!` orderings of `d` inputs. For 12 rating factors, that is 479 million orderings. The Song et al. (2016) random permutation estimator makes this tractable: sample `m` random orderings, compute the marginal contribution of adding each factor to an existing conditioning set, average. The cost is `Nv + m*(d-1)*No*Ni` model evaluations, where `No` (outer samples, default 1) and `Ni` (inner samples, default 3) control variance. Convergence is fast in practice — the library default of `m=50` orderings is sufficient for most 8-15 factor pricing models.

The remaining problem is evaluation cost when the dataset is large. Biessy (2024) ran on a moderately sized dataset; Rabitti and Tzougas (2025) applied the Song estimator to the French MTPL dataset of 678,000 policies and found runtime over 18 hours. Their fix: Conditional Latin Hypercube (CLH) sampling identifies a representative subsample of 2,500 policies that preserves the multivariate input distribution, including correlations. Hierarchical k-means then clusters the subsample. The Song estimator runs on the cluster centroids. Total runtime: approximately one minute. Accuracy: maintained — the paper reports relative error below 2% on all factor Shapley effects.

This is the default computational path in `insurance-sensitivity` when your dataset exceeds 10,000 policies.

---

## Installation and basic usage

```python
from insurance_sensitivity import SensitivityAnalysis

# Fitted GLM or GBM — any object with a .predict(X) method
# X_train: pandas DataFrame of rating factors
# exposure: array of policy exposures (earned years)

sa = SensitivityAnalysis(
    model=fitted_glm,
    X=X_train,
    exposure=exposure,
    clh_n_samples=2500,   # CLH subsample size (Rabitti-Tzougas)
    random_state=42,
)

result = sa.shapley()
print(result.to_df())
#    factor           shapley_effect  shapley_frac  sobol_s1  sobol_st
# 0  driver_age             0.0312        0.287      0.198     0.301
# 1  vehicle_group          0.0241        0.221      0.167     0.259
# 2  ncd                    0.0198        0.182      0.143     0.234
# 3  area                   0.0147        0.135      0.098     0.186
# 4  occupation             0.0089        0.082      0.071     0.107
# 5  annual_mileage         0.0067        0.062      0.054     0.089
# 6  vehicle_age            0.0043        0.040      0.038     0.061
# 7  policyholder_sex       0.0003        0.003      0.003     0.005
```

The `shapley_frac` column is the answer to the management question: driver age accounts for 28.7% of fitted premium variance in this book. The fractions sum to 1.0.

---

## Exposure weighting

Insurance policies have heterogeneous exposures. A mid-term policy written with 6 months remaining contributes 0.5 earned years; an annual policy renewing contributes 1.0. Standard variance-based sensitivity analysis treats every row in your data as equally weighted. That is wrong for insurance.

The library computes exposure-weighted variance throughout:

```
Var_w[Y] = sum_i w_i * (Y_i - Y_bar_w)^2 / sum_i w_i
```

where `w_i` is the policy exposure. The Shapley characteristic function becomes `v(S) = Var_w[E_w[Y | X_S]]`. This is the correct quantity when your portfolio has heterogeneous policy lengths — which is always.

There is no published Python library that does this. SALib v1.5.2 (the nearest alternative) has no exposure weighting and no Shapley effects. Standard Sobol indices from SALib on an exposure-heterogeneous portfolio will give subtly wrong answers. `insurance-sensitivity` does not.

---

## Correlated categoricals

UK motor has correlated rating factors. Driver age and NCD correlate. Postcode and occupation correlate. Vehicle group and vehicle age correlate. Standard Sobol indices assume independent inputs — the first-order index S1 will be inflated for any factor correlated with a high-importance factor.

Shapley effects handle this correctly by construction. The characteristic function conditions on all permutations of the other factors, averaging out correlations. No additional special handling is required.

For categorical factors (vehicle make, occupation code, area), the library uses empirical distribution sampling: rather than specifying a parametric distribution for each input, it samples from the observed joint distribution in the training data. This is the approach Biessy (2024) and Rabitti-Tzougas (2025) used on the French MTPL data, and it is the only approach that preserves the true correlation structure of the training dataset.

```python
# Mark which factors are categorical — library handles the rest
sa = SensitivityAnalysis(
    model=fitted_model,
    X=X_train,
    exposure=exposure,
    categorical_factors=['vehicle_make', 'occupation', 'area'],
)
```

---

## Sobol indices

Shapley effects are the right answer for correlated inputs. Sobol first and total-order indices are included for two reasons: they remain interpretable for approximately independent factor pairs, and they are the standard in other industries — if your model governance team has seen GSA before, they probably saw Sobol.

```python
sobol_result = sa.sobol()
print(sobol_result.to_df())
#    factor        S1      S1_ci   ST      ST_ci
# 0  driver_age  0.198   (0.187, 0.209)  0.301  (0.289, 0.313)
# ...

# Interaction indicator: factors where ST >> S1
sobol_result.interaction_flags(threshold=0.05)
```

The difference `ST - S1` for each factor indicates involvement in interactions. A factor with `ST` substantially above `S1` is participating in interaction effects that are not captured by main effects alone. This is a useful complement to the `insurance-interactions` library, which detects specific pairwise interactions — here, you see which individual factors are broadly involved in interactions without specifying the pairs.

---

## Log-scale decomposition

For multiplicative GLMs (Poisson frequency, Gamma severity), variance decomposition on the raw predicted premium surface can be dominated by scale differences between segments. The library offers log-scale decomposition as the default for models with a log link:

```python
sa = SensitivityAnalysis(
    model=fitted_glm,
    X=X_train,
    exposure=exposure,
    decompose_log_scale=True,  # default True for log-link models
)
```

This computes `Var_w[log(mu_hat)]` rather than `Var_w[mu_hat]`, and decomposes that. For multiplicative models, this is the more natural quantity: it tells you what fraction of the variation in the log-premium surface is attributable to each factor, which corresponds more directly to the regression coefficients you interpret.

---

## What this is for

The immediate use case is regulatory. FCA Consumer Duty requires fair value demonstration — which factors drive premium variation? FCA general insurance pricing data returns (PS21/5) ask for factor contribution documentation. PRA internal model validation (SS1/23) requires sensitivity documentation. All of these questions have a rigorous answer: Shapley effects. No existing Python library provided that answer.

The secondary use case is model development. If driver age accounts for 28.7% of premium variance but represents only 3% of your factor development effort, that is a misallocation. Shapley effects give you an objective basis for prioritising modelling work across factors.

The tertiary use case is factor selection. Biessy (2024) showed that Shapley-effect-ranked factors produce more homogeneous rating classes than factors ranked by p-value or regression coefficient. If you are considering dropping a factor from your model, its Shapley effect is the right quantity to look at — not its t-statistic.

---

## What it does not do

The Shapley-Owen effects extension (Ruess, 2024, arXiv:2409.19318), which attributes variance to subsets of factors rather than individual factors, is not in v0.1. This is useful for grouped factors (e.g., all geography-related factors combined) but expensive to compute and less commonly needed.

Deviance-based decomposition (the connection to `anova.glm` in R, which gives sequential deviance reduction by factor) is a future extension. The library provides a `sequential_deviance()` method as a reference comparison, but this is order-dependent and does not replace Shapley effects — it is there to show the difference.

---

**[insurance-sensitivity on GitHub](https://github.com/burning-cost/insurance-sensitivity)** — 112 tests, MIT-licensed, PyPI.

---

**Related articles from Burning Cost:**
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Full Predictive Distributions for Insurance Pricing](/2026/03/05/insurance-distributional/)
