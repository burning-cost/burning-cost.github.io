---
layout: post
title: "Which Rating Factors Actually Drive Your Premium Variance?"
date: 2026-03-10
author: Burning Cost
categories: [tools]
tags: [sensitivity-analysis, shapley-effects, sobol-indices, variance-decomposition, GLM, GBM, rating-factors, SHAP, exposure-weighting, CLH-subsampling, FCA, python, insurance-sensitivity]
description: "SHAP explains individual predictions. Shapley effects decompose portfolio-level variance. insurance-sensitivity brings Owen 2014, Song 2016 and Rabitti-Tzougas 2025 to insurance pricing — exposure-weighted, categorical-native, scalable to 678k policies."
---

Your GBM has SHAP values. Your GLM has standard errors. Neither one answers the question that keeps coming up in pricing committees: which rating factors are actually responsible for the variation in premiums across your portfolio?

SHAP values tell you, for a single policy, how much each feature shifted the prediction relative to a baseline. Run SHAP on a million policies and take the mean absolute value and you get something close to variable importance. But it is not a variance decomposition. "Vehicle group has the highest mean absolute SHAP value" is not the same as "vehicle group explains 34% of the variance in fitted premiums." One is a ranking; the other is a percentage of a total. Regulators, pricing committees, and fair value assessments want the second statement.

The standard tool for that is Sobol indices. The problem with Sobol indices is that they assume independent inputs. UK motor rating factors are not independent. Driver age correlates with NCD. Vehicle group correlates with postcode (certain cars cluster in certain areas). If you run Sobol first-order indices on correlated inputs, you systematically over-credit factors that are correlated with high-importance factors and produce a decomposition that does not add up to the total variance in any interpretable way.

Shapley effects (Owen 2014, Song et al. 2016) solve this. Same cooperative-game-theory logic as SHAP, but applied to variance decomposition rather than individual predictions. The effects always sum to V[Y], are never negative, and handle correlated inputs correctly because the Shapley formula averages over all possible orderings of factors — including the ones where the correlated factor arrives last and contributes nothing beyond what the first factor already captured.

[`insurance-sensitivity`](https://github.com/burning-cost/insurance-sensitivity) is library #38 in the Burning Cost portfolio. It implements Shapley effects with the extensions the insurance use case actually needs: exposure weighting, categorical factors without encoding, and CLH subsampling for large portfolios.

```bash
uv add insurance-sensitivity
```

---

## What variance decomposition gives you that SHAP does not

The distinction is worth being precise about.

SHAP produces a vector per prediction: for policy *i*, factor *j* contributed φᵢⱼ to the log-score. Aggregate across policies and you can rank factors by mean |φᵢⱼ| or mean φᵢⱼ². That is a useful diagnostic — it tells you which factors tend to move individual predictions by the most.

A Shapley effect gives you a scalar per factor: factor *j* is responsible for φⱼ of the total model output variance V[Y] across the portfolio. The effects sum to V[Y] exactly. Expressed as percentages, they partition 100% of premium variation across the factor set.

This second statement is what you need when:

- A pricing committee asks whether the model is primarily driven by vehicle factors or driver factors
- An FCA fair value review asks whether pricing variation is concentrated in factors that correlate with protected characteristics
- A senior actuary questions why premiums in postcode area X are 40% above the book average
- You want to compare the variance structure of a new model against the incumbent

The SHAP aggregate does not answer any of these questions directly. The Shapley effects do.

---

## The mechanics: Song 2016 kNN estimator

The practical challenge with Shapley effects is computing the conditional expectation E[Y | Xₛ = xₛ] for every subset S of factors and every permutation ordering. For a model with d = 15 factors, there are 2¹⁵ = 32,768 subsets. Direct enumeration is off the table.

Song, Nelson and Staum (2016) proposed a random-permutation estimator that makes this tractable. Rather than enumerating all subsets, draw m random permutations of the d factors. For each permutation, update each factor's Shapley effect incrementally as you step through the factor ordering. The conditional expectations E[Y | Xₛ] are estimated using k-nearest-neighbours on the training data — no parametric distributional assumption required.

The kNN estimator is the key innovation for insurance. Classical global sensitivity analysis methods (SALib, for example) require you to specify a parametric input distribution — independent uniform distributions, or a multivariate normal. You cannot easily specify the joint distribution of driver age, NCD level, and vehicle group from a live UK motor portfolio. The Song estimator instead uses the empirical joint distribution of your training data directly, conditioning on observed factor combinations.

With 256 permutations (the default), the estimator typically achieves standard errors of around ±1–2 percentage points on the Shapley effects for a 10-factor model. Run 1024 permutations for a final governance document; run 64 for quick exploration.

---

## Exposure weighting

The exposure weighting is not cosmetic.

A UK motor portfolio contains a mix of annual policies, mid-term renewals, and short-term policies that are in-force for as little as a few days. If you weight all rows equally, a large number of 30-day policies will have disproportionate influence on the estimated variance structure even though those policies represent little actual premium or risk exposure.

`SensitivityAnalysis` takes an `exposure_col` parameter that applies each policy's exposure weight (typically years of coverage) throughout:

- The total variance V[Y] is computed as the exposure-weighted variance of model predictions
- The kNN conditional-mean estimates are exposure-weighted
- The final Shapley effects sum to the exposure-weighted V[Y]

The result is a variance decomposition that represents the actual premium distribution across the portfolio, not a count-weighted average that over-represents short-term policies.

---

## Using the library

```python
from insurance_sensitivity import SensitivityAnalysis

sa = SensitivityAnalysis(
    model=fitted_glm,          # any model with .predict(X)
    X=training_df,             # training data including exposure column
    exposure_col='exposure',   # year fractions for each policy
    log_scale=True,            # decompose Var[log(fitted)]
                               # — right for a multiplicative GLM
    random_state=42,
)
```

The `log_scale=True` flag is the right choice for multiplicative pricing models. It applies `log()` to the fitted values before computing variance, so you are decomposing the variance of the log-premium surface rather than the raw premium surface. Raw premium variance is dominated by high-premium segments (commercial, high-value home) in ways that are not informative about which factors drive relative pricing decisions. The log scale is the natural scale for a multiplicative GLM and the one that corresponds to the log-relativity tables you already review.

```python
# Shapley effects — correct under correlated inputs
result = sa.shapley(n_perms=256)
print(result)
```

```
ShapleyResult(total_variance=0.1847)
  vehicle_group: 34.2%
  ncd_band: 22.1%
  driver_age: 18.4%
  area: 11.3%
  vehicle_age: 7.8%
  ... (10 factors total)
```

```python
result.plot_bar()   # horizontal bar chart with 95% confidence intervals
result.plot_pie()   # pie chart of percentage contributions
```

The bar chart shows 95% confidence intervals from the permutation variance. Wide intervals mean you need more permutations. Narrow intervals mean the ranking is stable.

---

## Handling categorical factors

Standard Sobol estimators require continuous inputs. Categorical factors — vehicle group, occupation, regional territory — require encoding that distorts the distance structure the kNN estimator relies on.

`insurance-sensitivity` handles categoricals natively by using empirical marginal sampling. When a factor is not in the conditioning set S, rather than sampling from a parametric marginal, it resamples the observed values from the training data. This preserves the realistic frequency distribution of categories (vehicle group G4 is more common than G17) without any encoding.

This matters. Encoding a 100-level vehicle group as integers 1–100 introduces a spurious ordinal structure that corrupts the kNN distance metric. One-hot encoding a 100-level factor creates 100 near-binary dimensions that dominate the distance metric. The empirical marginal sampling approach avoids both.

---

## CLH subsampling for large portfolios

The kNN step in the Song estimator scales roughly as O(n · d · log n) per conditional-expectation evaluation. For a 678,000-policy portfolio with 15 factors and 256 permutations, the full-sample estimator takes several hours.

Rabitti and Tzougas (2025) solved this in their European Actuarial Journal paper. Their key result: the Song estimator's accuracy depends on the marginal uniformity of the input distribution, not on the absolute sample size. A carefully chosen subsample of ~2,000–5,000 policies that preserves the marginal uniformity of each factor gives results very close to the full-sample estimate, at a fraction of the cost.

Their Conditional Latin Hypercube (CLH) algorithm selects the subsample by rank-transforming each numeric column to uniform [0, 1], stratifying into equal-width strata, and greedily selecting one observation per stratum per column with no observation selected twice. The result is a subsample that represents the full marginal distribution of every factor, and whose joint structure follows the observed correlations because whole rows are selected.

Rabitti and Tzougas demonstrated this on the French MTPL dataset (678,013 policies) and reported >1000× speedup with negligible accuracy loss.

```python
result = sa.shapley(
    n_perms=256,
    n_subsample=2500,   # representative subsample via CLH
                        # recommended for n > 10k
)
```

In practice: a 200,000-policy UK motor book that would take 45 minutes full-sample runs in under a minute with `n_subsample=2500`. We recommend validating on a smaller holdout first — run the full-sample estimate on 5,000 randomly sampled rows and compare against the CLH subsample estimate. If they agree within ±2 percentage points on the top factors, the subsampling is working correctly.

---

## Sobol as a fast alternative

When your factors are approximately independent — a severity model with largely uncorrelated risk characteristics, or a synthetic book for testing purposes — Sobol indices are faster and simpler:

```python
sobol = sa.sobol(n_samples=1024)
sobol.plot_bar()   # S1 and ST side by side
```

The Sobol estimate at n_samples=1024 costs d+2 × 1024 model evaluations (17,408 for a 15-factor model versus 256 × 15 × 3 = 11,520 for Shapley with default inner/outer settings, so they are comparable cost). The first-order index S1 measures the fraction of variance due to a factor alone; the total-order index ST includes all interactions.

If your factors are correlated, the library issues a `UserWarning` when |Pearson r| > 0.3 for any pair. We suggest treating Sobol results on correlated factors as a ranking tool only — the ST ordering is usually robust even when the absolute values are wrong — and using Shapley effects for any result that will appear in a governance document.

---

## Group attributions

For regulatory and management reporting, it is often more useful to attribute variance to groups of factors (all vehicle-related factors combined, all driver-related factors combined) rather than individual columns:

```python
groups = {
    'vehicle': ['vehicle_group', 'vehicle_age', 'cc_band'],
    'driver':  ['driver_age', 'ncd_band', 'licence_years'],
    'area':    ['postcode_area', 'garage_type'],
    'usage':   ['annual_mileage', 'overnight_location'],
}
result = sa.shapley(n_perms=256, groups=groups)
# effects DataFrame now has rows: vehicle, driver, area, usage
```

The group Shapley effects treat each group as a single player in the cooperative game. The vehicle group's Shapley effect is the average marginal contribution of revealing all vehicle factors simultaneously, across all orderings of the groups. This is the right statement for "how much of our premium variance is driven by vehicle characteristics versus driver characteristics."

---

## Comparing Shapley effects across models

One natural use is comparing the variance structure of a challenger model against the incumbent. If the challenger redistributes variance from vehicle group to driver factors, that is potentially a substantive pricing change — it implies the challenger is placing more weight on who is driving than what they are driving.

```python
sa_incumbent = SensitivityAnalysis(model=glm, X=X, exposure_col='exposure', log_scale=True)
sa_challenger = SensitivityAnalysis(model=gbm, X=X, exposure_col='exposure', log_scale=True)

incumbent_effects = sa_incumbent.shapley(n_perms=256).effects
challenger_effects = sa_challenger.shapley(n_perms=256).effects

import pandas as pd
comparison = incumbent_effects.merge(
    challenger_effects,
    on='factor',
    suffixes=('_glm', '_gbm'),
)
comparison['delta_pct'] = comparison['phi_pct_gbm'] - comparison['phi_pct_glm']
print(comparison.sort_values('delta_pct', ascending=False))
```

A large positive `delta_pct` means the GBM is placing substantially more weight on that factor than the GLM. A large negative value means the GLM was over-crediting a factor relative to the GBM's estimate. These are the factors worth examining in challenger validation.

---

## Where this fits versus SHAP and ANAM shape functions

The three tools answer different questions:

**SHAP** on a GBM: which features pushed *this prediction* above or below the average, and by how much? Useful for single-policy explainability, outlier investigation, and monitoring for individual predictions that have shifted.

**ANAM shape functions** ([`insurance-anam`](/2026/03/17/your-interpretable-model-isnt-interpretable-enough/)): what is the model's exact learned relationship between a feature and the log-linear predictor? The shape function is a property of the model architecture, not a post-hoc approximation.

**Shapley effects** (`insurance-sensitivity`): which factors account for the most variance in predicted premiums across the portfolio? This is a population-level statement, model-agnostic, valid for GLMs, GBMs, EBMs, or anything else with a predict method.

They are complementary. We would run Shapley effects after building any new model as a portfolio-level sense check, before reviewing individual factor shape functions in detail. If Shapley effects show that postcode accounts for 28% of variance in a frequency model you expected to be primarily driven by vehicle and driver characteristics, that is a signal to investigate before presenting the model to a pricing committee.

---

## The regulatory case for variance decomposition

Under Consumer Duty, fair value assessment requires demonstrating that pricing differentials are actuarially justified. If your model produces a 2:1 spread in premiums between the lowest and highest risk, being able to state "34% of that spread is driven by vehicle group, 22% by NCD history" is a more substantive piece of evidence than "vehicle group has the highest mean absolute SHAP value."

Under PRA SS1/23, model validators are expected to assess the relative importance of model inputs. Shapley effects provide a quantitative, reproducible measure of that importance that can be reproduced by an independent team from the model and training data.

The Biessy (ASTIN Bulletin, 54(1), 2024) paper provides the theoretical grounding for using global sensitivity analysis specifically in the context of rating factor selection. His empirical work on French MTPL data showed that GSA-based factor importance rankings differ materially from GLM parameter significance rankings — factors that are statistically significant but account for a small fraction of total variance are poor candidates for retention in a production tariff.

---

## Installation and supported models

```bash
uv add insurance-sensitivity
uv add "insurance-sensitivity[plots]"    # matplotlib bar and pie charts
uv add "insurance-sensitivity[polars]"   # polars DataFrame input
```

Python 3.10+. The model wrapper handles sklearn estimators, statsmodels GLM results, glum `GeneralizedLinearRegressor`, CatBoost, XGBoost, and LightGBM out of the box. For anything else, pass `predict_fn='my_method_name'` or pass a bare callable as `model`.

The source is at [github.com/burning-cost/insurance-sensitivity](https://github.com/burning-cost/insurance-sensitivity). The `notebooks/` directory has a worked example on synthetic UK motor data: 200,000 policies, 12 factors including four categoricals, a statsmodels GLM and a CatBoost challenger. The notebook runs the full Shapley comparison, CLH subsampling validation, and group attribution workflow in under three minutes on a laptop.

---

SHAP tells you why the model said what it said about one policy. Shapley effects tell you which factors are driving premium variation across the whole portfolio. You need both, and until now the second one required either a bespoke implementation or a trip through SALib's parametric input specification, which does not fit naturally onto a live insurance training dataset. This library closes that gap.

---

**Related articles:**
- [Actuarial Neural Additive Models: Exact Interpretability with Tweedie Loss](/2026/03/17/your-interpretable-model-isnt-interpretable-enough/) — intrinsic shape functions, for when you want to know what the model learned rather than what it's doing to variance
- [Your Rating Factor Might Be Confounded](/2026/03/05/your-rating-factor-might-be-confounded/) — why variance decomposition is not the same as causal attribution
- [EBMs for Insurance Tariff Construction](/2026/03/20/ebms-for-insurance-tariff-construction/) — Explainable Boosting Machines as an alternative to GLMs with additive shape functions
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) — using SHAP for a different question: per-policy explainability and relativity extraction
