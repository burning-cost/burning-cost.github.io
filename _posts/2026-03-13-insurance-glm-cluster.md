---
layout: post
title: "500 Vehicle Makes, One Afternoon, Zero Reproducibility"
date: 2026-03-13
categories: [libraries, pricing, glm]
tags: [factor-clustering, GLM, fused-lasso, R2VF, vehicle-make, categorical-banding, insurance-glm-tools, python, poisson, BIC, monotonicity]
description: "Every UK motor pricing actuary has spent a week in Excel collapsing vehicle makes. The judgement calls stack up: which makes merge, on what basis, with what documentation. insurance-glm-tools automates this with the R2VF algorithm — regularised regression ranking for nominal factors, split-coded fused lasso for fusion, unpenalised GLM refit. Poisson, Gamma, Tweedie. BIC lambda selection. Monotonicity enforcement. 156 tests."
---

The problem is not that you have 500 vehicle makes. The problem is that you cannot fit 500 dummies reliably, so you need to collapse them, and collapsing them manually means one actuary's judgment about whether Skoda and SEAT should share a band is baked into the model, undocumented, and wrong in a way that will not be obvious until someone runs the analysis three years later and gets a different answer.

UK motor pricing has hundreds of vehicle makes, over 350 occupation codes, and postcode districts that vary in claims frequency by a factor of four. Actuaries spend weeks at the start of every model cycle grouping these factors by hand. The groupings are sensible, but they are not reproducible, they are not statistically optimal, and they require re-doing every time the data changes.

[`insurance-glm-tools`](https://github.com/burning-cost/insurance-glm-tools) automates factor level clustering for insurance GLMs. It implements the R2VF algorithm (Ben Dror, arXiv:2503.01521, 2025): a two-step regularised approach that works for both ordered factors (vehicle age, NCD) and unordered factors (vehicle make, occupation). 156 tests, MIT-licensed, on PyPI.

```bash
uv add insurance-glm-tools
```

---

## What the algorithm does

The fused lasso penalises differences between adjacent level coefficients, driving similar levels to merge. For ordinal factors — vehicle age, NCD years — there is a natural ordering. You apply the fused lasso directly and levels with similar risk profiles collapse.

For nominal factors — vehicle make, occupation code — there is no natural ordering. You cannot penalise adjacent levels when adjacency is undefined. This is why standard fused lasso implementations do not work for insurance categorical factors. R2VF solves this with a two-step approach.

**Step 1 (Ranking):** Fit a ridge GLM with all factor dummies simultaneously. The resulting coefficients give a data-driven ordering: vehicle makes with similar predicted risk end up with similar coefficients and therefore similar rank positions. Makes that are very different in risk end up far apart in the ranking. This ordering has no particular meaning beyond giving fused lasso a sequence to penalise across.

**Step 2 (Fusion):** Re-encode each nominal factor as ordinal using the Step 1 ranking. Apply a standard fused lasso to all factors using the split-coding trick: transform the design matrix so that standard L1 (sklearn Lasso) achieves the fused lasso objective. No quadratic programming, no specialised solver. Where the penalty drives adjacent-rank differences to zero, those levels merge.

**Step 3 (Refit):** Fit an unpenalised GLM on the merged groupings. This removes the shrinkage bias introduced by the lasso in Step 2. The refit result has standard GLM properties and unbiased coefficient estimates.

The computational advantage over generalised fused lasso is significant. All-pairs GFL has O(K²) penalty terms for K levels — for 500 vehicle makes, that is 125,000 penalty terms. R2VF has O(K) terms because the Step 1 ranking imposes an ordering. Standard lasso is fast at O(K) terms. The full pipeline for a motor book with 500 makes, 350 occupation codes, and 50,000 policies runs in under two minutes on a single core.

---

## Usage

```python
from insurance_glm_tools.cluster import FactorClusterer

clusterer = FactorClusterer(
    family='poisson',
    link='log',
    lambda_='bic',              # BIC-selected regularisation
    min_exposure=500,           # merge groups below 500 earned years
    monotone_factors=['ncd'],   # enforce monotone decreasing NCD
    monotone_direction={'ncd': 'decreasing'},
)

clusterer.fit(
    X,
    y,
    exposure=exposure,
    ordinal_factors=['vehicle_age', 'ncd'],
    nominal_factors=['vehicle_make', 'occupation'],
)
```

The fit step is where everything happens: ridge ranking, fused lasso path, BIC selection, and monotonicity enforcement. The result is a set of level mappings: which original levels collapsed into which merged groups.

```python
# Review the groupings before committing
lm = clusterer.level_map('vehicle_make')
print(lm.to_df())
#    original_level  merged_group  coefficient  exposure
# 0           AUDI             0        -0.12    4521.3
# 1            BMW             0        -0.11    3892.1
# 2      LAND ROVER             0        -0.10    2341.8
# 3  MERCEDES-BENZ             0        -0.12    5102.4
# 4           FORD             1         0.08   18920.4
# 5         NISSAN             1         0.09    7841.2
# ...

print(f"Vehicle make: {lm.n_levels_original()} → {lm.n_groups()} groups")
# Vehicle make: 487 → 23 groups

print(f"Compression ratio: {lm.compression_ratio():.1f}x")
# Compression ratio: 21.2x
```

The level map is designed for review. An actuary can inspect which makes merged together, challenge groupings that look wrong, and adjust before refit. The library does not demand you accept the statistical grouping blindly — it gives you a starting point that is defensible and documented.

```python
# Merged design matrix — drop-in replacement for original columns
X_merged = clusterer.transform(X)

# Unpenalised GLM refit on merged factors
result = clusterer.refit_glm(X_merged, y, exposure=exposure)
print(result.summary())

# Diagnostics
diag = clusterer.diagnostics()
print(f"AIC before: {diag['aic_before']:.1f}, after: {diag['aic_after']:.1f}")
print(f"BIC before: {diag['bic_before']:.1f}, after: {diag['bic_after']:.1f}")
```

---

## BIC for lambda selection

The regularisation strength lambda controls how aggressively levels merge. Small lambda: many groups, fine discrimination but thin cells. Large lambda: few groups, stable estimates but lost signal.

Cross-validation is the standard answer in the ML literature. For insurance data, it is the wrong answer. Policies from the same customer appear across years; cross-validation folds will leak. Multi-year development patterns in the claims data mean earlier years are not exchangeable with later years. Running k-fold CV on a motor book will give you an optimistic estimate of out-of-sample performance and a lambda that overfits to leakage.

BIC selects lambda in-sample by penalising model complexity: `BIC = -2*log_likelihood + k*log(n)`, where `k` is the effective number of parameters (number of merged groups across all factors, plus other model terms) and `n` is the effective number of observations (exposure-weighted). This is appropriate when the goal is factor grouping — you want the grouping that best explains the data given its complexity, not the grouping that best predicts held-out data that you cannot cleanly separate.

---

## Monotonicity enforcement

NCD should be monotone decreasing: more NCD years, lower claims frequency. This is a known actuarial constraint, and it often does not hold naturally from fused lasso — particularly in thin cells at the extremes of the NCD scale where a handful of claims can invert the pattern.

The library enforces monotonicity as a post-processing step using isotonic regression (PAV algorithm). After the fused lasso step determines the groupings, the level map is checked for monotonicity violations and corrected. The correction re-merges any groups that violate the specified direction.

```python
clusterer = FactorClusterer(
    family='poisson',
    monotone_factors=['ncd', 'vehicle_age'],
    monotone_direction={
        'ncd': 'decreasing',
        'vehicle_age': 'increasing',  # newer vehicles, lower claims
    },
)
```

Monotonicity is enforced on the merged groups, not on individual levels. A merged group of NCD years 8-12 will be constrained to have lower claims frequency than a merged group of NCD years 4-7. Within a merged group, variation is subsumed into the group mean.

---

## GLM families

The library supports Poisson, Gamma, and Tweedie GLMs. All with a log link, which is standard for insurance pricing.

```python
# Frequency model
freq_clusterer = FactorClusterer(family='poisson', link='log')

# Severity model
sev_clusterer = FactorClusterer(family='gamma', link='log')

# Pure premium (combined)
pp_clusterer = FactorClusterer(family='tweedie', link='log', tweedie_p=1.5)
```

The Step 2 approximation — using sklearn Lasso on a transformed design matrix — applies an exposure-adjusted normal approximation to the GLM likelihood. The approximation is closest for Poisson and less exact for Gamma. The Step 3 refit is always an exact unpenalised GLM, so any approximation error from Step 2 affects only the lambda selection and grouping, not the final coefficient estimates.

---

## The reproducibility problem

The manual approach to factor grouping produces decisions that cannot be reconstructed. "SEAT and Skoda were merged because they have similar risk profiles in our book" is not a reproducible statement. The analyst who made that call may not be there in three years. The data generating that decision changes every model cycle. The grouping cannot be validated against alternative groupings.

`insurance-glm-tools` produces a documented, reproducible grouping. The level maps are serialisable. The lambda path is stored. The BIC criterion at each lambda is available for audit. Re-running the analysis on updated data produces a new grouping that follows the same procedure with no analyst judgment required.

This matters for model governance. PRA SS1/23 expects model risk documentation. FCA Consumer Duty expects fair value evidence. A factor grouping methodology that is statistically principled, documented, and reproducible is more defensible than one that is not — particularly when the Treating Customers Fairly implications of vehicle group relativities are being reviewed.

---

## What it is not

The library does not implement all-pairs generalised fused lasso. That approach can in principle find groupings that the R2VF ranking step misses — factors where similar risk profiles are not sorted adjacently by Step 1. In practice, Step 1 ridge ranking is good enough for insurance factors. If you have a factor where the ridge ranking is known to be unreliable, you can supply a custom ordering.

The library does not handle factors with a natural tree structure (e.g., ABI vehicle code hierarchy). Tree-based grouping via `insurance-interactions` is a better starting point there.

It does not implement the Henckaerts et al. (arXiv:1904.10890) tree-based grouping approach. That approach uses a GBM to bin continuous covariates and is better suited to numeric rating factors than to nominal categoricals. The two methods are complementary: tree-based for continuous factors, R2VF fused lasso for high-cardinality categoricals.

---

**[insurance-glm-tools on GitHub](https://github.com/burning-cost/insurance-glm-tools)** — 156 tests, MIT-licensed, PyPI.

---

**Related reading:**
- [Nested GLMs with Neural Network Embeddings for Insurance Ratemaking](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/) — neural embeddings for high-cardinality categoricals as an alternative to fused lasso factor collapsing; different trade-off between flexibility and interpretability
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) — SHAP-based GLM distillation from a GBM; the complement to GLM factor collapsing when you want to start from a non-parametric model
- [EBMs for Insurance Pricing: Better Than a GLM, Readable by a Pricing Committee](/2026/03/09/explainable-boosting-machines-for-insurance-pricing/) — additive models that handle high-cardinality features without the collapsing step; the alternative when factor interpretability is the primary constraint
