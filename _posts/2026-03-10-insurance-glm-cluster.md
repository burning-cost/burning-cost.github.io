---
layout: post
title: "500 Vehicle Makes, One Afternoon, Zero Reproducibility"
date: 2026-03-10
categories: [libraries, pricing, glm]
tags: [GLM, factor-clustering, fused-lasso, r2vf, regularisation, poisson, gamma, tweedie, vehicle-make, occupation, motor, python, insurance-glm-cluster, bic, credibility]
description: "Factor level collapsing is the highest-labour, lowest-reproducibility activity in GLM development. insurance-glm-cluster automates it using the R2VF algorithm (Ben Dror, arXiv:2503.01521) — ridge-regularised ranking followed by fused lasso fusion, with BIC lambda selection and min-exposure enforcement. No cvxpy required."
---

Ask any motor pricing actuary what they spent last Tuesday doing. Somewhere in the answer you will find: collapsing vehicle makes. Taking the 500 raw categories from the policy admin system, running the one-way relativities, identifying which makes cluster at similar risk levels, and drawing the grouping boundaries by hand. Then arguing about whether Alfa Romeo should sit with Audi or BMW. Then doing it again when the data extract changes.

This is the highest-labour activity in GLM development, and it produces results that vary by analyst, are difficult to audit, and cannot be reproduced without the analyst's notes. Every pricing team does it. No open-source Python tool helps with it.

[`insurance-glm-cluster`](https://github.com/burning-cost/insurance-glm-cluster) automates factor level collapsing for Poisson, Gamma, and Tweedie GLMs. It implements the R2VF algorithm (Ben Dror, Earnix, arXiv:2503.01521, March 2025) — the same method that powers Earnix's proprietary Auto-GLM "Smart Grouping" feature, now available in Python under MIT licence.

```bash
pip install insurance-glm-cluster
```

---

## Why the manual process is broken

The standard approach has three problems.

**Inconsistency.** Two actuaries collapsing the same vehicle make factor from the same data will produce different groupings. Not dramatically different, but enough to move rates and enough to be unexplainable in a model change documentation. The grouping reflects individual judgment about which one-way relativities are "close enough" to merge.

**No multivariate adjustment.** When you inspect vehicle make relativities in a one-way table, you are not controlling for the vehicle age, driver age, and NCD profile of each make's policyholders. Tesla drivers are young and urban. Land Rover drivers are older and rural. If you collapse vehicle makes by one-way relativity, you are partly collapsing demographics. The correct grouping merges makes with similar *adjusted* risk — adjusted for everything else in the model.

**No exposure credibility check.** After merging, how many earned car-years support each pricing band? The manual approach often produces merged groups that are thin at the extremes because the analyst eyeballs relativities, not exposure counts. A band covering three low-volume makes with 200 car-years total will have an unstable relativity.

R2VF addresses all three.

---

## The algorithm

R2VF ("Ranking to Variable Fusion") is a two-step approach that handles both ordinal factors (vehicle age, NCD) and nominal factors (vehicle make, occupation) within the same framework.

**For ordinal factors**, the problem is tractable directly: apply a fused lasso on the ordered dummy coefficients. The fused lasso penalty is `λ · Σ |β_j - β_{j-1}|`. When adjacent levels are sufficiently similar, the penalty drives their coefficient difference to exactly zero — they merge. The split-coding trick (reparameterise as `δ_j = β_j - β_{j-1}`) converts this to a standard L1 problem. No ADMM, no cvxpy.

**For nominal factors**, the ordering problem makes the naive all-pairs fused lasso intractable for K > 50. With 500 vehicle makes, you would need 124,750 penalty terms. R2VF solves this by adding a ranking step: fit a ridge-regularised GLM on all factor dummies simultaneously. The magnitude of each level's coefficient gives a data-driven ordering of that level by risk, adjusted for all other factors in the model. Once the nominal categories have an ordering, they become ordinal — and the standard fused lasso applies.

Ben Dror validated this on the NHTSA FARS 2022 accident dataset: 886 vehicle model codes, 51 US states, 59 body types. R2VF achieved CatBoost-parity log-loss with a covariate set one-third the size of ordinary Lasso, because it was merging genuinely similar levels rather than merely shrinking coefficients.

The refit step is straightforward: once the merged groupings are established, fit an unpenalised GLM on the collapsed factor dummies. This removes the shrinkage bias introduced by the lasso in step two. The final model has standard interpretable GLM coefficients.

---

## Using it

```python
from insurance_glm_cluster import FactorClusterer

clusterer = FactorClusterer(
    family='poisson',
    lambda_='bic',
    min_exposure=500,
    monotone_factors=['vehicle_age'],
    monotone_direction={'vehicle_age': 'increasing'},
)

clusterer.fit(
    X, y, exposure=exposure,
    ordinal_factors=['vehicle_age', 'ncd'],
    nominal_factors=['vehicle_make'],
)

# Inspect merged levels
lm = clusterer.level_map('vehicle_make')
print(lm.to_df())

# Transform and refit
X_merged = clusterer.transform(X)
result = clusterer.refit_glm(X_merged, y, exposure=exposure)
print(clusterer.diagnostics())
```

`lm.to_df()` returns a DataFrame with four columns: `original_level`, `merged_group`, `coefficient`, `exposure`. Every original vehicle make is mapped to its merged band. The coefficient is the unpenalised GLM estimate for that band. Exposure is the total earned car-years supporting it.

That output is also the audit trail. When a pricing committee or an FCA model change review asks why Alfa Romeo and SEAT are in the same band, you point to the R2VF coefficient from the multivariate-adjusted ridge regression, and the BIC improvement from merging them. That is a much more defensible answer than "they looked similar on the one-way".

---

## BIC lambda selection

Cross-validation for lambda selection is problematic with insurance data. Temporal autocorrelation means that randomly splitting training and validation gives you an optimistic estimate of holdout performance — consecutive policy years are correlated in ways a random split does not account for. Households with multiple policies make this worse.

BIC is the right criterion here:

```
BIC(λ) = -2 · ℓ(β̂(λ)) + K_eff(λ) · log(n)
```

where `K_eff` is the total number of distinct merged groups across all factors after applying lambda. As lambda increases, more levels merge, K_eff falls, and the penalty term decreases. BIC balances the log-likelihood (fit quality) against model complexity (number of bands). The optimal lambda is the one that minimises BIC over a grid.

This is how Earnix's Auto-GLM handles it, and it is the correct choice for actuarial work. The library also supports `lambda_='cv'` with temporal walk-forward splits for teams that prefer cross-validation — it uses [insurance-cv](https://github.com/burning-cost/insurance-cv) under the hood.

---

## Minimum exposure enforcement

After fusion, the library checks whether each merged group meets the `min_exposure` threshold. Groups that do not are absorbed into the adjacent group with the nearest coefficient value. This is a greedy procedure that runs after the regularisation, not a constraint on the optimisation itself.

The default threshold is `min_exposure=500` (earned car-years). For occupation codes, where claim counts are more informative than exposure, use `min_claims=50`. Both can be set simultaneously.

This step is what turns the statistical output into something a UK pricing actuary can actually use. A band with 200 car-years behind it is not a stable pricing level regardless of how clean the regularisation path looks.

---

## Monotonicity for ordered factors

For NCD, vehicle age, and driver age bands where the direction of the relativity is known a priori, the library enforces monotonicity post-fusion using isotonic regression on the merged group coefficients. This is a soft constraint: the regularisation step finds the groupings, and the isotonic step enforces the ordering on the resulting coefficients without re-fitting.

```python
clusterer = FactorClusterer(
    family='poisson',
    lambda_='bic',
    min_exposure=500,
    monotone_factors=['vehicle_age', 'ncd'],
    monotone_direction={
        'vehicle_age': 'increasing',
        'ncd': 'decreasing',
    },
)
```

If the data produces a non-monotone pattern after fusion, the library flags it and reports the magnitude of the imposed adjustment. A large adjustment is a signal: either the factor has a genuine non-linear relationship that monotonicity is masking, or there is a confound worth investigating. We recommend checking the diagnostics output before enforcing monotonicity blindly.

---

## What it does not do

R2VF operates within the GLM framework. It merges factor levels of existing categorical variables — it does not discretise continuous variables, discover interaction terms, or replace the modelling workflow that precedes it. You still need to decide which variables enter the model. You still need to inspect the merged groupings and override them where actuarial judgment requires it. The `FactorClusterer` is a better starting point than a one-way relativity table; it is not an autonomous tariff builder.

For continuous-to-categorical discretisation, look at [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) for smoothed age curves. For vehicle make/model hierarchies where the make/model tree structure carries information, the PHiRAT approach (Duval et al., arXiv:2304.09046) is a complement — it clusters child nodes within each parent using engineered risk features. PHiRAT support is on the roadmap.

For spatial factors (postcode sector), territory banding is a separate problem requiring spatial adjacency constraints. That belongs to [insurance-spatial](https://github.com/burning-cost/insurance-spatial).

---

## On the existing alternatives

The honest landscape:

**R's `genlasso`** implements the generalised lasso path algorithm for Gaussian responses only. No exposure, no Poisson, no Gamma. Fine for academic work; not usable for insurance frequency or severity models.

**`optbinning`** (Python) is excellent for credit scoring: binary target, Weight of Evidence, Information Value. Wrong distributional family for insurance claims data. Using WoE binning on a Poisson target produces theoretically incorrect groupings.

**`glmdisc`** (R and Python) uses a stochastic EM algorithm for discretisation and grouping in logistic regression. Logistic only.

**Earnix Auto-GLM** implements R2VF and it works — but it is proprietary, expensive, and only available inside the Earnix platform. If your team is already on Earnix, the "Smart Grouping" feature does what this library does. If you are not, there was no open alternative until now.

---

## A note on the paper

The R2VF algorithm was presented by Yuval Ben Dror (Earnix) at the Insurance Data Science Conference, London, June 2025. The arXiv preprint (2503.01521) was posted March 2025 — three months before the conference. The paper is eight pages and readable. We recommend reading it before using the library: understanding why the ranking step uses ridge rather than lasso (multicollinear vehicle makes), and why BIC is preferred over cross-validation, will make you a better user of the output.

The paper does not include Python code. This library is the first open-source implementation.

---

**[insurance-glm-cluster on GitHub](https://github.com/burning-cost/insurance-glm-cluster)** — MIT-licensed, PyPI.

**See also:**
- [Finding the interactions your GLM missed](/2026/02/27/finding-the-interactions-your-glm-missed/) — interaction detection after factor collapsing is done
- [Why your cross-validation is lying to you](/2026/02/23/why-your-cross-validation-is-lying-to-you/) — why BIC beats CV for lambda selection in insurance GLMs
- [Whittaker-Henderson smoothing for insurance pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — continuous variable smoothing, the complement to factor collapsing
