---
layout: post
title: "Your Factor Banding Is Made Up"
date: 2026-03-14
categories: [pricing, libraries]
tags: [GLM, factor-banding, R2VF, fused-lasso, territory, SKATER, embeddings, poisson, BIC, insurance-glm-tools, python]
description: "Automated GLM factor banding for UK insurance pricing: R2VF fused lasso, neural embeddings for high-cardinality categoricals, SKATER spatial clustering."
---

Ask a pricing actuary how they grouped vehicle age into bands and the honest answer is usually: "We tried a few splits, looked at the fitted relativities, and picked something that looked reasonable." That is not a method. That is pattern-matching after the fact, and it produces factor tables that are neither statistically optimal nor reproducible.

The problem is specific to GLMs. A GBM will find its own grouping structure during training — the tree splits define the bands. A GAM will fit a smooth curve and let you read off where the function changes meaningfully. A GLM does neither. It requires you to specify the grouping up front, before fitting, and then fits parameters for the groups you gave it. The quality of the model depends on the quality of your prior decisions about where to draw lines. Most teams draw those lines with a combination of actuarial intuition, volume checks, and whatever the previous actuary did.

[`insurance-glm-tools`](https://github.com/burning-cost/insurance-glm-tools) puts a statistical method where the guesswork currently lives. Three tools in one package: R2VF factor-level clustering for ordinal factors, neural network embeddings for high-cardinality categoricals, and SKATER spatial clustering for postcode territory grouping. All of them target the same gap: between "raw GLM with too many parameters" and "black box you can't put in Radar."

```bash
uv add insurance-glm-tools
```

---

## The factor banding problem, stated precisely

A Poisson GLM with vehicle age as a raw integer has one parameter per year: 0, 1, 2, ... 15, 16+ — seventeen parameters if you code it naively. Most of those parameters are estimated on thin exposure. The fitted relativities bounce around. You smooth them by hand, grouping years that look similar. The grouping is based on visual inspection of noisy estimates. The estimates are noisy partly because you haven't grouped yet. The whole thing is circular.

The correct approach — which actuaries have known about theoretically for decades and almost never implement — is to estimate the grouping and the relativities simultaneously, using a regularisation penalty that shrinks adjacent level differences toward zero. When two adjacent levels' difference is zero, they should be in the same group.

This is the fused lasso, applied to GLMs. The paper is Ben Dror (2025), arXiv:2503.01521, title "R2VF: Regularized Ratemaking via Variable Fusion." The library implements it.

---

## R2VF: what it does under the hood

The standard one-hot GLM design matrix for vehicle age (say 15 levels) has one column per level. R2VF replaces this with a split-coded design matrix: for a factor with levels 0, 1, ..., 14, each split column encodes "is the level above the k-th threshold?" A lasso penalty on the split coefficients δ_k causes adjacent splits to fuse when the data doesn't support keeping them separate.

The mechanics:

1. Build the split-coded matrix D from all ordinal factors in one shot
2. Fit a penalised Poisson GLM via IRLS — at each Newton step, the working response problem reduces to a weighted lasso, solved by sklearn's coordinate descent
3. Decode merged groups from the δ solution: where δ_k shrinks to zero, levels k and k+1 are in the same group
4. Compute BIC over a grid of lambda values; select the most parsimonious model that the data actually supports
5. Apply a minimum-exposure constraint so no group survives on fewer than `min_exposure` policy-years
6. Refit an unpenalised GLM on the merged encoding for clean coefficient estimates

The reason we implement IRLS with an L1 inner step rather than using sklearn's `PoissonRegressor` is stated in the source code comments and worth repeating here: sklearn's Poisson implementation uses an L2 (ridge) penalty, which shrinks coefficients toward zero but cannot shrink them to exactly zero. Level fusion requires exact zeros. The IRLS+lasso approach gives true L1 penalisation with a Poisson outer loop.

In code, this is three lines:

```python
from insurance_glm_tools.cluster import FactorClusterer

fc = FactorClusterer(family='poisson', lambda_='bic', min_exposure=500)
fc.fit(df, y, exposure=exposure, ordinal_factors=['vehicle_age', 'ncd_years', 'driver_age_band'])
X_merged = fc.transform(df)
```

BIC selects the regularisation strength automatically. The `min_exposure=500` constraint means no group is left standing unless it has at least 500 policy-years of exposure behind it, which is a defensible minimum for a UK motor book — adjust it for your portfolio size.

Inspect what you got:

```python
print(fc.level_map('vehicle_age').to_df())
```

```
┌─────────────────┬───────┬─────────────┬──────────────────┐
│ original_level  ┆ group ┆ coefficient ┆ group_exposure   │
╞═════════════════╪═══════╪═════════════╪══════════════════╡
│ 0               ┆ 0     ┆  0.031      ┆ 1842.3           │
│ 1               ┆ 0     ┆  0.031      ┆ ...              │
│ 2               ┆ 0     ┆  0.031      ┆ ...              │
│ 3               ┆ 1     ┆  0.000      ┆ 3214.7           │  ← base
│ 4               ┆ 1     ┆  0.000      ┆ ...              │
│ 5               ┆ 1     ┆  0.000      ┆ ...              │
│ 8               ┆ 2     ┆  0.042      ┆ 2891.1           │
│ 9               ┆ 2     ┆  0.042      ┆ ...              │
│ ...             ┆ ...   ┆  ...        ┆ ...              │
│ 13              ┆ 3     ┆  0.081      ┆ 987.4            │
│ 14              ┆ 3     ┆  0.081      ┆ ...              │
└─────────────────┴───────┴─────────────┴──────────────────┘
```

Vehicle age years 0–2 are in group 0 (newer vehicles, slightly elevated frequency). Years 3–7 are the base. Years 8–12 are group 2. Years 13–14 are group 3. Four groups from fifteen levels. BIC decided four was the right number, not five because someone split into quintiles and not three because it looked clean.

Unpenalised refit:

```python
result = fc.refit_glm(X_merged, y, exposure=exposure)
```

The output of `refit_glm` is a `GLMResults` object from statsmodels — full confidence intervals, deviance, AIC, BIC, the works. The banded factor table goes into Radar, Emblem, or whatever your rating engine is. It looks identical to a manually-banded GLM output. The difference is that the grouping decisions are reproducible, statistically justified, and attached to a paper trail.

---

## Where R2VF beats manual banding

The README benchmarks R2VF against manual quintile banding on 20,000 synthetic UK motor policies with 30 postcode districts and a known true grouping structure. The result is what you'd expect from a penalised method: R2VF produces fewer groups than quintile banding, with equivalent or better Poisson deviance, and higher Rand Index (recovery of the true DGP structure). The parsimony advantage is real — fewer groups means fewer parameters and a more credible estimate for each.

The benchmark is deliberately honest about the limits. When factor levels have genuine monotone structure that you want to preserve — NCD years going monotonically negative — R2VF will find it, but if you are imposing monotonicity as a business rule, isotonic regression constraints are the more principled approach. When exposure is so thin that even minimum-exposure constraints cannot save you, Bühlmann-Straub credibility weighting is better than any regularisation approach.

R2VF is the right tool when you have enough data to estimate groupings but currently do it by eye. It is not a substitute for actuarial judgment about what factors should look like; it is a check on whether what you have drawn actually reflects the data.

---

## High-cardinality categoricals: the embedding approach

Factor banding deals with ordinal factors where you want to merge adjacent levels. The second problem is different: high-cardinality nominal categoricals like vehicle make/model (300+ categories) or occupation class (150+ codes) where dummy-coding is theoretically correct but produces 300 parameters, mostly estimated on thin exposure.

The `insurance_glm_tools.nested` subpackage implements the Wang, Shi, Cao (NAAJ 2025) framework. Instead of dummy-coding, train a neural network to learn a dense embedding for each category from the GLM residuals, then use those embeddings in the outer GLM.

The pipeline is four phases:

1. **Base GLM** on structured factors (age band, NCD, vehicle group) — the usual GLM
2. **Embedding network** trained on base GLM residuals; each vehicle make/model is mapped to a compact vector, typically 10–20 dimensions
3. **Territory clustering** via SKATER (optional, requires spatial data): groups postcode sectors into contiguous territories
4. **Outer GLM** with structured factors + embedding dimensions + territory fixed effects

```python
from insurance_glm_tools.nested import NestedGLMPipeline

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncd_years + vehicle_group",
    embedding_epochs=20,
    # pass geo_gdf for spatial territory clustering, or omit for non-spatial
)
pipeline.fit(
    df, y, exposure,
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncd_years", "vehicle_group"],
)

relativities = pipeline.relativities()
```

The embedding dimensions are numeric features in the outer GLM. The outer GLM is still a GLM — interpretable, with a factor table, with statsmodels confidence intervals. The embedding is a pre-processing step that compresses the high-cardinality information into a form the GLM can handle without a parameter explosion.

The alternative most teams use is to group vehicle makes manually into "groups" — ABI vehicle groups, or proprietary groupings — which embeds the same prior decisions we criticised in the factor banding case. The embedding approach learns the grouping structure from the data. It does not require a prior view on whether a Volkswagen Golf is in the same risk category as a Seat Leon.

---

## Territory banding: postcode grouping with spatial constraints

The third tool in the package handles postcode territory grouping. The problem is similar to factor banding but with a hard constraint: territory groups must be spatially contiguous. You cannot group the SE1 postcode sector with a group in the North West just because their fitted claim frequencies happen to be similar. A territory is a geographic area, and the boundary must make sense on a map.

SKATER (Spatial 'K'luster Analysis by Tree Edge Removal) solves this by constructing a minimum spanning tree over postcode sector centroids, then pruning edges to create contiguous clusters of postcode sectors. The clusters respect spatial contiguity by construction — they cannot be geographically fragmented — and the number of territories is a parameter you specify based on your model governance requirements.

This is accessed via the nested pipeline when you have a GeoDataFrame of postcode sector polygons:

```python
uv add "insurance-glm-tools[spatial]"
```

```python
pipeline = NestedGLMPipeline(
    base_formula="age_band + ncd_years + vehicle_group",
    n_territories=15,   # how many territory groups to form
    embedding_epochs=20,
)
pipeline.fit(
    df, y, exposure,
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncd_years", "vehicle_group"],
    geo_gdf=postcode_gdf,          # GeoDataFrame with postcode sector polygons
)
```

The spatial pipeline requires geopandas, libpysal, and spopt. If you do not have spatial data, omit `geo_gdf` and the pipeline skips territory clustering entirely — you still get the embedding benefit.

The appeal of SKATER over the common alternative (fitting postcode district dummies, sorting them by relativity, and grouping by percentile) is that quintile grouping produces geographically incoherent territories. A North East postcode district and a South West postcode district with similar fitted relativities end up in the same group because the sorted list put them there. SKATER cannot do this. Its output is a set of territories that a compliance team can draw on a map and a regulator can inspect without finding oddities.

---

## Combining the tools

The reason both subpackages are in one library is that they compose. The natural workflow on a UK motor book:

1. Use R2VF to band vehicle age, driver age, NCD years, and any other ordinal factors
2. Pass the banded factors as the base GLM formula in `NestedGLMPipeline`
3. Let the embedding network handle vehicle make/model
4. If you have postcode data, let SKATER form the territory grouping

The outer GLM receives clean, statistically-defended factor encodings at every level. The factor tables that come out are interpretable, go into the rating engine without modification, and can be defended in a pricing review because every grouping decision has a statistical rationale.

---


The gap between "raw GLM with hand-drawn bands" and "black box GBM" is not a technical inevitability. GLMs remain the regulatory workhorse in UK personal lines — they produce factor tables, they compose with re-basing and credibility, and they map cleanly onto rating systems that were built around the additive log-linear structure. The reason teams reach for GBMs is partly genuine predictive performance and partly frustration with how labour-intensive good GLM feature engineering is.

Automating the labour-intensive parts — the banding decisions, the high-cardinality encodings, the postcode groupings — does not change the fundamental GLM structure. It changes the quality of the inputs. A GLM with R2VF-banded factors and SKATER territories is a better model than a GLM with manually-banded factors and quintile territories, not because the algorithm is smarter than the actuary but because BIC is more consistent than intuition and SKATER cannot accidentally create a geographically fragmented territory.

The tools do not replace actuarial judgment. The minimum exposure constraint, the number of territories, the choice of which factors to treat as ordinal versus nominal — those are all inputs the actuary controls. What gets automated is the execution of the grouping, not the decision about what grouping makes sense.

---

`insurance-glm-tools` is open source under the MIT licence at [github.com/burning-cost/insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools). Install with `uv add insurance-glm-tools`. Core requires Python 3.10+, numpy, scipy, scikit-learn, statsmodels, and torch. Spatial features require the `[spatial]` extra.

- [Nested GLMs with Neural Network Embeddings for Insurance](/2026/03/09/nested-glms-with-neural-network-embeddings-for-insurance/)
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/)
