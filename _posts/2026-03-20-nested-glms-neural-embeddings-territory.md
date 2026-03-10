---
layout: post
title: "Nested GLMs and Neural Embeddings for Territory Ratemaking"
date: 2026-03-20
categories: [libraries, pricing]
tags: [nested-GLM, entity-embeddings, territory, SKATER, postcode, high-cardinality, insurance-nested-glm, spatial-clustering, neural-network, statsmodels, geopandas, motor, python]
description: "1,800 UK postcode sectors cannot go into a GLM as dummies. insurance-nested-glm implements the Wang, Shi, Cao (NAAJ 2025) four-phase pipeline: base GLM, neural embedding of high-cardinality categoricals, SKATER spatially constrained clustering for territory bands, and a readable outer GLM with a full relativities table."
---

The standard territory problem in UK motor pricing is this: you have around 1,800 postcode sectors. A GLM cannot take 1,800 dummies - you do not have the degrees of freedom, the standard errors will be useless, and the resulting factor table cannot be filed. So you band them. You cluster on observed loss ratios, apply a five-group k-means, get five territory bands, and move on.

The problem with this approach is well understood but rarely fixed. K-means on loss ratios ignores geography. You end up with territories that are not contiguous - a band containing patches of central London, outer Leeds, and a suburb of Birmingham is not a territory in any meaningful sense. The credibility of individual sectors is ignored: a sector with 120 policies has the same weight in the clustering as one with 6,000. And the cluster features (raw loss ratios) conflate the territory effect with all the other rating factors in the mix - vehicle group, driver age, NCD - because you have not controlled for them.

[`insurance-nested-glm`](https://github.com/burning-cost/insurance-nested-glm) solves all three problems in a single pipeline.

```bash
uv pip install insurance-nested-glm
# With spatial clustering:
uv pip install "insurance-nested-glm[spatial]"
```

---

## The four-phase pipeline

The library implements the nested GLM framework from Wang, Shi, Cao (NAAJ 2025, 29(3)). The architecture is:

**Phase 1 - Base GLM.** Fit a standard Poisson or Gamma GLM on the structured factors you trust: age band, NCD, vehicle group, cover type. This produces a log-rate prediction for every policy that will serve as an offset in the next phase.

**Phase 2 - Embedding.** Train a shallow neural network with entity embeddings on the GLM residuals, with the base GLM log-prediction as an offset (CANN-style). The network maps each high-cardinality categorical - vehicle make/model, postcode sector - to a dense vector. Because the base GLM log-rate enters as an offset, the network learns only what the GLM missed, not the full pricing signal. This matters: it means the embeddings represent the residual structure of the high-cardinality variables, stripped of the main structured factors. The embeddings converge faster and overfit less than they would if training on the raw targets.

**Phase 3 - Territory clustering.** Use the learned postcode sector embeddings as input features to SKATER spatial clustering. Every resulting territory is geographically contiguous by construction. Small territories below an exposure threshold are merged into their nearest neighbour.

**Phase 4 - Outer GLM.** Fit a final Poisson or Gamma GLM on the structured factors, the embedding vectors as continuous regressors, and the territory fixed effect. The output is a statsmodels GLM - coefficient table, standard errors, LRT results, AIC. Not a black box.

The full pipeline runs with a single `fit()` call.

---

## Why this beats manual territory banding

The clustering in phase 3 is driven by the embedding vectors from phase 2, not raw loss ratios. That distinction matters.

Raw loss ratios are noisy (a sector with 80 policies has a highly variable observed frequency), confounded (they absorb all the compositional effects - old cars, young drivers, low NCD - that differ between sectors), and not exposure-weighted in standard k-means implementations. The clusters inherit all of those problems.

Embedding vectors are stable learned representations that encode the residual structure of postcode sector, controlling for the base GLM's structured factors. Two sectors with similar vectors - similar residual patterns after accounting for age, NCD, and vehicle group - will cluster together regardless of whether their raw loss ratios happened to be similar in the training period. This produces territories that are more stable over time and more interpretable: the territory factor represents a genuine geographic effect, not a noise-amplified mix of everything the GLM left on the floor.

SKATER enforces contiguity. Under UK insurance regulation, territory bands must be geographically interpretable. A band that is not contiguous cannot be defended. SKATER (Spatial K-luster Analysis by Tree Edge Removal) builds a minimum spanning tree over the spatial units and prunes tree edges to form contiguous subtrees. Every territory is a connected subgraph of the adjacency graph.

The library handles UK island geography automatically. Channel Islands, Isle of Man, Orkney, and Shetland all form disconnected components in a Queen contiguity adjacency graph. The `TerritoryClusterer` detects these components via BFS, clusters each independently, and merges the labels.

---

## Running the pipeline

```python
import pandas as pd
import numpy as np
from insurance_nested_glm import NestedGLMPipeline

df = pd.read_parquet("policies.parquet")
y = df["claim_count"].to_numpy()
exposure = df["earned_exposure"].to_numpy()

pipeline = NestedGLMPipeline(
    base_formula="age_band + ncb + vehicle_group",
    family="poisson",
    n_territories=200,
    min_territory_exposure=500,
    embedding_epochs=50,
)

pipeline.fit(
    df,
    y,
    exposure,
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncb", "vehicle_group"],
)

# Readable multiplicative relativities from the outer GLM
print(pipeline.relativities())

# Full statsmodels summary with standard errors, AIC
print(pipeline.summary())

# Predictions
pred = pipeline.predict(df, exposure)
```

With spatial territory clustering, pass a GeoDataFrame of postcode sector polygons:

```python
import geopandas as gpd

geo_gdf = gpd.read_file("postcode_sectors.gpkg")

pipeline.fit(
    df,
    y,
    exposure,
    geo_gdf=geo_gdf,
    geo_id_col="postcode_sector",
    high_card_cols=["vehicle_make_model"],
    base_formula_cols=["age_band", "ncb", "vehicle_group"],
)

fig = pipeline.plot_territories(geo_gdf, geo_id_col="postcode_sector")
fig.savefig("territories.png", dpi=150)
```

The postcode sector polygons are available from the ONS Open Geography Portal (postcode sector boundaries, available under the Open Government Licence). The `geo_id_col` must be present in both the policy DataFrame and the GeoDataFrame.

---

## Using the embedding step in isolation

If you want the embedding layer without the full pipeline - for example, to feed vehicle make/model embeddings into an EBM or CatBoost model - `EmbeddingTrainer` is available separately:

```python
from insurance_nested_glm import EmbeddingTrainer

trainer = EmbeddingTrainer(
    cat_cols=["vehicle_make_model"],
    epochs=50,
    hidden_sizes=(64, 32),
)
trainer.fit(df, y, exposure=exposure, offset=base_log_pred)

# Dense vectors, shape (n_policies, total_embedding_dim)
emb = trainer.transform(df)

# DataFrame: category level -> embedding coordinates
frames = trainer.get_embedding_frame()
print(frames["vehicle_make_model"].head())
```

Embedding dimension defaults to `min(50, ceil(n_levels / 2))` per column - the rule of thumb from Guo and Berkhahn (2016). For a vehicle make/model column with 2,800 levels that gives 50 dimensions. Override with `embedding_dims={"vehicle_make_model": 20}` if you want more compression.

The `offset` parameter is critical. Pass the base GLM log-prediction as the offset so the network learns only the residual structure of the high-cardinality variable. Without it, the network will learn the full pricing signal and overfit on the rare categories.

---

## The territory clusterer

`TerritoryClusterer` can also be used standalone, feeding in any embedding coordinates:

```python
from insurance_nested_glm import TerritoryClusterer

tc = TerritoryClusterer(n_clusters=200, min_exposure=500, method="skater")
tc.fit(geo_gdf, feature_cols=["emb_0", "emb_1", "emb_2"], exposure=unit_exposure)

print(tc.labels_)   # pd.Series of 1-indexed territory labels, aligned with geo_gdf
```

The `min_exposure` filter merges any territory below 500 earned car years into its nearest neighbour by centroid distance. This prevents the outer GLM from fitting a territory fixed effect on a level with one policy. Choose the threshold based on your portfolio size and the credibility standard you apply to other rating factors - we typically use 300-500 car years for a mid-size UK motor book.

---

## What you get out of the outer GLM

Because phase 4 fits a statsmodels GLM rather than a ML model, the output is auditable. `pipeline.relativities()` returns a pandas DataFrame with one row per coefficient - structured factors, territory fixed effects, and embedding regressors - expressed as multiplicative relativities (exp of the coefficient). Standard errors are included. You can compute likelihood ratio tests between nested models. AIC and BIC are available via `pipeline.outer_glm_.aic()` and `.bic()`.

The embedding regressors appear in the relativity table as continuous factors with p-values. If the embedding vectors carry no incremental explanatory power beyond the base factors and territory, they will have large standard errors and can be dropped. If they do carry power, the coefficient will be well-determined and you have a quantified case for including vehicle make/model as a pricing factor.

---

## When this beats manual territory banding

This approach is worth the additional complexity when your portfolio has enough geographic spread and volume that territory genuinely matters. For a 50,000-policy niche book written in a small geographic area, a three-band territory structure fitted by hand is probably sufficient.

The nested GLM pipeline earns its keep when:

- You have a large national motor book where territory is a meaningful 5-10% of total variance
- You are failing a regulatory peer review on territory because your current bands are not contiguous or defensible
- You have vehicle make/model or another high-cardinality categorical that you are currently dropping or crudely binning
- You want a territory structure that updates when you retrain the model rather than requiring annual manual re-banding

The pipeline currently supports Poisson (frequency) and Gamma (severity) families. Pure premium Tweedie and multi-peril models are on the roadmap.

---

## Installation

```bash
uv pip install insurance-nested-glm
uv pip install "insurance-nested-glm[spatial]"   # geopandas, libpysal, spopt
uv pip install "insurance-nested-glm[plot]"       # matplotlib territory maps
uv pip install "insurance-nested-glm[all]"        # everything
```

Requires PyTorch (for the embedding network) and statsmodels (for the outer GLM). Python 3.10+. Source at [github.com/burning-cost/insurance-nested-glm](https://github.com/burning-cost/insurance-nested-glm).

---

**See also:**
- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/) - the Bayesian alternative for territory smoothing, works well when you want posterior uncertainty on territory effects rather than a fixed effect in a GLM
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) - how to get BYM2 territory factors into Emblem or Radar once fitted
- [EBMs for Insurance Tariff Construction](/2026/03/20/ebms-for-insurance-tariff-construction/) - if you want a non-parametric main effects model rather than a GLM for the outer layer
