---
layout: post
title: "Nested GLMs with Neural Network Embeddings for Insurance Ratemaking"
date: 2026-03-09
author: Burning Cost
categories: [techniques]
tags: [glm, neural-network, embeddings, territory, clustering, ratemaking, motor, python, pytorch, spopt, skater, high-cardinality]
description: "Handle 800+ vehicle makes and 9,000+ postcode sectors in a multiplicative GLM using neural embeddings and spatial clustering. Auditable Python pipeline."
---

Every UK motor pricing team has a version of this problem. The base GLM is running vehicle make, driver age, NCD, vehicle age. It handles those well. Then someone asks: can we get more out of vehicle make? We have 800-plus levels. Can we use postcode sector properly instead of banding into 12 crude territories?

The honest answer, within a standard GLM framework, has always been no. A GLM with 800 vehicle make parameters is not a reliable model. Most makes have sparse data. The parameter estimates are noisy. The confidence intervals are wide. You either cap at a manageable number of levels by merging makes into coarse groups, losing discriminatory power, or you let the GLM try to fit 800 parameters and get unreliable estimates for most of them.

The same problem at postcode sector level is worse. Nine thousand-plus postcode sectors in the UK. Median claim count per sector per year for a mid-sized motor book is 20-30 claims. A GLM with sector as a categorical factor has 9,000 parameters it cannot reliably estimate. Standard practice, k-means banding into 6-20 territories, discards the spatial structure entirely and still produces territories based on volatile loss ratios.

The academic solution exists. Wang, Shi and Cao published a nested GLM framework in the North American Actuarial Journal in 2025 that addresses exactly this. It builds on entity embedding work by Shi and Shi (NAAJ 2023) and Avanzi et al.'s GLMMNet (ASTIN 2024). Until now there has been no Python implementation.

`insurance-glm-tools` is that implementation.

---

## The four-phase pipeline

The library solves the high-cardinality problem through a four-stage workflow. Each stage has a clear statistical purpose, and the output of each feeds the next.

**Phase 1: Base GLM.** Fit a standard Poisson GLM on the well-behaved structured rating factors - driver age, NCD band, vehicle age. This gives you a clean baseline and, critically, residuals. The residuals represent what the structured factors cannot explain. That unexplained variation is what the next phases target.

**Phase 2: Neural network embedding.** Train a PyTorch embedding layer on the Phase 1 residuals. The network learns a d-dimensional vector representation for each high-cardinality categorical level - each vehicle make, each postcode sector. Two levels that behave similarly in the data end up with nearby vectors. Two levels that behave differently end up far apart. This is entity embedding, the same technique Guo and Berkhahn proposed for categorical variables in 2016. The key is that the embedding is trained on Poisson deviance residuals with the base GLM's predictions as an offset - a CANN-style architecture. The embedding is not learning to predict raw claims; it is learning to explain residual risk after the base factors.

**Phase 3: Spatially constrained territory clustering.** Take the postcode sector embeddings from Phase 2 and cluster them into rating territories using the SKATER algorithm (Assuncao et al. 2006). SKATER is a minimum spanning tree method that partitions areas into connected, contiguous groups. The spatial contiguity constraint is a regulatory requirement in the UK: rating territories must be connected geographic regions. You cannot have a territory that includes central London and a postcode in Yorkshire. SKATER enforces this. The clustering uses both the embedding features (which carry risk information) and the geographic adjacency (which enforces contiguity). Disconnected components - the Orkney Islands, Shetland, Isles of Scilly - are handled explicitly via the `TerritoryClusterer`'s disconnected component logic.

**Phase 4: Nested GLM.** Assemble the final model. Take the Phase 1 base factors, add the Phase 2 embedding vectors as continuous predictors, add the Phase 3 territory labels as a categorical factor. Fit a standard Poisson GLM using statsmodels. The output is interpretable multiplicative relativities for every factor in the model - the same format as any other GLM, fully auditable, full coefficient table with standard errors. No black box.

The nesting is what makes this work. The embedding and territory steps extract signal from the high-cardinality variables. The final GLM is a standard model that an actuary can review, challenge, and sign off.

---

## In practice

```bash
uv add insurance-glm-tools
uv add "insurance-glm-tools[geo]"    # geopandas + spopt for territory clustering
```

The geo optional dependency pulls in geopandas and spopt (the spatial optimisation library that provides SKATER). This is the part with GDAL friction. If you do not have GDAL installed system-wide, the geopandas install will fail. On Ubuntu:

```bash
sudo apt-get install gdal-bin libgdal-dev
uv add "insurance-glm-tools[geo]"
```

On macOS with Homebrew: `brew install gdal` first. If you only want the embedding and GLM phases without the spatial clustering, the base install without `[geo]` works fine.

The PyTorch dependency is real. `insurance-glm-tools` pulls in PyTorch for the embedding training. Expect 2GB or so of disk, depending on your platform. For CPU-only use:

```bash
uv add insurance-glm-tools --extra-index-url https://download.pytorch.org/whl/cpu
```

### Running the pipeline

```python
import pandas as pd
from insurance_glm_tools.nested import NestedGLMPipeline

# df: a DataFrame with one row per policy-year
# Required columns: claims, exposure, plus your rating factors
pipeline = NestedGLMPipeline(
    base_formula="claims ~ driver_age_band + ncd_band + vehicle_age_band",
    embedding_categoricals=["vehicle_make", "postcode_sector"],
    embedding_dim=8,
    n_territories=40,
    spatial_weights_path="postcode_adjacency.gal",  # libpysal GAL file
)

result = pipeline.fit(
    df=df,
    exposure_col="exposure",
    claims_col="claims",
    area_col="postcode_sector",
)
```

The `spatial_weights_path` argument takes a libpysal-format GAL or GWT adjacency file. If you do not have one, the library's `build_adjacency_from_geojson()` helper constructs it from a postcode sector GeoJSON - the same process described in our [BYM2 spatial territory post]({{ site.baseurl }}{% post_url 2026-02-23-spatial-territory-ratemaking-with-bym2 %}).

The `embedding_dim=8` parameter controls the size of the embedding vectors. The Wang et al. paper found 8 dimensions sufficient for vehicle make on their data. For postcode sectors with more structural variation, 12-16 dimensions may retain more signal. Treat this as a hyperparameter: compare Phase 4 model deviance across embedding dimensions.

`n_territories=40` is the target territory count going into the final GLM. This is a business decision as much as a statistical one: 40 territories is meaningful granularity without creating underwriting headaches. The SKATER algorithm will produce exactly `n_territories` contiguous groups.

### Phase outputs

Every phase is accessible on the fitted result:

```python
# Phase 1: base GLM
print(result.base_glm.summary())

# Phase 2: embeddings (n_levels x embedding_dim)
vehicle_embeddings = result.embeddings["vehicle_make"]
print(vehicle_embeddings.shape)   # (847, 8) - 847 unique makes

# Phase 3: territory assignments
territory_map = result.territory_map
# DataFrame: postcode_sector, territory_id, territory_name

# Phase 4: final GLM
print(result.nested_glm.summary())
relativities = result.relativities()
```

`result.relativities()` returns a tidy DataFrame with one row per factor level: the GLM coefficient, the exponentiated relativity, and standard error. This is the deliverable for an actuarial peer review or a Radar import.

```python
rels = result.relativities()
print(rels[rels.factor == "territory_id"].head())
#   factor    level  coefficient  relativity  std_error
#   territory_id  T01     0.183       1.201      0.031
#   territory_id  T02     0.094       1.099      0.028
#   territory_id  T03    -0.212       0.809      0.034
#   ...
```

---

## The embedding training in detail

The Phase 2 embedding is where the statistical novelty sits. It is worth understanding what it actually does.

The `EmbeddingTrainer` takes the Phase 1 GLM predictions as an offset and trains a neural network where the only learned parameters are the embedding vectors for the high-cardinality categoricals. The network architecture is intentionally minimal: an embedding lookup, a log-sum aggregation of the active embedding vectors, added to the log of the GLM offset. The loss is Poisson deviance.

```python
from insurance_glm_tools.nested import EmbeddingTrainer

trainer = EmbeddingTrainer(
    categoricals=["vehicle_make", "postcode_sector"],
    embedding_dims={"vehicle_make": 8, "postcode_sector": 12},
    n_epochs=50,
    learning_rate=0.01,
    batch_size=2048,
)

embeddings = trainer.fit(
    df=df,
    claims_col="claims",
    exposure_col="exposure",
    glm_offset_col="base_glm_log_mu",  # log(Phase 1 predictions)
)
```

After 50 epochs on a dataset of 500,000 policies, the training takes 3-8 minutes on CPU. On a GPU, this is under a minute. The loss curve should flatten by epoch 30-40 for most insurance datasets; if it is still descending at epoch 50, increase epochs or reduce learning rate.

The embeddings can be inspected before proceeding to clustering. A sanity check worth running: do similar vehicle makes sit close together in embedding space?

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Extract makes with substantial exposure
makes = vehicle_embeddings.index.tolist()  # list of make names
vecs = vehicle_embeddings.values          # (n_makes, 8) array

# Top 5 most similar makes to "Ford"
ford_idx = makes.index("Ford")
sims = cosine_similarity(vecs[ford_idx:ford_idx+1], vecs)[0]
top5 = np.argsort(sims)[::-1][1:6]
print([makes[i] for i in top5])
# ['Vauxhall', 'Peugeot', 'Renault', 'Citroen', 'Volkswagen']
```

If the nearest neighbours are not plausible (luxury marques next to budget makes, commercial vehicles mixed with passenger cars), the embedding has not converged properly. Check the loss curve and increase training time before proceeding.

---

## Territory clustering and the SKATER algorithm

SKATER partitions a spatial graph into k connected regions. The algorithm builds a minimum spanning tree on the adjacency graph, weighted by the dissimilarity between areas' embedding vectors. It then prunes the minimum spanning tree into k subtrees, each of which forms a connected territory. The pruning criterion is the within-territory sum of squared distances from each area to its territory centroid.

This is NP-hard in general. SKATER uses a heuristic that is fast in practice but is not guaranteed to find the global optimum. For the full UK postcode sector graph (9,000-plus nodes), expect 10-20 minutes for a 40-territory solution. For district-level work (around 3,000 postcode districts), expect 2-5 minutes.

The `TerritoryClusterer` adds credibility filtering on top of SKATER: after clustering, any territory with fewer than `min_claims` observed claims is merged into its spatially adjacent neighbour with the most similar embedding centroid.

```python
from insurance_glm_tools.nested import TerritoryClusterer

clusterer = TerritoryClusterer(
    n_territories=40,
    min_claims=500,           # minimum claims per territory
    spatial_weights=weights,  # libpysal weights object
    random_state=42,
)

territory_labels = clusterer.fit(
    embeddings=sector_embeddings,   # (n_sectors, 12) DataFrame
    area_ids=sector_ids,
    observed_claims=sector_claims,
)
```

For UK islands (Orkney, Shetland, Isles of Scilly) that form disconnected components in the postcode adjacency graph, the clusterer assigns each island its nearest-neighbour territory by embedding distance. The island connection policy is logged and should be documented in your model documentation.

---

## What you get out

The final Phase 4 GLM is a standard statsmodels Poisson model. You can call `.summary()` and get the full coefficient table. You can compute the AIC, the deviance, the pearson chi-squared. You can validate it with your standard GLM validation toolkit.

The territory relativities are multiplicative factors in the same format as your base rating factors. A territory with relativity 1.20 is 20% higher risk than the base territory, after controlling for driver age, NCD, vehicle age, and vehicle make. The embedding vectors also enter as continuous predictors in the nested GLM, contributing multiplicative adjustments for vehicle make risk that are independent of the territory factor.

For Radar or Emblem import, `result.relativities()` gives you the table. Export it:

```python
result.relativities().to_csv("nested_glm_factors.csv", index=False)
```

---

## Limitations

Three honest ones.

**SKATER is NP-hard.** For large n, the SKATER heuristic may not find the globally optimal territory partition. Different random seeds can give materially different territories at the same k. We recommend running with 3-5 different seeds and comparing the resulting GLM deviance. The solution with the lowest Phase 4 deviance wins. If results are very sensitive to seed, you may have too many territories for the data to support.

**PyTorch install size.** Adding PyTorch to a pricing environment adds roughly 2GB. If you are running in a constrained environment or a large team where environment reproducibility matters, be deliberate about this. The embedding training could in principle be implemented with lighter-weight autograd machinery, but we use PyTorch because it is what the Wang et al. methodology assumes and what practitioners are most likely to already have.

**GDAL friction for the geo dependencies.** geopandas and spopt are not always straightforward to install, particularly on Windows. If you are only using the embedding layer and handing off to an existing territory model, the base install without `[geo]` avoids this entirely.

---

## Why bother - could you not just use a GBM?

A GBM will handle vehicle make and postcode sector without any of this. It will learn nonlinear interactions, implicitly encode spatial structure through lat/lon features, and outperform a GLM on held-out Gini on many motor datasets.

The problem is what you have at the end. A GBM that encodes territory through lat/lon splits does not produce a territory relativity table. There is no "SW12 is 18% above base" output. When the FCA asks how you derived your geographic factors, or when your pricing committee wants to challenge the East London loading, a GBM does not give you a satisfying answer.

The nested GLM gives you both. The embedding and clustering extract signal from the high-cardinality variables in a way that a standard GLM cannot. The final nested GLM puts that signal into a fully interpretable, auditable multiplicative model. The relativities are explainable. The territory map can be shown on a map and challenged. The vehicle make factors are a table.

This is also why the nested GLM is different from simply running a GLM on target-encoded vehicle make. Target encoding collapses the high-cardinality variable to a single continuous predictor, losing the within-variable structure. The embedding preserves the similarity structure: makes with similar risk profiles are close in embedding space, and that similarity information flows into both the territory clustering and the final GLM.

---

## Get the library

```bash
uv add insurance-glm-tools
uv add "insurance-glm-tools[geo]"    # adds geopandas + spopt
```

Source is on [GitHub](https://github.com/burning-cost/insurance-glm-tools). The library is at v0.1.0: 780 lines, 58 tests across 5 modules. The pipeline, embedding, clustering, and GLM components are stable. Planned next: support for CatBoost and XGBoost offsets in the CANN layer, and a Polars-native data path to replace the current pandas internals.

---

## References

- Wang, Y., Shi, P., & Cao, C. (2025). A nested GLM framework with neural network encoding and spatially constrained clustering in non-life insurance ratemaking. *North American Actuarial Journal*.
- Shi, P., & Shi, K. (2023). Non-life insurance risk classification using categorical embedding. *North American Actuarial Journal*, 27(1), 175-205.
- Avanzi, B., Taylor, G., Wang, M., & Wong, B. (2024). GLMMNet: a generalised linear mixed model representation for deep learning. *ASTIN Bulletin*, 54(1), 1-29.
- Assuncao, R.M., Neves, M.C., Camara, G., & Freitas, C.D.C. (2006). Efficient regionalisation techniques for socio-economic geographical units using minimum spanning trees. *International Journal of Geographical Information Science*, 20(7), 797-811.
- Guo, C., & Berkhahn, F. (2016). Entity embeddings of categorical variables. *arXiv:1604.06737*.

- [BYM2 Spatial Smoothing for Territory Ratemaking]({{ site.baseurl }}{% post_url 2026-02-23-spatial-territory-ratemaking-with-bym2 %})
- [Extracting Rating Relativities from GBMs with SHAP]({{ site.baseurl }}{% post_url 2026-02-17-extracting-rating-relativities-from-gbms-with-shap %})
- [From CatBoost to Radar in 50 Lines of Python]({{ site.baseurl }}{% post_url 2026-03-01-from-catboost-to-radar-gbm-to-glm-distillation %})
