---
layout: post
title: "BYM2 vs Contiguity-Constrained Clustering for Rating Territories"
date: 2026-03-31
categories: [spatial, territory-rating]
tags: [spatial, territory-rating, BYM2, ICAR, CCHA, SCHC, SKATER, pygeoda, spopt, clustering, Ward-linkage, contiguity, insurance-spatial, motor, postcode, uk-motor, Wang-Shi-Cao, NAAJ, GLM, neural-network, rating-territories]
description: "Wang, Shi, and Cao (NAAJ 2025) propose clustering NN-encoded residuals under a spatial contiguity constraint to produce hard territory zones. Here is how that differs from BYM2 smoothing, when each is the right choice, and how to run the clustering step in Python today."
---

There are two distinct things a pricing team might want from a spatial territory model. The first is a relativity surface: one factor per postcode sector, smoothed across neighbours, with uncertainty intervals, usable directly in a GLM or rating engine. The second is a territory map: a fixed number of contiguous zones, each with a single factor applied to every sector inside it.

These look like the same problem. They are not. BYM2 answers the first question. Contiguity-constrained hierarchical clustering — CCHA, or equivalently SCHC in the PySAL literature — answers the second. Wang, Shi, and Cao (NAAJ, Vol 29 No 3, 2025, DOI: 10.1080/10920277.2024.2442416) put a neural network embedding step in front of the clustering. That is the claimed novelty. The clustering itself is not new and is already available pip-installable.

This post explains the difference between the two approaches, when each is appropriate for a UK insurer, and how to run the CCHA step using `pygeoda` or `spopt` without waiting for the paper's authors to publish code they have not published.

---

## What the paper does

Wang et al. work within a three-component nested GLM framework.

**Stage 1: base GLM.** A standard Poisson or Gamma GLM with the tractable predictors — driver age, vehicle class, etc. Territory is excluded or coarsely banded. The residuals from this model carry the unmodelled spatial risk structure.

**Stage 2: neural network encoding.** A neural network with categorical embedding layers takes those residuals and encodes the high-cardinality categorical variables — most importantly raw location identifiers such as postal codes or FSAs — into a low-dimensional dense vector per geographic unit. The embedding step is what Wang et al. claim as their contribution. The idea is that the embedding aggregates information across all high-cardinality variables jointly, producing a richer location signal than the raw O/E alone. A learned 2D or 3D embedding vector per location then feeds into the clustering step.

**Stage 3: CCHA.** The embedding vectors are clustered subject to spatial contiguity constraints using contiguity-constrained hierarchical agglomeration with Ward's linkage. Only pairs of clusters sharing a geographic boundary are permitted to merge at each step. The result is exactly *k* contiguous territory zones, where *k* is set by the user.

The territories from Stage 3 re-enter the base GLM as a categorical variable and the whole system is fitted iteratively. On Brazilian auto insurance claim frequency data, it outperforms the vanilla GLM without territory.

---

## How CCHA works

CCHA is agglomerative Ward-linkage clustering with a contiguity mask. Ward's linkage minimises the total within-cluster variance at each merge; the contiguity mask means only geographically adjacent clusters are candidate merge partners. The spatial adjacency graph is built from unit boundary overlap in the usual way — Queen or Rook contiguity, same as the adjacency matrix BYM2 uses.

At each step the algorithm asks: of all adjacent cluster pairs, which merge minimises the increase in total within-cluster sum of squares? Merge that pair. Repeat until *k* clusters remain.

The algorithm is not new. It appeared in the geostatistics literature in the 1980s and has been standard in SAS actuarial toolkits since at least 2010 (NESUG proceedings, Zou and Diehl). In Python it is available as:

```python
# pygeoda — SCHC with Ward linkage
import pygeoda
w = pygeoda.queen_weights(gdf)               # adjacency from GeoDataFrame
clusters = pygeoda.schc(k=20, w=w, data=embedding_array, linkage="ward")

# spopt — SKATER (same class of algorithm, slightly different objective)
from spopt.region import Skater
model = Skater(gdf, w, attrs_name=["emb0", "emb1"], n_clusters=20)
model.solve()
gdf["territory"] = model.labels_
```

Both are pip-installable. `pygeoda` wraps the underlying C++ GeoDa library and handles adjacency graphs with islands (isolated areas get assigned to their nearest neighbour). `spopt.region.Skater` is pure-Python PySAL and is somewhat slower on large grids but easier to inspect.

If you already have `insurance-spatial` installed, you already have `libpysal`, which means `spopt` installs without conflict. `pygeoda` adds a separate C++ dependency but is the faster option for 9,000-sector UK postcode grids.

---

## What BYM2 does differently

BYM2 (Besag-York-Mollié 2, Riebler 2016 parameterisation) is a Bayesian GLM with a spatial random effect. Each geographic unit gets its own territory effect drawn from a prior that links it to its neighbours via an intrinsic conditional autoregressive (ICAR) structure. The fitting is MCMC; the output is a posterior distribution per sector.

The practical differences from CCHA are sharp:

**Output type.** BYM2 produces one relativity per sector with a full credible interval. CCHA produces *k* zones, each with a single factor. Under BYM2, Brighton BN1 1 and BN1 2 get slightly different factors if their claim experience differs. Under CCHA with *k*=20, they are either in the same territory and get identical factors, or in different territories and get factors that may jump discontinuously at the boundary even though the sectors are adjacent.

**Smoothing mechanism.** BYM2 smooths continuously. The ICAR prior pulls each sector toward its neighbours' values, with the strength of that pull governed by the rho parameter (proportion of variance attributable to spatial structure vs IID noise). There are no hard zone boundaries; the relativity surface is smooth. CCHA partitions. The boundary between territory 7 and territory 8 is a hard cut. On one side of a postcode boundary the factor may jump 15% purely because the algorithm decided that was the optimal partition given *k*=20.

**Handling thin data.** BYM2 borrows strength from neighbouring sectors through the ICAR prior. A rural sector with 18 claims gets pulled toward its well-exposed neighbours and produces a credible estimate. CCHA assigns it to a territory — but the territory-level factor for that zone will be dominated by the well-exposed sectors within it. There is implicit pooling, but no explicit uncertainty quantification per sector.

**rho and diagnostics.** BYM2 estimates rho — the proportion of spatial variance that is genuinely spatial rather than IID noise. A rho near 1.0 says the risk surface is spatially coherent; a rho near 0 says the territory variation is mostly random sector-to-sector noise. CCHA gives you no equivalent. You produce zones regardless of whether coherent spatial structure exists in the data to justify them.

**Computational speed.** CCHA on 9,000 postcode sectors runs in seconds on a laptop. BYM2 via MCMC with `nutpie` sampler takes 10-30 minutes depending on chain count and convergence. For quarterly territory refreshes this is acceptable. For iterative exploratory analysis trying different values of *k*, CCHA wins on iteration speed.

---

## The NN embedding step: does it matter?

The Wang et al. paper's distinguishing claim is that clustering on NN-learned embeddings outperforms clustering on raw O/E residuals, because the embedding aggregates information from all high-cardinality categorical variables jointly rather than relying on the spatial signal alone.

This is plausible. If your location residuals are noisy because you have thin postcode data, but your vehicle class and driver age distributions vary systematically by area, the NN embedding can pick up that joint structure and produce a denser location signal to cluster on.

Whether it is worth the additional complexity is a separate question. There is no public code. The paper uses Brazilian auto data; the embedding architecture and training details are specified at the level of a methods section rather than a reproducible specification. Without a reference implementation to validate against, reproducing it faithfully from the paper carries real risk of divergence from what the authors actually ran.

For a UK pricing team who wants contiguous territories today, the practical path is: run BYM2 via `insurance-spatial` to get posterior mean territory effects per sector, use those posterior means as the input feature vector to CCHA/SCHC, and cluster to *k* zones. You get spatially smoothed inputs to the clustering (which addresses the noise problem the NN embedding is intended to solve) without a PyTorch dependency and without implementing a neural network from a methods section.

```python
from insurance_spatial import AdjacencyMatrix, BYM2Model
import pygeoda, geopandas as gpd

# Stage 1: BYM2 smoothed relativities
adj = AdjacencyMatrix.from_geojson("postcode_sectors.geojson")
model = BYM2Model(claims=df["claims"], exposure=df["exposure"], adj=adj)
result = model.fit()
relativities = result.territory_relativities()  # posterior means per sector

# Stage 2: CCHA on smoothed signal
gdf = gpd.read_file("postcode_sectors.geojson")
gdf["relativity"] = relativities["relativity"].values
w = pygeoda.queen_weights(gdf)
clusters = pygeoda.schc(k=20, w=w, data=[gdf["relativity"].tolist()], linkage="ward")
gdf["territory"] = clusters.clusters
```

The adjacency matrix `adj` from `insurance-spatial` and the `queen_weights` from `pygeoda` encode the same neighbourhood structure. You are using BYM2 to produce a credibility-smoothed signal, then CCHA to partition it into hard zones.

---

## When to use each in UK insurance

**Use BYM2 (`insurance-spatial`) when:**

The deliverable is a per-sector factor table feeding directly into a GLM or rating engine. You want credible intervals to support pricing committee and Consumer Duty governance documentation. Your territory review is about refining the risk surface, not about drawing zone boundaries. This is the common case for UK motor and home lines running annual territory reviews.

**Use CCHA when:**

You need a bounded set of contiguous zones for a specific regulatory or operational purpose. UK examples where this is live: Flood Re zone construction (Flood Re uses five flood zones; the zone boundaries must be contiguous and defensible), commercial property territory banding for scheme pricing, or any situation where a third party (broker, MGA, Lloyd's syndicate) requires a categorical territory factor rather than a continuous relativity surface. US and Canadian regulatory frameworks typically impose cardinality constraints on territory zones that make CCHA-style algorithms a compliance requirement; UK lines currently do not, but FCA scrutiny of geographic rating has increased since the pricing practices market study.

**Use BYM2 then CCHA when:**

You need hard territory zones but your per-sector data is thin. The BYM2 smoothing step produces a stable, credibility-weighted signal per sector that is more appropriate as CCHA input than raw O/E. This is the pipeline we would run for a UK insurer needing to produce a new territory structure.

**Neither is appropriate when:**

Your spatial autocorrelation is driven by omitted covariates you have not included yet. If flood zone, subsidence band, crime rate quintile, and distance-to-coast are not in your base GLM, your territory residuals are proxying for these. Adding a spatial model on top of missing features is a modelling debt, not a modelling solution. Fix the feature set first.

---

## Limitations of the Wang et al. paper

The paper is methodologically clean but has three limitations worth flagging.

No public code. The NN encoding step is described at methods-section level without a reproducible specification. There is no arXiv preprint and no GitHub repository. Tandfonline does not have supplementary code. For a paper whose main contribution is a specific neural architecture applied to insurance ratemaking, this is a significant gap.

Single dataset. The validation uses Brazilian auto insurance claim frequency data. There is no sensitivity analysis to choice of *k*, no comparison against BYM2 or other spatial smoothing baselines, and no test on other lines or geographies. The demonstrated improvement over vanilla GLM is a low bar; it does not establish that CCHA on NN embeddings outperforms CCHA on BYM2-smoothed residuals, which is the relevant comparison.

No spatial diagnostics. The paper does not report Moran's I on the residuals from the final nested model, which would confirm whether the territory construction has actually absorbed the spatial autocorrelation. For a paper about spatial modelling, this is an odd omission.

---

## The bottom line

BYM2 smooths. CCHA partitions. They answer different questions, and trying to substitute one for the other produces the wrong answer to the wrong question.

For UK motor and home pricing, the common need is a smooth relativity surface with credible intervals — that is BYM2 via `insurance-spatial`. When you need hard territory boundaries — for Flood Re-style zones, commercial banding, or regulatory cardinality constraints — CCHA via `pygeoda.schc` or `spopt.region.Skater` is the tool. The Wang et al. NN embedding step is an interesting idea without a reference implementation; the simplest production-ready substitute is to feed BYM2 posterior means into CCHA rather than raw O/E residuals.

The entire CCHA clustering step is pip-installable today. There is no new library required.

---

**Paper:** Wang, R., Shi, H., and Cao, J. (2025). 'A Nested GLM Framework with Neural Network Encoding and Spatially Constrained Clustering in Non-Life Insurance Ratemaking.' *North American Actuarial Journal*, Vol 29(3), pp. 645-661. DOI: [10.1080/10920277.2024.2442416](https://doi.org/10.1080/10920277.2024.2442416).

Related posts:

- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/) — the ICAR prior, rho estimation, and why it scales to 9,000 postcode sectors
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) — the full pipeline from adjacency matrix to Emblem export
- [BYM2 Territory Modelling: Posterior Uncertainty and Year-on-Year Stability](/2026/03/15/your-territory-model-ignores-spatial-autocorrelation/) — why the posterior interval is the actual product
- [Spatial Error Correction vs Spatial Smoothing: Two Different Questions in Territory Rating](/2026/03/26/spatial-error-correction-vs-spatial-smoothing-territory-rating/) — frequentist spatial GBM vs Bayesian BYM2
