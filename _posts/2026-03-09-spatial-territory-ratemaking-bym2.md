---
layout: post
title: "Getting Spatial Territory Factors Into Production"
date: 2026-03-09
categories: [techniques]
tags: [spatial, territory, BYM2, ICAR, pymc, insurance-spatial, motor, python, catboost, polars, radar, emblem]
description: "How to go from a fitted CatBoost frequency model to BYM2 spatial territory factors for Emblem or Radar - with the actual data engineering, convergence checks, and Polars joins."
---

We [wrote previously]({{ site.baseurl }}{% post_url 2026-02-23-spatial-territory-ratemaking-with-bym2 %}) about why k-means territory banding is statistically wrong and what the BYM2 model does instead. That post covers the theory. This one covers the plumbing.

The practical problem: a CatBoost frequency model running on a motor book. It handles vehicle group, NCD, driver age, annual mileage. Territory is in there as a hashed high-cardinality categorical. The GBM learns something from it, but what? There is no extractable territory factor table. The peer reviewer wants one. The FCA's Consumer Duty review is asking how geographic factors are derived. We need an answer.

`insurance-spatial` is the piece of Python that sits between a base model and a rating engine. This post walks through the full pipeline.

---

## The two-stage structure

The library is built around a two-stage workflow, and understanding why the stages are separate matters before looking at any code.

Stage 1 is the existing model - CatBoost, GLM, whatever. It captures risk variation driven by vehicle, driver, and policy characteristics. We fit it without territory, or with a crude district-level offset, and produce a fitted expected frequency per policy.

Stage 2 is `insurance-spatial`. It takes the residual geographic variation - what Stage 1 missed because it had no good territory model - and fits a BYM2 spatial Poisson model to that residual. The output is a territory factor per postcode sector, with proper Bayesian credibility intervals, that slots back into the rating engine as a fixed offset.

The separation is deliberate. The spatial model is auditable on its own. We can re-run it annually without touching the base model. The territory factor table is a standard actuarial artefact: one row per sector, one log-relativity, signed off independently. That is what a peer reviewer can evaluate.

Running BYM2 on raw claims data and trying to simultaneously learn vehicle and territory effects makes the model harder to explain, harder to update, and harder to sign off. The two-stage approach keeps each piece legible.

---

## Stage 1: Aggregate to sector level

Start with a fitted CatBoost frequency model. The first data engineering step is to get from policy-level predictions to sector-level observed and expected claim counts.

```python
import polars as pl
import numpy as np

# df is a Polars DataFrame with the motor book
# Columns: policy_id, postcode_sector, exposure, claims, plus rating variables
# fitted_freq: CatBoost model's frequency prediction per policy (from .predict())

# Expected claims = fitted frequency * exposure per policy
df = df.with_columns([
    (pl.col("fitted_freq") * pl.col("exposure")).alias("expected_claims")
])

# Aggregate to sector level
sector_df = (
    df.group_by("postcode_sector")
    .agg([
        pl.col("claims").sum().alias("observed_claims"),
        pl.col("expected_claims").sum().alias("expected_claims"),
        pl.col("exposure").sum().alias("total_exposure"),
        pl.len().alias("n_policies"),
    ])
    .sort("postcode_sector")
)

print(sector_df.head(5))
# shape: (11200, 5)
# postcode_sector | observed_claims | expected_claims | total_exposure | n_policies
# AB10 1          |              14 |           11.83 |          987.3 |        412
# AB10 2          |               8 |            9.41 |          784.2 |        327
# ...
```

The `expected_claims` column is the key quantity. It is the base model's prediction of how many claims sector AB10 1 should have had, given the mix of vehicles and drivers in that sector, absent any territory effect. The ratio of `observed_claims` to `expected_claims` - the O/E ratio - is the residual geographic signal we want the BYM2 model to explain.

Filter out sectors with very low exposure before fitting. A sector with 3 policies and 0 claims contributes almost nothing to the model but adds a node to the graph the sampler has to handle. We drop sectors below 100 earned car-years, which removes roughly 200-300 sparse rural sectors from the UK motor graph.

```python
sector_df = sector_df.filter(pl.col("total_exposure") >= 100.0)

# Convert to numpy for insurance-spatial
observed = sector_df["observed_claims"].to_numpy().astype(np.int64)
expected = sector_df["expected_claims"].to_numpy()
sector_codes = sector_df["postcode_sector"].to_list()
```

---

## Stage 2: Build the adjacency

Polygon boundaries are required to build the adjacency graph. ONS does not publish official postcode sector shapefiles. The standard workaround is to derive approximate sector polygons from OS CodePoint Open (unit postcode centroids, free download), compute a Voronoi tessellation, dissolve by sector code, and clip to the UK coastline. The result is approximate but adequate - the adjacency structure cares about who neighbours whom, not the exact boundary line.

```python
from insurance_spatial.adjacency import from_geojson

adj = from_geojson(
    "postcode_sectors.geojson",
    area_col="PC_SECTOR",
    connectivity="queen",
    fix_islands=True,
)
# UserWarning: Graph had 4 connected components.
# Connected 3 island nodes to nearest mainland node.

print(f"Areas in graph:  {adj.n}")
print(f"Components:      {adj.n_components()}")    # must be 1 for ICAR
print(f"Mean neighbours: {adj.neighbour_counts().mean():.1f}")
```

The `fix_islands=True` argument handles disconnected components and the queen contiguity graph structure are covered in detail in [the theory post](/2026/02/23/spatial-territory-ratemaking-with-bym2/). Document the island-connection policy in your model documentation — it is a deliberate choice that affects the graph structure.

The adjacency was built from all sectors in the shapefile; we filtered some out for low exposure. So we need to restrict the graph to the sectors in our exposure dataset, in the same order.

```python
from insurance_spatial import AdjacencyMatrix
from scipy import sparse

# area_index() returns dict: sector_code -> row/column index in W
adj_idx = adj.area_index()
keep_indices = [adj_idx[s] for s in sector_codes if s in adj_idx]

# Submatrix restricted to retained sectors
W_sub = adj.W[np.ix_(keep_indices, keep_indices)]
W_sub = sparse.csr_matrix(W_sub)

adj_filtered = AdjacencyMatrix(W=W_sub, areas=sector_codes)
print(f"Filtered graph: {adj_filtered.n} areas, {adj_filtered.n_components()} component(s)")
```

Cache the scaling factor. It depends only on the graph topology, not the claims data, so we only need to recompute it when the adjacency structure changes - typically once per year when the boundary file is refreshed.

```python
# Run once, persist the value (e.g. to a pickle or a config file)
sf = adj_filtered.scaling_factor
print(f"BYM2 scaling factor: {sf:.4f}")
# BYM2 scaling factor: 0.6213

# On subsequent runs, inject the cached value to skip the eigendecomposition
adj_filtered = AdjacencyMatrix(W=W_sub, areas=sector_codes, _scaling_factor=sf)
```

The eigendecomposition underlying the scaling factor is O(N^3) in time. For N=3,000 (postcode districts) it takes a few seconds. For N=11,000 (postcode sectors) it runs for several minutes. Caching avoids recomputing it every quarterly model refresh.

---

## Pre-fit diagnostic: Moran's I

Before spending 20 minutes on MCMC, run Moran's I to confirm that spatial autocorrelation is present in the residuals. If it is not, BYM2 will fit with `rho` near zero and produce factors close to simple credibility weighting. Not wrong, but not adding much beyond what a Bühlmann-Straub model would give.

```python
from insurance_spatial.diagnostics import moran_i

# log(O/E) per sector: the spatial residual
log_oe = np.log(observed / expected)
log_oe = np.where(np.isfinite(log_oe), log_oe, 0.0)  # handle zero-claim sectors

test = moran_i(log_oe, adj_filtered, n_permutations=999)
print(test.interpretation)
# "Significant positive spatial autocorrelation (I=0.312, p=0.001).
#  Nearby areas have similar values. Spatial smoothing is warranted."

print(f"Moran's I:  {test.statistic:.3f}")
print(f"Z-score:    {test.z_score:.2f}")
```

A positive, significant Moran's I (p < 0.05 from the permutation test) means nearby sectors have similar residuals. The spatial model is warranted.

The `moran_i()` function is a pure-numpy permutation test with no dependency on external spatial statistics libraries (esda, PySAL). It reshuffles the observed values across areas repeatedly and asks how often the reshuffled data produces an I as extreme as observed. This is more reliable than the analytical normal approximation for the skewed, heavy-tailed distributions typical of insurance O/E ratios.

If Moran's I is not significant, see [the theory post](/2026/02/23/spatial-territory-ratemaking-with-bym2/) for the graceful degradation argument — the short version is: do not abandon the model. Note the test result in the model documentation either way.

---

## Fitting the model

```python
from insurance_spatial import BYM2Model

model = BYM2Model(
    adjacency=adj_filtered,
    draws=1000,
    chains=4,
    target_accept=0.9,
    tune=1000,
)

result = model.fit(
    claims=observed,
    exposure=expected,    # base model expected claims, not raw car-years
    random_seed=42,
)
```

The `exposure` argument here is the expected claims from the base CatBoost model, not raw policy-years. Passing expected claims as the offset means BYM2 is fitting the residual O/E ratio - the geographic variation unexplained after vehicle, driver, and policy characteristics. This is the two-stage pipeline in action.

If nutpie is installed, the library detects it automatically and runs the Rust NUTS implementation. If not, it falls back to PyMC's default sampler with a warning. For N=3,000 postcode districts: 4-8 minutes with nutpie, 20-30 minutes without. For N=11,000 postcode sectors with nutpie on 8 cores: budget 20-30 minutes. This is a model we refit quarterly at most; the runtime is not a practical obstacle.

```bash
uv add "insurance-spatial[nutpie]"
```

---

## Convergence checks before using anything

MCMC without diagnostics is not production-ready. The `result.diagnostics()` call is not optional.

```python
diag = result.diagnostics()

# Convergence - hard thresholds from Vehtari et al. 2021
print(f"Max R-hat:     {diag.convergence.max_rhat:.4f}")      # want < 1.01
print(f"Min ESS bulk:  {diag.convergence.min_ess_bulk:.0f}")  # want > 400
print(f"Min ESS tail:  {diag.convergence.min_ess_tail:.0f}")  # want > 400
print(f"Divergences:   {diag.convergence.n_divergences}")     # want 0

# Spatial proportion posterior
print(diag.rho_summary)
# parameter | mean  | sd   | q025 | q975
# rho       | 0.731 | 0.11 | 0.51 | 0.91
```

R-hat above 1.01 on any parameter means the chains did not mix and the posterior is unreliable. Do not use the result. Re-run with more tuning steps (`tune=2000`) or a higher `target_accept` (0.95). Persistent divergences after that usually indicate prior-data conflict or a very sparse area that is pulling against the spatial prior - worth examining which sectors have the widest posteriors after the next successful run.

The `rho` posterior is the key scientific output from the fitting process. A posterior mean of 0.73 with 95% CI of [0.51, 0.91] means roughly 73% of the residual geographic variation is spatially smooth - neighbouring sectors genuinely tend to have similar risk profiles after controlling for vehicle and driver mix. The spatial model is earning its complexity.

A `rho` posterior mean below 0.2 suggests the base model has already absorbed most of the geographic signal, perhaps through IMD score, urban density, or crime rate. BYM2 will still produce valid territory factors, but they will resemble simple credibility-weighted sector loss ratios more than smooth spatial relativities. Consider adding area-level covariates (see below) to push more geographic signal into the fixed effects and leave less for the spatial component.

---

## Extracting and consuming the factors

```python
# Relativities normalised to geometric mean 1.0
rels = result.territory_relativities(credibility_interval=0.95)

print(rels.head(5))
# area   | b_mean  | b_sd  | relativity | lower | upper | ln_offset
# AB10 1 |  0.1234 | 0.041 |      1.131 | 1.049 | 1.219 |    0.1231
# AB10 2 |  0.0812 | 0.039 |      1.085 | 1.008 | 1.167 |    0.0816
# AB10 3 | -0.0441 | 0.045 |      0.957 | 0.875 | 1.047 |   -0.0443
# ...
```

The `relativity` column is `exp(b_i)` normalised to geometric mean 1.0. The `lower` and `upper` columns are the 95% credibility interval on the multiplicative scale - computed from the full posterior, not a delta-method approximation. The `ln_offset` column is `log(relativity)`, ready to use as an additive term in a log-linear predictor.

Join back to the policy data in Polars:

```python
df_with_territory = (
    df.join(
        rels.select(["area", "ln_offset"]).rename({"area": "postcode_sector"}),
        on="postcode_sector",
        how="left",
    )
    .with_columns(
        pl.col("ln_offset").fill_null(0.0)  # sectors not in fitted model get 1.0
    )
)
```

The `fill_null(0.0)` handles sectors not in the fitted graph - typically the sparse rural sectors we filtered out earlier. Assigning them `ln_offset = 0.0` (relativity = 1.0) is the conservative choice: no territory loading for sectors with insufficient data.

For Emblem: export `rels.select(["area", "ln_offset"])` as CSV and import as an offset variable. The factor is already on the log scale.

For Radar: load the same CSV as a rate factor table with exact match on sector code.

---

## What the credibility intervals actually tell you

The width of the posterior interval per sector is the quantified evidence for how much confidence we have in that sector's factor.

```python
# Sectors where the 95% CI spans more than 2x from lower to upper bound
wide_intervals = (
    rels
    .filter(pl.col("upper") / pl.col("lower") > 2.0)
    .sort("b_sd", descending=True)
)
print(f"Sectors with wide posteriors: {len(wide_intervals)}")
```

A sector with `relativity = 1.34, lower = 1.01, upper = 1.78` does not warrant the same rate action as one with `relativity = 1.34, lower = 1.26, upper = 1.42`. The posteriors tell us which is which.

This is also the Consumer Duty argument for the BYM2 approach. When the FCA asks whether territory factors are proportionate to the evidence, the credibility interval is the answer. A wide posterior is the quantitative case for conservatism: we should not apply a 34% territory loading to a sector where the data support anywhere between +1% and +78%.

---

## Including area-level covariates

IMD score, crime rate, flood risk percentile - these can be passed as fixed effects. The spatial component then captures residual geographic variation after controlling for them.

```python
from sklearn.preprocessing import StandardScaler

covariate_cols = ["imd_score", "vehicle_crime_rate", "flood_risk_pct"]
X_cov = sector_df.select(covariate_cols).to_numpy()
X_cov_scaled = StandardScaler().fit_transform(X_cov)

result = model.fit(
    claims=observed,
    exposure=expected,
    covariates=X_cov_scaled,  # (N, P) float array, scaled: prior on beta is Normal(0,1)
    random_seed=42,
)
```

The `beta` posterior from the fitted result tells us how strongly each covariate predicts the residual geographic variation. A large positive beta on IMD score means the base CatBoost model was not fully absorbing deprivation - the spatial model is compensating. That is a signal to revisit Stage 1 feature engineering rather than permanently baking it into the spatial offset.

---

## What is in the library

The library is four modules, none of them large.

`adjacency.py` handles graph construction - `build_grid_adjacency()` for synthetic grids, `from_geojson()` for real boundaries via geopandas and libpysal. `AdjacencyMatrix` is a dataclass carrying the scipy sparse weight matrix, the ordered area list, and the scaling factor (computed lazily via eigendecomposition of the ICAR precision matrix, cached after first access).

`models.py` is the BYM2 model itself: around 60 lines of PyMC around `pm.ICAR`, with nutpie auto-detection and PyMC NUTS fallback. The model specification is exactly what appears in the docstring - no hidden transformations or reparameterisations beyond what the BYM2 literature specifies.

`diagnostics.py` provides `moran_i()` - a pure-numpy permutation implementation with no external spatial statistics dependency - and `convergence_summary()` which wraps ArviZ's R-hat and ESS across all model parameters.

`relativities.py` pulls the `b` samples from the posterior, normalises to geometric mean zero, exponentiates to multiplicative factors, and returns a Polars DataFrame with the full posterior summary including the `ln_offset` column ready for rating engine import.

44 tests, all passing. The test fixtures use a 5x5 synthetic grid with a known north-south risk gradient - the top row of cells has a true log-rate 0.5 higher than the bottom - so we can verify the model recovers direction and magnitude qualitatively without requiring access to real postcode data.

---

## Installation

```bash
uv add insurance-spatial
uv add "insurance-spatial[geo]"      # geopandas + libpysal for real boundaries
uv add "insurance-spatial[nutpie]"   # Rust NUTS sampler, strongly recommended
```

Source at [github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial). The core pipeline - adjacency, BYM2, diagnostics, relativities - is stable at v0.1.0. The covariate enrichment module (automated IMD, Environment Agency, and police.uk joins keyed by postcode sector) is next on the roadmap.

---

## References

- Riebler, A., Sorbye, S.H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. *Statistical Methods in Medical Research*, 25(4), 1145-1165.
- Gschlohl, S., Czado, C., & Held, L. (2019). Spatial statistical modelling of insurance risk: a spatial epidemiological approach to car insurance. *Scandinavian Actuarial Journal*.
- Besag, J., York, J., & Mollie, A. (1991). Bayesian image restoration, with two applications in spatial statistics. *Annals of the Institute of Statistical Mathematics*, 43(1), 1-59.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Burkner, P.C. (2021). Rank-normalization, folding, and localization: an improved R-hat for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.

---

**Related articles from Burning Cost:**
- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/)
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [From CatBoost to Radar in 50 Lines of Python](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/)
