---
layout: post
title: "Beyond Flood Zone 3: Building-Level Property Risk with Python"
date: 2026-03-31
categories: [pricing, flood, spatial]
tags: [flood, spatial, python, geopandas, rasterio, ea-data, flood-re, open-data, building-level, uk, nafra, pysal, property]
description: "EA NaFRA is open, 2m resolution, and free. So is the EPC register. So are OS building footprints. A UK pricing actuary who has actually tried to use them explains what you can and cannot do — and where Flood Re makes the whole exercise moot for another 13 years."
---

Two properties on the same street, both in the same postcode, both classified as EA Flood Zone 3. One sits on a raised terrace two metres above the nearest watercourse. The other has a basement flat with floor level below the 1-in-100-year flood depth. Under postcode-level flood pricing, they pay identical flood premiums. Under a building-level model, they should be separated by a factor of ten or more.

This gap is the pricing problem. It is not a subtle actuarial refinement — it is a material mispricing that accumulates at the tail, exactly where flood losses concentrate. Moriah et al. ([arXiv:2603.02418](https://arxiv.org/abs/2603.02418)) recently confirmed on 968,000 French home insurance policies that official flood zone classification alone leaves the overwhelming majority of flood risk unpriced at property level. [We covered that paper's methodology in detail earlier this week.](/2026/03/26/geolocated-weather-building-data-flood-insurance/) Here we are asking a different question: what can a UK pricing team actually do about it, using open data and Python, right now?

The honest answer has three parts. Open data gets you surprisingly far for research and internal benchmarking. Commercial data is required for production. And Flood Re makes building-level flood pricing largely irrelevant for the segment of the market that needs it most — until 2039.

---

## Why postcode-level flood risk is structurally crude

A UK postcode unit contains roughly 15 addresses. A postcode sector (the first five characters — SW1A 1) contains somewhere between 2,000 and 8,000 addresses. Most flood pricing operates at postcode sector level when it uses internal loss data, and at postcode unit level when it uses a third-party score lookup.

The EA Flood Zone classification — Zone 1 (low risk, <0.1%/yr), Zone 2 (medium, 0.1-1%/yr), Zone 3a (high, >1%/yr), Zone 3b (functional floodplain) — is a polygon layer. Assigning a zone to a property means point-in-polygon: find which zone polygon contains the property's centroid. But the zone boundary is a hydrological model output at roughly 50-100m resolution. Two adjacent properties near a zone boundary can be in different zones by a matter of metres of model uncertainty, and two properties in the same zone cell can have radically different ground elevations, basement configurations, and proximity to the actual channel.

The January 2025 NaFRA overhaul improved this. The new Risk of Flooding from Rivers and Sea (RoFRS) and Risk of Flooding from Surface Water (RoFSW) datasets include flood depth bands — for surface water, this means bands for >20cm, >30cm, >60cm, >90cm, >120cm inundation depth — which is materially more useful for insurance than the old zone categories. EA now identifies 6.3 million properties at flood risk in England, projecting to 8 million by mid-century. The 2025 data is free, on the Open Government Licence, and downloadable from the Defra Data Services Platform.

But even with depth bands, the fundamental limitation holds: this is grid-cell risk, not building risk. The 2m resolution surface water grid tells you the model's flood depth at a 2m × 2m square. It says nothing about whether the building at that location has a raised threshold, a semi-basement, a sump pump, or solid wall construction that resists flood ingress. It cannot tell you whether the nearest drainage channel was upgraded in 2022. These features matter enormously for claim probability, and none of them are in EA NaFRA.

---

## What the open data landscape actually looks like

The table below shows what is usable without a commercial data licence in England (NaFRA is England-only; Scotland and Wales have SEPA/NRW equivalents):

| Dataset | Resolution | Coverage | What it tells you |
|---|---|---|---|
| EA NaFRA RoFRS/RoFSW (2025) | ~2-50m grid | England | Flood zone category + depth bands |
| EA Flood Zones 1/2/3 | Polygon (~50-100m) | England | Planning zone |
| OS OpenMap Local buildings | Building polygon | GB | Footprint geometry |
| EPC register | Property address (UPRN) | England & Wales | Construction age, wall type, floor area |
| HadUK-Grid (Met Office) | 1km grid, daily 1836-present | UK | Historical rainfall |
| OS Terrain 50 | 50m DEM | GB | Elevation and slope |
| Copernicus GLO-30 | 30m DEM | Global | Elevation (better than SRTM for UK) |

What you cannot get for free: flood depth at actual building footprint level, flood defence status, property-level vulnerability scores, or any of the proprietary building attribute datasets (Verisk UKBuildings, OS NGD Enhanced Buildings with Verisk integration).

The BGS Groundwater Flooding Susceptibility dataset used to be freely accessible. From 1 April 2024, BGS introduced direct licence fees. If you need geological context — particularly relevant for groundwater flooding, which accounts for approximately £530M/yr in UK damages and is excluded from many policies — budget for it.

For commercial models: JBA operates at 5m resolution for hazard data and produces property-level flood scores as AAL percentages of TSI; Moody's RMS HD model runs at up to 1m resolution including defence status; Fathom UK (University of Bristol spin-out) operates at 10m with a clean API. All three are annual licence products.

---

## A practical Python pipeline with EA open data

Here is what building-level flood feature engineering actually looks like in Python using the open stack. This gets you meaningful features; it does not replace a commercial model.

**Step 0: coordinate system.** All EA and OS data is in OSGB36 (EPSG:27700). If your portfolio coordinates are in WGS84 (EPSG:4326), convert first. Mixing coordinate systems is the most common source of silent errors in spatial joins.

```python
import geopandas as gpd
from pyproj import Transformer

# Portfolio: DataFrame with 'lat', 'lon' columns
transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
portfolio["easting"], portfolio["northing"] = transformer.transform(
    portfolio["lon"].values, portfolio["lat"].values
)
gdf_portfolio = gpd.GeoDataFrame(
    portfolio,
    geometry=gpd.points_from_xy(portfolio["easting"], portfolio["northing"]),
    crs="EPSG:27700"
)
```

**Step 1: join EA flood zones.** Download the EA Flood Map for Planning polygons from the Defra Data Services Platform. The RoFRS layer is a ~700MB shapefile. Read it once, index it, and run a spatial join:

```python
# ea_rofrs = path to downloaded EA RoFRS shapefile
flood_zones = gpd.read_file(ea_rofrs)  # already in EPSG:27700
assert flood_zones.crs.to_epsg() == 27700

joined = gpd.sjoin(
    gdf_portfolio,
    flood_zones[["geometry", "Risk_band"]],
    how="left",
    predicate="within"
)
# Unmatched properties are outside any risk band — treat as 'Very Low'
joined["flood_zone"] = joined["Risk_band"].fillna("Very Low")
```

**Step 2: sample EA flood depth rasters.** The NaFRA 2025 surface water depth bands are distributed as GeoTIFFs. Use `rasterstats` to extract the flood depth at each property centroid:

```python
import rasterstats

# ea_depth_geotiff = path to downloaded depth band raster (e.g. depth > 30cm)
stats = rasterstats.point_query(
    gdf_portfolio.geometry,
    ea_depth_geotiff,
    interpolate="nearest"
)
joined["flood_depth_30cm"] = stats  # 1 = flooded at this depth, 0 or nodata = not
```

You can do this for each depth band (>20cm through >120cm) to build a depth profile per property.

**Step 3: distance to nearest watercourse.** The EA publishes the Watercourses layer as part of the Rivers and Sea dataset. Distance to the nearest channel is one of the strongest flood risk predictors in the Moriah et al. analysis:

```python
from scipy.spatial import cKDTree
import numpy as np

watercourses = gpd.read_file(ea_watercourses_path)

# Densify watercourse geometries to point clouds for distance approximation
wc_coords = np.array([
    [pt.x, pt.y]
    for geom in watercourses.geometry
    for pt in geom.coords
])
tree = cKDTree(wc_coords)

property_coords = np.column_stack([
    gdf_portfolio.geometry.x,
    gdf_portfolio.geometry.y
])
dist, _ = tree.query(property_coords, k=1)
joined["dist_to_watercourse_m"] = dist
```

**Step 4: terrain elevation and relative height.** Download OS Terrain 50 tiles (free from OS OpenData). Sample elevation at the property centroid. The useful feature is not absolute elevation but elevation relative to the nearest watercourse channel — a property 5m above the adjacent river is very different from a property 0.5m above it:

```python
import rasterio

with rasterio.open(os_terrain_50_path) as dem:
    coords = [(geom.x, geom.y) for geom in gdf_portfolio.geometry]
    elevations = [v[0] for v in dem.sample(coords)]

joined["elevation_m"] = elevations
```

Computing relative elevation requires knowing the channel bed elevation too, which requires either the OS Terrain product or EA channel survey data. For a first pass, using absolute elevation as a feature is still informative.

**Step 5: EPC building attributes.** The EPC register is downloadable in bulk from epc.opendatacommunities.org — roughly 30 million certificates for England and Wales, updated monthly, licensed under OGL. Match to your portfolio on UPRN (if you have it) or on address:

```python
import pandas as pd

epc = pd.read_csv(epc_bulk_path, usecols=[
    "uprn", "construction_age_band", "wall_description",
    "floor_description", "total_floor_area"
])
joined = joined.merge(epc, on="uprn", how="left")
```

Construction age matters for flood vulnerability — pre-1919 solid wall construction behaves very differently from post-1980 cavity wall construction when subject to inundation. Floor area is a useful severity predictor. About 30% of the UK housing stock lacks an EPC certificate, so expect significant missing data.

**Step 6: check for spatial autocorrelation in your residuals.** After fitting a GLM on these features, run Moran's I on the residuals using PySAL. If it is significantly positive, you have residual geographic structure that your features have not captured:

```python
from libpysal.weights import KNN
from esda.moran import Moran

w = KNN.from_dataframe(joined, k=8)
w.transform = "r"
moran = Moran(joined["glm_residual"], w)
print(f"Moran's I: {moran.I:.3f}, p-value: {moran.p_sim:.3f}")
```

Significant residual autocorrelation (typical for flood data) suggests you need a spatial random effect — BYM2 at postcode sector level via `insurance-spatial` is the natural next step.

---

## Moriah et al. confirms what to prioritise

The French study we covered separately used a sequential ablation design: start with standard underwriting variables, add official flood zone, add rainfall intensity, add building attributes. The ordering of marginal contribution is the actionable finding.

For a UK replication, the priority order for feature engineering is:

1. EA NaFRA flood depth bands (primary hazard signal — the 2025 update is substantially better than the old zone categories)
2. Distance to nearest watercourse
3. Elevation relative to local water level
4. EPC construction age (affects vulnerability — older buildings flood worse)
5. EPC wall and floor type (solid wall holds water longer)
6. HadUK-Grid historical rainfall intensity at 1km (the UK equivalent of ERA5-Land — finer resolution than ERA5 for the UK)
7. BYM2 spatial random effect on residuals

The important caveat from Moriah et al. is that severity is more sensitive to rainfall indicators, while frequency is more sensitive to building attributes. If you are trying to price expected annual loss, both matter. If you are trying to identify properties to decline or rate up, the building attribute layer moves the needle more.

---

## Flood Re and why this mostly does not matter until 2039

Building a carefully engineered building-level flood model for your UK home insurance portfolio makes a lot of actuarial sense. It makes almost no commercial sense for most of the properties that most need it.

Flood Re is the reason. Established in 2016 under the Water Act 2014, Flood Re operates as a reinsurance pool: UK personal lines home insurers cede the flood element of eligible policies to the pool at a fixed premium set by Council Tax band rather than actual flood risk. The 2024 fixed premiums run from £210/yr (Band A) to £1,200/yr (Band H) (source: Flood Re published premium schedule). An insurer with a Band D property in EA Flood Zone 3 that carries a modelled expected flood cost of £800/yr pays £320/yr to Flood Re — and cedes the flood risk entirely. The incentive to price that property at building level is eliminated by the cession mechanics.

For the ~250,000 policies currently ceded to Flood Re, granular flood modelling affects only the decision to accept or decline the risk before cession. Once the policy is written and ceded, the insurer is indifferent to whether the property's modelled flood cost is £300/yr or £3,000/yr — they pay the same Council Tax band premium either way.

There are eligibility exclusions that matter for pricing teams. New builds after 1 January 2009 are excluded — market pricing from day one. Purpose-built flats are excluded. Commercial properties are excluded. High-value policies above certain sum insured thresholds are excluded. For these segments, building-level flood pricing is commercially live now.

Flood Re's Transition Plan (published 2024) requires risk-reflective pricing by 2039. The explicit goal is that Council Tax band pricing ends and insurers price on actual flood risk. Three enabling conditions: flood defence investment, property-level flood resilience improvements, and better modelling. On the current trajectory, Flood Re itself acknowledges the 2039 deadline requires faster risk reduction to remain achievable.

The implication for pricing teams is a 13-year runway. Building-level flood models built and validated over the next five years will be the foundation for competitive underwriting when Flood Re eligibility expires. The 2024 data is already stark: UK flood claims hit £650M in 2024, a record, and average flood claim payouts reached £30,000 in Q1 2025, up 60% year-on-year (source: ABI weather-related claims data). Waiting until 2038 to develop the modelling capability is not a viable strategy.

---

## The honest gap: methodology vs data

The Python pipeline above produces a genuinely useful research-grade flood model. It will outperform a raw EA flood zone lookup. With EPC features and spatial blocking, it will produce calibrated building-level claim frequency predictions that identify differentiation within postcode sectors.

But it is not a production underwriting tool. The gap is data, not methodology.

EA NaFRA at 2m resolution does not model flood defence effectiveness — it models the landscape without defences in place. A property protected by a coastal barrier rated to 1-in-200 years looks identical in the open data to a property with no defence. JBA and RMS include defence status explicitly. This is a meaningful pricing difference.

OS OpenMap Local building footprints are relatively coarse. OS NGD Enhanced Buildings — with Verisk's UKBuildings dataset integrated, providing construction type, roof type, and building age for every property in GB — requires a commercial OS licence. The EPC register covers roughly 70% of the housing stock but with significant selection bias toward recently built and recently transacted properties.

Groundwater flooding — approximately 20% of UK flood claims by count — has no adequate open-data representation after the BGS licence change. The BGS 50m groundwater susceptibility dataset now requires a fee. For urban areas where groundwater flooding is the primary mechanism (think low-lying Thames Valley towns), this is a real modelling gap.

The commercial models (JBA at 5m, RMS at 1m, Fathom at 10m) solve all of these problems. They incorporate defence surveys, building-level vulnerability functions, and sub-metre terrain models that simply do not exist in open form. A pricing model built on open data and benchmarked against JBA flood scores will typically show R² values 20-30% lower than the commercial model — a gap that matters at the tail, where flood pricing decisions are made.

Our `insurance-spatial` library ([GitHub](https://github.com/burning-cost/insurance-spatial)) implements the BYM2 spatial smoothing layer that sits on top of either open-data or commercial flood features. The geospatial feature engineering pipeline described here — EA NaFRA join, raster sampling, watercourse distance, EPC merge — is a natural precursor to the spatial random effect that `insurance-spatial` handles. Open-source tools help with methodology, spatial statistics, and validation. They do not replace commercial hazard data for production pricing.

---

## What to actually do

For a pricing team starting from scratch on building-level flood:

Start with the EA NaFRA 2025 data — the 2025 update is meaningfully better than what was available before January, and it is free. Spatial join your portfolio to both the RoFRS and RoFSW layers. Build the depth band features described above. Add distance to watercourse and EPC construction age where available.

Benchmark the result against your existing commercial flood score (JBA, Fathom, or whatever you currently use). The open-data model will identify a substantial proportion of the discrimination that the commercial model finds, and the gaps it cannot close will tell you exactly what you are paying for in the commercial licence.

Use spatial blocked cross-validation throughout. Standard k-fold CV on geographic data is optimistic because nearby observations are correlated. Split your test folds by geographic blocks — k-means on coordinates is the simplest approach — and ensure test properties are spatially distant from training properties. A Gini improvement that survives spatial CV is real signal; one that does not is spatial leakage.

For the Flood Re-eligible segment of your book: build the model now as a validation and selection tool, even though pricing is bounded by Council Tax band. When 2039 arrives, you want a model with five years of live validation, not a model built from scratch under competitive pressure.

The open data is there. The Python stack works. The methodology, courtesy of Moriah et al. among others, is well-established. The question is whether the 2039 deadline feels real enough to justify the investment today. We think it does.
