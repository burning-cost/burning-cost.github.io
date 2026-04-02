---
layout: post
title: "Building-Level Flood Pricing: What Flood Re's 2039 Phase-Out Means for Your Models"
date: 2026-04-02
categories: [techniques, spatial, flood, regulation]
tags: [flood, flood-re, precipitation, bym2, spatial, glm, frequency-severity, haduk-grid, ea-flood-map, naFRA2, milre, ann-milre, insurance-spatial, uk-property, building-level, moriah-2026, arXiv-2603-02418]
description: "Flood Re ends in 2039. From that date, 350,000+ currently subsidised properties need risk-reflective pricing. We work through Moriah et al. 2026's sequential GLM using geolocated precipitation metrics, where UK data sources (EA NaFRA2, HadUK-Grid, OS Open Rivers) fit in, and what a pricing team would actually need to build this."
math: true
author: burning-cost
---

Flood Re ends in 2039. That is thirteen years away, which is long enough that most pricing teams have not yet treated it as urgent. It should not stay that way.

The scheme currently reinsures flood risk for roughly 350,000 UK properties — those at high flood risk, built before 1 January 2009, and ceded by their insurer to Flood Re Ltd in exchange for a fixed premium. From 2039, that backstop disappears. Insurers must price flood risk on their own books. The properties at greatest risk — many in Yorkshire, Somerset, parts of Wales and the Thames floodplain — will face market pricing for the first time.

The question is whether the market will have the tools to price them correctly, or whether the answer defaults to crude zone-based loadings that either overcharge low-risk properties inside a flood zone or miss high-risk properties outside one. A paper published on arXiv in March 2026 — Moriah, Vermet, Ailliot, Naveau, and Legrand (arXiv:2603.02418) — gives the clearest picture yet of how to do this properly. It is French, it uses ERA5-Land rather than HadUK-Grid, and the regulatory context differs. None of that makes it less useful.

---

## The problem with flood zones as rating factors

The EA Flood Map for Planning divides England into three zones. Flood Zone 3 means greater than 1% annual probability of flooding; Zone 2 is 0.1–1%; Zone 1 is below 0.1%. Following the NaFRA2 release in January 2025, the maps also carry surface water risk and sub-100m resolution depth estimates. They are the best open-access flood hazard data England has ever had.

They are still not adequate as a building-level pricing tool.

The EA's own guidance states the maps are "not suitable for showing whether an individual property is at risk." That is not false modesty — it is an accurate statement about a dataset designed for planning decisions, not insurance rating. Zone boundaries are polygon-level; two houses 50 metres apart on either side of a zone boundary may have more similar flood risk than two houses both labelled Zone 3 but with very different drainage characteristics, elevation relative to the nearest watercourse, and construction age.

More importantly: the majority of UK flood claims come from properties outside Flood Zones 2 and 3. Surface water flooding — overwhelmed drains, flash run-off from impervious surfaces, groundwater emergence — is not well captured by fluvial flood zone mapping. A property with no JBA or Fathom flood risk flag can flood. This is not a modelling edge case; it is routine claims experience.

The Moriah et al. finding that is directly relevant here: in a French home insurance portfolio of 968,000 policyholders over 2014–2022, with 10,800 flood claims at a 0.22% positive rate, the marginal lift from geolocated precipitation metrics was largest for properties outside the French equivalent of flood zones (TRI zones). That 95% of the portfolio outside statutory flood risk areas is exactly where current UK pricing is blind.

---

## What Moriah et al. 2026 actually does

The paper builds a sequential set of GLMs — four nested models — for both flood claim occurrence (Bernoulli/logit) and severity (Gamma/log). It measures the incremental predictive contribution of each data layer.

**Layer 1**: Insurance data only. Policy structure, sums insured, property attributes from underwriting. Gini 0.234 for occurrence, baseline for severity. This is what most pricing models currently use.

**Layer 2**: Static climate and hazard variables. TRI flood zone, proximity to watercourses, CatNat history, climatic region. Gini 0.377 (+61% improvement). This is roughly what a team using JBA or EA flood zones achieves.

**Layer 3**: Geolocated precipitation metrics. This is where the paper's main innovation sits — Gini 0.477 for occurrence, a further +26% over the static hazard layer.

**Layer 4**: Building and environmental attributes (structure type, age, ground floor area). Gini 0.546 for occurrence; +4% RMSE improvement in severity.

The precipitation metrics are constructed from ERA5-Land reanalysis data at ~0.1° resolution (roughly 10km cells). For each property location, the paper computes:

**MILRE** (Most Intense Local Relative Event): for each flood claim with a known date, it accumulates rainfall over windows of 1, 3, 5, 7, 10, and 30 days prior, extracts the maximum from the four nearest grid points, and transforms to an empirical CDF over the 1950–2024 historical record. MILRE = the maximum across all windows of that CDF value. A MILRE of 1.0 means the event was the most intense on record for that location. Crucially, it normalises for local climate — a 40mm event in Aberdeen and a 40mm event in Essex are treated differently because they represent different return periods locally.

**ann_MILRE**: the same construction applied annually — the maximum empirical CDF value of annual maximum precipitation across all windows. Used for the occurrence model where claim date is unknown.

Both metrics interact with other features in predictable ways: MILRE with movable assets (severity model); ann_MILRE with historical CatNat declaration count (occurrence model, because high precipitation in repeatedly flood-designated areas amplifies risk multiplicatively).

The modelling framework handles class imbalance — 0.22% positive rate — through observation weighting: $w_0 = n/(2 \times n_0)$, $w_1 = n/(2 \times n_1)$. This is straightforward to implement in scikit-learn or statsmodels. The GLM is evaluated via 5-fold cross-validation; the Gini and Critical Success Index figures quoted above are out-of-sample.

---

## UK data substitutions

The paper uses ERA5-Land (European Centre for Medium-Range Weather Forecasts reanalysis, 0.1° grid) and French regulatory data. The UK equivalents are:

**HadUK-Grid** (Met Office, via CEDA Archive): daily gridded precipitation at 1km resolution, from the 1960s through 2024 (v1.3.1.ceda). This is the correct UK substitute for ERA5-Land in MILRE computation — higher spatial resolution (1km vs 10km) and a UK-native product. Commercial insurers need a CEDA licence; the application is straightforward and costs well under the price of a JBA licence. Variables include daily precipitation, temperature, and sunshine hours. Format is NetCDF, accessible via xarray.

**EA Flood Map for Planning + NaFRA2**: the UK equivalent of France's TRI flood zones. NaFRA2 (January 2025) is the first nationally consistent probabilistic flood depth dataset at sub-100m resolution — the most significant advance in English flood data in a decade. Available via the DEFRA Data Services Platform under OGL v3. The WFS API at `environment.data.gov.uk` supports programmatic access. The same caveat applies here as with the EA guidance quoted above: the data is polygon-resolution, not building-level depth.

**OS Open Rivers**: polyline watercourse network, free download from OS Data Hub. Sufficient for computing distance-to-watercourse via a KDTree spatial join. For elevation relative to watercourse (the WCTRII-style composite variable from the paper), you need a DEM: OS Terrain 50 is open, Terrain 5 (2m resolution, better for buildings) requires a licence.

**EPC Register**: 25M+ certificates for England and Wales, free API at `epc.opendatacommunities.org`. Provides construction year, wall type, floor area, property type — linked to UPRN. Construction year is a direct proxy for flood damage potential: pre-1900 and pre-1960s solid-wall properties have significantly higher water damage claims in Norwegian and French data, and UK claims experience is consistent.

**France's PPRI (municipal flood prevention plans)** has no UK equivalent. France has a documented intermediate layer between national flood zones and building-level data. The UK equivalent — Local Flood Risk Management Strategies — is patchy and non-machine-readable. This is a genuine data gap; UK EA data jumps from national probability zones to building-specific modelling with nothing usable in between.

---

## How insurance-spatial fits

Our [insurance-spatial](https://github.com/burning-cost/insurance-spatial) library implements BYM2 spatial random effects via PyMC with adjacency matrices from libpysal. The BYM2 model accepts area-level covariates — log-rate fixed effects in the Poisson model — as a NumPy array of shape `(N, P)`.

The Moriah et al. pipeline maps cleanly onto a two-stage architecture:

**Stage 1 (building-level)**: A frequency-severity GLM with building-level features — MILRE, ann_MILRE, EA flood zone classification, distance to watercourse, elevation relative to watercourse, construction year, EPC floor type, sums insured. This is where the Moriah et al. innovation lives. The output is fitted values and area-level O/E ratios.

**Stage 2 (area-level)**: Pass those O/E ratios into BYM2 with area-level covariates (mean ann_MILRE for the LSOA, flood zone coverage %, etc.) to handle residual spatial correlation not explained by the building-level features. BYM2's spatial smoothing is exactly what you want here: it borrows strength across neighbouring areas while the rho parameter tells you how much of the residual variation is genuinely spatial versus idiosyncratic.

The gap is Stage 1: there is currently no open-source Python implementation of the MILRE/ann_MILRE pipeline with HadUK-Grid and EA data hooks. The modules needed are:

- `FloodFeatureExtractor`: property-level flood zone, distance to watercourse, elevation via OS Terrain 50 and EA WFS — approximately 250 lines of Python using rasterio, requests, and scipy.
- `PrecipitationRiskIndex`: MILRE and ann_MILRE from HadUK-Grid via xarray — approximately 200 lines, with the multi-window ECDF transformation as the core calculation.
- `SpatialBlockedCV`: cross-validation that blocks on geography to prevent autocorrelation leakage. This is non-negotiable for any spatial model — standard k-fold leaks because neighbouring properties end up in train and test sets.

These belong in a planned `insurance-geo` library. We will build them. The architecture decision is documented.

---

## Honest limitations

The Moriah et al. paper is French, calibrated on a French portfolio, and uses French regulatory data. Before drawing quantitative conclusions for a UK portfolio:

**The event type mix is different.** France's flood claims are dominated by fluvial and pluvial events, with the CatNat regime defining a claims-triggering threshold. UK domestic water damage claims include more escape-of-water, blocked drain, and groundwater events — not all of which are precipitation-driven in the same way. The MILRE metric will perform differently against a UK claims dataset; the Gini improvement figures from the paper should not be assumed to transfer.

**ERA5-Land vs HadUK-Grid.** The paper uses ERA5-Land at 0.1° (10km). HadUK-Grid at 1km is considerably higher resolution, which should help in mountainous and coastal areas where precipitation gradients are sharp. But the finer resolution means more data management complexity; a 1km grid over the UK at daily resolution from 1960 to 2024 is a large NetCDF file.

**Building-level depth data remains commercially locked.** JBA and Fathom's 5m flood maps are the products UK insurers actually use for high-resolution flood scoring. They are licensed and expensive. The EA's open NaFRA2 data is a genuine improvement but is not a substitute for commercial products for Zone 3 properties where building-specific depth matters.

**The WCTRII composite variable is powerful but requires clean data.** The paper's WCTRII variable — which partitions TRI zone classification by watercourse proximity and relative elevation into nine exposure categories — is the single strongest predictor in both occurrence and severity models. Replicating it for the UK requires a reliable DEM, clean watercourse geometry, and per-property location matching. UPRN accuracy matters here: a 20m geocoding error on a riverside property can flip its relative elevation category.

---

## What a pricing team should do now

The 2039 deadline is far enough away that the risk of underreaction is higher than overreaction. But the NaFRA2 data release in January 2025 and the Flood Pricing Certificate pilot expected in 2026 signal that the regulatory and data infrastructure is arriving ahead of the deadline.

Three things are worth doing now:

**Get the data pipeline running.** Register for CEDA Archive access (HadUK-Grid) and download a test set. Write the MILRE computation against a sample of claims with known event dates. Measure how much lift it adds over EA flood zone alone. If the answer is "a lot", that is useful to know years before 2039.

**Identify your Flood Re book.** Pull the properties currently ceded to Flood Re. These are, by definition, your highest-risk flood properties. They are also the ones that will require the most sophisticated pricing in 2039. Building a model validation dataset around them now, while their claims experience accumulates, is straightforward and cheap.

**Instrument for MILRE computation prospectively.** The ann_MILRE metric requires knowing which year each policy was in-force. The event-specific MILRE requires claim date. Most property loss databases already carry these. What they often lack is a reliable property latitude/longitude — they have UPRN or postcode only. The spatial join to HadUK-Grid grid points requires coordinates. Adding UPRN geocoding to the loss data pipeline now costs very little. Missing it when the model needs calibrating in 2034 will cost more.

The actuarial problem is not conceptually hard. The frequency-severity GLM framework is standard. The data is, largely, available and free. What does not yet exist is a clean open-source implementation that connects the UK property data ecosystem to the Moriah et al. methodology. That is the gap insurance-geo is designed to fill.

---

*arXiv:2603.02418 (Moriah, Vermet, Ailliot, Naveau, Legrand, March 2026). EA Flood Map for Planning: DEFRA Data Services Platform, OGL v3 March 2025 update. HadUK-Grid v1.3.1.ceda, Met Office / CEDA Archive. OS Open Rivers: OS Data Hub, OGL.*
