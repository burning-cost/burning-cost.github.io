---
layout: post
title: "What Actually Drives Flood Claims: Evidence from a Marginal Contribution Analysis"
date: 2026-03-26
categories: [pricing, flood, spatial]
tags: [flood, glm, spatial, rainfall, extreme-value, building-data, era5, catnat, france, severity, frequency, marginal-contribution, geographic-rating, open-data]
description: "Moriah et al. (2026) run a sequential model-building exercise on a French home insurance portfolio to measure what each data layer — hydrological zoning, rainfall intensity, building attributes — actually adds to flood claim prediction. The answer is more granular than the regulatory maps suggest."
---

Most UK pricing teams treating flood do one of two things. Either they buy a postcode-level flood score from a specialist supplier (JBA, Ambiental, Verisk) and treat it as a black box, or they fall back on the Environment Agency flood zone classification — Zone 1 (low), Zone 2 (medium), Zone 3a (high), 3b (functional floodplain) — and price accordingly. Both approaches conflate a lot that matters with a lot that does not.

Moriah, Vermet, Ailliot, Naveau, and Legrand (arXiv:2603.02418, March 2026) do something more useful: they take a French home insurance portfolio of 968,000 policyholders over 2014–2022 with 10,800 flood claims, and systematically measure what each additional layer of data contributes to model performance once standard underwriting information is already in the model. The paper is in French regulatory context but the methodology is directly applicable to UK pricing. The data sources are partially substitutable with UK equivalents.

---

## The sequential framework

The study uses GLMs throughout — deliberately, and correctly. When you want to measure the marginal value of data, a flexible machine learning model is the wrong tool. It will absorb everything, interpolate between signal and noise, and make attribution impossible. A logistic regression for claim occurrence and a log-linked Gamma GLM for claim cost are interpretable, stable across cross-validation folds, and honest about what they can and cannot explain.

The framework is a nested sequence:

1. **GLM ins** — standard underwriting variables: occupancy type, insured value, coverage options, geographic identifiers. Baseline.
2. **GLM ins+c** — adds expert hydrological and climatic variables: official flood risk maps (TRI and PPRI), river basin membership, climatic region, historical count of natural catastrophe decrees.
3. **GLM ins+r** — swaps the static climate variables for a dynamic rainfall intensity indicator (MILRE, defined below).
4. **GLM all** — adds fine-scale building and environmental attributes: footprint area, construction type, floor count, wall materials, slope, elevation, imperviousness, vegetation cover, distance to watercourses, building density in the immediate neighbourhood.

Performance is measured via 5-fold cross-validation on four metrics per model type: RMSE and Gini for cost, log-loss and Gini for occurrence. The cross-validation is honest — models are refitted in each fold, not just re-evaluated.

---

## The key innovation: MILRE

The Most Intense Local Relative Event indicator is the paper's main methodological contribution, and it is worth understanding precisely.

The authors use ERA5-Land reanalysis data — ECMWF's climate reanalysis product providing daily precipitation estimates on a ~9km grid globally, covering 1950 to present. For France this gives roughly 6,200 grid points. The data is free to download.

For each claim observation, MILRE is computed as follows. Take the four nearest ERA5-Land grid points. For each, compute rainfall accumulations over windows of {1, 3, 5, 7, 10, 30} days. For each window length _d_ and grid point, compute the empirical CDF rank of the observed accumulation within all historical accumulations at that point for that window:

```
F̂_nd(i*_nd) = (1/M_nd) Σ 1[i_ndk ≤ i*_nd]
```

where M_nd is the number of historical observations. MILRE is then:

```
MILRE = max_{n,d} F̂_nd(i*_nd)
```

The output is a number in [0, 1] — the extremeness of the most intense rainfall event at the location across all accumulation windows and nearby grid points, relative to local historical conditions. A value of 0.98 means the rainfall event was in the top 2% of what has historically been observed at that location.

This is better than raw millimetres for two reasons. First, it is locally standardised — 50mm in a wet Atlantic climate is less extreme than 50mm in a dry Mediterranean one, and MILRE captures this. Second, the multi-window maximum means it captures both flash flooding (short accumulation, high intensity) and groundwater saturation floods (long accumulation, lower daily intensity).

The annual variant ann_MILRE, used for frequency modelling, takes the annual maximum of MILRE across all days in the year. This acts as a spatio-temporal filter: years with high ann_MILRE at a location are high-risk years, years with low ann_MILRE are not.

---

## What each data layer actually contributes

The results tables are the most practically useful part of the paper. For cost modelling (severity):

| Model | RMSE change | Gini change | Deviance change |
|---|---|---|---|
| Baseline (underwriting only) | — | 0.056 | — |
| + hydrological/climate zoning | −1.0% | +162% | −4.5% |
| + rainfall intensity (MILRE) | −1.6% | +195% | −5.7% |
| + building + environment | −4.0% | +334% | −11.8% |

For claim occurrence (frequency):

| Model | Log-loss change | Gini change | CSI change |
|---|---|---|---|
| Baseline (underwriting only) | — | 0.234 | — |
| + hydrological/climate zoning | −5.8% | +61% | +59% |
| + rainfall intensity (MILRE) | −8.3% | +104% | +224% |
| + building + environment | −15.3% | +134% | +433% |

Some of these numbers warrant caution — Gini improvements of 100–400% sound extraordinary but are relative improvements on a very small absolute baseline (the baseline Gini for occurrence is 0.234). The RMSE improvements are more conservative and probably more representative of commercial impact. But the ordering is clear: building and environmental attributes dominate; the static regulatory maps help more for occurrence than for severity; the MILRE rainfall indicator is more informative than the static maps for both tasks.

The variable importance analysis (likelihood ratio tests) gives the clearest ranking. For severity, the dominant predictors are a composite indicator combining the TRI flood zone classification with watercourse proximity (WCTRII), followed by MILRE, followed by movable asset value and building size. For occurrence, building density within a 50m radius is the single strongest predictor, followed by amenity elements (outdoor facilities — a proxy for building type and vulnerability), followed by ann_MILRE and its interactions.

The building density finding is interesting. It is not obvious a priori that having more buildings within 50 metres should predict flood claim occurrence, but it makes sense on reflection: high building density correlates with impervious surface cover, which reduces infiltration and increases runoff; it also correlates with proximity to watercourses in older town layouts; and it may proxy for flood protection infrastructure that follows urban areas.

---

## The 95% problem

The single most commercially important finding in the paper is buried in the discussion section. Official TRI flood risk zones in France cover approximately 5% of the insured portfolio. That 5% generates 11% of flood events — so TRI does identify elevated risk. But 89% of flood events occur outside TRI zones entirely.

Within the non-TRI 95%, the baseline model plus climate zoning explains only 57% of total observed costs in the highest-risk sub-group. The full model (all variables) explains 97%. That is not a marginal improvement — it is the difference between pricing blindly and pricing with information.

UK pricing teams face an analogous problem. The EA flood zones cover a minority of addresses and are based on return period modelling that is now recognised to understate risk in many urban areas and to fail entirely for surface water flooding (the dominant flood mechanism by claim count, and one that correlates poorly with fluvial risk maps). Moriah et al. provide evidence that this is not a UK-specific regulatory failure — it is a structural limitation of official flood risk maps in general, and the gap can be substantially closed with ERA5-type rainfall indicators and building-level attributes.

---

## What this means for UK pricing

The UK has functional equivalents for most of the data sources in this paper.

**Rainfall data** — ERA5-Land is global and freely available from the Copernicus Climate Data Store. The MILRE construction requires daily precipitation grids and a historical archive. The ~9km ERA5-Land grid is coarser than ideal for urban surface water flooding but adequate for property-level aggregation. The Met Office also publishes HadUK-Grid at 1km resolution for the UK, which would improve spatial specificity.

**Flood zone data** — EA Flood Zone 2 and 3 (England), NRW flood maps (Wales), SEPA flood maps (Scotland) are the UK analogues of TRI/PPRI. All are freely available as GIS layers. The EA also publishes surface water flood risk maps separately, which have no direct French equivalent in this paper and would be an additional signal.

**Building attributes** — Ordnance Survey AddressBase Premium provides footprint geometry, construction year, and property type at building level. UPRN-linked insurance portfolios can join these directly. EPC certificates from MHCLG provide wall and roof construction type for the ~40% of properties with certificates. Neither is as comprehensive as the French cadastral data used in this paper, but both are materially useful.

**Environmental attributes** — CEH's Land Cover Map provides imperviousness and vegetation cover at 10m resolution. OS Terrain 50 gives slope and elevation. The Environment Agency publishes watercourse network layers.

The `insurance-spatial` and [`insurance-whittaker`](/insurance-whittaker/) repositories in the Burning Cost stack are relevant here. Geographic smoothing of the kind that Whittaker-Henderson implements naturally over rating factor grids applies directly to a postcode- or LSOA-level flood rating factor derived from ERA5 or EA zone data — for the methodology behind why standard GLM smoothing falls short for this, see [your rating table smoothing is wrong](/2026/03/18/your-rating-table-smoothing-is-wrong/) — you want a smooth, credible surface rather than a jagged empirical one. The conformal prediction work in `insurance-conformal` applies to the prediction intervals on flood severity estimates, particularly in the non-TRI/non-zone areas where there is genuine model uncertainty.

---

## Limitations of this paper

The data is French. The CatNat scheme in France operates differently from the UK market: CatNat is a government-backed reinsurance pool, so the claims data includes events that would be handled differently under UK commercial policies. The coverage structures differ. The French portfolio studied has homogeneous coverage type by construction (the authors filtered for this) — a UK portfolio would have more heterogeneous excess structures and flood exclusions that would need careful handling.

The rainfall variables are calibrated on French climate patterns. The MILRE logic transfers — it is dimensionless and locally normalised — but the specific ERA5 grid points, the accumulation window selection, and the tail behaviour will differ for UK climate regimes (more Atlantic influence, different orographic rainfall patterns, more significant coastal flood interaction in England than in interior France).

The paper uses GLMs throughout. This is a methodological choice justified by the attribution goal, but it means the absolute predictive performance is not a ceiling. A gradient boosted model with the same features would almost certainly improve on the reported Gini indices. The marginal contributions measured here are a lower bound on what these data sources can contribute with more flexible models.

The 5-fold cross-validation uses temporal splitting (2014–2022), but the paper does not report explicit out-of-time validation. Flood events are correlated in time by CAT year — a bad flood year will appear in multiple folds even with standard random splitting. The reported metrics may be optimistic by a small amount.

None of these limitations undermine the main findings. The ordering of data source contributions — static zoning < rainfall intensity < building attributes — is robust to methodology choice, and the 95% out-of-zone problem is structural rather than model-specific.

---

## What to take away

Three things are clear from this paper.

First, official flood maps are necessary but not sufficient for flood pricing. They identify the highest-risk concentrations but leave the majority of the risk unpriced. Rainfall intensity indicators derived from freely available reanalysis data fill a significant portion of that gap.

Second, building-level attributes matter more than climate variables. The paper's variable importance results are consistent: fine-scale physical characteristics of the building and its immediate environment dominate both frequency and severity models once rainfall intensity is already included. For UK pricing teams, this points toward using OS AddressBase footprints and EPC data as underwriting variables rather than relying solely on postcode-level flood scores.

Third, the MILRE framework is implementable. ERA5-Land is freely available, the CDF-transformation logic is straightforward, and the multi-window maximum is a small amount of data engineering. A UK insurer could construct this indicator for its own portfolio using open data within a few weeks of engineering time.

The paper is at [arXiv:2603.02418](https://arxiv.org/abs/2603.02418).
