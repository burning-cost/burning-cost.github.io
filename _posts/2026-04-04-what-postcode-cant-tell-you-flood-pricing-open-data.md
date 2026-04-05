---
layout: post
title: "What Postcode Can't Tell You: Flood Pricing Without a JBA Budget"
date: 2026-04-04
categories: [pricing, flood, spatial]
tags: [flood, flood-re, open-data, precipitation, milre, haduk-grid, ea-flood-map, nafra2, building-level, postcode, mga, uk-property, moriah-2026, arXiv-2603-02418, epc-register, os-open-rivers]
description: "You don't need a JBA licence to build a materially better flood model. A French study on 968,000 policies shows which open data sources actually move the needle — and the answer isn't the flood zone map you already have."
---

JBA Flood costs around £30,000 a year for a mid-sized insurer and more if you want the full 5m resolution stack with defence status. For a growing MGA writing a few thousand home policies, that is not a pricing data budget — it is a meaningful fraction of the technology budget. The standard response is to fall back on the EA flood zone classification, add a Zone 3 loading, and call it done.

We think this is leaving real predictive lift on the table, and a paper published in March 2026 makes that case with numbers. Moriah, Vermet, Ailliot, Naveau, and Legrand (arXiv:2603.02418) run a sequential GLM attribution exercise on a French home portfolio of 968,000 policyholders over 2014–2022 with 10,800 flood claims. They measure exactly what each additional data layer contributes once you already have standard underwriting variables.

The finding that matters for a team without a commercial data licence: the official flood zone classification — France's TRI maps, the UK equivalent of EA Flood Zones — adds meaningful lift over a baseline model, but a dynamic rainfall intensity indicator built from freely available reanalysis data adds more. And building-level attributes available at no cost from public registers add more still.

You do not need to buy the expensive product to build a substantially better model than your current one.

---

## What the EA flood zone map actually tells you

The EA Flood Map for Planning classifies English addresses into Zone 1 (less than 0.1% annual probability), Zone 2 (0.1–1%), Zone 3a (greater than 1%), and Zone 3b (functional floodplain). The NaFRA2 update, published in 2023, added surface water depth bands down to roughly 2m resolution.

These are planning tools that happen to be used in insurance pricing because they are the most structured publicly available flood hazard data we have. The EA's own documentation says they are "not suitable for showing whether an individual property is at risk." This is accurate: the zone boundaries come from hydraulic modelling at 50–100m resolution and say nothing about the specific building's floor level, basement configuration, proximity to a drainage channel, or whether the local drain was upgraded since the model was last run.

More practically: in the Moriah et al. study, official flood zones cover about 5% of the insured portfolio. That 5% generates 11% of flood events — so the zones work as a screen. But **89% of flood events occur outside the flood zones entirely** (in the Moriah et al. French study). Whatever model you build on zone classification alone is pricing the wrong 95% of your portfolio based on postcode and sum insured with no flood signal at all.

The same pattern holds in UK claims experience. Surface water flooding — blocked drains, flash run-off, groundwater emergence — does not correlate with fluvial zone mapping. This is not a modelling subtlety. It is routine, large-volume claims experience that every home insurer with a few years of data can see.

---

## What free data actually moves the needle

The Moriah et al. sequential framework tests four nested GLMs against a holdout. For claim occurrence, here is what each layer adds over the previous one (Gini coefficients, 5-fold CV):

| Model | Gini — occurrence |
|---|---|
| Underwriting data only | 0.234 |
| + static flood zones + climate region | 0.377 (+61%) |
| + rainfall intensity (MILRE) | 0.477 (+27% over previous) |
| + building and environmental attributes | 0.546 (+15% over previous) |

The rainfall intensity indicator — MILRE, discussed below — outperforms the static flood zones on marginal contribution. And MILRE is computed from ERA5-Land reanalysis data, which is free from the Copernicus Climate Data Store. The building attributes in Layer 4 come from sources with direct UK equivalents: cadastral registers, construction age, floor area. The UK equivalents are the EPC register and OS AddressBase — both free or very cheap.

The expensive commercial product (JBA, Fathom, RMS) sits between Layers 2 and 3 in this architecture: better spatial resolution for the static hazard layer, plus proprietary building vulnerability scores. It is strictly better. But it is not the only path to a materially improved model.

---

## The MILRE indicator: what it is and how to construct it

MILRE (Most Intense Local Relative Event) is the paper's main methodological contribution. The logic is:

For each property location, find the nearest grid points in a daily gridded precipitation dataset. For each grid point and each accumulation window (1, 3, 5, 7, 10, and 30 days), compute the empirical CDF rank of observed rainfall within all historical observations at that point and window. MILRE is the maximum CDF rank across all grid points and windows.

The output is a number in [0, 1]. A MILRE of 0.97 means the most intense local precipitation event was in the top 3% of what has historically occurred at that location. A MILRE of 0.40 means the event was unremarkable by local standards — it rained, but not unusually hard.

Two properties in different climatic regions with identical raw millimetre accumulations will have different MILRE values, which is correct: 40mm in three days is routine in the Scottish Highlands and extraordinary in East Anglia. The CDF normalisation makes the indicator locally calibrated rather than globally scaled.

For UK pricing, the right data source is **HadUK-Grid** rather than ERA5-Land. Met Office produces HadUK-Grid via CEDA Archive: daily gridded precipitation at 1km resolution, daily from 1891 for precipitation, in NetCDF format. Commercial licence applications to CEDA are straightforward and the cost is well below any commercial flood data product. At 1km versus ERA5-Land's 9km, HadUK-Grid provides substantially better spatial resolution for the convective rainfall events that drive UK surface water flooding.

The annual variant, ann_MILRE, is the annual maximum MILRE across all days in the year. It is the version used for frequency modelling when you do not have exact event dates on each policy. This is the variant most pricing teams would compute first: extract HadUK-Grid for each year, compute maximum CDF rank per property location, join to policy data by year-in-force.

---

## The UK open data stack for a building-level model

These are the sources a team can use without commercial licences, and what each contributes to the Moriah et al. framework:

**HadUK-Grid v1.3.1 (Met Office / CEDA Archive)**
Replaces ERA5-Land for MILRE computation. The research licence via CEDA is free; commercial use of HadUK-Grid requires a separate licence from the Met Office — contact the Met Office directly. Daily precipitation grids, 1km, daily precipitation from 1891. Format: NetCDF via xarray. This is Layer 3 in the framework — the rainfall intensity layer that outperforms static zone classification on marginal contribution.

**EA Flood Map for Planning + NaFRA2 (DEFRA Data Services Platform)**
Flood zone polygons and the 2025 surface water depth bands. OGL v3. Downloadable as shapefiles or accessible via WFS API at `environment.data.gov.uk`. This is Layer 2 — the static hazard layer. It is necessary but not sufficient.

**EPC Register (MHCLG, epc.opendatacommunities.org)**
25 million-plus certificates for England and Wales, linked to UPRN. Provides construction year, floor area, wall type, and property type. The construction year variable is particularly strong for flood severity: pre-1919 solid-wall construction has significantly higher water ingress exposure than modern cavity wall. Direct UK contribution to Layer 4.

**OS Open Rivers (OS Data Hub)**
Polyline watercourse network, free download, OGL. Useful for distance-to-watercourse via a KDTree spatial join. Distance to the nearest watercourse is one of the variables the paper identifies as a strong predictor in the composite WCTRII variable. For elevation relative to watercourse, you also need a DEM: OS Terrain 50 is open at 50m resolution. The 2m resolution OS Terrain 5 requires a licence but is licensed by many local authorities and some commercial data brokers under re-use terms.

**OS OpenMap Local — building footprints (OS Data Hub)**
Free polygon layer for Great Britain. Building footprint area, combined with construction year from EPC, gives a size-and-age profile that directly substitutes for the cadastral data in the French study.

The data gap versus the French study is PPRI — France's municipal flood prevention plans, which sit between national zone maps and building-level data. England has no equivalent layer that is machine-readable and nationally consistent. The Local Flood Risk Management Strategies produced under the Flood and Water Management Act 2010 are patchy, non-standardised, and not suitable for automated ingestion. This is a genuine limitation. UK pricing teams working with purely open data will have a weaker Layer 2 than the paper's French model, even though HadUK-Grid gives a stronger Layer 3.

---

## Why the timeline is now, not 2039

Flood Re ends in 2039. We have written about this before, and we will not rehearse the arithmetic again. The point for a smaller insurer or MGA is more immediate.

The FCA's 2026 access to insurance priorities explicitly name high-risk properties — including flood-risk properties — as a concern. The Flood Pricing Certificate pilot, anticipated under the Flood Re Transition Plan, will require participating insurers to document the basis of their flood pricing. "We apply an EA Zone 3 loading" is a defensible answer but a weak one. "We apply an EA Zone 3 loading plus a rainfall intensity component derived from HadUK-Grid, calibrated on our own claims experience" is substantially stronger — and the second answer requires about six weeks of data engineering work using free sources.

The reinsurance market is also moving. Post-Storm Darragh (December 2024) and the 2025 Somerset and Yorkshire events, cat programme renewals for UK property have come with increasing scrutiny of flood modelling methodology. Underwriters at Lloyd's and the London market are asking for more granular flood exposure analysis. An MGA that can demonstrate a building-level frequency-severity model — even one built on open data — is in a better position in those conversations than one relying on postcode zone classification.

The practical barrier is not data cost or methodology complexity. It is engineering time to connect the data sources. The Moriah et al. paper shows what the target architecture looks like. The UK open data exists. The missing piece is a clean open-source pipeline that connects them.

---

## What to do this week

Three concrete actions for a pricing team that wants to move from EA zone classification to a better model without buying a commercial licence:

**Register for CEDA Archive access.** It takes a working day. Download a sample year of HadUK-Grid daily precipitation for a test region. Verify you can read it with xarray and extract values at arbitrary (easting, northing) coordinates. This is the single highest-leverage action: HadUK-Grid is the foundation of the MILRE calculation.

**Join your portfolio to EPC.** Download the bulk EPC release from epc.opendatacommunities.org. Match to your portfolio by UPRN or address. You will get construction year and wall type for roughly 40–60% of your home book. These are immediate rating factors — construction year predicts flood severity (and fire, subsidence, and escape-of-water severity) and is defensible to a validation committee. Our [actuarial model validation guide](/2026/04/04/actuarial-model-validation-python/) covers how to produce the consistent documentation that a Flood Pricing Certificate review will require.

**Map distance to watercourse.** Download OS Open Rivers and compute the straight-line distance from each portfolio property to the nearest watercourse vertex via a KDTree. It takes under an hour of Python. Distance to watercourse is a strong predictor in both the occurrence and severity models in the French study and has never been easier to produce for a UK portfolio.

None of these replace a commercial flood model for a specialist flood-exposed book. But they are the difference between pricing on postcode and pricing on something closer to the physical risk. Across 89% of the portfolio that sits outside the EA flood zones, they represent the only risk signal you have.

---

*We have covered different aspects of the Moriah et al. methodology in earlier posts: the sequential GLM attribution in detail in [What Actually Drives Flood Claims](/2026/03/26/geolocated-weather-building-data-flood-insurance/); the NaFRA2 data and a Python pipeline for EA data integration in [Beyond Flood Zone 3](/2026/03/31/beyond-flood-zone-3-building-level-property-risk-python/); and the Flood Re 2039 implications for book strategy in [Building-Level Flood Pricing: What Flood Re's 2039 Phase-Out Means for Your Models](/2026/04/02/flood-re-2039-building-level-flood-pricing/).*

*arXiv:2603.02418 (Moriah, Vermet, Ailliot, Naveau, Legrand, March 2026). HadUK-Grid v1.3.1.ceda, Met Office / CEDA Archive. EA Flood Map for Planning, NaFRA2: DEFRA Data Services Platform, OGL v3. OS Open Rivers, OS OpenMap Local: OS Data Hub, OGL.*
