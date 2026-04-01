---
layout: post
title: "Territory Rating Beyond Postcode Lookup Tables"
date: 2026-04-01
categories: [spatial, territory-rating]
tags: [spatial, territory-rating, bym2, gwr, gwpr, inla, postcode, fairness, fca, consumer-duty, bayesian, conformal-prediction, factor-copula, spatial-embeddings, insurance-spatial, uk-motor, uk-property]
description: "UK territory rating is mostly postcode-to-band lookup tables. That creates both actuarial and regulatory risk. We work through the spatial statistics toolbox that does better: BYM2 for credibility-weighted area effects, INLA for speed at UK scale, contrastive embeddings for point-level data, and factor copulas for peril correlation. Plus the FCA angle no team can ignore."
math: true
author: burning-cost
---

Most UK pricing actuaries do territory rating the same way: a GLM with postcode sector dummies, or a lookup table mapping postcode to a territory band, or both. The postcode sector is a known quantity, the relativities come from the credibility-weighted experience study run two years ago, and the model passes through without comment.

This approach has two problems. The first is actuarial: postcode dummies do not borrow strength across neighbours, cannot handle new postcode sectors, and give no diagnostic to distinguish genuine spatial risk from statistical noise. The second is regulatory: the FCA and Lindholm et al. (2024) have independently shown that spatial pricing variables can proxy for protected characteristics. Consumer Duty's Price and Value outcome requires demonstrable actuarial justification. "Our postcode relativities reflect experience" is weaker than it used to be.

The spatial statistics literature has moved on considerably. This post works through the methods that matter — and, bluntly, the ones that do not.

---

## Why postcode dummies fail at the edges

A GLM with postcode sector fixed effects has an uncomfortable property: for postcode sectors with sparse experience, the estimate is underdetermined. Ten claims over three years in RH19 gives you a relativity with a standard error that would make the average actuary queasy — yet that relativity goes into the rating file weighted the same as SW1A's 40,000-policy estimate.

The standard answer is credibility weighting: blend the postcode estimate toward the regional or national mean. This helps, but it is purely statistical — the blending takes no account of the fact that RH19 is surrounded by other East Grinstead postcode sectors with similar housing stock, demographics, and crime rates. You have neighbours, and you are ignoring them.

The spatial statistics version of this is the **BYM2 model** (Besag, York, and Mollié, with the reparameterisation from Riebler et al. 2016). BYM2 decomposes the postcode sector random effect into two components:

- A spatially structured ICAR component: borrows strength from neighbours in the adjacency graph
- An unstructured IID component: captures idiosyncratic area effects not explained by neighbours

The rho parameter (ρ) directly quantifies the proportion of variance that is spatially structured. A high rho — say 0.85 — tells you the postcode effect is driven by spatial patterns that a well-specified covariate model should explain. A rho near zero suggests the variation is idiosyncratic, not spatial; adding more spatial smoothing would be wrong.

Our [insurance-spatial](https://github.com/burning-cost/insurance-spatial) library implements BYM2 via PyMC with adjacency matrices built from libpysal. The basic workflow is:

```python
from insurance_spatial import BYM2Model, AdjacencyMatrix

adj = AdjacencyMatrix.from_shapefile("postcode_sectors.shp")
model = BYM2Model(adj)
model.fit(claims=df["claims"], exposure=df["exposure"])
relativities = model.territory_relativities()
```

For a well-specified model on a UK insurer's motor book, you get credibility-weighted, spatially-smoothed relativities that work even for sectors with zero claims in the experience period — because the ICAR prior borrows from neighbours.

---

## GWR is not the answer

Geographically Weighted Regression turns up regularly in the spatial pricing literature as an alternative to global GLMs. The appeal is intuitive: instead of a single coefficient for "building age", fit a different coefficient at each location using a kernel-weighted local subsample. If building age matters more in old Victorian terraces in northern cities than in new-build commuter belts, GWR will find it.

Rivas-Lopez, Matilla-García, Míguez-Salido, and Bravo-Ovalle published a GWPR study in *Applied Spatial Analysis and Policy* in February 2025 (DOI: 10.1007/s12061-024-09632-4) applying this to Spanish home insurance water damage claims. GWPR outperformed a global GLM on AIC, MSE, and MSPE. The paper is technically careful. But the setup exposes GWPR's fundamental limitations for UK-scale territory rating.

**The dataset is 2,500 observations.** The authors explicitly reduced from the full dataset due to GWR's computational burden, which scales as O(n²) in naive implementation. UK insurers have millions of policies.

**GWR cannot extrapolate.** It is an interpolation method. For a postcode sector with no historical claims — a new development, a recently split sector, a boundary change — there is no principled way to generate a GWPR prediction. The local kernel has no data to weight. BYM2 handles this via ICAR smoothing across the adjacency graph.

**There is no credibility mechanism.** A postcode sector with five claims gets the same local regression as one with 5,000 — just with higher variance. This is actuarially incoherent. BYM2's hierarchical prior is the Bayesian analogue of credibility theory; GWR has no equivalent.

**Local multicollinearity is endemic.** Spatial covariates cluster. Urban density, building age, deprivation, and crime rates are all spatially autocorrelated. In the local kernel window, they are more correlated than in the full sample. This creates unstable, sign-reversed GWR coefficients that look like interesting spatial heterogeneity but are estimation artefacts.

Where GWR is genuinely useful: as a pre-modelling diagnostic. Run GWPR first to identify which GLM covariates have spatially varying effects. Then build a BYM2 model with appropriate covariate structure. We may add a thin wrapper around the `mgwr` Python package to insurance-spatial for this exploratory use — it would be a convenience layer over an existing library, not a new capability.

---

## INLA will make BYM2 practical at UK scale

The honest limitation of BYM2 via PyMC is speed. UK postcode sectors number approximately 8,000. MCMC sampling for a BYM2 model with N=8,000 spatial units takes hours — acceptable for an annual rerate, less so for anything iterative.

Gudmundarson and Tzougas published in the *European Actuarial Journal* in February 2026 (DOI: 10.1007/s13385-025-00443-6) fitting Besag negative-binomial models to climate-related property insurance claims in Greece 2012-2022 using R-INLA rather than MCMC. INLA — Integrated Nested Laplace Approximation — replaces the full MCMC posterior with a Laplace approximation that is 10-50x faster for models of this structure. For N=8,000 spatial units, that difference is between a two-hour run and a three-minute run.

The paper's key finding is substantive, not just methodological: a flood vulnerability index combining geographic location and infrastructure quality was the primary driver of spatial claim gradients, with low standard errors, confirming the index captures genuine risk rather than noise. This is how BYM2-style Besag models should be validated — look at what the spatial random effect is capturing and whether it reduces when you add informative covariates.

The Python path to INLA is `pyINLA`, currently less mature than R-INLA but under active development as of 2025. The alternative is an `rpy2` bridge to R-INLA, which is robust but adds an R dependency. We expect to add an INLA inference backend to insurance-spatial once pyINLA handles BYM2 cleanly — it is the single most impactful improvement we could make to the library for production UK territory rating.

---

## Spatial embeddings for point-level data

BYM2 is an area-level model. It aggregates claims to postcode sector, builds a spatial graph, and smooths the area effects. For most UK territory rating this is appropriate — we are setting a rate for the area, not the individual property.

But there is a complementary use case: enriching individual policy records with spatial context before the record enters a GLM or gradient-boosted model. For a surplus lines or commercial property book where the territory effect is at building level, area-level models miss within-postcode variation. And entity embeddings learned from historical claims — the obvious alternative — fail for new postcodes, boundary changes, and portfolio transfers.

Holvoet, Blier-Wong, and Antonio (arXiv:2511.17954, November 2025) address this with contrastive spatial embeddings. The architecture trains on paired satellite imagery and OpenStreetMap features using a SimCLR-style contrastive loss (temperature τ=0.07), producing a task-agnostic location embedding from lat/lon coordinates alone. At inference time, no satellite imagery access is needed — coordinates map directly to embeddings.

The key test: embeddings trained without French data still improve French real estate price predictions. Transferability is the point. The pretrained weights are available from the [GitHub release](https://github.com/freekholvoet/MultiviewSpatialEmbeddings).

The practical use case for a UK pricing team is: take an individual policy lat/lon, generate a spatial embedding, and include that embedding as a feature set in the GLM or GBM pricing model. This handles new buildings, new postcodes, and properties at the boundary of two postcode sectors — all cases where postcode dummies give the wrong answer.

One honest caveat: the embeddings are not validated on UK insurance frequency or severity targets. UK OSM coverage is dense and satellite imagery is high-quality, but whether the learned spatial features are risk-relevant for UK motor or property claims needs testing on a UK book. This is a validation task, not a theoretical objection.

---

## Factor copulas for flood and storm pricing

BYM2 models baseline territory risk — the expected claim rate at a postcode sector averaged across all years. For flood and storm lines, there is a separate problem: within a single event, policyholders' claims are spatially correlated. The December 2015 Cumbria floods and the November 2019 Yorkshire floods each generated thousands of claims with obvious spatial dependence. Standard territory models ignore this; they assume claims are independent conditional on the territory effect.

For gross territory pricing, ignoring within-event correlation is defensible. For reinsurance pricing and aggregate loss distributions, it is not.

Gao and Shi published in *ASTIN Bulletin* (55, pp. 242-262, 2025, DOI: 10.1017/asb.2025.7) a factor copula model for replicated spatial data — the technical term for multiple policyholders making claims from the same storm event. The model separates two correlation sources:

1. **Spatial correlation**: policyholders close together are more correlated due to physical proximity, modelled with an exponential distance-decay kernel
2. **Common shock**: all policyholders in the storm footprint are correlated because they experienced the same event, modelled as a single shared factor

Using Kansas hailstorm claims 2011-2015, they find the common shock factor dominates. The shared storm experience matters more than geographic proximity per se. This has a direct implication for aggregate loss distributions: the tail is fatter than independence models suggest.

The paper uses two-stage MLE — marginals first, then copula parameters — making it tractable from historical claims data without a full CAT model infrastructure. UK flood events produce exactly this data structure; UK insurers have Met Office footprint data to define event membership. This is not a competitor to BYM2 territory rating. It is a separate module for pricing the spatial dependence of aggregate losses, which matters for reinsurance structure and CAT-adjacent personal lines pricing.

---

## The FCA angle you cannot ignore

The FCA published a research note — "Motor Insurance Pricing and Local Area Ethnicity (England and Wales)" — analysing over six million motor policies. The finding is often mischaracterised. The FCA found that genuine actuarial risk differences between areas account for the overwhelming majority of premium differences between ethnically diverse and non-diverse postcodes. They did not find systematic overcharging beyond risk.

That sounds reassuring. It is not quite.

The qualification is that risk is itself partially shaped by socioeconomic factors that correlate with ethnicity — crime rates, parking conditions, vehicle type. Whether a risk driver that is itself partly caused by historical discrimination constitutes legitimate actuarial justification is genuinely contested. The FCA's Consumer Duty requires demonstrably fair value outcomes, not just actuarially defensible ones.

Lindholm, Richman, Tsanakas, and Wüthrich in the *Scandinavian Actuarial Journal* (9, pp. 935-970, 2024, DOI: 10.1080/03461238.2024.2364741) formalise why this is hard. They show that proxy discrimination — using postcode as a proxy for ethnicity — and demographic disparity — groups facing systematically different prices — cannot simultaneously be eliminated under actuarial constraints. You cannot make postcode rating both ethnicity-neutral and actuarially accurate in a world where risk and ethnicity are empirically correlated. The tension is mathematical.

The practical implication for pricing teams is not to abandon territory rating. It is to be able to *explain* the territory effect. BYM2 gives you two things no lookup table does:

First, the rho parameter. High rho means the spatial variation is structurally spatial — it is driven by geographic patterns that a covariate model should explain. Low rho means the variation is idiosyncratic. A high-rho territory model is more defensible under Consumer Duty than a pattern of residuals that looks more like demographic noise.

Second, covariate attribution. BYM2's random effect represents the spatial variation not accounted for by your rating factors. If you have included flood zone, deprivation score, crime index, and local property values as covariates, and the BYM2 spatial random effect is small, you have demonstrated that the territory effect reflects measurable risk drivers. That is a defensible audit trail.

The Lindholm et al. methodology for removing protected-characteristic proxying from spatial variables — based on inverse probability weighting — is technically clean and applicable to any territory rating variable. The political and legal question of whether UK insurers are required to apply it remains unresolved. The technical barrier to doing so does not.

---

## The uncertainty problem: GeoConformal

One underappreciated issue with territory relativities is interval calibration. A postcode sector relativity of 1.23 is a point estimate. The uncertainty around it is postcode-specific: a coastal flood-prone sector in rural Wales has fundamentally different uncertainty than a large urban London sector. Standard prediction intervals give the same width everywhere; the actual uncertainty varies by order of magnitude.

GeoConformal prediction (arXiv:2412.08661, December 2024, updated April 2025) extends conformal prediction to spatial settings by replacing uniform calibration weights with kernel distance-weighted calibration. The test point at a coastal flood-zone postcode draws calibration points from nearby flood-zone sectors; its interval is wider, and correctly so. Coverage in a housing price benchmark reached 93.67% versus 81% for bootstrap methods.

The insurance-spatial conformal module already implements this approach. The GeoConformal paper provides the theoretical grounding for why it works and what its coverage guarantees actually mean — local coverage (correct near this location) rather than marginal coverage (correct on average). For a pricing actuary, local coverage is the right target. A confidence interval on a territory relativity that happens to be correct on average across all of England is not particularly useful when you are filing rates for the Severn estuary.

---

## What this means in practice

The territory rating toolbox in 2026 is richer than postcode-to-band lookup tables suggest.

For **frequency/severity by area**: BYM2 with ICAR smoothing is the right foundation. It handles sparse data via partial pooling, quantifies the spatial signal via rho, and gives a defensible audit trail for Consumer Duty purposes. insurance-spatial implements this today.

For **scale**: INLA will replace MCMC for production-scale UK territory rating once pyINLA matures. The Gudmundarson & Tzougas 2026 paper demonstrates the approach on flood insurance data structurally similar to UK books. 10-50x speedup for 8,000 spatial units is not a nice-to-have; it is what makes the model usable in an annual rerate cycle.

For **individual policy enrichment**: Holvoet et al. 2025 pretrained spatial embeddings from sat/OSM data. Load the weights, pass coordinates, get spatial features. Validate on your book first.

For **aggregate peril pricing**: Gao & Shi 2025 factor copula. If you write flood or windstorm at any scale where reinsurance matters, ignoring within-event spatial dependence is leaving real error on the table.

For **regulatory defensibility**: the Lindholm et al. 2024 framework tells you exactly what your territory model must demonstrate. BYM2's rho and covariate attribution provide the evidence. The FCA has looked at six million policies. The next time they look, the question will be whether your spatial model can show its working.

GWR is a useful diagnostic. It is not a replacement for any of the above.
