---
layout: post
title: "Computer Vision for Property Insurance: What Actually Works in 2026"
date: 2026-03-26
categories: [property-insurance, computer-vision]
tags: [computer-vision, satellite-imagery, aerial-imagery, property-risk, underwriting, cape-analytics, vexcel, uk-insurance, pricing]
description: "Cape Analytics has 700k+ claim validations and a 400% loss ratio differential between best and worst roof condition. The UK market has none of that infrastructure. Here is an honest account of what imagery can and cannot do for property risk."
---

Aerial imagery vendors make a straightforward claim: a satellite or plane flies over your house, computer vision scores the roof, and the insurer prices more accurately. In the US this is real and deployed. In the UK it is mostly not yet operational — and the distinction matters if you are deciding where to allocate pricing resource in 2026.

This is the first of three posts on computer vision and LLMs for property risk. This one covers the vendor landscape, what imagery genuinely can and cannot tell you, the UK data infrastructure problem, and the regulatory position. The second post will work through the academic evidence in detail. The third covers proxy discrimination governance ([`insurance-governance`](/insurance-governance/) handles the model card documentation).

---

## What is deployed in the US

The US market has two production-grade aerial analytics products.

**Cape Analytics** (now partnered with EagleView following their 2023 collaboration) is the most studied. Their Roof Condition Rating version 5 incorporates validation against 20 million exposure records and over 700,000 claims. The headline result: properties rated Severe have a loss ratio roughly 400% of those rated Excellent. That is a material signal, not a rounding error. Cape's products include roof age (claimed 95% accuracy), exterior condition, and wildfire risk indicators — all extracted from aerial photography via proprietary computer vision models.

The validation caveat is worth stating: this is a vendor white paper, not a peer-reviewed academic study. Independent replication on Cape's dataset does not exist in the published literature. We cannot assess the methodology. But 700k claims is not a number you dismiss because of publication venue.

**Verisk** offers Aerial Imagery Analytics, built on a partnership with the Vexcel Data Program — which runs the world's largest commercial aerial imagery operation. Verisk's product has been adopted by Citizens Insurance in Florida, among others. The computer vision layer extracts visible roof defects, feeding their existing COPE data infrastructure.

**Nearmap** acquired Betterview in 2023, folding a property intelligence platform (roof scoring, COPE extraction) into their high-resolution aerial imagery product. The combined offering is marketed to US P&C insurers for both underwriting and claims.

The consistent pattern: these are not research projects or proof-of-concepts. US carriers are deploying this for policy issuance decisions. The regulatory pressure (more on that below) comes precisely because insurers were acting on these outputs without adequate governance.

For claims rather than underwriting, **Tractable** (London-headquartered) is deployed with Ageas UK, automating property damage estimates from smartphone photos. Their claimed metrics are 70% automation rate and settlement time reduced from months to one day. This is a post-loss application — the image comes after the event, not before it — and the validation loop is much tighter because you can directly compare automated estimates to settled claim costs.

**ICEYE**, a Finnish company running 40+ SAR satellites as of late 2024, provides flood extent data within 6–12 hours of an event for claims triage and parametric triggers. Their partnership with Munich Re went live in January 2026. SAR penetrates cloud cover, which matters for an event response product; it does not matter much for routine underwriting because the resolution is too coarse for individual property condition assessment.

---

## What UK infrastructure actually exists

The UK data picture is fragmented compared to the US, and the fragmentation is the real obstacle to deployment.

**Vexcel Data Program** covers UK urban areas at 7.5–15cm resolution (London, Birmingham, Manchester, Edinburgh, Glasgow, Belfast, Bristol) and rural areas at 15–20cm. That resolution is sufficient for roof material identification and large defect detection. The problem is update frequency: Vexcel do not publicly specify their UK refresh cycle, but it is likely a 12–18 month interval. For a roof condition model, an 18-month-old image creates real adverse decision risk — US state regulators have explicitly flagged stale imagery as a problem.

**Getmapping** provides UK aerial photography at up to 5cm resolution and explicitly lists insurance as an application. They partner with Ordnance Survey.

**OS MasterMap** with the Building Height Attribute provides building-level footprints and heights under commercial licence. The 3D Data Bundle (MasterMap + Terrain 5 + Building Height) is the standard starting point for structural risk analysis.

**Environment Agency National LiDAR Programme** is free at data.gov.uk and covers England with best-in-class terrain data — the right foundation for flood risk modelling at property level. Wales and Scotland have separate programmes under Natural Resources Wales and SEPA respectively.

**EPC Register** (England and Wales, available via MHCLG) is the most underused data source in UK home insurance pricing. Approximately 30 million certificates with free bulk download. The relevant fields are CONSTRUCTION_AGE_BAND, ROOF_DESCRIPTION, WALLS_DESCRIPTION, BUILT_FORM, and PROPERTY_TYPE. The limitation: roughly 65% of properties have an EPC, with new builds heavily over-represented. Pre-1900 Victorian terraces — the most interesting cohort from a risk perspective — are the least well-covered.

**JBA** flood data (5m grid, four return periods, UPRN-level resolution) is the commercial flood risk standard already in use by most UK insurers.

**Google Street View API** is commercially available and has been used in academic research for property feature extraction (see the second post in this series for the Kita & Kidzinski 2019 results). GSV Terms of Service explicitly prohibit use to identify individuals or for insurance risk rating without explicit consent — a constraint most teams do not check carefully enough.

What does not exist in the UK: a deployed Cape Analytics equivalent. No UK company offers a validated aerial property intelligence product with claims-calibrated condition scoring, at national scale, in production. Vexcel imagery is available; there is no analytics layer sitting on top of it in packaged form for UK insurers.

---

## What imagery reliably tells you

Based on the evidence from deployed US systems and the academic literature, these are the use cases where imagery has demonstrated genuine predictive signal:

**Roof condition and age** — this is the primary validated application. Roof failure drives claims across property damage, escape of water from roof penetration, and storm peril. The Cape Analytics signal (400% loss ratio differential) is the strongest quantitative result in the field. Roof material and pitch are also extractable with reasonable accuracy from sub-20cm imagery.

**Solar panel and home modification detection** — alterations that increase rebuild cost or change risk profile. Pools, large outbuildings, extensions, and solar installations are visible from above. The insurance relevance is primarily to declared sum insured accuracy and, for pools, liability exposure.

**Vegetation clearance for wildfire risk** — the clearest US application outside of roof condition. Defensible space assessment (clearance within 30 feet of structure) is measurable from aerial imagery. Less relevant in most of the UK but directly applicable to Scotland, parts of the Welsh hills, and moorland-adjacent properties.

**Area-level territory signals** — satellite imagery combined with OpenStreetMap features can generate spatial embeddings that improve sub-postcode territory rating. Holvoet et al. (November 2025, arXiv:2511.17954) demonstrated this on French real estate price prediction using contrastive learning on ~100k European locations. The embeddings consistently improved GLM, GAM, and GBM accuracy over raw coordinates. This is not individual-property imagery — it is neighbourhood-level signal — and it is the most accessible imagery application for a UK pricing team with limited data infrastructure.

---

## What imagery cannot tell you

This matters as much as what it can do.

**Interior condition** is invisible. Escape of water — burst pipes, faulty plumbing — is the largest claim type in UK home insurance by count, and it has nothing to do with roof condition or exterior appearance. An immaculate Victorian terrace with modern cladding and a 40-year-old lead pipe distribution system is a significant escape of water risk. Imagery tells you nothing.

**Subsidence indicators** require specialist survey. The settlement crack pattern that distinguishes shrinkable clay subsidence from normal thermal movement is not detectable from above, and frequently not from street level either.

**Electrical and heating installation age** — two major risk factors for fire claims — are structurally invisible. EPC records heating system type but not condition.

**Precise rebuild cost** cannot be reliably estimated from imagery alone. Rebuild cost depends on internal specification, materials, fittings, access constraints, and local labour costs — none of which are visible. Aerial imagery can establish footprint and height, which anchors a crude estimate, but the error range on that estimate is large enough to create material underinsurance risk if treated as primary.

**Flood risk from elevation alone** is an oversimplification. Ground elevation from LiDAR is a necessary input to flood modelling, but it is not sufficient. A property at low elevation with a culverted watercourse above it is more exposed than elevation alone suggests. JBA's 5m resolution scoring, which integrates channel geometry and surface drainage, is better than raw LiDAR elevation for pricing.

**Rural and non-standard properties** are where imagery is weakest. UK coverage thins out for rural properties; update frequency falls; and CV models trained on US suburban stock — or even UK urban stock — transfer poorly to thatched roofs, stone construction, and converted agricultural buildings. These are precisely the properties where standard market data is also inadequate and where a new data source would add most value. The coverage gap is in the wrong place.

---

## The regulatory position

US state regulators moved first, and the direction of travel is instructive.

Connecticut issued a bulletin in March 2024 addressing aerial imagery misuse in underwriting. Pennsylvania followed in May 2024, advising against sole reliance on satellite imagery — specifically calling out cosmetic issues (staining, streaking, discoloration) that do not represent material hazards. Multiple states now prohibit adverse underwriting actions (non-renewal, cancellation) based solely on imagery showing cosmetic rather than structural defects. The consistent requirement: image recency validation, distinction between cosmetic and material defects, consumer transparency on adverse decisions, and a mechanism for physical inspection to override an automated decision.

The UK has no FCA-specific guidance on aerial imagery in insurance pricing as of March 2026. That is not a green light — it means the existing framework applies.

The applicable rules are: FCA Consumer Duty (fair value requires defensible premium-setting data), UK Equality Act 2010 (indirect discrimination via proxy variables), and UK GDPR / ICO guidance on AI profiling. The proxy discrimination problem with imagery is material. Street-level and aerial imagery of residential neighbourhoods correlates with the ethnic composition of the area. Kita & Kidzinski (2019, arXiv:1904.05270) demonstrated that five image-derived features from Google Street View of insured addresses improved motor Gini by approximately 2 percentage points — but those features are plausibly proxies for neighbourhood demographics. The ICO's AI guidance explicitly states that proxy variables can encode protected characteristics in non-obvious ways.

FCA DP18/1 (2018) raised proxy discrimination concerns for postcode-based pricing. It has not been updated for satellite imagery. Research at City, St George's University (2023) found that standard UK motor pricing produces systematically higher premiums for some minority ethnic groups via postcode proxies — imagery would compound this, not neutralise it.

Any imagery-derived pricing feature needs disparate impact testing before deployment. This is not optional governance. For the testing framework, see [FCA proxy discrimination testing in Python](/2026/03/22/fca-proxy-discrimination-python-testing-guide/) which covers exactly this use case for continuous features. It is the minimum required to satisfy the Equality Act and Consumer Duty simultaneously. We cover the governance framework in detail in the third post.

---

## The honest verdict

Computer vision adds genuine signal for property peril pricing. The US evidence on roof condition is the clearest result in the field, and it is real. Claims-side applications (Tractable for photo-based damage assessment, ICEYE for flood event response) work and are deployed.

For a UK pricing actuary evaluating whether to invest in CV and imagery, the practical situation in March 2026 is:

The aerial imagery infrastructure that makes US deployments work — frequent, high-resolution coverage with an analytics layer validated against claims — does not exist in the UK at comparable scale. Vexcel has coverage but not the calibrated analytics layer. There is no UK equivalent of Cape Analytics. Building proprietary models requires training data (imagery matched to claims outcomes) that no UK insurer has publicly demonstrated they hold, combined with an imagery refresh cycle that is slower than US deployments.

The most accessible UK applications right now are: EPC-linked construction attribute enrichment (structured data, no imagery), JBA flood at UPRN level (already deployed by most), EA LiDAR for terrain-informed flood modelling, and spatial embeddings from satellite + OSM data for sub-postcode territory rating. These do not require building a CV pipeline.

The street view and aerial imagery applications — roof scoring, condition rating, non-standard construction detection — are technically feasible and intellectually compelling. The blockers are infrastructure (imagery coverage, freshness), training data (no UK ground-truth dataset linking imagery to claims), and governance (proxy discrimination risk that must be characterised before deployment, not after).

UK actuaries should watch this space closely but allocate budget cautiously. The question is not whether imagery contains signal — it does — but whether the signal is accessible, validatable, and deployable within UK data and regulatory constraints in the near term. Right now, the answer is: partially, in specific applications, with material effort and governance overhead.

---

*The second post in this series works through the academic evidence in quantitative detail: what Kita & Kidzinski (2019), Holvoet et al. (2025), and the March 2026 Moriah et al. paper actually show, and what a realistic LLM-based feature extraction pipeline looks like. The third post covers proxy discrimination governance for imagery-derived pricing features.*
