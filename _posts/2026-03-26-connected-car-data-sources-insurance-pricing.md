---
layout: post
title: "Connected-Car Data Sources for Insurance Pricing: Beyond the Black Box"
date: 2026-03-26
categories: [telematics, data, motor]
tags: [telematics, connected-car, ubi, ev, oem-api, ocpp, smartphone-telematics, uk-motor, gdpr, fca, insurance-telematics, insurance-monitoring, insurance-fairness, by-miles, marshmallow, admiral]
description: "OEM APIs, smartphone SDKs, and charging data — what connected-car data sources UK pricing teams can actually access, and how to turn them into rating factors."
---

The UK has roughly 1.3 million telematics policies, the largest market in Europe alongside Italy. Despite that, most of those policies deliver a single black-box score from a third-party provider, which the pricing team uses as a binary rating variable: either the policyholder has a telematics discount or they do not. The raw data — trip timestamps, accelerometer sequences, GPS traces — stays inside the vendor's system. The actuary prices on the score output, not the signal.

That is a manageable position when the data landscape is static. It is becoming harder to defend as the landscape shifts under three simultaneous pressures: OEM APIs now expose granular trip data directly, without a hardware device; smartphone-only products are making the opt-in threshold much lower; and for EVs specifically, charging behaviour data is becoming a viable usage proxy that no insurer has yet productised. This post maps the data sources available to UK pricing teams right now, what those sources actually contain, and how to turn them into rating factors — including the parts that remain genuinely hard.

This is the second post in our EV and connected-car series. [Post 1 covered EV motor insurance pricing](/2026/03/26/ev-motor-insurance-pricing-beyond-the-flat-surcharge/): the bimodal severity structure, battery write-offs, and why the flat surcharge misprice in both directions simultaneously.

---

## The data landscape

Connected-car data reaches insurers via four channels. They differ materially in data quality, access friction, coverage, and regulatory exposure.

### OEM cloud APIs

EV manufacturers build battery management systems that require continuous telemetry. That infrastructure produces a data exhaust that can — with consumer consent — be queried by third parties via standardised APIs. The normalisation layer is provided by aggregators: **Smartcar**, **Enode**, and **HighMobility** collectively cover 100+ EV and ICE models across Europe, presenting a unified API regardless of which OEM's data sits underneath.

What these APIs expose depends on the OEM:

- **State of Charge** (SoC): real-time and historical, available on virtually all EVs
- **Odometer**: continuous, not self-reported — eliminates the mileage-band gaming that plagues declared-mileage products
- **Charging sessions**: timestamp, duration, energy delivered (kWh), charger type (AC home, DC rapid), sometimes location category
- **Battery State of Health** (SoH): available on Tesla, some BMW models; degradation data across 4-5 year old EVs shows a median of ~93.5% SoH, falling to ~85% at 8-9 years (Thatcham/Generational data)
- **Hard events**: Tesla and some BMW models expose harsh braking and acceleration events; most OEMs do not
- **ADAS activations**: automatic emergency braking (AEB) interventions, lane-keep assist events — patchy coverage, but growing

The significant exception is **Tesla**, which does not participate in third-party data-sharing programmes. You cannot access Tesla fleet data through Smartcar, Enode, or HighMobility unless the individual policyholder connects their Tesla account directly. Given that Tesla is around 15-20% of UK BEV registrations (SMMT data, 2023-24), this gap is not trivial. Ford, Mercedes, and Volkswagen Group do participate in OEM data partnerships; LexisNexis has a confirmed data partnership with Mercedes.

**LexisNexis Telematics Exchange** operates in this space as a normalisation layer for insurers specifically: it ingests OEM API data, smartphone app data, and third-party device data and outputs a standardised driving behaviour score for UBI. LexisNexis Vehicle Build separately provides ADAS classification at VIN level across 2.5 million UK vehicles — this is available now and does not require any policyholder data-sharing consent.

### Aftermarket and aggregators

Before OEM APIs matured, the connected-car data market ran on two channels: OBD-II dongles plugged into the diagnostic port, and smartphone apps. The dongle market is declining for EVs — Tesla locked down CAN bus access for security reasons, and OBD-II coverage on BEVs is inconsistent. For ICE fleets and commercial lines, dongle-based telematics (via **Masternaut**, **Tantalum**) remains relevant, but it is not where the growth is.

**Wejo** and **Otonomo** (now merged as Wejo) aggregate raw trip data from OEM data programmes — primarily GM, Ford, and Stellantis — and package it for analytics buyers including insurers. Their data covers millions of connected vehicles and includes GPS trace, speed profile, and events. The catch: coverage is US-heavy. UK coverage through Wejo is thinner and weighted towards the Stellantis brands (Vauxhall, Peugeot, Citroën) that participate in their data programme. Useful for commercial motor if you write Stellantis fleet; less useful for private motor.

### Smartphone SDKs

The smartphone channel has overtaken hardware devices as the growth vector in UK UBI. No installation required; consent at app download; the phone's GPS and accelerometer provide speed, location, and motion data.

The main players in the UK:

- **The Floow**: the Sheffield-based telematics analytics provider powering multiple UK insurers (including LV=). Provides a white-label SDK plus analytics platform. Their score feeds into the insurer's rating engine as a vendor variable
- **Zendrive**: US-origin, now operating in UK commercial motor and fleet. Specialises in high-frequency trip classification using phone sensors
- **DriveScore** (formerly CityBee/Cambridge Mobile Telematics): standalone consumer app that also provides B2B scoring APIs

**Marshmallow** is the UK example of smartphone-only telematics at scale. Founded in 2017, targeting migrants and new-to-UK drivers who lack UK NCB history, they built their entire proposition on smartphone driving data as an alternative credit signal. They do not use a black box. Approximately 150,000 policyholders as of 2024; loss ratio reportedly competitive with the market. They are the strongest domestic evidence that smartphone-only data collection works at meaningful scale.

**By Miles** operates a different variant: pay-per-mile, using smartphone data to measure actual miles driven rather than declared annual mileage. Octopus Energy invested in the business in 2023 — an interesting signal that energy and insurance data are converging around EV charging behaviour. By Miles prices the per-mile rate on driving behaviour; the base premium covers the parked-car exposure.

**Admiral LittleBox** is the longest-running UK black-box telematics product and the largest by volume. It remains hardware-based (proprietary device, not OBD-II) with a companion app. Admiral reports that LittleBox policyholders have materially better loss ratios than comparable non-telematics policies — but as we covered in the [UBI adverse selection post](/2026/03/25/ubi-adverse-selection-telematics-discounts-drive-away-best-risks/), this is partly selection effect rather than purely a data quality story.

### OCPP charging data

The Open Charge Point Protocol (OCPP) governs communication between EV charge points and central management systems. OCPP 2.0.1 became IEC 63584 in 2024; OCPP 2.1 was released January 2025. UK regulation effective November 2024 mandates open data sharing and 99% uptime for rapid chargers. In practice, this means that charging networks — **bp pulse**, **Pod Point**, **Osprey**, **Gridserve**, **Ubitricity** — are sitting on session-level data (time, duration, kWh delivered, charger type, location) that is technically accessible and increasingly standardised.

No UK insurer has publicly announced a product built on charging network data. The commercial agreements to licence that data do not exist yet. But the regulatory foundation is there, and the actuarial logic is compelling: charging behaviour is a near-perfect proxy for mileage and driving pattern. A policyholder who rapid-charges three times a week at motorway service stations is doing high-mileage motorway driving. A policyholder who charges exclusively at home overnight is doing commuter-pattern urban driving. You do not need a GPS trace to distinguish these profiles.

---

## What the data actually contains

Setting aside the access question for a moment — what does the signal look like once you have it?

**Trip-level GPS traces** give: start/end timestamp, route, distance, duration, and speed profile at whatever resolution the device provides (1Hz for smartphone; 0.1Hz is common for OBD). From speed profile alone you can derive time of day, road type (by speed and map matching), urban versus motorway split, and implied average speed. These are stable, interpretable features.

**Accelerometer data** is richer and noisier. At 1Hz from a modern smartphone you get three-axis acceleration, from which you can derive harsh braking events (longitudinal decel below -3.5 m/s²), harsh cornering (lateral accel above threshold), and harsh acceleration. The threshold choices are not arbitrary — they come from the telematics literature — but they are calibrated to a reference population. The same g-force reading has different risk interpretation depending on speed and road context. This is why the naive event-counting approach systematically penalises urban drivers relative to motorway drivers at equivalent true risk: urban driving generates more events per kilometre regardless of driving quality. The HMM approach in [`insurance-telematics`](/insurance-telematics/) partially addresses this by learning road-type-specific latent states rather than counting events against a fixed threshold.

**Charging session data** (from OEM API or OCPP) gives: session start time, duration, kWh delivered, and charger type. For pricing, the most useful derived features are:

- **Charging frequency by type**: the ratio of home-AC charges to public-DC rapid charges. High rapid-charge frequency correlates with high annual mileage and motorway-centric use. It also accelerates battery SoH degradation
- **Charging time-of-day**: late-night charging (23:00–06:00) correlates with home charging, which correlates with home parking availability. Home parking is a material driver of material damage claims
- **Session energy delivered**: a cross-check on declared mileage. A 40kWh session on a vehicle with 250Wh/km efficiency implies 160km of driving since the last charge. Accumulated across sessions, this gives a better mileage estimate than the policyholder's declaration

**ADAS activation events**, where available, are double-edged. A high AEB activation rate either signals that the driver is operating in dangerous conditions (the AEB is working correctly to prevent crashes) or that the ADAS system is poorly calibrated or the vehicle is encountering false-positive triggers. The net actuarial interpretation is ambiguous; use with caution and always control for road type and mileage.

**Battery SoH** is actuarially cleaner: as we argued in post 1, SoH below ~80% is a direct input to total-loss probability in the severity model. It belongs in the severity sub-model rather than the frequency model.

---

## From raw data to pricing features

The engineering challenge is converting these raw signals into features stable enough to use in a GLM rating engine. Stability matters because a GLM relativity is a long-run parameter estimate — you do not want it to change materially because a policyholder went through a phase of motorway commuting for two months.

The pipeline for smartphone/OBD data:

1. **Clean raw trips**: remove GPS jumps (>200 km/h implied speed between consecutive points), fill short signal gaps, classify road type from speed percentiles or map-matching against OpenStreetMap
2. **Extract trip-level features**: harsh event rates per kilometre, night driving fraction, motorway percentage, urban fraction, trip duration distribution
3. **Fit HMM across the trip corpus**: the hidden Markov model identifies latent driving regimes — typically cautious, normal, aggressive — from the multivariate observation sequence (speed, longitudinal accel, lateral accel, heading change rate). The per-driver state occupancy fractions are more stable than trip-level event counts
4. **Aggregate to driver level with credibility weighting**: Bühlmann-Straub credibility prevents a new policyholder with 5 trips from getting an extreme score. At 30 trips the credibility weight is 30/(30+30) = 0.5; at 100 trips, 0.77

[`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) implements this full pipeline: raw trips in, GLM-ready driver features out. The output is a standard DataFrame with one row per policy — `state_2_fraction` (fraction of driving in aggressive regime), `state_entropy`, `night_driving_fraction`, `motorway_fraction` — that merges directly into your existing rating dataset.

The key features for the frequency model, in rough order of predictive power:

| Feature | Signal | Notes |
|---|---|---|
| `state_2_fraction` | Fraction of trips in high-intensity HMM state | Best single discriminator; Sun & Shi (2024) show 2.3× frequency ratio from 10th to 90th percentile |
| `night_driving_fraction` | Fraction of distance driven 23:00–05:00 | Stable; correlates with young driver risk independent of ADAS |
| `motorway_fraction` | Fraction of distance at motorway speed | Negative frequency predictor; reduces urban stop-start exposure |
| `harsh_event_rate` | Harsh braking + cornering events per 100km | Confounded by urban driving; use alongside `urban_fraction` as control |
| `trip_regularity` | Coefficient of variation of inter-trip intervals | Commuter pattern vs. irregular driving; different exposure distributions |
| `charge_type_ratio` | Home-AC charges / total charges (EV only) | Mileage and road-type proxy without GPS trace requirement |

For the severity sub-model, `battery_soh` (from OEM API) and `charger_type` (as a SoH degradation proxy) are the relevant connected-car inputs.

Feature stability over time is a monitoring requirement, not a one-off check. Driving behaviour drifts with life events — new job, house move, vehicle change — at a faster rate than traditional rating factors. [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) includes drift detection specifically designed for telematics features: PSI monitoring on the state occupancy distribution flags when a policyholder's regime mix has shifted materially since their last renewal.

---

## Regulatory constraints

**UK GDPR and data minimisation.** Connected vehicle data is personal data when linked to an identified driver. The lawful basis for telematics pricing is typically contract (consent to data use as a condition of the product) or legitimate interest. Data minimisation applies: you can collect trip-level accelerometer data for driver risk scoring, but you cannot then use the GPS trace for unrelated marketing purposes. The ICO has not published EV-specific telematics guidance; the existing UBI telematics framework (substantially informed by EDPB Guidelines 01/2020, which remain materially applicable under UK GDPR) applies.

**OEM data sharing and consumer consent.** When data reaches you via an OEM API aggregator, the consent chain runs: OEM collects data at vehicle/app setup → policyholder connects their account to your app → aggregator normalises and passes to you. Each link must be documented. Tesla's absence from third-party programmes means this chain does not exist for Tesla vehicles; you must obtain consent and data connection directly from the policyholder.

**FCA and Consumer Duty.** Two specific pressures:

First, the FCA has been explicit (PS21/5) that price walking — charging longstanding customers more than new business equivalents — is prohibited. Telematics discounts are not exempt. A customer who had a telematics discount at inception and later opts out cannot simply have the discount removed at renewal as a price walk. You need a fair value argument for the premium change that does not rely on a retention-pricing logic.

Second, Consumer Duty fair value assessment requires that telematics products demonstrably deliver value for customers who opt in. If you are only using the telematics score as a discount mechanism for low-risk drivers (and implicitly removing the discount for drivers who score poorly), you need to document that the pricing structure is fair and the score methodology is auditable. Opaque vendor scores with no explainability pathway are increasingly hard to defend under the Consumer Duty framework.

**FCA on proxy discrimination.** Telematics data contains location. Location data can be a proxy for protected characteristics — ethnicity correlates with residential geography at postcode level. The [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) library's proxy discrimination testing applies here: if your telematics-derived features retain location information (either directly or through road-type classifications correlated with postcode), the model should be tested for indirect discrimination. The FCA has been clear that algorithmic pricing must not use proxies that produce discriminatory outcomes even if the proxies themselves are not protected characteristics.

---

## What is actually working

Honest assessment of the UK market, March 2026:

**By Miles** has demonstrated that pay-per-mile with smartphone mileage verification is commercially viable. The Octopus Energy investment in 2023 is strategically significant — it validates the thesis that driving data and energy data are converging for EV customers, and that a pay-per-mile motor product and a smart-EV energy tariff are complementary. By Miles claims materially better loss ratios than the market average for their demographic; whether that is data quality or selection effect is genuinely unclear from public information.

**Marshmallow** has demonstrated smartphone-only UBI at scale with a non-standard segment (new-to-UK, no NCB history). Their loss ratio is reportedly competitive. The specific insight is that behavioural telematics data can substitute for absence of credit/claims history — a market gap that matters as EV ownership grows in segments that historically lack NCB.

**Admiral LittleBox** remains the volume benchmark: the longest-running, largest UK telematics book. Hardware-based, well-validated, loss ratio genuinely better than the non-telematics book. The gap has narrowed in recent years as adverse selection dynamics work through — the easy gains from selecting cautious young drivers into telematics have partially eroded.

What is not working: the integration of connected-car data (OEM API signals, charging behaviour) into the technical rate at any UK insurer, as far as we can determine from public information. Every UK telematics pricing product we are aware of uses either black-box hardware or smartphone behavioural scoring. Nobody is using OEM API odometer data as a continuous exposure variable (though the products exist; Smartcar has built the API). Nobody is using OCPP charging data as a usage proxy. These remain opportunities rather than deployed capabilities.

---

## The integration challenge

Assuming you have access to connected-car data — which is itself non-trivial — getting it into a GLM pricing model involves several practical problems that do not appear in the academic literature.

**Schema normalisation.** The Smartcar API, the Enode API, and a direct OEM integration each return data in different formats, at different latencies, with different field definitions. A "charging session" from Enode has different fields to a "charging transaction" from an OCPP network. Before you can build a feature engineering pipeline, you need an internal schema that normalises these into a single representation. This is plumbing work that takes months, not weeks.

**Missing data — not all trips captured.** Smartphone telematics coverage is not 100% of journeys. Signal loss, phone battery death, app background restrictions (iOS background location permissions changed in iOS 17), and trips taken in a different vehicle all produce gaps. Garage-parked vehicles sometimes show no trips for weeks. A driver feature computed on 60% of their trips has very different statistical properties to one computed on 90%. The pipeline must track coverage fraction per driver and either flag or credibility-weight accordingly. Policies with low coverage fractions should probably be treated as unscored, not scored on a small sample.

**Selection bias in opt-ins.** The UBI adverse selection problem applies at the data acquisition stage, not just the pricing stage. Drivers who consent to telematics are systematically not representative of the full portfolio. Any score model trained on the opted-in population will have population-level bias: the coefficients reflect the risk structure of the consenting group, not the whole book. When you apply those coefficients to a new opt-in cohort, the composition of who opts in this time may differ from who opted in when the model was trained. This is a model stability issue that traditional GLM validation frameworks do not specifically test for; [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) drift detection on the opted-in cohort's feature distribution is the practical check.

**Feature stability over model updates.** Smartphone app versions change. The accelerometer processing pipeline at the SDK provider changes. LexisNexis updates their ADAS classification. Any of these can shift the feature distribution discontinuously, breaking the coefficients estimated on historical data. This is a version-control and model governance problem as much as a statistical one. If you are using a vendor-supplied score, you need contractual commitments from the vendor about backwards compatibility of score distributions; the FCA's SS1/23 model risk requirements extend to material third-party models.

**Getting OEM data into a GLM in practice.** The standard workflow is: OEM API call at policy inception (consent required), store odometer and SoH as policy attributes, update at renewal. OEM data does not flow in real-time into most UK insurers' rating engines — it would require a mid-term endorsement process to react to. For GLM pricing, inception-time and renewal-time snapshots are sufficient; for dynamic pricing (telematics score updating weekly), you need a real-time pipeline that most insurers have not built.

---

## Where this is heading

The structural direction is clear: OEM API data will become easier to access as aggregators (Smartcar, Enode) mature, as more OEMs open their data programmes, and as the LexisNexis Telematics Exchange normalises the market. Charging data will eventually be licensed as a usage proxy — the regulatory framework is there; it just needs commercial agreements. Smartphone telematics will continue to grow because the opt-in friction is lower than any hardware device.

The pricing teams that build the data infrastructure now — normalised schema, credibility-weighted feature pipeline, monitoring for feature drift — will be in a position to absorb OEM API signals incrementally as they become available. The teams that wait for the data landscape to stabilise will still be buying a black-box vendor score in 2028.

The integration challenge is real. The data is not clean, coverage is not complete, the selection bias is not removable. But the alternative — a flat EV surcharge and a binary telematics discount with no visibility into what the data says — is harder to defend to the FCA and produces worse combined ratios than a team willing to do the plumbing.

---

**References**

- Smartcar UBI API documentation (2025)
- Enode EV data platform documentation (2025)
- LexisNexis Risk Solutions, *Telematics Exchange and Vehicle Build* (2024)
- Open Charge Alliance, OCPP 2.1 specification (January 2025)
- EDPB Guidelines 01/2020 on connected vehicles — still materially applicable under UK GDPR
- FCA, PS21/5 — General Insurance Pricing Practices (2021)
- FCA, Consumer Duty final rules (2023)
- Sun, R. and Shi, P. (2024). 'Hidden Markov models for telematics risk scoring', *North American Actuarial Journal* — driving regime segmentation via HMM outperforms naive event counts
- Thatcham/Generational battery SoH data (via EV Blueprint, March 2026)
- Insurance Business UK (2025) — UK insurer privacy concerns stalling telematics adoption

---

**Related**

- [EV Motor Insurance Pricing: Beyond the Flat Surcharge](/2026/03/26/ev-motor-insurance-pricing-beyond-the-flat-surcharge/) — the severity modelling problem this data is meant to solve
- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/13/insurance-telematics/) — the `insurance-telematics` library pipeline in detail
- [UBI Adverse Selection: When Telematics Discounts Drive Away Your Best Risks](/2026/03/25/ubi-adverse-selection-telematics-discounts-drive-away-best-risks/) — the portfolio composition dynamics downstream of opt-in UBI
- [Insurance Model Monitoring Beyond Generic Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — detecting telematics feature drift at renewal
