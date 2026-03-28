---
layout: post
title: "EV Motor Insurance Pricing: Beyond the Flat Surcharge"
date: 2026-03-26
categories: [pricing, techniques, research]
tags: [ev, electric-vehicles, motor-insurance, severity-modelling, battery, total-loss, thatcham, vrr, credibility, thin-data, bimodal-severity, uk-motor, glm, gbm]
description: "Why the standard flat EV surcharge is wrong in two directions simultaneously, what the claims data actually shows, and how to build a severity model that handles the bimodal structure correctly."
---

BEVs are 3.8% of the UK car parc right now. Your book's EV exposure is probably in that range  -  perhaps slightly higher if you write new-car or premium motor  -  and you are almost certainly pricing it with a flat surcharge somewhere between 25% and 40% above the equivalent ICE. That surcharge exists because the ABI data says EV premiums are higher, the Motor Insurance Taskforce confirmed it in December 2025, and your own combined ratio on the EV segment looks elevated. So far so defensible.

The problem is that the flat surcharge is wrong in two directions simultaneously. It overcharges a new Tesla Model 3 with a home charger and a Gen 3 ADAS stack. It undercharges a five-year-old Nissan Leaf parked on the street with a degraded battery and no lane assist. These two vehicles are not the same risk. Treating them as interchangeable  -  which is what a flat EV loading does  -  is not conservatism. It is cross-subsidy, and the first team that prices them differently will cherry-pick the Tesla and leave you with the Leaf.

This post is about how to get the modelling right. We cover what the data actually shows, why your current GLM is structurally misspecified on EV severity, what model architecture fixes it, and which rating factors you need that you are almost certainly not rating on yet.

---

## What the data says

The Finnish study published in *Traffic Injury Prevention* (doi:10.1080/15389588.2026.2612718) is the most useful frequency data point available to us: 76 billion km of driving across 2019-2023, BEVs showing a -15% lower at-fault injury crash rate than ICE equivalents. This is a large enough dataset that the result is credible, not noise. The implication is straightforward: if you are loading EV frequency above ICE, you are probably wrong. The EV loading is a severity play, not a frequency play. The Motor Insurance Taskforce Final Report (December 2025) confirmed this independently.

For severity, the best public data is Mitchell's *Plugged-In 2024 Year in Review*, which covers US and Canada claims. UK-specific insurer severity data exists but is not publicly available. The Mitchell numbers: BEV repairable severity is roughly 23% above ICE equivalents; BEV total loss rate is 10.2% versus approximately 7% for ICE. The total loss rate gap is the critical figure. It represents a 46% relative increase in write-off probability.

Repair severity has been falling 3% year-on-year as independent repairers gain experience with EV technology. That trend is directionally encouraging, but the total loss gap shows no sign of closing  -  because it is driven by battery economics, not by repairer capability.

The write-off dynamic is worth understanding precisely. Battery packs represent up to 40% of vehicle value. Replacement costs run £8,000-£15,000. Under the standard 50-60% actual cash value threshold for write-off, a battery that requires any non-trivial work becomes a write-off decision almost automatically  -  particularly given that most UK repairers lack the HV diagnostic capability to assess whether a pack needs full replacement or just module-level repair. Motor Insurance Taskforce committed in December 2025 to enabling independent repairer access to battery diagnostic data. If enacted and adopted, this changes the total loss calculus materially. But that is a 2027-2028 story, not today's.

---

## The structural problem with your current model

Standard severity modelling for motor uses a Gamma GLM. This is appropriate for ICE claims, where severity is roughly unimodal  -  a continuous distribution of repair costs with a long right tail for serious structural damage.

EV severity is not unimodal. It is bimodal:

- **Mode 1**: Minor damage with no battery involvement. Cosmetic, panel, glass, ADAS recalibration. Severity is comparable to or below ICE equivalents. The EV powertrain has fewer moving parts than an ICE; routine mechanical repairs are often simpler.
- **Mode 2**: Battery-involving damage. This is not a continuous extension of Mode 1  -  it is a step change into total loss territory. The battery pack, the HV wiring, the structural protection around the floor-mounted pack. A rear underride that dents the battery casing is not a £2,500 repair. It is a write-off decision.

If you fit a single Gamma on pooled EV claims, the model has to find a single distribution that covers both modes. It will overestimate minor claim severity  -  pulling the distribution right to accommodate the total loss mass  -  and systematically underestimate the probability of a total loss event. The resulting expected severity per claim is in the right neighbourhood (which is why your combined ratio doesn't look obviously wrong) but the model is misspecified, and it will rate incorrectly on the risk factors that drive the split between the two modes.

---

## The model architecture that works

The fix is a two-component structure. This is not unusual  -  hurdle models for frequency-severity are standard  -  but the specific decomposition for EV severity requires thinking carefully about what drives the branch point.

**Step 1: Separate battery-involving claims.** The cleanest flag is whether the battery or high-voltage system was assessed during the claim. If your claims system captures repair type, this flag is feasible to extract. If not, use the total loss flag as a proxy  -  it is imperfect (some battery claims are repairable) but good enough to start. Claims-handler notes are another source; keyword extraction on 'battery', 'HV', 'high voltage', 'write-off' gives a reasonable binary label.

**Step 2: Fit sub-models separately.** For non-battery claims, use a standard Gamma GLM with the same rating factors as ICE  -  parts, labour, ADAS calibration. For battery/total-loss claims, fit a logistic regression for total-loss propensity, and then conditional on total loss, compute expected loss as market value multiplied by (1 minus salvage rate). Market value is observable at policy inception. BEV salvage rates are lower than ICE due to battery storage safety requirements and limited secondary market for used packs  -  factor this in explicitly.

**Step 3: Blend for expected severity.**

> E[severity] = P(battery involvement) × E[TL loss | TL] + P(no battery) × E[repair loss | repair]

P(battery involvement) is itself a function of: vehicle age (as a SoH proxy), impact geometry where available from FNOL, and speed band if telematics data is present. These are richer signals than you will currently have for most EV policies, but the structure is right and you can improve the inputs over time.

**Step 4: Credibility-weight the EV parameters.** A mid-size UK insurer with 2% EV exposure has roughly 5,000 BEV policy years and 200-400 BEV claims. That is not enough to estimate EV-specific GLM parameters reliably in isolation. Use Bühlmann-Straub credibility with ICE parameters as the prior  -  at 200 claims, Z will be in the range 0.3-0.5, meaning substantial shrinkage toward the ICE prior is appropriate. The Thatcham VRR Damageability and Repairability scores provide structured external priors for severity by vehicle model: use them. They are built on 1,300 data points from 25,000 vehicle derivatives.

The data gap is real and we should say so plainly. UK insurer EV claims data is thin, development is limited (median BEV age is 1-2 years), and the fleet composition is shifting  -  82% of new BEV registrations are company-owned, which suppresses private-use frequency figures. Any parameter estimates should carry wide uncertainty intervals, and the credibility weighting is doing real work here rather than being a formality.

---

## Rating factors you are not rating on

The flat surcharge at vehicle type level is leaving money on the table because it ignores the factors that actually drive the bimodal split. Five factors warrant immediate investigation.

**Thatcham VRR sub-scores.** The Vehicle Risk Rating system launched in September 2024, replacing the 25-year-old group rating system over an 18-month dual-rating period. VRR scores vehicles 1-99 on five dimensions: Performance, Damageability, Repairability, Safety, and Security. The Damageability and Repairability sub-scores are directly relevant to EV severity modelling. Damageability measures how much structural damage results from a given impact; Repairability measures how accessible and repairable the damage is. For EVs, Repairability in particular captures some of the battery-adjacent structural complexity that drives write-off rates. These scores are VIN-level, available through Thatcham Research. If you are not rating on them today, you should be.

**Battery chemistry.** NMC (nickel manganese cobalt) and LFP (lithium iron phosphate) chemistries have different thermal runaway risk profiles and different replacement cost profiles. LFP is lower thermal runaway risk; NMC has higher energy density and higher replacement cost. This is not routinely available at quote  -  it requires VIN decode against OEM specification  -  but the data is accessible through vehicle data suppliers. The actuarial impact of getting this wrong is a mispriced severity distribution for a material subset of the fleet.

**Charger access type.** Whether a driver has a home charge point or relies on street/public charging matters for two reasons. First, it is a proxy for battery degradation rate  -  rapid DC charging (>150kW) accelerates SoH decline, and vehicles without home charging use rapid chargers more frequently. Second, it correlates with parking location, which is a material driver of material damage claims. This signal is not directly observable at quote, but it is inferable: property type from Land Registry data gives a reasonable proxy for home parking availability, and postcode-level charging infrastructure coverage is public data. Some insurers are beginning to ask directly; we expect this to become a standard question within two years.

**ADAS generation.** LexisNexis Vehicle Build classifies ADAS capability across four European markets including the UK, covering 2.5 million vehicles. Generation 3 systems (2024 vintage) have integrated sensor fusion  -  more sophisticated and more expensive to calibrate. Mitchell data shows BEV ADAS calibration events average 1.61 per repair versus 1.45 for ICE. The net effect on combined ratio depends on whether the frequency benefit of Gen 3 ADAS offsets the calibration cost per repair. We do not yet have clean UK data to answer this definitively; what we can say is that treating all ADAS generations as equivalent is wrong, and the LexisNexis data makes it possible to do better.

**Verified mileage.** EVs are disproportionately used as second cars for short-trip urban use. Self-reported declared mileage is unreliable for any vehicle; for EVs, where the actual usage pattern differs substantially from ICE norms, the problem is worse. OEM API access via providers such as Smartcar or Enode gives continuous odometer readings for 100+ EV models. For a telematics-based or usage-based product, this is the cleanest exposure variable available.

---

## The cross-subsidy opportunity

The market is broadly applying a 30-40% surcharge at vehicle type level (Motor Insurance Taskforce, December 2025). This is defensible as an aggregate position. It is not defensible at the segment level.

A new Tesla Model 3  -  SoH around 99%, Gen 3 ADAS, home charge point, high-income driver, company-fleet-adjacent ownership profile  -  is almost certainly being overloaded at +35%. Its total-loss probability is low (healthy battery, good structural protection, Thatcham Repairability score reflecting reasonable repair access), its frequency is below ICE (Finnish data, selection effects), and its ADAS stack provides measurable frequency benefit.

A five-year-old Nissan Leaf  -  SoH around 88%, no ADAS, street parking in an urban postcode, private ownership  -  is probably being underloaded at +35%. The degraded battery increases total-loss probability. The absence of a home charge point suggests public rapid-charging use, which degrades SoH further. The lack of ADAS means no frequency offset.

These are not fringe cases. They represent a meaningful fraction of the current BEV parc. The insurer that builds VRR Repairability score, battery SoH proxy, and charger type into their rating structure will price the Tesla competitively (winning the good risk) and price the Leaf appropriately (avoiding adverse selection on the bad risk). Everyone using the flat surcharge will be left with an increasingly adverse BEV mix as the fleet ages.

Thatcham's EV Blueprint (March 2026) sets out eight engineering requirements targeting unnecessary write-offs  -  resettable safety loops, modular battery design, standardised diagnostics. These are not mandated yet, but they signal where OEM design is heading. A vehicle built to the Blueprint specification has a materially different total-loss probability than one that is not. That difference should be reflected in pricing within the next rating generation.

---

## What to do now

Given the credibility constraints, we would not recommend rebuilding the EV pricing model from scratch on internal data alone. We would recommend three things in the near term.

First, instrument the data collection. Ensure your claims system is capturing battery involvement and HV system assessment as structured fields, not free text. If you do not have clean battery/non-battery claim splits, the bimodal model architecture above cannot be fitted. This is a claims operations question as much as an actuarial one.

Second, get Thatcham VRR data into your rating engine. The dual-rating period runs until mid-2026. VRR sub-scores are available now and they are actuarially grounded in insurer claims data. Damageability and Repairability are the priority for EV severity. There is no good reason not to rate on these.

Third, use the bimodal structure even if the parameters are blunt. Even a simple total-loss propensity flag  -  vehicle age over four years, street parking, no ADAS  -  combined with a severity uplift for battery-involving claims is better than the single Gamma. You will not be able to estimate the parameters precisely at current data volumes. Bühlmann-Straub credibility against the ICE prior is the right framework; accept that Z will be low and that you are mostly pricing off the prior for now. That is honest, and it is better than pretending the Gamma fit is adequate.

---

**Sources**

- ABI Premium Tracker (2024)  -  aggregate EV premium relativities
- Mitchell, *Plugged-In 2024 Year in Review*  -  US/Canada BEV claims data
- Motor Insurance Taskforce, *Final Report* (December 2025)  -  government position on EV repair access
- Thatcham Research, Vehicle Risk Rating (September 2024)  -  VRR methodology and sub-scores
- Thatcham Research, *EV Blueprint* (March 2026)  -  engineering requirements for repairable EVs
- Finnish study (2026). 'Electric vehicle safety: at-fault injury crash rates across 76 billion km', *Traffic Injury Prevention*. doi:10.1080/15389588.2026.2612718

---

- [Does Bühlmann-Straub Credibility Actually Work?](/2026/04/01/does-buhlmann-straub-credibility-actually-work/)  -  credibility weighting mechanics for thin segments
- [Spatial Panel GBMs: A Better Way to Price Geography](/2026/03/26/spatial-panel-gbm-geographic-insurance-pricing/)  -  when and how to handle geographic structure in GBM models
