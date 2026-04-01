---
layout: post
title: "Applying Bonus-Malus to Driving Behaviour, Not Just Claims"
date: 2026-04-01
categories: [telematics, techniques]
tags: [telematics, BMS, bonus-malus, UBI, PHYD, motor-pricing, weekly-dynamic, panel-data, Poisson-GLM, credibility, insurance-telematics, python, uk-insurance, young-drivers, aviva-drive, vitality-drive, marmalade, consumer-duty, Yanez, Guillen, Nielsen, ASTIN-2025, HMM, experience-rating]
description: "Yanez, Guillen and Nielsen (ASTIN Bulletin 2025) apply a bounded Bonus-Malus System not to claims but to telematics signals themselves, updating weekly. The result: Gini from 0.082 to 0.182, and a principled credibility-weighted memory of how a driver actually behaves over the year."
math: true
author: burning-cost
---

The standard telematics pricing question is: how do we turn a driver's behaviour this week into a premium for next year? The answer the industry has converged on — a score derived from harsh braking events, speed events, and night driving fraction, averaged or otherwise summarised over the observation period — gives a single number. That number ignores time.

A driver who was aggressive in January and careful from February onwards gets the same score as one who was careful in January and progressively worse. On a static scoring system, both look identical at renewal. They are not the same risk.

Yanez, Guillen and Nielsen's 2025 paper in the ASTIN Bulletin ([DOI: 10.1017/asb.2024.30](https://doi.org/10.1017/asb.2024.30), open access via Bayes Business School) addresses this by applying a Bonus-Malus System — normally reserved for claims history — to telematics signals themselves. The BMS score becomes a credibility-weighted trajectory of the driver's behaviour, updated every week, entering the claim frequency model as a covariate. The improvement in discriminatory power is material. The regulatory and product design implications for the UK are non-trivial.

---

## The concept: BMS as behavioural memory

Classical Bonus-Malus works on claims. You make a claim: you move up a ladder. You do not: you move down. The NCD system in UK personal motor is the familiar version — 0%, 20%, 30%, 40%, 50%, 65% discount levels, each policyholder carrying their place on the ladder from year to year. The BMS acts as a compressed memory of your claims history, but only that.

The Yanez/Guillen/Nielsen paper makes a different move. Apply the same mechanism not to whether a claim happened, but to the telematics signals themselves — distance driven at night, number of speed events, proportion of urban mileage — and update the resulting state weekly rather than annually. The BMS score $B_{it}$ for driver $i$ at week $t$ is a credibility-weighted summary of everything observed about that driver's behaviour up to that point. It then enters the claim frequency model directly:

$$\log \mathbb{E}[N_{it}] = \sum_k \alpha_k x_{ki} + \sum_j \beta_j z_{jit} + \gamma \cdot B_{it}$$

where $x$ are static characteristics (age, gender, vehicle power) and $z$ are current-week telematics covariates. The BMS term adds the behavioural trajectory on top of the contemporaneous signal.

The critical design choice is that the BMS is bounded. There is a finite ladder of levels. This matters because an unbounded score applied to telematics signals — which fluctuate substantially week to week — would create extreme instability: one week of aggressive driving could permanently damage a score built over months of careful behaviour. The bounded BMS applies credibility shrinkage: a driver with a long track record of safe driving absorbs a bad week without catastrophic score degradation. The credibility weight grows with weeks of observation, so the BMS gradually becomes a reliable signal rather than a volatile one.

---

## What the data showed

The paper uses Spanish data from 2018–2019: 19,214 drivers, 790,698 driver-week observations, 922 at-fault accidents, 181.4 million km of distance. The sample is predominantly young and novice — early UBI adopters in Spain, which is an important limitation we return to below.

The Gini index trajectory tells the core story:

| Model | Gini |
|-------|------|
| Demographics only (age, gender, vehicle) | 0.082 |
| Log-distance only (PAYD) | 0.115 |
| Full UBI with contemporaneous telematics | 0.180 |
| Full UBI with lagged signals (Heliyon 2024 best) | 0.182 |

This is the result from the precursor paper (Guillen, Perez-Marin and Nielsen, Heliyon 2024), which established the weekly Poisson GLM foundation with lagged telematics covariates. The ASTIN 2025 paper builds on this, adding the bounded BMS as an additional covariate. The BMS augmentation adds discriminatory power beyond the lagged-variable specification alone.

The step from 0.082 to 0.115 — pure demographics to pay-as-you-drive — confirms that distance driven is itself a powerful predictor. The step from 0.115 to 0.182 — PAYD to full UBI with behaviour signals — shows that how you drive matters independent of how much you drive. Neither finding is surprising. The contribution of the BMS is to show that how you have been driving over time adds information beyond what you are doing this week and what you drove last week.

The weekly premium range under the Poisson multiplicative model runs from EUR 0.015 to EUR 19.013, a ratio exceeding 1,000:1. We come back to this.

---

## How the weekly structure works

The Heliyon 2024 precursor established that lagged telematics variables predict claims beyond contemporaneous measurements. The key empirical finding: log-distance in the previous week carries a coefficient of 0.330 (p<0.001) even after controlling for the current week's distance. Urban driving proportion in the previous week: coefficient 0.013 (p<0.001). Historical behaviour matters beyond what you are doing right now.

The ASTIN 2025 paper extends this temporal structure from one-lag to a BMS-anchored trajectory. Rather than including lagged variables explicitly, the BMS state machine accumulates the full behavioural history with appropriate decay. Each week, the driver's current telematics signals update their BMS level; the updated level feeds into next week's risk assessment.

The two-step evaluation structure is useful for pricing practice. At the start of the week, risk is assessed from the BMS history — a prior estimate before the week's behaviour is observed. At the end of the week, the new signals update the BMS state, which then informs next week's start-of-week assessment. This creates a natural structure for mid-term premium adjustment: the weekly BMS update can trigger a pricing revision if the driver's trajectory deteriorates materially.

Three distributional specifications are compared. The Poisson multiplicative model (log-link, premiums multiply together) produces the widest premium range (EUR 0.015 to EUR 19.013). The linear probability additive model is more conservative (EUR 0 to EUR 7.747). Logistic regression is the widest of all (EUR 0.015 to EUR 21.164). The paper's primary results use Poisson.

---

## The relationship to HMM trip scoring

Our `insurance-telematics` library already implements a continuous-time hidden Markov model for intra-trip scoring, based on Chan, Badescu and Lin (ASTIN Bulletin 55(2), 2025). The HMM classifies individual observations within a trip — detecting latent driving states (steady motorway, stop-start urban, aggressive acceleration) and flagging anomalous deviations from a driver's normal regime.

The BMS framework in Yanez/Guillen/Nielsen operates at a completely different temporal scale. These are not competing approaches.

| | insurance-telematics HMM | Yanez/Guillen/Nielsen BMS |
|---|---|---|
| Temporal scale | Within trip (sub-second) | Weekly (52-period policy year) |
| State meaning | Latent driving regime | Behavioural risk class |
| Transition frequency | Markov at 1Hz | Markov at weekly frequency |
| Observation | Speed, acceleration per timestep | Weekly aggregated signal totals |
| Credibility | Bühlmann-Straub over trip count | Bounded BMS over week count |
| Output | State fraction features per trip | Weekly BMS score covariate |

The natural stacking is: the HMM classifies within-trip behaviour to produce weekly-aggregated features (fraction of time in aggressive state, count of anomalous deviations, maximum pseudo-residual magnitude). These features become the telematics signal inputs $z_{jit}$ to the weekly BMS. The BMS then tracks the evolution of those HMM-derived features over the policy year, providing a credibility-weighted trajectory. Intra-trip precision, inter-week memory.

---

## What the insurance-telematics library currently covers

The library (`src/insurance_telematics/`) implements:

- `hmm_model.py` — CTHMM intra-trip latent state classification (Chan et al. 2025 basis)
- `feature_extractor.py` — trip-level feature engineering (harsh events, night fraction, speed events)
- `scoring_pipeline.py` — Poisson GLM scoring with contemporaneous telematics covariates
- `risk_aggregator.py` — Bühlmann-Straub credibility weighting at driver level for sparse trip counts
- `trip_simulator.py` — synthetic trip generation for testing

The library is trip-centric. The unit of observation is a trip; credibility is accumulated over trips. There is no weekly panel structure, no BMS state machine applied to telematics signals, and no dynamic score that carries forward week-over-week.

The gap the ASTIN 2025 paper points to is clear: a `weekly_bms` module would need a weekly panel aggregator (restructuring trips to driver × week), a BMS state machine with a bounded ladder and transition probabilities calibrated from telematics signal distributions, and an extension to the scoring pipeline that accepts the weekly panel and the BMS state history as inputs. That module design is well-defined. We have not built it yet — the reason is below.

---

## UK application: where it fits and where it doesn't

### The products it maps onto

Aviva Drive, Vitality Drive, esure, Direct Line DrivePlus, and Marshmallow all produce a telematics-derived score that is computed continuously and applied at renewal or mid-term review. The BMS framework is, in essence, a principled formalisation of what these products already do informally.

Currently, those products aggregate behaviour into a score using proprietary rules — weighted combinations of harsh event counts, smoothed over the observation period. The BMS provides a statistically grounded alternative with explicit credibility properties: the scoring mechanism has a clear actuarial basis, the transition rules can be calibrated from portfolio data, and the discriminatory power is documented at Gini 0.182 against a 0.082 baseline.

Marmalade and Young Marmalade use black-box devices with a mid-term adjustment trigger: if the score falls below a threshold, the premium can be revised upward. The weekly BMS provides a clean audit trail for such adjustments — each week's state transition is observable, the credibility weighting is transparent, and the regulatory documentation of why a mid-term adjustment was triggered becomes straightforward.

### The weekly billing problem — and why it mostly doesn't matter

The paper models weekly premiums, and the EUR 0.015 to EUR 19.013 range assumes the policyholder is billed weekly. No UK personal motor product does this. Annual or monthly billing is the norm.

This is less of a constraint than it looks. The weekly BMS scoring process can run continuously without weekly billing. The accumulated BMS score at any point in the policy year is a sufficient statistic for risk assessment. A BMS score computed weekly but applied to pricing only at mid-term review or renewal is entirely consistent with UK product structures. You decouple the scoring cadence from the billing cadence.

### Consumer Duty and the 1,000:1 premium range

The 1,000:1 weekly premium ratio is the most significant regulatory concern. Under FCA Consumer Duty (PS22/9), products must deliver fair value, and a pricing mechanism that can produce premiums 1,000 times higher for the worst drivers than the best creates an obvious fair value question.

The bounded BMS is the partial answer: by constraining how rapidly a score can move in response to a single bad week, it prevents extreme premiums from appearing abruptly. A driver with six months of careful driving cannot be assigned a worst-case premium after a single aggressive week. The credibility weighting creates genuine protection against volatile score trajectories.

But the range itself is still wide. The Poisson multiplicative model is the most aggressive; the linear additive specification narrows it to 0:EUR 7.747 (which creates its own problem — zero premium for the safest drivers). In UK practice, floor and ceiling constraints on premium adjustment factors are common in telematics products, and the regulatory position would require such constraints to be documented in the fair value assessment.

A BMS ladder with named risk levels — "low risk," "moderate risk," "elevated risk" — that maps onto specific premium adjustment bands is more defensible under Consumer Duty than an opaque continuous score. It is also more communicable. A driver asking why their renewal premium increased can be told they spent four weeks at "elevated risk" level during the policy year; the BMS provides the audit trail.

### The young driver selection bias

The Spanish dataset skews heavily towards young and novice drivers — the early UBI adopter cohort. Calibration findings may not generalise to the full distribution of a standard UK motor portfolio where telematics adoption spans a wider age range (Marmalade targets under-30s; Aviva Drive is open to all ages; the full By Miles book is age-agnostic).

This is not a fatal limitation. It means a UK insurer implementing the framework would need to recalibrate the transition probabilities and signal coefficients from their own portfolio data rather than transferring Spanish parameters directly. That is standard practice for any GLM-based pricing model. The framework is portable; the parameter estimates are not.

---

## What we would build, and when

We think this is a blog rather than an immediate build for one reason: the transition matrix calibration for the BMS state machine requires partner data. Defining the BMS levels and their transition rules from telematics signals — as opposed to claims — is the part of the paper that is least fully specified in the open-access material. The paper's exact transition design is not extractable, and synthetic data from our `TripSimulator` would produce calibration that is structurally correct but not empirically grounded.

The build condition is concrete: if a UK insurer comes to us specifically wanting mid-term adjustment pricing tooling with an auditable weekly scoring mechanism, the module design is ready. The `dynamic_bms` module in `insurance-telematics` would need: a `WeeklyPanelAggregator` class that restructures trip-level data to driver × week format, a `BMSStateMachine` with configurable bounded ladder and update rules, and an extension to `scoring_pipeline.py` that accepts a weekly panel with BMS state history. Estimated build effort: three to four days including tests and documentation.

Until that conversation happens, the framework sits at BLOG: worth understanding, clearly the direction UK telematics pricing is heading, but not yet ready to implement without validated transition parameters.

---

## The broader shift this represents

Standard telematics pricing is a-priori with a behavioural modifier. You gather signals over the year, compute a score at renewal, and adjust the premium. The BMS framework is genuinely a-posteriori in the actuarial sense: the score is a credibility-weighted sufficient statistic of accumulated behavioural history, and it updates recursively as new information arrives. It brings the temporal structure of experience rating — which the non-life actuarial profession has decades of practice with on claims data — to the telematics signal domain.

That is a more defensible position than a proprietary score rule. It has an explicit actuarial basis, documented discriminatory power, credibility properties that can be discussed with regulators, and a natural connection to existing NCD practice. The Gini improvement from 0.082 to 0.182 — more than doubling discriminatory power over pure demographics — is large enough to matter commercially.

The question is whether UK telematics products are ready to move from informal scoring rules to formal BMS-based pricing. Aviva Drive and Vitality Drive are the most plausible early adopters. Marmalade is a natural fit for the young-driver focus. The actuarial infrastructure exists. What is missing is the validated transition calibration from UK portfolio data — which only a carrier with a live telematics book can provide.

---

## The paper

Yanez, J.S., Guillen, M. & Nielsen, J.P. (2025). 'Weekly Dynamic Motor Insurance Ratemaking with a Telematics Signals Bonus-Malus Score.' *ASTIN Bulletin: The Journal of the IAA*, 55(1), pp. 1–28. DOI: [10.1017/asb.2024.30](https://doi.org/10.1017/asb.2024.30). Open access: [openaccess.city.ac.uk](https://openaccess.city.ac.uk/id/eprint/33821/). Authors affiliated with Bayes Business School (City, University of London) and University of Barcelona.

The precursor: Guillen, M., Perez-Marin, A.M. & Nielsen, J.P. (2024). 'Weekly dynamic motor insurance pricing with telematics data.' *Heliyon*, 10(5). DOI: [10.1016/j.heliyon.2024.e26636](https://doi.org/10.1016/j.heliyon.2024.e26636).

---

## Related posts

- [Scoring a Trip as a Function, Not a Feature Vector](/telematics/techniques/2026/03/31/telematics-trip-scoring-functional-data-wavelets/) — the CTHMM and wavelet approaches that produce the weekly features the BMS then tracks
- [Cross-Product Claim Scores: Do They Work in the UK?](/techniques/pricing/2026/04/01/cross-product-claim-scores-uk-viability/) — the classical BM-panel framework for claims history, which this paper extends to telematics signals
