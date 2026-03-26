---
layout: post
title: "TPBI Multi-State Modelling After the Whiplash Reforms"
date: 2026-03-26
categories: [techniques]
tags: [tpbi, bodily-injury, whiplash-reform, multi-state, claims-lifecycle, oic-portal, civil-liability-act, ogden, motor, reserving, severity, frequency-severity]
description: "The Civil Liability Act 2018 split UK TPBI into two structurally different populations. Standard frequency-severity models treat them as one. Here is why that matters and what a multi-state lifecycle model looks like instead."
---

The UK motor market spent the decade before 2021 in a slow war of attrition with personal injury claims management companies. BI as a share of motor claims spend sat around 16% of total incurred cost (ABI data). Then the Civil Liability Act 2018 came into force in April 2021 and that figure dropped to roughly 9% by 2025. That four-year shift is the most structurally significant thing that has happened to UK TPBI pricing since the Jackson reforms of 2013.

The problem is that aggregate statistics hide what actually happened to the claims distribution. Small soft tissue injuries were capped and rerouted; large and complex injuries continued on a completely different trajectory. A standard frequency-severity model that aggregates across this structural split is now misspecified in a way that was not true before April 2021.

This post sets out what actually changed, why multi-state lifecycle modelling is the right framework for thinking about post-reform TPBI, and what a minimal implementation looks like.

---

## What the reforms actually did

The Civil Liability Act introduced a fixed tariff for whiplash and minor soft tissue injuries: £240 for claims up to three months, rising to £4,215 for injuries of 18 to 24 months. Crucially, it established the Official Injury Claim portal to replace the Claims Portal for RTA personal injury claims valued under £5,000. Claimants can navigate this portal without a solicitor.

The effect on volume was immediate. MoJ statistics recorded 386,000 personal injury claims in the first year post-reform against 650,000-plus in the pre-pandemic peak years. Average sub-£5k claim cost fell by roughly 25% as tariff replaced the previous negotiated settlement process.

So far, so good. But large BI claims — typically those above £25k involving multiple heads of damage, rehabilitation costs, or long-term care — were not meaningfully affected by the tariff structure. They continue to develop under the same cost pressures as before: Ogden discount rate at minus 0.25%, care cost inflation running well above general CPI, and periodical payment orders becoming more common for catastrophic injury. Our [insurance-severity library](https://github.com/burning-cost/insurance-severity) shows the large-loss tail thickening in recent development years rather than compressing.

This creates a bimodal claims landscape. Sub-£5k claims: volume compressed, unit cost capped, settled quickly via OIC portal. Large claims: fewer in number, but average severity increasing at 6-8% annually in real terms on recent experience. The arithmetic means are meaningless; what matters is the pathway a claim follows and when.

---

## Why standard frequency-severity fails here

Conventional GLM-based pricing works with a single frequency component and a single severity component, estimated jointly on historical claims. The implicit assumption is that a claim is a claim — the settlement amount is a draw from a stable severity distribution, independent of how the claim evolved.

That assumption broke down in April 2021. Post-reform, whether a claim ends up in the OIC portal or in litigation is now the primary determinant of its cost. Portal claims settle in roughly 180 to 250 days (MoJ reports 251 days median in Q2 2023, up from 227 pre-reform — the portal is actually slower, not faster, partly because of third-party validation steps). Litigated claims take two to four years and generate solicitor costs, medical report costs, and court fees in addition to general damages.

A frequency-severity model trained on 2019-2021 data has the pathway composition baked in from the pre-reform world. A model trained on 2022-2024 data is cleaner, but still does not distinguish between a £4,000 portal settlement and a £4,000 early-stage litigation settlement — which have completely different reserve development profiles and different probabilities of escalating to six figures.

The duration of a claim in each state matters for pricing in at least two ways. First, ULAE (unallocated loss adjustment expenses) accrue per unit time in active states, not per claim. Second, the transition probability out of litigation into settlement is not memoryless: a claim that has been litigated for 24 months has different settlement prospects than one at six months, even conditional on all observable covariates. Standard severity models have no natural way to express this.

---

## The multi-state framework

The claims lifecycle can be expressed as a continuous-time Markov chain (CTMC) with the following states:

```
Notified ──► OIC Portal ──► Settled (Paid)
    │                           ▲
    ▼                           │
Liability ──► Litigated ────────┤
Disputed                       ▼
                          Settled (Nil) / Withdrawn
```

Formally: a set of transient states {Notified, OICPortal, LiabilityDisputed, Litigated} and absorbing states {SettledPaid, SettledNil}. The generator matrix Q has entries q_ij representing the instantaneous rate of transition from state i to state j. Transition probabilities at horizon t follow from P(t) = expm(Q × t).

This is not novel mathematics. The health insurance actuarial literature has used this framework since at least Haberman and Pitacco's 1999 text on disability models — the illness-death model (healthy, sick, dead) is structurally identical to (notified, litigated, settled). What is novel is applying it to post-reform TPBI where the state space now explicitly includes the portal as a distinct routing option rather than a single undifferentiated settlement pathway.

Each transition intensity can carry covariates. For the Notified → OICPortal vs. Notified → LiabilityDisputed transition, the relevant covariates are vehicle type, claimant representation status at first notice, reported mechanism (rear-end vs. other), and accident date relative to April 2021. For the Litigated → SettledPaid transition, the relevant covariates include time in litigation, claim quantum band, and whether a medical examination has occurred.

A Poisson GLM reformulation (one model per transition, offset by time at risk) is equivalent to the CTMC under piecewise-constant intensity assumptions and is far easier to implement given existing tooling. Use statsmodels Poisson with a log offset for the exposure interval. This gives you covariate-adjusted transition rates that can then feed the matrix exponential calculation when you need occupancy probabilities at arbitrary horizons.

---

## The software gap

If you want the full CTMC machinery — P(t) = expm(Qt), occupancy probabilities, present values of future costs — the honest answer is that no Python library does this cleanly for panel data (periodic observations where the exact transition time is unknown). R's `msm` package (Jackson 2011) handles this correctly and remains the standard tool; it fits the generator matrix via maximum likelihood from panel observations using numerical integration over the matrix exponential.

In Python, the closest we have is: `scipy.linalg.expm` for the matrix exponential computation itself, and a hand-rolled likelihood function. Our `insurance-telematics` library contains a `ContinuousTimeHMM` class that does `expm(Q * dt)` for hidden-state inference; the same core computation applies here with observed states instead of latent ones. We are planning an `insurance-multistate` library that fills this gap properly — estimated generator matrix from panel data, covariate effects on transition intensities, and the standard insurance presets including a TPBILifecycle state space. For now, the practical path for most UK actuarial teams is either R msm or the Poisson GLM approximation.

The `PyMSM` package (Python) requires exact transition times, which you generally do not have in claims panel data. It is the wrong tool.

---

## Data sources

Three sources cover the post-reform picture adequately:

**OIC portal statistics.** The MoJ publishes quarterly Official Injury Claim statistics including claim volumes, stage at which claims resolve, and time to resolution. These are the only publicly available source of portal-specific settlement timing. They do not contain severity by state.

**ABI claims data and industry schemes.** The ABI's motor statistics report provides the BI share figures (16% to 9%) and average claim cost data. Internal claims data with a state flag (portal vs. litigated vs. neither) is necessary for fitting any multi-state model. If your internal data predates April 2021, you will need to reconstruct portal eligibility from claim value bands and accident date.

**Ogden tables (8th edition, 2020).** For large loss severity, the discount rate assumption dominates. The minus 0.25% rate continues to produce substantial multipliers for periodical payments: a 40-year-old with annual care costs of £80,000 has a capital value of roughly £3.5m at minus 0.25% versus £2.4m at 1.5% (the pre-2017 rate). Sensitivity analysis on the Ogden rate belongs inside any large-loss severity model; our `insurance-severity` library's `TruncatedGPD` class is appropriate for modelling the tail above £100k.

For claims inflation on the large-loss book, `insurance-trend`'s `SeverityTrendFitter` with `superimposed_inflation()` separates tariff-effect from general cost-of-living movements — exactly the decomposition needed when comparing 2019 and 2024 BI development.

---

## Monitoring settlement pattern shifts

The MoJ review expected in Spring 2026 may adjust the tariff schedule. Our March 2026 post on [BI claims trajectory under reform uncertainty](/2026/03/25/bi-claims-trajectory-whiplash-reform-uncertainty.html) assigned 45% probability to tariffs uprated with inflation (~17%), 25% to scope expansion, and 30% to partial reversion — the last scenario being the most dangerous for reserve adequacy despite having the second-lowest probability, because its distribution of outcomes is fat-tailed.

A multi-state model gives you a natural monitoring structure: track the empirical transition rates quarter on quarter and test whether the OICPortal → SettledPaid rate has shifted. If the MoJ changes the £5,000 portal threshold to £7,500 (a plausible scenario B outcome), a cohort of claims currently in the litigation pathway would instead route through the portal. The transition matrix shifts, and reserves built on the pre-change Q matrix are immediately stale.

`insurance-monitoring`'s drift detection is the right tool here: fit Q on a base window (2022-2024), then track the Litigated → SettledPaid and OICPortal → SettledPaid transition rates against those baselines. A significant shift in either direction is a trigger for reserve review, not just an annual scheduled update.

---

## The bottom line

Post-reform TPBI is two claims populations, not one. The standard frequency-severity GLM forces them into a single severity distribution, misweights the tail, and has no mechanism for detecting pathway shifts. A multi-state lifecycle model is not exotic actuarial theory — it is the same framework UK income protection actuaries have used for thirty years, applied to a state space that now has an explicit OIC portal branch.

The Poisson GLM approximation is sufficient for covariate estimation. The full CTMC apparatus — R `msm`, or a hand-rolled Python implementation using `scipy.linalg.expm` — is needed when you want occupancy probabilities at specified development horizons or present values of future cost emergence. Given the MoJ review pending this spring, the timing for building that machinery is now, not after the tariff change lands.

---

- MoJ: [Official Injury Claim statistics](https://www.gov.uk/government/collections/official-injury-claim-statistics) — quarterly portal volumes and resolution times
- ABI: [Motor insurance premium tracker and claims data](https://www.abi.org.uk/data-and-research/resources/motor/) — BI share of motor spend
- Jackson C.H. (2011). 'Multi-State Models for Panel Data: The msm Package for R.' *Journal of Statistical Software* 38(8):1–29
- Haberman S., Pitacco E. (1999). *Actuarial Models for Disability Insurance*. Chapman & Hall/CRC — the standard reference for illness-death multi-state models
- Ogden Tables, 8th edition (2020), Government Actuary's Department — Ogden discount rate minus 0.25% as of 2017 review; unchanged since
- Titman A.C., Sharples L.D. (2010). 'Semi-Markov Models with Phase-Type Sojourn Distributions.' *Biometrics* 66(3):742–752 — for duration-dependent transition rates
