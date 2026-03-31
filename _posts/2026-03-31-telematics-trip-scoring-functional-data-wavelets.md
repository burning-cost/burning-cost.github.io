---
layout: post
title: "Scoring a Trip as a Function, Not a Feature Vector"
date: 2026-03-31
categories: [telematics, techniques]
tags: [telematics, UBI, PHYD, functional-data-analysis, wavelets, MODWT, CTHMM, hidden-markov-model, insurance-pricing, insurance-telematics, python, pywt, scikit-fda, uk-insurance, young-drivers, by-miles, marmalade, cambridge-mobile-telematics, badescu, astin]
description: "Standard telematics pricing throws away most of the information in a trip by reducing it to 100+ scalar features. Two new papers from Toronto's Badescu group show what you recover when you treat the trip as a function: CTHMM anomaly detection at AUC 0.86, and MODWT wavelet scoring that captures behaviour across timescales from one second to a minute."
---

A telematics black-box records acceleration several times per second for the entire duration of every trip. For a two-hour motorway run, that is tens of thousands of measurements. The standard industry pipeline turns all of it into roughly 100 numbers — harsh braking count, night driving percentage, cornering g-force maximum — and throws the rest away.

We think this is the wrong framing. Not because the features are useless (they demonstrably predict claims), but because they encode a specific, largely arbitrary set of assumptions about what matters. The choice of 0.3g as the "harsh braking" threshold is not derived from data. The harsh braking count for a trip treats an isolated emergency stop the same as a hundred mild ones above threshold, provided the count matches. And drowsy driving — one of the genuine high-risk states — produces almost no threshold-crossing events at all. The standard scoring apparatus will call a drowsy driver safe.

Two papers from Andrei Badescu's group at Toronto are pushing in a different direction. Both treat the trip as a continuous function of time and apply statistical methods that operate on that function rather than summaries of it. The first (Chan, Badescu, Lin, *ASTIN Bulletin* 55(2), 2025, arXiv:2412.08106) uses a continuous-time hidden Markov model to detect when a driver deviates from their own normal. The second (Lee, Badescu, Lin, March 2026, arXiv:2603.15839, presented at the ASTIN Bulletin Conference in Zurich in January 2026) decomposes the acceleration signal across timescales using wavelet analysis and constructs a Bayesian risk index from multi-scale tail events.

These are not the same method. They are complementary, and they highlight different structural problems with standard feature engineering.

---

## Why summary statistics lose information

The standard pipeline produces a feature vector by binning observations and aggregating across the whole trip. This is reasonable as a first approximation, but it has six specific failure modes that functional approaches address.

**Threshold arbitrariness.** What counts as "harsh braking"? Different insurers use different values — 0.3g at CMT DriveWell, 0.4g elsewhere, different conventions for different road types. The threshold is not derived from claims data. CTHMM learns what constitutes "normal" for each driver from their observed acceleration distributions; MODWT wavelet decomposition avoids a discrete threshold entirely, working with continuous coefficient magnitudes.

**No intra-driver context.** A 0.4g deceleration on a motorway (emergency stop at 100 mph) and on a residential street (routine pulling up at lights) get identical treatment. CTHMM normalises every observation to the driver's own predicted distribution: it asks "how unexpected is this, given how this particular driver usually drives?"

**Temporal structure discarded.** A harsh braking feature count is the same whether 20 events were scattered across 20 trips or clustered in one bad 10-minute sequence. The wavelet approach via the maximum aggregation rule C_{i,t} = max|W̃_{i,j,t}| retains exactly when in the trip the worst events occurred, and at what timescale.

**Multi-scale patterns are invisible.** A single harsh event lasts under a second. A sustained aggressive driving episode — repeated rapid acceleration and braking in a 30-second urban sequence — is a qualitatively different risk signal. It appears at higher wavelet levels but cannot be captured by any per-event threshold.

**Drowsy driving scores as safe.** This is the most important failure. A drowsy driver rarely exceeds any acceleration threshold. Their score for harsh events: zero. Their actual risk: materially elevated. CTHMM detects drowsiness through extended sojourn in a low-variation latent state — the driver stays in the same state for too long. Wavelet decomposition produces uniformly low coefficients, which is also detectable as anomalous (though the wavelet risk index alone conflates "genuinely safe" with "inattentive" — more on this below).

**Channel correlation discarded.** Simultaneous hard braking and steering (an emergency manoeuvre) is more informative than either signal alone. Feature pipelines typically treat speed, longitudinal acceleration, and lateral acceleration independently.

---

## CTHMM: anomaly detection without threshold engineering

Chan, Badescu, and Lin model the raw telematics time-series as a continuous-time hidden Markov chain with S discrete latent states. Each state captures a distinct driving regime — think "steady motorway," "stop-start urban," "aggressive acceleration." The emission distributions for speed, longitudinal acceleration, and lateral acceleration are modelled jointly within each state.

The key mechanism for irregular GPS data is the transition probability matrix P(t) = exp(Qt), where Q is the instantaneous transition rate matrix. Standard discrete-time HMMs assume fixed observation intervals; GPS devices frequently log only when the reading changes significantly, producing gaps of 1–30+ seconds. The CTHMM handles this exactly rather than by approximate interpolation.

The anomaly index is built from pseudo-residuals. For each observation at time t, dimension d (speed, longitudinal acceleration, lateral acceleration), compute:

u_{ijt,d} = P(Y_{ijt,d} ≤ y_{ijt,d} | full prior history)

the conditional CDF of the current observation given everything observed before it. Under a well-specified model, Φ⁻¹(u) ~ N(0,1). Outliers are where |Φ⁻¹(u)| ≥ 3. The trip-level anomaly index is the proportion of such outliers, normalised to either the driver's own history (individual-specific) or the portfolio distribution (pooled).

Tested on a matched Australian rental car fleet (January 2019 – January 2023), the results are:

| Metric | CTHMM | Industry benchmark |
|--------|-------|-------------------|
| Trip-level AUC, accident classification | **0.86** | 0.65 |
| Driver-level AUC, at-fault claims | **0.78** | 0.65 |

The 0.65 AUC represents a published benchmark for telematics-only scoring using standard threshold features. The 0.13 improvement at driver level is meaningful — comparable to the lift other papers observe from adding a full year of claims history. No threshold engineering is required at any step.

Our `insurance-telematics` library already implements the CTHMM in `src/insurance_telematics/hmm_model.py`.

---

## MODWT wavelets: scoring across all timescales simultaneously

The wavelet approach from Lee, Badescu, and Lin (2026) starts from a different question: what if we decomposed the acceleration signal into components at different timescales and scored each trip on the distribution of extreme events across all those scales?

The transform used is the Maximal Overlap Discrete Wavelet Transform (MODWT) with a Daubechies D4 filter (in pywt notation: `'db2'`). The MODWT is non-decimating — output length equals input length at every level — and translation-invariant. A hard braking event at minute 2 of a trip and the same event at minute 20 receive identical treatment. Standard DWT downsamples at each level and destroys this property.

Six decomposition levels are used, selected because they capture 87.8% of signal variance. What each level captures depends on the sampling rate. For a 1Hz OBD/GPS device (typical UK black-box):

| Level | Timescale | Driving behaviour |
|-------|-----------|-------------------|
| 1 | 1–2s | Individual harsh braking or acceleration event |
| 2 | 2–4s | Short aggressive sequence |
| 3 | 4–8s | Sustained harsh manoeuvre |
| 4 | 8–16s | Multi-manoeuvre pattern |
| 5 | 16–32s | Behavioural episode (aggressive urban navigation) |
| 6 | 32–64s | Extended persistent behaviour |
| Scaling | >64s | Trip baseline (motorway vs urban context) |

The wavelet coefficients W̃_{i,j,t} represent the acceleration signal's content at level j and time t. The key innovation is maximum aggregation:

C_{i,t} = max_{j} |W̃_{i,j,t}|

This collapses the six-level array to a single scalar per time-point: the strongest wavelet response at any timescale at that moment. Prior wavelet-in-telematics work used level-wise energy (variance summed over time at each level), which discards temporal localisation — you know there was a harsh event somewhere in the trip but not when. The maximum rule retains both magnitude and timing.

The distribution of C_{i,t} values across the portfolio is modelled as a Gaussian-Uniform mixture. Two Gaussian components capture typical driving (means ≈ ±0.016g, sds ≈ 0.009g, each contributing roughly 48% of portfolio mass). Nine Uniform layers — four on the left tail, five on the right — model increasingly rare and severe events, with monotone-decreasing layer probabilities (outermost layer: ~0.07% of observations).

Each trip's tail events are counted by layer: N_{im} = count of time-points where C_{i,t} falls in layer m. These counts are modelled as Poisson with exposure E_i (the number of retained wavelet coefficients after thinning to approximate independence). A Gamma prior on each layer's rate, with empirical Bayes hyperparameters estimated from the portfolio, gives a closed-form posterior:

λ_{im} | data ~ Gamma(α_{0m} + N_{im}, β_{0m} + E_i)

The trip risk index is the exposure-weighted posterior mean of layer intensities, with severity weights that give disproportionate weight to rarer layers (γ = 1.7 in the paper's specification — the rarest left-tail layer receives nearly half the total score weight despite contributing under 0.1% of observations).

**The sequential update is practically important.** For driver-level scoring, each new trip adds to α_{dm} and β_{dm} without recomputing from scratch. A new driver starts at the portfolio mean (k=0 gives the prior) and accumulates evidence trip by trip. The cold-start problem is solved by construction.

---

## The registration problem: why classical FDA doesn't work here

Both CTHMM and MODWT fall under the broad heading of functional data analysis — treating the trip as an observation from an infinite-dimensional function space rather than a finite feature vector. But the classical FDA route (Ramsay and Silverman's B-spline basis expansion followed by functional PCA) runs into a structural problem before you get to any interesting statistics.

FPCA requires all trips to live on a common time domain. A 7-minute urban errand and a 90-minute motorway run need to be aligned. Options are: stretch both to [0,1] (which changes rates — 30 km/h means something different in a 7-minute trip vs a 90-minute one); landmark registration (align to specific events — but driving trips have no universal landmarks unlike, say, gait analysis with heel-strike); or PACE (Yao, Müller, Wang, *JASA* 100:577–590, 2005), which estimates FPC scores from pooled cross-driver covariance without individual smoothing.

The MODWT approach sidesteps the problem entirely. Translation invariance means event timing within a trip is irrelevant. Variable trip length is handled by the exposure term E_i in the Poisson model — a longer trip simply has more wavelet coefficients. No alignment, no warping, no preprocessing beyond getting the signal onto a regular time grid.

For UK deployments this matters practically: black-box OBD devices (Octo, VisionTrack) typically log GPS at 1Hz but may drop observations when the delta is below threshold, producing irregular intervals. CTHMM handles irregular intervals by design (P(t) = exp(Qt) for any gap t). MODWT needs a regular grid — irregular data needs interpolation or CTHMM-style extension.

---

## CTHMM vs MODWT: what each captures

These methods are not in competition; they measure different things.

CTHMM measures how much a driver deviates from their own normal. It is individualised and context-relative. A driver who is consistently moderate will flag a single aggressive trip; a consistently aggressive driver will not flag the same trip. This is the right framing for most UK young-driver policies where the goal is behaviour change as much as risk selection.

MODWT measures how extreme the absolute coefficient magnitudes are relative to the portfolio distribution. It is population-relative and severity-weighted. It captures multi-scale events without reference to the driver's history. It is a better fit for fleet-level risk ranking where you want to know who is worst in absolute terms, not who has changed.

Drowsiness is where they diverge most clearly. CTHMM detects drowsiness through extended sojourn in a low-variation latent state — the driver persists in a monotone state for longer than normal. The wavelet index assigns a low C_{i,t} throughout the trip, producing a low risk index. Low index is ambiguous: it could mean a genuinely smooth, safe driver, or it could mean a dangerously inattentive one. Used alone, the wavelet approach cannot tell the difference. The research brief flags this as an open problem.

A combined scoring approach — CTHMM anomaly index for driver-relative deviation detection, MODWT-MLTC for portfolio-relative severity — is the most defensible position. Both as inputs to a downstream GLM is the most natural implementation.

---

## Where UK operators stand

By Miles prices on mileage only — no acceleration or braking scoring at all. Young Marmalade uses CMT DriveWell, which is a threshold-based PHYD system validated by Pinnacle Consulting in the US. Direct Line DrivePlus uses Octo Telematics hardware with a similar threshold-feature pipeline. Marshmallow and Ticker are proprietary but follow the same general approach.

No UK operator, to our knowledge, is using CTHMM or wavelet-based functional scoring in production. The published AUC benchmark CMT's DriveWell claims (22× ratio between best and worst 10% of drivers) is a different metric to AUC, but the gap between the 0.65 industry telematics AUC and 0.78 from the CTHMM on real claims data represents a genuine opportunity. Even a fraction of that improvement translates to materially better risk selection on a book where the pricing of young drivers is the dominant challenge.

The implementation barrier is not the mathematics. The MODWT decomposition in pywt is four lines of Python. The Poisson-Gamma update is closed-form. The engineering work is the pipeline: GPS preprocessing, outlier filtering, stop removal, map-matching. Those steps are the same regardless of whether the downstream scoring uses threshold features or wavelets.

The FCA Consumer Duty constraint is real. A driver asking why their renewal premium increased cannot be told "your pseudo-residual anomaly index exceeded 0.14 over the quarter." Both CTHMM and wavelet scoring need an explainability layer that maps back to interpretable events — "hard braking on the M6 on 14 March" — before they can be used in a consumer-facing UK pricing workflow. This is solvable engineering, not a theoretical obstacle.

---

## What to read

- Chan, Badescu, Lin (2025). "Assessing Driving Risk Through Unsupervised Detection of Anomalies in Telematics Time Series Data." *ASTIN Bulletin* 55(2): 205–241. arXiv:2412.08106.
- Lee, Badescu, Lin (2026). "A Portfolio-Anchored Frequency-Severity Risk Index for Trip and Driver Assessment Using Telematics Signals." arXiv:2603.15839. Presented as Badescu, "Risk scoring algorithm on functional telematics data based on wavelet transform," ASTIN Bulletin Conference, ETH Zurich, January 2026.
- The `insurance-telematics` library ([GitHub](https://github.com/burning-cost/insurance-telematics)) implements the CTHMM from Chan et al. Wavelet scoring is the natural next extension.
