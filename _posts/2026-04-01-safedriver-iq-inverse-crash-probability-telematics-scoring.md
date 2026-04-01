---
layout: post
title: "SafeDriver-IQ: Telematics Scoring via Inverse Crash Probability"
date: 2026-04-01
categories: [telematics, pricing]
tags: [telematics, UBI, motor, inverse-probability, calibration, shap, random-forest, nhtsa, waymo, safedriver-iq, scoring, fca-consumer-duty, uk-insurance]
description: "Roy, Singh, and Das (arXiv:2603.14841) build a 0-100 driver safety score by inverting a crash classifier and multiplying in condition-specific penalty factors. The maths is clean, the performance is honest about its limitations, and the data requirements explain why this is not going live on UK motor books any time soon."
---

Most telematics scoring systems have a dirty secret: the score is not grounded in crash probability. It is a composite of counts and fractions — harsh braking rate, night driving percentage, cornering g-force — aggregated with weights that someone chose, validated against loss ratios on a historical book, and then frozen. The connection to actual claim frequency is indirect and unmeasured.

Roy, Singh, and Das (arXiv:2603.14841, March 2026) propose something more explicit. Start with a binary classifier that predicts crash probability. Invert it: the score for a driver or a trip is the model's estimated probability of *not* crashing, multiplied by 100. If the model says P(crash) = 0.08, the safety score is 92. This is SafeDriver-IQ.

The mechanics are straightforward. The philosophy is important. It means the score has a direct probabilistic interpretation at every point, not just a relative ranking. It means the score degrades in a structured way when conditions worsen — not through arbitrary weights but through calibration factors that are themselves estimated from data. And it means you can tie the score to an expected claim rate, which is what a pricing actuary actually needs.

---

## The core formula

The framework has two components. First, a base score derived directly from the classifier:

```
S_raw(x) = P(y = 0 | x) × 100
```

where `y = 0` is the safe outcome and `x` is the feature vector describing conditions at the moment of driving. A Random Forest classifier trained on 46,388 balanced samples (half crash records, half synthetic safe baselines) produces the posterior probability estimates. On test data: ROC-AUC 0.833, test accuracy 75.8%, train-test accuracy gap of 0.8 percentage points.

Second, a calibration layer. The raw score is then adjusted multiplicatively by condition-specific penalty factors:

```
S_cal(x) = S_raw(x) × ∏(k=1 to K) α_k(x)
```

Each `α_k(x)` is a value in (0, 1] representing a condition-specific multiplier: 0.60 for ice roads, 0.70 for snow, 0.75 for dark unlit roads, 0.88 for VRU presence. When conditions are benign, all multipliers are 1.0 and S_cal = S_raw. When multiple adverse conditions co-occur, the penalties compound multiplicatively.

This multiplicative structure is not cosmetic. The ablation study shows that removing lighting and environmental features together reduces ROC-AUC by 16.4 percentage points, versus 14.1 if you remove them independently and add the effects. The 2.3pp synergistic effect is precisely what the multiplicative calibration is designed to capture.

---

## Why multiplication and not addition

The conventional approach to composite scoring adds weighted features: harsh braking × 0.3 + night fraction × 0.2 + speed × 0.5. The choice of additive combination is rarely justified. It implies that the contribution of each feature is independent of the others.

The evidence here says otherwise. Night driving alone corresponds to a 2.1× baseline crash rate. Poor lighting alone is 2.3×. Night plus poor lighting together (which is a more concentrated subset) corresponds to 3.8× — not 4.4× as additive independence would predict, but substantially above either factor alone. VRU presence plus urban plus night together reaches 4.5×.

The multiplicative penalty structure S_cal = S_raw × α₁(x) × α₂(x) × ... is a direct encoding of this interaction. It is not additive independence and it is not arbitrary weighting: each α_k is estimated from crash rates in the stratified data and the product rule naturally compresses the extremes (because all α_k ≤ 1.0, their product cannot exceed any single factor, which avoids implausibly extreme scores for moderately bad conditions).

---

## The data: what the paper uses

The training set fuses two sources.

**NHTSA CRSS (Crash Report Sampling System):** 417,335 crash-level records from 2016 to 2023. These are actual US crash reports, standardised across police reporting jurisdictions. For VRU modelling the paper filters to 23,194 pedestrian and cyclist-involved crashes.

**Waymo Open Motion Dataset:** 500 naturalistic driving scenarios from Waymo's AV fleet — 455 real and 45 synthetically augmented. Stratified as 92.8% safe driving, 5.4% near-misses, 1.8% collisions. The Waymo data provides the "safe" class for external validation.

The safe training samples in the main classifier are synthetic rather than observed. The paper constructs them by taking crash records and modifying conditions systematically: 80% of poor lighting shifted to non-poor, 70% of night driving shifted to daytime, 90% of adverse weather shifted to clear, 85% of wet or icy roads shifted to dry. This is a reasonable procedure for balancing a heavily skewed dataset, but it introduces a bias: the safe-class training samples are counterfactually modified crash situations, not observed safe driving. There are 64 features across seven groups (temporal, environmental, location, VRU-specific, interaction terms, crash and vehicle characteristics, and metadata).

External validation on the Waymo scenarios is the most reassuring result: mean crash probability for safe episodes is 7.2%, near-misses 54.0%, collisions 92.0%. Monotonic ordering with that degree of separation, on data the model never saw and collected under a completely different methodology, is credible validation.

---

## Score distribution and risk stratification

The paper evaluates the scoring system across an 864-scenario factorial grid covering all combinations of conditions:

| Score range | Risk level | Comment |
|-------------|-----------|---------|
| 76–100 | Excellent | Baseline conditions, daytime, clear, no VRU |
| 61–75 | Low | Minor degradation — slight adverse weather or dusk |
| 41–60 | Medium | Night driving, or urban with pedestrians |
| 21–40 | High | Multiple co-occurring adverse factors |
| 0–20 | Critical | Ice + dark + VRU, or equivalent compound risk |

Across the 864 scenarios: mean 58.73, median 61.42, standard deviation 22.16, range 8.34 to 92.17. The distribution is platykurtic (kurtosis −0.89) — broadly spread rather than concentrated near a central value, which is useful: a score with most mass in the middle conveys little discriminating information.

The sensitivity analysis reveals the ordering of risk factors by effect size. Moving from daylight to dark unlit conditions reduces the score by 28.86 points (1.30 standard deviations). Daytime to night is −26.55 (1.20 SD). Low speed to high speed is −25.07 (1.13 SD). Snow reduces by −18.57 (0.84 SD). These are large, coherent effects — the kind of signal that a per-trip score needs if it is going to inform real-time ADAS feedback or, eventually, UBI pricing.

---

## Comparison to HMM-based telematics scoring

Our `insurance-telematics` library implements a 3-state Gaussian HMM approach: fit latent driving states (cautious, moderate, aggressive) from trip feature sequences, then use the per-driver fraction of time in the aggressive state as a GLM predictor. On a synthetic UK motor portfolio with a known data-generating process, the HMM approach delivers 5–10 percentage point Gini improvement over raw feature averages.

SafeDriver-IQ is doing something structurally different, and the comparison is instructive.

The HMM approach is **endogenous**: it learns risk from observed driving kinematics, using trip sequences as its input. It requires no external crash database. The signal it extracts — latent state fractions — is estimated entirely from the telematics data itself, with no labelled crash outcome for most drivers. Its weakness is that it requires enough trip history per driver to estimate state fractions reliably; Bühlmann-Straub credibility weighting helps, but cold-start drivers remain poorly characterised.

SafeDriver-IQ is **exogenous**: it calibrates against an external crash database and treats driving conditions as fixed inputs. Its feature vector is conditions-at-the-time-of-driving, not kinematics. It does not require trip history — a single set of condition readings produces a score immediately. But it requires the crash database to be representative of the scoring population.

For UBI pricing, you want both: an HMM-style signal capturing how this driver behaves behind the wheel, and a condition-adjustment layer capturing when they drive. SafeDriver-IQ's multiplicative calibration structure maps directly onto how you would implement that adjustment: a trip score is a product of a driver-level behavioural signal and a set of condition factors estimated from external data.

The wavelet approach from Lee, Badescu, and Lin (arXiv:2603.15839) is closer in spirit — it constructs a Bayesian trip risk index from the raw acceleration signal and anchors it to portfolio-level distributions. That paper and SafeDriver-IQ are complementary rather than competing: kinematics-based scoring (wavelet or HMM) tells you about driver behaviour; condition-adjusted scoring tells you about driving context.

---

## UK applicability: what the data gap means

The short answer: SafeDriver-IQ as implemented cannot be replicated on UK data today.

NHTSA CRSS is a US crash registry with standardised fields across all 50 states — police-reported crashes, sampled at roughly 6–8% nationally, with consistent field definitions since 2016. The UK has STATS19 (road casualties Great Britain), which is published by the Department for Transport and provides police-reported accident data back to 1979. STATS19 and NHTSA CRSS are not directly comparable, but STATS19 is an adequate UK substitute in terms of crash-level observations with environmental, lighting, road, and location characteristics.

What STATS19 does not provide is what SafeDriver-IQ's training most depends on: a "safe driving" comparison class. The paper constructs safe samples synthetically from crash records, which is a workaround for the absence of a naturalistic driving baseline. For UK application you would need either:

1. A naturalistic driving study with matched crash and non-crash observations under comparable conditions. The UK Naturalistic Driving Study (UK-NDS) and ADAS&ME datasets exist but are not publicly available at scale.
2. A fleet telematics dataset with matched crash records, large enough to construct a meaningful binary classifier. This means an insurer or fleet operator sharing telematics with labelled claims outcomes — achievable under commercial arrangements, but not off-the-shelf.

Waymo's Open Motion Dataset is US-centric. There is no UK equivalent at comparable scale. Deepmind's WayveScenes101 dataset (2024) covers some UK driving footage, but it is not annotated for crash probability modelling in the same way.

The FCA Consumer Duty adds a regulatory dimension. Pricing factors used in motor insurance must be explainable and demonstrably non-discriminatory (ICOBS 4.2, the fair value framework). SafeDriver-IQ's condition-adjusted score uses lighting, weather, time of day, and VRU presence — none of which are protected characteristics, and all of which are straightforwardly linked to crash frequency. The multiplicative penalty structure is more explainable than most proprietary telematics composites: you can tell a customer exactly which conditions reduced their score and by how much, because the α_k factors are explicit.

The SHAP decomposition (Equation 3 in the paper) reinforces this. The top five features by mean |SHAP|: POOR_LIGHTING (0.159), ADVERSE_WEATHER (0.115), IS_NIGHT (0.072), NIGHT_AND_DARK interaction (0.023), immediate lighting condition (0.022). Every entry is a condition feature, not a behavioural feature. A Consumer Duty fair value assessment would have an easy time justifying these factors.

---

## A conceptual Python sketch

To make the scoring approach concrete, here is how you would implement the core formula in Python, stripped of everything that requires the actual trained model:

```python
import numpy as np

# Condition-specific penalty factors from the paper (Table 3)
CONDITION_PENALTIES = {
    "ice_road": 0.60,
    "snow": 0.70,
    "dark_unlit": 0.75,
    "adverse_weather": 0.80,
    "vru_present": 0.88,
    "night": 0.90,
    "dawn_dusk": 0.95,
}

def compute_calibrated_score(
    crash_prob: float,
    active_conditions: list[str],
) -> float:
    """
    S_cal(x) = P(y=0|x) * 100 * prod(alpha_k(x))

    crash_prob: P(crash | conditions) from the fitted classifier
    active_conditions: list of condition keys from CONDITION_PENALTIES
    """
    s_raw = (1.0 - crash_prob) * 100.0
    multiplier = 1.0
    for condition in active_conditions:
        multiplier *= CONDITION_PENALTIES.get(condition, 1.0)
    return s_raw * multiplier


# Example: night driving on ice, no VRU
conditions = ["night", "ice_road"]
crash_prob = 0.18  # hypothetical model output
score = compute_calibrated_score(crash_prob, conditions)
# S_raw = 82.0
# Multiplier = 0.90 * 0.60 = 0.54
# S_cal = 82.0 * 0.54 = 44.3 (Medium risk band)
print(f"Calibrated score: {score:.1f}")
```

The production version replaces the hardcoded `crash_prob` with a Random Forest fitted to NHTSA CRSS (or STATS19 + synthetic safe samples), vectorises across trip observations, and applies SHAP for per-prediction explanation. The structure above captures the essential logic: the score is a direct transformation of posterior crash probability, not a composite of arbitrary feature weights.

---

## Why we are not building it yet

The BUILD phase for this item is blocked on data availability, and the reason is specific. We can implement the framework. We can adapt it to STATS19 structure. We cannot, without access to a UK naturalistic driving dataset at meaningful scale, construct the "safe class" training samples that SafeDriver-IQ's classifier depends on.

The synthetic safe-sample construction in the paper is a reasonable workaround for a US academic study, but it has a known limitation: it produces a safe class that is counterfactually modified crash data, not observed safe driving. For a classifier that will feed into a pricing score, that matters. The model may be learning to distinguish "crash record" from "crash record with conditions changed" rather than "crash" from "genuine safe trip." The Waymo validation mitigates this concern but does not eliminate it.

The options for UK implementation are: (a) engage with a UK insurer or fleet operator with matched telematics and claims data, (b) acquire access to UK-NDS or equivalent naturalistic study data, or (c) wait for a UK equivalent of the Waymo dataset to emerge at scale. None of those are achievable unilaterally.

What we can do now is implement the scoring architecture, validate the multiplicative calibration structure against synthetic data with a known DGP, and have the code ready for the point at which data becomes available. That is the current plan.

---

## What the paper gets right

The Waymo external validation is the most convincing result. Training on NHTSA CRSS and validating on Waymo (different country, different collection methodology, naturalistic rather than police-reported) with clean monotonic ordering across safe/near-miss/collision categories is stronger evidence of generalisation than any held-out test split from the same distribution. AUC 0.833 on a test set that was sampled from the same data-generating process as the training set tells you relatively little. Mean crash probability of 7.2% for Waymo safe episodes, 54.0% for near-misses, and 92.0% for collisions tells you the model has learnt something real.

The SHAP decomposition is also worth taking seriously. The most important features are lighting and weather, not crash characteristics like collision manoeuvre or impairment type. This matters for real-time use: the high-importance features are observable before a crash happens. A model whose predictions depend mainly on post-crash report fields would be useless for real-time scoring. The feature engineering here has avoided that trap.

The driver clustering result (KMeans, k=4, on 213,003 records) produces a finding with direct pricing implications: 29.6% of crash-involved drivers show zero measured aggression. Their crashes are environmentally caused. A kinematics-only telematics score would assign these drivers a low-risk classification on the basis of smooth driving behaviour — and miss the actual exposure entirely. This is the clearest statement in the paper of why condition-aware scoring is structurally necessary, not optional.

---

What SafeDriver-IQ adds to the telematics scoring literature is a principled probabilistic grounding that most existing approaches lack. The score has an interpretation at every point — not just a rank order, but an estimated probability. The condition penalties are explicit and estimable rather than expert-chosen. The architecture separates behavioural risk from contextual risk in a way that is compatible with how pricing actuaries want to think about the problem.

The data gap is real. UK STATS19 covers the crash side. The safe-class problem is what blocks production use, and solving it requires either naturalistic driving data at scale or a fleet dataset with matched telematics and claims outcomes. Neither is available off the shelf. Until that changes, this sits in the research library: well understood, architecturally sound, waiting for the data to make it buildable.
