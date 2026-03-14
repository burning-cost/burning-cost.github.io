---
layout: post
title: "HMM-Based Telematics Risk Scoring for Insurance Pricing"
date: 2026-03-13
categories: [libraries, pricing, telematics]
tags: [telematics, UBI, HMM, hidden-markov-model, driving-risk, GLM, insurance-telematics, motor, python, GPS, accelerometer, CTHMM]
description: "Most telematics scores are just event counts: harsh brakes per mile, hard corners per trip. That is not a risk model — it is a heuristic. insurance-telematics uses a continuous-time Hidden Markov Model to identify latent driving regimes from raw GPS and accelerometer data, then extracts actuarially interpretable features that plug directly into a Poisson frequency GLM."
---

Most telematics scores are event counts. Harsh braking events per kilometre. Hard cornering rate. Night driving percentage. These are cheap to compute, easy to explain, and wrong in a specific way: they treat every kilometre of driving as exchangeable. A harsh braking event on an urban road at 30 km/h is not the same as one at 110 km/h on a motorway. The denominator matters. The driving context matters.

The actuarial literature has known this for a while. Gao, Meng, and Wüthrich (ASTIN Bulletin, 2022) showed that model-based approaches substantially outperform naive event counting for claims frequency prediction. Sun and Shi (North American Actuarial Journal, 2024) demonstrated that a Hidden Markov Model on telematics sequences provides better risk discrimination than standard feature engineering, and crucially, that the latent states have direct actuarial interpretation: a driver who spends 40% of miles in the "aggressive urban" state has a claims frequency roughly double the portfolio average in their data.

[`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) implements this approach for UK pricing teams. Raw trip data in, GLM-ready driver risk features out. 84 tests, 12 modules, MIT-licensed, on PyPI.

```bash
uv add insurance-telematics
```

---

## The model

A standard discrete-time HMM assumes observations arrive at fixed intervals. Trip data does not: GPS pings arrive at 1Hz nominally, but gaps appear when the device loses signal, changes road type, or powers down at rest. A continuous-time HMM (CTHMM) handles this correctly by parameterising the holding time in each state with an exponential distribution, so a 3-second gap and a 30-second gap at a red light get treated differently.

The latent states represent driving regimes. With three states — the default — these typically learn something like:

- **State 0** (low intensity): steady motorway cruising, low lateral acceleration, long holding times
- **State 1** (medium intensity): mixed urban-suburban, moderate event rates
- **State 2** (high intensity): aggressive or distracted urban driving, frequent state transitions, high acceleration variance

The model does not label these states; it learns them from the data. The labels are interpretive post-hoc. What matters for pricing is the per-driver state occupancy distribution: what fraction of each driver's trip-seconds were spent in each regime.

The observation vector at each timestep is: `[speed_kmh, lon_accel, lat_accel, heading_change_rate]`. The CTHMM is fitted via EM on the full corpus of trips, with driver-level state sequences decoded via the Viterbi algorithm.

---

## Pipeline

```python
from insurance_telematics import TripPreprocessor, CTHMMFitter, StateFeatureExtractor, TelematicsGLMPipeline

# Step 1: clean raw trip CSVs
preprocessor = TripPreprocessor(
    min_trip_duration_s=60,
    speed_cap_kmh=200,
    impute_gaps=True
)
trips = preprocessor.fit_transform(raw_trips_df)

# Step 2: fit CTHMM on the trip corpus
fitter = CTHMMFitter(n_states=3, n_iter=100)
fitter.fit(trips)

# Step 3: extract per-driver state features
extractor = StateFeatureExtractor(fitter)
driver_features = extractor.transform(trips)
# Columns: state_0_frac, state_1_frac, state_2_frac,
#          mean_transition_rate, state_entropy,
#          total_trip_seconds, n_trips

# Step 4: build GLM-ready feature matrix
pipeline = TelematicsGLMPipeline(extractor)
X_telematics = pipeline.fit_transform(trips)
```

The output `X_telematics` is a standard pandas DataFrame with one row per policy. You merge it with your traditional rating factors and fit a Poisson frequency GLM exactly as normal. The telematics features are additive in log-space — they do not require any change to your GLM fitting procedure.

---

## What the features mean

The state occupancy fractions are the primary features. `state_2_frac` — the fraction of driving time in the high-intensity state — is typically the strongest predictor. In the Sun & Shi (2024) data, moving from the 10th to 90th percentile of this feature multiplied claims frequency by approximately 2.3x.

Two secondary features deserve attention:

**State entropy** measures how unpredictably a driver switches between regimes. A driver with high entropy is not consistently in any one state — they mix aggressive and calm driving erratically. This has a different risk interpretation from a driver who is consistently in the high-intensity state. The consistently aggressive driver is easier to price; the erratic driver has higher variance around the expected claim count.

**Mean transition rate** (the rate at which the driver switches states per unit time) correlates with driving complexity — more transitions suggest more complex, variable road environments, which may or may not be within the driver's control. This feature is most useful when combined with contextual data (road type, time of day).

---

## Calibration to premium relativities

The state features are continuous. To produce multiplicative relativities compatible with a GLM rating engine, you need to bin and calibrate. The library includes a `RelativitiesCalibrator` that:

1. Bins `state_2_frac` into deciles (or custom breakpoints)
2. Fits a monotonically constrained relativities curve via isotonic regression on the GLM coefficient
3. Outputs the relativity table in the same format as `shap-relativities` — ready for import into Radar or a rating engine

```python
from insurance_telematics import RelativitiesCalibrator

calibrator = RelativitiesCalibrator(feature='state_2_frac', n_bins=10, monotone='increasing')
calibrator.fit(X_telematics, claim_counts, exposures)
rel_table = calibrator.relativity_table()
# Returns DataFrame: bin_lower, bin_upper, relativity, credibility_weight
```

The credibility weights use Buhlmann-Straub credibility by default, so thin bins at the extremes of the distribution are blended toward the portfolio mean rather than taken at face value.

---

## Confounding: the urban mileage problem

There is a known confounding structure in telematics data that the library forces you to confront. Urban drivers generate more harsh events per kilometre regardless of driving quality — more stop-start, more tight corners, more cyclists and pedestrians requiring sharp responses. A driver with 80% urban mileage will score worse on naive event rates than a rural driver at similar true risk.

The CTHMM partially addresses this by learning road-type-specific emission distributions — the "high-intensity" state has different speed and acceleration signatures in urban versus motorway context. But the confounding does not disappear; it is attenuated. The `TripPreprocessor` accepts road type annotations (from HERE or OpenStreetMap) and the `CTHMMFitter` supports conditional emissions by road type when that data is available.

If you do not have road type data, the library flags the urban mileage fraction as a recommended control variable. Include it in the GLM alongside the telematics features to avoid attributing urban exposure to driving quality.

---

## UK market context

The UK has roughly 1.3 million telematics policies — the largest market in Europe alongside Italy. Adoption is concentrated in under-30s (Admiral Curve, By Miles, Marmalade, Hastings Direct's YouDrive). The FCA's pressure on age and postcode as rating factors has renewed interest in telematics as a non-discriminatory alternative: rather than pricing a 19-year-old solely on their age, a telematics score provides behavioural evidence that partially decouples rate from demographic proxy.

That regulatory trajectory is the main reason to build this capability now, even if your book does not currently write UBI policies. The tools for ingesting and scoring telematics data are infrastructure, not a one-off project.

---

## What it does not do

This library does not implement dynamic bonus-malus updating from telematics (updating the BM score weekly from trip data, as in Boucher et al.). That is a separate problem requiring a real-time scoring pipeline, not a batch GLM feature engineering step. It also does not handle image or video telematics — only GPS and accelerometer time series.

The CTHMM assumes the latent states are stationary across drivers. In practice, a professional delivery driver and a new 17-year-old learner are not drawn from the same latent state distribution. If your book has very heterogeneous driver populations, consider fitting separate CTHMMs by segment and merging the state features at the GLM stage.

---

**[insurance-telematics on GitHub](https://github.com/burning-cost/insurance-telematics)** — 84 tests, MIT-licensed, PyPI.

---

**Related reading:**
- [The Telematics Score That Forgets Where It's Been](/2026/03/12/insurance-jlm/) — joint longitudinal models that use the telematics risk score trajectory as a marker to predict time-to-claim; the downstream model that insurance-telematics feeds into
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/) — time-to-lapse modelling; pairing retention survival analysis with telematics risk segmentation to identify the customers worth retaining
- [Your NCD System Throws Away 95% of the Information in a Policyholder's Claim History](/2026/03/13/insurance-vine-longitudinal/) — longitudinal vine copulas for claim history; the alternative to NCD for capturing full claim history dependence across renewals
