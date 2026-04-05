---
layout: post
title: "HMM-Based Telematics Risk Scoring for Insurance Pricing"
date: 2026-03-13
categories: [libraries, pricing, telematics]
tags: [telematics, UBI, HMM, hidden-markov-model, driving-risk, GLM, insurance-telematics, motor, python, GPS, accelerometer, CTHMM]
description: "Continuous-time HMM for telematics risk scoring in UK motor pricing. Latent driving regimes from GPS data - actuarially interpretable features for Poisson GLM."
---

Most telematics scores are event counts. Harsh braking events per kilometre. Hard cornering rate. Night driving percentage. These are cheap to compute, easy to explain, and wrong in a specific way: they treat every kilometre of driving as exchangeable. A harsh braking event on an urban road at 30 km/h is not the same as one at 110 km/h on a motorway. The denominator matters. The driving context matters.

The actuarial literature has known this for a while. Jiang and Shi (North American Actuarial Journal, 2024) showed that model-based approaches substantially outperform naive event counting for claims frequency prediction. Hidden Markov Models on telematics sequences provide better risk discrimination than standard feature engineering, and crucially, the latent states have direct actuarial interpretation: a driver who spends a large fraction of miles in an "aggressive urban" state tends to have materially higher claims frequency than the portfolio average.

[`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) implements this approach for UK pricing teams. Raw trip data in, GLM-ready driver risk features out. 84 tests, 12 modules, MIT-licensed, on PyPI.

```bash
uv add insurance-telematics
```

---

## The model

A standard discrete-time HMM assumes observations arrive at fixed intervals. Trip data does not: GPS pings arrive at 1Hz nominally, but gaps appear when the device loses signal, changes road type, or powers down at rest. A continuous-time HMM (CTHMM) handles this correctly by parameterising the holding time in each state with an exponential distribution, so a 3-second gap and a 30-second gap at a red light get treated differently.

The latent states represent driving regimes. With three states - the default - these typically learn something like:

- **State 0** (low intensity): steady motorway cruising, low lateral acceleration, long holding times
- **State 1** (medium intensity): mixed urban-suburban, moderate event rates
- **State 2** (high intensity): aggressive or distracted urban driving, frequent state transitions, high acceleration variance

The model does not label these states; it learns them from the data. The labels are interpretive post-hoc. What matters for pricing is the per-driver state occupancy distribution: what fraction of each driver's trip-seconds were spent in each regime.

The observation vector at each timestep is: `[speed_kmh, lon_accel, lat_accel, heading_change_rate]`. The CTHMM is fitted via EM on the full corpus of trips, with driver-level state sequences decoded via the Viterbi algorithm.

---

## Pipeline

```python
from insurance_telematics import (
    load_trips,
    clean_trips,
    extract_trip_features,
    DrivingStateHMM,
    aggregate_to_driver,
    TelematicsScoringPipeline,
)

# Step 1: load and clean raw trip CSVs
trips_raw = load_trips("path/to/trips/*.csv")
trips = clean_trips(trips_raw)

# Step 2: extract trip-level features
trip_features = extract_trip_features(trips)

# Step 3: fit HMM on the trip corpus
hmm = DrivingStateHMM(n_states=3, n_iter=100)
hmm.fit(trip_features)

# Step 4: aggregate to driver level
driver_features = aggregate_to_driver(trip_features)
# Columns: aggregated feature means, n_trips, total_km,
#          credibility_weight, composite_risk_score, driver_id

# Step 5: fit end-to-end pipeline (clean → extract → HMM → aggregate → GLM)
# TelematicsScoringPipeline handles all stages internally
pipeline = TelematicsScoringPipeline(n_hmm_states=3)
pipeline.fit(trips_raw, claims_df)          # claims_df: driver_id, n_claims, exposure_years
predictions = pipeline.predict(trips_raw)   # driver-level frequency predictions
```

The driver features from `aggregate_to_driver()` are a standard Polars DataFrame with one row per driver. You merge with your traditional rating factors and fit a Poisson frequency GLM exactly as normal. The telematics features are additive in log-space - they do not require any change to your GLM fitting procedure.

---

## What the features mean

The state occupancy fractions are the primary features. `state_2_frac` - the fraction of driving time in the high-intensity state - is typically the strongest predictor. In practice, moving from the 10th to 90th percentile of this feature typically multiplies claims frequency by a factor of 2–3×, depending on portfolio mix and state definitions.

Two secondary features deserve attention:

**State entropy** measures how unpredictably a driver switches between regimes. A driver with high entropy is not consistently in any one state - they mix aggressive and calm driving erratically. This has a different risk interpretation from a driver who is consistently in the high-intensity state. The consistently aggressive driver is easier to price; the erratic driver has higher variance around the expected claim count.

**Mean transition rate** (the rate at which the driver switches states per unit time) correlates with driving complexity - more transitions suggest more complex, variable road environments, which may or may not be within the driver's control. This feature is most useful when combined with contextual data (road type, time of day).

---

## Calibration to premium relativities

The state features are continuous. To produce multiplicative relativities compatible with a GLM rating engine, bin the state occupancy fractions and include them in a Poisson GLM with monotone constraints. The driver-level features from `aggregate_to_driver()` feed directly into any Poisson GLM via `insurance-gam` or statsmodels, and [`shap-relativities`](/shap-relativities/) can extract the relativity table from the fitted model in the standard format ready for import into Radar or a rating engine.

---

## Confounding: the urban mileage problem

There is a known confounding structure in telematics data that the library forces you to confront. Urban drivers generate more harsh events per kilometre regardless of driving quality - more stop-start, more tight corners, more cyclists and pedestrians requiring sharp responses. A driver with 80% urban mileage will score worse on naive event rates than a rural driver at similar true risk.

The `DrivingStateHMM` partially addresses this by learning road-type-specific emission distributions — the "high-intensity" state has different speed and acceleration signatures in urban versus motorway context. But the confounding does not disappear; it is attenuated. Pass road type annotations (from HERE or OpenStreetMap) as additional columns in the trip DataFrame when available.

If you do not have road type data, the library flags the urban mileage fraction as a recommended control variable. Include it in the GLM alongside the telematics features to avoid attributing urban exposure to driving quality.

---

## UK market context

The UK has roughly 1.3 million telematics policies — the largest market in Europe alongside Italy. Adoption is concentrated in under-30s (Admiral Curve, By Miles, Marmalade, Hastings Direct's YouDrive). The FCA's pressure on age and postcode as rating factors has renewed interest in telematics as a non-discriminatory alternative: rather than pricing a 19-year-old solely on their age, a telematics score provides behavioural evidence that partially decouples rate from demographic proxy — though you should verify the telematics score itself does not proxy for protected characteristics using [insurance-fairness](/2026/03/20/fairness-auditing-without-sensitive-attributes/).

That regulatory trajectory is the main reason to build this capability now, even if your book does not currently write UBI policies. The tools for ingesting and scoring telematics data are infrastructure, not a one-off project.

---

## What it does not do

This library does not implement dynamic bonus-malus updating from telematics (updating the BM score weekly from trip data, as in Boucher et al.). That is a separate problem requiring a real-time scoring pipeline, not a batch GLM feature engineering step. It also does not handle image or video telematics - only GPS and accelerometer time series.

The `DrivingStateHMM` assumes the latent states are stationary across drivers. In practice, a professional delivery driver and a new 17-year-old learner are not drawn from the same latent state distribution. If your book has very heterogeneous driver populations, consider fitting separate HMMs by segment and merging the state features at the GLM stage.

---

**[insurance-telematics](/insurance-telematics/)** — 84 tests, MIT-licensed. [PyPI](https://pypi.org/project/insurance-telematics/).

---

**Related reading:**
- [The Telematics Score That Forgets Where It's Been](/2026/03/12/insurance-jlm/) - joint longitudinal models that use the telematics risk score trajectory as a marker to predict time-to-claim; the downstream model that insurance-telematics feeds into
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/) - time-to-lapse modelling; pairing retention survival analysis with telematics risk segmentation to identify the customers worth retaining
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) - once the HMM regime scores are available as features, `TweedieGBM` can estimate per-risk CoV that varies by driving style profile, not just by static risk characteristics
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) - telematics pricing models sit at Tier 2 or Tier 3 under most model risk frameworks given their dependence on proprietary data sources; `insurance-governance` automates the validation evidence required for the model register entry
- [Does HMM Telematics Risk Scoring Actually Work for Insurance Pricing?](/2026/03/28/does-hmm-telematics-risk-scoring-actually-work/) — benchmark showing 5-10pp Gini improvement from HMM state features versus raw trip averages on a state-structured DGP
- [Portfolio-Anchored Telematics Risk Scoring with Wavelets](/2026/03/26/wavelet-telematics-risk-index-lee-badescu-lin-2026/) — the Lee, Badescu, and Lin (2026) wavelet-based alternative that replaces event counts with principled Bayesian risk indices
