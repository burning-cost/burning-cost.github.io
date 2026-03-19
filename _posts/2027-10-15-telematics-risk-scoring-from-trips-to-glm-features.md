---
layout: post
title: "Telematics Risk Scoring: From Raw Trips to GLM Features"
date: 2027-10-15
categories: [pricing, libraries, tutorials]
tags: [telematics, hmm, hidden-markov-model, driving-states, GLM, poisson, risk-scoring, insurance-telematics, uk-motor, python, polars, catboost, jiang-shi-2024, buehlmann-straub]
description: "Most pricing teams either buy pre-scored telematics data from a box provider or ignore it entirely. insurance-telematics gives you the full pipeline in Python: raw 1Hz GPS/accelerometer trips to HMM-derived driving state features you can drop into your frequency GLM."
---

Most pricing teams sit in one of two camps on telematics. Either they buy a pre-scored variable from a box provider - a single "telematics score" that appears in the rating engine as a black-box input - or they do not use telematics at all because getting the data into a usable form is too painful. Neither is satisfying. In the first case you cannot audit the score, cannot explain it to underwriters, and cannot update it without going back to the supplier. In the second, you are leaving real risk information on the table.

The gap is tooling. Processing raw 1Hz GPS/accelerometer data into actuarially useful features is not conceptually hard, but the plumbing - GPS jump removal, kinematics derivation, state classification, driver-level aggregation with credibility weighting - is enough friction that most teams give up and reach for the vendor API. [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) removes that friction.

```bash
pip install insurance-telematics
```

---

## Why HMMs rather than raw feature averages

The naive approach to telematics scoring is to compute per-driver averages of behavioural signals: mean harsh braking rate, mean speeding fraction, mean night driving fraction. These are not useless. They are correlated with claims. The problem is that they are noisy at the driver level, especially for drivers with few trips. A driver who has ten recorded trips, three of which were through central London at rush hour, will show an elevated harsh braking rate that reflects road conditions more than their intrinsic driving style.

The HMM approach, formalised by Jiang and Shi in their 2024 NAAJ paper, treats driving style as a latent state that persists across trips rather than a trip-by-trip measurement. With three states - cautious, normal, aggressive - the model asks: what fraction of this driver's trips are consistent with the aggressive regime? That fraction is more stable across new observations than any individual harsh event count, because it integrates over the full sequence.

Concretely, the claim frequency model benefits because state fractions capture persistent driving style while averaging out situational noise. The benchmark in this library (300 synthetic drivers, 40 trips each, Poisson DGP) shows a 3-8 percentage point improvement in Gini coefficient compared to a GLM using raw feature averages instead. The HMM advantage is largest when the true data generating process is regime-structured - which is what the driving literature consistently finds for human behaviour.

---

## The pipeline: five stages

The library structures the work as five sequential steps. You can run them individually or wrap them in `TelematicsScoringPipeline` for a one-call fit.

```
Raw 1Hz trip data
  → clean_trips()          GPS jump removal, kinematics derivation, road type
  → extract_trip_features()   trip-level scalar features
  → DrivingStateHMM        latent state classification
  → aggregate_to_driver()  Bühlmann-Straub credibility weighting to driver level
  → Poisson GLM            predicted claim frequency
```

The input format is one row per second with `trip_id`, `timestamp`, `latitude`, `longitude`, `speed_kmh`, and optionally `driver_id`, `acceleration_ms2`, and `heading_deg`. Acceleration is derived from speed if absent. Column names are remappable via a `schema` dict if your data uses different names.

---

## Step 1: load and clean

```python
from insurance_telematics import TripSimulator, load_trips, clean_trips

# Generate a synthetic fleet for prototyping (use load_trips() for real data)
sim = TripSimulator(seed=42)
trips_raw, claims_df = sim.simulate(n_drivers=200, trips_per_driver=40)

# Clean: removes GPS coordinate jumps (>200 km/h implied speed between points),
# derives acceleration_ms2 from speed differences, classifies road type from speed
trips_clean = clean_trips(trips_raw)
```

`TripSimulator` generates a realistic synthetic fleet with three underlying regimes, Ornstein-Uhlenbeck speed processes, and Poisson claims. It is not representative of any real portfolio - the DGP is explicitly state-structured, which is the best case for the HMM - but it gets you to a working pipeline before your actual data is available.

---

## Step 2: extract trip-level features

```python
from insurance_telematics import extract_trip_features

features = extract_trip_features(trips_clean)
# Returns one row per trip with:
#   harsh_braking_rate    (events/km, decel < -3.5 m/s²)
#   harsh_accel_rate      (events/km, accel > +3.5 m/s²)
#   harsh_cornering_rate  (events/km, estimated from heading-change rate)
#   speeding_fraction     (fraction of time exceeding road-type limit)
#   night_driving_fraction (fraction of distance driven 23:00-05:00)
#   urban_fraction        (fraction of distance at < 50 km/h)
#   mean_speed_kmh, p95_speed_kmh, speed_variation_coeff
#   distance_km, duration_min
```

These are the inputs to the HMM. You can inspect them directly - they are interpretable, Polars DataFrames, one row per trip. The thresholds (3.5 m/s² for harsh events, 50 km/h for urban classification) are sensible defaults from the telematics literature but are parameterised if your fleet has different characteristics.

---

## Step 3: fit the HMM and extract driver state features

This is the step where the library earns its keep.

```python
from insurance_telematics import DrivingStateHMM

hmm = DrivingStateHMM(n_states=3, random_state=42)
hmm.fit(features)

# Assign each trip to a latent state (0=cautious, 1=normal, 2=aggressive)
states = hmm.predict_states(features)

# Aggregate to driver level: what fraction of trips was each driver in each state?
driver_hmm = hmm.driver_state_features(features, states)
# Returns one row per driver_id with:
#   state_0_fraction, state_1_fraction, state_2_fraction
#   mean_transition_rate (state changes per km)
#   state_entropy (Shannon entropy of state distribution)
```

State ordering is normalised on mean speed: state 0 is always the lowest-speed (most cautious) state, state 2 the highest-speed (most aggressive). This makes `state_2_fraction` the key GLM covariate - it is the fraction of trips assigned to the aggressive regime. Following Jiang and Shi (2024), this single feature typically dominates the frequency model in terms of coefficient magnitude and statistical significance.

`state_entropy` is a useful diagnostic. A driver with high entropy has genuinely variable behaviour across states - sometimes cautious, sometimes aggressive. A driver with low entropy consistently occupies one regime. High-entropy drivers are harder to price accurately because their state mix may shift with life circumstances (new job, house move) in ways the trailing observation window does not capture.

---

## Step 4: driver-level aggregation with credibility weighting

The trip-level feature averages also need to come through at driver level - they are useful GLM covariates alongside the HMM state fractions.

```python
from insurance_telematics import aggregate_to_driver

driver_risk = aggregate_to_driver(features, credibility_threshold=30)
# Returns one row per driver_id with distance-weighted feature means
# and a composite_risk_score (0-100)
# Drivers with < 30 trips are shrunk toward the portfolio mean
# via Bühlmann-Straub weights: z = n / (n + 30)
```

The `credibility_threshold` of 30 means a driver with 10 trips gets weight 10/40 = 0.25 - their score is 75% portfolio mean, 25% their own observed behaviour. A driver with 60 trips gets weight 60/90 = 0.67. This is not exotic - it is Bühlmann-Straub credibility applied to the telematics domain as described by Gao, Wang and Wüthrich (2021). The composite risk score is a diagnostic summary; use the individual features as GLM covariates.

---

## Step 5: join to the frequency model

At this point we have a driver-level DataFrame with HMM state fractions and credibility-weighted feature averages. We join it to the traditional rating factors and fit a frequency GLM.

```python
import polars as pl
from catboost import CatBoostRegressor

# Join HMM features to driver-level risk aggregation
driver_features = driver_risk.join(driver_hmm, on="driver_id", how="left")

# Join to claims and traditional rating factors
modelling_df = claims_df.join(driver_features, on="driver_id", how="left")

# Prepare for CatBoost - traditional factors plus telematics features
telem_cols = ["state_2_fraction", "state_entropy", "harsh_braking_rate",
              "speeding_fraction", "night_driving_fraction"]
trad_cols  = ["driver_age", "vehicle_age", "ncd_years"]
feature_cols = trad_cols + telem_cols

X = modelling_df.select(feature_cols).to_pandas()
y = modelling_df["claim_count"].to_numpy().astype(float)
exposure = modelling_df["exposure_years"].to_numpy()

model = CatBoostRegressor(
    iterations=400,
    learning_rate=0.04,
    depth=5,
    loss_function="Poisson",
    verbose=0,
    random_seed=42,
)
model.fit(X, y / exposure, sample_weight=exposure)
```

`state_2_fraction` enters the model as a continuous feature - no need to discretise it into bands before fitting. CatBoost handles the non-linearity. If you are using a GLM rather than a GBM, bin it into quintiles or use a spline in statsmodels; the library does not force you toward either approach.

The key point is that `state_2_fraction` and `state_entropy` are GLM-compatible features in exactly the same sense as `driver_age` or `ncd_years`. They are one number per driver, derived from the observation window. They can go into whatever model structure your team is already using.

---

## Using the one-call pipeline

If you want the full clean-to-GLM workflow without assembling the stages manually, `TelematicsScoringPipeline` does it in one fit/predict pair:

```python
from insurance_telematics import TelematicsScoringPipeline

pipe = TelematicsScoringPipeline(n_hmm_states=3, credibility_threshold=30)
pipe.fit(trips_raw, claims_df)
predictions = pipe.predict(trips_raw)
# predictions: Polars DataFrame with driver_id and predicted_frequency
```

The pipeline fits the HMM on the training trips, aggregates to driver level, and fits a Poisson GLM with all aggregated features as covariates. `predict()` runs the same clean-extract-aggregate-score chain on new trips. This is convenient for benchmarking and prototyping, less so for production where you will want to join telematics features to a broader model that includes traditional factors from your policy system.

---

## What the benchmark shows - and what it does not

On the synthetic fleet in the library benchmarks, HMM state features give a consistent 3-8pp Gini improvement over raw feature averages in a Poisson GLM. The top-to-bottom quintile loss ratio ratio is wider with HMM features: the model does a better job of putting the high-risk drivers in the high-risk deciles.

The honest caveat: the synthetic DGP is deliberately state-structured, which is the best case for an HMM. On a real portfolio where driving behaviour is genuinely continuous rather than regime-based, the gain may be smaller. And a 5pp Gini gain on frequency does not necessarily translate linearly into premium adequacy improvement - it depends on your current rating structure, how much of the frequency risk is already captured by traditional factors, and what correlation exists between driving behaviour and vehicle type (which is already rated).

We think the right use of this library is not as a silver bullet for frequency modelling but as the piece of infrastructure that lets you run that empirical test on your own data, with your own trips, with a transparent and auditable methodology. The vendor score gives you a number. This gives you a pipeline.

---

`insurance-telematics` is open source under MIT at [github.com/burning-cost/insurance-telematics](https://github.com/burning-cost/insurance-telematics). Requires Python 3.10+, Polars, hmmlearn, and SciPy.

- [When You Can't Fit a GLM from Scratch: Transfer Learning for Thin Segments](/2027/07/15/transfer-learning-for-thin-segments/) - credibility and borrowing strength for new or small books of business
- [Your Model Validation Is a Checklist, Not a Test](/2027/09/15/model-validation-pra-ss123/) - once the telematics model is built, running the SS1/23 validation suite
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/) - uncertainty at the policy level, complementary to point-estimate frequency models
