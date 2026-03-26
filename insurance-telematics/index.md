---
layout: page
title: "insurance-telematics"
description: "End-to-end pipeline from raw 1Hz GPS/accelerometer data to GLM-compatible risk scores for UK motor insurance pricing. HMM driving state classification."
permalink: /insurance-telematics/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-telematics"
pypi_package: "insurance-telematics"
---

End-to-end pipeline from raw telematics trip data to GLM-compatible risk scores for UK motor insurance pricing. HMM-based driving state classification, Bühlmann-Straub credibility aggregation to driver level, Poisson GLM integration.

**[View on GitHub](https://github.com/burning-cost/insurance-telematics)** &middot; **[PyPI](https://pypi.org/project/insurance-telematics/)**

---

## The problem

Most telematics scoring tools are either black-box APIs (you get a score, you cannot audit it) or academic scripts that do not run on your data. This library gives you the full pipeline in Python: load 1Hz GPS/accelerometer data, classify driving behaviour using a Hidden Markov Model, aggregate to driver-level risk scores, and produce a feature DataFrame you can drop into your Poisson frequency GLM alongside traditional rating factors.

The academic basis is Jiang & Shi (2024) in NAAJ: HMM latent states capture driving regimes (cautious, normal, aggressive) and the fraction of time in the aggressive state is more predictive of claim frequency than raw speed or harsh event counts alone.

Benchmark: HMM state features produce 3–8 Gini points improvement over raw trip averages on synthetic data with known driving regime DGP.

---

## Installation

```bash
pip install insurance-telematics
```

Dependencies: polars, numpy, scipy, hmmlearn, statsmodels, scikit-learn.

---

## Quick start

Five-line usage with the full pipeline class:

```python
from insurance_telematics import TripSimulator, TelematicsScoringPipeline

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=100, trips_per_driver=50)

pipe = TelematicsScoringPipeline(n_hmm_states=3)
pipe.fit(trips_df, claims_df)
predictions = pipe.predict(trips_df)
```

Step-by-step pipeline on real data:

```python
from insurance_telematics import (
    load_trips,
    clean_trips,
    extract_trip_features,
    DrivingStateHMM,
    aggregate_to_driver,
)

trips_raw = load_trips("trips.csv")
trips_clean = clean_trips(trips_raw)
features = extract_trip_features(trips_clean)

model = DrivingStateHMM(n_states=3)
model.fit(features)
states = model.predict_states(features)
driver_hmm_features = model.driver_state_features(features, states)

driver_risk = aggregate_to_driver(features, credibility_threshold=30)
driver_risk = driver_risk.join(driver_hmm_features, on="driver_id", how="left")
```

---

## Key API reference

### `TripSimulator`

Generates realistic synthetic fleet data for prototyping before your data is available. Three driving regimes (cautious/normal/aggressive), Ornstein-Uhlenbeck speed processes, synthetic Poisson claims.

```python
TripSimulator(seed=42)
```

- `.simulate(n_drivers, trips_per_driver)` — returns `(trips_df, claims_df)` as Polars DataFrames
- `trips_df` columns: `driver_id`, `trip_id`, `timestamp`, `speed_ms`, `accel_ms2`, `lat`, `lon`

### `load_trips`

Load trip data from CSV or Parquet. Validates required columns and returns a Polars DataFrame.

```python
trips = load_trips("trips.csv")          # or .parquet
trips = load_trips(trips_df)             # pass a DataFrame directly
```

Required columns: `driver_id`, `trip_id`, `timestamp`, `speed_ms`.

### `clean_trips`

Remove GPS jumps, derive acceleration and jerk, classify road type (urban/rural/motorway).

```python
trips_clean = clean_trips(trips_raw, max_speed_ms=70.0)
```

### `extract_trip_features`

Extract trip-level features: harsh braking rate, harsh acceleration rate, speeding fraction, night fraction, motorway fraction, mean speed, trip duration.

```python
features = extract_trip_features(trips_clean)
# Returns Polars DataFrame with one row per trip
```

### `DrivingStateHMM`

Hidden Markov Model for driving state classification. Classifies each trip into a latent driving state and produces driver-level state fraction features.

```python
DrivingStateHMM(
    n_states=3,         # number of latent states (3 = cautious/normal/aggressive)
    covariance_type="full",
)
```

- `.fit(features)` — fits Gaussian HMM on trip feature matrix
- `.predict_states(features)` — returns state sequence per trip
- `.driver_state_features(features, states)` — Polars DataFrame: driver_id + fraction in each state

### `ContinuousTimeHMM`

Continuous-time HMM variant for finer temporal resolution. Suitable for second-by-second state assignment.

### `aggregate_to_driver`

Aggregate trip-level features to driver level with Bühlmann-Straub credibility weighting. Drivers with fewer than `credibility_threshold` trips are shrunk toward the fleet mean.

```python
driver_risk = aggregate_to_driver(
    features,
    credibility_threshold=30,   # trips needed for full credibility
)
```

### `TelematicsScoringPipeline`

End-to-end pipeline: trips to GLM predictions in one object.

```python
TelematicsScoringPipeline(
    n_hmm_states=3,
    credibility_threshold=30,
)
```

- `.fit(trips_df, claims_df)` — fits HMM and Poisson GLM jointly
- `.predict(trips_df)` — returns Polars DataFrame with driver_id + predicted_frequency
- `.score_trips(trips_df)` — returns trip-level scores

---

## Related blog posts

- [HMM-Based Telematics Risk Scoring: From Trips to GLM Features](https://burning-cost.github.io/2026/03/13/insurance-telematics/)
- [Telematics Risk Scoring: From Raw Trips to GLM Features](https://burning-cost.github.io/2026/03/12/telematics-risk-scoring-from-trips-to-glm-features/)
