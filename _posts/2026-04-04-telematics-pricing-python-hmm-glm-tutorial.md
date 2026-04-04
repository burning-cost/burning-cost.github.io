---
layout: post
title: "Telematics Pricing in Python: From Raw Trip Data to a GLM in 30 Minutes"
date: 2026-04-04
author: Burning Cost
categories: [tutorials, telematics, pricing]
tags: [telematics, python, GLM, HMM, hidden-markov-model, motor, UBI, driving-states, insurance-telematics, poisson, polars, credibility, jiang-shi-2024, uk-motor]
description: "A practical Python tutorial for telematics pricing: load raw GPS trip data, classify driving regimes with a Hidden Markov Model, and produce GLM-ready risk features using insurance-telematics. Covers the full pipeline from 1Hz sensor data to Poisson frequency model."
---

UK motor insurers can now see how their policyholders actually drive. That should transform pricing. In practice, most teams are either buying a black-box score from a device provider and using it as one rating variable they cannot explain, or they are sitting on raw trip data they do not have the tooling to process. Neither is good enough.

This is a practical tutorial for doing it properly in Python. We will go from raw 1Hz GPS/accelerometer data — the format you actually receive from a telematics device or app — to a Poisson GLM with driving-regime features in about 30 minutes of setup time. The library handling the heavy lifting is [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics), MIT-licensed, on PyPI.

```bash
uv add insurance-telematics
# or: pip install insurance-telematics
```

---

## Why raw averages are not enough

The first instinct when handed a telematics dataset is to compute means: average harsh braking rate, average speeding fraction, average night driving proportion. These are not wrong — they correlate with claims — but they miss something important.

A driver who averages 0.07 harsh braking events per kilometre could be consistently moderate, or could be cautious on 90% of trips and genuinely aggressive on the remaining 10%. The average looks the same. The true claim rate does not: risk is a nonlinear function of peak intensity, not average intensity. A small number of highly aggressive trips does most of the damage. Raw averages hide that structure.

Jiang and Shi (2024, *North American Actuarial Journal* 28(4)) formalise the alternative: treat driving style as a latent state that is persistent across trips. Fit a Hidden Markov Model to the full sequence of trip-level observations. The fraction of trips a driver spends in the high-intensity state is more stable across new observations than any single harsh event count, and more predictive of claim frequency in any portfolio where the true DGP is regime-structured — which the driving behaviour literature consistently finds it is.

On a synthetic fleet of 5,000 drivers with 30 trips each and a known three-state DGP, HMM state features improve Gini by 5–10 percentage points over raw averages in a Poisson GLM. The HMM correctly identifies more than half of the genuinely high-risk drivers (top-quartile aggressive fraction), versus 25% at random. These are the [benchmark numbers from the library](https://github.com/burning-cost/insurance-telematics#validated-performance) — see the full validation at `notebooks/databricks_validation.py`.

---

## The pipeline

The library structures the work as five stages:

```
Raw 1Hz trip data (CSV or Parquet)
  → load_trips()             load and schema-map
  → clean_trips()            GPS jump removal, acceleration derivation, road type
  → extract_trip_features()  trip-level scalars: harsh braking, speeding, night fraction
  → DrivingStateHMM          classify each trip into a latent driving regime
  → aggregate_to_driver()    Bühlmann-Straub credibility weighting to driver level
```

The output is a Polars DataFrame: one row per driver, ready to join to your existing rating factors and feed into a GLM. Each stage is callable independently if you want to inspect intermediate results, or wrap the whole thing in `TelematicsScoringPipeline` for a single fit/predict call.

---

## Step 0: no data yet? Use the simulator

If your telematics data is still being negotiated with the device provider, start with `TripSimulator`. It generates a realistic synthetic fleet with three underlying driving regimes, Ornstein-Uhlenbeck speed processes, and Poisson claims. Not representative of any real portfolio — the DGP is explicitly state-structured — but it gets your pipeline running before the real data arrives.

```python
from insurance_telematics import TripSimulator

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=200, trips_per_driver=40)

# trips_df: one row per second
# columns: trip_id, driver_id, timestamp, latitude, longitude,
#          speed_kmh, acceleration_ms2, heading_deg
print(trips_df.shape)  # (200 drivers × 40 trips × ~8 min avg) ≈ 3.8M rows
```

For real data, replace the simulator with `load_trips()`. The expected schema is one row per second with `trip_id`, `timestamp`, `latitude`, `longitude`, `speed_kmh`. Everything else is optional or derivable. If your column names differ, pass a `schema` mapping:

```python
from insurance_telematics import load_trips

trips_df = load_trips(
    "raw_trips.parquet",
    schema={"gps_speed_kph": "speed_kmh", "device_id": "driver_id"}
)
```

---

## Step 1: clean the raw data

Raw GPS data is dirty in predictable ways: coordinate jumps where the device briefly loses signal, missing acceleration (derive it from speed deltas), ambiguous road type. `clean_trips()` handles all three.

```python
from insurance_telematics import clean_trips

trips_clean = clean_trips(trips_df)
# Removes GPS coordinate jumps implying > 200 km/h between consecutive points
# Derives acceleration_ms2 from speed differences where absent
# Assigns road_type from speed profile: urban (< 50 km/h), rural, motorway
```

Speed cap and jump threshold are parameterised if your fleet has atypical characteristics (e.g., a commercial fleet with legitimate high-speed motorway runs that would otherwise be clipped).

---

## Step 2: extract trip-level features

```python
from insurance_telematics import extract_trip_features

features = extract_trip_features(trips_clean)
# One row per trip. Key columns:
#   harsh_braking_rate       events/km (deceleration < -3.5 m/s²)
#   harsh_accel_rate         events/km (acceleration > +3.5 m/s²)
#   harsh_cornering_rate     events/km (from heading-change rate)
#   speeding_fraction        fraction of time exceeding road-type limit
#   night_driving_fraction   fraction of distance driven 23:00-05:00
#   urban_fraction           fraction of time at < 50 km/h
#   mean_speed_kmh, p95_speed_kmh, speed_variation_coeff
#   distance_km, duration_min
```

These nine features are the inputs to the HMM. They are interpretable and inspectable at this stage — you can already run a GLM using just these as raw averages and establish a baseline Gini before you add the HMM step. We recommend doing this: it lets you quantify the HMM lift on your own data rather than relying on benchmark numbers from a synthetic fleet.

---

## Step 3: fit the HMM

This is where the library earns its keep relative to doing the feature engineering manually.

```python
from insurance_telematics import DrivingStateHMM

hmm = DrivingStateHMM(n_states=3, random_state=42)
hmm.fit(features)

# Decode the most probable latent state for each trip
states = hmm.predict_states(features)

# Aggregate to driver level: fraction of trips in each state
driver_hmm = hmm.driver_state_features(features, states)
# One row per driver_id. Columns:
#   state_0_fraction   (cautious — lowest mean speed state)
#   state_1_fraction   (normal)
#   state_2_fraction   (aggressive — highest mean speed state)
#   state_entropy      (Shannon entropy of the state distribution)
#   mean_transition_rate  (state changes per km)
```

**State ordering.** The library normalises states by mean speed: state 0 is always the lowest-speed regime, state 2 the highest. The label "aggressive" is inferred from this ordering post-hoc, not assigned by the model. On most real portfolios the highest-speed state also shows the highest harsh event rates and the highest claim frequency — but check that assumption on your data before labelling anything in a regulatory document.

**State entropy.** A driver with high entropy does not consistently occupy any one regime. They are unpredictable: sometimes cautious, sometimes aggressive, apparently at random. This is a different risk profile from a consistently aggressive driver, and it may warrant a separate pricing treatment. High-entropy drivers are harder to price precisely because their state mix can shift with life changes — new commute route, new job, house move — in ways the trailing observation window may not have captured yet.

**Fit time.** For 200 drivers × 40 trips, the HMM fits in a few seconds. For 5,000 drivers × 30 trips, expect 30–90 seconds on a modern laptop or Databricks serverless. Above 50,000 drivers, batch by cohort.

---

## Step 4: credibility-weighted driver aggregation

Trip-level averages also need to come through at driver level, with credibility weighting for sparse drivers.

```python
from insurance_telematics import aggregate_to_driver

driver_risk = aggregate_to_driver(features, credibility_threshold=30)
# One row per driver_id. Distance-weighted feature means.
# Drivers with < 30 trips are shrunk toward the portfolio mean
# via Bühlmann-Straub: z = n_trips / (n_trips + 30)
```

A driver with 8 trips gets weight 8/38 = 0.21. Their score is 79% portfolio mean, 21% their own observed behaviour. This is not exotic — it is [Bühlmann-Straub credibility](/2026/03/23/does-buhlmann-straub-credibility-work-insurance-pricing/) — implemented in the [insurance-credibility library](/blog/2026/04/04/credibility-theory-python-buhlmann-straub-tutorial/) — applied to telematics, following Gao, Wang and Wüthrich (2021, *Machine Learning* 111). The threshold of 30 is a sensible default; it is parameterised if your trip length distribution is unusually short or long.

---

## Step 5: join to the frequency model

At this point we have two driver-level DataFrames — HMM state fractions and credibility-weighted averages — that we join together and then join to the claims and traditional rating factors.

```python
import polars as pl
import statsmodels.api as sm
import numpy as np

# Merge telematics features
driver_features = driver_risk.join(driver_hmm, on="driver_id", how="left")

# Join to claims and traditional factors
modelling_df = (
    claims_df
    .join(driver_features, on="driver_id", how="left")
)

# Prepare design matrix
# state_2_fraction is the primary telematics covariate
# Include urban_fraction as a control — see the confounding section above
telem_cols = ["state_2_fraction", "state_entropy", "speeding_fraction",
              "night_driving_fraction", "urban_fraction"]
trad_cols  = ["driver_age", "ncd_years", "vehicle_age"]  # your traditional factors

feature_cols = trad_cols + telem_cols

pdf = modelling_df.select(feature_cols + ["claim_count", "exposure_years"]).to_pandas()

X = sm.add_constant(pdf[feature_cols].astype(float))
y = pdf["claim_count"].astype(float)
exposure = pdf["exposure_years"].astype(float)

glm = sm.GLM(
    y,
    X,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(exposure)
).fit()

print(glm.summary())
```

`state_2_fraction` enters the GLM as a continuous covariate — log-linear in the Poisson model. If you want a relativity table for the rating engine, bin it into quintiles or deciles and calibrate. If you want smooth non-linear effects without arbitrary binning, `insurance-gam` handles that directly.

---

## The one-call version

If you want the full clean-to-prediction workflow without assembling stages manually:

```python
from insurance_telematics import TelematicsScoringPipeline

pipe = TelematicsScoringPipeline(n_hmm_states=3, credibility_threshold=30)
pipe.fit(trips_df, claims_df)
predictions = pipe.predict(trips_df)
# predictions: Polars DataFrame with driver_id and predicted_frequency
```

This is useful for prototyping and benchmarking. For production you will want to assemble the stages manually — you need to join telematics features to a broader model including traditional rating factors from your policy system, and `TelematicsScoringPipeline` does not know about those.

---

## The confounding problem you need to control for

There is a structural confound in telematics data that trips up every team encountering it for the first time. Urban drivers produce more harsh events per kilometre regardless of driving quality. More stop-start junctions, tighter corners, more emergency responses to cyclists and pedestrians. A driver with 80% urban mileage will show elevated harsh braking rates that reflect road environment, not intrinsic driving aggressiveness.

The HMM partially addresses this: urban driving has a distinct emission distribution in the feature space (lower speeds, higher lateral acceleration variance), so the model learns urban-specific patterns rather than treating all high-harsh-event trips equivalently. But the confounding attenuates rather than disappears.

The practical response is to include `urban_fraction` as an explicit control variable in the GLM alongside the telematics features. Do not exclude it from the model on the grounds that it is already correlated with state fractions. The correlation is partial; including it removes the portion of state_2_fraction that is really attributable to urban exposure.

If you have road type annotations from HERE or OpenStreetMap, pass them to `load_trips()` — the pipeline can condition the HMM emissions on road type, which improves urban/rural discrimination. Without them, the urban_fraction control is the right fallback.

---

## A note on the FCA and auditability

The FCA's fair value expectations and Consumer Duty requirements mean "the vendor gave us a score" is no longer a satisfying answer when the regulator asks how a pricing variable is constructed. FCA EP25/2 on proxy discrimination is directly relevant here: a telematics score derived from raw GPS data can proxy for protected characteristics — urban-heavy driving correlates with ethnicity; night driving correlates with occupation and sometimes religion.

The `insurance-telematics` pipeline is auditable at every stage. You can show exactly which trip-level features drive the HMM state assignments, and you can run the output features through [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) to check for proxy discrimination before the model goes live. A vendor black-box score gives you neither capability.

This is increasingly the substantive difference between "bought a score" and "built a model": regulatory accountability. The tooling now exists to do the latter without a multi-year data science build.

---

## What to do next

If you are starting from scratch on telematics:

1. **Install and run the simulator.** `TripSimulator` gets you a working pipeline in under an hour. You can benchmark HMM features against raw averages on synthetic data and confirm your tooling is correct before real data arrives.

2. **Load your actual trip data with `load_trips()`.** Use the `schema` parameter to remap column names. Inspect the output of `extract_trip_features()` before fitting the HMM — unusual distributions (e.g., very long mean trip duration, very high harsh event rates) indicate data quality issues that will distort the HMM.

3. **Fit with `n_states=3` first.** This recovers three regimes consistent with the Jiang-Shi (2024) framework. If your fleet has a distinct segment (e.g., a significant commercial/fleet component mixed into a personal lines book), try `n_states=4` and examine the emission distributions.

4. **Check the urban confound.** Run `driver_risk.select(["urban_fraction", "state_2_fraction"]).corr()` — if the correlation is above 0.4, you should include `urban_fraction` as a control in the GLM or investigate whether your HMM is partially recovering an urban/rural split rather than a risk split.

5. **Register the model.** Telematics models sit at Tier 2 or Tier 3 under most model risk frameworks because of the data source dependency. [Insurance-governance](https://github.com/burning-cost/insurance-governance) produces the validation evidence pack for the model register entry.

Library: [github.com/burning-cost/insurance-telematics](https://github.com/burning-cost/insurance-telematics) — MIT-licensed, Python 3.10+, PyPI: [`insurance-telematics`](https://pypi.org/project/insurance-telematics/).

---

**Related:**
- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/13/insurance-telematics/) — the full methodology post covering continuous-time HMMs and relativity calibration
- [Does HMM Telematics Risk Scoring Actually Work?](/2026/03/28/does-hmm-telematics-risk-scoring-actually-work/) — the benchmark: Gini improvement numbers on a 5,000-driver synthetic fleet
- [UBI Adverse Selection: Telematics Discounts Drive Away Your Best Risks](/2026/03/25/ubi-adverse-selection-telematics-discounts-drive-away-best-risks/) — the selection problem that can undermine telematics pricing even when the model is accurate
- [Portfolio-Anchored Telematics Risk Scoring with Wavelets](/2026/03/26/wavelet-telematics-risk-index-lee-badescu-lin-2026/) — the Lee-Badescu-Lin (2026) wavelet-based alternative to HMM state fractions
- [Building a GLM Frequency Model in Python: A Step-by-Step Tutorial](/blog/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — the Poisson frequency GLM into which telematics state features feed as rating factors
