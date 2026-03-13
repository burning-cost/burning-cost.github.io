---
layout: post
title: "HMM-Based Telematics Risk Scoring for Insurance Pricing"
date: 2026-03-10
categories: [libraries, pricing, telematics]
tags: [telematics, HMM, hidden-markov-model, UBI, usage-based-insurance, GPS, accelerometer, trip-simulator, driving-state, credibility, GLM, poisson, motor, python, insurance-telematics]
description: "Most UK insurers receive a black-box telematics score from vendors like Octo or The Floow. They cannot inspect it, tune it, or connect it to their rating methodology. insurance-telematics provides an end-to-end Python pipeline from raw 1Hz GPS/accelerometer data to GLM-compatible risk features — including a continuous-time HMM, Bühlmann-Straub credibility aggregation, and a trip simulator that removes the data access barrier entirely."
---

Most telematics pricing in the UK is outsourced. The insurer installs a black box or deploys an app, the data goes to Octo Telematics, Cambridge Mobile Telematics (CMT), or The Floow, and a pre-computed driver score comes back via API. The pricing team uses the score as a single rating factor. They cannot inspect the algorithm, challenge the feature weights, tune the cut-off thresholds, or connect the score to their own claims experience in any methodologically principled way. The score is a number. They trust it, or they do not, and either way they cannot do anything about it.

Teams with raw sensor data — 1Hz GPS and accelerometer readings from their own app — fare somewhat better on paper, but in practice they have no Python tooling to process it. The academic literature on telematics pricing is rich: Wüthrich (2017, EAJ) established speed-acceleration heatmaps as the foundational feature space, Gao, Wang & Wüthrich (2021, Machine Learning) showed that combined actuarial neural networks (CANN) integrating both traditional and telematics features outperform either alone, and Jiang & Shi (2024, NAAJ) built a discrete-time HMM framework that classifies trips by latent driving state and estimates long-run loss exposure from the stationary distribution. None of them released production-quality code.

[`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) fills that gap. End-to-end pipeline, 84 tests, runs on Databricks serverless.

```bash
uv add insurance-telematics
```

---

## The problem with naive feature engineering

The standard approach to telematics feature engineering — count harsh braking events per kilometre, compute the 85th percentile speed, calculate the fraction of night driving — is reasonable but has a structural weakness. It treats every trip as equivalent. A 3km school run and a 40km motorway commute generate the same feature statistics despite representing categorically different driving contexts. A driver who triggers ten harsh braking events on a 10km urban route and one who triggers ten on a 200km motorway journey get identical braking rates. The underlying risk profiles are not the same.

The HMM approach addresses this by modelling driving as a sequence of latent regime transitions. Rather than computing event rates directly, the model first classifies each second of sensor data into a driving state — cautious, normal, or aggressive — and then extracts features from the state sequence. Time spent in the aggressive state, mean transition rates between states, and state entropy become the GLM inputs. These features have a more direct actuarial interpretation: "this driver spent 22% of trip-seconds in the aggressive regime" is a cleaner risk signal than "this driver generated 4.7 harsh braking events per 100km."

---

## TripSimulator: removing the data access barrier

The first problem with building any telematics model is data access. Raw GPS/accelerometer data at 1Hz is expensive to collect, legally sensitive to share, and proprietary to the insurers who hold it. This is why there is no open benchmark dataset for insurance telematics and why most academic papers use proprietary data that cannot be reproduced.

`TripSimulator` generates synthetic fleet data with the statistical properties of real telematics, using a physically motivated model. Speed evolves as an Ornstein-Uhlenbeck process — mean-reverting to a regime-specific target with configurable volatility. The regime sequence (cautious, normal, aggressive) is drawn from a Dirichlet mixture, so each simulated driver has a characteristic regime distribution that persists across trips. Lateral acceleration (cornering) is derived from speed and heading change. Claims are generated from a Poisson process whose rate is a function of the driver's latent regime distribution.

```python
from insurance_telematics import TripSimulator, TelematicsScoringPipeline

# Simulate a fleet of 200 drivers, 30 trips each
sim = TripSimulator(n_drivers=200, trips_per_driver=30, seed=42)
trips_df, claims_df = sim.simulate()

# trips_df: ~6M rows (1Hz × 30s average trip duration × 6000 trips)
# columns: driver_id, trip_id, timestamp, lat, lon, speed_kmh, accel_ms2, heading
```

The OU speed process is parameterised per regime:

$$dV_t = \theta_r (\mu_r - V_t)\,dt + \sigma_r\,dW_t$$

where $\theta_r$ is the mean-reversion rate, $\mu_r$ the target speed, and $\sigma_r$ the volatility for regime $r \in \{\text{cautious}, \text{normal}, \text{aggressive}\}$. Cautious drivers (low $\mu$, high $\theta$) produce tight speed distributions centred on posted limits. Aggressive drivers (high $\mu$, low $\theta$, high $\sigma$) produce wide distributions with heavier tails above the limit.

---

## DrivingStateHMM: Gaussian emissions with enforced state ordering

The discrete-time HMM component wraps hmmlearn's `GaussianHMM` with two insurance-specific additions: enforced state ordering and actuarial feature extraction.

The ordering problem is standard with HMMs. The EM algorithm optimises transition and emission parameters jointly, but it has no concept of "cautious" versus "aggressive." States emerge from the data in arbitrary order, and the same model fitted on two different seeds will often label state 0 as aggressive and state 2 as cautious on one run, and reverse them on another. For insurance pricing this is unacceptable — you cannot change the meaning of a rating factor between model refits.

We enforce ordering by sorting states post-fit by mean speed of the dominant emission feature, with a consistency check: if mean speeds are not strictly ordered across states, fitting raises an exception rather than returning silently wrong results.

```python
from insurance_telematics import DrivingStateHMM, extract_trip_features

# Extract features from raw trips
features_df = extract_trip_features(trips_df)
# features_df: trip-level harsh_braking_rate, harsh_accel_rate, speed_p85,
#              night_fraction, urban_fraction, cornering_rate

# Fit HMM on speed sequences
hmm = DrivingStateHMM(n_states=3, covariance_type="full")
hmm.fit(trips_df)

# State 0: cautious, State 1: normal, State 2: aggressive
state_probs = hmm.predict_proba(trips_df)
# Returns per-trip state occupation probabilities: (n_trips, 3)
```

The three-state model is the default. Jiang & Shi (2024) use two states (cautious/aggressive); we add a middle state that captures the majority of urban driving and prevents the aggressive state from absorbing any driving above the mean. In practice, the three-state model fits better on the simulated data (lower BIC by 12% over two-state) and produces more stable relativities.

---

## ContinuousTimeHMM: handling irregular GPS intervals

GPS loggers in insurance telematics are not perfectly regular. Network outages, battery-saving modes, and urban canyon effects produce irregular time intervals between observations. At 1Hz nominal sampling, you might observe 0.8s, 1.1s, 2.3s, 1.0s between consecutive fixes. A discrete-time HMM assumes fixed intervals and will silently mishandle gaps — treating a 5-second gap the same as a 1-second gap.

`ContinuousTimeHMM` solves this properly. Rather than a transition matrix $A$ for a fixed time step, it works with a generator matrix $Q$ — the instantaneous transition rate matrix — and computes the actual transition matrix for time interval $\Delta t$ as:

$$A(\Delta t) = \exp(Q \cdot \Delta t)$$

using `scipy.linalg.expm`. The EM algorithm updates $Q$ rather than $A$, and forward-backward computations use the interval-specific $A(\Delta t_i)$ for each observation pair.

```python
from insurance_telematics import ContinuousTimeHMM

cthmm = ContinuousTimeHMM(n_states=3)
cthmm.fit(trips_df)  # Uses actual timestamp intervals

# Q is the generator matrix — off-diagonals are transition rates per second
# Q[0,2] < 0.001 means direct cautious→aggressive transitions are rare
print(cthmm.Q_)
```

The generator matrix $Q$ has the interpretation that off-diagonal element $Q_{ij}$ is the rate at which a driver in state $i$ transitions to state $j$ per unit time. For the simulated data fitted with `TripSimulator` defaults, $Q_{0,2}$ (cautious-to-aggressive) is near zero — direct state jumps without passing through normal are rare, which is actuarially sensible. The diagonal elements are negative and satisfy $Q_{ii} = -\sum_{j \neq i} Q_{ij}$.

The continuous-time formulation also handles longer gaps gracefully. A 30-second GPS outage is represented by $\exp(Q \cdot 30)$, which will typically produce a near-uniform transition probability — we have no information about state during the gap, and the model reflects that rather than assuming the driver stayed in the last observed state.

---

## Feature extraction and GLM integration

Once states are estimated, `extract_trip_features` produces the full feature set for GLM entry:

| Feature | Description |
|---|---|
| `time_in_aggressive` | Fraction of trip-seconds in state 2 |
| `mean_transition_rate` | Average $Q_{ij}$ for $i \neq j$, normalised |
| `state_entropy` | Shannon entropy of trip state distribution |
| `harsh_braking_rate` | Events per 100km (acceleration < −0.3g) |
| `harsh_cornering_rate` | Events per 100km (lateral acceleration > 0.3g) |
| `speeding_fraction` | Fraction of seconds above estimated road speed limit |
| `night_driving_fraction` | Fraction of trip-seconds between 22:00 and 06:00 |
| `urban_fraction` | Fraction of trip-seconds at speed < 50 km/h |
| `speed_p85` | 85th percentile speed across trip |

The road speed limit is estimated from a heuristic on GPS speed distribution (no map data required), which is a practical compromise. For production use, map-matching against an API such as OpenStreetMap or HERE is recommended — the library supports injecting a speed limit lookup function.

`aggregate_to_driver` applies Bühlmann-Straub credibility to combine per-trip features into driver-level scores, weighted by trip distance. A driver with three trips gets heavily shrunk towards the fleet mean; a driver with 150 trips carries almost their own observed experience. The credibility parameter $Z$ is estimated from the cross-trip variance structure using the standard Bühlmann-Straub estimator.

```python
from insurance_telematics import aggregate_to_driver

driver_features = aggregate_to_driver(
    features_df,
    weight_col="trip_distance_km",
    credibility_min_weight=5.0  # minimum weighted trips for full credibility
)
# driver_features: one row per driver, credibility-weighted feature means
# plus driver_credibility_z: the estimated Z for each driver (0-1)
```

---

## The full pipeline

`TelematicsScoringPipeline` chains preprocessing, HMM fitting, feature extraction, credibility aggregation, and Poisson GLM into a single sklearn-compatible object.

```python
from insurance_telematics import TripSimulator, TelematicsScoringPipeline

# Simulate data
sim = TripSimulator(n_drivers=500, trips_per_driver=40, seed=0)
trips_df, claims_df = sim.simulate()

# Fit end-to-end pipeline
pipeline = TelematicsScoringPipeline(
    hmm_type="continuous",   # use ContinuousTimeHMM
    n_states=3,
    glm_feature_subset=[
        "time_in_aggressive", "harsh_braking_rate",
        "speeding_fraction", "night_driving_fraction"
    ]
)
pipeline.fit(trips_df, claims_df)

# Score new trips
risk_scores = pipeline.predict(new_trips_df)
# Returns driver-level risk scores (0-100) calibrated to fleet mean = 50

# GLM relativities by feature
pipeline.relativities()
#   feature                  relativity  ci_lower  ci_upper
#   time_in_aggressive       1.82        1.61      2.06
#   harsh_braking_rate       1.34        1.21      1.48
#   speeding_fraction        1.47        1.29      1.67
#   night_driving_fraction   1.19        1.08      1.31
```

The GLM component is a Poisson frequency model with log link and log(exposure) offset, consistent with standard UK motor pricing practice. The telematics features enter multiplicatively alongside traditional rating factors — the library does not replace the base GLM, it adds a telematics score layer on top of it.

---

## Why this matters for the UK market

UK telematics penetration is around 5% of motor policies. That is roughly 1.3 million policies against a total book of approximately 27 million registered vehicles. The market is heavily concentrated: Admiral Curve (formerly Elephant Auto), By Miles (acquired by Direct Line in 2023), and Marmalade account for most of the volume. Young driver policies, where telematics is near-compulsory as a condition of affordability, drive the majority.

The FCA's pressure on age and postcode as rating factors creates a structural incentive to increase telematics penetration. If traditional rating factors become harder to use, driving behaviour becomes the natural alternative. A 19-year-old who demonstrably drives cautiously at low speeds between 09:00 and 17:00 is a different risk from one with identical demographics who drives at 80 mph on the M6 at 02:00. Telematics measures the difference that demographic factors approximate.

The tooling gap this library addresses is real. Octo, CMT, and The Floow are dominant for a reason: building telematics scoring from scratch requires GPS preprocessing, HMM implementation, credibility weighting, and GLM integration. None of those pieces existed in Python before `insurance-telematics`. Teams with raw data had no alternative to writing all of it themselves or outsourcing to a vendor and accepting the black box.

---

## Academic basis

The core methodology follows Jiang & Shi (2024, NAAJ 28(4):822-839), who applied a discrete-time HMM to classify trips by latent driving state and used the stationary distribution to estimate long-run loss frequency. Our continuous-time extension follows the generator matrix approach standard in survival analysis and, within insurance, in Markov chain NCD models. The combined actuarial-telematics neural network in Gao, Wang & Wüthrich (2021, Machine Learning 111:1787-1827) motivates treating telematics features as an additive GLM layer rather than a replacement for traditional factors. Wüthrich (2017, EAJ 7:89-108) established the foundational result that speed-acceleration feature spaces from GPS data predict claim frequency better than driver age alone — the result that made the entire field credible.

The Bühlmann-Straub credibility aggregation follows the implementation in our [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) library, which was the natural dependency.

---

## Installation and requirements

```bash
uv add insurance-telematics
```

Dependencies: NumPy, SciPy, pandas, hmmlearn, statsmodels. No GPU required. All 84 tests pass on Databricks serverless (Python 3.11) and standard CPython 3.10+.

The Databricks notebook demo is at `/notebooks/telematics_demo.py` in the repository — a worked example on simulated data covering the full pipeline from `TripSimulator` through `TelematicsScoringPipeline` with visualisation of state sequences and relativity tables.

**[github.com/burning-cost/insurance-telematics](https://github.com/burning-cost/insurance-telematics)**
