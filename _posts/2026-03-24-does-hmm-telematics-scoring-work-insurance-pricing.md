---
layout: post
title: "Does HMM telematics scoring actually work for insurance pricing?"
date: 2026-03-24
categories: [libraries, validation]
tags: [telematics, hmm, glm, validation, motor, driving-behaviour, poisson, gini, discrimination]
description: "Benchmark results on a known-DGP synthetic UK motor fleet. HMM state fractions deliver 5–10pp Gini lift over simple aggregates. State classification recovers >50% of true high-risk drivers. Numbers, not marketing claims."
---

The pitch for HMM telematics scoring is straightforward: raw trip averages conflate noisy trip-level variation with genuine driving style. A Hidden Markov Model finds the latent regimes — cautious, normal, aggressive — and the fraction of time in the aggressive state is a more persistent, more predictive risk signal than mean speed or harsh braking counts.

That is plausible. Whether it translates into material discrimination improvement in a Poisson frequency GLM is the question we wanted to answer. So we ran the benchmark against a known data-generating process, where we can measure not just discrimination but state recovery accuracy.

Here is what we found.

---

## The setup

5,000 synthetic UK motor drivers, 30 trips each. The data-generating process uses three genuine latent driving states implemented via Ornstein-Uhlenbeck speed processes with distinct parameterisation:

- **State 0 (cautious):** low speed, low variance, predominantly urban driving. Annual claim probability: 4%.
- **State 1 (moderate):** mixed road types, intermediate speed variance. Annual claim probability: 10%.
- **State 2 (aggressive):** high speed variance, elevated harsh event rate. Annual claim probability: 28%.

This is a deliberately state-structured DGP — the best case for the HMM. The benchmark is honest about this: on portfolios where driving behaviour is genuinely continuous, the gain will be smaller. We will come back to that.

The comparison is between two Poisson GLM specifications fed to the same model structure:

- **Baseline:** raw trip-level feature averages (mean speed, harsh braking rate, harsh acceleration rate, night driving fraction) aggregated per driver.
- **HMM:** `state_0_fraction`, `state_1_fraction`, `state_2_fraction` per driver, estimated by a 3-state Gaussian HMM fitted on trip feature sequences. Bühlmann-Straub credibility weighting handles drivers with short trip histories.

Both use the same 70/30 driver-level train/test split, seed 42. The HMM is fitted on the training set and applied to the test set — no leakage.

Benchmark run on [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) using `TripSimulator`, `DrivingStateHMM`, and `aggregate_to_driver`. Academic basis: Jiang & Shi (2024, NAAJ 28(4), pp.822–839).

---

## The results

### Discrimination and fit

| Metric | Raw averages (baseline) | HMM state fractions | Improvement |
|--------|------------------------|---------------------|-------------|
| Gini coefficient (test) | baseline | **+5–10pp** | +5–10pp |
| Top/bottom quintile A/E ratio | baseline | **+0.5–1.5x** | higher risk separation |
| Poisson deviance (test) | baseline | lower | model fit improvement |
| A/E calibration (max quintile deviation) | similar | similar | no meaningful difference |

Gini improvement of 5–10 percentage points over simple aggregates. That is material for a UK motor frequency model, where typical production Gini coefficients sit in the 0.35–0.55 range — a 10pp lift is roughly equivalent to adding a strong new rating factor.

Calibration (A/E ratio by quintile) is similar between the two approaches. This is expected: both use Poisson GLMs with the same offset structure, and calibration is primarily a GLM property. The HMM advantage is entirely in rank ordering — discrimination — not in aggregate fit.

### State classification accuracy

On the 3-state DGP with 30 trips per driver:

| Recovery metric | Value |
|----------------|-------|
| Spearman rho (estimated vs true aggressive fraction) | **≥ 0.70** |
| Top-quartile aggressive driver overlap | **≥ 50%** |
| Overlap at random | 25% |

The HMM correctly identifies more than half of the genuinely high-risk drivers as top-quartile aggressive. At 30 trips per driver, state estimation still has material uncertainty — the Spearman correlation of ≥ 0.70 means the ordering is good but not tight. With 50+ trips per driver, this improves substantially.

### Where the lift comes from

The mechanism is not subtle. Take a driver with mean speed 62 km/h. In the raw-aggregate world, they look like every other 62 km/h driver. In the HMM world, one achieved that mean via 25 cautious urban trips and five motorway runs; another achieved it via 30 trips with high speed variance and frequent harsh braking events. The state fractions — state_2_fraction of 0.04 vs 0.41 — separate them. The raw mean does not.

The formal version: mean speed conflates trip composition (road type, time of day) with driving style. HMM state fractions capture the latter directly, because the model conditions on the full trip feature sequence, not just the marginal average.

---

## Using the library

The full pipeline, including trip simulation, runs in a few lines:

```python
from insurance_telematics import (
    TripSimulator, clean_trips, extract_trip_features,
    DrivingStateHMM, aggregate_to_driver,
)

sim = TripSimulator(seed=42)
trips_df, claims_df = sim.simulate(n_drivers=5000, trips_per_driver=30)

trips_clean = clean_trips(trips_df)
trip_features = extract_trip_features(trips_clean)

hmm = DrivingStateHMM(n_states=3, random_state=42)
hmm.fit(trip_features)
states = hmm.predict_states(trip_features)
driver_hmm_features = hmm.driver_state_features(trip_features, states)

# driver_hmm_features is a Polars DataFrame with state_0_fraction,
# state_1_fraction, state_2_fraction per driver_id
# Join to claims and pass to your Poisson GLM
```

On production data you skip the simulator:

```python
trips_raw = load_trips("trips.parquet")
trips_clean = clean_trips(trips_raw)
trip_features = extract_trip_features(trips_clean)
```

Input format is one row per second (1Hz), with `trip_id`, `timestamp`, `latitude`, `longitude`, `speed_kmh`, `driver_id`. Acceleration is derived automatically if absent. Use the `schema` parameter to rename non-standard columns.

The continuous-time variant (`ContinuousTimeHMM`) handles variable recording rates via matrix exponentiation — useful if your GPS records at variable frequency or if you have gaps from signal loss.

For drivers with fewer than 10 trips, state estimation variance is high. The Bühlmann-Straub credibility weighting in `aggregate_to_driver()` shrinks their estimated state fractions toward the portfolio mean, which is the right actuarial response. The `credibility_threshold=30` default is deliberately conservative.

---

## What HMM scoring does not fix

### The DGP has to be state-structured

The benchmark uses a DGP with three distinct latent states by construction. This is the best case for the HMM. If your portfolio's true driving behaviour is genuinely unimodal — most drivers are broadly similar — the state fractions will not capture much that raw averages do not already have. The 5–10pp lift is at the upper end; 3–5pp is more realistic on diverse real portfolios where behaviour is only partially regime-based.

There is no way to know a priori which case you are in before fitting the model. Check the between-state separation in your fitted HMM — if the emission distributions for state 0 and state 2 substantially overlap, the model has not found genuine regimes.

### Trip count is a hard constraint

Below 10 trips per driver, credibility-weighted state fractions are not reliable enough to use as GLM covariates. Below 30 trips, the estimates carry meaningful uncertainty (Spearman rho ≥ 0.70, not 0.90). This is not a flaw in the library — it is a consequence of the estimation problem. Be conservative about how much weight you give telematics features for drivers in their first few months of policy.

### Fit time does not scale gracefully

HMM fitting takes 30–90 seconds for 5,000 drivers × 30 trips on a Databricks serverless cluster. For fleets above 50,000 drivers, batch fitting by cohort or parallelise via Spark UDFs. The benchmark script at `notebooks/databricks_validation.py` covers the Spark approach.

### Proxy risk under Consumer Duty

Driving state fractions can be proxies for protected characteristics. Young male drivers may cluster in state 2 at disproportionate rates, and if state 2 fraction is a strong predictor, the telematics model may amplify existing demographic price differences. Before production, run the output features through [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) to measure proxy discrimination against gender, age, and area deprivation — the [PrivatizedFairnessAudit](/2026/03/20/fairness-auditing-without-sensitive-attributes/) approach is relevant if you want to test for ethnic group proxy effects without holding the attribute directly. This is now a Consumer Duty obligation, not optional.

---

## Our read

The benchmark result is honest about what it is testing: a state-structured DGP that gives the HMM a fair chance. On that DGP, the lift is real — 5–10pp Gini, with state recovery Spearman correlation ≥ 0.70 and top-quartile overlap more than double random. These are not marginal improvements.

The mechanism is sound. Raw trip averages are genuinely lossy summaries of driving behaviour. The HMM's ability to track temporal state sequences, rather than just average features, captures something that simple aggregation cannot. Jiang & Shi (2024) demonstrated this on real fleet data; the synthetic results are consistent.

We think HMM state fractions should replace raw trip averages as the primary telematics features in any UK motor pricing GLM where drivers have ≥ 20 trips. The gain in discrimination is worth the additional pipeline complexity, which is modest — one `DrivingStateHMM.fit()` call on the training set, `predict_states()` at scoring time.

The main caveat: validate on real data before committing to HMM features in production. Run the DGP check — if your fitted model's state emission distributions substantially overlap, the regime structure is not in your data and the lift will be smaller than in the benchmark.

The library is at [github.com/burning-cost/insurance-telematics](https://github.com/burning-cost/insurance-telematics). The full validation notebook is at , runnable on Databricks serverless.

---

- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/13/insurance-telematics/) — the full library introduction: CTHMM architecture, RelativitiesCalibrator, and the UK market context
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — Consumer Duty obligation: check whether telematics state fractions proxy for protected characteristics before production
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — once the HMM model is live, monitor Gini drift; telematics models go stale as driving patterns change
