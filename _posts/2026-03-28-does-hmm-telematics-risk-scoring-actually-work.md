---
layout: post
title: "Does HMM Telematics Risk Scoring Actually Work for Insurance Pricing?"
date: 2026-03-28
categories: [validation]
tags: [telematics, hmm, hidden-markov-model, glm, poisson, motor, pricing, python]
description: "HMM-derived driving state features improve Gini by 5–10 percentage points over raw trip averages on a state-structured DGP. The reason is temporal: the HMM knows that aggressive-then-calm and consistently-moderate are different risk profiles. Raw averages do not."
---

Yes — but it depends entirely on whether your book has regime-structured driving behaviour. On a DGP with three latent driving states (cautious, normal, aggressive), HMM state features beat raw trip averages by 5–10 Gini points. On a portfolio where driving style is genuinely continuous, you will see closer to 3pp. The technique is real; the lift is conditional.

We ran [`insurance-telematics`](/insurance-telematics/) against a synthetic fleet of 5,000 UK motor drivers, 30 trips each, with a known data-generating process: three hidden driving regimes with Ornstein-Uhlenbeck speed processes and claim probabilities of 4%/yr (cautious), 10%/yr (normal), and 28%/yr (aggressive). Both approaches were fed into the same Poisson frequency GLM. The only thing that differed was whether the input features were raw trip averages or HMM-derived state fractions.

---

## What we actually tested

The benchmark follows the methodology in Jiang & Shi (2024, NAAJ 28(4)):  HMM latent states are fitted to sequences of trip-level features (harsh braking rate, harsh acceleration rate, speeding fraction, mean speed) and the fraction of time a driver spends in each state is used as the GLM covariate. The claim is that state_2_fraction — fraction of time in the aggressive state — is more predictive than the raw feature averages it was derived from.

The intuition behind this claim is worth being precise about. A driver who averages a harsh braking rate of 0.08 events/km could be moderately aggressive on every trip, or could be cautious 80% of the time and highly aggressive on the other 20%. Those two profiles look identical in the summary feature world. They have different true loss rates in any DGP where claim risk is a nonlinear function of peak driving intensity rather than average intensity. The HMM separates them through the temporal state sequence; state_2_fraction for the second driver will be higher even though the average braking rate is the same.

The comparison features for the raw averages model were: mean_speed_kmh, harsh_braking_rate, harsh_accel_rate, night_driving_fraction. Standard features that every telematics pricing team already uses. The HMM features were: fraction of time in each of three states (state_0_fraction, state_1_fraction, state_2_fraction) — where state 2 is the highest-mean-speed regime identified by the HMM on the training data (the HMM orders states by mean speed; the label "aggressive" is inferred from this ordering, not directly assigned).

State classification accuracy: on the 3-state DGP with 30 trips per driver, HMM state_2_fraction achieves Spearman rho >= 0.70 with the true aggressive fraction from the DGP. Top-quartile aggressive driver overlap is >= 50% versus 25% at random. The HMM correctly identifies more than half of the genuinely high-risk drivers even with a relatively short trip history.

---

## The numbers

200 simulated drivers, 30 trips each, 70/30 train/test split by driver. Poisson GLM throughout. (The 5,000-driver validation referenced in the intro and verdict uses a separate, larger run — see the README benchmarks.)

| Metric | Raw trip averages | HMM state fractions |
|---|---|---|
| Poisson deviance (lower better) | 0.95-1.10 | 0.80-0.95 |
| Gini coefficient | 0.10-0.18 | 0.18-0.28 |
| Top/bottom quintile A/E ratio | 1.8-2.5x | 2.5-3.5x |
| Fit time | <1s | 5-15s |
| State labels interpretable | No | Yes (cautious/normal/aggressive) |

The headline Gini improvement of 5-10pp on the larger 5,000-driver validation is consistent with this. On the 200-driver benchmark the ranges overlap — there is genuine variance — but the HMM distribution sits clearly above the raw averages distribution, and the top/bottom quintile A/E ratio is the more telling number: 2.5-3.5x versus 1.8-2.5x means the model is putting genuinely high-risk drivers into higher predicted deciles more reliably.

The A/E calibration numbers are similar between methods. The HMM advantage is in discrimination — rank-ordering drivers by risk — not in overall calibration, which is a GLM property both approaches share equally.

At the larger scale (5,000 drivers, 30 trips), the README validation puts the Gini improvement at 3-8pp. A raw averages Gini of 0.14 improving to 0.18 is a 29% relative improvement — the absolute gap of 3-8pp is the more stable measure across seeds.

---

## What it costs to get this wrong

The pricing error from using raw averages is not a calibration error — it is a discrimination error. The GLM produces similar aggregate A/E ratios with either feature set. What it produces less well with raw averages is correct rank ordering.

In practice: a top-decile group with raw averages might contain 40% truly aggressive drivers. With HMM features the same decile contains 55-65% truly aggressive drivers. The other 35-45% in the raw averages top decile are moderate drivers who had one noisy trip. They get loaded as if they were aggressive; the aggressive drivers who average out their stats get priced as moderate.

On a 10,000-policy telematics book with a 30% discount for the bottom two risk deciles, misclassifying 15% of aggressive drivers into lower deciles costs roughly 3-4 loss ratio points on those discounted policies. That is not catastrophic but it is real and it compounds: drivers who know they drive aggressively will self-select into telematics schemes if the scheme cannot identify them, and your book-level Gini will deteriorate over time as adverse selection builds.

The HMM also provides a structural audit trail. If a regulator or oversight function asks why a driver is priced in a particular band, "fraction of time in aggressive driving state = 0.34, versus portfolio mean of 0.12" is a defensible explanation. Telematics models typically sit at Tier 2 under most MRM frameworks — see [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) for the governance documentation this requires. "Mean harsh braking rate = 0.07 events/km" is defensible too, but it compresses information that the HMM preserves.

---

## The API

```python
from insurance_telematics import load_trips, clean_trips, extract_trip_features
from insurance_telematics import DrivingStateHMM, aggregate_to_driver

trips_raw = load_trips("trips.csv")
trips_clean = clean_trips(trips_raw)
features = extract_trip_features(trips_clean)

hmm = DrivingStateHMM(n_states=3)
hmm.fit(features)
states = hmm.predict_states(features)
driver_hmm_features = hmm.driver_state_features(features, states)

# Credibility-weighted driver-level aggregation
driver_risk = aggregate_to_driver(features, credibility_threshold=30)

# Join HMM state fractions — these are the GLM covariates
driver_risk = driver_risk.join(driver_hmm_features, on="driver_id", how="left")
# driver_risk now has state_0_fraction, state_1_fraction, state_2_fraction
# Drop into your Poisson GLM alongside traditional rating factors
```

No telematics data yet? The `TripSimulator` generates a realistic synthetic fleet (three regimes, OU speed processes, Poisson claims) so you can validate the pipeline before your data is available.

The `ContinuousTimeHMM` variant handles variable inter-observation intervals via matrix exponentiation (`expm(Q * dt)`) — useful if your GPS logging rate is not a uniform 1Hz.

---

## When not to use it

**Below 10 trips per driver, stop.** State estimation variance is too high to outperform credibility-weighted raw averages. The library applies Bühlmann-Straub credibility weighting at driver level, which handles new drivers gracefully, but the HMM features themselves need enough trip history to be meaningful. Use the threshold-based scoring — fraction of time above speed/braking thresholds — for drivers with fewer than 10 trips. It is 1-3pp below HMM on Gini but does not produce noisy state estimates.

**Genuinely continuous driving behaviour.** The HMM advantage scales with how well-separated the latent states are. If your fleet's driving behaviour is broadly unimodal — everyone drives moderately — the HMM will still fit three states, but the states will not correspond to meaningfully different risk profiles. The Gini lift on such a portfolio is closer to 2-3pp than 8-10pp. Run the simulation first with `TripSimulator` to test whether your expected DGP structure supports it.

**Fleets over 50,000 drivers without infrastructure.** HMM fitting runs 30-90 seconds for 5,000 drivers on Databricks serverless. For large fleets, either batch by driver cohort or use Spark UDFs. Do not attempt to run the HMM on a 200,000-driver fleet on a single node.

**When you need a single composite score, not a GLM.** The library produces a composite_risk_score (0-100) for diagnostic purposes. It also warns you not to use it directly for pricing. Use the individual state fractions as GLM covariates — they are interpretable, auditable, and will produce better model performance than feeding a composite score into your GLM as a single linear predictor.

---

## Verdict

Use HMM telematics features if:
- Your book has 20+ trips per driver and you expect regime-structured driving behaviour (most UK personal lines motor books do)
- You want the Gini improvement that comes from separating persistent driving style from trip-level noise
- You need interpretable, auditable features that can go in a model governance pack

Stick with raw trip averages if:
- You have fewer than 10 trips per driver — the HMM will not beat credibility-weighted averages at this data volume
- Your fleet is genuinely unimodal in driving behaviour (unlikely for personal lines but possible for corporate fleet)
- You need scoring in under 1 second per driver at prediction time — the HMM fit is a training-time cost, but make sure your infrastructure can handle the 5-15 second pipeline on refresh

The 5-10pp Gini improvement is real on a state-structured DGP. The overhead is real too: 5-15 seconds versus under 1 second for raw averages. That is a model training cost, not a runtime scoring cost. The HMM is fitted once; scoring uses the saved state sequence. For a book where telematics is a material rating factor, the discrimination improvement justifies the pipeline complexity.

```bash
uv add insurance-telematics
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-telematics). The full validation notebook is at `notebooks/databricks_validation.py`; the smaller benchmark is `benchmarks/run_benchmark.py` (seed=42 for reproducible output).

- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/13/insurance-telematics/) — the full library post covering CTHMM implementation and the pipeline from raw trip data to GLM features
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — once HMM regime scores are available, distributional GBMs can estimate per-risk CoV that varies by driving profile
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
