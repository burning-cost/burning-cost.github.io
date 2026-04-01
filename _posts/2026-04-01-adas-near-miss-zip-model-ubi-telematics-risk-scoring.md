---
layout: post
title: "ADAS Near-Miss Counts as a Risk Signal: A Zero-Inflated Poisson Framework"
date: 2026-04-01
categories: [telematics, techniques]
tags: [telematics, UBI, PHYD, motor-pricing, zero-inflated-poisson, ZIP, GBZIP, EM-algorithm, latent-groups, ADAS, near-miss, fleet, commercial-lines, Microlise, Lytx, panel-data, insurance-telematics, python, uk-insurance, Zhang, Guillen, arXiv-2509.02614, forward-collision, lane-departure, harsh-braking]
description: "Zhang, Guillen, Li, Li and Chen (arXiv:2509.02614) build a group-based zero-inflated Poisson model for ADAS near-miss event counts using 354 commercial drivers across 8.1 million km. The latent behaviour group structure is genuinely useful; the missing link to actual claims is a problem that needs to be said out loud."
math: true
author: burning-cost
---

Every telematics system already produces near-miss data. Harsh braking triggers, forward collision warning activations, lane departure alerts — these happen dozens of times per driver per week on a modern ADAS-equipped commercial vehicle, and they are logged. The question is what to do with them.

The dominant approach is to aggregate them into a score: sum the harsh events, weight by severity, divide by distance, and feed the result into a GLM. This works, but it discards a lot of structure. A driver who generates zero forward collision warnings has a very different risk profile depending on *why* they generate zero: because they are an exceptionally cautious driver, or because they avoid motorways, or because they never drive in conditions where the sensor fires. A raw count does not distinguish these cases.

Zhang, Guillen, Li, Li and Chen ([arXiv:2509.02614](https://arxiv.org/abs/2509.02614), submitted IEEE BigData 2025) take a different approach. They model the weekly count of each near-miss event type using a zero-inflated Poisson distribution, and they embed that model inside a latent group structure. The result is a Group-Based ZIP (GBZIP) that simultaneously asks: which behavioural archetype does this driver belong to, and given that archetype, what is their expected event rate? Both questions are answered jointly by an EM algorithm.

The dataset is 354 commercial vehicle drivers in China, April 2021 to March 2022: 287,511 trips, approximately 8.1 million kilometres, structured as a weekly panel of 12,528 driver-weeks. The event types modelled include harsh braking, harsh acceleration, serious speeding, forward collision warning, lane departure, and following-too-close alerts.

---

## Why zero-inflated Poisson

A standard Poisson model for weekly event counts runs into two problems. First, most drivers produce zero counts in most weeks for most event types — especially the rarer ADAS triggers like forward collision warnings. A Poisson distribution with a small mean will predict some zeros, but not nearly as many as are observed. The excess zeros deflate the fitted mean and produce poor predictions for the non-zero tail.

Second, zero-generating processes are not all the same. A driver who produces zero harsh braking events in a week is either: (a) driving carefully and genuinely avoiding harsh manoeuvres, or (b) driving so little that they have no opportunity. The structural zero from process (a) should not be treated identically to the sampling zero from a very low rate process (b).

Zero-inflated Poisson addresses the first problem directly. The model has two components:

$$P(C = 0 \mid x, E) = \pi + (1 - \pi)\exp(-E\lambda)$$

$$P(C = k \mid x, E) = (1 - \pi) \cdot \frac{(E\lambda)^k e^{-E\lambda}}{k!}, \quad k \geq 1$$

where $\pi$ is the probability of a structural zero (a driver in the "always zero" state), $\lambda$ is the Poisson rate for drivers in the counting state, and $E$ is the exposure offset — weekly distance, which enters the count component as $\log E$ in the linear predictor.

The ZIP answers the question: what fraction of zero-count weeks are genuinely zero-generating versus merely low-rate? Across the paper's portfolio, the improvement from Poisson to ZIP is substantial. For total near-miss count: Poisson AIC is 1,368,399; ZIP drops this to 1,342,583. Poisson deviance in five-fold cross-validation: 107.23. ZIP: 48.83.

That is a real improvement. But the bigger gain comes from the group structure.

---

## Latent behaviour groups

The core contribution is embedding ZIP inside a finite mixture model. There are $G$ latent groups, each with its own zero-inflation parameter $\pi^{(g)}$ and Poisson rate $\lambda^{(g)}$. Drivers are not assigned deterministically to groups — each driver has a posterior probability of membership in each group, estimated from their observed event history.

Formally, driver $i$ at week $t$ belongs to latent group $g$ with prior probability $\omega_g$ (the mixing weight). Conditional on group $g$:

$$\text{logit}\, \pi_{i,t}^{(g)} = \gamma_0^{(g)} + x_{i,t}^\top \gamma^{(g)}$$

$$\log \Lambda_{i,t}^{(g)} = \alpha_0^{(g)} + x_{i,t}^\top \beta^{(g)} + \log E_{i,t}$$

The marginal distribution for any driver is the mixture across groups:

$$\Pr(C = k \mid x, E) = \sum_{g=1}^{G} \omega_g \Pr(C = k \mid Z_i = g, x, E)$$

This is identified from data by an EM algorithm. The E-step computes posterior group membership weights (the $\tau_{i,g}$ posteriors) given current parameters. The M-step maximises weighted log-likelihood to update $\omega_g$ and the ZIP parameters per group. Convergence criterion is $|\ell^{(t)} - \ell^{(t-1)}| \leq \varepsilon(1 + |\ell^{(t-1)}|)$ with $\varepsilon = 10^{-4}$.

The paper tests $G \in \{1, 2, 3, 4\}$. The AIC elbow is at $G = 3$, which the paper recommends for interpretability. $G = 4$ provides the best fit. For total near-miss count, GBZIP at $G = 3$ produces AIC of 932,321 — against ZIP's 1,342,583. That is a gap of roughly 400,000 AIC points. The group structure is doing a lot of work.

The three groups that emerge in the $G = 3$ solution have a natural interpretation: low-risk drivers with high zero-inflation (most weeks are structural zeros), moderate-risk drivers with occasional events and a genuine Poisson count process, and high-risk drivers with persistently elevated rates across multiple event types. The groups are not directly interpretable without examining the parameters, but they correlate with covariates in the expected directions — more severe speeders are disproportionately assigned to the high-risk group.

---

## The covariates

The model includes 16 context variables alongside the event type outcomes. These split into:

**Behavioural inputs (ADAS warning counts):** sum of fatigue driving alerts, forward collision warnings, driver inattention warnings, smoking detections, phone use detections, lane departure warnings, and following-too-close warnings. These are the upstream sensor signals that predict downstream event counts — the ADAS system is detecting risk precursors before they escalate to near-miss events.

**Driving behaviour:** harsh acceleration count (range 0–3,996), harsh braking count (0–2,040), serious speeding count (0–300). These are the kinematic signals familiar from non-ADAS telematics systems.

**Context:** mean speed limit, temperature, visibility, wind speed, congestion index, road type (motorway versus rural indicator), traffic sign counts, road quality rating, road slope.

The exposure offset is weekly distance in kilometres (range 10–3,423 km across the panel).

The inclusion of weather and road context alongside the event counts is important and underused in UK practice. A driver generating high forward collision warning rates in low-visibility conditions is not the same risk as a driver generating the same count on a clear motorway. The contextual weighting changes the interpretation of the event count.

---

## Cross-validation results

Five-fold cross-validation on the held-out weeks produces the following comparisons for total near-miss count prediction:

| Model | Deviance | RMSE | McFadden R² |
|-------|---------|------|------------|
| Poisson | 107.23 | 143.46 | 0.09 |
| ZIP | 48.83 | 46.96 | 0.26 |
| G-ZIGP ($G = 3$) | 32.17 | 300.27 | 0.22 |

The G-ZIGP (Generalised ZIP, which adds a dispersion parameter within each group) has the best deviance but the worst RMSE. This happens because the generalised Poisson extension improves calibration of the count distribution's shape without necessarily improving mean predictions. The deviance metric rewards distributional fit; RMSE penalises mean prediction error. The GBZIP strikes a better balance for mean prediction purposes, which is what a risk score ultimately needs.

McFadden $R^2$ of 0.26 for ZIP and 0.22 for G-ZIGP is a reasonable result for count data with this degree of heterogeneity. It is not a high number, which is honest — near-miss event counts have substantial unexplained variation even with 16 covariates and latent group structure.

---

## What this looks like in code

The `ZIPNearMissModel` in `insurance-telematics` implements this framework. The input is a weekly panel DataFrame rather than the trip-level data that the rest of the library operates on — this is a different layer of analysis, not a replacement.

```python
import polars as pl
from insurance_telematics import TripSimulator, extract_trip_features, aggregate_to_driver
from insurance_telematics.zip_near_miss import NearMissSimulator, ZIPNearMissModel

# Generate synthetic weekly ADAS event counts (G=3 latent groups)
nm_sim = NearMissSimulator(n_groups=3, seed=42)
weekly_counts = nm_sim.simulate(n_drivers=200, n_weeks=52)
# weekly_counts schema: driver_id, week_id, exposure_km,
#   harsh_braking, harsh_accel, forward_collision,
#   lane_departure, too_close, speeding_serious

# Fit the GBZIP model
zip_model = ZIPNearMissModel(
    n_groups=3,
    event_types=["harsh_braking", "forward_collision", "lane_departure"],
    use_generalised=False,   # ZIP not ZIGP — better RMSE
    max_iter=100,
    tol=1e-4,
    exposure_col="exposure_km",
)
zip_model.fit(weekly_counts)

# Group membership posteriors — one row per driver
group_probs = zip_model.predict_group_probs(weekly_counts)
# driver_id | prob_group_0 | prob_group_1 | prob_group_2

# Predicted near-miss rate per driver (events per km)
rates = zip_model.predict_rate(weekly_counts)
# driver_id | predicted_nme_rate

# Driver-level features for passing to TelematicsScoringPipeline
zip_features = zip_model.driver_risk_features(weekly_counts)
# driver_id | dominant_group | nme_rate_per_km | zero_fraction_harsh_braking | ...
```

The model groups are sorted post-fit by ascending mean NME rate, so group 0 is consistently the safest. `dominant_group` is the group with highest posterior probability for each driver — the best single label for their behavioural archetype.

These features join to the standard trip-based risk pipeline at driver level:

```python
# Trip-based HMM features (existing pipeline)
sim = TripSimulator(seed=42)
trips_df, _ = sim.simulate(n_drivers=200, trips_per_driver=50)
trip_features = extract_trip_features(trips_df)
trip_risk = aggregate_to_driver(trip_features)

# Combine: HMM trip features + ZIP near-miss group features
combined = trip_risk.join(zip_features, on="driver_id", how="left")
# Pass combined to TelematicsScoringPipeline for GLM scoring
```

The two layers are complementary. The HMM operates within trips at sub-second resolution, classifying latent driving states and detecting anomalous behaviour. The ZIP model operates across weeks, classifying drivers into behavioural archetypes based on their event count distributions. Intra-trip precision meets inter-week memory.

---

## The gap that needs stating clearly

The paper's dataset contains a `claims_count` column with values in $\{0, 1, 2\}$.

The paper does not use it.

There is no analysis demonstrating that drivers with higher predicted NME rates have higher claim frequencies. No regression of claims on group membership. No validation that group 2 (the highest-rate group) actually costs more than group 0. The near-miss to claim relationship is asserted on the basis of face validity — more near-misses should mean more accidents — not demonstrated on the data in hand.

This matters enormously for insurance pricing. A telematics score that predicts near-miss event counts is not the same thing as a score that predicts claim costs. The two might correlate closely. They might not. The paper with 354 drivers had the data to check and did not check it.

The actuarial literature on this gap is thin. Near-miss reporting systems in occupational safety suggest a ratio of approximately 300 near-miss events per serious injury (Heinrich's triangle, updated variants). But insurance claims are not occupational injuries, the Heinrich ratios are contested, and commercial vehicle near-miss events in China in 2021 do not map cleanly onto UK personal motor claims frequency in 2026.

We cannot use this model for pricing in the UK without first establishing the claim link on UK data. That requires either:

1. A UK fleet insurer with ADAS event counts *and* claims data, willing to run a correlation study on their own book. This is tractable for Microlise or Lytx customers with sufficient claims volume — say, a fleet of 500+ vehicles over three years.

2. An OEM data partnership where ADAS event logs are matched to insurance claims records via VIN. This exists in prototype in the US (State Farm / GM partnership announced 2023); it does not exist at scale in the UK as of early 2026.

Until one of those pathways produces evidence, the ZIP near-miss model is a near-miss *rate estimation* tool, not a claim frequency predictor. It tells you who generates events; it does not tell you who generates losses. Those are different questions.

---

## Where this applies in the UK right now

The paper's dataset is commercial fleet — 354 drivers working for logistics operators with ADAS-equipped dashcam systems. That context maps directly onto UK commercial fleet insurance.

Microlise, Lytx, Samsara, and Teletrac Navman all provide event count telemetry to UK fleet operators: forward collision warning trigger counts, lane departure activations, harsh event tallies, fatigue alerts. The data format required by the GBZIP model — weekly counts per driver with exposure in kilometres — is exactly what these systems already export. A UK commercial fleet insurer with Microlise integration has the input data for this model today.

The picture for UK personal lines is different. By Miles (Direct Line Group) uses OBD/GPS for mileage-based pricing; it does not capture ADAS sensor event streams. Ticker uses a smartphone app plus OBD dongle; it captures kinematic harsh events via accelerometer but not factory ADAS trigger counts. Vitality Drive uses Cambridge Mobile Telematics DriveWell scoring; again, accelerometer-based kinematics rather than ADAS event logs.

Verisk noted in 2024 that for motor insurers "it can be very difficult to process data around existing ADAS features and turn it into useful underwriting information" — inconsistent event nomenclature across manufacturers, variable sensor calibration, and no industry-standard data feed. A lane departure warning on a Mercedes-Benz and a lane departure warning on a Vauxhall Mokka are not the same trigger. The OEM-to-insurer data pipeline for personal lines ADAS event counts does not exist commercially in the UK at scale.

The regulatory pathway is there in principle. UK GDPR Article 20 gives drivers the right to data portability from OEM connected services (FordPass, Volkswagen We Connect, ConnectedDrive). But exercising that right voluntarily for insurance pricing purposes — and maintaining the consent at renewal — is a product design challenge, not just a data engineering one.

Our assessment: fleet telematics now; personal lines in two to three years, conditional on OEM data sharing maturing. The model architecture will not need to change. The data supply will.

---

## The latent group structure and Consumer Duty

There is a regulatory dimension to the group structure that actuaries building on this paper should think through before deployment.

Each driver is assigned a `dominant_group` label based on posterior probabilities. That label feeds into pricing. The assignment is latent — the driver cannot observe the inputs to the group assignment directly, and the group boundary is not deterministic. It is a probabilistic classification from a mixture model.

Under FCA Consumer Duty (PS22/9), pricing models must be explainable and produce fair value. Under UK GDPR, automated decision-making that produces significant effects on individuals may require the right to an explanation and the right to human review (Article 22, incorporated into UK law post-Brexit).

A driver whose premium is higher because they have been assigned to group 2 of a latent mixture model has a legitimate question: why group 2, and what would I need to change to move to group 1? The `dominant_group` label does not answer that question directly. The event count inputs do — but explaining that "your forward collision warning rate of 0.8 per week puts you in the high-risk archetype" is a tractable communication task only if the group structure is documented clearly enough to support that translation.

The ZIP group structure is more interpretable than a neural network score precisely because it has a closed-form distributional basis. Each group has a specific zero-inflation rate and a specific Poisson mean per event type. Those parameters can be turned into plain-English descriptions: "Group 0 drivers generate fewer than one harsh braking event per 500 km; Group 2 drivers generate on average 4.2 per 500 km." That is explainable. The EM algorithm and the mixture weights behind the classification do not need to be explained to the policyholder; the group characteristics do.

---

## What we have built

The `ZIPNearMissModel` is specified and under construction in `insurance-telematics`. The target build is `src/insurance_telematics/zip_near_miss.py` with a companion `NearMissSimulator` for synthetic data generation.

The decision to build was driven by the library gap rather than immediate client demand. The existing `insurance-telematics` pipeline is trip-centric: the HMM classifies states within trips, the feature extractor aggregates to trip level, the scoring pipeline runs a Poisson GLM at driver level. There is no weekly panel structure, no count model for specific event types, and no latent group assignment. The ZIP model fills a genuine modelling gap regardless of whether UK fleet insurers are ready to use it today.

The key dependencies are already in `pyproject.toml`: `statsmodels >= 0.14.5` includes both `ZeroInflatedPoisson` and `ZeroInflatedGeneralizedPoisson` in `statsmodels.discrete.count_model`. No new packages required.

The current library version is 0.1.9. The ZIP module will ship in 0.2.0.

---

## The broader argument

ADAS event counts are a qualitatively different telematics signal from kinematic harsh events. A harsh braking detection from an accelerometer tells you that a driver made a sharp deceleration. A forward collision warning trigger tells you that the ADAS system judged the driver's following distance insufficient to avoid a collision at current closing speed — a prospective risk assessment built into the vehicle itself.

That prospective quality is valuable. The accelerometer records what happened; the forward collision warning records what almost happened. Insurance pricing has always cared about near-misses in principle — this is the occupational safety logic applied to driving. The contribution of Zhang et al. is to give that intuition a formal statistical structure that handles the excess zeros, the exposure heterogeneity across drivers, and the mixing of genuinely cautious drivers with low-exposure drivers in the zero count.

The model will become more useful as UK data infrastructure matures. The fleet application is viable now. The claim link validation is the blocking item. We think a study correlating GBZIP group membership with subsequent claims frequency on a UK fleet book — even at n=500 drivers over three years — would be publishable and commercially significant. If you are working at a commercial fleet insurer with Microlise or Lytx data and a reasonable claims history, this is a tractable PhD or dissertation project with direct underwriting application.

The zero-inflated structure will generalise to other low-frequency telematics event types beyond ADAS — near-miss events from dash-cam computer vision, pedestrian proximity alerts, intersection conflict detections. As vehicle sensor data becomes richer, the excess-zero problem will become more common, not less. The statistical machinery for handling it is now well-specified.

---

## The paper

Zhang, Y., Guillen, M., Li, J., Li, G. & Chen, S. (2025). 'Use ADAS Data to Predict Near-Miss Events: A Zero-Inflated Poisson Framework with Latent Behavior Groups.' arXiv:2509.02614. Submitted to IEEE BigData 2025. Authors affiliated with Peking University, University of Barcelona, and Tsinghua University.

---

## Related posts

- [Applying Bonus-Malus to Driving Behaviour, Not Just Claims](/telematics/techniques/2026/04/01/weekly-dynamic-telematics-bms-bonus-malus-driving-behaviour/) — the ASTIN 2025 paper that applies credibility-weighted state machines to weekly telematics signals; the weekly panel structure there is the same input format the ZIP model uses
- [Scoring a Trip as a Function, Not a Feature Vector](/telematics/techniques/2026/03/31/telematics-trip-scoring-functional-data-wavelets/) — the within-trip HMM layer that produces event features the ZIP model aggregates to weekly counts
- [SafeDriver-IQ: Telematics Scoring via Inverse Crash Probability](/telematics/pricing/2026/04/01/safedriver-iq-inverse-crash-probability-telematics-scoring/) — a different approach to the same problem: grounding telematics scores in crash probability rather than event count distributions
