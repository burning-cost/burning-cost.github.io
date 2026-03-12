---
layout: post
title: "Treating Competing Risks as Censored Is Biasing Your Retention and Home Insurance Pricing"
date: 2026-03-12
categories: [libraries, pricing, survival-analysis]
tags: [competing-risks, Fine-Gray, CIF, subdistribution-hazard, retention, home-insurance, Gray-test, IPCW, insurance-competing-risks, python]
description: "Standard survival analysis treats competing events as censored observations — which produces cumulative incidence functions that sum to more than one and cause-specific hazard estimates that overstate the probability of any single event. Fine-Gray (1999 JASA) is the correct method. insurance-competing-risks is the first Python implementation with peril-labelled insurance APIs."
post_number: 80
---

Retention modelling treats policy lapse as the event of interest. Customers who move to a different cover type before lapsing, or who the insurer non-renews for underwriting reasons, are typically treated as censored — as if they disappeared from the analysis for unknown reasons.

That is wrong. When a customer MTC (moves to cheaper cover), that is not a random censoring event independent of the lapse hazard. Customers most likely to lapse are also most likely to MTC. Treating MTC as censored overstates the lapse hazard for the customers who remain.

The same problem appears in home insurance peril pricing. A home policy that ends due to escape of water before a fire claim is not censored in the statistical sense — it ended for a competing reason. Standard Kaplan-Meier and Cox models produce CIFs that sum to more than one in the presence of competing risks. Fine-Gray (1999 JASA) is the correct method.

[`insurance-competing-risks`](https://github.com/burning-cost/insurance-competing-risks) is the first Python implementation designed for insurance, with peril-labelled APIs for both retention and home peril applications.

```bash
uv add insurance-competing-risks
```

## The competing risks problem

In classical survival analysis, the Kaplan-Meier estimator produces the cumulative incidence function (CIF) for an event by treating competing events as censored. The problem:

1. If censoring is informative (correlated with the event hazard), the estimator is biased
2. Cause-specific KM CIFs sum to more than 1.0 — a mathematical impossibility for probabilities
3. Regression coefficients from cause-specific Cox models do not have causal interpretations under competing risks

Fine and Gray (1999 JASA) introduced the subdistribution hazard, which directly models the probability of the event of interest in the presence of competing events. The subdistribution hazard for cause $j$ is:

$$\lambda_j^*(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t, J = j \mid T \geq t \text{ or } (T < t \text{ and } J \neq j))}{\Delta t}$$

Risk set: subjects who are event-free, plus those who experienced a competing event. This is the key difference from cause-specific hazards, and it ensures the CIFs sum to at most one.

## Retention: competing events for policy lifecycle

```python
from insurance_competing_risks import RetentionCompetingRisks

# Events: 1=lapse, 2=MTC, 3=NTU, 0=censored/renewed
crm = RetentionCompetingRisks(
    event_col="event_type",
    event_map={
        "lapse": 1,
        "mtc": 2,
        "ntu": 3
    },
    duration_col="policy_days"
)
crm.fit(X_train, events_train, durations_train)

# CIF for lapse at 12 months — accounting for MTC and NTU as competing events
cif_lapse = crm.predict_cif(X_test, cause=1, times=[365])

# Direct Fine-Gray regression coefficients
print(crm.summary(cause=1))
```

The `RetentionCompetingRisks` API explicitly labels the event types in insurance terminology. The underlying estimation uses the Fine-Gray subdistribution weighted partial likelihood.

## Home insurance: competing perils

```python
from insurance_competing_risks import PerilCompetingRisks

# Events: 1=fire, 2=flood, 3=escape_of_water, 4=subsidence, 0=no claim
crm = PerilCompetingRisks(
    event_col="first_peril",
    peril_map={
        "fire": 1,
        "flood": 2,
        "escape_of_water": 3,
        "subsidence": 4
    },
    duration_col="days_to_first_claim",
    exposure_col="earned_days"
)
crm.fit(X_train, events_train, durations_train)

# CIF curves per peril
cif_curves = crm.predict_cif_curves(X_test, times=np.arange(0, 366))
```

## Gray test for covariate significance

```python
from insurance_competing_risks import GrayTest

gray = GrayTest()
result = gray.test(
    durations=df["policy_days"],
    events=df["event_type"],
    groups=df["vehicle_age_band"],
    cause=1  # lapse
)
print(f"Gray test statistic: {result.statistic:.3f}, p={result.p_value:.4f}")
```

The Gray test is the non-parametric analogue of the log-rank test for subdistribution hazards. Use it before fitting a regression model to check whether covariates have any discriminating power.

## IPCW calibration

Inverse probability of censoring weighting (IPCW) calibration extends the standard Brier score to competing risks. It measures how well your CIF predictions are calibrated.

```python
from insurance_competing_risks import IPCWBrier, IPCWCalibration

# IPCW Brier score for cause 1 at 12 months
brier = IPCWBrier(cause=1, time=365)
score = brier.score(
    predicted_cif=crm.predict_cif(X_test, cause=1, times=[365]),
    durations=durations_test,
    events=events_test
)
print(f"IPCW Brier score: {score:.4f}")

# Calibration curve
cal = IPCWCalibration(cause=1, time=365, n_bins=10)
cal.plot(predicted_cif, durations_test, events_test)
```

## Why lifelines and scikit-survival don't cover this

lifelines has a Cox proportional hazards model and Kaplan-Meier. It does not implement Fine-Gray. The lifelines maintainers closed the request (#1498) noting it was on the roadmap but resources were constrained.

scikit-survival has a Fine-Gray estimator as of v0.22 but it is limited to univariate estimation — no regression, no covariate conditioning, no IPCW diagnostics. PyMSM is designed for clinical multi-state models with treatment coding assumptions that do not transfer to insurance.

`insurance-competing-risks` implements the full workflow: Gray test → Fine-Gray regression → CIF prediction → IPCW calibration, with insurance-specific peril and retention APIs.

---

`insurance-competing-risks` is available on [PyPI](https://pypi.org/project/insurance-competing-risks/) and [GitHub](https://github.com/burning-cost/insurance-competing-risks). 150 tests. Databricks notebook demonstrates both retention and home peril applications.

Related: [insurance-survival](https://burning-cost.github.io/2026/02/27/experience-rating-ncd-bonus-malus/) for standard survival modelling, [insurance-recurrent](https://burning-cost.github.io/2026/03/12/insurance-recurrent/) for repeat claimers, [insurance-uplift](https://burning-cost.github.io/2026/03/11/insurance-uplift/) for retention HTE.
