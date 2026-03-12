---
layout: post
title: "The Telematics Score That Forgets Where It's Been"
date: 2026-03-12
categories: [libraries]
tags: [joint-models, longitudinal, survival, telematics, dynamic-prediction, Wulfsohn-Tsiatis, EM-algorithm, Python, motor, repricing, insurance-jlm]
description: "Your telematics GLM uses the latest score as a static covariate. Two drivers at the same score — one improving, one deteriorating — get the same premium. A joint longitudinal-survival model sees the trajectory. insurance-jlm implements Wulfsohn-Tsiatis SREM with dynamic mid-term repricing in Python."
post_number: 71
---

There are approximately one million active black box policies in the UK. Every insurer running a telematics book is doing roughly the same thing with the resulting data: taking the most recent driving score, feeding it into a GLM as a static covariate, and pricing on that.

This discards trajectory information. Two drivers at a score of 68 — one who has been at 72 for three months and is declining, one who was at 58 three months ago and is improving — present materially different claim risks. The standard GLM cannot tell them apart. It treats both as equivalent risks because it sees only the current value, not the path.

There is a second problem. The observed score is not the true latent risk. It is measured with error: sensor noise, trip sampling variability, boundary effects on the score algorithm. If you put the noisy observed score directly into a Cox model, the association parameter is attenuated — Berkson attenuation bias. You systematically underestimate how strongly the driving trajectory predicts claims.

The joint longitudinal-survival model (JLM) solves both problems at once.

---

## What a joint model actually is

The Wulfsohn-Tsiatis (1997, *Biometrics*) shared random effects model has two linked submodels.

**Longitudinal submodel** — fits the trajectory:

```
y_i(t_ij) = m_i(t_ij) + ε_ij
m_i(t_ij) = x_i(t_ij)ᵀ β + z_i(t_ij)ᵀ b_i
```

where `b_i ~ N(0, D)` is a subject-specific random effect capturing the individual's persistent latent risk profile, and `ε_ij ~ N(0, σ²)` is measurement noise. `m_i(t)` is the true underlying trajectory — not what was observed.

**Survival submodel** — links true trajectory to event time:

```
h(t | b_i) = h_0(t) exp(γᵀ w_i + α m_i(t))
```

`α` is the association parameter: does a higher true driving score predict lower claim hazard? The model estimates this while accounting for the fact that the scores are measured with error. No Berkson attenuation.

The two submodels are estimated jointly via EM with Gauss-Hermite quadrature for the intractable integral over `b_i`. The result is an estimate of α with correct standard errors.

---

## The API

```python
import pandas as pd
from insurance_jlm import JointModel

# Long-format data: one row per telematics reading per policy
model = JointModel(
    long_model='linear',         # linear trajectory in time
    surv_model='cox',            # Cox PH for time-to-first-claim
    association='current_value', # α links current true score to hazard
    n_quad_points=7,             # Gauss-Hermite quadrature points
    se_method='bootstrap',       # bootstrap SEs; Louis formula as alternative
)

result = model.fit(
    data=telematics_df,        # long format: one row per reading
    id_col='policy_id',
    time_col='month',          # measurement time in policy months
    y_col='driving_score',     # observed telematics score (0-100)
    event_time_col='claim_month',
    event_col='claimed',       # 1=claim, 0=censored
    long_covariates=['age_band', 'vehicle_class'],
    surv_covariates=['age_band', 'vehicle_class', 'ncb_years'],
)

print(model.association_summary())
#    parameter  estimate     se  z_stat  p_value
# 0  alpha       -0.041  0.009  -4.56    <0.001
```

`alpha = -0.041` means: a one-unit increase in true driving score reduces claim hazard by about 4%. This is the attenuation-corrected estimate — a naive Cox model using observed scores would have produced something like -0.027.

---

## Dynamic prediction: the actual use case

Fitting the joint model is the first step. The insurance prize is dynamic prediction: updating the claim probability for an individual policy as new readings arrive, without refitting.

```python
from insurance_jlm import DynamicPredictor

predictor = DynamicPredictor(model)

# Policy with 4 months of readings so far
history = pd.DataFrame({
    'policy_id': ['P0042'] * 4,
    'month': [1, 2, 3, 4],
    'driving_score': [72, 68, 64, 61],  # declining trajectory
    'age_band': ['25-29'] * 4,
    'vehicle_class': ['hatchback'] * 4,
})

# P(no claim in next 8 months | survived 4 months, this history)
surv_prob = predictor.predict_survival(
    data=history,
    id_col='policy_id',
    landmark_time=4,
    horizon=8,
    n_mc=500,
)
print(surv_prob)
# policy_id   P(T > 12 | T > 4)  lower_95  upper_95
# P0042       0.831               0.801     0.861
```

Call this at each monthly upload. A policy whose score declines from 72 to 61 over four months has a materially different survival probability than one that was stable at 68. The dynamic predictor quantifies that difference.

For mid-term repricing: this is the probability that feeds into the repricing decision at month 4. A standard GLM gives you a static score from the initial quote. The joint model gives you a continuously updated probability conditional on everything observed so far.

---

## The convenience function for telematics books

```python
from insurance_jlm import jlm_from_telematics

# Merge telematics readings with claims data; the function handles long-format conversion
model = jlm_from_telematics(
    telematics_df=monthly_scores,   # policy_id, month, score
    claims_df=claims,               # policy_id, claim_month, claimed
    baseline_covariates=['age_band', 'vehicle_class', 'ncb_years'],
)

# Association parameter — does trajectory predict claims?
assoc = model.association_summary()

# Dynamic risk for entire active book
from insurance_jlm import DynamicPredictor
predictor = DynamicPredictor(model)
book_risk = predictor.batch_predict(current_readings, landmark_time=policy_months)
```

---

## What we actually built

The library has 120 tests across six modules.

**models/joint\_model.py** — the EM loop. E-step uses Gauss-Hermite quadrature via `numpy.polynomial.hermite`. M-step closes over `β`, `D`, `σ²` analytically and optimises `γ`, `α` numerically via `scipy.optimize.minimize`. Convergence is monitored on the observed log-likelihood, not the complete-data version. The constraint `dim(b_i) ≤ 4` is enforced at API level because GHQ computation grows exponentially with random effects dimension: beyond dim 4, 7 quadrature points per dimension means 7⁴ = 2,401 integration nodes per subject per EM iteration.

**models/longitudinal.py** — wraps `statsmodels.MixedLM` for the LME submodel. Extracts β, D, σ² as starting values for EM, then operates in closed form during iteration.

**models/survival.py** — wraps `lifelines.CoxPHFitter` for initial γ estimates. Breslow baseline hazard estimator for the EM loop.

**prediction/dynamic.py** — `DynamicPredictor` with Monte Carlo integration over the posterior of `b_i`. At each call, draws from the posterior `p(b_i | y_i(t), T_i > t, θ̂)` via rejection sampling from the prior scaled by the likelihood. Cheap enough for batch scoring a book of 50,000 active policies.

**data/loaders.py** — `jlm_from_telematics()` and `jlm_from_ncd()`. The NCD variant treats annual NCD level (0–5) as the longitudinal marker and time-to-lapse as the event — relevant for retention modelling.

**diagnostics/** — martingale residuals for the longitudinal submodel, Cox-Snell residuals for the survival submodel, Brier score and time-dependent AUC for dynamic prediction calibration.

---

## Limitations stated plainly

**Fitting time.** At 10,000 policies with 6 readings each, fitting takes around 12 minutes on a single core. At 100,000 policies it is 1–4 hours depending on convergence. The standard mitigation: subsample to 15,000–20,000 for model selection and association estimation, validate dynamic prediction on the full held-out book. EM is parallelisable across subjects; a `n_jobs` parameter is planned for v0.2.

**Gaussian marker assumption.** Driving scores on [0, 100] are not Gaussian. NCD levels are ordinal. The standard SREM assumes a Gaussian longitudinal submodel; this is misspecified for both. The mismatch matters most for extreme scores (near 0 or 100) where the linear model predicts out of range. v0.2 will add an ordered probit longitudinal submodel for NCD.

**Random effects dimension limit.** `dim(b_i) ≤ 4` in v0.1.0. A random intercept + slope model (`dim = 2`) is fine. A random intercept + slope + acceleration model (`dim = 3`) is on the boundary of practical. Anything higher requires MCMC or variational Bayes, which are not in scope for this library.

**MNAR risk from offline devices.** A device that goes offline (battery failure, disconnection) is not missing at random — offline periods correlate with claims. Standard EM assumes data is missing at random. If offline periods in your data are systematically associated with claim risk, the association estimate will be biased. There is no clean statistical fix for this; the engineering fix is to impute offline periods with a specific risk score value and flag them.

**No competing risks in v0.1.0.** Motor telematics books have two exit events: claim and lapse. The standard SREM handles a single event. The `jmstate` package handles multi-state models; in v0.2.0 we will add a competing-risks extension for the joint lapse/claim case.

---

## Installation

```bash
pip install insurance-jlm
```

Dependencies: `statsmodels`, `lifelines`, `scipy`, `numpy`. No PyTorch. No Stan. Pure Python with a standard scientific stack.

Source and issue tracker: [github.com/burning-cost/insurance-jlm](https://github.com/burning-cost/insurance-jlm)

---

## See also

- [HMM-Based Telematics Risk Scoring](https://burning-cost.github.io/2026/03/10/insurance-telematics/) — insurance-telematics produces the driving score time series that feeds into insurance-jlm as the longitudinal marker
- [Survival Models for Insurance Retention](https://burning-cost.github.io/2026/03/11/survival-models-for-insurance-retention/) — insurance-survival handles standard time-to-event modelling without the longitudinal component
- [Borrowing Experience You Don't Have](https://burning-cost.github.io/2026/03/12/borrowing-experience-you-dont-have/) — insurance-transfer for thin-segment pricing when your telematics book lacks enough claims for stable JLM estimation
- [Mixture Cure Models](https://burning-cost.github.io/2026/03/11/insurance-cure/) — insurance-cure for the non-claimer subpopulation; can be combined with JLM for a full latent class model

---

*insurance-jlm v0.1.0 — 120 tests, 6 modules. Implements the Wulfsohn-Tsiatis (1997) SREM with EM-GHQ estimation and landmark dynamic prediction.*
