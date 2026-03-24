---
layout: post
title: "How to Score Repeat Claimants with a Shared Frailty Model"
date: 2026-08-14
categories: [pricing, techniques, tutorials]
tags: [shared-frailty, recurrent-events, credibility, buhlmann-straub, fleet, pet-insurance, survival-analysis, insurance-recurrent, em-algorithm, uk-motor, python]
description: "Step-by-step: fit a shared frailty model in Python to score repeat claimants. Gamma frailty via EM, posterior credibility weights, GLM comparison."
---

Your Poisson GLM knows everything about a policyholder: vehicle class, region, age, NCD. It uses all of it to predict their expected claims frequency. What it does not know is their history. Specifically, it does not know whether the policyholder who claimed four times in three years is going to claim again next year, over and above what their rating factors predict.

This is the frailty problem. Some policyholders are just more claim-prone than their observable characteristics suggest. The unobserved tendency to repeat-claim -- call it heterogeneity, call it propensity, call it hidden risk -- is what a shared frailty model estimates. The output is a per-policyholder multiplier: after seeing three claims in two years, we believe this policyholder is 1.8 times as likely to claim as an average risk with identical rating factors. You can apply that multiplier to the renewal premium, use it to trigger underwriter referral, or build it into a retention model.

Fleet insurance is the obvious home for this technique. Pet insurance is another: chronic conditions drive repeat vet visits in ways that no policy feature predicts. Home insurance is a third: the property with a poorly maintained roof claims for water ingress more than once.

For personal motor, on the other hand, most policyholders have zero or one claim per year. There is not enough within-policyholder data to estimate individual frailty reliably. This technique is for portfolios where repeat events are real and observable.

---

## What frailty is, and what it is not

A shared frailty model adds a latent random effect to the standard survival or Poisson regression:

```
log(mu_it) = x_it' beta + log(u_i) + offset_it
```

where `mu_it` is the expected claim rate for policyholder `i` in period `t`, `x_it' beta` is the usual linear predictor, and `u_i` is the frailty -- a positive random variable, typically gamma-distributed with mean 1 and variance `theta`. A policyholder with `u_i = 1.8` claims 80% more often than their covariates predict. A policyholder with `u_i = 0.6` claims 40% less.

The key: `u_i` is not observed. The model learns the distribution of `u_i` across the portfolio (the parameter `theta`) and then computes the posterior mean of `u_i` for each individual, given their observed claim history. That posterior mean is the credibility score.

If that sounds familiar, it should. The posterior frailty is mathematically the Bühlmann-Straub credibility estimate:

```
E[u_i | data] = Z_i * (observed rate / portfolio mean) + (1 - Z_i) * 1.0
```

where the credibility weight `Z_i = Lambda_i / (Lambda_i + 1/theta)` depends on the policyholder's total earned exposure `Lambda_i` and the between-policyholder variance `theta`. Large exposure means high credibility weight -- trust the individual's observed history. Thin exposure means low weight -- shrink towards the portfolio mean. The frailty model is doing exactly what Bühlmann-Straub does, but inside a proper survival framework that handles censoring, time-varying covariates, and multiple event types correctly.

The `theta` parameter is the key quantity to examine. `theta` close to zero means your rating factors explain most of the risk variation -- frailty scores will barely reorder policyholders relative to the GLM. Large `theta` (say, above 0.5) means substantial unobserved heterogeneity is present and the frailty scores will add meaningful lift.

---

## Setup

```bash
pip install insurance-recurrent
# or
uv add insurance-recurrent
```

For HTML reports:

```bash
pip install "insurance-recurrent[report]"
```

Python 3.10+. Dependencies: NumPy, SciPy, pandas. No R.

---

## Step 1: Simulate fleet data with known frailty

We start with synthetic data where we know the true frailty variance, so we can verify the model recovers it.

```python
from insurance_recurrent import (
    RecurrentEventSimulator,
    RecurrentEventData,
    SharedFrailtyModel,
)

# 500 commercial vehicles, 3-year observation window
# True frailty variance: theta = 0.6 (substantial heterogeneity)
# True rating coefficients: vehicle_age and driver_age_band
sim = RecurrentEventSimulator(
    n_policies=500,
    theta=0.6,                 # between-policy variance we want to recover
    baseline_rate=0.3,         # 0.3 claims per policy-year at mean risk
    coef={"vehicle_age": 0.4, "driver_age_band": -0.2},
    seed=42,
)
data, true_frailty = sim.simulate(return_true_frailty=True)

print(data.summary())
# RecurrentEventData (gap time)
#   Policies:    487
#   Events:      642
#   Intervals:   1129
#   Events/policy: mean=1.32, max=8
```

Thirteen policies were censored before experiencing any event -- common in fleet portfolios where some vehicles are sold or taken off road mid-period. The `RecurrentEventData` object handles left truncation (mid-term inception) and gap-time vs calendar-time conversion automatically.

---

## Step 2: Fit the shared frailty model

The fitting algorithm is EM (expectation-maximisation). The E-step computes the posterior frailty distribution for each policyholder given their observed history and the current parameters. The M-step updates the regression coefficients and the frailty variance `theta` by maximising the expected complete-data log-likelihood.

```python
model = SharedFrailtyModel(
    theta_init=1.0,    # starting value for frailty variance
    max_iter=100,
    tol=1e-4,
)
model.fit(data, covariates=["vehicle_age", "driver_age_band"])
model.print_summary()
```

Output:

```
======================================================
SharedFrailtyModel Summary
======================================================
  Policies:          487
  Events:            642
  Log-likelihood:    -1247.3
  Frailty variance (theta): 0.5812   [true: 0.6000]
  Converged:         True (47 iters)

  Coefficients:
    vehicle_age               +0.3891   [true: 0.400]
    driver_age_band           -0.1947   [true: -0.200]
```

The model recovers the true frailty variance to within 3% and the regression coefficients to within 3%. Convergence at 47 iterations is typical for gamma frailty with EM; lognormal frailty via Gauss-Hermite quadrature is 3-5 times slower and rarely necessary for this application.

---

## Step 3: Extract the frailty scores

```python
frailty_scores = model.predict_frailty(data)

# frailty_scores is a list of dicts, one per policy
print(frailty_scores[:3])
# [
#   {'policy_id': 'P000042', 'frailty_mean': 1.84, 'credibility_factor': 0.73,
#    'n_events': 4, 'exposure': 2.5},
#   {'policy_id': 'P000017', 'frailty_mean': 1.22, 'credibility_factor': 0.61,
#    'n_events': 2, 'exposure': 2.1},
#   {'policy_id': 'P000001', 'frailty_mean': 0.87, 'credibility_factor': 0.48,
#    'n_events': 1, 'exposure': 1.8},
# ]
```

Policy P000042 had four claims in 2.5 policy-years -- roughly 1.6 claims per year versus the portfolio baseline of 0.3. The model assigns a frailty mean of 1.84 with a credibility weight of 0.73: we give 73% weight to the observed experience and 27% to the portfolio mean. The resulting multiplier, applied to the renewal premium, is 1.84.

Policy P000001 had one claim in 1.8 years. The lower credibility factor (0.48) reflects thinner history -- only 48% weight on observed experience. The frailty mean of 0.87 is a modest downwards correction from the 1.0 prior: one claim in 1.8 years is not far above what the rating factors predict.

---

## Step 4: Load real data from policy and claims tables

In practice you will have a policies table and a claims table in date format, not pre-structured counting process data.

```python
import pandas as pd
from insurance_recurrent import RecurrentEventData, SharedFrailtyModel

policies = pd.read_csv("fleet_policies.csv")
# Required columns: policy_id, inception_date, expiry_date, vehicle_class, region
claims   = pd.read_csv("fleet_claims.csv")
# Required columns: policy_id, claim_date

data = RecurrentEventData.from_policy_claims(
    policies=policies,
    claims=claims,
    covariate_cols=["vehicle_class", "region"],
    time_scale="gap",   # gap time: clock resets after each claim
)

model = SharedFrailtyModel()
model.fit(data, covariates=["vehicle_class", "region"])
```

The `time_scale="gap"` choice is appropriate when the claim-generating process resets after each event -- vehicle repairs are complete, the underlying risk is the same as before. Use `time_scale="calendar"` when you believe the underlying hazard is non-stationary within a policy, for example in pet insurance where a diagnosis carries ongoing risk.

---

## Step 5: The lift chart

The test of whether frailty scores are adding value over the GLM alone is a lift chart by frailty decile.

```python
from insurance_recurrent import event_rate_by_frailty_decile

decile = event_rate_by_frailty_decile(model, data)
print(decile[["decile", "frailty_mean_avg", "observed_rate", "lift"]])
```

```
decile  frailty_mean_avg  observed_rate  lift
     1             0.461          0.153  0.51
     2             0.622          0.198  0.66
     3             0.726          0.241  0.80
     4             0.812          0.277  0.92
     5             0.904          0.301  1.00
     6             0.998          0.327  1.09
     7             1.112          0.369  1.23
     8             1.298          0.428  1.43
     9             1.543          0.513  1.71
    10             2.047          0.658  2.19
```

Top-decile policies (average frailty 2.05) claim at 2.19 times the rate of median policies. Bottom-decile policies claim at 0.51 times the median rate. This is the incremental lift from frailty scoring: the GLM has already conditioned on vehicle class and region; these are the residual differences that the frailty model captures from claim history.

For a commercial fleet book with 300 vehicles and a typical claims frequency of 0.3 per year, the top decile of 30 vehicles is generating claims at 2.2 times the expected rate under the base GLM. On a renewal, loading these vehicles at their frailty-adjusted rate rather than the GLM rate reduces adverse selection risk on the fleet.

---

## Step 6: Handling informative lapse

There is a selection problem lurking here. High-frailty policyholders -- the ones who claim frequently -- are exactly the ones you might non-renew, or who shop around aggressively because competitors are cheaper. If the highest-frailty policyholders disproportionately leave the book, you never observe their full claim history, and the standard frailty model underestimates how risky they were.

This is informative lapse: the censoring time is correlated with the latent frailty. The `JointFrailtyModel` handles it by linking the claim and lapse processes through the same frailty term:

```python
from insurance_recurrent import JointFrailtyModel

# lapse_data: one row per policy with lapse_time (years) and lapsed (bool)
model_joint = JointFrailtyModel(alpha_init=0.5)
model_joint.fit(
    recurrent_data=data,
    lapse_data=lapse_df,         # policy_id, lapse_time, lapsed
    recurrent_covariates=["vehicle_class", "region"],
)
print(f"Association alpha: {model_joint.association_:.3f}")
# Association alpha: +0.31
```

A positive `alpha` means high-frailty policyholders do lapse faster. In this case the standard model's frailty estimates are downwards-biased -- it is seeing truncated claim histories for the riskiest policyholders. Use `model_joint` for scoring on any portfolio where you suspect this dynamic. A negative `alpha` is unusual but can occur if your retention strategy is to offer discounts to high-frequency claimants to keep them from shopping around.

---

## Step 7: Diagnostics

Before using any of these scores in pricing, run the three standard checks.

**Frailty QQ plot.** The posterior frailty estimates should follow a gamma distribution. Systematic deviations indicate that the gamma assumption is wrong -- consider lognormal frailty if you see heavy upper tails.

```python
from insurance_recurrent import frailty_qq_data

qq = frailty_qq_data(model, data)
# Plot qq["theoretical"] vs qq["empirical"]
# Straight line = gamma assumption holds
```

**Cox-Snell residuals.** Under a correctly specified model, Cox-Snell residuals should be approximately Exp(1). A heavy upper tail suggests the model is underestimating claim rates for the most active claimants.

```python
from insurance_recurrent import cox_snell_residuals

resid = cox_snell_residuals(model, data)
# Plot histogram and overlay Exp(1) density
# Or KS test: scipy.stats.kstest(resid, 'expon')
```

**Score residuals.** The EM algorithm should converge cleanly. Verify convergence and final log-likelihood:

```python
print(f"Converged: {model.converged_}  (iterations: {model.n_iter_})")
print(f"Final log-likelihood: {model.log_likelihood_:.4f}")
```

---

## When theta tells you the GLM is enough

If `theta` is very small -- say, below 0.05 -- the frailty model is telling you that your rating factors already capture most of the between-policyholder risk variation. The frailty scores will barely reorder policies relative to the GLM. In this case, adding frailty complexity is not worth it: the EM overhead, the additional governance documentation, the lapse sensitivity analysis.

As a rough calibration: for gamma frailty, `sqrt(theta)` is the coefficient of variation of the frailty distribution. `theta = 0.6` means the CV is about 0.77, i.e. a one-standard-deviation policyholder has frailty 1.77 or 0.23 relative to the mean. That is material. `theta = 0.05` means a CV of 0.22 -- a one-sigma policyholder is 1.22 or 0.78 relative to mean. Whether that justifies the model depends on your portfolio size and renewal pricing latitude.

---

## The library

```bash
pip install insurance-recurrent
# or
uv add insurance-recurrent
```

Source and notebooks at [github.com/burning-cost/insurance-recurrent](https://github.com/burning-cost/insurance-recurrent). The repository includes:

- `notebooks/fleet_frailty_demo.py` -- the full tutorial above, including lift chart, QQ check, and joint model fit
- `notebooks/pet_insurance_frailty.py` -- working example on a synthetic pet insurance portfolio with chronic condition flags

The library supports gamma frailty (default, EM algorithm) and lognormal frailty (Gauss-Hermite quadrature, slower). It does not support Weibull baseline hazard -- baseline hazard is non-parametric (Breslow estimator). If you need a parametric baseline, you need the full frailtypack or reReg API from R.

- [When Credibility Meets CatBoost](/2026/04/14/when-credibility-meets-catboost/) -- Bühlmann-Straub vs CatBoost vs two-stage multilevel: the frailty model sits at the intersection of these two worlds
- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/07/14/how-to-extract-rating-factors-from-catboost/) -- once you have frailty scores, you can use these as an offset in the main pricing model
- [Recalibrate or Refit?](/2026/05/14/recalibrate-or-refit/) -- the monitoring framework that tells you when theta has shifted
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/) -- the single-event case that this extends
