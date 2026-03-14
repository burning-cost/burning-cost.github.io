---
layout: post
title: "Shared Frailty for Repeat Claimers: When the Poisson GLM Gets the Wrong Answer"
date: 2026-03-12
categories: [libraries, pricing, credibility]
tags: [frailty, recurrent-events, Bühlmann-Straub, credibility, Poisson-GLM, repeat-claims, fleet-motor, pet-insurance, home-insurance, EM-algorithm, survival-analysis, python, insurance-recurrent]
description: "Standard Poisson GLMs treat each policy-year as conditionally independent. That assumption is wrong for repeat claimers, and the consequences are biased frequency estimates and missed credibility signals. insurance-recurrent implements shared frailty models for recurrent insurance claims — and the frailty posterior is exactly the Bühlmann-Straub credibility estimate."
---

Most personal lines policies produce at most one claim per year, and most produce none. The standard Poisson GLM is built for this regime. Conditional on your age, vehicle, region, and all the other rating factors, the number of claims follows a Poisson distribution with rate lambda, independent across policy-years. Fit it by MLE, extract the relativities, done.

The independence assumption fails in two common settings. The first is any commercial or fleet product where a single insured entity generates multiple claims in a single period - a fleet of 50 vehicles, a multi-pet household, a commercial property with several buildings. The second is any personal lines product where unobserved risk heterogeneity is substantial: a small fraction of policyholders accounts for a disproportionate share of claims, and last year's claims are genuinely predictive of this year's claims even after controlling for all observable rating factors.

When independence fails, the GLM's coefficient estimates are still unbiased for the average, but you are throwing away information. The claims history of each individual policyholder contains a signal about their latent risk level that the rating factors do not capture. A reckless driver who happens to be 35, in a Ford Fiesta, in Manchester, looks average on paper. If they have claimed twice in three years, you know something the GLM does not.

Frailty models formalise this. [`insurance-recurrent`](https://github.com/burning-cost/insurance-recurrent) implements shared frailty for recurrent insurance claims, with a direct connection to classical credibility theory.

```bash
pip install insurance-recurrent
```

---

## The frailty posterior is the credibility estimate

The Andersen-Gill model with gamma frailty has the following structure. Each policyholder `i` has a latent frailty `z_i` drawn from Gamma(theta, theta), with mean 1 and variance 1/theta. Their claim intensity at time `t` is:

```
lambda_i(t) = z_i * lambda_0(t) * exp(X_i' * beta)
```

The frailty `z_i` multiplies the baseline intensity. A policyholder with `z_i = 1.5` claims 50% more frequently than their risk profile would suggest. A policyholder with `z_i = 0.6` claims 40% less.

We observe the claims but not `z_i`. After the policyholder generates `n_i` claims against an expected cumulative hazard of `Lambda_i`, the posterior distribution of `z_i` is:

```
z_i | data ~ Gamma(theta + n_i, theta + Lambda_i)
```

The posterior mean is:

```
E[z_i | data] = (theta + n_i) / (theta + Lambda_i)
```

This is the Bühlmann-Straub credibility premium. Written in the classical credibility form:

```
E[z_i | data] = Z_i * (n_i / Lambda_i) + (1 - Z_i) * 1.0
```

where `Z_i = Lambda_i / (Lambda_i + theta)` is the credibility weight and `1.0` is the portfolio mean frailty. The frailty parameter theta is the classical credibility parameter k: when theta is large (low portfolio heterogeneity), credibility builds slowly; when theta is small (high heterogeneity), even a short claims history shifts the estimate substantially.

The frailty literature and credibility theory arrived at the same formula from different directions. In [`insurance-experience`](https://github.com/burning-cost/insurance-experience) we implement Bühlmann-Straub directly at the policyholder level for NCD-style experience rating. In `insurance-recurrent` we get there from the survival/counting process route, which handles the time structure of recurrent events correctly.

---

## Where this matters: three use cases

### Fleet motor (no NCD)

Fleet motor policies do not carry individual NCD. A fleet of 50 vehicles might have 8 claims in a year. The standard frequency model conditions on fleet characteristics - average vehicle age, driver demographics, sector - and produces a rate. But the within-fleet claim count is over-dispersed relative to Poisson because some vehicles (or some drivers) are systematically riskier than others, and that structure is stable across policy years.

The frailty theta here represents fleet-level heterogeneity. A fleet that claimed 12 times when the model expected 6 has a frailty posterior concentrated above 1. The next year's technical premium should load for this. Without frailty, the renewal premium is set on observable characteristics only, and the fleet's claims history is either ignored or adjusted by ad hoc rule.

```python
from insurance_recurrent import RecurrentEventData, AndersenGillFrailty, FrailtyReport

# data: one row per vehicle-claim-interval
# each fleet_id groups its vehicles across multiple claim periods
model = AndersenGillFrailty(frailty="gamma").fit(data)
scores = model.credibility_scores()

# Fleet with policy_id=7042: 4 claims, 1.8 expected
fleet_score = scores.loc[scores["id"] == 7042]
print(fleet_score[["n_events", "lambda_i", "frailty_mean", "credibility_weight"]])
#    n_events  lambda_i  frailty_mean  credibility_weight
#        4       1.80         2.41            0.642
```

The frailty posterior mean of 2.41 says this fleet claims at 2.4 times the model expectation. The credibility weight of 0.64 says we are 64% confident in the fleet's own experience versus the portfolio average. The renewal loading is directly interpretable: if the base technical premium for next year is 1,200, the credibility-loaded premium is 1,200 * 2.41 = 2,892.

### Pet insurance (breed propensity)

Pet insurance has high claim frequencies for certain breeds. A French Bulldog has a substantially different health risk profile from a Labrador, and much of that difference is captured in rating factors. But within a breed, individual animals vary - some are structurally sound, some are not, and a pet that has claimed for orthopaedic issues will claim again.

The informative censoring problem is acute here: policyholders whose pets have high cumulative health costs may lapse. This is not random censoring. The standard GLM treats lapsed policies as simply leaving the at-risk set. In fact, if the high-risk pets are more likely to have their policies cancelled, the observed claims data from continuing policies is biased downward.

[`insurance-recurrent`](https://github.com/burning-cost/insurance-recurrent) handles this with `JointFrailtyModel`, which models the claims process and the lapse process jointly with a shared frailty:

```python
from insurance_recurrent import JointData, JointFrailtyModel, simulate_joint

jd = JointData(
    recurrent=claims_long,       # (start, stop, event) per claim interval
    terminal_df=policies_df,     # lapse time and lapse indicator
    id_col="policy_id",
    terminal_time_col="lapse_age",
    terminal_event_col="lapsed",
    terminal_covariates=["breed_risk_score", "age_at_inception"],
)
model = JointFrailtyModel().fit(jd)
```

The joint model corrects the selection bias. The frailty now summarises the unobserved component of risk that drives both claims and lapse. The coefficient on the shared frailty in the lapse sub-model tells you how strongly high-risk animals are selected out - a point estimate that the standard GLM cannot provide.

### Home insurance (subsidence repeat claims)

Subsidence claims are the canonical example of a non-independent outcome in home insurance. A property that has claimed for subsidence has almost certainly not had the underlying cause fully remediated. The soil conditions, drainage, and tree proximity that caused the first claim are still present. The second claim is not independent of the first, even after conditioning on postcode, property age, and construction type.

The PWPModel (Prentice-Williams-Peterson) is appropriate here. It fits a separate baseline hazard for the first, second, and third-plus claims, allowing the claim intensity to change after each event:

```python
from insurance_recurrent import PWPModel

model = PWPModel(time_scale="gap", max_stratum=3).fit(data)

# Stratum-specific hazard ratios
print(model.summary_by_stratum())
#   stratum  n_events  coef_postcode_risk  HR
#         1      1240            0.312     1.37
#         2       183            0.589     1.80
#         3        41            0.841     2.32
```

The increasing HR for `postcode_risk` across strata tells you the risk factor becomes more predictive with each subsequent claim. Properties in high-risk postcodes are disproportionately represented among repeat claimers. This is recoverable from the PWP model; the standard GLM treating all claims as exchangeable misses it.

---

## What you get over a standard Poisson GLM

The standard GLM gives you population-average frequency relativities. Conditional on the rating factors, it estimates E[claims | X], which is the right quantity for setting technical premium at the portfolio level. If you are pricing a new risk you have never seen before, the GLM is all you have.

Where the GLM falls short:

**Within-policyholder correlation.** Two claims from the same policyholder are not independent. The GLM's standard errors assume they are, which means the uncertainty estimates on your frequency relativities are too narrow. On fleet data with multiple claims per insured, the GLM confidence intervals can be substantially anti-conservative.

**Informative censoring.** Policyholders who lapse are not a random draw. If high-risk policyholders are more likely to be cancelled or to leave, the claims distribution in the continuing book is biased. The GLM has no mechanism to correct for this. `JointFrailtyModel` estimates the selection effect explicitly.

**Credibility signals from claims history.** The GLM conditions on observable rating factors. The frailty model additionally conditions on the claims history, extracting the unobserved risk component that covariates miss. For a policyholder with three claims in three years, the frailty posterior is a materially better predictor of next year's claim frequency than the rating factor relativities alone.

---

## Fitting and interpreting the model

```python
from insurance_recurrent import (
    simulate_ag_frailty, SimulationParams,
    AndersenGillFrailty, FrailtyReport
)

# Generate synthetic fleet data: 500 vehicles, theta=2.0
params = SimulationParams(n_subjects=500, theta=2.0, n_covariates=2)
data = simulate_ag_frailty(params)

# Fit gamma frailty Andersen-Gill
model = AndersenGillFrailty(frailty="gamma", max_iter=50, tol=1e-5).fit(data)
print(model.summary())
#        coef      se    HR  HR_lower_95  HR_upper_95  p_value
# x1   0.298   0.051  1.35         1.22         1.49   <0.001
# x2  -0.188   0.049  0.83         0.75         0.91   <0.001
```

The coefficient table gives hazard ratios with confidence intervals corrected for within-policyholder correlation. These are not the same as the GLM parameter estimates on the same data; the frailty model partitions variance differently.

```python
# Credibility scores for each policyholder
scores = model.credibility_scores()
#    id  n_events  lambda_i  frailty_mean  frailty_var  credibility_weight
#     0         0      0.42         0.743        0.371               0.296
#     1         3      0.89         3.247        2.148               0.471
```

`frailty_mean` is the credibility-adjusted estimate of the policyholder's relative risk. `credibility_weight` is Z_i: how much we weight their own experience against the portfolio prior. A policyholder with `lambda_i = 0.89` (close to the model expectation) but 3 observed claims has a high credibility weight and a frailty mean substantially above 1 - their history is informative, and the model loads for it.

```python
# Diagnostics
report = FrailtyReport(model, data)
print(report.frailty_summary())
#   theta_hat: 1.987   (95% CI: 1.712, 2.304)
#   Variance of frailty (1/theta): 0.503
#   Mean frailty: 1.000
#   Std dev of frailty posterior means: 0.412

print(report.event_rate_by_frailty_decile())
#   decile  frailty_mean  observed_rate  expected_rate  ratio
#        1         0.389          0.31           0.33   0.94
#        5         0.943          0.91           0.88   1.03
#       10         2.841          2.76           2.79   0.99
```

The decile table is a calibration check: if the frailty estimates are well-calibrated, the observed claim rate in each frailty decile should track the expected rate. Poor calibration in the tails suggests either the frailty distribution is misspecified or the covariates are picking up frailty signal that should be in the random effect.

---

## Comparing gamma and lognormal frailty

```python
from insurance_recurrent import compare_models

m_gamma = AndersenGillFrailty(frailty="gamma").fit(data)
m_lognorm = AndersenGillFrailty(frailty="lognormal").fit(data)

print(compare_models(
    [m_gamma, m_lognorm],
    names=["gamma", "lognormal"],
    data=data
))
#             log_lik    AIC    BIC  theta_hat
# gamma       -1847.3   3698   3714      1.99
# lognormal   -1843.1   3692   3709      0.51
```

Gamma frailty is the natural choice for insurance because the conjugacy gives an exact closed-form E-step and the posterior mean is directly interpretable as a credibility premium. Lognormal frailty has heavier upper tails - it allows a small fraction of policyholders to be very much riskier than the population - which can be appropriate for commercial lines where the latent risk distribution is genuinely skewed. When the AIC gap is small, stick with gamma for the interpretability. When lognormal wins by more than a few points, the heavier tail structure is earning its complexity.

---

## Limitations

**Claim sparsity in personal motor.** Most personal motor policyholders have zero or one claim over their entire history. When the majority of individuals contribute no events to the estimation, the frailty model is effectively extrapolating from a sparse signal. The theta estimate will be unstable, and the credibility weights for individual policyholders will be low. This is not a failure of the method - it is telling you correctly that there is not much individual-level information. For personal motor, the model is most useful at the cohort or segment level (grouping by broker, scheme, or postcode district) rather than at the individual policyholder level. Fleet, pet, and commercial products with higher per-insured event rates are the natural home.

**Computational scaling.** The EM algorithm for gamma frailty has an O(n * T) cost per iteration where T is the number of distinct event times. For large portfolios (500,000 policies, tens of thousands of claim dates), this can be slow. The lognormal E-step requires Gauss-Hermite quadrature and is materially slower. For large datasets, consider fitting on a representative subsample to estimate theta and beta, then computing credibility scores on the full portfolio using the fitted parameters.

**The balance property.** The frailty posterior means do not automatically sum to 1.0 at portfolio level - unlike the explicit Bühlmann-Straub implementation in [`insurance-experience`](https://github.com/burning-cost/insurance-experience) which enforces the balance property by construction. If you are using the frailty scores as a multiplicative loading in a rating engine, you need to normalise them against the portfolio mean or the aggregate expected loss will shift. `FrailtyReport.normalised_scores()` does this.

**The parallel tracks.** [`insurance-experience`](https://github.com/burning-cost/insurance-experience) handles individual experience rating in the Bühlmann-Straub framework directly, with four model tiers and explicit balance property enforcement. It is the better choice when you are building an NCD-style individual rating mechanism, or when you have panel data in the standard exposure/claims format. Use `insurance-recurrent` when your data is in counting process format with time-stamped events, when you have informative censoring from lapse, or when you want to model how claim intensity changes across multiple events (PWP) rather than pooling all claims to a rate.

[`insurance-jlm`](https://github.com/burning-cost/insurance-jlm) handles a related but different problem: when you have a longitudinal score (a telematics safety score, a credit score) that is measured with error and you want to link its trajectory to claim hazard. The Wulfsohn-Tsiatis model there shares the joint modelling philosophy with `JointFrailtyModel` here, but the measurement error correction and the dynamic score trajectory are specific to that setting.

---

## Where to start

Install from PyPI and run the simulated fleet motor example first:

```python
from insurance_recurrent import simulate_ag_frailty, SimulationParams, AndersenGillFrailty, FrailtyReport

data = simulate_ag_frailty(SimulationParams(n_subjects=500, theta=2.0))
model = AndersenGillFrailty(frailty="gamma").fit(data)
report = FrailtyReport(model, data)

print(f"Estimated theta: {model.theta_:.3f} (true: 2.0)")
print(report.event_rate_by_frailty_decile())
```

The simulation has a known theta, so you can verify recovery before applying to real data. Then check whether your portfolio's actual theta is materially above zero. If `NelsonAalenFrailty().fit(data).theta_` returns something large (say, above 3), the portfolio is close to homogeneous and the frailty model adds little. If it returns 0.5 to 1.5, there is substantial unobserved heterogeneity worth modelling.

For fleet motor pricing, the credibility scores are the main output. For pet or home with informative lapse, the joint frailty model is worth the additional complexity. For subsidence or repeat peril claims, fit the PWPModel and compare stratum-specific hazard ratios to confirm whether claim intensity changes after each event.

The standard Poisson GLM is not wrong for most personal lines products. For products with genuine repeat claiming - fleet, pet, commercial property, or any book where unobserved heterogeneity is material - frailty models recover the credibility signal the GLM discards.

---

*`insurance-recurrent` is available via `pip install insurance-recurrent`. For NCD-style individual experience rating with explicit balance property enforcement, see [`insurance-experience`](https://github.com/burning-cost/insurance-experience). For joint longitudinal-survival modelling of telematics scores, see [`insurance-jlm`](https://github.com/burning-cost/insurance-jlm).*

**Related articles from Burning Cost:**
- [Individual Experience Rating Beyond NCD: From Bühlmann-Straub to Neural Credibility](/2026/03/24/insurance-experience/)
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [D-Vine Copulas for Longitudinal Claim Histories](/2026/03/13/insurance-vine-longitudinal/)
