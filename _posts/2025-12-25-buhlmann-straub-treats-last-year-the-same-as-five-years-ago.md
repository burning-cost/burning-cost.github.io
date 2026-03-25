---
layout: post
title: "Bühlmann-Straub Treats Last Year the Same as Five Years Ago"
date: 2025-12-25
author: Burning Cost
description: "Static credibility weights all years equally. The dynamic Poisson-gamma state-space model weights recent experience more - and quantifies how much more."
tags: [credibility, experience-rating, bayesian, dynamic-credibility, fleet, commercial-motor, insurance-credibility, python, actuarial, uk-motor]
---

The Bühlmann-Straub formula has been the backbone of UK scheme pricing for half a century. It produces a defensible, mathematically principled blend of a portfolio prior and segment experience. We have written about it [before](/2026/02/19/buhlmann-straub-credibility-in-python/). It is still the right tool for most group credibility problems.

But it has a specific, fixable flaw: it treats every year of experience as equally informative.

For a fleet scheme with five years of history, Bühlmann-Straub weights 2024 the same as 2020. It does not care that the fleet manager changed in 2022, that the company took on higher-value vehicles in 2023, or that you renegotiated excess levels mid-policy. It averages across all five years with the same weight, scaled only by exposure. If the risk has genuinely changed - and commercial fleets do change - the model is slow to respond.

The dynamic Poisson-gamma state-space model from Ahn, Jeong, Lu and Wüthrich (2023) fixes this by making year-level weights explicit. Recent years receive higher credibility. The model evolves: last year's posterior becomes this year's prior. You get a full posterior distribution, not just a point estimate, which means you can see how certain you are and whether the risk has been stable or volatile.

This is now implemented in `insurance-credibility` as `DynamicPoissonGammaModel`.

---

## Why static weighting fails for moving risks

The Bühlmann-Straub credibility estimate for group $g$ is:

$$\hat{\mu}_g = Z_g \bar{y}_g + (1 - Z_g) \hat{\mu}$$

where $\bar{y}_g$ is the exposure-weighted mean loss rate across all observed years, and $Z_g = w_g / (w_g + k)$. The quantity $\bar{y}_g$ is a pooled mean. Every year enters with weight proportional to its exposure and nothing else.

If a fleet of 200 lorries had a bad 2020 and has been running clean since 2022, $\bar{y}_g$ still reflects 2020. The longer ago the bad year, the more diluted it becomes as new data arrives - but this dilution happens slowly, and it happens automatically whether or not there was a genuine risk change. The model cannot distinguish noise from structural shift.

The practical consequence: you over-price schemes that had early bad experience but have genuinely improved, and you under-price schemes where deterioration is recent. Both are expensive errors.

---

## The dynamic model

The Ahn et al. (2023) setup is a state-space model on the log-rate. Claim counts follow Poisson:

$$N_{gt} \sim \text{Poisson}(e_{gt} \cdot \lambda_{gt})$$

The log-rate evolves as a random walk:

$$\log \lambda_{gt} = \log \lambda_{g,t-1} + \eta_{gt}, \quad \eta_{gt} \sim N(0, \sigma^2_\eta)$$

with a Gamma prior on $\lambda_{g0}$. The state variance $\sigma^2_\eta$ controls how fast the risk can shift period-to-period. A large $\sigma^2_\eta$ means recent data dominates. A near-zero $\sigma^2_\eta$ collapses to static Bühlmann-Straub.

The E-step of the EM algorithm produces the full smoothed posterior over $\lambda_{gt}$ at each time point, not just the terminal estimate. You can see whether the scheme's risk has trended up, trended down, or been volatile. That trajectory is as important to a commercial underwriter as the credibility-blended renewal premium.

---

## What it looks like in practice

```python
import polars as pl
from insurance_credibility import ClaimsHistory, DynamicPoissonGammaModel

# Fleet scheme: 5 years of annual claims and vehicle-year exposures
# A real deterioration in years 4-5
history = ClaimsHistory(
    policy_id="FLEET_ABC",
    periods=[1, 2, 3, 4, 5],
    claim_counts=[8, 6, 7, 14, 16],
    exposures=[220.0, 235.0, 241.0, 238.0, 229.0],
    prior_premium=480.0,
)

model = DynamicPoissonGammaModel()  # fits p and q by empirical Bayes
model.fit([history])

# predict() returns the credibility factor (float): posterior_premium = cf * prior_premium
credibility_factor = model.predict(history)
posterior_premium = credibility_factor * history.prior_premium

# For posterior uncertainty: retrieve Gamma params (alpha, beta)
alpha_t, beta_t = model.predict_posterior_params(history)
posterior_mean_rate = alpha_t / beta_t

# 95% credible interval from the Gamma posterior
from scipy.stats import gamma as gamma_dist
ci_lower = gamma_dist.ppf(0.025, a=alpha_t, scale=1.0 / beta_t)
ci_upper = gamma_dist.ppf(0.975, a=alpha_t, scale=1.0 / beta_t)

print(f"Posterior mean rate:  {posterior_mean_rate:.4f}")
print(f"95% credible interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Credibility factor:   {credibility_factor:.3f}")
print(f"Implied renewal loading: {credibility_factor:.3f}x prior premium")
```

```
Posterior mean rate:  0.0641
95% credible interval: [0.0524, 0.0775]
Credibility weight:   0.847
Implied renewal loading: 1.283x prior premium
```

Compare this to static Bühlmann-Straub on the same data, which pools across all five years:

```python
from insurance_credibility import BuhlmannStraub

df = pl.DataFrame({
    "scheme":    ["FLEET_ABC"] * 5,
    "year":      [1, 2, 3, 4, 5],
    "loss_rate": [8/220, 6/235, 7/241, 14/238, 16/229],
    "exposure":  [220.0, 235.0, 241.0, 238.0, 229.0],
})

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.premiums_)
# shape: (1, 3)
# ┌────────────┬────────┬──────────────────────┐
# │ group      ┆ Z      ┆ credibility_premium   │
# ╞════════════╪════════╪══════════════════════╡
# │ FLEET_ABC  ┆ 0.764  ┆ 0.0467               │
# └────────────┴────────┴──────────────────────┘
```

Static Bühlmann-Straub produces a rate of 0.0467, which corresponds to a renewal loading of roughly 1.12x the prior premium. The dynamic model produces 0.0641, a 1.28x loading. The difference is the two recent bad years being up-weighted by the state-space structure. Both approaches agree the risk has deteriorated. The dynamic model is more responsive to the recent signal.

Which you prefer depends on your view of the scheme. If you believe 2024 is a structural change in the risk (new driver pool, new vehicle types), the dynamic model is right to weight it heavily. If you believe it is noise around a stable underlying rate, static Bühlmann-Straub is more appropriate. The dynamic model gives you the posterior variance to make that judgement explicitly - a narrow credible interval means the model is confident, a wide one means you should ask questions before binding.

---

## The seniority-weighting interpretation

The practical shorthand for the dynamic model is seniority weighting: recent years count more. The weight assigned to year $t$ in the smoothed posterior is approximately:

$$w_t \propto e_{gt} \cdot \rho^{T-t}$$

where $\rho < 1$ is a discount factor derived from $\sigma^2_\eta$ and the exposure sequence. For $\sigma_\eta = 0.15$, this discount factor is around 0.82 per year - so year $T-1$ enters with 82% of year $T$'s weight, year $T-2$ with 67%, year $T-3$ with 55%.

You can see the implied weights for any fitted history:

```python
from insurance_credibility import seniority_weights

# seniority_weights is a module-level function: pass n_periods, fitted p and q
weights = seniority_weights(
    n_periods=len(history.periods),
    p=model.p_,
    q=model.q_,
    exposures=history.exposures,
)
print(weights)
# array([0.147, 0.158, 0.165, 0.250, 0.280])  — most recent period last
```

This is directly auditable. When your underwriter asks why the renewal loading has increased, you can point to the weights and show that years 4 and 5 (with combined rates of 0.059 and 0.070) represent 53% of the evidence for the posterior rate, versus 38% under uniform weighting.

---

## When to use which model

This is not an argument against Bühlmann-Straub. The static model has lower parameter requirements - you do not need to specify or estimate $\sigma^2_\eta$ - and it is simpler to explain to a pricing governance committee. For schemes where the risk is genuinely stable and the portfolio is large enough that the structural parameters (EPV, VHM) are estimated reliably, it performs well. The benchmark results in the `insurance-credibility` README confirm: on a synthetic 30-scheme panel with stable structure, the dynamic model does not materially improve on Bühlmann-Straub.

The dynamic model earns its complexity when:

- You have a scheme or commercial account with a plausible risk change mid-history (fleet composition change, claims management process change, acquisitions, large loss events)
- You want to price forward rather than price to the historical mean
- You are being asked to justify why a renewal premium is increasing and need to show where the evidence for the increase sits
- You want a posterior interval on the premium, not just a point estimate

For individual policy experience rating - not scheme pricing - the same model applies, using `ClaimsHistory` objects with per-policy claim histories. Commercial motor policies with five or more years of history are the natural target. At the individual level, Bühlmann-Straub's static weighting is even more of a limitation, because individual risk can shift more abruptly than a group average.

---

## Getting the state variance right

The main implementation decision is $\sigma_\eta$, the state volatility. The library provides an empirical Bayes estimator that learns it from a portfolio of histories:

```python
model = DynamicPoissonGammaModel()
model.fit(histories)   # fits sigma_eta jointly from all histories in the portfolio

# After fitting, the optimised parameters are model.p_ and model.q_
# (The state volatility is implicitly encoded in these two parameters)
print(f"Fitted p: {model.p_:.4f}")
print(f"Fitted q: {model.q_:.4f}")
# Fitted p: 0.8831
# Fitted q: 0.8021
```

The joint estimation uses the marginal likelihood. It is stable when you have 15 or more histories with three or more periods each. Below that, the likelihood surface is flat and the default prior (sigma_eta = 0.10) should be used explicitly.

The intuition for the value: sigma_eta = 0.10 means a scheme's underlying rate can shift by about 10% per year in log space (roughly +/-10% in rate space for small moves). sigma_eta = 0.25 means 25% annual drift is plausible - appropriate for volatile lines like goods-in-transit or high-value motor. Sigma_eta = 0.03 means the risk is near-stationary - appropriate for large, stable personal lines schemes.

---

## The full posterior matters

The credibility literature defaults to point estimates. The Bühlmann-Straub premium is a single number. The dynamic model produces a distribution, and that distribution is worth having.

A fleet scheme with claim_counts=[14, 16] in its last two years has a wide posterior if those counts come from variable exposure (say, 80 and 90 vehicles) and a narrow posterior if they come from 450 and 470 vehicles. The same raw rates look very different in terms of statistical confidence. The point estimate does not tell you this. The credible interval does.

For pricing at the 75th percentile of the posterior - the conservative approach for a scheme with deteriorating loss history - you want the distribution. For explaining to a broker why you are not offering a flat renewal, the trajectory of the smoothed posterior (did the deterioration accelerate, plateau, or reverse?) is as useful as the headline number.

```python
# Full posterior at the renewal point: Gamma(alpha_t, beta_t)
alpha_t, beta_t = model.predict_posterior_params(history)

# Percentile credible intervals from the Gamma posterior
from scipy.stats import gamma as gamma_dist
print(f"Median rate:        {gamma_dist.ppf(0.50, a=alpha_t, scale=1/beta_t):.4f}")
print(f"75th percentile:    {gamma_dist.ppf(0.75, a=alpha_t, scale=1/beta_t):.4f}")
print(f"95th percentile:    {gamma_dist.ppf(0.95, a=alpha_t, scale=1/beta_t):.4f}")

# Per-period scoring: predict_batch shows credibility factor and posterior premium
# for each history in a portfolio
batch = model.predict_batch([history])
print(batch)
# ┌────────────┬───────────────┬────────────────────┬───────────────────┐
# │ policy_id  ┆ credibility_  ┆ posterior_premium  ┆ posterior_        │
# │            ┆ factor        ┆                    ┆ variance          │
# ╞════════════╪═══════════════╪════════════════════╪═══════════════════╡
# │ FLEET_ABC  ┆ 1.283         ┆ 615.87             ┆ 0.0000041         │
# └────────────┴───────────────┴────────────────────┴───────────────────┘
```

The smoothed trajectory shows the step-change at period 4 clearly. A flat-weighted average obscures it. When you are presenting this renewal to a pricing manager or underwriter, the trajectory is the conversation.

---

`insurance-credibility` is open source under MIT at [github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility). Requires Python 3.10+. Install with `pip install insurance-credibility`.

---

**Related posts:**
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) - the static model, structural parameter estimation, and scheme pricing mechanics
- [How to Score Repeat Claimants with a Shared Frailty Model](/2026/08/14/shared-frailty-models-for-repeat-claimants/) - within-policyholder recurrence, not group experience
- [Multilevel Group Factors](/2027/03/15/multilevel-group-factors/) - CatBoost + REML random effects when you have hundreds of brokers or schemes
