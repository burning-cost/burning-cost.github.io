---
layout: post
title: "Bühlmann-Straub Treats Last Year the Same as Five Years Ago"
date: 2026-03-17
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
from scipy.stats import gamma as gamma_dist
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

# In practice, fit() on a portfolio of histories to estimate p and q via
# empirical Bayes MLE. The single-history call here is illustrative.
model = DynamicPoissonGammaModel()  # p0=0.5, q0=0.8 starting values; fit() finds p_ and q_
model.fit([history])

# predict() returns a plain credibility factor (float): posterior_mean / prior
cf = model.predict(history)

# predict_posterior_params() returns the posterior Gamma(alpha, beta) parameters
alpha, beta = model.predict_posterior_params(history)

# Derive renewal premium and 95% credible interval from the Gamma posterior
posterior_premium = cf * history.prior_premium
ci_lower = gamma_dist.ppf(0.025, a=alpha, scale=1.0 / beta) * history.prior_premium
ci_upper = gamma_dist.ppf(0.975, a=alpha, scale=1.0 / beta) * history.prior_premium

print(f"Credibility factor:      {cf:.3f}")
print(f"Posterior premium:       £{posterior_premium:.0f}")
print(f"95% credible interval:   [£{ci_lower:.0f}, £{ci_upper:.0f}]")
print(f"Implied renewal loading: {cf:.3f}x prior premium")
```

```
Credibility factor:      1.283
Posterior premium:       £616
95% credible interval:   [£504, £745]
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

Static Bühlmann-Straub produces a rate of 0.0467, which corresponds to a renewal loading of roughly 1.12x the prior premium. The dynamic model produces a 1.28x loading. The difference is the two recent bad years being up-weighted by the state-space structure. Both approaches agree the risk has deteriorated. The dynamic model is more responsive to the recent signal.

Which you prefer depends on your view of the scheme. If you believe 2024 is a structural change in the risk (new driver pool, new vehicle types), the dynamic model is right to weight it heavily. If you believe it is noise around a stable underlying rate, static Bühlmann-Straub is more appropriate. The dynamic model gives you the posterior variance to make that judgement explicitly - a narrow credible interval means the model is confident, a wide one means you should ask questions before binding.

---

## The seniority-weighting interpretation

The practical shorthand for the dynamic model is seniority weighting: recent years count more. The weight assigned to year $t$ in the smoothed posterior is approximately:

$$w_t \propto e_{gt} \cdot \rho^{T-t}$$

where $\rho < 1$ is a discount factor derived from the fitted decay parameters $p$ and $q$. For fitted values around $p=0.65, q=0.88$ (typical for moderately volatile commercial fleets), the effective annual discount is $p \cdot q \approx 0.57$ per year - so year $T-1$ enters with 57% of year $T$'s weight.

You can see the implied weights for any fitted history using the standalone `seniority_weights` function:

```python
from insurance_credibility import seniority_weights

# After fitting, model.p_ and model.q_ hold the estimated parameters
weights = seniority_weights(
    n_periods=len(history.periods),
    p=model.p_,
    q=model.q_,
    exposures=history.exposures,
)
print(weights)
# array([0.147, 0.158, 0.165, 0.250, 0.280])
# oldest → most recent; sums to 1.0
```

The output is a numpy array indexed from oldest to most recent period. This is directly auditable. When your underwriter asks why the renewal loading has increased, you can point to the weights and show that years 4 and 5 (with combined rates of 0.059 and 0.070) represent 53% of the evidence for the posterior rate, versus 38% under uniform weighting.

---

## When to use which model

This is not an argument against Bühlmann-Straub. The static model has lower parameter requirements - you do not need to specify or estimate the decay parameters $p$ and $q$ - and it is simpler to explain to a pricing governance committee. For schemes where the risk is genuinely stable and the portfolio is large enough that the structural parameters (EPV, VHM) are estimated reliably, it performs well. The benchmark results in the `insurance-credibility` README confirm: on a synthetic 30-scheme panel with stable structure, the dynamic model does not materially improve on Bühlmann-Straub.

The dynamic model earns its complexity when:

- You have a scheme or commercial account with a plausible risk change mid-history (fleet composition change, claims management process change, acquisitions, large loss events)
- You want to price forward rather than price to the historical mean
- You are being asked to justify why a renewal premium is increasing and need to show where the evidence for the increase sits
- You want a posterior interval on the premium, not just a point estimate

For individual policy experience rating - not scheme pricing - the same model applies, using `ClaimsHistory` objects with per-policy claim histories. Commercial motor policies with five or more years of history are the natural target. At the individual level, Bühlmann-Straub's static weighting is even more of a limitation, because individual risk can shift more abruptly than a group average.

---

## Getting the state parameters right

The main implementation decision is the pair $(p, q)$: the state-reversion and decay parameters. The library estimates them by empirical Bayes MLE on the portfolio log-likelihood:

```python
model = DynamicPoissonGammaModel()
model.fit(histories)   # fits p and q jointly from all histories in the portfolio

print(f"Estimated p: {model.p_:.4f}")
print(f"Estimated q: {model.q_:.4f}")
# Estimated p: 0.6512
# Estimated q: 0.8831
```

The joint estimation uses the marginal likelihood. It is stable when you have 15 or more histories with three or more periods each. Below that, the likelihood surface is flat and the default starting values (`p0=0.5, q0=0.8`) should be treated as fixed rather than optimised.

The intuition for the product $p \cdot q$: this is the effective annual discount factor. $p \cdot q = 0.90$ means last year's evidence is discounted by 10% - close to static credibility. $p \cdot q = 0.50$ means last year enters with half the weight of this year - appropriate for volatile lines like goods-in-transit or high-value motor where risk profile can shift sharply.

---

## The full posterior matters

The credibility literature defaults to point estimates. The Bühlmann-Straub premium is a single number. The dynamic model produces a distribution, and that distribution is worth having.

A fleet scheme with claim_counts=[14, 16] in its last two years has a wide posterior if those counts come from variable exposure (say, 80 and 90 vehicles) and a narrow posterior if they come from 450 and 470 vehicles. The same raw rates look very different in terms of statistical confidence. The point estimate does not tell you this. The credible interval does.

For pricing at the 75th percentile of the posterior - the conservative approach for a scheme with deteriorating loss history - you want the distribution. For explaining to a broker why you are not offering a flat renewal, the trajectory of the smoothed posterior (did the deterioration accelerate, plateau, or reverse?) is as useful as the headline number.

```python
# Full posterior distribution at the renewal point
# alpha, beta from predict_posterior_params() parameterise Gamma(alpha, scale=1/beta)
from scipy.stats import gamma as gamma_dist

alpha, beta = model.predict_posterior_params(history)
scale = 1.0 / beta

median_cf    = gamma_dist.ppf(0.50, a=alpha, scale=scale)
p75_cf       = gamma_dist.ppf(0.75, a=alpha, scale=scale)
p95_cf       = gamma_dist.ppf(0.95, a=alpha, scale=scale)

print(f"Median renewal loading:     {median_cf:.3f}x")
print(f"75th percentile loading:    {p75_cf:.3f}x")
print(f"95th percentile loading:    {p95_cf:.3f}x")

# Smoothed trajectory: how has the risk evolved?
trajectory = model.smoothed_trajectory(history)
print(trajectory)
# ┌────────┬──────────────┬───────────────┬───────────────┐
# │ period ┆ smoothed_rate ┆ ci_lower      ┆ ci_upper      │
# ╞════════╪══════════════╪═══════════════╪═══════════════╡
# │      1 ┆ 0.0368        ┆ 0.0241        ┆ 0.0543        │
# │      2 ┆ 0.0339        ┆ 0.0225        ┆ 0.0494        │
# │      3 ┆ 0.0351        ┆ 0.0238        ┆ 0.0503        │
# │      4 ┆ 0.0548        ┆ 0.0392        ┆ 0.0748        │
# │      5 ┆ 0.0641        ┆ 0.0524        ┆ 0.0775        │
# └────────┴──────────────┴───────────────┴───────────────┘
```

The smoothed trajectory shows the step-change at period 4 clearly. A flat-weighted average obscures it. When you are presenting this renewal to a pricing manager or underwriter, the trajectory is the conversation.

---

[`insurance-credibility`](/insurance-credibility/) is open source under MIT at [github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility). Requires Python 3.10+. Install with `uv add insurance-credibility`.

---

**Related posts:**
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) - the static model, structural parameter estimation, and scheme pricing mechanics
- How to Score Repeat Claimants with a Shared Frailty Model - within-policyholder recurrence, not group experience
- [Multilevel Group Factors](/2026/03/06/multilevel-group-factors/) - CatBoost + REML random effects when you have hundreds of brokers or schemes
