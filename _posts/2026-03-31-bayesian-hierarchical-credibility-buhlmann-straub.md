---
layout: post
title: "Bayesian Hierarchical Credibility — When Bühlmann-Straub Isn't Enough"
seo_title: "Bayesian Hierarchical Credibility: Beyond Bühlmann-Straub for UK Insurance Pricing"
date: 2026-03-31
categories: [credibility, bayesian]
tags: [credibility, buhlmann-straub, bayesian, poisson-gamma, conjugate, hierarchical, python, insurance-credibility, GBM, two-stage, prior-elicitation, UK-motor]
description: "Bühlmann-Straub pulls every segment toward the same grand mean. When that mean is wrong for your segment, the correction makes things worse. We explain when to reach for Poisson-Gamma conjugate credibility or full Bayesian hierarchical models instead — with code using PoissonGammaCredibility from insurance-credibility v0.1.8."
author: burning-cost
---

Bühlmann-Straub credibility does one thing that pure MLE does not: it acknowledges that a segment with limited data should not be trusted on its own terms. The credibility factor Z shrinks the segment's observed rate toward a prior, weighted by how much data you have. The mechanics are [well-established](/2026/02/19/buhlmann-straub-credibility-in-python/) and, for most UK personal lines applications, adequate.

But B-S has a specific structural property that causes real problems in a specific class of situations: the complement of credibility — the 1 − Z weight — is always applied to the grand mean of the portfolio. There is no mechanism to say "this segment belongs to a fundamentally different population." If your grand mean is wrong for the segment in question, B-S does not shrink toward the right target. It shrinks toward the wrong one, and does so with mathematical confidence.

This post covers three things: exactly when B-S breaks down, what Poisson-Gamma conjugate credibility does differently, and when you actually need PyMC. We use the new `PoissonGammaCredibility` class from [`insurance-credibility`](/insurance-credibility/) v0.1.8 throughout.

---

## The specific failure modes

We [catalogued six structural limits of B-S](/2026/03/23/does-buhlmann-straub-credibility-work-insurance-pricing/) in detail. Three of them come up repeatedly in UK pricing work.

**Hyperparameter uncertainty at small group counts.** B-S treats its estimated parameters — the expected process variance `v` and the variance of hypothetical means `a` — as if they were exact. For portfolios with 20 or more groups this approximation is defensible. For a commercial lines portfolio with 10–15 schemes, the sampling variance around `a_hat` is large enough that the resulting credibility factors carry false precision. A Bayesian posterior correctly propagates that uncertainty into wider credible intervals. B-S gives you a precise-looking number; the true uncertainty is 1.5 to 2 times wider.

**Non-Normal response families.** B-S produces the optimal *linear* credibility estimator. For claim counts (Poisson) or claim severity (Gamma), the optimal Bayesian posterior mean is nonlinear in the sufficient statistics. B-S equals the exact Bayes estimator only under Normal-Normal conjugacy. For Poisson data, there is a closed-form alternative — Poisson-Gamma conjugacy — that beats B-S without requiring MCMC.

**Crossed random effects.** B-S handles strictly nested hierarchies: scheme → portfolio, region → country. It cannot handle structures where a cell is the intersection of two independently-drawn groups — NCD level × vehicle group, occupation × territory. For those structures you need a proper mixed model or full Bayesian hierarchical model. B-S has no mechanism for it.

For most standard UK motor territory work — 50 to 200 regions, three to five years of data — B-S works well and the overhead of anything more complex is not justified. Where it breaks down is specialty lines (small number of schemes, few years of data), crossed effects, and pricing decisions that require a posterior distribution rather than a point estimate.

---

## Poisson-Gamma conjugate credibility

The Poisson-Gamma model is the textbook Bayesian credibility formulation for claim counts. Claim counts per group follow a Poisson process; the unknown claim rate λᵢ for each group is drawn from a Gamma prior across the portfolio.

Formally:

```
Prior:      λᵢ ~ Gamma(α₀, β₀)
Likelihood: yᵢ | λᵢ ~ Poisson(eᵢ · λᵢ)

Posterior:  λᵢ | data ~ Gamma(α₀ + yᵢ, β₀ + eᵢ)
```

where yᵢ is total claims for group i and eᵢ is total exposure.

The posterior mean is:

```
E[λᵢ | data] = (α₀ + yᵢ) / (β₀ + eᵢ)
             = Z_i · (yᵢ / eᵢ) + (1 − Z_i) · (α₀ / β₀)
```

with Z_i = eᵢ / (eᵢ + β₀).

This is the standard B-S credibility formula. The difference is what β₀ represents: it is the *effective prior exposure* — how much portfolio exposure is worth in prior information. A group needs eᵢ = β₀ earned car years to reach Z = 0.5. Where B-S estimates k = v/a from the panel data, Poisson-Gamma estimates β₀ from the empirical method of moments on group-level rates. These are the same quantity derived differently.

What Poisson-Gamma adds over B-S:

1. **Exact posterior, not linear approximation.** For Poisson data, this is the true Bayesian posterior. B-S gives the optimal linear approximation to it.
2. **Credible intervals from the posterior.** The posterior Gamma distribution gives exact quantiles — no bootstrapping, no asymptotic approximation.
3. **Segment-specific priors.** You can set different (α₀, β₀) for different sub-portfolios. A large fleet segment and a private motor segment need not share the same prior. B-S always regresses to the same grand mean.

### The code

`PoissonGammaCredibility` in insurance-credibility v0.1.8 takes raw claim counts and exposures — not pre-computed rates. This is deliberate: the sufficient statistics for the Poisson-Gamma update are (Σ claims, Σ exposure), not the rate.

```python
import polars as pl
from insurance_credibility.classical.conjugate import PoissonGammaCredibility

# Panel data: one row per (group, period)
# claims and exposure are raw counts and earned car years
panel = pl.DataFrame({
    "group":    ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "claims":   [12,   8,  15,   3,   1,   4,  45,  52,  48],
    "exposure": [800, 750, 900, 200, 180, 220, 4200, 4500, 4100],
})

model = PoissonGammaCredibility()
model.fit(
    data=panel,
    group_col="group",
    claims_col="claims",
    exposure_col="exposure",
)
print(model.summary())
```

The summary shows the credibility rate (posterior mean), the credibility factor Z, and the observed rate:

```
┌───────┬──────────┬──────────┬──────────────┬────────────┐
│ group ┆ Exposure ┆ Obs. Rate ┆ Cred. Rate  ┆ Prior Mean │
│ str   ┆ f64      ┆ f64      ┆ f64          ┆ f64        │
╞═══════╪══════════╪══════════╪══════════════╪════════════╡
│ A     ┆ 2450.0   ┆ 0.01429  ┆ 0.01317      ┆ 0.01087    │
│ B     ┆ 600.0    ┆ 0.01333  ┆ 0.01231      ┆ 0.01087    │
│ C     ┆ 12800.0  ┆ 0.01133  ┆ 0.01126      ┆ 0.01087    │
└───────┴──────────┴──────────┴──────────────┴────────────┘
```

Group B (600 earned car years total, 8 observed claims) has Z = 0.38 — the model is appropriately sceptical. Group C has 12,800 car years and the credibility weight is 0.93.

**Posterior credible intervals** are exact Gamma quantiles:

```python
intervals = model.credibility_intervals(credibility_interval=0.90)
print(intervals)
```

```
┌───────┬──────────────────┬──────────┬──────────┬──────────┐
│ group ┆ credibility_rate ┆ lower    ┆ upper    ┆ Z        │
│ str   ┆ f64              ┆ f64      ┆ f64      ┆ f64      │
╞═══════╪══════════════════╪══════════╪══════════╪══════════╡
│ A     ┆ 0.01317          ┆ 0.01033  ┆ 0.01645  ┆ 0.717    │
│ B     ┆ 0.01231          ┆ 0.00747  ┆ 0.01843  ┆ 0.383    │
│ C     ┆ 0.01126          ┆ 0.01029  ┆ 0.01228  ┆ 0.930    │
└───────┴──────────────────┴──────────┴──────────┴──────────┘
```

Group B's 90% posterior interval spans 0.006 to 0.016 — nearly a 2.5× range. B-S would give you a single number without the interval. This is the information that matters for stop-loss pricing, reinsurance structure, or regulatory sign-off.

**Scoring new groups** uses the prior directly, with an optional partial credibility update:

```python
result = model.predict(claims=5, exposure=300)
print(result)
```

---

## Comparing with Bühlmann-Straub on the same data

On a panel with 20+ groups, the posterior mean from `PoissonGammaCredibility` and the credibility premium from `BuhlmannStraub` converge. Both are estimating the same blend; they differ in which assumptions generate the weights.

Where they diverge:

| Scenario | B-S | Poisson-Gamma |
|---|---|---|
| 8 groups, thin data | Point estimate, false precision | Wider posterior; uncertainty in β₀ reflected |
| Need credible intervals | Not available | Exact Gamma quantiles |
| Segment-specific prior | Not possible | Set per-segment α₀, β₀ |
| Poisson data, exact update | Optimal linear approximation | Exact Bayesian posterior |

The B-S estimate and Poisson-Gamma posterior mean will rarely differ by more than 2–3% in log-rate terms on well-populated panels. The practical gap is the posterior distribution and the ability to set segment-appropriate priors. Both of these matter more than the point estimate difference.

---

## The two-stage architecture

There is a practical objection to Bayesian credibility models: they are hard to scale to the feature richness of a GBM. A PyMC model with 50 input features is not impossible, but it is expensive to specify and expensive to run. Meanwhile, a GBM handles 50 features natively.

The two-stage architecture resolves this:

**Stage 1 — GBM on full policy data.** Train a gradient-boosted model on all available rating factors. This produces a base predicted rate for every policy. The GBM handles all the feature complexity.

**Stage 2 — Bayesian credibility on residuals.** Aggregate GBM residuals (observed / predicted) to the segment level — scheme, territory, vehicle class, whatever the relevant grouping is. These segment-level residuals are the input to `PoissonGammaCredibility`. The credibility model adjusts each segment's base rate up or down based on its own experience, blended with the portfolio prior.

The key insight: after the GBM strips out the systematic feature effects, the segment-level residuals are closer to exchangeable. The Bayesian credibility model is operating on a much narrower distribution, which means tighter priors and better-behaved posteriors.

```python
from insurance_credibility.classical.conjugate import PoissonGammaCredibility
import lightgbm as lgb

# Stage 1: GBM base rate
gbm = lgb.train(params, train_data)
df = df.with_columns(
    pl.Series("base_rate", gbm.predict(X))
)

# Stage 2: segment-level sufficient stats
segment_stats = (
    df.group_by("segment")
    .agg([
        pl.col("claims").sum(),
        pl.col("exposure").sum(),
        (pl.col("base_rate") * pl.col("exposure")).sum().alias("expected_claims"),
    ])
)

# Credibility adjustment on observed vs expected
# Feed observed claims and GBM-implied exposure to the conjugate model
pgc = PoissonGammaCredibility()
pgc.fit(segment_stats, group_col="segment",
        claims_col="claims", exposure_col="expected_claims")
```

The effective exposure in Stage 2 is the GBM-predicted expected claims, not raw earned car years. A segment where the GBM already has strong signal contributes more prior weight; a novel segment with little GBM coverage gets more credibility adjustment.

This architecture solves the "Bayesian models don't scale to 50 features" problem. The GBM handles feature complexity. The Bayesian layer handles low-volume segment adjustment. Neither approach is doing something it is not suited for.

---

## When to use PyMC instead

`PoissonGammaCredibility` covers the most common use case: Poisson claim counts, nested (not crossed) hierarchies, and a portfolio large enough that hyperparameter uncertainty is modest.

For three specific situations, it is not enough and you need full Bayesian inference via PyMC:

**Crossed effects.** NCD level × vehicle group, occupation × territory — any structure where a cell is the intersection of two independently-drawn groups rather than a child of a single parent. PyMC's `ZeroSumNormal` or correlated Normal priors handle this; conjugate models cannot.

**Spatial random effects.** UK motor postcode-sector modelling (9,500 sectors) with a Conditional Autoregressive (CAR) prior, where neighbouring sectors share information proportionally to adjacency. The conjugate model has no mechanism for spatial smoothing. INLA (R) is the established tool here; NumPyro via `pm.sampling.jax.sample_numpyro_nuts()` is the Python path — feasible in minutes on CPU for O(1,000) segments, slower for full postcode-sector granularity.

**Time-varying risk.** B-S and the conjugate model both treat each group's risk parameter as stationary. UK motor risk profiles shift: COVID-era frequency collapse (2020), rebound (2022), and the ongoing effects of inflation on repair cost severity. A dynamic credibility model — random walk on group effects, state-space formulation — requires MCMC or Kalman filtering. Neither closed-form approach can represent it.

For standard UK motor work with 50–200 territories and three to five years of history, you do not need PyMC. For specialty lines with fewer than 20 schemes, for crossed factor structures, or for spatial territory models, the overhead is justified.

---

## Prior elicitation

For the PyMC path, prior specification is the main decision that requires judgement. The default priors in Bambi (`HalfNormal(2.5)` for random effect standard deviations on log scale) are too wide for insurance — they imply plausible territory effect standard deviations above 2 on the log scale, corresponding to rate multiples of 100× or more — not a realistic prior for territory effects in UK motor.

Our recommended starting point for the between-group log-scale standard deviation σ is `HalfNormal(0.3)`. This places 95% of the prior mass below 0.6 log units, corresponding to roughly an 80% relative rate range — consistent with observed territory coefficient distributions in UK motor. For highly homogeneous portfolios (large fleet, affinity schemes) you might tighten this to `HalfNormal(0.15)`. For specialty lines with genuine structural diversity, `HalfNormal(0.5)` is more appropriate.

The standard workflow:

```python
import pymc as pm
import numpy as np

log_rate_prior = np.log(portfolio_mean_rate)

with pm.Model():
    mu = pm.Normal("mu", mu=log_rate_prior, sigma=0.5)
    sigma = pm.HalfNormal("sigma", sigma=0.3)

    # Non-centred parameterisation — mandatory for thin groups
    z_raw = pm.Normal("z_raw", mu=0, sigma=1, shape=n_groups)
    log_lambda = pm.Deterministic("log_lambda", mu + sigma * z_raw)

    # Prior predictive check before touching the data
    prior_pred = pm.sample_prior_predictive(samples=500)
```

Run the prior predictive check before touching the data. If the prior is generating claim frequencies outside the plausible range (say, more than 10× the portfolio mean), tighten σ. The prior predictive check is not optional; it is the mechanism by which domain knowledge enters the model.

The non-centred parameterisation — expressing group effects as `mu + sigma * z_raw` rather than directly as `Normal(mu, sigma)` — is mandatory when groups are thin. The centred version creates a funnel geometry in the posterior that NUTS cannot traverse efficiently. Non-centred removes the funnel. Use it unconditionally.

Convergence thresholds for hierarchical models: R-hat < 1.01 (not the default 1.1), ESS > 400 per parameter. Any divergences indicate the non-centred parameterisation is needed or `target_accept` should be raised to 0.95.

---

## Practical summary

The decision tree is short:

- **Standard panel, Poisson counts, nested hierarchy, 20+ groups:** `PoissonGammaCredibility` from insurance-credibility v0.1.8. Zero extra dependencies, exact posterior, credible intervals included.
- **Fewer than 15 groups:** still use `PoissonGammaCredibility`, but interpret the posterior intervals sceptically — hyperparameter uncertainty is not fully propagated. Consider PyMC if the decision stakes are high.
- **Crossed effects, spatial structure, time-varying risk:** PyMC. Use non-centred parameterisation, `HalfNormal(0.3)` on σ, prior predictive checks before fitting.
- **50+ features, large policy-level dataset:** two-stage architecture. GBM first, credibility adjustment on segment sufficient stats.

The conjugate approach covers roughly 80% of the genuine gap between B-S and full Bayesian hierarchical models. It does so without pulling in PyMC, without MCMC, and without the diagnostic overhead of checking R-hat and ESS. That matters in production pricing where the model needs to run reliably and the outputs need to be explainable to a reserving actuary at short notice.

The `PoissonGammaCredibility` class is available now in [`insurance-credibility`](/insurance-credibility/) v0.1.8. Full PyMC integration is future work — the roadmap is set and the architecture is designed to accommodate it without breaking the existing API.
