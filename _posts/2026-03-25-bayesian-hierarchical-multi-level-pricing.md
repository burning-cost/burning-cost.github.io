---
layout: post
title: "Bayesian Hierarchical Multi-Level Pricing"
date: 2026-03-25
categories: [techniques]
tags: [bayesian, hierarchical, mixed-effects, credibility, neural-mixed-effects, pymc, insurance-credibility, scheme-pricing, mga, uk-motor, python]
description: "Bühlmann-Straub credibility breaks when your hierarchy has more than two levels or when the random effects interact with pricing factors. Here is when to upgrade to a full Bayesian hierarchical model, and how to do it with insurance-credibility and PyMC."
---

Bühlmann-Straub is correct — in the precise mathematical sense that it produces the minimum-variance linear estimator of each group's true rate given the data structure it assumes. The assumptions are: observations within a group are exchangeable, variance components are constant across groups, and the random effect enters linearly. On a simple two-level structure — risk nested within scheme, or scheme within book — these assumptions are usually defensible.

They break down in three common situations:

1. **Three or more levels**: broker → scheme → risk, or postcode sector → district → region. B-S can be extended hierarchically (Jewell 1975), but the variance component estimation becomes unstable at upper levels where you have few nodes and noisy between-node differences.

2. **Interaction between random effects and fixed pricing factors**: the broker effect on conversion is not constant across vehicle groups. A broker who specialises in commercial vehicles converts very differently on private car. Treating the broker effect as an additive offset to the main-effects model misses this.

3. **Thin segments with strong prior information**: you have a new MGA with six months of data and a credible view from the cedant's previous book about likely loss ratios. B-S has no mechanism for injecting informative priors. The choice is either ignore the prior information or use it informally via a manual loading.

The Bayesian hierarchical model handles all three. The price is computational complexity and the need to specify priors explicitly — which is either a cost or a feature depending on your perspective.

---

## The Wüthrich framework

Wüthrich (SSRN 4726206, 2025) derives neural mixed effects models for insurance pricing. The framework is a direct generalisation of classical credibility theory: the random effect θ_g for group g is drawn from a prior distribution, the observation given θ_g follows a Poisson (or gamma) distribution, and the posterior E[θ_g | data] is the credibility-weighted estimate.

The classical B-S result is a special case: with a normal-normal conjugate pair (normal prior on θ, normal likelihood), the posterior mean is exactly the B-S formula Z · X̄_g + (1-Z) · μ, where Z = w/(w+k). The Wüthrich contribution is to replace the main-effects model μ with a neural network: instead of a GLM that produces a portfolio mean, you have a neural net f(X; ω) that produces risk-level predictions, and the random effect θ_g adjusts those predictions for group-level systematic differences.

The resulting model is:

```
λ_ig = f(X_i; ω) · exp(θ_g)
```

where f(X_i; ω) is the neural network component (or GBM — the specific architecture matters less than the structure), θ_g ~ N(μ_g, σ²_g) is the group random effect, and λ_ig is the expected claim frequency for policy i in group g.

The posterior for θ_g given observations is analytically tractable for the Poisson-lognormal pair (Hinde 1982). Wüthrich provides the EM algorithm for fitting the variance components σ²_g jointly with the neural network weights ω.

For pure Bayesian inference — where you want proper posterior uncertainty on all parameters, not just point estimates — PyMC is the cleaner tool.

---

## When to use the hierarchical model

The decision rule is concrete. Use B-S (via `BuhlmannStraub` from [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility)) when:

- Two-level structure only
- Variance components are stable and you expect they will remain so
- You do not have informative prior beliefs about specific groups

Switch to the full Bayesian hierarchical model when any of these apply:

- Three or more nested levels (and you need premiums at all levels, not just the leaf)
- You have prior information from a cedant, previous book, or market data that should formally inform the estimate
- The random effects interact with pricing factors (e.g., broker effects vary by vehicle type)
- You need proper posterior intervals on group-level premiums, not just credibility factors

The last point matters more than it sounds. A B-S Z-factor of 0.6 tells you the credibility-weighted premium but not the uncertainty around it. A Bayesian model gives you the posterior distribution of the group premium — the 5th and 95th percentiles. For capital modelling, reinsurance treaty pricing, or scheme renewal terms, this is often the number that matters.

---

## The three-level geographic example

UK motor has a natural three-level geographic hierarchy: postcode sector (~9,500 units) → district (~3,000) → area (~124). Most pricing teams work at district level because sectors are too thin. The hierarchical model lets you work at sector level by borrowing strength from districts, which borrow from areas.

The `insurance-credibility` library implements this as `HierarchicalBuhlmannStraub`:

```python
import polars as pl
from insurance_credibility import HierarchicalBuhlmannStraub

# df has columns: area, district, sector, year, loss_rate, exposure
model = HierarchicalBuhlmannStraub(
    level_cols=["area", "district", "sector"]
)
model.fit(
    data=df,
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

# Variance components at each level
for name, lr in model.level_results_.items():
    print(f"{name}: v={lr.v_hat:.4f}, a={lr.a_hat:.4f}, k={lr.k:.2f}")

# Blended premiums at sector level (borrows from district, then area)
sector_premiums = model.premiums_at("sector")
```

The blending formula at each level is P_node = Z_node · X̄_node + (1-Z_node) · P_parent, where P_parent is the credibility premium from the level above. A sector with 50 policy-years gets a small Z and leans heavily on its district. A sector with 800 policy-years gets Z close to 1.0 and is trusted on its own data.

The hierarchical model prevents the pathology where a thin sector with three claims in one year gets a sector loading of 2.5× the area rate, which a simple ratio approach would produce.

---

## The full Bayesian model with PyMC

For MGA portfolios, scheme business, or any setting with strong informative priors, the PyMC implementation is more appropriate. The model below is for a two-level hierarchy (scheme within book), Poisson frequency, with a weakly informative prior on scheme effects:

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt

def build_hierarchical_model(
    scheme_idx: np.ndarray,   # integer index: which scheme for each policy
    n_schemes: int,
    log_glm_prediction: np.ndarray,   # log of GLM/GBM fixed-effects prediction
    claims: np.ndarray,
    exposure: np.ndarray,
    prior_scheme_mean: float = 0.0,   # informative prior from cedant data
    prior_scheme_sd: float = 0.15,    # how much you trust the cedant's numbers
) -> pm.Model:
    """
    Bayesian hierarchical Poisson model for scheme pricing.

    Random effect θ_g ~ N(μ_θ, σ_θ²) for each scheme g.
    Claims_i | θ ~ Poisson(exposure_i * glm_i * exp(θ_{g(i)}))

    The prior_scheme_sd encodes how much between-scheme variation you expect
    a priori. Set to 0.15 for UK motor (roughly ±30% 2σ range),
    tighter for narrow books.
    """
    with pm.Model() as model:
        # Hyperpriors on the random effect distribution
        mu_theta = pm.Normal("mu_theta", mu=prior_scheme_mean, sigma=0.1)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=prior_scheme_sd)

        # Non-centred parameterisation (faster sampling)
        theta_offset = pm.Normal("theta_offset", mu=0, sigma=1, shape=n_schemes)
        theta = pm.Deterministic("theta", mu_theta + sigma_theta * theta_offset)

        # Expected rate: GLM fixed effect * scheme random effect
        log_lambda = log_glm_prediction + theta[scheme_idx] + np.log(exposure)
        mu_obs = pt.exp(log_lambda)

        # Likelihood
        pm.Poisson("claims_obs", mu=mu_obs, observed=claims)

    return model
```

Fitting is standard:

```python
with build_hierarchical_model(
    scheme_idx=scheme_idx_array,
    n_schemes=n_schemes,
    log_glm_prediction=log_glm_pred,
    claims=claims_array,
    exposure=exposure_array,
    prior_scheme_mean=0.0,
    prior_scheme_sd=0.15,
) as model:
    trace = pm.sample(
        draws=2000,
        tune=1000,
        target_accept=0.9,
        chains=4,
        random_seed=42,
    )

# Posterior scheme effects: θ_g
theta_posterior = trace.posterior["theta"]
scheme_effect_mean = theta_posterior.mean(dim=["chain", "draw"]).values  # shape (n_schemes,)
scheme_effect_hdi  = pm.hdi(theta_posterior, hdi_prob=0.9)              # 90% HDI

# Scheme-level premium (relative to portfolio mean)
scheme_loading = np.exp(scheme_effect_mean)   # multiplicative adjustment
```

The non-centred parameterisation (`theta_offset`) is not aesthetic preference — it is necessary for efficient sampling when schemes are thin. The centred version (`theta ~ N(mu, sigma)` directly) produces funnel geometry in the posterior that causes NUTS to struggle. You will see divergences in the trace if you use the centred version on a portfolio where some schemes have fewer than 200 policy-years.

---

## Injecting cedant or market prior information

This is the practical advantage of Bayesian over B-S that matters for MGA and scheme pricing. Suppose you are pricing a new book from a cedant. The cedant provides a historical loss ratio of 72% and a view that between-scheme variance is moderate (schemes differ by about ±25% around the mean). You translate this into:

- `prior_scheme_mean = np.log(0.72 / portfolio_expected_lr)` — the cedant's prior on the book mean relative to your portfolio
- `prior_scheme_sd = 0.12` — consistent with ±25% 2σ scheme range on the log scale (since exp(2 × 0.12) ≈ 1.27)

After six months of live data, the posterior naturally blends the prior with the emerging experience. A scheme with 30 claims after six months still has meaningful uncertainty — the posterior HDI will be wide — but the centre of the distribution is already being pulled towards the data. After 18-24 months of full experience, the data dominates and the prior matters only at the hyperprior level (σ_theta).

This is exactly what B-S credibility does, but the Bayesian formulation makes the prior explicit and auditable. Under Solvency II internal model governance and the FCA's SR3/25 model risk framework, explicit parameterised priors are easier to validate than implicit ones embedded in a K ratio that emerges from method-of-moments estimation.

---

## Interpreting the variance components

The key diagnostics from either B-S or the Bayesian model are the variance components:

| Parameter | B-S notation | Bayesian analog | Interpretation |
|---|---|---|---|
| Within-group variance | v | σ²_within | Process noise: how much do outcomes vary around a group's true rate? |
| Between-group variance | a | σ²_theta | Genuine group heterogeneity: how much do true rates differ across groups? |
| Bühlmann's k | k = v/a | — | Noise-to-signal ratio |
| Credibility factor | Z = w/(w+k) | Posterior shrinkage | How much to trust a group's own data |

On UK commercial motor, typical values are v ≈ 0.015-0.025 (within-scheme year-to-year noise) and a ≈ 0.003-0.008 (genuine between-scheme rate variation). This gives k in the range 2-8 for typical fleet schemes. A scheme with 300 policy-years has Z = 300/(300+k). At k=5, Z = 0.98 — nearly fully trusted. At k=5 with only 30 policy-years, Z = 0.86 — still pulling heavily towards its own experience. The portfolio average is almost irrelevant for anything above 50 policy-years when the between-scheme variance is this large.

The Bayesian model gives you `sigma_theta` directly from the posterior. If the posterior 90% HDI for `sigma_theta` spans 0.05 to 0.25, that is an honest statement about how uncertain you are about between-scheme heterogeneity. The B-S v̂ and â are point estimates with no uncertainty quantification.

---

## MGA and delegated authority portfolios

The canonical use case where the classical model breaks hardest is an MGA portfolio where:

- The MGA writes through 15 schemes, 3 of which are new (no history)
- The remaining 12 have 2-5 years of data
- Underwriting authority varies: some schemes can write up to £10m CSL, others are restricted to standard risks

B-S treats the 3 new schemes as having Z=0 — they get the portfolio mean regardless of any prior information. In practice, the MGA onboarding process produces a view of expected loss ratios, risk appetite, and underwriting quality that should inform the starting premium. The Bayesian model encodes this as the prior on `mu_theta` per scheme (or a hierarchical prior on scheme quality by underwriting class).

For capital modelling, the scheme-level posterior also feeds directly into the aggregate distribution. The posterior samples for each scheme's random effect θ_g can be used to simulate correlated loss ratios across schemes — accounting for the fact that a bad year at the scheme level is partly driven by systematic factors (the MGA's underwriting culture) that affect all schemes.

```python
# Simulate aggregate loss ratio distribution from posterior
# Using posterior samples of theta for all schemes
theta_samples = trace.posterior["theta"].values  # (n_chains, n_draws, n_schemes)
theta_samples = theta_samples.reshape(-1, n_schemes)  # (total_draws, n_schemes)

# For each posterior draw, compute aggregate loss ratio
# given portfolio weights w_g (scheme earned premium shares)
w = earned_premium_by_scheme / earned_premium_by_scheme.sum()
aggregate_lr_samples = np.exp(theta_samples) @ w   # shape (total_draws,)

print(f"E[aggregate LR]:       {aggregate_lr_samples.mean():.3f}")
print(f"90th percentile:       {np.percentile(aggregate_lr_samples, 90):.3f}")
print(f"1-in-10 year event:    {np.percentile(aggregate_lr_samples, 90):.3f}")
```

The correlated scheme effects — because they share the hyperprior σ_theta — naturally produce heavier tails in the aggregate distribution than independent scheme models would. This is the right structure: bad underwriting years tend to be bad across schemes, not independently bad.

---

## What to change in your pricing process

The hierarchical model is not a drop-in replacement for a B-S spreadsheet. It requires:

**Tooling**: PyMC or Stan. Both run in Python. PyMC is the easier entry point — it uses NumPy-compatible syntax and the `pm.sample()` interface hides most of the MCMC complexity. Expect 2-5 minutes per model fit on a portfolio of 50 schemes and 5 years of data. This is not a real-time scoring model; it is an offline tool for pricing decisions.

**Prior specification**: someone needs to own the prior distributions and justify them. For scheme-level random effects, weakly informative priors (HalfNormal with σ=0.15-0.20 for the between-scheme standard deviation) are appropriate unless you have cedant data. Document the prior choices and check sensitivity.

**Convergence checking**: `pm.summary(trace)` gives R-hat statistics for each parameter. R-hat > 1.01 is a warning. R-hat > 1.1 means the chains have not converged and results should not be used. Thin-segment models with fewer than 20 observed claims per group will struggle — consider collapsing groups or strengthening the prior.

**Integration with the GLM**: the random effects model θ_g is a multiplicative adjustment on top of your GLM or GBM fixed-effects prediction. This means you need a clean separation between the risk-level rating model and the group-level experience model. If your current process bakes broker effects into the GLM factors, you need to strip them out before fitting the hierarchical model.

The [`insurance-credibility`](/2026/02/19/buhlmann-straub-credibility-in-python/) library handles the classical case with the `HierarchicalBuhlmannStraub` class. For the full Bayesian version with PyMC, the pattern above is our recommended starting point — it is the minimum viable model that gives you proper posterior uncertainty and the ability to inject informative priors. The Wüthrich (2025) neural mixed effects extension — replacing the GLM offset with a neural network trained jointly with the random effects — is the research frontier, but the pure-Bayesian version is sufficient for most pricing decisions.

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/) — the closed-form classical alternative that requires no MCMC and runs in under a second
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — when the Bayesian posterior intervals are too wide or too narrow, split conformal provides a distribution-free coverage guarantee
