---
layout: post
title: "Capital Modelling Basics in Python: Aggregate Distributions and Monte Carlo SCR"
description: "Build a working Solvency II SCR estimate from scratch using compound distributions and Monte Carlo simulation. Poisson/NegBin frequency, lognormal severity, 50k simulations, VaR at 99.5%. numpy and scipy only."
date: 2026-03-28
categories: [capital-modelling, tutorials]
tags: [SCR, solvency-ii, Monte-Carlo, aggregate-distributions, VaR, lognormal, negative-binomial, scipy, numpy, Python]
---

Capital modelling sits downstream of pricing in most insurance organisations, which means pricing actuaries often encounter it without having built one themselves. You know what a 1-in-200 loss means. You probably know the SCR is a VaR at 99.5%. But there is a gap between knowing the definition and being able to produce the number from a blank Python file.

This post closes that gap. We build a simple but structurally correct capital model using only numpy and scipy: fit a frequency and severity distribution, combine them into an aggregate loss distribution via Monte Carlo, and extract the Solvency II SCR. We are explicit about what this toy model is missing. The goal is for you to leave with working code you understand, not with the impression that capital modelling is solved.

---

## What we are building

Under Solvency II Article 101, the Solvency Capital Requirement is the Value-at-Risk of basic own funds at a 99.5% confidence level over a one-year horizon. In plain English: the capital you need to survive a 1-in-200 year loss year.

The standard formula approach uses prescribed factor tables. An internal model — or even an approximate internal model used for stress testing — builds the 1-in-200 from the ground up by simulating many possible loss years and reading off the 99.5th percentile. That simulation is what we are building here: purely underwriting risk, single line of business, one year.

---

## Building blocks: frequency and severity

The collective risk model decomposes aggregate loss into two independent components:

- **Frequency**: how many claims occur in the year, modelled as a random variable N
- **Severity**: how large each claim is, modelled as i.i.d. random variables X_1, X_2, ..., X_N

Total loss is the compound sum S = X_1 + X_2 + ... + X_N.

### Frequency distribution

The Poisson distribution is the default for claim counts. It has a single parameter (lambda = mean = variance) and is entirely tractable. The problem is that real claim counts are overdispersed: the variance exceeds the mean because of heterogeneity in the underlying exposure (some policyholders are just riskier than others, some years have worse weather, and so on).

The negative binomial distribution relaxes the Poisson's equidispersion constraint. It has two parameters: a mean mu and a dispersion r. Variance is mu + mu²/r. As r → ∞ it converges to Poisson. We use it here.

```python
import numpy as np
import scipy.stats as stats

# scipy's nbinom parameterises as (n, p) where:
#   n = r (dispersion parameter)
#   p = r / (r + mu)  (probability of "success" per trial)
LAMBDA_FREQ = 500   # expected annual claim count
R_DISP = 20         # dispersion: lower means more overdispersion

p_nb = R_DISP / (R_DISP + LAMBDA_FREQ)
freq_dist = stats.nbinom(n=R_DISP, p=p_nb)

print(f"Mean:     {freq_dist.mean():.1f}")   # 500.0
print(f"Variance: {freq_dist.var():.1f}")    # 13000.0  (vs 500 for Poisson)
```

**Fitting to observed data** uses method of moments: given a sample of annual claim counts, estimate r from the sample mean and variance.

```python
# claim_counts_obs: array of observed annual claim totals
sample_mean = claim_counts_obs.mean()
sample_var  = claim_counts_obs.var(ddof=1)

r_est = sample_mean**2 / (sample_var - sample_mean)
p_est = r_est / (r_est + sample_mean)

fitted_freq = stats.nbinom(n=r_est, p=p_est)
```

If `sample_var <= sample_mean`, your data is underdispersed and you would use Poisson or binomial instead. That is rare in non-life insurance.

### Severity distribution

The lognormal distribution — where log(X) is normal — is the workhorse severity distribution. It handles the right skew of most claim size distributions reasonably well at attritional and mid-range layers. It breaks down in the far tail, which is why cat models use different approaches, but for a first pass it is fine.

The lognormal's two parameters are mu_log and sigma_log, the mean and standard deviation of the underlying normal. A more intuitive parameterisation for actuaries is the mean and coefficient of variation (CV = standard deviation / mean) of the severity directly.

```python
MU_SEV = 2_000   # mean claim size (£)
CV_SEV = 0.8     # coefficient of variation

# Convert to lognormal parameters
sigma_log = np.sqrt(np.log(1 + CV_SEV**2))
mu_log    = np.log(MU_SEV) - 0.5 * sigma_log**2

sev_dist = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

print(f"Mean: {sev_dist.mean():.2f}")   # 2000.00
print(f"CV:   {np.sqrt(sev_dist.var()) / sev_dist.mean():.3f}")  # 0.800
```

**Fitting to a sample of individual claim amounts** uses maximum likelihood, which scipy handles directly:

```python
# observed_losses: array of individual paid claim amounts
sigma_hat, loc_hat, scale_hat = stats.lognorm.fit(observed_losses, floc=0)
# floc=0 fixes the location parameter at zero (standard for insurance losses)

mu_log_hat = np.log(scale_hat)
print(f"Fitted sigma_log: {sigma_hat:.3f}, mu_log: {mu_log_hat:.3f}")
```

For Pareto-tailed data (power law decay in the tail), replace `stats.lognorm` with `stats.pareto` or `stats.lomax`. The simulation machinery below does not change.

---

## Aggregate loss distribution via Monte Carlo

With frequency and severity distributions in hand, we simulate the aggregate loss S for N_SIMS independent years.

The efficient implementation avoids a Python loop over simulations. Instead, we draw all claim counts at once, draw all severities at once (total length = sum of all claim counts), then split the severity array at the cumulative count boundaries.

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# --- Parameters ---
LAMBDA_FREQ = 500      # expected annual claim count
R_DISP     = 20        # NegBin dispersion
MU_SEV     = 2_000     # mean severity (£)
CV_SEV     = 0.8       # coefficient of variation of severity
N_SIMS     = 50_000    # number of simulated years

# --- Distributions ---
p_nb      = R_DISP / (R_DISP + LAMBDA_FREQ)
freq_dist = stats.nbinom(n=R_DISP, p=p_nb)

sigma_log = np.sqrt(np.log(1 + CV_SEV**2))
mu_log    = np.log(MU_SEV) - 0.5 * sigma_log**2
sev_dist  = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

# --- Simulate ---
claim_counts   = freq_dist.rvs(size=N_SIMS, random_state=rng)
all_severities = sev_dist.rvs(size=claim_counts.sum(), random_state=rng)

# Split severity array into one chunk per simulated year
splits      = np.cumsum(claim_counts)[:-1]
agg_losses  = np.array([chunk.sum() for chunk in np.split(all_severities, splits)])
```

`agg_losses` is now an array of 50,000 simulated annual aggregate losses. The list comprehension over `np.split` is the only loop and it runs in under a second for these parameters.

---

## SCR calculation

The Solvency II SCR is the VaR at 99.5% — the loss level exceeded in only 1 of our 200 simulated years (or equivalently, the 99,750th value in a sorted array of 200,000 draws).

```python
mean_loss  = agg_losses.mean()
scr_var    = np.percentile(agg_losses, 99.5)
scr_capital = scr_var - mean_loss   # excess of expected loss

print(f"Mean aggregate loss:    £{mean_loss:,.0f}")
print(f"VaR at 99.5%:           £{scr_var:,.0f}")
print(f"SCR (excess over mean): £{scr_capital:,.0f}")
print(f"SCR as % of mean:       {100 * scr_capital / mean_loss:.1f}%")
```

With our parameters this produces approximately:

```
Mean aggregate loss:    £1,000,000
VaR at 99.5%:           £1,689,000
SCR (excess over mean): £689,000
SCR as % of mean:       69.0%
```

The 1-in-200 loss is about 1.7 times the expected loss for this parameterisation. That ratio is sensitive to the severity distribution's tail: heavier tails (higher CV, or switching to Pareto) push the ratio up sharply.

Visualise the distribution to check the result looks sensible:

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(agg_losses / 1e6, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
ax.axvline(
    scr_var / 1e6, color='crimson', linewidth=1.5,
    label=f'VaR 99.5% = £{scr_var/1e6:.2f}M'
)
ax.axvline(
    mean_loss / 1e6, color='navy', linewidth=1.5, linestyle='--',
    label=f'Mean = £{mean_loss/1e6:.2f}M'
)
ax.set_xlabel('Aggregate loss (£M)')
ax.set_ylabel('Simulated years')
ax.set_title('Aggregate loss distribution — 50,000 simulations')
ax.legend()
plt.tight_layout()
plt.show()
```

The histogram should be roughly bell-shaped with a long right tail. If it looks symmetric with thin tails, your severity CV is probably too low. If you have a large spike at zero (common in very low frequency lines), check the frequency parameters.

**Simulation error.** The 99.5th percentile is estimated from 50,000 draws. The standard error of a quantile estimate is approximately `sqrt(p*(1-p) / (n * f(q)^2))` where f(q) is the density at the quantile. In practice, re-running with different seeds and checking stability is more useful than the formula. For production use, 100,000 to 500,000 simulations are typical; 50,000 is enough to understand the shape.

---

## Where this model needs to go next

This is a single-line, single-peril model with independent frequency and severity. Real internal models add:

**Correlation between lines.** If your property and liability books both experience bad years together, naive independence overstates the diversification benefit. The standard approach is a Gaussian or t-copula on the marginal uniform transforms of each line's aggregate loss. We will cover copula-based dependency structures in a future post.

**Panjer recursion.** For discrete severity distributions and when you want the exact aggregate distribution rather than a simulation, Panjer's algorithm computes the compound distribution analytically. It is fast for moderate claim counts but becomes impractical above a few thousand claims per year. Worth knowing it exists.

**DFA frameworks.** Full dynamic financial analysis models the entire balance sheet: asset returns, reserving risk, reinsurance structure, and premium volatility together. Python packages like [aggregate](https://aggregate.readthedocs.io) (John Major's library) go significantly further than what we have built here.

**Reserving risk.** The Solvency II SCR includes risk from reserve deterioration on prior years, not just new business. Our model covers only premium risk (the aggregate loss from policies written this year). A complete model would add a reserve risk module, typically parameterised using bootstrap ODP results from your reserve analysis. See our [chain ladder post](https://burning-cost.github.io) for the reserving side.

**Extreme value theory.** For lines where the tail behaviour matters — marine, aviation, large commercial property — lognormal severity is not adequate in the far right tail. Generalised Pareto distribution fitting above a threshold (peaks-over-threshold method) gives a more defensible tail estimate.

---

## What this model is

A useful toy. If you have a single line where frequency and severity can be estimated from historical data, this approach gives you a back-of-envelope SCR that passes a sanity check. It is the kind of model you build to understand a new portfolio before the reserving team has finished their analysis, or to explain to a non-actuary why a cat-exposed line needs more capital than its expected loss suggests.

It is not a model you would submit to the PRA. The PRA expects dependency structures, parameter uncertainty, model uncertainty, data quality documentation, and an independent validation. But you cannot build any of those things if you do not understand this first.

The full working code for this post is below as a single runnable script.

---

## Complete script

```python
"""
Capital modelling basics: aggregate loss distribution and Monte Carlo SCR.
Requires: numpy, scipy, matplotlib (all standard).
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ── Parameters ─────────────────────────────────────────────────────────────
LAMBDA_FREQ = 500      # expected annual claim count
R_DISP      = 20       # NegBin dispersion (higher = less overdispersion)
MU_SEV      = 2_000    # mean severity (£)
CV_SEV      = 0.8      # coefficient of variation of severity
N_SIMS      = 50_000   # simulated years
SEED        = 42

# ── Distributions ──────────────────────────────────────────────────────────
rng   = np.random.default_rng(SEED)

p_nb      = R_DISP / (R_DISP + LAMBDA_FREQ)
freq_dist = stats.nbinom(n=R_DISP, p=p_nb)

sigma_log = np.sqrt(np.log(1 + CV_SEV**2))
mu_log    = np.log(MU_SEV) - 0.5 * sigma_log**2
sev_dist  = stats.lognorm(s=sigma_log, scale=np.exp(mu_log))

print(f"Frequency — mean: {freq_dist.mean():.1f}, variance: {freq_dist.var():.1f}")
print(f"Severity  — mean: £{sev_dist.mean():,.2f}, CV: {np.sqrt(sev_dist.var())/sev_dist.mean():.3f}")

# ── Monte Carlo simulation ──────────────────────────────────────────────────
claim_counts   = freq_dist.rvs(size=N_SIMS, random_state=rng)
all_severities = sev_dist.rvs(size=claim_counts.sum(), random_state=rng)
splits         = np.cumsum(claim_counts)[:-1]
agg_losses     = np.array([chunk.sum() for chunk in np.split(all_severities, splits)])

# ── SCR ────────────────────────────────────────────────────────────────────
mean_loss   = agg_losses.mean()
scr_var     = np.percentile(agg_losses, 99.5)
scr_capital = scr_var - mean_loss

print(f"\nMean aggregate loss:    £{mean_loss:,.0f}")
print(f"VaR at 99.5%:           £{scr_var:,.0f}")
print(f"SCR (excess over mean): £{scr_capital:,.0f}")
print(f"SCR as % of mean:       {100 * scr_capital / mean_loss:.1f}%")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(agg_losses / 1e6, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
ax.axvline(
    scr_var / 1e6, color='crimson', linewidth=1.5,
    label=f'VaR 99.5% = £{scr_var/1e6:.2f}M'
)
ax.axvline(
    mean_loss / 1e6, color='navy', linewidth=1.5, linestyle='--',
    label=f'Mean = £{mean_loss/1e6:.2f}M'
)
ax.set_xlabel('Aggregate loss (£M)')
ax.set_ylabel('Simulated years')
ax.set_title('Aggregate loss distribution — 50,000 simulations')
ax.legend()
plt.tight_layout()
plt.show()
```
