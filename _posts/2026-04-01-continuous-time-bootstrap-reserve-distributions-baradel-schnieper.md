---
layout: post
title: "Full Reserve Distributions: Continuous-Time Bootstrapping for IBNR and IBNER"
date: 2026-04-01
categories: [techniques, reserving]
tags: [reserving, IBNR, IBNER, stochastic-reserving, bootstrap, Schnieper, Mack, ODP, continuous-time, SDE, Poisson-measure, Brownian-motion, reserve-uncertainty, pricing-margin, capital-allocation, reinsurance, Baradel, arXiv-2603-11258, python]
description: "Baradel (arXiv:2603.11258) extends continuous-time bootstrapping to Schnieper's model, separating IBNR (new claims) from IBNER (cost development of known claims) and producing the full reserve distribution. Why this matters for pricing actuaries: margin-setting, capital allocation, and reinsurance purchasing all depend on the shape of the distribution, not just the mean."
seo_title: "Full Reserve Distributions via Continuous-Time Bootstrapping: Schnieper's Model Explained"
math: true
author: burning-cost
---

A reserving actuary's point estimate tells you the expected shortfall. It does not tell you how bad it could get, how asymmetric the risk is, or what the 99.5th percentile looks like for capital purposes. Pricing actuaries routinely consume reserve estimates as though they were facts, then add a fixed percentage margin and move on. That approach made more sense when the chain ladder was a spreadsheet and stochastic methods were reserved for actuarial research papers. It makes less sense now.

Baradel (arXiv:2603.11258, March 2026) provides a continuous-time bootstrap for Schnieper's reserving model — the one that separates claims not yet reported (IBNR) from changes in estimated costs of already-reported claims (IBNER). The method produces the full predictive distribution of total reserves, not just a mean and standard error. This post explains what the method does, how it differs from Mack and bootstrap ODP, and what it means in practice for a pricing actuary who needs to turn reserve uncertainty into a margin or a reinsurance structure.

---

## Why the distribution shape matters

Consider two reserve distributions with the same mean: £50m.

Distribution A is approximately Normal with standard deviation £8m. The 99.5th percentile sits around £70m. The skew is negligible.

Distribution B arises from a long-tailed development pattern on commercial liability claims — casualty business with medical inflation and social inflation risk. The mean is £50m but the distribution is heavily right-skewed. The 99.5th percentile is £95m. The 75th percentile is £58m, almost identical to Distribution A. But the tail is nearly three times as wide.

For pricing purposes, these are not equivalent reserves. They imply different premium loadings (if you are pricing the reserve risk into the product), different reinsurance attachment points, and different capital allocations under the SCR. A standard error from the Mack model tells you the mean and a rough symmetric confidence interval. It does not tell you whether you are in situation A or B.

This is the gap that stochastic reserving methods fill. We have covered Mack and bootstrap ODP [in previous posts](/2026/03/28/stochastic-reserving-python-bootstrap-odp/). Both are well-implemented in `chainladder`. Both operate on development triangles. Both work at the aggregate level — one cell per origin-period development-period combination.

Schnieper's model, and the continuous-time bootstrap that Baradel applies to it, goes further in two directions: it separates the sources of uncertainty, and it works with more granular claim-level information where available.

---

## Schnieper's model: what it adds

Hans Ulrich Schnieper introduced his model in 1991 ("Separating True IBNR Claims from IBNER Claims," ASTIN Bulletin). The insight is that the standard triangle conflates two different phenomena:

**IBNR** — Incurred But Not Reported. Claims that have occurred but have not yet been notified to the insurer. Pure new claim arrival risk. The uncertainty here is: how many more claims will emerge?

**IBNER** — Incurred But Not Enough Reserved. Claims that have been reported but whose estimated cost is still developing. Case reserve development risk. The uncertainty here is: for the claims we know about, how much will their final cost differ from the current case reserve?

The chain ladder, Mack, and bootstrap ODP do not distinguish between these. They model total cumulative paid or incurred losses and project the development factors forward. If IBNR is large (long-tailed class, late-reported claims) or IBNER volatility is high (casualty, liability, anything with protracted settlement), the aggregate model is blending two very different risk processes.

Schnieper's separation uses the information that reserving actuaries typically have but standard triangle methods throw away: the current case reserve for each reported claim. At each development period, you can observe:
- How many new claims have been reported (the IBNR arrival process)
- The aggregate case reserves for already-reported claims (the IBNER development process)

From these two series, Schnieper estimates the claim arrival intensity and the case reserve development pattern separately. The total reserve distribution is the sum of two independent components.

---

## Baradel's continuous-time framework

Baradel's paper (arXiv:2603.11258) follows the approach from his earlier paper on chain-ladder (arXiv:2406.03252, published in Insurance: Mathematics and Economics) — taking a classical discrete-time reserving model and recasting it in continuous time using stochastic calculus. This is not just an academic exercise; it changes the bootstrap.

The continuous-time formulation models two stochastic processes:

**Claim arrivals.** New IBNR claims arrive according to a Poisson random measure. In discrete time, this is modelled by the incremental new-claim counts at each development period. In continuous time, it is a point process with a deterministic intensity function $\lambda(t)$ to be estimated. The total IBNR at time $T$ is:

$$N(T) = \int_0^T \lambda(s)\, \mathrm{d}s + \text{martingale noise}$$

The intensity $\lambda(t)$ captures the pattern of claim emergence over development time. It is high in early development periods (many new claims arriving quickly after the accident date) and drops toward zero as the portfolio approaches full development.

**Case reserve development.** The aggregate case reserve for reported claims evolves as a diffusion — a process with a drift term reflecting systematic reserve strengthening or release, and a Brownian motion term reflecting idiosyncratic claim-by-claim volatility:

$$\mathrm{d}R(t) = \mu(t)\, \mathrm{d}t + \sigma(t)\, \mathrm{d}W(t)$$

Here $R(t)$ is the aggregate case reserve at development time $t$, $\mu(t)$ is the drift (systematic reserve development), and $\sigma(t)W(t)$ is the noise. The Brownian motion $W$ is assumed independent of the Poisson claim arrival process $N$.

The total reserve at the valuation date is then $R(T) + \mathbb{E}[N(\infty) - N(T)] \times \bar{c}$, where $\bar{c}$ is the expected cost per future IBNR claim. The continuous-time formulation gives Baradel the tools to derive the full joint distribution analytically and then bootstrap residuals to get the empirical distribution.

---

## The bootstrap procedure

The bootstrap is a residual bootstrap — the same general idea as bootstrap ODP, but applied to continuous-time residuals rather than Pearson residuals from a GLM.

**Step 1: Fit the intensity function $\lambda(t)$.** Estimate the claim arrival intensity from the historical new-claim counts. Baradel uses a non-parametric kernel smoother; in practice you would use whatever functional form fits your data — an exponential decay works for most short-tail classes.

**Step 2: Fit the diffusion parameters $\mu(t)$ and $\sigma(t)$.** These are estimated from the incremental changes in aggregate case reserves. $\mu(t)$ is the empirical mean of increments at each development lag; $\sigma^2(t)$ is the empirical variance.

**Step 3: Compute residuals.** For the IBNR process, residuals are the differences between observed new-claim counts and fitted intensity (normalised by $\sqrt{\lambda(t)}$ for variance stabilisation). For the IBNER process, residuals are normalised diffusion increments.

**Step 4: Bootstrap.** Draw with replacement from the IBNR residuals and IBNER residuals independently. Reconstruct simulated claim arrival paths and case reserve development paths. Compute the total reserve for each simulation. Repeat 10,000 times (or as many as your runtime budget allows).

The output is a vector of 10,000 simulated total reserves. Take quantiles directly. No distributional assumption, no lognormal fit, no Normal approximation. The shape that emerges is the shape of the data.

---

## How it compares to Mack and bootstrap ODP

It is worth being precise about what each method does and does not do, because they are often described as if they are approximate versions of the same thing.

**Mack (1993).** Analytical, distribution-free. Produces a mean and standard error for each origin year, with a covariance adjustment for the total. The confidence interval assumes approximate normality (or lognormality with a correction). Cannot produce percentile tables directly without a distributional assumption. Does not separate IBNR from IBNER. Works well for short-to-medium tail classes where the Normal approximation is reasonable.

**Bootstrap ODP (England and Verrall, 2002).** Simulation-based. Fits an over-dispersed Poisson GLM to the incremental triangle, resamples Pearson residuals, re-fits the model, and projects forward. Produces a full empirical distribution — no distributional assumption needed at the output stage. Does not separate IBNR from IBNER. Sensitive to triangle outliers (a single aberrant residual can propagate through many simulations). The standard implementation in `chainladder` uses 5,000 simulations and is fast.

**Baradel's continuous-time Schnieper bootstrap.** Simulation-based. Separates IBNR and IBNER processes. The continuous-time formulation means the residuals have cleaner theoretical properties — the noise is, by construction, a martingale difference sequence, which makes the bootstrap asymptotically valid under weaker conditions than the ODP bootstrap. Requires more data: you need incremental new-claim counts and incremental case reserve movements, not just paid or incurred aggregate totals. This is available from most reserving systems but not from a standard triangle extract.

The practical upshot: if you are working with personal lines data and a short-tail triangle, bootstrap ODP is fine and takes 10 lines of Python. If you are working with liability, casualty, commercial property, or any class where claim reporting patterns and reserve development are distinct risk processes you want to understand separately — Schnieper's approach gives you that decomposition.

---

## What this means for pricing actuaries

Reserving uncertainty feeds into pricing in three places. The continuous-time bootstrap changes what you can say in each.

**1. Risk margin and premium loading.** The Solvency II cost-of-capital approach to risk margin requires the full distribution of future SCRs, projected forward. In practice, most firms approximate this by scaling the best estimate. A better approach — and one the PRA has increasingly expected to see evidenced — is to derive the loading from a quantile of the reserve distribution. If the 75th percentile of your reserve distribution is 15% above the mean, and your target is to hold the 75th percentile in premium, your reserve loading is 15%. Deriving this from a simulation rather than a Mack standard error removes the normality assumption that makes Mack inadequate for long-tail classes.

**2. Capital allocation to lines of business.** Under an internal model (or a standard formula SF/RF adjustment), the reserve SCR for a line of business depends on the 99.5th percentile of the one-year reserve movement. The bootstrap gives you this directly. The standard Mack approach requires assuming a distribution for the reserve, then computing the tail analytically. For skewed classes — long-tail liability, casualty, credit — the Mack-derived tail is consistently underestimated.

**3. Reinsurance structure.** If you are purchasing aggregate stop-loss or per-period excess of loss on your reserve development, you are essentially buying protection against the right tail of the reserve distribution. The reinsurance pricing will be based on that tail. Understanding the tail shape — not just the mean and standard deviation — lets you evaluate whether the reinsurance price is reasonable and where the optimal attachment point sits. A bootstrap that produces asymmetric distributions tells you more than a Mack standard error that implicitly assumes symmetry.

---

## A sketch of the implementation

No open-source Python implementation of the continuous-time Schnieper bootstrap exists as of April 2026. The `chainladder` library (v0.9.1) implements Mack and bootstrap ODP but not the Schnieper separation or the continuous-time framework. This is a gap we are monitoring.

In the meantime, here is a sketch of the core simulation loop to illustrate the structure. This is not production code — it is a direct translation of the Baradel procedure to make the bootstrap structure concrete.

```python
import numpy as np
from scipy.stats import poisson

def fit_arrival_intensity(new_claims: np.ndarray, exposure: np.ndarray) -> np.ndarray:
    """
    Estimate claim arrival intensity lambda(t) from incremental new-claim counts.
    new_claims: shape (n_periods,), incremental new claims per development period
    exposure:   shape (n_periods,), number of origin years at risk in each period
    Returns:    shape (n_periods,), estimated intensity per origin year per period
    """
    return new_claims / exposure

def fit_diffusion_params(delta_reserves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate drift mu(t) and volatility sigma(t) from incremental case reserve changes.
    delta_reserves: shape (n_periods, n_origin_years), incremental reserve movements
    Returns: mu (n_periods,), sigma (n_periods,)
    """
    mu = np.nanmean(delta_reserves, axis=1)
    sigma = np.nanstd(delta_reserves, axis=1, ddof=1)
    return mu, sigma

def bootstrap_reserve_distribution(
    lambda_hat: np.ndarray,
    mu_hat: np.ndarray,
    sigma_hat: np.ndarray,
    ibnr_residuals: np.ndarray,  # shape (n_periods, n_origin_years)
    ibner_residuals: np.ndarray, # shape (n_periods, n_origin_years)
    n_future_periods: int,
    mean_cost_per_claim: float,
    n_simulations: int = 10_000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Returns array of shape (n_simulations,) with simulated total reserves.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total_reserves = np.zeros(n_simulations)

    for i in range(n_simulations):
        # --- IBNR component ---
        # Resample residuals for future periods
        ibnr_sim = np.zeros(n_future_periods)
        for t in range(n_future_periods):
            # Draw a residual from historical IBNR residuals at this lag
            residual = rng.choice(ibnr_residuals[t])
            # Reconstruct simulated new claim count
            ibnr_sim[t] = max(0.0, lambda_hat[t] + residual * np.sqrt(lambda_hat[t]))

        ibnr_reserve = ibnr_sim.sum() * mean_cost_per_claim

        # --- IBNER component ---
        ibner_sim = np.zeros(n_future_periods)
        for t in range(n_future_periods):
            residual = rng.choice(ibner_residuals[t])
            ibner_sim[t] = mu_hat[t] + sigma_hat[t] * residual

        ibner_reserve = ibner_sim.sum()

        total_reserves[i] = ibnr_reserve + ibner_reserve

    return total_reserves


# --- Usage ---
reserves = bootstrap_reserve_distribution(
    lambda_hat=lambda_hat,
    mu_hat=mu_hat,
    sigma_hat=sigma_hat,
    ibnr_residuals=ibnr_resids,
    ibner_residuals=ibner_resids,
    n_future_periods=8,
    mean_cost_per_claim=12_500.0,
    n_simulations=10_000,
)

percentiles = [50, 75, 90, 95, 99, 99.5]
print("Reserve distribution percentiles:")
for p in percentiles:
    print(f"  {p:5.1f}th: £{np.percentile(reserves, p):>12,.0f}")
```

A few implementation notes. The residual normalisation matters: IBNR residuals should be normalised by $\sqrt{\hat{\lambda}(t)}$ to stabilise variance before resampling, so the bootstrap does not over-represent high-intensity periods. IBNER residuals should be standardised by the estimated $\hat{\sigma}(t)$. The sketch above shows the structure; Baradel's paper (section 3.2) gives the exact normalisation.

The independence assumption — that IBNR arrivals and IBNER development are independent — is material. For some classes this is reasonable. For casualty lines where new claims arriving late tend to be more severe (solicitor-encouraged late reporting, for example), the independence assumption is wrong and the simulation will underestimate tail risk. Checking the empirical correlation between new-claim counts and contemporaneous reserve strengthening is a sensible diagnostic before applying the method.

---

## The point estimate problem

We started this post by saying that pricing actuaries consume reserve estimates as though they were facts. The continuous-time bootstrap does not change the mean — a calibrated bootstrap should reproduce the same central estimate as the chain ladder. What it does is make the uncertainty explicit, and do so in a way that does not require a distributional assumption.

For a short-tail personal lines book, Mack or ODP is probably enough. The Normal approximation is not terrible, and the added complexity of a continuous-time SDE calibration is not justified.

For commercial liability, professional indemnity, motor injury (bodily injury claims with long settlement patterns), or any book where IBNR and IBNER risks are distinct and material, the Baradel method gives you something the standard methods cannot: the full distribution, split by source, without normality assumptions.

That distribution feeds directly into pricing decisions. How much reserve risk loading to include in a commercial casualty rate. Where to set the aggregate stop-loss attachment on a quota share treaty. How much capital to allocate to the long-tail book versus the short-tail book in your internal model. These are not abstract questions. They are the questions that define whether a pricing actuary is doing their job properly.

The distribution matters. The point estimate does not.

---

*The paper: Nicolas Baradel, "Continuous-time modeling and bootstrap for Schnieper's reserving," arXiv:2603.11258, March 2026. For background on Mack and bootstrap ODP with Python code, see our earlier [stochastic reserving post](/2026/03/28/stochastic-reserving-python-bootstrap-odp/).*
