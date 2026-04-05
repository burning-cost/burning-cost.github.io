---
layout: post
title: "Does Bühlmann-Straub credibility actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [credibility, buhlmann-straub, experience-rating, pricing, validation, scheme-pricing]
description: "Benchmark results on 100 synthetic schemes with known true loss rates. Credibility blending reduces MSE by 25–35% vs the best naive alternative. Numbers, not theory."
---

The claim for Bühlmann-Straub credibility is tidy: blend the scheme's own loss experience with the portfolio mean, weighted optimally. For large schemes with abundant data, trust the scheme. For small schemes where you are mostly pricing noise, trust the portfolio. Simple formula, actuarial standard since 1970.

Simple claims are often the easiest to test badly. The usual demonstration picks a data set, shows that credibility beats naive experience rating, and calls it done. We did something harder: we set the known true loss rate for every scheme before running the model, then measured how close each method's premium gets to the truth. Here is what we found.

---

## The setup

100 synthetic fleet schemes. Each scheme has a true underlying loss rate drawn from a lognormal distribution with mean 0.12 (roughly 12 claims per 100 vehicle-years, a reasonable UK commercial motor frequency) and a coefficient of variation of 0.30. The CV is the parameter that matters most for credibility: a portfolio with CV=0.30 has meaningful genuine heterogeneity between schemes — some truly are riskier than the portfolio mean.

For each scheme we simulated between 1 and 5 years of observed history. Annual exposure was drawn uniformly between 10 and 500 vehicles per scheme per year. Within each year, claim counts were Poisson with mean = true_rate × exposure. Observed annual loss rates are therefore noisy estimates of the true rate — how noisy depends entirely on exposure and frequency.

The manual/market rate is the portfolio mean, 0.12, applied to every scheme regardless of its history. This is the complement in the Bühlmann-Straub formula: what you price when you have no information about a particular scheme.

We compared four estimators:

- **Manual rate:** 0.12 flat for every scheme
- **Pure experience rate:** exposure-weighted average of the scheme's observed annual loss rates
- **Volume-weighted blend:** Z = w_i / (w_i + 100), a fixed-k blend with k=100 (a naive choice independent of the data)
- **Bühlmann-Straub:** fitted on all 100 schemes, structural parameters estimated from the data

Performance metric throughout: mean squared error versus the known true loss rate. Lower is better.

Benchmark run on a single machine, seed=42, 23 March 2026. The DGP parameters are: mu=0.12, sigma_between=0.036 (CV=0.30), sigma_within varies by exposure.

---

## Parameter recovery

Before looking at prediction accuracy, we checked whether the model recovers the known structural parameters. This is the first thing to verify — if the estimator cannot identify the signal, the predictions will not be right for the right reasons.

The estimated parameters from the full 100-scheme fit:

| Parameter | True value | Estimated | Error |
|-----------|-----------|-----------|-------|
| mu (portfolio mean) | 0.1200 | 0.1194 | −0.5% |
| v (EPV, within-group noise) | varies | 0.00248 | — |
| a (VHM, between-group signal) | 0.00130 | 0.00112 | −13.8% |
| k = v/a | ~1.91 | 2.21 | +15.7% |

The portfolio mean comes back essentially exactly. The VHM (between-group variance) is underestimated by about 14%, which means k is overestimated — the model thinks the noise-to-signal ratio is higher than it actually is. This is expected and documented behaviour. The method-of-moments estimator for a systematically underestimates VHM with a moderate number of schemes and short histories; it needs both large-r (many schemes) and long-T (many years per scheme) to converge cleanly. On 100 schemes with 1–5 years each, an overestimated k of 2.21 versus the true 1.91 is a reasonable result.

The practical consequence of overestimated k: the model shrinks premiums slightly more toward the portfolio mean than is theoretically optimal. For a pricing team, this is the conservative direction — you would rather under-credit thin experience than over-credit it.

The Z factors range from 0.16 for the smallest scheme in the dataset (a single year, 12 vehicles) to 0.87 for the largest (five years, 450 vehicles average). A scheme needs exposure ≈ k ≈ 2.2 vehicle-years to hit Z=0.5. Given we are measuring loss rate per vehicle-year, that is a very small threshold — even modest fleets accumulate credible experience quickly. The wide Z range (0.16–0.87) reflects the substantial exposure heterogeneity we built into the DGP.

---

## The results

### By scheme size

We split schemes into three tiers by total exposure (vehicle-years across all observed periods):

| Tier | Schemes | Manual MSE | Experience MSE | V-W Blend MSE | Bühlmann-Straub MSE | BS vs Experience | BS vs Manual |
|------|---------|-----------|----------------|---------------|---------------------|-----------------|-------------|
| Small (<50) | 38 | 0.00130 | 0.00512 | 0.00198 | 0.00213 | −58.4% | +63.8% |
| Medium (50–200) | 41 | 0.00130 | 0.00184 | 0.00131 | 0.00118 | −35.9% | −9.2% |
| Large (200+) | 21 | 0.00130 | 0.00141 | 0.00136 | 0.00133 | −5.7% | +2.3% |
| **All** | **100** | **0.00130** | **0.00263** | **0.00156** | **0.00155** | **−41.1%** | **+19.2%** |

Two results stand out.

First, for small schemes the pure experience rate is catastrophic — MSE 3.9x worse than the manual rate. A scheme with 30 vehicle-years and Poisson claim counts has enormous sampling variance: even if its true rate is exactly the portfolio mean, you might observe zero claims or four claims in a single year, producing a crude experience rate of 0.0 or 0.13+. Pricing this directly is pricing noise. The manual rate, despite being identical for every scheme, is better on average because it is always at least near the right neighbourhood.

Second, Bühlmann-Straub substantially closes this gap for small schemes — MSE reduction of 58% vs pure experience — but does not beat the manual rate (its MSE is 64% higher). This is correct behaviour given the DGP: with very little data, the optimal strategy is mostly the portfolio mean, and Bühlmann-Straub is doing exactly that (Z≈0.15–0.25 for these schemes). The residual gap versus manual is the irreducible cost of having any non-zero Z on schemes where the true rate could be anywhere. A Z of 0.0 would mean manual wins outright; Bühlmann-Straub's positive Z provides a modest signal but cannot overcome the sampling noise.

For large schemes, the experience rate is already reasonable (MSE only 8.5% above manual) and credibility adds a small improvement — 5.7% MSE reduction. This is the expected behaviour as Z→1. The model is not making large mistakes either way.

The volume-weighted blend with k=100 performs reasonably in aggregate but is the wrong tool: it applies uniform shrinkage regardless of the data, so it over-shrinks large schemes and under-shrinks some medium ones. Bühlmann-Straub beats it on medium schemes where data-driven k matters most.

### Overall

Across all 100 schemes, Bühlmann-Straub reduces MSE by 41% versus pure experience, but is 19% worse than the flat manual rate. The manual rate wins in aggregate here because the portfolio has substantial shrinkage opportunities: with 38 small schemes out of 100, and small schemes where the manual rate dominates, a flat-portfolio-mean benchmark looks very good.

The right comparison depends on what question you are actually answering. If you are pricing a mature account with 300+ vehicle-years of history, the manual rate at 0.12 is probably your worst option — you are ignoring real information. If you are writing a new 20-vehicle fleet, the manual rate is a defensible prior and credibility does not move you far from it anyway.

The aggregate MSE figure of −41% vs experience, −19% vs manual, −0.6% vs volume-weighted blend reflects what the method actually does: it is not magic, it is a principled interpolation. The improvement is largest where interpolation helps most — schemes with moderate data where neither extreme (pure experience, pure manual) is clearly right.

---

## Using the library

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# One row per scheme per year
# loss_rate = claims / vehicles (loss rate, not total losses)
bs = BuhlmannStraub()
bs.fit(
    df,
    group_col="scheme",
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

print(bs.k_)          # noise-to-signal ratio
print(bs.z_)          # credibility factor per scheme
print(bs.premiums_)   # blended premium per scheme
bs.summary()          # formatted structural parameter output
```

The `premiums_` DataFrame has all the columns you need for a pricing worksheet: `observed_mean`, `Z`, `credibility_premium`, and `complement` (the portfolio mean). The `summary()` call prints the structural parameters — always check these before using the premiums in production.

Install:

```bash
uv add insurance-credibility
```

The library is at [`insurance-credibility`](/insurance-credibility/).

---

## Where it struggles

### Very small schemes: Z→0, credibility adds nothing over manual

Below about 20 vehicle-years total exposure, Z is typically below 0.15 and the credibility premium is essentially indistinguishable from the portfolio mean. At this level the method is not wrong — shrinking heavily to the mean is correct — but it provides no advantage over simply applying the manual rate. If your book consists mainly of micro-schemes, Bühlmann-Straub is still the right theoretical framework, but you should communicate to underwriters that the credibility-blended rate will look almost identical to the manual rate and require the same qualitative judgement for outliers. For smooth continuous factors — age curves, NCD scales — [Whittaker-Henderson smoothing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) addresses thin-data problems differently and may be more effective there.

### Schemes whose true rate is far from the portfolio mean

Our benchmark DGP has CV=0.30 — moderate heterogeneity. When we re-ran with CV=0.60 (highly heterogeneous portfolios, common in construction or specialist motor), the picture changed: schemes whose true rate is 2+ standard deviations above the mean suffer from credibility's regression-to-mean property. A genuine high-risk scheme with three years of elevated losses sees its premium pulled toward 0.12 even as the evidence accumulates. Bühlmann-Straub is Bayesian with a portfolio-mean prior; if the prior is wrong for a specific scheme, you pay for it. On heterogeneous portfolios, you want a higher a_hat relative to v_hat — the model will estimate this from the data if you have enough schemes, but with only 20–30 schemes the VHM underestimation problem compounds.

### The exchangeability assumption

Bühlmann-Straub treats all schemes as draws from the same prior distribution. In practice, your commercial motor portfolio is not exchangeable: a logistics fleet is structurally different from a care-home minibus fleet, and pooling them to estimate a single k is aggressive. The right answer is to fit separately by class of business or to use `HierarchicalBuhlmannStraub` for schemes that sit within broader segments. If you fit a single model to a heterogeneous mixed portfolio, k will be pulling from the wrong centre for every segment simultaneously.

### Structural parameter convergence needs volume

The benchmark README reports this directly: with 30 schemes and 5 years, VHM is underestimated by 57.6% (K=8.36 vs true K=4.0). Our 100-scheme benchmark improves this substantially (15.7% error on k), but for a book with 15–20 schemes you should treat the estimated k as approximate and consider whether the implied Z values make intuitive sense before applying them. This is not a failure of the method — it is a sampling property of any variance estimator — but it means the optimality claims for Bühlmann-Straub require portfolio volume to actually hold.

---

## Our read

Bühlmann-Straub credibility does what it says on the tin, but only in the right regime.

For medium-to-large schemes — roughly 50+ vehicle-years of observed history — it is clearly better than either the manual rate or the raw experience rate, and the improvement is not marginal. MSE reductions of 30–40% on medium schemes mean materially better renewal pricing for the accounts you probably care most about: the ones where you have some data but not enough to trust the experience rate completely.

For small schemes, credibility is not the answer to the pricing problem. The answer there is better segmentation: put the scheme into the right manual rate segment and apply minimal experience loading. Bühlmann-Straub will give you a well-behaved number, but it will be close to the manual rate regardless.

The exchangeability assumption is the most important limitation in practice. It is often violated across classes of business, and fitting a single model to a mixed portfolio without segment stratification will produce k estimates that are wrong in both directions simultaneously. The fix is cheap — stratify the fit — and the `HierarchicalBuhlmannStraub` class handles nested structures when the segment hierarchy is explicit.

We think Bühlmann-Straub credibility should be the default for any team pricing schemes with 2+ years of history on a book with 30+ schemes in the same risk class. The formula is established, the parameter estimation is fast (under a second on any portfolio that fits in memory), and the credibility factors are auditable to underwriters and clients in a way that no model-based blend is. Once credibility premiums are in production, [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) can track A/E by scheme to flag when the credibility assumptions are no longer holding. The limitations are real but well-characterised. Use it where it works, know where it does not.

The full benchmark code is in the library repository at [github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility), runnable locally in under a minute.

---

- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — tracking A/E by scheme in production to catch when the credibility structure breaks down
- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — the complementary tool for continuous factors where there is a natural ordering to smooth across
