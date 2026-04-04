---
layout: post
title: "MGA Entry Pricing: A Four-Stage Architecture from Day Zero to Year Three"
date: 2026-04-01
categories: [techniques, pricing]
tags: [mga, entry-pricing, abc, approximate-bayesian-computation, isotonic-regression, insurance-credibility, insurance-causal, insurance-thin-data, buhlmann-straub, dml, double-machine-learning, covariate-shift, fca, prod4, fair-value, lloyds, capacity, pcw, uk-insurance, personal-lines, motor-insurance, pet-insurance, market-data, selection-bias, python]
description: "An MGA launching on a UK PCW needs prices on day one with zero claims history. Here is the full architecture: market ABC as the prior, Bühlmann-Straub blending as claims arrive, DML correction for selection bias at year two, and covariate shift monitoring throughout."
math: true
author: burning-cost
---

An MGA launching on a UK PCW has a pricing problem with no clean solution. You need prices on day one. You have zero claims history. Your capacity provider needs documented evidence of pricing soundness. And the FCA's PROD 4 fair value requirement means "we priced to match competitors" is not a standalone answer.

The industry's common responses — reverse-engineering competitor GLMs from quote scrapes, buying Consumer Intelligence rate indices before you qualify for access, relying on the capacity provider's aggregate statistics — all have the same flaw: they tell you what the market charges, not what the risk costs. Those two things are not the same, particularly in a market where PCW rank dynamics mean carriers are pricing to acquisition volume rather than pure expected cost.

There is a principled four-stage architecture that takes an MGA from day zero to a mature claims-experience-driven pricing model. Each stage has a specific library or method, a clear trigger for moving to the next stage, and an explicit answer to the FCA question: "show us your documented risk basis."

---

## Stage 1: Market ABC (Day zero to launch)

The Goffard, Piette, and Peters method (ASTIN Bulletin 2025, arXiv:2502.04082) is designed exactly for this moment. It takes competitor commercial premiums as input and produces a posterior distribution over claim frequency λ and log-mean severity μ as output. No internal claims data required at any stage.

The mechanism is Approximate Bayesian Computation with a Sequential Monte Carlo sampler and an isotonic regression link. You sample candidate (λ, μ) pairs, simulate what the expected claims cost would be under those parameters, apply PAVA isotonic regression to map simulated costs to commercial premiums, and compare the result against observed market quotes. The loop runs for up to nine generations, tightening a tolerance threshold each time. The posterior you end up with is a distribution over which (λ, μ) values are consistent with the market quotes you collected.

Our [earlier post on the Python implementation](/2026/03/31/market-based-ratemaking-python-implementation/) has the full code — around 200 lines covering the forward model, PAVA link, loss ratio penalty, and ABC-SMC loop. We will not repeat it here.

**What to collect.** For UK motor, manually harvest PCW quotes across a synthetic risk grid: vehicle group, driver age band, territory, and NCD class. Aim for 200-500 profiles across 20-50 effective risk classes. Collect Tuesday-Wednesday mid-week when rates are most stable, over 2-3 weeks, and take the median by risk class across the collection window. Single-day snapshots are not representative — motor rates shift materially after weekend claims processing clears on Monday morning.

**The loss ratio corridor is the most important prior.** For UK motor in 2025-2026, use [55%, 75%] — combined ratios are running 105-115% and the loss component is in that range for well-managed books. For home contents use [45%, 65%]. For pet, [50%, 75%] given UK vet cost inflation. Getting this wrong biases all downstream estimates in the same direction. Do not treat it as a sensitivity parameter; calibrate it carefully from ABI statistics or reinsurer benchmarks before the loop runs.

**What the FCA wants.** A documented, quantitative link between your launch rates and the underlying risk. The ABC posterior provides that: here are the market quotes we collected, here is the model we fitted, here is the posterior over claim frequency and severity, here is the MAP estimate that became our base rate. That is a fair value evidence chain. A screenshot of competitor quotes with a note saying "we came in at third cheapest" is not.

**UK product fit.** Pet insurance and travel insurance are the cleanest applications — low-dimensional, homogeneous coverage structures, well-defined perils. Home contents is workable. Standard UK motor is harder (PCW quotes mix coverage tiers, telematics inclusions, rank-targeting noise), but viable as an intercept calibration: use the ABC posterior to calibrate your average rate level, not the full rating relativities.

---

## Stage 2: Bühlmann-Straub blending (Launch to Month 18)

The market ABC posterior gives you MAP estimates (λ̂, μ̂) and a posterior variance. Both feed directly into a Bühlmann-Straub credibility model.

The Bühlmann-Straub formula:

$$\hat{\lambda}_i = Z_i \bar{Y}_i + (1 - Z_i) \mu_0$$

where μ₀ = λ̂ from the ABC MAP, the within-group variance σ² comes from the posterior variance, and the credibility weight:

$$Z_i = \frac{n_i}{n_i + K}, \quad K = \frac{\sigma^2}{\tau^2}$$

At launch, nᵢ = 0, Z = 0, and your price is the ABC MAP estimate — the market prior. As claims arrive over the first 12-18 months, Z rises. By month 18 at reasonable volume, Z is typically 0.3-0.5 and your own experience is meaningfully influencing rates. By year three you have enough data to run a proper GLM and the credibility weight is approaching 1.

The key relationship between the ABC output and the credibility prior: the *spread* of the ABC posterior determines how quickly Z rises. A tight posterior — you collected 400 quotes, they were consistent, the ABC converged cleanly — means K is small and Z rises fast. Your own experience takes over quickly. A wide posterior — few quotes, heterogeneous coverage structures, the tolerance did not close far — means K is large, the market prior is uncertain, and you stay close to that prior for longer. This is exactly right: uncertainty in the starting estimate should translate into more data-hungry credibility blending.

The [`insurance-credibility`](/insurance-credibility/) library handles the blending:

```python
from insurance_credibility import BuhlmannStraub

# ABC posterior variance on lambda
lambda_posterior = particles[:, 0]
lambda_var = lambda_posterior.var()

model = BuhlmannStraub(
    collective_mean=lam_hat,       # mu_0: ABC MAP estimate
    within_variance=lambda_var,    # sigma^2 from ABC posterior
)
```

The library also handles NCD as a Markov chain for the transition and stationary distribution calculations described in [our NCD post](/2026/02/27/experience-rating-ncd-bonus-malus/). If your MGA is writing motor, the credibility model and the NCD model interact: the ABC prior is your starting frequency estimate, and the NCD factor is applied multiplicatively as the policyholder accumulates history.

---

## Stage 3: Selection bias correction (Year 2+)

By the end of Year 1 you have internal claims data. It is not a random sample of the market. Customers who bought your product at launch prices are the risks where your price was competitive — which means your observed claim frequency is systematically correlated with your pricing structure in ways a naive GLM cannot untangle.

If your ABC posterior put your French Bulldog rate 15% above the market median and your Labrador rate 3% below, the dogs you actually wrote skew towards Labradors. Your Labrador claims experience in Year 1 is rich; your French Bulldog experience is sparse and adversely selected from the fraction of French Bulldog owners willing to pay above-market rates. A straight frequency GLM on Year 1 data will produce biased relativities because of this selection mechanism — not random sampling variation, but structural confounding between your pricing decisions and your book composition.

Double Machine Learning corrects this. The treatment variable is your relative price position for each risk class at the time of sale (your price minus the market median for that profile). The outcome is claim frequency. DML estimates the causal effect of risk characteristics on frequency by partialling out the price-risk confounding in two residualisation steps: predict the treatment from risk covariates (to get residual pricing), predict the outcome from risk covariates (to get residual frequency), then regress residual frequency on residual pricing. The result is a frequency estimate that is not contaminated by the pricing selection effect.

The [`insurance-causal`](/insurance-causal/) library implements DML for this use case. It requires the market price at the time of sale for each policy — which you will have if you collected a PCW quote grid at launch and track your own price relative to that grid. Without the market price reference, the confounding cannot be separated.

This is the stage where the data collection protocol at Stage 1 pays dividends. If you recorded your price and the market prices at the time of writing each policy, you have the treatment variable. If you only recorded your own price and not the market context, you cannot do the correction cleanly.

---

## Stage 4: Covariate shift monitoring (Continuous)

The ABC assumption is that competitor quotes represent the same risk population as your target book. This breaks down if you attract a niche segment — policyholders for whom your pricing happened to be unusually competitive because of an idiosyncrasy in your rating structure. The ABC posterior is the right prior for the quote population; it may be a systematically biased prior for the actual written population.

The test is straightforward: compare the distribution of risk characteristics in your quote population (the synthetic grid you used to collect market data at Stage 1) against the distribution in your actual written business. If the distributions diverge materially — your written business is skewed towards certain ages, territories, or vehicle groups relative to the quote grid — your ABC prior is extrapolating to the wrong risk mix.

The [`insurance-thin-data`](/tools/) library provides a `CovariateShiftTest` class using Maximum Mean Discrepancy with a permutation test. Run it quarterly against the Stage 1 quote grid as your reference. A significant MMD test result means your written book has drifted from the population the ABC was calibrated on, and the credibility prior should be updated to reflect that.

Practically, this matters most in the first 12 months when Z is still low and your prices are heavily influenced by the ABC prior. If your book composition has drifted and you have not corrected the prior, you are mispricing the drift segment using the wrong expected frequency. By Year 2 when your credibility weight is 0.4-0.5, the drift effect is partially absorbed by your own data — but it is still worth monitoring, particularly if the drift is into a segment with materially different loss experience.

---

## The full pipeline

```
Stage 1 — Pre-launch
  PCW manual quotes (200-500 profiles)
  → ABC-SMC with Poisson-LogNormal (arXiv:2502.04082)
  → Posterior over (lambda, mu): MAP = launch rates, variance = credibility prior

Stage 2 — Launch to Month 18
  ABC MAP + posterior variance → BuhlmannStraub prior in insurance-credibility
  Credibility weight Z: 0 at launch, rises as claims arrive
  Z > 0.5 at ~Month 18 for well-written lines with 100+ claims

Stage 3 — Year 2+
  DML via insurance-causal: correct for selection bias in own claims data
  Treatment = relative price position at time of sale (your price vs market)
  Outcome = claim frequency
  Year 3+: sufficient data for own GLM; credibility-blend with ABC prior fading out

Stage 4 — Continuous
  CovariateShiftTest (insurance-thin-data): quarterly MMD against Stage 1 quote grid
  Alert if written book diverges from quote population
  Recalibrate ABC prior if drift is sustained
```

Each stage has a defined trigger. Move from Stage 1 to Stage 2 when you have your first claims. Move from Stage 2 to Stage 3 when you have enough claims volume to estimate price sensitivity (typically 500+ policies). Move to a full own GLM in Stage 3 when the credibility weight Z exceeds 0.8 on your core segments. Stage 4 runs throughout.

---

## The Lloyd's angle

New Lloyd's syndicates face a structurally identical problem. The February 2024 Capital Guidance tightened requirements for documented links between pricing assumptions and SCR inputs. A new syndicate writing personal lines (a small cohort — Lloyd's is primarily commercial and specialty — but growing post-Brexit) submits a Business Plan with pricing assumptions feeding into the SCR calculation. The 35% FAL uplift provides a buffer for model uncertainty, but it does not substitute for documented methodology.

The ABC posterior is a pricing assumption in a form the model office can use: here is our prior over claim frequency, here is the variance, here is the 80% credible interval. That is stronger than "we priced to target market." Reinsurers will often provide aggregate loss cost data as part of capacity negotiations, which can be used to sanity-check or constrain the ABC prior — if the reinsurer's data implies a frequency of 0.08 and your ABC posterior is centred at 0.12, one of those is wrong and you need to understand which.

---

## What not to do

**Do not copy competitor rates as your rates.** The ABCmethod inverts competitor quotes to recover risk parameters. Copying quotes directly gives you market rates, not expected costs. If the market is systematically mispricing a segment — as PCW rank dynamics encourage, since the cheapest quote wins the click regardless of whether it covers cost — you inherit those mispricing decisions. The isotonic link is the guard against this: it asks which (λ, μ) is consistent with the market's pricing, not what the market charged.

**Do not apply the Stage 1 ABC posterior beyond 24 months without re-estimation.** Market pricing evolves. If competitor rates have moved materially since your ABC calibration, your prior is stale. Rebuilding the ABC estimate from a fresh quote collection takes a few days; the payoff is a prior that reflects the current market rather than the market you launched into.

**Do not skip the loss ratio corridor calibration.** The corridor is an informative prior that constrains the ABC to accept only economically plausible parameter values. Set it from industry statistics, not from the paper's default. The paper's [40%, 70%] is calibrated to French pet insurers in May 2024. UK motor in 2026 is not French pet in 2024.

---

## Practical entry point

For a UK MGA launching on a PCW in the next six months, the minimum viable version of this architecture is:

1. Collect 200 PCW quotes manually across a synthetic risk grid (two-three weeks, 15-20 hours).
2. Run the ABC loop from our [Python implementation post](/2026/03/31/market-based-ratemaking-python-implementation/) with a UK-calibrated loss ratio corridor.
3. Load the MAP estimates into [`insurance-credibility`](/insurance-credibility/) as your Bühlmann-Straub prior.
4. Document the full chain for the FCA: quotes collected, methodology, posterior, rates.

Stages 3 and 4 are Year 2 problems. Stage 1 and 2 are solvable in the six weeks before launch. The documentation alone is worth more than the rate itself under Consumer Duty — the FCA cannot audit a rate without a methodology, but it can audit a methodology without a perfect rate.

---

## Related posts

- [From Competitor Quotes to Risk Parameters: Implementing Market-Based Ratemaking in Python](/2026/03/31/market-based-ratemaking-python-implementation/) — the ABC-SMC code and UK pet insurance worked example
- [Market-Based Ratemaking Without Claims History](/2026/03/25/market-based-ratemaking-no-claims-history/) — the Goffard/Piette/Peters method explained
- [Your Policyholders Are Playing a Game with Your NCD Ladder](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/) — why the ABC prior needs underreporting correction if you are using NCD-stratified market quotes
