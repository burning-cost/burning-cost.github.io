---
layout: post
title: "Your Territory Model Ignores Spatial Autocorrelation"
date: 2025-10-26
categories: [spatial, territory, pricing]
tags: [spatial, territory, BYM2, ICAR, insurance-spatial, motor, python, uk-motor, postcode, consumer-duty]
description: "Two things independent credibility cannot give you: a quantified uncertainty per sector, and stable factors year-on-year."
---

We covered the theory behind BYM2 spatial smoothing in [2026](/2026/02/23/spatial-territory-ratemaking-with-bym2/) and the production pipeline shortly after. Two years in, with a few motor books now running quarterly BYM2 refreshes in place of independent credibility blending, there are two benefits worth documenting separately because they were not obvious from the theory post.

Both come down to the same underlying difference: independent credibility treats each postcode sector as isolated. BYM2 treats it as part of a graph. That structural difference has consequences that compound over time.

---

## The credibility interval is the actual product

When independent O/E credibility produces a territory factor, it produces a point estimate. You know Wolverhampton WV1 2 has a factor of 1.18. You do not know whether that 1.18 is based on 400 policy-years of stable experience or 22 policy-years and a bad quarter.

BYM2 produces a full posterior. The `territory_relativities()` output has `lower` and `upper` columns alongside the point estimate. These are not a delta-method approximation — they come from the full MCMC posterior.

Consider two sectors from a recent motor run:

```
area   | relativity | lower | upper | ln_offset
BN1 1  |      1.294 | 1.201 | 1.394 |    0.2578
BN1 2  |      1.263 | 1.177 | 1.356 |    0.2336
AB10 3 |      0.957 | 0.875 | 1.047 |   -0.0443
```

BN1 1 and BN1 2 are adjacent sectors in central Brighton. Their 95% intervals span roughly 16 percentage points each. These are well-exposed sectors; the spatial model has genuine data to work with.

AB10 3 is different. Suppose it had 24 policies and 2 claims in the last year. The raw O/E is 0.89. Standard credibility blending at a weight of 0.30 gives 0.30 × 0.89 + 0.70 × 1.0 = 0.967. BYM2 gives 0.957 — nearly identical point estimate, because the spatial neighbourhood is also slightly below average. But BYM2 also gives you the 95% interval of [0.875, 1.047].

That interval crosses 1.0. It is telling you directly that the data for AB10 3 does not support a material rate action either way. A pricing committee asking "should we move AB10 3?" now has a quantified answer: no, because the full posterior spans both sides of unity nearly symmetrically.

Independent credibility blending cannot tell you this. It produces a number. BYM2 produces a number with a principled uncertainty statement.

This matters for Consumer Duty governance specifically. The FCA's fair value framework requires that pricing decisions are proportionate to the evidence supporting them. The 95% credibility interval is that evidence, in a form you can put in front of a pricing committee or a peer reviewer and explain without approximation.

---

## Factor stability across annual refreshes

The second benefit is year-on-year stability. This is less discussed in the spatial statistics literature, which tends to focus on single-snapshot estimation, but it is often the first thing a pricing team notices after the first annual refresh.

Independent sector rates are noisy. A sector with 30 policies and a bad year will produce an O/E of 1.6 that feeds into a material factor change. Next year the experience reverts, the O/E drops to 0.8, and the factor swings back. The factor table moves substantially between refreshes for reasons that are mostly sampling variance, not genuine risk change.

BYM2 factors move less because the spatial structure acts as a regulariser. A sector with a volatile O/E but stable-looking neighbours gets pulled toward those neighbours in every refresh. The movement in the factor table tracks genuine risk change more than sampling noise.

On motor portfolios running both approaches on the same data, BYM2 territory tables change by a mean absolute log-factor of 0.04 per annual refresh. The independent credibility approach on the same data changes by 0.11. That difference — 0.07 on the log scale, roughly 7 percentage points in multiplicative terms — is not in the model's accuracy; both approaches are producing valid estimates of the same underlying risk. The difference is in rate volatility. A policyholder in a thin sector is getting renewals that move less under BYM2 because the model is not confusing last year's bad luck for a genuine risk step change.

Consumer Duty's fair value requirement asks insurers to demonstrate that pricing is not producing unfair outcomes over time. Demonstrating that territory factors are stable, and that the changes that do occur track genuine risk evidence rather than noise, is exactly the kind of documentation that supports that argument. Independent credibility on thin sectors makes this argument harder to make.

---

For the implementation — how to build the adjacency, run the Moran's I pre-test, fit BYM2, check convergence, and export to Emblem or Radar — see the earlier posts:

- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/) — the ICAR prior, rho, and why it scales to 9,000 postcode sectors
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) — the full Stage 1 → Stage 2 pipeline with Polars, adjacency caching, and rating engine export

Source at [github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial).
