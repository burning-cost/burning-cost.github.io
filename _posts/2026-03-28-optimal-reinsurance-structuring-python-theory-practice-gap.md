---
layout: post
title: "Optimal Reinsurance Structuring: What Theory Delivers and What Pricing Teams Actually Do"
date: 2026-03-28
categories: [research]
tags: [reinsurance, XL, optimisation, solvency-ii, SCR, capital-modelling, python]
description: "Theory proves standard XL layers are optimal under VaR. Yet most pricing teams set retentions by trial and error. We review the academic results, the commercial tools, and the Python gap — then sketch a practical optimisation approach."
seo_title: "Optimal Reinsurance Structuring Python: Theory vs Practice for UK Insurance Pricing"
---

Most UK personal lines pricing teams treat reinsurance structure as a given. The capacity provider or the CRO hands down a treaty — £1m xs £250k per occurrence, say — and pricing absorbs it as a fixed cost. Net loss cost is burning cost applied to the net layer, premium loaded appropriately, job done.

This is understandable. Pricing teams have enough to do. But it leaves a real question unanswered: is that structure any good? Is £250k the right retention? Should the limit be £500k or unlimited? And does it actually reduce capital, or just cost money?

The academic literature has thought about this carefully for twenty years. The answers are cleaner than most practitioners realise — and the gap between what theory can tell you and what commercial tools actually deliver is wide enough to be worth mapping.

---

## Why reinsurance structure affects pricing

The connection is direct. A per-occurrence XL treaty changes three things simultaneously:

**Net loss cost.** Your burning cost is now the aggregate of retained losses — those below the retention plus those above the limit, if any. The ceded layer reduces expected net losses but you pay a reinsurance premium that typically exceeds the expected recovery (reinsurers have costs and profit margins too). The net effect on combined ratio depends on the loading.

**SCR.** Under Solvency II's standard formula, the non-life premium risk charge is calculated on net losses. A lower net loss distribution means a lower SCR, which means lower required capital, which means a lower required margin built into your technical price. The relationship is not one-to-one, and the standard formula handles it crudely — more on this below — but the direction is unambiguous.

**Required return on capital.** If your reinsurance programme genuinely reduces SCR, you need less capital to write the same book, and the required margin per unit of premium falls. If the reinsurance premium costs more than the margin saving from reduced capital, the structure is value-destructive even if it feels prudent.

Optimising the structure is therefore not an academic exercise. It has direct consequences for what the risk is worth to write.

---

## What theory says

The foundational result is Cai and Tan (2007), published in ASTIN Bulletin 37(1). They set up a clean single-period problem: an insurer with aggregate loss X, buying stop-loss reinsurance I(x) = (x − d)₊ under an expected-value premium principle with reinsurer loading θ. What retention d minimises the insurer's VaR?

The answer is analytic:

```
d* = F⁻¹(α / (1 + θ))
```

where F is the loss CDF and α is the confidence level (99.5% for Solvency II SCR). This is simply a quantile of the loss distribution, scaled by the effective loading factor. For a lognormal loss distribution with mean £2m and cv 0.4, and a reinsurer loading of 25%, the optimal retention at the 99.5% level sits at roughly the 79.6th percentile of gross losses. You can compute it from your loss model in seconds.

Cai and Tan also showed that the CVaR (CTE) criterion gives the same retention. The result is robust across both risk measures.

Chi and Tan (2011) extended this in the direction that matters most for practice. If you relax the convexity constraint on the ceded loss function — allowing structures where the ceded amount does not grow convexly with gross loss — the optimal form under VaR is no longer pure stop-loss but **truncated stop-loss**:

```
I(x) = min(x − d, u)
```

This is exactly the standard per-occurrence XL layer: the cedant retains the first d, the reinsurer pays up to u above that, and losses above d + u revert to the cedant. The standard XL treaty that every personal lines insurer buys is the theoretically optimal contract form under VaR minimisation. That result deserves to be more widely known among the pricing community.

Under CVaR, the picture is simpler still: pure stop-loss (no upper limit) is always optimal, regardless of the constraint class imposed on the ceded function. CVaR optimisation argues for unlimited top of tower; VaR optimisation gives you bounded layers with an upper limit.

The practical implication for UK insurers is this: Solvency II uses 99.5% VaR for the SCR. This means theory directly supports your standard XL structure — bounded layer, specific retention and limit. The question is whether the specific numbers are right, not whether the form is right.

---

## Bilateral optimality: why XL is what reinsurers also want

A limitation of Cai-Tan is that it is unilateral — it minimises insurer risk without asking whether the reinsurer is getting a fair deal. A 2022 result in Communications in Statistics extended this to bilateral Pareto-optimality.

Under law-invariant risk measures (which includes VaR and CVaR) combined with premium principles satisfying risk loading and convex order preservation, **layer (XL) reinsurance is always Pareto-optimal for both parties**. Neither insurer nor reinsurer can be made better off without making the other worse off, when the contract takes a layer form.

This is the theoretical reason XL per-occurrence treaties dominate personal lines reinsurance globally. It is not convention or historical accident. It is the bilateral optimum under standard risk measure conditions. A 2023 extension incorporates reinsurer default risk and shows layer contracts remain optimal, though with modified parameters — relevant post-Greensill and with ongoing counterparty quality concerns.

---

## How retention and limit are actually chosen

Theory gives you the optimal form and — in the single-stop-loss case — an analytic formula for the retention. In practice, most UK pricing teams and actuaries do something quite different.

**ReMetrica and Tyche** (Aon) are the industry-standard stochastic capital modelling platforms used by Lloyd's syndicates and larger UK carriers. They can simulate a gross loss distribution, apply treaty mechanics, and compute net metrics across thousands of programme combinations. They are powerful tools but they are not optimisers in the mathematical sense: they do not search the retention/limit space analytically, they run a pre-specified grid and report results. The actuary still decides which combinations to test, and interprets the Pareto frontier manually.

**Spreadsheet DFA models** are what most smaller insurers and MGAs actually use. Simulate aggregate losses, apply the treaty, compute solvency ratio and combined ratio, plot the frontier across a grid of retention and limit pairs. This is entirely reasonable engineering — grid search is a legitimate optimisation method when the parameter space is low-dimensional — but it is not informed by the theoretical structure. The optimal retention formula from Cai-Tan is never applied. The grid is chosen by feel.

The gaps between theory and practice are specific and worth naming.

**Gap 1: Market price vs optimal loading.** Cai-Tan assumes the reinsurer's loading θ is known and fixed. In practice, the loading is whatever the market charges. You take it or leave it. The theory tells you the optimal retention given θ; you can still use it if you calibrate θ from the quoted reinsurance premium, but this requires backing out the implied loading from the quote — something pricing teams rarely do explicitly.

**Gap 2: Layered towers with multiple reinsurers.** Theory handles a single reinsurer or — in the bilateral multi-reinsurer case — assigns layers across reinsurers with heterogeneous views. Practice has reinstatement premiums, aggregate deductibles, franchise clauses, and four reinsurers signing different lines at different net retentions. No clean theoretical treatment exists for the full commercial complexity.

**Gap 3: Aggregate stop-loss is theoretically better.** Under CVaR minimisation, pure stop-loss (which operates on aggregate losses) is optimal, not per-occurrence XL. Yet aggregate working covers are almost never placed in UK personal lines — reinsurers are unwilling to take frequency risk alongside severity risk without prohibitive pricing. The theoretically superior structure is operationally inaccessible.

**Gap 4: Loss distribution uncertainty.** Theory assumes the loss distribution is known. A small UK MGA with five years of data has a highly uncertain loss model. The optimal retention under one distributional assumption may be wrong by a factor of two under an alternative assumption. This model risk around the optimal decision is not addressed in the academic literature.

**Gap 5: Regulatory capital under standard formula.** This is the most practically important gap, and deserves its own section.

---

## How XL reinsurance affects your SCR — and the standard formula's limitation

Under Solvency II (and Solvency UK, unchanged on this point after the PRA reforms completed 31 December 2024), the non-life premium risk SCR uses a lognormal approximation with a sigma parameter for each segment. For segments with non-proportional (XL) reinsurance, the gross sigma is multiplied by an adjustment factor:

- Segments 1, 4, 5 (Motor Vehicle Liability, Fire and other property damage, General Liability): **factor = 80%**
- All other segments: **factor = 100%** (no adjustment)

Per EIOPA Q&A 2322, this 80% factor applies regardless of whether non-proportional reinsurance actually exists, and regardless of its specific terms. It is a blanket calibration. A £500k xs £100k programme gets the same 80% factor as a £1m xs £500k programme. The standard formula cannot compute the marginal SCR saving from changing retention from £500k to £250k.

This is a fundamental limitation. If you are on standard formula — which most UK MGAs and smaller carriers are — you cannot use the regulatory capital figure to optimise your programme. You would need either an internal model or a parallel shadow simulation that computes net loss distributions at different retention levels and derives the 99.5% VaR directly.

There is also a counterparty default risk charge to account for. XL reinsurance creates Type 1 counterparty exposure. The net SCR saving must be traded against the additional counterparty risk charge. For A-rated reinsurers, this charge is typically small in isolation but non-negligible in aggregate across a multi-layer programme.

Internal model firms — Lloyd's managing agents, larger carriers — can and do compute the true marginal capital benefit of each layer. The rest of the market is flying partially blind on this.

---

## The Python ecosystem gap

The Python actuarial ecosystem has filled most of the tooling gaps in pricing and reserving over the last five years. **chainladder-python** handles reserving triangles. **sktime**, **statsmodels**, and a dozen ML libraries cover frequency-severity modelling. For reinsurance, the picture is different.

There are several packages that apply treaty mechanics to simulated losses:

- **rippy** (pythactuary/rippy): simulation-based, supports per-occurrence XoL, uses numpy/scipy. No optimisation capability.
- **cbalona/reinsurance**: proportional and non-proportional layers, Dask-parallelised, composable model design. No optimisation capability.
- **GEMAct** (gpitt71/gemact-code, published in Annals of Actuarial Science 2023): the most complete actuarial package — collective risk models via FFT, aggregate loss distributions, coverage modifiers for stop-loss, XL, and reinstatements. This can compute the input distribution needed for optimisation. No optimisation built in.

There is no Python package that combines loss distribution computation, treaty mechanics, and optimisation over the retention/limit space. No package implements the Cai-Tan analytic formula. No package plots the Pareto frontier of solvency ratio versus profitability across programme combinations.

The equivalent in R is not much better. **ReIns** (TReynkens, companion to Albrecher, Beirlant and Teugels' 2017 textbook) provides EVT estimators and VaR/CTE premium calculations but does not optimise structures.

The commercial tools — ReMetrica, Tyche, Optalitix — fill this gap but are not open-source and not cheaply accessible to MGAs, smaller carriers, or in-house teams building their own analysis.

---

## A practical approach

The academic framework plus the European Actuarial Journal 2021 simulation paper (Springer, doi:10.1007/s13385-021-00281-2) together give a clear blueprint for what a practical Python implementation would look like.

**Step 1: Fit a loss model.** For UK motor or home, a collective risk model typically works: claim frequency as Poisson or negative binomial, severity as lognormal or Pareto depending on your tail behaviour. Use your own data; for development or benchmarking, public datasets work as a starting point.

**Step 2: Simulate gross losses.** Generate N (say, 100,000) annual aggregate loss scenarios. For per-occurrence XL, you need to preserve the individual loss granularity — do not aggregate before applying the treaty.

**Step 3: Define the search grid.** For a single-line per-occurrence XL, you have two parameters: retention d and limit u. A coarse grid might be 20 retention values (£50k to £2m) and 10 limit values (£250k to unlimited). That is 200 combinations — trivially fast.

**Step 4: For each programme, compute:**
- Net aggregate losses: gross losses minus treaty recoveries
- Net combined ratio (net losses + reinsurance premium + expenses, over net premium)
- SCR proxy: 99.5th percentile of net aggregate losses, or annualised net VaR
- Return on required capital

**Step 5: Calibrate the reinsurance premium.** Either use a quoted premium where available, or price it from the simulated ceded loss distribution using an expected value premium principle with an assumed loading. Back out the loading from market quotes if you have them.

**Step 6: Plot the Pareto frontier.** Solvency ratio (SCR proxy / capital held) on one axis, net combined ratio on the other. Each point is a programme combination. The efficient frontier is the programmes where you cannot improve the solvency ratio without worsening the combined ratio.

**Step 7: Apply the Cai-Tan analytic check.** Compute the theoretical optimal retention from d* = F⁻¹(α / (1 + θ)) where F is the gross aggregate loss distribution and θ is the reinsurer loading. This serves as a sense-check on where the grid search Pareto frontier peaks. If the simulation says the optimal retention is £300k and the analytic formula says £275k, you have confidence. If they diverge by a factor of three, something is wrong in your model.

The computation is genuinely affordable. Simulating 100,000 scenarios and applying 200 programme combinations takes a few seconds in numpy. The limiting factor is not computation — it is the willingness to build the pipeline at all.

GEMAct provides the aggregate loss distribution machinery. A small loop applying per-occurrence XL mechanics (net per scenario = sum(min(loss_i, d) + max(loss_i - d - u, 0)) over individual losses) adds the treaty layer. scipy.optimize or a simple numpy grid search handles the optimisation. matplotlib produces the Pareto plot.

This is not research — it is engineering. The pieces exist. No one has assembled them into an open-source package.

---

## What this means for a UK pricing team

If you are a motor or home pricing actuary and your firm buys a £1m xs £250k per-occurrence XL:

1. You probably do not know if £250k is the right retention. The structure was likely set when the programme was first placed and has been renewed with incremental adjustment.

2. The standard formula is not telling you the marginal capital benefit of changing the retention. You need a parallel simulation.

3. Theory says your layer form is right — bounded XL is optimal under VaR for the reasons Chi and Tan (2011) set out. But the specific retention should be derivable from your loss model and the reinsurance loading.

4. A grid search in Python over retention and limit values, producing a Pareto frontier of capital efficiency against net profitability, is a half-day of engineering work once the loss simulation is in place. It is not a six-month internal model project.

5. The optimal transport literature (arXiv:2312.06811, 2023) is not yet practical — it mainly provides alternative proofs of classical results. The simulated annealing approach (arXiv:2504.16530, 2025) is designed for catastrophe multi-peril problems, not personal lines XL. The relevant theory is Cai-Tan and Chi-Tan: twenty years old, still unused in most pricing teams.

We are building the Python implementation. When it is ready, we will publish it here.

---

*References: Cai J, Tan KS (2007), ASTIN Bulletin 37(1), 93–112. Chi Y, Tan KS (2011), ASTIN Bulletin. Albrecher H, Beirlant J, Teugels J (2017), Reinsurance: Actuarial and Statistical Aspects, Wiley. European Actuarial Journal 2021, doi:10.1007/s13385-021-00281-2. arXiv:2312.06811 (2023). EIOPA Q&A 2322 on standard formula non-proportional reinsurance adjustment. PRA SoP11/24 (November 2024).*
