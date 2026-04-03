---
layout: post
title: "The Adjuster Knows More Than Your Triangle"
date: 2026-04-03
categories: [reserving, individual-claims]
tags: [individual-claims-reserving, chain-ladder, linear-regression, neural-networks, rbns, ibnr, ptu-factors, mack, solvency-ii, pra-ss8-24, long-tail, bodily-injury, liability, wüthrich, richman, avanzi, splice, insurance-reserving-neural, arXiv-2603.11660, arXiv-2602.15385, arXiv-2601.05274, data-engineering, python]
description: "A new empirical paper from Richman and Wüthrich shows that individual claims models cut Mack RMSEP by 44% on accident data and correct severe bias on liability — but the gain comes from linear regression, not neural networks. The real blocker is data engineering."
author: burning-cost
math: true
---

The standard story about ML and reserving goes roughly like this: chain-ladder is a 1970s heuristic, neural networks are modern, therefore neural networks should produce better reserves. The conclusion is wrong — not because individual claims models do not work, but because the improvement has almost nothing to do with the ML.

Richman and Wüthrich published a paper in March 2026 (arXiv:2603.11660) that tests individual claims reserving on two real datasets: 66,639 accident insurance claims and 21,991 liability insurance claims, each over a 5×5 development triangle. The results are unambiguous. Individual methods substantially beat Mack chain-ladder on both datasets. On accident data, the RMSEP falls from 1,663 (Mack) to around 937 using an individual bootstrap — a 44% reduction. On liability, Mack chain-ladder carries a systematic bias of −4,204, more than twice its own RMSEP of 1,977. The individual linear regression model reduces that bias to −1,209.

Then come the neural networks. They produce "stable results." They add essentially nothing over the linear regression. The authors write: "even a linear regression model does an excellent job!" On datasets of this scale — 20,000 to 70,000 claims — the answer to whether neural networks add value is negative.

That is the counterintuitive result. The improvement is real and large. The method responsible is linear regression.

---

## Why linear regression beats chain-ladder

The triangle discards three things that matter.

First, claim type. A development triangle entry for accident year 2022, development year 3 blends bodily injury, property damage, and public liability claims. Their development patterns differ by an order of magnitude. The triangle assumes they develop identically.

Second, open/closed status. The triangle cannot distinguish a portfolio of ten large open claims from ten settled claims in the same cell. Whether a claim is still open is the strongest available signal for outstanding liability.

Third — and most important — the adjuster's case reserve. The current incurred estimate is the best single predictor of what a long-tail claim will eventually cost. Triangles operate on cumulative paid cash flows and discard case reserves entirely.

The empirical evidence on this last point is striking. Richman and Wüthrich's liability dataset shows that claims incurred alone reduces individual RMSE by 28% over cumulative paid alone (3.089 versus 4.265). The adjuster's estimate is not a noisy approximation of truth. On long-tail lines, it is better information than the historical payment pattern — better than whatever the development triangle is trying to capture.

Once you build a model that takes claim type, open/closed status, and case reserve as features, linear regression is already doing the heavy lifting. The non-linear relationships that would justify a neural network are not large enough, at 20k–70k claim scale, to show up in the metrics.

---

## When individual methods actually help

The evidence does not support blanket adoption. The honest assessment by line type:

**Long-tail, heterogeneous lines (casualty, bodily injury, public liability).** This is where individual methods are compelling. Chain-ladder produces materially biased reserves on liability data. The bias correction from conditioning on case reserve and claim status is substantial — not a rounding error. For a UK GI insurer with a significant EL, PL, or motor bodily injury book, the case for moving to individual reserving is strong.

**Short-tail, homogeneous lines (property, motor hull).** Individual methods provide marginal gain at best. Chain-ladder is near-unbiased on homogeneous portfolios. The data engineering investment is unlikely to pay off here.

**Very small portfolios (sub-1,000 claims).** Individual models do not have enough data to train. Lloyd's specialty lines, captives, and smaller commercial books remain dominated by aggregate methods. Simulation-based or Bayesian credibility approaches are more appropriate at that scale.

**Adjuster accuracy matters.** Case reserves are gold when adjusters are reliable. If reserve adequacy at the adjuster level is poor — which varies considerably by insurer and by line — the individual model's primary advantage is diminished. There is no published framework for detecting and down-weighting unreliable case estimates before feeding them into these models.

---

## The IBNR problem nobody has solved in Python

There is a gap in this literature that matters for practical adoption. The individual models above are RBNS models: they estimate outstanding liability for **reported** claims only. IBNR — claims incurred but not yet reported at valuation — is handled separately. Chain-ladder handles IBNR implicitly by extrapolating the triangle. Individual RBNS models do not.

The options for IBNR at individual level are: use a traditional aggregate estimate on top of the individual RBNS model; or use a survival analysis model for reporting delays (ReSurv, published on CRAN in July 2025, handles this well via Cox proportional hazards and XGBoost). ReSurv is R-only. No Python equivalent exists.

Our `insurance-reserving-neural` package covers RBNS neural reserving with FNN and LSTM, bootstrap uncertainty quantification at P10 through P99.5, and a chain-ladder comparison baseline. It is the only installable Python package for individual RBNS reserving. We are planning an individual IBNR module, but as of April 2026 the IBNR gap in Python is real.

The Delong, Lindholm and Wüthrich (2022) six-network architecture handles RBNS and IBNR jointly. It has not been replicated in any Python library. If you need a single integrated individual model for both components, it means implementing it yourself from the paper or using R.

---

## The projection-to-ultimate framework

Richman and Wüthrich's companion paper (arXiv:2602.15385, February 2026) explains the mathematical scaffolding. Rather than stepping development period by period in the usual chain-ladder fashion — j to j+1 to j+2 — a projection-to-ultimate (PtU) factor jumps directly from the current development period to ultimate in one step:

$$C_{i,J} = C_{i,I-i} \times F_{I-i}^{CL}$$

where $F_j = \prod_{l=j}^{J-1} f_l^{CL}$ is the product of chain-ladder link ratios from period $j$ to ultimate. This is what Lorenz and Schmidt called the "grossing-up method." The insight is that this framework applies cleanly at the individual claim level: replace the aggregate cell total with individual claim features, run a regression of ultimate on current features weighted by expected ultimate, and the individual model falls out.

The result is a framework that bridges aggregate chain-ladder and individual regression models. It is also why linear regression is so effective here — the PtU structure means you are estimating a single direct relationship between current claim state and ultimate cost, rather than chaining a sequence of period-to-period development predictions.

---

## The UK regulatory position

PRA SS8/24, which took effect on 31 December 2024 and supersedes SS5/14, is explicit: relying solely on triangle extrapolation "is unlikely to satisfy the Directive requirement for a probability-weighted average of future cash-flows, since not all possible future cash-flows may be represented in the data." This is not a prohibition on chain-ladder, but it creates deliberate regulatory space for individual-data approaches.

The 2023 PRA thematic review on claims reserving found that many insurers' aggregate methods failed to capture heterogeneous claims inflation — bodily injury and property damage inflating at different rates in the same triangle. Individual reserving models separate these components naturally by conditioning on claim type. That is a direct response to the specific concern the PRA raised.

For Lloyd's syndicates, CDR v3.2 mandates individual claim-level transaction data via ACORD standards. Syndicates with full CDR implementation already have the data infrastructure these models require. CDR adoption is still rolling out across the market, but the direction is clear.

---

## The real blocker: data engineering, not ML

Ask a UK reserving team why they have not moved to individual claims models and the answer is rarely "we need better neural networks." It is almost always some version of: the data is not in the right shape.

Chain-ladder needs 30 to 60 numbers — aggregate cumulative paid totals by accident year and development year. A standard triangle. Individual reserving models need a consistent transaction history per claim: every valuation date, every payment, every case reserve revision, open/closed status changes. For a book with 50,000 claims across 10 accident years, that means assembling something of the order of 500,000 claim-period rows with consistent identifiers, accurate dates, and non-missing case reserves.

Most UK insurers store this data, but assembling it in the right form requires dealing with system migrations, policy administration handoffs, claims system changes, and data quality issues at the transaction level that do not surface in aggregate triangles. The aggregate triangle hides data quality problems. The individual model exposes them.

Neural networks need approximately 50,000 claim-period observations for stable training (based on Avanzi, Lambrianidis, Taylor and Wong, arXiv:2601.05274, December 2025). Linear regression is viable with around 2,000 claims. For a team starting out, the right first step is a weighted linear regression with case reserve and claim status as features — not a neural network — and the data pipeline needed for linear regression is a fraction of the complexity.

The Richman and Wüthrich result says this is fine. Linear regression with the right features is the answer for most portfolios at most scales. The neural network is for when you have the data, the scale, and the evidence from the linear regression that non-linear terms add something.

---

## What we would do for a UK long-tail book

For a UK EL or motor BI book where chain-ladder is suspected to be biased, we would run this in order:

1. Assemble a claim-period panel: claim ID, accident date, valuation date, cumulative paid, case reserve (claims incurred), open/closed flag, claim type. Start with the most recent three accident years and the last five valuation dates. Get this right before touching any model.

2. Run Richman and Wüthrich's weighted linear regression (Listing 1 in arXiv:2603.11660). This is not complex code — it is a weighted OLS on log-transformed features. Compare total reserve to chain-ladder. If the individual model is materially lower with better-conditioned residuals, you have found the bias.

3. Add a chain-ladder comparison baseline so the validation documentation satisfies IFoA TAS M and SS8/24 requirements. Our `insurance-reserving-neural` package produces this as a standard output.

4. Decide whether a neural network is warranted. With fewer than 50,000 claim-period observations, the honest answer is probably not. With a large book and evidence that the linear residuals have structure, try an FNN and test whether it improves on held-out accident years.

5. Handle IBNR separately, with an aggregate estimate until an individual IBNR module exists in Python.

The literature has moved further than most UK reserving teams realise. The chain-ladder critique is now empirically grounded, not theoretical. But the solution is a better linear regression, not a transformer.

---

## Papers referenced

- Richman and Wüthrich, "One-Shot Individual Claims Reserving" (March 2026), arXiv:2603.11660
- Richman and Wüthrich, "Projection-to-Ultimate Factors" (February 2026), arXiv:2602.15385
- Avanzi, Lambrianidis, Taylor and Wong (December 2025), arXiv:2601.05274
- Delong, Lindholm and Wüthrich, *Scandinavian Actuarial Journal* 2022/1
- Hiabu, Hofman and Pittarello, "ReSurv" (CAS E-Forum 2025)
