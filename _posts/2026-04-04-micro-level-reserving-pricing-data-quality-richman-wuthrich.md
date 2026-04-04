---
layout: post
title: "Your Historical Loss Ratios Have a Reserving Problem"
date: 2026-04-04
categories: [reserving, pricing, actuarial]
tags: [micro-level-reserving, individual-claims, chain-ladder, loss-ratios, pricing-data-quality, rbns, ibnr, ptu-factors, richman, wüthrich, arXiv-2603.11660, arXiv-2602.15385, motor-bi, bodily-injury, liability, python, chainladder-python, insurance-reserving-neural, uk-personal-lines, solvency-ii, ss8-24]
description: "Richman and Wüthrich's March 2026 paper (arXiv:2603.11660) proves that aggregate chain-ladder produces materially biased ultimate estimates on liability lines. For UK personal lines pricing teams, this is not a reserving problem — it is a data quality problem that sits upstream of your loss ratio calibration."
author: Burning Cost
math: true
---

Pricing actuaries spend considerable effort on feature engineering, GLM structure, and exposure definitions. Most of them spend almost no time questioning the historical loss ratios that anchor the whole exercise. Those ratios come from the reserving team's ultimate estimates, which — for motor bodily injury, employers' liability, and public liability books — come from chain-ladder applied to aggregate development triangles.

Richman and Wüthrich published a paper in March 2026 (arXiv:2603.11660) that quantifies what it costs to do this. On a real liability dataset of 21,991 reported claims observed over a five-year development triangle, aggregate chain-ladder carries a bias of −4,204 — more than twice its own root mean squared error of prediction (RMSEP) of 1,977. That is not a small discrepancy. A reserve that is systematically short by more than 200% of its uncertainty measure is producing historical combined ratios that are materially wrong.

If you are pricing a UK motor BI or EL book, the chain-ladder ultimates you are regressing against may be systematically understating the true cost of claims settled in years two through five of development. The pricing model is not the problem. The training data is.

---

## What chain-ladder discards

The aggregate development triangle is a compression. Every claim in an accident year is summed into a single cumulative payment figure for each development period. From those sums, chain-ladder estimates development factors by fitting:

$$\hat{f}_j = \frac{\sum_i C_{i,j+1}}{\sum_i C_{i,j}}$$

This is well understood. What is less appreciated is what the aggregation throws away before the factor estimation even starts:

**Open versus closed status.** A cell with £5m cumulative paid from fifty large open claims has a very different expected development from the same £5m from fifty settled claims. The triangle treats them identically.

**Case reserve estimates.** For long-tail lines, the handler's current estimate of ultimate cost — updated quarterly, reflecting the specific facts of each claim — is the strongest available signal for outstanding liability. Richman and Wüthrich show that including claims incurred (paid plus case reserve) reduces individual RMSE by 28% over using cumulative paid alone on the liability dataset (RMSE of 3.089 versus 4.265). The aggregate triangle ignores this signal entirely.

**Claim type heterogeneity.** A motor BI triangle blends soft-tissue whiplash with catastrophic injury claims. A public liability triangle blends slips-and-falls with mesothelioma. Development patterns for these claim types differ by an order of magnitude. Development factors fitted to the blended triangle are wrong for both components.

The chain-ladder estimator on a heterogeneous long-tail book is not a neutral baseline. It is a biased one.

---

## What the Richman-Wüthrich paper actually shows

The paper (arXiv:2603.11660) is the second in a pair. The first (arXiv:2602.15385, February 2026) reformulates chain-ladder mathematically: rather than chaining development factors one period at a time, derive "projection-to-ultimate" (PtU) factors that jump directly from the current observation to the ultimate in a single step. This turns out to be equivalent to what chain-ladder does at aggregate level — but the PtU framing makes it obvious how to apply the same logic at claim level.

For claim $i$ observed at development lag $j$, fit:

$$\hat{C}_{i,J} = f\!\left(C_{i,j},\, X_{i,j}\right)$$

where $X_{i,j}$ is a vector of claim-level features and $J$ is the ultimate development period. Fit $f$ by weighted ordinary least squares on the historical data where ultimates are known.

The March 2026 paper tests this on two real datasets. The results on each differ in kind, not just magnitude.

On the accident dataset (66,639 claims), Mack chain-ladder achieves an RMSEP of 1,663. The individual linear regression reduces this to 937 — a 44% reduction. Bias is modest on both sides: chain-ladder runs −774 against the ground truth. This is a quality-of-fit improvement, not a bias correction.

On the liability dataset (21,991 claims), the picture is different and more serious. Chain-ladder carries a bias of −4,204 — the reserve is systematically short — against an RMSEP of 1,977. The bias exceeds twice the uncertainty measure. The individual linear regression reduces that bias to approximately −1,209. The method does not eliminate the under-reserving entirely on this small dataset, but it cuts it by roughly 70%. This is a bias correction, and it matters.

Then come the neural networks. The paper tests feedforward networks and transformers on the same features. They produce "stable results." They add essentially nothing over linear regression on datasets of this scale. On liability data with 21,991 claims, the adjuster's case reserve is doing the work that a neural network would need to learn from scratch. Given that the case reserve is already there, in the data, the neural network is redundant.

---

## Why this is a pricing problem, not a reserving problem

Reserving actuaries have their own reasons to care about this. But pricing actuaries have a more direct stake.

Consider how a UK personal lines motor pricing team calibrates their long-tail loss ratio for bodily injury. The typical approach is: take four or five years of accident-year ultimate loss ratios from the statutory reserving basis, regress them on relevant rating factors, and set an a priori expected loss ratio for projection. The current accident year is Bornhuetter-Ferguson weighted between the prior and the emerging experience.

Every step in this chain inherits the bias of the underlying reserve estimates. If chain-ladder is systematically under-reserving the liability book by amounts larger than its own uncertainty, then:

1. Your historical accident-year ultimates are understated for the long-tail years.
2. Your a priori loss ratio is calibrated against understated historical combined ratios.
3. Your BF weighting pulls the current year toward a prior that is too optimistic.
4. Your pricing margin for long-tail uncertainty is calibrated against a historical series that masked that uncertainty.

This is not a model error. It is a systematic bias in the training data, propagated through the pricing workflow. The pricing actuary did everything correctly and still produced a rate that is potentially insufficient.

The magnitude is not trivial. Richman and Wüthrich's liability result shows a bias of −4,204 on an aggregate reserve where the RMSEP is 1,977. That is directionally consistent with empirical findings from the PRA's 2023 thematic review on claims reserving, which found that many UK insurers were failing to capture heterogeneous claims inflation — bodily injury and property damage inflating at different rates within the same triangle.

---

## What the Python ecosystem offers

The honest answer: not enough for micro-level reserving end-to-end.

**What exists:**

`chainladder-python` (casact, maintained by the CAS open source community) is the Python standard for aggregate triangle methods. It handles chain-ladder, Bornhuetter-Ferguson, ODP bootstrap, and Clark growth curves competently. It can accept claim-level data as triangle inputs, but it then aggregates that data before fitting — it does not run claim-level models. The claim-level IBNR it produces is chain-ladder applied at individual claim level, not micro-level reserving in the Richman-Wüthrich sense.

`insurance-reserving-neural` (our package, in active development) covers RBNS neural reserving with feedforward and LSTM networks, PtU-structured linear regression as described in arXiv:2603.11660, bootstrap uncertainty quantification from P10 through P99.5, and a chain-ladder comparison baseline. It is the only installable Python package for individual RBNS micro-level reserving.

**What does not exist in Python:**

Individual IBNR in Python. The RBNS model above handles reported claims only. For claims that have occurred but not yet been reported at valuation, you need a separate model of reporting delays. The best available tool for this is ReSurv — a survival analysis approach using Cox proportional hazards and XGBoost, published on CRAN in July 2025. ReSurv is R-only. No Python equivalent exists.

The joint RBNS/IBNR architecture from Delong, Lindholm and Wüthrich (*Scandinavian Actuarial Journal*, 2022) — which handles both components simultaneously using six networks — has not been replicated in any Python library. Implementing it from scratch from the paper is not a weekend project.

The Richman-Wüthrich paper itself provides two R code listings (pseudocode format). No Python implementation accompanies the paper.

**The practical gap:** a UK pricing team wanting to test micro-level RBNS reserving against their chain-ladder ultimates can do so in Python today. A team wanting a full individual-level replacement for chain-ladder — RBNS and IBNR — cannot do so in Python without significant bespoke development.

---

## What a UK personal lines pricing team should actually do

The literature has moved. The chain-ladder critique is now empirically grounded on real data, not just theoretical. The question is what to do with that, given real constraints.

**If your book is short-tail, homogeneous property or motor own damage:** the bias problem is small. Chain-ladder reserves on homogeneous portfolios are close to unbiased. The data engineering investment to move to individual methods will not repay itself here. Focus elsewhere.

**If your book includes significant motor BI, EL, or PL:** the bias problem is real and quantifiable. The first step is not to implement micro-level reserving. It is to audit whether you are using reserve ultimates or paid-loss development as your loss ratio basis — and if you are using ultimates, to understand the vintage and methodology of those ultimates. Ask your reserving colleagues: has chain-ladder been checked for systematic bias against a selected-loss or individual-level benchmark on this book?

**If you want to run the experiment:** Richman and Wüthrich's linear regression is not complex. You need a claim-period panel — claim ID, valuation date, cumulative paid, case reserve (claims incurred), open/closed flag, claim type. For three accident years and five valuation dates on a book of 10,000 claims, that is 50,000 rows, which any half-decent claims system can produce. Fit weighted OLS of ultimate (observed on fully-developed claims) on those features. Compare aggregate total to chain-ladder. The gap, if material, is the magnitude of the data quality problem sitting upstream of your pricing.

**On neural networks:** on a UK personal lines motor book of up to 100,000 claims, the paper's evidence says linear regression will match or beat a feedforward network. Do not build a transformer before you have established that the linear residuals have exploitable structure. They probably don't.

**On Python tools:** `insurance-reserving-neural` covers the RBNS linear regression and neural network components described in arXiv:2603.11660. For the IBNR component, use aggregate chain-ladder or BF as a plug-in until a Python individual IBNR implementation exists.

---

## Hype versus reality

The hype: micro-level reserving with deep learning will transform actuarial practice and produce dramatically better reserves.

The reality, per the March 2026 evidence: individual methods do produce materially better reserves on long-tail lines — but the gain comes from linear regression, not deep learning. The mechanism is using case reserves and claim status as features, which the triangle ignores. The data engineering to assemble a clean claim-period panel is the main bottleneck, not the model sophistication. Neural networks add value on very large, heterogeneous books; on anything under 50,000 claim-period observations, they add noise.

For pricing actuaries specifically: the primary value of the Richman-Wüthrich result is not a new reserving method to implement. It is evidence that aggregate chain-ladder is systematically biased on long-tail liability data in a direction that makes historical loss ratios look better than they were. If you are pricing UK EL, PL, or motor BI, you should be asking your reserving team whether they have tested for this bias, and if not, why not.

The Python ecosystem does not yet have a complete individual-level reserving solution. R is ahead, particularly on the IBNR component via ReSurv. The gap will close, but it has not closed yet.

---

## Papers and tools referenced

- Richman and Wüthrich, "One-Shot Individual Claims Reserving" (March 2026), [arXiv:2603.11660](https://arxiv.org/abs/2603.11660)
- Richman and Wüthrich, "From Chain-Ladder to Individual Claims Reserving" (February 2026), [arXiv:2602.15385](https://arxiv.org/abs/2602.15385v2)
- Avanzi, Lambrianidis, Taylor and Wong, "On the use of case estimates and transactional payment data" (December 2025), arXiv:2601.05274
- Delong, Lindholm and Wüthrich, *Scandinavian Actuarial Journal* 2022/1 — joint RBNS/IBNR six-network architecture
- Hiabu, Hofman and Pittarello, "ReSurv" (CAS E-Forum 2025) — individual IBNR via survival analysis, R only
- `chainladder-python` — [github.com/casact/chainladder-python](https://github.com/casact/chainladder-python)
- `insurance-reserving-neural` — individual RBNS reserving in Python
