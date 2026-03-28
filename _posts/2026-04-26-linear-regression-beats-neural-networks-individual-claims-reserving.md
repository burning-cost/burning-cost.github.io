---
layout: post
title: "Linear Regression Beats Neural Networks for Individual Claims Reserving"
date: 2026-04-26
categories: [research]
tags: [reserving, IBNR, individual-claims, neural-networks, linear-regression, chain-ladder, python]
description: "Richman and Wüthrich's March 2026 paper shows linear regression with projection-to-ultimate factors closes 44% of the gap over chain ladder — and neural networks add nothing at typical portfolio sizes. The gain comes from features, not model complexity."
seo_title: "Linear Regression Beats Neural Networks for Individual Claims Reserving: What Richman-Wüthrich 2026 Means for Practice"
---

The individual claims reserving literature has spent roughly a decade building increasingly complex models. Gabrielli and Wüthrich (2020) gave us a six-module neural network architecture. Chaoubi et al. (2023) applied LSTMs. Avanzi et al. (2025) ran FNN against LSTM with full uncertainty quantification. Richman and Wüthrich themselves followed up their simulation machine with a transformer experiment. The implicit assumption running through all of this work was that more expressive models should eventually win.

Their March 2026 paper (arXiv:2603.11660) quietly kills that assumption. On a real accident insurance portfolio of 66,639 claims, linear regression with projection-to-ultimate factors achieves an RMSEP of 937 versus Mack chain-ladder's 1,663. That is a 44% reduction in prediction error. Neural networks and transformers, tested on the same data, do not improve on the linear regression result.

This is the most practically important finding in individual reserving since Wüthrich (2018) showed that individual claim covariates could improve chain-ladder at all. Let us work through what it means.

---

## The result in full

The setup is a 5×5 development triangle (5 accident years, 5 development periods) with individual claim data from an accident insurance portfolio. The benchmark is Mack chain-ladder, which operates on aggregate triangles and knows nothing about individual claims. RMSEP — root mean squared error of prediction — measures reserve accuracy for the RBNS (reported but not settled) component.

| Method | RMSEP |
|---|---|
| Mack chain-ladder | 1,663 |
| Individual linear regression (PtU) | 937 |
| Individual neural network | ~937 (no material improvement) |
| Individual transformer | ~937 (no material improvement) |

The liability insurance results are even starker. Mack chain-ladder produces a reserve *error* of −4,204 — a systematic underreserve — against an RMSEP of 1,977. The individual linear model reduces the error to −1,209 using claims incurred as the primary feature. The aggregate method is not just noisier; it is structurally biased in the wrong direction.

Neural networks and transformers were tested and found to add nothing over linear regression at N ≈ 66k claims. This is not a softly-hedged "marginal benefit" finding. The paper is clear.

---

## Why chain-ladder loses: information it cannot use

Chain-ladder is an aggregate method. It observes cumulative paid losses for a portfolio, aggregated by accident year and development period. Everything else it throws away.

At each valuation date, a UK insurer's claims system knows, for every open claim: the adjuster's current reserve (claims incurred), whether the claim is open or closed, the claimant's injury type and legal representation status, whether the claim has been litigated, how many payments have been made, and what the payments were for. None of this enters the chain-ladder calculation.

The individual linear regression model uses it all. The primary features in arXiv:2603.11660 are:

1. **Claims incurred** (the adjuster's case reserve)
2. **Claim status** (open/closed)
3. **Cumulative paid** (which is what chain-ladder uses too)
4. **Development period** (also in chain-ladder)

Features 1 and 2 are where the gains come from. Features 3 and 4 are what chain-ladder already has. The 44% RMSEP reduction is almost entirely attributable to using the adjuster's estimate and knowing whether a claim is still open.

On the liability data, claims incurred alone beats cumulative paid alone by 28% on RMSE. That single substitution — use the adjuster's number instead of cumulative paid — explains the bulk of the improvement.

---

## The feature hierarchy

This is the lesson that gets lost when the discussion centres on model architecture: **what you put in matters far more than how you model it**.

Avanzi et al. (arXiv:2601.05274, December 2025) make this explicit in a different experiment. They compare four models: FNN (feedforward neural network, no case estimates), FNN+ (with case estimates), LSTM (sequence model, no case estimates), LSTM+ (with case estimates). The key finding: FNN+ beats LSTM by a meaningful margin. Adding case estimates to a simple feedforward network outperforms a sophisticated sequence model that lacks them.

Together, the two papers establish a clear hierarchy for RBNS reserving:

1. **Case estimates (claims incurred)** — the adjuster's view of ultimate severity. Strong, interpretable, available today in every UK claims system. First feature to add.
2. **Claim status** — open, closed, reopened. Tells the model whether further development is possible. Near-zero implementation cost once you have claim-level data.
3. **Development pattern and exposure** — the PtU censoring mechanism that makes learning sets internally consistent. This is the technical contribution of arXiv:2603.11660; we covered the Python implementation in an [earlier post](/2026/03/26/ptu-reserving-python-implementation-gap/).
4. **Non-linear interactions** — where neural networks theoretically earn their place. At N ≈ 66k, they do not add anything detectable. At very large N or in lines with complex non-linear claim dynamics, this could change.
5. **Sequence memory** — what LSTM adds over FNN. Marginal over FNN+ at any scale tested so far (Avanzi et al. 2025). The adjuster's case reserve already summarises the claim history.

Model complexity sits at positions 4 and 5. Case estimates and claim status sit at positions 1 and 2. If you are building an individual reserving model from scratch, address the feature hierarchy before you address the architecture.

---

## What neural networks actually add

At N ≈ 66k, the honest answer is: nothing detectable. We think this is partly a sample size effect and partly a structural one.

The structural argument: adjuster case estimates already encode a substantial amount of non-linear information. An experienced adjuster's reserve reflects claim complexity, legal risk, likely settlement trajectory, and medical prognosis. When you hand that number to a linear regression as a feature, you are effectively importing a human expert model trained on years of claims experience. A neural network then has to discover non-linearities *beyond* what the adjuster already knows. At 66k claims across a 5×5 triangle, that is a hard task.

The sample size argument: modern neural architectures have large effective parameter counts. The generalisation benefit they provide over linear models typically materialises at much larger N. Avanzi et al. (2025) suggest ~50,000 claim-period rows as a minimum for stable FNN training. A transformer on tabular data needs more. At N = 66k with a 5×5 triangle, you are not in the regime where deep learning's advantages dominate.

When might complexity matter? We see three plausible cases:

**Very large long-tail portfolios** — UK employers' liability, professional indemnity, US casualty. Claims portfolios with 500k+ claim-period observations where non-linear interactions between injury severity, claimant age, legal representation, and development period might genuinely be learnable. No published evidence yet on whether this materialises.

**Non-stationary environments** — if settlement behaviour, legal inflation, or claims handling practice changes structurally during the training period, a model that can capture time-varying dynamics (via transformer attention or LSTM state) may outperform one that cannot. The 2020–2023 period in UK motor and BI had exactly this character. Linear regression with static features will not adapt.

**Heterogeneous multi-line portfolios** — if you are reserving across property, liability, and motor simultaneously and the development dynamics are fundamentally different, a model with sufficient capacity to learn separate embeddings per line could outperform fitting separate linear regressions. Though fitting separate linear regressions per line is also a reasonable response.

We do not think any of these cases are strong enough to justify leading with neural networks on a first implementation. They might be strong enough to justify a neural network experiment as a second step, once the linear baseline is working well.

---

## Practical implications for UK reserving teams

The academic result is clear. The question for a UK reserving actuary is whether it is achievable in practice. The barriers are real but surmountable.

**What you need that you probably have.** Every UK claims management system — Guidewire, Duck Creek, and most legacy systems — stores case reserves (claims incurred) and claim status at each valuation date. The data exists. Getting it into a clean quarterly snapshot format with consistent claim identifiers across development periods is a data engineering task, but not an extraordinary one. A mid-size insurer with a functioning data warehouse should be able to produce the input file for a PtU linear regression in weeks, not months.

**What you need that you might not have.** Reporting dates (FNOL dates) going back far enough to cover your longest development tail. Transaction-level payment histories, not just cumulative paid. Consistent claim identifiers across re-opened and merged claims. These are the harder data requirements, and they are where the 3–12 month data preparation estimate in the adoption barrier literature comes from. But note: PtU linear regression needs reporting dates and case reserves. It does not need transaction-level payments or sequence histories.

**The minimum viable experiment.** Two thousand claims with at least two development points each, case reserves at each valuation, claim status, and cumulative paid. That is achievable for most mid-size UK insurers on any line with a multi-period tail. A linear regression model with PtU factors on this data should materially beat the chain-ladder RMSEP for that portfolio. If it does not, you have learned something important about your case reserve quality.

**The regulatory position.** PRA SS8/24 does not prohibit individual ML reserving models. It requires documented back-testing against traditional methods, which is exactly what Richman and Wüthrich do in arXiv:2603.11660. A linear regression individual model has a shorter explanation than a neural network and a stronger theoretical basis. We think it passes the auditability test more cleanly than any of the deep learning alternatives — which matters in the current UK regulatory environment.

**Case reserve quality.** If your adjusters' case reserves are systematically biased — over-reserved on small claims, under-reserved on large ones, stale for long-running open claims — the individual model will amplify that bias rather than correct it. Before deploying an individual model in production, check your case reserve development factors by claim age and size. The model is only as good as the information you give it.

---

## The broader lesson

The individual claims reserving literature sometimes reads as a competition between architectures. Mack gives way to chain-ladder-with-covariates gives way to FNN gives way to LSTM gives way to transformer. Each new architecture is positioned as a step forward.

Richman and Wüthrich's March 2026 result breaks this framing. The step forward was always available: use the information in your claims system rather than discarding it. Once you do that — with a linear model, using adjuster estimates and claim status — the architecture competition produces no detectable winner at any portfolio size that has been tested.

That is not a finding that undermines the last decade of individual reserving research. Most of the theoretical framework — PtU factors, censored exposure, RBNS/IBNR separation, uncertainty quantification — is essential scaffolding for any individual model, simple or complex. What it does undermine is the idea that complexity is where the value lives.

For a UK reserving team deciding whether to invest in individual claims reserving, the April 2026 evidence says: start with linear regression and the right features. You can get most of the way to the theoretical frontier with a model that runs in seconds, that a non-specialist actuary can understand, and that does not require a GPU.

---

*The paper: Richman, R. and Wüthrich, M.V. (March 2026), arXiv:2603.11660. The companion paper on projection-to-ultimate factors: arXiv:2602.15385. The Avanzi et al. FNN/LSTM comparison: arXiv:2601.05274 (December 2025).*

*Our Python implementation of the PtU algorithm is described in a [separate post](/2026/03/26/ptu-reserving-python-implementation-gap/). The `insurance-reserving-neural` package (PyPI) covers FNN and LSTM individual RBNS reserving for teams that want to test the neural baseline.*
