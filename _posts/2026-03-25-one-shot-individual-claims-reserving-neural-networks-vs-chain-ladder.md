---
layout: post
title: "One-Shot Individual Claims Reserving: What the Richman-Wüthrich Paper Actually Shows"
date: 2026-03-25
author: Burning Cost
description: "arXiv:2603.11660 proposes direct projection-to-ultimate on individual claims data. The honest finding: linear regression on claim status and incurred already beats aggregate chain ladder. The neural network adds little on small datasets."
tags: [reserving, individual-claims, chain-ladder, neural-networks, micro-level, rbns, ibnr, liability, accident, richman, wuthrich, insurance-severity, python, actuarial]
---

Chain ladder works on triangles. It aggregates every claim in an accident year into a single number, applies a development factor to get to the next period, and calls that the reserve. The method is a century old, implemented in every reserving platform you can name, and defensible in front of any regulator who has ever looked at a Lloyd's quarterly return.

It is also discarding most of the information in your claims system.

Each open claim has a status (open, closed), an incurred estimate from the handler, a payment history, static covariates like reporting delay and claim type, and potentially years of transaction-level data. The aggregate triangle compresses all of that into a single development factor for the accident year. Individual claims reserving methods try to use the claim-level data directly. The question the Richman-Wüthrich paper [arXiv:2603.11660](https://arxiv.org/abs/2603.11660) asks is: what does it take to make that work in practice?

---

## Why individual reserving hasn't taken off

The academic literature on micro-level reserving stretches back at least to Norberg (1993) and has included Bayesian models, survival models, and recurrent neural networks. None of it has become standard practice. The reasons are partly computational — fitting a model at claim level on a book with 200,000 open claims is not a Friday afternoon task — but mainly methodological. The existing approaches tend to require bespoke implementation, strong parametric assumptions, or training architectures that are difficult to validate and explain to a sign-off committee.

The Richman-Wüthrich paper positions itself as an attempt to fix this. The framing is deliberate: build from chain ladder's intuition, not against it, and show that the method is robust enough to become a working standard rather than a research exercise.

---

## The "one-shot" idea

Classical individual reserving proceeds iteratively: predict next period's payment, step forward, predict again. This makes the forecast sensitive to compounding errors across development periods, and it requires a model for every time step rather than one model for the whole development.

The paper's core contribution is a projection-to-ultimate (PtU) factor approach. Instead of forecasting one period at a time, the model estimates a single multiplier that takes current cumulative payments directly to ultimate — hence "one-shot." This mirrors chain ladder structurally: CL development factors are also one-step projections applied to the latest diagonal. The difference is that these factors operate at claim level rather than accident-year level, and they can depend on claim-specific features rather than being a single number applied uniformly.

Formally, for claim *i* observed at development lag *j*, the model estimates:

```
Ĉ_{i,J} = f( C_{i,j}, X_{i,j} )
```

where C_{i,j} is cumulative paid to date, X_{i,j} is a vector of claim-level features, and J is the ultimate development lag. The function f is fitted by ordinary least squares — or a neural network — over the historical data where ultimates are known.

---

## Data structure and features

The paper works with two datasets: an accident insurance portfolio (66,639 claims, 5-year development, annual resolution) and a liability portfolio (21,991 claims, same structure). Both are presented as 5×5 upper triangles — the standard chain ladder setup, but observed at claim level.

The feature set is richer than anything that goes into a triangle. For each claim at observation point j:

- Cumulative paid to date, C_{i,j}
- Claims incurred (handler reserve + paid), I_{i,j}
- Claim status (open / closed)
- Static covariates: reporting delay, claim type (work vs leisure for accident; no equivalent for liability)
- Calendar month of observation

The distinction between the accident and liability datasets matters for what follows. Accident insurance claims are shorter-tailed with relatively predictable development patterns. The liability book has longer tails and, crucially, includes claims incurred information — handler reserve estimates that are updated continuously as the claim develops.

---

## What the neural network actually does (and doesn't do)

The model is a standard feedforward network applied to the feature vector. The paper tests it in section 4.4 and arrives at a finding that deserves emphasis: **the neural network does not provide significantly better predictions than linear regression on the same features**.

This is an honest result. The authors are not burying it. Their explanation is reasonable: on a dataset of 66,639 claims observed over 5 development years, the feature set is already informative enough that a linear function of C_{i,j-1} and claim status captures most of the predictive signal. The network has capacity the data doesn't need.

The paper also tests a transformer architecture, which would allow the model to condition on the full claim history rather than just the most recent state (the Markov assumption). Result: again, no meaningful improvement on these sample sizes. The authors acknowledge this may reverse on larger datasets and describe it as a proof of concept. That is the right framing. We would not read the transformer result as evidence against transformers for reserving — we would read it as evidence that 21,000 claims is not enough data to exploit sequence-level structure.

The headline finding for practitioners is not about the network architecture. It is about what features matter. On the liability dataset, **claims incurred information outperforms cumulative paid information** as the primary input. The handler's reserve estimate — continuously updated, incorporating case-specific judgment — is a better predictor of ultimate cost than the mechanical accumulation of payments to date. This is intuitive but underappreciated. Most individual reserving implementations use payment history as the primary signal. The paper's evidence suggests that discarding the case reserve and using only payments is a material modelling error on liability lines.

---

## Honest comparison with chain ladder

The paper computes Mack chain-ladder RMSEP alongside individual model errors, which gives a rare honest like-for-like. On the accident data, Mack CL produces an RMSEP of 1,663 for the aggregate reserve. The individual bootstrap estimate is 937 — a roughly 44% reduction in estimation uncertainty. On the liability data, the corresponding figures are 1,977 (Mack) and 1,201 (individual). The linear regression model, not the neural network, accounts for most of this improvement.

Where does individual reserving win?

- **Heterogeneous claims** with different development characteristics. Chain ladder's single development factor averages over claims that will settle quickly (minor injuries) and claims that will drag for years (serious casualties). The individual model can separate them from the first observation.
- **Claims with status information.** A claim flagged as closed pays nothing further by definition. Chain ladder has no mechanism to use this. An individual model with status as a feature immediately conditions on it.
- **Lines where case reserves are informative** — casualty, liability, professional indemnity. The incurred signal is strongest here.

Where does chain ladder hold its own?

- **Short-tailed homogeneous books.** Motor physical damage, household buildings. If claims develop predictably and the portfolio is large, the aggregate approach loses little information.
- **Data quality is poor at claim level.** Individual reserving requires clean, claim-level data with consistent coding. If your claims system has coding errors in claim status, inconsistent handler reserves, or missing reporting delays, the individual model will absorb that noise. Chain ladder is more robust to data quality problems because aggregation smooths over individual coding errors.
- **Governance.** A Mack chain-ladder reserve is explainable to a non-technical committee in ten minutes. An individual-level model with claim-status and handler-reserve features requires more documentation to sign off under PRA SS1/23. This is not a technical argument against individual methods, but it is a real adoption barrier.

The paper does not claim the neural network beats chain ladder. It claims the individual structure beats chain ladder, and that a linear regression on individual features already achieves most of that gain.

---

## Where insurance-severity fits in

The connection is indirect but genuine. [`insurance-severity`](https://github.com/burning-cost/insurance-severity) models the cross-sectional distribution of claim costs — what a claim of a given type is likely to cost at ultimate. The Richman-Wüthrich model is a development model — given a claim partway through its development, what will it cost at ultimate.

These are complementary, not competing. A reserving workflow that uses individual PtU factors for development pattern and a severity distribution model for the initial incurred estimate at FNOL is combining both. The DRN component of `insurance-severity` is particularly relevant here: it refines a GLM baseline distribution into a full predictive distribution, and you can use it to put uncertainty bounds on the per-claim ultimate estimate rather than just a point prediction.

```python
from insurance_severity import GLMBaseline, DRN

# baseline: your existing severity GLM for FNOL ultimate estimates
# X: claim-level features at FNOL (vehicle type, injury category, limit)
# y: known ultimates on closed claims

baseline = GLMBaseline(severity_glm)
drn = DRN(baseline, hidden_size=64, max_epochs=300, scr_aware=True)
drn.fit(X_train, y_train)

# Per-claim predictive distribution, not just a point estimate
dist = drn.predict_distribution(X_open)
p50 = dist.quantile(0.50)    # median: central estimate per claim
p95 = dist.quantile(0.95)    # 95th percentile: high-side per claim
```

The practical scenario: use the DRN to set the initial case reserve at FNOL, feed that incurred estimate into an individual PtU model, and update as claims develop. The Richman-Wüthrich findings on the liability data suggest that treating the incurred estimate as a dynamic covariate rather than ignoring it could materially reduce reserving error on casualty lines.

The EVT module in `insurance-severity` is also relevant for tail treatment. The paper works with liability insurance, where the tail of individual claim costs is heavy. A PtU model trained on mean squared error will have poor properties in the tail — a handful of catastrophic claims will drive the loss function and the model will underperform on attritional claims. One approach: fit separate PtU models for attritional and large claims, using `TruncatedGPD` or `CensoredHillEstimator` to handle the large-claim tail properly.

---

## Limitations

Several limitations are worth flagging, some acknowledged in the paper and some not.

**Sample size.** Both datasets are small for neural network training: 66,639 and 21,991 claims across 5 development years. The conclusion that "neural networks add nothing over linear regression" should be read as valid for these sample sizes. A UK motor book with 10 years of claim-level data and 500,000+ claims per accident year is a different setting. We would expect the network advantage to emerge on larger portfolios where heterogeneity is higher and linear approximations break down.

**5-year development horizon.** The datasets use 5×5 upper triangles. Liability lines with 10+ year tails are not tested. The one-shot PtU approach may be less stable when the projection horizon spans many years rather than a handful.

**Open claims only.** RBNS (Reported But Not Settled) reserving is what the paper addresses. IBNR (Incurred But Not Reported) requires a frequency model on top and is not covered here.

**No replication data.** The liability dataset is described in detail but not publicly released. The accident data comes from a Swiss source referenced in earlier Richman-Wüthrich work. Independent replication on UK data — motor bodily injury, EL/PL casualty — would be needed before putting this into production.

**Gradient requirements.** The paper uses least-squares loss throughout. For reserving purposes, under-reserving and over-reserving have asymmetric costs (Solvency II, tax). A loss function that penalises under-prediction more heavily than over-prediction would be more appropriate for operational use.

---

## What to take away

The strongest finding in this paper is not about neural networks. It is about what information is worth using. On liability lines, handler case reserves — filed under "claims incurred" — contain predictive signal that cumulative payments do not. Most reserving implementations ignore this. Claim status (open vs closed) is similarly powerful and similarly under-used in triangle methods.

We think the paper makes a credible case that individual reserving is ready for wider adoption on the right portfolios: liability lines, casualty, any book where claims have heterogeneous development patterns and case reserves are actively maintained. Chain ladder will remain the right answer for short-tailed homogeneous lines and wherever individual claims data quality is insufficient to support a claim-level model.

The neural network question is genuinely open. On 20,000 claims, linear regression wins. On a full UK motor bodily injury book with years of panel data — where the claim trajectory is a sequence of handler updates, payment events, court appointments, and settlement offers — a sequence model has room to demonstrate value. That is a different experiment to the one in this paper.

---

**Related posts:**
- [Conformal Reserve Ranges: Finite-Sample Coverage Guarantees for IBNR Intervals](/2026/03/16/reserve-range-conformal-guarantee/) — putting distribution-free uncertainty bounds on reserve estimates
- [Double GLM for Insurance: Every Risk Gets Its Own Dispersion](/2026/03/11/insurance-dispersion/) — per-policy dispersion modelling for the severity component that feeds reserving
- [Spliced Severity Distributions: When One Distribution Isn't Enough](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) — tail treatment for the severity inputs to a claims reserving model
- [Year-End Large Loss Loading](/2026/03/14/year-end-large-loss-loading/) — when severity tail uncertainty shows up in the aggregate reserve

---

**Paper:** Ronald Richman and Mario V. Wüthrich, *One-Shot Individual Claims Reserving*, arXiv:2603.11660 (March 2026).

**Code:** [`insurance-severity`](https://github.com/burning-cost/insurance-severity) — severity distribution modelling for the claim-level inputs. Install with `uv add insurance-severity`.
