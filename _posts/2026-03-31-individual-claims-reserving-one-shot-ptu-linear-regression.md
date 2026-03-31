---
layout: post
title: "The Simplest Individual Reserving Model That Works"
date: 2026-03-31
categories: [reserving, individual-claims]
tags: [RBNS, IBNR, PtU, linear-regression, chain-ladder, Richman-Wüthrich, individual-reserving, claims-incurred, case-reserves, OLS]
description: "A one-shot OLS regression on individual claims — using the adjuster's case reserve history as input — outperforms chain-ladder on liability data. Richman and Wüthrich's new paper (arXiv:2603.11660) shows neural networks add nothing. We explain the method, the data constraint that makes it work, and what it means for UK reserving practice."
author: burning-cost
---

Linear regression on individual claims beats feedforward neural networks and transformers. That is the headline finding from Ronald Richman and Mario Wüthrich's paper "One-Shot Individual Claims Reserving" (arXiv:2603.11660, submitted 12 March 2026). We think this result matters more than it sounds.

The method is called "one-shot" because it predicts ultimate cost from current partial development in a single step, without chaining J-1 age-to-age factors. The cohort consistency constraint — a single filter on which claims are eligible for the learning set — removes the structural IBNR contamination that makes naive individual projections biased. Once you have that constraint, OLS does the rest.

---

## Why chain-ladder fails on individual claims

Chain-ladder development factors are estimated from aggregate triangles where the numerator includes claims that were not even reported at the beginning of the period. Late reporters (IBNR claims) inflate the development factor. If you then apply that inflated factor to a single known open claim, you are systematically over-projecting it.

The compounding problem is a second issue. For the most recent accident year, you need J-1 sequential factors to reach ultimate. Each factor carries estimation error; they multiply. The most immature year — the one with the smallest paid amount and the largest reserve — has the worst-estimated chain and the largest factor product.

The Mack chain-ladder on the paper's liability dataset has an error-to-RMSEP ratio of 213%. The reserve is ‐4,204 against a true out-of-lower-triangle of 15,730. This is not a rounding error; it is a structural failure from mixing RBNS and IBNR development patterns in the same triangle.

---

## The one-shot approach

Instead of iterating through J-1 development factors, project directly from current cumulative paid to ultimate in one step:

```
C_hat_ultimate = F_j × C_current
```

where F_j is a Projection-to-Ultimate (PtU) factor — not a link ratio, but the full product of all remaining factors from current development age j to ultimate. This is mathematically equivalent to chain-ladder (see Proposition 2.2 in the companion paper, arXiv:2602.15385). The difference is in how F_j is estimated.

**The cohort consistency constraint** is the key insight. When estimating the PtU factor at development age j, restrict the learning set to claims that were already reported by age j. Both numerator and denominator of the factor use the same RBNS cohort. Late reporters are excluded entirely. This removes the IBNR contamination.

The algorithm runs backwards from the latest development age to age zero. At each step, the targets are the estimated ultimates from the previous (later) step, except for claims old enough to be fully settled.

Extending to regression is straightforward: replace the scalar PtU ratio with any regression function. The learning set at step j is claims where (a) reporting delay ≤ j and (b) the accident period is old enough that the ultimate is approximately known. Fit OLS or any sklearn-compatible model. The paper tests weighted GLM (chain-ladder equivalent), unweighted OLS, a feedforward network, and a transformer.

---

## OLS beats neural networks — and the reasons are structural

On the accident insurance dataset (66,639 claims, 5×5 triangle), unweighted OLS outperforms the feedforward network on every accident year:

| Accident year | OLS Ind.RMSE | FNN Ind.RMSE |
|---------------|-------------|-------------|
| 2 | 1.483 | 1.521 |
| 3 | 2.954 | 3.017 |
| 4 | 4.261 | 4.289 |
| 5 | 8.234 | 8.342 |

The transformer shows no improvement either. The authors' own conclusion: "The networks do not provide a significantly better predictive result."

This is not a surprise when you think about what the learning set looks like at each backward recursion step. A 5×5 triangle gives at most five accident-period cohorts per step, with cohort eligibility restricted by both reporting delay and accident period maturity. At the earliest development ages, the eligible sample can be a few hundred observations. Neural networks cannot generalise usefully at that scale. The signal is also approximately linear — paid amount at age j is a near-linear predictor of ultimate — and OLS captures it almost perfectly.

The practical implication is blunt: do not reach for a neural network until you have exhausted linear regression and have a principled reason to expect a nonlinear structure. For reserving datasets at any realistic UK portfolio size, that point has not arrived.

---

## Case reserves are the strongest feature

The liability dataset finding is the most important practical result in the paper. Three feature sets were tested on 21,991 liability claims:

| Feature set | Ind.RMSE | vs. paid only |
|-------------|---------|---------------|
| Cumulative paid only | 4.265 | — |
| Paid + claims incurred | 3.154 | −26% |
| **Claims incurred only** | **3.089** | **−28%** |

Claims incurred here means adjuster's case reserve plus cumulative paid — the standard "incurred" figure your claims system tracks. Using it as the sole predictor reduces individual RMSE by 27.5% relative to payment history. Adding payment history back in slightly worsens things.

This makes sense to anyone who has spent time in a claims department. An adjuster reviewing a liability claim at 18 months does not just see a payment total; they see injury severity notes, legal correspondence, and a view of likely settlement. Their case reserve encodes information that will not appear in the payment history for another two years. The regression is effectively treating the adjuster as a noisy sensor of the ultimate loss, which is exactly what the adjuster is.

The consequence for UK EL and PL books is direct. Bodily injury reserves carry much of the liability uncertainty. If your claims system tracks the case reserve history, you can capture a 28% reduction in individual claim RMSE over pure payment development — with a linear model and no hyperparameter tuning.

---

## The 15-line implementation

The minimum viable implementation is genuinely small. In Python, using scikit-learn:

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_ptu_ols(claims: pd.DataFrame, J: int) -> dict:
    """
    Backward recursive PtU regression.
    claims must have: claim_id, accident_period, reporting_delay,
                      development_period, claims_incurred (or cumulative_paid)
    J: maximum development periods.
    Returns dict: claim_id -> estimated ultimate.
    """
    ultimates = {}

    # Initialise from fully-settled claims at j = J-1
    for j in range(J - 1, -1, -1):
        learn = claims[
            (claims["reporting_delay"] <= j) &
            (claims["accident_period"] <= claims["accident_period"].max() - j)
        ].copy()

        # Targets: known ultimates for fully settled, estimated for the rest
        learn["target"] = learn["claim_id"].map(ultimates).fillna(
            learn["ultimate_actual"]   # ground truth where available
        )

        model = LinearRegression().fit(
            learn[["claims_incurred"]], learn["target"]
        )

        # Predict for RBNS claims currently at this development age
        to_predict = claims[
            (claims["development_period"] == j) &
            (claims["reporting_delay"] <= j)
        ]
        preds = model.predict(to_predict[["claims_incurred"]])
        for cid, pred in zip(to_predict["claim_id"], preds):
            ultimates[cid] = float(pred)

    return ultimates
```

The data engineering is harder than this. The main challenge is that `claims_incurred` here is not the current case reserve — it is the case reserve as at each historical development period. You need a panel of snapshots, not a single current value. Most UK claims systems record the current reserve and overwrite it. If yours does that, you can only use the paid-only feature set, and you forfeit the 28% improvement.

That is worth flagging to your CIO before starting this project.

---

## IBNR remains chain-ladder's problem

The individual model covers RBNS claims only — claims already reported at the evaluation date. IBNR reserves are computed as a residual:

```
IBNR_reserve_i = CL_total_i − sum(individual RBNS ultimates_i)
```

This is not a bottom-up individual IBNR estimate. It is chain-ladder's aggregate figure minus what the individual model accounts for. IBNR uncertainty is entirely chain-ladder's uncertainty.

For UK motor bodily injury and EL books in early development, IBNR can exceed RBNS. The individual model's precision gains on the RBNS component are bounded by whatever IBNR uncertainty remains. On a long-tailed line at accident year 1, IBNR is likely to dominate the reserve; a better RBNS model changes the total by less than it changes the component.

This is not a criticism — the authors are explicit about it. Individual RBNS modelling is a genuine improvement. But it is not a chain-ladder replacement. You need both.

---

## Where this applies in the UK market

**Best candidates:**

Motor bodily injury: long development, systematic case reserve tracking, moderate IBNR-to-RBNS ratio at years 2+. The liability finding (incurred > paid as predictor) applies directly. Typical UK MBI books run 5,000–15,000 claims per accident year — well within OLS range, not large enough to justify neural network complexity.

Employers' and public liability: same characteristics. Case reserves are central to how claims handlers manage these files; the adjuster information signal is strong. UK EL has the complication of very long development tails (20+ years), which is beyond the 5×5 empirical validation in the paper.

**Weaker candidates:**

Short-tailed lines (motor property damage, household contents): chain-ladder already works. RBNS reserves are small relative to premiums. Adding individual claim modelling complexity does not move the needle.

**Data infrastructure question first:** Before building the model, answer one question: does your claims system store the case reserve at each point in time, or does it overwrite? If it overwrites, the strongest feature is unavailable until you fix the data pipeline. That is a six-to-twelve month project at most large UK insurers, and it should be done regardless of which reserving method you ultimately use.

---

## Regulatory positioning

Solvency II (UK-retained) best estimate technical provisions require probability-weighted, discounted cash flows at the claim or contract level. Individual PtU distributions are in principle more aligned with what the regulation asks for than aggregate triangle methods. The chain-ladder is an approximation we have all been accepting by default.

The PRA's SS8/24 on internal model standards requires documented, validated methodology for any novel approach used in technical provisions. A paper submitted in March 2026 with no public code and no independent replication does not yet clear that bar. The practical path is to shadow-run the individual model alongside chain-ladder for a full reserving cycle — ideally two — before seeking approval for use in technical provisions. The comparison data you generate in the meantime is also your validation evidence.

---

## What is new, and what is not

The cohort consistency constraint (restricting learning sets to claims reported before period j) is the genuine intellectual contribution. Previous grossing-up methods — the approach dates to Lorenz-Schmidt in 1999 — applied PtU factors without this restriction, mixing RBNS and IBNR claims in the denominator. The bias was known but not formally addressed.

The ML extensibility framework (the learning set formulation in equation 4.2 of the paper) is also new: a clean specification that lets you swap any sklearn-compatible model into the backward recursion. The finding that OLS suffices empirically is the result of applying this framework to real data, not a prior assumption.

What is not new: PtU factors as a concept, RBNS/IBNR decomposition, and individual claims modelling in general. Gabrielli, Richman and Wüthrich's 2020 six-network architecture did individual reserving; so did Delong, Lindholm and Wüthrich in 2021. The contribution here is reducing complexity to something that can be implemented and validated in days.

We covered the RL-based individual reserving approach — Avanzi et al.'s MDP/Soft Actor-Critic method — in a post from last week. The contrast is instructive: that approach sits at the maximally complex end of the individual reserving spectrum. This one is at the minimally complex end. For any reserving team that has not yet implemented individual claim projection, the minimal version is the right starting point.

---

## Summary

A linear regression with the adjuster's case reserve as the input variable outperforms chain-ladder on liability data and outperforms neural networks on accident data. The implementation is trivial. The data engineering — getting clean case reserve snapshots out of your claims system — is not.

The model does not replace chain-ladder. IBNR remains a chain-ladder problem. For long-tailed UK lines in early development years, that matters.

Paper: Richman, R. & Wüthrich, M.V. (2026). "One-Shot Individual Claims Reserving." arXiv:2603.11660. Companion: arXiv:2602.15385.
