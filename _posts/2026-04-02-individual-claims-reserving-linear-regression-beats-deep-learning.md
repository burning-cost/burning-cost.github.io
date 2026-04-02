---
layout: post
title: "Individual Claims Reserving: Why a Linear Regression Beats a Transformer"
date: 2026-04-02
categories: [techniques, reserving, actuarial]
tags: [reserving, rbns, ibnr, chain-ladder, linear-regression, individual-claims, richman-wüthrich, ptu, ols, fnn, transformer, motor-bi, solvency-ii, ifoae, arXiv-2603-11660, arXiv-2602-15385]
description: "Richman & Wüthrich's one-shot individual claims reserving framework (arXiv:2603.11660) shows that a simple OLS model — using case estimates as the primary feature — matches or beats deep learning for RBNS reserving. We explain the mechanism, why case reserves matter more than payment sequences, and what a UK reserving team would need to implement this."
math: true
author: burning-cost
---

Two papers appeared on arXiv in early 2026 that, taken together, do something unusual in the ML-for-insurance literature: they show that a simple linear model is better than a neural network, explain precisely why, and provide a framework clean enough to implement from scratch in an afternoon. The papers are Richman and Wüthrich's *From Chain-Ladder to Individual Claims Reserving* (arXiv:2602.15385, February 2026) and *One-Shot Individual Claims Reserving* (arXiv:2603.11660, March 2026).

The headline finding: for RBNS (reported but not settled) claims, a linear regression using only case reserves as input reduces RMSE by 27.5% compared to a model using payment history alone — and matches a feedforward neural network on small-to-medium datasets. The transformer does not improve over the FNN. On a UK motor BI book, where the deep learning hype has been loudest, the message is that the main gain comes from how you structure the problem, not from the choice of model architecture.

---

## What is wrong with aggregate chain-ladder

Chain-ladder (CL) works on aggregate run-off triangles: rows are accident years, columns are development periods, and each cell is the total cumulative paid for all claims in that accident year at that development stage. The CL estimator fits development factors $f_j$ such that $\mathbb{E}[C_{i,j+1} \mid C_{i,j}] = f_j \cdot C_{i,j}$, then chains them to get an ultimate: $\hat{C}_{i,J} = C_{i,I-i} \cdot \prod_{l=I-i}^{J-1} f_l$.

This loses all the claim-level information. A motor BI accident year triangle contains, say, 40,000 claims. After aggregation, the triangle has five or ten numbers. The model is fit to ten numbers. It has no knowledge of how many claims are open or closed, what case reserves the adjusters have set, or how many late reporters are still to emerge. When you see that Mack CL on the liability data in these papers produces an error equal to 213% of its own RMSEP, this is why: aggregate CL is not a bad implementation of a good idea; it is throwing away the data.

There is a second, less-discussed problem: IBNR contamination. When fitting CL to an aggregate triangle, the denominator of the development factor at column $j$ includes accident years that are not yet fully reported. Claims still to emerge by period $j$ are implicitly counted at zero, which biases the factor estimates. The individual claims framework fixes this by construction.

---

## The projection-to-ultimate reformulation

The structural insight in arXiv:2602.15385 is a change of variable that looks trivial but has significant consequences. Define a multi-period factor:

$$F_j = \prod_{l=j}^{J-1} f_l$$

This is the "projection-to-ultimate" (PtU) factor from development period $j$: the expected ratio of ultimate losses to current cumulative paid at stage $j$. The CL ultimate estimate becomes:

$$\hat{C}_{i,J} = C_{i,I-i} \cdot F_{I-i}$$

Algebraically, this is identical to standard chain-ladder on aggregate triangles (the paper proves equivalence in Proposition 2.2). The reformulation adds no predictive power at the aggregate level. What it does is collapse the recursive multi-step prediction into a single direct prediction: ultimate equals current state times a factor. Any function — linear regression, neural network, transformer — can be substituted for that factor without needing to iterate through intermediate development periods.

At the individual claim level, the PtU factor for accident period $i$ at development stage $j$ is estimated as:

$$\hat{F}_{j-1}^{\text{RBNS}} = \frac{\sum \hat{C}_{i,J\mid\nu}}{\sum C_{i,j-1\mid\nu}}$$

where both sums run over the same cohort: claims reported by period $j-1$. This is the key correctness improvement over aggregate CL. The denominator no longer includes unreported claims; IBNR contamination is eliminated by construction.

---

## The one-shot algorithm

The full procedure in arXiv:2603.11660 is a backward recursion. Start with accident years old enough to be fully developed — you have observed ultimates for these. Work backwards, at each step:

1. Build a training dataset $\mathcal{L}_{j-1}$: all claims with reporting delay $\leq j-1$ and accident period $\leq I-j$. Targets are observed ultimates for old accident years, estimated ultimates (from the previous step of the recursion) for recent ones.
2. Fit a model $g_j(C_{i,j-1\mid\nu}, X_{i,j-1\mid\nu})$ by minimising squared loss on $\mathcal{L}_{j-1}$.
3. Apply a balance correction: scale all predictions by the ratio of summed targets to summed predictions, in-sample. This multiplicative scalar prevents systematic drift across the recursion.
4. Predict ultimates for accident year $I-j$: these become training targets in the next step.

The balance correction (step 3) is not optional. Without it, small biases at each stage of the recursion compound. With it, the aggregate reserve from the individual model is anchored to the aggregate data — which is desirable both for accuracy and for reconciliation with the existing triangle-based reserve carried on the balance sheet.

The linear regression variant of $g_j$ is:

$$\hat{\theta} = \arg\min \sum_{i,\nu} \left( \hat{C}_{i,J\mid\nu} - (\theta_0 + \theta_1 \cdot C_{i,j-1\mid\nu}) \right)^2$$

Two parameters. One development-period regression per $j$. No hyperparameter search, no random seeds, no GPU.

---

## Why case reserves matter more than payment history

This is the finding that most deserves attention. The liability insurance results in the paper test four feature combinations: paid only, case reserve only, claims incurred (paid + reserve), and paid + incurred. On the liability dataset ($N = 21{,}991$ claims):

| Feature set | Ind.RMSE |
|---|---|
| Paid only (RBNS CL) | 4.265 |
| Claims incurred (paid + reserve) | 3.089 |

A 27.5% RMSE reduction from adding one feature: the adjuster's case reserve.

The mechanism is not mysterious. Payment history tells you what has been paid out so far. For a long-tailed liability claim in development period 2 of a 10-period tail, cumulative paid is predominantly zero or a small early payment. It is almost uninformative about ultimate. The case reserve is the adjuster's forward-looking estimate of what remains to be paid — an expert assessment incorporating medical reports, solicitor letters, liability correspondence, and comparable settlements. It encodes the information that payment history cannot.

This also explains why the transformer does not improve on the FNN for small datasets. The transformer processes the full payment history sequence, using attention across development periods to identify patterns. For long-tailed claims, the early payment sequence is nearly flat. The transformer has no signal to attend to. The FNN using current incurred (paid + reserve) is using the single most informative observation at each point in time.

The implication for architecture design: before considering what model to use, consider what features are available. A linear regression with case reserves will outperform a transformer with payment history only. This is true on the paper's datasets and likely to remain true on any standard UK motor or liability book where adjusters engage meaningfully with case reserves from an early development stage.

---

## What this means for the Mack framework

The paper also contains a clean result connecting individual OLS to aggregate chain-ladder. The weighted linear regression:

$$\hat{f} = \arg\min \sum_{i,\nu} w_{i,\nu}(C_{i,j\mid\nu}) \cdot \left( \frac{C_{i,j+1\mid\nu}}{\max\{C_{i,j\mid\nu}, \epsilon\}} - f_j \right)^2$$

with weights $w_{i,\nu} = C_{i,j\mid\nu}$ and $\epsilon = 0.001$ is precisely the aggregate Mack CL estimator, applied at the individual claim level. Aggregate CL is a special case of individual OLS with specific weights and the constraint that the intercept is zero.

This is useful in two ways. First, it provides a formal validation test: the individual weighted OLS must reproduce Mack CL to floating-point tolerance when run on aggregate triangle data (after summing individual payments into cohort totals). Second, it means that the move from aggregate to individual methods is not a rupture — it is the same estimator, with fewer constraints on the feature space and without the cohort-consistency error.

---

## UK applicability

UK non-life insurers running Guidewire, Duckcreek, or legacy OPUS/TIA systems have exactly the data structure this framework requires: cumulative paid at each reserve review, case reserve at each review, claim status (open/closed/reopened), FNOL date, and accident date. A motor BI book of 200,000 claims per annum with 7-year development is a good fit for the FNN variant; a smaller book, or a team that wants an auditable model, should start with linear regression.

Three UK-specific complications are worth flagging before implementation.

**Claims inflation.** UK motor BI inflation ran at 8–12% over 2023–2025 (ABI data). A model trained on nominal cash flows learns an accident-year effect if accident year is not included as a feature. Including accident year and calendar month in $X_{i,j\mid\nu}$ is necessary; the model should treat these as continuous variables (month-of-year scaled to $[0, 1]$), not as one-hot dummies that cannot extrapolate.

**Ogden rate.** The discount rate used in lump-sum settlements for future losses is currently 0.5% (set 2019). Any model trained on data spanning an Ogden rate change will see step-changes in case reserves that are not claims development — they are a bulk repricing. An Ogden period indicator in $X$ is the minimal fix.

**PPO claims.** Periodical Payment Orders for catastrophic injury have effectively infinite development. Exclude these from the PtU model entirely and reserve them separately. Including them in the training data corrupts the PtU factor estimates for the standard development triangle.

---

## The tooling gap

There is no Python package for this. chainladder-python (CAS, v0.9.1, January 2025) is the standard Python reserving library and handles aggregate triangles well — Mack CL, ODP bootstrap, Clark LDF, Bornhuetter-Ferguson. It does not accept individual claim inputs. The two Richman-Wüthrich papers do not have public code linked as at April 2026.

In R, ReSurv (Hofman, Hiabu, Pittarello; CRAN) handles the IBNR component via survival analysis — reporting delay modelling to estimate unreported claim counts. That is the complement the PtU framework needs: RBNS from PtU, IBNR from a frequency/delay model. No integrated Python tool covering both exists.

We planned to build insurance-reserving for this cycle. The operator blocked new packages this cycle; we will return to it. In the meantime, the linear regression variant is short enough to implement directly: a backward recursion over development periods, two-parameter OLS per period, one balance correction scalar. For a team with individual claim data and a Python environment, the papers themselves (both open access on arXiv) contain the full algorithm in pseudocode.

---

## The practical verdict

The Richman-Wüthrich framework makes two claims we think are correct.

First: individual claim methods are strictly more powerful than aggregate triangle methods for RBNS reserving, because they use more information and avoid IBNR contamination in the estimation step. The margin over Mack CL on the paper's datasets is large — 69% vs 1.5% error-to-RMSEP on the accident insurance dataset. This is not a marginal improvement.

Second: the primary driver of individual reserving accuracy is the case reserve feature, not the model architecture. Linear regression with incurred beats neural networks with paid-only. For UK motor BI, where case reserves on open bodily injury claims are set by skilled adjusters from around 3 months post-FNOL, this feature is available and meaningful. A pricing actuary loading RBNS margins into frequency-severity rates should want individual RBNS estimates, not a triangle aggregate with an arbitrary tail factor on top.

The architecture question — FNN or transformer or linear regression — is secondary to sorting out your training data structure, getting cohort consistency right, and including case reserves. That is the conclusion of two papers that ran the experiments, and we think it is correct.

Both papers are open access. arXiv:2603.11660 contains the full algorithm with pseudocode and the performance tables. arXiv:2602.15385 contains the chain-ladder equivalence proof and the theoretical foundations. Start with 2603.11660 if you want to implement; read 2602.15385 if you want to understand why it works.
