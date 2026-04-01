---
layout: post
title: "Your Retention Model Is Wrong About When Customers Lapse"
date: 2026-04-01
categories: [techniques, causal-inference]
tags: [survival-analysis, causal-inference, heterogeneous-treatment-effects, targeted-learning, tmle, left-truncation, right-censoring, lapse, renewal, telematics, ps21-22, ipcw, python, arXiv-2603-26502, Pryce, Diaz-Ordaz, Keogh, Vansteelandt]
description: "Most retention models measure whether a customer lapses. surv-iTMLE measures when — and what your pricing intervention caused. We explain the estimand, why left truncation is more common in UK data than most teams realise, and what to use while the authors release code."
math: true
author: burning-cost
---

The standard approach to renewal pricing treats lapse as a binary event. Did the customer renew or not? You fit a logistic regression or gradient-boosted tree, add a price elasticity step, and use the binary renewal flag as both the target and the outcome for any post-hoc causal analysis. This is wrong in a specific, correctable way — and a new paper from UCL and Ghent (Pryce, Diaz-Ordaz, Keogh, and Vansteelandt, arXiv:2603.26502, March 2026) puts the right framework on the table.

The core issue is this: lapse is not a binary outcome. It is a time-to-event outcome — the customer lapses at day 14, day 47, or day 312 post-renewal. The binary model collapses that curve to a point. If you are measuring the causal effect of a price increase on lapse, or the heterogeneous effect of a telematics programme on time-to-first-claim by customer segment, the binary model loses the temporal structure entirely. And in many UK insurance datasets, the temporal structure is also accompanied by **left truncation** — a data feature that quietly invalidates most standard approaches, binary or survival.

---

## Left truncation: more common than you think

Right censoring is the standard survival problem. A customer is in the data, their policy is active at the extract date, and you do not know when or whether they will eventually lapse. Most survival analysis handles this correctly.

Left truncation is different. A record enters your dataset only because it has already survived to the observation window. In UK motor and home, this happens whenever policies start mid-year: a policy sold in October that you observe in a January data extract has already survived to January. Its inception-to-observation period is not random — the policy exists in your data precisely because it did not lapse between October and January. If you model the full cohort without accounting for this, you are conditioning on survival and your estimated hazards will be biased downward. Policies look stickier than they are.

The same problem affects commercial lines more severely. Multi-year commercial policies are routine. A data extract covering policies 1–3 years into a 5-year contract has left truncation as its dominant data structure, not an edge case.

And under the FCA's PS21/22 price walking prohibition, there is a measurement task that practically demands this be correct: measuring the heterogeneous effect of a first-year discount on renewal probability over time, at the individual level, to support Consumer Duty fair value assessments. Binary renewal is insufficient. Survival CATE is what the analysis requires.

---

## The estimand surv-iTMLE targets

The paper's target is the conditional survival probability difference:

$$\tau(x, t) = \mathbb{E}[S(t \mid X=x, A=1) - S(t \mid X=x, A=0)]$$

for all $t$ in some horizon $T_{\max}$. This is a **treatment effect curve over time**, indexed by both customer covariates $x$ and time $t$. The treatment $A$ might be a price increase band, telematics enrolment, or a fraud flag triggering enhanced review.

Two things this estimand provides that no scalar does:

1. **The shape of the lapse curve.** A 10% rate increase might produce near-identical one-year lapse rates between segments but very different shapes: price-sensitive comparison-site customers lapse in days 7–30 (within the cooling-off window), while inert customers lapse, if at all, at the renewal date. The binary outcome cannot distinguish these. The effect curve $\tau(x, t)$ shows both.

2. **Heterogeneity over $x$ at every $t$.** Young urban drivers with high NCD levels respond to telematics differently than older rural drivers — and the difference itself may be time-varying. Fitting one marginal treatment effect collapses this. $\tau(x, t)$ preserves it.

Existing methods each handle part of this. Causal survival forest (Cui et al., 2023, JRSSB) estimates heterogeneous effects but handles only right censoring, not left truncation. The `survtmle` R package (Benkeser) handles right censoring with competing risks but only estimates marginal average effects, not segment-level heterogeneity. Pointwise IPCW + causal forest approximates the target but produces jagged effect curves that can violate monotonicity of the estimated survival function.

The innovation in surv-iTMLE is handling all three problems jointly: left truncation + right censoring, heterogeneous effects over $x$, and smooth, bounded effect curves over $t$.

---

## How the algorithm works

surv-iTMLE is a targeted minimum loss-based estimator (TMLE) — a doubly-robust method from the semiparametric efficiency literature. The 'i' stands for *individual* targeting: rather than running a separate TMLE for each time point (which loses smoothness), the method targets all time points jointly via a shared fluctuation parameter. This is the key computational innovation.

The algorithm has three stages.

**Stage 1 — Nuisance estimation.** Using $k$-fold cross-fitting to avoid overfitting bias:

- Propensity model $g(A \mid X)$: the probability of receiving treatment given covariates — a standard logistic regression or ML classifier
- Conditional survival curve $Q(A, X, t)$: survival probability under each treatment arm, fitted via a survival forest or Cox model
- Censoring survival function $G(X, t)$: KM or forest on the censoring indicator
- For left truncation: additionally, the entry time hazard to weight out the truncation bias

**Stage 2 — TMLE fluctuation.** The clever covariate is:

$$H(A, X, t) = \left[\frac{\mathbf{1}(A=1)}{g(A=1 \mid X)} - \frac{\mathbf{1}(A=0)}{g(A=0 \mid X)}\right] \cdot \frac{1}{G(X, t)}$$

A single fluctuation parameter $\epsilon$ is estimated by logistic regression of the current $Q$ on $H$ across all time points jointly — not separately per time point. This shared $\epsilon$ is what makes the final curve smooth and bounded in $[0, 1]$. The targeted $Q^*$ solves the efficient influence function score equation.

**Stage 3 — Effect and inference.** The treatment effect curve is $\tau(x, t) = Q^*(1, x, t) - Q^*(0, x, t)$ on a time grid. Variance comes from the influence function: $\text{Var}[\tau] = \mathbb{E}[\text{EIF}^2]/n$. Simultaneous confidence bands over $t$ are available via bootstrap or influence function.

In simulations at $n = 500$ and $n = 1{,}000$, surv-iTMLE dominates causal survival forest, IPCW + causal forest, and pointwise TMLE on both bias and smoothness — particularly under left truncation, which renders all other estimators invalid.

---

## The Python gap (and what to do now)

There is no Python package for surv-iTMLE. No R package either. The paper was submitted 27 March 2026; code release appears pending or not planned. The same gap exists for the competing approaches: `grf::causal_survival_forest` requires an R bridge via `rpy2`, `SurvITE` (NeurIPS 2021) is research code requiring GPU, and the orthogonal survival learners of Frauen et al. (arXiv:2505.13072) have no released implementation.

The closest Python-native approximation is **IPCW preprocessing + CausalForestDML** from EconML. The idea: weight each observation by the inverse probability of censoring $1/\hat{G}(X, t_{\text{obs}})$, then pass the weighted survival indicators to a standard causal forest. This does not handle left truncation and does not enforce smoothness, but it is implementable today and gets you heterogeneous survival treatment effects in Python without R.

```python
from lifelines import KaplanMeierFitter, CoxTimeVaryingFitter
from econml.dml import CausalForestDML
import numpy as np

# Fit censoring model to get IPCW weights
# (KM on censoring indicator, stratified by treatment arm)
# For each observation: weight_i = 1 / G_hat(t_i | X_i)
# then define binary_outcome_i = event_indicator_i weighted

# Nuisance models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# IPCW-weighted causal forest
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(),
    model_t=LogisticRegression(),
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42,
)
# X: customer features; T: treatment indicator; Y: IPCW-weighted binary outcome
cf.fit(Y=ipcw_weighted_outcome, T=treatment, X=customer_features)
tau_hat = cf.effect(customer_features)
```

This approximation is the practical path for now. We are watching for the surv-iTMLE code release from Pryce and co-authors; when it arrives, the theoretically correct implementation is worth the effort for books where left truncation is material.

---

## Four use cases worth measuring

**1. Rate change lapse curves.** A rate increase applied to a renewals cohort creates a natural treated/control structure (high vs low increase). Binary DiD tells you the average lapse rate at the renewal date. Survival CATE shows the lapse curve over 0–365 days post-renewal — whether price-sensitive customers lapse in days 7–30 (shopping immediately) or at month 6. This feeds directly into CLV modelling and into the price elasticity curve your optimisation model uses.

**2. Telematics enrolment on time-to-first-claim.** The binary question — does telematics reduce claim frequency? — is already answered. The more informative question is whether enrolment shifts accident *timing*: fewer claims in the first 90 days suggest initial behavioural change; stabilisation thereafter suggests the effect is transient. The heterogeneous version reveals which driver segments sustain the effect and which do not.

**3. PS21/22 compliance analysis.** Under the FCA's price walking rules, firms must demonstrate that first-year discounts do not systematically disadvantage renewing customers. Measuring the heterogeneous effect of discount level on renewal probability over time — at the individual level — is the kind of evidence the FCA expects firms to have. A binary renewal flag at the policy anniversary is not sufficient to demonstrate fair value in the sense PS21/22 intends.

**4. Commercial lines mid-term cancellation.** Multi-year commercial policies with left-truncated data extracts are where the method's handling of truncation matters most. A pricing intervention applied mid-term — a premium audit, a coverage change, a rate revision — has an effect on cancellation timing that standard binary models cannot recover correctly.

---

## What we are not doing yet

We are not building a native Python implementation of surv-iTMLE from scratch. The algorithm is specified precisely in the paper, but implementing the joint temporal targeting step correctly — and validating the TMLE fluctuation numerics without reference code — is a substantial project with meaningful risk of getting the influence function wrong and producing confidence bands with incorrect coverage.

The practical recommendation:

- For right-censored data only (most standard personal lines): IPCW + `CausalForestDML` works today, is pip-installable, and gets you heterogeneous survival treatment effects with reasonable variance estimates.
- For left-truncated data (comparison-site mid-year cohorts, commercial multi-year): wait for the surv-iTMLE code release, or use `grf::causal_survival_forest` via `rpy2` for right-censored-only approximation and accept the truncation bias as a known limitation.
- For anyone running R: `grf::causal_survival_forest` (Cui et al. 2023) is the production-ready causal survival tool and has been since 2021.

We will reassess for a full Python build when Pryce and co-authors release code, or when EconML adds a native causal survival forest.

---

## The paper

Pryce, M., Diaz-Ordaz, K., Keogh, R. H., and Vansteelandt, S. (2026). 'Targeted learning of heterogeneous treatment effect curves for right censored or left truncated time-to-event data.' arXiv:2603.26502. Submitted 27 March 2026.

The application in the paper is NSCLC immunotherapy — heterogeneous survival benefit by tumour characteristics. The estimand and algorithm translate directly to insurance.

---

## Related posts

- [Causal Survival Forest in R via rpy2](/techniques/causal-inference/) — using `grf` from Python for right-censored survival treatment effects while the Python ecosystem catches up
- [Consumer Duty and the Limits of Binary Renewal Models](/pricing/ps21-22/) — why the FCA's fair value framework demands more than a logistic regression on the renewal flag
