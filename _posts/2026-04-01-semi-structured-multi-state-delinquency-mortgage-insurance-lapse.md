---
layout: post
title: "Semi-Structured Multi-State Models for Policy Lapse: A Mortgage Paper Worth Watching"
date: 2026-04-01
categories: [techniques]
tags: [multi-state, survival-analysis, neural-networks, semi-structured, lapse, competing-risks, GAM, mortgage, insurance-survival, Consumer-Duty, policy-lifecycle, arXiv-2603-26309]
description: "Medina-Olivares, Xia, Lessmann and Klein (arXiv:2603.26309, March 2026) build a semi-structured neural network model for mortgage delinquency transitions. The method combines a GLM/GAM structured predictor with an orthogonalised neural component. The gains over a pure GAM are modest. The architecture is directly transferable to insurance policy state modelling."
math: true
author: burning-cost
---

A mortgage and an insurance policy are the same modelling problem in different clothes. A mortgage moves through states — current, 30-days past due, 60-days past due, 90+ days, default, prepaid — and the probability of each monthly transition depends on borrower characteristics, loan terms, time-in-state, and macro conditions. An insurance policy moves through states — active, lapsed, reinstated, cancelled, claim — and the probability of each monthly transition depends on policyholder characteristics, pricing, channel, tenure, and market conditions. Both are discrete-time multi-state models. Both involve competing transitions. Both are routinely reduced to binary classification problems, which is wrong.

Medina-Olivares, Xia, Lessmann, and Klein ([arXiv:2603.26309](https://arxiv.org/abs/2603.26309), March 2026, Humboldt-Universität zu Berlin) address the mortgage case properly. The paper is worth reading for insurance pricing teams not because it solves everything, but because it crystallises an architecture that is directly applicable to policy lifecycle modelling and is, as of now, not packaged for the insurance domain.

The gains over the structured baseline are real but modest. We are not building against this paper yet.

---

## The modelling problem they solve

Standard mortgage credit risk models either ignore the multi-state structure — fitting a binary default/no-default logistic regression — or model each possible transition separately without a shared competing-risks framework. Neither approach handles competing transitions correctly.

At any time a mortgage can move from current to 30-days past due, or prepay, or stay current. These are competing outcomes: only one happens. Fitting a binary logistic model for default ignores prepayment as a competing exit, which biases the estimated default probability upward for borrowers who are likely to prepay. The same problem applies to insurance: fitting a binary lapse model ignores mid-term cancellation and claim as competing exits, and the estimated lapse probability is wrong as a result.

The authors work with a four-state framework: State 0 (fewer than 30 days past due), State 1 (30–59 days), State 2 (60–89 days), State 3 (90+ days past due, absorbing). Six permissible month-to-month transitions exist: forward progressions $0 \to 1$, $1 \to 2$, $2 \to 3$, and backward recoveries $1 \to 0$, $2 \to 0$, $2 \to 1$.

Binary logistic models are fitted separately for each transition. The competing-risks adjustment then transforms the binary transition probabilities $q_{ij}(t)$ into competing transition probabilities $\pi_{ij}(t)$ — the probability of transitioning from state $i$ to state $j$ next month, given that you are currently in state $i$ and may have multiple possible exits. For states with a single exit (e.g., $0 \to 1$ is the only transition out of State 0), $\pi_{i0,1}(t) = q_{i0,1}(t)$ directly. For multi-exit states, the normalisation is:

$$\pi_{i1,0}(t) = \frac{q_{i1,0}(t) \cdot (1 - q_{i1,2}(t))}{1 - q_{i1,0}(t) \cdot q_{i1,2}(t)}$$

and analogously for $\pi_{i1,2}(t)$. The authors show that this exact discrete-time formulation reduces MSE by roughly 50% compared to continuous-time approximations that are commonly applied to discrete monthly data. That result alone — before you get to the neural component — is worth noting if you are currently applying a continuous-time competing risks correction to monthly insurance data.

---

## The semi-structured architecture

Each of the six binary transition models uses the same architectural pattern. The predictor is additive:

$$\eta(t) = \underbrace{\beta_0 + \mathbf{x}^\top \boldsymbol{\beta} + \sum_k f_k(x_k)}_{\text{structured}} + \underbrace{g_\theta(\mathbf{x})}_{\text{neural}}$$

The structured component contains an intercept, linear effects for borrower and loan covariates (credit score, LTV, DTI, interest rate, occupancy type, property type), and smooth functions of duration via B-splines — the same setup as a GAM, implemented through additive smooth terms. High-cardinality categorical variables are encoded as weight-of-evidence scores within the structured component.

The neural component is a two-hidden-layer network (128 then 64 units, GeLU activations, $\ell_2$ regularisation at $10^{-3}$) that receives all covariates and outputs a single scalar that adds to the structured predictor. All covariates go into both components.

The critical constraint is orthogonalisation. Without it, the neural network would simply relearn the structured effects, offering no incremental value and destroying the interpretability of the structured component. The authors enforce orthogonality through QR factorisation: the structured design matrix $\mathbf{X}$ is decomposed as $\mathbf{X} = \mathbf{Q}\mathbf{R}$, giving a projection matrix $\mathbf{P} = \mathbf{Q}\mathbf{Q}^\top$. The neural outputs are constrained to the orthogonal complement $\mathbf{P}^\perp = \mathbf{I} - \mathbf{P}$ of the structured column space. This prevents the network from duplicating signal that the GAM already captures, so the structured coefficients remain interpretable and the neural contribution is genuinely additive.

This orthogonalisation approach is not new to the paper — it appears in earlier work on deep additive models — but its application to discrete-time multi-state transitions with competing risks is the contribution here.

---

## What the data looks like and what the results are

The training set is approximately 95,900 loans originated March 2016 to February 2018 from the Freddie Mac Single-Family Loan-Level Dataset, generating 4.3 million monthly observations. The test set covers 47,200 loans originated March 2018 to February 2019, with observations running to December 2022 — an out-of-time design with at least 36 months of follow-up per loan.

Against a GAM benchmark with the same structured predictor but no neural component, the semi-structured model produces consistent but modest gains at early prediction horizons. Multiclass AUC improvements are in the range of 0.005–0.020 across the six transitions; Brier score differences are below 0.01. Expected calibration error shows marginal improvement. The pattern is that the neural component adds most at short horizons, where nonlinear interactions between borrower characteristics matter more than macro trajectory.

Macroeconomic variables add limited incremental value in the out-of-time evaluation, which the authors flag explicitly. Our read: the macro variables are doing most of their work through the structured component's smooth temporal baseline; by the time you have a well-fitted duration effect, the macro signal is largely absorbed.

The authors also acknowledge that the hyperparameter search for the dominant $0 \to 1$ transition used only 10 random draws because of computational constraints. This is honest but means there may be gains on the table that the paper has not captured.

---

## The insurance policy lifecycle translation

The mortgage state space maps directly onto an insurance policy lifecycle:

| Mortgage state | Insurance equivalent |
|---|---|
| State 0: current | Active: policy in force, no arrears signal |
| State 1: 30–59 DPD | At-risk: payment instruction failed, first chase |
| State 2: 60–89 DPD | Suspended: coverage suspended pending payment |
| State 3: 90+ DPD | Lapsed/void: policy terminated by non-payment |
| Prepaid (absorbing) | Cancelled by customer: proactive mid-term cancellation |

The competing transitions become:
- Active $\to$ At-risk (first payment failure)
- At-risk $\to$ Active (reinstatement, payment recovered)
- At-risk $\to$ Suspended (second failure)
- Suspended $\to$ Active (late reinstatement)
- Suspended $\to$ Lapsed (final termination)
- Active or At-risk $\to$ Cancelled (customer-initiated)
- Active $\to$ Claim (claim while current)

Fitting binary logistic regression per transition and applying the exact discrete competing-risks normalisation is the architecture, mapped directly. The semi-structured neural component then adds capacity to capture nonlinear interactions — the interaction between tenure and NCD level in the active-to-at-risk transition, for example, or the interaction between channel, premium uplift, and reinstatement probability.

For UK insurers the regulatory angle is not abstract. The FCA's loyalty penalty ban (PS21/11, effective January 2022) changed the renewal pricing landscape: customers who previously renewed automatically now have demonstrably different price sensitivity, and the 2022 book no longer looks like the 2025 book at the transition level. A multi-state model with a structured component — capturing rating factors legibly for regulatory audit — and a neural component capturing changed behavioural interactions post-GIPP is a principled architecture for this problem.

Consumer Duty sharpens this further. Demonstrating that lapse risk is understood per-customer, and that customers are not being retained through friction rather than fair value, requires a model that distinguishes reinstatement probability (a customer who wants the product but is having a payment difficulty) from cancellation probability (a customer who has decided the product is not worth the price). Multi-state modelling makes this distinction structural rather than heuristic.

---

## What the existing stack provides

The `insurance-survival` library has the infrastructure for the competing-risks piece. `CauseSpecificCoxFitter` and `AalenJohansenEstimator` handle multi-state competing risks; the [competing risks post]({{ site.baseurl }}{% post_url 2026-03-12-insurance-competing-risks %}) covers the foundation, and the [survival models for lapse prediction post]({{ site.baseurl }}{% post_url 2026-03-28-survival-models-lapse-prediction %}) works through a practical lapse context.

What the stack does not have is the semi-structured architecture: the orthogonalised neural component grafted onto a GAM predictor for each transition, with the exact discrete competing-risks normalisation. That is the gap this paper fills architecturally.

The GAM piece is straightforward — `insurance-gam` has the additive smooth infrastructure. The neural component and the orthogonalisation procedure would require a new module, likely depending on PyTorch for the network layers. The exact competing-risks normalisation is a handful of numpy operations once the binary probabilities are in hand.

---

## Why we are watching but not building yet

The honest summary of the results: the semi-structured model is better than a pure GAM, but not by much. In a dataset of 4.3 million monthly observations, a multiclass AUC gain of 0.01 is statistically real. Whether it translates to better business decisions — better retention triggers, better lapse reserves, better renewal pricing — depends on the downstream use case and the quality of the model-to-decision pipeline, which the paper does not address.

There are also open questions for the insurance adaptation. The Freddie Mac data has no analogue of the aggregator/direct channel split, which is one of the dominant drivers of lapse behaviour in UK motor and home insurance. Whether the neural component adds value on that interaction — or whether a structured component with a channel indicator already captures it — is not answerable from this paper.

The orthogonalisation procedure requires care with high-cardinality encoding. The authors use weight-of-evidence for high-cardinality categoricals in the structured component; insurance rating factors (vehicle group, postcode, occupational class) have different distributional properties, and WoE encoding is not obviously correct in all cases.

We are also not convinced that the six-transition mortgage framework is the right granularity for most insurance policy lifecycle problems. For an annual motor book, the active $\to$ suspended $\to$ lapsed chain may have only a handful of observations per mid-transition state outside the very largest direct books. The data requirements are not trivial.

Technically credible, architecturally interesting, results modest, direct applicability to UK insurance requiring real translation work. We will revisit when there is an insurance-native application with UK policy data.

---

## What to read alongside this

The full paper is at [arXiv:2603.26309](https://arxiv.org/abs/2603.26309). For the `insurance-survival` infrastructure this would build on, the [competing risks post]({{ site.baseurl }}{% post_url 2026-03-12-insurance-competing-risks %}) and the [survival models for lapse prediction post]({{ site.baseurl }}{% post_url 2026-03-28-survival-models-lapse-prediction %}) are the right starting points. For the broader question of what survival modelling infrastructure exists today, the [insurance-survival retention post]({{ site.baseurl }}{% post_url 2026-03-11-survival-models-for-insurance-retention %}) is the reference.

If you are building multi-state policy lifecycle models now rather than waiting, the [multi-state claims lifecycle post]({{ site.baseurl }}{% post_url 2026-03-26-multi-state-claims-lifecycle-poisson-glm-substitution %}) covers the Poisson GLM substitution approach, which is deployable today with the existing stack. The semi-structured neural extension is a future-state architecture that this paper makes technically legible.
