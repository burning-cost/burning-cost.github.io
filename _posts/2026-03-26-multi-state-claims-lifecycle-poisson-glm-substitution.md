---
layout: post
title: "Claims Lifecycle Modelling: The Python Gap and How to Bridge It with Poisson GLMs"
date: 2026-03-26
categories: [techniques, reserving, pricing]
tags: [multi-state-models, CTMC, poisson-glm, claims-lifecycle, msm, python, reserving, motor-bi, OIC, litigation, transition-intensities, panel-data, survival, actuarial, uk]
description: "Python has no equivalent of R's msm package for continuous-time multi-state modelling of claims. We explain the mathematics, show why a Poisson GLM substitution works for most pricing purposes, and set out honestly where you still need the full apparatus."
author: Burning Cost
---

Every UK pricing actuary has encountered a claims file that contains states — notified, assessed, litigated, settled, withdrawn — but has done nothing with them beyond tabulating the frequencies. The settlement rate and the average time-to-settlement go into a pricing assumption and the state trajectory is discarded. If that describes your process, you are leaving information on the table. Specifically, you are discarding everything about which claims migrate to litigation and how long they stay there before settling, which is precisely the information that explains the variance in BI development costs.

The mathematical apparatus to use this information is called a continuous-time multi-state model. In R, you use [`msm`](https://cran.r-project.org/web/packages/msm/index.html) (Jackson, 2011; v1.8.2, November 2024). In Python, there is no equivalent. This post explains what the framework involves, why the Python gap exists, and how to get most of the benefit using Poisson GLMs you already know how to fit.

---

## The state space for a TPBI claim

The OIC portal publishes quarterly statistics that implicitly define the multi-state model for UK third-party bodily injury claims. In practical terms, the relevant states are:

- **Notified** — claim registered, liability not yet determined
- **OIC portal** — liability admitted, claim proceeding through the small claims portal
- **Litigated** — claim exited portal or contested, now in litigation
- **Settled (paid)** — claim resolved with payment
- **Settled (nil)** — claim closed without payment (withdrawn or defended successfully)

The Q1 2025 OIC statistics show approximately 21,000 claims per month entering the portal. The average settlement time is around 251 days (mid-2023 figure, trending upward as complex mixed-injury cases work through). That 251-day average is the mean sojourn across all states, weighted by transition probabilities. A multi-state model decomposes it: the fast-settling portal claims run around 180 days; litigated claims run over 450 days. The litigation propensity — P(Litigated | Notified) — is the single most important pricing input hidden inside that average.

---

## The continuous-time Markov chain framework

The formal setup is compact. You have a state space {1, 2, ..., K} and a **generator matrix** Q, which is a K×K matrix where:

- Off-diagonal entries q_{ij} ≥ 0 are transition intensities (instantaneous rate of moving from state i to state j)
- Diagonal entries q_{ii} = −∑_{j≠i} q_{ij} (rows sum to zero)

The interpretation of q_{ij}: in a short interval dt, the probability of transitioning from state i to state j is approximately q_{ij} · dt.

From Q you recover transition probabilities at any time horizon t via the **matrix exponential**:

```
P(t) = expm(Q · t)
```

where P_{ij}(t) is the probability of being in state j at time t given you started in state i. `scipy.linalg.expm` computes this in Python. The mean sojourn time in state i is −1/q_{ii}.

With covariates on each transition:

```
log(q_{ij}(x)) = log(q_{ij}^0) + β_{ij} · x
```

Each transition gets its own covariate vector and its own coefficient vector. A claim's litigation intensity depends on claim characteristics (injury type, liability dispute, legal representation) while the settlement intensity from litigation depends on different factors (claim age, medical evidence quality, solicitor conduct).

This is the health insurance framework of Haberman and Pitacco (1999) applied to property and casualty claims. The mathematics is identical; the states are different.

---

## Why R has this and Python does not

`msm` (Jackson, *Journal of Statistical Software*, 2011) solves a specific problem called the **panel data problem**: you observe the state at times t₁, t₂, ..., tₙ but you do not observe the exact times when transitions occurred. A claim file tells you the state on each handler note date; it does not tell you the exact moment liability was admitted or the exact moment a solicitor was instructed.

The likelihood for a panel observation — state r at time s, state u at time t — is P_{ru}(t − s), the matrix exponential entry. Full likelihood is a product over observed state pairs. `msm` maximises this efficiently, handles covariates on log-transition intensities, computes SEs via the numerical Hessian, and provides output functions for occupancy probabilities, sojourn times, and hazard ratios.

No Python package does this. The candidates are:

- **PyMSM** (Rossman et al., *JOSS* 2022): Cox proportional hazards per transition, requires exact transition times. Cannot handle panel data.
- **lifelines**: single-event or competing risks (Fine-Gray). No multi-state continuous-time framework.
- **hmmlearn**, **pomegranate**: discrete-time hidden Markov. No CTMC.
- **transitionMatrix** (PyPI): matrix manipulation only, no fitting from data.

The gap is specific: no Python package implements maximum likelihood estimation of Q from panel data (periodic state observations where transition times are unknown). This is not a trivial gap to fill — the matrix exponential appears inside the likelihood, which means numerical optimisation over a constrained matrix — but it is a solvable one. Our [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) library already uses `scipy.linalg.expm` inside a continuous-time HMM (for hidden driving states), which demonstrates the Python machinery is available. The panel-data MLE variant is structurally different — states are observed, no E-step needed — but the core computation is the same.

---

## The Poisson substitution: what you can do today

Here is the result that makes this tractable without building a new library. For piecewise-constant transition intensities — constant within defined time intervals — a multi-state CTMC model is equivalent to M independent Poisson GLMs, one per transition type.

The data restructuring is the key step. Start with your panel data: one row per (claim, observation date, state). Restructure into **person-period** format: one row per (claim, time interval, state), with:

- A binary indicator T_{k,m} for whether transition m occurred during the interval
- Exposure τ_{k,m}: time the claim spent at risk of transition m in that interval

Then fit, for each transition m:

```python
import statsmodels.api as sm

# One row per (claim, interval) at risk of transition m
glm_m = sm.GLM(
    T_km,                           # 0/1 transition indicator
    X_km,                           # covariate matrix
    family=sm.families.Poisson(),
    offset=np.log(tau_km)           # log-exposure offset
).fit()
```

The coefficient vector `glm_m.params` gives you covariate effects on the log transition intensity for transition m. Standard GLM output: deviance, AIC, residual diagnostics, likelihood ratio tests for variable selection.

This is the approach recommended in the fair long-term pricing literature (arXiv:2602.04791, Section 4.1-4.2). For a TPBI lifecycle model with five transitions (Notified→OIC, Notified→Litigated, OIC→Settled, Litigated→Settled, Litigated→Nil), you fit five Poisson GLMs. Each one uses the claim characteristics you already have in your pricing database: injury type, liability dispute flag, legal representation flag, accident circumstances, policyholder NCD level.

The litigation propensity model becomes a straightforward Poisson GLM over the Notified→Litigated and Notified→OIC transitions, with the OIC portal's October 2021 reform modelled as a structural break via a time dummy. This is a problem you can solve in an afternoon with `statsmodels`.

---

## What the Poisson substitution cannot do

The substitution is not a free lunch. Be clear about three limitations.

**You cannot directly compute P(t) = expm(Qt).** The Poisson GLMs give you transition intensities per interval but not the full generator matrix in a form ready for the matrix exponential. If you want the probability that a claim notified today is still in litigation 180 days from now — which is what you need for IBNR reserving — you need either the full CTMC machinery or a simulation approach that applies your fitted intensities step by step.

**Duration dependence requires more than Poisson GLMs.** The CTMC assumes exponentially distributed sojourn times — the Markov property says the probability of leaving a state does not depend on time already spent in it. For litigated claims this is demonstrably wrong: a claim that has been in litigation for two years has a different exit rate than one that entered litigation last month. The Titman-Sharples (2010) phase-type extension handles this by expanding each observable state into latent sub-states (phases), recovering an Erlang sojourn distribution that approximates any shape. This requires the full CTMC framework; it cannot be reduced to independent GLMs.

**The approximation requires intervals short enough that constant-intensity holds.** For annual panel data with slow-moving transitions this is fine. For claims data with daily resolution and rapid state changes in the first few weeks, the piecewise-constant approximation requires short enough intervals that the computational advantage over full MLE diminishes.

---

## Connecting to what you already have

If you have read our posts on [telematics risk scoring](/2025/07/28/telematics-risk-scoring-from-trips-to-glm-features/) or worked with the `insurance-telematics` library, the mathematical connection is direct. The continuous-time HMM fitted in that library uses `scipy.linalg.expm(Q * dt)` inside an EM algorithm. The multi-state claims model uses the same computation — `expm(Q * t)` — but for observed rather than hidden states. No E-step is needed when you know which state a claim is in at each handler note.

The actuarial literature on income protection (the Haberman-Pitacco health insurance canon) and the recent individual claims reserving literature (Bladt & Pittarello, arXiv:2311.07384, 2024; Buchardt et al., arXiv:2311.04318, 2025) are arriving at the same framework from different starting points. The reserving papers define the claim state space precisely — {IBNR, RBNS open, RBNS settled} — and estimate transition intensities from the development triangle. The pricing application is the same model with policy covariates added to each transition.

---

## What we are building

The Poisson GLM substitution covers most pricing use cases. For the remainder — occupancy probability curves, actuarial present values of claim development costs, duration-dependent litigation — we need a Python implementation of panel-data CTMC fitting.

We have scoped this as `insurance-multistate`. The core of v0.1 is a `TransitionModel` class that accepts panel data `(subject_id, time, state)`, estimates Q via maximum likelihood using `scipy.optimize.minimize` over the reparametrised log-intensities, and outputs transition probabilities at arbitrary horizons via `scipy.linalg.expm`. Covariates enter as log-linear offsets on each transition intensity.

The design deliberately avoids reimplementing `msm` in full — phase-type semi-Markov, hidden state extensions, and Bayesian estimation are out of scope for v0.1. The specific gap is the panel data MLE, which no existing Python package provides. That part is approximately 300 lines of clean Python. We will build it.

---

## The practical starting point

If you have a claims database with handler note dates and states, here is the sequence:

1. Define your state space. Five states for TPBI is the right level of granularity. Fewer loses the litigation split; more creates sparse transition counts.
2. Restructure to person-period format. One row per (claim, week or month, current state). Compute exposure τ as time in state.
3. Fit one Poisson GLM per transition. Log-exposure offset. Policy covariates as predictors.
4. Examine the structural break. Add a dummy for post-October 2021 to the Notified→Litigated transition. If the coefficient is large and negative, the Whiplash Reform Programme did what it was supposed to.
5. Use the fitted intensities to simulate claim trajectories. A Monte Carlo simulation with your per-transition GLMs gives occupancy probabilities without needing `expm(Qt)` directly.

The matrix exponential approach is cleaner and faster for computing P(t) at arbitrary t. But the GLM route is implementable with existing tools, interpretable to non-specialists, and correct under the piecewise-constant assumption. Start there, and the extension to full CTMC fitting becomes an upgrade rather than a prerequisite.
