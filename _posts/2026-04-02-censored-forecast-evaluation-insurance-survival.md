---
layout: post
title: "You Cannot Properly Score Expected Settlement Time"
date: 2026-04-02
categories: [techniques, evaluation, libraries]
tags: [survival-analysis, scoring-rules, crps, elicitability, censoring, insurance-survival, murphy-diagram, reliability-diagram, settlement-duration, income-protection, critical-illness, reserving, python, arXiv-2603.14835, Taggart, Loveday, Louis]
description: "Corollary 3 of Taggart, Loveday & Louis (2026) proves the mean is not provisionally elicitable under censoring — no proper scoring rule can uniquely identify E[T] as its minimiser when outcomes are right-censored. Every team scoring settlement duration models by RMSE is optimising an unscoreable functional. insurance-survival v0.6.0 ships CensoredForecastEvaluator as the correct alternative."
math: true
author: burning-cost
---

Most reserving teams have a settlement duration model. It predicts when claims will close. They evaluate it by RMSE or MAE against the closed claims in the validation period. This feels sensible — you have ground truth where you have it, you compute errors on that subset, you pick the model with the smallest error.

There is a theorem that says this approach is formally indefensible, and we now have a Python implementation that makes the correct alternative accessible.

The theorem is Corollary 3 of Taggart, Loveday & Louis (arXiv:2603.14835, March 2026): the mean functional E[T] is not provisionally elicitable under right-censoring. No scoring function S(x, t) exists such that the expected score E[S(x, t)] is uniquely minimised by the true expected settlement time when some observations are censored. You can minimise RMSE on the closed claims you observe — but whatever you are minimising is not the expected score of any proper rule. The minimiser has no identification guarantee.

This is not a minor statistical caveat. Most motor claims books have open claims at any valuation date. Income protection disability durations have many open claims that have not yet terminated. Critical illness lapse models are evaluated on in-force policies that may or may not lapse before the data extract. All of these are right-censored. If your model selection criterion is RMSE on the closed subset, you are selecting models by an unscoreable criterion.

---

## What provisional propriety means and why it matters

A scoring rule S(x, t) is strictly proper if E[S(x, t)] is uniquely minimised at x = F, the true distribution of T. You want proper scoring rules because only a proper rule gives you an honest incentive to predict truthfully — any other rule can be gamed, or can inadvertently select for a misspecified model.

Under censoring, the full distribution of T is not observable. Taggart et al. introduce **provisional strict propriety**: a scoring rule is provisionally strictly proper if E[S(F, T) | T > c] is uniquely minimised at F = F*, the true distribution, for every fixed censoring point c. Intuitively, you condition on what you can observe (the claim has not yet closed at the observation horizon c), and demand the incentive still works correctly.

Corollary 3 proves that E[T] — the scalar quantity most teams are trying to forecast — is not provisionally elicitable. No provisionally strictly proper rule has E[T] as its unique minimiser. The result applies to any summary that is not a quantile or an interval around quantiles.

Contrast this with quantiles: the pinball loss (also called the check function or quantile loss) is provisionally strictly proper. The q-quantile of T is provisionally elicitable. If you want to score a model by how well it predicts the median settlement time, the 90th percentile, or the interquartile range, you have a proper scoring rule available. If you want to score by mean settlement time, you do not.

---

## The full-distribution alternative: threshold-weighted CRPS

The correct approach for evaluating survival curve forecasts is the **threshold-weighted CRPS**, derived in Taggart et al. for the censored setting:

$$\text{twCRPS}_\tau(F, t) = \int_0^\tau w(\theta) \bigl(F(\theta) - \mathbf{1}[t \leq \theta]\bigr)^2 \, d\theta$$

where F is the predicted CDF (i.e., 1 minus the survival function), t is the observed event time (or censoring time), τ is the evaluation horizon, and w(·) is a weight function that allows you to emphasise specific parts of the time axis.

The censoring-adjusted formula — which is what `CensoredForecastEvaluator` computes — correctly handles observations censored before τ. For an observation with censoring time c < τ, the integral is split: below c the term is F(θ)², above c the term is (1 − F(θ))². The ordinary CRPS formula ignores this split and is biased. The paper derives this as Equations 18–19.

Setting w(·) = 1 gives the unweighted twCRPS. More usefully, you can set w(θ) = 1[θ ≤ 90] to emphasise settlement within the first 90 days, which is the KPI many motor claims operations track most closely.

---

## insurance-survival v0.6.0: CensoredForecastEvaluator

The new `CensoredForecastEvaluator` class implements the Taggart et al. framework. Survival functions are passed as callables `S: t -> [0, 1]`, not as raw data frames. The `from_matrix()` helper wraps the 2D matrix output from scikit-survival, pycox, or lifelines into the callable interface.

```python
from insurance_survival.evaluation import CensoredForecastEvaluator, from_matrix

evaluator = CensoredForecastEvaluator(tau=365.0)  # 1-year horizon
score = evaluator.twcrps(surv_fns, T_obs, event)

# Compare two models
ranking = evaluator.compare({
    'Cox PH': cox_surv_fns,
    'Random Survival Forest': rsf_surv_fns,
})

# Murphy diagram — where does each model win?
fig = evaluator.murphy_diagram({
    'Cox PH': cox_surv_fns,
    'Random Survival Forest': rsf_surv_fns,
}, T_obs, event)
```

`T_obs` is the vector of observed event or censoring times. `event` is the censoring indicator (1 = event occurred, 0 = censored). The evaluator integrates over `[0, tau]` using the trapezoid rule on a 200-point grid, which is sufficient for smooth parametric survival functions. For stepwise discrete hazard models (e.g., pycox DeepHit output), pass `n_grid=500` or higher.

The `compare()` method returns a DataFrame ranking models by mean twCRPS. The `murphy_diagram()` method is where the real insight lives.

---

## Murphy diagrams: seeing where your model fails

A Murphy diagram plots the elementary score at each threshold θ ∈ [0, τ] for each candidate model. Elementary scores are the building blocks of the CRPS — their integral over θ equals the total twCRPS — and they reveal whether model A beats model B uniformly across the time horizon, or only at certain parts of it.

The practical interpretation is direct. If your Cox PH model's Murphy curve is below your Random Survival Forest's curve for θ < 60 days but above it for θ > 180 days, Cox is better for predicting short-horizon settlement and RSF is better for long-duration claims. This is actionable: you might use Cox for fast-track claims and RSF for litigation-prone claims, with a simple routing rule based on claim characteristics at first notice.

RMSE on closed claims gives you one number that aggregates across the entire duration distribution. Murphy diagrams show you where in the distribution the advantage lies. For a reserving actuary asking "should I use model A or model B for this segment?", the Murphy diagram answers the question that RMSE cannot.

---

## Reliability diagrams under censoring

The third diagnostic is the reliability diagram: it checks whether the model's stated survival probabilities are empirically correct. If a model says S(90) = 0.6 (i.e., 40% probability of settling within 90 days), does roughly 40% of the relevant cohort actually settle within 90 days?

Under censoring you cannot compute empirical frequencies directly — some of the observations have not yet resolved. The implementation uses the Kaplan-Meier estimator per probability bin, which handles censored observations correctly. Each bin shows the mean predicted probability against the KM-estimated empirical frequency, with isotonic regression overlaid to identify systematic over- or under-confidence.

Motor books with large claims that linger — commercial vehicle, personal injury, liability — often show a characteristic pattern: models are well-calibrated for short settlement times but systematically overoptimistic at 12–18 months, because the training data underrepresents the tail. The reliability diagram surfaces this before you use the model for bulk reserving.

---

## Three UK insurance applications

**Motor claims settlement duration.** Right-censored at the valuation date: claims still open are censored observations. Set `tau=365.0` for a one-year evaluation horizon. Use the Murphy diagram to check whether your model's advantage (versus a simple log-normal baseline) holds uniformly across settlement times or only for fast-settling claims.

**Income protection disability duration.** Many claims are open and generating benefit payments at any valuation date. The evaluation problem is identical to motor settlement. The threshold-weighted CRPS gives you a model quality score that accounts for the censoring structure rather than pretending it does not exist.

**Critical illness lapse prediction.** In-force policies are right-censored — you observe the policy term so far, not its ultimate duration. Scoring lapse models on the closed policies only (those that have already lapsed or matured) biases selection toward models that predict well for short-duration policyholders. The censoring-corrected twCRPS evaluates across the full in-force book.

---

## The important caveat: provisional propriety requires fixed tau

Provisional propriety — and therefore the validity of the twCRPS as a proper scoring rule — requires that the evaluation horizon τ is the same for all observations. The underlying condition is that censoring is deterministic given τ: an observation is censored if and only if its true event time exceeds τ.

Real insurance data usually has random individual censoring: each claim has its own entry date, and the valuation date creates a different censoring time for each. A claim opened two years ago and still open has been observed for two years; a claim opened six months ago and still open has been observed for six months. Their censoring times differ.

`CensoredForecastEvaluator` emits a warning about this by default. For random-censoring scenarios, the IPCW-Brier score (already in `insurance_survival.metrics`) remains the appropriate complement: it uses inverse-probability-of-censoring weighting to construct an unbiased score when censoring times are random. The two methods are not alternatives — they answer slightly different questions. The threshold-weighted CRPS under fixed τ is the correct choice when you are evaluating over a defined policy or claims horizon; IPCW-Brier is the correct choice when individual observation windows differ materially.

---

## What scikit-survival and lifelines give you, and why it is not enough

scikit-survival ships the concordance index and the IPCW-Brier score. pycox adds IPCW-negative binomial log-likelihood. Neither library provides threshold-weighted CRPS, Murphy diagrams for censored data, or reliability diagrams under censoring.

The C-index has a specific and limited interpretation: it measures discrimination, the probability that a model ranks two randomly drawn subjects in the correct order. It says nothing about calibration — a model can have a C-index of 0.85 and be catastrophically miscalibrated at the 90-day threshold. The IPCW-Brier score is a proper scoring rule and is calibration-sensitive, but it returns a single scalar and does not decompose across thresholds.

Murphy diagrams and threshold-weighted CRPS give you the decomposition. For practical model selection in reserving or pricing, the decomposition is the useful output, not the scalar summary.

---

## Installation

```
uv add insurance-survival>=0.6.0
```

The `evaluation` subpackage has no new dependencies beyond what v0.5.0 already required: numpy, scipy, matplotlib, lifelines, and scikit-learn.

**Paper:** [arXiv:2603.14835](https://arxiv.org/abs/2603.14835) — Taggart, Loveday & Louis (2026) | **Library:** [insurance-survival on PyPI](https://pypi.org/project/insurance-survival/) | **GitHub:** [insurance-survival](https://github.com/burning-cost/insurance-survival)
