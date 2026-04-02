---
layout: post
title: "You Cannot Properly Score Expected Settlement Time"
date: 2026-04-02
categories: [techniques, evaluation, libraries]
tags: [survival-analysis, proper-scoring-rules, ipcw, censoring, twcrps, murphy-diagram, insurance-survival, settlement-duration, lapse, income-protection, python, arXiv-2603-14835, Taggart, Loveday, Louis]
description: "RMSE on closed claims is biased. Corollary 3 of Taggart, Loveday & Louis (arXiv:2603.14835) proves mean settlement time is not scoreable under right-censoring. CensoredForecastEvaluator in insurance-survival v0.6.0 gives you the proper alternative: threshold-weighted CRPS, Murphy diagrams, and reliability diagrams that handle open claims correctly."
math: true
author: burning-cost
---

Take a book of motor claims. Your settlement duration model predicts expected days to close. You evaluate it with RMSE — sensible enough. But your evaluation dataset is only closed claims. Open claims are excluded because you do not yet know their settlement time.

This is not a minor data quality nuisance. It is a fundamental bias. And there is a theorem that makes it worse: mean settlement time is not a proper scoring rule target under right-censoring. No scoring function exists that uniquely identifies the mean as its minimiser when some outcomes are unobserved. You are not just measuring on a biased subset — you are measuring a quantity that cannot be validly measured at all with the tools you are using.

Taggart, Loveday, and Louis formalised this in arXiv:2603.14835, submitted March 2026. Their Corollary 3 proves that the expected value functional is not *provisionally elicitable* under right-censoring. `CensoredForecastEvaluator` in `insurance-survival` v0.6.0 implements their framework.

---

## What provisional propriety means, and why it kills your RMSE

A scoring rule $S(F, t)$ is **proper** if the true distribution minimises the expected score. It is *strictly* proper if it is the *unique* minimiser. Propriety is why we use log-loss for classification and CRPS for distribution forecasts — they reward honest forecasts.

Under right-censoring, we do not observe $t$ directly. We observe $\tilde{t} = \min(t, C)$ and an indicator $\delta = \mathbf{1}(t \leq C)$, where $C$ is the censoring time. A scoring rule is **provisionally proper** if it remains proper when we replace the unobserved $t$ with the observed $(\tilde{t}, \delta)$.

Taggart et al. show that quantile forecasts and survival probabilities at fixed thresholds are provisionally elicitable. The mean is not. Concretely: your RMSE on closed claims does not converge to the true expected settlement time as you add more data. It converges to the conditional mean of the *uncensored* sub-population — the fast-settling claims — which is a systematically different and systematically lower number.

For a UK motor book, open claims at any extract date are disproportionately complex: disputed liability, serious injury, fraud suspected. If your model is evaluated only on closed claims, it is calibrated to the easy ones.

---

## What to use instead

The paper's central positive result is that the **threshold-weighted CRPS** is provisionally strictly proper under right-censoring for any fixed horizon $\tau$:

$$\text{twCRPS}_\tau(F, t) = \int_0^\tau w(s) \left( F(s) - \mathbf{1}(t \leq s) \right)^2 ds$$

where $w(s)$ is a weight function over the time axis. For unweighted evaluation, $w(s) = 1$. For a claim management team focused on 90-day settlements, set $w(s)$ high for $s \in [60, 120]$.

The censored version replaces $\mathbf{1}(t \leq s)$ with a split:

- For $s < \tilde{t}$: the claim was still open at $s$, so $\mathbf{1}(t \leq s) = 0$ is known with certainty. Integrand: $F(s)^2$.
- For $s \geq \tilde{t}$ and $\delta = 1$ (claim closed): $\mathbf{1}(t \leq s) = 1$. Integrand: $(1 - F(s))^2$.
- For $s \geq \tilde{t}$ and $\delta = 0$ (still open): this part of the integral is unknown. It is omitted — the score is computed only up to $\tilde{t}$.

This is Equations 18–19 of the paper. The key point is that censored observations contribute *partial* scoring information. Excluding them entirely, as closed-claim-only RMSE does, throws away valid information and introduces selection bias.

---

## CensoredForecastEvaluator

`insurance-survival` v0.6.0 adds an `evaluation` subpackage. The main class:

```python
import numpy as np
from lifelines import WeibullAFTFitter
from insurance_survival.evaluation import CensoredForecastEvaluator

# Fit two competing models
# waf1 = WeibullAFTFitter(...).fit(train_df, ...)
# waf2 = ...

# Build survival functions: callables S_i(t) -> float
def make_surv_fns(model, df):
    sf = model.predict_survival_function(df)    # DataFrame: time index x obs
    return [
        (lambda col: (lambda t: np.interp(t, col.index, col.values)))(sf[c])
        for c in sf.columns
    ]

surv_fns_1 = make_surv_fns(waf1, test_df)
surv_fns_2 = make_surv_fns(waf2, test_df)

# CensoredForecastEvaluator takes:
#   - surv_fns: list of callables S(t) -> float, one per observation
#   - t_obs:    observed times (min of event time, censoring time)
#   - events:   censoring indicator (1 = event observed, 0 = censored)
#   - tau:      fixed evaluation horizon

evaluator = CensoredForecastEvaluator(
    surv_fns=surv_fns_1,
    t_obs=test_df["duration"].values,
    events=test_df["event"].values,
    tau=365.0,          # evaluate over first 365 days
    n_grid=200,
)

score = evaluator.mean_twcrps()
print(f"Model 1 twCRPS: {score:.4f}")

# Compare two models with Murphy diagram
evaluator2 = CensoredForecastEvaluator(
    surv_fns=surv_fns_2,
    t_obs=test_df["duration"].values,
    events=test_df["event"].values,
    tau=365.0,
)

fig = evaluator.murphy_diagram(other=evaluator2, labels=["Weibull AFT", "Cox PH"])
fig.savefig("murphy_settlement.png", dpi=150)
```

The `from_matrix()` class method wraps 2D survival matrices (the output format of `pycox` and `scikit-survival`) into the callables that `CensoredForecastEvaluator` expects:

```python
# scikit-survival returns a structured array of step functions
# from_matrix wraps a (n_obs x n_timepoints) numpy matrix
evaluator = CensoredForecastEvaluator.from_matrix(
    surv_matrix=rsf_pred,        # shape (n_obs, n_timepoints)
    time_grid=time_grid,         # 1D array, length n_timepoints
    t_obs=test_df["duration"].values,
    events=test_df["event"].values,
    tau=365.0,
)
```

---

## What the Murphy diagram tells you

The Murphy diagram plots $\Delta\text{score}(\alpha) = S_1(\alpha) - S_2(\alpha)$ across threshold $\alpha \in [0, \tau]$, where $S_k(\alpha)$ is the mean elementary score at threshold $\alpha$. It shows *where on the time axis* each model has the advantage.

For motor settlement duration, a typical result: Cox PH wins at $\alpha < 30$ days (short claims), random survival forest wins at $\alpha > 180$ days (long-tail claims). A single twCRPS number averaging across all $\alpha$ would mask this. The Murphy diagram reveals it directly.

The actionable decision: if your claims management KPI is 30-day settlements, use Cox PH. If you are reserving the long tail, use the forest. Both conclusions from a single diagram.

---

## Reliability diagrams under censoring

Standard reliability diagrams bin predictions by forecast probability and check whether the empirical frequency matches. Under censoring, "empirical frequency" cannot be computed naively — censored observations in a bin are not failures.

`CensoredForecastEvaluator.reliability_diagram()` uses Kaplan-Meier within each bin to estimate the empirical survival probability at the evaluation threshold, then overlays isotonic regression to show the calibration curve. A diagonal reliability diagram means your model's stated 90-day settlement probability of 0.7 actually corresponds to 70% of claims settling within 90 days in that bin. Deviation below the diagonal means you are overestimating settlement speed — a reserving problem.

---

## Three UK applications

**Motor settlement duration.** Extract all open and closed claims as at a valuation date. Open claims are right-censored at the extract date. Previously: RMSE on closed claims only, excluding 35–45% of the book at any point. Now: twCRPS on the full extract. The calibration shift is typically 8–15 days at the 90th percentile — material for large loss reserving.

**Critical illness lapse.** In-force CI policies are right-censored — they have not yet lapsed. Lapse models evaluated only on the lapsed subset are calibrated to the early lapses (price shock, direct debit failure), not the long-term persisters. CensoredForecastEvaluator applied to year-1 and year-3 CI lapse with $\tau = 365$ and $\tau = 1095$ separately.

**Income protection disability duration.** Open IP claims are not just right-censored — many have been open for years, making the censored proportion dominant in active cohorts. Scoring disability duration models on closed recoveries only produces wildly optimistic duration estimates. twCRPS on the full cohort — including open claims via the censored integral split — gives a substantially different model ranking.

---

## The fixed-$\tau$ caveat

Provisional propriety requires a fixed horizon $\tau$ that is the same for all observations. This is a real constraint.

Real insurance data has random individual censoring: policies start on different dates, the valuation date is common, so each policy has a different maximum possible censoring time. `CensoredForecastEvaluator` emits a warning when you set `warn=True` (the default) if your censoring times are not identical across observations. The scores remain useful, but the strict theoretical propriety guarantee does not hold for the random-censoring case.

For practical purposes: use a $\tau$ that is comfortably below the minimum censoring time in your cohort. If your extract has policies as young as 30 days, a $\tau$ of 28 days is valid under fixed-censoring theory. For longer evaluation horizons on mixed-age cohorts, treat the scores as approximately proper and document the limitation.

---

## What scikit-survival and lifelines give you

`scikit-survival` provides the C-index and the Brier score. The Brier score is related to twCRPS at a single threshold — it is the integrand at a fixed $t$ without integration over the time axis. Neither library provides threshold-weighted CRPS, the Murphy diagram decomposition, or quantile elicitability results for censored data.

`lifelines` provides no proper scoring rules at all.

This gap is what `CensoredForecastEvaluator` fills.

---

## References

Taggart, R. J., Loveday, N., and Louis, S. (2026). 'On the evaluation of time-to-event, survival time and first passage time forecasts.' arXiv:2603.14835. Submitted March 2026.

---

## Related posts

- [The Survival Treatment Effect Your Retention Model Is Not Computing](/techniques/causal-inference/libraries/2026/04/02/surv-itmle-heterogeneous-survival-treatment-effects/) — IPCW and causal survival forests for censored renewal data
- [insurance-survival: Survival Modelling for Insurance Pricing](/libraries/pricing/insurance-survival/) — library overview, left truncation support, competing risks
