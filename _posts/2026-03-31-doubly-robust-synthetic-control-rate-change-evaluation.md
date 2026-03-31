---
layout: post
title: "Why your rate change evaluation should be doubly robust"
date: 2026-03-31
categories: [causal-inference, pricing]
tags: [synthetic-control, difference-in-differences, doubly-robust, rate-change-evaluation, causal-inference, insurance-causal-policy, FCA-Consumer-Duty, SDID, DRSC, Sun-Xie-Zhang-2025, Neyman-orthogonality, panel-data, model-risk]
description: "Sun, Xie & Zhang (arXiv:2503.11375) combine parallel trends and synthetic control into a single estimator that remains consistent if either assumption holds. We explain the math, the implementation in insurance-causal-policy, and why this matters for FCA Consumer Duty evidence packs."
author: burning-cost
---

Every UK pricing team eventually needs to answer a specific question: did that rate change actually cause the lapse increase we observed? Or was the market moving anyway?

The standard answer is a difference-in-differences analysis. Pick some control segments, check that trends were parallel before the change, estimate the ATT. Done. But the pre-trend test is diagnostic, not a proof. Parallel trends is an assumption about an unobservable counterfactual — what the treated segments would have done without the rate change. If you are evaluating a Q1 2022 rate change that partly overlaps with GIPP reform, "parallel trends" is doing a lot of work.

The alternative is a synthetic control: build a weighted combination of control segments that matches the treated segment's pre-treatment trajectory, then use the post-treatment divergence as your estimate. No parallel trends required — you are identifying from the SC matching quality. The problem is that the SC fit has its own assumption: the treated unit's counterfactual must lie in the span of the controls. In a motor book with five Scottish segments and twelve controls in England, that is also doing a lot of work.

Sun, Xie & Zhang (arXiv:2503.11375, 2025) asks whether you need to choose. Their doubly robust synthetic control (DRSC) estimator is consistent if *either* the SC weights are correctly specified *or* parallel trends holds. You only need one of the two assumptions to hold. This is not a mild improvement — it is a qualitatively different robustness guarantee.

---

## The identification logic

The standard DiD ATT is identified under parallel trends:

$$E[Y_T(0) - Y_{T-1}(0) \mid G=1] = E[Y_T(0) - Y_{T-1}(0) \mid G=0]$$

where $G=1$ denotes the treated group and $Y_t(0)$ is the potential outcome under no treatment. You cannot test this directly, only falsify it in pre-treatment periods.

The synthetic control ATT is identified under a different assumption: there exist group-level weights $w_g$ such that the treated group's conditional mean equals a weighted sum of control group conditional means across all pre-treatment periods:

$$E[Y_t(0) \mid G=1] = \sum_{g \neq 1} w_g \cdot E[Y_t(0) \mid G=g] \quad \forall t < T_{\text{post}}$$

These two assumptions are non-nested. Parallel trends allows arbitrary level differences between treated and controls but requires a common *trend*. Synthetic control allows arbitrary trend differences but requires the treated counterfactual to lie in the *span* of the controls. A dataset can satisfy one, both, or neither.

The DRSC moment function combines both into a single estimand. For the aggregate panel case (no individual-level covariates — which covers almost all insurance segment panels), the per-observation contribution to ATT estimation is:

$$\phi_i = \frac{1}{\pi_1} \left[ G_{1,i} - \sum_{g} w_g \cdot \frac{\pi_1}{\pi_g} \cdot G_{g,i} \right] \left( \Delta Y_i - \bar{m}_\Delta \right)$$

where $\Delta Y_i = Y_{T} - Y_{T-1}$, $\bar{m}_\Delta$ is the pooled control trend, $\pi_g = P(G=g)$ is the group's share of the sample (simplifying to group size ratios in the no-covariate case), and $w_g$ are SC weights estimated via unconstrained OLS on the pre-treatment conditional mean matrix.

The ATT is $\hat{\tau} = \text{mean}(\phi_i)$ across all units.

Double robustness follows from the structure of $\phi_i$. If the SC weights are correctly specified, the term in brackets becomes approximately zero for control units, so small errors in $\bar{m}_\Delta$ do not propagate. If parallel trends holds, the term $(\Delta Y_i - \bar{m}_\Delta)$ is approximately zero in expectation for controls, so errors in $w_g$ do not propagate. Consistency requires only one arm to be correct.

A further property — Neyman orthogonality — holds under parallel trends. This means that errors in the nuisance functions (the trend estimate $\bar{m}_\Delta$ and propensity scores) contribute at second order rather than first order to the ATT bias. In practice this means you can use flexible nonparametric estimators for the nuisance objects without the resulting bias dominating the ATT estimate. When the identifying assumption is SC-only, orthogonality fails, and you need $o_p(n^{-1/4})$ rates on your nuisance estimators — satisfied by most well-regularised ML methods, but worth being aware of.

Inference uses a multiplier bootstrap with Exp(1)-1 weights:

$$\hat{\tau}^*_b = \hat{\tau} + \text{mean}(W_i \cdot \phi_i), \quad W_i \stackrel{\text{iid}}{\sim} \text{Exp}(1) - 1$$

This bootstrap is valid under *either* identification regime — you do not need to know which assumption is doing the work to get valid confidence intervals.

---

## DRSC versus SDID

We already use SDID (Arkhangelsky et al., *AER* 2021) in [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy). How does DRSC compare?

The practical differences matter more than the theoretical ones.

**SC weight structure.** SDID solves a constrained quadratic program (via CVXPY): non-negative weights, sum-to-one constraint, L2 regularisation tuned to $(N_{\text{tr}} \cdot T_{\text{post}})^{1/4} \cdot \hat{\sigma}$. DRSC uses unconstrained OLS. Negative DRSC weights are not a bug — they indicate that the treated segment's pre-treatment trajectory requires extrapolation beyond the convex hull of the controls. SDID's non-negativity constraint hides this; DRSC surfaces it as diagnostic information.

**Small donor pools.** The most common situation in UK pricing is a small donor pool: five treated regions, ten to twelve control segments. SDID's constrained QP becomes unstable with $N_{\text{co}} < 12$. DRSC OLS weights are more stable (minimum-norm solution, no constraint to satisfy). We recommend DRSC as the primary estimator when $N_{\text{co}} < 15$.

**No CVXPY dependency.** DRSC uses numpy only. If you are running on a Databricks cluster where installing CVXPY requires a change request, this matters.

**Time weights.** SDID has explicit time weights that smooth over pre-treatment periods. DRSC uses only the last pre-period versus the post-period (first difference). If there was a market event immediately before the treatment that affected only treated segments, DRSC's $\Delta Y$ will be noisy. SDID's time weights are more robust in this case.

Our recommendation: run both. If the ATT estimates are close, the result is solid. If they diverge, investigate why — it usually points to a pre-treatment data quality issue or a violation of both identification assumptions simultaneously.

---

## Implementation in insurance-causal-policy

`DoublyRobustSCEstimator` is a first-class estimator in the library, alongside `SDIDEstimator` and `StaggeredEstimator`. Usage follows the same pattern:

```python
from insurance_causal_policy import DoublyRobustSCEstimator
from insurance_causal_policy import PolicyPanelBuilder

# Build a balanced panel from your claims data
panel = (
    PolicyPanelBuilder(claims_df, segments_df)
    .set_outcome("loss_ratio")
    .set_treatment_period(2023_Q2)
    .build()
)

# Fit DRSC
drsc = DoublyRobustSCEstimator(
    panel,
    outcome="loss_ratio",
    inference="bootstrap",   # valid under PT or SC — preferred
    n_replicates=500,
)
result = drsc.fit()

print(result.summary())
# ATT:  0.042  (95% CI: 0.018, 0.066)
# SE:   0.012
# p-value: 0.0008
# Pre-trend test p-value: 0.41  (cannot reject parallel trends)

# Inspect SC weights — negative weights flag extrapolation
print(result.weights.sc_weights.sort_values())
```

The `DRSCResult.weights` object includes both the SC weights and the propensity ratios ($\pi_1 / \pi_g$), which in the no-covariate case are just the treated-to-control group size ratios. The `pre_trend_pval` field gives the F-statistic p-value from regressing pre-treatment period indicators on the residualised outcome — a falsification test, not a proof of parallel trends, but a useful diagnostic.

The event study uses per-period bootstrap standard errors:

```python
import matplotlib.pyplot as plt

es = result.event_study
fig, ax = plt.subplots()
ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
ax.axhline(0, color="grey", linestyle="-", alpha=0.3)
ax.fill_between(es["period_rel"], es["ci_low"], es["ci_high"], alpha=0.2)
ax.plot(es["period_rel"], es["att"], marker="o")
ax.set_xlabel("Periods relative to treatment")
ax.set_ylabel("ATT on loss ratio")
```

One current gap: `FCAEvidencePack` accepts `SDIDResult | StaggeredResult` but not yet `DRSCResult`. The fix is a one-line type union change in `_evidence.py`. We will ship this in the next release. In the meantime, `result.to_fca_summary()` produces the same evidence-pack-formatted output as a dict.

---

## The regulatory context

The FCA's Consumer Duty (PRIN 2A, July 2023) requires firms to evidence that pricing outcomes are consistent with good customer outcomes. The standard is "reasonable steps and good judgement" — not formal causal inference. But a 2024 multi-firm review found firms "routinely failed to demonstrate causal attribution between rate changes and observed outcomes". A before-after comparison is not causal attribution. An ATT with a confidence interval and a pre-trend falsification test is.

The GIPP remedies (PS21/5, January 2022) created natural experiments: segments where renewal pricing was constrained versus segments where it was not. The FCA ran its own evaluation of those remedies using DiD methods (EP25/2, July 2025). When the regulator's own methodology is DiD-class, it is difficult for a firm to argue a naive before-after comparison in its Consumer Duty outcome monitoring report.

PRA supervisory statement SS3/19 on model risk management adds another hook. Rate change models are subject to post-implementation validation. DRSC can serve as that validation tool: did the model produce the predicted effect? The doubly robust property is directly relevant to model risk — it reduces the probability of a spurious null result when one of the two identification assumptions turns out to be wrong.

The key outputs for a regulatory evidence pack are: the ATT estimate with confidence interval, the pre-treatment falsification test, the SC weights (showing the synthetic control construction was plausible), and a plain-English interpretation. All of these come out of `result.summary()`.

---

## When to use which estimator

We are explicit about our recommendation rather than hedging:

- $N_{\text{co}} < 15$, or no CVXPY available: use DRSC as primary.
- $N_{\text{co}} \geq 20$, non-negative weights required for stakeholder communication: use SDID as primary.
- Evaluating a rate change that spans a market disruption (COVID-19, GIPP reform): use DRSC as primary, SDID as robustness check. The DR property provides a backstop when parallel trends breaks down post-disruption.
- Always run both and report convergence.

The paper (Sun, Xie & Zhang, arXiv:2503.11375) is available as a preprint. The implementation in `insurance-causal-policy` was built directly from the paper — no reference implementation existed at time of writing. The 56-test suite covers the key paths: OLS weight estimation, moment function computation, multiplier bootstrap, event study construction, and the under-identification warning when $T_{\text{pre}} < N_{\text{co}}$.
