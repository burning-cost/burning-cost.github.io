---
layout: post
title: "Your Propensity Model Is Wrong. TMLE Doesn't Mind."
date: 2026-03-12
categories: [libraries, pricing, causal-inference]
tags: [TMLE, targeted-learning, doubly-robust, SuperLearner, causal-inference, propensity-score, Poisson, insurance-tmle, python]
description: "Double robustness means your causal estimate remains consistent if either the outcome model or the propensity score is correctly specified — not both. TMLE delivers that guarantee with semiparametric efficiency. insurance-tmle is the first Python implementation with Poisson TMLE for insurance claim frequency and a proper SuperLearner ensemble."
post_number: 75
---

The standard causal inference workflow in pricing goes like this. Fit a propensity score model. Re-weight observations. Estimate the average treatment effect. Report a standard error that pretends the propensity model is the truth.

The problem: propensity score estimators are not doubly robust. If your propensity model is misspecified — and it will be — your ATE estimate is biased. The bias does not shrink with sample size. It compounds.

[`insurance-tmle`](https://github.com/burning-cost/insurance-tmle) implements Targeted Maximum Likelihood Estimation, the semiparametrically efficient doubly robust estimator. TMLE is consistent if *either* the outcome model *or* the propensity model is correctly specified. You need to get at least one right, not both. That is a substantially weaker requirement than what IPW asks for.

```bash
uv add insurance-tmle
```

## What TMLE does that IPW doesn't

The TMLE algorithm has two stages. First, estimate an initial outcome model — any flexible learner. Second, *target* that estimate with a one-step correction that uses the propensity score to remove confounding bias. The targeting step solves an efficient influence function estimating equation.

The result is an estimator that:
- Is doubly robust: consistent if either model is correct
- Achieves the semiparametric efficiency bound when both models are consistent
- Has honest standard errors via the efficient influence function (EIF) — no bootstrap needed
- Handles positivity violations gracefully through stabilised weights

## Poisson TMLE for claim frequency

The insurance application is claim frequency. We want to estimate the causal effect of a treatment (e.g., a pricing change, a telematics intervention) on claim counts, adjusting for confounders.

PoissonTMLE handles exposure offsets natively — the log(exposure) offset that appears in every insurance frequency GLM. This is not available in any existing Python TMLE implementation.

```python
from insurance_tmle import PoissonTMLE, SuperLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# SuperLearner with cross-validated meta-learning
sl_outcome = SuperLearner(
    learners=[GradientBoostingRegressor(), ...],
    meta_learner="nnls",
    cv_folds=5
)
sl_propensity = SuperLearner(
    learners=[GradientBoostingClassifier(), ...],
    meta_learner="logistic",
    cv_folds=5
)

tmle = PoissonTMLE(
    outcome_learner=sl_outcome,
    propensity_learner=sl_propensity,
    exposure_col="earned_exposure"
)
tmle.fit(X, A, Y, exposure=df["earned_exposure"])

result = tmle.estimate_ate()
print(result.ate)           # Causal rate ratio
print(result.ci_95)         # EIF-based confidence interval
print(result.eif_variance)  # Efficient influence function variance
```

## SuperLearner: the ensemble that makes TMLE work

TMLE's double robustness guarantee is strongest when you use a rich, flexible learner for both models. SuperLearner (van der Laan et al. 2007) is the correct choice: it cross-validates multiple learners and fits a non-negative least squares meta-learner to combine them.

```python
from insurance_tmle import SuperLearner
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

sl = SuperLearner(
    learners=[
        GradientBoostingRegressor(n_estimators=200),
        RandomForestRegressor(n_estimators=100),
        Ridge(alpha=1.0),
    ],
    meta_learner="nnls",
    cv_folds=10
)
```

The non-negative least squares constraint is important: it prevents negative weights that can produce extrapolation artefacts.

## CV-TMLE for small samples

Standard TMLE has first-order bias when the sample size is small relative to the complexity of the learners. CV-TMLE (Zheng & van der Laan 2011) cross-fits the targeting step to remove that bias.

```python
from insurance_tmle import CVTMLE

cv_tmle = CVTMLE(
    outcome_learner=sl_outcome,
    propensity_learner=sl_propensity,
    n_folds=10,
    exposure_col="earned_exposure"
)
cv_tmle.fit(X, A, Y)
result = cv_tmle.estimate_ate()
```

For thin-segment analyses — new business cohort studies, telematics intervention studies — CV-TMLE is the correct default.

## Subgroup analysis with StratumTMLE

```python
from insurance_tmle import StratumTMLE

stratum_tmle = StratumTMLE(
    strata_col="vehicle_group",
    outcome_learner=sl_outcome,
    propensity_learner=sl_propensity
)
stratum_tmle.fit(X, A, Y)

for stratum, result in stratum_tmle.stratum_results_.items():
    print(f"{stratum}: ATE={result.ate:.3f}, 95% CI {result.ci_95}")
```

## HTML report output

```python
from insurance_tmle import TMLEReport

report = TMLEReport(tmle_result=result, cv_tmle_result=cv_result)
report.save("tmle_causal_analysis.html")
```

The report includes: propensity score overlap diagnostics, efficient influence function distribution, one-step vs TMLE comparison, SuperLearner cross-validated risk, and covariate balance before/after.

## When to use TMLE over DML

[`insurance-causal`](https://github.com/burning-cost/insurance-causal) implements Double Machine Learning (DML). DML and TMLE are both doubly robust but differ in their approach:

- **DML** is easier to extend to heterogeneous treatment effects (CATE), integrates naturally with CausalForestDML, and is better suited to continuous treatments. Use DML when you need HTEs.
- **TMLE** has tighter finite-sample theory, produces efficient standard errors without bootstrap, and handles binary treatments more cleanly. Use TMLE when you need a single well-characterised ATE with valid confidence intervals.

For most pricing causal analyses — did this intervention reduce claims? — TMLE is the more principled choice.

---

`insurance-tmle` is available on [PyPI](https://pypi.org/project/insurance-tmle/) and [GitHub](https://github.com/burning-cost/insurance-tmle). The Databricks notebook in `notebooks/` runs the full workflow on synthetic motor data.

Related: [insurance-causal](https://burning-cost.github.io/2026/02/25/causal-inference-for-insurance-pricing/) for DML, [insurance-causal-policy](https://burning-cost.github.io/2026/03/19/your-rate-change-didnt-prove-anything/) for synthetic DID, [insurance-rdd](https://burning-cost.github.io/2026/03/11/insurance-rdd/) for regression discontinuity.
