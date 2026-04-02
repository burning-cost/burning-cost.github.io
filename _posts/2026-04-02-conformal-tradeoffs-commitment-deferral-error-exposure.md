---
layout: post
title: "Beyond Coverage: Commitment, Deferral, and Error Exposure in Conformal Pricing"
date: 2026-04-02
author: Burning Cost
categories: [conformal-prediction, model-deployment]
tags: [conformal prediction, automated underwriting, model monitoring, insurance-conformal]
description: "Coverage-matched conformal rules can have wildly different deployment profiles. Three operational metrics that matter for pricing deployment."
---

You have calibrated a conformal prediction model for automated motor underwriting. You have hit 90% coverage. You now face a choice between two threshold configurations, both with identical 90% coverage on the calibration set. Rule A commits to a quote on 85% of submissions with a 4% error exposure — 4% of auto-issued quotes are mispriced outside your tolerance. Rule B commits on 45% of submissions with 1% error exposure. Both pass the coverage test. Neither tells you anything about the other.

This is the gap Petrus Zwart's paper (arXiv:2602.18045, February 2026) addresses cleanly: coverage is a single number describing a single property of a deployed conformal system, and that property is the least interesting one for an insurer trying to decide what to deploy.

---

## Three numbers that actually matter

The three operational metrics Zwart formalises map directly onto the decisions a pricing or underwriting team makes when deploying an automated quoting rule.

**Commitment frequency** — the fraction of incoming submissions that receive a definitive auto-quote — is what your operations team calls the STP (straight-through processing) rate. Commercial pressure pushes this upward. A Lloyd's syndicate running a delegated authority arrangement for a niche commercial product might have 40% STP and regard that as acceptable. A UK personal lines motor book at 60% STP has a problem: the referral queue is unworkable.

**Deferral rate** — the fraction of submissions with no auto-quote, referred to a human underwriter — is the complement of STP for a well-designed rule. A UK commercial lines referral costs roughly £50–100 in underwriter time when you account for the full cost of interruption, documentation, and response lag. At scale, a 10 percentage point increase in deferral rate on a book writing 20,000 policies per year is approximately £100,000–£200,000 in friction costs. That is not an abstraction; it shows up in your expense ratio.

**Error exposure** — the fraction of committed (auto-issued) quotes that fall outside your pricing tolerance — is the adverse selection rate on the book you actually write. At 85% STP and 4% error exposure, roughly 3.4% of your written policies are mispriced. On a motor book with a target combined ratio of 96%, 3.4% mispricing on the auto-issued segment moves the needle materially — particularly if the errors concentrate in the high-severity tail, which they usually do, because the model scores that trigger auto-referral are correlated with prediction difficulty, and prediction difficulty is correlated with tail weight.

The fundamental point, which Zwart formalises through what he calls the Region-Class Label Table, is that these three metrics are not independently tunable. Within a threshold regime — holding fixed which inputs get committed, deferred, or hedged — changing thresholds reallocates probability mass across outcome categories. You cannot increase STP, cut deferral, and hold error exposure constant by tweaking lambda. The attainable set is constrained. Two practitioners with different cost structures should be choosing different points on the Pareto frontier, not both targeting 90% coverage.

---

## The small-book problem

The coverage guarantee from standard split conformal is a marginal average: P(Y in C(X)) >= 1-alpha, averaged over draws of the calibration set. This is fine when n is large enough that calibration-set variability is small. At n=300 — which describes every Lloyd's specialty syndicate, most MGA niche products, and a fair number of parametric weather books — it is not fine.

Zwart's simulation results are worth quoting directly. At n=300 calibration points and delta=0.10 (meaning we want at least 90% probability of achieving the coverage target), nominal split conformal has a **40% PAC violation rate** — it fails to achieve the target coverage 40% of the time, when you want that rate to be at most 10%. The DKWM correction (a standard small-sample fix) overcorrects: 0.04% violation rate, far too conservative, intervals too wide to be operationally useful.

SSBC (Small-Sample Beta Correction) hits the target: 9.6% violation rate at delta=0.10. It inverts the rank pivot — given a target (alpha*, delta), it selects the calibration quantile index that achieves calibration-conditional PAC coverage. In practice it uses less data than DKWM, so it does not pay the full width penalty of the conservative correction.

If you are deploying conformal pricing on any book with fewer than ~800 calibration policies, you should be using SSBC, not naive split conformal. The 40% PAC violation rate means you are frequently reporting a 90% coverage system that is actually running at 85% or worse on your specific calibration draw. FCA Consumer Duty aside, that is just a wrong number.

`insurance-conformal` does not yet include SSBC. It is approximately 20 lines of code — a quantile index selection using `scipy.stats.beta` — and we will add it as a utility function in v1.3.0. For now, if you are on a small book, the manual calculation is in the paper's Algorithm 1.

---

## SelectiveConformalRC covers most of this

The good news is that `SelectiveConformalRC` already handles the commitment/deferral semantics directly.

```python
from insurance_conformal.risk import SelectiveConformalRC

# xi: minimum fraction of submissions to auto-quote (STP floor)
# alpha: maximum tolerated risk on committed quotes
# run_grid_search: compute Pareto frontier over (STP, error_exposure)

scrc = SelectiveConformalRC(
    alpha=0.10,
    xi=0.80,
    run_grid_search=True,
)

scrc.calibrate(scores_cal, y_cal, lambda_1_grid, lambda_2_grid)

# Check where you landed: STP rate, error exposure, Pareto position
print(scrc.summary())
```

With `run_grid_search=True`, `SelectiveConformalRC` sweeps the threshold grid and returns the Pareto frontier as a DataFrame via `pareto_report()`:

```python
frontier = scrc.pareto_report()
# columns: lambda_1, lambda_2, selection_rate, conditional_risk, xi_lcb
# selection_rate = STP rate
# conditional_risk = error exposure on committed book
```

Every row on the frontier is a different (STP, error_exposure) combination at your target coverage level. Rule A and Rule B from the introduction are two rows on this table. The table does not tell you which to choose — that depends on your cost structure. But it makes the choice explicit, which is the precondition for making it deliberately rather than by accident.

`LCPModelSelector` (v1.2.0) gives you locally adaptive score selection before the selective controller. The composition is: `LCPModelSelector` finds the best non-conformity score per-prediction across your covariate space; `SelectiveConformalRC` controls commitment and error exposure on the resulting scores. One sets interval efficiency; the other sets deployment thresholds.

---

## Calibrate-and-audit

Zwart's two-stage workflow is worth adopting as a production pattern.

Stage one: fix thresholds on the calibration set. Freeze the region partition — which inputs get committed, deferred, hedged. Do not touch the thresholds after this point.

Stage two: audit on independent data. The key result is that once thresholds are frozen, the count of committed quotes in each outcome category on the audit set follows a Binomial distribution. This means you can do exact finite-sample inference on your operational KPIs — commitment rate, error exposure, deferral rate — without multiple comparison corrections, provided the metrics were pre-declared.

This matters for FCA Consumer Duty reporting. Exact Binomial inference on pre-declared metrics, with Clopper-Pearson confidence intervals, is a clean audit trail. It does not require asymptotic approximations. It does not have the multiple testing problems that arise when you check a dozen metrics and apply Bonferroni corrections. The statistical properties are unambiguous and documentable.

The `reporting` module in `insurance-conformal` has utilities for Clopper-Pearson intervals. The audit workflow — calibrate once, freeze, observe on held-out window — is the intended pattern for `SelectiveConformalRC`, though we have not formalised it as a named two-stage object yet. That is also a v1.3.0 candidate.

---

## What the paper does not cover

Two limitations are worth naming because they affect how directly you can apply this to UK pricing.

The paper's empirical work uses classification tasks (molecular toxicity prediction, Tox21 dataset). The region-class table is natural when Y is categorical — you are predicting class membership and the regions map cleanly to committed/deferred/hedged. For continuous pricing outputs (claim severity, burning cost), you need to discretise into loss bands before the machinery applies. That discretisation is not automatic, and the choice of bands affects the KPI calculations. The paper does not address this. We think the right approach is to discretise into quantile bands on the calibration set — but that is a modelling choice, not a free parameter, and it needs to be made deliberately.

Exposure weighting is not addressed. A committed quote on a £50,000 annual premium commercial risk deserves more weight in your error exposure calculation than a committed quote on a £200 personal lines motor risk. The KPI as defined — fraction of committed quotes that are mispriced — treats every committed quote equally. For a book with any exposure heterogeneity (which is every book), you should weight by premium or expected claim cost. The binomial inference structure is preserved under weighting; the weights just need to be fixed before calibration.

---

## Summary

Coverage is a necessary condition for a deployable conformal rule, not a sufficient one. Two rules with identical 90% coverage can have STP rates of 85% and 45%, error exposures of 4% and 1%, and those differences are commercially material. The metrics to track are commitment frequency (STP rate), deferral rate, and error exposure on the committed book.

For books with fewer than 800 calibration policies, use SSBC rather than naive split conformal. The 40% PAC violation rate on split conformal at n=300 is not a theoretical concern — it means your reported coverage number is frequently wrong.

`SelectiveConformalRC` with `run_grid_search=True` gives you the Pareto frontier over commitment and error exposure at your target coverage. That is the right tool for choosing a deployment threshold deliberately rather than arbitrarily.

The paper: Zwart, P.H. — "Conformal Tradeoffs: Operational Profiles Beyond Coverage" (arXiv:2602.18045, February 2026).

---

Related:
- [Locally Adaptive Score Selection for Conformal Intervals: insurance-conformal v1.2.0](/2026/04/02/lcp-model-selector-insurance-conformal-v120/) — the model selection layer that feeds into SelectiveConformalRC
- [Conditional Coverage and Conformal Prediction Model Selection: CVI and CC-Select](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — global score selection for conditional coverage quality
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the baseline split conformal implementation
