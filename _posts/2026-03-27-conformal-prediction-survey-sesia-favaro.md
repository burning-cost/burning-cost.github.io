---
layout: post
title: "What Conformal Prediction Actually Guarantees (And What It Doesn't)"
date: 2026-03-27
categories: [pricing, methodology]
tags: [conformal-prediction, prediction-intervals, solvency-ii, reserve-uncertainty, premium-adequacy, regulatory-capital, python]
description: "Sesia & Favaro's March 2026 survey of conformal prediction is the clearest account yet of what finite-sample distribution-free guarantees mean — and why the marginal/conditional distinction is the thing pricing teams should be thinking about."
---

A new survey paper landed on arXiv on 25 March 2026: "Elements of Conformal Prediction for Statisticians" by Matteo Sesia and Stefano Favaro ([arXiv:2603.23923](https://arxiv.org/abs/2603.23923)). It is explicitly a pedagogical overview rather than a methods paper, aimed at statisticians trained in classical inference who want to understand where conformal prediction sits relative to what they already know. That framing is exactly right for pricing actuaries.

We have been running [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — a library that puts conformal prediction intervals on Tweedie and Poisson pricing models — for long enough to know which misunderstandings cause problems in practice. The Sesia-Favaro paper resolves the most persistent one.

---

## The guarantee is real, but you should know what it covers

Conformal prediction makes one guarantee: for any pre-fitted model, any distribution over (X, Y), and any alpha between 0 and 1:

```
P(Y_test ∈ [lower, upper]) ≥ 1 − alpha
```

This holds in finite samples, requires no parametric assumptions on the error distribution, and requires only that the calibration and test data are exchangeable — roughly, that they come from the same process in no particular order. For insurance pricing, "train on years 1–4, calibrate on year 5, predict on year 6" satisfies this if the portfolio composition is not drifting dramatically.

No bootstrap. No normality assumption. No large-sample approximation. The guarantee is exact at the sample sizes you actually have.

The Sesia-Favaro paper makes this precise in a way that matters. The guarantee is **marginal** — it holds on average over the test distribution, not conditionally on the specific characteristics of each policy. A 90% interval covers 90% of all predictions. It does not necessarily cover 90% of high-value commercial property risks considered alone.

This distinction has direct implications for how you use conformal intervals in reporting.

---

## Why the marginal/conditional gap matters for pricing

A GLM produces a conditional mean prediction: the expected loss given the features of that risk. When we wrap it in a conformal predictor and call `predict_interval(X_test, alpha=0.10)`, the resulting intervals have a marginal 90% coverage guarantee — not a conditional one.

In `insurance-conformal`, we surface this directly through `coverage_by_decile()`:

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=fitted_catboost_tweedie,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)

# Check whether coverage is uniform across the predicted value distribution
cp.coverage_by_decile(X_test, y_test, alpha=0.10)
```

If the top predicted-value decile shows 72% empirical coverage instead of 90%, the marginal guarantee is satisfied but the interval is understating uncertainty for your most expensive risks. That is the failure mode to diagnose, not to dismiss.

The fix is the non-conformity score. Our default — `pearson_weighted`, using the Pearson residual scaled by `yhat^(p/2)` for Tweedie power `p` — corrects for heteroscedasticity in Tweedie/Poisson models and produces roughly uniform coverage across deciles. Switching from `raw` (unscaled absolute residual) to `pearson_weighted` narrows intervals by around 30% while maintaining the coverage guarantee: the heteroscedasticity correction eliminates unnecessary width for low-risk policies without shortchanging high-risk ones.

The Sesia-Favaro paper treats the conditional/marginal distinction formally. They describe why conditional coverage is harder — it is not achievable distribution-free without restricting the hypothesis class — and what the conditional alternatives (locally weighted conformal, conformalised quantile regression) cost in terms of assumptions. We implement both in `insurance-conformal`: `LocallyWeightedConformal` for adaptive-width intervals and `ConformalisedQuantileRegression` (split CQR, Romano et al. 2019) for heteroscedastic intervals from a pair of quantile models.

---

## Three applications where the guarantees matter

**Reserve uncertainty.** A conformal interval on a Tweedie claims regression gives you a distribution-free bound on the individual risk's expected cost range. Aggregate over the book and you have an interval on total incurred cost that you can report to the reserve committee without invoking Mack assumptions or parametric tail models. The interval is conservative — it will be wider than a well-specified lognormal — but it does not fail if your severity distribution has a heavier tail than you thought.

**Premium adequacy testing.** The `PremiumSufficiencyController` in `insurance_conformal.risk` applies conformal risk control (Angelopoulos et al., ICLR 2024) directly to underpricing risk. Rather than coverage probability, it controls expected loss: find the smallest loading factor lambda such that expected shortfall from underpriced policies stays below alpha% of premium income. Calibrate on one policy year's claims, apply to next year's book. No distributional assumption, finite-sample valid.

```python
from insurance_conformal.risk import PremiumSufficiencyController

psc = PremiumSufficiencyController(alpha=0.05, B=5.0)
psc.calibrate(y_cal, premium_cal)
result = psc.predict(premium_new)
# result["upper_bound"] = risk-controlled loading per policy
```

**Regulatory capital.** `SCRReport` wraps any calibrated conformal predictor and produces per-risk SCR upper bounds at `alpha=0.005` — the 99.5th percentile threshold Solvency II requires. The finite-sample validity is the argument for using these outputs as supplementary validation of internal model SCR components: unlike a parametric tail extrapolation, the conformal bound does not assume the tail behaves like a particular distribution.

```python
from insurance_conformal import InsuranceConformalPredictor, SCRReport

cp = InsuranceConformalPredictor(model=fitted_model)
cp.calibrate(X_cal, y_cal)
scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
print(scr.to_markdown())
```

EIOPA has not issued formal guidance on conformal methods as at March 2026. Present these as supplementary technical validation, not primary SCR figures.

---

## The exchangeability assumption is the thing to audit

The Sesia-Favaro survey is clear about where conformal prediction breaks. Marginal coverage holds under exchangeability. Temporal data is not exchangeable by default — year 6 motor claims are not drawn from the same distribution as year 1 motor claims, especially post-Ogden rate change or after a catastrophe event.

For time-series applications, `insurance-conformal` includes `RetroAdj` (Jun & Ohn 2025), which uses jackknife+ intervals with retrospective residual corrections and recovers from abrupt distribution shifts within 1–3 steps. The `RetroAdj` module was built specifically for the UK motor inflation scenario: a book that sees +30% large-loss inflation mid-year needs an online conformal method that adapts quickly, not one that takes 200 steps to recover.

The Sesia-Favaro paper covers the exchangeability requirement carefully, with a section on what breaks coverage and the conditions under which asymptotic results hold when exchangeability fails. Worth reading before applying split conformal to renewal pricing, where year-on-year portfolio selection effects can induce dependence.

---

## What the survey is and is not

This is a pedagogical review. It introduces no new algorithms. It does not address insurance-specific score modifications — the exposure-weighted Pearson residual that `insurance-conformal` uses is not in scope. Its value is precision: a clean account of what the finite-sample guarantee covers, what the split/full/cross-conformal variants trade off against each other, and how conformal risk control extends the basic coverage result.

The classification of conformal variants is particularly useful. Split conformal (which `insurance-conformal` uses throughout) trains once on a calibration set and gives the guarantee at the cost of using some data for calibration rather than training. Full conformal refits on each test point — correct for small models, untenable for a CatBoost that takes several hours to train. Cross-conformal amortises the calibration cost across folds. Sesia-Favaro explains the statistical tradeoffs clearly; the choice for insurance models is almost always split conformal on practical grounds.

For anyone on a pricing or capital modelling team who has encountered conformal prediction and wants a rigorous account of what it does — rather than a tutorial that stops at "run MAPIE and you get intervals" — this is the paper to read.

---

**Reference:** Sesia, M. and Favaro, S. (2026). Elements of Conformal Prediction for Statisticians. arXiv:2603.23923.

**Library:** [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — distribution-free prediction intervals for Tweedie/Poisson pricing models, with SCR reporting, premium sufficiency control, and online conformal for distribution shift.
