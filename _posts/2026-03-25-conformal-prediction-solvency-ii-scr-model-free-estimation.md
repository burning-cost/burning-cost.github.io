---
layout: post
title: "Conformal Prediction for Solvency II Capital Requirements: Model-Free SCR Estimation and Governance Validation"
date: 2026-03-25
categories: [capital, solvency-ii, techniques]
tags: [conformal-prediction, solvency-ii, scr, capital-modelling, insurance-conformal, distribution-free, actuarial, regulatory, uk, pipeline, solvency-capital-range]
description: "How solvency_capital_range() produces model-free 99.5% SCR bounds, how SCRReport produces the coverage validation table for regulatory submission, and why interval_width is the number you should be showing your CRO."
---

Solvency II Article 101 requires you to hold capital at the 99.5th percentile of the distribution of basic own funds over a one-year period. Every internal model achieves this by choosing a distributional form — lognormal, Pareto, Burr, Gaussian copula — and estimating its parameters from data. The regulator approves the model. The model produces a number. The number is the SCR.

What this process does not tell you is how sensitive the SCR is to the distributional choice, or how much of the stated 99.5% coverage is actually delivered on held-out data versus assumed by construction. Those are questions about model risk, and they are not answered by fitting more parameters.

Liang Hong's 2025 paper (arXiv:2503.03659) makes a precise version of this argument: parametric capital models achieve coverage asymptotically, in the limit as sample size grows. Conformal prediction achieves coverage at the actual sample size you have. For a regulatory requirement that applies today, on a finite book of business, the distinction is not academic.

The `solvency_capital_range()` function in [`insurance-conformal`](/insurance-conformal/) implements this as a lightweight functional interface. This post explains what it does, when to reach for it instead of `SCRReport`, and — more importantly — what the `interval_width` field is telling you that your parametric model cannot.

---

## The distributional choice problem

Take a UK commercial property book. You have 800 large loss observations from the past eight years. You fit a lognormal to the severity and read off the 99.5th percentile. You fit a Pareto to the same data and read off the 99.5th percentile again. The two estimates differ by 20–35% on a representative portfolio — not because your data is thin (800 observations is not nothing) but because the lognormal and Pareto tails diverge sharply at 1-in-200 and you have almost no data there to distinguish them.

The PRA's model validation expectations require you to demonstrate that your internal model's stated coverage is actually achieved on held-out data. For a parametric model, demonstrating this at the 99.5th percentile requires either an enormous hold-out set or a long time series — neither of which most UK commercial lines teams have. The validation is therefore largely qualitative, backed by expert judgement about whether the distributional choice is reasonable.

Conformal prediction does not make the tail uncertainty go away. What it does is make the uncertainty explicit and measurable. The prediction interval is wide when the tail is uncertain; it is narrow when the calibration data strongly constrains the 99.5th percentile. The `interval_width` field in `SolvencyCapitalRange` is that uncertainty, in pounds, per risk.

---

## What `solvency_capital_range()` gives you

Split conformal prediction works as follows. You have a fitted regression model — any model: GLM, GBM, neural network — that produces a point forecast `mu(x)` for each risk. You have a calibration set of risks with observed outcomes, not used for training. You compute a nonconformity score for each calibration observation — for a Tweedie model with power `p`, the Pearson-weighted residual `(y - mu(x)) / mu(x)^(p/2)` is the natural choice. You take the empirical `(1 - alpha)(1 + 1/n)`-quantile of those scores and call it `q_hat`.

Your 99.5% upper bound for a new risk is `mu(x) + q_hat * mu(x)^(p/2)`.

The coverage guarantee is:

```
P(Y_new <= upper_bound) >= 1 - alpha
```

This holds for any exchangeable data generating process. No lognormal. No Pareto. No copula. The coverage comes from the rank structure of the nonconformity scores, not from any assumption about the tail shape.

`solvency_capital_range()` wraps this in a dataclass:

```python
from insurance_conformal import InsuranceConformalPredictor, solvency_capital_range

# Fit and calibrate any conformal predictor
cp = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,  # 1.5 is typical for aggregate property losses
)
cp.calibrate(X_cal, y_cal)

# One call: per-risk SCR bounds at 99.5%
result = solvency_capital_range(cp, X_test, alpha=0.005)

print(result)
# SolvencyCapitalRange(n_risks=500, coverage_level=0.995,
#                      total_scr=8423.17, mean_interval_width=14.82)

print(result.scr_estimate[:3])    # per-risk SCR component: max(0, upper - E[Y])
print(result.upper_bound[:3])     # the 99.5% tail bound — your SCR-relevant figure
print(result.interval_width[:3])  # tail uncertainty, in loss units, per risk
```

The `scr_estimate` field is `max(0, upper_bound - expected_loss)` per risk: the excess of the conformal tail bound over the point forecast, which is the economic definition of required capital for an individual risk. The `total_scr` is the sum across all risks — provided for convenience, but see the limitations section below.

---

## A worked example: conformal vs parametric VaR

Here is a comparison on a synthetic commercial property portfolio. We run both a lognormal parametric estimate and the conformal bound, then compare the two on a test set.

```python
import numpy as np
from scipy import stats
from insurance_conformal import InsuranceConformalPredictor, solvency_capital_range

rng = np.random.default_rng(2026)
n = 2_000

# Synthetic portfolio: log-sum-insured, construction score, occupancy score
X = rng.normal(size=(n, 3))
beta = np.array([0.8, -0.4, 0.2])
mu_true = np.exp(X @ beta)

# True DGP: heavy-tailed compound — Poisson frequency, lognormal severity
# (This is what the parametric model assumes it knows; conformal does not.)
freq = rng.poisson(0.3, size=n)
sev = rng.lognormal(mean=np.log(mu_true), sigma=1.2, size=n)
y = freq * sev

X_train, X_cal, X_test = X[:800], X[800:1400], X[1400:]
y_train, y_cal, y_test = y[:800], y[800:1400], y[1400:]
mu_train, mu_cal, mu_test = mu_true[:800], mu_true[800:1400], mu_true[1400:]

# --- Parametric approach: lognormal VaR at 99.5% ---
sigma_mle = np.std(np.log(y_cal[y_cal > 0] + 1e-6))
log_mu_cal = np.log(mu_cal + 1e-6)
lognormal_99_5 = np.exp(log_mu_cal.mean() + sigma_mle * stats.norm.ppf(0.995))
parametric_scr = np.maximum(0, lognormal_99_5 - mu_test.mean())
print(f"Parametric lognormal SCR (per-risk mean): {parametric_scr:.2f}")

# --- Conformal approach ---
class MeanModel:
    def predict(self, X):
        return np.exp(np.asarray(X) @ beta)

cp = InsuranceConformalPredictor(
    model=MeanModel(),
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

result = solvency_capital_range(cp, X_test, alpha=0.005)

print(f"Conformal SCR (per-risk mean): {result.scr_estimate.mean():.2f}")
print(f"Conformal mean interval width: {result.mean_interval_width:.2f}")

# Empirical coverage check — the number that matters
empirical_coverage = np.mean(y_test <= result.upper_bound)
print(f"Empirical 99.5% coverage: {empirical_coverage:.3f}")
# Should be >= 0.995 by the conformal guarantee
```

The lognormal estimate will achieve 99.5% coverage on average, across many datasets from the same DGP. But on any given dataset, you do not know whether it is achieving the stated coverage or not, and you cannot test it at the 99.5th percentile without a very large hold-out set. The conformal coverage guarantee is right by construction — the empirical coverage check will confirm at least 99.5% coverage on the test set — and `interval_width` tells you how much uncertainty you are carrying in the tail bound itself.

---

## When to use `solvency_capital_range()` vs `SCRReport`

These are two different tools for different contexts. The distinction matters.

**Use `solvency_capital_range()`** when you need SCR estimates as inputs to a larger pipeline: a reserving system that uses per-risk tail bounds as inputs, a reinsurance optimisation that needs to price excess-of-loss layers from the upper bound distribution, or a stress-testing loop that reruns SCR estimates under distributional shift scenarios. The function returns a dataclass — a simple Python object — that you can pass directly into downstream code.

```python
# Embed SCR bounds inside a reinsurance pricing loop
def price_xl_layer(predictor, X, attachment, limit):
    result = solvency_capital_range(predictor, X, alpha=0.005)
    in_layer = np.clip(result.upper_bound - attachment, 0, limit)
    return in_layer.mean()  # expected layer loss at the 99.5% scenario
```

**Use `SCRReport`** when you are producing a regulatory submission and need a coverage validation table demonstrating that empirical coverage meets the 99.5% requirement across multiple alpha levels, formatted for a model governance committee.

`SCRReport` wraps any fitted and calibrated conformal predictor and adds the SCR-specific output columns:

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport

cp = InsuranceConformalPredictor(
    model=fitted_model,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

scr = SCRReport(predictor=cp, policy_ids=policy_ids_test)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
# Returns a Polars DataFrame:
# policy_id | expected_loss | upper_bound_99.5pct | scr_component | coverage_level

agg = scr.aggregate_scr()
# Returns: total_expected_loss, total_scr, scr_ratio, n_risks, alpha
print(f"Portfolio SCR ratio: {agg['scr_ratio']:.2f}x")
```

For regulatory conservatism, add a buffer over the bare Solvency II requirement:

```python
# 99.7% bound instead of 99.5% — requires >= 333 calibration observations
scr_conservative = scr.solvency_capital_requirement(X_test, alpha=0.003)
```

The coverage validation table is the output that matters for a model governance committee or regulator. Run it on a held-out set not used for training or calibration:

```python
val_table = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

```
## SCR Coverage Validation Report

| Alpha | Target | Empirical | N Covered | N Total | Shortfall | Meets Req |
|-------|--------|-----------|-----------|---------|-----------|-----------|
| 0.005 | 99.5%  | 99.600%   | 498       | 500     | 0.000%    | Yes       |
| 0.010 | 99.0%  | 99.200%   | 496       | 500     | 0.000%    | Yes       |
| 0.050 | 95.0%  | 95.400%   | 477       | 500     | 0.000%    | Yes       |
| 0.100 | 90.0%  | 90.600%   | 453       | 500     | 0.000%    | Yes       |
| 0.200 | 80.0%  | 80.800%   | 404       | 500     | 0.000%    | Yes       |
```

The `meets_requirement` column uses a 2pp tolerance reflecting finite-sample variation: with 500 test observations at alpha = 0.005, you expect 2–3 exceedances and coverage will bounce around 99.5% ± 0.5pp by chance alone. If `coverage_shortfall` is non-zero at alpha = 0.005, your calibration set is too small or your exchangeability assumption is violated.

`to_markdown()` formats the full table for inclusion in model governance documentation. The calibration set size guideline for 99.5%: at least 200 observations (strictly `ceil(0.995 * 201) = 200`), though 2,000 produces materially tighter bounds. For 99.9% you need at least 1,000.

Both call the same underlying conformal machinery. The choice is about what you are doing with the output.

---

## Exposure weighting

Most capital modelling workflows need to handle policies with different exposures — different policy periods, different vehicle-years, different years of cover. `solvency_capital_range()` handles this via the `exposure` parameter:

```python
# exposure: array of years on cover, e.g. 0.5 for mid-term policies
result = solvency_capital_range(
    cp, X_test,
    alpha=0.005,
    exposure=years_on_cover,
)
# All bounds are now exposure-scaled:
# upper_bound[i] = raw_upper_bound[i] * years_on_cover[i]
# total_scr = sum of exposure-scaled scr_estimate
```

The scaling applies elementwise to all bounds, so `total_scr` gives you the correct portfolio aggregate under uniform correlation (see limitations below). Without exposure weighting, you are implicitly assuming every policy has one year of cover, which is wrong for any portfolio with mid-term adjustments or cancellations.

---

## What `interval_width` is telling you

This is the field that most implementations leave out, and it is the one we think matters most for capital modelling.

`interval_width` is `upper_bound - lower_bound` per risk: the width of the conformal prediction interval in loss units. It measures how constrained the 99.5% tail estimate is by the available calibration data. A narrow interval on a large upper bound means the tail is well-determined by the calibration sample. A wide interval means there is genuine uncertainty about where the 99.5th percentile sits for that risk.

A parametric model gives you a single point estimate of the VaR. It does not tell you whether that estimate is tight or uncertain, because the uncertainty depends on how much tail data you have — and tail data is exactly what parametric models abstract away from. Conformal prediction makes the uncertainty explicit.

In practice:

```python
# Flag risks where SCR uncertainty is material
uncertain_risks = result.interval_width > 2 * result.scr_estimate
print(f"{uncertain_risks.mean():.1%} of risks have wide SCR intervals")
# These are the risks where distributional model choice matters most,
# and where additional tail data would most improve the capital estimate.
```

A risk with `interval_width > 2 * scr_estimate` has more uncertainty in the tail bound than the SCR component itself — the SCR estimate could easily be halved or doubled by the tail uncertainty alone. That is a number a CRO should see before approving a capital submission.

---

## Using CQR for heteroscedastic portfolios

For portfolios where severity varies substantially across risk types — commercial property with a mix of small and large sum-insured — the standard conformal predictor produces intervals that are too wide for small risks and too narrow for large ones. This is exactly the setting where Conformalized Quantile Regression (CQR) outperforms standard conformal.

`solvency_capital_range()` accepts any predictor implementing `predict_interval(X, alpha)`, including CQR:

```python
from insurance_conformal import ConformalisedQuantileRegression, solvency_capital_range

# CQR requires separate quantile models at a lower quantile (e.g. 0.1) and
# upper quantile (e.g. 0.9) — the conformalization step then corrects to 99.5%
cqr = ConformalisedQuantileRegression(
    model_lo=fitted_lo_quantile_model,
    model_hi=fitted_hi_quantile_model,
)
cqr.calibrate(X_cal, y_cal)

result = solvency_capital_range(cqr, X_test, alpha=0.005)
# result.upper_bound carries the 99.5% coverage guarantee
# result.interval_width will now vary across risks in proportion to their
# heteroscedasticity, rather than scaling uniformly with expected loss
```

The coverage guarantee is identical: at least 99.5% of risks will have their actual loss fall below `upper_bound`. The difference is efficiency: CQR produces tighter bounds on risks where the tail is more predictable from the features, and wider bounds on genuinely uncertain risks. See [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) for the full CQR workflow.

---

## When to use conformal SCR over parametric approaches

Use conformal SCR bounds when:

- You want a robustness check on your internal model that requires no additional distributional assumptions.
- You are entering a new line of business where you do not yet have sufficient data to justify a parametric tail choice, but you have access to analogous historical data for calibration.
- Your model governance process requires a quantitative demonstration that coverage claims are borne out on held-out data — the `coverage_validation_table()` output is exactly that.
- You have a book where the tail behaviour is genuinely uncertain: commercial property with heterogeneous construction types where neither lognormal nor Pareto provides a clean fit to large loss data.

Do not use conformal SCR bounds when:

- Your calibration set has fewer than 500 observations. Below that the 99.5% bound is determined by too few order statistics and will be unreliably wide.
- You have strong, well-validated prior knowledge of the tail distribution from a long run of historical data. Parametric models exploit this information efficiently; conformal prediction does not. If you genuinely know the tail is Pareto with shape 1.5 because you have 20 years of data confirming it, use the parametric model.
- You need a portfolio-level SCR that incorporates correlation structure. This library gives you marginal per-risk bounds.

The right use is as a distribution-free lower bound on what your capital requirement must be, plus a validation tool to test whether your parametric internal model's stated coverage is actually achieved on real data.


## Limitations

**Per-risk, not portfolio.** The 99.5% guarantee is per-risk and marginal. When you sum `scr_estimate` across risks to get `total_scr`, you are implicitly assuming perfect correlation — the worst case. For portfolio-level SCR that incorporates diversification, you still need a dependence model. Conformal prediction provides the marginal tail distributions; the aggregation is your problem.

**Exchangeability.** The coverage guarantee requires the calibration risks to be exchangeable with the risks you are scoring — drawn from the same distribution. If your calibration set is 2019–2022 and your in-force book reflects 2025 underwriting terms and exposure patterns, the exchangeability assumption is at risk. Check it explicitly: score the calibration set itself and verify that empirical coverage is at least 99.5%.

**Calibration set size at the tail.** The 99.5% bound is the `ceil(0.995 * (n_cal + 1))`-th order statistic of the nonconformity scores. With 200 calibration observations, the bound is determined by a single order statistic and will be unreliably wide. With 2,000, it is materially better constrained. For the 99.5% level, we recommend at least 500 calibration observations as a practical floor; 1,000 is more comfortable.

**Marginal, not conditional, coverage.** The guarantee is 99.5% on average across all risks, not 99.5% for every risk subgroup. A predictor that covers 99.8% of small commercial risks and 99.1% of large commercial risks has technically satisfied the marginal guarantee while under-covering the tail where it matters. Use `CoverageDiagnostics.coverage_by_decile()` from the same library to test conditional coverage by expected loss decile.

**Regulatory status.** EIOPA has not issued guidance on conformal methods as of March 2026. Present conformal SCR bounds to your regulator as a model-free validation cross-check, alongside your parametric internal model — not as an alternative to it.

---

## The practical workflow

For an internal model team, the most direct use of `solvency_capital_range()` is as a distributional assumption-free lower bound on the SCR. If your parametric model produces a lower SCR estimate than the conformal bound, that is a red flag: your distributional assumptions are producing a 99.5th percentile that is not actually achieved on held-out data. If the parametric SCR is higher, you have evidence that the distributional assumption is conservative, which may be intentional.

The `interval_width` comparison tells you something else: where on the book the distributional choice matters most. Wide conformal intervals on large commercial risks mean those risks have uncertain tails, and the lognormal-vs-Pareto choice is consequential there. Narrow conformal intervals on personal lines mean the tail is well-determined regardless of distributional form, and the parametric choice matters less.

That is a more useful output than a single SCR number, because it tells you where to focus model development and where the regulatory risk from distributional choice is concentrated.

```python
import polars as pl

# Combine conformal bounds with parametric estimates for comparison
comparison = pl.DataFrame({
    "expected_loss": result.upper_bound - result.scr_estimate,  # approx E[Y]
    "conformal_upper_99_5": result.upper_bound,
    "conformal_scr": result.scr_estimate,
    "interval_width": result.interval_width,
    "parametric_scr": parametric_scr_estimates,  # from your existing model
}).with_columns([
    (pl.col("parametric_scr") < pl.col("conformal_scr")).alias("parametric_below_conformal"),
    (pl.col("interval_width") / pl.col("conformal_scr").clip(lower_bound=1e-6)).alias("relative_uncertainty"),
])

# Risks where parametric model is below the conformal lower bound
flagged = comparison.filter(pl.col("parametric_below_conformal"))
print(f"{len(flagged)} risks where parametric SCR < conformal SCR")
print(f"Mean relative tail uncertainty: {comparison['relative_uncertainty'].mean():.2f}x")
```

The flagged risks are where your internal model is most exposed to distributional assumption risk. That is the output that belongs in a model validation report.

---

```bash
uv add insurance-conformal
```

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

- Hong (2025), "Conformal prediction of future insurance claims in the regression problem." [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — CQR and heteroscedastic interval construction
- [Which Uncertainty Quantification Method?](/2026/03/25/which-uncertainty-quantification-method-decision-framework/) — decision flowchart for choosing between conformal, GAMLSS, and distributional GBMs
