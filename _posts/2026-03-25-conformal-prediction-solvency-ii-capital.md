---
layout: post
title: "Conformal Prediction for Solvency II Capital"
date: 2026-03-25
categories: [capital, solvency-ii, techniques]
tags: [conformal-prediction, solvency-ii, scr, capital-modelling, insurance-conformal, distribution-free, actuarial, regulatory, uk]
description: "How SCRReport in insurance-conformal v0.6.2 produces distribution-free capital estimates at the 99.5% level required by Solvency II, without parametric tail assumptions."
---

Solvency II requires you to hold capital sufficient to cover a 1-in-200 year loss event — the 99.5th percentile of the aggregate loss distribution. Every internal model, every standard formula approximation, and every partial model achieves this by making distributional assumptions. You pick a tail. You fit parameters. You hope the tail you chose is the right one.

It usually is not, and the difference between a lognormal and a Pareto in the 99.5th percentile is not academic. On a commercial property book, the choice of severity tail shifts your individual risk SCR by 15–40% depending on the size of the risk. The EIOPA internal model approval process exists partly because this distribution uncertainty is real and consequential.

Conformal prediction offers a different route to the 99.5% bound. It does not require you to choose a distribution. Given a calibration set of historical losses, it produces a prediction upper bound that is guaranteed — finite-sample, for any underlying distribution — to contain the true loss at least 99.5% of the time. The guarantee comes from the data, not from an assumption about the tail.

Hong (arXiv:2503.03659) frames this explicitly within Solvency II: finite-sample validity, not asymptotic validity, is the right standard for regulatory capital modelling. A parametric model achieves coverage asymptotically — in the limit as sample size grows without bound — but regulators want you to hold capital today, not in a limiting sequence. Conformal prediction provides coverage on the actual sample you have.

---

## Why parametric tail estimation is genuinely hard

The standard actuarial approach to individual risk SCR runs roughly as follows. Fit a frequency-severity model: Poisson or negative binomial for frequency, lognormal or Pareto for severity. Combine via Monte Carlo. Read off the 99.5th percentile of the simulated total loss distribution. Apply correlations via copula or variance-covariance. Report your SCR.

Three sources of uncertainty are not reflected in the reported number.

**Distributional model risk.** The 99.5th percentile of a lognormal severity model and a Pareto severity model fitted to the same data are materially different. For UK commercial liability, the Pareto (1 + y/sigma)^(-alpha) tail is substantially heavier than the lognormal at the 1-in-200 level. Both models fit the bulk of the observed data well — the tail is thin by construction, and you have very few observations there. Choosing between them is a modelling judgement, not a statistical inference, and the choice can move your SCR materially.

**Parameter uncertainty.** Even once you have chosen a distribution, the parameters carry estimation uncertainty. A maximum likelihood estimate of the Pareto shape parameter on 200 large loss observations has substantial variance. Bootstrap confidence intervals on the 99.5th percentile quantile are wide. Reported SCRs rarely reflect this uncertainty.

**Conditional coverage failure.** Aggregate coverage at 99.5% does not imply 99.5% coverage for every risk subgroup. A model calibrated on the full portfolio may achieve correct average coverage while systematically under-covering high-value commercial risks and over-covering personal lines. For individual risk SCR components, conditional coverage by risk type is what matters, not aggregate coverage across all risks.

---

## What conformal prediction gives you

Split conformal prediction is disarmingly simple. You have a regression model that produces a point prediction mu(x) for each risk. You have a calibration set — risks not used to fit the model, with observed outcomes. You compute a nonconformity score for each calibration observation: typically a scaled residual of the form (y - mu(x)) / mu(x)^(p/2) for a Tweedie model with power p. You find the (1 - alpha)(1 + 1/n)-th empirical quantile of those calibration scores and call it q_hat.

Your prediction upper bound for a new risk with features x is then mu(x) + q_hat * mu(x)^(p/2).

The coverage guarantee is:

P(Y_new <= upper_bound) >= 1 - alpha

This holds for *any* joint distribution of (X, Y), provided the calibration observations and the new observation are exchangeable — roughly, drawn from the same distribution. It holds for finite n, not just in the limit. There is no lognormal, no Pareto, no Gumbel, no Gaussian copula. The guarantee comes entirely from the rank structure of the nonconformity scores.

For Solvency II, set alpha = 0.005 and you have a distribution-free upper bound at 99.5%.

The SCR component for a risk is max(0, upper_bound - expected_loss): the excess of the tail bound over the expected value. This is analogous to a parametric VaR estimate but without the distributional assumption, and it is conservative relative to parametric estimates when the calibration set is representative.

---

## The SCRReport API

`insurance-conformal` v0.6.2 implements this in `SCRReport`. It wraps any fitted and calibrated conformal predictor that implements `predict_interval(X, alpha)` and adds the SCR-specific output columns.

```bash
uv add insurance-conformal
```

The workflow is three steps: fit and calibrate a conformal predictor, construct `SCRReport`, run `solvency_capital_requirement()`.

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport

# Step 1: fit and calibrate a conformal predictor
# model is any sklearn-compatible model with a .predict() method
cp = InsuranceConformalPredictor(
    model=fitted_model,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal)

# Step 2: construct SCRReport, optionally with policy identifiers
scr = SCRReport(predictor=cp, policy_ids=policy_ids_test)

# Step 3: per-risk SCR bounds at 99.5%
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)
```

The output is a Polars DataFrame with five columns: `policy_id` (if provided), `expected_loss`, `upper_bound_99.5pct`, `scr_component`, and `coverage_level`. The `scr_component` column is `max(0, upper_bound - expected_loss)` per risk.

```python
# Aggregate to portfolio level
agg = scr.aggregate_scr()
# Returns: total_expected_loss, total_scr, scr_ratio, n_risks, alpha
print(f"Portfolio SCR ratio: {agg['scr_ratio']:.2f}x")
```

For regulatory submission documentation, `coverage_validation_table()` tests empirical coverage across multiple alpha levels on a held-out test set:

```python
val_table = scr.coverage_validation_table(X_test, y_test)
# Tests alphas [0.005, 0.01, 0.05, 0.10, 0.20] by default
# Returns: alpha, target_coverage, empirical_coverage, n_covered, n_total,
#          coverage_shortfall, meets_requirement

print(scr.to_markdown())
```

The `to_markdown()` output is a coverage validation report formatted for model governance documentation.

---

## A worked example

Here is a worked example using a synthetic commercial property portfolio. We use a Poisson frequency with log-linear structure over three risk features.

```python
import numpy as np
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport

rng = np.random.default_rng(2026)
n = 1_500
beta = np.array([0.6, -0.3, 0.15])

# Risk features: log-sum-insured, construction type score, occupancy score
X = rng.normal(size=(n, 3))
mu = np.exp(X @ beta)
y = rng.poisson(mu).astype(float)

# Split: train / calibrate / test
X_train, X_cal, X_test = X[:600], X[600:1000], X[1000:]
y_train, y_cal, y_test = y[:600], y[600:1000], y[1000:]

# Fit a Poisson GLM (or any model with .predict())
class PoissonModel:
    def __init__(self, beta):
        self.beta = beta
    def predict(self, X):
        return np.exp(np.asarray(X) @ self.beta)

model = PoissonModel(beta)  # using true parameters for this example

# Conformal predictor with Pearson-weighted nonconformity score
# pearson_weighted: (y - mu) / mu^(p/2), p=1.0 for Poisson
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    tweedie_power=1.0,
)
cp.calibrate(X_cal, y_cal)

# SCR bounds at 99.5%
scr = SCRReport(predictor=cp)
scr_bounds = scr.solvency_capital_requirement(X_test, alpha=0.005)

print(scr_bounds.head(5))
# expected_loss  upper_bound_99.5pct  scr_component  coverage_level
# 1.23           8.41                 7.18           0.995
# 0.87           6.22                 5.35           0.995
# 2.11           10.93                8.82           0.995
# ...

agg = scr.aggregate_scr()
print(f"Total expected loss:  {agg['total_expected_loss']:,.0f}")
print(f"Total SCR:            {agg['total_scr']:,.0f}")
print(f"SCR ratio:            {agg['scr_ratio']:.2f}x")
```

The calibration set size of 400 observations is sufficient to calibrate a 99.5% bound: you need at least `ceil((1 - alpha) * (n_cal + 1))` = 398 non-exceedances from 400 calibration observations. If you use alpha = 0.001 (99.9%) you need at least 1,000 calibration observations.

For regulatory conservatism, the module's docstring recommends alpha = 0.003 or alpha = 0.001 rather than exactly 0.005, to add a buffer over the bare Solvency II requirement:

```python
# Conservative: 99.7% bound instead of 99.5%
scr_conservative = scr.solvency_capital_requirement(X_test, alpha=0.003)
```

---

## Coverage validation for model governance

The coverage validation table is the part that matters for a regulator or model governance committee. Run it on a held-out set that was not used for training or calibration:

```python
val_table = scr.coverage_validation_table(X_test, y_test)
print(scr.to_markdown())
```

```
## SCR Coverage Validation Report

Coverage validation for conformal prediction intervals at multiple
alpha levels. The 'Meets Requirement' column uses a 2pp tolerance
for finite-sample variation.

| Alpha | Target | Empirical | N Covered | N Total | Shortfall | Meets Req |
|-------|--------|-----------|-----------|---------|-----------|-----------|
| 0.005 | 99.5%  | 99.600%   | 498       | 500     | 0.000%    | Yes       |
| 0.010 | 99.0%  | 99.200%   | 496       | 500     | 0.000%    | Yes       |
| 0.050 | 95.0%  | 95.400%   | 477       | 500     | 0.000%    | Yes       |
| 0.100 | 90.0%  | 90.600%   | 453       | 500     | 0.000%    | Yes       |
| 0.200 | 80.0%  | 80.800%   | 404       | 500     | 0.000%    | Yes       |
```

The key column for a regulator is `meets_requirement`. The 2pp tolerance reflects finite-sample variation: with 500 test observations at alpha = 0.005, you expect around 2–3 exceedances, and coverage will bounce around 99.5% ± 0.5pp by chance alone. Requiring exact coverage to 4 decimal places is not how finite-sample guarantees work.

The `coverage_shortfall` column shows where the guarantee is binding. If it is non-zero at alpha = 0.005, your calibration set is either too small or your exchangeability assumption is violated. The most common cause is distributional shift between the calibration and test periods.

---

## Limitations

**Exchangeability.** The coverage guarantee requires exchangeability between calibration and new observations. In practice this means the risks you calibrate on must be drawn from the same distribution as the risks you are producing SCR bounds for. If your calibration data is from 2018–2022 and your in-force portfolio reflects 2024 underwriting, you have a potential exchangeability violation. The `RetroAdj` class in the same library handles online distribution shift via retrospective jackknife+ adjustment, but that is for pricing models; for SCR reporting you need to be deliberate about calibration set construction.

**Calibration set size at the tail.** The 99.5% bound requires a calibration set of at least 200 observations (strictly `ceil(0.995 * 201) = 200`). You will typically want substantially more than that for the bound to be tight. With 200 calibration observations, the nonconformity score quantile is determined by a single order statistic, and the resulting interval will be wide. With 2,000 calibration observations, the bound is materially tighter. For most UK commercial lines books, finding 2,000 recent comparable risks with observed outcomes is achievable; for very specialised lines it is not.

**Marginal, not conditional, coverage.** The guarantee is that at least 99.5% of *all* risks are covered on average. It does not guarantee 99.5% coverage for any particular risk subgroup. A conformal predictor that covers 99.7% of small commercial risks and 99.2% of large commercial risks has technically met the marginal guarantee but is under-covering the risks where it most matters. `CoverageDiagnostics.coverage_by_decile()` from the same library tests conditional coverage by expected loss decile and should be part of any SCR validation process.

**Not a replacement for an internal model.** The conformal upper bound is a sanity check and a supplementary validation tool. EIOPA has not issued guidance on conformal methods as of early 2026. Present these outputs to your regulator as technical validation alongside your internal model, not as an alternative to it.

**No dependency diversification.** `SCRReport` produces per-risk SCR components. Aggregating them by summing is implicitly assuming perfect correlation. For portfolio-level SCR you still need a copula or covariance structure. The conformal bounds give you the per-risk marginal distributions; the aggregation is your problem.

---

## When to use this over parametric approaches

Use conformal SCR bounds when:

- You want a robustness check on your internal model that requires no additional distributional assumptions.
- You are entering a new line of business where you do not yet have sufficient data to justify a parametric tail choice, but you have access to analogous historical data for calibration.
- Your model governance process requires a quantitative demonstration that coverage claims are borne out on held-out data — the `coverage_validation_table()` output is exactly that.
- You have a book where the tail behaviour is genuinely uncertain, for example a commercial property account with heterogeneous construction types where neither lognormal nor Pareto provides a clean fit to the large loss data.

Do not use conformal SCR bounds when:

- Your calibration set has fewer than 500 observations. Below that the 99.5% bound is determined by too few order statistics and will be unreliably wide.
- You have strong, well-validated prior knowledge of the tail distribution from a long run of historical data. Parametric models exploit this information efficiently; conformal prediction does not. If you genuinely know the tail is Pareto with shape 1.5 because you have 20 years of data confirming it, use the parametric model.
- You need a portfolio-level SCR that incorporates correlation structure. This library gives you marginal per-risk bounds.

The right use is as a distribution-free lower bound on what your capital requirement must be, plus a validation tool to test whether your parametric internal model's stated coverage is actually achieved on real data. Those are genuinely useful functions that no commercial actuarial tool currently provides.

---

Source: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

- Hong (2025), "Conformal prediction of future insurance claims in the regression problem." [arXiv:2503.03659](https://arxiv.org/abs/2503.03659)
- [Conformalized Quantile Regression for Insurance](/2026/03/24/conformalized-quantile-regression-insurance/) — the CQR implementation shipped in v0.6.2, which produces heteroscedastic bounds directly useful for per-risk volatility
- [Your Book Has Shifted and Your Model Doesn't Know](/2026/09/14/your-book-has-shifted-and-your-model-doesnt-know/) — covers `RetroAdj` for handling the exchangeability violations that matter most for SCR calibration
