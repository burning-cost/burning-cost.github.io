---
layout: post
title: "Your SCR Is an Expected Shortfall. Stop Estimating It Via a Quantile Detour."
date: 2026-04-02
categories: [capital, solvency-ii, techniques, libraries]
tags: [expected-shortfall, CVaR, solvency-ii, SCR, quantile-regression, elicitability, Fissler-Ziegel, sandwich-standard-errors, insurance-quantile, tail-risk, python, uk, capital-modelling]
description: "ExpectedShortfallRegressor in insurance-quantile v0.5.0 implements direct ES regression via the i-Rock method. No quantile detour. Sandwich SEs. Directly relevant to Solvency II SCR."
---

Solvency II Article 101 defines the SCR as the Value at Risk of basic own funds at the 99.5th percentile over a one-year horizon. That is what the regulation says. What the regulation *means* is Expected Shortfall at 99.5%: the average loss in the worst 0.5% of outcomes, not just the boundary of that region. The VaR framing is a legal convenience. Every internal model actuary working in capital knows the real target is the tail expectation.

Which makes it puzzling that the standard workflow for estimating per-risk tail capital involves fitting a quantile model at some level α, then integrating the quantile function from α upward to get the expected shortfall — a post-hoc numerical computation on top of a model that was never trained to minimise error on the ES itself.

Li, Zhang, and He (2026) — arXiv:2602.18865, submitted February 2026 — showed why this is the wrong approach, proved what the right estimator looks like, and derived its asymptotic theory. We have implemented it in [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) v0.5.0 as `ExpectedShortfallRegressor`.

```bash
uv add insurance-quantile
```

---

## The superquantile is not the Expected Shortfall

This point needs stating plainly because it is not intuitive, and most practitioners have absorbed the opposite assumption.

The Rockafellar-Uryasev (2000) result says: minimise `E[loss - z] / (1 - α) + z` over z and you get CVaR. For a fixed x, this finds the ES at the portfolio level. It is clean, widely used, and correct for unconditional optimisation.

When you extend this to regression — fitting a model f(X) such that f(x) estimates ES(α | x) — you get **superquantile regression**: minimise `E[f(X)] / (1 - α) + z` as a function of both the regression coefficients and a scalar z. This is what `cvxpy` users implement when they try to build a direct ES model from first principles.

Li, Zhang, and He prove in Appendix A.2 of their paper that superquantile regression and ES regression produce **different coefficient vectors**, even in the linear model, even at the population level. There is a formal counterexample. The only exception is a degenerate case where the conditional distribution of Y given X is homoskedastic. UK insurance data is not homoskedastic.

The practical implication: if you are using a Rockafellar-Uryasev-style objective to fit your ES model, you are fitting a model whose coefficients are approximately correct for the mean but systematically wrong for the tail shape. The errors are larger for features with high heteroskedasticity — which is precisely the features driving your SCR differences across risks.

---

## What makes direct ES regression possible

ES is not a directly elicitable functional in the strict sense — you cannot write a single loss function such that the unique minimiser is the true ES. This was long cited as the theoretical barrier to direct ES estimation.

The Fissler-Ziegel (2016) result resolved this. While ES is not elicitable alone, the pair (VaR, ES) is *jointly* elicitable. There exists a strictly consistent scoring rule for the pair. This means you can fit a model that targets ES directly, as long as you also fit the quantile in the same objective or treat it as a nuisance parameter.

The i-Rock method from Li et al. takes a different route: rather than optimising the Fissler-Ziegel joint loss directly, it constructs a four-step procedure that is asymptotically equivalent to the optimal estimator. The key object is the Neyman-orthogonalised pseudo-value:

```
Z_i = Q_hat(α, X_i) + (Y_i - Q_hat(α, X_i))_+ / (1 - α)
```

This is constructed to satisfy `E[Z_i | X_i] = ES(α, X_i)` asymptotically, with the additional property that it is insensitive to first-order error in `Q_hat`. The first-stage quantile model can be imprecise without corrupting the ES estimate — this is Neyman orthogonality in the semiparametric estimation sense.

---

## The four-step procedure

`ExpectedShortfallRegressor` runs the Li et al. procedure end-to-end:

**Step 1.** Fit a first-stage quantile model at level α. By default this is `sklearn.linear_model.QuantileRegressor`; you can substitute CatBoost via `first_stage='catboost'`.

**Step 2.** Compute the Neyman-orthogonalised pseudo-values `Z_i` per observation.

**Step 3.** Partition the covariate space into bins using quantile-based breakpoints. The bin count is set automatically using the formula from the paper:

```
k = ceil(1.6 * sqrt(p) * (sqrt(n) / log(n))^(1/p))
```

where p is the number of covariates. Within each bin, run OLS of Z on X to get the local ES estimate `v_hat_m` at the bin centroid `x_bar_m`.

**Step 4.** Run second-stage quantile regression of `v_hat` on `x_bar` across all (M bins × T grid points) pseudo-observations. The resulting coefficients are the i-Rock ES regression estimates.

```python
import numpy as np
from insurance_quantile import ExpectedShortfallRegressor

esr = ExpectedShortfallRegressor(
    alpha=0.995,          # Solvency II level: ES at 99.5%
    first_stage="linear", # or "catboost" for nonlinear first stage
    n_bins_per_dim="auto",
)
esr.fit(X_train, y_train)
```

The API takes numpy arrays — not Polars DataFrames as `QuantileGBM` does — because this is a linear model for inference rather than a GBM for prediction. The coefficients have a direct interpretation: each `beta_j` is the marginal effect of feature j on the ES.

---

## Confidence intervals on tail risk

This is the part that does not exist anywhere else in Python.

Because the estimator has known asymptotic distribution — Theorem 4.2 of Li et al.:

```
sqrt(n) * (beta_hat - beta) -> N(0, D1^{-1} Omega1 D1^{-1})
```

where D1 and Omega1 are computable from the fitted model — we can construct a sandwich standard error for each coefficient. The `summary()` method produces a table analogous to a GLM coefficient table:

```python
summary = esr.summary()
print(summary)
```

```
               term      coef    std_err    z_stat   p_value
          intercept   4821.30     312.14     15.45    <0.001
        sum_insured      2.14       0.31      6.90    <0.001
       vehicle_age     -38.20      12.80     -2.98     0.003
         no_claims    -287.40      44.10     -6.52    <0.001
  commercial_route    1043.20     201.50      5.18    <0.001
```

`sum_insured` carries a positive coefficient: higher-value vehicles have higher ES, and you now have a confidence interval on that statement. `commercial_route` has a coefficient of £1,043 with a standard error of £202: it adds roughly £1k to expected shortfall at 99.5%, and we are 95% confident that number is above £647.

The sandwich estimator here is approximate. For n below about 500 the asymptotic coverage may be below nominal, and we recommend bootstrap standard errors in that case. For typical pricing datasets of 5,000+ non-zero severity observations, the sandwich SEs are reliable.

---

## The trapezoidal TVaR approach and when to use each

`insurance-quantile` already has `per_risk_tvar()`, which computes ES via numerical integration of the quantile function from α upward. That approach is nonparametric, requires no linearity assumption, and works well with a CatBoost quantile model backing it.

`ExpectedShortfallRegressor` is different in kind, not just in degree:

| | `per_risk_tvar()` | `ExpectedShortfallRegressor` |
|---|---|---|
| ES assumption | None (nonparametric) | Linear: ES(α, x) = x'β |
| SE on coefficients | No | Yes (sandwich) |
| Feature effects on ES | No (prediction only) | Yes (with CIs) |
| Data efficiency | Needs many quantile levels | Structured via binning |
| Good for | ES predictions at any risk | Inference on what drives ES |

We would use `per_risk_tvar()` for pricing — generating per-risk ES predictions to load into a premium formula. We would use `ExpectedShortfallRegressor` when the question is *which features drive tail exposure*, how much, and with what confidence. For Solvency II model documentation, the coefficient table from `summary()` is a better governance artefact than a plot of mean ES by decile.

---

## Solvency II application

For a UK general insurer running an internal model, the direct application is straightforward. The SCR for each segment or risk group is approximately its ES at 99.5%. The `ExpectedShortfallRegressor` gives you a linear model for this:

```python
from insurance_quantile import ExpectedShortfallRegressor

# Fit on ground-up non-zero severity, severity-only model
esr = ExpectedShortfallRegressor(alpha=0.995)
esr.fit(X_train, y_train)

# Predicted ES at 99.5% per new risk
scr_estimates = esr.predict(X_new)

# Coefficient table for regulatory documentation
esr.summary().to_csv("scr_model_summary.csv")
```

For each new risk, `predict()` returns the estimated ES at 99.5% under the linear model: the average severity in the worst 0.5% of outcomes for that risk's covariate profile.

The `summary()` output goes straight into the model documentation. A PRA model reviewer asking "what drives SCR differences across risks?" now has a table with coefficients and confidence intervals, not just a shapley plot or a partial dependence curve.

---

## What the package does not do

Three caveats worth stating explicitly.

**The linear form.** `ExpectedShortfallRegressor` estimates `ES(α, x) ≈ x'β`. If the true ES function is substantially nonlinear, the coefficients are a projection, not the truth. For heterogeneous books where, say, the ES curve bends sharply at a vehicle value threshold, the linear model will smooth over that. `per_risk_tvar()` with a CatBoost first stage handles nonlinearity better for prediction purposes.

**Small samples.** The binning step requires sufficient observations per bin. With fewer than about 200 non-zero severity claims, the bin counts become very small and the local OLS estimates within bins are unreliable. The paper's theory requires n → ∞; the finite-sample behaviour below n=500 is where the sandwich SEs are approximate rather than exact.

**Frequency is separate.** This is a severity model. It estimates `E[Y | Y > 0, Y is in the top (1-α) of its conditional distribution, X = x]`. The SCR premium loading is `frequency × severity_ES`. The frequency piece is a separate model — typically Poisson GLM or Poisson GBM.

---

## Availability

`ExpectedShortfallRegressor` is in `insurance-quantile` v0.5.0, committed at `12cb4b4`. This is the only Python implementation of the Li, Zhang, and He (2026) i-Rock estimator. No other pip-installable package has direct ES regression with sandwich standard errors; the closest available tooling is `statsmodels.QuantReg` plus manual integration, which implements the wrong objective.

```bash
uv add insurance-quantile
# or
pip install insurance-quantile
```

Source: [github.com/burning-cost/insurance-quantile](https://github.com/burning-cost/insurance-quantile). The library now has 412 tests passing, 44 of which are specific to the ES regressor.

---

**Related posts:**

- [How to Build a Large Loss Loading Model for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) — the GBM quantile route to per-risk TVaR; use this when linearity of ES is not defensible
- [Conformal Prediction for Solvency II Capital Requirements](/2026/03/25/conformal-prediction-solvency-ii-scr-model-free-estimation/) — model-free upper bounds on SCR using conformal prediction; no distributional assumption at all
- [Extreme Value Theory for UK Motor Large Loss Pricing](/2026/03/13/insurance-evt/) — GPD-based tail modelling for the unconditional severity distribution; complements ES regression when the i.i.d. tail structure is the target
