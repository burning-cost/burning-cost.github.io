---
layout: post
title: "GAMLSS in Python, Finally: insurance-distributional-glm"
date: 2026-03-10
categories: [libraries, pricing, capital]
tags: [gamlss, distributional-regression, glm, gamma, tweedie, nbi, zip, variance-modelling, capital-allocation, reinsurance, python, insurance-distributional-glm]
description: "R has had GAMLSS since 2005. Python has had nothing production-ready — until now. insurance-distributional-glm models every distribution parameter as a function of covariates: not just the mean, but the variance, shape, and tail. Per-risk volatility. Regulatory-grade relativities. Seven families."
---

The Gamma GLM you use for severity modelling makes one assumption you probably do not spend much time thinking about: the coefficient of variation is the same for every risk in the portfolio. Not the mean — that varies freely with age, vehicle value, territory, all the usual factors. The CV. Constant across the book.

This is wrong, and it is wrong in ways that matter.

A fleet policy covering 500 vehicles has a lower CV than a single private car, because losses average out. Territory A — say, rural Somerset — has lower claim variance than Territory B, a dense urban postcode with higher accident frequency and a more volatile mix of minor and serious damage. A driver with three prior fault claims in two years has a more dispersed loss distribution than a clean record at the same expected frequency. The Gamma GLM encodes none of this. It produces a mean estimate per risk and a single scalar dispersion for the whole book.

The technical name for modelling the full parameter set of a distribution — mean, variance, shape, tail — as functions of covariates is GAMLSS: Generalised Additive Models for Location, Scale and Shape. Rigby and Stasinopoulos introduced it in *JRSS Series C* in 2005. R has had a full implementation since then, with over 100 distributions and a mature ecosystem around it.

Python has had nothing.

[`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) is the first production-ready Python implementation of GAMLSS for insurance. Not a research prototype. Not a wrapper that breaks on real data. A GLM-based library with seven actuarially relevant distributions, full RS algorithm convergence, per-risk volatility scoring, and relativity tables you can put in front of a regulator.

```bash
pip install insurance-distributional-glm
```

---

## The problem with fixed dispersion

A standard Gamma GLM with log link models:

```
E[Y|X] = exp(X * beta)
Var[Y|X] = phi * E[Y|X]^2
```

where `phi` is a scalar estimated once across the whole dataset. The coefficient of variation is `sqrt(phi)` — the same for every risk. The model says that a fleet policy at £2,000 expected severity and a single car policy at £2,000 expected severity have identical variance. This is not a modelling choice you made consciously. It is a structural constraint baked into every GLM you have ever fitted.

The consequence is systematic mispricing of risk dispersion. Segments that are genuinely more volatile than average are under-reserved on volatility. Segments that are more stable are over-reserved. When you use those GLM outputs for capital allocation, reinsurance pricing, or IFRS 17 risk adjustments, you are carrying errors that the discrimination metrics — Gini, double lift, deviance ratio — will never surface, because they measure mean prediction accuracy, not distributional accuracy.

GAMLSS fixes this by giving `phi` its own linear predictor. Now:

```
E[Y|X]   = exp(X_mu * beta_mu)
Var[Y|X] = phi(X_sigma) * E[Y|X]^2
```

where `phi(X_sigma) = exp(X_sigma * beta_sigma)`. Different covariates, different coefficients, different degrees of freedom. Territory might drive variance even when it is not significant for the mean. Claim history might affect dispersion more than it affects the mean. The two models are fitted jointly; they interact through the residuals.

---

## The RS algorithm

The Rigby-Stasinopoulos algorithm makes this tractable. Rather than optimising all parameters simultaneously — which is non-convex and unstable — it iterates:

1. Hold `sigma` (dispersion) fixed. Fit `mu` using standard IRLS. This is exactly what your existing Gamma GLM does.
2. Hold `mu` fixed. Compute working responses for `sigma` from the deviance residuals. Fit `sigma` using IRLS on those working responses.
3. Repeat until convergence.

Each step is a standard GLM fit. If you understand IRLS — the weighted least squares procedure that underlies every GLM you have fitted — you already understand the RS algorithm. It is the same update, applied in sequence to each distribution parameter. For distributions with three or four parameters (the NBI and Tweedie families used here), the iteration extends naturally: fit `mu`, fit `sigma`, fit the shape parameter, cycle again.

Convergence is typically fast. On insurance datasets of 200,000 to 2 million records, three to five RS cycles are sufficient. The library defaults to a maximum of 20 iterations with a tolerance of 1e-6 on the log-likelihood.

---

## The API

```python
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma, NBI, ZIP

model = DistributionalGLM(
    family=Gamma(),
    formulas={
        'mu': ['age', 'vehicle_value', 'territory', 'ncb'],
        'sigma': ['territory', 'ncb']
    }
)
model.fit(X, y, exposure=exposure)
```

The `formulas` argument is the key design decision. `mu` takes the full covariate set — the usual GLM rating factors. `sigma` takes a smaller set. This is intentional: variance models benefit from parsimony. Territory and claim history are typically the strongest predictors of volatility; age and vehicle value, while dominant for the mean, often add noise rather than signal to the dispersion model.

After fitting:

```python
# Per-risk volatility scoring
vol = model.volatility_score(X)       # CV per risk, same shape as X

# Relativity tables for each parameter
mu_rel    = model.relativities(parameter='mu')
sigma_rel = model.relativities(parameter='sigma')
```

The `relativities()` method returns a dictionary indexed by covariate, with factor levels and their multiplicative relativities, standard errors, and 95% confidence intervals. The format is what UK pricing actuaries expect: one table per factor, base level normalised to 1.0, ready for the rating manual.

For model selection across families, the `choose_distribution` function runs all candidates and ranks by AIC:

```python
from insurance_distributional_glm.selection import choose_distribution

ranking = choose_distribution(X, y, [Gamma(), NBI(), ZIP()])
```

The ranking includes AIC, BIC, and log-likelihood for each candidate. We recommend Gamma for severity models, NBI for overdispersed frequency, and ZIP when zero inflation is structural rather than incidental (travel insurance, warranty products).

---

## Seven families

| Class | Distribution | Parameters modelled | Primary use |
|-------|-------------|---------------------|-------------|
| `Gamma` | Gamma | mu, sigma | Severity |
| `LogNormal` | Log-Normal | mu, sigma | Severity (heavier tail) |
| `InverseGaussian` | Inverse Gaussian | mu, sigma | Severity (very heavy tail) |
| `Tweedie` | Compound Poisson-Gamma | mu, sigma | Aggregate loss |
| `Poisson` | Poisson | mu | Frequency (equidispersed) |
| `NBI` | Negative Binomial I | mu, sigma | Overdispersed frequency |
| `ZIP` | Zero-Inflated Poisson | mu, sigma | Zero-inflated frequency |

All seven families have validated analytical derivatives for the IRLS update. We did not use finite differences. Numerical derivative approximations produce incorrect standard errors at the boundaries of the parameter space, which is precisely where insurance data tends to live (near-zero frequencies, high-severity tails). The analytical derivatives are slower to implement and easier to get wrong; they are also the only approach that gives you correct inference.

---

## Why no Python equivalent existed until now

R's `gamlss` package was published in 2005 by Rigby and Stasinopoulos. It now covers over 100 distributions and is well-maintained. The Python gap is not for lack of demand — actuarial pricing teams using Python have wanted this for years.

The existing Python options each fall short in the same way: they model only the location parameter.

**pyGAM** fits generalised additive models (splines for non-linear effects) but has a single linear predictor. No distributional regression.

**statsmodels GLM** is a rigorous implementation of single-parameter GLMs. The dispersion is a scalar, post-hoc estimate. There is no mechanism for modelling it as a function of covariates.

**scikit-learn** does not fit GLMs in the actuarial sense; its `TweedieRegressor` has no exposure offset, no analytical deviance, and no mechanism for distributional regression.

The gap is real and has been real for twenty years. We filled it.

---

## Practical uses

**Per-risk capital allocation.** Under Solvency II, the SCR is driven by the tail. If you are allocating capital to individual risks or sub-portfolios, you need a variance estimate per risk, not a book-level average. The `volatility_score()` output is the per-risk CV, which feeds directly into variance-based capital allocation methods (variance principle, standard deviation principle). Risks in the top quartile of CV need materially more capital per unit of expected loss than risks in the bottom quartile. The GLM cannot tell them apart; the GAMLSS model can.

**Reinsurance pricing.** The excess-of-loss cost depends on the tail of the severity distribution. The tail is determined by both `mu` and `sigma`. A risk with moderate expected severity but high dispersion contributes more to XL premium than a risk with the same expected severity and low dispersion. Pricing treaties off mean-only models systematically undercharges ceded premium for volatile segments and overcharges it for stable ones.

**Reserve volatility assessment.** The Merz-Wüthrich one-year risk measure, used for IFRS 17 risk adjustments, requires an estimate of process variance in addition to parameter uncertainty. A distributional model provides the process variance estimate directly, as a function of risk characteristics.

**PRA model validation.** Internal model validation under the PRA's supervisory statements requires demonstrating that the model captures not just the mean structure but the variance structure. A GAMLSS model produces explicit, interpretable evidence that the variance has been modelled, with factor-level relativities and confidence intervals. Telling your validator "we modelled sigma as a function of territory and claim history, and here are the relativities" is a materially stronger position than "we assume constant CV."

---

## How this relates to insurance-distributional

We have two distributional regression libraries. The question is which to use when.

[`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) uses CatBoost gradient boosted machines for both the mean and dispersion models. It handles interactions natively, requires no feature engineering, and scales to millions of records with minimal tuning. The cost is interpretability: the dispersion model is a GBM. You can extract SHAP values, but you cannot produce a relativity table with confidence intervals. It is the right choice when you have a large dataset, complex non-linear interactions, and a governance framework that accepts GBMs.

`insurance-distributional-glm` uses GLMs throughout. The mean model and the dispersion model are both GLMs with explicit factor levels, multiplicative structure, and analytical standard errors. It is slower on large datasets and requires more feature engineering. It is also the only option when the governance requirement is a relativity table — when the pricing actuary needs to sign off on a rating manual with explicit factors for dispersion, not just a model object. UK personal lines pricing, PRA internal model validation, Lloyd's syndicate filings: these are GLM-governance contexts.

Same conceptual framework. Different tooling for different regulatory environments.

---

## Quality

114 tests covering all seven families, all IRLS update paths, convergence behaviour, edge cases at distribution boundaries, and numerical stability under extreme exposures. CI runs on every commit.

The library is in active use. We are not aware of any other production-ready Python GAMLSS implementation.

```bash
pip install insurance-distributional-glm
```

[GitHub: burning-cost/insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm)

---

## See also

- [`insurance-distributional`](/2026/03/05/insurance-distributional) — distributional GBMs using CatBoost; same variance-modelling goal, non-linear models for GBM-governance contexts
- [`insurance-drn`](/2026/03/10/insurance-drn) — Distributional Refinement Networks; take any fitted GLM and refine it into a full predictive distribution using a neural correction layer
- [`insurance-calibration`](/2026/03/09/insurance-calibration) — balance property testing, Murphy decomposition, auto-calibration; once you have a distributional model, use this to validate it is well-calibrated at every price level
