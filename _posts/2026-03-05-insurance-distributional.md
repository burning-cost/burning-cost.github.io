---
layout: post
title: "Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean"
date: 2026-03-05
categories: [techniques, libraries]
tags: [distributional-regression, tweedie, catboost, gbm, dispersion, volatility, motor, pet, zip, zero-inflation, python, astin]
description: "insurance-distributional models the full conditional loss distribution, not just the mean. First open-source Python implementation of the ASTIN 2024 Best Paper."
---

Your Tweedie GBM produces one number per risk: the expected loss. That is what you price off. That is what goes into the technical premium. And for two risks sitting at the same expected loss, it treats them identically.

Here is where that breaks down.

Take two motor risks both priced at £350 pure premium. Risk A is a 45-year-old driving a 1.4 Focus in Swindon: low frequency, modest severity. Risk B is a 22-year-old with a modified 2.0 Golf in Manchester: also low expected frequency (young drivers claim rarely in any given year), but when they do, it is expensive. The conditional distributions of these two risks are completely different. The Tweedie GBM, even a well-specified one, returns £350 for both, because it is predicting E[Y|x]. It has nothing to say about Var[Y|x].

That distinction matters in at least four places: safety loading, underwriter referrals, IFRS 17 risk adjustment, and capital allocation under Solvency II. In all of these, it is the variance (specifically the coefficient of variation per risk) that should be driving the decision. You are currently using a constant, or a blunt segmented estimate, because no tool in your standard stack produces CoV per risk.

We built [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) to fix that. It is the 20th library in the Burning Cost suite, and the first open-source implementation of the approach from the ASTIN 2024 Best Paper (So & Valdez, *Applied Soft Computing*, doi:10.1016/j.asoc.2025.113226; arXiv 2406.16206).

---

## Why no one has done this yet

The gap is not theoretical. Everyone in actuarial pricing knows that Var[Y|x] matters. The gap is in available tooling.

The obvious starting point is GAMLSS (Generalized Additive Models for Location, Scale and Shape, introduced by Rigby and Stasinopoulos in *JRSS Series C* (2005)). GAMLSS models all parameters of the conditional distribution (mean, dispersion, shape) as smooth functions of covariates. It works. It is also GAM-based, which means splines, R only, slow on large datasets, and unable to handle high-cardinality categoricals natively. It is the statistical ancestor of distributional GBM, not the tool you want on a UK personal lines book.

The natural extension is distributional gradient boosting. Three implementations exist, each with a blocking problem for insurance.

**XGBoostLSS** (März, arXiv 1907.03178) supports 26 distributions across XGBoost and LightGBM. Its dependency stack includes XGBoost, PyTorch, and Pyro (Uber's probabilistic programming framework), which is substantial. More critically: it has no Tweedie distribution. Tweedie compound Poisson-Gamma is the standard distribution for motor and home pure premium, so this is not an obscure gap. XGBoostLSS cannot do your core motor model distributionally.

**NGBoost** (Stanford ML Group, ICML 2020) uses natural gradients for distributional prediction. Also no Tweedie. Also slower than tree-based methods. Also uses scikit-learn trees rather than CatBoost, which means no native handling of high-cardinality categoricals.

**CatBoostLSS**, the one you actually want given CatBoost's superiority for UK personal lines with vehicle make/model, occupation, and postcode features, has a banner on its GitHub repository: "Repo is currently not maintained." Last significant activity was 2020. No PyPI package. It was abandoned.

So the state of play as of March 2026: the ASTIN award-winning approach to distributional GBM for insurance has been published (So & Valdez 2024), peer-reviewed, awarded, and has no open-source implementation anyone can install and use.

---

## What insurance-distributional does

```bash
uv add insurance-distributional
```

Four distribution classes. Same fit/predict interface throughout.

| Class | Distribution | Parameters | Primary use |
|-------|-------------|------------|-------------|
| `TweedieGBM` | Compound Poisson-Gamma | mu, phi | Motor/home pure premium |
| `GammaGBM` | Gamma | mu, phi | Severity-only models |
| `ZIPGBM` | Zero-Inflated Poisson | lambda, pi | Pet/travel/breakdown frequency |
| `NegBinomialGBM` | Negative Binomial | mu, r | Overdispersed frequency |

No PyTorch. No Pyro. CatBoost, NumPy, SciPy, Polars.

---

## The motor pure premium model

`TweedieGBM` is the core class for UK motor and home pricing.

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

pred = model.predict(X_test, exposure=exposure_test)

pred.mean             # E[Y|X] — pure premium
pred.variance         # Var[Y|X] = phi * mu^p, per risk
pred.cov              # CoV = SD/mean, per risk
pred.volatility_score()  # same as cov — for safety loading and referrals
```

`power=1.5` is standard for UK motor. You fix it at fit time; the library does not estimate it from data, following the So & Valdez approach.

The `pred` object is a `DistributionalPrediction`, a container that bundles all derived quantities so you do not have to recompute them. The two numbers you care about are `mean` (your existing pricing output) and `volatility_score()` (the new thing).

---

## How the dispersion model works

Standard Tweedie GBMs estimate a single scalar dispersion phi for the whole book. `TweedieGBM` estimates phi as a function of covariates: a separate GBM fitted to squared Pearson residuals from the mean model.

The method is the Smyth-Jørgensen double GLM, from their 2002 *ASTIN Bulletin* paper (32(1):143-157). The response variable for the dispersion model is:

```
d_i = (y_i - mu_hat_i)^2 / mu_hat_i^p
```

Under the Tweedie model, E[d_i] = phi_i. So we fit a Gamma regression with log link on d_i to estimate phi(x). The claim is not new; Smyth and Jørgensen made it in 2002. What is new is doing it with a GBM rather than a GLM, which lets phi vary non-linearly across the same feature space you use for the mean model.

The practical consequence: risks with the same expected loss can have materially different dispersion estimates depending on their feature profile. A young driver's phi is typically higher than a middle-aged driver's with the same mean. Previously you had no way to quantify that per risk. Now you do.

Coordinate descent handles the two-model estimation:

1. Fit the mean model (CatBoost Tweedie loss with log offset for exposure).
2. Compute squared Pearson residuals from step 1.
3. Fit the dispersion model (CatBoost on log-squared-residuals, weighted by exposure).
4. Repeat for n_cycles (default 1; one cycle is sufficient for most insurance datasets).

If you want a scalar dispersion (faster, simpler, matches a standard Tweedie GBM):

```python
model = TweedieGBM(power=1.5, model_dispersion=False)
```

---

## Zero-inflated frequency: pet insurance

Pet insurance has a structural zero problem that motor does not. Most pets do not claim in any given year. Not because their claims were Poisson-zero, but because they are genuinely no-claim pets. The zeros are a mixture of two populations: pets that happened not to claim this year, and pets that never claim. Treating the whole zero mass as Poisson is a misspecification.

`ZIPGBM` separates the two:

```python
from insurance_distributional import ZIPGBM

model = ZIPGBM()
model.fit(X_train, y_train)
pred = model.predict(X_test)

pred.mu     # observable mean = (1-pi)*lambda
pred.pi     # structural zero probability per pet
pred.variance  # (1-pi)*lambda*(1 + pi*lambda) — ZIP variance formula

# Underlying Poisson rate for the claiming population
model.predict_lambda(X_test)
```

The estimation follows So (arXiv 2307.07771): two-stage coordinate descent with EM soft labels for the pi classifier. Stage one fits a Poisson GBM for lambda(x), weighting observations by their probability of being a genuine Poisson zero rather than a structural zero. Stage two fits a CatBoostClassifier for pi(x) on soft EM labels derived from the current lambda estimate.

The observable mean `(1-pi)*lambda` is what you price off. The `pi` and `lambda` outputs separately tell you something useful: a risk with high pi is structurally unlikely to claim; a risk with high lambda (but lower pi) is a different risk profile that claims frequently when it does enter the claiming population. Pet breed, age, and whether the policy includes dental cover all affect pi and lambda differently.

---

## The volatility score and what to do with it

The key output is `volatility_score()`, the coefficient of variation per risk:

```
CoV(Y|x) = sqrt(Var[Y|x]) / E[Y|x]
```

For Tweedie: `CoV = sqrt(phi(x)) * mu(x)^(p/2-1)`. For ZIP: derived from the ZIP variance formula above. It is dimensionless and directly comparable across risks of different sizes.

Five things you can do with it that you cannot currently do with a standard pricing model:

**Safety loading.** `P = mu * (1 + k * CoV)` where k is your risk appetite parameter. Currently you apply the same k to every risk. With per-risk CoV, k becomes optional, since the loading already varies. The Risk A / Risk B example from the introduction: if both have mu=350 but Risk B has CoV 40% higher, the correct premium for Risk B is proportionally higher even though the expected loss is identical.

**Underwriter referral rules.** Flag risks where CoV exceeds a threshold. No commercial pricing tool outputs CoV per risk: not Emblem, not Radar, not Guidewire. The rule "refer any risk with CoV > 0.8" is currently unimplementable in standard pricing workflows.

**Reinsurance attachment optimisation.** XL attachment decisions depend on the distribution of outcomes, not just the mean. High-phi segments drive tail exposure disproportionately. Identifying them lets you price the protection rationally rather than loading proportionally to mean.

**IFRS 17 risk adjustment.** IFRS 17 requires a risk adjustment for non-financial risk: the compensation required for bearing uncertainty. Standard approaches use quantile/VaR loadings applied to aggregate reserves. Per-risk CoV provides a more granular input, particularly for portfolios with heterogeneous variance profiles.

**Capital allocation (Solvency II).** Under the standard formula, SCR is allocated at product/LoB level. With per-risk conditional variance, you can identify which risks consume disproportionate SCR per unit of premium. High-CoV risks at technical premium are candidates for repricing; high-CoV risks below technical premium are candidates for non-renewal.

---

## Scoring and calibration

Mean predictions from Tweedie models are typically validated with deviance. Distributional predictions require additional diagnostics because you are now asserting a specific functional form for the whole distribution, not just the mean.

```python
from insurance_distributional import (
    tweedie_deviance,
    coverage,
    pit_values,
    gini_index,
)

# Standard actuarial metric — same as before, for the mean
tweedie_deviance(y_test, pred.mean, power=1.5)

# Does the predicted 90% interval contain 90% of observations?
cov = coverage(y_test, pred, levels=(0.80, 0.90, 0.95))
# Requires v0.1.3+. Earlier versions had a phi estimation bug (near-zero phi
# due to in-sample mu overfitting) that caused coverage intervals to collapse.
# v0.1.3 fixes this with K=3 cross-fitting and Gamma deviance loss.

# PIT histogram — should be uniform for a well-calibrated model
pit = pit_values(y_test, pred, n_samples=5000)

# Gini on mean (standard) vs Gini on volatility score (new)
gini_mean = gini_index(y_test, pred.mean)
gini_vol  = gini_index(y_test, pred.volatility_score())
```

The `gini_vol` figure is worth checking: it answers whether the dispersion model has genuine discriminating power. A volatility score with Gini near zero means the dispersion model is not adding information beyond the mean. In our experience on UK motor data, the Gini on CoV is typically 40–60% of the Gini on the mean: meaningful but not as strong as the mean model, which is what you would expect given that dispersion is noisier to estimate.

The `coverage` function uses Monte Carlo sampling from the fitted distribution. For Tweedie, this is a compound Poisson-Gamma simulation using the standard Jørgensen (1987) parameterisation. Use v0.1.3 or later: earlier versions had a phi estimation bug that made coverage metrics unreliable. The mean prediction (`pred.mean`) was unaffected in all versions.

---

## The CRPS

For comparing distributional models against each other, proper scoring rules are the right tool. The Continuous Ranked Probability Score (CRPS) is:

```python
model.crps(X_test, y_test)       # Monte Carlo CRPS
model.log_score(X_test, y_test)  # negative log-likelihood
```

CRPS has the same units as the target (pounds, claim counts) and is strictly proper, minimised only when you report the true distribution. Unlike the log-score, it does not blow up for extreme observations in the tail. Use it to compare `TweedieGBM(model_dispersion=True)` against `TweedieGBM(model_dispersion=False)` and against a standard Tweedie GBM. The distributional model should win on CRPS. If it does not, the dispersion model has not learned anything useful from your data.

---

## Practical notes for UK personal lines

**CatBoost, not XGBoost.** CatBoost's ordered target statistics for categorical features handle vehicle make/model, occupation code, and postcode area without encoding. The Chevalier & Côté EAJ 2025 comparison confirmed that CatBoost outperforms XGBoost and other GBM frameworks when the dataset has many high-cardinality categoricals, which UK personal lines always does. Pass them directly:

```python
model = TweedieGBM(
    power=1.5,
    cat_features=["vehicle_group", "occupation_code", "postcode_area"],
    catboost_params_mu={"iterations": 500, "depth": 8, "task_type": "GPU"},
    catboost_params_phi={"iterations": 200},
)
```

**Exposure handling.** `fit()` takes an `exposure` parameter (policy years in a vector). The mean model uses `log(exposure)` as a baseline offset, which is standard for rate-per-year models. The dispersion model weights observations by exposure: a full-year policy is a more reliable data point than a mid-term cancellation.

**Power parameter.** Fix `power=1.5` for motor pure premium. This is the standard for UK motor and what So & Valdez used. You can try `power=1.6` or `power=1.4` if your Tweedie deviance suggests a different power for your specific data, but do not attempt to estimate it jointly with the other parameters in v1, as identification is fragile.

**Pet insurance.** Use `ZIPGBM`, not `TweedieGBM`. Claim rates for cats are around 15–25% per year, for dogs 20–30%. The structural zero proportion is high enough that ZIP materially outperforms standard Poisson on log-score. The pi model will find breed, age, and indoor/outdoor status as strong predictors of structural zero probability.

---

## Design decisions we want to be explicit about

We made three design choices that differ from the existing literature.

**Separate models per parameter, not joint multi-output.** CatBoost's `MultiTargetCustomObjective` supports joint estimation but has sparse documentation and known GPU incompatibilities. Separate models are simpler to debug and maintain. The statistical cost (no shared tree structure) is negligible for the typical insurance pricing use case where the feature sets for mu and phi overlap almost completely.

**No PyTorch.** XGBoostLSS uses Pyro for autodiff-based gradient computation. Pyro alone is 500MB+ of dependencies and is a probabilistic programming framework designed for variational inference: overkill for four standard distributions whose log-likelihoods are well-understood and whose gradients SciPy handles cleanly. We computed the gradients analytically. The Tweedie and Gamma gradients are in every actuarial textbook.

**One coordinate descent cycle by default.** The GAMLSS EM algorithm tradition uses multiple cycles. In practice, on UK motor and home datasets, one cycle produces nearly identical results to three or five cycles. The convergence is fast because CatBoost is initialised from unconditional MLE estimates; the first cycle does most of the work. Set `n_cycles=3` if you want to verify this holds for your data.

---

## Research context

The theoretical foundation was established by Smyth and Jørgensen in 2002. The extension to GBMs is So & Valdez (2024), the paper this library implements, which won the ASTIN Best Paper at the IAA Joint Colloquium in Brussels in September 2024. Their paper presents the maths, derives the gradients analytically for zero-inflated Tweedie CatBoost, and reports strong empirical results on telematics data. They did not release code.

The broader distributional GBM literature (Chevalier & Côté EAJ 2025, cyc-GBM, PGBM, NGBoost) is well-developed for Poisson frequency and Gamma severity models. None of it covers Tweedie. The compound Poisson-Gamma is the primary distribution for UK pure premium pricing and has been absent from every probabilistic GBM framework until now.

This library fills that gap. It is not a research prototype. It is production-quality Python with a test suite, type annotations, and an API designed for actuarial pricing workflows.

---

## Getting started

```bash
uv add insurance-distributional
```

Minimum viable motor workflow:

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)
pred = model.predict(X_val, exposure=exposure_val)

# The two numbers you need
pred.mean             # your existing technical premium
pred.volatility_score()  # new: CoV per risk, for loading and referrals

# Check the distribution model is calibrated (requires v0.1.3+)
from insurance_distributional import coverage
coverage(y_val, pred, levels=(0.80, 0.90, 0.95))
```

Source code and tests on [GitHub](https://github.com/burning-cost/insurance-distributional). The library is built around CatBoost, NumPy, SciPy, and Polars. No heavy probabilistic programming framework required.

The expected loss is necessary. It is not sufficient. Two risks at £350 are not the same risk if their conditional variances differ by a factor of two. Distributional GBM is how you quantify that difference.

---

*`insurance-distributional` is open source under MIT licence. It is the first open-source implementation of the So & Valdez ASTIN 2024 Best Paper approach. Source, tests, and documentation on [GitHub](https://github.com/burning-cost/insurance-distributional).*

- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/) - tail quantiles and TVaR; the distributional approach in this post gives you the full predictive distribution, the quantile approach gives you specific percentiles with coverage guarantees
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - distribution-free prediction intervals that work when the parametric assumptions are uncertain; a complement to the distributional GBM coverage checks
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/) - once the distributional model is fitted, the Murphy decomposition and PIT histogram diagnostics tell you whether the dispersion model is adding genuine information or overfitting
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) - the governance documentation layer: how to register a distributional pricing model, document the two-stage dispersion estimation, and produce the model risk artefact for sign-off
