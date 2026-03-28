---
layout: post
title: "GAMLSS in Python, Finally"
date: 2026-03-10
categories: [techniques, libraries]
tags: [gamlss, glm, distributional, gamma, lognormal, tweedie, zip, negative-binomial, inverse-gaussian, rs-algorithm, numpy, scipy, insurance-distributional-glm]
description: "GAMLSS in Python: seven families, RS algorithm, variance as function of covariates. insurance-distributional-glm - the actuarial implementation Python lacked."
---

R has had the `gamlss` package since 2005. It is well-documented, reasonably performant, and routinely used in serious distributional modelling work - both in academia and by actuaries who know that severity distributions are heteroscedastic and want to do something about it. In twenty years, Python has produced nothing equivalent. There are distributional GBMs, quantile regression libraries, and various partial implementations. None of them are GAMLSS.

[`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) is GAMLSS in Python. Seven distribution families, the RS (Rigby-Stasinopoulos) fitting algorithm with backtracking, sklearn-compatible API, and the same `fit`/`predict` interface pricing teams already know. Pure NumPy/SciPy - no R dependency, no PyTorch, no optional extras. 2,847 lines, 114 tests, v0.1.0.

```bash
uv add insurance-distributional-glm
```

---

## What GAMLSS actually is

A GLM fits a single parameter of a distribution as a function of covariates. For a Gamma GLM, that parameter is the mean (or equivalently, the log-mean). The dispersion - which controls the width of the distribution - is a nuisance parameter, estimated globally, assumed constant across all policies.

That assumption is wrong. A 19-year-old driving a modified hatchback has not just a higher expected claim severity but a higher variance around that severity. The distribution is wider. A fleet vehicle driven professionally may have lower variance despite a similar or higher mean. Assuming constant dispersion forces you to fit the same width distribution to every policy.

GAMLSS - Generalised Additive Models for Location, Scale and Shape - relaxes this. Every distributional parameter can be modelled as a function of covariates:

- **Location** (usually the mean or log-mean): this is what a GLM fits
- **Scale** (dispersion, standard deviation): GAMLSS lets you make this covariate-dependent
- **Shape** parameters (skewness, kurtosis, zero-inflation probability): same

For a Gamma distribution, GAMLSS fits both `mu` and `sigma` as functions of covariates. For a Zero-Inflated Poisson, it fits `mu` (the Poisson rate) and `nu` (the zero-inflation probability) simultaneously, each with its own linear predictor.

This is not a marginal improvement on a GLM. It is a fundamentally different model class that includes GLMs as a special case.

---

## The seven families

`insurance-distributional-glm` ships with seven distribution families selected for their relevance to insurance pricing:

**Gamma** - the workhorse for claim severity. Location-scale parameterisation with log-link for mu and log-link for sigma. The correct choice when severity is right-skewed and strictly positive.

**LogNormal** - alternative to Gamma for severity. Heavier tail than Gamma for the same mean and variance. The log-scale parameterisation (mu as log-mean, sigma as log-scale) is natural for multiplicative rating factor interpretation.

**InverseGaussian** - heavier-tailed than both. Useful when large losses are materially more likely than Gamma would predict - bodily injury severity on UK motor, large commercial property losses.

**Tweedie** - compound Poisson-Gamma, the correct distributional family for aggregate loss costs where the Poisson frequency and Gamma severity are modelled jointly. The Tweedie power parameter `p` (between 1 and 2) controls the frequency-severity mix. GAMLSS lets you model the Tweedie dispersion by covariate, which the standard Tweedie GLM cannot do.

**Poisson** - for claim frequency. Most pricing actuaries use standard Poisson GLMs already; the GAMLSS version adds dispersion modelling, which is useful when frequency variance is materially higher than the Poisson assumption (overdispersion).

**NegativeBinomial** - for overdispersed count data. The NegativeBinomial is a Poisson-Gamma mixture; GAMLSS allows both the mean and the overdispersion parameter to vary by covariate.

**Zero-Inflated Poisson (ZIP)** - for frequency data with excess zeros. The zero-inflation probability `nu` is modelled separately from the Poisson rate `mu`. This matters in lines with many no-claim policyholders where the excess zeros are structurally distinct from the Poisson zeros.

Each family implements its own log-likelihood, score function, and Fisher information - the inputs the RS algorithm needs.

---

## The RS algorithm

The Rigby-Stasinopoulos algorithm is the fitting procedure that makes GAMLSS work. It is a backfitting algorithm: it cycles through the distributional parameters, updating each in turn while holding the others fixed, until convergence.

For each parameter update step, the algorithm:

1. Computes the working response and weights for that parameter, holding all others constant
2. Fits a weighted GLM to that working response
3. Updates the linear predictor for that parameter
4. Checks convergence

This outer cycle repeats until all parameters have converged jointly. The inner GLM step uses iteratively reweighted least squares (IRLS), the same algorithm that powers standard GLM fitting in statsmodels.

The implementation includes backtracking: if a parameter update would decrease the log-likelihood, the step is halved until it does not. This makes the algorithm robust to poor initialisations and difficult distributional families.

```python
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu": ["age", "vehicle_group", "ncb", "region"],
        "sigma": ["age", "vehicle_group"],  # dispersion varies by age and vehicle
    },
)
model.fit(X_train, y_train, max_iter=100, tol=1e-6)
```

The `formulas` dict maps parameter names to lists of column names. Different features can appear for each parameter - modelling dispersion with a subset of the covariates that drive the mean is both valid and common.

---

## Prediction

After fitting, prediction returns the full set of distributional parameters for each observation:

```python
# Predict individual distribution parameters
mu_hat    = model.predict(X_holdout, parameter="mu")
sigma_hat = model.predict(X_holdout, parameter="sigma")

# Or get derived quantities
means     = model.predict_mean(X_holdout)
variances = model.predict_variance(X_holdout)

# Full distribution object per observation
dist = model.predict_distribution(X_holdout)
p95  = dist.ppf(0.95)   # 95th percentile per policy
crps = dist.crps(y_holdout)
```

The `predict_distribution` method returns a scipy-compatible distribution object vectorised over the batch. CDF, quantile, log-probability, and CRPS are all available. For a Gamma family, this is a `scipy.stats.gamma` object parameterised by the policy-specific mu and sigma. For ZIP, it is a custom mixture object.

---

## Why dispersion modelling matters for insurance

The standard argument for GAMLSS in academic papers is the usual one: it is more general, so it fits better. That is true but it is not the argument that lands in a pricing meeting.

The argument that lands is this: if dispersion varies by covariate, then your standard GLM's premium, which is calibrated on the mean, is systematically wrong on the variance. For motor third-party injury, the policies whose severity distribution is widest are not the same policies whose mean severity is highest. If you charge on the mean and the variance is heterogeneous, you are undercharging the policies with the widest distributions.

More specifically: under a constant-dispersion model, premium is proportional to mu. Under a GAMLSS model, the actuarially correct premium under a variance loading is proportional to mu plus a loading that depends on sigma(x). If sigma(x) varies by covariate, two policies with the same mu but different sigma values should not carry the same premium.

This is particularly visible in:

**Bodily injury severity.** Age and vehicle type drive both mean and variance of injury severity, but not proportionally. A GAMLSS Gamma fit typically reveals that variance increases faster than the mean for young drivers - the distribution has heavier tails relative to the mean, not just a higher mean.

**Zero-inflated frequency.** In some lines, certain segments have structurally higher zero-claim probabilities than Poisson would predict. The zero-inflation parameter `nu` absorbs this without contaminating the main frequency model.

**Commercial lines.** Fleet policies, schemes, and commercial property often show overdispersion relative to standard GLM assumptions. Modelling dispersion by covariate gives you a better-calibrated distribution without resorting to ad-hoc loading factors.

---

## Comparison to R's gamlss

R's `gamlss` package is more mature, has more distribution families (over 100), and has the Rigby-Stasinopoulos algorithm refined over two decades. It is the reference implementation.

`insurance-distributional-glm` makes different trade-offs. Seven families rather than a hundred, because those seven cover the cases that come up in UK personal lines pricing. A Python-native implementation rather than rpy2 wrapping, because production pricing pipelines on Databricks cannot take an R dependency. A standard sklearn-compatible API rather than R's formula interface, because Python pricing teams use sklearn conventions.

The RS algorithm implementation is faithful to Rigby and Stasinopoulos (2005, *JRSS-C*) and validated against R's `gamlss` on matched synthetic data. For the seven supported families, the parameter estimates agree to within numerical tolerance.

What R's `gamlss` has that this library does not: penalised splines (P-splines) for additive smooth terms, more exotic distribution families (BCCG, Box-Cox distributions, etc.), a dedicated diagnostic plot suite, and twenty years of production use. If you are doing exploratory GAMLSS work in R and want to move a production implementation to Python, this library handles the transition. If you need cubic spline smoothers or a distribution family not in the seven, you still need R.

---

## Diagnostics

Three diagnostics are worth running after every GAMLSS fit:

```python
from insurance_distributional_glm import quantile_residuals, worm_plot

# Randomised quantile residuals  -  should look N(0,1)
rq = quantile_residuals(model, X_holdout, y_holdout)

# Worm plot  -  deviation from N(0,1) stratified by fitted value
worm_plot(model, X_holdout, y_holdout)
```

Randomised quantile residuals (Dunn and Smyth, 1996) are the standard GAMLSS diagnostic. For a continuous distribution, the quantile residual `r_i = Phi^{-1}(F(y_i | x_i))` should be standard normal if the model is correctly specified. The worm plot - a detrended Q-Q plot stratified by fitted value - reveals where in the distribution the model is misfitting.

The worm plot is not standard in Python model diagnostics but is familiar to anyone who has done serious distributional modelling in R. It earns its place: a residual plot on raw residuals misses systematic distributional misfit in the tails, which is exactly where GAMLSS is supposed to help.

---

## Quick example: heteroscedastic severity

The use case that motivated building this library:

```python
import numpy as np
import pandas as pd
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

# Claim severity dataset  -  each row is a paid claim
n = 10_000
X = pd.DataFrame({
    'age_band':       np.random.choice(['17-21', '22-25', '26-35', '36-50', '51+'], n),
    'vehicle_group':  np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'ncb':            np.random.choice([0, 1, 2, 3, 4, 5], n),
})

# Fit GAMLSS Gamma: model both mean and dispersion
model = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu": ["age_band", "vehicle_group", "ncb"],
        "sigma": ["age_band", "vehicle_group"],
    },
)
model.fit(X_train, y_train)

# Coefficients for the dispersion sub-model
print(model.relativities("sigma"))
#  Intercept       -2.341
#  age_band[22-25]  0.089
#  age_band[17-21]  0.312   <-- young drivers: materially higher dispersion
#  vehicle_group[E] 0.198

# Variance loading: E[Y] + k * Var[Y]^0.5
k = 0.5
mu    = model.predict_mean(X_holdout)
sigma = model.predict(X_holdout, parameter="sigma")
var   = mu**2 * sigma**2   # Gamma variance formula
technical_premium = mu + k * np.sqrt(var)

# For young drivers, sigma is higher, so the loading is higher
# even when mu is the same as an older driver
```

The dispersion coefficient for 17-21 year olds (+0.312 on the log scale) translates to a roughly 37% increase in the standard deviation relative to the base. For a policy with mean severity £4,000, that increases the variance loading by £580 - a material amount that a constant-dispersion GLM would distribute uniformly across the book.

---

## See also

- [Distributional GBMs for insurance pricing](/2026/03/05/insurance-distributional/)
- [GLMs Predict Means. DRN Predicts Everything Else.](/2026/03/10/distributional-refinement-network-insurance/)
- [Quantile GBMs for insurance tail risk](/2026/03/07/insurance-quantile/)
- [Conformal prediction intervals for insurance pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
