---
layout: post
title: "Double GLM for Insurance: Every Risk Gets Its Own Dispersion"
date: 2026-03-11
categories: [techniques, libraries]
tags: [glm, dglm, dispersion, tweedie, gamma, inverse-gaussian, poisson, negative-binomial, irls, reml, solvency-ii, margin-loading, python, insurance-dispersion]
description: "Double GLM gives every UK insurance policy its own dispersion parameter. insurance-dispersion - policy-level Solvency II variance and risk-adequate loading."
---

R actuaries have had access to the `dglm` package since Smyth and Dunn wrote it in the late 1990s. It gets around 12,000 CRAN downloads a month. The paper behind it - Smyth (1989), *JRSS-B* 51:47-60 - is one of the more cited pieces of statistical methodology in the actuarial literature.

Python has had nothing equivalent. statsmodels issue #7639, opened in 2020, asks for Double GLM support. No PR has been merged. Five years later, the issue is still open.

[`insurance-dispersion`](https://github.com/burning-cost/insurance-dispersion) fills that gap. It is a Python implementation of the Double GLM (DGLM): joint modelling of mean AND dispersion, where every observation gets its own dispersion parameter phi_i rather than one scalar phi estimated for the entire portfolio. Six distribution families, REML correction, overdispersion likelihood ratio test, mean and dispersion factor tables. Pure NumPy/SciPy and formulaic - no ML dependencies. 1,651 lines, 79 tests, v0.1.0.

```bash
uv add insurance-dispersion
```

---

## What the standard GLM gets wrong about dispersion

A standard GLM models the conditional mean `E[Y|x] = mu_i`. The variance of each observation is:

```
Var[Y_i] = phi * V(mu_i)
```

where `V(mu)` is the variance function (mu^2 for Gamma, mu for Poisson, mu^p for Tweedie) and `phi` is the dispersion parameter. That phi is a single scalar - estimated once, applied equally to every risk in the portfolio.

This means: if the mean model says two policies have the same mu, the GLM assigns them identical variance. That is often wrong. A 19-year-old driving a modified hatchback in Manchester does not just have a higher expected severity than a 45-year-old in a standard family car - the distribution around that severity is wider. The 19-year-old's best outcomes and worst outcomes are further apart. The GLM cannot see this. Its one-phi-fits-all assumption forces the same relative uncertainty onto every risk.

The Double GLM relaxes this directly:

```
Mean submodel:       g(mu_i) = x_i^T beta
Dispersion submodel: h(phi_i) = z_i^T alpha
Var[Y_i]           = phi_i * V(mu_i)
```

phi_i is now a function of covariates z_i. The mean model and dispersion model can use different feature sets - it is entirely valid (and common) to have age and vehicle group in both, while the mean model also carries NCD and the dispersion model omits it. The two models are coupled but not identical.

---

## The algorithm: alternating IRLS

The fitting procedure is alternating iteratively reweighted least squares. At each iteration:

**Mean step.** Fix the current phi_i estimates. Run one IRLS step of a weighted GLM, using weights 1/phi_i. Update mu_i.

**Dispersion step.** Fix the updated mu_i. Compute the unit deviances:

```python
d_i = dev_resid(y_i, mu_i, phi=1)   # raw unit deviance per observation
```

Fit a Gamma GLM with log link to the d_i values. The expectation of d_i under the DGLM is phi_i, so the Gamma GLM recovers exp(z_i^T alpha) = phi_i without bias. Update alpha, then phi_i.

Repeat until the relative change in the dispersion deviance drops below 1e-7. Convergence typically takes 5-20 iterations on insurance-scale datasets.

```python
from insurance_dispersion import DGLM
import insurance_dispersion.families as fam
import pandas as pd

model = DGLM(
    formula="claim_severity ~ age_band + vehicle_group + ncb",
    dformula="~ age_band + vehicle_group",
    family=fam.Gamma(),
    data=df_train,
    method="reml",      # REML correction on by default
)
result = model.fit()
```

The `method='reml'` default applies the Smyth-Verbyla (1999) correction. It adjusts the dispersion pseudo-responses by the hat diagonal from the mean model - removing the contribution of estimating the mean parameters from the dispersion likelihood. For insurance mean models with many factor levels, REML materially reduces the bias in phi estimates. We use it by default.

---

## Why this matters for insurance pricing

### Risk-adequate margin loading

Standard premium loading under a variance principle is:

```
premium_i = mu_i + k * sqrt(Var[Y_i])
           = mu_i + k * sqrt(phi * V(mu_i))   # standard GLM
           = mu_i + k * sqrt(phi_i * V(mu_i)) # DGLM
```

Under a standard GLM, two risks with the same mu get the same loading - because phi is constant. Under the DGLM, the loading varies with phi_i. A segment with twice the dispersion of another carries a 41% higher uncertainty loading even when the pure premiums are identical.

This is not a theoretical nicety. In UK motor bodily injury, young drivers and high-performance vehicles carry materially wider severity distributions than the mean model captures. Loading them identically to standard risks cross-subsidises in the wrong direction: volatile risks pay the same loading as stable risks, so the stable risks overpay and the volatile risks underpay.

### Solvency II capital at policy level

The SCR calculation at 99.5th percentile requires `Var[Y_i]` at individual policy level - or at minimum, at a granular segment level. Without the DGLM, you have three options: assume phi constant (wrong), use separate models per segment (loses credibility), or move to GAMLSS (requires framework adoption and is harder to explain to a model risk committee than two coupled GLMs).

The DGLM delivers individual-level variance with a workflow that plugs into existing GLM infrastructure. The result object exposes this directly:

```python
result = model.fit()

# Per-policy variance prediction
var_pred = result.predict(df_holdout, which="variance")
# var_pred = phi_i * V(mu_i) per observation

# 99.5th percentile loading for SCR
from scipy import stats
p995_loading = stats.norm.ppf(0.995) * np.sqrt(var_pred)
```

### Tweedie: frequency and severity moving independently

With constant phi, the Tweedie distribution constrains frequency and severity to move proportionally for a given mean. If mu doubles, both the frequency component and the severity component scale in a fixed ratio determined by the Tweedie power p. You cannot have frequency increase faster than severity in one segment without changing p globally.

With phi_i varying by segment, the Tweedie dispersion absorbs segment-specific variation in the frequency-severity mix. A segment with higher phi can have proportionally more severity variance relative to its mean than a low-phi segment with the same pure premium. This is the actuarially correct behaviour for lines where frequency and severity drivers are genuinely different.

---

## Reading the results

After fitting, the result object gives you the same outputs you would expect from a GLM, plus the dispersion-specific ones:

```python
result = model.fit()

# Fitted phi per training observation
result.phi_                 # array of length n_train

# Log-likelihood
result.loglik
result.aic

# Predict on new data
mu_hat  = result.predict(df_holdout, which="mean")
phi_hat = result.predict(df_holdout, which="dispersion")
var_hat = result.predict(df_holdout, which="variance")
```

---

## Overdispersion LRT

Before committing to a DGLM, you want to test whether dispersion actually varies - or whether the constant-phi GLM is adequate. The library provides a likelihood ratio test:

```python
lrt = result.overdispersion_test()

print(f"LRT statistic: {lrt['statistic']:.1f}  df: {lrt['df']}  p-value: {lrt['p_value']:.3e}")
print(lrt['conclusion'])
# LRT statistic: 187.4  df: 6  p-value: < 0.001
# Reject constant phi (dispersion varies by covariates)
```

The test statistic is -2 times the difference in log-likelihoods between the DGLM and the constrained model (constant phi). Under H0 it is asymptotically chi-squared with degrees of freedom equal to the number of dispersion parameters minus one (the intercept). A p-value below 0.05 tells you the constant-phi assumption is rejected.

In our experience on UK motor severity data: the LRT almost always rejects for models with age and vehicle group in the dispersion formula. The dispersion variation is real and material. The question is not usually whether to model it, but which covariates belong in the dispersion formula.

---

## Factor tables

The mean relativities and dispersion relativities are both available in the standard actuarial format:

```python
# Mean relativities — familiar exp(beta) table
mean_table = result.mean_relativities()
print(mean_table)
#                    relativity  lower_95  upper_95
# age_band
# 17-21               1.000        1.000     1.000   (base)
# 22-25               0.847        0.811     0.884
# 26-35               0.721        0.693     0.750
# 36-50               0.668        0.643     0.694
# 51+                 0.702        0.674     0.731
#
# vehicle_group
# A                   1.000        1.000     1.000   (base)
# B                   1.089        1.054     1.125
# C                   1.163        1.127     1.200
# D                   1.291        1.248     1.335
# E                   1.487        1.433     1.543

# Dispersion relativities — new
disp_table = result.dispersion_relativities()
print(disp_table)
#                    phi_relativity  lower_95  upper_95
# age_band
# 17-21               1.000           1.000     1.000   (base)
# 22-25               0.874           0.801     0.954
# 26-35               0.731           0.673     0.794
# 36-50               0.663           0.610     0.720
# 51+                 0.712           0.654     0.775
#
# vehicle_group
# A                   1.000           1.000     1.000   (base)
# B                   1.043           0.985     1.104
# C                   1.112           1.052     1.176
# D                   1.224           1.157     1.295
# E                   1.398           1.318     1.482
```

The dispersion relativities are multiplicative in the same sense as the mean relativities. A phi_relativity of 1.398 for vehicle group E means the conditional variance of a group-E claim is 39.8% higher than a group-A claim at the same mean severity. That gap translates directly into a higher uncertainty loading under any variance-based premium formula.

---

## The full example: UK motor severity

```python
import numpy as np
import pandas as pd
from insurance_dispersion import DGLM
import insurance_dispersion.families as fam

# UK motor severity dataset — one row per paid claim
np.random.seed(42)
n = 8000

df = pd.DataFrame({
    "age_band":      np.random.choice(["17-21","22-25","26-35","36-50","51+"], n,
                         p=[0.08, 0.12, 0.30, 0.32, 0.18]),
    "vehicle_group": np.random.choice(["A","B","C","D","E"], n,
                         p=[0.20, 0.25, 0.25, 0.20, 0.10]),
    "ncb":           np.random.choice([0,1,2,3,4,5], n,
                         p=[0.10, 0.12, 0.15, 0.20, 0.22, 0.21]),
    "claim_severity": np.random.gamma(shape=2.0, scale=2000.0, size=n),
})

train = df.sample(frac=0.8, random_state=0)
test  = df.drop(train.index)

model = DGLM(
    formula="claim_severity ~ age_band + vehicle_group + ncb",
    dformula="~ age_band + vehicle_group",
    family=fam.Gamma(),
    data=train,
    method="reml",
)
result = model.fit()

# Overdispersion test first — confirm we need the DGLM
lrt = result.overdispersion_test()
print(f"LRT p-value: {lrt['p_value']:.4f}")

# Mean and dispersion factor tables
print(result.mean_relativities())
print(result.dispersion_relativities())

# Risk-adequate premium: mu + k * sqrt(phi * V(mu))
k = 0.5   # half-standard-deviation loading
mu  = result.predict(test, which="mean")
phi = result.predict(test, which="dispersion")
var = phi * mu**2   # Gamma variance: phi * mu^2

technical_premium = mu + k * np.sqrt(var)

# Two risks with same mean severity but different vehicle groups
# group A (phi_rel=1.0) vs group E (phi_rel=1.398)
# Same mu = £3,000: loading difference
mu0, phi_A, phi_E = 3000.0, 1.0, 1.398
loading_A = k * np.sqrt(phi_A * mu0**2)  # = 1,500
loading_E = k * np.sqrt(phi_E * mu0**2)  # = 1,774
print(f"Margin loading — Group A: £{loading_A:.0f}, Group E: £{loading_E:.0f}")
# Margin loading — Group A: £1,500, Group E: £1,774
```

The £274 per-policy difference in loading comes entirely from the dispersion model. The mean model assigns these two risks identical pure premiums. Without the DGLM, you cannot make this distinction at all.

---

## Relationship to GAMLSS

Our [`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) library implements GAMLSS: a broader framework where every distributional parameter - mean, dispersion, shape, zero-inflation probability - can be modelled as a function of covariates. GAMLSS includes DGLM as a special case.

So why would you use the DGLM when GAMLSS exists?

Because the workflows are different. GAMLSS requires adopting a new modelling framework with its own fitting algorithm, diagnostics, and mental model. The DGLM slots into existing GLM infrastructure - same families, same link functions, same interpretation of factor tables. If your pricing team already has a working Gamma GLM for severity, the DGLM is an extension that adds a second formula and a second factor table. GAMLSS is a replacement.

For teams that want to add dispersion modelling without rebuilding their pricing stack, the DGLM is the right tool. For teams doing novel distributional work - modelling zero-inflation or shape parameters by covariate - GAMLSS is the right tool.

---

## The Python gap

R's `dglm` package has had this functionality since 1998. Around 12,000 downloads a month on CRAN. Smyth and Dunn wrote it alongside the 1989 theory. The method is established actuarial practice in R pricing teams.

Python pricing teams have not had access to it. The statsmodels issue asking for DGLM support has been open since 2020 with no resolution. The Python actuarial ecosystem has distributional GBMs, GAMLSS, quantile regression - but not the most basic tool for heteroscedastic GLM work.

`insurance-dispersion` fills that gap. Six distribution families (Gaussian, Gamma, Inverse Gaussian, Tweedie, Poisson, Negative Binomial), formulaic-based formula interface, REML correction, overdispersion LRT, and factor tables in the format pricing teams expect.

---

## Getting started

```bash
uv add insurance-dispersion
```

Minimum viable workflow for Gamma severity:

```python
from insurance_dispersion import DGLM
import insurance_dispersion.families as fam

model = DGLM(
    formula="claim_severity ~ age_band + vehicle_group + ncb + region",
    dformula="~ age_band + vehicle_group",
    family=fam.Gamma(),
    data=df_train,
)
result = model.fit()

# Test whether DGLM is warranted
lrt = result.overdispersion_test()

# Factor tables
result.mean_relativities()
result.dispersion_relativities()

# Per-policy variance for Solvency II or margin loading
result.predict(df_holdout, which="variance")
```

Source and tests on [GitHub](https://github.com/burning-cost/insurance-dispersion).

One scalar phi for the whole portfolio is an assumption, not a fact. The DGLM lets you test that assumption, reject it when the data warrants it, and price accordingly.


- [GAMLSS in Python, Finally](/2026/03/10/insurance-distributional-glm/) - the broader framework when you need shape and zero-inflation parameters, not just dispersion
- [Distributional GBMs for insurance pricing](/2026/03/05/insurance-distributional/) - per-risk dispersion estimation via gradient boosting rather than GLMs
- [GLMs Predict Means. DRN Predicts Everything Else.](/2026/03/10/distributional-refinement-network-insurance/) - neural refinement of GLM/GBM predictive distributions
- [Quantile GBMs for insurance tail risk](/2026/03/07/insurance-quantile/) - TVaR and ILF estimation without distributional assumptions
- [Conformal prediction intervals for insurance pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - distribution-free coverage guarantees when parametric assumptions are uncertain
