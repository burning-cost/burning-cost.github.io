---
layout: page
title: "insurance-frequency-severity"
description: "Sarmanov copula joint frequency-severity modelling for UK personal lines insurance. Tests the independence assumption your pricing model makes."
permalink: /insurance-frequency-severity/
schema: SoftwareApplication
github_repo: "https://github.com/burning-cost/insurance-frequency-severity"
pypi_package: "insurance-frequency-severity"
---

Sarmanov copula joint frequency-severity modelling with GLM marginals. Tests and corrects for the dependence between claim count and average severity — the assumption every standard two-model GLM pricing framework makes and almost certainly gets wrong.

**[View on GitHub](https://github.com/burning-cost/insurance-frequency-severity)** &middot; **[PyPI](https://pypi.org/project/insurance-frequency-severity/)**

---

## The problem

Every UK motor pricing team runs two GLMs and multiplies the results:

```
Pure premium = E[N|x] × E[S|x]
```

This assumes claim count N and average severity S are independent given rating factors x. In UK motor, the NCD structure suppresses borderline claims: policyholders with frequent small claims are aware of the NCD threshold and do not report near-miss incidents. The result is a systematic negative correlation between claim count and average severity.

Vernic, Bolancé, and Alemany (2022) found this mismeasurement amounts to €5–55+ per policyholder on a Spanish auto book. The directional effect in UK motor is the same; the magnitude depends on your book.

The library gives you three tools to measure and correct for this:

1. **Sarmanov copula** — bivariate distribution for NB/Poisson frequency × Gamma/Lognormal severity. Handles the discrete-continuous mixed margins problem. Plug in your fitted GLM objects; the library estimates omega (copula parameter) via IFM.
2. **Gaussian copula** — standard approach from Czado et al. (2012). Uses PIT approximation for the discrete margin.
3. **Garrido conditional** — simplest baseline: adds N as a covariate in the severity GLM. No copula required.

---

## Installation

```bash
pip install insurance-frequency-severity
pip install "insurance-frequency-severity[neural]"   # neural two-part model (requires torch)
```

---

## Quick start

```python
import numpy as np
import statsmodels.api as sm
from insurance_frequency_severity import JointFreqSev, DependenceTest

rng = np.random.default_rng(42)
n = 5000
claim_count = rng.poisson(0.10, size=n)
avg_severity = np.where(claim_count > 0,
                        rng.gamma(shape=3.0, scale=800.0, size=n), np.nan)
X = np.column_stack([rng.normal(35, 8, n), rng.normal(5, 2, n)])

X_df = sm.add_constant(X)
freq_glm = sm.GLM(claim_count, X_df,
                  family=sm.families.NegativeBinomial(alpha=0.8)).fit()

mask = claim_count > 0
sev_glm = sm.GLM(avg_severity[mask], X_df[mask],
                 family=sm.families.Gamma(link=sm.families.links.Log())).fit()

import pandas as pd
claims_df = pd.DataFrame({"claim_count": claim_count, "avg_severity": avg_severity})

model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
model.fit(claims_df, n_col="claim_count", s_col="avg_severity")

X_new = sm.add_constant(rng.normal(35, 8, (100, 2)))
corrections = model.premium_correction(X_new)
# corrections: multiplicative adjustment to apply to the naive E[N]*E[S] estimate
```

Test for dependence before correcting:

```python
from insurance_frequency_severity import DependenceTest

test = DependenceTest()
result = test.test(claim_count, avg_severity)
print(f"Sarmanov omega: {result.omega:.4f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Reject independence: {result.significant}")
```

---

## Key API reference

### `JointFreqSev`

Primary class. Fits a Sarmanov copula to existing marginal GLMs and estimates the dependence parameter omega.

```python
JointFreqSev(
    freq_glm,       # fitted frequency GLM (statsmodels or sklearn-compatible)
    sev_glm,        # fitted severity GLM (statsmodels or sklearn-compatible)
    copula="sarmanov",   # sarmanov | gaussian | conditional
)
```

- `.fit(df, n_col, s_col)` — estimates omega via IFM (Inference Functions for Margins)
- `.premium_correction(X)` — multiplicative correction to E[N]*E[S]; returns numpy array
- `.omega_` — estimated copula parameter
- `.summary()` — prints dependence test results and correction magnitude

### `ConditionalFreqSev`

Garrido conditional method: adds N as a covariate in the severity GLM. No copula estimation required. Simplest approach for smaller books.

```python
from insurance_frequency_severity import ConditionalFreqSev

model = ConditionalFreqSev(freq_glm=freq_glm)
model.fit(claims_df, n_col="claim_count", s_col="avg_severity")
model.sev_with_n_glm   # re-fitted severity GLM including N
```

### Copula classes

```python
from insurance_frequency_severity import (
    SarmanovCopula,         # discrete-continuous Sarmanov
    GaussianCopulaMixed,    # Gaussian with PIT approximation for count margin
    FGMCopula,              # Farlie-Gumbel-Morgenstern (weak dependence)
)
```

All copula classes implement `.fit(n, s, freq_glm, sev_glm)`, `.omega_`, `.loglik_`, and `.aic_`.

### `DependenceTest`

Tests independence of claim count and severity given covariates.

```python
from insurance_frequency_severity import DependenceTest

test = DependenceTest()
result = test.test(claim_count, avg_severity, X=X)
# result.omega, result.p_value, result.significant
```

### `CopulaGOF`

Goodness-of-fit tests for copula selection.

```python
from insurance_frequency_severity import CopulaGOF, compare_copulas

gof = CopulaGOF(fitted_model)
gof.pit_test()         # probability integral transform uniformity test
gof.tail_dependence()  # upper/lower tail dependence coefficients

results = compare_copulas(sarmanov_model, gaussian_model)
# AIC/BIC/LogLik comparison table
```

### `JointModelReport`

Produces a governance-readable report with dependence statistics, correction magnitudes, and model fit.

```python
from insurance_frequency_severity import JointModelReport

report = JointModelReport(model)
report.to_markdown("fs_report.md")
```

### Neural two-part model (optional)

```python
from insurance_frequency_severity.dependent import DependentFSModel

model = DependentFSModel(hidden_dim=64, n_layers=3)
model.fit(X_train, n_train, s_train)
model.predict_premium(X_new)
```

---

## Related blog posts

- [Your Frequency-Severity Independence Assumption Is Costing You Premium](https://burning-cost.github.io/2026/03/08/frequency-severity-independence-is-costing-you-premium/)
- [CatBoost Frequency-Severity on freMTPL2](https://burning-cost.github.io/2026/03/22/catboost-frequency-severity-fremtpl2-insurance-pricing/)
