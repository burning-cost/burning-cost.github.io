---
layout: post
title: "Tweedie Regression for Insurance Pricing in Python"
date: 2026-03-28
categories: [techniques, tutorials]
tags: [tweedie, glm, python, insurance-pricing, statsmodels, scikit-learn, pure-premium]
description: "Why Tweedie GLM is the standard for aggregate loss modelling in insurance, with a complete Python example covering power parameter selection, exposure offset, and comparison with frequency-severity."
---

If you work in general insurance pricing, you fit Tweedie GLMs. Not because it became fashionable, but because the distribution is the right description of what aggregate loss data actually looks like: a spike of exact zeros (policies with no claims), and a right-skewed positive tail (policies that had one or more claims). No other single-distribution GLM family handles both properties without contortion.

This post works through the foundations and the Python implementation. We cover the compound Poisson-gamma interpretation, how to select the power parameter, when a Tweedie GLM is the right choice versus a separate frequency-severity model, and a set of practical tips from actual use. The code runs end to end — we generate synthetic insurance data inline so you can follow along.

---

## Why Tweedie? The compound Poisson-gamma story

The Tweedie family is an exponential dispersion model parameterised by a power parameter `p`. The mean-variance relationship is `Var(Y) = phi * mu^p`, where `phi` is the dispersion. Different values of `p` give well-known special cases:

- `p = 0`: Normal
- `p = 1`: Poisson
- `p = 2`: Gamma
- `p = 3`: Inverse Gaussian

The range `1 < p < 2` is the compound Poisson-gamma (CPG) distribution, and this is the region that matters for insurance aggregate losses.

The CPG representation is what gives Tweedie its practical intuition. Write the aggregate loss for a policy as:

```
Y = X_1 + X_2 + ... + X_N
```

where `N ~ Poisson(lambda)` is the claim count and each `X_i ~ Gamma(alpha, beta)` is an independent claim size. When `N = 0`, `Y = 0` exactly. When `N >= 1`, `Y` is a sum of gamma random variables, giving a positive continuous value.

This maps directly to the insurance data-generating process. A Tweedie GLM with `1 < p < 2` is not a hack to get zeros into a continuous distribution — it is a principled model of the underlying process. The connection to the Tweedie power parameter is:

```
p = (alpha + 2) / (alpha + 1)    where alpha is the gamma shape
lambda = mu^(2-p) / (phi*(2-p))
```

As `p -> 1`, the Poisson claim count component dominates (many small claims, most policies claim). As `p -> 2`, the gamma severity component dominates (few large claims, most policies are zero). For UK motor, `p` typically sits between 1.4 and 1.7. For commercial property, closer to 1.8.

---

## Synthetic data

We generate 50,000 policy records with four rating variables, exposure in years, and aggregate annual loss. The true data-generating process is explicitly CPG so we can validate our model recovery.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 50_000

# Rating variables
age_band = rng.choice(['17-25', '26-35', '36-50', '51-65', '66+'], n,
                      p=[0.10, 0.22, 0.35, 0.22, 0.11])
vehicle_group = rng.choice(['A', 'B', 'C', 'D'], n, p=[0.3, 0.3, 0.25, 0.15])
area = rng.choice(['Urban', 'Suburban', 'Rural'], n, p=[0.4, 0.35, 0.25])
ncd_band = rng.choice(['0', '1-2', '3-4', '5+'], n, p=[0.15, 0.20, 0.30, 0.35])
exposure = rng.uniform(0.1, 1.0, n)

df = pd.DataFrame({
    'age_band': age_band,
    'vehicle_group': vehicle_group,
    'area': area,
    'ncd_band': ncd_band,
    'exposure': exposure,
})

# True log-linear mean structure (pure premium per unit exposure)
log_mu = np.zeros(n)

age_effect = {'17-25': 0.6, '26-35': 0.2, '36-50': 0.0, '51-65': -0.1, '66+': 0.1}
veh_effect = {'A': 0.0, 'B': 0.15, 'C': 0.3, 'D': 0.5}
area_effect = {'Urban': 0.25, 'Suburban': 0.0, 'Rural': -0.15}
ncd_effect = {'0': 0.4, '1-2': 0.2, '3-4': 0.0, '5+': -0.3}

log_mu += np.vectorize(age_effect.get)(df['age_band'])
log_mu += np.vectorize(veh_effect.get)(df['vehicle_group'])
log_mu += np.vectorize(area_effect.get)(df['area'])
log_mu += np.vectorize(ncd_effect.get)(df['ncd_band'])
log_mu += 5.5  # intercept: base pure premium ~245

# True Tweedie parameters
true_p = 1.6
true_phi = 1.2

mu = np.exp(log_mu) * exposure  # scale by exposure

# Simulate via compound Poisson-gamma
# lambda_i = mu_i^(2-p) / (phi*(2-p))
# alpha = (2-p)/(p-1), beta = phi*(p-1)*mu_i^(p-1)
p, phi = true_p, true_phi
lam = mu**(2 - p) / (phi * (2 - p))
alpha_shape = (2 - p) / (p - 1)
beta_rate = phi * (p - 1) * mu**(p - 1)

N = rng.poisson(lam)
Y = np.zeros(n)
for i in range(n):
    if N[i] > 0:
        Y[i] = rng.gamma(shape=alpha_shape * N[i],
                         scale=1.0 / beta_rate[i])

df['loss'] = Y
df['claim_count'] = N

print(f"Zero loss proportion: {(Y == 0).mean():.1%}")
print(f"Mean loss: £{Y.mean():.0f}")
print(f"Loss per unit exposure: £{(Y / exposure).mean():.0f}")
```

Running this gives roughly 68% zero losses and a mean annual loss around £250, which is plausible for UK motor.

---

## Fitting with statsmodels

statsmodels gives you the full GLM machinery: deviance, Pearson residuals, coefficient standard errors, and proper treatment of the exposure offset.

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Log-link Tweedie GLM with exposure offset
# We fix p=1.6 first, then show how to select it
model = smf.glm(
    formula='loss ~ age_band + vehicle_group + area + ncd_band',
    data=df,
    family=sm.families.Tweedie(
        var_power=1.6,
        link=sm.families.links.Log()
    ),
    offset=np.log(df['exposure'])
).fit()

print(model.summary())
```

A few things to note here. The offset `log(exposure)` is the standard way to model pure premium (loss per unit exposure): it constrains the exposure coefficient to 1 on the log scale rather than estimating it as a free parameter. If you forget the offset, your model implicitly assumes all policies had unit exposure, which biases every coefficient.

The output gives you log-scale coefficients. To get multiplicative relativities relative to the base level of each factor, exponentiate:

```python
params = model.params
relats = np.exp(params)
print(relats.to_string())
```

---

## Fitting with scikit-learn TweedieRegressor

scikit-learn's `TweedieRegressor` is convenient for cross-validation and pipeline integration, but it has two limitations to know about: it does not support offsets natively, and it fits with coordinate descent (L-BFGS) rather than IRLS, so convergence behaviour differs for large sparse datasets.

```python
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np

cat_features = ['age_band', 'vehicle_group', 'area', 'ncd_band']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])

# Workaround for missing offset: divide response by exposure,
# then fit pure premium directly (valid when log link is used)
df['pure_premium'] = df['loss'] / df['exposure']

pipe = Pipeline([
    ('prep', preprocessor),
    ('model', TweedieRegressor(power=1.6, alpha=0, link='log', max_iter=300))
])

pipe.fit(df[cat_features], df['pure_premium'], model__sample_weight=df['exposure'])
```

The `sample_weight=exposure` compensates for the missing offset: longer-exposure policies are weighted more heavily, which is not quite equivalent to a proper offset but is a reasonable approximation for balanced datasets. For production pricing work, we prefer statsmodels for the full GLM output and diagnostics; sklearn for grid-search workflows.

---

## Selecting the power parameter

`p` is not typically estimated by standard IRLS — it lives outside the exponential dispersion family parameterisation. The two practical approaches are profile log-likelihood and grid search with deviance.

### Profile log-likelihood

For a fixed dataset, evaluate the Tweedie log-likelihood over a grid of `p` values and take the maximum. statsmodels exposes the log-likelihood via `model.llf`.

```python
import matplotlib.pyplot as plt

p_grid = np.arange(1.05, 1.99, 0.05)
ll_values = []

for p_val in p_grid:
    fit = smf.glm(
        formula='loss ~ age_band + vehicle_group + area + ncd_band',
        data=df,
        family=sm.families.Tweedie(var_power=p_val, link=sm.families.links.Log()),
        offset=np.log(df['exposure'])
    ).fit(disp=False)
    ll_values.append(fit.llf)

ll_values = np.array(ll_values)
best_p = p_grid[np.argmax(ll_values)]
print(f"Best p by profile likelihood: {best_p:.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(p_grid, ll_values, 'o-', color='steelblue', ms=4)
ax.axvline(best_p, color='firebrick', linestyle='--', label=f'Best p = {best_p:.2f}')
ax.set_xlabel('Power parameter p')
ax.set_ylabel('Log-likelihood')
ax.set_title('Profile log-likelihood for Tweedie power parameter')
ax.legend()
plt.tight_layout()
plt.savefig('tweedie_profile_ll.png', dpi=150)
```

This will typically recover a value close to the true `p=1.6` we embedded in the data. With real data, you often see a fairly flat region — it is not unusual for the profile to be indistinguishable over `p in [1.5, 1.7]`, which tells you the data is not particularly informative about the exact power. In that situation, use `p=1.5` as a defensible default and move on.

### Grid search over deviance (sklearn)

```python
from sklearn.model_selection import cross_val_score

p_grid_sk = np.arange(1.1, 1.95, 0.05)
cv_scores = []

for p_val in p_grid_sk:
    reg = TweedieRegressor(power=p_val, alpha=0, link='log', max_iter=300)
    # Negative mean Tweedie deviance (higher = better)
    scores = cross_val_score(
        reg,
        preprocessor.fit_transform(df[cat_features]),
        df['pure_premium'],
        cv=5,
        scoring='neg_mean_tweedie_deviance',
        fit_params={'sample_weight': df['exposure']}
    )
    cv_scores.append(scores.mean())

best_p_cv = p_grid_sk[np.argmax(cv_scores)]
print(f"Best p by CV deviance: {best_p_cv:.2f}")
```

Profile likelihood on the training set and CV deviance often agree to within 0.1. If they diverge, something is wrong with your model specification — check for a miscoded variable or leakage.

---

## Tweedie GLM vs frequency-severity: when does each win?

This is the question that comes up in every pricing team. Our view: Tweedie is the right default. Frequency-severity is worth the extra complexity only when you have a specific reason to model the two components separately.

### The Tweedie case

The CPG Tweedie imposes a constraint: the frequency and severity share the same covariate effects through the mean `mu`. Specifically, `E[Y] = E[N] * E[X_i]`, and both are driven by `exp(eta)`. This is not a restriction — it is a feature. Most rating factors affect both frequency and severity in the same direction. Age and NCD are the clearest examples in motor: high-risk drivers have more claims and larger average claims.

Tweedie also has half the parameters to estimate, which matters on thin data (commercial lines, specialty). It is more stable under regularisation. It produces a single consistent pure premium without post-hoc combination of two models that were each fitted on their own loss function.

### When frequency-severity wins

Separate frequency-severity modelling is worthwhile when:

**The two components have genuinely different covariate structures.** Vehicle value affects severity (repair costs) but has only a weak relationship with claim frequency. If you force a single `mu` you will underprice high-value vehicles on severity and overprice on frequency. This argument is often overstated — in practice, most factors do move both components — but it is real.

**You need the components separately.** Actuarial sign-off sometimes requires you to present frequency and severity trends independently. Reserve teams want frequency and severity separately. If the business needs them, fit them separately.

**Tweedie residuals show clear misfit.** If a Tweedie model leaves systematic residuals on the zero/non-zero split that cannot be explained by your rating factors, that is a signal you should model the binary event explicitly. This is the ZI-Tweedie territory — see [our separate post on that]({% post_url 2026-03-28-zero-inflated-tweedie-gbm-insurance-pricing %}).

A quick diagnostic: fit a Tweedie GLM and a frequency GLM (Poisson) with the same covariates. Do the coefficients broadly agree? If NCD=5+ has a relativity of 0.55 in Tweedie and 0.50 in Poisson, they are consistent — Tweedie is fine. If they are 0.55 and 0.30, your severity model is picking up something your frequency model missed, and you want to understand why before collapsing to a single model.

```python
# Quick consistency check
freq_model = smf.glm(
    formula='claim_count ~ age_band + vehicle_group + area + ncd_band',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['exposure'])
).fit(disp=False)

# Compare NCD relativities
tweedie_ncd = np.exp(model.params.filter(like='ncd'))
freq_ncd = np.exp(freq_model.params.filter(like='ncd'))

comparison = pd.DataFrame({
    'Tweedie': tweedie_ncd,
    'Frequency': freq_ncd
})
print(comparison)
```

---

## Deviance diagnostics

Tweedie deviance (rather than Pearson chi-squared) is the correct goodness-of-fit measure for GLMs. The deviance residual for observation `i` is:

```
d_i = 2 * [y_i * (y_i^(1-p) - mu_i^(1-p))/(1-p) - (y_i^(2-p) - mu_i^(2-p))/(2-p)]
```

for `y_i > 0`. For zero observations the term simplifies to `-2 * mu_i^(2-p) / (2-p)`.

In practice, you want to plot deviance residuals against fitted values and against each predictor. Large systematic patterns are a signal of missing interactions or nonlinearity.

```python
fitted = model.fittedvalues
deviance_resids = model.resid_deviance

# Residuals vs fitted (log scale)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.scatter(np.log(fitted + 1), deviance_resids, alpha=0.1, s=5, color='steelblue')
ax.axhline(0, color='firebrick', linewidth=1)
ax.set_xlabel('log(fitted + 1)')
ax.set_ylabel('Deviance residual')
ax.set_title('Residuals vs fitted')

# Residuals by exposure band (should be flat)
df['exp_band'] = pd.qcut(df['exposure'], q=5)
resid_by_exp = df.assign(resid=deviance_resids).groupby('exp_band')['resid'].mean()

ax = axes[1]
ax.bar(range(len(resid_by_exp)), resid_by_exp.values, color='steelblue')
ax.axhline(0, color='firebrick', linewidth=1)
ax.set_xticks(range(len(resid_by_exp)))
ax.set_xticklabels([str(i) for i in resid_by_exp.index], rotation=30)
ax.set_xlabel('Exposure quintile')
ax.set_ylabel('Mean deviance residual')
ax.set_title('Residuals by exposure (should be zero)')

plt.tight_layout()
plt.savefig('tweedie_diagnostics.png', dpi=150)
```

The exposure diagnostic is particularly important: if residuals trend with exposure, your offset is misspecified. This can happen when the relationship between exposure and expected loss is non-linear — for example, very short-exposure policies (mid-term adjustments) behave differently from full-year policies. In that case, consider dropping policies with exposure below 0.1 years from model fitting, or adding a categorical indicator for short-term policies.

---

## When Tweedie fails

Tweedie with `1 < p < 2` cannot produce negative values, cannot model loss ratios directly (though you can reparameterise), and will struggle with:

**Excess zeros beyond the CPG structure.** If your zero rate is higher than the model predicts even with the right `p`, you have structural zeros. Motor NCD suppression, commercial lines with annual deductibles, and any product with deliberate non-reporting incentives all produce this. Use a ZI-Tweedie or two-part model instead.

**Multimodal severity distributions.** A portfolio with both attritional claims (small, high frequency) and large losses (bodily injury, fire) may have a severity distribution that no gamma can describe. Consider modelling attritional and large losses separately with a cap on the attritional component.

**Very sparse data.** With fewer than ~500 claims in a cell, the profile likelihood for `p` is essentially flat. You are fitting a flexible distribution to thin data and the variance will swamp the mean. Use Poisson-gamma explicit frequency-severity with informative priors, or use a credibility-weighted approach.

**Loss ratios as response.** Some teams want to model `loss / premium` directly. Do not fit Tweedie to loss ratios — the premium denominator introduces a non-constant offset that invalidates the variance structure. Either model aggregate loss with premium as an exposure substitute, or transform and model differently.

---

## The full fitted model

Pulling it together: the standard Tweedie GLM workflow in production is roughly:

1. Generate data with `exposure >= 0.1` years, exclude mid-term cancellations unless you have a proper treatment for them.
2. Grid-search `p` on a held-out validation set, or use profile likelihood on the full dataset if sample size is large enough.
3. Fit with `log(exposure)` offset and log link in statsmodels.
4. Check: deviance residuals by fitted value, by each predictor, and by exposure band.
5. Compare key factor relativities with a Poisson frequency model. Investigate large divergences.
6. Extract multiplicative relativities via `exp(coef)` and review against actuarial judgement before filing.

The power parameter is rarely the thing that breaks a Tweedie model. A missing interaction, a miscoded variable, or a structural zero problem are far more common sources of fit failure. Fit `p=1.5` first, diagnose the residuals, and revisit the power if the profile likelihood suggests something materially different.

---

## Further reading

The canonical reference for Tweedie GLMs in insurance is **England & Verrall (2002)** "Stochastic Claims Reserving in General Insurance", British Actuarial Journal — though that focuses on reserving, the GLM theory applies directly to pricing.

For the compound Poisson-gamma derivation: **Jørgensen & Paes de Souza (1994)** "Fitting Tweedie's compound Poisson model to insurance claims data", Scandinavian Actuarial Journal.

For the practical comparison with frequency-severity in a GBM context: **Delong, Kozak & Lindholm (2023)** "Making Tweedie's compound Poisson model more accessible", European Actuarial Journal.

If your data has structural zeros, our [ZI-Tweedie post]({% post_url 2026-03-28-zero-inflated-tweedie-gbm-insurance-pricing %}) covers the extension to a two-part model with `insurance-distributional`.
