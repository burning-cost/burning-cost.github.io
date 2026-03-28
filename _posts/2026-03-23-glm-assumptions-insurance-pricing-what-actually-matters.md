---
layout: post
title: "GLM Assumptions in Insurance Pricing: What Actually Matters"
date: 2026-03-23
author: Burning Cost
categories: [pricing, techniques, tutorials]
tags: [GLM, assumptions, Tweedie, Poisson, gamma, quasi-likelihood, overdispersion, residuals, diagnostics, link-function, dispersion, insurance-conformal, insurance-gam, motor-insurance, home-insurance, python, uk-insurance]
description: "Which GLM assumptions actually matter for insurance pricing, which ones you routinely violate without consequence, and the diagnostics worth running before signing off a production model."
---

Every GLM textbook lists the assumptions. The response follows a particular distribution family. The link function is correctly specified. The observations are independent. The systematic component is linear in the parameters. All four are routinely violated in real insurance pricing work. Two of those violations matter a great deal. Two of them you can mostly ignore. Knowing which is which is what separates a pricing actuary who validates models from one who recites them.

This post goes through each assumption in turn - what it means, what happens when it breaks, and what you should actually do about it on a live motor or home book.

---

## The four assumptions: a brief taxonomy

A GLM specifies: the conditional distribution of the response (the *family*), how the mean relates to the linear predictor (the *link function*), that the observations are *independent*, and that the log-mean (or whatever the canonical link maps to) is a *linear function of the parameters*.

These are not equally important. We will treat them in order of practical consequence.

---

## 1. Distribution family: this is where the real decisions are

Choosing the wrong distribution family is the GLM assumption most likely to produce wrong premiums, not just wrong standard errors.

For a frequency model on a UK motor book, the Poisson distribution is usually the right starting point. Not because claims counts are perfectly Poisson - they are not - but because the Poisson GLM is equivalent to maximising the multinomial likelihood for exposure-weighted count data, which is exactly what you want. The variance assumption (Var = mu) is almost certainly wrong, but as we will discuss below, the Poisson MLE has a robustness property that makes this less catastrophic than it sounds.

For severity, the Gamma distribution is the standard choice, and it is generally well-calibrated for attritional motor and home claims. Its variance-mean relationship (Var = phi * mu^2, i.e., constant coefficient of variation) matches what we observe empirically: large expected claims have larger absolute variance, but roughly stable CV. If your severity has a heavier tail than Gamma - large loss property or BI-heavy motor - the inverse Gaussian (Var ~ mu^3) is worth trying, or a spliced model where you model attritional and large losses separately.

For pure premium directly, Tweedie with power parameter p between 1 and 2 is the correct family. The compound Poisson-Gamma structure of Tweedie (1 < p < 2) matches the actual data-generating process: a Poisson number of claims, each with Gamma-distributed severity. For UK private motor, p typically estimates between 1.5 and 1.7. Getting p wrong by 0.2 or 0.3 produces miscalibrated variance estimates that distort safety loadings and reinsurance pricing even when the mean predictions look fine.

The Tweedie power parameter is not a nuisance to be ignored. Estimate it from data:

```python
from scipy.optimize import minimize_scalar
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie
from statsmodels.genmod.families.links import log as LogLink
import numpy as np

def tweedie_deviance(p, y, mu):
    """Compute Tweedie deviance for a given power p."""
    eps = 1e-12
    y_pos = np.where(y > 0, y, eps)
    d = 2 * np.where(
        y > 0,
        y_pos * (y_pos**(1 - p) - mu**(1 - p)) / (1 - p)
        - (y_pos**(2 - p) - mu**(2 - p)) / (2 - p),
        mu**(2 - p) / (2 - p)
    )
    return d.mean()

# Fit a base GLM at p=1.5, then profile over p
base_model = sm.GLM(
    y_train,
    X_train_sm,
    family=Tweedie(var_power=1.5, link=LogLink()),
    offset=np.log(exposure_train),
).fit()

mu_hat = base_model.fittedvalues

def neg_profile(p):
    if p <= 1.01 or p >= 1.99:
        return 1e10
    return tweedie_deviance(p, y_train, mu_hat)

result = minimize_scalar(neg_profile, bounds=(1.1, 1.9), method='bounded')
print(f"Estimated Tweedie p: {result.x:.3f}")
```

On a typical UK motor book dominated by accidental damage: p ≈ 1.5–1.6. Add bodily injury to the mix and p climbs to 1.65–1.75.

---

## 2. The independence assumption: when it breaks and when it doesn't

The independence assumption says that knowing the outcome for policy i tells you nothing about the outcome for policy j. On a motor book, this is wrong in at least two ways.

**Spatial clustering.** Policies in the same postcode are exposed to the same local crime rates, road quality, and weather events. Their claims are positively correlated. For attritional frequency - the grinding background of TPBI and own-damage claims - this correlation is small enough that ignoring it costs you little. Standard errors are slightly underestimated. Point estimates are unaffected.

For catastrophe exposure - a flood affecting fifty home insurance policies in the same district - spatial correlation matters enormously. The GLM independence assumption means the model has no mechanism to distinguish fifty independent claims from fifty simultaneous correlated claims. It will get the expected aggregate loss approximately right (claims are additive in expectation) but will catastrophically understate the aggregate variance. If you use GLM confidence intervals to set your cat loading, you are underestimating aggregate risk.

**Frequency-severity dependence.** The two-part model (separate Poisson frequency GLM, separate Gamma severity GLM) assumes frequency and severity are conditionally independent given the rating factors. Every published study that has tested this finds it to be wrong - young drivers who claim are selecting higher-severity claims, high-NCD drivers claim less often but for more when they do. The dependence is modest in most books (Spearman rank correlations in the 0.10-0.20 range) but it is systematically non-zero. We wrote about this in more detail in [Your Frequency-Severity Independence Assumption Is Costing You Premium]({{ site.baseurl }}{% post_url 2025-05-14-frequency-severity-independence-is-costing-you-premium %}).

The practical upshot: for attritional modelling of non-catastrophe lines, the independence assumption is wrong but not materially so for pricing point estimates. For aggregate variance estimates, cat loading, and reinsurance tower pricing, it is wrong in a way that matters and requires explicit treatment.

---

## 3. The linear predictor: more robust than you think, but not infinitely so

GLMs assume the link-transformed mean is a linear function of the rating factors:

```
log(mu) = beta_0 + beta_1 * driver_age_band + beta_2 * vehicle_group + ...
```

This is a modelling assumption, not a physical law. Driver age effects on frequency are not piecewise-constant across bands. The vehicle group effect is not the same across all driver ages. What the linear predictor buys you is interpretability and stability: a multiplicative rating factor structure that a pricing committee can audit, challenge, and import into a rating engine.

What it costs you is predictive power. Nonlinear age effects, interactions between age and vehicle type, geographies that do not respond linearly to the factors you have included - these are all left in the residuals.

The common solution is to put more structure in the bands: narrow driver age bands where the effect is steep (17–20, 21–24, 25–29), wider bands where it flattens out. This is actuarial craft. The statistical complement is to test for residual nonlinearity using GAMs  -  [insurance-gam](/2026/03/14/insurance-gam-interpretable-nonlinearity/) provides Poisson and Gamma GAMs with REML smoothing  -  then decide whether the nonlinearity is real enough to warrant a more complex factor structure.

`insurance-gam` fits a Poisson or Gamma GAM with spline terms for continuous rating factors, using a smoothing penalty that prevents overfitting thin data. The output is a smooth fitted curve that you can compare against your banded GLM factors to see where the bands are doing violence to the underlying signal:

```python
from insurance_gam import InsuranceGAM

gam_freq = InsuranceGAM(
    family="poisson",
    smooth_terms=["driver_age", "vehicle_age", "ncd_years"],
    factor_terms=["vehicle_group", "region_band"],
)
gam_freq.fit(df_train, response="claim_count", exposure="exposure")

# Compare smooth driver age effect vs banded GLM
gam_freq.plot_partial("driver_age", compare_glm=glm_result)
```

The GAM partial effect for driver age is the benchmark for whether your banding is capturing the right shape. If the GAM curve bends sharply between age 19 and 23 but your GLM uses a single "17-24" band, you are assigning the same relativity to two risks that are materially different. That is a linearity violation worth fixing, not just accepting.

Interactions are the harder case. A GLM with a log link does have implicit interactions - the multiplicative structure means the driver age relativity applies proportionally to whatever the vehicle group effect is. But crossed multiplicative interactions (driver age × vehicle group where the young-driver penalty is larger for high-performance vehicles) are not captured without explicit interaction terms. The problem is identifying which interactions are real. We cover that in [Finding the Interactions Your GLM Missed]({{ site.baseurl }}{% post_url 2026-02-27-finding-the-interactions-your-glm-missed %}).

---

## 4. The variance function: overdispersion and what to do about it

This is where the Poisson assumption requires the most practical nuance.

A Poisson GLM assumes Var(Y) = mu. On any real motor or home book, the observed variance exceeds this. Policyholders are not Poisson processes with identical risk characteristics within each cell. There is residual heterogeneity - unobserved risk factors, policyholder selection effects, genuine volatility beyond the Poisson rate - that makes the observed variance higher than a pure Poisson would predict. This is overdispersion.

The standard test is to compute the Pearson chi-squared statistic divided by the residual degrees of freedom:

```python
# After fitting a Poisson GLM in statsmodels
pearson_chi2 = result.pearson_chi2
df_resid     = result.df_resid
dispersion   = pearson_chi2 / df_resid

print(f"Estimated dispersion: {dispersion:.3f}")
# Poisson assumption: 1.0. Motor frequency: typically 1.1–1.5.
```

For UK motor frequency, we consistently see estimated dispersion between 1.1 and 1.5. This is material: it means standard errors computed under the Poisson assumption are 5–22% too small, and confidence intervals and hypothesis tests based on those standard errors are anticonservative.

**Does this invalidate the GLM coefficients?**

No. This is the key robustness result. A Poisson GLM estimated by maximum likelihood produces consistent estimates of the mean function under any distribution with the same variance-mean relationship, not just the Poisson. This is the quasi-likelihood argument: even if claims are not Poisson, the score equations for the Poisson log-likelihood are the same as the quasi-likelihood score equations for any distribution with Var(Y) proportional to mu. The coefficients are still right (asymptotically). It is the standard errors and test statistics that are wrong.

The fix is simple: use quasi-Poisson, which scales the standard errors by the square root of the estimated dispersion:

```python
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
# Quasi-Poisson: same coefficients, corrected standard errors
glm_quasi = sm.GLM(
    claims,
    X,
    family=Poisson(),
    offset=np.log(exposure),
).fit(scale="x2")   # scale="x2" uses Pearson chi-squared / df for dispersion

print(glm_quasi.summary())
# Coefficients: identical to standard Poisson
# Std errors: inflated by sqrt(dispersion) ≈ 1.1–1.2 on typical motor data
```

The choice between Poisson MLE and quasi-Poisson matters for:
- Hypothesis tests on rating factors (significance testing)
- Confidence intervals on relativities
- AIC-based model selection (quasi-likelihood cannot use AIC; use QIC or the scaled deviance)

It does not matter for:
- Premium calculations (same coefficients, same predictions)
- Deviance residuals as a diagnostic
- GLM vs GBM Gini comparisons

For Gamma severity, the analogous issue is that Gamma Var = phi * mu^2 requires estimating the dispersion phi. Statsmodels estimates phi from the Pearson chi-squared by default. If you have reason to believe phi varies across your book (heterogeneous volatility by claim type or driver segment), a distributional GLM with a sigma submodel is warranted. We described how to validate that in [How Do You Know Your Sigma Model Is Working?]({{ site.baseurl }}{% post_url 2026-03-08-validating-gamlss-sigma-models %}).

---

## Practical diagnostics: what to run before sign-off

### Deviance residuals vs fitted values

This is the GLM equivalent of the standard regression residual plot. For a Poisson model, compute the deviance residuals:

```python
import numpy as np
import matplotlib.pyplot as plt

residuals = result.resid_deviance
fitted    = np.log(result.fittedvalues)   # log scale for interpretability

plt.scatter(fitted, residuals, alpha=0.1, s=5)
plt.axhline(0, color='k', linewidth=0.8)
plt.xlabel("log(fitted)")
plt.ylabel("Deviance residual")
plt.title("Poisson GLM: deviance residuals vs fitted")
```

What you are looking for: no systematic trend (residuals should be centred at zero across the fitted range), and no obvious fan-out (which would indicate the variance function is wrong). Fanning in a Poisson model is unusual, but a consistent positive trend - residuals rising as fitted values increase - indicates underdispersion at high-risk segments, often a sign of model misspecification at the top of the risk distribution.

### Cook's distance and influential observations

Large losses are the main source of influence in a severity GLM. A single £500k subsidence claim on a home book can pull coefficients for property age or construction type materially. Cook's distance identifies which observations are doing this:

```python
influence = result.get_influence()
cooks_d   = influence.cooks_distance[0]

# Flag observations with Cook's D above the conventional threshold
threshold = 4 / len(y)
flagged   = np.where(cooks_d > threshold)[0]
print(f"{len(flagged)} influential observations ({100*len(flagged)/len(y):.1f}% of data)")
```

On a severity model with genuine large losses, you will have influential observations. The question is whether they are legitimate (the distribution genuinely produces those values and you want the model to account for them) or data quality issues (coding errors, reserve releases misclassified as claims). Inspect them. Do not remove them mechanically.

### Actual/expected by band: the rating actuary's residual check

The statistical diagnostics above are necessary but not sufficient. The one check that catches the most practical problems is computing actual/expected ratios by band for each rating factor, after controlling for the others:

```python
import polars as pl

def ae_by_factor(df: pl.DataFrame, factor: str,
                 actual: str, fitted: str, exposure: str) -> pl.DataFrame:
    return (
        df.group_by(factor)
        .agg([
            pl.col(actual).sum().alias("actual"),
            pl.col(fitted).sum().alias("fitted"),
            pl.col(exposure).sum().alias("exposure"),
        ])
        .with_columns(
            (pl.col("actual") / pl.col("fitted")).alias("ae_ratio")
        )
        .sort(factor)
    )

ae_driver_age = ae_by_factor(
    df_holdout.with_columns(pl.Series("fitted", result.predict(X_holdout_sm))),
    factor="driver_age_band",
    actual="claim_count",
    fitted="fitted",
    exposure="exposure",
)
```

An A/E ratio that is systematically above 1.0 for young drivers and below 1.0 for older drivers after a full GLM refactoring indicates either that the banding is wrong, that a relevant interaction is missing, or that the linearity assumption is failing in the way the GAM check would have identified. Any individual band A/E outside [0.85, 1.15] on a well-populated segment is a conversation worth having.

---

## Tweedie, Poisson, and Gamma: which to use when

The insurance pricing world converged on a particular workflow:

**Frequency model:** Poisson GLM with log link and exposure offset. Correct for overdispersion with quasi-Poisson if dispersion > 1.2. Do not use negative binomial unless you have a specific reason (NB adds a parameter that is rarely identifiable on annual motor data with standard rating factors).

**Severity model:** Gamma GLM with log link, fitted on positive claims only. Use inverse Gaussian only if you have established empirically that the variance-mean relationship is cubic rather than quadratic - which is rare for attritional motor, more plausible for BI-heavy books.

**Pure premium directly:** Tweedie GLM with estimated p. This is appropriate when you have a stable book with a long development history. The advantage over the two-part model is simplicity and the natural handling of the zero-claims mass. The disadvantage is that you lose the ability to separately analyse frequency and severity trends, which matters for monitoring and reserving. Most UK pricing teams use the two-part model for this reason, not because Tweedie is statistically inferior.

**When to use quasi-likelihood:** Always use quasi-Poisson (not standard Poisson MLE) for significance testing of rating factors, for confidence intervals on relativities, and whenever you are comparing models with different numbers of parameters. Quasi-Poisson has no AIC, so for model selection use the deviance difference with quasi-likelihood chi-squared tests (F-test on the deviance ratio with estimated dispersion).

```python
# Model comparison under quasi-Poisson: use F-test, not chi-squared
from statsmodels.stats.anova import anova_lm

model_base   = sm.GLM(y, X_base,   family=Poisson(), offset=offset).fit(scale="x2")
model_extend = sm.GLM(y, X_extend, family=Poisson(), offset=offset).fit(scale="x2")

# F-test: correct for quasi-likelihood. Chi-squared is anticonservative here.
f_stat = (
    (model_base.deviance - model_extend.deviance) / (model_extend.df_resid - model_base.df_resid)
) / model_extend.scale

from scipy.stats import f as f_dist
p_value = f_dist.sf(f_stat, dfn=(model_extend.df_resid - model_base.df_resid), dfd=model_extend.df_resid)
print(f"F-statistic: {f_stat:.3f}, p-value: {p_value:.4f}")
```

Proper MLE (not quasi-likelihood) remains appropriate for: cross-validation comparisons on the same dataset (deviance or log-score), Bayesian credibility extensions, and any context where the full probabilistic model is used downstream - for instance, if you are constructing prediction intervals from the parametric Tweedie distribution. For distribution-free alternatives, [conformal prediction for insurance](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) provides valid coverage without any distributional assumption.

Speaking of prediction intervals: if your GLM assumptions are materially violated, parametric intervals derived from the fitted distribution will be unreliable. The coverage will be wrong. For intervals that hold without distributional assumptions, `insurance-conformal` constructs finite-sample valid prediction intervals using conformal prediction. These are tighter than parametric Tweedie intervals on heteroskedastic books and have guaranteed coverage regardless of whether the distribution family was correctly specified. See [Conformal Prediction Intervals for Insurance Pricing Models]({{ site.baseurl }}{% post_url 2026-02-19-conformal-prediction-intervals-for-insurance-pricing %}) for the full setup.

---

## The link function: log is almost always right, and here is why

The canonical link for Poisson is the log link. For Gamma it is the reciprocal. In practice, everyone uses the log link for both, and this is correct.

The log link produces a multiplicative rating structure: each factor level multiplies the base premium by a relativity. This is what the industry runs in Radar, Emblem, and every other rating engine. It is regulatory-stable, auditable, and matches how underwriters think. More importantly, it constrains predictions to be positive, which is required for any insurance pricing model.

The identity link (log-transformed response, linear predictor for log-premium) is sometimes used for simplicity. It is wrong: it does not guarantee positive predictions and does not produce multiplicative relativities on the original scale. Avoid it.

The log link also has excellent numerical properties: the Poisson log-linear model is globally convex, so IRLS converges reliably. The reciprocal link for Gamma is not globally convex on the original scale, which is one reason practitioners prefer log even when the Gamma canonical link is theoretically preferred.

---

## Common violations in real portfolios: a practical checklist

**Motor frequency:**
- Overdispersion (dispersion 1.1–1.5): always present, use quasi-Poisson for inference
- Spatial clustering: present, minor effect on attritional point estimates, use spatial smoothing for territory factors
- Linearity: driver age and NCD effects have real nonlinearity at the edges; use GAM to check band structure
- Independence (F/S): present and non-trivial if you are computing combined ratios rather than just frequency point estimates

**Home frequency:**
- Same as motor for attritional perils
- Peril correlation (flood, escape of water, storm arriving together): the independence assumption breaks badly for cat events; do not use GLM confidence intervals for aggregate PML estimates

**Home severity:**
- Gamma is reasonable for attritional claims (escape of water, accidental damage)
- Large losses (subsidence, structural damage) have heavier tails; consider a spliced model or Pareto above a retention threshold
- Cook's distance analysis is especially important here because a handful of large losses can dominate coefficient estimates

**Telematics/UBI:**
- Score variables are highly correlated and often proxy for unobserved risk factors (urban driving, driver occupation)
- Linearity assumption is often violated: the effect of a telematics score on frequency is typically non-linear, flattening off above moderate risk thresholds
- Use a GAM to check the shape before banding; [Telematics Risk Scoring: From Trips to GLM Features]({{ site.baseurl }}{% post_url 2025-07-28-telematics-risk-scoring-from-trips-to-glm-features %}) covers the full pipeline

---

## What we recommend

Use Poisson with quasi-likelihood scaling for all frequency inference. Check your Tweedie power parameter empirically - do not default to p=1.5 without testing. Use GAMs to validate your banding structure before locking in rating factors; the linearity assumption is the one most likely to produce systematic pricing errors that accumulate quietly over renewal cycles.

Ignore the independence assumption for attritional point estimates. Do not ignore it if you are computing aggregate variance, setting cat loads, or pricing reinsurance.

For prediction intervals - coverage-guaranteed uncertainty bands per risk - the distributional assumptions in your GLM are a fragile foundation. Conformal prediction gives you the intervals without requiring those assumptions to hold.

---

**Related:**
- [Your Frequency-Severity Independence Assumption Is Costing You Premium]({{ site.baseurl }}{% post_url 2025-05-14-frequency-severity-independence-is-costing-you-premium %}) - empirical evidence and a Python implementation of Sarmanov copula marginals
- [How Do You Know Your Sigma Model Is Working?]({{ site.baseurl }}{% post_url 2026-03-08-validating-gamlss-sigma-models %}) - when the constant-dispersion assumption fails in a severity model and how to validate a sigma submodel
- [Finding the Interactions Your GLM Missed]({{ site.baseurl }}{% post_url 2026-02-27-finding-the-interactions-your-glm-missed %}) - testing for residual nonlinearity and interaction effects
- [Conformal Prediction Intervals for Insurance Pricing Models]({{ site.baseurl }}{% post_url 2026-02-19-conformal-prediction-intervals-for-insurance-pricing %}) - distribution-free prediction intervals when your GLM assumptions are in doubt
- [Tweedie Regression for Insurance: What sklearn Doesn't Tell You About Exposure]({{ site.baseurl }}{% post_url 2026-03-21-tweedie-regression-insurance-what-sklearn-doesnt-tell-you %}) - the exposure offset problem and how to estimate p correctly
