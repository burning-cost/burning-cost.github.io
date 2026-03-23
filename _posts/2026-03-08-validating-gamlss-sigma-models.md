---
layout: post
title: "GAMLSS Sigma Submodel Validation: GAIC, Quantile Residuals and Worm Plots"
date: 2026-03-08
categories: [pricing, techniques, tutorials]
tags: [gamlss, distributional-glm, diagnostics, worm-plot, quantile-residuals, volatility-scoring, sigma, variance, gamma, insurance-distributional-glm, python, model-validation]
description: "Four-step workflow to validate a GAMLSS sigma submodel before production: GAIC comparison, holdout quantile residuals, worm plots, and volatility score calibration against empirical CV."
---

[`insurance-distributional-glm`](/2026/03/10/insurance-distributional-glm/) fits GAMLSS models that allow the dispersion parameter sigma to vary by risk. A sigma submodel with two or three covariates will almost always reduce training-set log-likelihood. That does not mean it has learned something real — it might be fitting the idiosyncratic variance of your training cohort rather than a structural relationship that holds on unseen data.

Validating a sigma submodel is a distinct problem from validating the mean model. Standard GLM diagnostics are silent on whether the dispersion structure is genuine. The checks described here are designed to answer one question: is this sigma submodel capturing real heterogeneity in claim severity variance, or is it noise that will produce miscalibrated safety loadings in production?

Here is the four-step validation workflow we use before signing off a sigma submodel as production-ready.

---

## Step one: the GAIC test

Before examining diagnostics, establish that adding covariates to sigma is justified at all. Fit two models: one with an intercept-only sigma (equivalent to a standard GLM with constant dispersion), and one with your candidate sigma covariates.

```python
import numpy as np
import polars as pl
from insurance_distributional_glm import DistributionalGLM
from insurance_distributional_glm.families import Gamma

# Severity data: one row per paid claim
# age_band: 0=17-25, 1=26-35, 2=36-50, 3=51+
# vehicle_group: A-E, 0-indexed
# region: 0-10

model_null = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_group", "region"],
        "sigma": [],                # intercept-only: constant dispersion
    },
)
model_null.fit(df_train, y_train)

model_sigma = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "vehicle_group", "region"],
        "sigma": ["age_band", "vehicle_group"],
    },
)
model_sigma.fit(df_train, y_train)

delta_gaic = model_null.gaic(penalty=2) - model_sigma.gaic(penalty=2)
print(f"DELTA GAIC: {delta_gaic:.1f}")
```

The GAIC with penalty=2 is standard AIC. A positive delta means the sigma covariates are worth adding after penalising for the extra parameters. We use a threshold of 10 as our floor — anything below that is ambiguous, and the governance overhead of explaining a non-standard dispersion submodel to a pricing committee is not worth it for a marginal AIC gain.

On a UK motor severity book with 40,000+ claims and genuine heterogeneity by age and vehicle group, you should typically see a delta GAIC of 30-100. If you are seeing 5-8, your sigma covariates are probably fitting noise. Use BIC (penalty=log(n)) for a stricter test on larger datasets; on 40k observations, log(n) ≈ 10.6, which is a meaningfully harder bar.

```
DELTA GAIC (AIC, penalty=2):  61.4   -> sigma covariates justified
DELTA GAIC (BIC, penalty=10.6): 28.1   -> survives the harder test
```

If the model passes this gate, proceed. If not, use the null model.

---

## Step two: quantile residuals on held-out data

The GAIC test only tells you whether the sigma submodel is improving fit relative to a constant dispersion baseline. It does not tell you whether the distributional assumption is correctly specified at all.

Quantile residuals are the right tool for this. For a continuous response and a correctly specified model, the randomised quantile residual `r_i = Phi^{-1}(F(y_i; theta_hat_i))` is standard normal. Departures from normality indicate distributional misfit: overdispersion, underdispersion, heavy tails, skewness in the wrong direction.

The key: run this on your **holdout set**, not the training data. If the sigma model is overfitting, the training-set quantile residuals will look fine and the holdout residuals will not.

```python
from insurance_distributional_glm import quantile_residuals

# Holdout set -- not used in fitting
resids = quantile_residuals(model_sigma, df_holdout, y_holdout, seed=42)

# The residuals should look N(0,1). Check mean, std, and the Shapiro-Wilk p-value.
from scipy import stats

sw_stat, sw_p = stats.shapiro(resids[:5000])   # Shapiro-Wilk caps at 5000
print(f"Mean: {resids.mean():.3f}  Std: {resids.std():.3f}")
print(f"Shapiro-Wilk p: {sw_p:.4f}")
```

A well-specified model on holdout data looks like this:

```
Mean: 0.012   Std: 1.003
Shapiro-Wilk p: 0.312
```

An overfitting sigma model — fitted with too many sigma covariates on too few claims — looks like this:

```
Mean: 0.028   Std: 1.041
Shapiro-Wilk p: 0.003
```

The Shapiro-Wilk p-value of 0.003 flags that the residuals are not normal. Not dramatically non-normal — the mean and standard deviation look almost right — but the tails are systematically off. The model is assigning confidence intervals that are too wide for some segments and too narrow for others in a way that the global statistics conceal.

Be honest about what Shapiro-Wilk can and cannot do here. On large holdout sets (10,000+ observations), even small deviations from normality will produce significant p-values. Do not use a p < 0.05 rule mechanically. Look at the QQ plot. The question is whether the tails deviate materially, not whether the test rejects at the 5% level.

---

## Step three: the worm plot

Quantile residuals give you a global distributional check. The worm plot gives you a stratified check: does the distributional fit hold across the range of fitted values?

```python
from insurance_distributional_glm import worm_plot

# Requires: uv add "insurance-distributional-glm[plots]"
# Split the holdout into 4 groups by fitted mu, plot separately
worm_plot(model_sigma, df_holdout, y_holdout, n_groups=4, seed=42)
```

The worm plot splits the holdout into quartiles by fitted mu, then draws a detrended QQ plot (residual minus theoretical normal quantile) for each group. A well-fitting model shows flat worms near zero in all four panels. Systematic patterns indicate specific failures:

- **Worm shifts upward across all groups**: underdispersion — the model's predicted variance is too high
- **Worm shifts downward**: overdispersion — predicted variance too low
- **S-shape in the worm**: the variance-mean relationship is wrong (for Gamma, this means sigma is systematically biased at one end of the mu range)
- **U-shape**: kurtosis is wrong — the tails are heavier or lighter than Gamma predicts

The most informative failure mode for a sigma submodel is an S-shape that varies across the four panels. If the high-mu panel shows a downward S and the low-mu panel shows an upward S, your sigma formula is learning the wrong covariates: it is confusing high-mu segments (which have naturally higher absolute variance) with high-CV segments (which have higher variance relative to the mean). The sigma submodel should be predicting CV — which is what the Gamma sigma parameter is — not absolute variance.

This confusion is common when sigma formula covariates overlap heavily with mu formula covariates. The fix is to refit with a smaller sigma formula, removing covariates that are strongly collinear with the mean submodel.

---

## Step four: volatility score against empirical CV

The three tests above are diagnostics — they tell you whether the model is self-consistent. The final check is whether the predicted volatility is calibrated against reality.

The `volatility_score` method returns the coefficient of variation CV = sqrt(Var[Y|X]) / E[Y|X] per observation, derived from the fitted distributional parameters. This is the model's prediction of how volatile each policy's severity is, relative to its mean.

To validate it, compute empirical CV by segment on the holdout and compare:

```python
import polars as pl

# Predicted CV per policy
cv_pred = model_sigma.volatility_score(df_holdout)

# Add to holdout frame
df_val = df_holdout.with_columns(pl.Series("cv_pred", cv_pred))
df_val = df_val.with_columns(pl.Series("y", y_holdout))

# Empirical CV by age_band x vehicle_group on holdout
empirical_cv = (
    df_val
    .group_by(["age_band", "vehicle_group"])
    .agg([
        pl.col("y").mean().alias("mu_emp"),
        pl.col("y").std().alias("sd_emp"),
        pl.col("cv_pred").mean().alias("cv_pred_mean"),
        pl.len().alias("n"),
    ])
    .with_columns(
        (pl.col("sd_emp") / pl.col("mu_emp")).alias("cv_emp")
    )
    .filter(pl.col("n") >= 50)    # only segments with enough data
    .sort(["age_band", "vehicle_group"])
)

print(empirical_cv.select(["age_band", "vehicle_group", "cv_emp", "cv_pred_mean", "n"]))
```

```
age_band  vehicle_group  cv_emp  cv_pred_mean     n
       0              0   0.614         0.601   184
       0              2   0.701         0.689   127
       0              4   0.812         0.798    89
       1              0   0.481         0.469   312
       1              2   0.513         0.501   248
       1              4   0.601         0.587   201
       2              0   0.447         0.441   418
       2              2   0.468         0.462   355
       2              4   0.533         0.519   289
       3              0   0.441         0.438   376
       3              2   0.452         0.448   321
       3              4   0.497         0.488   267
```

The model is tracking empirical CV well: young drivers (age_band=0) in the highest vehicle group (group=4) have a CV of 0.812 empirically, and the model predicts 0.798. Older drivers (age_band=3) have empirical CV around 0.44-0.50, and the model predicts 0.44-0.49. The sigma submodel is genuine signal, not fitting noise.

A sigma model that is overfitting produces a very different picture: predicted CV varies more than empirical CV, with the high-predicted-CV segments showing empirical CV no different from the book average. When you see that, go back to the GAIC and reduce the sigma formula.

The Spearman correlation between `cv_emp` and `cv_pred_mean` across segments is a useful single-number summary. Anything above 0.85 across segments with n >= 50 warrants production confidence. Below 0.5, the sigma submodel is probably not capturing the right signal.

---

## The practical severity loading calculation

Once the sigma submodel passes these tests, the premium calculation changes:

```python
mu    = model_sigma.predict(df_new, parameter="mu")
sigma = model_sigma.predict(df_new, parameter="sigma")
var   = mu**2 * sigma**2        # Gamma variance formula: Var = (CV * mu)^2

# Safety loading: expected loss + lambda * standard deviation
lambda_ = 0.35                   # loading factor, calibrated to target return
technical_premium = mu + lambda_ * np.sqrt(var)

# For constant-dispersion model, sigma is scalar:
# technical_premium_null = mu + lambda_ * (sigma_constant * mu)
# The GAMLSS version loads high-CV risks more than low-CV risks
# even when their means are identical
```

For a young driver (age_band=0) in vehicle group 4 with a £3,500 expected severity and CV of 0.80, the standard deviation is £2,800. The safety loading at lambda=0.35 is £980. For a mature driver (age_band=2) with the same expected severity but a CV of 0.52, the standard deviation is £1,820 and the loading is £637. The constant-dispersion model would assign both the same loading based on the book-average CV of roughly 0.55.

That is a £343 pricing difference between two policies with identical expected severity. The young driver is underpriced by the standard model; the mature driver is overpriced. On a book of 80,000 policies, the aggregate mismatch concentrates in whichever segments are systematically under or over the book-average CV.

---

## When the sigma model should not be used

Honest about the limitations:

**Too few claims per segment.** The sigma submodel is estimated from the variability in claim severities, which requires sufficient claims to identify the relationship. If your sigma formula includes a covariate whose lowest-frequency level has 80 claims, the sigma coefficient for that level will have wide uncertainty. We recommend at least 150-200 claims per covariate level before trusting a sigma coefficient. The standard errors are not printed by default — call `model.coefficients['sigma']` and compare the magnitude of the coefficients against the scale of sigma (typically log(0.4) to log(0.9) for motor severity). A coefficient with magnitude below 0.05 is probably not identifiable from noise on fewer than 1,000 claims.

**Sigma covariates that are collinear with mu covariates.** Adding the same set of covariates to both the mu and sigma formulas creates identifiability pressure: the algorithm has to distinguish "this segment has a higher mean" from "this segment has a higher CV". When the covariates are the same, that distinction is hard. Use a strict subset of covariates for sigma — the one or two factors where you have a priori reason to believe the CV differs.

**Single-peril books with fewer than 10,000 claims.** The GAIC test provides no protection on very small datasets because the log-likelihood surface is noisy. The holdout quantile residual test does help, but only if your holdout is large enough to produce stable distributional statistics (we suggest at least 2,000 holdout claims for a continuous response). On smaller datasets, the constant-dispersion GLM is almost always the right choice, with manual dispersion parameters per major segment if the evidence warrants it.

---

## Putting it together

The validation protocol in order:

1. GAIC comparison between null sigma and candidate sigma formula. Threshold: delta GAIC > 10 (AIC) or delta GAIC > log(n) (BIC). Below this, stop.
2. Quantile residuals on holdout. QQ plot and Shapiro-Wilk. Look at tails, not just the summary statistics.
3. Worm plot on holdout split by mu quartile. S-shapes and systematic shifts indicate the wrong sigma formula.
4. Volatility score vs. empirical CV by segment. Spearman correlation > 0.85 across well-populated segments is the production gate.

The full workflow runs in under five minutes on a standard motor severity dataset. Steps 1 and 4 are the ones that get challenged in a pricing review — the GAIC test because it is unfamiliar to actuaries used to standard F-tests, and the volatility score comparison because it requires segment-level empirical statistics that not every team has readily to hand. Both are worth the effort. A sigma model that passes all four steps is one you can sign off confidently.

---

`insurance-distributional-glm` v0.1.0 is at [github.com/burning-cost/insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm). The diagnostics module (`quantile_residuals`, `worm_plot`) and `volatility_score` are all included in the base install. Worm plots require matplotlib: `uv add "insurance-distributional-glm[plots]"`.

---

**Related:**
- [GAMLSS in Python, Finally](/2026/03/10/insurance-distributional-glm/) — the introductory post on fitting the model
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/03/04/per-risk-volatility-scoring-with-distributional-gbms/) — a GBM-based approach to the same problem if you need more flexibility in the mean submodel
- [Your Frequency-Severity Independence Assumption Is Costing You Premium](/2026/03/08/frequency-severity-independence-is-costing-you-premium/) — the other structural assumption most teams leave unchallenged
