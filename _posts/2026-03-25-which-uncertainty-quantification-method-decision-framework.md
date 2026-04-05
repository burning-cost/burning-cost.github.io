---
layout: post
title: "Which Uncertainty Quantification Method? A Decision Framework for Insurance Pricing Actuaries"
date: 2026-03-25
categories: [pricing, techniques]
tags: [uncertainty-quantification, conformal-prediction, gamlss, credibility, bootstrap, GAM, insurance-conformal, insurance-distributional, insurance-credibility, insurance-gam, solvency-ii, scr, uk-motor, decision-framework, python]
description: "A structured decision framework for choosing between conformal prediction, distributional GBM, Bühlmann-Straub credibility, GLM bootstrap, and GAM uncertainty. Model type, data size, regulatory use, and interpretability all drive the answer."
---

There is no shortage of uncertainty quantification methods in the actuarial toolkit. There is a shortage of clear guidance on which one to use when. Most posts on this topic either describe a single method in isolation or give you a list of options without telling you how to choose. This is the post we wish existed when we started building pricing models that had to satisfy both a CFO wanting capital numbers and a regulatory team wanting auditable intervals.

The framework below has five decision nodes. Work through them in order. The recommendations at each leaf are opinionated — we explain why, and we give you the code.

---

## The five decision nodes

**Node 1: What is the primary regulatory use case?**

The answer to this determines the entire downstream choice more than any other factor.

- **SCR / Solvency II capital model** → go to Node 2
- **Pricing interval or risk margin** → go to Node 3
- **Audit or model validation** → go to Node 4

**Node 2: Is the regulatory output a per-risk bound or a portfolio aggregate?**

- **Per-risk bound** (individual policy SCR component, individual risk XL pricing) → **Recommendation A: [Conformal prediction](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)**
- **Portfolio aggregate** (internal model aggregate distribution, reinsurance aggregate) → **Recommendation B: [Distributional GBM](/2026/03/05/insurance-distributional/)**

**Node 3: What is your model type?**

- **GLM / GAM** → go to Node 5
- **GBM / ensemble** → go to Node 3a

**Node 3a: Do you need a full conditional distribution or just a prediction interval?**

- **Full distribution** (layer pricing, CRPS optimisation, IFRS 17 risk adjustment) → **Recommendation B: Distributional GBM**
- **Interval only** → **Recommendation A: Conformal prediction**

**Node 4: How large is the dataset, and is there a meaningful grouping structure?**

- **Small to medium data (under 20,000 exposures) with groups** (schemes, territories, NCD classes) → **Recommendation C: Bühlmann-Straub credibility**
- **Larger data with a GLM that must be fully auditable** → **Recommendation D: GLM + bootstrap**

**Node 5: Is interpretability a hard constraint?**

- **Hard constraint** (regulatory model, board-level reporting, Fair Pricing rules) → **Recommendation E: GAM with EBM uncertainty**
- **Not a hard constraint** → **Recommendation A: Conformal prediction**

---

## Decision table summary

| Primary use | Model type | Data / structure | Interpretability | Recommendation |
|---|---|---|---|---|
| SCR per-risk bound | Any | Any | Any | A: Conformal |
| SCR portfolio aggregate | Any | Any | Any | B: Distributional GBM |
| Pricing interval | GBM | Any | Not required | A: Conformal |
| Pricing interval | GBM | Any | Not required | B: Distributional GBM (full dist.) |
| Pricing interval | GLM/GAM | Any | Hard constraint | E: GAM EBM |
| Audit / validation | Any | Small + groups | Any | C: Bühlmann-Straub |
| Audit / validation | GLM | Large, no groups | Hard constraint | D: GLM bootstrap |

---

## Recommendation A: Conformal prediction — [`insurance-conformal`](/insurance-conformal/)

**When to use it.** Your model is a fitted GBM (CatBoost, LightGBM) or GLM, you need a prediction interval with a statistical coverage guarantee, and you do not want to make parametric assumptions about the error distribution. This is the right tool for Solvency II per-risk upper bounds, for pricing intervals that need to be consistent with Solvency II Article 120 internal model validation requirements, and for any situation where "I calibrated this on 5,000 policies so the interval is actually 91.3% rather than exactly 90%" is an acceptable answer. The finite-sample guarantee — P(y ∈ [lower, upper]) ≥ 1 − alpha — is distribution-free and holds for any calibration set size above around 200 observations.

The non-conformity score matters. For Tweedie or Poisson models, use `pearson_weighted` (the default). Raw residual scores give wider intervals in the tails and narrower intervals at low predicted values, which is exactly backwards for insurance. Check `coverage_by_decile()` after calibrating — if any decile deviates more than 5pp from the target, switch to `deviance`.

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.scr import SCRReport

# Wrap your fitted CatBoost or sklearn model
cp = InsuranceConformalPredictor(
    model=fitted_catboost,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

# Calibrate on a held-out period (e.g. year 5 of a train-on-years-1-4 model)
cp.calibrate(X_cal, y_cal)

# 90% prediction intervals for the renewal book
intervals = cp.predict_interval(X_test, alpha=0.10)
# pl.DataFrame with columns: lower, point, upper

# Coverage diagnostic — check this before submitting to a regulator
cp.summary(X_test, y_test, alpha=0.10)

# Solvency II per-risk SCR upper bounds at 99.5th percentile
scr = SCRReport(predictor=cp)
scr_table = scr.solvency_capital_requirement(X_test, alpha=0.005)
```

One honest caveat: conformal prediction gives you *marginal* coverage, not *conditional*. It guarantees that 90% of all risks are covered on average. It does not guarantee that 90% of high-value risks specifically are covered. Use `coverage_by_decile()` to check this. If you need conditional coverage — equal coverage at every risk level — the literature has no fully satisfying answer yet, and neither do we.

---

## Recommendation B: Distributional GBM — [`insurance-distributional`](/insurance-distributional/)

**When to use it.** You need the *full conditional distribution* of losses, not just an interval. The motivating cases: pricing an excess-of-loss layer (you need E[max(Y − d, 0) ∧ l | x] computed from the conditional distribution, not from a point prediction), IFRS 17 risk adjustment (you need CoV per risk to calculate the risk adjustment at policy level), or capital modelling where a scalar phi assumption has been challenged. Also use this when you have evidence of heteroscedastic dispersion — risks with the same expected loss but materially different volatility — which is the norm in UK motor rather than the exception.

The `TweedieGBM` model fits a joint mean and dispersion model using the Smyth-Jørgensen double GLM with CatBoost. The key output beyond the mean is `pred.volatility_score()`, which is the per-risk coefficient of variation. That number is what turns a distributional model into something actionable for pricing.

```python
from insurance_distributional import TweedieGBM
from insurance_distributional import coverage, pit_values

model = TweedieGBM(power=1.5)
model.fit(X_train, y_train, exposure=exposure_train)

pred = model.predict(X_test, exposure=exposure_test)

# Point prediction and distributional outputs
mu      = pred.mean          # E[Y | X]
phi     = pred.variance      # Var[Y | X] — per-risk, not portfolio scalar
cov_per_risk = pred.volatility_score()  # CoV = sqrt(phi) * mu^(p/2 - 1)

# Proper scoring: CRPS is the most informative for distributional models
crps_score = model.crps(X_test, y_test)

# PIT histogram — should be uniform if distributional calibration is good
pit = pit_values(y_test, pred)
```

If you are pricing XL layers or per-risk excess structures, use `FlexCodeDensity` instead of `TweedieGBM`. The FlexCode density estimator fits a nonparametric conditional density using a cosine basis, which makes no parametric assumptions about the shape of the severity distribution. It is slower and requires more data (we recommend at least 5,000 observations for the severity component) but is strictly more general.

---

## Recommendation C: Bühlmann-Straub credibility — [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility)

**When to use it.** You have a book with meaningful grouping structure — schemes, territories, affinity partners, NCD classes — and not enough exposure in each group to rely solely on group-level experience. The Bühlmann-Straub model is the actuarial standard answer to this question and has been since 1970. It is not glamorous, it is not machine learning, and it is exactly right for the job. It produces credibility factors Z_i ∈ [0, 1] for each group that blend the group's own weighted-average loss rate with the portfolio collective mean. The structural parameters v (within-group variance) and a (between-group variance) are estimated from the data, not assumed.

The critical diagnostic is Bühlmann's k = v/a. A large k (say k > 100) means the groups are genuinely homogeneous and the data is noisy — you should trust the collective. A small k (say k < 5) means the groups differ substantially — even thin experience should move the premium. Both outcomes are informative.

For experience rating at individual policy level — adjusting a priori premiums based on a policyholder's own claims history — use `StaticCredibilityModel` from `insurance_credibility.experience` rather than `BuhlmannStraub`.

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# Panel data: one row per (group, period)
# e.g. commercial fleet schemes across five underwriting years
df = pl.DataFrame({
    "scheme":     ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
    "year":       [2021, 2022, 2023] * 3,
    "loss_rate":  [0.55, 0.60, 0.58, 0.80, 0.75, 0.82, 0.40, 0.42, 0.38],
    "exposure":   [1000, 1200, 1100, 300, 350, 320, 5000, 4800, 5200],
})

bs = BuhlmannStraub()
bs.fit(df,
       group_col="scheme",
       period_col="year",
       loss_col="loss_rate",
       weight_col="exposure")

bs.summary()
# Structural parameters: mu_hat, v_hat (EPV), a_hat (VHM), k = v/a
# Group results: z_ (credibility factors), premiums_

print(bs.z_)       # credibility factors — scheme A gets ~0.72, scheme C gets ~0.95
print(bs.premiums_)  # blended credibility premiums per scheme
```

The uncertainty in the credibility premium is quantified implicitly by Z_i itself: a low Z_i means the interval around the credibility premium is wide. For explicit confidence intervals on the structural parameters, the bootstrap is appropriate, but in practice the credibility factor tells you what you need to know — how much to trust a group's own experience.

---

## Recommendation D: GLM + bootstrap

**When to use it.** Your model is a GLM, it must be fully auditable, and the regulator or model governance committee will not accept "this is a black-box interval from a machine learning procedure." The parametric bootstrap is the conservative, explainable choice: refit the GLM on B bootstrap samples of the training data, compute B sets of predicted values, and take the empirical quantiles of those predictions as your interval. This approach cannot fail a model validation review because it is entirely transparent about its assumptions.

The limitations are real. The parametric bootstrap is slow for large GLMs, it understates interval width when the model is misspecified (and it is always misspecified), and it gives you parameter uncertainty intervals rather than prediction intervals — the two are different things. For a Poisson frequency model, the bootstrap interval around mu_i reflects uncertainty in the fitted relativities, but it does not account for the process variance of realised claims around mu_i. Add the Poisson variance term explicitly if you need a genuine prediction interval rather than a parameter uncertainty interval.

```python
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Fit baseline GLM
glm = smf.glm(
    "claims ~ C(age_band) + C(vehicle_group) + C(ncd) + offset(np.log(exposure))",
    data=train_df,
    family=sm.families.Poisson(),
).fit()

mu_base = glm.predict(test_df)

# Parametric bootstrap: resample with replacement, refit, predict
B = 500
rng = np.random.default_rng(2026)
boot_predictions = np.zeros((B, len(test_df)))

for b in range(B):
    idx = rng.choice(len(train_df), size=len(train_df), replace=True)
    sample = train_df.iloc[idx].reset_index(drop=True)
    try:
        glm_b = smf.glm(
            "claims ~ C(age_band) + C(vehicle_group) + C(ncd) + offset(np.log(exposure))",
            data=sample,
            family=sm.families.Poisson(),
        ).fit(disp=False)
        boot_predictions[b] = glm_b.predict(test_df)
    except Exception:
        boot_predictions[b] = mu_base  # fallback: use baseline if singular

# 90% parameter uncertainty interval
lower = np.percentile(boot_predictions, 5, axis=0)
upper = np.percentile(boot_predictions, 95, axis=0)

# For a genuine prediction interval (not just parameter uncertainty),
# add the Poisson process variance:
# pred_lower_i = scipy.stats.poisson.ppf(0.05, mu=lower_i)
# pred_upper_i = scipy.stats.poisson.ppf(0.95, mu=upper_i)
```

Use 500 bootstrap samples as a minimum. Below that, the quantile estimates become unstable. For large portfolios (over 500,000 rows), consider the residual bootstrap (resample Pearson residuals rather than rows) which is faster and preserves the covariate structure.

---

## Recommendation E: GAM with EBM uncertainty — [`insurance-gam`](https://github.com/burning-cost/insurance-gam)

**When to use it.** You need interpretability as a hard requirement — every shape function must be inspectable, editable by a senior actuary, and explainable to a non-technical audience — and you also need uncertainty quantification on those shape functions. The `InsuranceEBM` from `insurance-gam` wraps interpretML's Explainable Boosting Machine, which is a GAM with pairwise interactions fitted by gradient boosting. Each feature's contribution is a learned shape function that can be plotted, inspected, and adjusted.

Bagging in EBM gives you uncertainty on each shape function. The `inner_bags` parameter controls the number of internal bagging models; 100 inner bags gives stable uncertainty estimates on the shape functions at moderate computational cost. The uncertainty here is on the *model* (parameter uncertainty), not on future realisations of claims.

For regulatory models in particular, the combination of interpretability and uncertainty quantification is powerful: you can show a reviewer not just the relativities but also the confidence bands around them, and you can demonstrate that the shape of the curve is stable across bootstrap samples.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

ebm = InsuranceEBM(
    loss="tweedie",
    variance_power=1.5,
    interactions=5,  # number of pairwise interaction terms to fit
    ebm_kwargs={
        "outer_bags": 8,    # bagging for uncertainty estimates on shape functions
        "inner_bags": 0,    # set >0 for within-bag uncertainty (slower)
        "max_bins": 256,
    },
)

ebm.fit(X_train, y_train, exposure=exposure_train)

# Point predictions
mu_hat = ebm.predict(X_test, exposure=exposure_test)

# Shape function relativities with confidence bands from outer bagging
rel = RelativitiesTable(ebm)
rel.table("driver_age")
# Returns a DataFrame: feature | level | relativity | lower_ci | upper_ci | n_obs

# For pricing use: actuary can inspect and constrain any shape function
# before generating the final relativity table
```

The honest limitation: EBM is not competitive with CatBoost on predictive performance for large, complex insurance portfolios. The deviance difference is typically small (2-5% on held-out Gini) but it is consistently in favour of CatBoost. If your portfolio is large enough that predictive performance dominates over interpretability, use conformal prediction on a CatBoost model instead.

---

## What we do not recommend

**Quantile regression alone.** Quantile regression gives you the alpha-th quantile of Y | X, not a calibrated prediction interval. Two quantile models at 5% and 95% do not jointly guarantee 90% coverage — they are each individually calibrated at their respective quantiles, but the joint coverage depends on their correlation structure. Use conformal prediction if you want a coverage guarantee.

**Bayesian GLMs for large portfolios.** MCMC-based Bayesian GLMs are computationally prohibitive at personal lines scale (millions of policies, dozens of rating factors). Variational Bayes approximations are faster but introduce approximation error that is hard to quantify. The bootstrap gives you equivalent parameter uncertainty estimates at a fraction of the computational cost, with full transparency. Use Bayesian methods when you have a genuine strong prior — structured expert knowledge that dominates the data — which is rare in personal lines.

**Neural network uncertainty (MC dropout, deep ensembles).** These methods are powerful and well-studied in the machine learning literature. They are not ready for regulatory insurance pricing in the UK. The uncertainty estimates are sensitive to hyperparameter choices, they are not auditable in the way a regulator expects, and there is no equivalent of conformal prediction's finite-sample guarantee. Watch this space; it will change.

---

## Quick-reference installation

```bash
# Conformal prediction — any model type, distribution-free intervals
uv add insurance-conformal

# Distributional GBM — full conditional distribution, per-risk dispersion
uv add insurance-distributional

# Credibility — group-level blending, experience rating
uv add insurance-credibility

# GAM / EBM — interpretable shape functions with uncertainty
uv add "insurance-gam[ebm]"
```

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/)
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture for UK Insurance Pricing](/2026/03/14/insurance-gam-interpretable-nonlinearity/)

The libraries are independent. Install only what you need. All four are compatible with polars DataFrames natively and with standard numpy arrays.

---

## The short version

If you only read one sentence: **conformal prediction on a fitted GBM is the right default for most pricing interval and per-risk SCR use cases.** The only reasons to use something else are that you need a full conditional distribution (distributional GBM), you have a grouping structure and thin data (credibility), you have a hard GLM-only interpretability constraint (bootstrap), or your model is a GAM and shape function uncertainty matters more than interval efficiency (EBM).

Everything else is a special case.
