---
layout: post
title: "Fitting a Motor Insurance GLM in Python: Poisson Frequency and Gamma Severity with statsmodels"
date: 2026-03-24
author: Burning Cost
categories: [pricing, tutorials, glm]
tags: [GLM, poisson, gamma, statsmodels, frequency-severity, exposure, offset, overdispersion, factor-table, AE-ratio, deviance, quasi-poisson, motor-insurance, uk-insurance, python, tutorial, insurance-glm-tools, insurance-dispersion, insurance-datasets]
description: "A practical statsmodels tutorial for pricing actuaries: Poisson frequency model with exposure offset, Gamma severity model, overdispersion tests, factor table extraction, and A/E validation on synthetic UK motor data."
---

Every UK motor insurer runs a GLM. Most of those GLMs were built in Emblem, or in R using `glm()`, by actuaries who have now been asked to "do it in Python." The Python documentation for Tweedie regression points at sklearn. The sklearn documentation does not mention exposure offsets. The statsmodels documentation does not mention insurance pricing.

This tutorial is for the pricing actuary who wants to build a frequency/severity GLM in Python from first principles, with correct exposure handling, overdispersion testing, factor table extraction, and validation diagnostics. We use synthetic UK motor data from `insurance-datasets` so the correct answer is known in advance.

---

## Setup

```bash
uv add statsmodels insurance-datasets insurance-glm-tools insurance-dispersion polars numpy scipy scikit-learn
```

```python
import numpy as np
import polars as pl
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.stats import f as f_dist, norm as scipy_norm
from sklearn.preprocessing import OneHotEncoder
from statsmodels.genmod.families import Poisson, Gamma
from statsmodels.genmod.families.links import log as LogLink
```

---

## The data

`insurance-datasets` generates synthetic UK motor policies from a known Poisson frequency / Gamma severity data-generating process. True parameters are exported alongside the data, so you can check whether the GLM recovers them.

```python
from insurance_datasets import load_motor

df, true_params = load_motor(n=50_000, random_state=42)
# df: Polars DataFrame with columns:
#   policy_id, inception_date, exposure (years),
#   driver_age, ncd_years, vehicle_group, vehicle_age, region_band,
#   claim_count (integer), claim_amount (£, NaN where no claim)
# true_params: {"freq": {factor: coefficient}, "sev": {factor: coefficient}}

print(f"Claim frequency: {df['claim_count'].sum() / df['exposure'].sum():.4f} claims per year")
print(f"Claimants: {(df['claim_count'] > 0).sum()} of {len(df)} ({100*(df['claim_count']>0).mean():.1f}%)")
```

You will see roughly 8–9% of policyholders claiming, an average severity around £1,800, and exposure varying between 0.05 and 1.0 (monthly direct debit payers, mid-term cancellations, new business late in the year). The exposure variation is intentional — it is what makes the offset non-trivial.

---

## Data preparation

statsmodels accepts a numpy matrix or scipy sparse matrix for the design matrix. We use Polars for the manipulation, then encode categoricals and pass the result through.

```python
CAT_FACTORS  = ["ncd_years", "vehicle_group", "region_band"]
CONT_FACTORS = ["driver_age", "vehicle_age"]

X_cont = df.select(CONT_FACTORS).to_numpy().astype(float)
X_cat  = df.select(CAT_FACTORS).to_numpy().astype(str)

enc = OneHotEncoder(drop="first", sparse_output=True)
X_cat_enc = enc.fit_transform(X_cat)

# Intercept + continuous + encoded categoricals
X = sp.hstack([
    np.ones((len(df), 1)),
    X_cont,
    X_cat_enc,
], format="csc")

feature_names = (
    ["intercept"]
    + CONT_FACTORS
    + enc.get_feature_names_out(CAT_FACTORS).tolist()
)

exposure = df["exposure"].to_numpy()
y_freq   = df["claim_count"].to_numpy().astype(float)
```

### Train/test split

We split by inception date, not randomly. Testing on a future period is the honest evaluation for a production pricing model.

```python
from insurance_cv import walk_forward_split

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,
    test_months=6,
    ibnr_buffer_months=3,
)
last_split = splits[-1]
train_idx, test_idx = last_split.get_indices(df)

X_tr, X_te     = X[train_idx], X[test_idx]
y_tr, y_te     = y_freq[train_idx], y_freq[test_idx]
exp_tr, exp_te = exposure[train_idx], exposure[test_idx]
```

---

## Frequency model: Poisson GLM with exposure offset

The canonical frequency model is:

```
log(E[N_i]) = log(exposure_i) + X_i * beta
```

The `log(exposure_i)` term is an *offset* — it enters the linear predictor with a fixed coefficient of exactly 1.0, not estimated. This is what makes the model predict a *rate* (claims per year) rather than an expected count for whatever exposure period happens to be observed on each policy.

```python
freq_model = sm.GLM(
    y_tr,
    X_tr,
    family=Poisson(),
    offset=np.log(exp_tr),   # <-- the crucial line
)
freq_result = freq_model.fit()
print(freq_result.summary())
```

### Interpreting the output

The summary prints log-scale coefficients, standard errors, z-statistics, and p-values. To convert to multiplicative rating relativities:

```python
coefs = freq_result.params
se    = freq_result.bse
z     = coefs / se

factor_table = pl.DataFrame({
    "factor":     feature_names,
    "log_coef":   coefs,
    "relativity": np.exp(coefs),
    "std_err":    se,
    "z_stat":     z,
    "p_value":    2 * scipy_norm.sf(np.abs(z)),
})

print(factor_table.filter(pl.col("factor").str.starts_with("ncd")).sort("factor"))
```

A relativity of 0.72 on `ncd_years_5` means a driver with 5 years of no-claims history is predicted to have 28% fewer claims than the 0-NCD base, all other factors equal.

### Sanity check on the exposure offset

The simplest sanity check is to predict on the test set and confirm that aggregate predicted claims match aggregate actual claims.

```python
# predict() returns E[N_i] = rate_i × exposure_i for each policy
pred_count_te = freq_result.predict(X_te, offset=np.log(exp_te))
pred_rate_te  = pred_count_te / exp_te   # annual rate per policy

print(f"Aggregate predicted claims: {pred_count_te.sum():.0f}")
print(f"Actual claims in test:      {y_te.sum():.0f}")
print(f"Aggregate A/E ratio:        {y_te.sum() / pred_count_te.sum():.4f}")
```

An A/E ratio near 1.0 confirms aggregate calibration. That is necessary but not sufficient — you also need calibration within each factor band, which we cover below.

---

## Overdispersion test

A Poisson GLM assumes Var(N) = mu. On any real motor book, observed variance exceeds this because of unmodelled risk heterogeneity — drivers within the same NCD band and vehicle group differ in ways the model cannot see. Ignoring overdispersion does not bias the coefficient estimates, but it makes standard errors wrong, so hypothesis tests and confidence intervals are anticonservative.

```python
dispersion = freq_result.pearson_chi2 / freq_result.df_resid

print(f"Estimated dispersion (phi): {dispersion:.3f}")
# If phi > 1.0, overdispersion is present.
# Typical UK motor frequency: phi between 1.1 and 1.5.
```

If dispersion exceeds about 1.1, use quasi-Poisson for all inference. The coefficients are identical — only the standard errors change.

```python
freq_result_quasi = sm.GLM(
    y_tr,
    X_tr,
    family=Poisson(),
    offset=np.log(exp_tr),
).fit(scale="x2")   # "x2" scales standard errors by sqrt(Pearson chi^2 / df)

# Confirm that coefficients are unchanged
np.testing.assert_allclose(freq_result.params, freq_result_quasi.params, rtol=1e-8)

print(f"Standard Poisson SE (first 3): {freq_result.bse[:3].round(4)}")
print(f"Quasi-Poisson SE  (first 3): {freq_result_quasi.bse[:3].round(4)}")
# Quasi-Poisson SEs are inflated by sqrt(phi) ≈ 1.05–1.22 for typical motor books
```

**For production models:** always use quasi-Poisson standard errors for significance tests on rating factors, confidence intervals on relativities, and any report going to a pricing committee.

The F-test replaces the chi-squared test for comparing nested models under quasi-likelihood:

```python
def quasi_poisson_ftest(result_base, result_full):
    """F-test for comparing nested quasi-Poisson models.

    The standard chi-squared deviance test is anticonservative under
    overdispersion. This F-test divides by the estimated dispersion.
    """
    delta_dev = result_base.deviance - result_full.deviance
    delta_df  = result_base.df_resid - result_full.df_resid
    phi       = result_full.scale
    f_stat    = (delta_dev / delta_df) / phi
    p_val     = f_dist.sf(f_stat, dfn=delta_df, dfd=result_full.df_resid)
    return f_stat, p_val, int(delta_df)
```

---

## Severity model: Gamma GLM

The severity model is fitted only on policies with at least one claim. The Gamma GLM with log link models E[S | N > 0], where S is average claim amount.

```python
claimed_mask = y_tr > 0
y_sev_tr     = df["claim_amount"].to_numpy()[train_idx][claimed_mask]
X_sev_tr     = X_tr[claimed_mask]

# No exposure offset for severity.
# We are modelling average severity per claim, not severity per policy-year.
sev_model  = sm.GLM(
    y_sev_tr,
    X_sev_tr,
    family=Gamma(link=LogLink()),
)
sev_result = sev_model.fit()

sev_table = pl.DataFrame({
    "factor":     feature_names,
    "relativity": np.exp(sev_result.params),
    "std_err":    sev_result.bse,
})
print(sev_table.filter(pl.col("factor").str.starts_with("vehicle")).sort("factor"))
```

### The Gamma dispersion parameter

The Gamma GLM estimates a single dispersion parameter phi = 1/alpha (where alpha is the Gamma shape), assuming the same relative volatility across all risk levels. For attritional UK motor claims, this is approximately correct. Where it breaks down: books mixing BI-heavy and OD-only claims where severity volatility differs systematically by claim type, or books with unmodelled large-loss contamination.

If your severity residuals show systematic patterns in |residual| against vehicle group or claim type, the fix is a Double GLM where phi is itself modelled as a function of covariates. `insurance-dispersion` implements this:

```python
from insurance_dispersion import DoubleGLM

dglm = DoubleGLM(family="gamma")
dglm.fit(y_sev_tr, X_sev_tr)
print(dglm.dispersion_summary())
# Shows phi_hat per covariate level.
# Systematic variation means the constant-phi model understates uncertainty
# for high-variance segments and overstates it for stable ones.
```

---

## Combining frequency and severity: pure premium

```python
# Annual frequency rate per policy (not count for this exposure period)
freq_rate_te = freq_result_quasi.predict(X_te, offset=np.log(exp_te)) / exp_te

# Expected severity for every policy (severity model applies to all risks,
# not just those that claimed in this period)
sev_pred_te = np.exp(X_te.dot(sev_result.params))

# Pure premium = E[N/year] × E[S | claim]
pure_premium_te = freq_rate_te * sev_pred_te

print(f"Pure premium — mean:   £{pure_premium_te.mean():.2f}")
print(f"Pure premium — median: £{np.median(pure_premium_te):.2f}")
print(f"Pure premium — P90:    £{np.percentile(pure_premium_te, 90):.2f}")
```

---

## Factor tables

A GLM coefficient table is not the same as a rating factor table. For a factor like NCD years (0–9), you want a table showing the relativity for each level relative to the base, with the base level explicitly included.

```python
def factor_table_from_glm(result, feature_names: list[str],
                           enc: OneHotEncoder, cat_factors: list[str],
                           cont_factors: list[str]) -> dict[str, pl.DataFrame]:
    """Extract per-factor relativity tables from a fitted statsmodels GLM result."""
    tables = {}
    params = result.params
    bse    = result.bse

    # Continuous factors: single-row table (relativity is per unit change)
    for col in cont_factors:
        idx = feature_names.index(col)
        tables[col] = pl.DataFrame({
            "factor":     [col],
            "level":      ["(continuous — per unit)"],
            "log_coef":   [params[idx]],
            "relativity": [np.exp(params[idx])],
            "std_err":    [bse[idx]],
        })

    # Categorical factors: base level (relativity=1.0) plus each non-reference level
    offset = 1 + len(cont_factors)   # 1 for intercept column
    for i, col in enumerate(cat_factors):
        non_base_cats = enc.categories_[i][1:]   # drop="first" removes the base
        n    = len(non_base_cats)
        idx  = slice(offset, offset + n)

        tables[col] = pl.DataFrame({
            "factor":     [col] * (n + 1),
            "level":      [str(enc.categories_[i][0])] + list(non_base_cats.astype(str)),
            "log_coef":   [0.0] + list(params[idx]),
            "relativity": [1.0] + list(np.exp(params[idx])),
            "std_err":    [float("nan")] + list(bse[idx]),
        })
        offset += n

    return tables

freq_tables = factor_table_from_glm(
    freq_result_quasi, feature_names, enc, CAT_FACTORS, CONT_FACTORS
)

print("\nFrequency relativities for NCD years:")
print(freq_tables["ncd_years"])
```

---

## A/E validation by factor band

Statistical diagnostics catch distributional problems. The check that catches most actuarial problems is computing actual/expected ratios by band after controlling for the other factors.

```python
pred_freq_te = freq_result_quasi.predict(X_te, offset=np.log(exp_te))
df_test      = df[test_idx]

def ae_by_factor(df_test: pl.DataFrame, factor: str,
                 actual: np.ndarray, fitted: np.ndarray,
                 exposure: np.ndarray) -> pl.DataFrame:
    return (
        pl.DataFrame({
            factor:     df_test[factor].to_numpy(),
            "actual":   actual,
            "fitted":   fitted,
            "exposure": exposure,
        })
        .group_by(factor)
        .agg([
            pl.col("actual").sum(),
            pl.col("fitted").sum(),
            pl.col("exposure").sum(),
        ])
        .with_columns(
            (pl.col("actual") / pl.col("fitted")).alias("ae_ratio")
        )
        .sort(factor)
    )

for factor in CAT_FACTORS:
    tbl = ae_by_factor(df_test, factor, y_te, pred_freq_te, exp_te)
    print(f"\nA/E by {factor}:")
    print(tbl)
```

Any factor band where A/E is consistently outside [0.85, 1.15] on a well-populated segment is telling you the model is not capturing the risk correctly. Common causes: incorrect banding (adjacent risk levels treated as identical when they differ), a missing interaction, or nonlinearity that the linear predictor cannot represent.

For vehicle age and driver age, the A/E check will often reveal that initial banding decisions need revising. `insurance-glm-tools`'s `FactorClusterer` finds the statistically optimal grouping using a fused lasso penalty:

```python
from insurance_glm_tools.cluster import FactorClusterer

fc = FactorClusterer(family="poisson", lambda_="bic", min_exposure=300)
fc.fit(
    df[train_idx].select(["vehicle_age", "ncd_years"]).to_pandas(),
    y_tr,
    exposure=exp_tr,
    ordinal_factors=["vehicle_age", "ncd_years"],
)

print("\nOptimal vehicle age grouping (BIC-selected):")
print(fc.level_map("vehicle_age").to_df())
```

BIC selects the regularisation strength that prevents over-splitting on noisy data, and `min_exposure=300` prevents groups with thin data from standing alone. The output replaces the hand-drawn band structure with one that is reproducible and directly justifiable in a model documentation pack.

---

## Model selection: the F-test, not AIC

When you add a rating factor to a quasi-Poisson GLM, how do you know it is genuinely improving the model? AIC is not defined under quasi-likelihood. The chi-squared deviance test is anticonservative because it ignores the overdispersion. The correct test is the F-test on deviance reduction, using the estimated phi as the denominator.

```python
# Base model: NCD years and vehicle group only
enc_base   = OneHotEncoder(drop="first", sparse_output=True)
X_cat_base = enc_base.fit_transform(df[train_idx].select(["ncd_years", "vehicle_group"]).to_numpy().astype(str))
X_base     = sp.hstack([np.ones((len(train_idx), 1)), X_cont[train_idx], X_cat_base], format="csc")
freq_base  = sm.GLM(y_tr, X_base, family=Poisson(), offset=np.log(exp_tr)).fit(scale="x2")

# Full model adds region_band
f_stat, p_val, ddf = quasi_poisson_ftest(freq_base, freq_result_quasi)
print(f"F-test for region_band: F({ddf}, {freq_result_quasi.df_resid:.0f}) = {f_stat:.3f}, p = {p_val:.4f}")
# p < 0.05 means region_band adds explanatory power after controlling for the base factors
```

---

## Recovering the true parameters

Because we used synthetic data, we can verify that the GLM recovers the known data-generating coefficients. This is the test that statsmodels tutorials never show you, because they use real data where the ground truth is unknown.

```python
print(f"\n{'Factor':<30} {'True':>10} {'Estimated':>12} {'Bias':>10}")
print("-" * 64)

for factor, true_val in true_params["freq"].items():
    if factor in feature_names:
        idx = feature_names.index(factor)
        est = freq_result_quasi.params[idx]
        print(f"{factor:<30} {true_val:>10.4f} {est:>12.4f} {est - true_val:>10.4f}")
```

With 50,000 policies and a clean Poisson DGP, the GLM recovers true log-coefficients to within ±0.01–0.02 for well-populated factor levels. Thin levels — rare vehicle groups, low-exposure region bands — will have wider estimation error, which is exactly where the A/E checks above will flag problems.

---

## What this tutorial has not covered

A production UK motor pricing GLM requires more than what we have shown here:

**Factor interactions.** The log-link GLM has implicit multiplicative interactions (young driver penalty × vehicle group relativity). Crossed interactions — where the young-driver penalty is larger for high-performance vehicles — require explicit interaction terms. `insurance-interactions` tests for these systematically using CANN and NID scores, so you are not guessing which interactions matter.

**Spatial smoothing for territory factors.** Postcode district factors estimated directly from a GLM are noisy because exposure per district varies enormously. `insurance-spatial` fits a BYM2 hierarchical model that borrows strength from neighbouring districts before banding.

**Credibility blending for thin segments.** New vehicle groups, unusual occupation classes, recently-added area codes — these have insufficient exposure for stable GLM estimates. `insurance-credibility` blends the direct GLM estimate with the portfolio complement using Bühlmann-Straub weights.

**Large-loss treatment.** The Gamma severity model is disproportionately influenced by a handful of large claims — BI claims above £50k, major property damage. Cook's distance diagnostics identify which claims are doing this; the decision about whether to cap them or model them separately belongs in the actuarial sign-off.

**Formal validation pack.** Under PRA SS1/23, a motor pricing model requires quantitative pass/fail tests: Gini with bootstrap CI, Hosmer-Lemeshow goodness-of-fit, PSI on score distributions, calibration slope. `insurance-governance` automates these and produces an auditable HTML report in under five seconds.

---

## Summary

Three things in a statsmodels GLM for motor insurance pricing that are non-negotiable:

1. **Exposure offset.** `offset=np.log(exposure)` in the `GLM()` call. Without it, the model predicts an expected count for an arbitrary period rather than an annual rate. This is the most common mistake in Python GLM implementations.

2. **Quasi-Poisson for inference.** Use `.fit(scale="x2")` and the F-test for model comparison. Standard Poisson standard errors understate uncertainty by 10–22% on a typical UK motor book (dispersion 1.1–1.5).

3. **A/E by band as the primary validation check.** Statistical diagnostics catch distributional problems; A/E by factor band catches actuarial ones. Both are required before signing off a model.

---

**Libraries used in this post:**

- `statsmodels` — GLM with Poisson and Gamma families, exposure offset, quasi-likelihood
- [`insurance-datasets`](https://github.com/burning-cost/insurance-datasets) — synthetic UK motor data with known DGP for ground-truth validation
- [`insurance-glm-tools`](https://github.com/burning-cost/insurance-glm-tools) — automated factor banding via R2VF fused lasso
- [`insurance-dispersion`](https://github.com/burning-cost/insurance-dispersion) — Double GLM for covariate-driven dispersion in severity models
- [`insurance-cv`](https://github.com/burning-cost/insurance-cv) — temporal walk-forward splits for honest prospective evaluation

**Related posts:**

- [GLM Assumptions in Insurance Pricing: What Actually Matters]({{ site.baseurl }}{% post_url 2026-03-23-glm-assumptions-insurance-pricing-what-actually-matters %}) — which violations matter, which you can live with, and the diagnostics worth running
- [Tweedie Regression for Insurance: What sklearn Doesn't Tell You About Exposure]({{ site.baseurl }}{% post_url 2026-03-21-tweedie-regression-insurance-what-sklearn-doesnt-tell-you %}) — why `TweedieRegressor` fails on books with exposure variation, and the statsmodels alternative
- [Your Frequency-Severity Independence Assumption Is Costing You Premium]({{ site.baseurl }}{% post_url 2025-05-14-frequency-severity-independence-is-costing-you-premium %}) — the NCD-driven correlation between frequency and severity, and how to correct for it analytically
- [Optimal Binning for GLM Rating Factors: Beyond the Eyeball Test]({{ site.baseurl }}{% post_url 2026-03-14-your-factor-banding-is-made-up %}) — R2VF automated factor clustering for vehicle age, NCD, and geography
- [Why k-Fold CV Is Wrong for Insurance]({{ site.baseurl }}{% post_url 2026-03-21-why-k-fold-cv-is-wrong-for-insurance %}) — why k-fold is wrong for insurance and how to set up temporal holdouts correctly
