---
layout: post
title: "GLMs for UK Insurance Pricing in Python: What the Generic Tutorials Miss"
date: 2026-03-22
author: Burning Cost
categories: [python, glm, insurance-pricing]
tags: [glm, python, uk-insurance, glum, polars, catboost, freMTPL2, poisson, gamma, frequency-severity, exposure, NCD, ABI-group, postcode, FCA, Consumer-Duty, Emblem, Radar, IBNR, offsets, insurance-distill, shap-relativities, insurance-cv]
description: "GLM insurance Python for UK pricing actuaries: exposure handling, Consumer Duty, frequency-severity split, and Emblem/Radar deployment with glum."
---

Every GLM insurance Python tutorial does the same thing. Load freMTPL2, fit a Poisson with sklearn, plot a lift curve, call it a day. If you are a UK pricing actuary who has used Emblem or Radar professionally, those tutorials are missing about 80% of what you actually care about.

This is not a GLM tutorial. We assume you know what a link function is, you have fitted frequency-severity models before, and your question is: *how do I do this properly in Python, with the same rigour I would apply in Emblem?*

The answer centres on [glum](https://github.com/Quantco/glum) for the GLM fitting, Polars for data handling, and a set of UK-specific choices that differ from what generic Python tutorials describe.

```bash
uv pip install glum polars scikit-learn catboost
uv pip install scikit-learn-intelex  # optional, speeds up preprocessing on x86
```

---

## Why glum, not sklearn

`sklearn.linear_model.PoissonRegressor` is correct. We are not disputing that. The reasons to prefer glum for production pricing work are practical:

- glum uses the Cholesky factorisation path by default, which is 3-10x faster than sklearn's IRLS on large portfolios (100k+ policies)
- glum exposes `P1` (L1) and `P2` (L2) penalty matrices per coefficient, so you can regularise different factors differently - useful when you want light regularisation on postcode but stronger regularisation on a sparse interaction term
- glum supports the full offset term natively in `fit()`, which is how you handle IBNR adjustments and large loss loadings without hacking sample weights
- glum returns coefficient standard errors. sklearn does not. You need these for the LR test when deciding whether to include an interaction

The performance difference matters in practice. A UK personal lines motor book with 500k in-force policies and 5 years of history is a big matrix. glum handles it. sklearn's PoissonRegressor can stall.

---

## Exposure: the thing everyone gets wrong

The most common mistake in Python GLM tutorials is treating exposure as a `sample_weight`. It is not.

In a Poisson GLM for claim frequency, the model is:

```
log(E[claims]) = log(exposure) + X @ beta
```

The `log(exposure)` term is an **offset** - it enters the linear predictor directly, not the variance. When you pass exposure as a sample weight, you are reweighting the deviance contributions, which is approximately correct for balanced datasets but will give subtly wrong coefficient estimates when exposure is correlated with your rating factors.

In UK motor, exposure is measured in earned car-years - the fraction of the policy period falling in the accident year. A policy incepting 1 October with a 12-month term contributes 0.25 earned car-years to the current accident year and 0.75 to the next.

```python
import polars as pl
import numpy as np
from sklearn.datasets import fetch_openml

# freMTPL2freq is the standard actuarial benchmark dataset
# French motor third-party liability, ~678k policies
raw = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
df_pd = raw.data.copy()
df_pd["ClaimNb"] = raw.target

# Convert to Polars - we work in Polars throughout
df = pl.from_pandas(df_pd).with_columns([
    pl.col("Exposure").clip(0.0, 1.0).alias("exposure"),  # cap at 1.0 car-year
    pl.col("ClaimNb").cast(pl.Int32).alias("claim_count"),
])

print(df.select(["exposure", "claim_count", "VehPower", "BonusMalus", "Region"]).head())
```

The exposure cap at 1.0 matters. freMTPL2 has some exposure values slightly above 1.0 due to mid-term adjustments; leaving them uncapped biases the offset.

For a UK portfolio you will have something cleaner: `exposure = (min(policy_expiry, year_end) - max(policy_inception, year_start)).days / 365.25`. Compute this before modelling, cap at 1.0, and use it as the offset.

**Mid-term adjustments (MTAs)** complicate this. A UK motor policy that changes vehicle mid-year generates two records: pre-MTA and post-MTA. The standard treatment is to split the policy record at the MTA date and assign separate exposure fractions. Many UK systems store MTAs as separate transactions; you may need to aggregate to a policy-year record before fitting. Do not sum exposures naively - if a policy has three MTAs in a year, the three exposure fractions should sum to at most 1.0.

**Earned vs written** is the other distinction. A GLM fitted on earned exposure predicts accident-year frequency. A GLM fitted on written exposure predicts policy-year frequency. For UK personal lines pricing (rating future new business), you want earned exposure aligned to the accident year your claims data covers. For reserving inputs, you might want written.

---

## UK rating factors: what differs from generic examples

freMTPL2 uses French-specific variables. The mapping to UK motor factors is approximate:

| freMTPL2 variable | UK equivalent | Notes |
|---|---|---|
| `BonusMalus` (0-350 scale) | NCD years (0-5 typically) | French scale runs higher; UK NCD is capped at 5 years protected/unprotected |
| `Region` (22 French regions) | Postcode sector (~ 9,000 in UK) | UK postcode is the highest-cardinality factor |
| `VehPower` | ABI vehicle group (1-50) | Composite score: performance, security, repairability |
| `VehBrand` | Make and model | UK uses ABI group as the rating dimension, not raw make |
| `VehAge` | Vehicle age | Direct equivalent |
| `DrivAge` | Age of youngest named driver | UK uses youngest, not policyholder age |
| `Density` | Urban/rural flag or IMD decile | Postcode density is correlated with UK area factor |

### NCD in a UK GLM

No Claims Discount in the UK is both a rating factor and a policy structure. The standard treatment is to include NCD years (0-5) as a categorical variable or a step-function smooth. Treating it as continuous is wrong: the marginal difference between 0 and 1 year NCD is much larger than between 4 and 5 years.

In Emblem, NCD is almost always treated as a group factor with manual relativities confirmed by data. In Python with glum, you encode it as a categorical and use L2 regularisation to shrink towards the actuarial prior:

```python
from glum import GeneralizedLinearRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import scipy.sparse as sp

# Prepare features
ncd_encoder = OneHotEncoder(
    categories=[list(range(6))],  # NCD 0-5
    drop="first",
    sparse_output=True,
)
```

### Postcode sector

Postcode sector (first 4-5 characters of UK postcode: `SW1A 1`, `M1 1`) has ~9,000 levels. At that cardinality, one-hot encoding is impractical and standard L2 regularisation is not granular enough - you need a hierarchical or credibility-weighted approach.

The standard UK actuarial treatment is to fit a separate spatial smoothing model on postcode (Whittaker smoother, kriging, or a hierarchical GLM with postcode within postcode district) and use the smoothed postcode relativity as an offset in the main GLM. This separates the signal from the sparse postcode cell problem.

In Python, a Whittaker smoother over postcode sector-level A/E ratios is the closest equivalent to what Emblem's geographic smoothing does. Our [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) library implements this directly for postcode-level rating factor derivation.

### Voluntary excess

Voluntary excess is a selection factor that generic tutorials never mention. Policyholders who choose a higher voluntary excess tend to be lower-risk (self-selection). The net effect depends on the compulsory excess structure and the book mix. Most UK GLMs include a small number of excess bands (no voluntary, £100, £250, £500+) as a group factor. The selection effect means the coefficient is not directly interpretable as the pure excess removal effect - it is a mix of selection and exposure adjustment.

### ABI vehicle group

The ABI group (1-50 for cars) is a composite score maintained by Thatcham Research covering repair costs, car performance, and security. It is the standard UK motor rating dimension for vehicle risk. freMTPL2's `VehPower` is a rough proxy.

Unlike a raw power or engine size variable, ABI group already encodes actuarial judgement from Thatcham. Including it in a GLM as a continuous variable (assuming linearity across 1-50) misses the non-linearities between groups. We bin it into 8-10 bands or use it as a small-cardinality categorical.

---

## The frequency-severity split

UK actuarial practice fits two separate GLMs: one for claim frequency (Poisson, outcome = claim count, offset = log(exposure)) and one for claim severity (Gamma, outcome = average severity conditional on a claim, weights = claim counts). The pure premium is frequency times severity.

This is different from fitting a single Tweedie model on loss ratios. The reasons to maintain the split:

1. Frequency and severity have different rating factor relationships. NCD is a strong frequency factor but a weak severity factor. Vehicle value is a strong severity factor. A single Tweedie conflates these signals.
2. Regulatory reporting in the UK typically requires frequency and severity components separately (for Lloyd's, for Solvency II internal models, for FCA pricing practice reviews).
3. For Consumer Duty purposes, you need to be able to explain which component of price is driven by which factor. "Frequency effect of NCD = 0.78, severity effect of NCD = 0.94" is an auditable statement. A single Tweedie coefficient is not.

```python
import polars as pl
import numpy as np
from glum import GeneralizedLinearRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load and prep freMTPL2 ---
from sklearn.datasets import fetch_openml
import pandas as pd

raw_freq = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
raw_sev  = fetch_openml("freMTPL2sev",  version=1, as_frame=True, parser="auto")

df_freq = pl.from_pandas(
    raw_freq.data.assign(ClaimNb=raw_freq.target.astype(int))
).with_columns([
    pl.col("Exposure").clip(0.0, 1.0).alias("exposure"),
    pl.col("ClaimNb").alias("claim_count"),
    pl.col("BonusMalus").cast(pl.Float64),
    pl.col("VehPower").cast(pl.Int32),
    pl.col("VehAge").cast(pl.Int32),
    pl.col("DrivAge").cast(pl.Int32),
])

# Severity: aggregate claim amounts per policy
df_sev_raw = pl.from_pandas(
    raw_sev.data.assign(ClaimAmount=raw_sev.target.astype(float))
)
df_sev_agg = df_sev_raw.group_by("IDpol").agg([
    pl.col("ClaimAmount").sum().alias("total_claim_amount"),
    pl.len().alias("sev_claim_count"),
])

# Join severity back onto frequency data
df = df_freq.join(df_sev_agg, on="IDpol", how="left").with_columns([
    pl.col("total_claim_amount").fill_null(0.0),
    pl.col("sev_claim_count").fill_null(0).cast(pl.Int32),
    # Average severity (only defined for policies with claims)
    (pl.col("total_claim_amount") / pl.col("sev_claim_count"))
        .alias("avg_severity"),
])

print(f"Policies: {len(df):,}")
print(f"Policies with claims: {df.filter(pl.col('claim_count') > 0).shape[0]:,}")
print(f"Mean exposure: {df['exposure'].mean():.3f}")
print(f"Mean frequency: {(df['claim_count'].sum() / df['exposure'].sum()):.4f}")
```

### Frequency GLM

```python
# Features for frequency model
freq_features = ["VehPower", "VehAge", "DrivAge", "BonusMalus",
                 "VehBrand", "VehGas", "Density", "Region"]

# Build feature matrix - work in pandas for sklearn compatibility
X_pd = df.select(freq_features).to_pandas()
y_freq = df["claim_count"].to_numpy().astype(float)
exposure = df["exposure"].to_numpy()

# Preprocessing: categorical one-hot, continuous pass-through
cat_features = ["VehBrand", "VehGas", "Region"]
num_features = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_features),
    ("num", "passthrough", num_features),
])

X_enc = preprocessor.fit_transform(X_pd)

# glum Poisson frequency model with log(exposure) offset
freq_model = GeneralizedLinearRegressor(
    family="poisson",
    link="log",
    alpha=1e-4,          # L2 penalty - light regularisation
    solver="irls-cd",    # default; use 'lbfgs' for very large matrices
    max_iter=100,
    fit_intercept=True,
)

freq_model.fit(
    X_enc,
    y_freq,
    offset=np.log(exposure),   # <-- offset, not sample_weight
)

y_pred_freq = freq_model.predict(X_enc, offset=np.log(exposure))

# Gini coefficient (one-sample, standard actuarial metric)
from sklearn.metrics import auc
def gini_coefficient(y_true, y_pred, w):
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    w_sorted = w[order]
    cum_claims = np.cumsum(y_sorted * w_sorted)
    cum_exposure = np.cumsum(w_sorted)
    cum_claims  /= cum_claims[-1]
    cum_exposure /= cum_exposure[-1]
    lorenz_area = auc(cum_exposure, cum_claims)
    return 1 - 2 * lorenz_area

gini_freq = gini_coefficient(y_freq, y_pred_freq, exposure)
print(f"Frequency Gini: {gini_freq:.4f}")
# Typical range for freMTPL2: 0.28 - 0.35
```

### Severity GLM

```python
# Filter to policies with at least one claim for severity model
sev_mask = df["sev_claim_count"].to_numpy() > 0
X_sev = X_enc[sev_mask]
y_sev = df["avg_severity"].to_numpy()[sev_mask]
w_sev = df["sev_claim_count"].to_numpy()[sev_mask].astype(float)

sev_model = GeneralizedLinearRegressor(
    family="gamma",
    link="log",
    alpha=1e-4,
    max_iter=100,
    fit_intercept=True,
)

# For severity: claim counts are the weights (not an offset)
# We are modelling E[severity | claim], weighted by number of claims
sev_model.fit(X_sev, y_sev, sample_weight=w_sev)

y_pred_sev_at_claims = sev_model.predict(X_sev)
print(f"Mean predicted severity: £{y_pred_sev_at_claims.mean():,.0f}")
```

### Pure premium

```python
# Predict pure premium = freq * severity for all policies
y_pred_sev_all = sev_model.predict(X_enc)
y_pred_pp = y_pred_freq * y_pred_sev_all

print(f"Mean predicted pure premium: £{y_pred_pp.mean():,.2f}")
```

---

## Offsets for IBNR and large loss loading

In UK commercial and some personal lines work, you need to adjust the GLM for two additional factors that do not belong in the rating factor structure:

**IBNR loading**: accident years near the data cutoff are underreported. If your latest accident year is 2024, claims from Q3/Q4 2024 are still developing. You can estimate an IBNR factor per accident quarter using a chain-ladder development method, then include `log(ibnr_factor)` as an offset in the frequency GLM. This prevents the GLM from treating recent underreported accident years as genuinely lower frequency.

**Large loss loading**: severity distributions have heavy tails. A single large loss in a thin rating cell can distort the Gamma GLM coefficients. The standard UK actuarial treatment is to cap individual claim amounts at a threshold (e.g., the 95th percentile), fit the capped severity GLM, and apply a separate large loss loading factor to the final pure premium.

```python
# Large loss capping - cap severity at 95th percentile before fitting Gamma GLM
sev_cap = float(np.percentile(y_sev, 95))
y_sev_capped = np.minimum(y_sev, sev_cap)
large_loss_factor = y_sev.mean() / y_sev_capped.mean()

sev_model_capped = GeneralizedLinearRegressor(
    family="gamma",
    link="log",
    alpha=1e-4,
    max_iter=100,
)
sev_model_capped.fit(X_sev, y_sev_capped, sample_weight=w_sev)

print(f"Large loss loading factor: {large_loss_factor:.4f}")
# Apply this multiplicatively to pure premium predictions
# y_pred_pp_final = y_pred_freq * sev_model_capped.predict(X_enc) * large_loss_factor
```

For IBNR offsets, you would compute these externally from your reserving triangle and pass them as an additional column:

```python
# Schematic - you supply ibnr_factors from your reserving model
# ibnr_factors: Polars Series, one per policy, typically 1.0 for mature years,
# 1.05-1.20 for the most recent accident quarter

ibnr_offset = np.log(df["ibnr_factor"].to_numpy())  # df must have this column
freq_model.fit(
    X_enc,
    y_freq,
    offset=np.log(exposure) + ibnr_offset,
)
```

---

## FCA pricing practices and Consumer Duty implications

The FCA's General Insurance Pricing Practices (GIPP) rules (effective 1 January 2022) changed what a UK GLM is allowed to do in personal lines. Consumer Duty (July 2023) pushed further. The direct implications for GLM construction:

**Price walking is prohibited.** Your GLM cannot include renewal tenure as a variable. "Number of previous renewals" or "years as customer" cannot be rating factors. If your data includes any proxy for renewal history - policy inception date pattern, marketing channel mapped to tenure segment - you need to check whether it is correlated with renewal number. If it is, including it in your GLM is a regulatory problem.

**Fair value obligation.** Under Consumer Duty, pricing teams must be able to demonstrate that each rating factor contributes to a genuine assessment of risk, not to market segmentation or price optimisation at the expense of some customer groups. This means the actuarial justification for each factor needs to be auditable: the lift it provides, the deviance improvement, the economic interpretation. A factor that improves Gini but has no plausible risk mechanism is a liability.

**Proxy discrimination (Consumer Duty, also FCA DP21/1).** A factor that is uncorrelated with risk but correlated with a protected characteristic - via a proxy route - is discriminatory under Equality Act and Consumer Duty. Postcode, in particular, is under scrutiny: inner-city postcodes correlate with ethnicity. A UK GLM that includes dense urban postcodes as a straight rating factor needs a proxy discrimination test alongside the standard actuarial tests.

The [insurance-fairness](https://github.com/burning-cost/insurance-fairness) library implements the Bayes Business School optimal transport method for proxy discrimination testing in Python. For a GLM rate review, running this before sign-off is now a reasonable standard of care.

```bash
# install insurance-fairness
uv add insurance-fairness
```

```python
from insurance_fairness import detect_proxies

# Detect which rating factors act as statistical proxies for a protected attribute
result = detect_proxies(
    df=df_pl,
    protected_col="postcode_minority_pct",   # from Census lookup, in your Polars df
    factor_cols=freq_features,
    exposure_col="exposure",
)
print(result.summary())
```

We covered this in detail in [How to Detect Proxy Discrimination in a UK Motor Insurance GLM](https://burning-cost.github.io/2026/03/22/fca-proxy-discrimination-python-testing-guide/).

---

## Cross-validation for GLMs: why k-fold is wrong

Standard k-fold cross-validation on insurance data gives optimistic CV scores. The reasons are temporal: claims from policy years in your training set may not have fully developed, and randomly-assigned test folds see future claim patterns as training data.

For GLM model selection - choosing regularisation strength, deciding whether an interaction term is worth including - you need temporal walk-forward splits. glum does not handle this internally; you structure the CV loop yourself.

[insurance-cv](https://github.com/burning-cost/insurance-cv) provides walk-forward splits that respect policy year structure and include an IBNR development buffer:

```bash
uv add insurance-cv
```

```python
from insurance_cv import walk_forward_split, InsuranceCV
import polars as pl

# Build walk-forward folds — expanding training window, 6-month IBNR buffer
splits = walk_forward_split(
    df=df,
    date_col="inception_year",
    ibnr_buffer_months=6,
)
cv = InsuranceCV(splits=splits, df=df)

for fold, (train_idx, test_idx) in enumerate(cv.split(X_enc)):
    X_train = X_enc[train_idx]
    X_test  = X_enc[test_idx]
    y_train = y_freq[train_idx]
    y_test  = y_freq[test_idx]
    w_train = exposure[train_idx]
    w_test  = exposure[test_idx]

    m = GeneralizedLinearRegressor(family="poisson", link="log", alpha=1e-4)
    m.fit(X_train, y_train, offset=np.log(w_train))
    y_hat = m.predict(X_test, offset=np.log(w_test))

    gini = gini_coefficient(y_test, y_hat, w_test)
    print(f"Fold {fold}: Gini = {gini:.4f}")
```

The fold-by-fold Gini trajectory is the key output - not the mean. If Gini declines monotonically across folds (earlier training year predicting later test year), that tells you the model is degrading and needs a trend term or more frequent refitting.

---

## Getting coefficients out: the Emblem/Radar deployment path

You have fitted a glum GLM. The rating engine (Emblem, Radar, or a bespoke system) expects a factor table: one row per rating factor level, with a multiplicative relativity.

glum's coefficient vector is in log space: `exp(coef)` gives you the multiplicative relativity. The complication is that your preprocessing (OneHotEncoder + passthrough) has scrambled the feature names.

```python
# Reconstruct feature names after ColumnTransformer preprocessing
ohe_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
all_feature_names = list(ohe_feature_names) + num_features

assert len(all_feature_names) == len(freq_model.coef_)

# Build factor table
import polars as pl

factor_table = pl.DataFrame({
    "feature": all_feature_names,
    "log_coef": freq_model.coef_,
    "relativity": np.exp(freq_model.coef_),
    "std_err": freq_model.std_errors_,  # glum provides these; sklearn does not
    "z_stat": freq_model.coef_ / freq_model.std_errors_,
}).sort("relativity", descending=True)

print(factor_table.head(10))
# | feature              | log_coef | relativity | std_err | z_stat |
# | Region_R24           | 0.412    | 1.510      | 0.031   | 13.3   |
# | BonusMalus           | 0.018    | 1.018      | 0.001   | 24.1   |
# ...
```

The `std_errors_` attribute is why we use glum. You can compute confidence intervals and decide whether a coefficient is stable enough to include in the rating table, which is exactly the decision you would make in Emblem when reviewing factor credibility.

### One-hot encoding and the reference level

When you `drop="first"` in OneHotEncoder, the dropped level is your reference. In Emblem, the reference level is typically set to the highest-exposure cell. In Python, `drop="first"` drops the first alphabetical category, which is rarely the highest-exposure cell. This does not affect model predictions but makes the relativities harder to interpret and compare to Emblem output.

Fix: use `drop="if_binary"` for binary features and manually set the reference level for your key factors. Or rebase the factor table after fitting:

```python
# Rebase: set reference level to exposure-weighted mean = 1.0 (log = 0.0)
# This is the actuarial standard: all relativities are relative to the base risk

def rebase_factor_table(ft: pl.DataFrame, feature_prefix: str, exposure_col: pl.Series) -> pl.DataFrame:
    # ft has columns: feature, relativity
    # Returns rebased table with exposure-weighted mean relativity = 1.0
    mask = ft["feature"].str.starts_with(feature_prefix)
    rels = ft.filter(mask)["relativity"].to_numpy()
    # Placeholder: proper rebasing requires the exposure per level
    # In practice, extract from the design matrix
    return ft  # expand for production use
```

---

## The GBM-to-GLM path

If you are following the standard UK actuarial workflow - fit a GBM to find the right structure, then distil into a GLM for deployment - the Python path is:

1. Fit a CatBoost frequency model (see [Extracting Rating Relativities from GBMs with SHAP](/2026/03/02/how-to-extract-rating-factors-from-catboost/))
2. Use [shap-relativities](https://github.com/burning-cost/shap-relativities) to extract a continuous age curve and group-level relativities - this replaces the manual Emblem factor derivation step
3. Use [insurance-distill](https://github.com/burning-cost/insurance-distill) to fit a surrogate GLM on the GBM's pseudo-predictions, with optimal binning - this is the Emblem-compatible factor table
4. Validate the surrogate GLM's Gini against the GBM

```bash
uv add "shap-relativities[all]"
uv add "insurance-distill[catboost]"
```

```python
from catboost import CatBoostRegressor, Pool
from insurance_distill import SurrogateGLM
from shap_relativities import SHAPRelativities

# 1. Fit CatBoost - use Pool for native categorical support
cat_cols = ["VehBrand", "VehGas", "Region"]
X_pd_cat = df.select(freq_features).to_pandas()

train_pool = Pool(
    data=X_pd_cat,
    label=y_freq,
    weight=exposure,
    cat_features=cat_cols,
)

cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    loss_function="Poisson",
    eval_metric="RMSE",
    verbose=0,
)
cb_model.fit(train_pool)

# 2. Extract SHAP relativities
sr = SHAPRelativities(model=cb_model, X=df.select(freq_features))
relativities = sr.extract_relativities()
age_curve = sr.extract_continuous_curve("DrivAge")

# 3. Surrogate GLM
X_polars = df.select(freq_features)  # Polars DataFrame
surrogate = SurrogateGLM(
    model=cb_model,
    X_train=X_polars,
    y_train=y_freq,
    exposure=exposure,
    family="poisson",
)
surrogate.fit(max_bins=10)

report = surrogate.report()
print(report.metrics.summary())
# Gini (GBM):           0.3241
# Gini (GLM surrogate): 0.3087
# Gini ratio:           95.2%

# Export factor tables
surrogate.export_csv("output/factors/", prefix="motor_freq_")
```

The `export_csv` output is directly loadable into Emblem or Radar as a factor import file (with some header reformatting). We covered the Emblem import format in detail in [From CatBoost to Radar in 50 Lines of Python](https://burning-cost.github.io/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/).

---

## What to do with interactions

A 12-factor UK motor GLM has 66 pairwise interactions. You will not test them all in Emblem. In Python, you can run an automated interaction search using likelihood-ratio tests and get a ranked shortlist.

The [insurance-interactions](https://github.com/burning-cost/insurance-interactions) library does this with CANN residual analysis. We covered this in [Finding the Interactions Your GLM Missed](https://burning-cost.github.io/2026/02/27/finding-the-interactions-your-glm-missed/).

The key point for glum: once you have identified the interaction candidates, you can include them explicitly in the design matrix or use glum's `P2` penalty matrix to apply a stronger regularisation to interaction terms:

```python
# Stronger regularisation on interaction terms vs main effects
# glum accepts a per-coefficient L2 penalty vector
n_main = X_enc.shape[1]
n_interactions = X_int.shape[1]  # your interaction columns

# 10x stronger penalty on interactions than main effects
alpha_vector = np.concatenate([
    np.full(n_main, 1e-4),
    np.full(n_interactions, 1e-3),
])

freq_model_with_interactions = GeneralizedLinearRegressor(
    family="poisson",
    link="log",
    alpha=1.0,         # scale factor; actual penalty = alpha * P2
    P2=alpha_vector,   # per-coefficient penalty
)
```

---

## A note on glum vs Emblem parity

You will not get identical coefficients from glum and Emblem on the same data. The reasons:

- Emblem uses exact IRLS with no regularisation by default; glum with `alpha=0` approaches this but the solvers differ slightly
- Emblem handles aliased parameters (empty cells) by zeroing the coefficient; glum's IRLS-CD solver handles this differently
- Emblem's exposure treatment uses a multiplicative offset that is implicit in the model structure; the explicit `offset=log(exposure)` in glum is mathematically equivalent but numerically different path

The test for parity is not "do the coefficients match to 4 decimal places" but "do the predicted frequencies match to within rounding error, and does the factor table tell the same actuarial story."

For a GLM with ~100k policies and well-populated cells, you should get Gini coefficients within 0.005 of each other and relativities within 2-3% on any factor with more than 1,000 exposure units.

---

## Summary

The gap between the generic Python GLM tutorial and what a UK pricing actuary needs is mostly about rigour in the inputs and outputs, not in the modelling library itself:

- **Exposure as offset**, not sample weight
- **Frequency-severity split**, not a single Tweedie
- **UK-specific factors**: NCD, ABI group, postcode sector with credibility smoothing, voluntary excess
- **IBNR and large loss offsets** to prevent recent-year bias
- **Temporal cross-validation** to get honest CV scores
- **Coefficient standard errors** for factor significance decisions (use glum, not sklearn)
- **Consumer Duty and FCA GIPP compliance**: no tenure factors, proxy discrimination testing on postcode
- **Factor table export** in a format the rating engine can consume

glum handles all of this correctly. The code in this post is the working foundation; adapt the feature list to your UK portfolio and your Emblem/Radar deployment format.

---

*Libraries referenced in this post: [glum](https://github.com/Quantco/glum), [insurance-cv](https://github.com/burning-cost/insurance-cv), [insurance-distill](https://github.com/burning-cost/insurance-distill), [shap-relativities](https://github.com/burning-cost/shap-relativities), [insurance-fairness](https://github.com/burning-cost/insurance-fairness), [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)*

- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — the LRTW framework and `insurance-fairness` library for the proxy discrimination audit mentioned in this post
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture for UK Insurance Pricing](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — when you need to move beyond the GLM while keeping the factor-table output format
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — how to take the GLM's validation metrics (Gini CI, A/E by segment, Hosmer-Lemeshow) and package them into a compliant MRM artefact
