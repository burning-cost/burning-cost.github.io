---
layout: post
title: "How to Build a Burning Cost Model for Insurance Pricing in Python"
date: 2026-03-23
categories: [python, pricing, tutorials]
tags: [burning-cost, frequency-severity, glm, poisson, gamma, glum, polars, exposure, large-loss, IBNR, python, uk-motor, insurance-pricing]
description: "Build a burning cost model in Python: frequency-severity split, exposure offsets, large loss capping, IBNR adjustment, and combined pure premium for UK pricing."
---

A burning cost model is the foundation of most UK personal lines pricing. The name comes from reinsurance: the "burning cost" of a programme is the actual incurred losses divided by the premium or exposure, giving you the rate of loss. In a pricing context, a burning cost model is a predictive model of expected claim cost per unit of exposure - the pure premium before expenses, profit loading, or tax.

The majority of UK pricing teams build this as two separate models - a frequency model and a severity model - fitted independently and then multiplied together. This post explains why that structure exists, what the practical preparation steps are before you fit anything, and how to build both models in Python with glum and Polars.

If you have used Emblem or Radar professionally and want to understand how to replicate the same workflow in Python, this is the post for you. If you want to go directly to CatBoost-based burning cost, we have [a separate post on that using freMTPL2](/2026/03/22/catboost-insurance-pricing-frequency-severity-fremtpl2/).

---

## Why split frequency and severity

The combined claim cost for a policy is:

```
Pure premium = E[frequency] * E[severity]
```

You could model pure premium directly - fit a Tweedie GLM to total claims / exposure and be done with it. We do not recommend this for most UK pricing work. Here is why.

Frequency and severity respond differently to rating factors. Vehicle group affects both, but in opposite directions for some risks: a van might have higher frequency (more miles, more incidents) but lower severity (slower speeds, better structural protection) than a sports car. Modelling them together forces a single set of coefficients to capture what are genuinely different relationships.

Frequency and severity also have different data requirements. Severity data is noisy - a single large claim can move the mean materially. Frequency data is much more stable. Separating them lets you apply different credibility loadings, use different regularisation strengths, and make different decisions about how much historical data to include.

Finally, it makes governance easier. When you present a rate change to underwriting, being able to say "frequency is up 3% and severity is flat" is far more interpretable than "pure premium is up 3%." The PRA's SS1/23 requirements on model explanation are easier to satisfy when your model has components that correspond to business intuitions.

---

## Data preparation: the steps before you touch a model

Most Python tutorials skip straight to `model.fit()`. In practice, most of the work is in preparing the claims data correctly. Three steps matter most.

### 1. Large loss capping

Before fitting a severity model, you need to decide what to do with large individual claims. The standard UK practice is to cap claims at a per-claim excess-of-loss threshold before fitting the attritional severity model, and then add a separate large loss loading on top.

The rationale: if you include uncapped large losses in your severity GLM, the model will try to explain the variation using your rating factors. A postcode that happened to have three large fire losses this year will get a higher severity factor, not because it is genuinely a worse risk for large losses, but because you have three observations in a class of 200 policies. The GLM will overfit.

The cap should be set actuarially, not statistically. A common starting point is the 99th percentile of the per-claim distribution, but you should also examine your reinsurance structure. If you buy per-risk excess-of-loss cover with a retention of £150,000, then capping at £150,000 is defensible and aligns your attritional model with the net position.

```python
import polars as pl

# Cap individual claims at £150k (net of reinsurance retention)
LARGE_LOSS_CAP = 150_000

claims = pl.read_parquet("claims.parquet")

claims = claims.with_columns(
    pl.col("incurred").clip(upper_bound=LARGE_LOSS_CAP).alias("incurred_capped"),
    (pl.col("incurred") > LARGE_LOSS_CAP).cast(pl.Int8).alias("is_large_loss"),
)

# Large loss loading to be added back at the end
excess_losses = (
    claims.filter(pl.col("is_large_loss") == 1)
    .select((pl.col("incurred") - pl.col("incurred_capped")).sum())
    .item()
)
attritional_total = claims.select(pl.col("incurred_capped").sum()).item()
large_loss_ratio = excess_losses / attritional_total
print(f"Large loss loading: {large_loss_ratio:.3%}")
```

Store this loading separately. You will add it back after combining the frequency and severity predictions.

### 2. IBNR adjustment

Your claims data is always incomplete. Claims that occurred in the most recent accident periods have not all been reported yet - incurred but not reported (IBNR). If you fit your frequency model on raw claim counts, you will systematically underestimate frequency for recent periods.

The standard fix is to apply a development factor to bring each accident period's reported claims up to an ultimate estimate. A chain-ladder development is fine for this purpose - you do not need a full stochastic reserve model.

```python
# Apply development factors by accident year (simplified chain-ladder)
dev_factors = {
    2021: 1.000,
    2022: 1.002,
    2023: 1.015,
    2024: 1.089,
    2025: 1.241,
}

policies = pl.read_parquet("policies.parquet")

policies = policies.with_columns(
    (
        pl.col("claim_count")
        * pl.col("accident_year").replace(dev_factors, default=1.0)
    ).alias("claim_count_ult"),
)
```

Note that you are developing claim counts, not amounts. Severity development (the change in average case reserve over time) is a separate question and usually handled through the capping threshold rather than explicit development factors.

### 3. Exposure: earned car-years, not policy count

The frequency model's dependent variable is claim count, and the offset is log(exposure). Exposure is measured in earned car-years: the proportion of the policy period falling within the accident year.

A 12-month policy incepting 1 October 2024 earns 0.25 car-years in accident year 2024 and 0.75 in accident year 2025. Getting this right matters: UK motor portfolios have seasonal inception patterns, so if you measure exposure as policy count rather than earned car-years, you will get biased frequency estimates that vary by accident period without any corresponding true risk change.

```python
policies = policies.with_columns(
    (
        (
            pl.min_horizontal(pl.col("expiry_date"), pl.lit(pl.date(2025, 12, 31)))
            - pl.max_horizontal(pl.col("inception_date"), pl.lit(pl.date(2025, 1, 1)))
        ).dt.total_days()
        / 365.25
    ).clip(lower_bound=0.0).alias("earned_exposure_2025")
)
```

Do this for each accident year in your experience data and stack the results into a single modelling dataset where each row is a policy-year combination.

---

## The frequency model

With prepared data, the frequency model is a Poisson GLM with a log link and log(exposure) as an offset. We use [glum](https://github.com/Quantco/glum), which is substantially faster than sklearn for large datasets and exposes the offset term cleanly.

```bash
uv pip install glum polars scikit-learn
```

```python
import numpy as np
from glum import GeneralizedLinearRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Rating factors for UK motor (adjust to your book)
CAT_FEATURES = ["vehicle_group", "ncd_band", "postcode_district", "cover_type"]
NUM_FEATURES = ["driver_age", "vehicle_age"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
    ("num", "passthrough", NUM_FEATURES),
])

freq_model = GeneralizedLinearRegressor(
    family="poisson",
    link="log",
    alpha=0.01,          # L2 regularisation - tune by cross-validation
    fit_intercept=True,
)

X = preprocessor.fit_transform(train_df.to_pandas())
y = train_df["claim_count_ult"].to_numpy()
offset = np.log(train_df["earned_exposure"].to_numpy())

freq_model.fit(X, y, offset=offset)
```

The `offset` parameter in glum's `fit()` is the log-exposure term. It enters the linear predictor directly: the model fits `log(E[y]) = offset + X @ beta`, which is exactly the Poisson GLM you would fit in Emblem.

A few practical decisions to document:

- **Which accident years to include?** We typically use five years of data for frequency, weighted by development factors. Older years get included if there is no structural break in the portfolio mix, excluded if there was a significant book composition change.
- **Postcode granularity.** Fitting at postcode district (the first part of the postcode, e.g. "SW1") is usually fine for frequency. For severity, you may want to aggregate further to postcode area ("SW") if district-level data is thin.
- **NCD band.** NCD is both a rating factor and a proxy for driver experience. It will be correlated with age. Check the correlation matrix before fitting and consider whether to include both or use NCD as your primary experience measure.

---

## The severity model

The severity model is fitted on claim records - one row per claim, not per policy-year. The dependent variable is the capped incurred cost. The natural distribution is Gamma with a log link.

```python
# Filter to claims only (rows where a claim occurred)
claims_df = train_df.filter(pl.col("claim_count_ult") > 0)

sev_model = GeneralizedLinearRegressor(
    family="gamma",
    link="log",
    alpha=0.05,    # Severity needs more regularisation than frequency
    fit_intercept=True,
)

X_sev = preprocessor.transform(claims_df.to_pandas())
y_sev = claims_df["incurred_capped"].to_numpy()

# No offset in severity model - it's already per-claim
sev_model.fit(X_sev, y_sev)
```

Note: no offset in the severity model. Exposure does not belong here - you are modelling the expected cost of a claim that has already occurred, conditional on the claim count.

The Gamma family assumes variance proportional to the square of the mean, which fits the right-skewed distribution of attritional claim costs reasonably well. If your claims data has a spike at the minimum claim threshold (common in motor where there is a claims management incentive structure), consider fitting a truncated model or excluding claims below the threshold.

---

## Combining into a burning cost estimate

The pure premium (burning cost) is:

```python
# Predict frequency and severity on a rating universe
freq_pred = freq_model.predict(X_universe, offset=np.zeros(len(X_universe)))  # per car-year
sev_pred  = sev_model.predict(X_universe)

# Burning cost = frequency * severity * large loss loading
burning_cost = freq_pred * sev_pred * (1 + large_loss_ratio)
```

The `offset=np.zeros(...)` in the frequency prediction gives you the predicted claims per car-year for each risk profile. Multiplying by severity gives you the expected annual claim cost per car-year.

One subtlety: the large loss loading we calculated earlier is a portfolio-level ratio. For a more sophisticated model, you would want a per-risk large loss loading based on the exposure size of each risk cell. For most personal lines work, the portfolio-level loading is sufficient.

---

## What this does not cover

A production burning cost model needs several things we have not covered here:

**Trend.** The frequency and severity models above use historical data. Fitting a separate trend component - using a gas model or a temporal spline - is necessary before the burning cost can be used for rating. We cover this in [our post on between-update trend](/2026/03/08/gas-models-for-between-update-trend/).

**Rating factor extraction.** If you need to deploy into Radar or Emblem as a factor table, you cannot use the Python GLM predictions directly. You need to extract multiplicative relativities from the model. We cover this in [extracting rating relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) and with our [`shap-relativities`](https://github.com/burning-cost/shap-relativities) library.

**Model validation.** Before you present burning cost estimates to pricing leadership or use them to set rates, you need a formal validation against holdout data: Gini coefficient, double lift chart, and calibration by decile at minimum. The PRA's SS1/23 sets out expectations for model validation; we cover what that looks like in practice in [model validation under PRA SS1/23](/2026/03/11/model-validation-pra-ss123/).

**Credibility weighting.** Where a rating cell has thin data, the pure GLM estimates are unreliable. Blending with a market prior or a higher-level estimate using credibility weights is standard practice. We cover the Bühlmann-Straub approach in [our post on credibility models in Python](/2026/02/19/buhlmann-straub-credibility-in-python/).

---

## The complete workflow

The minimal burning cost pipeline looks like this:

1. Cap claims at the large loss threshold, store the excess as a loading
2. Develop claim counts to ultimate using chain-ladder factors by accident year
3. Calculate earned car-year exposure for each policy-period combination
4. Fit a Poisson GLM (frequency) with log(exposure) as offset
5. Fit a Gamma GLM (severity) on claims-level data
6. Multiply predictions and apply the large loss loading
7. Apply trend factors before using for rating

What is not on this list: any step labelled "open the claims spreadsheet and argue about whether to include the 2019 flood year." The decisions above - capping threshold, development factors, which accident years to include - should be documented and version-controlled alongside the model code. If your burning cost review process involves these decisions being relitigated at each cycle, that is a governance problem, not a data problem.

Our [`insurance-governance`](https://github.com/burning-cost/insurance-governance) library includes templates for logging these decisions as part of a reproducible pricing cycle.
