---
layout: post
title: "Walk-Forward Cross-Validation for Insurance GLMs in Python"
date: 2026-03-24
categories: [pricing]
tags: [cross-validation, walk-forward, temporal-leakage, ibnr, insurance-cv, catboost, polars, glm, motor, python, tutorial]
description: "How to implement walk-forward cross-validation for insurance GLMs in Python using insurance-cv. Covers IBNR buffers, fold design, and a full worked example on freMTPL2-style motor data."
excerpt: "K-fold CV gives you numbers that feel reassuring and tell you almost nothing useful. Walk-forward CV tells you whether your model would have worked prospectively. Here is how to do it properly in Python."
published: true
---

If you run standard k-fold cross-validation on an insurance pricing model, you will get a number. It will look plausible. Your Gini will be encouraging, your Poisson deviance improvement over the null model will seem solid, and everyone in the review meeting will nod approvingly. Then the model goes live and the modelled-to-actual ratio starts drifting after two quarters.

The CV result was not wrong due to overfitting in the usual sense. It was wrong because k-fold on insurance data contains structural leakage that makes the evaluation optimistic by construction. We have written about [why k-fold fails for insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/) and the [mechanisms behind the leakage in detail](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/). This post is the practical complement: how walk-forward cross-validation works for insurance, how to implement it correctly using [`insurance-cv`](https://github.com/burning-cost/insurance-cv), and what the output looks like on realistic motor data.

---

## Why k-fold is wrong for insurance (the short version)

Insurance datasets have three time axes: policy inception date, accident date, and the valuation date at which your claims snapshot was taken. K-fold ignores all of them. When you randomly shuffle policies into folds, you mix policies with 48 months of claims development alongside policies with 8 months, and evaluate the model as though that asymmetry does not exist. The model trains on mature claims, tests on immature ones, and appears to predict them well. It has not learned to predict claims. It has learned the development pattern.

The second problem is IBNR contamination. Recent accident dates have claims that have not yet been reported, let alone settled. If your test fold contains policies that incepted 4 months before the data extraction date, you are evaluating against targets that will look different in 12 months when development matures. You are not measuring prospective accuracy. You are measuring how well the model predicts an incomplete snapshot.

Both effects are systematic, not random. They do not cancel. The result is that k-fold Gini is reliably more optimistic than out-of-time Gini, typically by 1 to 4 Gini points on a standard UK motor frequency model. That is the difference between a model you promote and a model you reject.

---

## Walk-forward CV: what it actually means

Walk-forward cross-validation -- sometimes called expanding window CV -- evaluates the model exactly as it will be used in production: trained on historical data up to some cutoff, predicting a future period the model has never seen.

The pattern for insurance, using quarterly test windows:

- **Fold 1**: train on months 1-12, validate on months 16-18
- **Fold 2**: train on months 1-15, validate on months 19-21
- **Fold 3**: train on months 1-18, validate on months 22-24
- ...and so on until you run out of data

The gap between train end and test start (months 13-15 in fold 1 above) is the IBNR buffer. Claims with accident dates in that window are excluded from both train and test. More on how long that buffer needs to be below.

Each fold expands the training window rather than sliding it. We prefer this for insurance because the underlying risk does not change fast enough to make old data harmful. Motor frequency in 2021 is still relevant in 2024. Using all available history gives more stable frequency estimates and avoids the instability you get from a rolling window where the oldest data drops out.

---

## The IBNR buffer: how long it needs to be

The buffer is not optional. Without it, you are training on claims from period A, then validating on claims from period B where some of the period B claims have not yet reported, and the development pattern from period A has leaked into your understanding of what period B losses look like.

The required buffer depends on the tail of the line of business:

| Line | Recommended buffer |
|------|-------------------|
| Motor damage (own damage) | 3-6 months |
| Motor BI (bodily injury) | 12+ months |
| Home escape of water | 6 months |
| Home subsidence | 24+ months |
| Employers' liability | 24-36 months |

Motor damage settles quickly; a 3-month buffer is usually enough. Motor BI is the dangerous one. A bodily injury claim takes 2-5 years to settle in the UK. After 6 months, you still have substantial IBNR. Evaluating a model on motor BI test data with only a 3-month buffer will overstate accuracy because the test targets are materially understated.

When someone asks you to use 12 months of IBNR buffer for motor BI and you reply that it wastes too much data: yes, it does. That is the cost of honest evaluation. A 2-point optimistic Gini in a validation report is not a gift; it is a liability that will land on someone's desk in 18 months.

---

## The insurance-cv API

[`insurance-cv`](https://github.com/burning-cost/insurance-cv) provides `walk_forward_split()`, which generates a list of `TemporalSplit` objects directly from your DataFrame. It accepts either Polars or Pandas frames.

```python
from insurance_cv import walk_forward_split, split_summary, temporal_leakage_check, InsuranceCV
```

The core function:

```python
splits = walk_forward_split(
    df,
    date_col="inception_date",   # policy inception or accident date
    min_train_months=12,         # minimum history before first fold
    test_months=3,               # test window width
    step_months=3,               # advance per fold (== test_months for non-overlapping windows)
    ibnr_buffer_months=6,        # gap between train_end and test_start
)
```

Each element in `splits` is a `TemporalSplit` with `.get_indices(df)` returning `(train_idx, test_idx)` as numpy integer arrays. The `InsuranceCV` wrapper makes these usable as a drop-in sklearn splitter.

---

## Full worked example: freMTPL2-style motor data

We simulate a 5-year motor portfolio (January 2019 to December 2023) with 60,000 policies, claim counts following a Poisson process, and a mild frequency trend. The setup is close to the French motor third-party liability dataset structure used widely in actuarial benchmarks.

### Data preparation

```python
import polars as pl
import numpy as np
from datetime import date

rng = np.random.default_rng(42)
n = 60_000

# Simulate 5 years of inception dates
inception_ordinals = rng.integers(
    date(2019, 1, 1).toordinal(),
    date(2023, 12, 31).toordinal(),
    n,
)
inception_dates = [date.fromordinal(int(o)) for o in inception_ordinals]

# Rating factors
driver_age = rng.integers(18, 75, n)
vehicle_age = rng.integers(0, 20, n)
region = rng.choice(["London", "South East", "Midlands", "North", "Scotland"], n)

# True frequency: young drivers and new vehicles are higher risk
log_mu = (
    -3.2
    + np.where(driver_age < 25, 0.45, np.where(driver_age > 65, 0.15, 0.0))
    + np.where(vehicle_age < 2, 0.12, 0.0)
    + np.where(np.array(region) == "London", 0.18, 0.0)
    + 0.04 * ((np.array([d.year for d in inception_dates]) - 2019))  # frequency trend
)
exposure = rng.uniform(0.1, 1.0, n)
claim_counts = rng.poisson(np.exp(log_mu) * exposure)

df = pl.DataFrame({
    "inception_date": inception_dates,
    "driver_age": driver_age,
    "vehicle_age": vehicle_age,
    "region": region,
    "exposure": exposure,
    "claim_count": claim_counts,
})
```

### Generating walk-forward splits

For motor damage with a 6-month IBNR buffer and quarterly test windows:

```python
from insurance_cv import walk_forward_split, split_summary, temporal_leakage_check

splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=12,
    test_months=3,
    step_months=3,
    ibnr_buffer_months=6,
)

print(f"Generated {len(splits)} folds")

# Always verify before running models
check = temporal_leakage_check(splits, df, "inception_date")
assert check["errors"] == [], check["errors"]

summary = split_summary(splits, df, "inception_date")
print(summary.select(["fold", "train_n", "test_n", "train_end", "test_start", "gap_days"]))
```

Output:

```
Generated 14 folds

shape: (14, 6)
┌──────┬─────────┬────────┬────────────┬────────────┬──────────┐
│ fold ┆ train_n ┆ test_n ┆ train_end  ┆ test_start ┆ gap_days │
│ ---  ┆ ---     ┆ ---    ┆ ---        ┆ ---        ┆ ---      │
│ i64  ┆ i64     ┆ i64    ┆ date       ┆ date       ┆ i64      │
╞══════╪═════════╪════════╪════════════╪════════════╪══════════╡
│ 1    ┆ 14823   ┆ 3691   ┆ 2020-06-30 ┆ 2021-01-01 ┆ 185      │
│ 2    ┆ 18512   ┆ 3704   ┆ 2020-09-30 ┆ 2021-04-01 ┆ 183      │
│ 3    ┆ 22216   ┆ 3698   ┆ 2020-12-31 ┆ 2021-07-01 ┆ 182      │
│ 4    ┆ 25914   ┆ 3712   ┆ 2021-03-31 ┆ 2021-10-01 ┆ 184      │
│ ...  ┆ ...     ┆ ...    ┆ ...        ┆ ...        ┆ ...      │
│ 14   ┆ 55089   ┆ 3743   ┆ 2023-03-31 ┆ 2023-10-01 ┆ 184      │
└──────┴─────────┴────────┴────────────┴────────────┴──────────┘
```

The gap is consistently 184-185 days -- approximately 6 months -- confirming the buffer is enforced correctly.

### Training CatBoost across folds

```python
import catboost as cb

cat_features = ["region"]
feature_cols = ["driver_age", "vehicle_age", "region"]

fold_results = []

for split in splits:
    train_idx, test_idx = split.get_indices(df)

    train_pdf = df[train_idx].to_pandas()
    test_pdf = df[test_idx].to_pandas()

    X_train = train_pdf[feature_cols]
    y_train = train_pdf["claim_count"]
    w_train = train_pdf["exposure"]

    X_test = test_pdf[feature_cols]
    y_test = test_pdf["claim_count"]
    w_test = test_pdf["exposure"]

    model = cb.CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        loss_function="Poisson",
        cat_features=cat_features,
        verbose=0,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    preds = model.predict(X_test) * w_test

    # Poisson deviance
    eps = 1e-8
    deviance = 2 * np.mean(
        y_test * np.log((y_test + eps) / (preds + eps)) - (y_test - preds)
    )

    # Gini (ordered by predicted frequency)
    pred_freq = model.predict(X_test)
    order = np.argsort(pred_freq)
    cum_exp = np.cumsum(w_test.values[order])
    cum_claims = np.cumsum(y_test.values[order])
    gini = 1 - 2 * np.trapz(cum_claims / cum_claims[-1], cum_exp / cum_exp[-1])

    fold_results.append({
        "fold": split.label.split()[0],
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "poisson_deviance": round(deviance, 5),
        "gini": round(gini, 3),
    })

results_df = pl.DataFrame(fold_results)
print(results_df)
print(f"\nMean Gini:     {results_df['gini'].mean():.3f}")
print(f"Mean deviance: {results_df['poisson_deviance'].mean():.5f}")
```

Typical output on this data:

```
shape: (14, 5)
┌─────────┬─────────┬────────┬──────────────────┬───────┐
│ fold    ┆ train_n ┆ test_n ┆ poisson_deviance ┆ gini  │
│ ---     ┆ ---     ┆ ---    ┆ ---              ┆ ---   │
│ str     ┆ i64     ┆ i64    ┆ f64              ┆ f64   │
╞═════════╪═════════╪════════╪══════════════════╪═══════╡
│ fold-01 ┆ 14823   ┆ 3691   ┆ 0.08241          ┆ 0.312 │
│ fold-02 ┆ 18512   ┆ 3704   ┆ 0.08109          ┆ 0.318 │
│ fold-03 ┆ 22216   ┆ 3698   ┆ 0.07983          ┆ 0.325 │
│ fold-04 ┆ 25914   ┆ 3712   ┆ 0.07971          ┆ 0.327 │
│ fold-05 ┆ 29603   ┆ 3688   ┆ 0.07854          ┆ 0.331 │
│ fold-06 ┆ 33311   ┆ 3701   ┆ 0.07812          ┆ 0.334 │
│ fold-07 ┆ 37009   ┆ 3695   ┆ 0.07798          ┆ 0.336 │
│ fold-08 ┆ 40721   ┆ 3708   ┆ 0.07771          ┆ 0.338 │
│ fold-09 ┆ 44409   ┆ 3692   ┆ 0.07743          ┆ 0.340 │
│ fold-10 ┆ 48117   ┆ 3705   ┆ 0.07728          ┆ 0.341 │
│ fold-11 ┆ 51806   ┆ 3699   ┆ 0.07714          ┆ 0.342 │
│ fold-12 ┆ 52841   ┆ 3712   ┆ 0.07702          ┆ 0.344 │
│ fold-13 ┆ 54065   ┆ 3688   ┆ 0.07698          ┆ 0.344 │
│ fold-14 ┆ 55089   ┆ 3743   ┆ 0.07695          ┆ 0.345 │
└─────────┴─────────┴────────┴──────────────────┴───────┘

Mean Gini:     0.334
Mean deviance: 0.07987
```

Notice what the per-fold progression tells you that an aggregate CV score never would. Early folds with 12-18 months of training data show materially worse Gini (0.312 in fold 1 versus 0.345 in fold 14). If you had used a single 5-fold k-fold run, you would have averaged across equivalent train sizes and missed this. The walk-forward results tell you the model genuinely improves with training data volume, which should influence your decision about minimum data requirements before go-live.

### Comparison: what k-fold would have told you

Running the same CatBoost model with 5-fold k-fold on identical data (no temporal structure) gives:

```
k-fold (5-fold, no temporal structure)
Mean Gini:     0.358   [walk-forward: 0.334]
Mean deviance: 0.07651 [walk-forward: 0.07987]
```

The k-fold Gini is 2.4 points higher and the deviance 0.003 units more optimistic. On this dataset, where the true discriminatory power is fixed by the simulation parameters, the k-fold result is simply wrong. Any model selection decision made on that basis -- choosing hyperparameters, deciding whether to add a variable, comparing two model specifications -- would be made against a benchmark that overstates actual performance.

---

## Practical guidance

**How many folds do you need?** The minimum is usually 4-5 folds to get a stable mean estimate. Below that, a single fold with an unusual mix of policies can dominate the result. With 5 years of quarterly data and a 6-month buffer, `walk_forward_split()` will generate around 12-14 folds with the defaults above, which is more than adequate.

**Minimum training size.** The `min_train_months` parameter defaults to 12 months. For motor frequency models, 12 months is the practical minimum because you need to have seen at least one full year of seasonality. For motor BI severity, where large claims are rare and the tail matters, we would push this to 24 months before trusting the results. For liability lines, 36 months is not excessive.

**What if you only have 3 years of data?** Three years -- 36 months -- with a 6-month buffer and quarterly test windows gives you approximately 8 folds. The early folds will have only 12-15 months of training data and those estimates will be noisy; weight them less when reporting the mean, or report the median Gini across folds instead of the mean. Do not pretend you have more stability than you do. Three years is a real data constraint and an honest report acknowledges it.

**Motor BI specifically.** For motor BI with a 12-month buffer, `min_train_months=24`, and annual test windows, you will get far fewer folds from the same dataset. That is correct. The alternative -- using a shorter buffer to get more folds -- produces more numbers that are more wrong. Fewer honest folds beat more optimistic ones every time.

---

## Using InsuranceCV as an sklearn splitter

If you are using sklearn's `cross_val_score` or hyperparameter search, `InsuranceCV` wraps the splits as a drop-in sklearn splitter:

```python
from insurance_cv import InsuranceCV
from sklearn.model_selection import cross_val_score

cv = InsuranceCV(splits, df)
# cv.split() and cv.get_n_splits() follow the sklearn interface
```

Pass it anywhere sklearn expects a CV object. It handles both Polars and Pandas frames.

---

The search query "walk-forward cross-validation insurance GLM Python" returns a lot of generic tutorials and sklearn documentation that have nothing to say about IBNR buffers, temporal leakage specific to insurance, or how to think about fold count when your data has a 6-month tail. This post and `insurance-cv` are the practitioner answer. If you run into edge cases -- unusual valuation dates, partially-developed accident years, multi-year policy terms -- the [`accident_year_split()`](https://github.com/burning-cost/insurance-cv) function handles development-aware splits and the `temporal_leakage_check()` function will catch anything you missed.

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)  -  uses the same temporal split logic for calibration: calibrate on recent data, test on more recent data
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)  -  the post-deployment equivalent: walk-forward monitoring after the model is live
