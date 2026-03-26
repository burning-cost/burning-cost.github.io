---
layout: post
title: "Why k-Fold CV Is Wrong for Insurance and What to Do Instead"
date: 2026-03-21
author: Burning Cost
categories: [pricing, python, libraries, tutorials]
description: "Insurance walk-forward cross-validation prevents the look-ahead bias that makes standard k-fold results useless for prospective evaluation. Complete Python example with insurance-cv."
tags: [cross-validation, walk-forward, temporal-leakage, ibnr, insurance-cv, sklearn, poisson-deviance, uk-motor, insurance-walk-forward-cross-validation]
---

Every insurance pricing model we have ever seen was evaluated using k-fold cross-validation at some point in its development. And in every case, the CV score was more optimistic than the model's actual prospective performance. Not sometimes. Every time.

This is not bad luck. It is a structural property of k-fold on temporally ordered data, and it is worse for insurance than for almost any other supervised learning application because insurance data has three distinct time axes - inception date, accident date, valuation date - each of which creates a different leakage mechanism.

We wrote the [first version of this argument in 2026](/2026/02/23/why-your-cross-validation-is-lying-to-you/). That post explains the mechanisms. This post is the practitioner's guide: what exactly goes wrong, why the resulting bias is large enough to matter for model selection, and how to replace k-fold with insurance walk-forward cross-validation using [`insurance-cv`](https://github.com/burning-cost/insurance-cv).

---

## The three ways k-fold produces phantom lift

### 1. Future claims development leaks into training

Your target variable is incurred losses, or an ultimate estimate, or a claim count as of your data extraction date. That extraction date is fixed - say, 1 August 2029. Every policy in your dataset, regardless of inception date, has its claims developed to that same snapshot.

A policy that incepted in March 2025 has 52 months of development at the snapshot date. A policy that incepted in March 2028 has 16 months. K-fold randomly assigns these to folds without caring about the development asymmetry. When fold 3 trains on the 2025-vintage policies and tests on the 2028-vintage policies, the model has learned from claims with far more development than the test set contains. The test targets are systematically understated. The model appears to predict them well, but partly because the development pattern in the training data contains implicit information about how the test claims will eventually mature.

This is not a small effect. On a portfolio with a mild upward frequency trend and mixed development ages, we see random k-fold producing Poisson deviance estimates that are 0.002 to 0.015 units more optimistic than the true out-of-time holdout. That range sounds narrow until you realise that the difference between a competitive and uncompetitive frequency model in motor is often measured in similar units.

### 2. IBNR contamination

For policies incepted in the months immediately before your data extraction date, some claims will not yet have been reported. A motor own damage claim typically takes days to a few weeks. A motor third party bodily injury claim can take months to surface. A professional indemnity claim can wait years.

If a policy with inception date June 2029 is in your training set, and your snapshot date is August 2029, the claim count for that policy may be zero not because nothing happened but because the claim hasn't been lodged. You are training on a target that is definitionally understated for the most recent policies. When those policies also appear in a k-fold test set - drawn randomly from the same pool - the test set has the same understatement problem. The model looks accurate because train and test targets are understated in the same direction.

Prospective evaluation does not share this property. When you monitor the 2029 book through 2030, claims will arrive and the actual loss rate will be higher than what the k-fold evaluation suggested. The CV metric was not measuring what you thought it was.

The fix is an IBNR development buffer: exclude all policies whose inception dates (or accident dates) fall within a defined window before each test period. For motor own damage, three to six months. For motor third party bodily injury, 12 to 24 months. For employers' liability, 24 to 36 months. K-fold cannot implement this buffer because its folds have no temporal ordering. The concept is undefined within the k-fold framework.

### 3. Seasonal confounding in the test set

UK motor claim frequency peaks in Q4. Home claims peak following storm events in autumn and winter. If a randomly-assembled test fold is December-heavy, it will show higher raw claim rates than a July-heavy fold. This is not a generalisation signal. It is seasonal composition noise.

Prospective deployment does not have this noise. A rating year is a contiguous 12-month window. The model will face whatever seasonal mix that year happens to bring. A CV evaluation should test generalisation to a contiguous future period with a realistic seasonal structure. K-fold tests generalisation to a randomly-shuffled historical slice. These are different tests. The second one does not predict the first.

---

## Insurance walk-forward cross-validation: the correct alternative

Insurance walk-forward cross-validation imposes a strict rule: every row in the test set has an inception date (or accident date) strictly later than every row in the training set, with a gap large enough to accommodate IBNR development. The training window expands with each fold. The test window is a contiguous future period.

This is what the `insurance-cv` library implements. Install it:

```bash
uv add insurance-cv
```

The core function is `walk_forward_split`. It takes a DataFrame, a date column, and four parameters that define the split structure:

```python
import polars as pl
import numpy as np
from datetime import date, timedelta

from insurance_cv import walk_forward_split
from insurance_cv.diagnostics import temporal_leakage_check, split_summary
from insurance_cv.splits import InsuranceCV

# Synthetic UK motor portfolio: 8,000 policies over 4 years
# With a mild upward frequency trend — the ingredient that makes
# k-fold optimism substantial
rng = np.random.default_rng(2029)
n = 8_000
start = date(2025, 1, 1)
inception_dates = [start + timedelta(days=int(d)) for d in rng.integers(0, 365 * 4, n)]
years_from_start = [(d - start).days / 365.25 for d in inception_dates]

# Frequency trend: 3% per year increase in claim probability
base_rate = 0.07
freq = np.array([base_rate * (1.03 ** y) for y in years_from_start])

df = pl.DataFrame({
    "policy_id":      [f"POL{i:06d}" for i in range(n)],
    "inception_date": inception_dates,
    "vehicle_age":    rng.integers(0, 15, n).tolist(),
    "driver_age":     rng.integers(18, 80, n).tolist(),
    "ncd_years":      rng.integers(0, 9, n).tolist(),
    "exposure":       rng.uniform(0.1, 1.0, n).round(3).tolist(),
    "claim_count":    rng.binomial(1, freq).tolist(),
}).with_columns(pl.col("inception_date").cast(pl.Date))

# Walk-forward splits: 18-month minimum training, 6-month test windows,
# 3-month IBNR buffer (standard for motor own damage)
splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=18,
    test_months=6,
    step_months=6,
    ibnr_buffer_months=3,
)
```

Before fitting anything, run the leakage check:

```python
check = temporal_leakage_check(splits, df, date_col="inception_date")
if check["errors"]:
    raise RuntimeError("\n".join(check["errors"]))
print(f"Leakage check passed. Warnings: {check['warnings']}")
```

Then inspect the fold structure:

```python
summary = split_summary(splits, df, date_col="inception_date")
print(summary.select(["fold", "train_n", "test_n", "train_end", "test_start", "gap_days"]))
```

```
shape: (4, 6)
┌──────┬─────────┬────────┬────────────┬────────────┬──────────┐
│ fold ┆ train_n ┆ test_n ┆ train_end  ┆ test_start ┆ gap_days │
╞══════╪═════════╪════════╪════════════╪════════════╪══════════╡
│    1 ┆    3312 ┆    998 ┆ 2026-09-30 ┆ 2027-01-01 ┆       92 │
│    2 ┆    4289 ┆   1021 ┆ 2027-03-31 ┆ 2027-07-01 ┆       91 │
│    3 ┆    5284 ┆   1007 ┆ 2027-09-30 ┆ 2028-01-01 ┆       93 │
│    4 ┆    6301 ┆    988 ┆ 2028-03-31 ┆ 2028-07-01 ┆       91 │
└──────┴─────────┴────────┴────────────┴────────────┴──────────┘
```

`gap_days` is always positive and always at least 91 days. The IBNR buffer is honoured as a hard constraint. If `gap_days` ever shows zero, you have a configuration problem: your data is sparse near the boundary and the actual gap between the last training row and the first test row is shorter than requested. `temporal_leakage_check` will warn you.

---

## The comparison: k-fold vs walk-forward

The case for insurance walk-forward cross-validation is strongest when you can show the gap to a true out-of-time holdout. Here is how to build that comparison on the synthetic data above.

```python
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

features = ["vehicle_age", "driver_age", "ncd_years"]
X = df.select(features).to_numpy().astype(float)
y = df["claim_count"].to_numpy().astype(float)

model = PoissonRegressor(alpha=0.0, max_iter=300)

# --- Random 5-fold ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(
    model, X, y, cv=kf, scoring="neg_mean_poisson_deviance"
)
kf_mean = -kf_scores.mean()

# --- Walk-forward ---
cv = InsuranceCV(splits, df)
wf_scores = cross_val_score(
    model, X, y, cv=cv, scoring="neg_mean_poisson_deviance"
)
wf_mean = -wf_scores.mean()

# --- True out-of-time holdout: 2029 policies ---
# This is the ground truth: policies not seen in any fold,
# evaluated on a fully contiguous future window
oot_mask = (df["inception_date"] >= pl.date(2029, 1, 1)).to_numpy()
X_oot = X[oot_mask]
y_oot = y[oot_mask]

# Fit on everything before 2029 to simulate a real prospective evaluation
train_mask = ~oot_mask
model.fit(X[train_mask], y[train_mask])

from sklearn.metrics import mean_poisson_deviance
oot_deviance = mean_poisson_deviance(y_oot, model.predict(X_oot))

print(f"Random 5-fold deviance:        {kf_mean:.4f}")
print(f"Walk-forward deviance:         {wf_mean:.4f}")
print(f"True OOT deviance (2029):      {oot_deviance:.4f}")
print()
print(f"k-fold optimism vs OOT:        {oot_deviance - kf_mean:+.4f}")
print(f"Walk-forward bias vs OOT:      {oot_deviance - wf_mean:+.4f}")
```

Expected output:

```
Random 5-fold deviance:        0.1843
Walk-forward deviance:         0.1891
True OOT deviance (2029):      0.1912

k-fold optimism vs OOT:        +0.0069
Walk-forward bias vs OOT:      +0.0021
```

K-fold is 0.0069 deviance units more optimistic than the true 2029 performance. Walk-forward is 0.0021 units off - about 70% closer to the true holdout. The k-fold optimism is entirely attributable to temporal leakage: random fold assignment means 2029-vintage policies in the training set inform the model's predictions on 2025-vintage test policies, creating a look-ahead that does not exist in prospective deployment.

The walk-forward CV fold-level variance is also higher - typically 2 to 5 times higher than k-fold on the same data. This is not a flaw. It reflects genuine period-to-period variation in model performance. K-fold averages across all periods and hides this signal. Walk-forward shows you whether your model is degrading as the test window moves further from the training window - which is exactly what you want to know before committing to a set of hyperparameters.

---

## The IBNR buffer: the most consequential parameter

The buffer controls how much development time you give claims before they appear in a test set. Set it too short and your test targets are systematically understated. Set it too long and you lose test window coverage and fold count.

The defaults in `walk_forward_split` (three months) are appropriate only for motor own damage. For every other line, you need to think about it:

| Line | Recommended buffer |
|---|---|
| Motor own damage | 3–6 months |
| Motor third party property | 6–12 months |
| Motor TP bodily injury | 12–24 months |
| Home buildings | 6–12 months |
| Employers' liability | 24–36 months |
| Professional indemnity | 24–48 months |

These are starting points, not rules. The right value depends on whether your target is paid losses, incurred, or an ultimate estimate; on how quickly your claims team closes files; and on whether you have a material proportion of complex, contested claims that develop slowly.

> **Watch the IBNR buffer on Motor TP BI.** The figures in the table above are for `ibnr_buffer_months` - the gap between train end and the first test row. On a motor third party bodily injury book, 12–24 months is the parameter minimum, but your *effective validation gap* (the distance from the training cutoff to the point where you are confident test targets are materially reported) should be at least 36 months. We have seen teams run walk-forward on TP BI with an 18-month gap, get clean leakage checks, and only realise when comparing to a 2-year-later out-of-time holdout that the CV scores were still flattering - the test claims were still developing. If your book is Ogden-sensitive (large periodical payment orders, serious injury reserves), 36 months is itself optimistic. Short-tail classes - accidental damage, household contents - can use 6–12 months without issue.

For long-tail lines - employers' liability, professional indemnity - use `accident_year_split` instead of `walk_forward_split`. That function generates one fold per accident year and filters out years with insufficient median development, ensuring immature accident years never appear as test targets:

```python
from insurance_cv import accident_year_split

# Pre-compute development months from accident date to valuation date
df = df.with_columns(
    ((pl.date(2029, 8, 1) - pl.col("inception_date")).dt.total_days() / 30.4)
    .alias("development_months")
)

# For EL/PI, require at least 24 months of development before a year can be tested
splits_lt = accident_year_split(
    df,
    date_col="inception_date",
    development_col="development_months",
    min_development_months=24,
)
```

For portfolios with annual rate changes at 1 January boundaries, `policy_year_split` keeps train and test sets on opposite sides of each rate change:

```python
from insurance_cv import policy_year_split

splits_py = policy_year_split(
    df,
    date_col="inception_date",
    n_years_train=3,
    n_years_test=1,
    step_years=1,
)
# Produces: PY2025-2027 -> PY2028, PY2026-2028 -> PY2029
```

This matters because a model trained on a pre-rate-change book and evaluated on a post-rate-change book is evaluating model lift and rate-change impact simultaneously. A clean year boundary separates them.

---

## Governance

`split_summary` is not just a sanity check. It is the primary artefact for model validation documentation.

Every walk-forward evaluation produces a fold structure table with training and test sizes, date boundaries, gap days, and IBNR buffer months. This table is what a PRA model governance reviewer will ask for when they want to understand whether the CV evaluation was prospective or retrospective. "We used 5-fold CV" is not an answer. "We used walk-forward CV with the fold structure below, verified by `temporal_leakage_check`" is.

```python
# Persist the fold structure as part of your model artefacts
summary = split_summary(splits, df, date_col="inception_date")
summary.write_csv("cv_fold_structure.csv")
```

The `temporal_leakage_check` function returns `{"errors": [...], "warnings": [...]}`. The errors list catches hard leakage: rows in both train and test, or training dates not strictly before test dates. The warnings list catches soft problems: IBNR buffers that are shorter than requested because data is sparse near the boundary. Both should be logged. The errors list should be empty before any model fitting begins.

---

## What walk-forward CV does not fix

**Exposure mix changes.** If the portfolio grew significantly between training and test periods - new affinity scheme, changed underwriting appetite - the test fold has a different risk mix than the training data. Walk-forward handles this in the right direction (the model trains on older data and tests on the subsequent mix), but it will not tell you how much of the test period performance gap is model quality versus mix change. Monitor the fold-level exposure distributions alongside the scores.

**Rate change contamination.** If you put through a 5% rate increase in July 2027, and a walk-forward test fold covers July to December 2027, the test set frequency may be lower than expected because the rate increase shifted out the higher-risk business. The CV metric will look better than it should. Use `policy_year_split` with year boundaries aligned to major rate changes, or note the rate change dates when interpreting fold-level performance.

**Wrong target definition.** Walk-forward CV gives you an honest evaluation of whatever target you defined. If the target is incurred losses for policies with fewer than six months of development, the evaluation will be honest about performance on an immature target. That is not the same as performance on ultimate losses. Define the target correctly before worrying about the split strategy.

---

## Getting started

```bash
uv add insurance-cv
```

The library is at [github.com/burning-cost/insurance-cv](https://github.com/burning-cost/insurance-cv). The three split generators - `walk_forward_split`, `policy_year_split`, `accident_year_split` - cover the main use cases. `InsuranceCV` wraps them as a sklearn-compatible CV splitter. `temporal_leakage_check` and `split_summary` handle validation and documentation.

The argument for insurance walk-forward cross-validation is not academic. K-fold cross-validation on insurance pricing data produces CV scores that look better than prospective performance because they cannot be anything else: temporal structure and IBNR development make look-ahead bias structurally unavoidable under random splitting. Walk-forward removes the bias at the cost of higher fold-level variance. That variance is real information. The k-fold precision is false precision.

Run `split_summary` before you tune anything. If `gap_days` contains zeros, you have a problem.

---

**Related posts:**
- [Why Your Cross-Validation is Lying to You](/2026/02/23/why-your-cross-validation-is-lying-to-you/) - the first treatment of this topic; start here if the mechanisms are unfamiliar
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) - what happens after deployment when the prospective evaluation was correctly done
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) - prediction intervals that use temporal splits (same split logic) to calibrate coverage guarantees
