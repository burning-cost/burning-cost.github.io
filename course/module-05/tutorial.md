# Module 5: Conformal Prediction Intervals for Insurance Pricing

In Module 4 you extracted SHAP relativities from a CatBoost model and produced rating factor tables comparable to a GLM output. The model tells you what to charge. This module addresses a question the model does not answer on its own: how confident should you be in that number?

A CatBoost Tweedie model predicts a pure premium of £342 for a specific risk. That is the expected loss given the features. But how uncertain is that estimate? For a typical NCD-5 driver in area C with a mid-range vehicle group, the model has seen many similar risks in training and the prediction is reliable. For a 19-year-old with 9 conviction points in vehicle group 47 - a combination that appears a handful of times in the training data - the £342 point estimate is real, but the actual outcome could be anywhere from £0 to £15,000.

Most pricing workflows use the point estimate alone and ignore the uncertainty. This module shows why that is a problem for reserving, minimum premiums, and FCA Consumer Duty, and how conformal prediction provides uncertainty estimates with a mathematical coverage guarantee that you can audit and defend.

By the end of this module you will have:

- Calibrated a conformal predictor on held-out insurance data and generated 90% prediction intervals
- Validated that coverage is consistent across risk deciles, not just overall
- Flagged uncertain risks for underwriting referral using relative interval width
- Built risk-specific minimum premium floors from the conformal upper bound
- Produced a portfolio reserve range estimate with explicit assumptions
- Logged everything to MLflow and written intervals to Unity Catalog for audit

---

## Part 1: Why uncertainty matters - the actuarial argument

Before writing any code, you need to understand why point estimates are insufficient for the three tasks where pricing teams most need uncertainty quantification.

### Minimum premiums

Every commercial minimum premium policy applies some flat uplift: "minimum premium = 1.3x technical premium, subject to a floor of £250." The uplift is chosen based on experience and judgment. It does not vary by risk. A policy with a narrow loss distribution (stable, well-understood risk profile, many similar training observations) gets the same 30% uplift as a policy with a wide loss distribution (unusual risk profile, sparse training data, high inherent volatility).

For the narrow-distribution risk, the 1.3x uplift may be excessive - you are charging a floor well above what the actual upper quantile of losses suggests. For the wide-distribution risk, 1.3x may be dangerously low - the genuine 90th percentile of losses for that risk is far above the technical premium times 1.3.

Both directions create Consumer Duty problems. Overcharging stable risks is a fair value issue. Undercharging volatile risks is a solvency adequacy issue. A risk-specific minimum premium based on the actual calibrated upper bound of losses addresses both simultaneously.

### Reserving

Reserve teams need a range, not just a point estimate. The typical approach - "reserves are 115% of technical premium" - bakes in a fixed margin based on historical reserve adequacy. It does not vary by the composition of the current book. If the current book skews towards more volatile risks (younger drivers, higher vehicle groups, more thin-cell combinations), the fixed percentage understates reserve uncertainty. If it skews towards stable risks, the percentage overstates it.

Conformal prediction intervals aggregate to portfolio-level range estimates that reflect the actual composition of the book. The upper bound of the sum of individual intervals is a genuine 90th-percentile portfolio loss estimate, not a rule of thumb.

### Underwriting referral

The decision about which risks to refer to a human underwriter is often discretionary. A conformal approach makes it systematic: flag the risks where the model's prediction interval is wide relative to the point estimate. These are the risks where the model is genuinely uncertain because the training data is sparse in that region of the feature space. A systematic, data-driven referral process is more defensible to the FCA than an underwriter's discretionary judgment applied to the same risks.

The key conceptual distinction - one that regularly confuses the pricing committee - is between risk level and model uncertainty. A young driver with conviction points is high risk. But if we have 5,000 such drivers in training, the model is not uncertain about that combination. A 72-year-old driving a high-group vehicle with a specific configuration of features might be low-to-moderate risk but appear only a dozen times in training data - the model is very uncertain. Both dimensions exist independently.

---

## Part 2: What conformal prediction guarantees - the theory in plain English

You do not need to read a statistics paper to use conformal prediction correctly. But you do need to understand what the coverage guarantee says, because the FCA may ask about its mathematical basis.

### The guarantee

Given:
- A trained model
- A calibration dataset drawn from the same distribution as future test data
- A target coverage level `1 - alpha` (e.g. 90%)

The conformal predictor produces intervals such that, over repeated test observations:

**P(y_new is inside the interval) >= 1 - alpha**

This means: if you produce 90% prediction intervals using a correctly calibrated conformal predictor, at least 90% of future observations will fall inside their intervals. The "at least" is precise: the actual coverage is guaranteed to be at least the target, not exactly the target.

### What it does NOT guarantee

The guarantee is **marginal**: it applies across all test observations combined. It does not say that 90% of young drivers will be covered, or 90% of high-vehicle-group policies will be covered. Without additional care, the intervals could achieve 99% coverage in the bottom risk decile and only 72% in the top decile, and the marginal number is still 90%.

For insurance this is unacceptable. The top decile of risks - the ones contributing most to reserve uncertainty and the most likely to generate large adverse outcomes - is where we most need reliable coverage. A reserve range that achieves 90% coverage across the whole portfolio but only 72% coverage for the largest risks is not useful for the reserving team.

The solution is the **variance-weighted non-conformity score** (specifically the Pearson-weighted score explained in Part 5), which makes intervals scale with the predicted loss level. Combined with the coverage-by-decile diagnostic (Part 7), this ensures the intervals are valid where you need them to be valid.

### The exchangeability requirement

The conformal guarantee requires that the calibration data and test data are **exchangeable**: the joint distribution of any combined sample should be invariant to permutation. In practice this means calibration data and test data must come from the same underlying distribution.

For insurance, temporal trends break this. If claims have been inflating at 8% per year, a calibration observation from 2022 and a test observation from 2024 are not exchangeable - the 2024 observation has been through two additional years of inflation. This is why we calibrate on recent business, why the data split is temporal (not random), and why recalibration is necessary when the book changes.

**The practical rule:** calibrate on the most recent 20% of your data before the test period. Recalibrate at least annually, or quarterly if your book changes quickly.

### The formal finite-sample guarantee

For a calibration set of size `n`, the exact conformal guarantee is:

**P(y_new inside interval) >= 1 - alpha - 1/(n + 1)**

The `1/(n+1)` term is the finite-sample correction. For n=1,000 observations, it is 0.001 - negligible. For n=100, it is 0.01 - still small but worth noting if your calibration set is very small. Exercise 2 explores this empirically.

---

## Part 3: Setting up the notebook

### Create the notebook

Go to your Databricks workspace. In the left sidebar, click **Workspace**. Navigate to your user folder - this is usually listed under **Workspace > Users > your-email@company.com**.

Click the **+** button (at the top of the sidebar or next to your folder name) and choose **Notebook**. Name it `module-05-conformal-intervals`. Keep the default language as Python. Click **Create**.

The notebook opens with one empty cell. At the top of the notebook you will see a cluster selector - it usually shows "Detached" or your cluster name. If it says "Detached," click it and choose your cluster from the dropdown. Wait for the cluster name to appear with a green circle next to it. Do not run any cells until the cluster is connected.

If your cluster is not in the list, it may not be running. Go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes to start. Once the cluster shows "Running" with a green icon, come back to the notebook and connect to it.

### Install the libraries

In the first cell of your notebook, type this and run it by pressing **Shift+Enter**:

```python
%pip install "insurance-conformal[catboost]" catboost polars mlflow --quiet
```

You will see pip installation output scrolling for 30-60 seconds. Wait for it to finish completely. The last few lines will say something like:

```
Successfully installed insurance-conformal-0.x.x catboost-1.x.x ...
Note: you may need to restart the Python kernel to use updated packages.
```

Once you see that, run this in the next cell:

```python
dbutils.library.restartPython()
```

This restarts the Python session so the newly installed packages are available. Any variables from before the restart are gone - that is expected. The `%pip install` cell must always be the very first cell in the notebook, before any other code.

**What you should see after the restart:** the cell runs silently and the notebook is ready for the next cell. There is no output from `dbutils.library.restartPython()`.

### What each library does

- **insurance-conformal** - the core library for this module, from `github.com/burning-cost/insurance-conformal`. It provides the `InsuranceConformalPredictor` class, which calibrates conformal predictors, generates prediction intervals, and runs coverage diagnostics. The `[catboost]` extra installs the CatBoost integration.
- **catboost** - the gradient boosted tree library from Modules 3 and 4. We use it here for the base Tweedie model.
- **polars** - the data manipulation library from all previous modules.
- **mlflow** - the experiment tracking library built into Databricks.

### Confirm the imports work

In a new cell, type this and run it (Shift+Enter):

```python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from datetime import date
from catboost import CatBoostRegressor, Pool
from insurance_conformal import InsuranceConformalPredictor

print(f"Polars:   {pl.__version__}")
print(f"MLflow:   {mlflow.__version__}")
print(f"NumPy:    {np.__version__}")
print("InsuranceConformalPredictor: imported OK")
print("All imports OK")
```

**What you should see:**

```
Polars:   0.x.x
MLflow:   2.x.x
NumPy:    1.x.x
InsuranceConformalPredictor: imported OK
All imports OK
```

If you see `ModuleNotFoundError: No module named 'insurance_conformal'`, the install cell did not complete. Check that you ran the `%pip install` cell first, then `dbutils.library.restartPython()`. If the error persists, run the install cell again and restart again.

---

## Part 4: Building the synthetic motor dataset

We use the same synthetic UK motor portfolio as Modules 3 and 4. If you saved it to a Delta table, you can read it back. If not, we regenerate it here. The dataset has 100,000 policies with a superadditive interaction between young drivers and high vehicle groups - the same interaction the GBM found in Module 3.

Add a markdown cell to keep the notebook organised. In Databricks, cells starting with `%md` render as formatted text rather than code:

```python
%md
## Part 4: Data preparation
```

Now in a new cell, paste this and run it:

```python
rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors
areas             = ["A", "B", "C", "D", "E", "F"]
area              = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])
vehicle_group     = rng.integers(1, 51, size=n)
ncd_years         = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age        = rng.integers(17, 86, size=n)
conviction_points = rng.choice([0, 3, 6, 9], size=n, p=[0.78, 0.12, 0.07, 0.03])
annual_mileage    = rng.integers(3_000, 35_000, size=n)
exposure          = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency (log link, multiplicative effects)
INTERCEPT         = -3.10
area_effect       = {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.35, "E": 0.50, "F": 0.70}
conviction_effect = {0: 0.0, 3: 0.25, 6: 0.55, 9: 0.90}

log_mu = (
    INTERCEPT
    + np.array([area_effect[a] for a in area])
    + (-0.15) * ncd_years
    + 0.010   * (vehicle_group - 25)
    + np.where(driver_age < 25, 0.55, np.where(driver_age > 70, 0.20, 0.0))
    + np.array([conviction_effect[c] for c in conviction_points])
    + np.where((driver_age < 25) & (vehicle_group > 35), 0.30, 0.0)  # interaction
)

claim_count = rng.poisson(np.exp(log_mu) * exposure)

# Severity: Gamma-distributed, log link
sev_log_mu = (
    7.80
    + np.array([area_effect[a] * 0.3 for a in area])
    + 0.015 * (vehicle_group - 25)
    + np.array([conviction_effect[c] * 0.2 for c in conviction_points])
)
incurred = np.where(
    claim_count > 0,
    rng.gamma(shape=3.0, scale=np.exp(sev_log_mu) / 3.0, size=n) * claim_count,
    0.0,
)

# Assign accident years (chronological - essential for the temporal split)
accident_year = rng.choice(
    [2019, 2020, 2021, 2022, 2023, 2024],
    size=n,
    p=[0.12, 0.14, 0.16, 0.18, 0.20, 0.20],
)

df = pl.DataFrame({
    "accident_year":      accident_year.astype(np.int32),
    "area":               area,
    "vehicle_group":      vehicle_group.astype(np.int32),
    "ncd_years":          ncd_years.astype(np.int32),
    "driver_age":         driver_age.astype(np.int32),
    "conviction_points":  conviction_points.astype(np.int32),
    "annual_mileage":     annual_mileage.astype(np.int32),
    "exposure":           exposure,
    "claim_count":        claim_count.astype(np.int32),
    "incurred":           incurred,
}).with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
).sort("accident_year")

print(f"Dataset: {len(df):,} rows")
print(f"Accident years: {df['accident_year'].min()} - {df['accident_year'].max()}")
print(f"Overall claim frequency: {claim_count.mean():.4f}")
print(f"Mean pure premium: £{incurred.mean():.2f}")
print(f"Zero-claim rows: {(claim_count == 0).mean():.1%}")
df.head(5)
```

**What this does:** generates 100,000 synthetic motor policies with realistic rating factor distributions, a true Poisson frequency model, and Gamma severity. The `incurred / exposure` computation produces the pure premium (loss cost per year of exposure) that we model with Tweedie. The sort by `accident_year` is essential - without it, the temporal split below is meaningless.

**What you should see:**

```
Dataset: 100,000 rows
Accident years: 2019 - 2024
Overall claim frequency: 0.0536
Mean pure premium: £157.xx
Zero-claim rows: 94.x%
```

The exact numbers will match this if you use `seed=42`. If you see a `KeyError` or `NameError`, check that the cell above (the imports) ran successfully first.

---

## Part 5: The three-way temporal split

This is the most important structural decision in the module. The split determines whether the conformal coverage guarantee is valid.

### Why three sets, not two

Standard machine learning uses two sets: training (to fit the model) and test (to evaluate it). Conformal prediction needs three:

1. **Training set** - the model learns the relationship between features and losses
2. **Calibration set** - the conformal predictor measures how well the model's predictions are calibrated; this set is entirely unseen during model training
3. **Test set** - we evaluate whether the intervals produced using the calibration quantile actually achieve the stated coverage on new data

The calibration set must be unseen during model training. If any calibration observation influenced the model's parameters, the coverage guarantee can fail - the model may have partially memorised those observations, producing artificially small residuals and a calibration quantile that is too optimistic.

### Why temporal, not random

The exchangeability assumption underpinning the coverage guarantee requires that calibration data comes from the same distribution as test data. In insurance, time matters: claim frequencies, severities, and the mix of risks change from year to year. A random split where calibration data includes policies from 2020 and test data also includes policies from 2020 treats them as if they came from the same "snapshot," which hides temporal trends in the residuals.

The temporal split - training on older data, calibrating on middle data, testing on the most recent data - is the correct approach. It matches the operational reality: you train on historical business, calibrate on recent business, and make predictions for future business.

In a new cell, type this and run it:

```python
%md
## Part 5: Temporal data split
```

```python
# The DataFrame is already sorted by accident_year (we did this above)
n   = len(df)
X_COLS = ["vehicle_group", "driver_age", "ncd_years", "area", "conviction_points", "annual_mileage"]
CAT_FEATURES = ["area"]

train_end = int(0.60 * n)   # first 60% of rows (by accident year) = training
cal_end   = int(0.80 * n)   # next 20% = calibration

# Training set: oldest 60% of policies
X_train = df[:train_end][X_COLS].to_pandas()
y_train = df[:train_end]["pure_premium"].to_pandas()
e_train = df[:train_end]["exposure"].to_numpy()

# Calibration set: next 20%
X_cal   = df[train_end:cal_end][X_COLS].to_pandas()
y_cal   = df[train_end:cal_end]["pure_premium"].to_pandas()

# Test set: most recent 20%
X_test  = df[cal_end:][X_COLS].to_pandas()
y_test  = df[cal_end:]["pure_premium"].to_pandas()

# Verify the temporal split is what we think it is
train_years = df[:train_end]["accident_year"].unique().sort()
cal_years   = df[train_end:cal_end]["accident_year"].unique().sort()
test_years  = df[cal_end:]["accident_year"].unique().sort()

print(f"Training set:     {len(X_train):,} rows, years: {train_years.to_list()}")
print(f"Calibration set:  {len(X_cal):,}  rows, years: {cal_years.to_list()}")
print(f"Test set:         {len(X_test):,}  rows, years: {test_years.to_list()}")
```

**What this does:** splits the sorted DataFrame at the 60% and 80% marks. Because we sorted by `accident_year` before splitting, the training set contains the oldest policies, the calibration set contains the next-most-recent, and the test set contains the most recent.

**What you should see:** three sets of roughly 20,000, 20,000, and 20,000 rows (they will not be exactly equal because accident years are not uniformly distributed). The years should not overlap between sets - if they do, the sort did not work.

```
Training set:     60,xxx rows, years: [2019, 2020, 2021, 2022]
Calibration set:  20,xxx rows, years: [2022, 2023]
Test set:         20,xxx rows, years: [2023, 2024]
```

There will be some year overlap at the boundaries (e.g. 2022 appears in both training and calibration) because the rows within a year are ordered by `rng.choice` rather than strictly chronologically. This is acceptable - the important thing is that the majority of calibration rows are more recent than the majority of training rows.

---

## Part 6: Choosing the Tweedie power and fitting the base model

### The Tweedie family

CatBoost's Tweedie loss function models the compound Poisson-Gamma distribution that characterises aggregate insurance losses. The `variance_power` parameter `p` controls the variance-to-mean relationship:

- `p = 1`: Poisson (variance proportional to mean). Appropriate for claim counts only.
- `p = 2`: Gamma (variance proportional to mean squared). Appropriate for claim severity only.
- `1 < p < 2`: Compound Poisson-Gamma. **This is the distribution of aggregate losses** - a point mass at zero (no claims) combined with a positive continuous distribution when claims occur.

For UK motor pure premiums, `p = 1.5` is the standard choice. This sits in the middle of the compound Poisson-Gamma range and reflects both the frequency structure (lots of zeros from no-claim policies) and the severity structure (right-skewed losses when claims occur). Using p=1.3 or p=1.7 makes a small difference to the fit. Using p outside the range (1, 2) makes a large difference and is inappropriate for aggregate loss data.

**Practical note:** if your book has a very low claims rate (e.g. liability, where most policies never claim), you might choose p closer to 1.0. If severity is the dominant driver of variation (e.g. catastrophe-exposed property), p closer to 2.0 is more appropriate. For standard UK motor, 1.5 is correct.

### Build the Pool objects and fit the model

In a new cell:

```python
%md
## Part 6: Training the base Tweedie model
```

```python
# CatBoost Pool objects package features, labels, and metadata together
train_pool = Pool(X_train, y_train, cat_features=CAT_FEATURES)
cal_pool   = Pool(X_cal,   y_cal,   cat_features=CAT_FEATURES)
test_pool  = Pool(X_test,  y_test,  cat_features=CAT_FEATURES)

tweedie_params = {
    "loss_function":    "Tweedie:variance_power=1.5",
    "eval_metric":      "Tweedie:variance_power=1.5",
    "learning_rate":    0.05,
    "depth":            5,
    "min_data_in_leaf": 50,    # prevents overfitting to small insurance cells
    "iterations":       500,
    "random_seed":      42,
    "verbose":          100,   # print progress every 100 trees
}

model = CatBoostRegressor(**tweedie_params)
model.fit(train_pool, eval_set=cal_pool, early_stopping_rounds=50)

# Sanity-check predictions on the test set
preds_test = model.predict(test_pool)
print(f"\nTest set predictions:")
print(f"  Min: {preds_test.min():.2f}")
print(f"  Median: {np.median(preds_test):.2f}")
print(f"  Mean: {preds_test.mean():.2f}")
print(f"  Max: {preds_test.max():.2f}")
print(f"  Actual mean pure premium: {y_test.mean():.2f}")
```

**What this does:** creates the three CatBoost Pool objects, sets the Tweedie hyperparameters, and trains the model using the calibration set as an early-stopping validation set. The `verbose=100` setting prints the loss every 100 iterations.

**Why `min_data_in_leaf=50`:** without a minimum leaf size, deep trees can split on cells with only a handful of observations. These splits produce very precise but unreliable predictions for thin-cell risks. Thin-cell risks are exactly where conformal intervals will be widest - we need stable base model predictions in those regions, not wildly varying predictions from overfit splits.

**A note on early stopping:** using the calibration pool for early stopping means the model's iteration count has been influenced by the calibration data. This introduces a very minor dependency. In practice the effect on coverage is negligible. However, if you need strict separation for a regulatory audit, use a separate validation pool (drawn from the training set, not the calibration set) for early stopping and keep the calibration set entirely unseen during model fitting.

**What you should see:** training output like this, with the Tweedie loss printed every 100 iterations:

```
0:      learn: 2.05xxx  test: 2.07xxx   best: 2.07xxx (0)   total: ...
100:    learn: 1.82xxx  test: 1.85xxx   best: 1.84xxx (87)  total: ...
...
Stopped by early stopping after xxx iterations
```

The test set mean prediction should be close to (but not identical to) the actual mean pure premium. A large discrepancy here would suggest a misconfigured model.

---

## Part 7: The non-conformity score - why raw residuals fail for insurance

This section explains the most important technical choice in conformal prediction for insurance data. Read it carefully before moving to the calibration step.

### How conformal calibration works

The conformal predictor does not change the base model. The base model still produces point predictions. What conformal calibration does is:

1. Take the calibration set (data the model has never seen)
2. Compute a "non-conformity score" for each observation - a measure of how surprising that observation is given the model's prediction
3. Sort those scores and store the `(1 - alpha)` quantile

When predicting intervals for a new observation, the predictor asks: "what outcome range would have a non-conformity score below the stored quantile?" That range becomes the prediction interval.

The coverage guarantee follows from the sorting step: if calibration and test observations are exchangeable, a new test observation's score is equally likely to be any rank in the combined distribution. So the probability that its score falls below the stored quantile is at least `1 - alpha`.

### Why raw residuals produce wrong intervals for insurance

The simplest non-conformity score is the absolute residual: `|y - ŷ|`. This fails for insurance data.

Insurance losses are right-skewed and heteroscedastic. A risk with predicted pure premium £500 has more absolute variance than a risk with predicted pure premium £50. The natural variance of insurance losses scales with the mean: this is exactly what the Tweedie variance model captures. A miss of £100 on a £100-risk is a 100% error. A miss of £100 on a £1,000-risk is a 10% error. The raw residual treats them identically.

The consequence: the calibration quantile is set primarily by the majority of low-risk policies (there are more of them and they have smaller absolute residuals). When applied to high-risk policies, the same fixed-width interval is too narrow. The high-risk policies have genuine residuals that are larger in absolute terms, but the calibration quantile reflects the scale of the low-risk majority.

On typical UK motor data, raw residual intervals achieve approximately 90% overall coverage but only 72-75% coverage in the top risk decile. The aggregate number looks fine. The top decile - where reserves are concentrated - is seriously under-covered.

### The Pearson-weighted score

The `pearson_weighted` score normalises by the predicted Tweedie variance:

```
score = |y - ŷ| / ŷ^(p/2)
```

For Tweedie `p=1.5`, this divides by `ŷ^0.75`. This is the Pearson residual for a Tweedie model - it removes the mean-variance relationship and produces scores that are approximately homoscedastic across risk levels. A miss of £100 on a £100-risk and a miss of £1,000 on a £1,000-risk produce similar scores, reflecting similar fractional errors.

The result: interval widths scale with the predicted loss level. A £500-predicted risk has a wider absolute interval than a £50-predicted risk, which is correct - the genuine uncertainty is larger. Coverage is approximately flat across deciles because the calibration quantile is now set at the right scale for every risk level.

The `tweedie_power=1.5` parameter you pass to the predictor must match the `variance_power=1.5` you used when training the model. If they differ, the Pearson normalisation uses the wrong exponent and coverage will be wrong in ways that may not be obvious from the marginal coverage alone.

---

## Part 8: Calibrating the conformal predictor

Now we calibrate the predictor. This step is fast - it runs on the calibration set once and stores the sorted scores.

In a new cell:

```python
%md
## Part 8: Calibrating the conformal predictor
```

```python
cp = InsuranceConformalPredictor(
    model=model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)

cp.calibrate(X_cal, y_cal)

# Inspect the calibration scores
cal_scores = cp.calibration_scores_
print(f"Calibration set size: {len(X_cal):,}")
print(f"Number of calibration scores: {len(cal_scores):,}")
print(f"\nCalibration score distribution:")
print(f"  Min:    {cal_scores.min():.4f}")
print(f"  Median: {np.median(cal_scores):.4f}")
print(f"  90th percentile (alpha=0.10 quantile): {np.quantile(cal_scores, 0.90):.4f}")
print(f"  95th percentile (alpha=0.05 quantile): {np.quantile(cal_scores, 0.95):.4f}")
print(f"  Max:    {cal_scores.max():.4f}")
```

**What this does:** creates the conformal predictor, links it to the trained base model, and runs calibration on the held-out calibration set. The calibration step computes the Pearson residual for every calibration observation and sorts them. The 90th percentile score becomes the threshold for 90% prediction intervals.

**What you should see:**

```
Calibration set size: 20,xxx
Number of calibration scores: 20,xxx

Calibration score distribution:
  Min:    0.0000
  Median: 0.xxxx
  90th percentile (alpha=0.10 quantile): 2.xxxx
  95th percentile (alpha=0.05 quantile): 3.xxxx
  Max:    xx.xxxx
```

The distribution of Pearson residuals is right-skewed: most calibration observations have small residuals (the model predicts them well) and a small number have very large residuals (zero-loss risks where the model predicts a positive premium, or rare large claims). This is expected for insurance data.

### What just happened

The `cp.calibration_scores_` array contains the Pearson residuals for every calibration observation, sorted from smallest to largest. To generate a 90% prediction interval for a new risk, the library will find the range of outcomes that would produce a Pearson residual below the 90th percentile. Any outcome within that range is "consistent" with the model's prediction at the 90% confidence level.

Think of it as the model saying: "for this risk, I predict £342. Looking at how wrong I was on similar calibration observations, and scaling by the expected variance, the range of outcomes I cannot rule out at 90% confidence is [£X, £Y]."

---

## Part 9: Generating and interpreting prediction intervals

In a new cell:

```python
%md
## Part 9: Prediction intervals
```

```python
# 90% intervals: will cover the true outcome >= 90% of the time
intervals_90 = cp.predict_interval(X_test, alpha=0.10)

# 95% intervals: more conservative - for minimum premium floor applications
intervals_95 = cp.predict_interval(X_test, alpha=0.05)

# 80% intervals: used later in the practical minimum premium hybrid
intervals_80 = cp.predict_interval(X_test, alpha=0.20)

# Examine the structure
print("Output columns:", intervals_90.columns)
print("\nFirst 5 rows:")
print(intervals_90.head(5))
```

**What you should see:**

```
Output columns: ['point', 'lower', 'upper']

First 5 rows:
┌──────────┬──────────┬──────────┐
│ point    ┆ lower    ┆ upper    │
│ ---      ┆ ---      ┆ ---      │
│ f64      ┆ f64      ┆ f64      │
╞══════════╪══════════╪══════════╡
│ xxx.xx   ┆ 0.00     ┆ xxx.xx   │
...
```

The `lower` column is clipped at zero because insurance losses cannot be negative. Many rows will have `lower = 0.0` - this is correct for zero-inflated Tweedie data where a substantial fraction of policies will have no claims.

Now examine the interval widths:

```python
widths_90 = intervals_90["upper"] - intervals_90["lower"]
point_est = intervals_90["point"].to_numpy()

print("90% Interval width distribution:")
print(f"  Min:    {widths_90.min():.2f}")
print(f"  Median: {widths_90.median():.2f}")
print(f"  Mean:   {widths_90.mean():.2f}")
print(f"  90th percentile: {widths_90.quantile(0.90):.2f}")
print(f"  Max:    {widths_90.max():.2f}")

# Key ratio: how much do intervals scale with risk level?
rel_widths = widths_90.to_numpy() / np.clip(point_est, 1e-6, None)
print(f"\nRelative width (interval width / point estimate):")
print(f"  Median: {np.median(rel_widths):.2f}")
print(f"  90th percentile: {np.quantile(rel_widths, 0.90):.2f}")
print(f"  Max:    {rel_widths.max():.2f}")
```

**Interpreting the output:** with `pearson_weighted` intervals, the absolute width scales roughly in proportion to the point estimate. A risk with a £1,000 point estimate will have an approximately 10x wider absolute interval than a risk with a £100 point estimate. This is correct - the genuine uncertainty is larger for larger risks. The relative width (interval / point estimate) should be more stable across risks, though it will be wider for thin-cell risks where the model is uncertain.

---

## Part 10: The coverage diagnostic - the gate you must pass before using intervals

Do not skip this step. Do not use intervals for minimum premiums, reserving, or underwriting referral without passing this check.

The coverage diagnostic answers: "do the 90% intervals actually achieve 90% coverage on test data, and is that coverage consistent across risk levels?"

In a new cell:

```python
%md
## Part 10: Coverage validation
```

```python
# Run the full coverage-by-decile diagnostic
diag_90 = cp.coverage_by_decile(X_test, y_test, alpha=0.10)

print("Coverage by decile (target: 0.90)")
print("Decile 1 = lowest predicted premium, Decile 10 = highest predicted premium")
print()
print(diag_90.to_string())
```

**What this does:** bins the test set into 10 equal-count groups by predicted pure premium (so decile 1 is the 10% of risks with the lowest predicted loss, decile 10 is the 10% with the highest predicted loss), then measures what fraction of each decile's actual outcomes fall inside the prediction interval.

**What you should see:** something like this:

```
Coverage by decile (target: 0.90)
Decile 1 = lowest predicted premium, Decile 10 = highest predicted premium

┌────────┬──────────┬──────────────────┐
│ decile ┆ coverage ┆ n                │
│ ---    ┆ ---      ┆ ---              │
│ i64    ┆ f64      ┆ u32              │
╞════════╪══════════╪══════════════════╡
│ 1      ┆ 0.921    ┆ 2,xxx            │
│ 2      ┆ 0.912    ┆ 2,xxx            │
│ 3      ┆ 0.905    ┆ 2,xxx            │
│ 4      ┆ 0.897    ┆ 2,xxx            │
│ 5      ┆ 0.893    ┆ 2,xxx            │
│ 6      ┆ 0.889    ┆ 2,xxx            │
│ 7      ┆ 0.891    ┆ 2,xxx            │
│ 8      ┆ 0.886    ┆ 2,xxx            │
│ 9      ┆ 0.882    ┆ 2,xxx            │
│ 10     ┆ 0.878    ┆ 2,xxx            │
└────────┴──────────┴──────────────────┘
```

### Reading the diagnostic

**All deciles within 5pp of target (85-95% for a 90% interval).** This is a pass. The `pearson_weighted` score should achieve this on well-behaved insurance data.

**Monotone decline from bottom to top decile.** A small monotone decline (e.g. 92% in decile 1 to 88% in decile 10) is normal and acceptable - some residual heteroscedasticity in the scores is expected. If the decline is steep (e.g. 95% to 72%), the Pearson score has not fully normalised the variance and you should investigate whether the model has structural bias against large risks.

**Coverage below 85% in any decile.** This is a failure. The most common cause is distribution shift between calibration and test data: if the most recent business has had higher claims inflation, the calibration quantile (set on older business) will be too low for the test period. Recalibrate on more recent data (Part 14 covers this).

**Non-monotone pattern** (e.g. low in the middle deciles, high at the extremes). This usually indicates a specific risk segment in the test set that was absent from or underrepresented in calibration. Investigate which features differ between the problematic deciles and the well-covered deciles.

Now run the automated checks:

```python
coverages = diag_90["coverage"].to_list()
spread    = max(coverages) - min(coverages)
min_cov   = min(coverages)

print(f"\nMarginal coverage (all deciles combined): {sum(coverages)/len(coverages):.3f}")
print(f"Min decile coverage:  {min_cov:.3f}")
print(f"Max decile coverage:  {max(coverages):.3f}")
print(f"Coverage spread (max - min): {spread:.3f}")
print()

if spread > 0.10:
    print("WARNING: Coverage spread > 10pp. Try nonconformity='deviance' as an alternative.")
    print("If spread persists, the base model may have structural bias against large risks.")
elif min_cov < 0.85:
    print("WARNING: Minimum decile coverage below 85%. Check for distribution shift.")
    print("Recalibrate on more recent data before using intervals downstream.")
else:
    print("Coverage check PASSED. Intervals may be used for downstream applications.")
    print("Log these results to MLflow for audit.")
```

### Log the coverage diagnostics to MLflow

Coverage metrics are part of the model audit trail. Log them alongside the model parameters:

```python
with mlflow.start_run(run_name="module05_conformal_baseline") as run:
    conf_run_id = run.info.run_id

    # Log hyperparameters
    mlflow.log_params({
        "nonconformity_score":  "pearson_weighted",
        "tweedie_power":        1.5,
        "alpha_90":             0.10,
        "calibration_n":        len(X_cal),
        "model_depth":          5,
        "learning_rate":        0.05,
    })

    # Log coverage metrics
    mlflow.log_metric("marginal_coverage_90",   float(sum(coverages) / len(coverages)))
    mlflow.log_metric("min_decile_coverage_90", float(min(coverages)))
    mlflow.log_metric("max_decile_coverage_90", float(max(coverages)))
    mlflow.log_metric("coverage_spread_90",     float(spread))

    print(f"MLflow run ID: {conf_run_id}")
    print("Coverage metrics logged.")
```

---

## Part 11: Application 1 - Flagging uncertain risks for underwriting referral

The conformal interval gives you two dimensions for every risk:

1. **Risk level** - the point estimate (£342 is an expensive risk; £87 is a cheap one)
2. **Model uncertainty** - the relative interval width ((upper - lower) / point estimate)

These are independent. A risk can be expensive and well-understood (young driver with conviction points, but there are thousands in training data). A risk can be cheap and poorly understood (unusual feature combination that appears rarely in training).

Underwriting referral should be based on model uncertainty, not risk level. High-risk policies the model understands can be quoted automatically with high confidence. Uncertain policies - regardless of price - should go to a human.

In a new cell:

```python
%md
## Part 11: Underwriting referral flag
```

```python
# Extract arrays for computation
point  = intervals_90["point"].to_numpy()
lower  = intervals_90["lower"].to_numpy()
upper  = intervals_90["upper"].to_numpy()

# Relative width: how wide is the interval relative to the point estimate?
# This measures model uncertainty, not risk level
rel_width = (upper - lower) / np.clip(point, 1e-6, None)

# Set threshold at the 90th percentile -> exactly 10% referral rate
width_threshold = np.quantile(rel_width, 0.90)
flag_for_review = rel_width > width_threshold

print(f"Relative width threshold (90th percentile): {width_threshold:.4f}")
print(f"Policies flagged for review: {flag_for_review.sum():,} ({100*flag_for_review.mean():.1f}%)")
```

**What you should see:**

```
Relative width threshold (90th percentile): x.xxxx
Policies flagged for review: 2,xxx (10.0%)
```

The flag rate is exactly 10% by construction - you set the threshold at the 90th percentile of relative widths, so the top 10% are flagged. If the underwriting director wants a 5% referral rate, use the 95th percentile. If they want 15%, use the 85th percentile.

Now characterise the flagged risks:

```python
# Build a combined analysis frame
X_test_pl = pl.from_pandas(X_test.reset_index(drop=True))
X_test_pl = X_test_pl.with_columns([
    pl.Series("flagged",    flag_for_review),
    pl.Series("rel_width",  rel_width),
    pl.Series("point_est",  point),
    pl.Series("actual",     y_test.values),
])

# Profile: flagged vs unflagged
for flag_val, label in [(True, "FLAGGED (uncertain)"), (False, "Not flagged")]:
    sub = X_test_pl.filter(pl.col("flagged") == flag_val)
    print(f"\n{label}: {len(sub):,} policies")
    print(f"  Mean point estimate:    £{sub['point_est'].mean():.2f}")
    print(f"  Mean actual incurred:   £{sub['actual'].mean():.2f}")
    print(f"  Mean driver age:        {sub['driver_age'].mean():.1f}")
    print(f"  Mean vehicle group:     {sub['vehicle_group'].mean():.1f}")
    print(f"  % with convictions:     {(sub['conviction_points'] > 0).mean() * 100:.1f}%")
    print(f"  Mean relative width:    {sub['rel_width'].mean():.3f}")
```

**What you should see:** flagged risks skew towards younger drivers, higher vehicle groups, and conviction points. They are generally more expensive than unflagged risks. But the key point - which you need to explain to the underwriting director - is that flagging is based on training data density, not just risk level.

The conversation you should be prepared to have:

"Why are we flagging young drivers with conviction points? Surely we know they are high risk."

Your answer: "Yes, they are high risk and we know that well. We have thousands of such drivers in training data. But we have very few 19-year-olds with 9 conviction points in vehicle group 47. The model's prediction for that specific combination is uncertain, not the prediction for young drivers in general. We are flagging the thin-cell combinations where we genuinely lack data, not the common high-risk profiles where the model is confident."

This distinction matters for Consumer Duty. Referring risks for human review because the **model is uncertain** is a different and more defensible process than discretionary referrals based on underwriter judgment.

```python
# Verify: coverage is similar for flagged and unflagged groups
# (the flag is based on width, not on coverage failure)
for flag_val, label in [(True, "Flagged"), (False, "Not flagged")]:
    mask    = flag_for_review == flag_val
    covered = ((y_test.values[mask] >= lower[mask]) & (y_test.values[mask] <= upper[mask]))
    print(f"{label} actual coverage: {covered.mean():.3f}")
```

Both groups should show coverage close to 90%. If the flagged group has materially lower coverage (e.g. below 85%), it suggests the `pearson_weighted` score is not fully correcting for the heteroscedasticity in those thin cells.

---

## Part 12: Application 2 - Risk-specific minimum premium floors

### The problem with flat multipliers

Your current minimum premium policy probably reads something like: "Minimum premium = 1.3x technical premium, subject to a floor of £250."

The 1.3x multiplier has no principled basis in the distribution of losses. It is a rule of thumb chosen based on historical reserve adequacy experience. Applied uniformly, it:

- Overcharges stable, well-understood risks (their 95th percentile loss is well below 1.3x the technical premium)
- Undercharges volatile, uncertain risks (their 95th percentile loss may be 2x or 3x the technical premium)

Both create problems. Overcharging stable customers is a Consumer Duty fair value issue. Undercharging volatile risks understates the reserve requirement and creates solvency adequacy risk.

### Using the conformal upper bound as a floor

The 95% conformal upper bound is the loss level we expect to be exceeded only 5% of the time. Using it as a minimum premium floor means: "we price this risk so that it will be unprofitable for us no more than 5% of the time, as measured on the calibration data."

That is a principled, auditable justification. Not "we apply 1.3x because that is what we have always done," but "we apply a floor calibrated to the 95th percentile of predicted losses, validated on historical business, with documented coverage of 95% across all risk deciles."

In a new cell:

```python
%md
## Part 12: Minimum premium floors
```

```python
# Extract upper bounds at different alpha levels
upper_90 = intervals_90["upper"].to_numpy()   # 90% upper bound (10% exceedance)
upper_95 = intervals_95["upper"].to_numpy()   # 95% upper bound (5% exceedance)
upper_80 = intervals_80["upper"].to_numpy()   # 80% upper bound (20% exceedance)

# Three floor approaches
floor_conventional = np.maximum(1.3 * point, 250)    # current policy
floor_conformal_95 = upper_95                          # principled 95% upper bound
floor_practical    = np.maximum(1.5 * point, upper_80) # hybrid: 1.5x vs 80% upper

print(f"{'Approach':<35} {'Median':>10} {'Mean':>10} {'95th pctile':>14}")
print("-" * 72)
for label, floor in [
    ("Conventional (1.3x, floor £250)",     floor_conventional),
    ("Conformal 95% upper bound",           floor_conformal_95),
    ("Practical (1.5x vs 80% upper)",       floor_practical),
]:
    print(f"{label:<35} {np.median(floor):>10.2f} {np.mean(floor):>10.2f} {np.quantile(floor, 0.95):>14.2f}")
```

**What you should see:** the three approaches produce similar medians but different tails. The conformal floor will be higher than the conventional floor for volatile risks (wide intervals) and lower for stable risks (narrow intervals).

Now find which risks the two approaches disagree on:

```python
# Where conformal floor is HIGHER than conventional: conventional undercharges these risks
higher_than_conventional = floor_conformal_95 > floor_conventional
print(f"\nRisks where conformal floor > conventional floor: {higher_than_conventional.sum():,} ({higher_than_conventional.mean():.1%})")
if higher_than_conventional.any():
    sub = X_test_pl.filter(pl.Series(higher_than_conventional))
    print(f"  Their profile: mean age {sub['driver_age'].mean():.1f}, "
          f"mean vehicle group {sub['vehicle_group'].mean():.1f}, "
          f"{(sub['conviction_points'] > 0).mean() * 100:.1f}% with convictions")

# Where conformal floor is LOWER than conventional: conventional overcharges these risks
lower_than_conventional = floor_conformal_95 < floor_conventional
print(f"\nRisks where conformal floor < conventional floor: {lower_than_conventional.sum():,} ({lower_than_conventional.mean():.1%})")
if lower_than_conventional.any():
    sub_lo = X_test_pl.filter(pl.Series(lower_than_conventional))
    print(f"  Their profile: mean age {sub_lo['driver_age'].mean():.1f}, "
          f"mean vehicle group {sub_lo['vehicle_group'].mean():.1f}, "
          f"mean NCD {sub_lo['ncd_years'].mean():.1f} years")
```

**Interpreting the results:**

- Risks where the conformal floor is higher: these are volatile risks (wide intervals) where the conventional 1.3x multiplier does not cover the genuine uncertainty. The conformal floor is the correct answer here - it reflects the actual 95th percentile of losses for that risk profile.

- Risks where the conformal floor is lower: these are stable risks (narrow intervals) where the conventional floor is excessive. On Consumer Duty grounds, the conformal floor is more defensible: you are not applying an arbitrary multiplier to a well-understood risk.

The FCA evidence: the coverage-by-decile diagnostic shows that the 95% intervals achieve 95% coverage across all risk deciles. This is the mathematical basis for the floor - not a rule of thumb, but a calibrated threshold validated on recent business.

---

## Part 13: Application 3 - Portfolio reserve range estimates

Individual prediction intervals aggregate to portfolio-level range estimates. This section shows how, and is explicit about the assumptions that make the aggregation valid or invalid.

In a new cell:

```python
%md
## Part 13: Portfolio reserve ranges
```

```python
# Portfolio point estimate: sum of all individual predictions
portfolio_point = point.sum()

# --- Method 1: Naive (worst-case correlation) ---
# Assume all risks simultaneously hit their lower (or upper) bounds.
# This is the catastrophe scenario: all risks move together.
portfolio_lower_naive = lower.sum()
portfolio_upper_naive = upper.sum()

# --- Method 2: Independence (CLT approximation) ---
# Assume individual risks are independent. Sum the variances, take the square root.
# IMPORTANT WARNING (read before presenting to reserving team):
# - This uses a symmetric normal approximation to individual losses.
#   Tweedie losses are right-skewed, so this understates individual variance.
# - This assumes zero correlation across risks.
#   Systemic events (weather, economic shocks) violate this assumption.
# - The independence range is an optimistic lower bound, not a central estimate.
#
# For a 90% interval, width ≈ 2 × 1.645 × sd, so sd ≈ width / 3.29
approx_sd        = (upper - lower) / 3.29
portfolio_sd     = np.sqrt((approx_sd ** 2).sum())

portfolio_lower_indep = max(0, portfolio_point - 1.645 * portfolio_sd)
portfolio_upper_indep = portfolio_point + 1.645 * portfolio_sd

print("Portfolio reserve range estimates")
print("=" * 55)
print(f"Point estimate (sum of technical premiums): £{portfolio_point:,.0f}")
print()
print("90% Range (Naive - perfect correlation, worst case):")
print(f"  Lower: £{portfolio_lower_naive:,.0f}")
print(f"  Upper: £{portfolio_upper_naive:,.0f}")
print()
print("90% Range (Independence - CLT approximation, optimistic):")
print(f"  Lower: £{portfolio_lower_indep:,.0f}")
print(f"  Upper: £{portfolio_upper_indep:,.0f}")
print()
diversification_benefit = portfolio_upper_naive / portfolio_upper_indep
print(f"Diversification benefit (naive / independence upper): {diversification_benefit:.1f}x")
print()
print("NOTE: True portfolio range lies between these bounds.")
print("Naive bound: relevant for catastrophe scenarios (correlated weather, economic shocks).")
print("Independence bound: relevant for idiosyncratic risk (individual accidents, theft).")
print("Add a catastrophe overlay separately for systemic events.")
```

**What you should see:** the naive upper bound is much larger than the independence upper bound. For a portfolio of 20,000 test policies, the diversification benefit is typically 3-10x - the independence bound is far lower than the naive bound because portfolio diversification reduces the aggregate uncertainty when risks are uncorrelated.

**How to present this to the reserving team:**

Present both bounds explicitly with their assumptions:

- "The independence bound (£X) is the expected 90th percentile reserve if all claims events are independent. This is appropriate for standard idiosyncratic risk."
- "The naive bound (£Y) is the expected 90th percentile reserve if all risks are simultaneously adversely affected. This is the catastrophe scenario."
- "The true range lies between these. For storm or flood events, the relevant bound is closer to the naive. For everyday claims events, it is closer to the independence."

```python
# Segmented reserve range by area: useful for the reinsurance conversation
area_col    = X_test.reset_index(drop=True)["area"]
seg_results = {}

for area_val in sorted(area_col.unique()):
    mask = area_col.values == area_val
    seg_results[area_val] = {
        "n_risks":     mask.sum(),
        "total_point": point[mask].sum(),
        "total_lower": lower[mask].sum(),
        "total_upper": upper[mask].sum(),
    }

print(f"\n{'Area':<6} {'Risks':>7} {'Point £':>12} {'Lower £':>12} {'Upper £':>12} {'Width ratio':>13}")
print("-" * 65)
for area_val, seg in seg_results.items():
    ratio = seg["total_upper"] / max(seg["total_lower"], 1)
    print(f"{area_val:<6} {seg['n_risks']:>7,} {seg['total_point']:>12,.0f} "
          f"{seg['total_lower']:>12,.0f} {seg['total_upper']:>12,.0f} {ratio:>13.2f}x")
```

Areas with a high upper/lower ratio have the most reserve uncertainty. These are candidates for area-specific stop-loss reinsurance cover.

---

## Part 14: Recalibration - the key operational advantage

The practical value of conformal prediction for insurance operations is that calibration is separable from training. When claim frequencies or severities drift, you do not need to retrain the base model. You recalibrate the predictor on recent business.

Retrain: 15-25 minutes for a CatBoost model on 60,000 observations.
Recalibrate: under 10 seconds on 2,000 observations.

In a new cell:

```python
%md
## Part 14: Recalibration
```

```python
import time

# Simulate a quarterly recalibration using only the most recent 2,000 observations
# from the calibration period
X_cal_arr = X_cal.reset_index(drop=True)
y_cal_arr = y_cal.reset_index(drop=True)

X_cal_recent = X_cal_arr.tail(2_000)
y_cal_recent = y_cal_arr.tail(2_000)

print(f"Recalibrating on {len(X_cal_recent):,} most recent calibration observations...")
t0 = time.time()
cp.calibrate(X_cal_recent, y_cal_recent)
recal_time = time.time() - t0
print(f"Recalibration complete in {recal_time:.2f} seconds")

# Check coverage after recalibration
diag_recal = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
recal_coverages = diag_recal["coverage"].to_list()
print(f"\nCoverage after recalibration on {len(X_cal_recent):,} observations:")
print(f"  Marginal: {sum(recal_coverages)/len(recal_coverages):.3f}")
print(f"  Min decile: {min(recal_coverages):.3f}")
print(f"  Max decile: {max(recal_coverages):.3f}")
```

**What you should see:** coverage after recalibration will be close to the full-calibration-set result. With 2,000 calibration observations, coverage estimates will have slightly higher variance (the coverage measurement is less precise with fewer calibration points) but should still pass the 85-95% check.

```python
# Compare full calibration (20,000 obs) vs recent-only (2,000 obs)
print("\nCalibration set size comparison:")
print(f"{'Size':<12} {'Marginal coverage':>20} {'Min decile coverage':>22}")
print(f"{'2,000 (recent)':.<12} {sum(recal_coverages)/len(recal_coverages):>20.3f} {min(recal_coverages):>22.3f}")
print(f"{'20,000 (full)':.<12} {sum(coverages)/len(coverages):>20.3f} {min(coverages):>22.3f}")
```

### When recalibration is sufficient vs when you need to retrain

**Recalibration restores coverage** when the base model's predictions are still directionally correct but the error scale has shifted. This covers:
- Claims inflation (all losses have risen by a percentage)
- Frequency drift (all risks have a higher claims rate by a constant multiplier)

**Recalibration fails** when the model's rankings have deteriorated - it no longer correctly identifies which risks are more or less likely to claim. The diagnostic: if recalibration restores marginal coverage (overall coverage returns to 90%) but coverage-by-decile remains poor (top decile stays at 75%), the model is calibrated at the portfolio level but not within deciles. The model itself is wrong about relative risk, not just about scale.

**The operational rule: recalibrate quarterly, retrain annually, unless coverage-by-decile shows a structural break sooner.**

---

## Part 15: Writing results to Unity Catalog

Intervals and coverage diagnostics go to Delta tables. This gives you version history, time travel, and a permanent audit trail linked to the MLflow run ID.

In a new cell:

```python
%md
## Part 15: Writing to Delta tables
```

```python
import pandas as pd

# Recalibrate on full calibration set before writing production results
cp.calibrate(X_cal, y_cal)

# Regenerate intervals with the full calibration
intervals_90 = cp.predict_interval(X_test, alpha=0.10)
intervals_95 = cp.predict_interval(X_test, alpha=0.05)

upper_90 = intervals_90["upper"].to_numpy()
upper_95 = intervals_95["upper"].to_numpy()
point    = intervals_90["point"].to_numpy()
lower    = intervals_90["lower"].to_numpy()

rel_width        = (upper_90 - lower) / np.clip(point, 1e-6, None)
width_threshold  = np.quantile(rel_width, 0.90)
flag_for_review  = rel_width > width_threshold
floor_conformal  = upper_95
floor_practical  = np.maximum(1.5 * point, intervals_80["upper"].to_numpy())

# Build the output DataFrame
intervals_to_write = intervals_90.to_pandas().copy()
intervals_to_write["model_run_date"]       = str(date.today())
intervals_to_write["mlflow_run_id"]        = conf_run_id
intervals_to_write["alpha"]                = 0.10
intervals_to_write["nonconformity_score"]  = "pearson_weighted"
intervals_to_write["tweedie_power"]        = 1.5
intervals_to_write["flag_for_review"]      = flag_for_review.tolist()
intervals_to_write["relative_width"]       = rel_width.tolist()
intervals_to_write["floor_conformal_95"]   = floor_conformal.tolist()
intervals_to_write["floor_practical"]      = floor_practical.tolist()

print("Writing intervals to pricing.motor.conformal_intervals...")
(
    spark.createDataFrame(intervals_to_write)
    .write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("pricing.motor.conformal_intervals")
)
print(f"Written {len(intervals_to_write):,} rows.")
```

**What this does:** creates a Delta table with one row per policy in the test set. Each row has the point estimate, lower and upper bounds, the underwriting referral flag, relative width, and both minimum premium floors. The `mlflow_run_id` links this table to the MLflow experiment entry where the model and coverage metrics are stored.

Now write the coverage log in append mode:

```python
# Coverage diagnostics: append so we can track over time
diag_final    = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
diag_to_write = diag_final.to_pandas().copy()
diag_to_write["model_run_date"] = str(date.today())
diag_to_write["mlflow_run_id"]  = conf_run_id
diag_to_write["test_years"]     = str(test_years.to_list())
diag_to_write["calibration_n"]  = len(X_cal)

print("Appending to pricing.motor.conformal_coverage_log...")
(
    spark.createDataFrame(diag_to_write)
    .write.format("delta")
    .mode("append")
    .saveAsTable("pricing.motor.conformal_coverage_log")
)
print("Done.")
```

The coverage log is append-mode. Every calibration run adds a new set of diagnostic rows with a timestamp. Query the history:

```python
spark.sql("""
    SELECT model_run_date, decile, coverage, calibration_n, mlflow_run_id
    FROM pricing.motor.conformal_coverage_log
    ORDER BY model_run_date DESC, decile ASC
""").show(30)
```

A declining top-decile coverage trend over successive runs signals that the calibration is going stale. Set an alert: if the top decile falls below 85% for a 90% interval, trigger recalibration before the next reserve cycle.

---

## Part 16: Limitations for regulatory presentation

When presenting conformal prediction intervals to regulators or a pricing committee, you must be explicit about the assumptions and limitations. Eight documented limitations follow. Document all of them, even the ones that do not apply to your specific use case - regulators and actuaries are more comfortable with a methodology that acknowledges its boundaries than one that claims to be universally applicable.

### Limitation 1: Marginal, not conditional, coverage

The coverage guarantee is marginal: at least 90% of all test observations are covered. It is not conditional: it does not guarantee 90% coverage within every risk segment. Coverage-by-decile validation tests this empirically, but passes or fails at the decile level, not at the level of individual feature combinations.

**Mitigation:** run coverage-by-decile and by key feature values (area, driver age band, vehicle group band) before using intervals for any downstream purpose.

### Limitation 2: Exchangeability assumption

The guarantee requires calibration and test data to be exchangeable (from the same distribution). Temporal drift breaks this. Inflation, regulatory changes (e.g. FCA pricing remedies in January 2022), and changes in the book composition (new distribution channels, portfolio acquisitions) all introduce distribution shift that can invalidate the guarantee.

**Mitigation:** calibrate on the most recent available data. Monitor coverage on recent live business quarterly. Recalibrate when coverage falls below 85%.

### Limitation 3: The base model must be directionally correct

Conformal calibration adjusts the width of intervals to achieve coverage, but it cannot fix a model that is systematically wrong about which risks are more or less dangerous. If the base model's risk ranking has degraded (Gini coefficient fallen substantially), wider intervals will not restore meaningful coverage-by-decile.

**Mitigation:** track the base model's Gini coefficient over time alongside coverage metrics. If coverage-by-decile deteriorates despite recalibration, and Gini has fallen, retrain the base model.

### Limitation 4: Asymmetric intervals and the CLT aggregation

Individual intervals for Tweedie models are asymmetric: the distance from the point estimate to the upper bound is much larger than the distance to the lower bound (which is clipped at zero for many policies). The CLT-based portfolio range aggregation uses a symmetric normal approximation, which understates individual variance and therefore understates portfolio-level range.

**Mitigation:** present the independence range as an optimistic lower bound on portfolio uncertainty. For more accurate portfolio ranges, simulate from the Tweedie distribution at the calibrated quantile scale.

### Limitation 5: Minimum calibration set size

The finite-sample correction to the coverage guarantee is `1/(n+1)`. For n=1,000, this is 0.001 - negligible. But the precision of the coverage estimate itself follows a binomial distribution with standard deviation `sqrt(alpha*(1-alpha)/n)`. For n=500 and alpha=0.10, the 95% confidence interval on coverage is approximately ±2.6pp. For small books, the calibration set may be too small to precisely verify that coverage is achieving the target.

**Mitigation:** use at least 2,000 calibration observations. For books with fewer total policies, consider using cross-conformal prediction (Exercise 2 explores the sample-size tradeoff).

### Limitation 6: Feature distribution shift (covariate shift)

Calibration was run on a specific distribution of features. If the future book has substantially different feature distribution - new markets, different vehicle groups, changed distribution channel demographics - the coverage guarantee is weakened because the calibration scores may not represent the scale of prediction error on the new feature distribution.

**Mitigation:** compare the feature distribution of the calibration set to the current book quarterly. Flag large shifts (e.g. mean vehicle group rising by 5+ points) as a trigger for recalibration.

### Limitation 7: Intervals do not account for parameter uncertainty in the base model

The base CatBoost model was trained on a specific dataset. A different training set (e.g. with different random seed or a different bootstrapped subsample) would produce different predictions. The conformal intervals account for prediction error given the fitted model but not for the uncertainty in the model parameters themselves.

**Mitigation:** for high-stakes applications (e.g. reserve range for a reinsurance negotiation), complement conformal intervals with a model uncertainty estimate from bootstrap resampling of the base model.

### Limitation 8: The pearson_weighted score assumes Tweedie is the correct distributional family

The Pearson normalisation divides by `ŷ^(p/2)`, which is the correct normalisation for a Tweedie model. If the true distribution departs substantially from Tweedie - for example, if severity is bimodal (small attritional claims plus occasional large claims with a different distribution) - the normalisation may not fully homogenise the scores, resulting in residual heteroscedasticity in coverage.

**Mitigation:** validate coverage-by-decile and additionally validate by severity quantile (do intervals covering large-claim policies achieve similar coverage to those covering small-claim policies?). If heteroscedasticity persists, try `nonconformity="deviance"` as an alternative score.

---

## Part 17: Checkpoint - verify the complete notebook

Before moving to the exercises, check that your notebook has completed successfully.

Run this verification cell:

```python
# Checkpoint: verify all components have run correctly
print("Module 5 checkpoint")
print("=" * 40)

# 1. Base model
try:
    test_pred = model.predict(test_pool)
    print(f"[PASS] Base model: predicts {len(test_pred):,} test observations")
except Exception as e:
    print(f"[FAIL] Base model: {e}")

# 2. Conformal predictor calibrated
try:
    n_cal_scores = len(cp.calibration_scores_)
    print(f"[PASS] Conformal predictor: {n_cal_scores:,} calibration scores stored")
except Exception as e:
    print(f"[FAIL] Conformal predictor: {e}")

# 3. Coverage check
try:
    diag = cp.coverage_by_decile(X_test, y_test, alpha=0.10)
    min_cov = diag["coverage"].min()
    status  = "PASS" if min_cov >= 0.85 else "FAIL"
    print(f"[{status}] Coverage: minimum decile coverage = {min_cov:.3f} (threshold: 0.85)")
except Exception as e:
    print(f"[FAIL] Coverage diagnostic: {e}")

# 4. MLflow logging
try:
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    run    = client.get_run(conf_run_id)
    print(f"[PASS] MLflow run '{run.info.run_name}' logged (run_id: {conf_run_id[:8]}...)")
except Exception as e:
    print(f"[FAIL] MLflow: {e}")

print()
print("If all four checks show [PASS], proceed to the exercises.")
print("If any show [FAIL], rerun the relevant Part section above.")
```

---

## What just happened - a summary before the exercises

You have built a complete conformal prediction pipeline for UK motor insurance data.

The pipeline has five components:

1. **Base model** (Tweedie CatBoost, `p=1.5`): produces point estimates of pure premium. Trained on the oldest 60% of the data. Early stopping uses the calibration pool as validation.

2. **Conformal predictor** (Pearson-weighted non-conformity score): stores the sorted calibration residuals as a lookup table. Takes under 10 seconds to calibrate. Separable from model training: you can recalibrate without retraining.

3. **Coverage diagnostic** (coverage-by-decile): the mandatory check that intervals are valid across the risk spectrum, not just on average. Must pass before using intervals downstream.

4. **Downstream applications**: underwriting referral flag (relative width threshold), minimum premium floors (conformal upper bound), portfolio reserve ranges (naive and independence aggregation).

5. **Operational loop**: coverage monitoring table accumulates over time. Quarterly recalibration refreshes the coverage. Annual retrain refreshes the base model. Coverage-by-decile drives the decision on which action is needed.

The eight documented limitations are the governance deliverable: this is what you present to the FCA or the reserving committee when they ask about the mathematical basis and the assumptions. The more precisely you can state the limitations, the more credible the methodology is.

---

## What comes next

Module 6 covers credibility and Bayesian methods. The thin cells that produce wide conformal intervals in this module are the same thin cells where credibility blending matters most: cells with too few observations to produce a reliable estimate from first principles. Bayesian hierarchical models with informative priors borrow strength from similar cells, which in turn narrows the prediction intervals without sacrificing coverage. The two methodologies are complementary, not competing.
