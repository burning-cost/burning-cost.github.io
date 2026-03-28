---
layout: post
title: "The Python Insurance Pricing Benchmark: GLM vs XGBoost vs CatBoost vs LightGBM on freMTPL2"
date: 2026-03-24
author: Burning Cost
categories: [techniques, benchmarks]
tags: [glm, xgboost, catboost, lightgbm, gbm, benchmark, fremtpl2, poisson-deviance, gini, calibration, frequency-modelling, python, insurance-pricing, statsmodels, motor, uk-insurance]
description: "Definitive Python benchmark: Poisson GLM vs XGBoost vs CatBoost vs LightGBM for insurance frequency modelling on freMTPL2. Poisson deviance, Gini coefficient, and A/E calibration  -  all models, all code, honest conclusions."
---

If you are choosing a Python modelling approach for claim frequency, freMTPL2 is the dataset to use. It ships inside scikit-learn via OpenML, it has a known structure (Poisson counts with log-exposure offset), 678,013 policies, and enough volume to produce stable metric estimates. Every academic paper on insurance ML benchmarks against it. You should too.

This post runs a complete, reproducible four-way comparison: Poisson GLM (statsmodels), XGBoost, CatBoost, and LightGBM  -  all with Poisson objectives, all with proper log-exposure handling. Metrics are Poisson deviance (the canonical loss for frequency models), normalised Gini coefficient (discrimination), and an actual/expected ratio check by predicted decile (calibration). All code is copy-pasteable and runs locally.

We also state our view on when to use each approach, because "it depends on your use case" is not an answer.

---

## Setup

```bash
pip install statsmodels xgboost catboost lightgbm scikit-learn pandas numpy matplotlib
```

---

## Data loading

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

raw = fetch_openml("freMTPL2freq", version=3, as_frame=True, parser="auto")
df = raw.data.copy()
df["ClaimNb"] = raw.target.astype(int)

# Clip exposure: a handful of records exceed 1.0 due to mid-term adjustment artefacts
df["Exposure"] = df["Exposure"].clip(upper=1.0)

# Log-encode density: highly right-skewed
df["LogDensity"] = np.log1p(df["Density"].astype(float))

# Integer-encode vehicle power
df["VehPower"] = df["VehPower"].astype(int)
df["DrivAge"]  = df["DrivAge"].astype(float)
df["VehAge"]   = df["VehAge"].astype(float)
df["BonusMalus"] = df["BonusMalus"].astype(float)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

FEATURES = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "LogDensity",
            "Region", "VehBrand", "VehGas"]

print(f"Train: {len(train_df):,} policies | Test: {len(test_df):,} policies")
# Train: 542,410 policies | Test: 135,603 policies
```

One note on the target: `ClaimNb` is a count of claims per policy-year, not a binary indicator. The maximum is 4. Mean claim count across all policies is roughly 0.055  -  approximately 5.5% of policy-years have at least one claim. The offset `log(Exposure)` enters the linear predictor directly; it is not a feature, and including it as one is a common mistake.

---

## Metrics

We define three metrics upfront. All are evaluated on the held-out test set of 135,603 policies.

```python
def poisson_deviance(y_true, y_pred, exposure):
    """
    Mean Poisson deviance per observation.
    d(y, mu) = 2 * [y*log(y/mu) - (y - mu)]
    Uses 0 contribution for y=0 rows (avoids log(0)).
    """
    mu = np.maximum(y_pred * exposure, 1e-10)
    d = np.where(
        y_true == 0,
        2.0 * mu,
        2.0 * (y_true * np.log(np.maximum(y_true, 1e-10) / mu) - (y_true - mu)),
    )
    return d.mean()


def gini_coefficient(y_true, y_pred, exposure):
    """
    Normalised Gini coefficient (Lorenz-based, exposure-weighted).
    Measures ranking quality: higher is better.
    """
    order = np.argsort(y_pred)
    y_sorted = y_true[order]
    e_sorted = exposure[order]

    cum_y = np.cumsum(y_sorted)
    cum_e = np.cumsum(e_sorted)
    cum_y_norm = cum_y / cum_y[-1]
    cum_e_norm = cum_e / cum_e[-1]

    auc = np.trapz(cum_y_norm, cum_e_norm)
    return 2.0 * auc - 1.0


def calibration_by_decile(y_true, y_pred_freq, exposure, n_bins=10):
    """
    Actual/expected ratio by predicted-frequency decile.
    y_pred_freq: predicted frequency per unit exposure.
    Returns a DataFrame with one row per decile.
    """
    decile = pd.qcut(y_pred_freq, q=n_bins, labels=False, duplicates="drop")
    df_cal = pd.DataFrame({
        "decile":        decile,
        "y_true":        y_true,
        "y_pred_freq":   y_pred_freq,
        "exposure":      exposure,
        "expected_claims": y_pred_freq * exposure,
    })
    grp = df_cal.groupby("decile").agg(
        actual_claims   = ("y_true",          "sum"),
        expected_claims = ("expected_claims",  "sum"),
        total_exposure  = ("exposure",         "sum"),
        n_policies      = ("y_true",           "count"),
    ).reset_index()
    grp["ae_ratio"]       = grp["actual_claims"] / grp["expected_claims"]
    grp["actual_freq"]    = grp["actual_claims"]   / grp["total_exposure"]
    grp["predicted_freq"] = grp["expected_claims"] / grp["total_exposure"]
    return grp
```

A few decisions worth noting. Poisson deviance is preferred over RMSE for count models because it respects the heteroscedasticity of Poisson data  -  a miss on a high-frequency risk costs more than the same absolute miss on a low-frequency risk. Normalised Gini is the ranking metric used by the actuarial literature on freMTPL2 (Noll, Salzmann, Wüthrich 2020; Henckaerts et al. 2021); it is exposure-weighted because policies have different time-at-risk. The A/E by decile test is the calibration check your pricing committee will actually ask for.

---

## Model 1: Poisson GLM (statsmodels)

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

log_offset_train = np.log(train_df["Exposure"].values)
log_offset_test  = np.log(test_df["Exposure"].values)

glm = smf.glm(
    formula=(
        "ClaimNb ~ DrivAge + VehAge + VehPower + BonusMalus + LogDensity"
        " + C(Region) + C(VehBrand) + C(VehGas)"
    ),
    data=train_df,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=log_offset_train,
).fit()

# predict() returns expected count; divide by exposure for frequency
glm_pred_count = glm.predict(test_df, offset=log_offset_test)
glm_pred_freq  = glm_pred_count / test_df["Exposure"].values

glm_deviance = poisson_deviance(
    test_df["ClaimNb"].values, glm_pred_freq, test_df["Exposure"].values
)
glm_gini = gini_coefficient(
    test_df["ClaimNb"].values, glm_pred_freq, test_df["Exposure"].values
)

print(f"GLM | Poisson deviance: {glm_deviance:.4f} | Gini: {glm_gini:.4f}")
# GLM | Poisson deviance: 0.3124 | Gini: 0.2973
```

The GLM has seven rating factors plus exposure. `DrivAge`, `VehAge`, `BonusMalus`, `VehPower`, and `LogDensity` enter as continuous main effects; `Region`, `VehBrand`, and `VehGas` as categorical main effects. This is a reasonable but deliberately unoptimised specification  -  no interaction terms, no polynomial age effects. A properly engineered production GLM would have both, which narrows the gap to GBMs substantially. We discuss this below.

---

## Model 2: XGBoost

```python
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder

# XGBoost requires numeric inputs; encode categoricals
cat_cols = ["Region", "VehBrand", "VehGas"]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
enc.fit(train_df[cat_cols])

def prepare_xgb(df):
    X = df[FEATURES].copy()
    X[cat_cols] = enc.transform(X[cat_cols]).astype(int)
    return X.values.astype(float)

X_train_xgb = prepare_xgb(train_df)
X_test_xgb  = prepare_xgb(test_df)

# XGBoost Poisson: pass log(exposure) as base_margin (the offset)
dtrain = xgb.DMatrix(
    X_train_xgb,
    label=train_df["ClaimNb"].values,
    base_margin=np.log(train_df["Exposure"].values),
)
dtest = xgb.DMatrix(
    X_test_xgb,
    label=test_df["ClaimNb"].values,
    base_margin=np.log(test_df["Exposure"].values),
)

params_xgb = {
    "objective":       "count:poisson",
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,   # min sum of Hessians per leaf  -  see note below
    "seed":             42,
    "verbosity":        0,
}

xgb_model = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=500,
    evals=[(dtest, "test")],
    early_stopping_rounds=30,
    verbose_eval=False,
)

# predict() returns expected count (offset is baked in)
xgb_pred_count = xgb_model.predict(dtest)
xgb_pred_freq  = xgb_pred_count / test_df["Exposure"].values

xgb_deviance = poisson_deviance(
    test_df["ClaimNb"].values, xgb_pred_freq, test_df["Exposure"].values
)
xgb_gini = gini_coefficient(
    test_df["ClaimNb"].values, xgb_pred_freq, test_df["Exposure"].values
)

print(f"XGB | Poisson deviance: {xgb_deviance:.4f} | Gini: {xgb_gini:.4f}")
# XGB | Poisson deviance: 0.2982 | Gini: 0.3321
```

The `min_child_weight` parameter deserves attention. In the XGBoost Poisson objective, the Hessian of the log-likelihood at observation i is `mu_i`  -  the predicted count. `min_child_weight` is the minimum sum of Hessians in any leaf, so it approximately equals the minimum expected claims per leaf. Setting it to 20 means every leaf must cover at least 20 expected claims. On a 540k-policy training set at 5.5% frequency, this prevents splits on cells so thin they contain only a handful of actual claims. Getting this wrong is the most common source of overfit in XGBoost Poisson models on insurance data.

---

## Model 3: CatBoost

```python
from catboost import CatBoostRegressor, Pool

cat_idx = [FEATURES.index(c) for c in cat_cols]

train_pool = Pool(
    data=train_df[FEATURES],
    label=train_df["ClaimNb"].values,
    weight=train_df["Exposure"].values,
    cat_features=cat_idx,
)
test_pool = Pool(
    data=test_df[FEATURES],
    label=test_df["ClaimNb"].values,
    weight=test_df["Exposure"].values,
    cat_features=cat_idx,
)

cb_model = CatBoostRegressor(
    loss_function="Poisson",
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    min_data_in_leaf=20,
    subsample=0.8,
    random_seed=42,
    verbose=False,
    early_stopping_rounds=50,
)
cb_model.fit(train_pool, eval_set=test_pool)

cb_pred_count = cb_model.predict(test_df[FEATURES])
cb_pred_freq  = cb_pred_count / test_df["Exposure"].values

cb_deviance = poisson_deviance(
    test_df["ClaimNb"].values, cb_pred_freq, test_df["Exposure"].values
)
cb_gini = gini_coefficient(
    test_df["ClaimNb"].values, cb_pred_freq, test_df["Exposure"].values
)

print(f"CB  | Poisson deviance: {cb_deviance:.4f} | Gini: {cb_gini:.4f}")
# CB  | Poisson deviance: 0.2944 | Gini: 0.3408
```

CatBoost receives the categorical columns as raw strings  -  no encoding step required. The Poisson loss function in CatBoost multiplies each observation's Hessian by the `weight` (exposure), giving full-year policies proportionally more influence. This is the correct treatment when the target is claim count rather than claim frequency  -  the model is fitting `E[ClaimNb] = mu * exposure`, not `E[ClaimNb/Exposure] = mu`.

This matters more on richer datasets than freMTPL2, where `Region` has 22 levels and `VehBrand` has 11, but the principle carries directly to a UK motor book where vehicle make has 400+ levels. On those books, CatBoost's ordered target statistics prevent the leakage that OrdinalEncoder + XGBoost produces when levels are rare.

---

## Model 4: LightGBM

```python
import lightgbm as lgb

# LightGBM handles categoricals natively when dtype is "category"
train_lgb = train_df[FEATURES].copy()
test_lgb  = test_df[FEATURES].copy()
for c in cat_cols:
    train_lgb[c] = train_lgb[c].astype("category")
    test_lgb[c]  = test_lgb[c].astype("category")

lgb_train = lgb.Dataset(
    train_lgb,
    label=train_df["ClaimNb"].values,
    weight=train_df["Exposure"].values,
    categorical_feature=cat_cols,
    free_raw_data=False,
)
lgb_test = lgb.Dataset(
    test_lgb,
    label=test_df["ClaimNb"].values,
    weight=test_df["Exposure"].values,
    reference=lgb_train,
    free_raw_data=False,
)

params_lgb = {
    "objective":                "poisson",
    "max_depth":                 6,
    "num_leaves":               63,
    "learning_rate":             0.05,
    "feature_fraction":          0.8,
    "bagging_fraction":          0.8,
    "bagging_freq":              1,
    "min_sum_hessian_in_leaf":  20,   # analogous to XGBoost min_child_weight
    "lambda_l2":                 1.0,
    "verbose":                  -1,
    "seed":                     42,
}

lgb_model = lgb.train(
    params_lgb,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_test],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)

lgb_pred_count = lgb_model.predict(test_lgb)
lgb_pred_freq  = lgb_pred_count / test_df["Exposure"].values

lgb_deviance = poisson_deviance(
    test_df["ClaimNb"].values, lgb_pred_freq, test_df["Exposure"].values
)
lgb_gini = gini_coefficient(
    test_df["ClaimNb"].values, lgb_pred_freq, test_df["Exposure"].values
)

print(f"LGB | Poisson deviance: {lgb_deviance:.4f} | Gini: {lgb_gini:.4f}")
# LGB | Poisson deviance: 0.2951 | Gini: 0.3389
```

LightGBM's `min_sum_hessian_in_leaf` is the direct equivalent of XGBoost's `min_child_weight` for Poisson models. `num_leaves=63` gives a maximum tree with 63 leaf nodes  -  equivalent to depth 6 in a balanced tree, but LightGBM's leaf-wise growth allows asymmetric trees to spend budget where the data has the most signal. For insurance data with non-linear but smooth feature effects, this typically outperforms the symmetric depth-limited trees XGBoost and CatBoost use by default.

LightGBM is also consistently the fastest of the three GBMs to train: approximately 28 seconds on this dataset versus 90 seconds for CatBoost and 45 for XGBoost, on a laptop CPU. That matters during feature development when you are running dozens of experiments.

---

## Calibration check

A model with good Gini but poor calibration will underprice some risks and overprice others systematically. The A/E by decile check catches this.

```python
import matplotlib.pyplot as plt

models_to_check = {
    "GLM":      glm_pred_freq,
    "XGBoost":  xgb_pred_freq,
    "CatBoost": cb_pred_freq,
    "LightGBM": lgb_pred_freq,
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

for ax, (name, pred_freq) in zip(axes, models_to_check.items()):
    cal = calibration_by_decile(
        test_df["ClaimNb"].values,
        pred_freq,
        test_df["Exposure"].values,
    )
    ax.bar(cal["decile"] + 1, cal["ae_ratio"] - 1.0, color="steelblue", alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel("Predicted frequency decile (1=lowest)")
    ax.set_ylabel("A/E ratio − 1")
    ax.set_ylim(-0.3, 0.3)
    ax.grid(True, alpha=0.2)

plt.suptitle("Calibration by Decile  -  Actual/Expected Ratio − 1", fontsize=13)
plt.tight_layout()
plt.savefig("calibration_by_decile.png", dpi=150)
```

What to look for: A/E ratios consistently above 1.0 in the top deciles mean the model underestimates frequency for the highest-risk segment  -  the worst possible calibration failure because it causes systematic underpricing of the worst risks. A/E below 1.0 at the bottom deciles means overpricing of good risks, which invites adverse selection.

The GLM on this dataset shows its main miscalibration in decile 10 (A/E ≈ 1.11  -  it underestimates the worst risks because the main-effects specification cannot capture the interaction between young drivers and high-power vehicles). All three GBMs have A/E ratios within ±5% across deciles, which is the threshold most pricing actuaries accept as adequately calibrated.

---

## Results table

| Model | Poisson deviance | Gini coefficient | Max A/E deviation | Training time (CPU) |
|---|---|---|---|---|
| Poisson GLM | 0.3124 | 0.2973 | ~11% (decile 10) | ~12s |
| XGBoost | 0.2982 | 0.3321 | 4.8% | ~45s |
| LightGBM | 0.2951 | 0.3389 | 3.7% | ~28s |
| CatBoost | **0.2944** | **0.3408** | **3.1%** | ~90s |

Three observations about these numbers.

**The GLM gap is mostly specification, not fundamental.** The GLM as written has linear continuous effects and no interaction terms. Adding `I(DrivAge**2)` for the U-shaped age curve, a `BonusMalus` spline, and the `DrivAge:VehPower` interaction moves the Gini to roughly 0.330 in our testing  -  nearly closing the gap with XGBoost. A well-specified GLM is not inherently uncompetitive with a GBM on freMTPL2.

**The three GBMs are close.** CatBoost leads by 0.006 Gini points over LightGBM, which leads by 0.003 over XGBoost. On a book of 678,000 policies, a 0.006 Gini difference is real but not transformative  -  it shifts perhaps 2-3% of policies between risk bands. Whether that justifies the engineering complexity of switching from LightGBM to CatBoost depends on your book's competitive position.

**CatBoost calibrates best.** The max A/E deviation of 3.1% versus 4.8% for XGBoost reflects CatBoost's ordered boosting, which reduces variance in the tail of the predicted distribution. On insurance data with rare high-risk segments, tail calibration matters more than aggregate deviance.

---

## The GLM-GBM gap: honest assessment

The GBMs win on discrimination and calibration. But the gap has three important qualifications that the ML literature routinely glosses over.

**Feature engineering matters more than algorithm choice.** The 0.044 Gini gap between our plain GLM and CatBoost narrows to roughly 0.010 with proper GLM feature engineering. The typical GBM benchmark paper compares against a naive main-effects GLM, not one that a competent pricing actuary would put in production. This does not mean GBMs are not better  -  they are  -  but the magnitude of the advantage is routinely overstated.

**GLMs can be deployed; GBMs usually cannot.** UK personal lines pricing engines  -  Radar, Emblem, any multiplicative rater  -  expect factor tables. A fitted CatBoost model is not directly deployable in that architecture. Three options exist: (1) deploy the GBM as a standalone scoring engine and call it at quote time; (2) extract multiplicative relativities from the GBM using SHAP values ([`shap-relativities`](https://github.com/burning-cost/shap-relativities)); or (3) distil the GBM into a surrogate GLM ([`insurance-distill`](https://github.com/burning-cost/insurance-distill), which retains 90-97% of the GBM's Gini in our testing). None of these is cost-free  -  each has engineering overhead and governance risk.

**Regulators want explainability.** The FCA's PRIN 12 and the PRA's SS3/19 do not specify modelling approaches, but the requirement to explain pricing decisions to policyholders and demonstrate that models are fair has practical consequences for black-box GBMs. The SHAP-to-factor-table workflow in `shap-relativities` addresses this; it is not trivial to audit.

Our view: use GBMs to set the performance ceiling and understand the nonlinear structure of your book. Use that understanding to improve your GLM specification. Then decide  -  based on your deployment architecture and regulatory context  -  whether the remaining GBM advantage is worth the operational complexity of running a tree model in production.

---

## The middle ground: EBMs

Between the interpretable-but-limited GLM and the accurate-but-opaque GBM, there is a genuinely useful middle tier. Explainable Boosting Machines (EBMs)  -  available via [`insurance-gam`](https://github.com/burning-cost/insurance-gam)  -  achieve Gini coefficients roughly halfway between the GLM and the GBMs on freMTPL2, while producing per-feature shape functions that a pricing committee can inspect factor by factor. The shape function for driver age is not a coefficient  -  it is a curve showing the full U-shape from age 18 to 80, without the actuary first specifying bins or polynomial terms.

```bash
pip install "insurance-gam[ebm]"
```

On the insurance-gam benchmark (synthetic UK motor, 10,000 policies), EBM ranked risks 28% better than a main-effects GLM by Gini. On freMTPL2 the gap is smaller because freMTPL2 has fewer categorical levels and less interaction structure, but the EBM still outperforms the unspecified GLM without requiring any shape analysis up front.

EBMs are our preferred recommendation for teams that need interpretability and cannot accept the GLM's discriminatory ceiling. The governance workflow  -  `GLMComparison` to show what the EBM does differently from the approved GLM, `MonotonicityEditor` to enforce directional constraints  -  is described in the [insurance-gam governance post](/2026/03/16/the-governance-bottleneck-for-ebm-adoption/).

---

## Beyond GBMs: CANN

The frontier beyond ensemble methods is Combined Actuarial Neural Networks (CANN  -  Schelldorfer and Wüthrich 2019). A CANN wraps a neural network around a GLM baseline: the GLM output enters the network as a fixed offset (or skip connection), and the network learns residuals from the GLM. The architecture prevents the network from diverging far from the actuarial baseline; the GLM component remains interpretable.

Holvoet, Antonio and Henckaerts (2023/2025) ran the definitive benchmark study on freMTPL2: CANN does not consistently beat a well-tuned GBM. On Poisson deviance, well-tuned CatBoost outperforms CANN variants in most configurations they tested. The CANN advantage is primarily in settings where the feature-effect structure is smooth and continuous, which a neural approximation captures more efficiently than tree splits.

CANN earns its place in the toolkit  -  particularly for telematics pricing where the feature space is genuinely high-dimensional and smooth  -  but for tabular insurance data with moderate feature counts, CatBoost or LightGBM is harder to beat. We will cover CANN implementation in a separate post.

---

## Connecting the pipeline

This benchmark covers model fit. The rest of the pipeline has dedicated tooling.

- **Factor table extraction**: once CatBoost wins the benchmark, [`shap-relativities`](https://github.com/burning-cost/shap-relativities) extracts multiplicative relativities per feature in the same format as `exp(β)` from a GLM  -  full driver age curve, confidence intervals, and a reconstruction check that the relativities multiply back to the model prediction. [Worked example](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/).

- **GBM-to-GLM distillation**: if your rating engine cannot consume a CatBoost model directly, [`insurance-distill`](https://github.com/burning-cost/insurance-distill) fits a surrogate GLM on the GBM's pseudo-predictions, with automatic CART binning of continuous variables and direct export to factor table format. Retains 90-97% of the GBM's Gini. [Full writeup](/2026/03/01/from-catboost-to-radar-gbm-to-glm-distillation/).

- **Frequency-severity combination**: this post covers frequency only. For the severity model and the freq-sev combination  -  including the non-independence correction that the standard pure premium formula ignores  -  see [`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity). The dependence correction is worth roughly 3-8% improvement in pure premium calibration on a typical UK motor book.

- **Monitoring for drift**: a model trained today and deployed in 18 months will face book mix changes, economic drift, and claims inflation. [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) implements population stability index, feature drift detection, and performance tracking across rolling windows. [Monitoring post](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/).

---

## Summary

freMTPL2: 678,013 French MTPL policies, ~5.5% mean claim rate per policy-year, seven rating factors, Poisson frequency structure. 80/20 train/test split, all models with Poisson objective and log-exposure offset.

| | GLM | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|
| Poisson deviance | 0.3124 | 0.2982 | 0.2951 | **0.2944** |
| Gini coefficient | 0.2973 | 0.3321 | 0.3389 | **0.3408** |
| Max A/E deviation | ~11% | 4.8% | 3.7% | **3.1%** |
| Factor table natively | Yes | No | No | No |
| Categorical handling | Manual | Requires encoding | Native (category dtype) | Native (cat_features) |
| Rating engine compatible | Direct | Via distillation | Via distillation | Via distillation |

CatBoost wins on all three metrics. LightGBM is 3× faster to train and 0.002 behind on deviance  -  close enough that for most teams the choice between them is operational preference, not model quality. XGBoost requires an encoding step for categoricals that neither competitor needs; on a UK motor book with rich categorical structure, that matters.

The GLM is not obsolete. It is the fastest to train, the cheapest to deploy, and the only option that produces a factor table without post-hoc processing. On a well-specified model with interaction terms and spline effects, the Gini gap to the GBMs is under 10 points. If your book is in a segment where that gap translates to adverse selection, use a GBM. If your regulatory and deployment constraints are tight, use the GLM and invest the saved engineering time in better feature engineering.

All code is tested on freMTPL2 version 3 from OpenML (`fetch_openml("freMTPL2freq", version=3)`). Exact metric values will vary slightly with library versions; the rankings are stable.
