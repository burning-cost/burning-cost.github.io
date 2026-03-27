---
layout: post
title: "When You Can't Fit a GLM from Scratch: Transfer Learning for Thin Segments"
date: 2026-03-10
categories: [pricing, libraries, tutorials]
tags: [transfer-learning, thin-data, GLMTransfer, MMD, covariate-shift, negative-transfer, poisson-glm, insurance-thin-data, uk-motor, electric-vehicles, python, JASA, Tian-Feng]
description: "GLMTransfer borrows statistical strength from a related source book to price thin target segments. Motor-to-fleet, home-to-landlord, and fleet roll-outs."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-transfer` has been consolidated into [`insurance-thin-data`](https://pypi.org/project/insurance-thin-data/). For credibility-based thin-segment blending, see [insurance-credibility](/insurance-credibility/). Install with `pip install insurance-thin-data` — all functionality described here is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-thin-data)
</div>


Every UK motor pricing team has at least one segment they are pricing badly. Not because the actuaries are incompetent. Because the segment is too thin to estimate reliably on its own, and nobody has found a satisfying method for borrowing strength from elsewhere.

The usual options are: blend toward the book mean using credibility weights (defensible but primitive), use the same factors as the nearest analogue (usually another vehicle class or channel, justified by analogy rather than data), or accept the wide confidence intervals and apply a manual loading. None of these is wrong, exactly. None of them is principled.

The problem is concrete. Suppose your motor book has 80,000 policies. Of those, 1,200 are battery electric vehicles - mostly added in the last two years as you started competing on EV motor. That is enough data to notice that EVs have a different frequency distribution than ICE equivalents. It is not enough to estimate stable GLM coefficients for age, vehicle age, postcode sector, and NCD simultaneously. The MLE will fit the noise.

What you actually have is 80,000 policies in a related book with a mature, well-estimated set of GLM coefficients. The question is whether you can borrow from those coefficients in a way that is statistically honest - not just applying the ICE factors to the EV segment and calling it a day, but explicitly modelling how much the EV segment deviates from the source.

[`insurance-thin-data`](https://github.com/burning-cost/insurance-thin-data) implements this via `GLMTransfer`, which is the Tian and Feng (JASA, 2023) two-step penalised GLM method adapted for the insurance workflow. The short version: pool source and target data to get a stable starting point, then fine-tune on target data alone to correct for whatever is genuinely different about the segment.

```bash
uv add insurance-thin-data
```

---

## The correct order of operations

Transfer learning done wrong produces confident nonsense. Borrowing from a source that is genuinely incompatible with your target will shrink your EV coefficients toward the wrong values and reduce your predictive accuracy below what the thin-data GLM would have achieved on its own. This is called negative transfer, and it is a real risk.

The correct order:

1. Run the MMD shift test. Quantify how different the source and target distributions are.
2. Check which features drive the shift. Not all drift is harmful.
3. Fit `GLMTransfer`. The debiasing step handles moderate drift; severe drift may require excluding the source.
4. Run the negative transfer diagnostic. Confirm that borrowing actually helped.

Skip step 1 and you are flying blind.

---

## Step 1: MMD shift test

Maximum Mean Discrepancy (MMD) is a kernel-based measure of how different two distributions are. `CovariateShiftTest` computes it with a permutation test to give you a p-value that does not rely on distributional assumptions. The kernel is mixed - RBF for continuous features, indicator kernel for categorical - which matters because insurance feature matrices contain both.

```python
import numpy as np
from insurance_thin_data.transfer import CovariateShiftTest

# X_source: features for ICE policies (n=80000 after removing EVs)
# X_target: features for EV policies (n=1200)
# Both matrices: same columns, same encoding
# Features: driver_age, vehicle_age, ncd_years, postcode_density,
#           annual_mileage, vehicle_group (integer-encoded), fuel_type_flag

cat_cols = [5, 6]  # vehicle_group, fuel_type_flag are categorical

shift_test = CovariateShiftTest(
    categorical_cols=cat_cols,
    n_permutations=1000,
    random_state=42,
)

result = shift_test.test(X_source, X_target)
print(result)
```

```
ShiftTestResult(MMD²=0.0312, p=0.003 [significant],
n_source=80000, n_target=1200)
```

The p-value of 0.003 confirms significant distributional shift - not surprising given that EV drivers are systematically younger, have higher annual mileage, and skew toward urban postcodes relative to the ICE book. Significant MMD does not mean transfer is impossible. It means you need the debiasing step to work properly, and you need to check which features are driving the divergence.

```python
top_drifted = shift_test.most_drifted_features(result, top_n=5)
feature_names = [
    "driver_age", "vehicle_age", "ncd_years",
    "postcode_density", "annual_mileage", "vehicle_group", "fuel_type_flag"
]

for idx, score in top_drifted:
    print(f"  {feature_names[idx]}: MMD² = {score:.4f}")
```

```
  fuel_type_flag: MMD² = 0.0198
  annual_mileage: MMD² = 0.0091
  vehicle_age: MMD² = 0.0073
  postcode_density: MMD² = 0.0041
  driver_age: MMD² = 0.0029
```

`fuel_type_flag` is trivially different - the target is entirely EV. That is definitional, not a problem. `annual_mileage` and `vehicle_age` are the features to watch: if the GLM coefficient for annual mileage turns out very different between source and target, the pooled estimate will be pulled toward the wrong value and the debiasing step will need to work hard to correct it.

---

## Step 2: fit GLMTransfer

`GLMTransfer` is a two-step algorithm from Tian and Feng (JASA, 2023).

**Pooling step:** Pool source and target data. Fit an L1-penalised GLM. The large source sample dominates, giving you stable coefficient estimates. The penalty prevents overfitting to the combined dataset.

**Debiasing step:** Fit the *difference* between the target-optimal coefficients and the pooled coefficients, using target data only, with its own L1 penalty. Only meaningful differences survive the penalty. If the EV segment genuinely has a different mileage effect, that difference will be estimated and applied. If the age effect is identical between segments, the debiasing step will shrink that delta to zero and borrow the pooled estimate fully.

The final coefficients are pooled + delta - a data-driven blend that is simultaneously informed by the large source sample and corrected for systematic differences.

```python
from insurance_thin_data.transfer import GLMTransfer

model = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,   # regularisation in pooling step
    lambda_debias=0.02,  # regularisation in debiasing step
    scale_features=True,
    fit_intercept=True,
)

model.fit(
    X_target, y_target, exposure_target,
    X_source=X_source,
    y_source=y_source,
    exposure_source=exposure_source,
)
```

The `lambda_pool` and `lambda_debias` hyperparameters control the degree of regularisation. With thin target data, `lambda_debias` should be set conservatively - a larger value keeps the delta small and leans more heavily on the source. Selecting both requires a manual cross-validation loop: sklearn's `cross_val_score` cannot pass the source data through its fold machinery cleanly, so write the loop directly.

```python
import numpy as np
from sklearn.model_selection import KFold

def poisson_deviance(y_true, y_pred):
    y_pred = np.maximum(y_pred, 1e-10)
    ratio = y_true / y_pred
    return 2.0 * np.mean(
        np.where(y_true > 0, y_true * np.log(ratio) - (y_true - y_pred), y_pred - y_true)
    )

kf = KFold(n_splits=5, shuffle=True, random_state=42)
deviances = []

for train_idx, val_idx in kf.split(X_target):
    m = GLMTransfer(family="poisson", lambda_pool=0.005, lambda_debias=0.02, scale_features=True)
    m.fit(
        X_target[train_idx], y_target[train_idx], exposure_target[train_idx],
        X_source=X_source, y_source=y_source, exposure_source=exposure_source,
    )
    preds = m.predict(X_target[val_idx], exposure_target[val_idx])
    deviances.append(poisson_deviance(y_target[val_idx], preds))

print(f"CV Poisson deviance: {np.mean(deviances):.4f} ± {np.std(deviances):.4f}")
```

```
CV Poisson deviance: 0.3821 ± 0.0412
```

For comparison, a target-only Poisson GLM on the same 1,200 policies fits at 0.4203 ± 0.0681. The transfer model reduces deviance by roughly 9% and also reduces the cross-fold standard deviation - a direct reflection of the coefficient stability that the pooling step buys you.

---

## Step 3: inspect what changed

The interpretability of this approach comes from comparing the pooled coefficients with the debiasing delta. Features where delta is near zero are priced identically in EV and ICE. Features where delta is large are the ones where the EV segment genuinely differs.

```python
import pandas as pd

feature_names_with_intercept = ["intercept"] + [
    "driver_age", "vehicle_age", "ncd_years",
    "postcode_density", "annual_mileage", "vehicle_group", "fuel_type_flag"
]

coef_df = pd.DataFrame({
    "feature": feature_names_with_intercept,
    "beta_pooled": model.beta_pooled_,
    "delta": model.delta_,
    "beta_final": model.beta_pooled_ + model.delta_,
}).set_index("feature")

print(coef_df.round(4))
```

```
                  beta_pooled    delta  beta_final
feature
intercept             -2.8103   0.1241     -2.6862
driver_age             0.0312   0.0021      0.0333
vehicle_age           -0.0088  -0.0412     -0.0500
ncd_years             -0.0821  -0.0019     -0.0840
postcode_density       0.0094   0.0038      0.0132
annual_mileage         0.0281   0.0190      0.0471
vehicle_group          0.1144  -0.0003      0.1141
fuel_type_flag        -0.0073   0.0000     -0.0073
```

Two findings stand out. `vehicle_age` has a large negative delta: EVs depreciate more predictably, and older EVs face higher repair costs from battery degradation - a different relationship than ICE vehicles where age has a weaker and less consistent effect on frequency. `annual_mileage` also shows a meaningful positive delta: EV drivers tend to be higher-mileage commuters, and the mileage-frequency relationship is steeper in this segment than in the general book.

The age and NCD effects are largely inherited from the source. That is the correct result. There is no strong a priori reason why age or NCD should work differently for EV versus ICE motor risk; the data confirms this and the algorithm borrows those coefficients wholesale.

---

## Step 4: negative transfer diagnostic

Never deploy without running this.

```python
from insurance_thin_data.transfer import NegativeTransferDiagnostic

# Split target into train/test before fitting
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te, exp_tr, exp_te = train_test_split(
    X_target, y_target, exposure_target,
    test_size=0.25, random_state=42,
)

diag = NegativeTransferDiagnostic()
result = diag.run(
    X_test=X_te,
    y_test=y_te,
    exposure_test=exp_te,
    transfer_model=model,
    X_train=X_tr,
    y_train=y_tr,
    exposure_train=exp_tr,
    X_source=X_source,
    y_source=y_source,
    exposure_source=exposure_source,
)

print(result)
```

```
TransferDiagnosticResult(
  transfer=True [beneficial]
  deviance_transfer=0.3714
  deviance_target_only=0.4198
  NTG=-0.0484 (-11.5%)
  n_test=300
)
```

NTG of -0.048 means the transfer model outperforms the target-only GLM by 11.5% on the holdout. Transfer was beneficial. Had the NTG been positive - meaning the transfer model was worse - the right response is to increase `lambda_debias`, which forces the debiasing step to correct more aggressively for source-target mismatch, or to exclude the source entirely and fall back to the regularised target-only model.

A positive NTG on a book with severe MMD shift often means the debiasing regularisation is too weak. The source is being pooled in fully and the delta is too small to correct it. Increase `lambda_debias` until NTG turns negative, or until the gain becomes negligible.

---

## What this does not fix

The Tian-Feng estimator handles the case where the source and target share the same feature set and the same functional form - a Poisson log-linear GLM - but differ in their coefficient values. It is not designed for:

**Structural differences in what features matter.** If EVs require features that ICE vehicles do not (battery health indicator, charge location, overnight parking), those features have no source estimates to borrow from. The debiasing step will estimate their effects from target data alone, which is fine - but that means you get no regularisation benefit from the source for those features. On thin data, that can mean unstable estimates for EV-specific features even when the shared-feature coefficients are well-stabilised.

**Very large shift with few target observations.** Below roughly 200 target policies, the debiasing step is fitting on near-nothing. `lambda_debias` will shrink delta to approximately zero regardless of the true shift. Effectively, you are just applying the pooled model with minimal correction. That may still be better than the standalone GLM on 200 policies - the pooled estimator benefits from the source sample size - but you should not expect the debiasing step to do meaningful work.

**Multiple source populations with different relevance.** If you want to transfer from both UK motor ICE and an Irish EV book simultaneously, set `X_source` as a list of two arrays and use `delta_threshold` to let the algorithm screen out sources that are harmful. The greedy screening will exclude sources where the required debiasing delta is large, keeping only genuinely helpful sources in the pooled step.

---

## Why not just use credibility blending instead

[Bühlmann-Straub credibility](/2026/02/19/buhlmann-straub-credibility-in-python/) is the standard actuarial answer to thin data. It shrinks cell-level estimates toward the grand mean with weights determined by exposure and within-cell variance. It works well when you have many thin cells and you are estimating a scalar mean per cell. It does not handle the full coefficient vector of a multi-factor GLM.

The specific failure mode: credibility blending a scalar frequency per risk cell does not tell you whether the age-frequency relationship is different in the EV segment. It blends the level, not the shape. If EV policyholders are systematically younger and the age effect is steeper for EVs, credibility blending will underprice young EV drivers even after the level correction.

`GLMTransfer` estimates the full coefficient vector including interactions with the segment. For a single-factor segment problem, credibility is fine and simpler to explain. For anything involving differential factor effects across segments, transfer learning is the right framework.

---

`insurance-thin-data` is open source under MIT at [github.com/burning-cost/insurance-thin-data](https://github.com/burning-cost/insurance-thin-data). Install with `uv add insurance-thin-data`. The transfer module requires Python 3.10+, NumPy, SciPy, and scikit-learn; no PyTorch dependency unless you are using the CANN fine-tuning backend.

- [Foundation Models for Thin Segments: TabPFN and TabICLv2 in Insurance Pricing](/2026/03/13/insurance-tabpfn/) - when you have no related source book at all, TabICLv2 in-context learning works directly from the target data
- [Your New Business Mix Changed. Your Model Didn't Notice.](/2026/03/06/channel-mix-drift-your-model-didnt-notice/) - covariate shift detection when the portfolio composition drifts without a new segment being added
- [Your Group Factors Are Not All Worth Modelling](/2026/03/06/multilevel-group-factors/) - ICC diagnostics for deciding which group effects are worth the modelling overhead
