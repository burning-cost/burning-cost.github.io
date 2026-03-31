---
layout: post
title: "Explainable Boosting Machines for Insurance Pricing — An Interpretable Alternative That Actually Works"
seo_title: "Explainable Boosting Machines for Insurance Pricing: Near-XGBoost Performance, Full Interpretability"
date: 2026-03-31
categories: [interpretability]
tags: [EBM, explainable-boosting-machine, interpretML, insurance-pricing, GLM, GAM, XGBoost, CatBoost, Poisson-deviance, shape-functions, relativities, EU-AI-Act, transparency, freMTPL, Krupova-2025, insurance-gam, python, monotonicity, motor-insurance, actuarial]
description: "EBMs achieve near-XGBoost predictive performance on insurance claims data while remaining fully interpretable by design — no post-hoc SHAP required. We show the Poisson frequency workflow, relativities extraction, and the honest comparison with GLMs and CatBoost+SHAP."
author: burning-cost
---

The standard ML interpretability playbook runs like this: fit a black-box GBM, run SHAP to explain it, present the waterfall plots to the pricing committee, move on. The problem is that SHAP explanations are not the model. They are an approximation of the model's behaviour derived from cooperative game theory. The thing you present to Lloyd's is not the thing you priced with. For EU AI Act purposes, that distinction is not cosmetic — Article 13 requires that high-risk AI systems be transparent and interpretable *as deployed*, not as approximated by a secondary tool.

Explainable Boosting Machines (EBMs) sidestep the problem entirely. Interpretability is intrinsic to the model structure, not bolted on afterwards. And a paper published in March 2025 by Krupová, Rachdi, and Guibert at Paris Dauphine (arXiv:2503.21321) confirms what other benchmarks have suggested: EBMs with pairwise interactions achieve near-XGBoost performance on insurance claims data, and clearly outperform GLMs, GAMs, and CART on both frequency and severity tasks.

This post explains how EBMs work, shows the frequency modelling workflow with real code, and gives an honest account of when to use EBM versus a GLM versus CatBoost.

---

## What an EBM actually is

EBMs are a specific instantiation of Generalised Additive Models with pairwise interactions (GA²M), introduced by Lou, Caruana, Gehrke, and Hooker at KDD 2013. The model takes the form:

```
g(E[Y]) = β₀ + Σ fᵢ(xᵢ) + Σᵢ<ⱼ fᵢⱼ(xᵢ, xⱼ)
```

Each fᵢ is a univariate shape function — a piecewise-constant lookup table learned by gradient boosting, applied in cyclic round-robin fashion across features. The pairwise terms fᵢⱼ are 2-D lookup tables selected automatically using the FAST algorithm, which evaluates all candidate pairs on the residuals of the main-effects model and picks the k pairs with the highest residual R² improvement.

The key property: every shape function is a direct object in the model. You can read off fᵢ(xᵢ) as a table of scores, convert to relativities with exp(score − base_score), and show it to an actuary in exactly the format they expect from a GLM factor analysis. There is no approximation. That *is* the model.

InterpretML (Microsoft Research, open source) has supported Poisson deviance and Gamma deviance since version 0.7, so the main-effects and interaction structure maps directly onto the actuarial frequency-severity framework without modification.

---

## Fitting a frequency model

The freMTPL2 dataset from the R CASdatasets package is the standard European motor benchmark: 679,013 policies, claims count as target, exposure in earned car years. The target is claim frequency, which follows a Poisson distribution with mean proportional to exposure.

```python
import polars as pl
import numpy as np
from insurance_gam.ebm import InsuranceEBM

# Load freMTPL2 frequency data (standard benchmarking dataset)
df = pl.read_csv("freMTPL2freq.csv")

features = [
    "Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
    "VehBrand", "VehGas", "Density", "Region",
]
X = df.select(features)
y = df["ClaimNb"].to_numpy()
exposure = df["Exposure"].to_numpy()

# 80/20 train/test split by policy
n = len(df)
rng = np.random.default_rng(42)
idx = rng.permutation(n)
train_idx, test_idx = idx[: int(0.8 * n)], idx[int(0.8 * n) :]

X_train = X[train_idx]
X_test  = X[test_idx]
y_train = y[train_idx]
y_test  = y[test_idx]
exp_train = exposure[train_idx]
exp_test  = exposure[test_idx]

# Fit EBM with Poisson deviance, pairwise interactions, 5 bagging rounds
model = InsuranceEBM(
    loss="poisson",
    interactions="3x",        # auto-select top 3×n_features pairs
    outer_bags=5,
    inner_bags=0,
    learning_rate=0.05,
    max_bins=256,
)
model.fit(X_train, y_train, exposure=exp_train)
```

The `exposure` parameter is passed as a log offset — internally this sets `init_score=np.log(exposure)` before the first boosting round, which is the correct actuarial treatment. Passing `sample_weight=exposure` instead is a common mistake: it reweights the deviance contribution of each policy rather than shifting the rate baseline, which produces a biased intercept.

For an out-of-sample check:

```python
from insurance_gam.ebm._diagnostics import deviance

y_pred_test = model.predict(X_test, exposure=exp_test)
d = deviance(y_test, y_pred_test, loss="poisson", weights=exp_test)
print(f"Out-of-sample Poisson deviance: {d:.4f}")
```

---

## Extracting relativities

This is where EBMs earn their place in an actuarial workflow. The shape function for each feature is a direct object in the model — no SHAP, no approximate attribution, no post-hoc processing. We wrap the extraction in `RelativitiesTable`, which follows the standard actuarial convention: the base level is the modal bin (highest training weight), and all other bins are expressed as multipliers relative to it.

```python
from insurance_gam.ebm import RelativitiesTable

rt = RelativitiesTable(model)

# Relativity table for driver age
age_table = rt.table("DrivAge")
print(age_table)
# ┌────────────────┬───────────┬────────────┐
# │ bin_label      ┆ raw_score ┆ relativity │
# │ ---            ┆ ---       ┆ ---        │
# │ str            ┆ f64       ┆ f64        │
# ╞════════════════╪═══════════╪════════════╡
# │ (17.0, 22.0]   ┆  0.4821   ┆ 1.6197     │
# │ (22.0, 26.0]   ┆  0.2134   ┆ 1.2380     │
# │ (26.0, 31.0]   ┆  0.0000   ┆ 1.0000     │ ← base
# │ (31.0, 42.0]   ┆ -0.0891   ┆ 0.9147     │
# │ (42.0, 60.0]   ┆ -0.1203   ┆ 0.8867     │
# │ (60.0, 75.0]   ┆  0.0312   ┆ 1.0317     │
# │ (75.0, 100.0]  ┆  0.1894   ┆ 1.2087     │
# └────────────────┴───────────┴────────────┘

# Export all features to Excel (one sheet per feature)
rt.export_excel("frequency_relativities.xlsx")

# Or visualise as a bar chart
rt.plot("DrivAge", kind="line")
```

The relativity table for driver age is directly comparable to what a GLM would produce. The shape function captures the U-curve — young drivers at 1.62×, the safest cohort (26–31) at base, a modest recovery for older drivers reflecting the classic mobility-reduction effect. A GLM would need explicit polynomial terms or grouped factor levels to approximate this shape; EBM discovers it automatically and stores it exactly.

The `export_excel()` call writes a workbook with one sheet per main-effect feature. This is the format pricing committees actually use. Interaction terms have 2-D score arrays and don't flatten into a single relativity table; a heatmap is the appropriate visualisation for those.

---

## The Krupová et al. benchmark

The Paris Dauphine paper (arXiv:2503.21321, March 2025) is the most thorough published benchmark of EBMs on insurance data. Using a proprietary French dataset with roughly 5 million frequency observations and 100,000 severity observations from ADDACTIS France, they find:

- **Frequency (Poisson deviance)**: EBM with interactions achieves near-XGBoost performance. GLM and GAM are clearly below both.
- **Severity (Gamma deviance)**: Same pattern. EBM with interactions ≈ XGBoost. CART is well behind.
- **FAST interaction selection**: The built-in algorithm correctly identifies vehicle-age × driver-age and bonus-malus × driver-age as the dominant interaction pairs — both consistent with actuarial expectation.

Worth being precise about what the paper contributes: it is an empirical validation study, not an algorithmic advance. InterpretML already supported Poisson and Gamma deviance before the paper was written. The paper's value is confirming, on a large proprietary dataset with proper actuarial benchmarking, that EBM is competitive with XGBoost and substantially better than GLM. The insurance-gam library already implements a more complete actuarial workflow than anything described in the paper — exposure offset handling, relativities extraction, monotonicity enforcement, diagnostics — but the benchmark result is a credible external validation.

Multi-dataset benchmarks from CAS E-Forum 2024 covering 20 insurance datasets (across North American, European, and UK portfolios) report average performance ranks: CatBoost at 2.93, EBM at 4.08, GLM at 6.14. The EBM–CatBoost gap is typically under 2% relative Poisson deviance. The GLM–EBM gap is typically 5–15% depending on dataset non-linearity.

---

## EBM vs GLM vs CatBoost+SHAP — when to use each

Here is our position, stated directly:

**Use EBM** when you need ML-level performance and full interpretability. This is the default recommendation for UK personal lines pricing teams building frequency or severity models from scratch. The shape functions are natively auditable, EU AI Act Article 13 compliance is straightforward (you can produce a complete transparency report from the model objects without any secondary explanation tool), and the relativities output matches the format that Lloyd's, the FCA, and pricing committees expect. The [insurance-gam](https://github.com/burning-cost/insurance-gam) library provides the full workflow including exposure offset, monotonicity enforcement, and Excel export.

**Use a GLM** when regulatory precedent is the binding constraint — for instance, a Lloyd's model approval process where the syndicate governance requires an established actuarial standard. GLMs have a 30-year track record, satisfy the balance property by construction (MLE + canonical link forces Σŷ = Σy on training data), and are understood by every pricing actuary in the market. When the regulatory environment demands GLM, use GLM. EBMs do not yet have that precedent in the UK Lloyd's market.

**Use CatBoost+SHAP** when predictive performance is the only objective and interpretability is genuinely secondary — internal risk ranking, ML scoring for underwriting triage, research contexts. CatBoost leads multi-dataset benchmarks (mean rank 2.93 vs EBM's 4.08) and the margin is real even if small. But SHAP explanations are an approximation of the model, not the model itself. The TreeSHAP values for a CatBoost ensemble are exact for that ensemble — but the ensemble is a black box. Under EU AI Act Annex III, insurance risk assessment and pricing is a high-risk AI application. Presenting SHAP as your Article 13 transparency mechanism will not survive regulatory scrutiny, and the FCA's Consumer Duty framework has a similar flavour of expectation.

The "both approaches have merits" answer is not useful here. EBM is the right default for regulated pricing. CatBoost is right for unregulated ranking tasks. GLM is right when the regulatory process requires it specifically.

---

## One gap to be honest about

EBMs do not automatically satisfy the balance property. GLMs do — under a canonical link (log + Poisson), the MLE score equations force the sum of predictions to equal the sum of observed claims on training data. EBM is fitted by cyclic gradient boosting minimising Poisson deviance, which does not enforce those score equations. In practice the model will be close to balanced, but regularisation (bagging, smoothing) introduces small global bias.

The fix is straightforward:

```python
# Check balance ratio on training data
y_pred_train = model.predict(X_train, exposure=exp_train)
balance_ratio = y_train.sum() / y_pred_train.sum()
print(f"Balance ratio: {balance_ratio:.4f}")  # should be close to 1.0

# Apply log-scale correction to the model intercept if needed
if abs(balance_ratio - 1.0) > 0.005:
    model.ebm_.intercept_ += np.log(balance_ratio)
    print("Intercept corrected for global balance.")
```

This adjusts the scalar intercept in the InterpretML model object directly — no shape functions are changed, and all relativities remain valid. It is one line of code and it gives you the same guarantee that a GLM provides by construction. We have this on the roadmap for a future `balance_check()` function in insurance-gam.

---

## The EU AI Act angle

EU AI Act Annex III classifies insurance risk assessment and pricing as high-risk AI. Article 13 requires that high-risk systems provide sufficient transparency for users to interpret outputs, and that technical documentation covers performance metrics, training data, known limitations, and human oversight requirements.

For a CatBoost+SHAP model, compiling that documentation requires assembling information from two separate objects: the model (which you cannot directly inspect) and the SHAP explainer (which approximates the model). The transparency obligation is about the *deployed system*, and the deployed system is CatBoost, not SHAP.

For an EBM, every piece of Article 13-relevant information is directly readable from the model object: the shape function for each feature (the model's representation of that variable's effect), the selected interaction pairs (explicitly stored as 2-D lookup tables), performance metrics computed from the same object, and monotonicity constraints as metadata. The transparency report writes itself. The compliance gap between EBM and CatBoost+SHAP under EU AI Act is structural, not a matter of documentation effort.

The Act's obligations apply from August 2026 for most high-risk applications. Pricing teams building new models now should be building for that deadline.

---

## Getting started

Install:

```bash
pip install insurance-gam
```

The `insurance_gam.ebm` subpackage wraps InterpretML's `ExplainableBoostingRegressor` with actuarial conventions: exposure as a log offset, Polars DataFrame support, relativities extraction, GLM comparison, and monotonicity enforcement. The `insurance_gam.ebm._comparison.GLMComparison` class runs a direct deviance comparison between the EBM and a Poisson GLM on the same data, which is useful for internal model governance sign-off.

For the full benchmark workflow — including the FAST interaction pair analysis and the A/E diagnostics that Krupová et al. use — see the [insurance-gam documentation](https://github.com/burning-cost/insurance-gam/tree/main/docs) and the freMTPL2 example notebook.

---

*Krupová, Rachdi, Guibert (2025) "Explainable Boosting Machine for Predicting Claim Severity and Frequency in Car Insurance", arXiv:2503.21321, HAL hal-05007738. Lou, Caruana, Gehrke, Hooker (2013) "Accurate Intelligible Models with Pairwise Interactions", KDD 2013.*
