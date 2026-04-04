---
layout: post
title: "What PRA SS1/23 Validation Looks Like on Real Data: 677K French Motor Policies"
date: 2026-03-28
categories: [governance, validation, pricing, techniques]
tags: [model-validation, ss1-23, pra, governance, hosmer-lemeshow, gini, rag-status, fremtpl2, catboost, poisson-glm, model-risk, calibration, discrimination, insurance-governance]
description: "Most governance tooling is tested on toy examples with clean DGPs and inflated Gini coefficients. We ran the full insurance-governance validation suite on 677K freMTPL2 policies — Poisson GLM and CatBoost side by side — and found three things that change how teams should set RAG thresholds."
---

Governance tooling is routinely validated on synthetic data with clean data-generating processes. That is convenient for library development and useless for understanding what the tools will find in production. The Gini is higher, the calibration is cleaner, and the Hosmer-Lemeshow p-values are comfortably above 0.05. Everything looks good. Nothing prepares you for what happens on a real book.

We benchmarked [`insurance-governance`](/insurance-governance/) against freMTPL2 — 677,991 French MTPL policies from OpenML (dataset ID 41214, Noll, Salzmann & Wüthrich, 2020) — running a full SS1/23-aligned validation suite on both a Poisson GLM and a CatBoost GBM. The findings are not about whether the models are good. They are about what real-data validation looks like and what that means for how teams should calibrate their RAG thresholds.

---

## The setup

Two models, same dataset, identical validation settings.

**Model 1 (GLM):** Poisson regression via statsmodels, numeric features only — driver age, vehicle age, vehicle power, BonusMalus, and log-density. No categoricals. This is the kind of model a UK motor team might have been running in 2018.

**Model 2 (GBM):** CatBoost, all nine freMTPL2 features including the four categoricals — vehicle brand, fuel type, area code, and region. This is the current-generation model.

```python
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import statsmodels.api as sm
from catboost import CatBoostRegressor

data = fetch_openml(data_id=41214, as_frame=True)
df = data.frame.copy()

# fetch_openml can return numeric columns as object dtype — coerce explicitly
for col in ["DrivAge", "VehAge", "VehPower", "BonusMalus", "Density", "Exposure", "ClaimNb"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["log_Density"] = np.log1p(df["Density"])
df["ClaimFreq"] = df["ClaimNb"] / df["Exposure"].clip(lower=1e-6)
```

80/20 train/validation split. Validation results drive the `ModelValidationReport`.

```python
# GLM fit
numeric_features = ["DrivAge", "VehAge", "VehPower", "BonusMalus", "log_Density"]
X_glm = sm.add_constant(df_train[numeric_features].astype(float))
offset = np.log(df_train["Exposure"].clip(lower=1e-6).astype(float))

glm = sm.GLM(
    df_train["ClaimNb"],
    X_glm,
    family=sm.families.Poisson(),
    offset=offset,
).fit()

y_pred_glm_val = glm.predict(
    sm.add_constant(df_val[numeric_features].astype(float)),
    offset=np.log(df_val["Exposure"].clip(lower=1e-6)),
)
```

```python
# CatBoost fit — accepts string categoricals natively
cat_features = ["VehBrand", "VehGas", "Area", "Region"]
gbm = CatBoostRegressor(
    loss_function="Poisson",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    random_seed=42,
    verbose=0,
)
gbm.fit(
    df_train[all_features],
    df_train["ClaimFreq"],
    sample_weight=df_train["Exposure"],
    cat_features=cat_features,
)
y_pred_gbm_val = gbm.predict(df_val[all_features]) * df_val["Exposure"]
```

---

## The validation API

`ModelValidationReport` runs the same five-test suite regardless of model type. The model card records the metadata the SS1/23 framework specifies: purpose, methodology, limitations, owner, monitoring frequency, and monitoring triggers. SS1/23 applies formally to banks; for insurers, the equivalent expectation flows from Solvency II Articles 120–126. Either way, none of this metadata is optional in a well-governed pricing function.

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

glm_card = ValidationModelCard(
    name="Poisson GLM — freMTPL2 frequency",
    version="1.0",
    purpose="Estimate claim frequency for French MTPL pricing",
    methodology="Log-linear Poisson GLM with numeric rating factors",
    model_type="GLM",
    target="ClaimNb",
    features=numeric_features,
    limitations="Excludes vehicle brand, fuel type, area, region. No interaction terms.",
    owner="Pricing team",
    development_date="2026-03-28",
    validation_date="2026-03-28",
    validator_name="Burning Cost",
    monitoring_owner="Pricing team",
    monitoring_frequency="Quarterly",
    monitoring_triggers=["A/E > 1.10", "Gini drop > 0.03", "PSI > 0.20"],
)

glm_report = ModelValidationReport(
    model_card=glm_card,
    y_val=df_val["ClaimNb"].values,
    y_pred_val=y_pred_glm_val.values,
    exposure_val=df_val["Exposure"].values,
    y_train=df_train["ClaimNb"].values,
    y_pred_train=y_pred_glm_train.values,
    exposure_train=df_train["Exposure"].values,
    X_train=df_train,
    X_val=df_val,
    segment_col="VehGas",
    monitoring_owner="Pricing team",
    monitoring_triggers=["A/E > 1.10", "Gini drop > 0.03"],
    random_state=42,
)

results = glm_report.run()
rag = glm_report.get_rag_status()
```

`run()` returns a list of `TestResult` objects — one per test in the fixed five-test suite: discrimination (Gini), calibration (A/E ratio), Hosmer-Lemeshow, stability (PSI), and overfitting check. `get_rag_status()` aggregates them into a single `RAGStatus` with `overall`, `discrimination`, `calibration`, `stability`, and `overfitting` fields.

```python
glm_report.generate("/tmp/glm_validation.html")
glm_report.to_json("/tmp/glm_validation.json")
```

The HTML report is the document you attach to the model governance record. The JSON is the machine-readable version that feeds into your monitoring database.

---

## Finding 1: real-world Gini is lower than your synthetic benchmarks

On freMTPL2, the Poisson GLM (numeric features only) achieves a Gini in the 0.15–0.30 range. The CatBoost GBM with all nine features reaches 0.25–0.40. These are representative of what well-specified frequency models achieve on large heterogeneous real portfolios.

Synthetic benchmarks routinely produce Gini coefficients in the 0.40–0.60 range. The reason is not that synthetic models are better — it is that synthetic data-generating processes have clean, low-noise relationships. Real insurance portfolios have substantial irreducible noise: a driver with a safe profile still has accidents, and a driver with a risky profile still has claim-free years. That irreducible noise depresses the Gini ceiling regardless of model quality.

The consequence for RAG threshold calibration is significant. If your governance framework flags discrimination as Amber below 0.35 and Red below 0.20 — thresholds calibrated on synthetic benchmarks — it will over-flag production frequency models. A Gini of 0.22 on a 677K-policy freMTPL2-like book is not evidence that the model is poor. It is consistent with a well-specified model operating near the ceiling of what the data supports.

We think governance teams should calibrate RAG thresholds to their portfolio's own history, not to synthetic benchmarks or to published Gini ranges from academic papers. The `insurance-governance` defaults are a starting point, not the answer. The `ModelValidationReport.get_rag_status(custom_thresholds={"discrimination_green": 0.20, "discrimination_amber": 0.12})` call takes a dict of overrides if you want to document explicitly that you have calibrated to portfolio history.

---

## Finding 2: Hosmer-Lemeshow catches what global A/E misses

This is the most practically important result from the benchmark.

On 677K policies, the H-L test has enough statistical power to detect segment miscalibration that is invisible in the global A/E ratio. The Poisson GLM — which excludes vehicle brand, fuel type, area code, and region — produces a global A/E close to 1.0. The overall calibration looks fine.

H-L disagrees. The H-L test partitions policyholders into deciles by predicted frequency and tests whether the actual claim count in each decile matches the expected count. With 677K observations, even a 3–4% discrepancy in a decile is statistically significant. The GLM without categoricals leaves residuals concentrated in the urban, high-density, young-driver segment — exactly the cohort where area code and region would have improved calibration. That segment averages out against the suburban and rural segments in the global A/E.

```python
for r in results:
    if r.test_name == "hosmer_lemeshow":
        print(f"H-L statistic: {r.statistic:.2f}")
        print(f"H-L p-value:   {r.p_value:.4f}")
        print(f"RAG:           {r.rag.upper()}")
        if r.details:
            print(r.details)
```

The GBM, which includes area and region, passes H-L comfortably. The delta between GLM and GBM on H-L — not on Gini — is the most useful diagnostic for deciding whether adding categorical features to a pricing model is worth the governance overhead.

This is why H-L belongs in the fixed five-test suite, not as an optional extra. A model validation that only reports global A/E and Gini is not telling you whether the model is well-calibrated across the rating structure. It is telling you whether the errors average out.

---

## Finding 3: PSI on a random split tells you nothing

freMTPL2 has no date field. The only available split is random. On a random split, PSI between training and validation is near-zero by construction — the two samples are drawn from the same distribution. The stability test passes cleanly.

This is the expected result. It is not a model quality signal. In production, where your training data is from 2023 and your validation data is from 2025, PSI will be materially higher — particularly on features like BonusMalus (which accumulates with the claim cycle), average age (which shifts as the book matures), and vehicle age (which trends as the fleet ages).

```python
for r in results:
    if r.test_name == "stability_psi":
        print(f"PSI score: {r.statistic:.4f}  [{r.rag.upper()}]")
        # On a random split, expect GREEN (~0.01-0.03)
```

We flag this because teams sometimes mistake "PSI is green on freMTPL2" for evidence that the library is working correctly. It is. The test is also uninformative here because the experimental design makes PSI near-zero by construction. For PSI to be a useful monitoring signal, you need a temporal split. If you are evaluating `insurance-governance` on open datasets, freMTPL2 is a good test for everything except temporal stability.

---

## Risk tier scoring

Once the validation tests are complete, `RiskTierScorer` converts the findings into a risk tier for governance purposes. The tier determines the frequency and depth of ongoing validation obligations.

```python
from insurance_governance import RiskTierScorer

scorer = RiskTierScorer()

glm_tier = scorer.score(
    gwp_impacted=50_000_000,       # £50M GWP example
    model_complexity="low",        # Poisson GLM
    deployment_status="production",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
    validation_months_ago=0,
    drift_triggers_last_year=0,
)

gbm_tier = scorer.score(
    gwp_impacted=50_000_000,
    model_complexity="high",       # CatBoost with 9 features
    deployment_status="production",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
    validation_months_ago=0,
    drift_triggers_last_year=0,
)

print(f"GLM tier: {glm_tier.tier}")
print(f"GBM tier: {gbm_tier.tier}")
```

The GBM scores higher on model complexity, which pushes it toward a higher tier if the GWP input is material. On a £50M book, both models end up in the same tier — complexity alone does not override GWP exposure. On a £500M book, the CatBoost model's complexity score would push it into Tier 1 (highest oversight), while the GLM might remain at Tier 2. That is a defensible governance outcome: more complex models on large books warrant more scrutiny.

---

## The model-agnostic promise

The API call is identical for GLM and GBM. The only difference is the prediction array passed in.

```python
# GLM: statsmodels predictions already in claim count units
glm_report = ModelValidationReport(model_card=glm_card, y_pred_val=glm_preds, ...)

# GBM: CatBoost predicts frequency, multiply by exposure for counts
gbm_preds_counts = gbm.predict(df_val[all_features]) * df_val["Exposure"].values
gbm_report = ModelValidationReport(model_card=gbm_card, y_pred_val=gbm_preds_counts, ...)
```

There are no model-specific code paths in `ModelValidationReport`. The five tests — discrimination, calibration, H-L, stability, overfitting — all operate on the prediction and outcome arrays. It does not matter whether those predictions came from a GLM, a GBM, a neural net, or a lookup table.

This is not a marketing claim. We verified it empirically by running the same validation call on statsmodels output and on CatBoost output in the same notebook. The test suite ran identically.

---

## CatBoost vs GLM: Gini lift is real but not dramatic

The GBM's access to categorical features (region, vehicle brand, area code) produces a higher Gini than the numeric-only GLM. The lift is meaningful — and the governance machinery correctly penalises the GBM for its higher complexity. But the margin is not as large as the feature count difference would suggest.

The reason is that freMTPL2 is a cross-sectional portfolio of average difficulty. The geographic signal in region and area is real but not enormous. The vehicle brand signal exists but is noisy. On a portfolio with stronger geographic concentration, more granular categoricals, or telematics data, the GBM advantage would widen. The benchmark tells you the lift you get from switching to a GBM on a dataset of this type; it does not tell you the lift you will get on your specific book.

What the benchmark does tell you is that both models can be validated with the same governance infrastructure. If you are managing a mixed estate — GLMs in motor, GBMs in home, something else in commercial — `ModelValidationReport` handles all of them uniformly. The governance pack, including the HTML report and JSON export, follows the same format regardless of model architecture. That matters for the governance function's ability to maintain consistent standards across a model inventory.

---

## Running it yourself

The benchmark notebook is at [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance) (`notebooks/benchmark_fremtpl2.py`). It downloads freMTPL2 from OpenML and produces the full side-by-side validation for both models.

```
uv add insurance-governance statsmodels catboost scikit-learn
```

One note on `fetch_openml`: several numeric columns come back as Python object dtype. Coerce them explicitly with `pd.to_numeric(errors='coerce')` before fitting. CatBoost accepts string categoricals natively via the `cat_features` list; statsmodels needs `pd.get_dummies()` or equivalent. The notebook handles both.

This is the ninth of ten flagship external benchmarks across the insurance-governance library, using the same 677K-row freMTPL2 dataset as the [insurance-monitoring drift detection benchmark](/2026/03/27/model-drift-detection-fremtpl2-insurance-monitoring/) and the [insurance-causal DML benchmark](/2026/03/28/causal-inference-fremtpl2-bonusmalus-dml/). Running the same dataset through all three libraries gives you a complete picture: validated model, monitored for drift, with causal estimates of the endogenous rating factors.
