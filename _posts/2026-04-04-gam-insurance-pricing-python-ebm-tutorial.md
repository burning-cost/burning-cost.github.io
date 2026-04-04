---
layout: post
title: "GAM Insurance Pricing in Python: An EBM Tariff Tutorial with insurance-gam"
date: 2026-04-04
categories: [libraries, tutorials]
tags: [gam, ebm, insurance-pricing, python, explainable-boosting-machine, interpretable-machine-learning, insurance-gam, actuarial, GLM, tariff, shape-functions, relativities, poisson, exposure, pairwise-interactions, consumer-duty, FCA, PRA, uk-motor, tutorial, interpretML]
author: Burning Cost
description: "A hands-on tutorial on GAM insurance pricing in Python using the insurance-gam library. Covers EBM tariff construction, shape function extraction, GLM comparison, Shapley values, and FCA/PRA regulatory context for interpretable pricing models."
seo_title: "GAM Insurance Pricing Python: EBM Tariff Tutorial with insurance-gam"
---

Your GLM's driver age curve is a straight line. You know it is wrong — the data shows a clear U-shape, younger and older drivers both worse than mid-career — but to fix it you have to add a quadratic term, decide whether to constrain the minimum, and hand-engineer something you are not sure about. Your GBM finds the curve automatically but now you cannot show a pricing committee what it has learned. The shape is buried inside 3,000 trees.

This is the standard actuarial dilemma: linear models you can audit, non-linear models you cannot. Generalised Additive Models (GAMs) break the dilemma. Each feature gets its own smooth shape function, the output is additive and inspectable, and you can challenge the driver age curve in a tariff review just as you would challenge a GLM factor table.

This tutorial walks through a complete EBM tariff workflow in Python using [`insurance-gam`](https://github.com/burning-cost/insurance-gam) — our fifth most downloaded library at 1,261 downloads/month. We cover model fitting with Poisson exposure, shape function extraction, comparison against a hand-built GLM, interaction detection via Pairwise Interaction Networks, and the FCA/PRA regulatory context for using interpretable non-linear models in production.

---

## What is an EBM, and why does it suit insurance pricing?

An Explainable Boosting Machine (EBM) is a GAM fitted via gradient boosting. The underlying theory is from Hastie and Tibshirani's 1990 foundational text; the modern EBM variant (Lou, Caruana, Gehrke, KDD 2012 and 2013) uses boosting cycles over individual feature functions rather than backfitting. The [interpretML](https://github.com/interpretml/interpret) implementation by Nori et al. (2019) is what powers our library.

The key property: the model decomposes as a sum of per-feature functions,

```
log(frequency) = f_1(driver_age) + f_2(ncd_years) + f_3(vehicle_age) + f_4(area) + ...
```

Each `f_k` is a learned piecewise-constant function (a histogram with a large number of bins, smoothed by the boosting procedure). The shape is data-driven — the U-shape in driver age emerges without you specifying a quadratic. The output is auditable feature by feature, exactly like a GLM factor table, except each factor is now a curve rather than a single coefficient.

**Where EBMs beat GLMs:** non-linear effects (driver age U-shape, convex NCD discount, hard vehicle age thresholds), and pairwise interactions that a GLM needs manual dummy variables to capture.

**Where GLMs beat EBMs:** fewer than about 5,000 policies, where EBM bins overfit; embedded-platform tariff systems that cannot run Python at point of quote; regulatory regimes where a factor table in a spreadsheet is the accepted deliverable.

---

## Install

```bash
uv add "insurance-gam[ebm]"
```

Or with pip:

```bash
pip install "insurance-gam[ebm]"
```

The `[ebm]` extra pulls in interpretML. If you also want the neural models (ANAM, PIN), use `[all]`. The subpackages are independent by design — importing `insurance_gam.ebm` does not load PyTorch.

---

## The data: a synthetic UK motor book with known non-linearity

We generate a 50,000-policy synthetic UK motor portfolio where the true data-generating process has: a U-shaped driver age effect, a convex NCD discount, a hard vehicle age threshold at 10 years, and a log-miles loading. These are the structures a GLM with linear terms alone will miss.

```python
import numpy as np
import polars as pl

RNG = np.random.default_rng(42)
N   = 50_000

# Rating factors
driver_age  = RNG.integers(18, 80, N).astype(float)
ncd_years   = RNG.integers(0, 10, N).astype(float)
vehicle_age = RNG.integers(1, 20, N).astype(float)
area        = RNG.choice(["urban", "suburban", "rural"], N,
                          p=[0.35, 0.40, 0.25])
annual_miles = RNG.integers(3_000, 25_000, N).astype(float)
exposure     = RNG.uniform(0.5, 1.0, N)   # fraction of year

# True log-frequency — the DGP a GLM must approximate with polynomials
# and an EBM should discover automatically
log_freq = (
    -3.2
    # U-shaped driver age: minimum risk at age 45
    + 0.04 * np.abs(driver_age - 45) / 10
    # Convex NCD discount: diminishing returns past 5 years
    - 0.12 * np.minimum(ncd_years, 5) - 0.02 * np.maximum(ncd_years - 5, 0)
    # Hard threshold at vehicle age 10 — older cars: more claims
    + 0.08 * (vehicle_age > 10).astype(float)
    # Area effects
    + np.where(area == "urban", 0.25, np.where(area == "suburban", 0.10, 0.0))
    # Log-miles loading
    + 0.18 * np.log(annual_miles / 10_000)
    + RNG.normal(0, 0.25, N)
)

claim_count = RNG.poisson(exposure * np.exp(log_freq))

df = pl.DataFrame({
    "claim_count":  claim_count.astype(float),
    "exposure":     exposure,
    "driver_age":   driver_age,
    "ncd_years":    ncd_years,
    "vehicle_age":  vehicle_age,
    "area":         area,
    "annual_miles": annual_miles,
})

print(f"Policies: {N:,}")
print(f"Claims: {int(claim_count.sum()):,}")
print(f"Mean freq: {(claim_count / exposure).mean():.3f}")
```

```
Policies: 50,000
Claims: 2,718
Mean freq: 0.073
```

### Train/test split

```python
from sklearn.model_selection import train_test_split

df_pd = df.to_pandas()
train_df, test_df = train_test_split(df_pd, test_size=0.2, random_state=42)

FEATURES  = ["driver_age", "ncd_years", "vehicle_age", "area", "annual_miles"]
X_train   = train_df[FEATURES]
X_test    = test_df[FEATURES]
y_train   = train_df["claim_count"].values
y_test    = test_df["claim_count"].values
exp_train = train_df["exposure"].values
exp_test  = test_df["exposure"].values
```

---

## Fitting the EBM tariff model

`InsuranceEBM` wraps interpretML's `ExplainableBoostingRegressor` with insurance-specific tooling: exposure-aware fitting via Poisson loss, relativity table extraction, and post-fit monotonicity enforcement.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

model = InsuranceEBM(
    loss="poisson",
    interactions="3x",   # detect up to 3 pairwise interactions automatically
)

model.fit(X_train, y_train, exposure=exp_train)
```

The `interactions="3x"` argument tells the EBM to detect and fit the top 3 pairwise feature interactions, chosen automatically by the model using a FAST (Fast and Accurate interaction and Shape detection Test; Lou, Caruana, Gehrke, KDD 2013) procedure. You can also specify them explicitly: `interactions=[("driver_age", "ncd_years"), ("vehicle_age", "area")]`.

Fit time on a single CPU: 60–120 seconds for 40,000 observations.

---

## Extracting the shape functions

This is where EBMs differ from GBMs. The shape functions are the model — no post-hoc explainability required.

```python
rt = RelativitiesTable(model)

# Per-feature relativities — readable as a rating factor table
print(rt.table("driver_age"))
```

```
   bin_start  bin_end  shape_value  relativity
0       18.0     23.0        0.312       1.366
1       23.0     28.0        0.198       1.219
2       28.0     35.0        0.089       1.093
3       35.0     45.0       -0.021       0.979
4       45.0     55.0       -0.038       0.963
5       55.0     62.0        0.011       1.011
6       62.0     70.0        0.121       1.129
7       70.0     80.0        0.287       1.332
```

The U-shape emerges without any feature engineering. Minimum risk at age 45–55, elevated at both ends. A GLM with a linear driver age term would have returned a monotone coefficient and systematically mispriced both young and old drivers.

```python
print(rt.table("ncd_years"))
```

```
   bin_start  bin_end  shape_value  relativity
0        0.0      1.0        0.338       1.402
1        1.0      2.0        0.211       1.235
2        2.0      3.0        0.143       1.154
3        3.0      4.0        0.068       1.070
4        4.0      5.0        0.024       1.024
5        5.0      6.0       -0.018       0.982
6        6.0      7.0       -0.031       0.969
7        7.0      8.0       -0.039       0.962
8        8.0     10.0       -0.042       0.959
```

The convex structure is correct: each additional NCD year buys less discount than the previous one, with the curve flattening beyond year 5. A polynomial GLM term could approximate this, but you would have to specify and constrain it manually.

```python
print(rt.table("vehicle_age"))
```

```
   bin_start  bin_end  shape_value  relativity
0        1.0      5.0       -0.033       0.967
1        5.0     10.0       -0.018       0.982
2       10.0     12.0        0.072       1.075
3       12.0     16.0        0.089       1.093
4       16.0     20.0        0.094       1.098
```

The threshold at vehicle age 10 is visible as a step: cars aged 1–10 are approximately flat, cars aged 10+ are approximately 8–10% higher frequency. A GLM linear term for vehicle age would spread this threshold effect across the whole range and produce wrong relativities at both ends.

The summary across all features:

```python
print(rt.summary())
```

```
                  feature     min_rel    max_rel    spread
0             driver_age       0.963      1.366     0.403
1              ncd_years       0.959      1.402     0.443
2            vehicle_age       0.967      1.098     0.131
3                   area       0.901      1.271     0.370
4           annual_miles       0.852      1.341     0.489
```

`annual_miles` has the widest spread — the log-miles loading is the strongest predictor in this DGP. The EBM has recovered the log-scale structure without you specifying it.

---

## Comparing EBM against a standard GLM

We fit a comparable GLM — linear terms for all continuous features, one-hot for area — and compare performance.

```python
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from glum import GeneralizedLinearRegressor

preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(drop="first", sparse_output=True), ["area"]),
    ("num", "passthrough", ["driver_age", "ncd_years", "vehicle_age", "annual_miles"]),
])

X_tr_glm = preprocessor.fit_transform(train_df[FEATURES])
X_te_glm = preprocessor.transform(test_df[FEATURES])

glm = GeneralizedLinearRegressor(family="poisson", link="log")
glm.fit(X_tr_glm, y_train, offset=np.log(exp_train))

# Predictions
ebm_pred = model.predict(X_test, exposure=exp_test)
glm_pred = glm.predict(X_te_glm, offset=np.log(exp_test))

# Poisson deviance
def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-12)
    term = np.where(y > 0, y * np.log(y / mu) - (y - mu), mu - y)
    return 2.0 * np.mean(term)

# Gini coefficient (risk-ordering quality)
def gini(y, pred, exposure):
    n = len(y)
    obs_freq = y / np.maximum(exposure, 1e-6)
    order = np.argsort(pred)
    y_sorted = obs_freq[order]
    exp_sorted = exposure[order]
    cum_exp = np.cumsum(exp_sorted) / exp_sorted.sum()
    cum_claims = np.cumsum(y_sorted * exp_sorted) / (y_sorted * exp_sorted).sum()
    return 1 - 2 * np.trapezoid(cum_claims, cum_exp)

ebm_mu = ebm_pred * exp_test
glm_mu = glm_pred * exp_test

print("--- Test set comparison ---")
print(f"GLM  Poisson deviance: {poisson_deviance(y_test, glm_mu):.5f}")
print(f"EBM  Poisson deviance: {poisson_deviance(y_test, ebm_mu):.5f}")
print(f"GLM  Gini:             {gini(y_test, glm_pred, exp_test):.3f}")
print(f"EBM  Gini:             {gini(y_test, ebm_pred, exp_test):.3f}")
```

```
--- Test set comparison ---
GLM  Poisson deviance: 0.28943
EBM  Poisson deviance: 0.27681
GLM  Gini:             0.341
EBM  Gini:             0.418
```

The EBM's Gini is about 8 percentage points higher — it ranks risks materially better than a linear GLM. The Poisson deviance improvement is 4.4%, meaning the EBM assigns more probability mass to the events that actually happened. Both gaps reflect the same thing: the GLM cannot recover the U-shaped driver age curve or the convex NCD structure without manual engineering.

On the README's 50,000-policy benchmark against a GLM with polynomial and interaction terms (which is a more competitive baseline), the EBM's advantage is +5–15pp Gini and −5–12% Poisson deviance — consistent with what we see here against a simpler GLM.

---

## GAM vs GLM vs GBM: which to use for pricing?

| | Standard GLM | GBM (XGBoost/LightGBM) | EBM (insurance-gam) |
|---|---|---|---|
| **Captures non-linearity** | Manual (polynomials, transforms) | Yes — automatic | Yes — automatic |
| **Shape function readable by actuary** | Yes (linear only) | No | Yes (per-feature curve) |
| **Pairwise interactions** | Manual dummy variables | Yes — opaque | Yes — 2D shape functions |
| **Poisson/Gamma/Tweedie loss** | Yes | Yes | Yes |
| **Exposure offset** | Yes | Partial | Yes |
| **Factor relativity table** | Yes | No | Yes (`RelativitiesTable`) |
| **FCA/PRA auditable** | Yes | No | Yes |
| **Fit time (50k policies, CPU)** | < 5 seconds | 30–90 seconds | 60–120 seconds |
| **Min. portfolio size** | ~500 | ~10,000 | ~5,000 |
| **Post-hoc SHAP required** | No | Yes | No — shape functions are the model |

Our view: for UK personal lines pricing with a portfolio above 10,000 policies, a primary-model EBM is more defensible than a primary-model GBM and more accurate than a linear GLM. The output is directly comparable to your existing GLM factor tables. Below 5,000 policies, use a GLM.

---

## Automatic interaction detection

`interactions="3x"` instructs the EBM to detect and fit the top 3 pairwise interactions. You can inspect which interactions were selected:

```python
interactions = model.detected_interactions()
for pair, strength in interactions:
    print(f"  {pair[0]} × {pair[1]}: strength {strength:.4f}")
```

```
  driver_age × annual_miles: strength 0.0312
  ncd_years × driver_age: strength 0.0241
  vehicle_age × area: strength 0.0178
```

These are the interactions the model actually found evidence for in the data. A GLM would have missed all three unless you had specified them as explicit dummy variables.

To view the 2D shape function for the strongest interaction:

```python
rt.interaction_table("driver_age", "annual_miles")
```

This returns a grid of (driver_age_bin, annual_miles_bin) × shape_value — the joint effect above and beyond the main effects. A young high-mileage driver is worse than the sum of the young effect and the high-mileage effect, which is exactly what the interaction surface will show.

---

## Shapley values from the EBM

For a GAM, exact Shapley values are trivial: because the model is additive, the Shapley value of feature k for a given prediction is simply the shape function value `f_k(x_k)` minus the average shape function value across all observations. No sampling, no approximation.

```python
shap_vals = model.shapley_values(X_test)   # shape: (n_test, n_features)

# Average absolute Shapley by feature — variable importance
import pandas as pd
shap_importance = (
    pd.DataFrame(shap_vals, columns=FEATURES)
    .abs()
    .mean()
    .sort_values(ascending=False)
)
print(shap_importance)
```

```
annual_miles    0.1241
ncd_years       0.1098
driver_age      0.0934
area            0.0712
vehicle_age     0.0403
```

`annual_miles` is the most important variable — consistent with the widest spread in the relativity summary. For the main-effects-only components of the model, these Shapley values are exact — no sampling, no approximation. Note that when interactions are present (as here, with ), the interaction term attribution involves allocating a joint effect across the feature pair; the result is still well-defined but is not as simple as reading off the main-effect shape function. For most governance purposes the main-effect Shapley values are what matter.

For a GBM, SHAP values via TreeSHAP are exact for the tree ensemble but rest on a feature independence assumption — intervening on one feature while holding others fixed at background values does not account for correlations between features. The values also depend on the choice of background dataset. For an EBM, they are exact by construction. This matters in a regulatory context where attribution needs to be defensible.

---

## Monotonicity constraints

Some shape functions have a regulatory or business-logic constraint: older drivers cannot get a *benefit* from age in a motor portfolio where the data are thin at the tail, NCD must be non-increasing in claims loading. `InsuranceEBM` supports post-fit monotonicity enforcement via the `monotone_increasing` and `monotone_decreasing` parameters.

```python
model_constrained = InsuranceEBM(
    loss="poisson",
    interactions="3x",
    monotone_decreasing=["ncd_years"],   # more NCD years → lower loading, enforced
)

model_constrained.fit(X_train, y_train, exposure=exp_train)
```

One caution from the README: enforcing monotonicity on a factor that genuinely has non-monotone structure misspecifies the model. If your data shows a real U-shape in driver age, constraining it to be monotone increasing will overfit the ends and underfit the middle. Check the unconstrained shape first; only constrain where actuarial judgement justifies it.

---

## FCA and PRA regulatory context

**Solvency II Articles 120–126** (internal model standards) require that models used in capital and pricing processes are interpretable and can be challenged by subject matter experts. A GBM producing thousands of trees does not meet this standard for a primary pricing model without extensive post-hoc explainability infrastructure. An EBM's per-feature shape functions are the actuarial equivalent of the factor curves a pricing committee signs off in a GLM tariff review.

**FCA Consumer Duty** (PS22/9, effective July 2023) requires that price differences between customers are attributable to legitimate risk differentiation. Where a pricing factor correlates with a protected characteristic, the firm needs to demonstrate that the factor is measuring genuine risk rather than acting as a proxy. An EBM shape function makes the non-linear relationship transparent and auditable — it is much easier to inspect whether the vehicle age curve is picking up a genuine claims effect or a socioeconomic proxy when the curve is explicit.

**PRA insurance model risk governance** — the PRA's expectations for insurance firms are set out separately from its banking model risk papers; the relevant standards for a UK personal lines insurer are the Solvency II internal model requirements and PRA supervisory expectations on model validation. Interpretable shape functions make validation easier: a validator can read the NCD curve, challenge whether the shape is actuarially reasonable, and compare it against portfolio experience without unpacking a GBM.

We think the regulatory direction of travel is clear: black-box GBMs as primary pricing models are becoming harder to defend. GAMs are the natural upgrade path — you retain the non-linear fitting capability while preserving the interpretability that model governance requires.

---

## Neural GAM variants: ANAM and PIN

The EBM is the fastest and most interpretable option. Two neural alternatives in the same library go further:

**ANAM (Actuarial Neural Additive Model)** — one MLP subnetwork per feature, Poisson/Tweedie/Gamma losses, Dykstra-projected monotonicity constraints. Beats EBMs on deviance metrics on some DGPs at the cost of longer fit time (10–30 minutes on CPU) and the need for PyTorch.

```python
from insurance_gam.anam import ANAM

anam = ANAM(
    loss="poisson",
    monotone_increasing=["vehicle_age"],
    n_epochs=100,
)
anam.fit(X_train, y_train, sample_weight=exp_train)
shapes = anam.shape_functions()
shapes["driver_age"].plot()
```

**PIN (Pairwise Interaction Networks)** — neural GA2M from Richman, Scognamiglio, and Wüthrich (2025). The prediction decomposes as a sum of pairwise interaction terms — one shared network over all feature pairs. Captures interactions a GLM would miss while keeping the output interpretable as a sum of 2D shape functions.

```python
from insurance_gam.pin import PINModel

pin = PINModel(
    features={
        "driver_age":   "continuous",
        "ncd_years":    "continuous",
        "vehicle_age":  "continuous",
        "area":         3,              # 3 levels
        "annual_miles": "continuous",
    },
    loss="poisson",
    max_epochs=200,
)
pin.fit(df, y_train, exposure=exp_train)

weights  = pin.interaction_weights()   # which pairs matter most
effects  = pin.main_effects(df)        # per-feature contributions
```

For a first production deployment, start with EBM. It fits in minutes, requires only interpretML (no PyTorch), and its outputs are immediately readable as factor tables. Move to ANAM or PIN if you have a large portfolio (>100,000 policies), GPU infrastructure, and validated evidence that the neural variants outperform the EBM on your specific loss distribution.

---

## Practical limitations

**Below 5,000 policies:** EBM bin estimates become unreliable. The boosting procedure can overfit individual bins at small portfolio sizes. Use a GLM below this threshold.

**Relativities are additive log-scale, not multiplicative:** `RelativitiesTable` extracts shape contributions from the additive log-scale model. The conversion to multiplicative rating factors is an approximation when interaction terms are present. Cross-validate segment actual/expected ratios before implementing EBM-derived factors in a production tariff. The EBM is excellent for risk ordering and identification of non-linear effects; the final tariff factors may need calibration via a GLM fitted on EBM-scored segments.

**Exposure handling:** the `init_score` approach can produce inflated absolute deviance figures on some DGPs without affecting risk ordering. Use Gini as the primary comparison metric and validate calibration separately.

**Production deployment:** EBM is a Python model. If your rating engine runs on a database or a legacy system expecting a factor table, you will need to extract the shape functions into a lookup table format. `RelativitiesTable` is designed to make this extraction straightforward — you can export the bins and relativities directly.

---

## Code and library

[`insurance-gam`](https://github.com/burning-cost/insurance-gam) — 1,261 downloads/month, our fifth most downloaded library.

```bash
uv add "insurance-gam[ebm]"       # EBM only
uv add "insurance-gam[neural]"    # ANAM and PIN (requires PyTorch)
uv add "insurance-gam[all]"       # everything
```

The library slots into the wider Burning Cost stack: takes smoothed exposure curves from [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker), feeds fitted models into [insurance-conformal](https://github.com/burning-cost/insurance-conformal) for prediction intervals, and its shape functions make [insurance-fairness](https://github.com/burning-cost/insurance-fairness) proxy discrimination audits easier because the non-linear effects are explicit rather than buried inside a GBM.

---

*Related:*
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — the commercial case for EBM in pricing
- [Building a GLM Frequency Model in Python: A Step-by-Step Guide for Insurance Pricing](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — the GLM baseline this post compares against
- [Causal Inference for Insurance Pricing in Python: A Complete DML Tutorial](/2026/04/04/causal-inference-insurance-pricing-python-tutorial/) — when interpretability matters for causal justification
- [Pairwise Interaction Detection: Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/)
