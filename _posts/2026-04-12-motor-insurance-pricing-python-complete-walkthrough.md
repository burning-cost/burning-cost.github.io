---
layout: post
title: "Motor Insurance Pricing in Python: A Complete Walkthrough"
date: 2026-04-12
categories: [tutorials]
tags: [motor, GLM, frequency-severity, python, pricing, tutorial, FrEMTPL2, poisson, gamma, tweedie]
description: "End-to-end motor insurance pricing in Python using the French MTPL dataset. Frequency-severity GLMs, exposure offsets, coefficient interpretation, validation, and calibration to gross premium. The complete workflow for pricing actuaries and data scientists."
---

Motor insurance pricing is fundamentally a prediction problem with an awkward structure: most policyholders never claim, the ones who do claim have losses that span four orders of magnitude, and the exposure to risk varies continuously across the portfolio. A single predictive model handles that structure poorly. The industry-standard solution, used everywhere from Lloyds syndicates to tier-1 personal lines direct writers, is the frequency-severity split.

This post builds that model from scratch in Python, using the FrEMTPL2 dataset — the standard benchmark in the academic and practitioner literature. We cover data loading and exploratory analysis, a Poisson GLM for claim frequency, a Gamma GLM for claim severity, combining them into a pure premium, validation on held-out data, and a brief account of how you turn a pure premium into something you can actually charge. Every code block is self-contained and reproducible.

This is the entry point. At the end we point to posts that extend each component: GBMs as an alternative to GLMs, SHAP for extracting factor tables from GBMs, conformal prediction for uncertainty bounds, and model monitoring for production.

---

## Why frequency-severity rather than a single model

The direct approach would be to model expected loss per unit of exposure: pounds per policy-year. This is a Tweedie regression — a single model that handles the zero-inflated, right-skewed distribution of insurance losses by mixing a Poisson mass at zero with a Gamma tail.

Tweedie models work. But the frequency-severity split is preferred in practice for three reasons.

First, the drivers of claim frequency and the drivers of claim severity are often different. Young drivers have high frequency (more incidents); high-performance cars have high severity (more expensive repairs). A combined model has to learn both relationships simultaneously, which makes sense statistically but obscures the interpretation. When you explain a rate change to an underwriter or a pricing committee, you want to say "frequency is up 4% in the 17–25 band because of more minor scrapes; severity is flat" — not "Tweedie prediction increased."

Second, data quality differs between frequency and severity. Frequency is stable. Severity is affected by inflation trends, large-loss volatility, and development (claims that are not yet fully settled). Modelling them separately lets you apply different credibility adjustments and inflation indices to each.

Third, the FCA's expectations under the pricing practices regulation (effective January 2022) require firms to demonstrate that pricing factors are justified and not discriminatory. A factor table from a frequency-severity GLM is auditable line by line. A Tweedie GBM prediction is less transparent, which creates a burden of explainability work downstream.

We use the frequency-severity split throughout this post. The Tweedie model is covered in [our CatBoost post](/2026/03/22/catboost-insurance-pricing-frequency-severity-fremtpl2/) as a validation tool.

---

## Data: FrEMTPL2

The French Motor Third-Party Liability dataset (freMTPL2) comes from a French insurer and covers approximately 678,000 motor policies. It is the benchmark used in Noll, Salzmann and Wüthrich (2020), in the scikit-learn documentation, and in most published insurance pricing papers since 2019. We use it because it is publicly available, large enough to give stable estimates, and has the structure of a real pricing dataset: varying exposure, sparse claims, geographic variation, and a mix of continuous and categorical features.

```python
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

# freMTPL2freq: policy features and claim counts
# freMTPL2sev: individual claim amounts (one row per claim)
freq_raw = fetch_openml(data_id=41214, as_frame=True, parser="auto")
sev_raw  = fetch_openml(data_id=41215, as_frame=True, parser="auto")

freq = freq_raw.frame.copy()
sev  = sev_raw.frame.copy()

print(freq.shape)   # (678013, 12)
print(sev.shape)    # (26639, 2)
```

The frequency table has one row per policy-year. The severity table has one row per claim. The link between them is `IDpol`.

### Column overview

```python
print(freq.dtypes)
```

Key columns:

| Column | Type | Description |
|---|---|---|
| `IDpol` | int | Policy identifier |
| `ClaimNb` | int | Number of claims in the exposure period |
| `Exposure` | float | Policy-years exposed to risk (0–1) |
| `Area` | category | Urbanisation band (A–F, A = most rural) |
| `VehPower` | int | Vehicle power band |
| `VehAge` | int | Vehicle age in years |
| `DrivAge` | int | Driver age in years |
| `BonusMalus` | int | Bonus-malus coefficient (50 = maximum bonus) |
| `VehBrand` | category | Vehicle brand (B1–B14) |
| `VehGas` | category | Fuel type (Diesel / Regular) |
| `Density` | int | Population density of driver's commune |
| `Region` | category | French administrative region (22 regions) |

### Basic EDA

```python
# Exposure distribution
print(freq["Exposure"].describe())
# Mean exposure is around 0.53 policy-years — most policies are mid-term
# (new business, cancellations, renewals within the year)

# Claim frequency
total_claims = freq["ClaimNb"].sum()
total_exposure = freq["Exposure"].sum()
raw_frequency = total_claims / total_exposure
print(f"Total claims: {total_claims:,}")
print(f"Total exposure: {total_exposure:,.0f} policy-years")
print(f"Raw claim frequency: {raw_frequency:.4f}")
# Approximately 6.7% — roughly one claim per 15 policy-years
```

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Exposure distribution
axes[0].hist(freq["Exposure"], bins=50, edgecolor="none", color="#2c7bb6")
axes[0].set_title("Exposure distribution")
axes[0].set_xlabel("Policy-years")
axes[0].set_ylabel("Policies")

# Claims per policy
claim_counts = freq["ClaimNb"].value_counts().sort_index()
axes[1].bar(claim_counts.index[:6], claim_counts.values[:6], color="#2c7bb6")
axes[1].set_title("Claims per policy")
axes[1].set_xlabel("Number of claims")
axes[1].set_ylabel("Policies")

# Severity distribution (log scale)
axes[2].hist(np.log1p(sev["ClaimAmount"]), bins=60, edgecolor="none", color="#d7191c")
axes[2].set_title("log(1 + claim amount)")
axes[2].set_xlabel("log(€+1)")
axes[2].set_ylabel("Claims")

plt.tight_layout()
plt.savefig("fremtpl2_eda.png", dpi=150)
plt.show()
```

The severity distribution is roughly log-normal with a long right tail. Average claim amount is around €2,300; the 95th percentile is around €7,000; the 99th percentile exceeds €30,000. This is exactly the shape that a Gamma GLM with log link handles well.

---

## Feature engineering

Before fitting, we clean types and create the features we actually want to model on.

```python
# Type coercion
freq["ClaimNb"] = freq["ClaimNb"].astype(int)
freq["Exposure"] = freq["Exposure"].astype(float).clip(0.001, 1.0)

# Driver age: bin into actuarially meaningful bands
age_bins   = [17, 22, 26, 32, 42, 52, 62, 75, 200]
age_labels = ["17-21", "22-25", "26-31", "32-41", "42-51", "52-61", "62-74", "75+"]
freq["DrivAgeBand"] = pd.cut(
    freq["DrivAge"],
    bins=age_bins,
    labels=age_labels,
    right=True
)

# Vehicle age: bin similarly
vage_bins   = [0, 1, 3, 6, 10, 15, 200]
vage_labels = ["0-1", "2-3", "4-6", "7-10", "11-15", "16+"]
freq["VehAgeBand"] = pd.cut(
    freq["VehAge"],
    bins=vage_bins,
    labels=vage_labels,
    right=True
)

# BonusMalus: cap extreme values; model as continuous
freq["BonusMalus"] = freq["BonusMalus"].clip(50, 230)

# Density: log-transform for GLM linearity assumption
freq["LogDensity"] = np.log1p(freq["Density"])

# Log exposure for offset
freq["LogExposure"] = np.log(freq["Exposure"])
```

The binning of driver age deserves a word. We are not doing this to reduce dimensionality — `DrivAge` as a continuous variable in a GLM would be fitted as a linear relationship on the log scale, which is wrong (the relationship is U-shaped: young and old drivers both have higher frequency). Binning captures the non-linearity while keeping the GLM framework. An alternative is a GAM with spline terms; we touch on that in the GAM post.

---

## Frequency model: Poisson GLM

The Poisson GLM models expected claim count given exposure:

$$\log(E[\text{ClaimNb}_i]) = \log(\text{Exposure}_i) + \beta_0 + \beta_1 x_{1i} + \cdots$$

The log(Exposure) term is an *offset* — it enters the linear predictor with a fixed coefficient of 1, not estimated. This enforces that expected claims scale linearly with exposure, which is the correct assumption for a Poisson process. A common mistake is to include exposure as a feature and let the model estimate its coefficient; that would allow the model to learn a non-linear relationship between exposure and claims, which makes no physical sense.

We use `statsmodels` for the frequency GLM. It gives us full summary output including standard errors, p-values, and diagnostics without extra work.

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm

freq_formula = """
ClaimNb ~ DrivAgeBand + VehAgeBand + C(Area) + C(VehBrand)
        + C(VehGas) + C(Region) + BonusMalus + LogDensity
        + VehPower + offset(LogExposure)
"""

freq_model = smf.glm(
    formula=freq_formula,
    data=freq,
    family=sm.families.Poisson()
).fit()

print(freq_model.summary())
```

### Interpreting coefficients as relativities

The Poisson GLM has a log link, so coefficients are additive on the log scale and multiplicative on the claim frequency scale. The relativity for a coefficient β is `exp(β)`.

```python
# Extract relativities for driver age bands
coefs = freq_model.params
confs = freq_model.conf_int()

driv_age_coefs = coefs[coefs.index.str.startswith("DrivAgeBand")]
driv_age_lower = confs.loc[confs.index.str.startswith("DrivAgeBand"), 0]
driv_age_upper = confs.loc[confs.index.str.startswith("DrivAgeBand"), 1]

relativities = pd.DataFrame({
    "band"      : driv_age_coefs.index.str.replace("DrivAgeBand[T.", "", regex=False).str.replace("]", ""),
    "relativity": np.exp(driv_age_coefs.values),
    "lower_95"  : np.exp(driv_age_lower.values),
    "upper_95"  : np.exp(driv_age_upper.values),
})

print(relativities.to_string(index=False))
```

A relativity of 1.45 for the 17–21 band means a driver in that band is expected to have 45% more claims than the base level (22–25), all else equal. This is the format actuaries, underwriters, and pricing committees work with.

### Residual diagnostics

```python
# Pearson residuals vs fitted values
fitted_freq  = freq_model.fittedvalues
pearson_resid = freq_model.resid_pearson

plt.figure(figsize=(8, 4))
plt.scatter(np.log(fitted_freq + 1e-6), pearson_resid,
            alpha=0.05, s=2, color="#2c7bb6")
plt.axhline(0, color="black", linewidth=0.8)
plt.xlabel("log(fitted claims)")
plt.ylabel("Pearson residual")
plt.title("Frequency model: residuals vs fitted")
plt.tight_layout()
plt.savefig("freq_residuals.png", dpi=150)
plt.show()
```

What you want to see: residuals centred on zero with no systematic pattern. What you often see in insurance data is overdispersion — the variance of the residuals is larger than the Poisson model predicts. In freMTPL2 the Poisson model fits reasonably well; in UK motor data with postcode-level clustering, a negative binomial GLM is sometimes preferred.

---

## Severity model: Gamma GLM

The severity model is fitted only on policies with at least one claim. We use the claim-level severity table joined to the frequency table for features.

```python
# Join features onto severity data
sev_with_features = sev.merge(
    freq[["IDpol", "DrivAgeBand", "VehAgeBand", "Area", "VehBrand",
          "VehGas", "Region", "BonusMalus", "LogDensity", "VehPower",
          "Exposure"]],
    on="IDpol",
    how="left"
)

# Remove zero and implausible claims
sev_clean = sev_with_features[sev_with_features["ClaimAmount"] > 0].copy()

print(f"Claims for severity model: {len(sev_clean):,}")
print(sev_clean["ClaimAmount"].describe())
```

The Gamma GLM with log link models expected claim amount:

$$\log(E[\text{ClaimAmount}_i]) = \beta_0 + \beta_1 x_{1i} + \cdots$$

No exposure offset here — we are modelling the cost of a claim that has already occurred, not a rate per unit of exposure.

```python
sev_formula = """
ClaimAmount ~ DrivAgeBand + VehAgeBand + C(Area) + C(VehBrand)
            + C(VehGas) + C(Region) + BonusMalus + LogDensity
            + VehPower
"""

sev_model = smf.glm(
    formula=sev_formula,
    data=sev_clean,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

print(sev_model.summary())
```

```python
# Dispersion parameter — how well the Gamma assumption fits
dispersion = sev_model.scale
print(f"Gamma dispersion (phi): {dispersion:.4f}")
# Values around 0.5–1.5 are typical for motor claims
```

### Comparing frequency and severity relativities

A useful sanity check: plot the frequency relativity and severity relativity for the same factor side by side. For driver age, you expect frequency to peak in the 17–21 band and decline monotonically thereafter; you might expect severity to show less variation (or even the reverse, since experienced drivers have more complex journeys). Where the relativities conflict, think about why.

```python
# Severity relativities for driver age
sev_coefs = sev_model.params
sev_driv  = sev_coefs[sev_coefs.index.str.startswith("DrivAgeBand")]

sev_rels = np.exp(sev_driv.values)
frq_rels = relativities["relativity"].values  # from above

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(age_labels) - 1)  # bands that appear in model
width = 0.35
ax.bar(x - width/2, frq_rels, width, label="Frequency", color="#2c7bb6")
ax.bar(x + width/2, sev_rels,  width, label="Severity",  color="#d7191c")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(x)
ax.set_xticklabels(age_labels[:-1], rotation=30)
ax.set_ylabel("Relativity vs base")
ax.set_title("Frequency vs severity relativities by driver age")
ax.legend()
plt.tight_layout()
plt.savefig("freq_sev_relativities.png", dpi=150)
plt.show()
```

---

## Combining: pure premium

The pure premium (also called the technical rate or burning cost) is the expected loss per unit of exposure:

$$\text{PurePremium}_i = \hat{\lambda}_i \times \hat{s}_i$$

where $\hat{\lambda}_i$ is the predicted claim frequency and $\hat{s}_i$ is the predicted claim severity. Both are in-sample predictions here; in practice you predict on new business.

```python
# Frequency predictions (expected claims per policy-year)
freq["PredFreq"] = freq_model.predict(freq)

# Severity predictions for all policies (not just claimants)
freq["PredSev"] = sev_model.predict(freq)

# Pure premium
freq["PurePremium"] = freq["PredFreq"] * freq["PredSev"]

print(freq["PurePremium"].describe())
```

```python
# Portfolio distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(freq["PurePremium"], bins=100, color="#2c7bb6", edgecolor="none")
axes[0].set_xlabel("Pure premium (€)")
axes[0].set_title("Distribution of pure premiums")

axes[1].hist(np.log1p(freq["PurePremium"]), bins=100, color="#2c7bb6", edgecolor="none")
axes[1].set_xlabel("log(1 + pure premium)")
axes[1].set_title("log-scale pure premium")

plt.tight_layout()
plt.savefig("pure_premium_dist.png", dpi=150)
plt.show()
```

The distribution should be roughly log-normal with a long right tail. Policies at the extremes — very young drivers with powerful cars in urban areas — will have pure premiums several times the portfolio mean.

---

## Validation

### Train/test split

We fit on 80% of policies and validate on the held-out 20%. Because we have a time dimension (exposure periods), a proper time-based split would be ideal; for FrEMTPL2, which does not have policy dates, a random split is acceptable.

```python
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    freq.index, test_size=0.2, random_state=42
)

train = freq.loc[train_idx].copy()
test  = freq.loc[test_idx].copy()

# Refit frequency model on training data
freq_model_train = smf.glm(
    formula=freq_formula,
    data=train,
    family=sm.families.Poisson()
).fit()

# Refit severity model on training claims only
train_claims_idx = sev_clean["IDpol"].isin(train["IDpol"])
sev_model_train  = smf.glm(
    formula=sev_formula,
    data=sev_clean[train_claims_idx],
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

# Predict on test set
test["PredFreq"]     = freq_model_train.predict(test)
test["PredSev"]      = sev_model_train.predict(test)
test["PurePremium"]  = test["PredFreq"] * test["PredSev"]
test["ExpectedLoss"] = test["PurePremium"] * test["Exposure"]
```

### Lift chart: predicted vs actual loss ratio by decile

```python
# Observed loss per policy (aggregate to IDpol level from severity table)
actual_loss = (
    sev_clean[sev_clean["IDpol"].isin(test["IDpol"])]
    .groupby("IDpol")["ClaimAmount"]
    .sum()
    .reset_index()
    .rename(columns={"ClaimAmount": "ActualLoss"})
)

test = test.merge(actual_loss, on="IDpol", how="left")
test["ActualLoss"] = test["ActualLoss"].fillna(0)

# Sort by predicted pure premium, bucket into deciles
test["Decile"] = pd.qcut(test["PurePremium"], 10, labels=False) + 1

decile_summary = test.groupby("Decile").agg(
    Policies       = ("IDpol", "count"),
    Exposure       = ("Exposure", "sum"),
    ExpectedLoss   = ("ExpectedLoss", "sum"),
    ActualLoss     = ("ActualLoss", "sum"),
    PredPurePrem   = ("PurePremium", "mean"),
).assign(
    ActualLossRate = lambda d: d["ActualLoss"]  / d["Exposure"],
    PredLossRate   = lambda d: d["ExpectedLoss"] / d["Exposure"],
    LiftRatio      = lambda d: (d["ActualLoss"] / d["ExpectedLoss"])
)

print(decile_summary[["Policies", "PredLossRate", "ActualLossRate", "LiftRatio"]].to_string())
```

```python
# Double lift chart
fig, ax = plt.subplots(figsize=(9, 5))
x = decile_summary.index
ax.plot(x, decile_summary["PredLossRate"],  marker="o", label="Predicted", color="#2c7bb6")
ax.plot(x, decile_summary["ActualLossRate"], marker="s", label="Actual",    color="#d7191c")
ax.set_xlabel("Decile (1 = lowest predicted risk)")
ax.set_ylabel("Loss rate (€ per policy-year)")
ax.set_title("Lift chart: predicted vs actual loss rate by decile")
ax.legend()
plt.tight_layout()
plt.savefig("lift_chart.png", dpi=150)
plt.show()
```

What you want: the two lines to track each other closely. Systematic divergence in the tails — the model predicts the wrong ordering of risk — indicates that you are missing important rating factors or that your feature engineering is not capturing a non-linear relationship.

### Gini coefficient

The Gini coefficient measures the model's ability to rank risks. A Gini of 0 means the model has no discriminatory power (predicts the same rate for everyone); a Gini of 1 means it perfectly separates high and low risk. UK motor models typically achieve Gini coefficients of 0.35–0.55 on frequency.

```python
def gini_coefficient(actual, predicted, weight=None):
    """
    Gini coefficient for a regression/pricing model.
    Uses the insurance definition: 2 * AUC(actual_binary) - 1
    applied via a sorted concentration curve.
    """
    df = pd.DataFrame({"actual": actual, "predicted": predicted})
    if weight is not None:
        df["weight"] = weight
    else:
        df["weight"] = 1.0

    df = df.sort_values("predicted")
    df["cum_weight"]  = df["weight"].cumsum() / df["weight"].sum()
    df["cum_actual"]  = (df["actual"] * df["weight"]).cumsum() / (df["actual"] * df["weight"]).sum()

    # Trapezoid rule
    auc = np.trapz(df["cum_actual"], df["cum_weight"])
    return 2 * auc - 1

# Frequency Gini on test set
# Binary actual: did this policy have at least one claim?
test["HasClaim"] = (test["ClaimNb"] > 0).astype(int)

gini_freq = gini_coefficient(
    actual    = test["HasClaim"],
    predicted = test["PredFreq"],
    weight    = test["Exposure"]
)

gini_pp = gini_coefficient(
    actual    = test["ActualLoss"],
    predicted = test["PurePremium"],
    weight    = test["Exposure"]
)

print(f"Frequency Gini:    {gini_freq:.4f}")
print(f"Pure premium Gini: {gini_pp:.4f}")
```

For this dataset and model specification, you should see a frequency Gini around 0.28–0.32. That is lower than production UK motor models because FrEMTPL2 does not include NCD, telematics, or postcode-level data — the most powerful pricing factors in real UK pricing.

---

## Calibration: from pure premium to gross premium

The pure premium is the expected loss. Before you can charge it, you need to load it for the costs of doing business. The full premium build-up looks like this:

```
Pure premium (expected loss)
+ Large loss loading       — the expected cost of claims above a working layer, which distort the GLM
+ Development loading      — adjustment for IBNR (claims incurred but not yet reported)
+ Reinsurance cost         — the net premium for any excess-of-loss programme protecting the layer
─────────────────────────────
= Risk premium (net of expenses)
+ Expense loading          — acquisition costs, claims handling, overheads (typically 20–35% of gross)
+ Profit margin            — target combined ratio below 100%
─────────────────────────────
= Gross premium (what the customer pays)
```

In Python, the calculation is straightforward once you have the loadings:

```python
# Illustrative loadings — in practice these come from actuarial reserving and
# expense analysis, not from the pricing model itself
large_loss_loading   = 1.05   # 5% uplift for excess claims
development_loading  = 1.03   # 3% IBNR adjustment
reinsurance_cost     = 0.02   # £0.02 per £1 of risk premium
expense_ratio        = 0.28   # 28% of gross premium
target_loss_ratio    = 0.65   # target combined ratio ~ 93%

# Risk premium
freq["RiskPremium"] = (
    freq["PurePremium"]
    * large_loss_loading
    * development_loading
    / (1 - reinsurance_cost)
)

# Gross premium (from risk premium given expense and profit target)
# risk_premium = gross_premium * target_loss_ratio
# gross_premium = risk_premium / target_loss_ratio * (1 / (1 - expense_ratio))
# Simplification: gross = risk / (target_loss_ratio * (1 - expense_ratio))
freq["GrossPremium"] = freq["RiskPremium"] / (target_loss_ratio * (1 - expense_ratio))

print(freq["GrossPremium"].describe())
```

In a real pricing exercise, the large loss loading, development loading, and reinsurance cost are set by a separate team (typically reserving or capital). The pricing model's job is to distribute the total risk premium correctly across the portfolio. The loadings determine the overall level; the model determines the relativities.

One important constraint: the sum of modelled risk premiums must equal the total expected loss (i.e., the model must be calibrated to the observed loss ratio). If your frequency and severity models are systematically over- or under-predicting, you apply a scalar correction before computing gross premiums.

```python
# Calibration check
actual_total_loss = test["ActualLoss"].sum()
predicted_total   = test["ExpectedLoss"].sum()
calibration_ratio = actual_total_loss / predicted_total

print(f"Calibration ratio: {calibration_ratio:.4f}")
# Should be close to 1.0; if not, investigate whether your
# train/test split, development lags, or model bias are the cause
```

---

## Where to go from here

This post has built the foundation: a frequency-severity GLM on FrEMTPL2, fully validated. The logical next steps are:

**GBMs as an alternative.** CatBoost handles categorical features natively, does not require binning, and typically outperforms GLMs by 3–5 Gini points on real UK motor data. The flip side is interpretability. [CatBoost for Insurance Pricing: Frequency-Severity on freMTPL2](/2026/03/22/catboost-insurance-pricing-frequency-severity-fremtpl2/) uses the same dataset and gives you a direct comparison.

**Extracting GLM-format factor tables from a GBM.** Once you have a GBM, you need a factor table for the pricing committee and for Radar/Emblem. [SHAP Relativities for Insurance GBMs](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) shows how to get `exp(β)`-equivalent multiplicative relativities with confidence intervals from any CatBoost model — without being constrained to additive structure.

**Uncertainty quantification.** Point estimates are not enough for individual risk decisions. [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) provides distribution-free coverage guarantees per policy — which is more honest about model uncertainty than bootstrapped GLM confidence intervals.

**Production monitoring.** Once the model goes live, you need to know when it has gone stale. [Insurance Model Monitoring in Python](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) covers Gini drift, A/E ratio monitoring, and sequential testing using the `insurance-monitoring` library — all exposure-weighted, all designed for the pricing context rather than generic ML drift detection.

---

## Summary

The frequency-severity GLM is the right starting point for motor insurance pricing in Python. It is interpretable, auditable, well-understood by regulators, and produces factor tables that pricing committees can challenge. The FrEMTPL2 dataset gives you a public benchmark to test against.

The model we have built here — Poisson frequency, Gamma severity, combined pure premium, validated on a held-out test set with Gini and lift chart — is the standard workflow used at every UK personal lines insurer. The specifics vary (different features, different credibility approaches, time-based splits, large loss truncation) but the structure is the same.

Where GLMs fall short is in capturing interactions between features: a young driver in a powerful car in an urban area is riskier than the product of the three individual factors implies. GBMs handle interactions natively. The rest of the site is about where to go from there.
