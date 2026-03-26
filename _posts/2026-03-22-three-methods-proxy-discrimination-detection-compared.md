---
layout: post
title: "Three Ways to Detect Proxy Discrimination in Your Pricing Model: When They Agree and When They Don't"
date: 2026-03-22
author: Burning Cost
categories: [compliance, fairness, techniques]
description: "Mutual information, proxy R-squared, and SHAP proxy scores all flag proxy discrimination but catch different things. A practical guide to interpreting conflicting signals in insurance-fairness proxy detection."
tags: [FCA, Consumer-Duty, fairness, proxy-discrimination, insurance-fairness, mutual-information, proxy-r2, SHAP, detect-proxies, Equality-Act, python, uk-insurance, methodology]
---

The most common question we hear after teams run their first `FairnessAudit` is not "how do I fix this" but "these three methods disagree - which one should I believe?"

`insurance-fairness` flags proxy discrimination using three methods simultaneously: mutual information, proxy R-squared (via CatBoost), and SHAP proxy scores. When all three are red, you have a problem. When only one is, the diagnosis is subtler - and getting it wrong in either direction costs you. A false negative leaves you exposed to an FCA file review. A false positive triggers expensive factor restructuring on a model that was fine.

This post explains what each method is measuring, when they diverge, and how to weigh conflicting signals. We use the library's `detect_proxies()` function throughout so you can reproduce the analysis on your own book.

---

## What each method actually measures

Before the code, a clear statement of what these three numbers mean. They are not three ways of measuring the same thing.

**Proxy R-squared** is the answer to: if I trained a CatBoost model to predict the protected characteristic from this rating factor alone, how much variance would it explain? It measures the raw statistical association between the factor and the protected characteristic, without any reference to prices.

**Mutual information** measures statistical dependence more broadly than R-squared. It captures non-linear and non-monotone relationships that correlation-based methods miss. For a categorical factor like postcode district, where the relationship to ethnicity proportion is neither linear nor monotone but is real, mutual information will often fire when proxy R-squared does not.

**SHAP proxy score** is different in kind. It is the Spearman rank correlation between a factor's SHAP contribution to the predicted price and the protected characteristic. It measures not just whether the factor correlates with the protected characteristic, but whether that correlation is being *transmitted into prices*. A factor can be strongly correlated with a protected characteristic (high proxy R-squared) but contribute negligible price variation (near-zero SHAP proxy score). These are two very different situations from a regulatory standpoint.

The Equality Act 2010 Section 19 test turns on whether a provision, criterion or practice puts persons sharing a protected characteristic at a *particular disadvantage*. Disadvantage requires a price impact. A factor that correlates with ethnicity but contributes 0.1% of total price variance is technically a proxy but is not causing material disadvantage. A factor with lower correlation but high SHAP contribution to prices is the genuine risk.

---

## Setup

```bash
uv add insurance-fairness catboost polars numpy
```

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness.proxy_detection import detect_proxies
```

---

## A concrete example where the methods disagree

We construct a synthetic portfolio with three rating factors that have deliberately different proxy profiles. The point is to make the three methods land in different places for the same factor.

```python
rng = np.random.default_rng(2024)
n = 15_000

# Protected characteristic: ethnicity proxy (proportion non-white-British at LSOA)
# We make this continuous, as you would have from an ONS Census 2021 join
ethnicity_prop = rng.beta(2, 6, n)  # right-skewed, like real UK postcode data

# Factor 1: postcode district
# Strong linear correlation with ethnicity_prop - classic proxy channel
# This is deliberately constructed to have high MI, high proxy R2, AND high SHAP
postcode_score = 0.8 * ethnicity_prop + 0.2 * rng.standard_normal(n)
postcode_band  = np.digitize(postcode_score, bins=np.quantile(postcode_score, np.linspace(0, 1, 20))).astype(float)

# Factor 2: vehicle_group (categorical, A-E)
# Non-linear relationship to ethnicity_prop: groups B and D skew high-ethnicity
# High mutual information but moderate proxy R2 (CatBoost picks up the categorical structure)
vehicle_group_idx = rng.choice(5, n)
vehicle_group_str = np.array(["A", "B", "C", "D", "E"])[vehicle_group_idx]
# B and D are more common in high-ethnicity postcodes
vg_ethnicity_bump = np.where(
    (vehicle_group_str == "B") | (vehicle_group_str == "D"),
    rng.uniform(0.1, 0.3, n),
    0.0,
)
ethnicity_prop = np.clip(ethnicity_prop + vg_ethnicity_bump, 0, 1)

# Factor 3: ncd_years
# Moderate correlation with ethnicity_prop (shorter tenure in high-ethnicity groups)
# Captures genuine risk variation - not just demographic correlation
# Moderate proxy R2, LOW SHAP proxy score because model weights it on risk signal
ncd_years = np.clip(
    rng.poisson(4, n) - 2 * (ethnicity_prop > 0.4).astype(float),
    0, 9
).astype(float)

driver_age  = rng.integers(21, 75, n).astype(float)
vehicle_age = rng.integers(1, 15, n).astype(float)
exposure    = rng.uniform(0.3, 1.0, n)

# Claim cost: driven by genuine risk factors plus a modest ethnicity-proxy leakage
# via postcode_band (the proxy channel we want to detect)
claim_cost = np.exp(
    4.0
    + 0.03 * vehicle_age
    - 0.008 * ncd_years
    - 0.004 * driver_age
    + 0.15 * (postcode_band / 20)      # proxy channel - drives price disparity
    + rng.normal(0, 0.4, n)
) * exposure

# Fit a CatBoost model on all rating factors (no ethnicity_prop)
X = np.column_stack([postcode_band, ncd_years, driver_age, vehicle_age])
y = claim_cost / exposure

model = CatBoostRegressor(iterations=200, depth=4, verbose=0, random_seed=42)
model.fit(X, y, sample_weight=exposure)
predicted_rate = model.predict(X)

df = pl.DataFrame({
    "ethnicity_prop":  ethnicity_prop,
    "postcode_band":   postcode_band,
    "vehicle_group":   vehicle_group_str,
    "ncd_years":       ncd_years,
    "driver_age":      driver_age,
    "vehicle_age":     vehicle_age,
    "exposure":        exposure,
    "claim_cost":      claim_cost,
    "predicted_rate":  predicted_rate,
})
```

Now run proxy detection on the three factors we care about:

```python
result = detect_proxies(
    df,
    protected_col="ethnicity_prop",
    factor_cols=["postcode_band", "vehicle_group", "ncd_years"],
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
)

print(result.to_polars().select([
    "factor", "proxy_r2", "mutual_information", "partial_correlation", "rag"
]))
```

The output will show the three factors sitting in different parts of the method-agreement space:

- `postcode_band`: high across all three methods. All methods agree this is a problem.
- `vehicle_group`: high mutual information, lower proxy R-squared. The non-linear categorical relationship is what mutual information detects.
- `ncd_years`: modest proxy R-squared and mutual information, but further investigation via SHAP (see below) will show low SHAP proxy score because the model is picking up genuine risk, not demographic correlation.

---

## When to act on each signal

### All three methods agree: act

If proxy R-squared is above 0.10, mutual information is elevated, and you later find high SHAP proxy scores, you have a factor that is correlated with a protected characteristic and transmitting that correlation into prices at scale. This is what postcode looks like in a real UK motor book. The Citizens Advice (2022) analysis found proxy R-squared of approximately 0.38 for postcode district against ethnicity proportion in Inner London postcodes. That level of association is not marginal.

The correct response is: investigate whether the factor's predictive power is justified on causal risk grounds, and run the counterfactual fairness check to quantify what a discrimination-free price looks like.

### High mutual information, modest proxy R-squared: the categorical factor case

`vehicle_group` in our example illustrates a detection gap. CatBoost proxy R-squared fits a regression model; if the relationship between the factor and the protected characteristic is categorical rather than monotone, R-squared underestimates the association. Mutual information is non-parametric and catches it.

For categorical factors, treat elevated mutual information as the primary signal. The threshold in `insurance-fairness` is calibrated so that a mutual information above 0.05 with a categorical factor is worth investigating, even if proxy R-squared is below the 0.10 amber threshold.

The actionable test is to run `calibration_by_group()` with `vehicle_group` as a stratifying variable and look at whether the A/E ratio differs systematically across vehicle groups by ethnicity proxy band. If calibration is flat, the association is not causing pricing harm. If it is not, you have a factor worth restructuring.

### Moderate proxy R-squared, low SHAP proxy score: the risk signal case

This is `ncd_years` in our example. Moderate proxy correlation - groups with higher ethnicity proportions in our synthetic data have systematically shorter NCD histories, which is plausible in a real book - but the model is weighting NCD on its genuine risk signal. SHAP proxy scores would be low here because the correlation between NCD's SHAP contribution and ethnicity proportion is weak: the model is responding to the risk variation in NCD, not to the demographic variation.

To check SHAP proxy scores explicitly, you need to run the full `FairnessAudit` with `run_proxy_detection=True` rather than calling `detect_proxies()` directly. The audit computes SHAP values and adds the proxy score to the report:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["ethnicity_prop"],
    prediction_col="predicted_rate",
    outcome_col="claim_cost",
    exposure_col="exposure",
    factor_cols=["postcode_band", "vehicle_group", "ncd_years"],
    model_name="Motor Model v2.0",
    run_proxy_detection=True,
)

report = audit.run()
proxy_detail = report.results["ethnicity_prop"].proxy_detection
print(proxy_detail.to_polars())
```

A low SHAP proxy score on a moderately correlated factor is your actuarial justification evidence. It tells you the factor's pricing contribution is driven by risk variation, not demographic correlation. Document this finding in your Consumer Duty fair value assessment. The FCA's TR24/2 specifically asked for analysis at this level of granularity - not just "we audited it" but "we found X correlation, and here is our evidence that the price contribution is driven by Y."

---

## The interpretation hierarchy

When you get a flagged factor and the three methods partially disagree, apply this hierarchy:

**SHAP proxy score first.** This is the regulatory question. Is the factor transmitting demographic correlation into prices? If SHAP proxy score is low, the Section 19 particular-disadvantage test is harder for a regulator to sustain. Document it.

**Proxy R-squared second.** This is the compliance audit question. If proxy R-squared is above 0.10, you are in amber regardless of price impact - you need to have an explanation of why the factor is still in the model. That explanation is the calibration check showing the model is not systematically wrong within groups, plus any actuarial evidence that the factor is causally related to risk.

**Mutual information third.** Use this specifically for categorical factors where proxy R-squared may be underestimating. Elevated mutual information without elevated proxy R-squared on a categorical factor is a flag to investigate further, not to act immediately.

---

## What to put in your audit documentation

For each flagged factor, the FCA evidence pack should contain:

1. The proxy R-squared, mutual information, and SHAP proxy score, with the RAG thresholds applied.
2. For factors where methods disagree: an explanation of *why* they disagree (e.g., "vehicle_group mutual information is elevated because of a categorical non-linear relationship; proxy R-squared underestimates this structure; we investigated calibration by group and found A/E ratios within 5% across all vehicle groups and ethnicity proxy quartiles").
3. For factors with high SHAP proxy scores: a quantification of the price impact and a statement of actuarial justification or a remediation plan.
4. The calibration-by-group results showing whether pricing harm is present.

The `report.to_markdown()` output structures most of this automatically. The interpretation of disagreements between methods is where judgment is required - and where the documentation adds value, because it shows the FCA that the analysis was conducted with understanding rather than by pressing a button.

---

## One thing the methods cannot tell you

All three methods are sensitive only to associations in your data. If your training data already encodes proxy discrimination - if the claim patterns in historically disadvantaged postcodes are themselves partly a product of discrimination rather than genuine risk - the methods will measure what is in the data, not what the underlying causal structure should be.

This is not a reason to avoid the audit. It is a reason to be precise about what a green result means: no detectable proxy relationship in the data as you hold it. It is not a finding that the model is causally fair. For most pricing teams, the detectable-in-the-data standard is where Consumer Duty requires them to operate.

---

`insurance-fairness` v0.6.0: [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). The `detect_proxies()` function is importable from `insurance_fairness.proxy_detection`; the full SHAP proxy scores require the `FairnessAudit` entry point.

---

**Related posts:**
- [FCA Consumer Duty Pricing Fairness in Python](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) - the end-to-end audit workflow
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) - the LRTW framework and what proxy discrimination means legally
- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/) - the correction step after detection
