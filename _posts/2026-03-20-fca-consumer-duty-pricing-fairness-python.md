---
layout: post
title: "FCA Consumer Duty Pricing Fairness in Python"
date: 2026-03-20
featured: true
author: Burning Cost
categories: [compliance, fairness, libraries, tutorials]
description: "The FCA expects pricing teams to demonstrate their models don't proxy-discriminate under Consumer Duty. Most teams do this in Excel. Here is how to do it properly in Python, using insurance-fairness."
tags: [FCA, Consumer-Duty, fairness, proxy-discrimination, insurance-fairness, FairnessAudit, ProxyDetection, calibration, disparate-impact, Equality-Act, PRIN-2A, python, uk-insurance, tutorial]
---

There is a specific obligation in PRIN 2A.9 that most UK pricing teams are not meeting, and the FCA has been explicit about this in its Consumer Duty guidance and supervisory work. The obligation is to monitor whether your pricing model produces systematically different outcomes for customers with different protected characteristics - age, gender, disability, ethnicity - and to produce quantitative evidence of that monitoring on a regular basis.

Most teams know about this obligation. What they do not have is a reliable way to discharge it. The typical approach is some combination of: an Excel workbook with A/E ratios by age band, a paragraph in the fair value assessment explaining why the model is actuarially justified, and possibly an R script somebody wrote in 2023 that no longer runs cleanly. The FCA looked at what firms were producing and called it "high-level summaries with little substance." The FCA's multi-firm review of Consumer Duty implementation (2024) found that most firms were producing "high-level summaries with little substance" — a finding that preceded the enforcement actions following the Citizens Advice 2022 ethnicity penalty report, two of which named insurers specifically on fair value grounds.

This post shows how to run a proper proxy discrimination audit in Python using `insurance-fairness`. The API is built around the specific evidence the FCA needs, not around generic ML fairness benchmarks.

---

## What the FCA actually requires

Before the code: what does a defensible Consumer Duty fair value assessment need to contain?

The FCA has not issued a prescriptive checklist. What it has said, in FG22/5 and its subsequent supervisory work, is that monitoring must be ongoing, segmented by customer group, and backed by quantitative evidence. The two distinct obligations that are easy to conflate:

**PRIN 2A.9 (fair value monitoring)** requires evidence that pricing outcomes - not just pricing processes - are fair across groups. An A/E ratio of 1.0 for females and 0.95 for males does not mean your model is discriminating. It might mean males are less profitable because of genuine risk differences. But you need to know the number, have a view on whether it is acceptable, and document that view.

**Equality Act 2010, Section 19 (indirect discrimination)** requires that no rating factor puts persons sharing a protected characteristic at a particular disadvantage without justification. Postcode is the canonical case in UK insurance. Citizens Advice estimated a £280/year ethnicity penalty in motor insurance in 2022, driven by postcodes that encode group identity. The postcode factor in your tariff may be actuarially clean and still constitute indirect discrimination under Section 19.

The proxy discrimination problem is what makes this hard. A model that does not use gender as a rating factor can still discriminate by gender if vehicle group, occupation band, or payment method are correlated with gender and the model weights those factors heavily. Proving or disproving that this is happening requires more than correlation checks - you need to trace the proxy relationship from factor to protected attribute to price impact.

---

## Setup

```bash
uv add insurance-fairness catboost polars
```

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import FairnessAudit
from insurance_fairness import detect_proxies, calibration_by_group, demographic_parity_ratio
```

---

## Build a synthetic UK motor portfolio

We use synthetic data here so the example is self-contained. The structure mirrors a real UK motor book: 10,000 policies, CatBoost frequency model, and a deliberate proxy relationship baked in - postcode district correlates with gender, which then feeds into claim cost. This is the mechanism the library needs to detect.

```python
rng = np.random.default_rng(42)
n = 10_000

# Rating factors
vehicle_age       = rng.integers(1, 16, n).astype(float)
driver_age        = rng.integers(17, 80, n).astype(float)
ncd_years         = rng.integers(0, 10, n).astype(float)
vehicle_group     = rng.choice(["A", "B", "C", "D", "E"], size=n)
postcode_district = rng.choice(
    ["SW1", "E1", "N1", "SE1", "W1", "EC1", "WC1", "M1", "B1", "LS1"],
    size=n,
)
exposure = rng.uniform(0.3, 1.0, n)

# Gender (protected characteristic — not in model)
# Drawn with postcode-conditional probability so postcode becomes a genuine proxy.
# Central London postcodes have higher male concentrations in this synthetic book.
postcode_male_prob = {
    "SW1": 0.65, "W1": 0.62, "EC1": 0.60,
    "WC1": 0.55, "N1": 0.52, "SE1": 0.50,
    "E1":  0.48, "M1": 0.47, "B1": 0.45, "LS1": 0.43,
}
p_male = np.array([postcode_male_prob[pc] for pc in postcode_district])
gender = np.where(rng.random(n) < p_male, "M", "F")

# Claim cost: genuine risk differences, plus a gender effect that is now partially
# recoverable via postcode — the proxy channel
claim_cost = np.exp(
    4.2
    + 0.04 * vehicle_age
    - 0.008 * ncd_years
    - 0.005 * driver_age
    + 0.10 * (gender == "M").astype(float)  # true gender effect
    + rng.normal(0, 0.45, n)
)

# Fit CatBoost on observable factors only (no gender)
X = np.column_stack([vehicle_age, driver_age, ncd_years])
y = claim_cost / exposure

model = CatBoostRegressor(iterations=200, depth=4, verbose=0, random_seed=42)
model.fit(X, y, sample_weight=exposure)

predicted_rate = model.predict(X)

df = pl.DataFrame({
    "gender":            gender,
    "vehicle_age":       vehicle_age,
    "driver_age":        driver_age,
    "ncd_years":         ncd_years,
    "vehicle_group":     vehicle_group,
    "postcode_district": postcode_district,
    "exposure":          exposure,
    "claim_cost":        claim_cost,
    "predicted_rate":    predicted_rate,
})
```

---

## Run the full fairness audit

`FairnessAudit` is the main entry point. Pass it the fitted model, the policy-level data, and the list of protected characteristic columns to audit. The `factor_cols` argument controls which rating factors are screened for proxy relationships - include every factor in your production tariff, not just the model inputs.

```python
audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",   # rate per unit exposure, not total premium
    outcome_col="claim_cost",
    exposure_col="exposure",
    factor_cols=[
        "postcode_district",
        "vehicle_age",
        "ncd_years",
        "vehicle_group",
        "driver_age",
    ],
    model_name="Motor Frequency Model v3.1",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()
```

The console output gives you the headline RAG status and per-characteristic metrics. The substantive output is the Markdown file:

```python
report.to_markdown("fair_value_evidence_2026q2.md")
```

This file contains: proxy detection results for all five rating factors, calibration by group with exposure-weighted A/E ratios across ten premium deciles, demographic parity ratio in log-space, disparate impact ratio, and a regulatory mapping section linking each finding to PRIN 2A.9 and Equality Act Section 19. There is a sign-off table at the end for the certifying actuary.

---

## Proxy detection in detail

If you want to inspect the proxy detection results without running the full audit, `detect_proxies` is available as a standalone function:

```python
from insurance_fairness import detect_proxies

proxy_result = detect_proxies(
    df,
    protected_col="gender",
    factor_cols=[
        "postcode_district",
        "vehicle_age",
        "ncd_years",
        "vehicle_group",
        "driver_age",
    ],
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
)

print(proxy_result.flagged_factors)
print(proxy_result.to_polars())
```

The detection uses three methods simultaneously:

**Proxy R-squared**: a CatBoost model predicts the protected characteristic from each rating factor alone. A proxy R-squared above 0.05 is amber; above 0.10 is red. This threshold structure comes from the library's calibration against known discrimination cases - the 0.10 threshold was the level at which the Citizens Advice postcode-ethnicity relationship became detectable in the 2022 dataset.

**Mutual information**: model-free, captures non-linear relationships that R-squared misses. Particularly useful for categorical factors like postcode district, where the relationship with a protected group is categorical rather than monotone.

**SHAP proxy scores**: the Spearman correlation between a factor's SHAP contribution to the predicted price and the protected characteristic. This is the link that matters for the Section 19 justification: a factor can correlate with a protected characteristic (high proxy R-squared) but contribute negligible price variation (low SHAP proxy score). The Section 19 argument is strongest when SHAP proxy score is also high - the factor is not just correlated with the protected characteristic, it is transmitting that correlation into prices.

The Spearman correlation check that most teams currently run will return nothing on `postcode_district` because the relationship is categorical and non-linear. CatBoost proxy R-squared catches it. That is why the FCA's thematic review found firms' manual checks insufficient: the checks were not designed to detect the relationships that matter.

---

## Calibration by group

The most defensible evidence format for a PRIN 2A.9 compliance file is calibration by group - whether the model is equally well-calibrated across protected groups at each premium level. A model with A/E = 0.95 overall for females might look acceptable, but if the under-prediction is concentrated at high predicted premiums, that is a systematic overcharge of high-risk female policyholders.

```python
from insurance_fairness import calibration_by_group

cal = calibration_by_group(
    df,
    protected_col="gender",
    prediction_col="predicted_rate",
    outcome_col="claim_cost",
    exposure_col="exposure",
    n_deciles=10,
)

print(f"Max calibration disparity: {cal.max_disparity:.4f}  [{cal.rag}]")
# Largest (A/E_female / A/E_male - 1) across all (decile, group) cells,
# exposure-weighted
```

The `max_disparity` compares the female and male A/E ratios within each predicted-premium decile and takes the maximum absolute difference, exposure-weighted. A value below 0.05 is green; 0.05–0.15 is amber; above 0.15 is red. These are not FCA thresholds - the FCA has not issued any - but they are calibrated against what a supervision team would want to investigate.

---

## Three things most teams get wrong

**Testing on the wrong outcome metric.** The Consumer Duty obligation under Outcome 4 is about fair value: whether the product delivers equivalent financial outcomes to different groups. Testing calibration of a frequency model tells you whether the model is unbiased at predicting claims, not whether policyholders are getting fair value. Run the calibration check on total premium relative to total claims, with exposure weighting, not on model output alone.

**Ignoring intersectional groups.** The Equality Act 2010 prohibits discrimination on the basis of individual protected characteristics. PRIN 2A.9 requires monitoring of groups. These are not the same thing. A model that is well-calibrated for females overall may be poorly calibrated for young female drivers specifically, and the aggregate A/E check will not surface it. If you have sufficient exposure, run `calibration_by_group` on cross-products of your protected characteristics - not just gender and age separately, but (gender × age band) jointly. The library supports this via the `protected_col` argument accepting a pre-constructed interaction column.

**Treating the audit as a one-time exercise.** The FCA's Consumer Duty supervisory reviews found firms producing fair value assessments that were undated, not reproducible, and disconnected from the models actually in production. The output from `FairnessAudit.run()` includes the model name, the run date, the policy count, and version metadata from the data. Run it as part of your quarterly model review cycle. Put the Markdown file in your model documentation repository alongside the model artefacts. The FCA's expectation is ongoing monitoring, not an annual checkbox.

---

## What a "green" result actually means

A green RAG status from `FairnessAudit` means: no rating factor has proxy R-squared above 0.10 for any protected characteristic, the maximum calibration disparity across (protected group × premium decile) cells is below 0.05, and the demographic parity log-ratio is within ±0.05. It does not mean your model definitely does not discriminate. It means the tests we can run on observable data do not find evidence of discrimination at the standard thresholds.

The honest framing for a pricing committee paper: "Our proxy discrimination audit for Q2 2026 found no rating factors with proxy R-squared above 0.10 for gender, and maximum calibration disparity of X across (gender × predicted premium decile) cells. This is consistent with the model not discriminating on gender. We note that the audit cannot detect discrimination by characteristics we do not hold in our policy data - ethnicity, religion, disability status - and we address this in a separate analysis using ONS Census 2021 LSOA-level ethnicity proxies." That framing is more defensible than a paragraph asserting the model is actuarially justified.

---

[`insurance-fairness`](/insurance-fairness/) v0.6.0 is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). Python 3.10+. The full API - including `MulticalibrationAudit`, `DoubleFairnessAudit` for Consumer Duty Outcome 4, and `PrivatizedFairnessAudit` for books where you lack protected attribute data - is in the README.

---

**Related posts:**
- [The FCA Is Investigating Home and Travel Insurers: Is Your Pricing Model Defensible?](/2026/03/19/the-fca-is-investigating-home-and-travel-insurers/) - the regulatory context, what the FCA's Consumer Duty review actually found, and what a complete evidence pack requires
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) - what to do when you cannot observe ethnicity, religion, or disability status directly
- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/) - the correction step after detection: WassersteinCorrector and causal path decomposition
