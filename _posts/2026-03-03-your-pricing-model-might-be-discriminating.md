---
layout: post
title: "Proxy Discrimination in UK Motor Pricing: Detection and Correction"
date: 2026-03-03
categories: [techniques, compliance]
tags: [fairness, proxy-discrimination, FCA, Consumer-Duty, Equality-Act, LRTW, GBM, postcode, python, motor, insurance-fairness]
description: "Detect and correct proxy discrimination in UK insurance using SHAP and insurance-fairness. Protected characteristic leakage detection under FCA Consumer Duty."
---

Your GBM uses postcode as a feature. That is completely standard practice in UK motor insurance. Postcode predicts claim frequency. The model validation looks fine.

But postcode also correlates with ethnicity. In London, in particular, postcode bands are not race-neutral. The ONS Census 2021 shows that Inner East London LSOAs have non-white British populations above 70%. Your model does not explicitly use ethnicity. It does not need to. Postcode is doing that job for it.

This is proxy discrimination. It is the subject of Equality Act 2010 Section 19. And as of July 2023, it is directly in scope of FCA Consumer Duty, specifically PRIN 2A.4, which requires firms to monitor whether their products provide fair value across groups defined by protected characteristics.

The FCA has signalled this is a priority area. In TR24/2 (August 2024), they found that fair value assessments from reviewed firms were too high level and lacked the granularity to adequately evidence good outcomes across customer groups. The next step is enforcement.

We built [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) to give pricing teams an audit-ready answer to this problem. This post explains the regulatory exposure, the technical problem of proxy discrimination in insurance models, and how to use the library to measure and mitigate it.

---

## Why "we don't use race" is not a defence

The intuition that discrimination requires intent - that using a protected characteristic explicitly is the only thing the law prohibits - is wrong, and has been wrong in English law since the Race Relations Act 1976. Section 19 of the Equality Act 2010 defines indirect discrimination as: applying a provision, criterion or practice (PCP) that puts people sharing a protected characteristic at a particular disadvantage, where you cannot justify it as a proportionate means of achieving a legitimate aim.

Pricing models do not need to include race. They need to produce premium predictions that correlate with race to create indirect discrimination.

Citizens Advice published analysis in 2022 finding that motor insurance customers in postcodes where more than 50% of residents are people of colour paid an average of £280 more per year than otherwise similar customers in majority-white postcodes. Their estimate of the total annual excess was £213 million. The FCA took that analysis seriously.

The legitimate aim defence - actuarial justification that the pricing factor genuinely reflects risk, not demographic correlation - is available, but it requires you to demonstrate it. Not to assert it. The regulator will want to see the work.

Proxy detection is that work.

---

## What proxy discrimination looks like in a GBM

Consider a standard UK motor frequency model: CatBoost, Poisson objective, features including driver age, vehicle group, NCD, occupation, and postcode area.

The model has never seen ethnicity. It cannot, because you do not collect it. But it has seen postcode, and postcode in your training data correlates with both claim frequency and, through the demographic composition of postcode areas, ethnicity.

When you train the model, it learns:
1. Claim frequency varies by postcode (true - driven by urban density, traffic, theft rates, road quality).
2. Some of that postcode effect captures risk-relevant variation (the causal channel you want).
3. Some of it captures demographic correlation with protected characteristics (the proxy discrimination channel you do not want).

The GBM cannot distinguish these. It optimises Poisson deviance. It does not know or care that postcode is a proxy for ethnicity as well as a predictor of risk. The resulting model bakes both effects into the postcode feature importance.

The same logic applies to occupation in home insurance (manual occupations correlate with socioeconomic status, which correlates with both risk and protected characteristics), to vehicle group in motor (performance vehicles correlate with age, income, and indirectly race and disability), and to annual mileage (self-reported mileage is lower in groups that work from home, which correlates with occupation, which correlates with protected characteristics).

Every rating factor that correlates with a protected characteristic is a potential proxy discrimination channel. Postcode is the most visible one, but it is not the only one.

---

## The LRTW framework: what you actually need to measure

The academic literature on fair insurance pricing converged on a specific framework: Lindholm, Richman, Tsanakas and Wüthrich, published in ASTIN Bulletin 2022, with three subsequent papers extending it through 2024.

The LRTW definition: a pricing model f(X) exhibits proxy discrimination if it is not conditionally independent of a protected attribute S given the observed rating factors X.

In plain terms: if knowing a customer's ethnicity still gives you information about what your model will charge them - even after you know all their risk factors - your model is discriminating.

The LRTW discrimination-free price is defined as:

```
p_DF(x) = E[f(X, S) | X = x] = integral f(x, s) * p(s | x) ds
```

This marginalises the model's best-estimate price over the conditional distribution of the protected attribute given the non-protected features. It is the price that uses risk information without implicitly using protected-characteristic information.

Three things that follow from this that matter for your audit:

**Unawareness is not discrimination-free.** Simply leaving S out of the model does not fix proxy discrimination when X correlates with S. The correct fix is to know the conditional distribution p(s | x) - typically from ONS Census data linked to postcodes - and use it to marginalise.

**Proxy discrimination is not the same as demographic disparity.** A discrimination-free model will still charge different average premiums to different demographic groups, because risk genuinely correlates with protected characteristics. The law is concerned with proxy discrimination - charging more than the risk-based rate implies because of demographic correlation - not with demographic disparities themselves. Libraries that implement demographic parity (equal average premiums across groups) are solving the wrong problem for UK insurance.

**Actuarial justification requires decomposition.** A regulator who asks "why does this group pay more?" needs you to be able to say "this much is genuine risk differential, this much is unexplained by risk." The LRTW framework gives you that decomposition. A raw comparison of average premiums by ethnicity does not.

---

## Installing the library

```bash
uv add insurance-fairness
```

The library requires a trained CatBoost model and a Polars DataFrame. It assumes a Poisson or Tweedie frequency model with a log link.

---

## Step 1: measuring proxy correlation

Before you compute discrimination-free prices, you need to know which features are proxies and how strongly they correlate with protected characteristics. The library's primary entry point is `FairnessAudit`, which runs a full audit in one call. For more granular control, you can call `detect_proxies()` directly.

```python
import polars as pl
from insurance_fairness import FairnessAudit
from catboost import CatBoostRegressor

# df_train is your training data; postcode_district is linked to ONS Census
# ethnicity_prop is proportion of non-white-British population at LSOA level,
# from ONS Census 2021, joined via postcode -> LSOA lookup

model = CatBoostRegressor()
model.load_model("frequency_model.cbm")

audit = FairnessAudit(
    model=model,
    data=df_train,
    protected_cols=["ethnicity_prop"],
    prediction_col="predicted_freq",
    outcome_col="claim_count",
    exposure_col="policy_years",
    factor_cols=["postcode_district", "vehicle_group", "occupation", "ncd_years", "age_band"],
    model_name="UK Motor Frequency v3.2",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()
```

```
============================================================
Fairness Audit: UK Motor Frequency v3.2
Date: 2026-03-07
Policies: 243,891 | Exposure: 201,432.7
Overall status: AMBER
============================================================

Protected characteristic: ethnicity_prop
----------------------------------------
  Demographic parity log-ratio: +0.2714 (ratio: 1.3117) [AMBER]
  Max calibration disparity: 0.0841 [GREEN]
  Disparate impact ratio: 0.7623 [AMBER]
  Flagged proxy factors (2): postcode_district, occupation
```

The audit also exposes the proxy detection results in a structured form. To see the full per-factor breakdown:

```python
proxy_result = report.results["ethnicity_prop"].proxy_detection
proxy_df = proxy_result.to_polars()
print(proxy_df.select(["factor", "proxy_r2", "mutual_information", "partial_correlation", "rag"]))
```

```
shape: (5, 5)
┌────────────────────┬──────────┬───────────────────┬────────────────────┬───────┐
│ factor             ┆ proxy_r2 ┆ mutual_information ┆ partial_correlation ┆ rag   │
│ ---                ┆ ---      ┆ ---                ┆ ---                 ┆ ---   │
│ str                ┆ f64      ┆ f64                ┆ f64                 ┆ str   │
╞════════════════════╪══════════╪═══════════════════╪════════════════════╪═══════╡
│ postcode_district  ┆ 0.3847   ┆ 0.2913             ┆ 0.5821              ┆ red   │
│ occupation         ┆ 0.1134   ┆ 0.0872             ┆ 0.2247              ┆ amber │
│ vehicle_group      ┆ 0.0621   ┆ 0.0514             ┆ 0.1183              ┆ green │
│ age_band           ┆ 0.0198   ┆ 0.0231             ┆ -0.0412             ┆ green │
│ ncd_years          ┆ 0.0087   ┆ 0.0119             ┆ 0.0203              ┆ green │
└────────────────────┴──────────┴───────────────────┴────────────────────┴───────┘
```

The proxy R-squared for `postcode_district` is 0.38 - a CatBoost model trained on postcode alone explains 38% of the variance in the ethnicity proportion proxy. That is not marginal. Occupation at 0.11 sits just above the amber threshold (the library flags anything above 0.10). Age and NCD are not significant proxies here.

This table is your evidence base. If you can show the regulator that occupation had proxy R-squared of 0.11 and that you investigated it, that is a substantially better position than having no record of the analysis.

---

## Step 2: measuring disparate impact

The `FairnessReport` already contains the disparate impact result. You can also inspect calibration-by-group:

```python
# Disparate impact ratio
di = report.results["ethnicity_prop"].disparate_impact
print(f"Disparate impact ratio: {di.ratio:.4f} [{di.rag.upper()}]")
print(f"Group means: {di.group_means}")

# Calibration by group — actual-to-expected ratios by decile
cal = report.results["ethnicity_prop"].calibration
print(f"Max calibration disparity from A/E=1: {cal.max_disparity:.4f} [{cal.rag.upper()}]")
```

```
Disparate impact ratio: 0.7623 [AMBER]
Group means: {'0': 0.0412, '1': 0.0540}
Max calibration disparity from A/E=1: 0.0841 [GREEN]
```

The disparate impact ratio compares mean predicted frequency between groups. A ratio below 0.80 is flagged amber - not because the US four-fifths rule applies in UK law (it does not), but as a diagnostic threshold. The calibration check is separately green: the model is not systematically miscalibrated within groups, which means the disparity in mean predictions largely reflects genuine risk variation rather than model error.

The key discipline here: a disparate impact ratio of 0.76 is not automatically discrimination. Risk genuinely varies by postcode. The question is how much of the disparity is genuine risk variation versus demographic correlation. Use the discrimination-free pricing step in `insurance-fairness.optimal_transport` to decompose it - that library takes these audit findings and produces corrected premiums.

---

## Step 3: producing the audit report

The library outputs structured data rather than a PDF - the right format depends on your governance infrastructure:

```python
import json

# Save a Markdown report for model governance documentation
report.to_markdown("fairness_audit_2026Q1.md")

# Save a JSON record for MI submissions and audit trail
audit_dict = report.to_dict()
with open("fairness_audit_2026Q1.json", "w") as f:
    json.dump(audit_dict, f, indent=2)

# Print the report status and flagged factors
print(f"Overall RAG: {report.overall_rag.upper()}")
print(f"Flagged factors: {report.flagged_factors}")
```

```
Overall RAG: AMBER
Flagged factors: ['occupation', 'postcode_district']
```

The Markdown report covers: the protected characteristics assessed, the proxy R-squared and mutual information scores per rating factor, the disparate impact measurements, the calibration-by-group results, and the overall RAG status. The JSON output is machine-readable and supports structured submissions to your Chief Actuary sign-off process.

---

## The fairness-accuracy trade-off is real, and smaller than you think

The common objection to discrimination-free pricing: it will hurt your loss ratio. If you remove the postcode-ethnicity correlation from your prices, you will misprice some risks.

This is true, in theory. In practice, the magnitude is smaller than the theory implies, for two reasons.

First, the genuine risk information in postcode - urban density, road quality, theft rates, traffic patterns - is not removed. Only the component of the postcode effect that is not explained by risk-relevant features but is explained by demographic composition gets adjusted. If your model already includes urban density, road type, and crime statistics, the residual postcode-ethnicity correlation being removed is smaller.

Second, the calibration check in the audit already shows you whether the model is miscalibrated within groups. If the max calibration disparity is 0.08 (as in the example above), the model's predictions are well-calibrated for each group - the pricing disparity reflects genuine risk, not systematic mispricing. That is the output that justifies the analysis to a regulator.

The library's `calibration_by_group()` function can be called independently on a holdout period to verify that calibration holds out-of-sample:

```python
from insurance_fairness import calibration_by_group

cal_holdout = calibration_by_group(
    df=df_holdout,
    protected_col="ethnicity_prop",
    prediction_col="predicted_freq",
    outcome_col="claim_count",
    exposure_col="policy_years",
    n_deciles=10,
)
print(f"Holdout max calibration disparity: {cal_holdout.max_disparity:.4f} [{cal_holdout.rag.upper()}]")
```

When the proxy discrimination component is genuinely a proxy and not causal, the discrimination-free adjustment should not materially damage predictive accuracy. If calibration deteriorates sharply after the adjustment, that is information: postcode's predictive power is not purely proxy. You should investigate what genuine causal channel postcode is capturing that your other features are not - and consider adding those features (urban density index, road quality scores, telematics) to reduce the proxy correlation at source.

---

## What a pricing team should actually do

The practical question is not "is our model perfectly discrimination-free" - it is not, and neither is anyone else's. The practical question is "can we demonstrate that we have taken this seriously."

Our recommended sequence:

**1. Run the full audit on your current model.** A single `FairnessAudit(...).run()` call gives you proxy R-squared for all rating factors, disparate impact ratios, and calibration-by-group. The ONS Census 2021 postcode-to-ethnicity linkage is the starting point for motor. You need an hour of data engineering, not weeks.

**2. Examine the flagged factors.** Anything with proxy R-squared above 0.10 goes into the `report.flagged_factors` list. For each flagged factor, understand the evidence: is the proxy correlation driving genuine risk signal or demographic correlation? The calibration-by-group check is your primary test.

**3. Document everything.** The FCA in TR24/2 was explicit: they want adequate granularity and evidence of good outcomes. "We looked and found it was fine" is not adequate. "We ran these specific tests, found these specific results, and took these specific actions" is. Use `report.to_markdown()` to produce the evidence document.

**4. If the proxy component is material, compute discrimination-free prices.** The `insurance-fairness.optimal_transport` subpackage handles this: it takes a CausalGraph specifying which variables are proxies versus justified mediators, and applies the Lindholm marginalisation to produce corrected premiums. See [Discrimination-Free Pricing in Python](/2026/03/10/insurance-fairness-ot/) for that workflow.

**5. Brief the Chief Actuary and your Consumer Duty owner.** This is not a modelling curiosity. It is a regulated obligation with enforcement behind it. The right people need to be informed of the findings.

What you should not do: wait until the FCA writes to you. The £213 million annual excess identified by Citizens Advice represents a real regulatory target. The firms that can produce audit-ready documentation quickly are the ones that have done this work before they were asked.

---

## On the limits of the approach

Three things to be honest about.

**The protected proxy is imperfect.** We use postcode-level ONS ethnicity proportions as a proxy for individual protected characteristics because insurers do not collect individual-level ethnicity data. This means our estimates of p(S | X) are noisy, particularly in mixed postcodes. The proxy audit will underestimate proxy correlation where demographic mixing is high. That is a conservative bias - actual proxy discrimination may be higher than measured.

**Multiple protected characteristics.** This post focuses on ethnicity because it has the most prominent evidence base (the Citizens Advice analysis) and the clearest postcode proxy. The same methods apply to disability, religion, and sex - pass multiple column names to `protected_cols` and `FairnessAudit` runs them all. But building the proxies requires different data sources and is harder. The library supports custom proxy columns: if you have a disability proxy from your distribution data, you can use it.

**The LRTW framework assumes your model is predictively correct.** The audit checks calibration within groups, but if training data is sparse for some demographic groups, the model may be miscalibrated within those groups in ways the decile analysis does not catch. Both proxy discrimination and within-group miscalibration are worth solving; they require different tools.

---

## Getting started

```bash
uv add insurance-fairness
```

Source and issue tracker on [GitHub](https://github.com/burning-cost/insurance-fairness). The library requires CatBoost models and Polars DataFrames; postcode-to-LSOA linkage data and ONS Census 2021 ethnic group tables at LSOA level are available from the ONS open data portal.

Start with a `FairnessAudit` on your current production model. If `report.overall_rag` comes back green and the flagged_factors list is empty, document that and file it. If the overall status is amber or red, you need to understand the disparity decomposition before your next Consumer Duty review.

The fairness-accuracy trade-off is real. But "we did not audit this because we were worried about our loss ratio" is not a position you can sustain with the regulator. The Citizens Advice analysis is public. The FCA's expectations are documented. The question is not whether you will address this - it is whether you address it now or after a letter arrives from Stratford.

- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/)
- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/)
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
