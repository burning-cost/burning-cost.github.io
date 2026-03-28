---
layout: post
title: "FCA Proxy Discrimination Testing in Python: A Practical Guide"
date: 2026-03-22
categories: [fairness, regulation, techniques]
tags: [fairness, FCA, proxy-discrimination, EP25/2, Consumer-Duty, insurance-fairness, python, Equality-Act, pricing, audit]
description: "How to run a proxy discrimination test on an insurance pricing model in Python: EP25/2 requirements, the insurance-fairness library, factor-level proxy detection, and FCA-ready audit output. The definitive practical tutorial."
seo_title: "Proxy Discrimination Testing for Insurance Pricing in Python"
---

The FCA's Evaluation Paper EP25/2, published in early 2025, is the most specific regulatory guidance yet on what a proxy discrimination audit for a non-life pricing model should look like. It is not Consumer Duty in general. It is not the Gender Directive. It is the question: do your rating factors act as conduits for protected characteristics you are not permitted to use directly?

This post explains what EP25/2 actually requires, why the general-purpose fairness tooling you might reach for first does not answer that question, and how to run a conformant proxy discrimination audit in Python using the `insurance-fairness` library.

---

## What EP25/2 actually requires

EP25/2 does not specify a test statistic or a software tool. What it specifies is the question firms need to be able to answer, and the standard of evidence required to answer it.

The core obligation, in plain terms: a firm using a pricing model must be able to demonstrate that its rating factors do not systematically proxy for protected characteristics in a way that produces discriminatory pricing outcomes. The word "systematically" matters - the FCA is not looking for zero correlation (which is impossible, since almost every rating factor correlates with something that correlates with something protected). It is looking for material correlation that has not been investigated and documented.

Three things follow from this that shape what a technical audit needs to produce.

**First, the test is about factor-level correlation, not output-level disparity.** A model that charges different average premiums to men and women is not necessarily discriminating - risk differs. A model that uses a factor which is a strong predictor of gender, without having considered that relationship, is a problem. EP25/2 wants you to examine the inputs, not just compare the group average outputs.

**Second, conditional independence is the legal standard.** The Lindholm, Richman, Tsanakas and Wüthrich (LRTW) framework, now the academic reference point for FCA enforcement discussions, defines discrimination as: a pricing model that is not conditionally independent of a protected attribute S given the observed rating factors X. In other words, if knowing a customer's protected characteristic tells you something about what the model will charge them - even after you know all their risk factors - the model is discriminating. This is a stricter and more precise test than demographic parity.

**Third, the audit trail is part of the obligation.** EP25/2 specifically criticised the quality of evidence submitted by firms in the preceding thematic review, TR24/2 (August 2024). "Too high level and lacking the granularity to adequately evidence good outcomes" is the FCA's summary of what it received. The audit needs to be reproducible, factor-level, and documented - not a one-line assertion that the model was reviewed.

---

## Why Fairlearn and AIF360 do not fit

The two most widely used Python fairness libraries are Fairlearn (Microsoft) and AIF360 (IBM). Both are well-built. Neither was designed for this problem.

**They implement demographic parity.** The core metric in both libraries is a comparison of model outputs across protected groups - are the average predictions, positive prediction rates, or error rates similar across groups? This is the right question for fraud classification or credit approval. For insurance pricing, it is the wrong question. UK insurers are explicitly permitted to charge different average premiums to different groups when risk differs. A pricing model that achieves demographic parity is not compliant - it is probably mispricing some risks.

**They do not test rating factor inputs.** Fairlearn's `MetricFrame` and AIF360's `ClassificationMetric` both take model outputs and ask how those outputs vary by protected characteristic. They cannot tell you whether `occupation` carries ethnicity information, or whether `postcode_district` is a stronger predictor of the protected characteristic than it is of claim frequency. Factor-level proxy detection - the thing EP25/2 requires - is not in either library's scope.

**They do not handle exposure weights.** Insurance portfolios are not rows. A single-month policy contributes a twelfth of the exposure of an annual policy on the same risk. Both libraries treat every row equally. For calibration-by-group or demographic parity computations in insurance, unweighted metrics are wrong.

**They do not produce insurance-specific audit outputs.** A Fairlearn `MetricFrame` is not an FCA evidence document. It does not contain a regulatory mapping, a RAG status, or the factor-level proxy scores that a pricing committee sign-off requires. The output format matters: if you want to put something in a model risk register or send it to a Consumer Duty owner, you need a structured document, not a Python object.

The right tool is one built for the specific regulatory question: do your non-protected rating factors proxy for protected characteristics? That is what [`insurance-fairness`](/2026/03/20/fca-consumer-duty-pricing-fairness-python/) is for.

---

## The three-step workflow

Install the library:

```bash
uv add insurance-fairness
```

The library requires a trained CatBoost model and a Polars DataFrame. It produces a structured `FairnessReport` with RAG statuses, factor-level proxy scores, and a Markdown output suitable for governance documentation.

### Step 1: Fit the model and prepare the data

The audit takes a policy-level DataFrame with a prediction column already populated. The typical setup for a UK motor frequency model:

```python
import polars as pl
from catboost import CatBoostRegressor
from insurance_fairness import FairnessAudit

# Load your policy data and trained model
df = pl.read_parquet("motor_policies_2025.parquet")
model = CatBoostRegressor()
model.load_model("frequency_model_v4.cbm")

# Add predictions to the DataFrame if not already present
X = df.select(["postcode_district", "vehicle_group", "occupation", "ncd_years", "age_band"])
df = df.with_columns(
    pl.Series("predicted_freq", model.predict(X.to_pandas()))
)
```

The `ethnicity_prop` column here is a postcode-level ONS Census 2021 continuous proxy: the proportion of non-white-British residents at LSOA level, joined via postcode-to-LSOA lookup. It is a floating-point number between 0 and 1, not a binary flag. The library handles continuous protected characteristic proxies natively.

### Step 2: Run the proxy audit

```python
audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["ethnicity_prop"],
    prediction_col="predicted_freq",
    outcome_col="claim_count",
    exposure_col="policy_years",
    factor_cols=["postcode_district", "vehicle_group", "occupation", "ncd_years", "age_band"],
    model_name="Motor Frequency v4  -  Q1 2026",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()
```

The audit runs three complementary proxy detection methods for each (factor, protected characteristic) pair:

- **CatBoost proxy R-squared**: fits a gradient boosting model to predict `ethnicity_prop` from each rating factor alone, using exposure as sample weights. The held-out R-squared measures how much a single factor explains of the protected characteristic proxy. The amber threshold is 0.05; red is 0.10.
- **Mutual information**: model-free, captures non-linear dependencies. A postcode band that is not monotonically ordered with ethnicity proportion - higher in inner London, lower in the Home Counties, higher again in Bradford - will show up here when a linear or rank correlation misses it. Measured in nats.
- **Partial Pearson-residualised Spearman correlation**: the association between a factor and the protected characteristic after controlling for the other rating factors. This answers: does postcode carry ethnicity information *beyond* what vehicle group, occupation, and NCD already explain?

All three are exposure-weighted throughout.

### Step 3: Generate the audit report

```python
# Structured Markdown for governance documentation
report.to_markdown("fairness_audit_motor_2026q1.md")

# Machine-readable JSON for audit trail and MI submissions
import json
with open("fairness_audit_motor_2026q1.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

The Markdown report maps each finding to the specific FCA regulatory requirement: PRIN 2A.4 (Price and Value Outcome), TR24/2, and Equality Act 2010 Section 19. It contains the factor-level proxy scores, RAG statuses, calibration-by-group results, and the overall audit status. It is structured to go directly into a pricing committee pack or model risk register.

---

## Reading the output

The proxy detection scores for each rating factor:

```python
proxy_result = report.results["ethnicity_prop"].proxy_detection
print(proxy_result.to_polars().select(
    ["factor", "proxy_r2", "mutual_information", "partial_correlation", "rag"]
))
```

```
shape: (5, 5)
┌────────────────────┬──────────┬───────────────────┬─────────────────────┬───────┐
│ factor             ┆ proxy_r2 ┆ mutual_information ┆ partial_correlation ┆ rag   │
│ ---                ┆ ---      ┆ ---                ┆ ---                 ┆ ---   │
│ str                ┆ f64      ┆ f64                ┆ f64                 ┆ str   │
╞════════════════════╪══════════╪═══════════════════╪═════════════════════╪═══════╡
│ postcode_district  ┆ 0.3847   ┆ 0.2913             ┆ 0.5821              ┆ red   │
│ occupation         ┆ 0.0832   ┆ 0.0721             ┆ 0.1934              ┆ amber │
│ vehicle_group      ┆ 0.0621   ┆ 0.0514             ┆ 0.1183              ┆ amber │
│ age_band           ┆ 0.0198   ┆ 0.0231             ┆ -0.0412             ┆ green │
│ ncd_years          ┆ 0.0087   ┆ 0.0119             ┆ 0.0203              ┆ green │
└────────────────────┴──────────┴───────────────────┴─────────────────────┴───────┘
```

What each result means in practice:

**`postcode_district`, proxy R-squared 0.38, red.** A CatBoost model trained on postcode district alone explains 38% of the variance in the ethnicity proxy. This is not a borderline result. Postcode encodes substantial demographic information; the model's postcode relativity is doing pricing work that partially reflects ethnicity. This does not automatically mean the model is unlawfully discriminating - postcode also reflects urban density, theft rates, road quality, and traffic patterns that genuinely predict claims. But it means the factor warrants full decomposition: how much of the postcode premium variation is genuine risk, and how much is demographic correlation?

**`occupation`, proxy R-squared 0.08, amber.** Sits between the amber threshold (0.05) and the red threshold (0.10). Manual occupations correlate with socioeconomic status, which correlates with multiple protected characteristics. An amber result means: document this, monitor it, understand the evidence. It does not mean remove occupation from the model.

**`age_band` and `ncd_years`, green.** These factors have low proxy R-squared for ethnicity. That is the expected result - NCD years and driver age are not strong proxies for ethnicity in UK motor data.

The partial correlation for `occupation` (0.19) is noticeably higher than its proxy R-squared (0.08). This indicates occupation carries ethnicity-correlated information *beyond* what the other factors explain - after controlling for postcode, vehicle group, and NCD, occupation still tells you something about ethnicity. Worth investigating whether your occupation banding creates avoidable concentration.

The overall report also contains the calibration-by-group check: actual-to-expected claim rates within each decile of the protected characteristic proxy. A well-calibrated model with a high proxy R-squared for postcode means the price variation is tracking genuine risk, not systematic mispricing. A miscalibrated model on top of a high proxy R-squared is the worst outcome.

---

## What to do about amber and red results

The FCA does not require a zero-proxy model. It requires evidence of engagement.

For a **red** result (proxy R-squared above 0.10): the factor needs explicit investigation and documentation. The minimum requirement is: quantify how much of the factor's premium variation is attributable to genuine risk differentiation versus demographic correlation. The `insurance-fairness.optimal_transport` subpackage handles this - it implements the LRTW discrimination-free pricing calculation, which marginalises the model's output over the conditional distribution of the protected characteristic given the non-protected features. This produces a decomposition: here is the discrimination-free price, here is the adjustment relative to the uncorrected model, here is the magnitude of the proxy discrimination channel.

For an **amber** result: document the finding, the proxy R-squared, the mutual information, and the calibration result. Record the conclusion: either "the calibration is strong, the factor reflects genuine risk, and we are monitoring it at each model review" or "we are taking the following mitigation action." Either conclusion is defensible if it is documented.

For **green** results: the audit trail records that the test was run and passed. File it.

The specific trigger for deeper investigation should be: amber or red proxy R-squared *combined* with a calibration disparity above the amber threshold (0.10). That combination - a factor that proxies for a protected characteristic and a model that is systematically miscalibrating one demographic group - is the pattern that the FCA's enforcement concern is focused on.

If remediation is required, the options in order of preference are: (1) add features that capture the legitimate causal channel more directly, reducing the proxy correlation at source (urban density index, telematics features, road quality scores instead of raw postcode); (2) apply LRTW marginalisation to correct the prices directly; (3) document the justification that the remaining correlation reflects genuine risk variation under Equality Act 2010 Section 19 indirect discrimination's proportionate means / legitimate aim defence.

---

## Integration with model governance

The `FairnessAudit` is designed to run as part of the standard model review cycle, not as a one-off exercise. The right integration is:

```python
# Run as part of annual model review
report = audit.run()
report.to_markdown(f"governance/fairness_audit_{review_date}.md")

# Check overall status and fail the review if red
if report.overall_rag == "red":
    raise RuntimeError(
        f"Fairness audit RED status. Flagged factors: {report.flagged_factors}. "
        "Escalate to Chief Actuary before model deployment."
    )

# Append to the model risk register
print(f"Audit date: {report.audit_date}")
print(f"Overall RAG: {report.overall_rag.upper()}")
print(f"Flagged proxy factors: {report.flagged_factors}")
```

The `insurance-governance` library's model registry is designed to accept `FairnessReport` outputs as structured evidence items in the model risk register. Linking the two means that every model deployment is associated with a dated fairness audit, the audit output is retrievable on demand, and changes in RAG status between review cycles are logged automatically. See [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) for the governance integration.

The EP25/2 obligation is ongoing. The FCA is explicit that Consumer Duty monitoring requires regular review, not a single assessment at model launch. Annual re-run on the current in-force book is the minimum; quarterly on books with significant mix shifts is more defensible.

---

## The data engineering question

The biggest practical obstacle for most firms is not the statistical method - it is assembling the protected characteristic proxy column.

For ethnicity in UK motor and home insurance, the standard approach is the ONS Census 2021 ethnic group tables at Lower Layer Super Output Area (LSOA) level, joined to your policy data via postcode-to-LSOA lookup. The ONS publishes both: the `TS021` ethnic group tables and the National Statistics Postcode Lookup (NSPL). The join is a standard left join on postcode sector. The result is a floating-point ethnicity proportion per policy, which you pass as `ethnicity_prop` in the protected characteristic column. The library handles continuous protected characteristic proxies natively; you do not need to binarise.

For gender, the situation has changed since the Gender Directive (2013). Insurers no longer hold gender at quote, so building a gender proxy requires either a name-based estimation method or a proxy from fleet composition data. This is harder and the proxies are noisier.

For disability, the Equality Act 2010 definition covers a broad range of conditions that insurers do not ask about and would not be permitted to use. A disability proxy requires either self-reported data or area-level DWP statistics. The library accepts any numeric column as a protected characteristic - the statistical machinery is the same regardless of which protected characteristic the column represents.

---

The regulatory timetable here is not abstract. TR24/2 (August 2024) told the market that the quality of evidence firms were producing was inadequate. EP25/2 (2025) set the analytical standard more precisely. The next intervention from Stratford will be enforcement, not another paper.

A `FairnessAudit` on your current production model takes an afternoon of data engineering and a few minutes of compute. The output is a dated, factor-level, RAG-rated document with regulatory mapping. That is what the FCA is asking for.

```bash
uv add insurance-fairness
```

Source and issue tracker at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).

---

**Related posts:**

- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) - the LRTW framework, the Citizens Advice data, and the full audit workflow
- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) - direct comparison of why general-purpose ML fairness tools miss the FCA's specific concern
- [Discrimination-Free Pricing in Python](/2026/03/10/insurance-fairness-ot/) - LRTW marginalisation and causal path decomposition for correcting proxy discrimination
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) - governance integration for fairness audit outputs
