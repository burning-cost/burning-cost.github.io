---
layout: post
title: "Insurance Fairness Audit in Python: A Complete FCA Consumer Duty Walkthrough"
date: 2026-04-04
author: Burning Cost
categories: [tutorials, fairness, regulatory]
tags: [insurance-fairness, python, proxy-discrimination, fca, consumer-duty, equality-act-2010, fairness-audit, bias-metrics, demographic-parity, calibration-by-group, uk-motor, postcode, ethnicity, ps22-9, fg22-5, prin-2a, polars, catboost, pricing-committee]
description: "Step-by-step Python tutorial for running an insurance fairness audit. Covers proxy discrimination detection, exposure-weighted bias metrics, FCA Consumer Duty mapping, and generating an audit report your pricing committee can sign off. Uses the insurance-fairness library (2,152 downloads/month)."
---

Your postcode rating factor probably correlates with ethnicity. That is not a hypothesis — Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, totalling £213m annually, driven entirely through proxy factors. The question is not whether your model might be doing this. The question is whether you can demonstrate it is not, in terms that satisfy the FCA's Consumer Duty requirements and would survive an FCA file review.

This tutorial walks through a complete insurance fairness audit in Python using [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness), our most-downloaded library at 2,152 downloads/month. By the end you will have run proxy detection across your rating factors, computed exposure-weighted bias metrics, and produced a Markdown audit report mapped to FCA PRIN 2A, FG22/5, and Equality Act s.19. The whole workflow takes under 15 minutes on a 50,000-policy book.

```bash
pip install insurance-fairness
# or
uv add insurance-fairness
```

---

## Why "we don't model protected characteristics" is not enough

The standard defence from pricing teams is: "We don't use gender, ethnicity, or religion in our model — so we can't be discriminating." This misunderstands what Section 19 of the Equality Act 2010 requires.

Section 19 defines indirect discrimination as applying a "provision, criterion or practice" that puts people with a protected characteristic at a particular disadvantage compared to others — *even if the protected characteristic is not directly used*. A postcode rating factor does not reference ethnicity. It does not need to: if policyholders in high-diversity postcode districts are systematically overcharged relative to their actual risk, because the postcode factor is picking up area demographics rather than (or in addition to) genuine risk variation, that is indirect discrimination.

The FCA's position on this is not ambiguous. Consumer Duty PRIN 2A requires firms to demonstrate fair outcomes for all groups of customers. FCA FG22/5 (Consumer Duty guidance) states firms should consider how algorithmic tools could produce biased outcomes for different groups. The FCA's 2024 multi-firm review of Consumer Duty implementation found most Fair Value Assessments were "high-level summaries with little substance" — the FCA's words, not ours.

The standard statistical approach to checking for proxy correlation — a Spearman correlation between postcode and an ethnicity proxy — misses the problem almost entirely. The postcode-ethnicity relationship is non-linear: London inner postcodes, outer postcodes, and rural postcodes cluster in ways that a linear correlation measure cannot detect. On a 20,000-policy synthetic UK motor portfolio replicating the Citizens Advice finding, Spearman returns |r| ≈ 0.10 and flags nothing. CatBoost proxy R² returns 0.62 — unambiguously RED.

This is the gap `insurance-fairness` is designed to close.

---

## When do you need a fairness audit?

Not every model update requires a full audit. Here is how we think about the triggers:

**Always required:**
- New model launch (personal motor, home, SME) — prior to pricing committee sign-off
- Major model rebuild (new rating factors, new algorithm, e.g. GLM to GBM)
- Annual Consumer Duty fair value assessment

**Triggered by specific signals:**
- FCA supervisory request or thematic review covering your class of business
- Introduction of a new geographic factor (postcode, LSOA, ward-level variable)
- Addition of occupation, employer, or employment-status factors — these correlate with religion and ethnicity
- Consumer complaints alleging discriminatory pricing — a spike in complaints mentioning "why do I pay more than my neighbour" should trigger immediate audit
- Claims experience divergence by area that is not explained by peril exposure
- Portfolio acquisition where you inherit an audited model that you cannot validate

**Lower priority but worth scheduling:**
- Annual refresh of existing stable GLM with no structural changes
- Changes to loadings (security, claims history) that do not alter the rating factor structure

For Consumer Duty purposes, you need to be able to show this assessment has been made actively and documented, not just assert that you run audits on major changes.

---

## The data you need

A fairness audit requires three things beyond your existing modelling dataset:

**1. A protected characteristic column, or a proxy for it.**

Most UK insurers do not hold individual-level ethnicity data. The practical alternative is to join ONS 2021 Census LSOA ethnicity proportions to your postcode data and use the area-level diversity proportion as a proxy. This is an area-level proxy, not individual-level — correlations with this proxy are correlations with area demographics, not individual ethnicity. That limitation should be stated explicitly in your audit report. It is still materially better than no proxy at all.

For gender, most insurers have the data directly (even if they cannot use it as a rating factor post-Equality Act s.11). For age, the data is typically available and s.13 direct discrimination by age in insurance is subject to a sector-specific exemption under Schedule 3 paragraph 20 — but indirect discrimination through age-correlated factors still applies.

**2. Your model's predictions at policy level.**

The predicted premium or predicted frequency/severity — whichever your rating model produces — at the individual policy level. Not average predictions by rating cell.

**3. Exposure and outcome columns.**

Earned exposure (car-years or home-years) and actual claims outcome. The bias metrics are all exposure-weighted. Using policy counts rather than earned exposure produces incorrect results for any book where policy durations vary.

---

## Step 1: load data and construct the audit object

We will use a synthetic UK motor portfolio with 20,000 policies and a known postcode-ethnicity proxy structure. Replace this with your actual data.

```python
import polars as pl
from insurance_fairness import FairnessAudit

# Your policy-level DataFrame. Required columns:
#   - postcode_district: string, your geographic rating factor
#   - vehicle_age, ncd_years, vehicle_group: other rating factors
#   - ethnicity_proxy: ONS 2021 area-level diversity proportion (0–1)
#     joined to postcode via LSOA lookup
#   - predicted_rate: model output, pure premium per car-year
#   - claim_amount: total claim cost in the period (0 for nil claims)
#   - exposure: earned car-years

df = pl.read_parquet("motor_portfolio_q4_2025.parquet")

audit = FairnessAudit(
    model=None,                # pass None if you only have predictions, not a fitted model object
    data=df,
    protected_cols=["ethnicity_proxy"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2025",
    run_proxy_detection=True,
)
```

If you have a fitted CatBoost or sklearn model object, pass it as `model=`. The proxy detection uses the fitted model to compute SHAP-linked price impact alongside the statistical measures. Without a model object, proxy detection falls back to mutual information and CatBoost proxy R² (trained internally), which is still the right approach.

---

## Step 2: run the audit and read the summary

```python
report = audit.run()
report.summary()
```

Output:

```
Motor Model Q4 2025 — Fairness Audit Summary
Overall RAG: RED

Proxy detection (ethnicity_proxy):
  postcode_district  proxy R²=0.62 [RED]   MI=0.41 [RED]
  vehicle_group      proxy R²=0.09 [GREEN] MI=0.12 [AMBER]
  ncd_years          proxy R²=0.03 [GREEN] MI=0.06 [GREEN]
  vehicle_age        proxy R²=0.02 [GREEN] MI=0.04 [GREEN]

Calibration by group (ethnicity_proxy, 10 deciles):
  Max A/E disparity: 0.081 [AMBER]
  — High-diversity deciles (8–10) show systematic A/E > 1.05

Demographic parity log-ratio: +0.082 (ratio 1.085) [AMBER]
  — High-diversity group pays ~8.5% more per car-year after controlling for exposure

Flagged factors: postcode_district
```

The overall RAG is RED because at least one factor has returned a RED proxy signal. That single finding is sufficient for the overall audit to be RED — the AMBER calibration and demographic parity figures are secondary evidence.

**Reading the proxy R² threshold.** The proxy R² thresholds (amber: >0.05, red: >0.10) are operationally derived from the library's validation work, not FCA-prescribed. A proxy R² of 0.62 for postcode-ethnicity is not close to any threshold — it is unambiguous. A factor at 0.08 (amber) warrants investigation and documentation; whether it constitutes indirect discrimination turns on proportionality, not on whether it crosses a statistical threshold.

---

## Step 3: understand what the metrics are measuring

### Calibration by group (the primary Equality Act metric)

Calibration by group answers: does the model overpredict or underpredict for protected-characteristic groups relative to what they actually claim?

The Actual/Expected (A/E) ratio within a protected-group × risk-decile cell tells you whether the model is correctly pricing the risk profile of that group. An A/E of 1.08 in the high-diversity deciles means those policyholders are paying 8% more than their observed claims justify — they are being charged for risk the model assigns to them but they do not actually exhibit.

This is computed exposure-weighted:

```python
from insurance_fairness import calibration_by_group

cal = calibration_by_group(
    df,
    protected_col="ethnicity_proxy",
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    n_deciles=10,
)

print(f"Max A/E disparity: {cal.max_disparity:.4f} [{cal.rag}]")
print(cal.to_polars())  # Full decile-by-group table
```

Equal calibration (A/E ≈ 1.0 within all groups) is the most defensible outcome under Equality Act s.19, because it means the model is correctly pricing each group's actual risk. It does not, however, establish that the model is free of indirect discrimination — it establishes that the model is well-calibrated conditional on the proxy-contaminated risk profile. If the postcode factor is encoding ethnicity, high-diversity policyholders' "risk profile" in the model includes a component that is actually demographic, not actuarial. Calibration by group cannot disentangle these; proxy detection can.

### Demographic parity ratio (log-space, for multiplicative models)

Demographic parity measures whether different protected groups pay systematically different premiums, after accounting for exposure.

For GLMs and GBMs that operate in log-space (i.e., multiplicative models), the right metric is the log-ratio of group mean predictions — not the raw ratio. The raw ratio applied to a multiplicative model double-counts the log transformation.

```python
from insurance_fairness import demographic_parity_ratio

dp = demographic_parity_ratio(df, "ethnicity_proxy", "predicted_rate", "exposure")
print(f"Log-ratio: {dp.log_ratio:+.4f} (ratio: {dp.ratio:.4f}) [{dp.rag}]")
# Log-ratio: +0.0820 (ratio: 1.0854) [AMBER]
```

A ratio of 1.085 means the high-diversity group is charged approximately 8.5% more per car-year on average than the low-diversity group, after weighting by exposure. For context: the Citizens Advice (2022) estimate of the ethnicity penalty in UK motor was approximately 6–9% of annual premium, so this synthetic result is consistent with the order of magnitude of the real finding.

### Proxy detection (the distinctive metric)

This is what makes `insurance-fairness` different from generic fairness libraries. It does not just ask whether group outcomes differ — it asks *why*.

```python
from insurance_fairness import detect_proxies

result = detect_proxies(
    df,
    protected_col="ethnicity_proxy",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    run_proxy_r2=True,
    run_mutual_info=True,
    run_partial_corr=True,
)

print(result.flagged_factors)     # ['postcode_district']
print(result.to_polars())         # Full results table sorted by proxy R²
```

The CatBoost proxy R² trains a classifier to predict the protected characteristic from a single factor. An R² of 0.62 for postcode_district means that 62% of the variance in the area-level ethnicity proxy is explained by the postcode district alone. This is the relationship the model is exploiting, whether intentionally or not.

Mutual information is complementary: it detects non-linear associations that R² can understate. A factor with low proxy R² but high mutual information indicates the relationship is present but non-monotone — potentially more concerning from a discrimination standpoint because it is harder to detect with standard correlation approaches.

---

## Step 4: probe deeper with counterfactual fairness

Proxy detection tells you *which factors* are proxies. Counterfactual fairness tells you *what the premium impact is* of the proxy contamination: what would an individual policyholder pay if we could remove the proxy correlation while keeping everything else constant?

```python
from insurance_fairness import IndirectDiscriminationAudit

indirect = IndirectDiscriminationAudit(
    protected_attr="ethnicity_proxy",
    proxy_features=["postcode_district"],
    exposure_col="exposure",
)
result = indirect.fit(
    X_train=df.drop(["claim_amount", "exposure"]),
    y_train=df["claim_amount"],
    X_test=df.drop(["claim_amount", "exposure"]),
    y_test=df["claim_amount"],
)

print(f"Proxy vulnerability: {result.proxy_vulnerability:.2f}")
# Proxy vulnerability: 0.14
# Interpretation: the average policyholder's premium would change by 14%
# if the postcode-ethnicity proxy correlation were removed
print(result.segment_report)
```

The proxy vulnerability score (Côté, Côté & Charpentier, 2025) is the mean absolute difference between the "unaware" premium — what the model charges not knowing the protected characteristic — and the "proxy-free" premium — what it would charge if the postcode factor were stripped of its ethnicity correlation. On this synthetic portfolio, 14% means some policyholders are paying materially more because of where they live in ways attributable to area demographics rather than genuine risk.

---

## Step 5: generate the FCA-ready audit report

```python
report.to_markdown("motor_fairness_audit_q4_2025.md")
```

The Markdown report includes:

- **Regulatory mapping table** — each finding mapped to the relevant FCA instrument: PRIN 2A, FG22/5, PS22/9, Equality Act s.19
- **RAG summary with evidence** — overall status, per-factor proxy results, calibration table, demographic parity figures
- **Flagged factors with remediation options** — for each RED/AMBER factor, a list of options including: remove and replace, apply optimal transport correction, add to proportionality justification, escalate to legal review
- **Sign-off table** — a structured table for the pricing committee chair and chief actuary to counter-sign; includes model name, audit date, auditor name, and a declaration field

The report is not a compliance determination — it is documented evidence. Whether postcode_district constitutes indirect discrimination depends on whether its use can be *proportionately justified* — i.e., whether it represents a legitimate aim pursued by proportionate means. That proportionality assessment is a legal and actuarial judgement the library cannot make for you. What the report gives you is the evidence base for that assessment, mapped to the regulatory instruments your legal counsel needs to reference.

---

## Step 6: multicalibration — going beyond aggregate metrics

Aggregate A/E by group can look acceptable while hiding problems at the intersection of groups and risk segments. A model might be well-calibrated for female policyholders on average but systematically overcalibrated for female policyholders in the top loss-cost decile. Multicalibration catches this.

```python
from insurance_fairness import MulticalibrationAudit

mc_audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
mc_report = mc_audit.audit(
    y_true=df["claim_amount"].to_numpy(),
    y_pred=df["predicted_rate"].to_numpy(),
    protected=df["ethnicity_proxy"].to_numpy(),
    exposure=df["exposure"].to_numpy(),
)

# If cells are failing, apply iterative correction
from insurance_fairness import IterativeMulticalibrationCorrector
corrector = IterativeMulticalibrationCorrector()
y_pred_corrected = corrector.correct(
    df["predicted_rate"].to_numpy(),
    df["ethnicity_proxy"].to_numpy(),
    mc_report,
    df["exposure"].to_numpy(),
)
```

Denuit, Michaelides & Trufin (2026) establish the theoretical basis for multicalibration in insurance pricing — a model that is multicalibrated cannot systematically overcharge any identifiable subgroup. It is a strictly stronger requirement than aggregate calibration by group, and it is the right target for Consumer Duty purposes.

---

## Handling the "we don't hold protected characteristics" objection

This objection is common and partially valid. You genuinely cannot run `protected_col="ethnicity"` if you have not collected it. The area-level ONS proxy approach covers ethnicity reasonably well for postcode-based factors. For other characteristics:

**Religion and belief** — occupation-level proxies are available from ONS occupation classification data. Certain occupations correlate strongly with religious community (e.g., taxi driver with British South Asian Muslim communities, nurse with various faith communities). If occupation is a rating factor, this warrants checking.

**Disability** — harder to proxy at area level. Disability Living Allowance claim rates by LSOA are publicly available from DWP but the signal is weak for pricing purposes. If your model contains any indicator that correlates with health status (telematics night driving, for example), disability proximity is worth flagging.

**Gender** — you almost certainly hold this. Even if you do not rate on it, running the gender analysis costs you 30 seconds and gives you evidence for the Consumer Duty documentation.

**Age** — the direct age factor is protected under the Equality Act but benefits from the Schedule 3, paragraph 20 actuarial exemption. Age *interactions* with other factors — where the postcode loading is different for young drivers than for older drivers — warrant scrutiny even where the direct age factor is justified.

The audit report's limitations section is the right place to document which protected characteristics you could not audit directly and what proxy approach you used. This is better practice than not documenting it.

---

## The full one-call workflow

If you want the complete audit in a single call:

```python
import polars as pl
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,             # or None if you only have predictions
    data=df,
    protected_cols=["ethnicity_proxy", "gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2025",
    run_proxy_detection=True,
)

report = audit.run()          # 2–10 minutes depending on book size and n_factors
report.summary()              # Console output with RAG statuses
report.to_markdown("audit_q4_2025.md")   # FCA-ready report

# Access structured results programmatically
print(report.overall_rag)              # 'red', 'amber', or 'green'
print(report.flagged_factors)          # ['postcode_district']
print(report.results["gender"])        # ProtectedCharacteristicReport object

# JSON for governance system integration
import json
with open("audit_q4_2025.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

The `to_dict()` output is designed to feed into [insurance-governance](https://github.com/burning-cost/insurance-governance), which generates the full pricing committee sign-off pack including the fairness audit evidence section.

---

## What a RED finding means in practice

A RED proxy detection result does not mean your model is illegal. It means:

1. You have evidence of a potential indirect discrimination mechanism that requires investigation.
2. You need to assess whether the use of the factor can be *proportionately justified* under Equality Act s.19(2)(b) — i.e., that it is "a proportionate means of achieving a legitimate aim".
3. You need to document this assessment in writing, with the fairness audit as the evidence base.
4. If you conclude the factor cannot be proportionately justified, you need to either replace it, apply a correction, or escalate to legal advice.

The proportionality test in insurance has not been definitively settled by case law. The relevant guidance is FCA FG22/5 (Consumer Duty guidance), the Equality and Human Rights Commission's (EHRC) Technical Guidance on Services, and the ABI's guidance on the Equality Act (updated 2023). Your legal counsel needs to assess proportionality; the fairness audit provides the quantified evidence for that assessment.

What you cannot do — under Consumer Duty PRIN 2A and under FCA's Finalised Guidance FG22/5 — is proceed without having made this assessment. "We didn't know" is not available as a defence when the audit tooling exists and the FCA has explicitly flagged the issue.

---

## Decision framework: audit workflow at a glance

```
New model or major model update?
  → Run FairnessAudit.run() before pricing committee sign-off
  → Required outputs: audit report, sign-off table, overall RAG

Proxy detection RED on any factor?
  → Legal review of proportionality justification
  → Options: replace factor, optimal transport correction, limit factor weight
  → Document decision in pricing committee minutes

Calibration by group AMBER or worse?
  → Investigate whether A/E disparity is statistically significant (use cal.to_polars() for decile detail)
  → If significant: trigger multicalibration audit
  → Document whether disparity reflects genuine risk differences or proxy contamination

Demographic parity ratio > 1.05 (after exposure weighting)?
  → Cross-reference with proxy detection results
  → If proxy RED and parity AMBER: likely connected — address at source
  → If proxy GREEN and parity AMBER: investigate other factors or data quality

Annual Consumer Duty Fair Value Assessment?
  → Run full audit for all models in scope
  → Report.to_dict() → insurance-governance → sign-off pack
```

---

## What this is not

This library produces a documented audit trail. It does not:

- Constitute legal advice on whether your model constitutes indirect discrimination
- Provide a regulatory safe harbour — a GREEN overall RAG does not mean you are compliant
- Replace the proportionality assessment that only your legal counsel can make
- Guarantee the FCA will find your approach satisfactory — no automated tool can

The FCA's 2024 multi-firm review found that most firms' Consumer Duty documentation was inadequate. The Burning Cost position is that a well-evidenced, quantified audit — even one that flags problems — is a materially better regulatory position than no audit, or a superficial one. The FCA responds better to firms that have identified and addressed issues than to firms that claim everything is fine without evidence.

---

## Performance at scale

On a 50,000-policy book with six rating factors and two protected characteristics:

| Task | Time | Notes |
|------|------|-------|
| Calibration by group (10 deciles) | < 2s | Primary Equality Act metric |
| Demographic parity ratio | < 1s | Log-space for multiplicative models |
| Mutual information scores (all factors) | < 5s | Non-linear relationship detection |
| Proxy R² per factor (CatBoost) | 15–60s per factor | Dominant cost; subsample above 100k policies |
| Full audit with proxy detection (6 factors) | 3–8 min | Produces FCA-ready Markdown |

For books above 100,000 policies, pass `proxy_detection_sample_n=50000` to subsample the proxy R² computation. The calibration and parity metrics use the full book regardless.

---

## Related libraries

- [insurance-governance](https://github.com/burning-cost/insurance-governance) — converts the fairness audit report into a full pricing committee sign-off pack with FCA Consumer Duty documentation section
- [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) — monitors protected-group A/E ratios on the live book post-deployment; alerts when fairness metrics drift
- [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) — establishes whether a premium change *caused* a shift in protected-group outcomes, rather than just correlating with one; relevant for proportionality assessments

Library: [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness) — MIT-licensed, Python 3.10+, PyPI: [`insurance-fairness`](https://pypi.org/project/insurance-fairness/).

---

**Related:**
- [Your Pricing Model Might Be Discriminating](/2026/03/07/your-pricing-model-might-be-discriminating/) — the founding post: Citizens Advice £213m estimate, Section 19 Equality Act mechanics, and what "proxy discrimination" actually means in insurance
- [The Fairness Impossibility You Cannot Optimise Away](/2026/04/04/nsga2-multi-objective-fairness-pareto-boonen-2512-24747/) — Chouldechova's theorem made operational: you cannot satisfy demographic parity, equalised odds, and calibration simultaneously; Pareto optimisation is the right response
- [Detection vs Mitigation: insurance-fairness and EquiPy Are Not Competing](/2026/04/04/insurance-fairness-equipy-comparison/) — auditing (what you need for the FCA) versus correction (what you need after the audit finds a problem)
- [The Mills Review: What the FCA's Long-Term AI Inquiry Means for Pricing Teams](/2026/03/26/fca-long-term-ai-review/) — the regulatory environment in which all of this sits
