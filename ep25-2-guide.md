---
layout: page
title: "Proxy Discrimination Audit Guide"
description: "A step-by-step compliance guide for UK personal lines pricing actuaries. How Consumer Duty and the Equality Act 2010 require proxy discrimination auditing, how to conduct one, and what the board needs to see."
permalink: /ep25-2-guide/
---

Citizens Advice estimated in 2022 that the ethnicity penalty in UK motor insurance averages £280 per year — £213m annually across the market. The mechanism is not complicated: postcode drives price; postcode correlates with ethnicity; therefore price correlates with ethnicity. The legal obligation to address this falls on firms under Consumer Duty (PS22/9, PRIN 2A) and Equality Act 2010 Section 19 — not EP25/2, which is about something else entirely (evaluating GIPP price-walking remedies).

This guide covers how to conduct a defensible proxy discrimination audit on a personal lines pricing model, and what "good" looks like when you put it in front of a board or a supervisor.

---

## The regulatory obligation

**FCA Evaluation Paper EP25/2** (published July 2025) evaluated the outcomes of the GIPP price-walking remedies introduced in 2022. That is its actual scope: whether renewal price-walking has been eliminated. It is not the source of proxy discrimination requirements.

The proxy discrimination obligation comes from two documents:

- **Consumer Duty (PS22/9, PRIN 2A)** — requires firms to actively monitor whether products provide fair value for different groups of customers, including groups defined by protected characteristics
- **Equality Act 2010, Section 19** — independently prohibits indirect discrimination: a practice that puts persons sharing a protected characteristic at a particular disadvantage, unless it is a proportionate means of achieving a legitimate aim

TR24/2 (FCA multi-firm review, 2024) found most insurers relied on inadequate manual checks to satisfy these obligations — which is the immediate supervisory pressure.

The obligation is not to achieve equal prices across groups. It is to demonstrate that price differences reflect genuine risk differences rather than the rating model acting as a proxy for a characteristic it should not price on.

**Protected characteristics** relevant to personal lines pricing: sex, ethnicity, disability, religion or belief, age (for underwriting, not rating, purposes). Race and ethnicity receive the most scrutiny given the Citizens Advice evidence.

**What proxy discrimination is.** A rating variable acts as a proxy when it carries statistical information about a protected characteristic beyond what is explained by the risk it is legitimately measuring. Postcode is the canonical example: it predicts theft rates (legitimate) and also predicts ethnicity (not legitimate as a pricing input). The fact that you did not intend to price on ethnicity is irrelevant — what matters is whether the model's output is influenced by ethnicity via the proxy.

**Why manual correlation checks are not enough.** A Pearson or Spearman correlation between postcode band and ethnicity proportion catches a linear relationship. It will miss:

- Non-linear proxies: a variable that is monotonically correlated with protected status only within certain levels
- Interaction proxies: two variables that individually show low correlation but jointly predict protected status well
- Indirect proxy chains: vehicle group predicts theft; theft patterns correlate with area; area correlates with ethnicity
- Model amplification: a factor with modest proxy correlation but high SHAP importance contributes more discriminatory variation to the final price than the raw correlation implies

TR24/2 (the FCA's 2024 multi-firm review) found that most insurers were relying on exactly these inadequate manual checks and that this was not sufficient to meet their Consumer Duty and Equality Act obligations.

---

## The audit workflow

The workflow below assumes a CatBoost pricing model and uses `insurance-fairness` throughout. Install it first:

```bash
pip install insurance-fairness insurance-datasets polars catboost
```

---

### Step 1: Identify protected characteristics

The protected characteristics you need to test against are not all directly observable in your data. For UK motor:

| Characteristic | Typical data source |
|---|---|
| Sex | Policy data (often held; use with care post-Test-Achats) |
| Ethnicity | Not typically held — use ONS Census 2021 LSOA proxy |
| Disability | Not typically held — flag if any proxy known |
| Religion | Not typically held |

For ethnicity, the standard approach for UK insurers joins three public datasets:

1. **ONS Postcode Directory (ONSPD)** — maps full postcode to LSOA code. Available from the ONS Geography Portal, updated quarterly.
2. **Census 2021 Table TS021** — ethnic group counts by LSOA. Available from NOMIS.
3. Divide non-white-British count by total LSOA population to produce a continuous `ethnicity_proxy` column (proportion 0–1).

```python
import polars as pl

# postcode_lsoa: from ONSPD (postcode -> lsoa21cd)
# lsoa_ethnicity: from TS021 (lsoa21cd -> pct_non_white_british)

df = (
    df
    .join(postcode_lsoa, on="postcode", how="left")
    .join(lsoa_ethnicity, on="lsoa21cd", how="left")
)
```

This gives you a continuous ethnicity proxy per policy. The library handles continuous protected columns natively.

For development and testing, `insurance-datasets` provides synthetic motor data with an injected area-band/ethnicity-proxy correlation that replicates the Citizens Advice finding structure:

```python
from insurance_datasets import load_motor

motor = load_motor(n_policies=50_000, seed=42)
# Columns include: area (A-F), driver_age, vehicle_group,
# ncd_years, claim_count, claim_amount, exposure
```

---

### Step 2: Proxy detection

Before computing any fairness metrics, establish which rating factors carry information about the protected characteristics. This is the step most teams skip.

```python
from insurance_fairness.proxy_detection import detect_proxies

result = detect_proxies(
    df,
    protected_col="ethnicity_proxy",       # continuous 0-1 ONS proportion
    factor_cols=[
        "area",          # postcode band — expected to flag
        "vehicle_group",
        "ncd_years",
        "driver_age",
        "annual_mileage",
        "occupation_class",
    ],
    run_proxy_r2=True,        # CatBoost R-squared per factor
    run_mutual_info=True,     # catches non-linear relationships
    run_partial_corr=True,    # controls for other factors
)

print(result.to_polars())
# factor           proxy_r2   mutual_info   partial_corr   flag
# area             0.312      0.198         0.287          RED
# vehicle_group    0.041      0.031         0.028          GREEN
# ncd_years        0.018      0.012         0.015          GREEN
```

**Interpreting proxy R-squared.** A CatBoost model predicts the protected characteristic from each rating factor in isolation. R-squared > 0.05 is amber; R-squared > 0.10 is red. These are triggers for investigation, not bright-line compliance tests — the FCA has not prescribed thresholds.

**Why mutual information matters.** Proxy R-squared uses a tree model and captures most non-linear relationships. Mutual information is a model-free complement — it will occasionally flag something the R-squared misses, particularly for ordinal factors with step-function relationships.

**A factor can be both legitimate and a proxy.** `area` has a genuine theft signal and a genuine ethnicity proxy signal. Detecting proxy correlation does not mean the factor must be removed — it means you need to quantify the discriminatory component and decide whether the pricing benefit is proportionate. That is the proportionality test under Equality Act Section 19.

---

### Step 3: Bias metrics

With proxies identified, quantify the bias in the model's output.

```python
from insurance_fairness import (
    FairnessAudit,
    calibration_by_group,
    demographic_parity_ratio,
    disparate_impact_ratio,
)

# Full audit
audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["ethnicity_proxy", "gender"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["area", "vehicle_group", "ncd_years", "driver_age", "annual_mileage"],
    model_name="Motor Model v3.2 — Q1 2025",
    run_proxy_detection=True,
)

report = audit.run()
report.summary()
```

The three metrics that matter most, and what they mean in plain English:

**Calibration by group (the primary test).**

```python
cal = calibration_by_group(
    df,
    protected_col="ethnicity_proxy",
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    n_deciles=10,
)
print(f"Max calibration disparity: {cal.max_disparity:.4f}  [{cal.rag}]")
```

This splits policyholders into deciles by predicted premium and measures A/E separately for high-ethnicity-proxy and low-ethnicity-proxy groups within each decile. If the model is well-calibrated by group, price differences reflect risk differences. Max disparity < 0.05 is green; 0.05–0.10 is amber; > 0.10 is red.

This is the most defensible metric under the Equality Act proportionality test. A model that is calibrated by group is not over- or under-pricing any protected group — the premium differences are explained by risk.

**Demographic parity ratio.**

```python
dp = demographic_parity_ratio(
    df, "ethnicity_proxy", "predicted_premium", "exposure"
)
print(f"Log-ratio: {dp.log_ratio:+.4f}  (ratio: {dp.ratio:.4f})")
```

This measures whether average premiums differ across groups. Reported in log-space because pricing models are multiplicative. A ratio of 1.10 means the high-proxy group pays 10% more on average. The FCA does not require parity — risk differences are allowed — but large ratios (> 1.15) warrant a written explanation.

**Disparate impact ratio.**

```python
di = disparate_impact_ratio(
    df, "ethnicity_proxy", "predicted_premium", "exposure"
)
print(f"Disparate impact: {di.ratio:.4f}  [{di.rag}]")
```

The 80% rule from US employment law: if the less-favoured group's selection rate (here, the proportion quoted below a reference premium) is below 80% of the majority group's rate, that is a material disparate impact. This metric is widely recognised and is the most likely to appear in a regulatory question. Below 0.80 is red.

---

### Step 4: Mitigation

Proxy detection followed by bias metric computation tells you the size of the problem. Mitigation is the hard part.

**The Lindholm correction (marginalisation).** The academically rigorous approach is from Lindholm, Richman, Tsanakas, and Wüthrich (2022). For each policy, compute the premium the model would produce on average if the protected characteristic were drawn from its marginal distribution rather than the observed value. This produces a "discrimination-free price" that retains the legitimate risk signal from proxies while removing the protected-characteristic component.

```python
from insurance_fairness import counterfactual_fairness

cf = counterfactual_fairness(
    model=model,
    df=df,
    protected_col="ethnicity_proxy",
    feature_cols=["area", "vehicle_group", "ncd_years", "driver_age", "annual_mileage"],
    exposure_col="exposure",
    method="lrtw_marginalisation",   # Lindholm et al. 2022
)
cf.summary()
# Counterfactual premium impact: +4.1%
# (high-ethnicity-proxy policies would pay 4.1% less under marginalised price)
```

The LRTW marginalisation approach is published in ASTIN Bulletin 52(1) and was awarded by the American Academy of Actuaries. Using it rather than a bespoke method gives you academic defensibility if the audit is challenged.

**Feature removal trade-offs.** Removing `area` entirely eliminates the proxy signal but also removes a legitimate theft predictor. The pareto frontier — how much discrimination is reduced per unit of predictive performance lost — is the right frame for this decision.

```python
from insurance_fairness.pareto import fairness_accuracy_frontier

frontier = fairness_accuracy_frontier(
    model=model,
    df=df,
    protected_col="ethnicity_proxy",
    factor_cols=["area", "vehicle_group", "ncd_years", "driver_age"],
    outcome_col="claim_amount",
    exposure_col="exposure",
)
# Returns: list of (gini, disparate_impact_ratio) points
# at different levels of area banding / removal
```

We are not advocating for removing `area`. We are advocating for quantifying the trade-off explicitly and documenting that the pricing benefit was judged proportionate to the discriminatory impact. That written proportionality judgement is what the Equality Act Section 19 defence requires.

**What feature removal does and does not do.** Removing a proxy factor from the model does not remove proxy discrimination if another variable in the model also correlates with the protected characteristic. A model with `area` removed that retains `occupation_class` (which correlates with both ethnicity and area) will still show proxy bias — the signal has moved, not disappeared. Run the full proxy detection after any feature change.

---

### Step 5: Documentation — what the board needs to see

Consumer Duty requires board-level sign-off on fair value assessments. The board does not need the technical detail; it needs:

1. **Evidence that the audit was run** — date, model version, dataset size
2. **Which protected characteristics were tested**
3. **The result** — RAG status per characteristic, per metric
4. **What was done about amber or red findings**
5. **Who attested to the methodology**

```python
report.to_markdown("motor_fairness_audit_q1_2025.md")
```

The `to_markdown()` output includes a sign-off table — populated by the lead pricing actuary and countersigned by the Head of Pricing or Chief Actuary. This goes into the pricing committee file alongside the technical validation.

The report structure:

- Executive summary: overall RAG, date, model name, exposure count
- Per-characteristic results: calibration grid, parity ratio, disparate impact ratio
- Proxy detection table: all factors, all scores, flags
- Methodology section: references to Lindholm et al. and the FCA regulatory mapping
- Sign-off table: name, role, date, attestation statement

File it. Keep it. Consumer Duty requires this documentation to exist, be current, and be reviewed by the board — not just that the model passed a check at launch.

---

## What "good" looks like

| Metric | Green | Amber | Red |
|---|---|---|---|
| Max calibration disparity | < 0.05 | 0.05–0.10 | > 0.10 |
| Demographic parity log-ratio | \|log-ratio\| < 0.05 | 0.05–0.10 | > 0.10 |
| Disparate impact ratio | > 0.90 | 0.80–0.90 | < 0.80 |
| Proxy R-squared (per factor) | < 0.05 | 0.05–0.10 | > 0.10 |

A green across all metrics is sufficient for an unqualified sign-off.

An amber finding on a single metric does not require mitigation but requires a written explanation. For example: "Demographic parity log-ratio for the ethnicity proxy is +0.072 (amber). This reflects genuine risk differences between area bands — calibration by group is green at 0.031, confirming the model is not systematically over-pricing high-proxy-group policyholders. No mitigation action taken. Reviewed annually."

A red finding on calibration by group is a material concern and warrants a written board-level decision — either a mitigation plan with a timeline, or a proportionality judgement setting out why the pricing benefit outweighs the discriminatory impact.

These thresholds are ours, not the FCA's. The FCA has not published numerical thresholds. The justification for these specific numbers: the 0.10 calibration threshold is equivalent to a 10% A/E gap between protected and non-protected groups, which is broadly the level at which most chief actuaries would consider a pricing basis inadequate. The 0.80 disparate impact threshold follows US Title VII case law and is widely cited in algorithmic fairness literature.

---

## Common mistakes

**Using Pearson correlation only.** The tree-based proxy R-squared catches monotonic non-linear relationships that Pearson misses. Occupation class is the classic example: in UK motor data, certain occupation codes are predominantly held by one ethnic group, but the relationship is step-function, not linear. Pearson gives you 0.04; proxy R-squared gives you 0.18. One of those numbers should trigger an investigation.

**Ignoring non-linear proxy chains.** Vehicle group predicts claim severity. High-value vehicles cluster in certain postcodes. Those postcodes have ethnic concentration. The chain — vehicle group → postcode → ethnicity — means vehicle group is a proxy even though its direct correlation with ethnicity is modest. The proxy detection module tests each factor in isolation; it does not automatically detect chain effects. Run mutual information scores and review the flagged factors for plausible chains.

**Treating feature removal as mitigation.** Removing `area` from the model and declaring the audit clean is the most common mistake we see. If `occupation_class` remains in the model and correlates with the ethnicity proxy, the model will still fail the calibration-by-group test. Feature removal shifts the proxy signal; it does not eliminate it unless all correlated features are removed (at which point you have also destroyed predictive accuracy). Run the full audit after any feature change.

**Running the audit once at model launch.** Consumer Duty requires ongoing monitoring. The book mix shifts: new distribution channels, geographic expansion, rate changes that interact differently with risk segments. An audit that passed in March 2024 on the training distribution does not guarantee compliance on the live book in March 2025. The library is fast enough (2–10 minutes for 50,000 policies) to run quarterly as part of the standard A/E monitoring cycle.

**Conflating calibration with parity.** A model can be perfectly calibrated by group and still show large demographic parity differences if risk genuinely differs by group. Conversely, a model can show demographic parity (equal average prices) and fail calibration (it is overpricing low-risk members of the disadvantaged group and underpricing high-risk members). Neither calibration nor parity alone is sufficient — you need both. The FCA's primary interest is calibration because that maps to the proportionality test; but unexplained parity differences attract scrutiny and should have a written explanation.

---

## References

- FCA Consumer Duty (PS22/9): Policy Statement and Final Rules (July 2022)
- FCA Multi-Firm Review TR24/2: Outcomes Monitoring under the Consumer Duty (August 2024)
- FCA Evaluation Paper EP25/2: Evaluation of General Insurance Pricing Practices Remedies (22 July 2025) — covers GIPP price-walking remedies, not proxy discrimination
- PRIN 2A: Consumer Duty — FCA Handbook
- Equality Act 2010, Section 19: Indirect Discrimination
- Lindholm, Richman, Tsanakas, Wüthrich (2022). Discrimination-Free Insurance Pricing. ASTIN Bulletin 52(1), 55–89.
- Lindholm, Richman, Tsanakas, Wüthrich (2026). Sensitivity-Based Measures of Discrimination in Insurance Pricing. European Journal of Operational Research.
- Citizens Advice (2022). Discriminatory Pricing: Exploring the Ethnicity Penalty in the Insurance Market.

---

## Library reference

This guide uses [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) throughout.

```bash
pip install insurance-fairness
```

Full worked notebook: [`fairness_audit_demo.py`](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py) — runs on Databricks Free Edition serverless compute, installs its own dependencies, generates synthetic data inline.

Related libraries: [`insurance-causal`](https://github.com/burning-cost/insurance-causal) for establishing whether a rating factor causally drives risk versus acting as a proxy; [`insurance-governance`](https://github.com/burning-cost/insurance-governance) for the PRA SS1/23 model validation report that the fairness audit feeds into.
