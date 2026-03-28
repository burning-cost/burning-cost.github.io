---
layout: post
title: "What a Real Fairness Audit Finds: Gender Bias Testing on 67,856 Motor Policies"
date: 2026-03-27
categories: [fairness, pricing, techniques]
tags: [fairness-audit, proxy-discrimination, gender-bias, consumer-duty, equality-act-2010, fca, multicalibration, indirect-discrimination, insurance-fairness, motor, catboost]
description: "We ran insurance-fairness against ausprivauto0405  -  a real Australian motor dataset with an explicit Gender field. Here is what FairnessAudit, MulticalibrationAudit, and IndirectDiscriminationAudit found, and what that means for how you audit UK pricing models."
---

Most fairness audit walkthroughs use toy data. An artificial dataset where gender neatly predicts claims, so the audit catches the problem, everyone looks clever, and you learn nothing about what happens on a real book.

We ran `insurance-fairness` against a real published dataset  -  67,856 Australian private motor insurance policies with an explicit Gender field  -  to see what these tools actually find when you throw messy real data at them. The results were instructive, and not in the way you might expect.

**A note before we start:** ausprivauto0405 is Australian motor data from 2004-05, sourced from the CASdatasets R package (Dutang & Charpentier, 2024). We used it because it has a real Gender variable and real claim outcomes  -  not because it tells us anything about the UK market. Everything here is methodology validation. Do not treat any numbers in this post as benchmarks for Equality Act 2010 compliance or FCA Consumer Duty reporting.

---

## The setup

The dataset has 67,856 policies after cleaning, with columns for vehicle value, vehicle age, vehicle body type, driver age, and claim occurrence. We split off 25% as a test set, then fitted a CatBoost classifier on the training set to predict claim occurrence  -  with Gender deliberately excluded from the model features.

The point is to replicate production reality: your UK model does not use gender as a rating factor (it has been prohibited since *Test-Achats* in 2012 and the subsequent ABI guidance), but gender may still be correlated with the factors you do use. The question is whether that residual correlation creates discriminatory outcomes.

```python
from insurance_fairness import (
    FairnessAudit,
    equalised_odds,
    MulticalibrationAudit,
    IndirectDiscriminationAudit,
)

audit = FairnessAudit(
    model=None,
    data=df_audit,                        # Polars DataFrame
    protected_cols=["gender"],
    prediction_col="predicted_freq",
    outcome_col="claim_occ",
    exposure_col="exposure",
    factor_cols=["VehValue", "VehAge", "VehBody", "DrivAge"],
    model_name="CatBoost claim frequency  -  ausprivauto0405",
    run_proxy_detection=True,
    run_counterfactual=False,
    proxy_catboost_iterations=100,
    n_bootstrap=0,
)

report = audit.run()
report.summary()
```

The `model=None` argument tells `FairnessAudit` to work from pre-computed predictions rather than re-scoring the model  -  useful when you want to audit predictions that already exist in your monitoring database.

---

## What FairnessAudit measures

`FairnessAudit.run()` returns a `FairnessReport` with a `ProtectedCharacteristicReport` for each protected column. For gender, it computes four things:

**Demographic parity**: the ratio of mean predicted frequencies between Male and Female policyholders, expressed as a log-ratio. A log-ratio of zero means the model assigns identical average premiums across groups. The benchmark is ±0.10 (green); beyond ±0.20 the RAG goes red. In this dataset, males have a raw claim frequency of approximately 0.07 per car-year versus females at 0.06  -  so some gap is actuarially expected. The interesting question is whether the gap in *predictions* matches the gap in *actual claims* after conditioning on vehicle and driver age factors.

**Calibration by group**: the maximum A/E disparity across groups. If the model predicts male claim frequency accurately but systematically underestimates female frequency (or vice versa), that will show here. The metric is `cal.max_disparity`  -  the largest difference between group-level A/E ratios. Green is below 0.05; red is above 0.15.

**Disparate impact ratio**: the ratio of favourable outcomes (low premium predictions, per the 80% rule from US fair lending practice). This is the most commonly cited fairness metric in regulatory discussions. Below 0.80 is the traditional threshold for concern, though the FCA has not formalised that threshold in a UK insurance context.

**Proxy detection**: for each rating factor, `FairnessAudit` fits a CatBoost model to predict gender from that factor alone and reports the R-squared. It also computes mutual information. The `flagged_factors` list contains anything where proxy R-squared exceeds 0.05 or mutual information is materially above zero.

```python
gr = report.results.get("gender")

if gr.proxy_detection:
    pdr = gr.proxy_detection
    for s in pdr.scores:
        print(f"{s.factor:<20} proxy_r2={s.proxy_r2:.4f}  mi={s.mutual_information:.4f}  [{s.rag.upper()}]")

print(f"Flagged: {pdr.flagged_factors or 'none'}")
```

On this dataset, DrivAge is the factor most correlated with gender  -  young drivers skew male, mature drivers skew female. That correlation is why driver age is a legitimate rating factor (it predicts claims), but it also means that any model heavy on driver age will implicitly pick up some gender signal. Whether that constitutes indirect discrimination under Section 19 of the Equality Act depends on whether the proxy relationship has an objective justification  -  which in this case it does, because driver age predicts accidents independently of gender.

---

## Multicalibration: the actuarial fairness test

`MulticalibrationAudit` asks a more demanding question than standard calibration. A model is *multicalibrated* if, within every predicted-frequency bin, both male and female predictions are equally accurate:

```
E[Y | mu(X) = p, gender = g] = p   for all bins p and all g
```

This condition was formalised for insurance by Denuit, Michaelides & Trufin (2026, arXiv:2603.16317). It unifies actuarial accuracy with group fairness: a model that passes A/E overall but fails within specific prediction bands for a specific gender is both actuarially suspect and potentially discriminatory.

```python
mc = MulticalibrationAudit(n_bins=8, alpha=0.05)
mc_report = mc.audit(
    y_true=df["ClaimOcc"].values.astype(float),
    y_pred=y_pred_all,
    protected=gender_binary,      # 1 = Male, 0 = Female
    exposure=df["Exposure"].values,
)

print(f"Is multicalibrated: {mc_report.is_multicalibrated}")
print(f"Overall calibration p-value: {mc_report.overall_calibration_pvalue:.4f}")
for g_name, pval in mc_report.group_calibration.items():
    label = "Male" if str(g_name) == "1" else "Female"
    print(f"  {label}: p={pval:.4f}")
```

The `MulticalibrationReport.bin_group_table` gives you a Polars DataFrame with one row per (bin, gender) cell: the observed and expected counts, the A/E ratio for that cell, and a p-value for whether the deviation from 1.0 is statistically significant. `worst_cells` ranks the top 10 cells by |A/E − 1|, which is exactly what you would put in a pricing committee paper when reporting multicalibration failures.

If the audit fails  -  `is_multicalibrated = False`  -  you can apply a credibility-weighted correction:

```python
corrected_preds = mc.correct(y_pred_all, gender_binary, mc_report, exposure=df["Exposure"].values)
```

The correction is conservative: cells below `min_credible` observations (default 1,000) get a blended adjustment towards no change, which prevents overcorrecting on noise from thin subgroups.

---

## Indirect discrimination: the Côté et al. framework

The most technically demanding part of the benchmark is `IndirectDiscriminationAudit`, which implements the five benchmark premiums from Côté, Côté & Charpentier (CAS Working Paper, October 2025).

The key insight is that you do not need a causal graph to measure proxy discrimination. You only need two models: one trained with the protected attribute visible (the "aware" model h_A), and one trained without it (the "unaware" model h_U). The difference between them, per policyholder, measures how much the unaware model is implicitly re-learning gender from the other features:

```
proxy_vulnerability = mean |h_U(x) - h_A(x)|
```

```python
indirect = IndirectDiscriminationAudit(
    protected_attr="Gender",
    proxy_features=[],                           # no known proxies to remove additionally
    model_class=GradientBoostingClassifier,
    model_kwargs={"n_estimators": 100, "max_depth": 3, "random_state": 42},
    exposure_col="Exposure",
    random_state=42,
)

result = indirect.fit(X_train, y_train, X_test, y_test)
print(f"Portfolio proxy vulnerability: {result.proxy_vulnerability:.5f}")
print(result.summary)
```

`IndirectDiscriminationResult` contains the scalar `proxy_vulnerability`  -  the exposure-weighted mean absolute difference between aware and unaware predictions  -  along with a `summary` DataFrame with per-segment breakdown and a `benchmarks` dict containing all five model outputs per policyholder.

A proxy vulnerability near zero means the unaware model cannot infer gender from the remaining features: your rating factors do not collectively act as a gender proxy. A high vulnerability means they do, even if no single factor is strongly correlated.

On ausprivauto0405, vehicle body type (VehBody) is moderately correlated with gender  -  utility vehicles skew male in this dataset. That is both a legitimate rating variable (different vehicle types genuinely have different claim profiles) and a partial gender proxy. The audit quantifies the overlap.

---

## Equalised odds as a binary decision test

For completeness, we also computed equalised odds  -  relevant if your model is used to make a binary accept/decline decision rather than to set a continuous premium:

```python
threshold = float(np.mean(y_pred_all))
df_eo = df_audit.with_columns(
    pl.when(pl.col("predicted_freq") >= threshold)
      .then(pl.lit(1)).otherwise(pl.lit(0)).alias("predicted_binary"),
    pl.col("claim_occ").cast(pl.Int32),
)

eo = equalised_odds(
    df=df_eo,
    protected_col="gender",
    prediction_col="predicted_binary",
    outcome_col="claim_occ",
    exposure_col="exposure",
)

print(f"TPR disparity: {eo.tpr_disparity:.4f}  [{eo.rag.upper()}]")
print(f"FPR disparity: {eo.fpr_disparity:.4f}")
```

`equalised_odds` returns a result with `tpr_disparity` and `fpr_disparity`  -  the difference in true positive and false positive rates across groups  -  along with a RAG classification. For a pure pricing model (no hard underwriting decisions), demographic parity and calibration are more directly relevant than TPR/FPR, but the function is there for decisioning contexts.

---

## What this means for UK auditing

The regulatory pressure is real. FCA Consumer Duty (PRIN 2A, live since July 2023) requires firms to monitor for differential outcomes by customer group. FCA TR24/2 (2024) called out pricing practices specifically. The Equality Act 2010 Section 19 creates liability for indirect discrimination where a pricing practice puts a protected group at a particular disadvantage without objective justification.

These three tools  -  `FairnessAudit`, `MulticalibrationAudit`, and `IndirectDiscriminationAudit`  -  address different layers of that obligation:

- `FairnessAudit` gives you the top-line evidence pack: demographic parity, calibration by group, disparate impact ratio, proxy detection scores.
- `MulticalibrationAudit` gives you the actuarial precision test: calibration within every (bin, group) cell, not just in aggregate. A model can pass overall A/E checks while systematically underpricing high-risk female policyholders in the top frequency band.
- `IndirectDiscriminationAudit` gives you the indirect discrimination quantification: how much is the unaware model implicitly reconstructing protected group membership from the rating factors you do use?

None of these tools tells you whether your model is lawful. That is a legal question that depends on objective justification arguments specific to your book, your underwriting process, and your regulatory history. What they tell you is what the evidence shows  -  and whether that evidence is strong enough that your compliance team needs to be in the room.

We think pricing teams should be running these checks as part of the standard model review cycle, not as a one-off exercise when there is regulatory heat. The `report.to_markdown("audit_2025.md")` method on `FairnessReport` produces a formatted audit pack you can attach to the model governance record directly.

---

## Running it yourself

The library and notebook are at [github.com/pricing-frontier/insurance-fairness](https://github.com/pricing-frontier/insurance-fairness). The benchmark notebook (`notebooks/ausprivauto_fairness_databricks.py`) is Databricks-compatible and downloads the dataset directly from the CASdatasets GitHub repository.

```
uv pip install insurance-fairness rdata requests catboost
```

The `rdata` package handles parsing the `.rda` format that CASdatasets uses. Everything else is standard PyPI.
