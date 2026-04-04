---
layout: post
title: "Detection vs Mitigation: insurance-fairness and EquiPy Are Not Competing"
date: 2026-04-04
categories: [fairness, insurance-fairness]
tags: [proxy-discrimination, fca-consumer-duty, equality-act, fairness-python, equipy, wasserstein, optimal-transport, demographic-parity, insurance-fairness-python, arXiv-2503.09866, charpentier]
description: "EquiPy (Fernandes Machado, Charpentier et al.) does distributional fairness correction via Wasserstein barycenters. insurance-fairness does proxy discrimination auditing for UK regulatory compliance. These solve different problems. A UK pricing team needs both."
author: burning-cost
math: true
---

Two Python libraries targeting insurance fairness, published months apart. One from a SCOR Foundation-funded academic group including Arthur Charpentier, who has probably written more on actuarial fairness than anyone alive. One from us, built against the FCA's Consumer Duty and Equality Act 2010 obligations. They are not the same tool. They are not competing. Understanding why tells you something important about what fairness work in UK insurance actually requires.

---

## The core difference

**EquiPy** ([arXiv:2503.09866](https://arxiv.org/abs/2503.09866), Fernandes Machado, Grondin, Ratz, Charpentier and Hu) is a **mitigation tool**. It takes a fitted model's predictions and post-processes them so that the output distribution is the same across demographic groups — Strong Demographic Parity. It does not touch the model. It adjusts the scores.

**insurance-fairness** is a **detection and audit tool**. It takes a fitted model and asks: does this model's behaviour constitute proxy discrimination under UK law? It produces documented evidence — proxy detection results, calibration diagnostics, bias metrics, an FCA-mapped Markdown report — that a pricing committee can sign off and that will stand up to a regulatory file review.

These are different jobs. One assumes you have already decided the model is biased and want to correct it. The other helps you determine whether it is biased, in the specific way the FCA and the Equality Act care about, with evidence you can show a regulator.

---

## What EquiPy does

EquiPy implements the fairness mechanism from Charpentier's group's 2023 paper on sequentially fair mechanisms for multiple sensitive attributes. The central construct is a Wasserstein barycenter: given group-conditional prediction distributions, find the distribution that minimises the average Wasserstein distance to each group's conditional, then transport each group's predictions to that common target.

The key insight — and where the "sequential" in sequential fairness comes in — is the multi-attribute case. When you have both gender and age band as sensitive attributes, achieving fairness on both simultaneously is not simply stacking two single-attribute corrections. EquiPy handles this via `MultiWasserstein`, which processes attributes in sequence and tracks the resulting fairness at each step. The epsilon parameter controls the accuracy-fairness trade-off for each attribute: 0 means full fairness constraint, values above 0 relax it.

The API is clean. You fit on a calibration set and transform predictions for a test set:

```python
from equipy.fairness import FairWasserstein, MultiWasserstein
import numpy as np

# Single sensitive attribute
fw = FairWasserstein(sigma=0.001)
fair_preds = fw.fit_transform(
    y_calib, sensitive_calib,   # calibration: model scores + group labels
    y_test, sensitive_test,     # test: scores to be corrected
    epsilon=0.0,                # 0 = full Strong Demographic Parity
)

# Multiple sensitive attributes: gender then age band
mw = MultiWasserstein(sigma=0.001)
mw.fit(y_calib, sensitive_features_calib)  # sensitive_features: DataFrame with columns [gender, age_band]
fair_preds = mw.transform(
    y_test,
    sensitive_features_test,
    epsilon=[0.0, 0.1],  # per-attribute trade-off
)

# See the sequential fairness path
print(mw.get_sequential_fairness())
# {'gender': 0.042, 'age_band': 0.031}  ← unfairness remaining after each correction
```

The visualisation tools — waterfall plots showing the sequential unfairness reduction, phase diagrams plotting risk against unfairness at different epsilon values — are well designed and directly useful for understanding the accuracy-fairness frontier.

The current version is 0.0.11a0.dev0, pre-alpha, published March 2025. The pinned dependencies (scikit-learn==1.3.0, pandas==2.0.3, matplotlib==3.7.2) will conflict with anything that has moved forward in the past year. This is normal for academic pre-release code, not a criticism — it means you should isolate it in its own environment.

---

## What insurance-fairness does

insurance-fairness approaches the problem from UK regulatory obligations rather than from a fairness criterion.

The starting question is: does this pricing model constitute indirect discrimination under Section 19 of the Equality Act 2010, via proxy factors? The FCA's Consumer Duty (PRIN 2A, FG22/5) requires firms to demonstrate fair outcomes across groups sharing protected characteristics. Citizens Advice (2022) estimated a £280/year ethnicity penalty in UK motor insurance, driven entirely through postcode — proxy discrimination without any protected attribute in the model.

The library's approach to proxy detection is the piece that matters most, and it is different from anything in the academic fairness literature:

```python
from insurance_fairness import detect_proxies

result = detect_proxies(
    df,
    protected_col="ethnicity_proxy",      # ONS 2021 Census LSOA ethnicity proportion, joined by postcode
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "occupation"],
    run_proxy_r2=True,    # CatBoost proxy R-squared
    run_mutual_info=True, # mutual information (catches non-linear relationships)
    run_partial_corr=True,
)
print(result.flagged_factors)   # ['postcode_district']
print(result.to_polars())
# shape: (4, 4)
# ┌──────────────────────┬───────────┬──────┬──────────────┐
# │ factor               ┆ proxy_r2  ┆ mi   ┆ rag          │
# │ ---                  ┆ ---       ┆ ---  ┆ ---          │
# │ str                  ┆ f64       ┆ f64  ┆ str          │
# ╞══════════════════════╪═══════════╪══════╪══════════════╡
# │ postcode_district    ┆ 0.62      ┆ 0.41 ┆ red          │
# │ occupation           ┆ 0.08      ┆ 0.09 ┆ amber        │
# │ vehicle_age          ┆ 0.02      ┆ 0.03 ┆ green        │
# │ ncd_years            ┆ 0.01      ┆ 0.02 ┆ green        │
# └──────────────────────┴───────────┴──────┴──────────────┘
```

The proxy R-squared of 0.62 for postcode_district is the critical result. Standard Spearman correlation returns |r| ≈ 0.10 on the same data — below any threshold you would set — and finds nothing. The CatBoost proxy R-squared finds it because postcode encodes ethnicity non-linearly: London inner areas vs outer areas vs rural areas have very different ethnic compositions that produce a non-monotone relationship. Manual correlation checks miss this entirely. Across 50 seeds on a 20,000-policy synthetic UK motor portfolio replicating the Citizens Advice finding, the library returns 100% detection rate; Spearman returns 0%.

The full audit workflow is:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)
report = audit.run()
report.summary()
# Overall RAG: RED
# Proxy detection: postcode_district — proxy R²=0.62 [RED], MI=0.41 [RED]
# Calibration by group (gender): max A/E disparity 0.081 [AMBER]
# Demographic parity log-ratio: +0.082 (ratio 1.085) [AMBER]
report.to_markdown("audit_q4_2024.md")
```

The `to_markdown()` output includes cross-references to PRIN 2A, FG22/5, and Equality Act section 19 — the specific provisions a pricing committee or FCA reviewer would look for. The report includes a sign-off table. This is not window dressing; the FCA's multi-firm Consumer Duty review (2024) found that most Fair Value Assessments were "high-level summaries with little substance." A report that maps each finding to the specific regulatory obligation it evidences is substantively different from a statistical summary.

The library also contains a growing set of correction tools — marginal fairness via distortion risk measures (Huang & Pesenti, 2025), discrimination-insensitive reweighting (Miao & Pesenti, 2026), multicalibration correction — but these are supplementary to the core audit function.

---

## Why they are complements, not alternatives

The question "should I use EquiPy or insurance-fairness?" is the wrong question. They sit at different points in the workflow:

```
1. BUILD model (GBM, GLM)
        ↓
2. AUDIT: Does it discriminate? How? Against what legal standard?
   → insurance-fairness: proxy detection, calibration by group, FCA-mapped report
        ↓
3. DECIDE: Is this correctable? What does the regulator expect?
   → legal/actuarial judgement; is the factor justifiable under s.19?
        ↓
4. CORRECT if required
   → insurance-fairness: discrimination-insensitive reweighting, marginal fairness correction
   → EquiPy: distributional correction via Wasserstein barycenter post-processing
        ↓
5. RE-AUDIT corrected model
   → insurance-fairness: run FairnessAudit on the corrected predictions
```

EquiPy's correction is theoretically stronger in one respect: the Wasserstein barycenter approach gives you Strong Demographic Parity — the full conditional distribution is equalised, not just the mean. Our `demographic_parity_ratio` is a Level 1 mean-parity check. We have tail-parity auditing (`TailParityAudit`, following Le, Denis and Hebiri, [arXiv:2604.02017](https://arxiv.org/abs/2604.02017)) on the roadmap, but full-distribution equalisation is not what the FCA's current guidance targets.

insurance-fairness's correction is more insurance-specific: the marginal fairness correction (v0.5.0) operates on distortion risk measures — Expected Shortfall, Wang transform — which are the risk measures relevant to a UK pricing actuary, not the regression score framing in EquiPy. The discrimination-insensitive reweighter (v0.6.3) operates at training time on the data weights, which preserves the model structure rather than post-processing the outputs.

For a UK pricing team working under Consumer Duty and Equality Act obligations, the practical answer is: use insurance-fairness for the mandatory audit work; consider EquiPy when you need to understand the distributional fairness properties of the corrected scores, particularly for multi-attribute intersectional analysis.

---

## What EquiPy does better

This is worth being explicit about.

**The multi-attribute sequential framework is more rigorous.** `MultiWasserstein` handles the case where fairness with respect to gender and age band simultaneously is the target. insurance-fairness audits protected characteristics independently and does not currently provide an intersectional distributional correction. For a UK pricing team that is asked to demonstrate fairness across gender, age, and disability simultaneously — which Consumer Duty Outcome 4 can require — EquiPy's sequential framework is the right tool for the corrective step.

**The visualisation is better.** Waterfall plots showing the sequential reduction in unfairness across attributes, and phase diagrams showing the full (Risk, Unfairness) frontier, are exactly what you want to show a pricing committee when explaining the accuracy cost of a fairness intervention. We have RAG status outputs; we do not yet have this kind of frontier analysis tooling.

**The academic pedigree is impeccable.** Charpentier's group have been the most productive academic team working on insurance fairness for a decade. The paper behind EquiPy (arXiv:2309.06627) is methodologically rigorous. The connection to multi-marginal optimal transport theory is exact, not approximate. For teams that need to defend the methodology to an academic or Solvency II model governance audience, EquiPy's theoretical foundations are straightforward to cite.

---

## What EquiPy does not do

**It does not produce regulatory evidence.** EquiPy returns corrected prediction arrays. It does not tell you whether the original predictions constituted indirect discrimination. It does not produce a document you can put in a Consumer Duty evidence pack. It does not map its findings to PRIN 2A, FG22/5, or section 19 of the Equality Act.

**It does not detect proxies.** EquiPy takes in sensitive attributes and corrects for them. It does not tell you which of your rating factors are acting as proxies for those sensitive attributes. If you do not know that postcode_district has proxy R² = 0.62 against ethnicity, EquiPy will not tell you. You need the detection step first.

**It is pre-alpha.** Version 0.0.11a0.dev0 is not a production release. The pinned dependencies are a significant practical constraint — scikit-learn==1.3.0 is two major versions behind as of April 2026. This is not a reason not to use it; it is a reason not to use it in a live pricing system without isolation.

**The fairness criterion may not match regulatory expectations.** Strong Demographic Parity — the full conditional distribution of predictions is identical across groups — is a stronger criterion than the FCA's current guidance targets. The FCA is focused on proxies for protected characteristics and the resulting price disparity; it has not (as of the April 2025 EP25/2 guidance) required full distributional equalisation. A model corrected to Strong Demographic Parity may satisfy a regulator who cares about mean disparity and proxy contamination; it also may introduce pricing inaccuracies that a regulator focused on affordability in the other direction will question. This is an active policy question, not a settled one.

---

## Practical guidance

**If you have never audited your pricing model for proxy discrimination:** start with insurance-fairness. Run `detect_proxies`. Find out whether your postcode or occupation factor has a proxy R-squared above 0.10. If it does, you have a compliance problem to diagnose before you consider correction.

**If you have a clean audit and want to understand the distributional fairness properties of your model:** EquiPy's (Risk, Unfairness) phase diagrams are valuable. They show you the frontier — how much accuracy you would sacrifice to achieve different degrees of Strong Demographic Parity — and they do this for multiple sensitive attributes in sequence.

**If you need to make a corrective intervention and demonstrate it to the FCA:** the correction tool should be matched to the type of model. For a GLM or distortion risk measure premium, `MarginalFairnessPremium` in insurance-fairness applies a closed-form correction that preserves the actuarial structure of the model. For a score-based model where post-processing is acceptable and you need distributional equalisation, EquiPy's `FairWasserstein` is well suited.

**If you are working on intersectional fairness across multiple protected attributes:** EquiPy's `MultiWasserstein` is currently the better tool for the corrective step. We do not yet have a comparable multi-attribute distributional correction.

Neither library is a compliance safe harbour. Whether a specific model constitutes indirect discrimination under section 19 requires legal and actuarial judgement. Both libraries produce evidence and corrections; neither substitutes for the judgement.

---

## References

Fernandes Machado, A., Grondin, S., Ratz, P., Charpentier, A. and Hu, F. (2025). EquiPy: Implementing Algorithmic Fairness for Multiple Sensitive Attributes in Python. arXiv:2503.09866. Submitted 12 March 2025.

Charpentier, A., Hu, F., Ratz, P. and others (2023). A Sequentially Fair Mechanism for Multiple Sensitive Attributes. arXiv:2309.06627.

Le, N.S., Denis, C. and Hebiri, M. (2026). Demographic Parity Tails for Regression. arXiv:2604.02017.

Huang, F. and Pesenti, S.M. (2025). Marginal Fairness: Fair Decision-Making under Risk Measures. arXiv:2505.18895.

Miao, K.E. and Pesenti, S.M. (2026). Discrimination-Insensitive Pricing. arXiv:2603.16720.

Citizens Advice (2022). Priced Out: An investigation into how motor insurance premiums vary by race. Available at citizensadvice.org.uk.

---

Related on Burning Cost:

- [Fairness in the Tail: Why Mean Demographic Parity Is the Wrong Test](/2026/04/04/demographic-parity-tails-regression-insurance-fairness/) — Le et al. tail parity framework
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/) — `FairnessAudit` and proxy detection in practice
- [Per-Policyholder Proxy Discrimination Scores](/2026/04/03/sensitivity-proxy-discrimination/) — instance-level vulnerability scores for Consumer Duty evidence
- [NSGA-II Multi-Objective Fairness for Insurance Pricing](/blog/2026/04/04/nsga2-multi-objective-fairness-pareto-boonen-2512-24747/) — the Boonen et al. Pareto-front approach that uses EquiPy-style mitigation as one objective alongside accuracy
