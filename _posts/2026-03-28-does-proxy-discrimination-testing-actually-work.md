---
layout: post
title: "Does Proxy Discrimination Testing Actually Work?"
date: 2026-03-28
categories: [validation]
tags: [fairness, proxy-discrimination, fca, consumer-duty, equality-act, pricing, python]
description: "Manual Spearman correlation missed postcode as an ethnicity proxy in 100% of 50 benchmark runs. CatBoost proxy R-squared caught it in 100% of runs. The difference is the non-linearity of the relationship."
---

The question pricing teams should be asking is not "does my model use a protected characteristic?" — it does not, because that would be illegal. The question is "does my model use factors that are correlated with a protected characteristic in a way that a court would call indirect discrimination under Section 19 of the Equality Act 2010?"

That is a harder question, and the standard approach to answering it is not adequate.

---

## What most teams currently do

The conventional proxy check is a Spearman rank correlation between each rating factor and the protected characteristic. Set a threshold — typically |r| > 0.25 — and flag anything above it. It is fast, it produces a number, and it fits in a single column of a compliance spreadsheet.

We benchmarked this against [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) across 50 random seeds on 20,000 synthetic UK motor policies with a known postcode-ethnicity proxy relationship — the structure replicating the Citizens Advice (2022) finding that postcode loading drives a roughly £280/year ethnicity penalty in UK motor.

---

## The benchmark result

| Metric | Manual Spearman (threshold |r| > 0.25) | Library proxy R² + MI |
|---|---|---|
| Postcode flagged as proxy | No (|r| ≈ 0.10, below threshold) | Yes (proxy R² ≈ 0.62, RED) |
| Factors correctly flagged (out of 6) | 0/6 | 1–2/6 |
| Detection rate (50 seeds) | 0/50 | 50/50 |
| Mean postcode Spearman |r| | ~0.10 | N/A |
| Mean postcode proxy R² | N/A | ~0.62 (std ≈ 0.02) |
| Captures non-linear proxy | No | Yes |
| Quantifies financial impact | No | Yes (£/policy) |

The Spearman check detects nothing, across all 50 seeds, at any reasonable threshold. The library detects the proxy with a mean R² of 0.62 in every single run.

This is not a failure of the threshold. The Spearman |r| of 0.10 is genuinely small. The issue is that postcode area encodes ethnicity non-linearly: Inner London postcodes have high ethnic diversity, outer postcodes lower, rural postcodes the lowest — but within each postcode area the relationship is not monotone, and the Spearman statistic, which measures rank correlation, cannot detect non-monotone categorical-to-continuous relationships.

The library fits a CatBoost model predicting the protected attribute from each rating factor in isolation. Proxy R² of 0.62 means postcode area explains 62% of variance in the diversity score. That number survives scrutiny under Equality Act Section 19 and is the kind of finding a regulator can act on.

---

## The financial impact

The benchmark also quantifies the premium differential. High-diversity policyholders pay roughly £70–90 more per year than low-diversity policyholders, driven through the postcode area loading. This is the number a pricing committee needs to decide whether to act.

A Spearman correlation of 0.10 does not produce this number. A proxy R² of 0.62 demands it.

---

## The API

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    df=policies_df,
    protected_col="diversity_score",
    rating_factors=["postcode_area", "vehicle_group", "ncd_years", "age_band", "mileage", "payment_method"],
    premium_col="annual_premium",
    exposure_col="exposure",
)

report = audit.run()
print(report.proxy_summary())
# postcode_area: proxy_r2=0.62, mi=0.41, flag=RED
# vehicle_group: proxy_r2=0.08, mi=0.06, flag=GREEN

report.to_markdown("fairness_audit.md")  # FCA-ready report with regulatory mapping
```

The `to_markdown()` output includes explicit regulatory cross-references to PRIN 2A, TR24/2, and Equality Act Section 19. It is structured for a pricing committee pack — one document that can go to legal, compliance, and the pricing team simultaneously.

---

## The action/outcome distinction

One finding from the benchmark is less obvious but more important. Minimising premium disparity (action fairness) does not minimise loss ratio disparity (outcome fairness). On a 20,000-policy TPLI portfolio, the policy with the most equal premiums produced the most unequal loss ratios.

This is the compliance gap FCA TR24/2 (August 2024) described. Firms were auditing at the point of quoting — checking whether premiums are equal across groups — and missing the Consumer Duty Outcome 4 obligation, which is a post-sale value question. A firm paying claims at the same rate across groups regardless of risk has cross-subsidised protected groups, which looks fair but is not economically sustainable. A firm pricing to true risk produces premium differentials that may look unfair but are the correct actuarial answer.

The `DoubleFairnessAudit` class in the library computes the Pareto front across action fairness, outcome fairness, and revenue. A pricing committee can see every operating point along the trade-off, with quantified evidence of the choice being made. A firm that can only show a single demographic parity ratio cannot demonstrate the same level of considered decision-making to the FCA.

---

## The FCA context

TR24/2 (August 2024) found most insurers' Fair Value Assessments were "high-level summaries with little substance". Six Consumer Duty investigations are open; two directly involve insurers on fair value grounds. This is live enforcement risk, not theoretical.

Citizens Advice (2022) put the ethnicity penalty at £280/year and £213 million per year in aggregate for UK motor — and this is driven by postcode factors that every standard motor pricing model uses. The question is whether your book has the same structure. A Spearman check at |r| = 0.10 will not tell you.

---

## Limitations

The proxy detection works on the factors in isolation — it tests whether postcode area is a proxy for ethnicity when considered alone. It does not test for indirect proxying through combinations of factors. If vehicle group and payment method are individually clean but jointly correlated with a protected characteristic, the single-factor proxy test misses it. The SHAP proxy score addresses this partially by tracing price impact back through the full model structure.

Computation time for proxy R² is 12–15 seconds per factor on 50,000 policies (80 CatBoost iterations). For a 20-factor tariff, run on Databricks. The mutual information check is a useful first pass — under 5 seconds for the full factor set — to identify candidates before running CatBoost proxy detection.

---

## Verdict

Use proxy discrimination testing if:
- You use postcode area or any geographic factor in UK motor or home pricing — the Citizens Advice (2022) finding is strong enough that you should assume the proxy exists until you have evidence otherwise
- You are preparing for an FCA Consumer Duty review or have TR24/2 on your risk register
- You are going to commission the work anyway as part of a Fair Value Assessment — you might as well do it properly once rather than producing a number that will not survive scrutiny

The Spearman correlation check at |r| > 0.25 is not adequate for demonstrating absence of proxy discrimination. A CatBoost proxy R² of 0.10 or above on any rating factor warrants investigation. At 0.62, you have a documented problem and a £/policy number that quantifies it.

```bash
uv add insurance-fairness
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-fairness). The Monte Carlo sensitivity benchmark (`benchmarks/benchmark_sensitivity.py`) runs 50 seeds and produces the detection rate table.

- [Your Pricing Model Might Be Discriminating](/2026/03/07/your-pricing-model-might-be-discriminating/)
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)
