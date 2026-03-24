---
layout: page
title: "insurance-fairness"
description: "Proxy discrimination auditing aligned to FCA Consumer Duty. Non-linear proxy detection, SHAP price-impact scoring, counterfactual analysis."
permalink: /insurance-fairness/
---

[GitHub](https://github.com/burning-cost/insurance-fairness) &middot; `pip install insurance-fairness` &middot; [Full documentation](https://github.com/burning-cost/insurance-fairness#readme)

Quantifies indirect discrimination risk from rating variables correlated with protected characteristics. Proxy R², mutual information, and SHAP proxy scores where standard correlation checks fail. Equality Act 2010 proportionality documentation built in.

---

## Expected Performance

Benchmarked on 20,000 synthetic UK motor policies with a known postcode-ethnicity proxy structure. Run on Databricks serverless in 4.1s end-to-end.

- **100% detection rate across 50 seeds** — the library flags postcode as a proxy in every independent draw; Spearman correlation misses it in 0 out of 50 (0% detection)
- **Proxy R² = 0.78 for postcode_area** vs Spearman r = 0.06 — the non-linear relationship is invisible to rank correlation but trivial for CatBoost to recover
- **SHAP proxy score = 0.75** — the proxy is not dormant; it actively propagates into model prices via the postcode loading channel
- **Financial impact:** high-diversity policyholders (inner London postcodes) pay ~12% more on average; postcode loading contributes ~£58.57/policy to that differential, totalling ~£238k/year for the high-diversity group in the benchmark portfolio
- **Proxy R² computation time: 0.5s** for 6 factors at n=20,000 (CatBoost, 80 iterations)
- **Zero false positives** across all clean factors in the benchmark

| Factor | Spearman r (manual) | Proxy R² (library) | Library status |
|--------|--------------------|--------------------|----------------|
| postcode_area | 0.06 | **0.78** | RED |
| vehicle_group | 0.02 | 0.00 | GREEN |
| ncd_years | -0.01 | 0.00 | GREEN |

The Spearman threshold (|r| > 0.25) gives 0/6 factors flagged. The library gives 1/6 — the correct answer.

Full benchmark methodology: `benchmarks/run_benchmark.py` in the repo.

---

## Related Libraries

- [insurance-causal](https://github.com/burning-cost/insurance-causal) — establishes whether a rating factor genuinely drives risk or proxies a protected characteristic
- [insurance-governance](https://github.com/burning-cost/insurance-governance) — PRA SS1/23 model risk reports incorporating fairness findings
