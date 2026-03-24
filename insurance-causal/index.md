---
layout: page
title: "insurance-causal"
description: "Double machine learning for deconfounding rating factors. Causal forest for heterogeneous treatment effects."
permalink: /insurance-causal/
---

[GitHub](https://github.com/burning-cost/insurance-causal) &middot; `pip install insurance-causal` &middot; [Full documentation](https://github.com/burning-cost/insurance-causal#readme)

Standard GLM coefficients are biased wherever rating variables correlate with distribution channel or policyholder selection. DML removes that bias without a structural model. v0.4.0 adds causal forest heterogeneous treatment effects: GATES aggregates, CLAN segment profiling, and RATE/AUTOC/QINI targeting evaluation for segment-level price response.

---

## Expected Performance

Benchmarked on synthetic UK motor data with a known ground-truth treatment effect of −0.15 (n=20,000, unobserved confounder DGP). Run on Databricks serverless, 2026-03-21.

- **DML bias vs true effect: ~2–5%**, compared to 15–20% for a naive Poisson GLM — the GLM estimate would lead a pricing team to set a telematics discount 15–20% too aggressively
- **95% CI covers the true effect** with DML; the naive GLM CI does not
- **Causal forest segment RMSE: ~0.06–0.10** vs ~0.08–0.12 for a GLM with interaction terms, with the additional benefit of per-policy CATEs and formal heterogeneity tests
- **Minimum practical n: ~1,000** with `cv_folds=3`; at n &lt; 500 per segment the library warns and reduces iterations automatically
- **Small-sample improvement (v0.3.0+):** at n=5,000, bias reduced from 30–55% (v0.2.x) to 8–20% through sample-size-adaptive nuisance parameters
- **Fit time:** ~60s for 5-fold CatBoost at n=20,000 on Databricks serverless

| Metric | Naive Poisson GLM | DML (insurance-causal) |
|--------|------------------|------------------------|
| Bias (% of true effect) | ~15–20% | ~2–5% |
| 95% CI covers truth? | No | Yes |
| Per-policy CATE | No | Yes |
| Confounding correction | No | Yes (DML) |

Full benchmark methodology: `notebooks/benchmark.py` in the repo.

---

## Related Libraries

- [insurance-fairness](https://github.com/burning-cost/insurance-fairness) — proxy discrimination auditing: causal inference establishes whether a rating factor genuinely drives risk
- [insurance-optimise](https://github.com/burning-cost/insurance-optimise) — demand-curve pricing using the elasticity estimates from this library
- [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) — SDID-based causal evaluation of rate changes
