---
layout: page
title: "insurance-distill"
description: "GBM-to-GLM distillation. Fits a surrogate Poisson/Gamma GLM to CatBoost predictions and exports multiplicative factor tables for Radar/Emblem."
permalink: /insurance-distill/
---

[GitHub](https://github.com/burning-cost/insurance-distill) &middot; `pip install insurance-distill` &middot; [Full documentation](https://github.com/burning-cost/insurance-distill#readme)

Bridges the gap between GBM model development and legacy rating engine deployment. The surrogate GLM learns the GBM's predictions — not the raw claims — which eliminates individual claim noise and gives a cleaner signal for the GLM to fit. Factor tables are multiplicative (log link) by construction: compatible with Radar, Emblem, Guidewire, and most UK personal lines rating engines.

---

## Expected Performance

Benchmarked on Databricks serverless compute using the default `tree` binning strategy.

- **Fit times: 0.4s (n=10k), 1.8s (n=50k), 9.1s (n=250k)** — scales roughly linearly with rows; the dominant cost is the GLM fit in glum
- **Full workflow end-to-end: 0.7s (n=10k), 2.5s (n=50k), 12.3s (n=250k)**
- **Fidelity R²: 90–97%** match between GBM predictions and distilled GLM factors on typical synthetic motor portfolios (7 factors, 80/20 holdout)
- **Segment deviation: 3.6%** max vs 21.4% for a direct GLM — the surrogate is 6× more faithful at cell level, which is the artefact a pricing actuary presents to a CRO
- **Gini ratio guidance:** 10 bins per continuous variable is a reasonable default; dropping to 5 bins typically costs 2–4 Gini ratio points
- **Large portfolio guidance:** above 500,000 policies, pass a stratified subsample to `SurrogateGLM.fit()` and run `report()` on the full dataset — factor tables are evaluated on all data regardless

| Task | n=10,000 | n=50,000 | n=250,000 |
|------|----------|----------|-----------|
| `SurrogateGLM.fit()` | 0.4s | 1.8s | 9.1s |
| Full workflow | 0.7s | 2.5s | 12.3s |

Full benchmark methodology: `benchmarks/` directory in the repo.

---

## Related Libraries

- [shap-relativities](https://github.com/burning-cost/shap-relativities) — alternative approach: extract relativities directly from CatBoost SHAP values rather than fitting a surrogate
- [insurance-gam](https://github.com/burning-cost/insurance-gam) — when you want interpretable shape functions during model building, not just at deployment
