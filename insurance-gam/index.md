---
layout: page
title: "insurance-gam"
description: "Interpretable GAMs for insurance pricing. EBM tariffs, Actuarial NAM, Pairwise Interaction Networks, and exact Shapley values."
permalink: /insurance-gam/
---

[GitHub](https://github.com/burning-cost/insurance-gam) &middot; `pip install insurance-gam` &middot; [Full documentation](https://github.com/burning-cost/insurance-gam#readme)

EBM and Neural Additive Model for interpretable deep learning in insurance pricing. Shape functions per rating factor give the transparency of a GLM with the predictive power of a neural network. Factor table output suitable for pricing committee review without post-hoc SHAP.

---

## Expected Performance

Benchmarked on 10,000 synthetic UK motor policies (75/25 train/test). DGP includes four non-linear effects a standard GLM cannot fully represent: U-shaped driver age hazard, exponential NCD discount, hard vehicle age threshold, and log-miles loading. Run on Databricks serverless, 2026-03-22.

- **EBM Gini: −0.329** vs GLM Gini: −0.455 — EBM ranks risks ~28% better than the GLM on this non-linear DGP (higher absolute value = better discrimination)
- **Shape functions are directly auditable** — the driver age curve and NCD discount are recovered from data without any feature engineering, polynomial terms, or SHAP post-processing
- **Where the GLM is competitive:** on a correctly-specified DGP where polynomial terms capture the main non-linearity, deviance is essentially at oracle; EBM's advantage is in shape recovery and risk ordering
- **Fit time:** EBM 60–120s (single-threaded boosting loop) vs GLM &lt;1s on Databricks serverless — fit cost is one-off; scoring time is fast for both
- **Deviance caveat:** EBM exposure handling via offsets can introduce calibration scale error on some DGPs; use Gini as the primary comparison metric and validate deviance on your specific DGP

| Model | Poisson Deviance | Gini |
|-------|-----------------|------|
| Oracle (true DGP) | 0.2508 | −0.460 |
| Poisson GLM (linear+quad) | 0.2528 | −0.455 |
| InsuranceEBM (interactions=3x) | see note | −0.329 |

Use InsuranceEBM when risk ordering matters more than calibrated counts — reinsurance pricing, underwriting scores, portfolio selection — or when you need shape functions the GLM cannot express.

Full benchmark methodology: `benchmarks/run_benchmark_databricks.py` in the repo.

---

## Related Libraries

- [shap-relativities](https://github.com/burning-cost/shap-relativities) — extracts multiplicative relativities from GBMs when you need a simpler output format
- [insurance-distill](https://github.com/burning-cost/insurance-distill) — distils GBM predictions into surrogate GLM factor tables for legacy rating engines
