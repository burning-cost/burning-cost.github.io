---
layout: page
title: "insurance-credibility"
description: "Bühlmann-Straub credibility in Python. Caps thin segments, stabilises NCD factors, blends new models with incumbents."
permalink: /insurance-credibility/
---

[GitHub](https://github.com/burning-cost/insurance-credibility) &middot; `pip install insurance-credibility` &middot; [Full documentation](https://github.com/burning-cost/insurance-credibility#readme)

Credibility models for UK non-life pricing: Bühlmann-Straub group credibility and Bayesian experience rating. Mixed-model equivalence checks. The actuarial answer to the thin data problem — gives thin segments proportionally less weight than thick segments, not the same weight or zero weight.

---

## Expected Performance

Benchmarked on a synthetic panel of 30 scheme segments, 5 accident years, 64,302 total policy-years with known structural parameters. Run on Databricks serverless.

- **Credibility beats raw experience on thin and medium tiers** — on thin schemes (&lt;500 exposure), credibility MAE (0.0069) beats raw (0.0074) and portfolio average (0.0596) by wide margins
- **Ties on thick tiers** (2000+ exposure) where Z approaches 1.0 — credibility and raw converge, which is correct behaviour
- **Portfolio average is the worst method in all tiers** — it ignores genuine between-scheme variation and costs you on large schemes where the evidence is unambiguous
- **Portfolio mean recovery: within 1.4%** of true value (mu_hat=0.6593 vs true=0.6500)
- **Experience rating vs flat NCD table:** credibility shrinkage outperforms raw frequency ratio (single bad year gets partial weight); A/E calibration deviation is lower than NCD's discrete bands
- **Fit time: under 5 seconds** on a 150-row panel
- **K estimation:** needs at least 50–100 policies with 2+ years of history; conservative K (over-estimated at small panel size) means the model shrinks more aggressively than theory dictates — safe for thin groups

| Tier | Raw MAE | Portfolio avg MAE | Credibility MAE | Winner |
|------|---------|------------------|-----------------|--------|
| Thin (&lt;500 exp) | 0.0074 | 0.0596 | **0.0069** | Credibility |
| Medium (500–2000) | 0.0030 | 0.0423 | **0.0029** | Credibility |
| Thick (2000+) | 0.0014 | 0.0337 | 0.0014 | Tie (Z≈1.0) |

Full benchmark methodology: `benchmarks/benchmark.py` in the repo.

---

## Related Libraries

- [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) — smooths the rating curves that credibility factors are applied to
- [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) — transfer learning and GLM bootstrapping for segments too thin for credibility
