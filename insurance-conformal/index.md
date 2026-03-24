---
layout: page
title: "insurance-conformal"
description: "Distribution-free prediction intervals for insurance GBM and GLM pricing models with finite-sample coverage guarantees."
permalink: /insurance-conformal/
---

[GitHub](https://github.com/burning-cost/insurance-conformal) &middot; `pip install insurance-conformal` &middot; [Full documentation](https://github.com/burning-cost/insurance-conformal#readme)

Five non-conformity scores tuned for Tweedie and Poisson claims. Frequency-severity conformal intervals, online retrospective adjustment (RetroAdj), Solvency II SCR bounds. The `pearson_weighted` score is the default and produces narrower intervals than parametric approaches at identical coverage targets. Current: v0.6.0.

---

## Expected Performance

Benchmarked on 50,000 synthetic UK motor policies (CatBoost Tweedie forecast, Gamma DGP with heteroskedastic tails), temporal 60/20/20 split. Run on Databricks serverless, 2026-03-21.

- **13.4% narrower intervals** than parametric Tweedie at identical 90% aggregate coverage (£3,806 vs £4,393 mean width)
- **Locally-weighted conformal:** 11.7% narrower than parametric, 90.6% top-decile coverage — the right choice when per-decile coverage matters for reinsurance attachment decisions
- **Aggregate coverage at 90%:** conformal 90.2% vs parametric 93.1% — the parametric approach over-covers low-risk policies and wastes premium capacity
- **RetroAdj vs ACI after 30% claims inflation:** RetroAdj recovers 90% coverage in ~15–30 steps; ACI takes ~80–150 steps at the same gamma — 3–8x faster recovery after abrupt shifts
- **Calibration set guidance:** interval widths are stable above n_cal ≈ 2,000; below that, widths fluctuate 20–30% across seeds
- **Benchmark time:** 4s on 50,000 policies

| Metric | Parametric Tweedie | Conformal (pearson_weighted) | LW Conformal |
|--------|-------------------|------------------------------|--------------|
| Aggregate coverage @ 90% | 0.931 | 0.902 | 0.903 |
| Worst-decile coverage | 0.904 | 0.879 | **0.906** |
| Mean interval width (£) | 4,393 | **3,806** | 3,881 |
| Distribution-free guarantee | No | Yes | Yes |

Always run `coverage_by_decile()` after calibration — the marginal guarantee holds on average, not per-decile.

Full benchmark methodology: `benchmarks/benchmark_gbm.py` in the repo.

---

## Related Libraries

- [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) — monitor whether interval coverage degrades after deployment
- [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) — Sarmanov copula joint modelling; conformal intervals for two-part models
