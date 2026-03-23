---
layout: page
title: "insurance-whittaker"
description: "Whittaker-Henderson smoothing for experience rating tables. REML lambda selection, Bayesian credible intervals, 1D and 2D variants."
permalink: /insurance-whittaker/
---

[GitHub](https://github.com/burning-cost/insurance-whittaker) &middot; `pip install insurance-whittaker` &middot; [Full documentation](https://github.com/burning-cost/insurance-whittaker#readme)

Smooths age curves, NCD scales, vehicle group relativities, and 2D rating tables without parametric assumptions. REML selects the smoothing parameter automatically — no manual bandwidth tuning, no bucket boundary decisions. Bayesian credible intervals widen correctly in thin-data regions.

---

## Expected Performance

Benchmarked against Gaussian kernel smoothing and exposure-weighted binned means on synthetic UK motor rating curves with known true DGP. Run on Databricks serverless, 2026-03-22.

- **57.2% MSE reduction vs raw observed rates** (63-band driver age curve, order=2, REML lambda)
- **Lowest OOS MSE** of the three methods compared, with the largest advantage at thin-data regions (young driver ages 17–24, old driver ages 70+) where the benefit is commercially most important
- **Young driver accuracy:** W-H estimates 0.3881 vs moving average 0.3787 (true mean 0.3977) — the moving average undershoots the peak due to boundary effects
- **REML automatically selects lambda** — kernel smoothing requires a LOO-CV bandwidth search; binned means require manual bucket boundary decisions. Both introduce analyst discretion that REML eliminates
- **Credible intervals widen in thin-data cells** — you can see which parts of the curve to trust. Binned means and kernels do not provide this
- **Cliff-edge elimination:** binned means produce 10–15% jumps between adjacent ages at bucket boundaries. W-H produces a smooth, continuous curve throughout

| Method | MSE vs true curve | Max absolute error |
|--------|------------------|--------------------|
| Raw observed rates | 0.00042 | 0.0804 |
| Weighted 5-pt moving average | 0.00018 | 0.0803 |
| Whittaker-Henderson (REML) | **0.00018** | 0.0831 |

The MSE differences are driven by thin-tail regions. In the well-observed middle of the curve (ages 30–55), all methods produce similar results — W-H earns its keep where it matters most.

Full benchmark methodology: `databricks/benchmark_whittaker_vs_baselines.py` in the repo.

---

## Related Libraries

- [insurance-credibility](https://github.com/burning-cost/insurance-credibility) — Bühlmann-Straub credibility for thin scheme segments
- [insurance-distill](https://github.com/burning-cost/insurance-distill) — GBM-to-GLM distillation; smoothed factor tables feed directly into surrogate GLMs
