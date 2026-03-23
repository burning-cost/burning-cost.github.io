---
layout: page
title: "insurance-monitoring"
description: "Exposure-weighted drift detection for deployed pricing models. Anytime-valid A/B testing, calibration change detection, and interpretable drift attribution."
permalink: /insurance-monitoring/
---

[GitHub](https://github.com/burning-cost/insurance-monitoring) &middot; `pip install insurance-monitoring` &middot; [Full documentation](https://github.com/burning-cost/insurance-monitoring#readme)

Post-deployment monitoring for insurance pricing models. PSI/CSI, actual-vs-expected, Gini drift z-test, anytime-valid champion/challenger testing (mSPRT), calibration drift detection (PITMonitor), and feature-attributed drift with FDR control (InterpretableDriftDetector). Current: v0.8.0.

---

## Expected Performance

All benchmarks on synthetic UK motor data. Run on Databricks serverless.

- **MonitoringReport vs manual A/E:** detects three failure modes (covariate shift, calibration drift, discrimination decay) that aggregate A/E misses entirely — runs in under 40 seconds on 14,000 policies including bootstrap variance estimation
- **Time-to-detection:** PSI fires within the first monthly batch (~1,000–1,500 policies) on a covariate shift that aggregate A/E never detects because the errors self-cancel at portfolio level
- **mSPRT vs t-test with peeking:** fixed-horizon t-test with monthly checking inflates FPR to ~25% (5x nominal); mSPRT holds at ~1% at all stopping times
- **PITMonitor FPR: ~3%** vs repeated Hosmer-Lemeshow testing at ~46% (9x inflation) — H-L raises alarms throughout stable periods and causes teams to distrust the monitoring system
- **InterpretableDriftDetector:** correctly attributes drift to drifted features with zero false positives; Benjamini-Hochberg FDR control gives materially more power than Bonferroni at d ≥ 5 rating factors
- **PITMonitor changepoint estimation:** recovers the true drift onset within ±30 steps on typical runs

| Method | Nominal FPR | Actual FPR (monthly peeking) |
|--------|------------|------------------------------|
| Fixed-horizon t-test | 5% | ~25% |
| mSPRT (SequentialTest) | 5% | ~1% |

| Method | Empirical FPR (stable phase) | FPR inflation |
|--------|------------------------------|---------------|
| Repeated Hosmer-Lemeshow | ~46% | 9x |
| PITMonitor | ~3% | 0.6x |

Full benchmark scripts: `benchmarks/` directory in the repo.

---

## Related Libraries

- [insurance-conformal](https://github.com/burning-cost/insurance-conformal) — distribution-free prediction intervals; flag when interval coverage degrades
- [insurance-governance](https://github.com/burning-cost/insurance-governance) — PRA SS1/23 model validation reports incorporating monitoring outputs
