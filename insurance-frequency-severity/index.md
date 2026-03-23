---
layout: page
title: "insurance-frequency-severity"
description: "Sarmanov copula joint frequency-severity modelling with GLM marginals. Analytical premium correction for freq-sev dependence."
permalink: /insurance-frequency-severity/
---

[GitHub](https://github.com/burning-cost/insurance-frequency-severity) &middot; `pip install insurance-frequency-severity` &middot; [Full documentation](https://github.com/burning-cost/insurance-frequency-severity#readme)

Tests the independence assumption every two-part pricing model makes. IFM estimation, analytical premium correction (closed-form, no simulation at scoring time), Garrido conditional severity, dependence tests. When freq-sev dependence is present, an independent two-part model systematically misprices the highest-risk policies.

---

## Expected Performance

Benchmarked against an independent two-part model (Poisson GLM × Gamma GLM) on 12,000 synthetic UK motor policies (8,437 train / 3,563 test) with known positive freq-sev dependence via a latent risk score. Run on Databricks serverless, 2026-03-16.

- **28.6% MAE reduction** in pure premium vs oracle (£10.60 vs £14.84 per policy)
- **Portfolio total premium bias reduced from +22.95% to −6.77%** — the independent model over-charges at portfolio level; the correction brings it within 7%
- **Correction factors are small but consistent:** mean 0.943, range p10 0.939–p90 0.950 — high-risk decile gets a 4.8% correction, low-risk 6.0%
- **Fit time overhead: +21%** over the independent model (0.128s vs 0.105s) — effectively zero for the improvement it provides
- **When to use:** run `DependenceTest` first; use the correction when the test indicates positive, statistically significant dependence (p &lt; 0.05)
- **When NOT to use:** when you cannot reject independence, or fewer than ~500 claims — omega estimate is too noisy

| Metric | Independent model | Sarmanov copula | Change |
|--------|------------------|-----------------|--------|
| Pure premium MAE | 14.84 | **10.60** | −28.6% |
| Portfolio total bias | +22.95% | −6.77% | −16.2pp |
| Fit time (s) | 0.105 | 0.128 | +21% |

Full benchmark methodology: `benchmarks/benchmark_insurance_frequency_severity.py` in the repo.

---

## Related Libraries

- [insurance-conformal](https://github.com/burning-cost/insurance-conformal) — conformal prediction intervals for frequency-severity two-part models (Graziadei et al. protocol)
- [insurance-severity](https://github.com/burning-cost/insurance-severity) — heavy-tail severity with composite Pareto models for the severity component
