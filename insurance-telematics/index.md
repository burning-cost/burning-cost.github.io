---
layout: page
title: "insurance-telematics"
description: "End-to-end pipeline from raw 1Hz GPS/accelerometer data to GLM-compatible risk scores via HMM driving state classification."
permalink: /insurance-telematics/
---

[GitHub](https://github.com/burning-cost/insurance-telematics) &middot; `pip install insurance-telematics` &middot; [Full documentation](https://github.com/burning-cost/insurance-telematics#readme)

HMM driving state classification (cautious/normal/aggressive) from raw telematics trip data. Bühlmann-Straub credibility aggregation to driver level. Outputs GLM-compatible state fraction features. The state fractions capture persistent driving style rather than trip-level noise.

---

## Expected Performance

Benchmarked against raw trip-level feature averages (mean speed, harsh braking rate, harsh acceleration rate, night fraction) in a Poisson GLM on 300 synthetic drivers with 40 trips each. Full methodology: `notebooks/benchmark_telematics.py`.

- **3–8pp Gini improvement** over raw trip averages — the improvement comes from state fractions capturing persistent driving style rather than noisy trip-level averages
- **Better loss ratio separation:** top-to-bottom quintile loss ratio ratio is larger with HMM features — the model puts high-risk drivers into higher predicted deciles more reliably
- **A/E calibration is similar between methods** — the HMM advantage is in discrimination (rank ordering), not overall calibration
- **Full pipeline fit time: 30–90s** on 300 drivers (clean + extract + HMM 200 iterations + GLM). For large fleets, HMM fitting should be parallelised via Spark
- **HMM advantage proportional to DGP structure:** on portfolios where driving style is genuinely regime-based (cautious/aggressive), the benefit is clearest; on portfolios where behaviour is continuous, gains may be smaller

The HMM advantage is because raw averages conflate brief aggressive episodes with sustained aggressive behaviour. A driver who is cautious 90% of the time but occasionally aggressive has different risk from one who is aggressive 50% of the time — state fractions distinguish them; mean speed does not.

Full benchmark methodology: `notebooks/benchmark_telematics.py` in the repo.

---

## Related Libraries

- [insurance-causal](https://github.com/burning-cost/insurance-causal) — establishes whether HMM state fractions causally drive claims or proxy for other risk factors
- [insurance-fairness](https://github.com/burning-cost/insurance-fairness) — telematics scores can act as proxies for protected characteristics; audit before production
- [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) — monitors whether telematics-derived GLM factors remain well-calibrated over time
