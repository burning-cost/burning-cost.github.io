---
layout: post
title: "Does Automated Model Monitoring Actually Work?"
date: 2026-03-27
categories: [validation]
tags: [model-monitoring, drift-detection, A-E, PSI, pricing, python]
description: "We planted three simultaneous model failures in a 50,000-policy UK motor book. The aggregate A/E never triggered. The library detected the first problem after 1,500 policies. Here is what that means in practice."
---

The aggregate actual-to-expected ratio is the monitoring check every UK pricing team runs. It is also, in our benchmarks, blind to two of the three most common model failure modes. We planted deliberate failures in a 50,000-policy synthetic book and watched what each method could see.

The answer to the headline question is: yes, automated monitoring works, but only if you monitor the right things. Monitoring only the aggregate A/E ratio is not monitoring — it is creating a false sense of security.

---

## The experiment

We set up a 50,000-policy reference period and a 15,000-policy monitoring period with three simultaneous failure modes:

1. **Covariate shift** — young drivers (18–30) oversampled 2× in the monitoring period
2. **Calibration drift** — new vehicles (age < 3) have claims inflated 25%
3. **Discrimination decay** — 30% of monitoring-period predictions randomised

Then we ran both a manual aggregate A/E check and [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring)'s `MonitoringReport`.

---

## What the aggregate A/E sees

The aggregate A/E ratio in the monitoring period sits close to 1.0. The overpriced older vehicles cancel the underpriced young drivers at portfolio level. Manual check says: no action needed.

This is not a contrived failure mode. A model that is 15% cheap on young urban drivers (say 20% of the book) and 3.75% expensive on everyone else produces an aggregate A/E of exactly 1.00. Both errors continue growing as the portfolio composition shifts. Neither is visible without segment-level decomposition.

Discrimination decay — 30% of predictions randomised — is invisible to A/E monitoring entirely. A model losing its ranking power needs a refit, not a recalibration. The aggregate A/E cannot tell you which one you need.

---

## What the library sees

| Check | Manual A/E | MonitoringReport |
|---|---|---|
| Aggregate A/E ratio | Yes | Yes |
| Statistical CI on A/E | No | Yes |
| Segment-level A/E drift | No | Yes |
| Driver age covariate shift (PSI) | No | Yes — RED flag |
| Gini discrimination decay | No | Yes |
| Murphy decomposition (REFIT vs RECALIBRATE) | No | Yes |
| Structured recommendation | No | Yes |
| Policies to first alarm | Never | ~1,500 (PSI RED) |

PSI on driver age hits the RED threshold (PSI > 0.25) after approximately 1,000–1,500 policies — roughly one month into the monitoring period on a 1,250-policy/month book. The aggregate A/E never breaches its 5% threshold across the entire 15,000-policy monitoring window.

The timing matters. Detecting a covariate shift one month in means you can investigate and recalibrate before the claims deterioration materialises in the A/E. Detecting it 18 months in — which is when most teams notice via a deteriorating combined ratio — means you have been mispricing that segment for a year and a half.

---

## Failure mode detection by method

Which monitoring method catches which problem:

| Failure mode | Aggregate A/E | PSI / CSI | Segment A/E | Gini drift test | Murphy decomp |
|---|---|---|---|---|---|
| Covariate shift (feature dist change) | No | Yes | Partial | No | No |
| Calibration drift (global scale) | Yes | No | Yes | No | Yes |
| Discrimination loss (ranking decay) | No | No | No | Yes | Yes |
| Sudden changepoint | Delayed | Delayed | Delayed | Delayed | No |

No single check is sufficient. The aggregate A/E catches calibration drift but is blind to covariate shift and discrimination loss — two of the three failure modes in this benchmark.

---

## The sequential testing result

The benchmark also covers champion/challenger testing, which most teams run as a fixed-horizon t-test with monthly reviews. Under H₀ (no true effect), a fixed-horizon t-test peeked monthly reaches an actual false positive rate of ~25% — five times the nominal 5%. You will declare a winner on roughly one in four genuine no-difference experiments if you check monthly and use a fixed-horizon test.

The mSPRT sequential test holds at ~1% FPR under the same peeking schedule, across 10,000 Monte Carlo simulations. Under H₁ (challenger 10% cheaper on frequency), mSPRT detects the effect in a median of 8 months on a 500-policy-per-arm-per-month book. A pre-registered t-test at 24 months would reach the same conclusion 16 months later.

---

## The API

```python
from insurance_monitoring import MonitoringReport
import polars as pl

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,   # pl.DataFrame with named rating factors
    feature_df_current=feat_cur,
    reference_exposure=exp_ref,
    exposure=exp_cur,
)

# MonitoringReport runs all checks on construction
print(report.recommendation)   # REFIT / RECALIBRATE / NO_ACTION
print(report.to_polars())      # traffic-light per check
```

The `recommendation` is deliberately opinionated: REFIT means ranking has degraded and a new model is required; RECALIBRATE means ranking is intact but predictions are off-scale and a multiplicative adjustment will fix it. That distinction is worth weeks of effort — recalibration is a morning's work, a full refit is a project.

---

## Limitations

The benchmark uses synthetic data with planted failures. In practice, covariate shifts are gradual and overlapping rather than step functions. PSI will still detect them earlier than aggregate A/E, but the timing advantage will vary with the severity of the shift.

The Gini drift z-test requires a bootstrap step (200 replicates by default) for statistical inference. At 50k reference / 15k monitoring policies this takes 3–5 minutes. For weekly monitoring on large books run this on Databricks; local runs are comfortable at 10k/4k.

The Murphy decomposition (REFIT vs RECALIBRATE) uses the Lindholm-Wüthrich (SAJ 2025) framework and requires Poisson or Tweedie outcomes. For binary outcomes (renewal/lapse), use the Gini drift z-test as the discrimination check and segment A/E for calibration.

---

## Verdict

Use automated monitoring if:
- You are running monthly or quarterly model health checks and currently relying on aggregate A/E as the primary signal — you need at minimum PSI per rating factor and a Gini drift test alongside the A/E
- You run champion/challenger experiments with interim looks — mSPRT is essential; fixed-horizon tests with monthly reviews are running at five times the stated false positive rate
- You need PRA SS3/17 model risk audit trail documentation

Do not expect it to:
- Replace the judgement call about whether an observed shift is worth acting on — that requires understanding the book and the claims environment
- Detect failures outside the training distribution — if an entirely new risk type enters the book the library can flag it as covariate shift, but it cannot assess whether the model is appropriate for it

```bash
uv add insurance-monitoring
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-monitoring). Start with `benchmarks/benchmark.py` for the failure-mode detection matrix, then `benchmarks/benchmark_sequential.py` for the champion/challenger results.

- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) — the full technical post explaining PSI, segmented A/E, and the Gini z-test that powers this benchmark
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the governance layer: how to register model monitoring findings in the MRM inventory
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/)
