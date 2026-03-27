---
layout: post
title: "Does automated model monitoring actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [model-monitoring, validation, drift-detection, psi, csi, gini, murphy-decomposition, motor, pricing]
description: "Aggregate A/E at 0.94 looks fine. The model has been mispricing under-25s for eight months. Benchmark results on a synthetic UK motor book with three planted failure modes."
---

The claim for automated model monitoring is straightforward: deployed pricing models go stale, aggregate A/E monitoring misses most of it, and feature-level drift statistics catch the problems earlier. That is the kind of claim a pricing actuary can test. Three failure modes, one synthetic book, run the numbers.

So we ran the benchmark. Known data-generating process, 50,000-policy UK motor reference portfolio, three deliberately induced failure modes planted in 15,000 monitoring-period policies. We compared what a manual aggregate A/E check finds against what [`insurance-monitoring`](/insurance-monitoring/)'s full monitoring stack finds. Here is what the numbers show.

---

## The setup

50,000 synthetic motor policies in the reference period. 15,000 in the monitoring period. The model was trained on the reference period and deployed. Then we planted three failure modes in the monitoring cohort:

1. **Covariate shift** — young drivers (18–30) oversampled 2× in the monitoring period. The portfolio is ageing down; the model was calibrated on an older book.
2. **Calibration drift** — claims for new vehicles (age < 3 years) inflated 25%. The model is cheap on one segment.
3. **Discrimination decay** — 30% of monitoring-period predictions randomised. The model's ranking is substantially broken.

The three failure modes are designed to interact: the covariate shift and calibration drift partially cancel at portfolio level. Cheaper pricing on young drivers partially offsets expensive pricing on older-vehicle segments. This is the canonical scenario where aggregate A/E looks acceptable while individual segments are badly mispriced.

Benchmark run 23 March 2026, seed=42, Polars-native throughout.

The comparison is a manual aggregate A/E ratio (the standard dashboard metric on most UK motor pricing teams) against `MonitoringReport` from `insurance-monitoring`, which wraps exposure-weighted PSI per feature, A/E with Poisson confidence intervals, the Gini drift z-test (arXiv 2510.04556), and the Murphy decomposition.

---

## The results

### What aggregate A/E sees

| Period | A/E ratio | 95% CI | Verdict |
|--------|-----------|--------|---------|
| Reference | 0.9624 | — | In band |
| Monitoring | 0.9420 | 0.918–0.968 | INVESTIGATE |

The monitoring-period A/E of 0.9420 falls just outside the standard 0.95–1.05 green band. A manual check raises an INVESTIGATE flag — which is better than nothing. But it cannot tell you whether this is a recalibration job (adjust the intercept, done in hours) or a refit (rebuild the model, done in weeks). It also cannot tell you which segment is driving it, or whether the book composition has shifted so that this problem will get worse next quarter even if you do nothing.

Critically, in this scenario the aggregate A/E does not breach threshold at all until late in the monitoring period. The errors are cancelling. The 0.94 reading after 15,000 policies is the result of a model that is simultaneously too cheap on young urban drivers and too expensive on older-vehicle segments; the net at portfolio level looks almost acceptable.

### What granular monitoring sees

The full detection picture at the same 15,000-policy snapshot:

| Check | Manual A/E | MonitoringReport |
|-------|-----------|-----------------|
| Aggregate A/E | Yes (0.9420) | Yes (0.9420) |
| Statistical CI on A/E | No | Yes (0.918–0.968) |
| Driver age covariate shift (PSI = 0.21) | Not detected | AMBER |
| Vehicle age calibration drift | Not detected | Detected (Murphy) |
| Discrimination decay (30% randomised predictions) | Not detected | REFIT flag |
| Gini change | Not computed | −0.012 |
| Gini drift p-value | N/A | 0.76 (underpowered at n=4,000) |
| Murphy local MCB | Not computed | 0.0090 — REFIT |
| Murphy global MCB | Not computed | 0.0002 |
| Final recommendation | INVESTIGATE | REFIT |

Two of the three planted failure modes are completely invisible to aggregate A/E. Covariate shift does not show in A/E until claims develop — which takes 12 months minimum for UK motor. Discrimination decay does not show in A/E at all: a model whose ranking is broken but whose average prediction is still calibrated will produce A/E near 1.0 indefinitely.

The Murphy decomposition is the thing that makes the REFIT vs RECALIBRATE decision defensible. Local MCB of 0.0090 against global MCB of 0.0002 means the model's local structure is broken — rank-preserving scale adjustments will not fix it. REFIT. When that refit is triggered, consider whether the replacement model should be an [interpretable GAM](/2026/03/14/insurance-gam-interpretable-nonlinearity/) whose shape functions will make the next monitoring comparison easier to explain. If the numbers were reversed (global MCB dominating), you multiply all predictions by the A/E ratio, ship it, and schedule a proper refit. The distinction is weeks of work.

### Time-to-detection: aggregate A/E vs PSI

The snapshot comparison is informative. The operational question is sharper: if you watch policies accumulate month by month, which approach raises the alarm first?

The benchmark walks through the monitoring cohort in 500-policy batches. On a 1,250-policy/month book, this represents roughly one batch every two weeks. In the 50,000/15,000 scenario:

- **Aggregate A/E (5% breach threshold):** Never fires. The covariate shift and calibration drift cancel at portfolio level. The aggregate A/E stays within 0.95–1.05 across the entire 15,000-policy monitoring period.
- **PSI on driver_age (RED threshold, PSI > 0.25):** Fires at approximately 1,000–1,500 policies — roughly one month into monitoring on a 1,250-policy/month book.

Eight to ten times earlier. And in this scenario, aggregate A/E would never have triggered an investigation at all.

The reason is structural. PSI measures the compositional shift directly, before claims are observed. You do not need a year of claims development to know that your monitoring-period book contains twice as many under-25s as your training data. That is a feature distribution comparison; it runs on the policy data the day the policies are issued.

---

## What breaks

### The Gini test is underpowered below about 5,000 monitoring policies

At n=4,000 monitoring policies, the Gini drift z-test returns p=0.76 even against a 30% prediction randomisation. This is correct behaviour — the test has insufficient power at that sample size to distinguish a Gini drop of 0.012 from noise. At 15,000 policies, the same DGP produces z ≈ −1.9, p ≈ 0.06. If your monitoring window is thin — a quarterly book with fewer than 5,000 renewals — treat the Gini test as indicative only. The Murphy local MCB component is more sensitive at small n and will often flag REFIT before the Gini test reaches significance.

### A/E is unreliable on immature accident periods

The library makes no IBNR adjustment. Pass it 9-month-old motor claims and it will report a model that over-predicts, because 25–30% of ultimate claims have not emerged yet. For UK motor, only accident years that are at least 12 months developed are suitable inputs; apply chain-ladder factors first for anything more recent. This is not a monitoring problem specifically — it affects every A/E calculation — but it catches teams off guard when they pipe in the most recent quarter and see an A/E of 0.75.

### PSI flags that a distribution shifted — it does not explain why

PSI on driver_age hitting RED tells you the age distribution in your monitoring cohort differs from your training data. It does not tell you whether that is a genuine portfolio mix shift (new broker, changed marketing), a data quality issue (missing DOBs defaulting to 25), or a seasonal effect. PSI is the alarm; investigation is still required. `InterpretableDriftDetector` gets closer to attribution by measuring how each feature's contribution to model loss has changed, but domain judgement remains necessary.

### False positive control at scale

The default PSI threshold (RED at 0.25) was calibrated for credit-scoring applications where feature distributions are more stable than motor insurance. On UK motor with seasonal and economic-cycle variation in vehicle mix and driver demographics, you will see PSI values of 0.1–0.15 on stable books during normal portfolio evolution. The `MonitoringThresholds` module lets you customise these — tightening for large books where you have high statistical confidence in distribution estimates, loosening for specialist lines where natural variation is higher. The defaults are a starting point.

### Repeated testing inflates false positive rates

The PITMonitor comparison with repeated Hosmer-Lemeshow is stark. Repeated H-L checked every 50 observations produces a 46% false alarm rate during the stable phase — nine times the nominal 5%. The library's e-process approach holds empirical FPR at ~3% across 300 simulations. If you are running a calibration test monthly in production and stopping to investigate every time it trips, and you are using a fixed-horizon test, you are almost certainly investigating non-existent problems roughly half the time.

---

## Our read

The benchmark result is unambiguous. On a synthetic UK motor book with realistic offsetting drift — the pattern where a model is cheap on one segment and expensive on another — aggregate A/E monitoring never raises an alarm. PSI fires within the first monthly batch. Two of three failure modes are entirely invisible to A/E; the granular monitoring stack catches all three.

The limitations are real. The Gini test needs volume. IBNR is your problem to handle before you pass in the data. PSI thresholds need calibration for your book. But these are limitations you should expect from any monitoring tool, and they are smaller than the limitation of running aggregate A/E alone on a book where segment errors offset each other.

We think the central argument in the library's README is correct: aggregate A/E is a portfolio-level accounting check, not a model health check. A model that prices under-25s 15% cheap and over-25s 4% expensive will report A/E near 1.0 if 20% of the book is under-25. That is not a model that is performing well — it is a model that is performing well on average in a way that is about to get much worse as the book composition evolves.

The Murphy decomposition's REFIT vs RECALIBRATE distinction is the piece of the library that is hardest to replicate with a manual spreadsheet approach, and it is the decision that has the most operational consequence. Getting it wrong costs weeks.

The library is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). The full benchmark is in . The Databricks notebook is at [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py).

---

- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — the full library introduction: exposure-weighted PSI, Gini drift z-test, Murphy decomposition, and mSPRT sequential testing
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — once monitoring flags a problem and a refit is done, the re-validation evidence flows back into the model inventory via run_id
- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — when the Murphy decomposition says refit, this is what the replacement model should look like
