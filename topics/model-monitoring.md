---
layout: page
title: "Insurance Model Monitoring in Python"
description: "Exposure-weighted PSI, CSI, A/E ratios, Gini drift z-tests, and double-lift monitoring for deployed insurance pricing models. PRA SS1/23 and FCA-aligned. Python library."
permalink: /topics/model-monitoring/
---

The standard model monitoring workflow for insurance pricing has a structural defect: it monitors outputs when it should be monitoring inputs. An A/E ratio only moves after claims develop. On a UK motor book, frequency claims develop over three to twelve months. By the time your quarterly A/E flags a segment as mispriced, you have renewed that cohort twice at the wrong rate.

Input monitoring is detectable immediately. A distribution shift in driver age at new business is observable at quote time, months before any claim. Exposure-weighted PSI on model inputs fires fast. Gini coefficient drift on a validation cohort fires at the end of each quarter. Together they give you a layered early warning system that A/E cannot replicate.

The [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) library implements this layered approach: exposure-weighted PSI and CSI for feature drift, actual-vs-expected ratios with Poisson confidence intervals at segment level, and Gini drift z-tests for discriminatory power. All metrics are calibrated to insurance data — exposure offsets, fractional policy years, and zero-inflated frequency distributions are handled natively.

**Library:** [insurance-monitoring on GitHub](https://github.com/burning-cost/insurance-monitoring) &middot; `pip install insurance-monitoring`

---

## Tutorials and introductions

- [Insurance Model Monitoring in Python: Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — the main library tutorial: PSI, CSI, and Gini drift on a deployed CatBoost model
- [Insurance Model Monitoring in Python: Gini Drift, A/E Ratios and Double-Lift Curves](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/) — end-to-end monitoring workflow with double-lift chart generation
- [Insurance Model Monitoring in Python: Gini, A/E, and Double-Lift](/2026/03/22/insurance-model-monitoring-gini-ae-double-lift-python/) — companion post with full Python code walkthrough

---

## Techniques and problem framing

- Your Model Drift Alert Is Too Late — the case for monitoring inputs rather than outputs
- [Three-Layer Drift Detection: What PSI and A&E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) — the full monitoring stack: feature drift, score drift, and performance drift
- [Monthly Covariate Shift Monitoring: When to Reweight and When to Retrain](/2026/03/15/covariate-shift-detection-book-mix-changes/) — density-ratio methods for detecting and correcting covariate shift
- [Motor Model Mispricing Caught by Monitoring: A Case Study](/2026/03/23/motor-model-mispricing-caught-by-monitoring/) — worked example of a real monitoring alert and the underlying cause
- [Gini Drift Testing: Statistical Power and Sample Size](/2026/03/25/gini-drift-testing-statistical-power/) — how many policies you need for a Gini drift test to have meaningful power

---

## Benchmarks and validation

- [Does automated model monitoring actually work for insurance pricing?](/2026/03/27/does-automated-model-monitoring-actually-work/) — empirical test of whether PSI/CSI alerts precede A/E deterioration
- [Does Automated Model Monitoring Actually Work?](/2026/03/27/does-automated-model-monitoring-actually-work/) — extended validation on simulated drift scenarios

---

## Library comparisons

- [Alibi Detect vs insurance-monitoring: Drift Detection for Insurance Pricing Models](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) — Alibi Detect is general-purpose; here is where it fails on insurance-specific requirements
- [NannyML vs insurance-monitoring: Drift Detection Built for Insurance Portfolios](/2026/03/21/nannyml-vs-insurance-monitoring-drift-detection-insurance/) — NannyML's estimated performance monitoring versus exposure-weighted insurance metrics
- [Why Evidently Isn't Enough for Insurance Pricing Model Monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) — Evidently's column-level drift vs. the actuarial metrics that matter for a pricing sign-off
