---
layout: post
title: "Does Automated Model Monitoring Actually Work for Insurance Pricing?"
date: 2026-03-23
featured: true
author: Burning Cost
categories: [monitoring, model-risk, libraries, benchmarks]
tags: [insurance-monitoring, PSI, CSI, ae-ratio, gini-drift, mSPRT, sequential-testing, PITMonitor, model-drift, calibration, murphy-decomposition, does-it-work, benchmarks, python, uk-insurance, polars]
description: "We tested insurance-monitoring against aggregate A/E on synthetic drift scenarios. The headline result: aggregate A/E misses localised drift that PSI catches in roughly one month. Here is what actually happened and what it cannot do."
---

This is the fourth post in our series where we test our own libraries honestly against standard approaches. The test is simple: does the library actually detect what it claims to detect, and does it do so before the damage is done?

This time: `insurance-monitoring` versus the standard approach every pricing team runs - aggregate actual-to-expected monitoring.

---

## The problem we were trying to solve

Pricing models do not fail loudly. They do not crash; they do not return an error. They silently misprice. An underwriting year goes by. The loss ratio edges up. The rate action is 6%, when it should have been 9%, because nobody knew the model had started misfiring ten months earlier.

The aggregate A/E ratio is the standard signal. Compute actual claims divided by model expected claims across the whole portfolio. If it is between 0.95 and 1.05, the model is within tolerance. If it is outside, investigate.

The problem is not that A/E is wrong. The problem is that errors cancel.

A model that is 15% cheap on under-25 drivers and 15% expensive on mature drivers reads 1.00 at portfolio level if those two segments are about the same size. No alarm. Both segments are being actively mispriced. The net A/E looks perfect.

This is not a contrived edge case. It is the normal operating condition for a deployed pricing model. UK motor books see continuous compositional change: comparison site growth shifts age and channel mix, telematics adoption clusters by segment, NCD discounting changes who shops and who stays. The model was trained on one portfolio and is running on a different one, incrementally, every month. Aggregate A/E will often be stable precisely because the errors stay balanced as both sides of the cancellation grow together.

---

## What we tested

We ran `insurance-monitoring` against a manual aggregate A/E check on four scenarios. Three benchmark scripts cover the core claims.

**Scenario 1: Three simultaneous failure modes (benchmarks/benchmark.py)**

50,000-policy reference portfolio. 15,000-policy monitoring cohort. Three failure modes planted deliberately:

1. Covariate shift: young drivers (18-30) oversampled 2x in the monitoring period
2. Calibration drift: new vehicles (age < 3 years) have claims inflated 25%
3. Discrimination decay: 30% of monitoring-period predictions randomised

This is deliberately adversarial. In practice you rarely get all three at once. But we wanted to know whether the library can distinguish the failure modes - not just signal "something is wrong" but say which something and which response it warrants.

**Scenario 2: Time-to-detection race (benchmarks/benchmark.py, walk-forward mode)**

Same DGP, but we walk through the monitoring cohort in 500-policy batches, one per simulated month. At each step: does aggregate A/E breach its 5% threshold? Does PSI on driver_age breach the RED threshold (PSI > 0.25)?

**Scenario 3: Sequential testing FPR (benchmarks/benchmark_sequential.py)**

10,000 Monte Carlo simulations under H0 - no true effect. Analyst checks results monthly for 24 months. Fixed-horizon t-test versus mSPRT.

**Scenario 4: Repeated calibration testing (benchmarks/benchmark_pit.py)**

500 well-calibrated Poisson observations, then 500 with 15% rate inflation. Hosmer-Lemeshow checked every 50 observations versus PITMonitor updated per observation.

---

## What happened: the headline numbers

### Scenario 1 results at fixed snapshot (50k reference / 15k monitoring)

| Check | Manual A/E | MonitoringReport |
|---|---|---|
| Aggregate A/E ratio | 0.942 | 0.942 |
| Statistical CI on A/E | Not computed | Yes - Garwood exact |
| Segment A/E drift | Not detected | Detected |
| Driver age covariate shift (PSI) | Not detected | AMBER (PSI = 0.21) |
| Discrimination decay | Not detected | REFIT flag |
| Murphy local MCB | Not computed | 0.0090 (REFIT signal) |
| Gini drift p-value | N/A | 0.76 at n=15k |
| Structured recommendation | INVESTIGATE | REFIT |

The aggregate A/E of 0.942 sits just outside the 0.95-1.05 green band, so a manual check correctly raises an alert. But it cannot tell you which of the three failure modes is present, whether to recalibrate or refit, or which segment to examine first. `MonitoringReport` identifies all three failure modes and recommends REFIT, derived from the Murphy decomposition: local MCB (0.0090) substantially exceeds global MCB (0.0002), which means the model's ranking is broken, not just its overall level. A recalibration would have fixed the wrong thing.

The Gini drift test returns p=0.76 at 15,000 monitoring policies. This is not a failure of the test - it is the test correctly reporting that 15,000 observations is insufficient power to detect a Gini drop of -0.012. The test is appropriately conservative. At the same DGP, 50,000 monitoring policies produces z=-1.9, p=0.054.

### Scenario 2: time-to-detection

This is the result that matters most operationally.

In the 50,000/15,000 scenario with 2x young driver oversampling and 25% new-vehicle claims inflation, the covariate shift and calibration drift partially cancel at portfolio level. The aggregate A/E stays within the 0.95-1.05 band for the entire 15,000-policy monitoring period. It never fires.

PSI on driver_age hits the RED threshold (PSI > 0.25) at approximately 1,000-1,500 policies. On a book writing 1,250 new policies per month, that is roughly one month into the monitoring period.

The aggregate A/E would not have detected this scenario at all. PSI detected it within the first monthly batch.

This is the central operational argument for the library. PSI measures compositional shift directly from feature distributions, before any claims arrive. It does not require a year of claims development. It does not require the errors to be large enough to breach an aggregate threshold. It measures whether the population you are monitoring is the same population the model was trained on - which is the right leading indicator.

### Scenario 3: sequential testing FPR

| Method | Nominal FPR | Actual FPR (monthly peeking) |
|---|---|---|
| Fixed-horizon t-test | 5% | ~25% |
| mSPRT (SequentialTest) | 5% | ~1% |

The 25% FPR for the fixed-horizon t-test assumes the analyst checks monthly and stops if the test is significant - which is standard practice in UK motor champion/challenger runs. Nobody runs a 24-month test and genuinely does not look until it completes. The mSPRT holds at 5% at all stopping times by design (the e-process guarantee from Johari et al. 2022).

Under H1 (challenger 10% cheaper on frequency), mSPRT reaches a decision in a median of 8 months on 500 policies per arm per month. A pre-registered t-test at 24 months would get there eventually, but with 16 months of unnecessary waiting and five times the false positive rate during interim checks.

### Scenario 4: PITMonitor vs repeated Hosmer-Lemeshow

| Method | Nominal FPR | Empirical FPR (stable phase) |
|---|---|---|
| Hosmer-Lemeshow (every 50 obs) | 5% | ~46% (10 looks) |
| PITMonitor | 5% | ~3% (300 simulations) |

With 10 monthly checks on a stable model, Hosmer-Lemeshow raises a spurious alarm roughly 46% of the time. This is not a deficiency in the test - it is a deficiency in applying a fixed-horizon test in a repeated-testing context. Teams that do this train themselves to ignore alarms. When real calibration drift occurs, the signal is buried in noise.

PITMonitor's e-process (Henzi, Murph, Ziegel 2025, arXiv:2603.13156) stays near zero during the stable phase and rises sharply when calibration genuinely shifts. In the benchmark, it correctly identifies the changepoint at t~500 within ±30 steps.

---

## What the library gets right

**The cancellation problem.** This is the headline. Aggregate A/E is structurally blind to scenarios where the model is wrong in offsetting directions across segments. PSI per feature is not fooled by this, because it measures compositional shift at the feature level, not the portfolio level.

**The RECALIBRATE vs REFIT distinction.** This is the operationally expensive question. A recalibration - adjusting the intercept or adding an offset - takes an afternoon. A full refit takes two to eight weeks of data science time, plus governance, plus potentially regulatory notification under PRA SS3/17. Getting this wrong in either direction is costly. The Murphy decomposition (Lindholm & Wüthrich, Scandinavian Actuarial Journal 2025) resolves it: if global MCB dominates, the model is wrong in the mean but right in its ranking - recalibrate. If local MCB dominates, the ranking itself is broken - refit. A standard A/E dashboard cannot answer this question.

**The repeated-testing problem.** Most monitoring tools ignore it. Both SequentialTest and PITMonitor solve it explicitly by constructing e-processes with anytime-valid guarantees. The FPR stays at the nominal level regardless of how often you check.

**Exposure weighting.** Unweighted PSI treats a 30-day moped policy the same as a 12-month fleet policy. If your short-term business skews young, unweighted PSI understates driver age drift. Exposure-weighted PSI - which `insurance-monitoring` does via the `exposure_weights` parameter - gives the correct population-level picture.

---

## What it cannot do

We said we were being honest, so here is the other half.

**PSI and CSI do not tell you why the distribution shifted.** A RED flag on driver_age means the age distribution has changed. It does not tell you whether that is organic growth, a new comparison site channel, a competitor pulling from a specific segment, or a data quality issue upstream. The `InterpretableDriftDetector` (TRIPODD, Panda et al. 2025, arXiv:2503.06606) adds feature-attribution to tell you *which* features are driving model loss changes, but distinguishing benign mix change from model-invalidating shift still requires domain judgement.

**The Gini drift test is underpowered below 5,000-10,000 monitoring observations.** At n=4,000, the test returns p=0.76 against a 30% randomisation of predictions. That is not a false negative - it is an honest report of insufficient evidence. If your monitoring window is thin (quarterly book, specialist product, new entrant cohort), the Gini test should be treated as indicative only. The Murphy decomposition local MCB component is more sensitive at small n.

**A/E ratios are unreliable on immature accident periods.** For frequency models, you need at least 12 months of claims development before the A/E is reliable. For liability, 24+ months. The library does not apply IBNR adjustment - it works on the numbers you give it. Calibrating on undeveloped data makes the model look like it is over-predicting when it is not. Apply chain-ladder factors first.

**PITMonitor detects changes in calibration, not absolute miscalibration.** A model that has been biased since launch and remains consistently biased will not trigger PITMonitor. The e-process tests for changes in the PIT distribution. Use `CalibrationChecker` at model sign-off to catch pre-existing miscalibration; use `PITMonitor` for post-launch change detection. Conflating the two will miss a stubbornly miscalibrated model entirely.

**The REFIT vs RECALIBRATE recommendation is a heuristic, not an oracle.** The Murphy decomposition decision rule (local MCB > global MCB implies REFIT) is derived from the arXiv 2510.04556 framework, but the thresholds are set operationally. When both global and local miscalibration are present simultaneously, the recommendation can point to RECALIBRATE when a full refit is warranted. Treat it as a structured starting point for a model review, not a decision that bypasses actuarial judgement.

---

## Our position

Aggregate A/E monitoring is not optional - you need to run it. But running only aggregate A/E is like checking your building's fire alarm system by asking "did we have any fires last year?" It is a lagging indicator. By the time the aggregate number moves, the misfiring has been compounding for months.

The specific combination that PSI-per-feature detection fires at 1,000-1,500 policies while aggregate A/E never fires at all in the same scenario is the result that we think justifies adding granular monitoring to a pricing stack. That is not a cherry-picked scenario. It is the standard case: a book with covariate shift and calibration drift that partially cancel at portfolio level. This describes most deployed UK motor books six months after their last refit.

The library is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).

```bash
uv add insurance-monitoring
```

Run the benchmarks yourself to verify:

```bash
python benchmarks/benchmark.py
python benchmarks/benchmark_sequential.py
python benchmarks/benchmark_pit.py
```

---

## See also

- [Three-Layer Drift Detection: What PSI and A/E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) - the conceptual case for multi-layer monitoring
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) - detailed walkthrough of each monitoring layer
- [NannyML vs insurance-monitoring](/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/) - how the library compares to general-purpose tools
- [Alibi Detect vs insurance-monitoring](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) - the same comparison against Alibi Detect
