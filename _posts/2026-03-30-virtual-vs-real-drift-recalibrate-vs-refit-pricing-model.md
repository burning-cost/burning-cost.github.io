---
layout: post
title: "Virtual vs Real Concept Drift: When to Recalibrate vs Refit Your Pricing Model"
date: 2026-03-30
categories: [model-monitoring]
tags: [concept-drift, virtual-drift, real-drift, murphy-decomposition, gini-drift, recalibrate, refit, insurance-monitoring, arXiv-2510.04556, calibration, discrimination, python, uk-insurance]
description: "Your A/E ratio is drifting. Is it the portfolio mix, or has the underlying loss mechanism changed? The answer determines whether you update the intercept or commission a refit. A diagnostic framework — with code — for getting it right."
seo_title: "Virtual vs Real Concept Drift in Insurance Pricing Models: Recalibrate or Refit?"
---

Your quarterly monitoring report lands and the overall A/E ratio has moved to 1.09. The model is 9% cheap. Someone raises the idea of a recalibration — "just adjust the intercept". Someone else says that feels like a sticking plaster. The meeting ends without a decision.

This is one of the most consequential calls in model governance: do you recalibrate (a half-day job, stays within existing governance) or refit (three months, Model Committee sign-off, new validation cycle)? In practice, the decision is made on instinct and competing pressures. It should be made on evidence.

The formal framework for that evidence comes from Brauer, Menzel, and Wüthrich (arXiv:2510.04556, December 2025). Their contribution is a diagnostic decision tree based on three tests that tell you *which kind* of drift you have. The kind determines the fix.

---

## The distinction that matters: virtual vs real

These two terms get used loosely. They have precise meanings.

**Virtual drift** (also called covariate shift): the distribution of inputs P(X) has changed, but the relationship between inputs and outcomes P(Y|X) has not. Your portfolio now writes more young drivers than it did when the model was trained. But for any given young driver with given characteristics, the model's predicted rate is still correct — the model's learned structure is valid. The portfolio has moved to a part of the feature space where the model was always right; there is just more volume there now.

In virtual drift, a single global recalibration — multiplying all predictions by `sum(actual) / sum(predicted)` — mostly works. The model's ranking of risks is unchanged. The factors and relativities are right. The level is wrong because the mix changed.

**Real drift** (concept drift proper): P(Y|X) has changed. The actual underlying loss rate for a given risk profile is no longer what the model was trained on. Maybe claims inflation has hit young drivers harder than old. Maybe a fraudulent repair network has embedded itself in one postcode group. Maybe a new ADAS technology has genuinely changed the risk profile of vehicles with it. The model's structure — its relativities, its factor ordering, its implicit assumptions about the relationship between driver age and claims frequency — is wrong.

A recalibration scales a wrong shape by a constant. The overall A/E improves. The segment-level A/E stays broken.

---

## Why this is hard to diagnose from A/E alone

A/E ratio at portfolio level cannot distinguish these cases. Both virtual and real drift can produce a 1.09 overall A/E. Segment-level A/E helps — if it is flat across all segments, that points to virtual drift; if it is systematically wrong in specific cohorts (young versus old, urban versus rural), that points to real drift. But segment-level A/E is a visual check, not a test, and most breakdowns are noisy enough to be inconclusive.

The [Murphy decomposition](/2026/02/28/recalibrate-or-refit/) gives you GMCB and LMCB as numbers. GMCB >> LMCB means a global scale fix is the right action. LMCB >> GMCB means the model structure is wrong. That is the right starting point, but it still does not answer the prior question: is the model's *ranking* intact? Because if ranking has degraded, no amount of recalibration recovers it — you are scaling a model that is no longer ordering risks correctly.

The piece that was missing until the Brauer-Menzel-Wüthrich paper was a formal hypothesis test for Gini drift. Without it, a Gini point estimate that moved from 0.41 to 0.38 might be noise (50,000 policies, two quarters of data) or it might be real signal. There was no principled way to decide.

---

## The three diagnostics

Brauer et al. formalise three tests that together constitute a decision framework. The logic is sequential.

### 1. Gini ranking test

Test whether the model's discrimination — its ability to rank high-risk from low-risk policyholders — has changed between the reference period and the monitoring period.

The paper establishes that the Gini estimator is asymptotically normal:

```
sqrt(n) * (Ĝ - G) → N(0, σ²)
```

where σ² is estimated by non-parametric bootstrap. For two independent periods, the variances are additive, giving a two-sample z-test:

```
z = (G_monitor - G_reference) / sqrt(Var_bootstrap(G_ref) + Var_bootstrap(G_mon))
```

If the Gini has dropped significantly, the model's ranking structure has changed. This is real drift — recalibration will not fix it. Stop here and schedule a refit.

If the Gini is stable, the ranking is intact. The model still orders risks correctly. Whatever is causing the A/E drift is in the calibration, not the structure.

The paper recommends `alpha = 0.32` (one-sigma rule) for routine monitoring — flagging earlier at the cost of more false positives that require manual review. Use `alpha = 0.05` for governance escalation or regulatory reporting.

### 2. Murphy decomposition (GMCB vs LMCB split)

Once you know the ranking is stable, the next question is: what kind of calibration problem do you have? The [Murphy score decomposition](/2026/02/28/recalibrate-or-refit/) splits total deviance as:

```
D(y, ŷ) = UNC - DSC + MCB
MCB = GMCB + LMCB
```

- **GMCB**: miscalibration that a single global multiplier (`alpha = sum(Y) / sum(Ŷ)`) removes. Virtual drift, or pure price-level inflation, shows up here.
- **LMCB**: miscalibration that persists after balance correction. The model's price structure is wrong in specific segments — even after adjusting the level.

If GMCB dominates: recalibrate. If LMCB dominates: refit, even if the Gini test passed (the ranking metric is averages-based; a model can maintain portfolio-level Gini while being structurally wrong in pockets).

### 3. Local autocalibration test

The autocalibration check bins policies by predicted rate and tests whether each cohort is self-financing (E[Y | ŷ] = ŷ within each bin). It answers: is the LMCB concentrated in specific prediction intervals, or spread uniformly?

Concentrated LMCB in specific deciles (say, the highest-rate decile is systematically underestimated) is a governance red flag — it suggests a specific segment of the portfolio is being mispriced, not just an overall level shift.

---

## The decision tree

Put the three tests together:

```
Is Gini drift significant?
├── YES → Real drift. REFIT. (Ranking has degraded; recalibration is insufficient.)
└── NO  → Is LMCB > GMCB?
          ├── YES → Real drift, ranking-stable. REFIT. (Structure is wrong; level is not.)
          └── NO  → Is GMCB material?
                    ├── YES → Virtual drift. RECALIBRATE. (Level is wrong; structure is fine.)
                    └── NO  → OK. (No material drift.)
```

Three outcomes: refit, recalibrate, or leave it alone. Each has a different governance path. The Gini test is the critical fork — it separates the cases where recalibration is safe from the cases where it is misleading.

---

## What insurance-monitoring gives you right now

We built all three tests into [insurance-monitoring](https://github.com/actuarialopensource/insurance-monitoring). They are standalone functions; what does not yet exist is a single `DriftReport` entry point that runs the full sequence automatically. That is the next priority for the library. For now, you run them in order:

```python
import numpy as np
from insurance_monitoring.gini_drift import GiniDriftTest
from insurance_monitoring.calibration._murphy import murphy_decomposition
from insurance_monitoring.calibration._autocal import check_auto_calibration
from insurance_monitoring.drift import psi

# --- Step 0: covariate shift check ---
# High PSI flags virtual drift in the input features.
# This does not tell you whether model structure is affected — that is
# what the Gini test is for.
driver_age_psi = psi(
    reference=ref_df["driver_age"].to_numpy(),
    current=mon_df["driver_age"].to_numpy(),
    n_bins=10,
)
print(f"Driver age PSI: {driver_age_psi:.3f}")
# < 0.10: stable | 0.10-0.25: monitor | > 0.25: act

# --- Step 1: Gini ranking test ---
gini_test = GiniDriftTest(
    reference_actual=ref_df["claims"].to_numpy(),
    reference_predicted=ref_df["predicted_freq"].to_numpy(),
    monitor_actual=mon_df["claims"].to_numpy(),
    monitor_predicted=mon_df["predicted_freq"].to_numpy(),
    reference_exposure=ref_df["exposure"].to_numpy(),
    monitor_exposure=mon_df["exposure"].to_numpy(),
    n_bootstrap=500,   # 500 for governance reports; 200 for routine monitoring
    alpha=0.32,        # one-sigma rule for routine monitoring
    random_state=42,
)
result = gini_test.test()
print(gini_test.summary())

if result.significant and result.delta < 0:
    print("Gini has declined significantly. Real drift. Schedule refit.")
    # Stop here — Murphy decomposition will tell you more, but the verdict is refit.

# --- Step 2: Murphy decomposition ---
murphy = murphy_decomposition(
    y=mon_df["claims"].to_numpy(),
    y_hat=mon_df["predicted_freq"].to_numpy(),
    exposure=mon_df["exposure"].to_numpy(),
    distribution="poisson",
)
print(f"Verdict: {murphy.verdict}")
print(f"GMCB: {murphy.global_mcb:.4f}  LMCB: {murphy.local_mcb:.4f}")
print(f"DSC: {murphy.discrimination:.4f}  MCB: {murphy.miscalibration:.4f}")

# Verdict is one of: 'OK', 'RECALIBRATE', 'REFIT'
# RECALIBRATE: alpha = sum(Y) / sum(Ŷ); update intercept or multiply predictions
# REFIT: structural change required; escalate to Model Committee

# --- Step 3: Local autocalibration ---
if murphy.verdict != "OK":
    autocal = check_auto_calibration(
        y=mon_df["claims"].to_numpy(),
        y_hat=mon_df["predicted_freq"].to_numpy(),
        exposure=mon_df["exposure"].to_numpy(),
        n_bins=10,
    )
    print(f"Autocalibration p-value: {autocal.p_value:.4f}")
    print(f"Worst bin ratio (obs/pred): {autocal.worst_bin_ratio:.3f}")
```

The `murphy.verdict` field encodes the GMCB vs LMCB logic directly. `'RECALIBRATE'` means GMCB >= LMCB; `'REFIT'` means LMCB > GMCB. The local autocalibration check tells you which deciles are the problem — useful for a refit brief.

---

## What the library does not yet handle

Two gaps are worth naming.

**IBNR**: the framework assumes mature, finalised loss data. If you are running quarterly monitoring on partially-developed claims (as most UK motor teams do), you need an IBNR adjustment layer before passing data to these tests. The library does not provide this. A naïve run on partially-developed data will flag calibration drift that is purely development pattern, not model drift.

**Gradual trend vs level shift**: `GiniDriftTest` and `murphy_decomposition` compare two windows — reference vs monitor. They detect level shifts between the windows. They do not detect monotonic trend across rolling quarters. If your Gini is declining 0.01 each quarter, these batch tests will miss it until the cumulative drop crosses the significance threshold. The `PITMonitor` class in the library handles sequential monitoring, but it does not yet integrate with the Murphy decomposition output.

Both are on the roadmap.

---

## Governance language

One underrated contribution of the Brauer-Menzel-Wüthrich framework is that it gives pricing teams cleaner language for governance escalation.

Before this framework: "The A/E is 1.09. We think we should recalibrate but it might need a refit."

After: "The Gini test shows no significant ranking degradation (z = -0.8, p = 0.42). The Murphy decomposition shows GMCB = 0.0034, LMCB = 0.0004. The miscalibration is dominated by a global level shift. We are recalibrating. This is within existing change governance — no Model Committee referral required."

Or: "The Gini test shows significant ranking degradation (z = -2.4, p = 0.016). LMCB > GMCB. The model's structural accuracy has changed. This is real concept drift. A refit is required. Escalating to Model Committee."

The first statement tells the committee exactly why recalibration is safe. The second tells them exactly why it is not. Both are defensible under internal model risk management policy, with documentation aligned to model risk management best practice (the PRA's SS1/23 principles for banks provide a useful reference framework, though the direct regulatory obligation for insurers flows through Solvency II Article 124 and PRA Supervisory Statement SS4/16).

---

## What is new here vs existing practice

The short answer: the Gini hypothesis test.

Tracking Gini over time is standard. Computing a p-value for whether a given change is noise or signal was not possible before Theorem 1 of arXiv:2510.04556 established the asymptotic distribution. That is the single most actionable change for any UK pricing team currently watching a Gini trend and not knowing what to do with it.

The GMCB/LMCB split already existed in actuarial literature. The autocalibration check has analogues in Hosmer-Lemeshow tests. The new contribution is the *integrated decision tree* and the formal test for ranking degradation that sits at the top of it.

PSI remains the right tool for detecting covariate shift in input features. The Brauer-Menzel-Wüthrich framework does not replace it — it sits on top of it. PSI tells you the portfolio has changed. The Gini test tells you whether that matters for model performance. They answer different questions.

---

## Further reading

- [Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question](/2026/02/28/recalibrate-or-refit/) — the operational mechanics of reading GMCB/LMCB once an alert fires
- [Detecting Model Decay: Gini Drift Testing with Statistical Power](/2026/03/25/gini-drift-testing-statistical-power/) — deeper treatment of the asymptotic theory and power calculations for the Gini test
- [Does PSI Actually Catch Pricing Model Drift?](/2026/03/28/does-psi-model-monitoring-actually-catch-pricing-model-drift/) — empirical comparison: where PSI succeeds, where it fails, and what the Gini test catches that PSI misses
- [Three-Layer Drift Detection: What PSI and A/E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) — building a monitoring stack from feature drift through to discrimination testing
