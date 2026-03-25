---
layout: post
title: "Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question"
date: 2026-05-14
categories: [monitoring, pricing, techniques]
tags: [murphy-decomposition, calibration, model-monitoring, recalibrate, refit, poisson, gini-drift, insurance-monitoring, uk-motor, pra-ss123, python]
description: "Assumes familiarity with the Murphy decomposition framework. Focuses on the operational question: given a monitoring alert, how do you read GMCB vs LMCB..."
---

Your monitoring dashboard has gone amber. The A/E ratio is 1.07 — the model is 7% cheap across the portfolio. The question on the table: do we adjust the intercept this afternoon, or do we commission a full model refit?

This question comes up every time a monitoring alert fires. It sounds like it should have a principled answer. In most UK pricing teams, it is answered by committee instinct: how bad does 1.07 feel? how long has it been since the last refit? who has capacity? These are not the right inputs.

The right input is the Murphy decomposition.

---

## What the decomposition actually measures

A/E ratio of 1.07 tells you the model's predictions are systematically 7% low. It does not tell you *why*. The Murphy decomposition — from Lindholm & Wüthrich (SAJ 2025) and Brauer et al. (arXiv:2510.04556, December 2025) — gives you the two components of miscalibration as separate numbers: GMCB (the portion a single scaling factor fixes) and LMCB (the portion that requires a structural change to the model). The post [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/) covers the full framework, including the balance property test and auto-calibration. This post is about what you do with the GMCB/LMCB split when a monitoring alert fires.

**Global scale error (GMCB)**: the model's shape is correct — it ranks risks right, the relativities between segments are accurate — but the overall level is wrong by a constant. Every prediction needs multiplying by 1.07. This costs an afternoon: update the GLM intercept, or multiply the GBM output by 1.07 in the score function. The ranking, the factor tables, the relativities — none of it changes.

**Local structural error (LMCB)**: the model's shape is wrong. It overprices young drivers and underprices mature drivers in a way that partly cancels at portfolio level. Or it has the vehicle group relativities in the wrong order because the composition of claims has shifted since training. An intercept adjustment does not fix this — you are multiplying a wrong shape by a constant and calling it a recalibration. The errors reduce but they do not go away, and they stay systematically in the wrong places.

---

## The decomposition

For a Poisson frequency model, total deviance decomposes as:

```
D(Y, Ŷ) = UNC + DSC + MCB
```

- **UNC** (uncertainty): the baseline deviance of predicting the portfolio mean for everyone. This is fixed by the data and tells you nothing about the model.
- **DSC** (discrimination): the skill gained by the model's ranking. A higher DSC is better — it means the model distinguishes well between high and low risk. This is what your Gini coefficient approximates.
- **MCB** (miscalibration): the excess deviance from wrong price levels. Zero means the model is perfectly calibrated; positive means it is leaving deviance on the table through miscalibration.

MCB decomposes further:

```
MCB = GMCB + LMCB
```

- **GMCB** (global MCB): the portion of miscalibration fixed by multiplying all predictions by a single constant — i.e., the portion attributable to a wrong intercept. This is the cheap fix.
- **LMCB** (local MCB): the portion not fixable by any global scale adjustment. This requires a model refit.

The decision rule: if GMCB >> LMCB, recalibrate. If LMCB >= GMCB, refit.

---

## Running it in practice

```bash
uv add insurance-monitoring
```

```python
import numpy as np
from insurance_monitoring.calibration import murphy_decomposition, CalibrationChecker

rng = np.random.default_rng(42)
n = 80_000
exposure = rng.uniform(0.5, 2.0, n)
y_hat = rng.gamma(2, 0.05, n)

# Scenario 1: global scale error only
# Model is 8% cheap but the shape is right
y_global = rng.poisson(exposure * y_hat * 1.08) / exposure

result_global = murphy_decomposition(y_global, y_hat, exposure, distribution='poisson')
print(f"GMCB: {result_global.global_mcb:.6f}")
print(f"LMCB: {result_global.local_mcb:.6f}")
print(f"Verdict: {result_global.verdict}")
# GMCB: 0.000341
# LMCB: 0.000019
# Verdict: RECALIBRATE
```

GMCB at 18x LMCB: nearly all of the miscalibration is a scale problem. Multiply predictions by 1.08 and you are done.

```python
# Scenario 2: structural error
# Young drivers (age < 30) systematically under-priced by 25%
# Mature drivers (age > 50) over-priced by 12% to partly compensate
driver_ages = rng.integers(18, 75, n)
scale = np.where(driver_ages < 30, 1.25,
        np.where(driver_ages > 50, 0.88,
                 1.02))
y_structural = rng.poisson(exposure * y_hat * scale) / exposure

result_structural = murphy_decomposition(y_structural, y_hat, exposure, distribution='poisson')
print(f"GMCB: {result_structural.global_mcb:.6f}")
print(f"LMCB: {result_structural.local_mcb:.6f}")
print(f"Verdict: {result_structural.verdict}")
# GMCB: 0.000089
# LMCB: 0.000218
# Verdict: REFIT
```

LMCB at 2.4x GMCB: multiplying by a constant reduces the errors but leaves the structural pattern intact. You need to refit on recent data.

The two scenarios produce broadly similar A/E ratios at portfolio level. The decomposition separates them.

---

## The full picture: add the Gini drift test

Murphy decomposition answers the RECALIBRATE vs REFIT question for calibration errors. But there is a second failure mode — discriminatory power decay — that GMCB/LMCB does not catch.

If the model's ranking has degraded (the Gini has fallen from 0.42 at deployment to 0.36 eighteen months later), the model's shape may look fine by the MCB test. The claims environment has shifted in a way that makes the historical feature-response relationships less predictive, but the model is still internally consistent. MCB is low. But the model is doing worse work than it was.

The Gini drift z-test (Theorem 1, arXiv:2510.04556) is the right test for this:

```python
from insurance_monitoring.discrimination import gini_coefficient, gini_drift_test

gini_ref = gini_coefficient(act_ref, pred_ref, exposure=exp_ref)
gini_cur = gini_coefficient(act_cur, pred_cur, exposure=exp_cur)

result = gini_drift_test(
    reference_gini=gini_ref,
    current_gini=gini_cur,
    n_reference=len(act_ref),
    n_current=len(act_cur),
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur, current_predicted=pred_cur,
)
print(f"Gini: {gini_ref:.3f} → {gini_cur:.3f}")
print(f"z = {result['z_statistic']:.2f}, p = {result['p_value']:.3f}")
# Gini: 0.421 → 0.363
# z = -2.71, p = 0.007
```

p = 0.007 is a statistically significant degradation. At that point, RECALIBRATE is no longer on the table regardless of what GMCB says. The model needs refitting even if the MCB test looked clean.

---

## Combining both in `MonitoringReport`

Running these separately is fine for investigation. For routine monthly monitoring, `MonitoringReport` runs all of it in one call and applies the decision tree:

```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    exposure=exposure_cur,
    reference_exposure=exposure_ref,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years", "area"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

df = report.to_polars()
print(df)
# metric                   value    band
# ae_ratio                 1.07     amber
# gini_current             0.363    amber
# gini_p_value             0.007    red
# csi_driver_age           0.18     amber
# csi_vehicle_age          0.04     green
# murphy_discrimination    0.038    --
# murphy_miscalibration    0.000307 --
# murphy_gmcb              0.000089 --
# murphy_lmcb              0.000218 --
# recommendation           REFIT    red
```

The decision tree implementation: if the Gini p-value is below 0.10, that triggers REFIT regardless of MCB. If Gini is stable but LMCB >= GMCB, that also triggers REFIT. If Gini is stable and GMCB >> LMCB, RECALIBRATE. If nothing is significantly off, MONITOR_CLOSELY or NO_ACTION depending on A/E.

When the Murphy distribution is set, this is the logic that runs. Without it, the framework falls back to A/E + Gini alone — still more than most teams are running, but less sharp on the RECALIBRATE vs REFIT question.

---

## Where teams typically get this wrong

**Treating all amber A/E as a recalibration**. A/E of 1.06 across the portfolio, trending from 1.01 three months ago, and the team adjusts the intercept. If LMCB is actually higher than GMCB, this is suppressing the signal. The portfolio goes green for another quarter while the structural problem grows.

**Treating all red A/E as requiring a refit**. Model is 12% cheap, team mobilises a full refit. If GMCB >> LMCB, that refit produces a model with the same shape as before, a corrected intercept, and eight weeks elapsed. The intercept adjustment would have had the same effect on actual pricing in ninety minutes.

**Not exposure-weighting**. Standard PSI and standard A/E ratios from credit scoring do not weight by earned exposure. A portfolio with a mix of annual and monthly policies, or a book where telematics customers accumulate differently than standard, produces different segment proportions by policy count versus by exposure. The monitoring library weights by exposure throughout. If you are using unweighted PSI, your green/amber/red thresholds are wrong for UK personal lines.

**Running monitoring on immature accident periods**. The A/E ratio requires that most claims have been reported and developed. For UK motor, the practical minimum is 12 months of run-off on each accident period. The calibration sign-off documentation on `insurance-monitoring` recommends applying chain-ladder development factors before passing actuals to any of the calibration functions for recent accident months.

---

## The governance case

PRA SS1/23 (effective May 2024) requires model validators to document the basis for decisions about model updates. The Murphy decomposition provides exactly the right level of evidence: a quantitative split between GMCB and LMCB with a documented decision rule.

A model risk note that reads "A/E of 1.07 observed, Murphy GMCB of 0.000341 versus LMCB of 0.000019, GMCB/MCB ratio = 0.95, recalibration applied per monitoring framework" is a substantively different document from "A/E of 1.07 observed, intercept adjusted to restore balance." The first can be audited against the framework. The second requires a validator to take the pricing team's word for it.

```python
from insurance_monitoring.calibration import murphy_decomposition

result = murphy_decomposition(y, y_hat, exposure, distribution='poisson')

# Audit-trail output for model risk log
note = (
    f"Murphy decomposition (Poisson, n={len(y):,}): "
    f"GMCB={result.global_mcb:.6f}, "
    f"LMCB={result.local_mcb:.6f}, "
    f"GMCB/MCB={result.global_mcb / (result.global_mcb + result.local_mcb):.2f}. "
    f"Verdict: {result.verdict}."
)
print(note)
# Murphy decomposition (Poisson, n=80,000): GMCB=0.000341, LMCB=0.000019,
# GMCB/MCB=0.95. Verdict: RECALIBRATE.
```

One line in the model risk log. Reproducible from the data. No committee instinct required.

---

## Installation

```bash
uv add insurance-monitoring
```

Python 3.10+. Polars-native throughout. No scikit-learn dependency. The calibration suite requires only `polars >= 0.20`, `numpy >= 1.24`, and `scipy >= 1.10` for the Garwood confidence intervals on A/E. `matplotlib >= 3.7` is optional for the calibration plot functions.

The library is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). The `examples/model_drift_monitoring.py` script runs the full three-scenario benchmark — covariate shift, calibration deterioration, Gini decay — on synthetic motor data and produces a structured monitoring report with Murphy decomposition outputs.

- [Three-Layer Drift Detection: What PSI and A/E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/)
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/)
- [When Credibility Meets CatBoost](/2026/04/14/when-credibility-meets-catboost/)
