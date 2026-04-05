---
layout: post
title: "Insurance Model Monitoring in Python: A Practitioner's Guide"
date: 2026-04-04
author: Burning Cost
categories: [monitoring, python, tutorials]
tags: [insurance-model-monitoring, model-monitoring, PSI, CSI, gini-drift, ae-ratio, murphy-decomposition, recalibrate, refit, insurance-monitoring, python, uk-motor, uk-insurance, model-risk, drift-detection, pricing-actuary]
description: "How to set up insurance model monitoring in Python from scratch: PSI, Gini drift, and A/E with the insurance-monitoring library. Know when to redeploy, recalibrate, or refit."
---

Most UK pricing teams monitor their deployed models the same way: an analyst runs the Gini and A/E ratio at quarter-end, compares them to last quarter, and emails the result to the pricing manager. If nothing looks dramatically wrong, the model stays. If the A/E ratio hits 1.10, a conversation starts.

This is not model monitoring. This is model hoping.

The problem with ad hoc quarterly checks is not that actuaries are lazy — it is that a single A/E ratio at portfolio level is a profoundly weak signal. A model that overprices young drivers by 12% and underprices mature drivers by 10% reads 1.01 at portfolio level. The adverse selection consequence — cheap competitors sweeping your mature drivers while you retain the overpriced young ones — arrives 18 months later in the loss ratio.

This post is about setting up insurance model monitoring in Python properly: what metrics to track, what thresholds to use, what the alerts mean, and when each type of action is actually warranted. We use the [`insurance-monitoring`](/insurance-monitoring/) library throughout.

```bash
uv add insurance-monitoring
```

---

## The three questions monitoring must answer

Every monitoring check needs to answer three distinct questions. They are not substitutes for each other.

**1. Has the book shifted?** Feature distribution drift — the composition of your in-force portfolio has changed since the model was trained. Driver age distribution has shifted because of claim-free renewal accumulation. Vehicle mix has moved with new model releases. Postcode mix has changed because of house price moves affecting where your policyholders live. If the book has shifted, the model is being applied to a population it was not built on.

**2. Has the model's ranking degraded?** Discrimination drift — the model still predicts the right average frequency, but it has lost the ability to rank risks. It charged the right aggregate but the wrong individuals. This is invisible to an A/E ratio check and lethal to adverse selection management.

**3. Is the model miscalibrated, and if so, how?** Calibration drift separates into two very different problems. Global miscalibration — the model is off by a flat scale factor across the entire book — is fixable in an afternoon by adjusting the intercept. Local miscalibration — specific segments are badly wrong, and they partly cancel at portfolio level — requires a refit. These look identical in the A/E ratio headline and require completely different responses.

The decision logic flows from these three questions: feature drift alone is a flag; Gini degradation or local miscalibration triggers a refit; only global miscalibration alone triggers a recalibration.

---

## Setting up the monitor

The first step is fitting the monitor on the reference dataset — the data used to train and validate the model. This establishes the baseline Gini distribution that later quarters are compared against.

```python
import numpy as np
from insurance_monitoring import ModelMonitor

rng = np.random.default_rng(42)

# Reference period: 40,000 UK motor policies, 2022-2023
n_ref = 40_000
exposure_ref = rng.uniform(0.5, 1.0, n_ref)
driver_age_ref = rng.normal(38, 10, n_ref).clip(17, 80)
vehicle_age_ref = rng.uniform(1, 10, n_ref)
ncd_years_ref = rng.integers(0, 10, n_ref).astype(float)

# True frequency: younger drivers, older vehicles cost more
lam_ref = np.exp(
    -2.8
    + 0.03 * np.maximum(30 - driver_age_ref, 0)
    + 0.04 * vehicle_age_ref
    - 0.05 * ncd_years_ref
)
counts_ref = rng.poisson(exposure_ref * lam_ref)
y_ref = counts_ref / exposure_ref      # observed claim rates
y_hat_ref = lam_ref                    # model predictions (well-calibrated at training)

monitor = ModelMonitor(
    distribution="poisson",
    n_bootstrap=400,
    alpha_gini=0.32,    # one-sigma early warning — raises review, not automatic action
    alpha_global=0.32,
    alpha_local=0.32,
    random_state=0,
)
monitor.fit(y_ref, y_hat_ref, exposure_ref)
```

The `alpha=0.32` default implements the "one-sigma rule" from Brauer, Menzel & Wüthrich (arXiv:2510.04556, October 2025): flag at one standard deviation rather than the conventional 1.96. A false alarm triggers a model review, not an automatic refit. The cost of a false alarm is a conversation; the cost of a missed true alarm is 18 months of adverse selection. Calibrate `alpha` to your team's actual cost tolerance, but 0.32 is the right starting point for quarterly pricing model monitoring.

---

## Quarterly check: the three scenarios

We simulate three realistic monitoring scenarios a UK pricing team would encounter in 2025 after training on 2022-2023 data.

### Scenario 1: No drift — redeploy

The book composition and frequency experience are stable. The monitor confirms the model remains fit for purpose.

```python
n_new = 8_000
exposure_new = rng.uniform(0.5, 1.0, n_new)
driver_age_new = rng.normal(38, 10, n_new).clip(17, 80)
vehicle_age_new = rng.uniform(1, 10, n_new)
ncd_years_new = rng.integers(0, 10, n_new).astype(float)

lam_new = np.exp(
    -2.8
    + 0.03 * np.maximum(30 - driver_age_new, 0)
    + 0.04 * vehicle_age_new
    - 0.05 * ncd_years_new
)
y_actual_new = rng.poisson(exposure_new * lam_new) / exposure_new
y_hat_new = lam_new    # model predictions still correct

result = monitor.test(y_actual_new, y_hat_new, exposure_new)
print(result.decision)          # => REDEPLOY
print(result.gini_significant)  # => False
print(result.gmcb_significant)  # => False
print(result.lmcb_significant)  # => False
```

### Scenario 2: Claims inflation — recalibrate

Two years of above-trend claims inflation has lifted the overall frequency level. The model's rank ordering is intact — it still correctly identifies which risks are cheap and expensive — but its absolute level is now uniformly low. A/E of approximately 1.09. Recalibration (multiply all predictions by the balance factor) fixes this without any structural change.

```python
# Claims inflation: true frequency is 9% higher than the model's training period
lam_inflated = lam_new * 1.09
y_actual_inflated = rng.poisson(exposure_new * lam_inflated) / exposure_new
# Model predictions unchanged — still calibrated to 2022-2023 frequency level
y_hat_stale = lam_new

result_inflation = monitor.test(y_actual_inflated, y_hat_stale, exposure_new)
print(result_inflation.decision)            # => RECALIBRATE
print(result_inflation.gini_significant)    # => False (ranking still good)
print(result_inflation.gmcb_significant)    # => True  (flat level shift detected)
print(result_inflation.lmcb_significant)    # => False (no structural error)
print(f"Multiply all predictions by: {result_inflation.balance_factor:.3f}")
```

Recalibration here is a single number: multiply all predictions by `balance_factor`. This is an afternoon's work in any deployment: update the intercept in a GLM, or multiply the GBM score function output by the factor. No model rebuild required.

### Scenario 3: Book mix shift — refit

Two years of claim-free renewals have aged the NCD distribution — the proportion of policyholders at maximum NCD has increased substantially. The model was trained on a younger NCD mix and now systematically overprices high-NCD risks while underpricing low-NCD ones. Portfolio A/E looks roughly flat, but the Gini test and local calibration test catch the structural error.

```python
# Book mix shift: portfolio has aged (more experienced drivers, higher NCD)
driver_age_shifted = rng.normal(44, 10, n_new).clip(17, 80)   # +6 years
ncd_years_shifted = rng.integers(4, 10, n_new).astype(float)  # shifted to higher NCD

lam_shifted = np.exp(
    -2.8
    + 0.03 * np.maximum(30 - driver_age_shifted, 0)
    + 0.04 * vehicle_age_new
    - 0.05 * ncd_years_shifted
)
y_actual_shifted = rng.poisson(exposure_new * lam_shifted) / exposure_new
# Stale model: predicts the training-period mean, ignoring the population shift
y_hat_mixed = np.full(n_new, lam_ref.mean())

result_shift = monitor.test(y_actual_shifted, y_hat_mixed, exposure_new)
print(result_shift.decision)           # => REFIT
print(result_shift.gini_significant)   # => True (rank ordering degraded)
print(result_shift.lmcb_significant)   # => True (NCD segment errors not cancelling)
print(result_shift.summary())          # governance-ready paragraph for model risk committee
```

The Gini drop and the local miscalibration together make the case unambiguous. A recalibration here would multiply a wrong-shaped model by a constant — the errors get smaller in aggregate but stay in the wrong places. The model needs rebuilding on recent data.

---

## Feature drift: CSI before the decision

The `ModelMonitor` decision tells you what to do. The `MonitoringReport` tells you why. Running CSI across rating factors before interpreting the monitoring result is good practice: it surfaces the root cause early and directs the refit effort.

```python
import polars as pl
from insurance_monitoring.drift import csi

ref_df = pl.DataFrame({
    "driver_age":   driver_age_ref,
    "vehicle_age":  vehicle_age_ref,
    "ncd_years":    ncd_years_ref,
})

cur_df = pl.DataFrame({
    "driver_age":   driver_age_shifted,
    "vehicle_age":  vehicle_age_new,
    "ncd_years":    ncd_years_shifted,
})

for col in ["driver_age", "vehicle_age", "ncd_years"]:
    csi_val = csi(
        ref_df[col].to_numpy(),
        cur_df[col].to_numpy(),
        weights_ref=exposure_ref,
        weights_new=exposure_new,
    )
    band = "GREEN" if csi_val < 0.1 else ("AMBER" if csi_val < 0.2 else "RED")
    print(f"{col:20s}  CSI={csi_val:.3f}  [{band}]")

# driver_age            CSI=0.187  [AMBER]
# vehicle_age           CSI=0.031  [GREEN]
# ncd_years             CSI=0.241  [RED]
```

`ncd_years` is the culprit. This directs the refit effort: focus on getting recent data that covers the current NCD distribution. It is not necessary to rebuild every feature — but NCD relativities will need re-estimating on data that reflects today's book.

The exposure-weighting in `csi()` is not cosmetic. A short-period policy contributes the same count as an annual policy to unweighted PSI. For a UK motor portfolio where younger drivers disproportionately take monthly-pay policies, unweighted CSI understates the true distributional shift in your earned exposure. Always weight by earned car-years.

---

## What the thresholds actually mean

| PSI / CSI value | Interpretation | Action |
|:---|:---|:---|
| < 0.10 | No meaningful shift | Continue monitoring |
| 0.10 – 0.20 | Moderate shift | Flag for investigation; check whether performance metrics are affected |
| > 0.20 | Significant shift | Refit or recalibrate depending on Gini and calibration tests |

The 0.10 / 0.20 thresholds are the industry standard from the PSI literature (Siddiqi, 2006) and remain appropriate for insurance portfolios when applied to exposure-weighted distributions. They are not PRA or FCA-mandated thresholds — they are empirically grounded rules of thumb. Set your governance thresholds by reference to them, document the rationale, and revisit after your first annual validation cycle.

For discrimination drift, there is no equivalent industry standard threshold. The `alpha=0.32` default in `ModelMonitor` is calibrated to the cost of false alarms in a quarterly review process. For a formal annual governance attestation (which PRA SS1/23 model risk principles require firms to evidence), use a conventional significance level (`alpha=0.05`) and document the test methodology.

---

## Connecting to governance

A monitoring output is not self-evidencing. The result of `monitor.test()` needs to go somewhere.

```python
import json

result_dict = result_shift.to_dict()
# Keys: decision, decision_reason, gini_z, gini_p, gini_significant,
#       gmcb_score, gmcb_p, gmcb_significant, lmcb_score, lmcb_p,
#       lmcb_significant, balance_factor, n_new, distribution, ...

with open("motor_freq_q1_2026_monitoring.json", "w") as f:
    json.dump(result_dict, f, indent=2)
```

Feed this JSON into your model risk register. Under PRA SS1/23 principles, insurers need to evidence that pricing models are being monitored with defined quantitative thresholds and documented pass/fail outcomes — not narrative assertions. A quarterly JSON file with the three test results and the decision is the minimum viable governance artefact. Better still, pair it with [`insurance-governance`](/insurance-governance/) to ingest the monitoring output directly into the model risk committee pack — the [actuarial model validation guide](/2026/04/04/actuarial-model-validation-python/) documents how validation and monitoring connect in the same governance chain.

[`insurance-monitoring`](/insurance-monitoring/) also integrates with MLflow if your team uses that for experiment tracking:

```bash
uv add insurance-monitoring[mlflow]
```

---

## The minimal monitoring setup

If you start nowhere, start here. This is the smallest viable quarterly monitoring check for a UK motor frequency model.

```python
import pickle
import numpy as np
from insurance_monitoring import ModelMonitor

# Step 1: Fit on training period data (run once when model is deployed)
monitor = ModelMonitor(distribution="poisson", n_bootstrap=400, random_state=0)
monitor.fit(y_train_actual, y_train_predicted, exposure_train)

# Serialise so you can reload without the original training data next quarter
with open("motor_freq_monitor.pkl", "wb") as f:
    pickle.dump(monitor, f)

# Step 2: Each quarter, load and run
with open("motor_freq_monitor.pkl", "rb") as f:
    monitor = pickle.load(f)

result = monitor.test(y_q1_actual, y_q1_predicted, exposure_q1)
print(result.decision)    # REDEPLOY / RECALIBRATE / REFIT
print(result.summary())   # human-readable governance paragraph
```

That is it. Six lines to go from trained model to a principled three-way recommendation. Set a quarterly calendar reminder, commit the result JSON to your model risk register, and you have moved from model hoping to insurance model monitoring in Python that would withstand a PRA supervisory visit.

For the full worked example including feature CSI, champion/challenger A/B testing with anytime-valid sequential testing, and integration with `insurance-governance` for model risk committee packs, see the [insurance-monitoring examples on GitHub](https://github.com/burning-cost/insurance-monitoring/tree/main/examples).

---

*Related:*
- [Actuarial Model Validation in Python: Automated Reports for UK Insurance Pricing](/2026/04/04/actuarial-model-validation-python/) — the point-in-time validation that produces the thresholds monitoring should fire against
- [glum Insurance Pricing in Python: Fitting, Intervals, Monitoring, Fairness, Governance](/2026/04/04/glum-insurance-pricing-python-workflow/) — the complete workflow showing how monitoring sits alongside model fitting and governance
- [Does PSI Model Monitoring Actually Catch Pricing Model Drift?](/2026/03/28/does-psi-model-monitoring-actually-catch-pricing-model-drift/) — benchmark results showing when PSI is a weak signal and A/E CI is the more sensitive test
