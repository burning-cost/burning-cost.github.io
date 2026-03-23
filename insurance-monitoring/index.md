---
layout: page
title: "insurance-monitoring"
description: "Model drift detection for deployed insurance pricing models. PSI/CSI, Gini drift z-test, anytime-valid A/B testing via mSPRT, PITMonitor for calibration drift. Current: v0.8.0."
permalink: /insurance-monitoring/
---

Model drift detection, calibration monitoring, and champion/challenger testing for deployed insurance pricing models. Polars-native throughout. Current version: v0.8.0.

**[View on GitHub](https://github.com/burning-cost/insurance-monitoring)** &middot; **[PyPI](https://pypi.org/project/insurance-monitoring/)**

---

## The problem

Your aggregate A/E ratio looks fine. Your model has been mispricing under-25s for eight months.

The aggregate A/E is blind to who is inside the portfolio. A model that is 15% cheap on young drivers and 15% expensive on mature drivers reads 1.00 and raises no alarm. This library monitors features, not just the headline number.

The three distinct failure modes require different tools:
- **Covariate shift** — PSI per feature detects when the book composition has changed
- **Calibration drift** — A/E with Poisson CI and Murphy decomposition identifies whether to recalibrate or refit
- **Discrimination decay** — Gini drift z-test detects whether the model's ranking has degraded (the difference between a cheap recalibration and a full refit)

---

## Installation

```bash
pip install insurance-monitoring
```

---

## Quick start

```python
import polars as pl
import numpy as np
from insurance_monitoring import MonitoringReport

rng = np.random.default_rng(42)
n_ref, n_cur = 10_000, 4_000

report = MonitoringReport(
    reference_actual=act_ref,
    reference_predicted=pred_ref,
    current_actual=act_cur,
    current_predicted=pred_cur,
    feature_df_reference=feat_ref,
    feature_df_current=feat_cur,
    features=["driver_age", "vehicle_age", "ncd_years"],
    murphy_distribution="poisson",
)

print(report.recommendation)
# 'RECALIBRATE' | 'REFIT' | 'NO_ACTION' | 'INVESTIGATE' | 'MONITOR_CLOSELY'

df = report.to_polars()
# metric | value | band
```

---

## Key API reference

### `MonitoringReport`

The primary entry point. Runs the full monitoring suite in one call.

```python
MonitoringReport(
    reference_actual,       # numpy array, reference period outcomes
    reference_predicted,    # numpy array, reference period predictions
    current_actual,         # numpy array, current period outcomes
    current_predicted,      # numpy array, current period predictions
    feature_df_reference,   # Polars DataFrame, reference features
    feature_df_current,     # Polars DataFrame, current features
    features,               # list of feature column names
    murphy_distribution,    # "poisson" | "normal" | None
    gini_bootstrap=False,   # bool, use bootstrap CI for Gini drift test
)
```

- `.recommendation` — str: RECALIBRATE | REFIT | NO_ACTION | INVESTIGATE | MONITOR_CLOSELY
- `.to_polars()` — Polars DataFrame with metric/value/band rows
- `.results_` — dict of individual test results

### Drift detection

```python
from insurance_monitoring.drift import psi, csi, ks_test, wasserstein_distance

# Population Stability Index
psi_val = psi(reference, current, bins=10)

# Characteristic Stability Index (per feature)
csi_results = csi(feature_df_ref, feature_df_cur, features=["age", "vehicle_group"])
```

### `InterpretableDriftDetector` (v0.7.0)

Feature-interaction-aware drift attribution with exposure weighting and BH FDR control.

```python
from insurance_monitoring import InterpretableDriftDetector

detector = InterpretableDriftDetector(
    model=fitted_catboost,
    loss="poisson_deviance",
    error_control="fdr",     # fdr | bonferroni
)
detector.fit_reference(X_ref, y_ref, weights=exposure_ref)
result = detector.test(X_cur, y_cur, weights=exposure_cur)

result.drifting_features   # list of features with significant drift
result.p_values            # dict of p-values per feature
result.plot()              # bar chart of attributed drift contributions
```

### Calibration monitoring

```python
from insurance_monitoring.calibration import (
    ae_ratio,               # A/E with Poisson confidence interval
    hosmer_lemeshow,        # Hosmer-Lemeshow test
    check_balance,          # balance property test
    murphy_decomposition,   # UNC/DSC/MCB decomposition
    CalibrationChecker,     # combined calibration suite
)

ae = ae_ratio(actual, predicted, exposure)
# ae.ratio, ae.ci_lower, ae.ci_upper, ae.significant
```

### `PITMonitor` (v0.7.0)

Anytime-valid calibration change detection via PIT e-process martingale. Holds at alpha = 0.05 FPR regardless of how often you check.

```python
from insurance_monitoring import PITMonitor

monitor = PITMonitor(alpha=0.05)
monitor.fit(y_train, pred_train)           # fits PIT distribution
alarm = monitor.update(y_new, pred_new)    # anytime-valid update
# alarm.triggered: bool
# alarm.e_value: float (alarm when > 1/alpha)
```

### `SequentialTest` (v0.5.0+, tested v0.8.0)

Anytime-valid A/B testing for champion/challenger experiments using mixture SPRT.

```python
from insurance_monitoring import SequentialTest

test = SequentialTest(metric="frequency", alpha=0.05)
test.add_observations(champion_claims, challenger_claims,
                      champion_exposure, challenger_exposure)
result = test.test()
# result.reject: bool — valid at any stopping time
# result.e_value: float
# result.confidence_sequence: (lower, upper)
```

Benchmark: peeking t-test reaches ~25% FPR with monthly checks; mSPRT holds at 5% at all stopping times.

### Discrimination monitoring

```python
from insurance_monitoring.discrimination import (
    gini_coefficient,               # Gini with optional bootstrap CI
    gini_drift_test,                # two-sample drift test
    gini_drift_test_onesample,      # one-sample: monitor vs stored training Gini
    GiniDriftBootstrapTest,         # class-based with .plot() and .summary()
)
```

### `GiniDriftBootstrapTest` (v0.6.0)

Produces a governance-ready bootstrap histogram for pricing committee packs.

```python
from insurance_monitoring import GiniDriftBootstrapTest

test = GiniDriftBootstrapTest(
    reference_actual=y_train,
    reference_predicted=pred_train,
    n_bootstrap=200,
)
result = test.test(current_actual=y_cur, current_predicted=pred_cur)
result.plot()     # bootstrap histogram with CI shading
result.summary()  # governance paragraph
```

---

## Benchmark

| Test | Result |
|------|--------|
| Peeking t-test FPR (monthly checks) | ~25% |
| mSPRT FPR at all stopping times | ~5% |
| PITMonitor FPR (calibrated model) | ~3% |
| Repeated Hosmer-Lemeshow FPR | ~46% |
| InterpretableDriftDetector false positives | 0 in benchmarks |

---

## Related blog posts

- [Insurance Model Monitoring Beyond Generic Drift](https://burning-cost.github.io/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)
- [Gini, A/E, and Double-Lift: The Three Tests Every Monitoring Dashboard Needs](https://burning-cost.github.io/2026/03/22/insurance-model-monitoring-gini-ae-double-lift/)
- [insurance-monitoring vs Evidently](https://burning-cost.github.io/2026/03/22/insurance-model-monitoring-evidently-alternative/)
- [insurance-monitoring vs NannyML](https://burning-cost.github.io/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/)
- [insurance-monitoring vs alibi-detect](https://burning-cost.github.io/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/)
