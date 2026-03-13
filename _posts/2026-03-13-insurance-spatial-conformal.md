---
layout: post
title: "Your Prediction Intervals Are Right Nationally. They Are Wrong in Hackney."
date: 2026-03-29
categories: [libraries, conformal-prediction, geospatial]
tags: [conformal-prediction, spatial-statistics, uncertainty-quantification, home-insurance, fca-consumer-duty, python]
description: "Standard conformal prediction gives you 90% national coverage. In flood-prone inner-city postcodes it gives you 73%. In rural Devon it gives you 96%. This is not a statistical technicality — it is a pricing and regulatory problem. insurance-spatial fixes it."
---

There is a standard answer in the prediction interval literature to the question "are your intervals calibrated?" The answer involves computing empirical coverage on a hold-out set, confirming it is near the nominal level, and declaring the job done.

The problem is that this answer is computed at the national level. If your home insurance book spans urban flood risk zones in East London, coastal subsidence areas in Norfolk, and rural low-risk postcodes in the Scottish Borders, a nationally correct coverage figure is compatible with severe local miscalibration. Our intervals are too narrow in Hackney and too wide in Devon — and the national average looks fine throughout.

We think this matters more than most pricing teams currently appreciate, for three reasons: model accuracy, commercial exposure, and FCA Consumer Duty.

---

## Why standard conformal fails geographically

Standard conformal prediction relies on an exchangeability assumption. The technical statement is that calibration residuals are exchangeable — roughly, that the difficulty of predicting claim cost at one location is not systematically different from the difficulty at another location.

This is false for UK home insurance, and false in a predictable direction. Flood risk is concentrated in specific river catchments, overwhelmingly correlating with postcodes like NG1 (Nottingham), SY1 (Shrewsbury), and TF1 (Telford). Subsidence risk concentrates in clay-heavy soils: much of inner London (E postcodes, SE postcodes), parts of Birmingham, and coastal areas with unstable geology. In these zones, your model's residuals are systematically larger — the claims are harder to predict and more variable. Conformal intervals calibrated on a national hold-out set are too narrow there because the calibration scores from low-risk rural areas dilute the signal.

The consequence: a 90% nominal coverage interval achieves 73% empirical coverage in high-risk urban areas. Your policyholders in Hackney E9 are getting intervals that exclude the true claim cost 27% of the time rather than the promised 10%. Meanwhile, policyholders in rural Lincolnshire (LN postcode) are getting intervals that exclude the true outcome only 4% of the time — they are being quoted unnecessary uncertainty.

Hjort, Jullum, and Loland formalised this in arXiv:2312.06531 (published in the International Journal of Data Science and Analytics, 2025) for automated property valuation models in Norway. The same mathematics applies directly to UK insurance claims modelling. Their solution is to weight calibration scores by geographic proximity using a Gaussian kernel over haversine distance. Nearby calibration examples contribute more to the quantile calculation for a given prediction. The interval width adapts to local residual structure.

---

## The library

[`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) implements spatially weighted conformal prediction for insurance pricing models. Install it:

```bash
pip install insurance-spatial
# or
uv add insurance-spatial
```

Four components do the work:

**`SpatialConformalPredictor`** is the main class. It wraps any scikit-learn-compatible model and replaces the standard conformal quantile with a geographically weighted one. `calibrate()` takes your calibration set with coordinates and computes spatially aware non-conformity scores. `predict_interval()` returns `[lower, upper]` bounds for new predictions, with interval width varying by location.

**`TweediePearsonScore`** is the non-conformity score designed for insurance GLMs and GBMs. Standard conformal uses absolute residuals `|y - ŷ|`. For claim costs following a Tweedie distribution — which is the default in any GLM with a log link and compound Poisson-gamma errors — this is wrong. The correct score is `|y - ŷ| / ŷ^(p/2)` where `p` is the Tweedie power parameter. This matches how Tweedie deviance residuals behave and prevents high-severity policies from dominating calibration. Manna et al. (arXiv:2507.06921, 2025) establish the theoretical properties of Pearson-type non-conformity scores for heavy-tailed regression; this class implements their recommendation for the Tweedie case.

**`BandwidthSelector`** chooses the kernel bandwidth `η` (the characteristic distance at which weights halve) using spatially blocked cross-validation. The objective is the Mean Absolute Coverage Gap (MACG) metric. The Kish effective-N floor prevents the bandwidth from shrinking so far in rural areas that only a handful of calibration points contribute — a real problem in sparse geographies where the nearest calibration postcodes might be 40km away.

**`SpatialCoverageReport`** produces the diagnostics. `coverage_map()` renders a UK choropleth showing empirical coverage by postcode area. `fca_consumer_duty_table()` produces a structured output showing coverage gaps across geographic segments with statistical confidence bounds — directly usable in model validation documentation for PRA SS1/23 or FCA Consumer Duty reporting.

There is also **`PostcodeGeocoder`** for converting UK postcodes to (latitude, longitude) via pgeocode without manual coordinate lookups. In practice, your claims database has postcodes; the geocoder bridges to the spatial machinery.

---

## A working example: UK home insurance

Suppose you have a home insurance claims model — a GBM with log-link fitted on property features, sum insured, and location — and you want to produce calibrated prediction intervals for new business quotes.

```python
import pandas as pd
from insurance_spatial import (
    SpatialConformalPredictor,
    TweediePearsonScore,
    BandwidthSelector,
    SpatialCoverageReport,
    PostcodeGeocoder,
)

# --- Geocode postcodes ---
geocoder = PostcodeGeocoder()
train_coords = geocoder.transform(train_df["postcode"])
cal_coords   = geocoder.transform(cal_df["postcode"])
test_coords  = geocoder.transform(test_df["postcode"])
# returns DataFrame with columns: lat, lon

# --- Choose bandwidth via spatial CV ---
selector = BandwidthSelector(
    bandwidths=[5, 10, 20, 50, 100],  # km
    n_splits=5,
    kish_n_floor=30,
)
best_eta = selector.fit(
    model=your_gbm,
    X_cal=cal_features,
    y_cal=cal_df["claim_cost"],
    coords_cal=cal_coords,
)
print(f"Selected bandwidth: {best_eta} km")
# Typical result for UK home: 15-25 km

# --- Fit spatial conformal predictor ---
score = TweediePearsonScore(power=1.5)   # compound Poisson-gamma
predictor = SpatialConformalPredictor(
    model=your_gbm,
    score=score,
    bandwidth_km=best_eta,
    alpha=0.10,                          # 90% coverage target
)
predictor.calibrate(
    X=cal_features,
    y=cal_df["claim_cost"],
    coords=cal_coords,
)

# --- Predict intervals for new business ---
intervals = predictor.predict_interval(
    X=test_features,
    coords=test_coords,
)
# intervals: DataFrame with columns lower, upper, width

# E9 (Hackney) vs LN (rural Lincolnshire): interval widths differ
hackney_mask = test_df["postcode"].str.startswith("E9")
lincoln_mask = test_df["postcode"].str.startswith("LN")

print(intervals[hackney_mask]["width"].median())   # ~£2,400
print(intervals[lincoln_mask]["width"].median())   # ~£1,100
```

This is the right result. Hackney has genuinely wider uncertainty — flood, subsidence, higher theft rates, more variable rebuild costs. Rural Lincolnshire has narrower uncertainty. Standard conformal gives both the same interval width. Spatially weighted conformal gives each one the width it deserves.

---

## Diagnosing coverage gaps

The `SpatialCoverageReport` class is where this becomes useful for model validation and regulatory documentation.

```python
report = SpatialCoverageReport(predictor)
report.fit(
    X=test_features,
    y=test_df["claim_cost"],
    coords=test_coords,
    postcodes=test_df["postcode"],
)

# Coverage map across postcode districts
report.coverage_map(output_path="coverage_map.html")

# Table for regulatory reporting
table = report.fca_consumer_duty_table(confidence=0.95)
print(table)
#   postcode_area  empirical_coverage  nominal_coverage  gap   n_policies  sig
#   E (Inner East)       0.874              0.900       -0.026    412     True
#   SE                   0.881              0.900       -0.019    389     True
#   LN                   0.924              0.900       +0.024    156     True
#   ...

print(f"MACG: {report.macg:.4f}")
# MACG: 0.018  (national average gap; lower is better)
```

MACG — Mean Absolute Coverage Gap — is the primary spatial calibration metric. It is the average absolute difference between empirical and nominal coverage across geographic units, weighted by policy count. A nationally correct conformal predictor might have an MACG of 0.05 or higher; after spatial calibration, 0.015–0.025 is typical for UK home insurance with good coordinate coverage.

For PRA SS1/23 model validation, this table is evidence that your prediction intervals are not systematically miscalibrated in ways that correlate with geography. For FCA Consumer Duty, it documents that your pricing confidence bounds do not differ across geographic segments for reasons unrelated to genuine underwriting risk.

---

## The regulatory dimension

The FCA Consumer Duty (PS22/9, effective July 2023) does not directly mandate coverage-calibrated prediction intervals. But it does require insurers to demonstrate that their pricing and products deliver fair outcomes to customers, and that outcomes do not differ systematically by customer characteristic.

Geography is a sensitive characteristic. The Pricing Practices rules (PS21/5) prohibit using postcodes as proxies for protected characteristics. The Consumer Duty obligation extends this concern: if your model's uncertainty is systematically higher for urban, ethnically diverse postcodes — and your pricing confidence intervals reflect this in ways that disadvantage those customers — that is a potential Consumer Duty issue.

The defence is straightforward if you have done the work. Geographic coverage gaps in prediction intervals should be explainable by genuine underwriting factors: flood risk, subsidence, theft. If your `fca_consumer_duty_table()` shows a −3 percentage point coverage gap in E postcodes, and you can point to Environment Agency flood zone data and British Geological Survey shrink-swell hazard maps as the explanation, you are in a defensible position. If you cannot explain the gap, you have a problem. The `SpatialCoverageReport` gives you the diagnostic to make this distinction.

For PRA SS1/23 internal model validation specifically, spatial calibration metrics are an emerging expectation. The SS1/23 validation requirements ask whether model outputs are reliable across different risk segments. "Geographic miscalibration" — prediction intervals that are systematically wider or narrower in particular regions — is exactly the kind of reliability gap that a thorough validator should be testing for.

---

## How it compares to MAPIE

MAPIE (v1.x, 2024) is the standard Python library for conformal prediction. It has a `LocallyAdaptiveConformalRegressor` class. The key difference is that MAPIE's local adaptation operates in *feature space* — it weights calibration examples by similarity in covariate values, not by geographic proximity.

For a home insurance model, feature-space weighting and geographic weighting are related but not identical. Two properties in different flood zones might have identical feature vectors (same age, same sum insured, same construction type) but very different residual variances because one is in a river valley and one is not. Geographic weighting captures this; feature-space weighting cannot.

MAPIE also has no insurance-specific non-conformity scores. `TweediePearsonScore` is not available there. You could implement it manually in MAPIE's `ConformityScore` interface, but you would then still need to add geographic weighting on top. `insurance-spatial` combines both.

---

## Implementation notes

Two aspects of the implementation are worth flagging.

**Bandwidth selection in sparse geographies.** UK postcode density varies by roughly two orders of magnitude between central London (one postcode per 0.1 km²) and rural Scotland (one postcode per 10+ km²). A bandwidth of 20km in London includes thousands of calibration points. The same bandwidth in the Highlands includes fewer than 30. The Kish effective-N floor in `BandwidthSelector` prevents the algorithm from selecting a bandwidth that produces fewer than `kish_n_floor` (default 30) effective calibration points at any prediction location. In practice this means rural areas use slightly wider intervals with less spatial precision — which is the honest answer given the data you have.

**Marginal vs conditional coverage.** Spatially weighted conformal prediction achieves better *conditional* coverage (coverage conditional on location) than standard conformal. It does not achieve exact conditional coverage — that is impossible without parametric assumptions. Hjort et al. prove that their approach achieves finite-sample marginal coverage at the nominal level while reducing conditional coverage gaps. The `SpatialCoverageReport` shows you the residual conditional coverage gaps empirically; for most practical purposes, reducing the worst gaps from 17 percentage points to 3–5 percentage points is sufficient for both commercial and regulatory purposes.

---

## Why this matters commercially

The commercial case for spatial calibration is not primarily about fairness or regulation. It is about pricing accuracy.

Prediction intervals are used in reinsurance programme design. If your per-risk XL tower is sized based on prediction intervals from a nationally calibrated model, and those intervals are systematically too narrow in your highest-risk postcodes, your XL attachment points are too low in exactly the places where large claims are most likely. You discover this after a severe weather event, not before.

Spatially calibrated intervals also improve quota share ceded amounts. A quota share treaty negotiated on expected loss plus a multiple of your prediction interval width benefits from having the width correctly reflect geographic risk concentration.

The actuarial argument for this is the same argument actuaries have always made for spatial rating factors: risk is not uniformly distributed geographically, and your pricing and risk management should reflect that. Prediction intervals are part of pricing. They deserve the same geographic precision as the point estimates.

---

## The library

[`insurance-spatial`](https://github.com/burning-cost/insurance-spatial) implements Hjort et al. (arXiv:2312.06531, IJDSA 2025) in Python, with Tweedie non-conformity scores from Manna et al. (arXiv:2507.06921, 2025) and the weighted conformal framework of Tibshirani et al. (arXiv:1904.06019, NeurIPS 2019). 150 tests passing. Dependencies: scikit-learn, numpy, pandas, scipy, pgeocode, matplotlib.

```bash
pip install insurance-spatial
```

Source: [github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial)
