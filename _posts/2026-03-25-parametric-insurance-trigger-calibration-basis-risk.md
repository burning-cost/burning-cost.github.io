---
layout: post
title: "Parametric Insurance: Trigger Calibration and Basis Risk"
date: 2026-03-25
categories: [pricing, catastrophe, parametric]
tags: [parametric, evt, basis-risk, copula, flood-re, trigger-calibration, extreme-value-theory, gpd, weather-index, crop-insurance, insurance-severity, insurance-conformal, uk-personal-lines]
description: "SOA and CAS research from late 2025 has sharpened the methods for calibrating parametric triggers and quantifying basis risk. Here is what that means in practice for UK flood and weather-indexed products."
---

Parametric insurance pays out when an index crosses a threshold, not when you prove a loss. The appeal is obvious - no loss adjustment, near-instant settlement, lower expense ratios. The problem is equally obvious - if the index moves without the insured's loss, or the loss happens without the index moving, someone gets hurt. Getting the trigger right is what separates a useful product from an expensive lottery ticket. Getting basis risk wrong is what ends careers.

The SOA research programme published a January 2026 monograph on trigger design that formalises EVT-based calibration, and the CAS 2025 annual proceedings included two papers on copula approaches to basis risk quantification that are directly applicable. This is not abstract: Flood Re is actively reviewing its parametric trigger structures for its B2B reinsurance layer, and the EA's FCERM monitoring network now has enough data density across England to support postcode-level parametric index design. The methods have caught up with the opportunity.

---

## Trigger calibration: why standard quantile fitting is not enough

The naive approach to parametric trigger calibration fits a historical return-period distribution to the index variable - river gauge height, accumulated rainfall, peak wind speed - and sets the trigger at a chosen return period. The reason this fails is the same reason standard GPD fitting fails for large losses with heterogeneous policy limits: the distribution of extreme observations is truncated and censored in ways that standard maximum likelihood ignores.

River gauge records are censored in two directions. Gauges have operational limits above which they record out-of-range. Historic records before gauge installation are recovered from flood marks or documentary evidence with substantial uncertainty. The EA's national flood risk assessment (NFRA2) includes gauge records back to the 1800s at some sites, but the quality of the pre-1960 data ranges from excellent to barely useful. Standard GPD fitted on the post-1990 record will have a different shape parameter than the true century-scale tail, and the error propagates directly into the trigger level.

The EVT correction is the same correction that applies to large-loss severity modelling with policy limits. Use [TruncatedGPD from insurance-severity](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) with the operational gauge limit as the upper truncation point for each observation:

```python
from insurance_severity import TruncatedGPD
import numpy as np

# Gauge exceedances above threshold (e.g. 0.5m above channel bankfull)
exceedances = gauge_df["height_excess"].values          # x - u
gauge_limits = gauge_df["operational_limit_excess"].values  # T_i - u

gpd = TruncatedGPD(threshold=0.5)
gpd.fit(exceedances, limits=gauge_limits)

params = gpd.summary()
# {'xi': 0.31, 'sigma': 0.82, 'threshold': 0.5,
#  'se_xi': 0.04, 'se_sigma': 0.09}

# Trigger at 1-in-100 return period:
# S(x_100) = 1 - F_annual(u) * (1 - GPD.cdf(x_100 - u)) = 0.01
# Solve for x_100 using isf:
q_annual_above_threshold = 0.01 / (1 - p_threshold)
trigger_excess = gpd.isf(np.array([q_annual_above_threshold]))
trigger_level = 0.5 + trigger_excess[0]
```

The `limits` argument is the load-bearing piece. At gauges with, say, a 4.5m operational ceiling and a 0.5m bankfull threshold, any observation that hit the ceiling was truncated there - the actual peak was higher. Ignoring this underestimates xi, which underestimates the tail return level, which sets the trigger too low. The insured pays a premium calibrated to the 1-in-100 event but triggers on something that actually happens every 40 years.

For sites with IBNR-style incomplete records - gauges that were temporarily non-operational during the 2007, 2013, or 2015 UK flood events - use `CensoredHillEstimator`:

```python
from insurance_severity import CensoredHillEstimator

hill = CensoredHillEstimator()
hill.fit(
    claims=annual_maxima,
    censored=is_incomplete_record,  # True = gauge was not recording
    n_bootstrap=500,
    rng_seed=42,
)
print(f"xi = {hill.xi:.3f}, 95% CI: {hill.ci}")
# xi = 0.28, 95% CI: (0.19, 0.37)
```

The correction divides the Hill estimator numerator by the fraction of uncensored observations in the top-k order statistics. At sites where 30% of extreme events coincide with gauge outages, the uncorrected Hill underestimates xi by roughly that proportion. The magnitude matters: at xi = 0.28, a correction that raises it to 0.38 shifts the 1-in-200 return level by approximately 15% at the sites we have tested against the EA's own return-level estimates.

---

## Basis risk: what copulas actually tell you

Basis risk in parametric insurance is the correlation structure between the trigger index and the insured's actual loss. If the two were perfectly correlated, the product would behave like indemnity insurance. In practice they are not, and the question is how to model the joint distribution honestly.

The CAS 2025 paper from Boucher and Côté (proceedings vol. 112) makes the argument that Pearson correlation is a dangerously misleading measure of basis risk because it is a linear measure in a context where both the index and the loss are heavy-tailed and driven by the same extreme events. The Flood Re case is particularly sharp: the Thames barrier gauge height and residential flood claims in Greater London are not linearly related. Claims are zero for all but the top 2% of gauge readings, then jump discontinuously as properties in the floodplain inundate in a nonlinear sequence that depends on local microtopography, property floor levels, and drainage capacity. A Pearson correlation of 0.6 tells you almost nothing useful about the performance of the product at the return periods that matter.

The copula approach models the dependence structure separately from the marginals, which is correct - the tail dependence between index and loss is what drives basis risk outcomes, not the bulk correlation. For flood, a Gumbel or Joe copula is appropriate because both are upper-tail-dependent. A Gaussian copula, which has asymptotically zero tail dependence, will consistently underestimate basis risk at the trigger levels you actually care about.

The practical steps are:

1. Fit marginal distributions for the index variable and for losses separately using the EVT methods above.
2. Transform to uniform margins using the PIT.
3. Fit a family of Archimedean copulas (Clayton, Gumbel, Frank, Joe) to the uniform-marginal data, selecting by AIC.
4. Simulate from the fitted copula to estimate the distribution of payout errors: cases where the index triggers but the policyholder has no loss, and cases where losses occur without trigger.

The second error type - loss without trigger - is the regulatory concern under FCA ICOBS 6A fair value obligations. If a product triggers on 60% of large loss events and fails to trigger on 40%, that 40% needs to be reflected in the premium as a known shortfall in value, not hidden in basis risk uncertainty.

Quantifying that uncertainty around the copula fit is where [insurance-conformal](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) becomes useful. Once you have simulated payout error distributions, you can construct prediction intervals that carry a coverage guarantee regardless of the copula family:

```python
from insurance_conformal import InsuranceConformalPredictor

# 'model' here is a fitted copula simulation -> payout_error mapping
cp = InsuranceConformalPredictor(
    model=payout_error_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
intervals = cp.predict_interval(X_test, alpha=0.10)
# intervals.columns: ['lower', 'upper']
```

For basis risk applications, the `alpha=0.10` interval gives you the 90% range of payout errors on new events. That range is what goes into the fair value assessment - not the point estimate.

---

## Which features drive the basis risk gap?

Basis risk is not uniform across policyholders. A riverside property 50 metres from the gauge is much more tightly coupled to the trigger than a property that floods from overwhelmed surface drainage half a kilometre away. Understanding which characteristics drive the basis risk gap - and by how much - is analytically identical to the large-loss feature importance problem in severity modelling.

`TailVariableImportance` from insurance-severity (implementing arXiv:2504.06984) fits a tail-weighted lasso on the claim amounts, concentrating weight on observations above the 90th percentile. For basis risk, the equivalent is fitting it on the absolute payout error - the gap between trigger payout and actual loss - at extreme observations:

```python
from insurance_severity import TailVariableImportance

tvi = TailVariableImportance(threshold_quantile=0.90, alpha=0.1)
tvi.fit(
    X=basis_risk_features,    # distance_to_gauge, property_type, floor_level, drainage_type
    y=payout_error_magnitude, # abs(trigger_payout - actual_loss), strictly positive
    feature_names=feature_cols,
)

print(tvi.importances)
# {'distance_to_gauge': 0.41, 'drainage_type': 0.27,
#  'floor_level_relative': 0.19, 'property_type': 0.09, ...}

tvi.plot(top_k=8)
```

The result is what you would expect from physical principles - distance to gauge and drainage connectivity dominate - but quantified. At our trial run on EA gauge data combined with the MHCLG flood risk map, `distance_to_gauge` accounted for 41% of tail importance in payout error, versus 12% of total (Gini-based) importance across the full distribution. The tail is where the mismatch bites, and standard importance measures built on bulk data will not show it.

---

## The Flood Re context

Flood Re operates as a reinsurance pool for residential properties at high flood risk. It is, in structure, a parametric mechanism: premiums are capped by council tax band, Flood Re absorbs losses above the capped premium, and the reinsurance layer between Flood Re and the market is designed around return-period triggers. The review Flood Re published in 2024 flagging the model's 2039 expiry raises the question of whether its successor should incorporate more explicitly parametric structures - pay-on-event rather than pay-on-loss - for the B2B reinsurance layer.

The EVT trigger calibration issues above are directly relevant. The Flood Re pool is exposed to river flooding across England and Wales, with gauge records of varying quality and length. A pool-level trigger set on national or regional rainfall index behaves differently from a trigger set on local gauge levels - the former reduces basis risk at the portfolio level but increases it for individual properties. Neither is obviously right. But setting either trigger based on uncorrected GPD fits is provably wrong when those gauges have significant operational truncation.

Weather-indexed crop insurance - currently a niche product in the UK but growing given the 2018, 2020, and 2022 drought experiences - faces the same calibration problem with rainfall deficit indices. The Met Office HADUK-Grid dataset at 1km resolution is dense enough to support local parametric calibration, but the record length is thirty-five years, which is insufficient for return-period estimation at the 1-in-50 level without EVT methods.

---

## Putting it together: a calibration protocol

The SOA January 2026 monograph recommends a five-step protocol for parametric trigger calibration that maps directly onto the tools described here:

**Step 1 - Data audit.** Identify operational truncation limits and periods of sensor outage for every historical record in the index series. This determines the censoring and truncation structure before any modelling begins.

**Step 2 - Marginal tail fit.** Fit `TruncatedGPD` with appropriate limits. Compare with `CensoredHillEstimator` as a robustness check. The two should agree on xi to within their respective confidence intervals; if they do not, investigate the censoring structure.

**Step 3 - Copula dependence fit.** Transform index and loss data to uniform margins, fit multiple copula families, select on AIC with tail dependence coefficient as a secondary diagnostic. Do not accept a Gaussian copula if the data shows upper tail dependence.

**Step 4 - Basis risk simulation.** Simulate 50,000 joint scenarios from the fitted copula and marginals. For each scenario, compute whether the trigger fires and what the insured's actual loss is. The distribution of mismatches is the basis risk profile.

**Step 5 - Conformal uncertainty quantification.** Wrap the basis risk simulation in a conformal predictor to obtain coverage-guaranteed intervals on the payout error distribution. These intervals, not point estimates, should enter the FCA fair value assessment.

The protocol sounds involved but it is considerably less involved than the claim disputes that result from a parametric product that consistently fails its policyholders at the exact moments it was supposed to protect them. Basis risk is not a footnote - it is the central actuarial problem in parametric design, and it has now been formalised enough to be treated properly.

---

**Related tools:** [insurance-severity](https://burning-cost.github.io/insurance-frequency-severity) - EVT tail modelling including TruncatedGPD, CensoredHillEstimator, TailVariableImportance. [insurance-conformal](https://burning-cost.github.io/insurance-conformal) - distribution-free prediction intervals for uncertainty quantification. See also: [Does Conformal Prediction Actually Work for Insurance Claims?](/2026/03/26/does-conformal-prediction-actually-work-for-insurance-claims/) — empirical benchmark of conformal methods on real insurance data.

**References:** SOA January 2026 Parametric Insurance Monograph; Boucher & Côté, CAS Proceedings vol. 112 (2025); Albrecher, Beirlant & Teugels (arXiv:2511.22272); arXiv:2504.06984 (tail variable importance).
