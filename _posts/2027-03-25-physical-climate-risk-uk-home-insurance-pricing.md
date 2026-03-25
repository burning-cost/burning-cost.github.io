---
layout: post
title: "Physical Climate Risk in UK Home Insurance Pricing"
date: 2027-03-25
categories: [pricing, techniques, tutorials]
tags: [climate-risk, flood, weather, home-insurance, ukcp18, flood-re, whittaker, conformal, monitoring, insurance-whittaker, insurance-conformal, insurance-monitoring, trend, peril-modelling, uk-personal-lines, python]
description: "How UK home insurers should model physical climate risk: UKCP18 projections, Flood Re's 2039 exit, ABI claims data, and practical code using insurance-whittaker, insurance-conformal, and insurance-monitoring."
---

The ABI reported £4.6 billion in property insurance payouts in the first nine months of 2025 alone — weather damage accounted for £936 million of that figure, £143 million more than the same nine months of 2024. Full-year 2024 came in at £5.7 billion, the largest annual total on record. Full-year 2023 saw a record £573 million from weather claims on its own, driven by Storms Babet, Ciaran, and Debi landing in a single autumn.

None of this is noise. It is the signal that the frequency-severity assumptions baked into most UK home pricing models at validation time are now wrong by a compounding margin each year.

This post covers how to think about the problem and what to do about it in your pricing stack.

```bash
uv add insurance-whittaker insurance-conformal insurance-monitoring
```

---

## What UKCP18 actually tells you

The UK Climate Projections 2018 (UKCP18) are the Met Office's current probabilistic projections for UK climate to 2100. The key results for home insurance pricing are not subtle:

- **Rainfall intensity.** The 1-in-100-year daily rainfall event becomes a 1-in-50 year event in most of England and Wales under the central RCP8.5 pathway by the 2050s. The upper end of the probability range makes it a 1-in-25 year event. These are 15-minute-resolution projections. Your flood accumulation model is calibrated on historical events that pre-date this shift.

- **Subsurface moisture.** Extended summer dry spells, increasing in duration and severity under all scenarios, drive clay shrinkage and subsidence. UKCP18 projects longer and more intense summer droughts in southern England. Ground investigation data from the 2018 and 2022 summers gives you a preview: the CILA reported a marked spike in subsidence claims notifications in H2 2022 following that summer's heat.

- **Sea level and coastal surge.** Mean sea level rise of 20-30 cm by 2070 under the central scenario, with the upper bound significantly higher. For coastal property in Kent, Essex, Somerset, and parts of West Wales, this changes the effective return period of flood events at the lower end of the hazard curve meaningfully.

Three things to be clear about before using UKCP18 in pricing:

1. These are probabilistic projections, not point forecasts. The 10th-90th percentile range is wide for anything beyond about 20 years. You are not being given a number; you are being given a distribution.
2. The near-term signal (next 10-15 years) is tighter than the long-term signal. The 2039 Flood Re transition sits within the near-term window where the projections are most useful.
3. UKCP18 tells you about hazard — the physical event. Translating that into insured loss requires exposure and vulnerability assumptions that are yours to make. The projections do not do that work for you.

The appropriate use of UKCP18 in pricing is to stress your trend assumptions, not to replace historical data with projection outputs. We will come back to this.

---

## Flood Re: the 2039 problem

Flood Re is the UK's flood reinsurance backstop, launched in 2016. It allows household insurers to cede flood risk on high-flood-risk properties into a shared pool, priced on council tax band rather than actual risk. As of 2024, around 250,000 policies are in the scheme.

The scheme has a designed end date: 2039. The intent, embedded in the scheme's original design and reaffirmed in the 2023 quinquennial review, is that Flood Re will transition properties back to risk-reflective market pricing over the 2025-2039 period. The mechanism is a staircase of increasing cession premiums that makes Flood Re progressively less attractive to cede into, forcing market pricing before the backstop disappears entirely.

What this means for pricing:

**If you are currently ceding into Flood Re**, your flood book is cross-subsidised. The risk is in the pool, not in your model. When the backstop goes, you will need either a credible individual-risk flood pricing capability, or you will exit the high-risk segment. Neither option is quick to build. The 2039 deadline is 13 years away. That sounds comfortable and it is not — the development and validation cycle for a market-quality postcode-level flood pricing model, including exposure data build, model development, back-testing against recent event losses, and regulatory review, is typically 3-5 years for a serious insurer starting from scratch.

**If you are not ceding into Flood Re**, your pricing is presumably already risk-reflective, but the transition creates a market repricing event. Properties that have been cross-subsidised become correctly priced. Customer flow from those properties creates adverse selection pressure on the market. Your portfolio composition changes.

**The UKCP18 interaction.** The Flood Re scheme was designed under assumptions about future flood risk that predate some of the most useful UKCP18 outputs. The 2039 backstop removal was calibrated partly on the expectation that flood risk management investment (FCERM) would reduce exposure over the period. That investment is proceeding but not at the pace originally assumed. The net result: in 2039, market participants will be pricing individually for properties in areas where the underlying hazard has shifted materially from the state at Flood Re's inception.

---

## The modelling challenge: trend under climate shift

Standard peril modelling assumptions — stationary frequency and severity, independent of calendar year after rating factors — fail in two ways under climate shift.

First, the trend is not constant. Historical trend smoothing approaches that fit a single gradient to five or ten years of data will underfit in the near term (when the trend was lower) and overfit if the most recent years are heavily weather-affected. You need a smoother that can represent a changing gradient without overfitting to individual event years.

Second, the uncertainty in the trend is substantial and should propagate into the rates. A point estimate of flood frequency trend that ignores estimation uncertainty will systematically underprice in the tail.

[`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker) handles both of these. The Whittaker-Henderson smoother is the right tool here: it fits a penalised curve to claims frequency over calendar time, selects the smoothing parameter automatically via REML, and returns posterior credible intervals that reflect how uncertain the trend is given the data you have.

### Fitting a flood frequency trend

Suppose you have annual claim counts and policy-year exposures for your flood and storm damage perils, going back 12 years. Here is the approach:

```python
import numpy as np
from insurance_whittaker import WhittakerHendersonPoisson

# Years 2013-2024, annual flood frequency (claims per policy year)
years = np.arange(2013, 2025)
flood_counts  = np.array([412, 389, 445, 398, 521, 467, 503, 578, 612, 643, 698, 751])
policy_years  = np.array([85_000, 87_200, 89_100, 90_500, 91_800, 93_000,
                           94_100, 95_300, 96_400, 97_600, 98_800, 100_000])

smoother = WhittakerHendersonPoisson(order=2, lambda_method="reml")
result = smoother.fit(years, flood_counts, exposure=policy_years)

print(f"Smoothed flood rate 2024: {result.fitted_rate[-1]:.5f}")
print(f"95% CI: ({result.ci_lower_rate[-1]:.5f}, {result.ci_upper_rate[-1]:.5f})")
print(f"Effective degrees of freedom: {result.edf:.1f}")
```

The `order=2` penalty smooths second differences — it allows an accelerating trend without interpolating event spikes. The REML criterion balances fidelity to the data against smoothness; years with heavy storm losses (you can see the spike pattern in those counts) pull the fitted curve up but do not create a false step change in the trend.

The result gives you both the smoothed frequency estimate and a credible interval. That interval matters. If the 95% upper bound on the 2024 flood rate is 20% above the point estimate, that spread should inform your rate adequacy calculation, not be discarded.

For trend projection: the `WHResultPoisson` object gives you `fitted_rate` for the in-sample years. For a near-term projection (say, 2025-2030), apply the UKCP18 hazard scaling factors to the endpoint of the smoothed curve, treating them as a multiplicative trend assumption. Do not fit a smooth through projected values — the projections are not data.

---

## Prediction intervals for extreme weather claims

Once you have a frequency model, the severity tail is the other half of the problem. For flood and storm perils, the severity distribution in a bad year is driven by a small number of very large losses. Your GLM predicts expected severity. What you need to price adequately — and to set your reinsurance programme correctly — is the conditional prediction interval, particularly in the upper tail.

[`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) provides distribution-free prediction intervals with valid coverage guarantees, even when the underlying severity distribution has a heavy tail and the parametric assumptions of your Tweedie model are wrong.

The `ConformalisedQuantileRegression` class is the right tool when you have a quantile model for severity:

```python
from catboost import CatBoostRegressor
from insurance_conformal import ConformalisedQuantileRegression

# Fit quantile models for the 5th and 95th percentiles of claim severity
lo = CatBoostRegressor(loss_function="Quantile:alpha=0.05", iterations=400, depth=6)
hi = CatBoostRegressor(loss_function="Quantile:alpha=0.95", iterations=400, depth=6)
lo.fit(X_train, y_train)
hi.fit(X_train, y_train)

# Conformally calibrate to guarantee marginal coverage
cqr = ConformalisedQuantileRegression(model_lo=lo, model_hi=hi)
cqr.calibrate(X_cal, y_cal)

intervals = cqr.predict_interval(X_test, alpha=0.10)
# intervals["lower"] and intervals["upper"] carry the 90% coverage guarantee
# regardless of quantile model misspecification
```

The conformal calibration step adjusts the raw quantile model outputs so the actual 90% interval contains at least 90% of observations — not as an assumption but as a finite-sample guarantee. In the context of weather-exposed home insurance, where the true severity distribution shifts with each storm event and your training data may not represent recent cat-exposed experience, that guarantee is worth having.

For the most weather-exposed segment — properties with high flood score and high sum insured — the uncalibrated quantile model's 95th percentile is typically too narrow. The conformal correction widens it to the empirically correct level. That width difference directly affects your reinsurance XL attachment pricing.

If you want to run this per-peril rather than across the whole home book, run a separate `ConformalisedQuantileRegression` for each peril class. Do not pool flood and storm damage with escape of water — the severity distributions are too different for pooled calibration to be useful.

---

## Detecting when your climate assumptions are stale

A pricing model built on 2019-2023 weather data is already potentially stale for flood frequency in 2025. The question is how to detect this quickly, with formal error control, rather than waiting for an adverse A/E result in the annual review.

[`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) has `PITMonitor`, an anytime-valid calibration change detector. It constructs an e-process over probability integral transforms (PITs) and raises an alarm when calibration has shifted — with a guaranteed false alarm rate of at most `alpha` over any monitoring horizon, no matter how frequently you check.

For a flood frequency model:

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

# Initialise at model deployment — one monitor per peril is right
# Run a separate monitor per peril class; do not pool flood and storm
flood_monitor = PITMonitor(alpha=0.05, rng=2025)

for month_batch in incoming_claims:
    for row in month_batch:
        # PIT for Poisson frequency model: CDF evaluated at observed count
        mu = row.exposure * row.flood_rate_hat
        pit = float(poisson.cdf(row.flood_claims, mu))
        alarm = flood_monitor.update(pit)
        if alarm:
            print(
                f"Flood frequency calibration alarm at step {alarm.time}, "
                f"estimated changepoint: {alarm.changepoint}"
            )
            # Trigger recalibration workflow
```

The `PITMonitor` detects changes, not absolute miscalibration. A model that consistently over-predicts will not alarm here — use `CalibrationChecker` at deployment for that check. What `PITMonitor` catches is the model that was calibrated and has since drifted: exactly the situation when a sustained shift in flood frequency has moved the book outside the training distribution.

The `alarm.changepoint` estimate tells you roughly when the drift began. If the changepoint is within the last 12 months, you have a recent event driving the change and recalibration may be sufficient. If it is 24-36 months back, you have a structural shift and you need to revisit the frequency model itself.

For climate risk specifically, we recommend running `PITMonitor` per peril class at monthly granularity. The anytime-valid property means you can check monthly without inflating false alarm rates — the usual problem with repeated H-L or A/E monitoring that leads actuaries to check quarterly to control false positives.

---

## Putting it together: a practical framework

The full workflow for climate-adjusted home peril pricing looks like this:

1. **Trend estimation with credible intervals.** Fit `WhittakerHendersonPoisson` to each weather peril (flood, storm, subsidence) separately. Extract both the point estimate and the posterior 95% CI. Use the CI width as input to your rate uncertainty loading, not just the point estimate.

2. **Near-term hazard scaling.** Apply UKCP18 near-term scaling factors (10-30 year window) to the smoothed frequency endpoint. The Met Office UKCP18 regional Marine Projections and Land Projections are the right source. Do not use projections beyond 30 years for pricing — the uncertainty is too wide to be actionable.

3. **Severity tail intervals.** Fit `ConformalisedQuantileRegression` per peril to ensure your prediction intervals at the 90th and 95th percentile are empirically valid. Use these intervals for large loss loading and reinsurance attachment pricing.

4. **Flood Re exposure identification.** Flag properties currently ceded into Flood Re and model the repricing step at the 2039 transition. The repricing cannot be left until 2038 — distribution partners and customers will need to be managed through the transition. If you do not have postcode-level flood scoring for these properties now, build it now.

5. **Production monitoring.** Deploy `PITMonitor` per peril from the day the new pricing model goes live. Monthly update, per-peril monitors, `alpha=0.05`. When the alarm fires, the `changepoint` estimate tells you where to look.

---

## What this does not solve

A pricing model with correct trend estimation, valid prediction intervals, and production monitoring for drift is better than most UK home insurers have today. It does not solve:

**Tail dependence between perils.** Storm damage and flood are correlated — the same weather event drives both. Your aggregate exposure in a bad year is higher than the sum of independent peril models predicts. This requires either a joint severity model (see [`insurance-conformal`'s `JointConformalPredictor`](https://github.com/burning-cost/insurance-conformal)) or explicit catastrophe scenario analysis.

**Climate non-stationarity in the exposure base.** The properties most exposed to flood and subsidence risk under UKCP18 are also the properties where insurance pricing discipline is most commercially sensitive. Correct pricing may reduce GWP in some segments. That is a board-level decision, not a modelling decision, but the modelling should at least make the risk visible.

**Physical vulnerability changes.** UKCP18 tells you about hazard frequency. The insurance loss also depends on building vulnerability — how much damage a flood event of a given depth causes, for a given property type. Vulnerability functions are typically based on historical loss data from events prior to the period when the most recent retrofitting took place. If you are pricing a portfolio with a significant proportion of recently flood-mitigated properties (FCRM-funded property-level schemes), your historical severity may overstate future loss.

The modelling problem is tractable. The bigger challenge is that the industry's current pricing tools — most of them built for stationary risk — were not designed to handle a trend that compounds annually at an uncertain rate. The three libraries above provide the building blocks. The architectural decision to use them is the one that needs making.

---

Source: [ABI property insurance payouts — year-to-date record £4.6 billion (Nov 2025)](https://www.abi.org.uk/news/news-articles/2025/11/year-to-date-property-insurance-payouts-hit-record-4.6-billion/) | [ABI full-year 2024 weather claims](https://www.abi.org.uk/news/news-articles/2025/2/more-action-needed-to-protect-properties-as-adverse-weather-takes-record-toll-on-insurance-claims-in-2024/) | [UKCP18 Marine and Land Projections](https://www.metoffice.gov.uk/research/approach/collaboration/ukcp/index)

- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/) — when a single Tweedie phi is not enough for variance-aware home pricing
- [How to Build a Large Loss Loading Model for Home Insurance](/2026/10/14/large-loss-loading-for-home-insurance/) — the per-risk tail loading approach for high-sum-insured properties
- [Recalibrate or Refit?](/2026/05/14/recalibrate-or-refit/) — how to diagnose whether a drifting model needs a recalibration or a full rebuild
