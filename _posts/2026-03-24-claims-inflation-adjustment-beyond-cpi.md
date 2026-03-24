---
layout: post
title: "Claims Inflation Adjustment in Pricing Models: Beyond CPI"
date: 2026-03-24
categories: [pricing]
tags: [claims-inflation, trend, cpi, ogden, motor, home, severity, frequency, loss-cost, insurance-trend, superimposed-inflation, ons, pricing-actuary]
excerpt: "CPI-adjusting your historical claims data before fitting a pricing model introduces systematic bias - here is what to use instead."
---

Many pricing teams CPI-adjust their historical claims data before fitting a frequency-severity model. The logic is sensible: claims from 2019 are not directly comparable to claims from 2024 in nominal terms, so you inflate them to today's money before fitting. The problem is that CPI is the wrong index, and in some lines it is badly wrong.

This matters because an incorrect inflation adjustment does not just change the intercept of your model. It changes the factor relativities whenever the inflation rate varies systematically across segments - which, in motor and home, it does.

---

## Why CPI is inadequate

CPI is a consumer expenditure basket. It measures the price change of a representative bundle of goods and services for UK households. Motor insurance repair costs are not in that basket in any meaningful proportion. The ONS category that comes closest is 12.5.4.1 (Motor vehicle insurance premiums), which tracks insurance prices rather than repair costs. That is measuring the output, not the input.

The actual inflation drivers for motor claims are:

**Parts prices.** The supply chain disruptions of 2021-2023 pushed parts costs for common vehicles up 20-35% in some segments. Modern vehicles increasingly require manufacturer-specific electronic components that are not available through the aftermarket. A 2023 Kia Sportage bumper sensor suite costs materially more than a 2018 Ford Focus equivalent. CPI does not track this.

**Labour costs.** Automotive technician wages have increased substantially since 2021, partly due to genuine skills shortages and partly due to the electrification transition requiring different expertise. The ONS ASHE data for vehicle technicians (SOC 5231) shows average earnings growth running ahead of the all-employee median since 2022.

**ADAS recalibration.** Advanced Driver Assistance Systems - lane-keep assist cameras, forward collision radar, parking sensor suites - require recalibration after any repair that disturbs their mounting. Windscreen replacement on most vehicles built after 2019 now includes a camera recalibration step. Bumper repairs require sensor array checks. These costs did not exist at scale five years ago. They are now a material fraction of average repair cost for new vehicles.

**Theft mix.** Catalytic converter theft ran at near-epidemic levels in 2021-2022, particularly affecting hybrid vehicles. The average cost per catalytic converter claim (replacement plus fitting) was substantially above the overall book average. When theft patterns change, the mix of claim types changes, and average cost changes for compositional reasons rather than price inflation reasons.

For home claims, the inflation drivers are different but equally non-CPI:

**Rebuild costs.** The BCIS House Rebuilding Cost Index tracks materials and labour for property reinstatement. It has historically run ahead of CPI due to skilled trades scarcity and regional concentration of housebuilding capacity. Since 2022 it has run significantly ahead, with tender price inflation reaching double digits in 2022 before moderating.

**Escape of water.** Average repair costs for escape of water claims have increased partly because of the shift to plastic pipework (which is cheaper to replace but spreads water further before detection) and partly because of the increasing complexity of fitted kitchens and bathrooms. A claim in a premium property with underfloor heating and a smart home installation has a fundamentally different cost structure than the equivalent claim from 2015.

---

## What to use instead

The correct approach is to use a sector-specific index and decompose the residual.

For UK motor, the ONS publishes the SPPI G4520 series (Maintenance and repair of motor vehicles), series code HPTH. This is the relevant producer price index for repair cost inflation. It is free, public, and available from the ONS API with no authentication required. It runs monthly.

For UK home, the relevant ONS series is D7DO (CPI 04.3.2, Services for maintenance and repair of dwellings). For reinstatement cost inflation, BCIS data is more authoritative but requires a subscription.

Even a sector-specific index will not capture everything. The residual after deflating by an appropriate index is called superimposed inflation: the cost growth that the index does not measure. ADAS recalibration costs are largely superimposed inflation on HPTH, because they represent new cost types not previously in the repair price basket.

The [`insurance-trend`](https://github.com/burning-cost/insurance-trend) library handles this decomposition. You provide your historical claims series and the external index, and it fits a severity trend on the deflated series, reporting the index-measured component and the residual superimposed component separately.

```bash
uv add insurance-trend
```

```python
from insurance_trend import SeverityTrendFitter, ExternalIndex

# ONS HPTH: maintenance and repair of motor vehicles
motor_repair_index = ExternalIndex.from_ons(
    'HPTH',
    start_date='2019-01-01',
    frequency='quarters',
)

fitter = SeverityTrendFitter(
    periods=['2019Q1','2019Q2','2019Q3','2019Q4',
             '2020Q1','2020Q2','2020Q3','2020Q4',
             '2021Q1','2021Q2','2021Q3','2021Q4',
             '2022Q1','2022Q2','2022Q3','2022Q4',
             '2023Q1','2023Q2','2023Q3','2023Q4',
             '2024Q1','2024Q2','2024Q3','2024Q4'],
    total_paid=[8.2e6, 8.6e6, 8.9e6, 8.4e6,
                5.8e6, 4.1e6, 7.2e6, 8.0e6,
                8.3e6, 8.9e6, 9.4e6, 9.1e6,
                10.2e6, 11.1e6, 11.8e6, 11.4e6,
                12.1e6, 12.8e6, 13.2e6, 12.7e6,
                12.9e6, 13.3e6, 13.6e6, 13.1e6],
    claim_counts=[1840, 1910, 1960, 1820,
                  1730, 840,  1200, 1650,
                  1720, 1780, 1830, 1760,
                  1790, 1850, 1880, 1800,
                  1770, 1820, 1850, 1780,
                  1750, 1800, 1830, 1760],
    external_index=motor_repair_index,
    periods_per_year=4,
)

result = fitter.fit(detect_breaks=True, n_bootstrap=1000)
print(result.summary())
print(f"Superimposed inflation: {fitter.superimposed_inflation():.2%}")
```

The fit is performed on the deflated series: average cost divided by the re-based index. The reported `trend_rate` is therefore residual trend after removing the index-measured component. `superimposed_inflation()` returns this residual. On a typical recent UK motor book, we would expect the HPTH-deflated series to still show 2-4% annual superimposed inflation from ADAS complexity effects.

---

## Applying trend factors in a frequency-severity framework

The purpose of all this is to put historical claims on a current-cost basis before fitting your pricing model. The mechanics depend on whether you fit on policy-level data or aggregated triangles.

For a policy-level frequency-severity model, the approach is to trend each historical claim to the pricing date before fitting the severity model. The trend factor for a claim occurring at time $t$ is:

$$\text{TF}(t) = \left(1 + r\right)^{T - t}$$

where $r$ is the annual trend rate and $T$ is the pricing date (the mid-point of the future policy period you are pricing for). A claim from Q1 2022 being used in a model pricing 2025 policies has an approximate trend period of 3.5 years at the mid-point.

In practice, using a single trend rate across the whole projection horizon is an approximation. The compound factor from 2022 to 2025 should use the trend rate appropriate to each sub-period: the 2022-2023 inflation spike was not the same rate as the expected 2024-2025 moderation. The `insurance-trend` library's `trend_factor()` method handles period-specific compounding:

```python
# Trend factor from 2022Q1 (period index 12) to 2025Q2 (period 25)
# Using the fitted piecewise model
factor = result.trend_factor(periods=13)  # 13 quarters forward
print(f"Trend factor: {factor:.4f}")      # e.g. 1.241
```

Applied to a severity model, this means your fitting dataset has each historical claim inflated by its individual trend factor before it enters the regression. A claim worth £2,100 in 2022Q1, trended at a 1.241 factor, enters the severity model at £2,606. The GLM then fits to trended costs, so its predictions are automatically on the future-pricing basis.

This is cleaner than fitting on nominal costs and then applying a portfolio-level adjustment at the end, because the latter approach assumes inflation has been uniform across all claim types and rating factor combinations - which it has not.

---

## The Ogden rate problem

For liability lines, court award inflation has a distinct driver: the Ogden discount rate. The rate changed from -0.75% to -0.25% in July 2019, which materially reduced the lump sum multipliers applied to serious injury claims. Claims settled after the change had smaller capitalised values than equivalent claims settled before it.

This creates a structural break in any liability severity series that straddles July 2019. The level shift is not inflation in any ordinary sense; it is a regulatory change with a specific date. If you fit a single trend through this break, the regression slope absorbs some of the level shift as a trend estimate, giving you biased trend rates on both sides of the boundary.

The practical fix is to treat July 2019 as a known break point and fit separate trend lines before and after it. `insurance-trend`'s `detect_breaks=True` option uses the PELT algorithm to detect structural breaks automatically; you can also force a break at a specific index with `changepoints=[n]` if you know where it occurred.

For motor bodily injury, the 2019 Ogden change was large enough relative to within-period noise that PELT will typically detect it on any book with material injury exposure. For smaller books, the signal may not stand out clearly enough and you should force the break manually.

The next Ogden review is ongoing - the Ministry of Justice has been expected to consult on a further change. Any pricing team modelling BI severity should have a break-point sensitivity built into their trend framework, not a single uninterrupted regression.

---

## Claims mix changes

There is a category of severity change that is not inflation at all: changes in the mix of claim types. If your theft claim frequency doubles because catalytic converter theft surged, your average severity changes because theft claims have a different average cost from own damage claims. This would show up as severity inflation even on an inflation-adjusted basis, because it is a compositional shift.

The correct response is to model frequency and severity separately by peril or claim type where the data supports it. A motor book with materially separated theft, windscreen, and own damage components will produce more stable severity trends on each sub-component than on the blended average. The blended average severity will oscillate with peril mix changes in a way that makes trend estimation genuinely difficult.

This is especially relevant for home insurance, where the frequency of escape-of-water, storm, fire, and accidental damage claims each has its own inflation driver. A home book that has seen its storm frequency increase due to more severe weather will show apparent average cost increases even if the per-claim costs are stable, because storm claims are above average cost. Modelling frequency and severity by peril is standard practice in property cat modelling; it should be standard practice in personal lines home pricing too.

---

## Putting it together: a realistic UK motor example

Suppose you are pricing a mid-market motor book for 2025. Your experience data runs from 2020 to 2024. You exclude 2020 Q2-Q3 from the frequency trend because lockdown-period frequencies are uninformative about the structural level (force a break or exclude those quarters entirely). For the severity trend you use HPTH deflation and detect the post-supply-chain-disruption moderation as a structural break around 2023Q3.

Selected trend rates might look like:

| Component | Selected annual rate | Source |
|---|---|---|
| Frequency | -1.2% | PELT piecewise log-linear, post-2021 period, 95% CI: (-2.1%, -0.3%) |
| Severity (HPTH-deflated) | +3.4% | PELT piecewise, post-disruption period, 95% CI: (+2.1%, +4.8%) |
| Superimposed inflation | +2.3% | Residual after HPTH deflation |
| Total severity trend | +5.7% | Deflated trend + superimposed component |
| Combined loss cost | +4.4% | (1 - 0.012) × (1 + 0.057) - 1 |

On a mid-point-to-mid-point basis from 2022 (mid-point of the 2020-2024 experience period) to 2025.5 (mid-point of 2025 policies), that is approximately 3.5 years at 4.4% annual combined: a trend factor of about 1.163. Applied to a base pure premium of £450, that projects to £523.

The difference between using this approach and simply applying CPI (which ran at about 6% annualised over this period in a way that would overstate severity inflation in 2022-2023 and potentially understate the structural superimposed component) is material. It is not a rounding error; it is a pricing decision.

---

## Why this matters for model fitting, not just rate indication

Most of this discussion has focused on trend factors for rate indication purposes. But the same issue applies to the historical data you use to fit your frequency-severity model.

If you fit a severity model on 2020-2024 claim data without adjusting for the 2021-2023 inflation spike, your model will produce predictions that are anchored to the 2020-2022 cost environment. The claims from 2023-2024 will have high deviance residuals. The factors most correlated with high-cost claims - vehicle group, claim type, policy area - may appear to be drifting in value when what is actually happening is that inflation hit those segments harder.

The [insurance-monitoring library](https://github.com/burning-cost/insurance-monitoring) will flag this as response drift: an elevated AE ratio and a degraded Gini coefficient. The signal it is giving you is correct, but the cause is not model mis-specification - it is inflation that was not accounted for during fitting.

The fix is to trend all claims in your fitting dataset to a common basis date before fitting. This is a one-time step in the data preparation pipeline; once it is there, future model refits use consistent cost-basis claims automatically.

`insurance-trend`'s `LossCostTrendFitter` gives you the combined trend parameters. Applying the per-claim trend factors is then a join in Polars:

```python
import polars as pl

# trend_table: Polars DataFrame with columns [accident_quarter, trend_factor]
# generated from result.projection on the base-to-pricing-date range

claims_trended = (
    claims
    .join(trend_table, on="accident_quarter", how="left")
    .with_columns(
        (pl.col("paid_amount") * pl.col("trend_factor")).alias("trended_amount")
    )
)
```

The severity model then fits on `trended_amount`. Everything else in the GLM specification is unchanged.

---

`insurance-trend` is open source at [github.com/burning-cost/insurance-trend](https://github.com/burning-cost/insurance-trend). Install with `uv add insurance-trend`. The ONS HPTH and D7DO index integrations require no API key. BCIS and other subscription data can be loaded from CSV.

- [Tracking Trend Between Model Updates with GAS Filters](/2026/03/08/gas-models-for-between-update-trend/)
- [Your GLM Trend is a Straight Line Through a Structural Break](/2026/03/18/your-glm-trend-is-a-straight-line-through-a-structural-break/)
- [Migrating from Emblem to Python: What Actually Changes](/2026/03/24/migrating-from-emblem-to-python/)
