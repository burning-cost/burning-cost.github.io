---
layout: post
title: "Trend Selection Is Not Actuarial Judgment: A Python Approach"
date: 2026-03-13
categories: [techniques, libraries, pricing]
tags: [trend, frequency, severity, loss-cost, superimposed-inflation, ons-api, structural-breaks, ruptures, statsmodels, python, insurance-trend]
description: "Python insurance trend library: log-linear OLS/WLS, bootstrap CIs, ONS deflation, superimposed inflation, structural break detection - insurance-trend."
---

Every rate filing contains a trend selection. It sits somewhere between the loss development section and the credibility section, usually a table with three columns: component, selected rate, source. Frequency: -2.0%, selected. Severity: +7.5%, selected. Loss cost: +5.4%, calculated.

The "source" column says something like "actuarial judgment" or "five-year linear regression." Neither tells you anything useful. Neither tells you whether the regression accounted for seasonality. Neither tells you whether the trend was fitted through the COVID lockdown years or whether those quarters were excluded. Neither tells you whether the severity trend is gross of the 2022–2023 inflation spike or net of it. And neither tells you what the confidence interval around that +7.5% looks like, which matters because the difference between +5% and +10% severity trend, compounded over a two-year policy horizon, is roughly a 5-point swing in the loss ratio.

Trend is the largest unexamined number in the rate. It gets less scrutiny than the development tail, despite being more consequential for future-year pricing. This is partly because the tooling is poor: actuaries do trend in Excel, or in R scripts that nobody else can run, or in proprietary systems that produce a number but not reproducible code.

[`insurance-trend`](https://github.com/burning-cost/insurance-trend) is our 30th open-source library. Nine modules, 2,273 lines, 154 tests, v0.1.0. It brings frequency trend, severity trend, and combined loss cost trend into Python with composable fitters, ONS API integration, automatic structural break detection via ruptures, and 1,000-replicate bootstrap confidence intervals.

```bash
uv add insurance-trend
```

---

## The model

The library fits a log-linear model on frequency and severity separately:

```
log(freq_t) = α + β·t + Σ γ_k·seasonal_k + ε_t
```

The annual trend rate is `exp(β × periods_per_year) − 1`. On quarterly data that means `exp(β × 4) − 1`. The logged left-hand side is the key decision: it constrains the trend to be multiplicative, which is the correct assumption for insurance claims. A 5% trend on a £1,000 average cost produces a £50 increment in year 1, a £52.50 increment in year 2. That is not what an additive model does, and the actuarial literature has always been clear that multiplicative is right.

Seasonal dummies are included by default for quarterly data (Q1, Q2, Q3 with Q4 as base). This matters for motor: roadworks, school holidays, and adverse weather create persistent seasonal patterns in frequency. If you fit trend through seasonal variation without dummy controls, the slope estimate absorbs some of the pattern. We have seen this produce trend estimates 1–2 percentage points different from the seasonally adjusted figure on typical UK motor data.

WLS is available via the `weights` parameter for anyone who wants to down-weight older periods. The default is equal weights (OLS). We do not pick a default decay scheme because the right answer is domain-specific and should be explicit.

---

## Frequency trend

```python
from insurance_trend import FrequencyTrendFitter

fitter = FrequencyTrendFitter(
    periods=['2019Q1', '2019Q2', '2019Q3', '2019Q4',
             '2020Q1', '2020Q2', '2020Q3', '2020Q4',
             '2021Q1', '2021Q2', '2021Q3', '2021Q4',
             '2022Q1', '2022Q2', '2022Q3', '2022Q4',
             '2023Q1', '2023Q2', '2023Q3', '2023Q4'],
    claim_counts=[1840, 1910, 1960, 1820,
                  1730, 840,  1200, 1650,
                  1720, 1780, 1830, 1760,
                  1790, 1850, 1880, 1800,
                  1770, 1820, 1850, 1780],
    earned_exposure=[18400, 18600, 18800, 18200,
                     18000, 17200, 17800, 18100,
                     18200, 18400, 18500, 18300,
                     18400, 18600, 18700, 18500,
                     18300, 18400, 18500, 18300],
    periods_per_year=4,
)

result = fitter.fit(n_bootstrap=1000, ci_level=0.95)
print(result.summary())
```

```
Method          : piecewise
Trend rate (pa) : -0.0183
95% CI          : (-0.0312, -0.0058)
R-squared       : 0.8741
Changepoints    : [4]
Bootstrap N     : 1000
```

The `changepoints: [4]` flag is the library detecting that index 4 (the Q1 2020 observation) marks a structural break in the log-frequency series. The `piecewise` method fits a separate slope to each segment and reports the trend from the final segment. That is almost certainly the right thing to do: a single regression through 2019–2023 that includes the Q2 2020 lockdown nadir would produce a spuriously steep positive trend as frequency recovered from its artificial floor.

The detection uses the PELT algorithm from the `ruptures` package with an L2 cost function and a default penalty of 3.0. PELT is O(n) in the number of observations; it does not brute-force all possible segmentations. The penalty controls the false positive rate; at 3.0 it is conservative enough for typical insurance quarterly series of 16–24 observations. The library warns when breaks are detected and asks you to review them for actuarial plausibility:

```
UserWarning: Structural breaks detected at indices [4].
Consider reviewing these dates for actuarial plausibility
(e.g. COVID lockdown, regulatory changes). Pass changepoints=[]
to suppress piecewise fitting.
```

You can pass `changepoints=[4]` to force the break at a specific index, `changepoints=[]` to suppress piecewise fitting entirely, or `detect_breaks=False` to skip automatic detection. The mechanism is explicit; nothing happens silently.

The 95% bootstrap CI runs 1,000 parametric replicates, resampling residuals from the OLS fit, refitting on each perturbed series, and collecting the slope distribution. The interval `(-3.12%, -0.58%)` tells you the trend is negative with reasonable confidence but not tight: the uncertainty spans 2.5 points. That uncertainty matters for pricing. If you are building a rate indication on this segment and the trend could plausibly be -0.6% or -3.1%, those are meaningfully different pricing answers.

---

## Severity trend and superimposed inflation

The severity side is where the interesting actuarial problem lives. Average cost per claim in UK motor has done something unusual since 2021: it has risen sharply even after accounting for general price inflation. The reason is structural. Modern vehicles have ADAS sensors embedded in the windscreen, bumper, and door mirrors. Replacing a bumper on a 2018 SEAT Leon requires recalibrating the parking sensor suite. A windscreen replacement on many current vehicles requires recalibrating the lane-keep assist camera. These parts and procedures do not appear in the ONS consumer price basket in proportion to their weight in motor repair costs.

The decomposition this requires is: how much of the severity trend is general economic inflation (CPI, RPI, or the ONS SPPI for vehicle repair — series code HPTH), and how much is structural cost inflation that a general index will not capture? The residual is superimposed inflation. It matters because economic inflation is at least partially offset by investment return; superimposed inflation is a pure cost. And it matters for reserving: IBNR on recent periods should be inflated at the total trend, not just the economic index.

```python
from insurance_trend import SeverityTrendFitter, ExternalIndex

# Fetch ONS HPTH: SPPI G4520, Maintenance & repair of motor vehicles
# No authentication, no API key — ONS public API
motor_repair_index = ExternalIndex.from_ons(
    'HPTH',
    start_date='2019-01-01',
    frequency='quarters',
)

fitter = SeverityTrendFitter(
    periods=['2019Q1', '2019Q2', '2019Q3', '2019Q4',
             '2020Q1', '2020Q2', '2020Q3', '2020Q4',
             '2021Q1', '2021Q2', '2021Q3', '2021Q4',
             '2022Q1', '2022Q2', '2022Q3', '2022Q4',
             '2023Q1', '2023Q2', '2023Q3', '2023Q4'],
    total_paid=[8.2e6, 8.6e6, 8.9e6, 8.4e6,
                5.8e6, 4.1e6, 7.2e6, 8.0e6,
                8.3e6, 8.9e6, 9.4e6, 9.1e6,
                10.2e6, 11.1e6, 11.8e6, 11.4e6,
                12.1e6, 12.8e6, 13.2e6, 12.7e6],
    claim_counts=[1840, 1910, 1960, 1820,
                  1730, 840,  1200, 1650,
                  1720, 1780, 1830, 1760,
                  1790, 1850, 1880, 1800,
                  1770, 1820, 1850, 1780],
    external_index=motor_repair_index,
    periods_per_year=4,
)

result = fitter.fit(detect_breaks=True)
print(result.summary())
print(f"Superimposed inflation: {fitter.superimposed_inflation():.2%}")
```

```
Method          : piecewise
Trend rate (pa) : 0.0641
95% CI          : (0.0489, 0.0801)
R-squared       : 0.9102
Changepoints    : [8]
Bootstrap N     : 1000

Superimposed inflation: 0.0284
```

When an external index is provided, the fit is performed on the deflated severity series: total paid divided by the re-based index. So `trend_rate` is the trend in severity after stripping out the index-measured inflation. `superimposed_inflation()` returns the residual: the severity cost growth that the economic index did not capture. In this example the 2.84% superimposed component represents ADAS recalibration costs, labour time complexity, and parts pricing dynamics — the structural drivers that the ONS SPPI basket does not fully weight.

The ONS HPTH series is free, public, and updated monthly. No API key, no commercial licence. Other indices catalogued in the library include L7JE (CPI 12.5.4.1 Motor vehicle insurance), D7DO (CPI 04.3.2 services for maintenance and repair of dwellings — useful for household books), and CZEA (RPI Maintenance of motor vehicles). For BCIS data, which is subscription-only, `ExternalIndex.from_csv()` handles the load.

```python
# For household property books, the ONS building repair index
property_index = ExternalIndex.from_ons('D7DO')

# For BCIS or any other subscription data
bcis_index = ExternalIndex.from_csv(
    'bcis_tender_price.csv',
    date_col='quarter',
    value_col='index',
)
```

---

## Combined loss cost trend

For most pricing purposes you want the combined loss cost trend: `(1 + freq_trend) × (1 + sev_trend) − 1`. `LossCostTrendFitter` does this in one call, running both component fits internally and returning a `LossCostTrendResult` with the decomposition.

```python
from insurance_trend import LossCostTrendFitter, ExternalIndex

motor_repair_index = ExternalIndex.from_ons('HPTH', start_date='2019-01-01')

fitter = LossCostTrendFitter(
    periods=['2019Q1', '2019Q2', '2019Q3', '2019Q4',
             '2020Q1', '2020Q2', '2020Q3', '2020Q4',
             '2021Q1', '2021Q2', '2021Q3', '2021Q4',
             '2022Q1', '2022Q2', '2022Q3', '2022Q4',
             '2023Q1', '2023Q2', '2023Q3', '2023Q4'],
    claim_counts=[1840, 1910, 1960, 1820,
                  1730, 840,  1200, 1650,
                  1720, 1780, 1830, 1760,
                  1790, 1850, 1880, 1800,
                  1770, 1820, 1850, 1780],
    earned_exposure=[18400, 18600, 18800, 18200,
                     18000, 17200, 17800, 18100,
                     18200, 18400, 18500, 18300,
                     18400, 18600, 18700, 18500,
                     18300, 18400, 18500, 18300],
    total_paid=[8.2e6, 8.6e6, 8.9e6, 8.4e6,
                5.8e6, 4.1e6, 7.2e6, 8.0e6,
                8.3e6, 8.9e6, 9.4e6, 9.1e6,
                10.2e6, 11.1e6, 11.8e6, 11.4e6,
                12.1e6, 12.8e6, 13.2e6, 12.7e6],
    external_index=motor_repair_index,
)

result = fitter.fit()
print(result.summary())
print(result.decompose())
```

```
=== Loss Cost Trend Summary ===
Frequency trend (pa)         : -1.83%
Severity trend (pa)          :  6.41%
Combined loss cost trend (pa):  4.49%
Superimposed inflation (pa)  :  2.84%
```

```python
{
    'freq_trend': -0.0183,
    'sev_trend': 0.0641,
    'combined_trend': 0.0449,
    'superimposed': 0.0284,
}
```

The combined trend of 4.49% is not the sum of -1.83% and 6.41%; it is the product. The difference is small here but becomes material at higher trend rates or over longer projection horizons.

The trend factor method on the result converts this to a compound multiplier for any projection horizon:

```python
# Trend factor from 2023Q4 to 2025Q4 — 8 quarters, 2 years
factor = result.trend_factor(8)
print(f"Trend factor (2 years): {factor:.4f}")  # e.g. 1.0917
```

For a portfolio with an average loss cost of £450 at the mid-point of the experience period, a 4.49% annual trend to a future policy mid-point two years out means a projected loss cost of £450 × 1.092 = £491. That is the number that goes into the rate. The confidence interval around 4.49% — roughly ±2.5 points given the component CIs — means the projected loss cost range is roughly £475–£510. The "selected" trend table in a rate filing should reflect that range.

---

## The Ogden problem

The Ogden rate changed from -0.75% to -0.25% in July 2019 (England, Wales, and Northern Ireland). This is a structural break in the severity series for liability-heavy lines: bodily injury claims settled close to the change date had materially different reserve values than those settled six months earlier. If you are fitting severity trend on a liability book that spans the 2019 boundary, the regression slope will absorb some of the level shift as a slope estimate, biasing both the pre-change and post-change trends.

PELT will typically detect this as a structural break if the signal-to-noise ratio is adequate. The level change from the Ogden reform was substantial relative to within-period noise on most liability books. If it does not detect it automatically, pass `changepoints=[2]` (for a 2019Q1-indexed quarterly series) to force the break.

COVID creates the same problem on the frequency side, with greater magnitude. The lockdown quarters (Q2 and Q3 2020 for most UK personal lines) have frequencies so far below the structural trend that including them without a break will pull the post-2021 trend upward, because the regression is trying to fit through the recovery as well as the steady state. Automatic PELT detection handles this correctly in most cases; we tested it on synthetic series with a lockdown-sized frequency dip and it reliably places the break at the dip entry and exit.

The 2022–2023 inflation shock is the current live version of this problem for severity. Average repair costs in UK motor climbed approximately 25–30% in nominal terms between Q1 2021 and Q4 2023. The slope of a single regression through this period will be dominated by the inflation jump. The post-2023 trend, as parts costs stabilise and supply chains normalise, will be lower than the 2021–2023 slope suggests. Whether PELT separates these or not depends on whether the inflation period looks like a level shift (break) or a sustained slope change (the log-linear trend picks it up). Our view: use the ONS index deflation and let the residual superimposed inflation estimate reveal the structural component.

---

## Alternative: UnobservedComponents

For practitioners who prefer a smooth state-space decomposition to a piecewise linear fit, `method='local_linear_trend'` invokes `statsmodels.tsa.statespace.structural.UnobservedComponents` with a local linear trend specification:

```python
result_uc = fitter.fit(method='local_linear_trend', n_bootstrap=200)
print(result_uc.summary())
```

The UnobservedComponents model has a time-varying slope, allowing the trend rate to evolve rather than holding it constant within segments. The reported trend rate is the mean of the smoothed state slope over the last four quarters: essentially a recent-period estimate of where the trend is now. Bootstrap CIs are capped at 200 replicates for this method because fitting a state-space model is substantially more expensive than OLS.

We think `local_linear_trend` is appropriate when the trend genuinely seems to be changing gradually — a slow moderation of frequency decline as market saturation approaches, for example — and `log_linear` with explicit structural breaks is appropriate when the discontinuities are abrupt and actuarially attributable to specific events. In practice: use `log_linear` by default, switch to `local_linear_trend` as a cross-check, and document why the two diverge if they do.

---

## Output and diagnostics

`TrendResult` is a dataclass. All fields are accessible directly:

```python
result = fitter.fit()

result.trend_rate      # float, e.g. -0.0183
result.ci_lower        # float
result.ci_upper        # float
result.method          # 'log_linear', 'piecewise', or 'local_linear_trend'
result.fitted_values   # Polars Series
result.residuals       # Polars Series, multiplicative: actual/fitted - 1
result.changepoints    # list of int
result.projection      # Polars DataFrame: period, point, lower, upper
result.r_squared       # float

result.trend_factor(8) # compound factor over 8 periods
result.plot()          # three-panel matplotlib Figure
result.summary()       # human-readable string
```

The three-panel plot shows actual vs fitted on the original scale, multiplicative residuals, and the forward projection fan with CI bands. The projection compounds from the last fitted value using the per-period trend rate, with CI bounds scaled from the annual bootstrap intervals.

`LossCostTrendResult` wraps two `TrendResult` objects and adds `combined_trend_rate`, `superimposed_inflation`, and the combined projection fan:

```python
lc_result = fitter.fit()
lc_result.frequency    # TrendResult for frequency
lc_result.severity     # TrendResult for severity
lc_result.decompose()  # dict with all four components
lc_result.trend_factor(8)  # compound factor over 8 periods
lc_result.plot()       # multi-panel diagnostic figure
```

The projected loss cost convenience method skips the intermediate result entirely if you just need the forward DataFrame:

```python
projection = fitter.projected_loss_cost(future_periods=8, ci=0.95)
print(projection)
# Polars DataFrame: period (1–8), point, lower, upper
```

---

## What this replaces

A typical Excel-based trend analysis on a quarterly motor book takes a working day to build and is not reproducible — different actuaries will make different choices about which periods to include, whether to log-transform, whether to add seasonal dummies, and whether to acknowledge the COVID break. The choices are implicit in the model structure, not documented.

`insurance-trend` makes those choices explicit and parameterised. The default configuration — log-linear OLS with quarterly seasonal dummies, PELT structural break detection at penalty 3.0, 1,000-replicate bootstrap CI — is defensible and documented. If the peer reviewer wants different assumptions, they change named parameters and rerun. The code is the documentation.

The ONS integration removes the last reason to do this manually. The HPTH series has been public and free since the ONS introduced the SPPI publication. Building it into severity trend analysis is a fifteen-minute task. It has not been standard practice because there was no Python library that made it straightforward. Now there is.

---

`insurance-trend` is open source under the MIT licence at [github.com/burning-cost/insurance-trend](https://github.com/burning-cost/insurance-trend). Install with `uv add insurance-trend`. 154 tests, all passing. Requires Python 3.10+, NumPy, Polars, statsmodels, and ruptures.

- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/)
- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/)
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
