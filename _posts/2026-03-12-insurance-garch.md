---
layout: post
title: "GARCH for Claims Inflation: Modelling Volatility That Clusters"
date: 2026-03-12
categories: [libraries, pricing, uncertainty]
tags: [GARCH, GJR-GARCH, claims-inflation, volatility, PRA, motor, home, time-series, bootstrap, backtesting, python, insurance-garch]
description: "The UK motor and home inflation episode of 2021–2024 exposed a familiar failure: actuaries presenting single-number trend assumptions into a regime where the variance of inflation was itself time-varying. Robert Engle solved this problem in 1982 on UK RPI data. insurance-garch brings GARCH to insurance pricing teams."
---

Every pricing committee in the UK has seen the same slide. A chart with claims inflation on the y-axis, time on the x-axis, and a straight trend line drawn through the last twelve quarters. Maybe a dotted extension into the future. Maybe a label that says "3.8% p.a." Someone in the room asks about uncertainty. The actuary says something like "we think plus or minus a point is reasonable." Nobody writes it down. The committee nods and moves on.

That is not a risk model. That is a point estimate wearing a blazer.

The 2021–2024 UK motor and home claims inflation episode made this embarrassing in real time. Motor severity was running above 20% year-on-year in late 2022. Home claims were at 118% combined ratio in 2023. The volatility was not random noise around a stable trend — it was clustered. Supply chain disruption hit, then energy costs, then used car prices, then repair labour. Each shock amplified the next. Actuaries who had been told to present a single number were presenting single numbers into a regime where the variance of inflation was itself time-varying.

This is precisely the problem ARCH models were built to solve.

Robert Engle won the 2003 Nobel Prize in Economics partly for his 1982 paper introducing ARCH — written on UK inflation data. The original motivation was: quarterly UK retail price inflation in the 1970s exhibited volatility clustering. Calm periods followed calm periods. Volatile periods followed volatile periods. A model that assumed constant variance was systematically mispredicting uncertainty. Forty years later, UK insurance pricing teams are making the same mistake Engle was correcting.

We built [`insurance-garch`](https://github.com/burning-cost/insurance-garch) to fix it.

## What the library does

`insurance-garch` wraps Kevin Sheppard's `arch` package (v7.2 — a production-grade GARCH implementation used in academic finance for a decade) with an insurance-specific workflow. The design principle is simple: a pricing actuary who has used `insurance-trend` should be able to pick this up in an afternoon.

```bash
pip install insurance-garch
```

```python
from insurance_garch import (
    ExposureWeightedSeries,
    ClaimsInflationGARCH,
    VolatilityScenarioGenerator,
    GARCHBacktest,
)

# Convert raw claims + exposure to log-rate series
ews = ExposureWeightedSeries(claims=severity_qtly, exposure=earned_exposure)

# Fit GJR-GARCH(1,1) with Student-t errors (the default)
model = ClaimsInflationGARCH(ews)
result = model.fit()

# Generate 10,000 bootstrap simulation paths, 8 quarters forward
scenarios = VolatilityScenarioGenerator(result, horizon=8).generate(n_sims=10_000)

# Base (p50), stressed (p90), shocked (p99) — Bank of England fan chart
scenarios.fan_chart()
df = scenarios.to_dataframe()  # base / stressed / shocked columns

# Validate the model
bt = GARCHBacktest(result, alpha=0.05).run()
print(bt.summary())
```

The output of `scenarios.to_dataframe()` is something you can hand to a pricing committee: three named paths (base, stressed, shocked) for the next eight quarters, with an honest explanation of what those percentiles mean.

## The five modules

`series.py` handles the first and most annoying problem: you rarely have a clean inflation series to hand. `ExposureWeightedSeries` converts raw claims and exposure into a log-rate series, applying a continuity correction to handle zero-claim periods. `CalendarYearInflationSeries` extracts the calendar-year diagonal from a development triangle — the correct representation of calendar-year inflation effects for reserving applications.

`model.py` is the estimation core. `ClaimsInflationGARCH` supports four volatility specifications — GARCH, GJR-GARCH, EGARCH, TARCH — crossed with three error distributions: normal, Student-t, skewed-t. That is twelve combinations. `GARCHSelector` fits all twelve and ranks by BIC. We use BIC rather than AIC because with 40–120 quarterly observations (typical for UK personal lines), AIC over-penalises; BIC is the right default at these sample sizes.

The default specification is GJR-GARCH with Student-t errors, and that choice is deliberate. GJR-GARCH (Glosten, Jagannathan and Runkle, 1993) adds an asymmetry term: negative shocks to inflation — sudden cost spikes — increase future volatility more than positive shocks of the same size. That is exactly the behaviour we saw in 2021–2022. Motor repair costs shot up faster than they came down. The Student-t handles the fat tails: Ogden rate rulings, PPO awards, and large weather events are not normal-distribution events.

`forecast.py` generates scenarios using bootstrap simulation rather than analytical forecasts. At horizons beyond two or three quarters, analytical GARCH forecasts deteriorate for fat-tailed distributions. Bootstrap resampling from empirical residuals preserves the actual tail behaviour in the historical data.

`backtest.py` implements Kupiec (1995) unconditional coverage and Christoffersen (1998) conditional coverage tests. These are the standard VaR backtesting framework. Kupiec asks: did the right fraction of observations fall outside the VaR? Christoffersen asks: were the exceptions independent of each other, or did they cluster? A model that clusters its breaches has failed at the one thing GARCH models are supposed to prevent.

`report.py` generates a self-contained HTML report with inline figures. No external dependencies, no server required. Attach it to a model documentation pack.

## The PRA angle

The PRA Dear Chief Actuary letter of June 2023 — following an earlier letter in October 2022 — was unusually direct. A quarter of reviewed firms could not provide analysis quantifying their allowance for claims inflation. The PRA explicitly said annual assumption updates may be too infrequent. It required stress and scenario testing covering mild plausible to severe outcomes, and it flagged the need for consistency of inflation assumptions across pricing, reserving, and capital.

Read those requirements against the library: time-varying volatility from GARCH addresses the infrequency problem. `VolatilityScenarioGenerator` with its base/stressed/shocked paths directly produces the scenario range the PRA asked for. `GARCHBacktest` provides auditable historical validation. And shipping a common Python library means pricing, reserving, and capital teams can use the same model.

We are not claiming this library makes you PRA-compliant. Compliance is a matter for your own legal and actuarial sign-off. But the gap between "we think plus or minus a point is reasonable" and a documented GARCH model with backtested VaR is large, and the regulator has noticed.

## Where it fits in the stack

`insurance-garch` complements `insurance-trend` rather than replacing it. `insurance-trend` handles the level and structural breaks in a claims inflation series — finding the trend, identifying regime changes. `insurance-garch` handles the uncertainty envelope around that trend: how wide should the confidence interval be, how does it widen over the forecast horizon, what does the tail look like?

The two libraries can be chained directly. `ExposureWeightedSeries.from_trend_result()` accepts a `SeverityTrendFitter` result from `insurance-trend` and uses the residuals as the GARCH input series. The GARCH is modelling the volatility that the trend model left unexplained.

## Honest caveats

GARCH is not a crystal ball. It is a model of conditional heteroskedasticity in historical data. It has no view on structural breaks not yet in the data — it would not have predicted the COVID supply chain shock from 2019 data alone. The scenarios it generates are conditional on the historical volatility regime continuing; if you believe the UK insurance market has entered a structurally different period, the historical calibration needs revisiting.

The sample size problem is real. Quarterly data for a single line of business from a mid-sized insurer might give you 50–80 observations. GARCH can be fit on that, but the parameter uncertainty is material. Use `GARCHSelector` and look at the BIC spread across specifications — if the best and worst specifications are within 2 BIC points, you do not have strong evidence for your preferred model.

The default bootstrap simulation uses 10,000 paths. On a laptop, that runs in a few seconds for an 8-quarter horizon. For 20-quarter horizons with large datasets, budget more time.

## Getting started

The repository is at [`github.com/burning-cost/insurance-garch`](https://github.com/burning-cost/insurance-garch). The notebook `notebooks/01_garch_motor_inflation_demo.py` runs the full workflow on synthetic UK motor data calibrated to the 2021–2023 inflation episode — it is the fastest way to see whether the library fits your workflow.

92 tests, 0 failures. The architecture decisions — BIC selection, bootstrap forecasting, GJR default — are documented in the repository and in the test suite. If you disagree with a default, every parameter is overridable.

The pricing committee slide with a single trend line is still going to exist. But now you can put the fan chart next to it.

---

*insurance-garch is part of the Burning Cost open-source stack. Earlier libraries in the series: [insurance-trend](https://github.com/burning-cost/insurance-trend), [insurance-copula](https://github.com/burning-cost/insurance-copula), [insurance-conformal](https://github.com/burning-cost/insurance-conformal).*

---

**Related reading:**
- [Trend Selection Is Not Actuarial Judgment: A Python Approach](/2026/03/13/insurance-trend/) — GLM-based trend models for the structural component; GARCH captures the volatility around that trend
- [When Did Your Loss Ratio Actually Change?](/2026/03/13/insurance-changepoint/) — before fitting any trend or volatility model, test whether the series has had a structural break that invalidates the stationarity assumption
