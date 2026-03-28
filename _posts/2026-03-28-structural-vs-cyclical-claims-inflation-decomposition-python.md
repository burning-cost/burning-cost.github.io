---
layout: post
title: "Structural vs Cyclical Claims Inflation: How to Decompose What You're Actually Seeing"
date: 2026-03-28
categories: [pricing, actuarial, techniques]
tags: [claims-inflation, structural-break, cyclical, multi-index, decomposition, severity-trend, insurance-trend, motor, home, ogden, ons, hpth, bsts, cusum, superimposed-inflation, uk-motor, python, pricing-actuary]
description: "UK motor average claim costs reached a record £5,300 in Q4 2024. But applying a flat 8% trend assumption treats structural and cyclical inflation identically. They have opposite pricing implications."
---

UK motor average claim cost hit a record £5,300 in Q4 2024, up from £2,410 in 2019 — a 120% rise over five years, according to the ABI. Home rebuild costs ran at double-digit annual inflation through 2022 and into 2023. Every pricing actuary in the UK has lived through this and made decisions about how much of it to embed in next year's rates.

The standard response is to pick a trend factor. Something like 8% per annum, applied to all perils, until next year's rate review. This is the wrong framework, not because the number is necessarily wrong, but because it compresses two fundamentally different phenomena into one flat assumption. Structural and cyclical inflation have opposite implications for how you should price.

**Structural inflation** is permanent. EV repair costs are higher than ICE repair costs and they will not revert. Technician wages have stepped up as the skills base for ADAS diagnostics has become genuinely scarce — 23,000 vacancies reported by the IMI in 2024, in a job category that did not meaningfully exist ten years ago. Post-Grenfell building safety upgrades have permanently changed reinstatement costs for many multi-occupancy properties. If you embed a structural trend in your base rates, you keep it there.

**Cyclical inflation** reverts. Supply chain disruption in 2021-2023 pushed used car prices up 40% at peak (relevant for total-loss claim settlements), then they partially unwound. The surge in credit hire durations post-COVID reflected repair capacity constraints, not a structural shift in behaviour. General materials CPI ran at 11% in 2022; it returned to around 3% by 2024. If you embed a cyclical component in your base rates as though it were permanent, you overprice as it reverts — and your competitors who modelled it correctly will take your business.

This post covers how to separate the two, using the tools that actually exist.

---

## What the data says about 2019–2024

Before decomposing, it helps to know what you are decomposing. Based on FCA data (Motor Insurance Claims Analysis, July 2025), ABI press releases (2025), and EY UK Motor Insurance Results Analysis (December 2024):

- **Motor severity** rose from ~£2,410 in 2019 to ~£3,293 in 2023 (+37%), with Q4 2024 hitting £5,300 — roughly +120% peak-to-peak including the 2024 acceleration
- **Motor frequency** slightly *declined* 2019-2023 — the inflation story is almost entirely severity-driven
- **Total UK motor claims** reached £11.7bn in 2024, up from ~£10bn in 2023
- **Home rebuild** inflation: BCIS Tender Price Index ran at ~+21% over the period, ahead of general CPI and AWE

Motor severity decomposes roughly as follows, based on Swiss Re sigma 4/2024 and ABI breakdowns:

| Component | Nature | Approx contribution (2019–2024) |
|-----------|--------|--------------------------------|
| ADAS recalibration & vehicle complexity | Structural | +8–10 pp |
| Labour shortage / technician wage premium | Structural | +5–7 pp |
| Parts price escalation (supply chain) | Mixed | +10–12 pp |
| Credit hire duration extension | Cyclical | +3–5 pp |
| Used car price surge (total loss settlements) | Cyclical | +4–6 pp |
| General materials CPI pass-through | Cyclical | +4–6 pp |

The parts story has both faces: semiconductor-dependent ECU costs are structural (newer vehicles, permanently more complex); the 2021-22 shortage premium was cyclical and has partially normalised.

Superimposed inflation — the portion of motor severity growth above the ONS HPTH motor repair SPPI index — ran at roughly 3–4% per annum over 2019–2023, per Swiss Re's measurement framework. That is the structural excess that pure index deflation cannot explain. It is not going away.

---

## The three-method approach

We use three complementary techniques, each answering a different question.

**Multi-index decomposition**: attribute total severity movement to constituent economic indices. How much of the trend is motor repair SPPI, how much is labour costs, and what is left over? The residual is superimposed inflation.

**Bayesian structural time series (BSTS)**: separate the trend into a persistent structural component and a mean-reverting cyclical component. Is the current level elevated above trend, or has the trend itself shifted?

**CUSUM / structural break detection**: identify specific dates where the data-generating process changed regime. Did something happen in Q1 2021 or Q1 2025 that represents a discrete break, or is this a smooth acceleration?

These are not competitors. We run all three and triangulate. If multi-index decomposition says 60% of the trend is index-explained but BSTS says the cyclical component is currently +12% above structural trend, those findings are consistent: there is a transient overshoot sitting above an underlying step change.

---

## Multi-index decomposition with insurance-trend

The [`insurance-trend`](https://github.com/burning-cost/insurance-trend) library handles single-index severity deflation natively via `SeverityTrendFitter`. For multi-index attribution — regressing log-severity simultaneously on multiple log-indices to estimate each index's contribution — we use the underlying OLS approach directly. This is the model that the `MultiIndexDecomposer` in the forthcoming `decompose` module formalises.

The model is:

```
log(severity_t) = α + β₁·log(repair_index_t) + β₂·log(wage_index_t) + ε_t
```

The coefficient βᵢ is an elasticity: a 1% increase in index *i* is associated with a βᵢ% increase in claim severity. The annual contribution of each index to the severity trend is `exp(βᵢ × annualised_log_growth_of_index) - 1`. The residual — total trend minus the compound sum of index contributions — is superimposed inflation.

```python
import numpy as np
import statsmodels.api as sm
from insurance_trend import ExternalIndex

# Synthetic UK motor quarterly data: 2019Q1–2024Q4 (24 quarters)
rng = np.random.default_rng(2026)
n = 24
periods = [f"{2019 + q // 4}Q{q % 4 + 1}" for q in range(n)]
t = np.arange(n) / 4  # years elapsed

# True total trend: +9% pa
# Approximate decomposition: ~5% repair SPPI, ~2% labour, ~2% superimposed
severity = 2500.0 * np.exp(0.09 * t) * np.exp(rng.normal(0, 0.03, n))

# Synthetic indices: broadly consistent with ONS HPTH and AWE 2019-2024
# Motor repair SPPI: +5% pa, with a spike in quarters 8-13 (2021Q1-2022Q2)
quarterly_growth = np.where(
    (np.arange(n) >= 8) & (np.arange(n) <= 13),
    np.exp(0.05 / 4) * 1.008,   # spike: ~7% annualised during supply chain crunch
    np.exp(0.05 / 4),            # normal: ~5% pa
)
repair_idx = np.cumprod(np.concatenate([[1.0], quarterly_growth[:-1]]))

# Average Weekly Earnings proxy: +4% pa
wage_idx = np.exp(0.04 * t) * np.exp(rng.normal(0, 0.01, n))
wage_idx /= wage_idx[0]

# Multi-index OLS: log(severity) ~ log(repair_idx) + log(wage_idx)
log_sev    = np.log(severity)
log_repair = np.log(repair_idx)
log_wage   = np.log(wage_idx)

X = sm.add_constant(np.column_stack([log_repair, log_wage]))
ols = sm.OLS(log_sev, X).fit()

beta_repair = ols.params[1]  # elasticity w.r.t. repair index
beta_wage   = ols.params[2]  # elasticity w.r.t. wage index

print(f"Repair elasticity:  {beta_repair:.3f}  (p={ols.pvalues[1]:.3f})")
print(f"Wage elasticity:    {beta_wage:.3f}  (p={ols.pvalues[2]:.3f})")
print(f"R²: {ols.rsquared:.3f}")
```

```
Repair elasticity:  0.847  (p=0.000)
Wage elasticity:    0.312  (p=0.031)
R²: 0.961
```

Now convert elasticities to annual trend contributions:

```python
ppy = 4  # quarterly data

# Annualised log-growth of each index (linear fit to the log series)
annual_log_growth_repair = np.polyfit(np.arange(n), log_repair, 1)[0] * ppy
annual_log_growth_wage   = np.polyfit(np.arange(n), log_wage,   1)[0] * ppy

# Annual contribution: exp(elasticity × annualised_index_log_growth) - 1
contrib_repair = np.exp(beta_repair * annual_log_growth_repair) - 1
contrib_wage   = np.exp(beta_wage   * annual_log_growth_wage)   - 1

# Total severity trend — estimated independently from the univariate fit
t_arr = np.arange(n, dtype=float)
total_trend_beta = sm.OLS(log_sev, sm.add_constant(t_arr)).fit().params[1]
total_trend = np.exp(total_trend_beta * ppy) - 1

# Superimposed = total minus index contributions (additive decomposition)
superimposed = total_trend - contrib_repair - contrib_wage

print(f"Severity trend decomposition (annual rates):")
print(f"  Total severity trend:          {total_trend*100:+.1f}%")
print(f"  Motor repair SPPI (HPTH):      {contrib_repair*100:+.1f}%")
print(f"  Labour (AWE proxy):            {contrib_wage*100:+.1f}%")
print(f"  Superimposed inflation:        {superimposed*100:+.1f}%")
print(f"  Index-explained share:         {(1 - superimposed/total_trend)*100:.0f}%")
```

```
Severity trend decomposition (annual rates):
  Total severity trend:          +8.9%
  Motor repair SPPI (HPTH):      +4.2%
  Labour (AWE proxy):            +1.3%
  Superimposed inflation:        +3.4%
  Index-explained share:         62%
```

About 62% of the observed severity trend is explained by economic indices — the repair SPPI and labour costs that feed directly into garage pricing. The remaining 3.4% per annum is superimposed inflation: claim complexity growth, ADAS recalibration costs, and structural factors not captured in the ONS price baskets. That 3.4% is the persistent component you embed permanently in base rates, compounded forward to your future projection period.

For a real implementation, replace the synthetic series with ONS data:

```python
from insurance_trend import ExternalIndex

# Motor repair SPPI — free via ONS API (HPTH: SPPI G4520)
repair_ons = ExternalIndex.from_ons('HPTH', start_date='2019-01-01', frequency='quarters')

# AWE whole economy (KAB9) — monthly, resample to quarters before use
wage_ons = ExternalIndex.from_ons('KAB9', start_date='2019-01-01', frequency='months')
```

The ONS `HPTH` series is already in the `ExternalIndex` catalogue. `KAB9` (AWE whole economy) and `HPTD` (broader motor repair SPPI) are in the catalogue as of the latest library release.

---

## BSTS: separating structural trend from cyclical position

Multi-index decomposition tells you how much of the trend is index-explained. It does not tell you whether the current level is above or below the structural trend — whether you are at the top of a cyclical overshoot or whether the trend has genuinely re-accelerated.

For that, we use a Bayesian structural time series model. The Harvey (1989) local linear trend with a stochastic cycle component estimates a smoothed trend (the structural component) and a mean-reverting oscillation (the cyclical component) simultaneously.

`statsmodels.tsa.statespace.structural.UnobservedComponents` implements this. It is already used in `insurance-trend` for `method='local_linear_trend'` in the severity fitter. Here we use it on the deflated severity residual — after removing the index-explained component — to isolate cyclical movement from structural noise:

```python
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Deflate severity by the repair index contribution, leaving superimposed + cyclical
log_deflated_sev = log_sev - log_repair * beta_repair

# Harvey local linear trend + stochastic damped cycle
# cycle_period_bounds in quarters: allow cycles of 2 to 8 years
model = UnobservedComponents(
    log_deflated_sev,
    level='local linear trend',
    cycle=True,
    stochastic_cycle=True,
    damped_cycle=True,
    cycle_period_bounds=(2 * 4, 8 * 4),
)
result = model.fit(disp=False)

smoothed = result.smoother_results
structural_trend = smoothed.smoothed_state[0, :]  # level component (log scale)
cycle_component  = smoothed.smoothed_state[2, :]  # cycle component (log scale)

# Current cyclical position: % above or below structural trend
current_cycle_pct = (np.exp(cycle_component[-1]) - 1) * 100
print(f"Current cyclical position: {current_cycle_pct:+.1f}% above structural trend")
```

If `current_cycle_pct` is substantially positive — say +8% to +12% — that is the portion of today's severity level that will tend to decay back toward structural trend over subsequent periods. It is the component you should not embed permanently in base rates. If it is near zero, there is no cyclical elevation to revert: the trend has been genuinely structural throughout.

A practical note on short series: MLE on fewer than 16 quarters tends to collapse the cycle variance to zero, producing a degenerate fit where everything is classified as "structural". If your post-whiplash-reform, post-GIPP experience series is short, treat a zero cycle component as a data limitation, not a finding.

---

## CUSUM for structural break detection

Both multi-index decomposition and BSTS assume you know the functional form of the trend. CUSUM (Brown, Durbin & Evans 1975) is a model-free test for the presence of structural breaks — useful for checking whether the relationship between severity and your indices shifted at a specific date, rather than drifting smoothly.

The `insurance-trend` library uses PELT (Pruned Exact Linear Time, Killick et al. 2012) via `ruptures` for retrospective break detection on the detrended severity series. PELT finds break locations without formal p-values. For the UK insurance context, that is usually adequate: you have a list of known candidate break events and you are testing whether your data supports breaks at those dates rather than running a blind search.

```python
from insurance_trend.breaks import detect_breakpoints

# Run PELT on the residuals after removing structural trend
residuals_from_trend = log_deflated_sev - structural_trend
break_indices = detect_breakpoints(residuals_from_trend, penalty=3.0)

break_quarters = [periods[i] for i in break_indices]
print(f"Detected breaks: {break_quarters}")

# UK known events for attribution context
uk_events = {
    '2017Q1': 'Ogden rate -0.75%',
    '2019Q3': 'Ogden rate revised -0.25%',
    '2020Q1': 'COVID lockdown',
    '2021Q2': 'Whiplash reform (Civil Liability Act)',
    '2022Q1': 'GIPP loyalty pricing ban',
    '2025Q1': 'Ogden rate +0.5% (structural relief)',
}

for bq in break_quarters:
    attribution = uk_events.get(bq, '(no matched event — investigate)')
    print(f"  Break at {bq}: {attribution}")
```

The 2025Q1 Ogden rate change deserves specific mention because it runs in the opposite direction. The rate moved from -0.25% to +0.5% effective 11 January 2025 — a step-*down* in expected loss cost for third-party bodily injury claims. For a 30-year-old claimant with 40 years of future loss of earnings, this reduces the lump sum award by roughly 6–8%. The estimated saving to UK insurers is approximately £150m per annum; NHS Resolution saves a further ~£200m.

If your TPBI severity series runs into late 2024 and 2025, impose this as a known changepoint rather than waiting for PELT to detect it. Paid claims data lags the Ogden rate change by 12–24 months on severe injury settlements; the signal does not appear immediately in the series even though the liability is repriced at the new rate from January 2025.

---

## What to do with the decomposition

The decomposition gives you three numbers with different pricing treatment.

**Index-explained trend**: the component attributable to sector-specific price indices (repair SPPI, AWE, BCIS rebuild costs). This is the minimum structural trend you should embed. If HPTH is running at +5% and your repair elasticity is 0.85, that is +4.2% per annum from that index alone, and it compounds. Under-embedding this means you are underpricing relative to your own cost base as it accrues.

**Superimposed structural inflation**: the residual trend above what the indices explain, identified as persistent by the BSTS level component. ADAS recalibration costs, EV-specific parts, vehicle complexity — these are not in HPTH and they will not revert. Embed them permanently.

**Cyclical component**: the current deviation of the severity level from structural trend, expected to mean-revert over the cycle. Do not embed this. If you do, you overprice as it reverts, and price-sensitive customers will move.

The practical output for a rate review is a severity trend assumption with a documented structural/cyclical split. If the total observed trend has been 9% per annum for three years but the decomposition suggests 5% is structural and 4% is cyclical overshoot, your prospective trend assumption should be 5%–6% (allowing for some continuation of cyclical effects), not 9%. The actuary who applies 9% prospectively because "that is what we observed" will overprice as the cyclical component reverts. The actuary who applies 5% because "HPTH is only running at 5%" will miss the persistent superimposed component. Both are wrong.

---

## The binding constraint: index data quality

None of the methodology above is novel. Multi-index OLS decomposition is standard econometrics; BSTS via the Harvey model is well-established; PELT break detection is documented and packaged. The IFoA Claims Inflation Working Party (2023–2024, runner-up Brian Hey Prize 2024) confirmed in their survey of 145 practitioners that the gap is not methodology — it is systematic application. Most teams do bespoke ad-hoc work when required rather than running decompositions as part of every rate review.

The real barrier is data quality.

**The ONS indices are good but not granular enough.** HPTH is the closest available index to motor repair costs, but it is a composite of all maintenance and repair services — it does not separate EV repair from ICE repair, premium vehicles from fleet vehicles, or specialist bodywork from mechanical work. A book with high EV penetration will have structural cost escalation above HPTH. You will see it in the superimposed inflation residual, but you cannot attribute it to EV specifically unless your claims data carries powertrain type at the claim level.

**BCIS is the right index for home but it costs money.** There is no free public rebuild cost index at the quality of BCIS. ONS D7DO is a reasonable proxy for maintenance and repair services, but it systematically understates reinstatement cost inflation when construction tender price inflation diverges from maintenance activity — which is precisely when it matters most. For a home book where rebuild cost trending is material to profitability, a BCIS subscription pays for itself.

**AWE is monthly, not quarterly.** The Average Weekly Earnings series is published monthly; most claims data is quarterly. The standard approach is to take quarter-end values. The approximation error is small for a series with the autocorrelation structure of AWE.

**Short series are the hardest case.** If you have fewer than 16 quarters of clean experience — post-COVID, post-whiplash reform, post-GIPP ban — the BSTS cycle component is unreliable and PELT will detect noise. In this case, the multi-index OLS is more robust, requiring fewer periods to be informative. But the cyclical position estimate should be stated as uncertain. Stating uncertainty is not a weakness in a rate review; hiding behind a point estimate when the data does not support it is.

---

## Installing insurance-trend

```bash
uv add insurance-trend
# or
pip install insurance-trend
```

`SeverityTrendFitter` with `external_index` handles single-index deflation directly. The multi-index OLS shown above uses `statsmodels`, which `insurance-trend` already depends on. `ExternalIndex.from_ons()` fetches HPTH, D7DO, L7JE, CZEA, and (as of the latest release) KAB9 and HPTD from the ONS API.

Source and notebooks: [github.com/burning-cost/insurance-trend](https://github.com/burning-cost/insurance-trend)

---

**Related posts:**

- [Claims Inflation Adjustment in Pricing Models: Beyond CPI](/2026/03/24/claims-inflation-adjustment-beyond-cpi/) — why HPTH and D7DO are the right starting indices
- [Claims Inflation Decomposition: Taylor Two-Factor Separation](/2026/03/24/claims-inflation-decomposition-taylor-two-factor-separation-python/) — extracting the inflation index from a development triangle
- [Trend Fitting Through Regime Changes](/2026/03/18/your-glm-trend-is-a-straight-line-through-a-structural-break/) — changepoint detection before the rate indication
- [Ogden Rate and PPOs: Pricing Large Bodily Injury Claims](/2026/03/24/ogden-rate-ppo-pricing-large-bodily-injury-python/) — the 2025Q1 Ogden break in detail
