---
layout: post
title: "Extreme Value Theory for UK Motor Large Loss Pricing"
date: 2026-03-13
categories: [libraries, pricing, severity, reinsurance]
tags: [EVT, extreme-value-theory, GPD, GEV, threshold-selection, censored-MLE, profile-likelihood, return-levels, XL-pricing, reinsurance, Solvency-II, TPBI, motor, subsidence, flood, insurance-evt, python]
description: "EVT for UK motor large loss pricing: censored GPD for open TPBI claims, profile likelihood CIs, excess layer pure premiums. insurance-evt Python library."
---

Large bodily injury claims do not behave like ordinary claims. A TPBI settlement for a catastrophic spinal injury can take fifteen years and run to tens of millions. The claim sits open on the register, its ultimate development unknown, while the actuary is expected to estimate the 1-in-200 year return level for the Solvency II internal model. Standard severity distributions fail at both ends of this problem: they cannot handle the censoring and they have no principled basis for extrapolation to return periods far outside the observed data.

Extreme Value Theory exists precisely for this. The Pickands-Balkema-de Haan theorem gives us a rigorous basis for modelling exceedances above a high threshold: under mild regularity conditions, the excess distribution converges to a Generalised Pareto Distribution (GPD) as the threshold rises. This is not an empirical observation but a mathematical result, which is why EVT is the right framework for large loss pricing, capital modelling, and reinsurance layer pricing.

The Python ecosystem has not caught up. `pyextremes` is the most complete existing library, but it requires a `DatetimeIndex` — it was built for time series of environmental extremes. Insurance claim severity data is not a time series: it is a cross-sectional array of settled amounts with an open/closed flag. `scipy.stats.genpareto` has the distribution but none of the workflow. R users have had `ReIns` (with censored MLE and ExcessGPD) since 2017. Python actuarial teams have had nothing.

[`insurance-evt`](https://github.com/burning-cost/insurance-evt) fills that gap.

```bash
uv add insurance-evt
```

---

## The threshold problem

Everything in GPD fitting depends on the threshold. Set it too low and the GPD approximation is not yet valid — you get biased parameter estimates. Set it too high and you have too few exceedances to estimate anything precisely. There is no clean analytical solution; the standard diagnostic workflow is three plots examined in sequence.

The Hill estimator comes first. It estimates 1/xi (the tail index) using only the top-k order statistics, and gives you a quick read on whether the tail is Pareto-type at all. A stable plateau at positive values confirms a heavy tail; a plot that trends steadily downward as k increases suggests either a lighter-tailed distribution or contamination.

```python
import numpy as np
from insurance_evt import ThresholdSelector

# claims: array of settled large loss amounts above a reporting threshold
# For UK motor TPBI, this might be claims above £500k
sel = ThresholdSelector(claims)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sel.hill_plot(ax=axes[0], bootstrap_ci=True)
sel.mrl_plot(ax=axes[1])
sel.parameter_stability_plot(ax=axes[2])
```

The Mean Residual Life (MRL) plot is the workhorse. Under the GPD approximation, `E[X - u | X > u]` should be linear in `u` above the true threshold. You look for the lowest threshold where the plot becomes convincingly linear — that is your starting point. The parameter stability plot then shows whether xi and the modified scale `sigma* = sigma - xi*u` are stable across a range of thresholds. Stability is a necessary (not sufficient) condition.

`auto_threshold()` is available but it is a sanity check, not a verdict:

```python
u_suggested = sel.auto_threshold()  # KS goodness-of-fit based suggestion
# Always examine the plots above before trusting this
```

The TPBI preset gives you context for where the parameters should land:

```python
from insurance_evt import TPBI

print(TPBI.threshold_percentile)   # 95.0
print(TPBI.typical_xi_range)       # (0.30, 0.55)
print(TPBI.notes)
# "Right-censoring is significant: large TPBI claims settle over 5-15 years..."
```

If your fitted xi falls well outside `(0.30, 0.55)` for UK motor TPBI, that is a signal to re-examine your data and threshold, not to accept the estimate. The preset encodes what the market has seen empirically; your data may differ, but large departures deserve scrutiny.

---

## Censored MLE for open claims

This is the feature that motivated the library. UK motor TPBI claims above £500k typically have 15-30% open at any point in time. These are not missing data — they are lower bounds on the ultimate settlement. Treating them as settled claims biases xi upward, sometimes substantially. Poudyal and Brazauskas (2023) showed the magnitude of this bias and derived the corrected censored likelihood.

The fix is straightforward: open claims contribute `log(1 - F(x))` to the likelihood (the log-survival function at the current reported amount) rather than `log f(x)` (the log-density). The parameter estimates then reflect that the claim will ultimately exceed the current amount.

```python
from insurance_evt import GPDFitter

# open_flag: boolean array, True = claim is still open/unsettled
fitter = GPDFitter(
    claims,
    threshold=500_000,
    right_censoring=open_flag,   # handles open claims correctly
)
result = fitter.fit()

print(result)
# GPDFitResult(xi=0.4123 ± 0.0934, sigma=£312k ± £41k,
#              n=94, n_censored=22, threshold=£500k)
```

The `n_censored` field tells you how many claims triggered the censored likelihood path. If that number is large relative to `n` — say, above 10% — the censoring correction is material and the difference from naive MLE is worth documenting.

For reinsurance teams who only see claims above their attachment point (a left-truncation problem rather than a right-censoring one):

```python
fitter = GPDFitter(
    claims,
    threshold=500_000,
    left_truncation=2_000_000,  # reinsurer only sees claims above £2m
)
result = fitter.fit()
```

The likelihood is conditioned on the claim exceeding the attachment point, giving unbiased parameter estimates from the reinsurer's truncated view.

---

## Profile likelihood confidence intervals

Standard practice is to report xi with a symmetric Wald confidence interval: `xi ± 1.96 * xi_se`. This is wrong for tail parameters, and wrong in a systematic direction. The log-likelihood for xi is right-skewed — there is more uncertainty about how heavy the tail could be than about how light it could be. Wald intervals miss this asymmetry entirely.

The profile likelihood CI is the correct approach (Coles, 2001; McNeil, Frey, Embrechts, 2005). You fix xi at a candidate value, maximise the likelihood over sigma, and find the set of xi values where the profile log-likelihood exceeds `L_max - chi2(1, 0.95)/2 = L_max - 1.92`. The upper bound of a profile likelihood CI is typically wider than the Wald upper bound. For capital work, that is the number you should be using.

```python
xi_lo, xi_hi = fitter.profile_likelihood_ci("xi", alpha=0.05)
print(f"xi: {result.xi:.3f} (95% profile CI: {xi_lo:.3f}, {xi_hi:.3f})")
# xi: 0.412 (95% profile CI: 0.218, 0.681)
# vs Wald: 0.412 ± 0.183 => (0.229, 0.595) — upper bound is 13% too low
```

The asymmetry matters most at small samples (below 50 exceedances), which is exactly when it tends to come up in practice. Large loss registers above £500k do not typically have thousands of events.

---

## Return levels and the 1-in-200

Given a GPD fit and an annual arrival rate for exceedances, return levels follow directly:

```
x_T = u + (sigma / xi) * [(lambda * T)^xi - 1]
```

where `lambda` is the annual rate of claims above the threshold and `T` is the return period in years.

```python
from insurance_evt import ReturnLevelCalculator

# lambda = exceedances per year, estimated from the claims data
n_years = 10
lambda_annual = result.n_exceedances / n_years  # e.g. 9.4 per year

calc = ReturnLevelCalculator(result, lambda_annual=lambda_annual)

rl_200 = calc.return_level(200)
ci_lo, ci_hi = calc.return_level_ci(200, method="profile")

print(f"1-in-200: £{rl_200/1e6:.1f}m (95% CI: £{ci_lo/1e6:.1f}m to £{ci_hi/1e6:.1f}m)")
# 1-in-200: £8.7m (95% CI: £4.1m to £24.3m)
```

The CI asymmetry on the return level is far more pronounced than on xi itself. With xi=0.41 and a 200-year return period, the upper CI is typically 2.5 to 3 times the lower CI. Capital figures based on symmetric return level intervals are understating the uncertainty. Solvency II internal model teams should use the upper profile likelihood bound as the conservative estimate for SCR sensitivity testing.

The full Solvency II report wraps this into a structured output suitable for model documentation:

```python
from insurance_evt import SolvencyIIReport

report = SolvencyIIReport(calc).generate()

print(report["interpretation"])
# "The 1-in-200 year claim severity is estimated at £8.7m (95% CI: £4.1m to £24.3m,
#  profile likelihood). Fitted GPD: xi = 0.412 (shape), sigma = £312k (scale),
#  threshold = £500k, based on 94 exceedances at annual rate 9.40/year."

for w in report["warnings"]:
    print(w)
# "Censoring: 22 of 94 exceedances (23%) are right-censored (open claims).
#  Censored MLE applied. Final development of open claims could move xi."
# "Heavy tail: xi = 0.412 (> 0.5 not triggered). ..."
# "Stationarity assumed: standard EVT requires iid claims. ..."
```

The warnings are generated automatically based on sample size, censoring rate, xi magnitude, and a blanket stationarity caveat. For weather-driven perils (flood, subsidence), the stationarity assumption deserves particular attention: climate trends mean the 1976 or 1995 events are poor analogues for the future, and standard EVT extrapolation assumes they are.

---

## Excess layer pricing

This is the second feature that does not exist elsewhere in Python. The ExcessGPD formula computes the annual expected cost of an XL reinsurance layer directly from the fitted GPD, using the closed-form integral of the survival function:

```
layer PP (M xs M+L) = lambda * integral_{M-u}^{M-u+L} [1 - F_GPD(y)] dy
```

For `xi != 0`, the integral has a closed form in terms of the GPD survival function. For `xi = 0` (exponential limit), it reduces to `lambda * sigma * [exp(-a/sigma) - exp(-b/sigma)]`.

R users have had this via the `ReIns` package (`ExcessGPD` function, Reynkens et al., 2017) for years. This is the first Python implementation.

```python
from insurance_evt import ExcessLayerCalculator

xl = ExcessLayerCalculator(result, lambda_annual=lambda_annual)

# Pricing a single layer: £1m xs £1m (covers claims between £1m and £2m above threshold)
pp = xl.layer_pure_premium(retention=1_000_000, limit=1_000_000)
print(f"Layer PP: £{pp:,.0f}/year")

# Layer table across a tower of covers
layer_df = xl.layer_table(
    retentions=[500_000, 1_000_000, 2_000_000, 5_000_000],
    limits=[500_000, 1_000_000, 2_000_000, None],   # None = unlimited top layer
)
print(layer_df.to_string(index=False))
#  retention     limit  mean_excess  pure_premium  rate_on_line
#    500,000   500,000      758,341       312,440         0.625
#  1,000,000 1,000,000    1,099,234       187,223         0.187
#  2,000,000 2,000,000    1,891,452        84,112         0.042
#  5,000,000       inf    4,341,117        23,887           NaN
```

The `rate_on_line` column (pure premium / limit) is in the format reinsurance underwriters work with. For unlimited covers it is not defined, hence `NaN`.

A practical note: the formula requires `xi < 1`. If the profile likelihood upper bound for xi exceeds 1, the mean excess is theoretically infinite and the layer pricing formula breaks down. In that situation, cap the layer and use the profile CI upper bound as the sensitivity. An uncapped layer priced against a distribution with xi > 1 is mispriced by construction.

---

## GEV for subsidence

Not all perils suit the threshold excess approach. UK domestic subsidence has a strong clustering structure: events in 1976, 1995, 2003, 2018, and 2022 drove the majority of large claims. The excess-over-threshold approach assumes approximately independent observations, which is not true when all 300 large claims in a year trace back to the same hot/dry summer. Annual block maxima are a more defensible unit of analysis.

The GEV (Generalised Extreme Value) distribution is the asymptotically correct model for block maxima, analogous to how GPD is the right model for threshold exceedances. `insurance-evt` fits GEV with multi-start optimisation (the GEV likelihood surface has more local optima than GPD) and includes a domain of attraction test:

```python
from insurance_evt import GEVFitter
from insurance_evt import SUBSIDENCE

print(SUBSIDENCE.preferred_method)  # 'gev'
print(SUBSIDENCE.typical_xi_range)  # (0.05, 0.35)

# annual_maxima: array of worst claim (or aggregate) per year
gev_fitter = GEVFitter(annual_maxima)
gev_result = gev_fitter.fit()

print(gev_result)
# GEVFitResult(xi=0.189 ± 0.071, mu=£145k, sigma=£89k, n=28)
```

The domain of attraction test checks whether the data is consistent with Frechet (xi > 0, heavy tail), Gumbel (xi = 0, light tail), or Weibull (xi < 0, bounded tail) domains. Subsidence typically falls in the Frechet domain due to the occasional extreme summer, though the tail is lighter than TPBI because damage is bounded by the structural value of the property.

---

## Where it fits in the toolkit

`insurance-evt` sits at the extreme end of the severity distribution. It models the part of the distribution that determines capital requirements and reinsurance costs. For the bulk of the severity distribution — the part that drives the average claim and GLM pricing factors — the better tools are [`insurance-severity`](https://github.com/burning-cost/insurance-severity) (spliced body/tail with covariate-dependent thresholds) and [`insurance-ilf`](https://github.com/burning-cost/insurance-ilf) (increased limits factors with bootstrap confidence intervals).

The typical large loss workflow we would use:

1. Fit the composite distribution to all claims to get the body correctly specified
2. Use `ThresholdSelector` to identify the GPD attachment point for the tail
3. Fit `GPDFitter` with censoring if the open claim rate is above 10%
4. Use `ReturnLevelCalculator` with profile likelihood CIs for Solvency II internal model input
5. Use `ExcessLayerCalculator.layer_table()` for reinsurance structure pricing

The key thing the library does not do: non-stationary EVT. For flood and subsidence, climate trends mean the iid assumption is increasingly untenable. Every `SolvencyIIReport` output includes a stationarity warning as a result. A non-stationary GEV interface is on the roadmap for v0.2. Until then, if you are fitting flood claims that span the pre/post Flood Re structural break (2016), fit the two periods separately.

---

**[insurance-evt on GitHub](https://github.com/burning-cost/insurance-evt)** — MIT-licensed, PyPI. 2,744 lines, 144 tests, 9 modules.

---

## See Also

- **[insurance-severity](https://burning-cost.github.io/2026/03/13/insurance-composite/)** — Composite severity regression for the full distribution, with covariate-dependent thresholds and TVaR
- **[insurance-ilf](https://github.com/burning-cost/insurance-ilf)** — Increased limits factors from severity distributions with bootstrap CIs; use alongside EVT for excess layer rating factor benchmarks
- **[insurance-quantile](https://burning-cost.github.io/2026/03/07/insurance-quantile/)** — Quantile GBMs for tail risk, when you want covariate-dependent tail modelling rather than unconditional EVT
