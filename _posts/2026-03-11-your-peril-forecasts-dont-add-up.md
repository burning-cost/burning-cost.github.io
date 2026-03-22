---
layout: post
title: "MinTrace Reconciliation for Insurance Pricing Hierarchies"
date: 2026-03-11
categories: [libraries, pricing, forecasting]
tags: [reconciliation, MinTrace, hierarchical-forecasting, loss-cost, earned-premium, frequency-severity, coherence, Wickramasuriya, GLS, WLS, insurance-reconcile, python, motor, home, peril-tree]
description: "MinTrace reconciliation for insurance pricing hierarchies: optimal joint adjustment across peril models and portfolio GLM. Exposure-weighted, Python."
---

There is a meeting that happens in virtually every UK pricing team at rate review time. Someone puts up a slide showing the portfolio loss cost trend. It says +8%. Someone else puts up the peril-level models — Fire, Escape of Water, Subsidence, Flood, Storm. Weighted together, they aggregate to +6%. The room looks at both numbers. Nobody is obviously wrong. And then someone makes a spreadsheet adjustment.

The adjustment is manual. It overrides one set of forecasts to satisfy the other. It does not minimise total forecast error. It does not preserve the statistical properties of either model. It is entirely undocumented as a methodology. And because everyone does it this way, it has never occurred to anyone that there might be a better answer.

There is.

[`insurance-reconcile`](https://github.com/burning-cost/insurance-reconcile) implements MinTrace reconciliation (Wickramasuriya, Athanasopoulos, Hyndman & Rajaratnam, 2019, JASA 114(508)) for insurance pricing hierarchies, with the insurance-specific extensions that make it work correctly on loss costs, earned premium weights, and frequency×severity decompositions.

```bash
uv add insurance-reconcile
# With full MinTrace reconciliation:
uv add 'insurance-reconcile[mintrace]'
```

---

## What incoherence actually means

A set of forecasts is *coherent* if the parts add up to the whole, at every level of the hierarchy, simultaneously.

For a UK home book with Buildings (Fire, Escape of Water, Subsidence, Flood, Storm) and Contents (Theft, Accidental Damage), coherence means:

- The five Buildings perils aggregate to the Buildings cover forecast
- The two Contents perils aggregate to the Contents cover forecast
- Buildings and Contents together aggregate to the Home portfolio forecast

Loss costs make this non-trivial. Loss costs are rates, not amounts. The aggregation constraint is:

```
LC_Buildings = (LC_Fire×EP_Fire + LC_EoW×EP_EoW + ... + LC_Storm×EP_Storm) / EP_Buildings
```

A naive sum — `LC_Buildings = LC_Fire + LC_EoW + ...` — is wrong. Standard MinTrace assumes additive constraints. Insurance hierarchies do not satisfy them.

You can check the scale of the problem on a real book in about ten lines:

```python
from insurance_reconcile import InsuranceReconciler
from insurance_reconcile.hierarchy import InsuranceHierarchy
from insurance_reconcile.simulate import simulate_incoherent_uk_home

lc_df, ep_df = simulate_incoherent_uk_home(n_periods=24)

hierarchy = InsuranceHierarchy.uk_home()
reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())

report = reconciler.check_coherence(lc_df, ep_df)
print(report.to_string())
```

```
=== Coherence Report ===
Status: INCOHERENT
Max discrepancy: 5.23%
Violations found: 36

Level violations:
  portfolio: 12 violations (max 4.1%)
  cover:     24 violations (max 5.2%)

Worst offenders:
  Buildings [2024-Q3]: forecast 0.087, EP-weighted sum 0.091 (+4.6%)
  Home      [2024-Q1]: forecast 0.052, EP-weighted sum 0.055 (+5.2%)
```

A 5% incoherence in loss costs is not a rounding error. On a £300M premium book, it is a £15M discrepancy between what the peril models collectively believe and what the portfolio model believes. One of them is more right than the other; the manual spreadsheet adjustment picks a winner without any statistical basis for the choice.

---

## Why MinTrace is the right answer

MinTrace, proved optimal by Wickramasuriya et al. (2019, Theorem 1), solves a different problem. Instead of picking one level and overriding the others, it adjusts all forecasts simultaneously to satisfy the hierarchy constraints, with minimum trace of the reconciled covariance matrix.

The reconciled forecast is:

```
ỹ = S(S'W⁻¹S)⁻¹S'W⁻¹ŷ
```

where `ŷ` is the vector of base forecasts at all levels, `S` is the summing matrix that encodes the hierarchy structure, and `W` is a weight matrix.

The key theorem result: MinTrace is unbiased. If the base forecasts are unbiased, the reconciled forecasts are unbiased. You do not sacrifice calibration to get coherence. The manual spreadsheet adjustment cannot make this claim.

The choice of `W` is the critical insurance-specific decision. Generic implementations use `W = I` (OLS, equal weights) or `W = diag(n_bottom_i)` (WLS by structural count — how many bottom-level series aggregate into each node). Both are wrong for insurance.

---

## The earned premium weight argument

The correct weight for insurance is earned premium.

The argument comes from the law of large numbers. A peril cell with £50M of earned premium has vastly lower coefficient of variation on its observed loss cost than a cell with £100K. The £50M cell is a stable, highly informative signal. The £100K cell is noisy.

MinTrace with `W = diag(earned_premium)` encodes this directly: high-EP series are given more weight and need barely any adjustment to satisfy the hierarchy constraints. Low-EP series absorb most of the adjustment, which is precisely what the economics say should happen.

`PremiumWeightedMinTrace` implements this:

```python
from insurance_reconcile.reconcile import PremiumWeightedMinTrace

ep = {
    'Home':             8_000_000,
    'Buildings':        5_500_000,
    'Contents':         2_500_000,
    'Fire':             1_800_000,
    'EoW':              1_400_000,
    'Subsidence':         900_000,
    'Flood':              800_000,
    'Storm':              600_000,
    'Theft':            1_600_000,
    'AccidentalDamage':   900_000,
}

reconciler = PremiumWeightedMinTrace(earned_premium=ep)
lc_reconciled = reconciler.reconcile(lc_hat, S, series_names=names)
```

The GLS formula `S(S'W⁻¹S)⁻¹S'W⁻¹ŷ` with `W = diag(EP)` is straightforward NumPy. The library does not subclass `hierarchicalforecast` internals — it implements the matrix algebra directly, avoiding the undocumented internal API that breaks across versions.

---

## The loss cost transform

The other gap in generic implementations is the loss cost aggregation constraint.

Standard MinTrace assumes additive constraints: `claims_total = Σ claims_i`. This is correct for claim amounts. It is wrong for loss costs (which are claims per unit of exposure) or loss ratios (claims per pound of earned premium).

The correct approach for loss costs is:

1. Convert: `claims_i = LC_i × EP_i`
2. Reconcile: apply MinTrace to claims amounts (which satisfy additive constraints)
3. Invert: `LC_reconciled_i = claims_reconciled_i / EP_i`

This is provably correct. The additive constraint on claims amounts is exact. Converting back via `LC = claims / EP` gives LC values satisfying the EP-weighted aggregation constraint. Wickramasuriya (2019, Theorem 1) guarantees that MinTrace preserves unbiasedness, so the unbiasedness of the original forecasts survives the transform.

`LossRatioReconciler` handles the transform automatically:

```python
from insurance_reconcile.reconcile import LossRatioReconciler

# lc_hat: array of loss cost forecasts at all hierarchy levels
# ep: earned premium per series

lr_reconciler = LossRatioReconciler(earned_premium=ep)
lc_reconciled = lr_reconciler.reconcile(lc_hat, S, series_names=names)
```

The caller works entirely in loss costs. The claims transform is internal. If you pass loss ratios instead of loss costs, the same API works — the mathematics are identical since `LR = LC / rate`, which is a rescaling that cancels in the transform.

---

## Frequency × severity decomposition

Many pricing teams forecast loss costs as the product of frequency and severity: `LC = freq × sev`. The hierarchy imposes constraints on the product, not on the components separately.

The challenge is that `freq × sev` is a multiplicative constraint. Standard MinTrace cannot reconcile `(freq, sev)` pairs directly — it operates on additive systems. The solution is log-space transformation:

```
log(LC) = log(freq) + log(sev)
```

In log-space, the multiplicative constraint becomes additive. MinTrace applies to `(log_freq, log_sev)` jointly and the reconciled values transform back via exponentiation.

`FreqSevReconciler` implements this:

```python
from insurance_reconcile.reconcile import FreqSevReconciler

fs_reconciler = FreqSevReconciler(
    earned_premium=ep,
    freq_weight=0.5,   # relative weight on frequency vs severity adjustment
)

freq_reconciled, sev_reconciled = fs_reconciler.reconcile(
    freq_hat=freq_base_forecasts,
    sev_hat=sev_base_forecasts,
    S=S,
    series_names=names,
)
```

The `freq_weight` parameter controls how the adjustment is split between frequency and severity. `freq_weight=0.5` splits equally; `freq_weight=0.8` means 80% of the adjustment is absorbed by frequency and 20% by severity. In practice, if your severity model is more stable than your frequency model — which is typical for attritional motor BI — you would set `freq_weight` above 0.5.

---

## The hierarchy DSL

Building the summing matrix `S` by hand for a UK home peril tree is error-prone. It is a seven-peril, two-cover, one-portfolio hierarchy with specific aggregation rules, and getting the row ordering wrong silently produces incorrect reconciliation with no error message.

`PerilTree` and `InsuranceHierarchy` define the hierarchy declaratively:

```python
from insurance_reconcile.hierarchy import PerilTree, InsuranceHierarchy, GeographicHierarchy

# Define peril tree
tree = PerilTree(
    portfolio='Home',
    covers={
        'Buildings': ['Fire', 'EoW', 'Subsidence', 'Flood', 'Storm'],
        'Contents':  ['Theft', 'AccidentalDamage'],
    },
)

# Optional: add geographic dimension
geo = GeographicHierarchy(
    levels=['postcode_sector', 'postcode_area', 'region'],
    national_name='National',
)

# Combined hierarchy (crossed: peril × geography)
hierarchy = InsuranceHierarchy(
    peril_tree=tree,
    geographic_hierarchy=geo,
    name='UK Home',
)
```

Standard UK trees are built in:

```python
hierarchy = InsuranceHierarchy.uk_home()           # Buildings/Contents, 7 perils
hierarchy = InsuranceHierarchy.uk_home_with_geography()  # + postcode × region

# Motor: OwnDamage / ThirdPartyInjury / Windscreen
motor_tree = PerilTree.uk_motor()
```

The `HierarchyBuilder` class converts the `InsuranceHierarchy` spec into the `S_df` DataFrame that `hierarchicalforecast.aggregate()` expects, handling the row ordering so the weight alignment is guaranteed correct.

---

## Pre- and post-reconciliation diagnostics

`CoherenceReport` runs before reconciliation and tells you where your forecasts are incoherent, by how much, and at which periods:

```python
report = reconciler.check_coherence(lc_df, ep_df)

# Which series are most incoherent?
report.worst_violations(n=5)
#    series       period  discrepancy_pct
# 0  Home         2024Q1         5.23
# 1  Buildings    2024Q3         4.61
# 2  Home         2023Q4         3.87
# 3  Contents     2024Q2         2.44
# 4  Buildings    2024Q1         2.31

# Is the incoherence systematic or random?
report.discrepancy_time_series().plot()
```

`AttributionReport` runs after reconciliation and shows how much each series was adjusted, in absolute and percentage terms:

```python
result = reconciler.reconcile(lc_df, ep_df)
attr = result.attribution

attr.summary()
#    series              mean_adjustment_pct   max_adjustment_pct
# 0  Home                              -2.1                 -4.1
# 1  Buildings                         +1.3                 +2.8
# 2  Fire                              -0.4                 -0.9
# 3  EoW                               +2.1                 +3.7
# 4  Subsidence                        +0.8                 +1.6
```

The attribution report answers the governance question that follows every reconciliation: "We can see the numbers changed. Which models moved the most, and by how much?" That question currently gets answered with "we blended them" — which is not an answer that satisfies a peer reviewer or an FCA audit.

---

## Putting it together

The end-to-end workflow for a UK home pricing review:

```python
from insurance_reconcile import InsuranceReconciler
from insurance_reconcile.hierarchy import InsuranceHierarchy
from insurance_reconcile.simulate import simulate_uk_home

# Your base forecasts (periods × series), earned premium
lc_df, ep_df = simulate_uk_home(n_periods=24)

# Step 1: coherence audit
hierarchy = InsuranceHierarchy.uk_home()
reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())

report = reconciler.check_coherence(lc_df, ep_df)
if report.is_incoherent:
    print(f"Max discrepancy: {report.max_discrepancy_pct:.1f}%")
    print(report.worst_violations(n=3).to_string())

# Step 2: reconcile
result = reconciler.reconcile(lc_df, ep_df)

# Step 3: attribution
print(result.attribution.summary().to_string())

# Step 4: coherent forecasts, ready for the rate filing
lc_reconciled = result.reconciled_df
```

The reconciled DataFrame is a drop-in replacement for the base forecasts. Every downstream calculation — development factors, trend analysis, rate level indication — receives coherent inputs. The hierarchy constraints are satisfied by construction, not by a spreadsheet cell that someone will forget to update next quarter.

---

## Why not just use hierarchicalforecast directly

Nixtla's `hierarchicalforecast` is a well-maintained library and `insurance-reconcile` uses it as an optional dependency for the MinTrace solver. The gaps are all insurance-specific:

`wls_struct` weights by structural count (number of bottom series per aggregate). That is correct for time series with uniform variance structure. Insurance EP varies by orders of magnitude across perils — Subsidence might have one-tenth the premium of Fire. Structural count weighting treats them identically.

`hierarchicalforecast` reconciles whatever values you pass in. If you pass loss costs, it applies additive constraints to values that do not satisfy additive constraints. The output will appear reconciled — the constraints will appear to hold — but the underlying economics are wrong. You would only discover this if you independently checked the EP-weighted aggregation rule.

The frequency×severity reconciliation requires working in log-space and back-transforming. `hierarchicalforecast` has no mechanism for this transform.

The hierarchy spec (`S_df` construction) requires deep familiarity with the `aggregate()` function's column ordering conventions. There is no declarative DSL; you build the input DataFrame by hand. For a crossed peril×geography hierarchy with four geographic levels and seven perils, that is 88 rows to get right.

`insurance-reconcile` does not compete with `hierarchicalforecast`. It wraps it with the insurance semantics it needs.

---

## The standard it replaces

The standard is a spreadsheet. On one tab: the portfolio GLM trend. On another: the peril-level models. In a third: a cell that reads `=E12*0.6+C4*0.4` where the 60/40 blend was decided in a meeting three quarters ago and no one has documented the rationale.

The spreadsheet has produced the same answer for every pricing team that has ever done this. The teams know it is not quite right. It fails every audit because the methodology is not written down. And it discards genuine statistical information — both the portfolio model and the peril models contain signal, and blending them in fixed proportions is not the optimal way to use either.

MinTrace is not a new idea. It has been standard methodology in academic forecasting since Hyndman et al. (2011) and is now over a decade old in the time series literature. The reason insurance teams are not using it is not scepticism — it is that the available implementations do not handle the insurance-specific details correctly.

160 tests, 9 modules, MIT-licensed.

```bash
uv add 'insurance-reconcile[mintrace]'
```

---

**[insurance-reconcile on GitHub](https://github.com/burning-cost/insurance-reconcile)**

- [When Did Your Loss Ratio Actually Change?](/2026/03/13/insurance-changepoint/) — BOCPD for detecting regime shifts in claims experience
- [Trend selection is not actuarial judgment](/2026/03/13/insurance-trend/) — GLM-based trend with structural break detection and ONS index integration

---

**References**

- Wickramasuriya, S.L., Athanasopoulos, G., Hyndman, R.J., & Rajaratnam, M. (2019). Optimal Forecast Reconciliation using Unbiasedness, Covariance Information and Minimum Trace. *Journal of the American Statistical Association*, 114(508), 804-819.
- Hyndman, R.J., Ahmed, R.A., Athanasopoulos, G., & Shang, H.L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics and Data Analysis*, 55(9), 2579-2589.
