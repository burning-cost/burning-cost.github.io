---
layout: post
title: "Your Peril GLMs Don't Add Up — MinTrace Reconciliation for Insurance Pricing"
seo_title: "MinTrace Reconciliation for Insurance Pricing: Making Peril GLMs Coherent"
date: 2026-04-03
categories: [techniques, libraries]
tags: [reconciliation, mintrace, hierarchical-forecasting, loss-cost, peril-modelling, uk-home, uk-motor, insurance-reconcile, python, wickramasuriya, coherence, frequency-severity]
description: "Separate peril GLMs routinely disagree with each other by 3–8% at cover level. insurance-reconcile applies premium-weighted MinTrace to make your hierarchy coherent without discarding either model's information."
math: true
author: burning-cost
---

Take a UK home book. You have a buildings GLM and a contents GLM. Each was fitted separately on cleaned, validated data by people who know what they are doing. You also have separate peril-level GLMs for fire, escape of water, subsidence, flood, and storm under buildings, and theft and accidental damage under contents.

At some point, someone multiplies the peril GLMs back up the tree and compares to the cover-level models. The numbers do not agree. Buildings cover says £127/RY. The sum of peril GLMs — weighted by exposure — says £121/RY. The difference is 5%. Nobody did anything wrong. The models were fitted on slightly different views of history, with different development factors, different attritional/large splits, possibly different rate changes. Disagreements of this size are normal.

The standard resolution is to pick one and override the other, or blend them in a spreadsheet with weights someone chose by feel. Both approaches throw away information. The portfolio model knows something about portfolio-level trend that the peril models do not. The peril models know something about peril mix shift that the portfolio model smooths over.

MinTrace reconciliation (Wickramasuriya, Athanasopoulos & Hyndman, JASA 2019) finds the minimum-variance linear combination of all models in the hierarchy that satisfies the constraints exactly. We built `insurance-reconcile` to apply it with correct insurance semantics — premium-weighted aggregation, rate-to-amount transforms, and frequency × severity decomposition.

---

## Why standard MinTrace needs insurance-specific wrapping

The Nixtla `hierarchicalforecast` library implements MinTrace well. The gaps are specific to insurance.

**Aggregation semantics.** Loss costs are rates, not amounts. Buildings cover loss cost is not the sum of peril loss costs. It is the premium-weighted average:

$$\text{LC}_\text{Buildings} = \frac{\sum_p \text{LC}_p \cdot \text{EP}_p}{\sum_p \text{EP}_p}$$

Standard MinTrace sums. This means vanilla reconciliation produces answers that are simultaneously mathematically optimal and actuarially incoherent.

**Weights.** Standard WLS_STRUCT weights each series by the count of bottom-level series feeding into it. For insurance, the economically correct weight is earned premium. A fire peril with £5M EP has a much lower coefficient of variation than a subsidence peril with £0.5M EP. The weight matrix should encode that we trust the high-volume cells more.

**Frequency × severity.** You do not model loss cost directly. You model frequency and severity separately, then multiply. That multiplication is a non-linear constraint: $\text{LC} = \text{freq} \times \text{sev}$. You cannot reconcile frequency and severity hierarchies independently — the product constraint must be respected. Log-space reconciliation makes this tractable: $\log(\text{LC}) = \log(\text{freq}) + \log(\text{sev})$ is additive.

---

## What incoherence looks like in practice

```python
from insurance_reconcile import InsuranceReconciler
from insurance_reconcile.hierarchy import InsuranceHierarchy
from insurance_reconcile.simulate import simulate_incoherent_uk_home

hierarchy = InsuranceHierarchy.uk_home()

# Simulate independently-run models with typical incoherence
lc_incoherent, ep_df = simulate_incoherent_uk_home(n_periods=24)

reconciler = InsuranceReconciler(hierarchy, earned_premium=ep_df.mean())
report = reconciler.check_coherence(lc_incoherent, ep_df)

print(report.to_string())
```

```
=== Coherence Report ===
Status: INCOHERENT
Max discrepancy: 5.23%
Violations found: 36

Top violations by magnitude:
  Buildings cover (period 2025-06): actual=132.41, expected=125.18, discrepancy=+5.78%
  Buildings cover (period 2025-07): actual=131.09, expected=124.88, discrepancy=+4.97%
  Contents cover (period 2025-06): actual=87.22, expected=84.06, discrepancy=+3.76%
  ...
```

The coherence check works without `hierarchicalforecast` — it only needs numpy and pandas. You can run it in any environment as a validation step before a rate filing or board pack.

---

## Reconciling with premium weights

```python
from insurance_reconcile.reconcile import PremiumWeightedMinTrace
from insurance_reconcile.hierarchy import HierarchyBuilder

builder = HierarchyBuilder(hierarchy)
S_df, tags = builder.build_S_df(bottom_series=hierarchy.all_perils)

ep = {
    'Fire': 5_000_000, 'EoW': 4_000_000, 'Subsidence': 1_000_000,
    'Flood': 500_000,  'Storm': 1_500_000,
    'Theft': 2_000_000, 'AccidentalDamage': 1_000_000,
    'Buildings': 12_000_000, 'Contents': 3_000_000,
    'Home': 15_000_000,
}

rec = PremiumWeightedMinTrace(earned_premium=ep, nonnegative=True)
lc_reconciled = rec.reconcile(lc_incoherent, S_df.values, series_names=list(S_df.index))
```

The reconciled output satisfies the hierarchy to within floating-point precision. More importantly, it does so with minimum total variance — the premium weighting ensures that fire and escape of water (the high-EP perils) barely move, while flood and subsidence (small EP, high volatility) absorb most of the adjustment. This is the actuarially correct behaviour.

---

## Frequency × severity reconciliation

If you model frequency and severity separately — and you should — reconcile them together:

```python
from insurance_reconcile.reconcile import FreqSevReconciler

rec = FreqSevReconciler(earned_premium=ep)
freq_reconciled, sev_reconciled = rec.reconcile(
    freq_hat, sev_hat, S_df.values, series_names=list(S_df.index)
)
```

The reconciler works in log space: $\log(\text{LC}) = \log(\text{freq}) + \log(\text{sev})$ is a linear additive constraint in that space. After reconciliation it back-transforms. Non-negativity is guaranteed by construction — you cannot get a negative loss cost out of a log-space reconciliation.

This matters because frequency trend and severity trend behave differently. Post-Whiplash Reform, UK motor BI frequency fell sharply while average severity rose as the residual book concentrated on more serious cases. A portfolio-level trend applied uniformly to loss cost misses this dynamic entirely. Reconciling frequency and severity hierarchies separately, and then re-combining, propagates the peril-level insight upward while keeping the portfolio view coherent.

---

## Declaring the hierarchy

The peril tree DSL avoids hand-building the summing matrix, which is a reliable source of errors:

```python
from insurance_reconcile.hierarchy import PerilTree, InsuranceHierarchy

tree = PerilTree(
    portfolio='Home',
    covers={
        'Buildings': ['Fire', 'EoW', 'Subsidence', 'Flood', 'Storm'],
        'Contents':  ['Theft', 'AccidentalDamage'],
    },
)

hierarchy = InsuranceHierarchy(tree)
```

For motor, a typical structure would be covers (TP, TPFT, Comp) at the top, perils (BI, PD, Theft, Windscreen, Fire) below, and optionally geography as a second dimension. The `GeographicHierarchy` class handles postcode → area → region → national stacking.

---

## When MinTrace works and when it does not

MinTrace is a linear method. It finds the optimal blend of your existing models under the assumption that their errors are stationary. There are cases where this assumption fails.

**Structural breaks.** If you took a major rate change or significant UW action mid-period, the before/after mix produces a non-stationary error structure. Reconcile each regime separately, or exclude the transition period.

**Conflicting signals.** If the portfolio model and the peril models genuinely disagree about the direction of trend — not just level — MinTrace will split the difference. This is not always wrong, but you should know it is happening. The coherence report attribution flags which level is driving the adjustment.

**Bootstrapped versus credibility estimates.** If some of your peril cells are credibility-blended to the class mean rather than fully graduated, they will have lower apparent variance (the credibility shrinkage suppressed it). MinTrace will treat them as more reliable than they are. Apply credibility before reconciliation, not after.

We think the 3–8% discrepancy that most UK pricing teams carry silently in their hierarchies is worth fixing. It creates inconsistencies between pricing, reserving, and planning that compound over time, and incoherence between peril-level and cover-level models is the kind of inconsistency that Consumer Duty (PRIN 2A) fair value assessments are expected to surface — if your perils say one thing and your cover model says another, that gap needs an owner. MinTrace does not require you to rebuild your models. It resolves the disagreement between the models you have.

---

## Installation

```bash
pip install insurance-reconcile
# Full reconciliation:
pip install 'insurance-reconcile[mintrace]'
```

The base install (numpy, pandas, scipy) handles coherence checking and the hierarchy DSL. The `[mintrace]` extra adds `hierarchicalforecast>=1.5.0` for the MinTrace computation. Nixtla's implementation handles the numerical edge cases — ridge protection for `mint_shrink`, non-negative QP via OSQP — that we do not replicate.

Source: [github.com/burning-cost/insurance-reconcile](https://github.com/burning-cost/insurance-reconcile)
