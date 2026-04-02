---
layout: post
title: "LifetimeBoundsCalculator: Worst- and Best-Case Mortality for Variable Annuity Pricing"
date: 2026-04-02
categories: [libraries]
tags: [mortality, survival, variable-annuity, annuity-pricing, life-tables, insurance-survival, actuarial, cmi, arXiv-2603.06238]
description: "insurance-survival v0.5.0 adds LifetimeBoundsCalculator — closed-form worst- and best-case mortality bounds between integer ages for variable annuity and guaranteed benefit pricing."
author: burning-cost
---

Fractional-age mortality assumptions are a known source of pricing error in variable annuity products. The standard approaches — uniform distribution of deaths (UDD), constant force of mortality (CFM), and the Balducci assumption — give different fractional-year survival probabilities, and there is no universally correct choice. What the literature has not previously provided is a clean answer to: what are the extreme values? What is the highest survival probability we could assign at age 65.4, and what is the lowest, given only that annual survival probabilities are fixed?

[insurance-survival v0.5.0](https://pypi.org/project/insurance-survival/) adds `LifetimeBoundsCalculator`, implementing the closed-form mortality bounds from Dupret and Motte (arXiv:2603.06238, March 2026). The result is a tool that computes worst-case and best-case contract values — not by Monte Carlo sampling over assumed mortality distributions, but by applying the mathematical extremes that any valid fractionation of the life table must lie between.

---

## The mathematical result

The key theorem in Dupret and Motte is simple to state. Given annual survival probabilities {p_x+j} for j = 0, 1, ..., the two extreme survival functions on the real line are:

**Upper survival** (mortality concentrated at year-ends): S_upper(t, s) is constant on each interval [j, j+1), equal to the cumulative survival at year-start j. Death is deferred to the last moment of each year.

**Lower survival** (mortality concentrated at year-beginnings): S_lower(t, s) is constant on each interval (j-1, j], equal to the cumulative survival at year-end j. Death is pulled forward to the first moment of each year.

At every integer age, both bounds coincide exactly with the life table. Between integers, the gap between them bounds every other valid fractional-age assumption. UDD, CFM, and Balducci all lie strictly between S_lower and S_upper for all t — verified both in the paper and in the library's test suite.

The "upper" and "lower" labels refer to the survival function, not necessarily the contract value. For a life annuity, upper survival implies longer expected lifetime and therefore higher contract value. For a guaranteed minimum death benefit (GMDB) in a variable annuity, higher survival defers death and therefore reduces the present value of the death benefit — so the bound directions reverse. The library's `spread_pct` always reports an absolute spread, with the direction documented per contract type.

---

## The API

```python
from insurance_survival.mortality.bounds import LifetimeBoundsCalculator, BoundsResult

# Construct from annual survival probabilities (hat_p_j = p_{x+j})
hat_p = [0.9942, 0.9931, 0.9918, 0.9903, 0.9886,
         0.9866, 0.9843, 0.9817, 0.9787, 0.9753]  # age 65-74 male, illustrative

calc = LifetimeBoundsCalculator(
    hat_p=hat_p,
    starting_age=65,
    discount_rate=0.04,
)
```

Or via the CMI S3 table classmethod (CMI 2023 improvement scale built in):

```python
calc = LifetimeBoundsCalculator.from_cmi(
    table_code='S3PML',      # S3 Pensioners Male Lives
    starting_age=65,
    n_years=30,
    improvement_scale='CMI2023',
    discount_rate=0.04,
)
```

The four CMI S3 table variants supported are S3PML (Pensioners Male Lives), S3PFL (Female Lives), S3AML (Annuitants Male Lives), and S3AFL (Female Lives) — the standard tables for UK pension scheme and annuity work.

---

## Computing bounds

For a 10-year deferred annuity paying £1 per year:

```python
result: BoundsResult = calc.annuity_bounds(
    t=0,       # current age offset from starting_age
    T=10,      # contract duration in years
    income_fn=lambda s: 1.0,  # constant £1 income
    n_steps=1000,
)

print(result.upper)       # upper bound on annuity present value
print(result.lower)       # lower bound on annuity present value
print(result.spread)      # upper - lower in £
print(result.spread_pct)  # spread as % of midpoint
```

For a guaranteed minimum death benefit paying benefit_fn(s) at death in year s:

```python
result = calc.death_benefit_bounds(
    t=0,
    T=10,
    benefit_fn=lambda s: 100_000.0,  # flat £100k death benefit
)
```

For arbitrary payoff structures (GMAB, GMWB, structured products):

```python
def gmab_payoff(s):
    # Guaranteed minimum accumulation benefit: pays max(guarantee, account_value)
    account_value = 100_000 * (1.06 ** s)
    guarantee = 120_000.0
    return max(account_value, guarantee)

result = calc.general_bounds(t=0, T=10, payoff_fn=gmab_payoff, n_steps=500)
```

The survival bounds schedule — the dense grid of S_upper and S_lower values — is available as a Polars DataFrame:

```python
df = calc.survival_bounds_schedule(t=0, T=10, n_steps=200)
# Columns: s (fractional age offset), survival_upper, survival_lower, spread
```

---

## A practical pricing example

A UK life office is pricing a 10-year GMAB on a £100k single premium. The policy guarantees return of premium at maturity regardless of fund performance. The pricing actuary needs worst-case and best-case present values for the mortality-dependent component.

Using CMI S3PML at age 65 with a 4% discount rate:

```python
calc = LifetimeBoundsCalculator.from_cmi('S3PML', 65, 10, 'CMI2023', 0.04)

# Worst case for GMAB: policyholder survives (no death benefit, but guarantee costs more)
# Best case for GMAB: policyholder dies early (death benefit cheaper)
result = calc.general_bounds(t=0, T=10,
                             payoff_fn=lambda s: 100_000 * max(1.0, 1.06**s),
                             n_steps=1000)

print(f"Upper bound: £{result.upper:,.0f}")
print(f"Lower bound: £{result.lower:,.0f}")
print(f"Spread:      £{result.spread:,.0f} ({result.spread_pct:.2f}%)")
```

The library's own verification for a similar UK 65-year-old male, 10-year annuity, no discounting produces a spread of 1.58% between the two bounds. For discounted contracts, the spread narrows because discounting compresses early vs late cash flows.

The `fractional_age_comparison()` method shows where UDD, CFM, and Balducci sit relative to the bounds:

```python
df = calc.fractional_age_comparison(t=0, T=5, contract_type='annuity')
# Columns: assumption, value, relative_to_upper, relative_to_lower
```

For standard UK pension scheme work, all three classical assumptions typically produce values well within the spread. The bounds are tightest at shorter durations (integer-age mortality matters less) and widest for products with high sensitivity to fractional-year mortality — guaranteed income drawdown products where the benefit changes in the first year of retirement.

---

## The `spread_by_duration` diagnostic

Before pricing, `spread_by_duration()` gives a quick view of how sensitive each product type is to fractional-age mortality as a function of contract length:

```python
df = calc.spread_by_duration(contract_type='annuity', max_years=30)
# Columns: duration_years, upper, lower, spread_pct
```

For a UK male aged 65 on S3PML, the spread peaks around durations 5–15 years and narrows toward 30 years as the cumulative survival probabilities dominate over fractional-year effects. The shape of this curve tells the pricing actuary which products are worth spending time on the fractionation assumption and which are not.

---

## Why this matters for VA pricing governance

Variable annuity pricing under Solvency II requires stress testing of longevity assumptions. The standard approach is to apply CMI improvement scale scenarios — improved longevity across the board. What `LifetimeBoundsCalculator` adds is a complementary analysis: given any fixed life table, what is the range of prices consistent with valid fractional-age assumptions? This is a model risk quantification that sits alongside the scenario tests rather than replacing them.

For products with narrow margins — guaranteed income products, for example — the 1–2% spread from fractional-age assumptions can be larger than the profit margin. Knowing this before signing off on pricing basis documentation is better than discovering it in validation.

The library is on [PyPI](https://pypi.org/project/insurance-survival/) and source is on [GitHub](https://github.com/burning-cost/insurance-survival). The paper is at [arXiv:2603.06238](https://arxiv.org/abs/2603.06238).
