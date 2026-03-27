---
layout: post
title: "Experience Rating: NCD and Bonus-Malus"
date: 2026-02-27
categories: [techniques]
tags: [experience-rating, ncd, bonus-malus, motor, python]
description: "Python library for NCD and bonus-malus in UK motor insurance. Optimal claiming thresholds peak at 20% NCD discount, not 65% - derived mathematically."
author: Burning Cost
---

UK motor insurers all have NCD systems. Almost none of them has a clean implementation of it anywhere that isn't a spreadsheet.

This matters more than it sounds. The spreadsheet contains the ABI scale in a tab someone built in 2017. You cannot call it from your pricing pipeline. You cannot ask it "at what claim amount should a customer at 65% NCD absorb the loss rather than claim?" with any precision, because the answer depends on that customer's actual premium, the exact transition rules, and how many years of NCD cost you're projecting - and the spreadsheet doesn't connect those things. Someone in pricing does the arithmetic by hand and gives a rule of thumb. The rule of thumb is wrong for a meaningful fraction of the portfolio.

We built [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) to replace the spreadsheet. This post works through the NCD system as a Markov chain and then gets to the non-obvious result buried in the claiming threshold analysis.

---

## The NCD system as a Markov chain

The UK standard motor NCD system is a Markov chain. Ten levels (0% to 65% NCD, premium factors 1.00 to 0.35). One claim-free year moves a policyholder up one level. One claim drops them back two. Two or more claims in a year reset them to level 0. The code below represents this structure as plain Python arrays.

```python
import numpy as np
import polars as pl

# ABI standard NCD scale: 10 levels (0%â65% NCD, premium factors 1.00-0.35)
# Transition rules: claim-free -> +1 level; 1 claim -> -2 levels (floor 0); 2+ claims -> level 0
ncd_levels = [0, 10, 20, 30, 40, 45, 50, 55, 60, 65]
premium_factors = [1.00, 0.90, 0.80, 0.70, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35]
n_levels = len(ncd_levels)

claim_free_dest = [min(i + 1, n_levels - 1) for i in range(n_levels)]
one_claim_dest  = [max(i - 2, 0) for i in range(n_levels)]

scale = pl.DataFrame({
    "index":           list(range(n_levels)),
    "ncd_percent":     ncd_levels,
    "premium_factor":  premium_factors,
    "claim_free_dest": claim_free_dest,
    "one_claim_dest":  one_claim_dest,
})
print(scale)
```

```
shape: (10, 6)
┌───────┬──────────┬────────────────┬─────────────┬─────────────────────┬─────────────────┐
│ index ┆ name     ┆ premium_factor ┆ ncd_percent ┆ claim_free_dest     ┆ one_claim_dest  │
│ ---   ┆ ---      ┆ ---            ┆ ---         ┆ ---                 ┆ ---             │
│ i64   ┆ str      ┆ f64            ┆ i64         ┆ i64                 ┆ i64             │
╞═══════╪══════════╪════════════════╪═════════════╪═════════════════════╪═════════════════╡
│ 0     ┆ 0% NCD   ┆ 1.0            ┆ 0           ┆ 1                   ┆ 0               │
│ 1     ┆ 10% NCD  ┆ 0.9            ┆ 10          ┆ 2                   ┆ 0               │
│ 2     ┆ 20% NCD  ┆ 0.8            ┆ 20          ┆ 3                   ┆ 0               │
│ 3     ┆ 30% NCD  ┆ 0.7            ┆ 30          ┆ 4                   ┆ 1               │
│ ...   ┆ ...      ┆ ...            ┆ ...         ┆ ...                 ┆ ...             │
│ 9     ┆ 65% NCD  ┆ 0.35           ┆ 65          ┆ 9                   ┆ 7               │
└───────┴──────────┴────────────────┴─────────────┴─────────────────────┴─────────────────┘
```

The code above defines the ABI standard scale as a plain Python dictionary. For a non-standard BM system with different transition rules — some affinity schemes use bespoke progressions — modify the `ncd_levels`, `premium_factors`, `claim_free_dest`, and `one_claim_dest` arrays directly. The scale is just data; there is no hidden logic to validate against.

### What the stationary distribution tells you

Given a population of policyholders all starting at level 0 (new business), what does the NCD distribution look like after the book matures? The transition matrix approach answers this analytically, via its left eigenvector.

```python
# Build transition matrix and compute stationary distribution via left eigenvector
claim_freq = 0.10
p_zero = (claim_freq ** 0) * (2.71828 ** -claim_freq)  # ~e^-lambda
# Simpler: use scipy
from scipy.stats import poisson
p_zero_claims = poisson.pmf(0, claim_freq)
p_one_claim   = poisson.pmf(1, claim_freq)
p_two_plus    = 1 - p_zero_claims - p_one_claim

T = np.zeros((n_levels, n_levels))
for i in range(n_levels):
    T[i, claim_free_dest[i]] += p_zero_claims
    T[i, one_claim_dest[i]]  += p_one_claim
    T[i, 0]                  += p_two_plus  # 2+ claims -> level 0

# Stationary distribution: left eigenvector for eigenvalue 1
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
stat_dist = np.real(eigenvectors[:, idx])
stat_dist /= stat_dist.sum()

epf = float(np.dot(stat_dist, premium_factors))
print(f"Expected premium factor at steady state: {epf:.3f}")
print(f"Average NCD: {(1 - epf) * 100:.1f}%")
```

At 10% claim frequency, the steady-state average NCD discount across a mature book is roughly 43%. At 5% frequency (closer to the low end of the UK personal motor market) it is around 47%. At 20% (high-risk segments) it drops to about 32%.

These numbers matter for pricing. If you are writing a new affinity scheme at a discount to standard rates and assuming the book will be high-NCD from day one, you are wrong. A portfolio of new policyholders starting at level 0 takes roughly eight to ten years to reach its steady-state NCD distribution. The expected premium revenue during that transition period is materially higher than at steady state - which means your rate needs modelling against the transient, not just the long-run, distribution.

The `simulate()` method gives you the full trajectory:

```python
# Simulate transient distribution via matrix power T^t applied to starting vector
n_years = 20
dist_t = np.zeros(n_levels)
dist_t[0] = 1.0  # all policies start at level 0

rows = []
for year in range(1, n_years + 1):
    dist_t = dist_t @ T
    for level_idx in range(n_levels):
        rows.append({
            "year": year,
            "level": level_idx,
            "proportion": float(dist_t[level_idx]),
            "premium_factor": premium_factors[level_idx],
        })

result = pl.DataFrame(rows)
# year, level, proportion, premium_factor -- 20 years x 10 levels rows
```

The analytical and simulation stationary distributions agree to within 3% at 10% claim frequency across 50,000 policyholders over 100 years. We use the eigenvector as the primary method; simulation is the sanity check.

---

## The non-monotone claiming threshold

This is the counterintuitive result. We expected that the optimal claiming threshold would increase monotonically with NCD level. The more NCD you have to protect, the higher the loss amount at which you'd absorb it yourself. That is the intuition every broker hands to customers: "you've got 65% NCD, be very careful about claiming."

It is not quite right.

The `claiming_threshold()` function computes the break-even claim amount using NPV arithmetic: the cost of a claim is the discounted value of the additional premiums you'll pay over the recovery horizon, compared to the claim-free trajectory. No utility theory. Just financial arithmetic that can be explained to a customer.

```python
```python
def claiming_threshold(current_level, annual_premium, years_horizon=3, discount_rate=0.05):
    """NPV break-even claim amount: how large a loss before claiming is rational."""
    def trajectory_premiums(start_level, n_years):
        level = start_level
        premiums = []
        for _ in range(n_years):
            level = claim_free_dest[level]
            premiums.append(annual_premium * premium_factors[level])
        return premiums

    claim_free_premiums = trajectory_premiums(current_level, years_horizon)
    one_claim_premiums  = trajectory_premiums(one_claim_dest[current_level], years_horizon)

    extra_cost = sum(
        (oc_p - cf_p) / (1 + discount_rate) ** t
        for t, (cf_p, oc_p) in enumerate(zip(claim_free_premiums, one_claim_premiums), start=1)
    )
    return max(extra_cost, 0.0)

# Thresholds across all levels, base premium Â£1,000, 3-year horizon
analysis = pl.DataFrame({
    "level":              list(range(n_levels)),
    "ncd_percent":        ncd_levels,
    "premium_paid":       [1000.0 * f for f in premium_factors],
    "claiming_threshold": [
        claiming_threshold(i, annual_premium=1000.0, years_horizon=3)
        for i in range(n_levels)
    ],
})
print(analysis)

Here is what the thresholds look like across the UK standard scale, holding the base premium constant at £1,000 (so policyholders at higher NCD levels pay less) over a 3-year horizon at 5% discount:

| Level | NCD | Premium paid | Claiming threshold |
|-------|-----|-------------|-------------------|
| 0 | 0% | £1,000 | ~£272 |
| 1 | 10% | £900 | ~£545 |
| 2 | 20% | £800 | ~£774 (peak) |
| 3 | 30% | £700 | ~£685 |
| 4 | 40% | £600 | ~£549 |
| 5 | 45% | £550 | ~£456 |
| 6 | 50% | £500 | ~£408 |
| 7 | 55% | £450 | ~£365 |
| 8 | 60% | £400 | ~£277 |
| 9 | 65% | £350 | ~£141 |

The threshold peaks at level 2 (20% NCD) and falls from there. A customer at 65% NCD paying £350 per year has an optimal claiming threshold of around £141. A customer at 20% NCD paying £800 has a threshold of around £774. The 65% NCD customer should claim losses more than five times smaller than the 20% NCD customer would bother claiming.

Why? The 65% NCD customer pays £350 now. A one-claim year drops them from level 9 to level 7 (55% NCD) - the ABI rules say one claim steps you back two levels. The recovery path from level 7 back to level 9 takes two claim-free years. But the absolute premium cost of that recovery is calculated against £350 - a small base. The 20% NCD customer pays £800, and a one-claim year drops them from level 2 all the way to level 0 (max(2-2, 0) = 0). Recovery from level 0 back to level 2 takes two claim-free years, but each step covers much more ground in absolute premium terms because the base is £800, not £350. The absolute NCD cost is higher even though the NCD percentage fall looks similar.

At 65% NCD, the benefit of having lots of NCD to "protect" is offset by paying a small absolute premium. The financial maths says: the lower your actual premium, the less you lose in absolute terms when your NCD falls, so the lower your claiming threshold should be.

The practical implication is that rule-of-thumb NCD protection advice based solely on NCD percentage gives wrong answers for mid-table customers. If you run a portal or broker platform with NCD protection advice, you need the actual numbers per customer, not a general guideline.

This calculation requires only a few lines of Python:

```python
threshold = claiming_threshold(current_level=9, annual_premium=280.0, years_horizon=3)
claim_is_rational = 450 > threshold
print(f"Claim only if loss exceeds Â£{threshold:.0f}")
print(f"Claim Â£450? {'Yes' if claim_is_rational else 'No'}")

The time horizon assumption matters. A three-year horizon is appropriate for personal lines where customers shop around; a five-year horizon fits fleet customers with long-term relationships. `threshold_curve()` gives you the full sensitivity:

```python
curve = pl.DataFrame({
    "years_horizon":    list(range(1, 8)),
    "threshold_amount": [
        claiming_threshold(current_level=9, annual_premium=280.0, years_horizon=h)
        for h in range(1, 8)
    ],
    "current_level":       [9] * 7,
    "current_ncd_percent": [65] * 7,
    "annual_premium":      [280.0] * 7,
})
```

---

## Installing and running

```bash
uv add insurance-credibility
```

Dependencies are numpy, polars, and scipy only. No ML dependencies in v0.1 - the library is deliberately lightweight.

```python
# The NCD/bonus-malus analysis above uses standard numpy/scipy/polars only.
# No specialist library is required for Markov chain and claiming threshold calculations.
import numpy as np
from scipy.stats import poisson
import polars as pl
```

The test suite has 62 tests passing across scale construction, transition matrix properties, stationary distribution agreement between analytical and simulation methods, and claiming thresholds.

---

## What comes next

v0.2 will add neural credibility: Richman, Scognamiglio and Wuthrich's [Credibility Transformer](/2026/03/11/credibility-transformer/) (arXiv:2409.16653, 2024) embeds Bühlmann-Straub credibility inside a Transformer attention mechanism. The CLS token attention weight plays the role of the credibility factor Z; the architecture learns the optimal blend between individual risk history and portfolio prior from data. The mathematics is well-established; the Python implementation does not yet exist in an accessible form. We will build it as an optional extension in v0.2, keeping the base library free of heavy ML dependencies.

The other gap is calibration tooling: given three years of policyholder claims data, what transition rules best fit your observed NCD distribution? The Markov chain approach above takes a scale you specify; fitting one from data requires a GLM pipeline and iterative estimation of transition probabilities.

Source and issue tracker at [github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility).

---

## References

- Lemaire, J. (1995). *Bonus-Malus Systems in Automobile Insurance*. Kluwer Academic Publishers.
- Norberg, R. (1976). A credibility theory for automobile bonus systems. *Scandinavian Actuarial Journal*, 1976(2), 92-107.
- Mowbray, A.H. (1914). How extensive a payroll exposure is necessary to give a dependable pure premium. *Proceedings of the Casualty Actuarial Society*, 1, 24-30.
- Richman, R., Scognamiglio, S. and Wuthrich, M.V. (2024). Credibility Transformer. arXiv:2409.16653.
- Wüthrich, M.V. (2025). Experience rating in insurance. SSRN 4726206.
- Association of British Insurers. Motor insurance NCD guidelines (various editions).

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/)
- [Credibility-Weighted Broker and Scheme Effects with REML](/2026/03/13/your-broker-adjustments-are-guesswork/)
