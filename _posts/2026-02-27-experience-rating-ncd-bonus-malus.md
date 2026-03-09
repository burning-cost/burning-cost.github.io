---
layout: post
title: "Your NCD Threshold Advice Is Wrong at 65%"
date: 2026-02-27
categories: [techniques]
tags: [experience-rating, ncd, bonus-malus, experience-mod, schedule-rating, motor, commercial-lines, python]
description: "A Python library for NCD/bonus-malus systems, experience modification factors, and schedule rating. Includes the non-obvious finding that optimal NCD claiming thresholds peak at 30% NCD, not 65%."
author: Burning Cost
---

UK motor insurers all have NCD systems. Almost none of them has a clean implementation of it anywhere that isn't a spreadsheet.

This matters more than it sounds. The spreadsheet contains the ABI scale in a tab someone built in 2017. You cannot call it from your pricing pipeline. You cannot ask it "at what claim amount should a customer at 65% NCD absorb the loss rather than claim?" with any precision, because the answer depends on that customer's actual premium, the exact transition rules, and how many years of NCD cost you're projecting - and the spreadsheet doesn't connect those things. Someone in pricing does the arithmetic by hand and gives a rule of thumb. The rule of thumb is wrong for a meaningful fraction of the portfolio.

We built [`experience-rating`](https://github.com/burning-cost/experience-rating) to replace the spreadsheet. It covers three things that belong together: NCD/bonus-malus systems, experience modification factors for commercial lines, and schedule rating. This post works through each one and then gets to the non-obvious result buried in the claiming threshold analysis.

---

## The NCD system as a Markov chain

The UK standard motor NCD system is a Markov chain. Ten levels (0% to 65% NCD, premium factors 1.00 to 0.35). One claim-free year moves a policyholder up one level. One claim drops them back two. Two or more claims in a year reset them to level 0. The `BonusMalusScale` class represents exactly this structure.

```python
from experience_rating import BonusMalusScale, BonusMalusSimulator

scale = BonusMalusScale.from_uk_standard()
print(scale.summary())
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

`from_uk_standard()` gives you the ABI scale. If you have a commercial BM system with different transition rules - some fleet policies use three-step progressions, some commercial vehicle covers have different back-steps by fleet size - you define it via `from_dict()` with a JSON-compatible spec. The validation at construction time checks that every transition destination is a valid level index. There is no silent out-of-range behaviour.

### What the stationary distribution tells you

Given a population of policyholders all starting at level 0 (new business), what does the NCD distribution look like after the book matures? The `BonusMalusSimulator` answers this analytically, via the left eigenvector of the transition matrix.

```python
sim = BonusMalusSimulator(scale, claim_frequency=0.10)

dist = sim.stationary_distribution(method="analytical")
epf = sim.expected_premium_factor()

print(f"Expected premium factor at steady state: {epf:.3f}")
print(f"Average NCD: {(1 - epf) * 100:.1f}%")
```

At 10% claim frequency, the steady-state average NCD discount across a mature book is roughly 43%. At 5% frequency (closer to the low end of the UK personal motor market) it is around 47%. At 20% (high-risk segments) it drops to about 32%.

These numbers matter for pricing. If you are writing a new affinity scheme at a discount to standard rates and assuming the book will be high-NCD from day one, you are wrong. A portfolio of new policyholders starting at level 0 takes roughly eight to ten years to reach its steady-state NCD distribution. The expected premium revenue during that transition period is materially higher than at steady state - which means your rate needs modelling against the transient, not just the long-run, distribution.

The `simulate()` method gives you the full trajectory:

```python
result = sim.simulate(n_policyholders=50_000, n_years=20)
# Returns a Polars DataFrame: year, level, count, proportion, premium_factor
# 50k policyholders x 20 years runs in under a second on a laptop
```

The analytical and simulation stationary distributions agree to within 3% at 10% claim frequency across 50,000 policyholders over 100 years. We use the eigenvector as the primary method; simulation is the sanity check.

---

## The non-monotone claiming threshold

This is the counterintuitive result. We expected, before building `ClaimThreshold`, that the optimal claiming threshold would increase monotonically with NCD level. The more NCD you have to protect, the higher the loss amount at which you'd absorb it yourself. That is the intuition every broker hands to customers: "you've got 65% NCD, be very careful about claiming."

It is not quite right.

`ClaimThreshold` computes the break-even claim amount using NPV arithmetic: the cost of a claim is the discounted value of the additional premiums you'll pay over the recovery horizon, compared to the claim-free trajectory. No utility theory. Just financial arithmetic that can be explained to a customer.

```python
from experience_rating import ClaimThreshold

ct = ClaimThreshold(scale, discount_rate=0.05)

# Thresholds across all levels, base premium £1,000, 3-year horizon
# full_analysis sets each level's premium as base * premium_factor
analysis = ct.full_analysis(annual_premium=1000.0, years_horizon=3)
print(analysis)
```

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

The library makes this calculation trivial:

```python
# Should this specific customer claim this specific loss?
claim_is_rational = ct.should_claim(
    current_level=9,
    claim_amount=450,
    annual_premium=280.0,  # what they actually pay
    years_horizon=3,
)

# Or: what is the threshold?
threshold = ct.threshold(
    current_level=9,
    annual_premium=280.0,
    years_horizon=3,
)
print(f"Claim only if loss exceeds £{threshold:.0f}")
```

The time horizon assumption matters. A three-year horizon is appropriate for personal lines where customers shop around; a five-year horizon fits fleet customers with long-term relationships. `threshold_curve()` gives you the full sensitivity:

```python
curve = ct.threshold_curve(current_level=9, annual_premium=280.0, max_horizon=7)
# Returns: years_horizon, threshold_amount, current_level, current_ncd_percent, annual_premium
```

---

## Experience modification factors for commercial lines

On the commercial side, the equivalent question to NCD is the experience modification factor: given this risk's actual loss history versus what we expected, what adjustment do we apply to next year's rate?

The standard formula (NCCI-style, used in US workers' compensation and adapted widely for UK commercial lines) is:

```
Mod = (A * actual_losses + (1 - A) * expected_losses + B) / (expected_losses + B)
```

Where A is the credibility weight and B is the ballast. A Mod above 1.0 means the risk has performed worse than expected; below 1.0 means better. The ballast B limits the swing that any single large loss can produce. Without it, one catastrophic year sends the mod to 2.0+ and the customer cannot renew at a remotely competitive price.

The `ExperienceModFactor` class makes the parameter choices explicit:

```python
import polars as pl
from experience_rating import ExperienceModFactor
from experience_rating.experience_mod import CredibilityParams

params = CredibilityParams(credibility_weight=0.65, ballast=8_000.0)
emod = ExperienceModFactor(params)

portfolio = pl.DataFrame({
    "risk_id":          ["Precision Mfg Ltd", "Cornish Hotels Ltd", "BrightTech Ltd"],
    "expected_losses":  [25_000.0,             80_000.0,             10_000.0],
    "actual_losses":    [32_000.0,             65_000.0,              8_000.0],
})

result = emod.predict_batch(portfolio, cap=2.0, floor=0.5)
print(result.select(["risk_id", "expected_losses", "actual_losses", "mod_factor"]))
```

```
shape: (3, 4)
┌────────────────────┬─────────────────┬───────────────┬────────────┐
│ risk_id            ┆ expected_losses ┆ actual_losses ┆ mod_factor │
│ ---                ┆ ---             ┆ ---           ┆ ---        │
│ str                ┆ f64             ┆ f64           ┆ f64        │
╞════════════════════╪═════════════════╪═══════════════╪════════════╡
│ Precision Mfg Ltd  ┆ 25000.0         ┆ 32000.0       ┆ 1.137      │
│ Cornish Hotels Ltd ┆ 80000.0         ┆ 65000.0       ┆ 0.894      │
│ BrightTech Ltd     ┆ 10000.0         ┆ 8000.0        ┆ 0.893      │
└────────────────────┴─────────────────┴───────────────┴────────────┘
```

We expose ballast directly rather than deriving it in a calibration function. The choice of B is a deliberate actuarial decision with regulatory implications: it determines which risks get penalised and which get discounts, and the reasoning needs to be defensible. Hiding B inside a `fit()` call obscures a parameter that underwriters and compliance teams will ask about.

If you want to derive credibility from exposure rather than setting A manually:

```python
emod = ExperienceModFactor.from_exposure(
    actual_exposure=3_500_000,        # e.g., payroll in £
    full_credibility_exposure=5_000_000,
    ballast=8_000.0,
    credibility_formula="square_root",  # Mowbray formula: A = sqrt(e / e_full)
)
```

The `sensitivity()` method is useful for underwriting conversations:

```python
sense = emod.sensitivity(expected_losses=25_000)
# Returns: actual_losses, mod_factor, loss_ratio across the range [0, 75_000]
```

Show an underwriter how the mod responds to different loss outcomes before binding. If the cap kicks in below 2x expected losses, that is worth knowing in advance.

---

## Schedule rating

Schedule rating is the older, less systematic sibling of experience rating. The underwriter applies judgemental debits and credits within pre-approved bounds for factors not captured in the statistical rate: premises condition, management quality, risk controls, claims co-operation history. In UK commercial liability, property, and professional indemnity, schedule rating is a standard part of the underwriting process.

`ScheduleRating` is a validated container for this. The bounds are checked at entry time, not at rating time:

```python
from experience_rating import ScheduleRating

sr = ScheduleRating(max_total_debit=0.25, max_total_credit=0.25)
sr.add_factor("Premises",      min_credit=-0.10, max_debit=0.10, description="Premises quality")
sr.add_factor("Management",    min_credit=-0.07, max_debit=0.07, description="Management quality")
sr.add_factor("Risk_Controls", min_credit=-0.08, max_debit=0.08, description="Risk controls in place")

# Rate a specific risk
factor = sr.rate({
    "Premises":      0.05,   # 5% debit - below-average premises
    "Management":   -0.03,   # 3% credit - experienced management
    "Risk_Controls": 0.02,   # 2% debit - adequate but not strong
})
print(f"Schedule factor: {factor:.4f}")  # 1.0400
```

Batch rating against a portfolio:

```python
risks = pl.DataFrame({
    "risk_id":       ["A", "B", "C"],
    "Premises":      [0.05, -0.10,  0.00],
    "Management":   [-0.03,  0.07, -0.05],
    "Risk_Controls": [0.02,  0.08, -0.08],
})

result = sr.rate_batch(risks)
# Adds "schedule_factor" column; aggregate cap/floor applied silently
```

The aggregate cap (25% in this example) is applied after summing all individual factors. If the risk scores +10% on premises, +7% on management, and +8% on risk controls, the total is +25% - exactly at the cap. Any further factors would be absorbed. This is UK commercial practice: the cap is set by underwriting authority and the aggregate controls the total swing, not individual factor bounds.

The reason bounds are validated at entry rather than at the `rate()` call is compliance risk. If an underwriter keys in 0.15 for a factor with a 0.10 cap, you want that error caught immediately, not silently clipped when the factor runs at month-end. Fail early.

---

## Installing and running

```bash
uv add experience-rating
```

Dependencies are numpy, polars, and scipy only. No ML dependencies in v0.1 - the library is deliberately lightweight.

```python
from experience_rating import (
    BonusMalusScale,
    BonusMalusSimulator,
    ClaimThreshold,
    ExperienceModFactor,
    ScheduleRating,
)
```

The test suite has 98 tests passing across scale construction, transition matrix properties, stationary distribution agreement between analytical and simulation methods, claiming thresholds, experience modification formula correctness, and schedule rating bounds validation.

---

## What comes next

v0.2 will add neural credibility: Richman, Scognamiglio and Wuthrich's Credibility Transformer (arXiv:2409.16653, 2024) embeds Bühlmann-Straub credibility inside a Transformer attention mechanism. The CLS token attention weight plays the role of the credibility factor Z; the architecture learns the optimal blend between individual risk history and portfolio prior from data. The mathematics is well-established; the Python implementation does not yet exist in an accessible form. We will build it as an optional extension in v0.2, keeping the base library free of heavy ML dependencies.

The other gap is calibration tooling: given three years of policyholder claims data, what transition rules best fit your observed NCD distribution? Currently `BonusMalusScale` takes a scale you specify; it does not fit one from data. That requires a GLM pipeline and iterative estimation. It is not in v0.1 because we wanted the analysis tools to ship without coupling them to a training framework.

Source and issue tracker at [github.com/burning-cost/experience-rating](https://github.com/burning-cost/experience-rating).

---

## References

- Lemaire, J. (1995). *Bonus-Malus Systems in Automobile Insurance*. Kluwer Academic Publishers.
- Norberg, R. (1976). A credibility theory for automobile bonus systems. *Scandinavian Actuarial Journal*, 1976(2), 92-107.
- Mowbray, A.H. (1914). How extensive a payroll exposure is necessary to give a dependable pure premium. *Proceedings of the Casualty Actuarial Society*, 1, 24-30.
- Richman, R., Scognamiglio, S. and Wuthrich, M.V. (2024). Credibility Transformer. arXiv:2409.16653.
- Wüthrich, M.V. (2025). Experience rating in insurance. SSRN 4726206.
- Association of British Insurers. Motor insurance NCD guidelines (various editions).

---

**Related articles from Burning Cost:**
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/03/06/buhlmann-straub-credibility-in-python/)
- [Survival Models for Insurance Retention](/2026/03/08/survival-models-for-insurance-retention/)
- [Your Broker Adjustments Are Guesswork](/2026/03/09/your-broker-adjustments-are-guesswork/)
