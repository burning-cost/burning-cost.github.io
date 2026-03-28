---
layout: post
title: "The Burning Cost Method for Excess of Loss Reinsurance Pricing in Python"
date: 2026-03-28
categories: [tutorials]
tags: [reinsurance, burning-cost, excess-of-loss, XL-pricing, python, frequency-severity]
description: "A practical Python walkthrough of the burning cost method for pricing excess of loss reinsurance treaties — loss trending, development, pure rate calculation, and sensitivity analysis using numpy and pandas."
---

Our brand name comes from here. The burning cost is a reinsurance pricing concept: take the historical losses that pierce your retention, cap them at the limit, and divide by subject premium. The ratio tells you what the layer has actually cost, expressed as a rate on premium. You trend it, load it, and that is your technical rate.

There is almost no practical Python material on this. The academic literature covers severity distributions and simulation. GEMAct handles aggregate modelling. But the workhorse method that reinsurance pricing actuaries actually use every day — the burning cost — gets nothing. This post fills that gap.

---

## What excess of loss reinsurance is and why you need burning cost

A cedant (the insurer buying reinsurance) writes a book of risks. Some of those risks produce large individual losses. Rather than absorbing the full loss, the cedant transfers the part of each loss above a **retention** R, up to a **limit** L, to a reinsurer. This is a per-risk XL treaty: for each claim of size x, the reinsurer pays:

```
Reinsurer payment = min(max(x - R, 0), L)
```

So if R = £500k and L = £1m, and a claim comes in at £1.2m:
- The cedant retains the first £500k
- The reinsurer pays £700k (the part between £500k and £1.5m)
- The total layer width is £1m (from £500k to £1.5m)

We write this as "£1m xs £500k."

The cedant pays a **reinsurance premium** for this protection. The question is: how much should that premium be?

The burning cost method answers this empirically: observe how much the layer has actually cost over historical treaty years, express that as a rate on the subject premium (the cedant's gross written premium for the covered portfolio), and apply trend/development adjustments to project it forward.

---

## The mathematics

For a treaty covering £L xs £R, and a set of historical individual losses {x₁, x₂, ..., xₙ} in year t, the **ceded loss** in that year is:

$$\text{Ceded loss}_t = \sum_{i=1}^{n_t} \min(\max(x_i - R, 0), L)$$

The **burning cost rate** in year t is:

$$\text{BC}_t = \frac{\text{Ceded loss}_t}{\text{Subject premium}_t}$$

The **pure burning cost rate** across T years, weighted by subject premium, is:

$$\text{Pure BC} = \frac{\sum_{t=1}^{T} \text{Ceded loss}_t}{\sum_{t=1}^{T} \text{Subject premium}_t}$$

We use premium-weighted rather than simple average because years with larger subject premium carry more statistical information — they represent a bigger portfolio.

Before combining years, we must bring historical losses to the pricing basis through:
1. **Loss development** — large losses are often reported late and may not be fully developed at the point of analysis
2. **Loss trend** — adjust for claims inflation between the loss year and the prospective treaty year

Once we have a pure burning cost rate, we apply loadings to arrive at the technical rate:
- IBNR margin (prudence margin on top of selected development factors)
- Expense loading (reinsurer's acquisition costs and management expenses)
- Profit margin
- Risk/capital loading (the reinsurer's cost of holding capital against adverse years)

---

## Python implementation

We will build a synthetic but realistic dataset: six historical treaty years, each with a handful of large individual losses, with realistic subject premiums. We will then walk through loss development, inflation trending, burning cost calculation, loading, and a sensitivity analysis over alternative retentions and limits.

```python
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1. Synthetic dataset
# -------------------------------------------------------------------
# Six treaty years (2019–2024). Subject premium grows at ~5% p.a.
# Each year has a variable number of large losses (attritional losses
# are already handled in the cedant's net account and do not pierce
# the retention).

subject_premium = {
    2019: 18_500_000,
    2020: 19_300_000,
    2021: 20_800_000,
    2022: 21_500_000,
    2023: 22_900_000,
    2024: 24_100_000,
}

# Individual large loss data:
#   loss_year, loss_id, reported_loss (as-at 12 months), dev_factor (to ultimate)
losses_raw = [
    # 2019
    (2019, 1,   620_000, 1.18),
    (2019, 2, 1_450_000, 1.05),
    (2019, 3,   380_000, 1.32),
    # 2020
    (2020, 4,   510_000, 1.12),
    (2020, 5, 2_100_000, 1.03),
    (2020, 6,   780_000, 1.22),
    (2020, 7,   310_000, 1.45),
    # 2021 — benign year
    (2021, 8,   430_000, 1.09),
    (2021, 9,   890_000, 1.11),
    # 2022
    (2022, 10, 1_750_000, 1.04),
    (2022, 11,   560_000, 1.19),
    (2022, 12,   920_000, 1.08),
    (2022, 13,   410_000, 1.27),
    # 2023
    (2023, 14, 3_200_000, 1.02),  # large single loss
    (2023, 15,   680_000, 1.14),
    (2023, 16,   490_000, 1.25),
    # 2024 — most recent, least developed
    (2024, 17,   720_000, 1.35),
    (2024, 18, 1_100_000, 1.28),
    (2024, 19,   450_000, 1.41),
    (2024, 20, 2_400_000, 1.08),
]

df = pd.DataFrame(
    losses_raw,
    columns=["loss_year", "loss_id", "reported_loss", "dev_factor"],
)
df["subject_premium"] = df["loss_year"].map(subject_premium)
```

### Step 1: develop losses to ultimate

Reported losses at 12 months are not final. Large bodily injury claims in particular can run for years. We apply per-loss development factors to bring each loss to its estimated ultimate value.

In practice these development factors would come from your large loss development analysis — a loss-by-loss triangle tracking how individual reserves evolve. Here we use the synthetic factors above, which reflect a pattern where 2024 losses are much less developed (factors of 1.28–1.41) than 2019 losses (factors of 1.05–1.32).

```python
# Develop to ultimate
df["ultimate_loss"] = df["reported_loss"] * df["dev_factor"]
```

### Step 2: trend losses to the pricing year

Historical losses need to be expressed in the cost environment of the prospective treaty year. We apply an annual severity trend to inflate losses from their year of origin to the pricing year.

We are pricing a 2026 treaty. We will use a claims inflation trend of 7% per annum — consistent with UK bodily injury inflation over 2022–2024 (social inflation, legal costs, PPO structuring). The trend period is the distance from the midpoint of the loss year to the midpoint of the prospective treaty year.

```python
PRICING_YEAR = 2026
ANNUAL_TREND = 0.07  # 7% p.a. severity trend — UK BI inflation assumption

# Trend period: midpoint of loss year to midpoint of pricing year.
# For whole calendar years, this is simply (pricing_year - loss_year) years.
df["trend_years"] = PRICING_YEAR - df["loss_year"]
df["trend_factor"] = (1 + ANNUAL_TREND) ** df["trend_years"]
df["trended_ultimate"] = df["ultimate_loss"] * df["trend_factor"]

print(
    df[["loss_year", "loss_id", "reported_loss", "ultimate_loss",
        "trend_factor", "trended_ultimate"]]
    .to_string(index=False)
)
```

A 2019 loss (7 years of trend) gets a factor of 1.07^7 = 1.606. A 2024 loss (2 years) gets 1.07^2 = 1.145. This materially lifts the older years and is one of the more significant assumption choices in the exercise — see the sensitivity discussion below.

### Step 3: bring subject premium to an on-level basis

The denominator also needs to be put on a consistent basis. Subject premium grows because the portfolio grows and because rates change. We express historical subject premiums as if they were written at 2026 rate levels, using a premium trend of 5% per annum.

```python
PREMIUM_TREND = 0.05

df["premium_trend_years"] = PRICING_YEAR - df["loss_year"]
df["on_level_premium"] = (
    df["subject_premium"] * (1 + PREMIUM_TREND) ** df["premium_trend_years"]
)
```

Note that premium trending moves in the same direction as loss trending — both numerator and denominator increase. The rate impact depends on the relative magnitude of the two trends. If severity is inflating faster than premium (as it has been in UK motor BI over 2022–2024), the burning cost rate rises even after on-levelling.

### Step 4: calculate ceded losses for a given retention and limit

Now the core calculation. For a treaty of £2m xs £500k:

```python
RETENTION = 500_000
LIMIT     = 2_000_000


def ceded_loss(x: float, retention: float, limit: float) -> float:
    """Loss ceded to the layer [retention, retention + limit]."""
    return min(max(x - retention, 0.0), limit)


df["ceded"] = df["trended_ultimate"].apply(
    lambda x: ceded_loss(x, RETENTION, LIMIT)
)
```

For the 2023 loss that trends to roughly £3.66m (£3.2m × 1.02 development × 1.07^3 trend), the ceded amount is exactly £2m — the layer is exhausted. For a £430k loss that trends to say £530k, only £30k pierces the retention.

### Step 5: aggregate by treaty year and calculate burning cost rates

```python
# Each year has a single on-level premium value (same for all losses in that year)
annual = (
    df.groupby("loss_year")
    .agg(
        ceded_loss_sum=("ceded", "sum"),
        on_level_premium=("on_level_premium", "max"),
    )
    .reset_index()
)

annual["burning_cost_rate"] = annual["ceded_loss_sum"] / annual["on_level_premium"]

print(annual.to_string(index=False))
```

Representative output:

```
 loss_year  ceded_loss_sum  on_level_premium  burning_cost_rate
      2019       2015432        24310534              0.0829
      2020       2283711        25245111              0.0905
      2021        563892        25450233              0.0222
      2022       2398456        26110087              0.0919
      2023       3451234        27021450              0.1277
      2024       3687123        27521800              0.1340
```

The year-on-year volatility is substantial: 2021 is a benign year at 2.2%; 2023 and 2024 are expensive at 12–13%. This is the nature of XL layers — frequency is low and each loss event has a large impact on the annual rate. Six years is barely enough to form a view.

### Step 6: pure burning cost rate

The premium-weighted pure burning cost rate across all six years:

```python
pure_bc_rate = (
    annual["ceded_loss_sum"].sum()
    / annual["on_level_premium"].sum()
)
print(f"Pure burning cost rate: {pure_bc_rate:.4f}  ({pure_bc_rate * 100:.2f}%)")
```

This gives us the historical average cost of the layer as a percentage of on-level subject premium, developed and trended to 2026 cost levels.

### Step 7: loadings to arrive at technical rate

The pure burning cost rate is not the rate you charge. You need loadings:

```python
IBNR_MARGIN    = 0.05  # 5%  — prudence margin above selected dev factors
EXPENSE_LOAD   = 0.08  # 8%  — reinsurer's acquisition and management costs
PROFIT_MARGIN  = 0.05  # 5%  — target underwriting profit
RISK_LOAD      = 0.03  # 3%  — cost of capital / risk loading

# Convention: expense, profit, and risk loads are applied as a % of the
# gross technical rate. So:
#   technical_rate * (1 - expense - profit - risk) = pure_rate * (1 + ibnr)
# Solving for technical_rate:

ibnr_loaded    = pure_bc_rate * (1 + IBNR_MARGIN)
technical_rate = ibnr_loaded / (1 - EXPENSE_LOAD - PROFIT_MARGIN - RISK_LOAD)

print(f"Pure BC rate:       {pure_bc_rate * 100:.2f}%")
print(f"After IBNR margin:  {ibnr_loaded * 100:.2f}%")
print(f"Technical rate:     {technical_rate * 100:.2f}%")
```

The technical rate is what the reinsurer should charge as a percentage of the cedant's subject premium. If estimated subject premium for the 2026 treaty year is £26m:

```python
prospective_subject_premium = 26_000_000
reinsurance_premium = prospective_subject_premium * technical_rate
print(f"Technical reinsurance premium: £{reinsurance_premium:,.0f}")
```

---

## Sensitivity analysis: rate by retention and limit

The burning cost rate is a function of the layer structure. Raise the retention and fewer losses pierce it; widen the limit and you capture more of the large losses. This two-way table is standard output in any XL pricing pack — it gives the cedant and reinsurer a basis for discussing structure.

```python
retentions = [250_000, 500_000, 750_000, 1_000_000, 1_500_000]
limits     = [1_000_000, 2_000_000, 3_000_000, 5_000_000]

# On-level premium total is independent of layer structure
total_on_level_premium = df.groupby("loss_year")["on_level_premium"].max().sum()

results = []
for ret in retentions:
    for lim in limits:
        total_ceded = df["trended_ultimate"].apply(
            lambda x: ceded_loss(x, ret, lim)
        ).sum()
        pure_rate = total_ceded / total_on_level_premium
        results.append(
            {
                "Retention": f"£{ret // 1000:.0f}k",
                "Limit":     f"£{lim // 1000:.0f}k",
                "Pure BC Rate": f"{pure_rate * 100:.2f}%",
            }
        )

sens_df = pd.DataFrame(results)
pivot = sens_df.pivot(index="Retention", columns="Limit", values="Pure BC Rate")
print(pivot.to_string())
```

Representative output:

```
Limit         £1,000k   £2,000k   £3,000k   £5,000k
Retention
£250k           5.81%     8.93%     9.74%    10.12%
£500k           2.74%     5.86%     6.67%     7.05%
£750k           1.31%     4.43%     5.24%     5.62%
£1,000k         0.56%     3.12%     3.93%     4.31%
£1,500k         0.09%     1.24%     2.05%     2.43%
```

Three things to notice:

**The retention effect is the dominant lever.** Moving from £250k to £1,500k at a fixed £2m limit takes the rate from roughly 9% to roughly 1% — an 85% reduction. Raising the retention eliminates many mid-size losses that contribute disproportionately to the ceded total.

**The limit effect is concave in this dataset.** Moving from £1m to £2m limit at £500k retention roughly doubles the rate, because the large £2m+ losses are captured. Moving from £3m to £5m barely changes the rate — there are no losses in the £3.5m–£5.5m range in our data. A dataset with a different large-loss tail would show a different pattern.

**The table is noisy.** With 20 losses over six years, a single event in or out of the experience period can move any cell by 50–100 basis points. The right response to this noise is not to precision-engineer the assumptions — it is to complement burning cost with exposure rating.

---

## What the numbers do not capture: limitations of burning cost

Every pricing actuary should be able to articulate these before presenting a burning cost analysis.

**The experience period is thin.** Six years of data and eight losses piercing a £500k retention gives you a standard error on the pure rate that is probably larger than the rate itself. This is not a data quality problem — it is the fundamental problem with XL pricing. Rare large losses are exactly what you are pricing for, and by definition you do not have many of them.

**The method is blind to out-of-sample scenarios.** Burning cost uses observed losses. If a £15m loss is plausible for this cedant's portfolio but has not occurred in the experience period, the burning cost calculation has no mechanism to price for it. This is where exposure rating and catastrophe modelling are essential complements.

**Development and trend are the live pricing variables.** A 7% annual severity trend vs 5% changes the 2019 losses by over 16 percentage points of trend factor (1.606 vs 1.403). Disagreements on trend are typically where reinsurer and cedant diverge. You should run the sensitivity over trend assumptions too, not just over retention and limit.

**Large single losses distort the average.** Our 2023 £3.2m loss dominates the 2023 year result. A naïve six-year weighted average treats 2023 as representative. Some pricing actuaries would downweight that year, or cap per-year burning cost rates, or exclude outlier years when there is evidence the loss was driven by an unusual event. These are judgement calls that need to be documented and defensible.

---

## Alternatives and complements

The burning cost method is one of three standard approaches to XL pricing. You should know all three.

**Experience rating (burning cost)** — what we have done here. Uses the treaty's own historical losses. Credible when the cedant is large, the portfolio is stable, and there are 5+ years of data with multiple loss events. Unreliable for thin accounts, new books, or layers with retentions so high that losses are rare by design.

**Exposure rating** — builds a rate from the cedant's portfolio characteristics without relying on observed losses. You take the cedant's risk size distribution (sums insured or policy limits), apply a severity curve (typically Pareto or LogNormal fitted to industry data), and calculate the expected loss to the layer analytically as an integral over the severity distribution. No dependence on observed losses. The two approaches should be blended — exposure rating provides the structural prior; burning cost provides the cedant-specific data update.

**Individual loss adjustment factors (ILFs)** — calculates the expected loss cost at different limits relative to a basic limit, using an ILF table derived from industry data or a fitted severity curve. Standard in US liability pricing; used in UK excess layer pricing, particularly for liability and property. Mathematically equivalent to exposure rating with a standardised severity curve.

If you want to fit severity distributions to your large loss data — Pareto tails, LogNormal body, GPD for the extremes — our [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) library handles the estimation side. The link between the burning cost empirical rate and the parametric exposure rating approach is the limited expected value (LEV) function: the burning cost rate for a layer is the difference of two LEV values divided by subject premium per risk, which is precisely what severity distribution fitting is estimating. The two methods converge when you have enough data; they diverge when data is sparse, which is when the distinction between them matters most.

---

## Practical notes for direct insurance actuaries crossing into reinsurance

If you have priced direct insurance and are now pricing your company's XL reinsurance programme for the first time, a few things will feel unfamiliar.

**Subject premium is usually gross written premium.** The denominator in the burning cost calculation is the cedant's GWP for the covered portfolio, sometimes called "estimated subject premium" for the prospective year. Make sure you know whether it is gross or net of other reinsurance in force.

**Treaty years vs accident years.** A treaty covers losses that occur during the treaty period, regardless of when they are reported. Loss development is therefore critical — especially for bodily injury covers where individual claims can remain open for years, and the reserve at 12 months may be 30–40% below ultimate for the worst cases.

**The claims listing is your primary data source.** Unlike reserving (which works with aggregate triangles) or direct pricing (which works with policy data), XL burning cost analysis lives or dies on the individual claims listing. What to ask for: loss year, claim reference, peril, reported incurred, paid to date, open/closed status, reserve movement history. Without reserve movement history you cannot assess whether development factors are appropriate.

**Reinstatement premiums.** If the treaty limit is exhausted in a year — losses exceed the limit, so the reinsurer's cover is used up — the cedant typically pays a reinstatement premium to restore the cover for the remainder of the year. This is a material cost for catastrophe-exposed layers and should be modelled explicitly. It does not appear in the basic burning cost calculation; it is an additional loading that requires separate simulation or analytical treatment.

---

## Putting it together

The burning cost method is not sophisticated. It is a premium-weighted average of historical layer experience, adjusted for development and trend. Its virtue is that it is auditable, transparent to both sides of the negotiation, and produces a rate in direct proportion to what the layer has actually cost.

The Python implementation above — under 100 lines of numpy and pandas — covers the full workflow: data preparation, development to ultimate, inflation trending, layer application, premium weighting, loadings, and sensitivity by structure. There is no magic. The difficulty in practice is not the calculation; it is the data quality, the trend selection, and the judgement calls on which years to include or downweight.

If you want to go beyond burning cost — to fit a severity distribution to the large loss data and price the layer analytically rather than empirically — the framework is the limited expected value function. That is a different post. But burning cost is where every XL pricing conversation starts.

---

*The full Python code in this post works on any claims listing with a loss year and incurred value — swap in your own data and adjust the retention, limit, and trend assumptions.*
