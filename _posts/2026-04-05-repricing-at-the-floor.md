---
layout: post
title: "Repricing at the floor: how to find headroom in UK motor without triggering Consumer Duty"
date: 2026-04-05
categories: [motor, regulation, trend, monitoring, credibility]
tags: [consumer-duty, PRIN-2A, FCA, UK-motor, NCR, EY, fair-value, insurance-trend, insurance-monitoring, insurance-credibility, loss-cost-trend, AE-ratio, buhlmann-straub, rate-adequacy, pricing, python]
description: "UK motor NCR is at 111% (EY Q4 2024). The market is at or near technical floor. Here is how to identify underpriced segments using loss cost trending, A/E monitoring, and Bühlmann-Straub credibility — while building the fair value documentation Consumer Duty requires before you touch a rate."
author: Burning Cost
---

EY's Motor Insurance Results Tracker has UK motor NCR at 111% for Q4 2024. The market got there through a brutal two-year repricing cycle driven by parts inflation, paint and materials costs, and courtesy car exposure from supply chain disruption. Most carriers have largely repriced the book — but "largely" is doing a lot of work in that sentence.

The problem now is that the easy headroom is gone. The segments where you were obviously underpriced by 20% have been addressed. What remains is a harder question: where, specifically, are you still underpriced, and by how much, and can you demonstrate to the FCA that your rate increases in those segments represent fair value rather than a selective squeeze on captive customers?

This post works through that problem using three of our libraries: `insurance-trend` for loss cost trending and rate adequacy, `insurance-monitoring` for A/E ratio surveillance by segment, and `insurance-credibility` for Bühlmann-Straub credibility weighting when you are blending thin portfolio experience with market benchmarks.

The regulatory constraint runs through every section. Consumer Duty came into force on 31 July 2023 for open products. PRIN 2A.3 is the fair value outcome: the price a customer pays must be reasonable relative to the overall benefits. Before you touch a rate in a segment where customers are likely to be less price-sensitive — older vehicles, higher NCD bands, rural postcodes — you need documented fair value evidence that your rate change is driven by genuine cost developments, not margin recovery at the expense of a captive segment.

---

## The rate adequacy question

A combined ratio of 111% does not mean every segment is equally inadequate. The average hides enormous dispersion. Young drivers (17–24, comprehensive) have repriced aggressively. Telematics books have better data and have followed experience more tightly. The segments that are most likely to still be under-earning are those where:

1. You have historically been reluctant to price to experience because loss volumes are thin.
2. The rating factors you use are lagged — garaging postcode, for example, captures 2019 flood risk, not 2024.
3. You have grandfathered renewal customers who last shopped more than three years ago.

The first step is establishing where your loss cost trend has diverged from your filed rates. That is a `LossCostTrendFitter` job.

```python
pip install insurance-trend
```

```python
from insurance_trend import LossCostTrendFitter, BreakEventCalendar

# Quarterly motor data — 2020Q1 through 2024Q4
# claim_counts, earned_exposure, total_paid are lists or arrays
fitter = LossCostTrendFitter(
    periods=[
        "2020Q1", "2020Q2", "2020Q3", "2020Q4",
        "2021Q1", "2021Q2", "2021Q3", "2021Q4",
        "2022Q1", "2022Q2", "2022Q3", "2022Q4",
        "2023Q1", "2023Q2", "2023Q3", "2023Q4",
        "2024Q1", "2024Q2", "2024Q3", "2024Q4",
    ],
    claim_counts=[
        820, 790, 760, 810,   # 2020 — COVID suppression visible
        870, 860, 890, 910,   # 2021 — recovery
        940, 960, 980, 970,   # 2022 — parts inflation onset
        950, 940, 930, 920,   # 2023 — frequency softening
        900, 895, 885, 880,   # 2024
    ],
    earned_exposure=[
        9800, 9750, 9700, 9800,
        9900, 9950, 10000, 10050,
        10100, 10150, 10200, 10180,
        10150, 10100, 10080, 10060,
        10000, 9980, 9960, 9940,
    ],
    total_paid=[
        39_400_000, 38_000_000, 36_600_000, 39_000_000,
        43_500_000, 43_000_000, 44_500_000, 46_000_000,
        49_700_000, 52_000_000, 54_400_000, 54_100_000,
        54_800_000, 55_100_000, 55_500_000, 55_900_000,
        56_700_000, 57_200_000, 57_800_000, 58_400_000,
    ],
    periods_per_year=4,
)

result = fitter.fit()
print(result.summary())
```

The `result.summary()` output will give you frequency trend, severity trend, and combined loss cost trend with 95% bootstrap confidence intervals. For a typical UK motor book over this window you would expect to see severity trend in the range of +8% to +12% per annum (reflecting the parts and labour cost environment) with frequency trending flat to slightly negative.

The rate adequacy gap is the difference between cumulative earned rate change and cumulative loss cost trend since your last full repricing exercise. If your loss cost is trending at +9% per annum and you have earned rate increases of +7% per annum over the same period, you are falling behind by approximately 2 points per annum — compounding.

`BreakEventCalendar` maps detected structural breaks in your trend to known UK market events: the October 2022 Ogden rate review, the May 2023 whiplash reforms, the various supply chain shocks.

```python
from insurance_trend import BreakEventCalendar

# result.frequency.changepoints contains integer indices into the periods list
# Convert to period labels before passing to BreakEventCalendar
freq = result.frequency
break_periods = [freq.periods[i] for i in freq.changepoints]

calendar = BreakEventCalendar()
attribution = calendar.attribute(break_periods)
print(attribution.summary())
```

This gives you the audit trail that a fair value assessment needs: your trend is +X%, of which Y% is attributable to documented market events, and Z% is unexplained residual that requires further investigation before you use it to justify a rate increase.

---

## Detecting underpriced segments: A/E monitoring

Portfolio-level trend is useful for setting an overall rate adequacy target. Segment-level A/E ratios are what you need to identify *where* to act.

The A/E ratio for a frequency model is actual claim count divided by expected claim count (predicted frequency × earned exposure). A/E > 1.0 means the model is underpredicting — you are underpriced relative to actual experience. Sustained A/E > 1.05 over three or more quarters in a segment is a signal that requires documented response.

```python
pip install insurance-monitoring
```

```python
import polars as pl
from insurance_monitoring import ae_ratio

# Policy-level data with model predictions
# actual_claims: int — whether this policy had a claim this period
# predicted_freq: float — model output (claims per car-year)
# earned_exposure: float — fraction of year this policy was on risk
# segment: str — rating factor cell (e.g. vehicle group × NCD band)

ae_by_segment = ae_ratio(
    actual=policy_df["actual_claims"],
    predicted=policy_df["predicted_freq"],
    exposure=policy_df["earned_exposure"],
    segments=policy_df["segment"],
)

# ae_by_segment is a Polars DataFrame with columns:
# segment | actual | expected | ae_ratio | n_policies

# Identify segments with sustained A/E > 1.05
elevated = ae_by_segment.filter(pl.col("ae_ratio") > 1.05)
print(elevated.sort("ae_ratio", descending=True))
```

Running this quarterly and tracking the time series of A/E by segment is the core of segment-level monitoring. What you are looking for is persistence — a single quarter at A/E 1.08 in a small segment might be noise. Three consecutive quarters in the same segment is a finding.

For segments where A/E has been elevated for two or more consecutive quarters, `ae_ratio_ci` gives you a confidence interval:

```python
from insurance_monitoring import ae_ratio_ci

# For a specific underperforming segment
ci = ae_ratio_ci(
    actual=segment_df["actual_claims"],
    predicted=segment_df["predicted_freq"],
    exposure=segment_df["earned_exposure"],
    alpha=0.05,   # 95% CI
)
print(f"A/E: {ci['ae']:.3f}  95% CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
```

If the lower bound of the CI exceeds 1.0, the segment is genuinely underpriced at the 95% confidence level. That is the evidentiary threshold for escalation to pricing committee.

---

## Credibility weighting: blending thin experience with market benchmarks

The segments that are most often underpriced are precisely the segments where you have the least data. A rural postcode area with 200 earned vehicle-years per quarter has noisy A/E estimates — you cannot price to raw own-experience without introducing enormous random variation.

Bühlmann-Straub credibility gives you the correct blend: weight own experience against the portfolio (or market) mean in proportion to how much your own experience can be trusted, given volume.

```python
pip install insurance-credibility
```

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# Panel data: one row per (territory, year)
# loss_rate = observed loss ratio (total paid / earned premium)
# exposure = earned vehicle-years

territory_panel = pl.DataFrame({
    "territory": [
        "North West", "North West", "North West",
        "South East", "South East", "South East",
        "Scotland",   "Scotland",   "Scotland",
        "Wales",      "Wales",      "Wales",
        "East Anglia","East Anglia","East Anglia",
    ],
    "year": [2022, 2023, 2024] * 5,
    "loss_rate": [
        0.72, 0.78, 0.81,   # North West — deteriorating
        0.68, 0.71, 0.73,   # South East — stable
        0.81, 0.79, 0.83,   # Scotland — high base
        0.65, 0.68, 0.70,   # Wales — relatively benign
        0.74, 0.80, 0.85,   # East Anglia — sharply deteriorating
    ],
    "exposure": [
        12_000, 11_800, 11_500,
        28_000, 28_500, 29_000,
        6_500,  6_200,  6_000,
        4_800,  4_700,  4_600,
        3_200,  3_100,  3_000,  # thin — low credibility
    ],
})

bs = BuhlmannStraub()
bs.fit(
    territory_panel,
    group_col="territory",
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)
print(bs.summary())
```

The summary output shows you the Bühlmann k (the noise-to-signal ratio) and the credibility factor Z for each territory. A territory like East Anglia with 3,000 vehicle-years per year will have a credibility factor substantially below 1.0 — perhaps 0.4 to 0.6 — meaning its credibility-weighted loss rate is a 40–60% blend of its own experience and the portfolio mean.

This is the correct way to handle the thin-data problem. Raw own-experience would suggest East Anglia needs a +15% rate increase based on the 2022–2024 trend. Bühlmann-Straub might give you +8% after credibility weighting. Both numbers need to be in your rate filing with their supporting evidence. The FCA does not object to charging East Anglia customers more than the portfolio average if you have credible evidence of higher loss costs. It objects to using that logic as cover for capturing value from customers who have no competitive alternative.

---

## The Consumer Duty constraint: documenting fair value before you act

Consumer Duty (PRIN 2A.3) requires that the price charged must be reasonable relative to the overall benefits of the product. For pricing actuaries, that creates three obligations that sit alongside the rate adequacy analysis above.

**First: your rate change must be driven by cost, not by retention curve exploitation.** The FCA's position, set out in its January 2025 review of fair value outcomes, is that using price optimisation to charge customers more because they are less likely to switch is not consistent with PRIN 2A.3. This is not new — the GIPP rules that took effect on 1 January 2022 already banned loyalty penalties for home and motor. Consumer Duty extends the principle: you cannot charge a captive segment more than a new customer for equivalent risk, and you cannot price a segment at a premium simply because its members are price-insensitive.

The practical implication: any segment where your rate increase is larger than your loss cost trend justification should be documented and reviewed by your fair value governance function before implementation.

**Second: you need contemporaneous documentation.** The FCA expects firms to maintain fair value assessments that are updated when material rate changes are made, not retrospectively reconstructed after a supervisory request. The monitoring outputs above — the A/E ratios, the trend analysis, the credibility-weighted indication — are your primary evidence. Keep them. Date them.

**Third: the fair value assessment must consider the full product.** PRIN 2A.3 applies to the price/benefit package, not to the premium in isolation. If you are repricing a segment upward while simultaneously narrowing claims settlement terms or adding excess clauses, the combined effect must be assessed. This is where actuarial and product governance need to work together.

A minimal fair value evidence pack for a segment rate change should contain:

- Trend analysis showing loss cost development over the most recent four to eight quarters, with confidence intervals.
- Segmented A/E analysis showing that the segment in question has genuinely underperformed relative to model expectations.
- Credibility-weighted rate indication with the blending assumptions documented.
- Competitor benchmark — at least three publicly available comparative quotes for representative risks in the segment.
- Sign-off from your Chief Actuary or Pricing Director that the rate change is cost-led, not retention-led.

None of this is onerous if the monitoring infrastructure is already in place. It is onerous if you are trying to reconstruct the evidence after the fact.

---

## Putting it together: a rate adequacy workflow

The workflow for a quarterly rate adequacy review in the current market looks like this:

1. Run `LossCostTrendFitter` on the whole book and on major segments. Establish the rolling four-quarter trend in frequency and severity. Compare against your last filed rates.

2. Run `ae_ratio` segmented by your primary rating factors. Flag any cell where A/E has been above 1.05 for two or more consecutive quarters. These are your repricing candidates.

3. For each flagged segment, run `BuhlmannStraub` to produce a credibility-weighted rate indication. Thin segments get blended toward the portfolio mean. Credible segments (typically Z > 0.7) can be priced more aggressively to own experience.

4. For each proposed rate change above +5%, produce a fair value evidence pack before implementation.

5. Document everything with dates. The FCA will ask for contemporaneous evidence.

The market is at or near technical floor. NCR of 111% is not sustainable, and there is pressure across the market to restore margins. But the firms that do it cleanly — with credible, documented cost-led evidence for every rate change — are building a supervisory relationship that will matter when the next Consumer Duty review cycle arrives. The firms that grab margin opportunistically in captive segments are not.

---

## Libraries used

- [`insurance-trend`](https://github.com/burning-cost/insurance-trend) — `LossCostTrendFitter`, `BreakEventCalendar`
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — `ae_ratio`, `ae_ratio_ci`
- [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) — `BuhlmannStraub`

All three are MIT-licensed, pure Python, and Polars-native with pandas interoperability.
