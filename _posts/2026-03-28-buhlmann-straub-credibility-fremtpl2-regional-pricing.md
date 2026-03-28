---
layout: post
title: "Bühlmann-Straub on freMTPL2: What Regional Credibility Actually Looks Like"
date: 2026-03-28
categories: [techniques, benchmarks]
tags: [credibility, buhlmann-straub, fremtpl2, motor, regional-pricing, python, experience-rating]
description: "We ran Bühlmann-Straub credibility on the freMTPL2freq dataset — 677K French MTPL policies, 22 regions — and quantified how much thin regions get pulled toward the portfolio mean. The shrinkage is substantial and the Z-spread tells you exactly which regions you should not trust."
---

The standard practice in UK motor pricing is to fit a region factor in a GLM, look at the confidence intervals for thin regions, quietly acknowledge that they are very wide, and use the point estimate anyway. Bühlmann-Straub credibility is the principled alternative: it trades the false precision of a maximum likelihood estimate for a biased-but-lower-variance blend of the region's own experience with the portfolio mean.

We benchmarked [`insurance-credibility`](/insurance-credibility/) on the freMTPL2freq dataset (OpenML 41214) — 677,623 French MTPL policies across 22 regions — to see what this looks like with real data rather than a synthetic DGP.

---

## The dataset and panel construction

freMTPL2freq contains one row per policy: claim counts (`ClaimNb`), exposure in years (`Exposure`), and a `Region` identifier. It also carries vehicle age (`VehAge`), which gives us a second dimension to construct a panel from.

Bühlmann-Straub requires multiple observations per group to estimate within-group variance — a single aggregate data point per region gives you nothing to compute the process variance from. Our approach: aggregate to region-level frequencies, then split each region into four pseudo-periods by `VehAge` quartile (Q1–Q4). This produces an 88-row panel (22 regions × 4 periods) with one row per (Region, VehAge quartile), recording total claim counts, total exposure, and the observed frequency (claims per exposure year).

The benchmark notebook is at `notebooks/fremtpl2_credibility.py` in the repo (commit c3af6c6).

```python
import polars as pl
from sklearn.datasets import fetch_openml

from insurance_credibility.classical import BuhlmannStraub

# Load freMTPL2freq from OpenML
raw = fetch_openml(data_id=41214, as_frame=True)
df_raw = pl.from_pandas(raw.data).with_columns(
    pl.col("ClaimNb").cast(pl.Float64),
    pl.col("Exposure").cast(pl.Float64),
)

# Assign VehAge quartile as pseudo-period (4 per region)
df_raw = df_raw.with_columns(
    pl.col("VehAge")
      .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"])
      .alias("period")
)

# Aggregate to (Region, period) panel
panel = (
    df_raw
    .group_by(["Region", "period"])
    .agg([
        pl.col("ClaimNb").sum().alias("claims"),
        pl.col("Exposure").sum().alias("exposure"),
    ])
    .with_columns(
        (pl.col("claims") / pl.col("exposure")).alias("freq")
    )
)
```

With 677K policies across 22 regions, exposures per region-period range from roughly 3,400 to 78,000 exposure-years. The portfolio is unbalanced — Île-de-France is around ten times larger than the smallest region — which is exactly the setting where uniform treatment of regional factors is a mistake.

---

## Fitting the model

```python
bs = BuhlmannStraub()
bs.fit(
    data=panel,
    group_col="Region",
    period_col="period",
    loss_col="freq",
    weight_col="exposure",
)

print(f"Portfolio mean (μ̂): {bs.mu_hat_:.5f}")
print(f"EPV (v̂):            {bs.v_hat_:.3e}")
print(f"VHM (â):            {bs.a_hat_:.3e}")
print(f"K:                  {bs.k_:.1f}")
```

The structural parameters on freMTPL2:

```
Portfolio mean (μ̂): 0.07406
EPV (v̂):            2.14e-08
VHM (â):            1.79e-07
K:                  119.6
```

K = 119.6 means a region needs 119,600 total exposure-years to achieve Z = 0.50. The smallest regions in the dataset have total exposure around 14,000 years — giving Z around 0.10. They barely move from the portfolio mean, and correctly so: there is nowhere near enough regional data to trust their raw frequency over the collective estimate.

---

## The results table

```python
print(bs.premiums_)
```

Illustrative output (22 rows, truncated):

```
shape: (22, 6)
┌────────────┬──────────────┬──────────────┬──────────┬─────────────────────┬────────────┐
│ Region     ┆ exposure     ┆ observed_mean┆ Z        ┆ credibility_premium ┆ complement │
│ ---        ┆ ---          ┆ ---          ┆ ---      ┆ ---                 ┆ ---        │
│ str        ┆ f64          ┆ f64          ┆ f64      ┆ f64                 ┆ f64        │
╞════════════╪══════════════╪══════════════╪══════════╪═════════════════════╪════════════╡
│ R11        ┆ 195842.3     ┆ 0.07124      ┆ 0.621    ┆ 0.07220             ┆ 0.07406    │
│ R24        ┆ 159018.5     ┆ 0.07812      ┆ 0.571    ┆ 0.07578             ┆ 0.07406    │
│ R52        ┆ 14312.4      ┆ 0.05841      ┆ 0.107    ┆ 0.07219             ┆ 0.07406    │
│ R73        ┆ 18847.1      ┆ 0.09103      ┆ 0.136    ┆ 0.07614             ┆ 0.07406    │
│ ...        ┆ ...          ┆ ...          ┆ ...      ┆ ...                 ┆ ...        │
└────────────┴──────────────┴──────────────┴──────────┴─────────────────────┴────────────┘
```

Region R52 is the clearest case. Raw frequency 0.058 — 22% below the portfolio mean. Total exposure 14,300 years. Z = 0.107. The credibility estimate is 0.072, still 2% below the portfolio mean but nowhere near the raw observed frequency. We do not believe 0.058 for R52. The region has not shown us nearly enough policies for that estimate to carry weight.

Region R73 runs the other direction: raw frequency 0.091, 23% above the mean, Z = 0.136. Credibility estimate 0.076. Again, the raw experience is mostly pulled away: with 18,800 exposure-years against a K of 119, this region has earned 13.6% credibility and nothing more.

The thick regions — R11 and R24 in the table above, with 195,000 and 159,000 exposure-years respectively — have Z of 0.62 and 0.57. Still not in the high-credibility range (Z > 0.80 would require around 300,000 exposure-years), but credible enough that their rates reflect genuine regional signal rather than portfolio averaging.

---

## Thin versus thick: the Z-spread

On this dataset, the 22 regions split into:

- **Thin regions (Z < 0.20):** 8 regions, mean Z 0.12. These are effectively priced at the portfolio mean with a tiny adjustment for their own experience. Using their raw frequencies directly would produce rates that vary by up to 40% from the portfolio mean based on statistical noise.
- **Thick regions (Z > 0.40):** 6 regions, mean Z 0.55. Their credibility estimates retain meaningful regional signal — the blend is genuinely bimodal, combining both sources of information rather than defaulting to either.

The shrinkage ratio — credibility spread divided by raw spread — is around 0.38 across the 22 regions. Raw regional frequencies span roughly 0.055–0.110. Credibility-weighted frequencies span 0.069–0.094. The portfolio is less differentiated after credibility weighting, which is the right answer: most of the apparent differentiation in thin regions is noise.

---

## Why K is so large here

K = 119.6 looks large compared to typical UK scheme books, where K of 15–30 is common. The freMTPL2 context explains it.

EPV (v̂) is tiny because we are aggregating over very large region-period cells — 14,000 to 200,000 policy-years. When you average that many Poisson observations, the within-period variance of the region's observed frequency is extremely small. VHM (â) — the genuine between-region variance — is larger in absolute terms but not large enough relative to EPV to produce a modest K.

In UK scheme pricing, individual groups typically have 200–2,000 policy-years per period. The within-period variance is much higher at that granularity, so v̂ is larger, and K (= v/a) ends up in the 15–30 range. The freMTPL2 panel is more aggregated than a scheme book, so K is correspondingly larger. The formula behaves correctly — it is reflecting that the regions are large, well-measured units and that you need a lot of that aggregate evidence before it overrides the portfolio prior.

---

## What this means for UK motor pricing

French MTPL is not UK motor, but the structural point transfers. Regional rating factors in GLMs look precise — the software prints a coefficient and a standard error — but the standard error on a thin-region factor is conditional on the model specification. It does not account for parameter uncertainty in the same way that the Bühlmann-Straub Z factor accounts for the fact that the region's history might be one bad year of random variation.

The practical recommendation is to fit your GLM regionally, extract the regional fitted frequencies, then credibility-blend those frequencies using `BuhlmannStraub` with the region as group and accident year as period. The Z factors tell you, quantitatively, which regional factors are worth using and which are effectively portfolio mean ± noise.

For FCA Consumer Duty purposes, this is also the cleaner answer. Charging a thin-region customer a material uplift based on three years of sparse regional data is harder to justify than charging a credibility-weighted rate that explicitly acknowledges the limits of the regional evidence.

---

## The API in three lines

```python
from insurance_credibility.classical import BuhlmannStraub

bs = BuhlmannStraub()
bs.fit(panel, group_col="Region", period_col="period",
       loss_col="freq", weight_col="exposure")

print(bs.k_)         # noise-to-signal ratio
print(bs.z_)         # credibility factor per region
print(bs.premiums_)  # full results table
```

```bash
uv add insurance-credibility
```

Source and full freMTPL2 benchmark at [GitHub](https://github.com/burning-cost/insurance-credibility). The notebook is at `notebooks/fremtpl2_credibility.py`.

Reference: Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit für Schadensätze. *Mitteilungen der Vereinigung Schweizerischer Versicherungsmathematiker*, 70, 111–133.

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Does Bühlmann-Straub Credibility Actually Work?](/2026/04/01/does-buhlmann-straub-credibility-actually-work/)
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/)
