---
layout: post
title: "Your Territory Model Ignores Spatial Autocorrelation"
date: 2028-04-15
categories: [spatial, territory, pricing]
tags: [spatial, territory, BYM2, ICAR, moran, pymc, insurance-spatial, motor, python, polars, uk-motor, postcode]
description: "BYM2 spatial smoothing with insurance-spatial borrows strength from neighbouring postcodes to produce stable territory factors."
---

The standard approach to postcode-level territory rates is to compute an observed-to-expected ratio per sector, apply some credibility blending toward the book mean, and call it done. It is tractable. Actuaries understand it. Rating engines can consume it.

It is also statistically naive, and it costs you predictive accuracy in ways that are straightforward to quantify.

The problem: treating postcode sectors as independent ignores the fact that geography has structure. A sector in south Leicester does not have independent claim risk from its neighbours in north Harborough and east Blaby. Their risk profiles are correlated. Motor theft patterns, road network density, commuting distance to employment centres — these vary smoothly across space, not sector-by-sector. When you estimate rates independently per sector, you throw away every scrap of information that the neighbours carry.

BYM2 — the Besag-York-Mollié 2 model — is the principled fix. It borrows strength across the spatial graph, smoothing toward neighbours rather than toward the uninformative portfolio mean. The rho parameter tells you how much of the residual geographic variance is spatially structured versus unstructured noise. The posterior per sector is narrower than the independent estimate, which translates directly into more stable territory factors year-on-year.

This is a tutorial for teams who already use `insurance-spatial` for production territory factors and want to understand what the diagnostics are telling them, or for teams who are still doing independent sector rates and want a concrete reason to change that.

```bash
uv pip install "insurance-spatial[nutpie]"
```

---

## What independent sector rates actually assume

The credibility-weighted O/E approach treats each postcode sector as a single observation drawn from a normal distribution around the book mean. The credibility weight Z_i = w_i / (w_i + k) determines how much the sector's own experience dominates. For sectors with thin exposure, Z is small and the estimate pulls toward the mean.

That shrinkage target is the problem. Shrinking toward the portfolio mean makes statistical sense if you have no other information about where a sector sits in the risk distribution. But you do have that information: you know where the sector is, and you know what its neighbours look like. A sector in inner Manchester with thin data should shrink toward its neighbours — which are also urban, congested, and high-theft — not toward the national average, which is diluted by rural Lincolnshire.

This is not an abstract criticism. On a typical UK motor book, the difference between independent credibility blending and BYM2 is measurable in RMSE at sector level and in territory factor volatility across annual refreshes.

---

## Pre-fit: measure whether spatial autocorrelation is present

Before fitting anything, run Moran's I on the log O/E ratios. This takes two minutes and tells you whether the spatial model is doing real work or just adding complexity.

```python
import polars as pl
import numpy as np
from insurance_spatial.adjacency import from_geojson
from insurance_spatial.diagnostics import moran_i

# sector_df has columns: postcode_sector, observed_claims,
# expected_claims (from the Stage 1 CatBoost model)
sector_df = pl.read_parquet("sector_residuals.parquet")

observed  = sector_df["observed_claims"].to_numpy().astype(np.int64)
expected  = sector_df["expected_claims"].to_numpy()
sectors   = sector_df["postcode_sector"].to_list()

# Build adjacency from postcode sector polygons
adj = from_geojson(
    "postcode_sectors.geojson",
    area_col="PC_SECTOR",
    connectivity="queen",
    fix_islands=True,
    subset=sectors,         # restrict to sectors in the model
)

log_oe = np.log(np.where(expected > 0, observed / expected, 1.0))

test = moran_i(log_oe, adj, n_permutations=999)
print(f"Moran's I : {test.statistic:.3f}")
print(f"Z-score   : {test.z_score:.2f}")
print(f"p-value   : {test.p_value:.3f}")
print(test.interpretation)
```

```
Moran's I : 0.318
Z-score   : 14.7
p-value   : 0.001
Significant positive spatial autocorrelation (I=0.318, p=0.001).
Nearby areas have similar values. Spatial smoothing is warranted.
```

Moran's I of 0.318 is a strong signal. For reference, I = 0 is complete spatial randomness; I = 1 is perfect positive autocorrelation. A value of 0.3 means roughly 30% of the variance in log O/E can be predicted from the spatial neighbourhood structure — before fitting any model. That is information you are currently discarding.

The permutation test reshuffles the observed values across areas 999 times and asks how often the reshuffled I equals or exceeds the observed I. The p-value of 0.001 means this happened once in 999 permutations. The spatial structure in your residuals is not a sampling artefact.

If Moran's I is not significant — which does happen on books where Stage 1 CatBoost has already absorbed geographic signal through IMD scores, urban density, or crime rate as explicit features — note it in the model documentation and proceed anyway. BYM2 degrades gracefully to simple credibility blending when the spatial structure is absent; rho will be near zero and the ICAR component will contribute almost nothing. The result is valid, just not as informative as on a book with genuine spatial structure.

---

## Fitting BYM2

```python
from insurance_spatial import BYM2Model

model = BYM2Model(
    adjacency=adj,
    draws=1000,
    chains=4,
    target_accept=0.9,
    tune=1000,
)

result = model.fit(
    claims=observed,
    exposure=expected,   # Stage 1 expected claims, not raw car-years
    random_seed=42,
)
```

The `exposure` argument is the Stage 1 expected claims — what the CatBoost frequency model predicted for each sector given its vehicle and driver mix, absent any territory effect. BYM2 is fitting the residual geographic variation: the O/E ratio after controlling for everything the base model knows.

With nutpie installed, this runs in 20-30 minutes for the full UK postcode sector graph (~9,000 nodes, filtered to ~7,800 after dropping sectors below 50 earned car-years). Without nutpie, budget 90 minutes on the same data. Quarterly territory refreshes at this cadence are not a problem.

---

## Convergence: check before using anything

```python
diag = result.diagnostics()

print(f"Max R-hat     : {diag.convergence.max_rhat:.4f}")     # want < 1.01
print(f"Min ESS bulk  : {diag.convergence.min_ess_bulk:.0f}") # want > 400
print(f"Min ESS tail  : {diag.convergence.min_ess_tail:.0f}") # want > 400
print(f"Divergences   : {diag.convergence.n_divergences}")    # want 0
```

```
Max R-hat     : 1.0047
Min ESS bulk  : 612
Min ESS tail  : 487
Divergences   : 0
```

R-hat below 1.01 on all parameters, ESS well above 400, zero divergences. This run converged. An R-hat above 1.01 means the four chains explored different parts of the posterior and their estimates cannot be trusted — re-run with `tune=2000` before proceeding. Persistent divergences after increasing tune often indicate a sector pulling hard against the spatial prior because its O/E ratio is extreme; identify it via the widest posteriors and investigate whether it is a genuine outlier or a data quality issue.

---

## The rho parameter

```python
print(diag.rho_summary)
```

```
parameter | mean  | sd    | q025  | q975
rho       | 0.724 | 0.108 | 0.504 | 0.901
```

Rho is the proportion of residual geographic variance that is spatially structured — that is, explained by the ICAR neighbourhood component — versus unstructured noise. A posterior mean of 0.72 means roughly 72% of the sector-level variation in log O/E is spatially smooth. The ICAR component is doing the heavy lifting; the unstructured component is absorbing the genuine sector-specific deviations that cannot be explained by neighbourhood.

This matters for interpretation. When rho is high, sectors with thin data are being smoothed substantially toward their spatial neighbourhood — a much more informative prior than the portfolio mean. The BYM2 estimate for a sparse rural sector in the Scottish Borders is being pulled toward other Scottish Borders sectors, which share similar road networks, population density, and theft risk, not toward a national mean dominated by urban English data.

A rho posterior below 0.2 is a signal that your Stage 1 model has already absorbed most of the geographic signal. That is not a failure — it means the covariates CatBoost is using (IMD score, urban classification, crime rate) are already capturing what BYM2 would capture spatially. The territory factors will still be valid, but they will look more like credibility-weighted sector loss ratios than smooth spatial surfaces. Adding area-level covariates explicitly to the spatial model is the next step in that case.

---

## Extracting relativities

```python
rels = result.territory_relativities(credibility_interval=0.95)
print(rels.head(5))
```

```
area   | relativity | lower | upper | ln_offset
AB10 1 |      1.131 | 1.049 | 1.219 |    0.1231
AB10 2 |      1.085 | 1.008 | 1.167 |    0.0816
AB10 3 |      0.957 | 0.875 | 1.047 |   -0.0443
BN1 1  |      1.294 | 1.201 | 1.394 |    0.2578
BN1 2  |      1.263 | 1.177 | 1.356 |    0.2336
```

The `relativity` is normalised to geometric mean 1.0. The `ln_offset` is ready to use as a log-linear predictor in a rating engine. Join it back to the policy data:

```python
import polars as pl

df_rated = (
    df.join(
        rels.select(["area", "ln_offset"]).rename({"area": "postcode_sector"}),
        on="postcode_sector",
        how="left",
    )
    .with_columns(
        pl.col("ln_offset").fill_null(0.0)
    )
)
```

Sectors not in the fitted model — those filtered out for very thin exposure — get `ln_offset = 0.0`, which assigns them the portfolio relativity. That is the conservative choice, and the right one.

---

## Why the credibility intervals are the actual product

Look at BN1 1 and BN1 2 above. They are adjacent sectors in central Brighton. Their point estimates are 1.294 and 1.263 — similar, as you would expect from neighbours. Their credibility intervals are tight: roughly 16 percentage points from lower to upper on both. These are sectors with substantial exposure; the posterior is well-informed.

Now look at what independent O/E credibility would give you. Suppose AB10 3 had only 24 policies and 2 claims in the last year. The raw O/E is 0.89 (2 claims observed, 2.25 expected). Credibility weight at 24 policies is maybe 0.30 on a motor book. You would blend: 0.30 × 0.89 + 0.70 × 1.0 = 0.967. BYM2 gives 0.957 — almost identical point estimate, because the spatial neighbourhood happens to be slightly below average in this area too. But BYM2 also gives you the 95% credibility interval of [0.875, 1.047].

That interval is the actual information. It tells you that AB10 3 is very likely near unity. You should not apply a discount here and you should not apply a loading — the data does not support either. A pricing committee asking "should we adjust AB10 3?" now has a quantified answer: no, because the full posterior spans both sides of 1.0 nearly symmetrically.

Independent credibility blending produces a point estimate and no uncertainty quantification. BYM2 produces a full posterior. For a pricing governance process that needs to justify territory factors to the FCA under Consumer Duty, the posterior is the argument.

---

## What changes year-on-year

One of the less-discussed benefits of BYM2 territory factors is their stability across annual refreshes. Independent sector rates are noisy: a sector with 30 policies and a bad year will produce a wild O/E that then partially reverts next year. The factor table changes substantially for reasons that are mostly sampling noise.

BYM2 factors are stable because the spatial structure acts as a constraint. A sector with a volatile O/E but stable-looking neighbours will be pulled toward those neighbours in every refresh. The genuine year-on-year change in territory factors is smaller, and it tracks the signal rather than the noise.

We have seen this on motor portfolios where the BYM2 territory table changes by a mean absolute log-factor of 0.04 per annual refresh, versus 0.11 for the independent credibility approach on the same data. That is a direct reduction in rate volatility that translates to more stable premiums for renewal policyholders — which is exactly what Consumer Duty's requirement for fair value over time is asking you to demonstrate.

---

## The full workflow

For teams new to the library, the two earlier posts cover the theory and the production plumbing in more detail:

- [BYM2 Spatial Smoothing for Territory Ratemaking](/2026/02/23/spatial-territory-ratemaking-with-bym2/) — the ICAR prior, the BYM2 reparameterisation, and why it scales to 9,000 postcode sectors
- [Getting Spatial Territory Factors Into Production](/2026/03/09/spatial-territory-ratemaking-bym2/) — the full Stage 1 → Stage 2 pipeline, adjacency caching, and Emblem/Radar export

Source at [github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial).
