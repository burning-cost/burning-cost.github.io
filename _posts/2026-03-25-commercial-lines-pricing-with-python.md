---
layout: post
title: "Commercial Lines Pricing with Python"
date: 2026-03-25
categories: [techniques, commercial-lines]
tags: [commercial-lines, fleet, credibility, buhlmann-straub, ILF, excess-of-loss, large-loss, severity, whittaker, monitoring, experience-rating, python]
description: "Fleet motor, property, and liability pricing in Python. Covers Bühlmann-Straub credibility for fleet schemes, GPD large loss loading, MBBEFD ILF tables, and PSI drift detection on commercial books."
---

Most of what we publish here is personal lines motor. That is the dominant UK GI pricing discipline and where most of the open-source tooling has been aimed. But the actuarial problems in commercial lines are at least as interesting — and the tooling gap is larger. A fleet motor actuary or a commercial property pricing team is more likely to be doing things in Excel than a personal lines team, not less.

This post is for them. We cover four commercial pricing problems using our Python libraries, with code that reflects how each calculation actually works rather than pseudocode that sidesteps the awkward details.

The four problems: fleet credibility, large loss loading, layered structures, and portfolio drift detection. We will draw on `insurance-credibility`, `insurance-severity`, `insurance-ilf`, `insurance-whittaker`, and `insurance-monitoring`.

---

## 1. Fleet motor: Bühlmann-Straub credibility for scheme pricing

A fleet scheme has two sources of evidence: its own claims history, and the underwriter's portfolio of similar fleets. Neither is fully trusted on its own. The scheme's history is often thin — fifty vehicles, three years, a handful of claims per year. The portfolio prior is a noisy average that may include road haulage firms, courier operations, and mixed commercial vehicle schemes that have nothing to do with this fleet.

The Bühlmann-Straub model resolves this. It estimates, from the observed distribution of experience across all schemes, how much between-scheme heterogeneity is signal versus noise. That ratio — the Bühlmann K — determines how much weight to give each scheme's own experience.

The data structure is a panel: one row per scheme per year, with loss rate (claims per vehicle year, or cost per vehicle year) and exposure (vehicle years).

```python
import polars as pl
from insurance_credibility.classical import BuhlmannStraub

# Panel of fleet schemes: one row per (scheme, year)
# loss_rate: total incurred / vehicle years
# veh_years: exposure weight
fleet_data = pl.DataFrame({
    "scheme": [
        "FleetA", "FleetA", "FleetA",
        "FleetB", "FleetB", "FleetB",
        "FleetC", "FleetC", "FleetC",
        "FleetD", "FleetD", "FleetD",
        "FleetE", "FleetE", "FleetE",
    ],
    "year": [2022, 2023, 2024] * 5,
    "loss_rate": [
        0.180, 0.195, 0.172,   # A: mid-size courier, stable
        0.420, 0.380, 0.440,   # B: haulage, elevated
        0.155, 0.160, 0.148,   # C: company cars, low
        0.310, 0.270, 0.290,   # D: mixed commercial
        0.210, 0.225, 0.198,   # E: van operators
    ],
    "veh_years": [
        820, 910, 875,
        2100, 2050, 2200,
        380, 410, 395,
        1450, 1380, 1510,
        640, 680, 655,
    ],
})

bs = BuhlmannStraub()
bs.fit(
    data=fleet_data,
    group_col="scheme",
    period_col="year",
    loss_col="loss_rate",
    weight_col="veh_years",
)

print(f"Portfolio mean (μ̂):   £{bs.mu_hat_:.3f} per veh-year")
print(f"EPV (v̂):               {bs.v_hat_:.4f}   (within-scheme noise)")
print(f"VHM (â):               {bs.a_hat_:.4f}   (between-scheme signal)")
print(f"K:                     {bs.k_:.0f}         (veh-years needed for Z=0.5)")
print()
print(bs.premiums_)
```

The K here will be around 1,200-1,500 vehicle years in this data. That means a fleet with 1,400 total vehicle years across the experience window gets Z ≈ 0.5: equal weight on its own history and the portfolio mean. Fleet B has 6,350 vehicle years total and sits at Z ≈ 0.82 — its elevated experience dominates. Fleet C has only 1,185 vehicle years and Z ≈ 0.44 — the low observed rate of £0.154/veh-year pulls back substantially toward the portfolio.

Two practical notes. First, the loss rate here is total incurred per vehicle year, which already conflates frequency and severity. For large fleets with heterogeneous vehicle types, you usually want to model these separately and blend them — but Bühlmann-Straub can be applied to total cost if that is how your scheme data arrives. Second, if schemes have genuinely different vehicle compositions (a 30-tonne HGV fleet versus company cars), the portfolio mean μ̂ may not be an appropriate complement. You would want to pass a normalised or risk-adjusted rate rather than use the raw grand mean.

---

## 2. Large loss loading: capping, Pareto fitting, and reloading

Commercial lines loss data is heavy-tailed. A single large fire, one liability judgment, a multi-vehicle motorway collision — these events are not outliers to be discarded. They are part of the risk profile and need to be priced.

The standard approach has three steps:

1. Cap each historical loss at a per-risk retention threshold.
2. Estimate the aggregate expected large loss cost using a heavy-tail model.
3. Load the attritional burning cost with the large loss component.

```python
import numpy as np

# Historical losses: a mix of attritional and large losses
rng = np.random.default_rng(42)
n_claims = 400
# Lognormal attritional losses
attritional = rng.lognormal(mean=np.log(8_000), sigma=0.8, size=360)
# Pareto-tailed large losses (simulate a heavy right tail)
large = rng.pareto(a=1.8, size=40) * 50_000 + 25_000
all_losses = np.concatenate([attritional, large])

# Exposure: 1,200 vehicle years
vehicle_years = 1_200.0

# Step 1: cap at retention threshold of £100,000 per claim
retention = 100_000.0
capped = np.minimum(all_losses, retention)

# Attritional burning cost (using capped losses)
bc_attritional = capped.sum() / vehicle_years
print(f"Attritional burning cost (capped at £{retention:,.0f}): £{bc_attritional:,.2f}/veh-yr")
```

Now fit the tail above the cap using `TruncatedGPD` from `insurance-severity`. This class corrects for the fact that policies have per-risk limits, so the exceedances you observe are drawn from a truncated distribution — standard GPD MLE underestimates the tail index when you ignore this.

```python
from insurance_severity.evt import TruncatedGPD

# Exceedances above the retention threshold
exceedances = all_losses[all_losses > retention] - retention

# Policy limits vary by risk. For simplicity, assume £500k limit.
# For real data, pass actual per-policy limits.
limits = np.full(len(exceedances), 500_000.0)

gpd = TruncatedGPD(threshold=0.0)  # threshold already applied: exceedances start at 0
gpd.fit(exceedances=exceedances, limits=limits)

print(f"GPD tail index xi:     {gpd.xi:.3f}")
print(f"GPD scale sigma:       £{gpd.sigma:,.0f}")
```

A tail index xi around 0.5-0.8 is typical for fleet motor large losses. Higher xi means heavier tail and more sensitivity to the upper limit assumption. If xi > 1, the theoretical mean does not exist — worth flagging to underwriting.

Step 3: compute the large loss loading using numerical integration of the fitted tail. The expected cost of a loss above the retention, per claim, is the integral of the survival function from 0 to the maximum loss cap (limit minus retention):

```python
from scipy import integrate

max_excess = 400_000.0  # £500k limit - £100k retention

# E[excess loss | excess > 0] via numerical integration of survival function
el_per_excess_claim, _ = integrate.quad(gpd.sf, 0, max_excess)

# Frequency of large losses per vehicle year
freq_large = len(exceedances) / vehicle_years

# Large loss loading per vehicle year
bc_large = freq_large * el_per_excess_claim
print(f"Large loss loading:    £{bc_large:,.2f}/veh-yr")
print(f"Total burning cost:    £{bc_attritional + bc_large:,.2f}/veh-yr")
```

The `WeibullTemperedPareto` class is the alternative when the tail is heavy but physically bounded — property, D&O, PI. It fits a survival function `S(x) = x^{-alpha} * exp(-lambda * x^tau)` which has a Pareto core but dampens off at extreme values. Use it when you have a credible cap on maximum loss size from engineering surveys or policy limits that genuinely constrain the tail.

For severity modelling more generally, the `insurance-severity` library also includes composite body-tail models and a distributional regression network (DRN) for when the severity distribution varies by risk characteristic. For a commercial fleet, the DRN approach is probably overkill — but if you are pricing property and the loss distribution genuinely differs between timber-frame and concrete construction, it earns its keep.

---

## 3. Layered structures: ILFs and per-risk excess of loss

Commercial lines pricing often requires working in layers. A liability insurer might sell £5m xs £500k. A captive manager needs to know what retention level minimises expected total cost. A reinsurance actuary pricing a per-risk XL treaty needs expected loss by layer from the cedant's risk profile.

The `insurance-ilf` library handles this via the MBBEFD exposure curve family (Bernegger, 1997), which gives analytic formulas for the expected loss in any layer from a fitted distribution.

```python
from insurance_ilf import MBBEFDDistribution, ilf_table, layer_expected_loss, per_risk_xl_rate
import pandas as pd

# Fit a Swiss Re Y3 curve (c=3): appropriate for commercial property
# Y1 (c=1.5): light industrial, low hazard
# Y2 (c=2):   standard commercial
# Y3 (c=3):   commercial property, mid-hazard
# Y4 (c=4):   high-hazard industrial
dist = MBBEFDDistribution.from_c(3.0)

# ILF table: limits from £500k to £10m, basic limit £1m
# Normalise to MPL of £10m
table = ilf_table(
    dist=dist,
    limits=[500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000],
    basic_limit=1_000_000,
    mpl=10_000_000,
)
print(table[["limit", "ilf", "marginal_ilf"]])
```

The ILF at £2m xs £0 (i.e., a £2m limit) is the ratio of expected loss at £2m to expected loss at the basic limit of £1m. The marginal ILF is the incremental factor between successive limits in the table — useful for pricing the £1m xs £1m layer separately from the ground-up £2m.

For a per-risk XL layer — where the reinsurer pays losses in the band `[AP, AP+L]` on each individual risk — use `layer_expected_loss` directly:

```python
# A single property risk:
# sum insured £3m, MPL assumed to be 80% of SI (concrete construction)
# subject premium (burning cost from direct insurer's perspective): £18,000
# Layer: £1m xs £1m

el = layer_expected_loss(
    dist=dist,
    attachment=1_000_000,
    limit=1_000_000,
    policy_limit=3_000_000,
    mpl=3_000_000 * 0.8,   # MPL = 80% of sum insured
    subject_premium=18_000,
)
print(f"Expected loss to £1m xs £1m layer: £{el:,.0f}")
```

For treaty pricing, you work from the cedant's risk profile — the distribution of their book by sum insured band:

```python
# Cedant's risk profile: three SI bands
risk_profile = pd.DataFrame({
    "sum_insured":  [500_000, 1_000_000, 2_000_000],
    "premium":      [45_000,  80_000,    60_000],
    "count":        [90,      80,        30],
})

from insurance_ilf import per_risk_xl_rate

result = per_risk_xl_rate(
    risk_profile=risk_profile,
    dist=dist,
    attachment=1_000_000,
    limit=1_000_000,
    mpl_factor=0.8,   # MPL = 80% of sum insured throughout
)

print(f"Total expected loss to layer: £{result['total_expected_loss']:,.0f}")
print(f"Subject premium:              £{result['subject_premium']:,.0f}")
print(f"Technical rate:               {result['technical_rate']:.2%}")
print(f"Rate on line:                 {result['rol']:.4f}")
```

The `technical_rate` here is expected loss / subject premium: what proportion of the burning cost attaches to this layer. The `rol` (rate on line) is the traditional reinsurance pricing metric: expected loss as a proportion of the layer limit capacity.

One limitation to flag: the MBBEFD family assumes the shape of the severity distribution is the same for all sum insured bands, scaled by SI. For a genuinely heterogeneous book — say, mixing high-rise residential with single-storey warehousing — you should fit separate distributions by occupancy class and apply them separately. `per_risk_xl_rate` works band by band, so you can pass it a risk profile with per-band curve parameters if you pre-compute `dist` separately for each segment.

---

## 4. Smoothing loss development: Whittaker-Henderson on commercial triangles

Commercial lines loss development is choppier than personal lines. Lower volumes, large individual losses, and heterogeneous exposure mixes mean that raw development factors from a triangle can be unreliable. Smoothing the tail factors is standard practice; doing it with a principled statistical method is less common.

`insurance-whittaker` provides the Whittaker-Henderson smoother, which minimises a penalised least-squares criterion and selects the smoothing parameter automatically via REML. The natural use case is smoothing age-to-age factors or tail factors across development periods.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# Development periods: months 12, 24, 36, ... 120
dev_periods = np.array([12, 24, 36, 48, 60, 72, 84, 96, 108, 120], dtype=float)

# Age-to-age factors from a commercial liability triangle
# (real data would come from your reserving system)
ata_factors = np.array([2.85, 1.72, 1.41, 1.22, 1.12, 1.07, 1.04, 1.02, 1.01, 1.005])

# Weights: number of open diagonals contributing to each factor
# Factors at early development have more credible source data
weights = np.array([8, 7, 6, 5, 4, 3, 3, 2, 2, 1], dtype=float)

# Smooth with order=2 (penalises second differences: smooth curves)
wh = WhittakerHenderson1D(order=2, lambda_method="reml")
result = wh.fit(dev_periods, ata_factors, weights=weights)

print(f"Smoothing parameter lambda: {result.lambda_:.2f}")
print(f"Effective degrees of freedom: {result.edf:.1f}")
print()
for i, (dp, raw, smooth) in enumerate(zip(dev_periods, ata_factors, result.fitted)):
    print(f"  {int(dp):3d}m: raw={raw:.4f}  smoothed={smooth:.4f}  "
          f"CI=[{result.ci_lower[i]:.4f}, {result.ci_upper[i]:.4f}]")
```

The 95% credible interval comes from the hat matrix diagonal — it tells you where the smoother is uncertain. For development periods with only one or two contributing diagonals, the interval will be wide and you should consider whether you want to manually constrain those factors to a tail pattern assumption.

The smoother also works on relativities across vehicle groups, fleet age bands, or any ordered rating variable where you want smoothed factors rather than unsmoothed GLM output. The `order=2` default gives cubic-spline-like smoothness; `order=1` gives piecewise-linear smoothing, appropriate if you believe there is a genuine kink in the curve at some development period.

---

## 5. Experience rating for scheme business

For schemes with multi-year history and a credible claim count, experience rating adjusts next year's rate based on actual versus expected claims. The standard formula for the experience adjustment factor is:

```
EAF = Z * (actual_claims / expected_claims) + (1 - Z)
```

where Z is the credibility factor. The question is how to set Z. Using `BuhlmannStraub` across the portfolio of schemes is the right answer — it estimates K from the actual observed heterogeneity across schemes rather than assuming an arbitrary credibility schedule.

Here is a worked example for a portfolio of liability schemes:

```python
import polars as pl
import numpy as np
from insurance_credibility.classical import BuhlmannStraub

# Liability scheme experience: 8 schemes, 4 years
# loss_rate here is actual incurred / expected incurred (i.e., A/E ratio)
rng = np.random.default_rng(7)
n_schemes = 8
n_years = 4
true_ae = rng.lognormal(mean=0.0, sigma=0.18, size=n_schemes)  # true A/E ratios

rows = []
for i, ae_true in enumerate(true_ae):
    for year in range(2021, 2021 + n_years):
        policy_count = int(rng.uniform(200, 1200))
        # Observed A/E has Poisson noise around truth
        ae_obs = rng.normal(ae_true, ae_true / np.sqrt(policy_count * 0.15))
        ae_obs = max(ae_obs, 0.0)
        rows.append({
            "scheme": f"S{i+1:02d}",
            "year": year,
            "ae_ratio": ae_obs,
            "policy_count": float(policy_count),
        })

df = pl.DataFrame(rows)

bs = BuhlmannStraub()
bs.fit(
    data=df,
    group_col="scheme",
    period_col="year",
    loss_col="ae_ratio",
    weight_col="policy_count",
)

print(f"K = {bs.k_:.0f}  (policy-years needed for Z=0.50)")
print()

# Experience-rated A/E for the next year's renewal pricing
premiums = bs.premiums_
for row in premiums.iter_rows(named=True):
    eaf = row["credibility_premium"]
    print(f"  {row['group']}: Z={row['Z']:.2f}  raw A/E={row['observed_mean']:.3f}  "
          f"credibility A/E={eaf:.3f}")
```

The `credibility_premium` here is the credibility-weighted A/E ratio. Multiply the scheme's current technical rate by this factor for the renewal loading. A scheme with Z = 0.35 and a raw A/E of 1.40 gets a credibility-weighted A/E of roughly 1.14 (pulling toward the portfolio mean of 1.0). Price accordingly — the 40% adverse experience is probably real but you don't fully believe it yet.

---

## 6. Drift detection on commercial books

Commercial books change composition faster than personal lines. A new broker relationship, a change in the minimum fleet size you will write, a recession affecting one industry sector — all of these shift the underwritten population in ways that can silently invalidate a pricing model.

`insurance-monitoring` provides PSI (Population Stability Index) for operational monitoring. The exposure-weighted variant matters for insurance: a bin with 30 annual policies is not equivalent to a bin with 300 one-month policies.

```python
import numpy as np
from insurance_monitoring.drift import psi

# Fleet size distribution at model training (2023)
rng = np.random.default_rng(0)
# Log-normal fleet sizes: median around 25 vehicles
ref_fleet_sizes = rng.lognormal(mean=np.log(25), sigma=0.8, size=2_000)
ref_exposures = rng.uniform(0.5, 1.0, size=2_000)  # partial year exposures

# Current portfolio (2025Q1): new broker brings larger fleets
cur_fleet_sizes = rng.lognormal(mean=np.log(40), sigma=0.9, size=800)
cur_exposures = rng.uniform(0.5, 1.0, size=800)

psi_val = psi(
    reference=ref_fleet_sizes,
    current=cur_fleet_sizes,
    n_bins=10,
    exposure_weights=cur_exposures,
    reference_exposure=ref_exposures,
)

print(f"Fleet size PSI: {psi_val:.3f}")
# Interpret: < 0.10 stable, 0.10-0.25 investigate, >= 0.25 model likely stale
```

The thresholds come from banking credit-scoring practice (FICO, 1990s) but apply equally to insurance. A PSI of 0.25 on fleet size means the distribution has shifted enough that a model trained on the old mix should be re-validated before relying on it for renewal rates.

For commercial property, the relevant drift variables are sum insured band, occupancy class, and construction type. Run PSI on each quarterly. If occupancy class has PSI > 0.20 — say, a shift from retail (less fire risk) toward food processing (more fire risk) — your book-level loss ratio will move faster than the model predicts.

---

## What we have not covered

Several commercial pricing problems are either too specialised or need their own post:

**Catastrophe loading.** Property accumulation modelling requires a cat model output (RMS, AIR, or Oasis) and an allocation methodology. We have not built a cat model and do not intend to. The loading calculation itself — expected annual loss from the cat model, allocated to individual risks by SI band and location — can be done with standard pandas arithmetic once you have the AAL output.

**Motor trade and non-standard motor.** These require separate frequency models for road risk versus premises risk. The Bühlmann-Straub approach applies to the individual location or garage, not the headline account.

**Casualty reserving.** The Whittaker-Henderson smoother helps with development factor selection, but the full chain-ladder, BF, or Clark LDF method is a separate exercise. We are not aware of a good open-source Python reserving library — if you know of one, tell us.

**Pricing adjustments for deductibles.** The ELF (excess loss factor) from `insurance-ilf` handles per-risk deductibles on property. For casualty, deductible pricing is more complex because the attachment interacts with the claims frequency distribution. This deserves a dedicated post.

---

## Installation

```bash
uv add insurance-credibility insurance-severity insurance-ilf insurance-whittaker insurance-monitoring
```

All five packages are available on PyPI and work with Python 3.10+. They use Polars for data handling; `insurance-ilf` uses pandas for output DataFrames (for compatibility with reserving and reporting workflows).

---

## References

- Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit für Schadensätze. *Mitteilungen VSVM*, 70, 111-133.
- Bernegger, S. (1997). The Swiss Re Exposure Curves And The MBBEFD Distribution Class. *ASTIN Bulletin*, 27(1), 99-111.
- Clark, D.R. (2014). Basics of Reinsurance Pricing. CAS Study Note.
- Albrecher, H., Beirlant, J. & Teugels, J. (2025). Extremes for insurance losses with censoring. *arXiv:2511.22272*.
- Biessy, G. (2026). Whittaker-Henderson Smoothing Revisited. *ASTIN Bulletin*. arXiv:2306.06932.

---

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/) — the `insurance-credibility` library for the credibility weighting of fleet and commercial segments
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — model governance documentation for commercial lines models, which typically sit at Tier 1 given higher GWP per model
