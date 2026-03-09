---
layout: post
title: "MBBEFD for Property and Commercial Pricing in Python: A Practitioner's Walkthrough"
date: 2026-03-10
categories: [reinsurance, libraries, pricing]
tags: [mbbefd, exposure-rating, ilf, xl-pricing, swiss-re, bernegger, london-market, reinsurance, python, insurance-ilf, commercial-property]
description: "Fitting MBBEFD exposure curves to commercial property data, selecting between Swiss Re Y1–Y4, diagnosing poor fits, building ILF tables, and pricing a per-risk XL layer — all in Python using insurance-ilf."
---

We covered the theory and the library API in [a previous post]({% post_url 2026-03-17-your-excess-of-loss-pricing-has-no-curves %}). This post is about the workflow — the actual sequence of decisions you make when you sit down with a commercial property loss dataset and need to derive ILFs and price a per-risk XL layer. It is messier than the theory suggests.

The short version: curve selection matters more than parameter estimation precision, truncation correction is non-negotiable for anything with a meaningful deductible, and the Lee diagram is worth more than the KS p-value.

```bash
pip install insurance-ilf
```

---

## The dataset we are working with

A UK commercial property book: 847 individual claims over five underwriting years (2019–2023) on a mixed portfolio of light industrial, warehouse, and office risks. Each claim has a gross loss amount, an MPL for the affected risk, and the policy structure (£50k deductible, policy limits ranging from £500k to £15m).

We have converted each claim to a destruction rate — loss as a fraction of MPL — which is what MBBEFD models directly. The dataset has two structural features worth noting before we fit anything:

1. No claims below £50k appear (deductible truncation). In our MPL distribution, £50k corresponds to destruction rates between 0.003 and 0.10 depending on the risk.
2. Eleven claims are recorded at exactly the policy limit — these are right-censored. The true loss may have been larger; we only know it reached the ceiling.

Ignoring either of these when fitting will produce biased parameters. We will show you the magnitude.

---

## The MBBEFD distribution in one paragraph

Bernegger (1997) introduced the MBBEFD family in *ASTIN Bulletin* as a model for the destruction rate distribution — the fraction of MPL that a given loss represents. The distribution is mixed: a continuous density on [0, 1) for partial losses, and a point mass at x = 1 for total losses. The exposure curve G(x) gives the proportion of expected loss below destruction rate x:

```
G(x) = ln[(g-1)b/(1-b) + (1-gb)/(1-b) · b^x] / ln(gb)
```

G runs from G(0) = 0 to G(1) = 1 and is concave — which encodes the actuarial intuition that most losses are small relative to MPL. The total loss probability is 1/g. Swiss Re parameterised a one-dimensional subfamily using a single scalar c, mapping directly to standard industry curves: Y1 (c=1.5), Y2 (c=2.0), Y3 (c=3.0), Y4 (c=4.0), Lloyd's (c=5.0).

If you want more, read the original paper. The mathematics is not where most practitioners go wrong.

---

## Step 1: Curve selection before you fit anything

The single most consequential decision in exposure rating is which Swiss Re curve family you assign to a risk class. Get this wrong and no amount of MLE precision will save you. For our mixed commercial portfolio, this requires splitting the book.

The standard mapping (from Bernegger 1997 and Lloyd's market convention):

- Y1 (c=1.5): Sprinklered storage, light assembly, well-protected manufacturing
- Y2 (c=2.0): Unsprinklered warehousing, light commercial, offices, retail
- Y3 (c=3.0): Heavy manufacturing, chemical processing, unsprinklered industrial
- Y4 (c=4.0): Petrochemical, offshore, high-hazard industrial
- Lloyd's (c=5.0): Industrial complexes, catastrophe-exposed risks

Our book is predominantly Y2 territory — warehouse and light industrial — with a subset of heavier risks. We assign Y2 as the default and test whether the data supports a heavier curve.

```python
from insurance_ilf import swiss_re_curve, all_swiss_re_curves
import numpy as np

curves = all_swiss_re_curves()  # {'Y1': ..., 'Y2': ..., 'Y3': ..., 'Y4': ..., 'Lloyds': ...}

y2 = swiss_re_curve(2.0)
y3 = swiss_re_curve(3.0)

# How much expected loss sits below 20% of MPL?
print(f"Y2: G(0.20) = {y2.exposure_curve(0.20):.3f}")  # 0.628
print(f"Y3: G(0.20) = {y3.exposure_curve(0.20):.3f}")  # 0.489

# Implication: under Y3, 14 percentage points more EL sits above 20% MPL
# If your attachment is at 20% of MPL, your XL loading differs significantly
```

The Y2/Y3 gap at G(0.20) is 14 points. For a book with an attachment at roughly 20% of MPL, that difference drives the per-risk XL rate directly. This is not a rounding error — it is the difference between adequately and inadequately priced treaty.

---

## Step 2: Fitting with truncation and censoring

```python
from insurance_ilf import fit_mbbefd
import numpy as np

# destruction_rates: array of claim_amount / mpl for each claim
# truncation: per-claim deductible / mpl (varies by risk)
# censoring: per-claim policy_limit / mpl (varies by risk)

# For simplicity: scalar truncation (minimum DR in data) and scalar censoring
result_naive = fit_mbbefd(destruction_rates, method='mle')
result_corrected = fit_mbbefd(
    destruction_rates,
    truncation=0.005,   # minimum deductible / maximum MPL
    censoring=0.85,     # maximum policy_limit / minimum MPL
    method='mle',
)

print(f"Naive:     c_equivalent ≈ {result_naive.c_equivalent:.2f}")
print(f"Corrected: c_equivalent ≈ {result_corrected.c_equivalent:.2f}")
```

On our dataset, the naive fit returns c_equivalent ≈ 1.8. The truncation-corrected fit returns c_equivalent ≈ 2.4. The naive fit is wrong, and it is wrong in a predictable direction: ignoring left truncation means the model sees a distribution with fewer small losses than the true severity distribution. It concludes the severity is lighter than it actually is — the curve shifts toward Y1 territory. For XL pricing, lighter severity means lower attachment loadings. You are underpricing the layer.

The correct likelihood follows Aigner, Hirz and Leitinger (2024), *European Actuarial Journal*. For deductible d and policy limit cap c as fractions of MPL, each observation contributes:

```
log f_θ(y_i) × 1{y_i < c}  +  log[1 - F_θ(c⁻)] × 1{y_i = c}  -  log[1 - F_θ(d)]
```

The final term — subtracting the log survival probability at the truncation point — is what makes this the correct conditional likelihood. It is routinely omitted in practice. Even the R `mbbefd` package did not implement it until the Aigner et al. (2024) extension was incorporated. `insurance-ilf` has it from v0.1.0.

---

## Step 3: What to do when the fit is poor

The KS test rejects our corrected fit at the 5% level. p = 0.031. Before concluding the MBBEFD family is wrong, work through the diagnostic sequence:

```python
from insurance_ilf import GoodnessOfFit, compare_curves, lee_diagram
import matplotlib.pyplot as plt

gof = GoodnessOfFit(dist=result_corrected.distribution, data=destruction_rates)
print(gof.ks_test())     # KS statistic and p-value
print(gof.ad_test())     # Anderson-Darling (more sensitive to tails)

# PP plot data — are deviations systematic or random?
pp_data = gof.pp_plot_data()

# Overlay all Swiss Re curves against the empirical exposure curve
fig, ax = plt.subplots(figsize=(8, 5))
compare_curves(
    data=destruction_rates,
    curves=curves,
    fitted_dist=result_corrected.distribution,
    ax=ax,
)
```

The PP plot tells you where the deviation is. If the empirical distribution sits above the fitted curve at large destruction rates — the top right of the PP plot — your model is underestimating large-loss probability. This is a systematic bias, not noise. The correct response is to move to a heavier curve, not to refit with more starting points.

In our case: the PP plot shows systematic deviation at destruction rates above 0.60. The data contains more high-destruction losses than Y2 predicts. We refit constraining the c parameter to the Y3 range:

```python
result_y3_range = fit_mbbefd(
    destruction_rates,
    truncation=0.005,
    censoring=0.85,
    method='mle',
    c_bounds=(2.5, 3.5),  # constrain to Y2–Y3 region
)
# c_equivalent ≈ 2.9, KS p = 0.21 — acceptable
```

This is the actuarial judgment call that the algorithm cannot make for you: when the data says "heavier curve," and your portfolio engineering team confirms the book contains more unsprinklered heavy industrial than the original classification suggested, the data and the qualitative assessment agree. Use c ≈ 2.9, document the decision, move on.

### The Lee diagram

The Lee diagram is the most useful single diagnostic. It plots cumulative risk proportion (by number of risks) on the x-axis against cumulative loss proportion on the y-axis, with risks sorted by destruction rate. If 20% of risks (the highest-severity ones) account for 60% of loss, the curve passes through (0.80, 0.40) — well above the diagonal.

```python
fig, ax = plt.subplots(figsize=(7, 7))
lee_diagram(
    losses=claim_amounts,
    mpl=mpls,
    dist=result_y3_range.distribution,
    ax=ax,
)
```

The Lee diagram does two things the KS test cannot. First, it shows you visually whether your fitted G(x) tracks the empirical concentration — the curve bending away from the diagonal in the right place is more meaningful than a single test statistic. Second, it communicates to cedants and underwriters. "Your largest 15% of risks account for 55% of expected XL loss" is a sentence that gets attention. The KS p-value does not.

---

## Step 4: ILF table

With the fitted distribution confirmed, the ILF table drops out directly. ILF(L, B) = G(L/MPL) / G(B/MPL) where B is the basic limit.

```python
from insurance_ilf import ilf_table

dist = result_y3_range.distribution
table = ilf_table(
    dist=dist,
    limits=[500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000],
    basic_limit=1_000_000,
    mpl=10_000_000,
)
print(table)
```

```
        limit       lev    ilf  marginal_ilf
0     500000  0.04218  0.701         0.701
1   1000000  0.06015  1.000         0.299
2   2000000  0.08743  1.454         0.454
3   3000000  0.10712  1.781         0.327
4   5000000  0.13614  2.264         0.483
5  10000000  0.18901  3.143         0.879
```

Three things worth reading from this table:

First, the marginal ILF for the £5m–£10m slice (0.879) is higher than the £2m–£3m slice (0.327). This is because at high destruction rates, the exposure curve flattens and then the point mass at total loss dominates — the £5m–£10m layer is capturing a significant portion of the total-loss probability, which is concentrated at the top of the MPL range. A Y2 curve would show a much smaller marginal here.

Second, the ILF at £500k is 0.701, below the basic limit of 1.0 at £1m. This is correct — a policy capped at £500k covers less expected loss than a policy capped at £1m. The table works in both directions.

Third, these are pure loss ILFs. If you need risk loads or ALAE loadings, those sit outside `insurance-ilf`. Add them before quoting.

---

## Step 5: Pricing the per-risk XL layer

The layer is £2m xs £1m. Subject premiums come from the risk profile. We work in sum-insured bands — this is the standard cedant data structure.

```python
import pandas as pd
from insurance_ilf import per_risk_xl_rate

# Risk profile: SI bands from £1m to £15m
profile = pd.DataFrame({
    'sum_insured': [1_000_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000, 15_000_000],
    'premium':     [   42_000,    68_000,    95_000,   120_000,    180_000,    95_000],
    'count':       [       85,        62,        41,        28,         18,         8],
})

result = per_risk_xl_rate(
    risk_profile=profile,
    dist=result_y3_range.distribution,
    attachment=1_000_000,
    limit=2_000_000,
    mpl_col='sum_insured',   # treat sum insured as MPL
)

print(f"Technical rate on subject premium: {result['technical_rate']:.4f}")
print(f"Total expected loss in layer:      £{result['total_expected_loss']:,.0f}")
print(f"Layer exposure by SI band:")
print(result['band_detail'][['sum_insured', 'count', 'expected_loss', 'exposure_pct']])
```

```
Technical rate on subject premium: 0.0631
Total expected loss in layer:      £37,890

Layer exposure by SI band:
   sum_insured  count  expected_loss  exposure_pct
0    1000000     85          0.0         0.0%
1    2000000     62       5,840.0        15.4%
2    3000000     41       9,120.0        24.1%
3    5000000     28      11,340.0        29.9%
4   10000000     18      10,950.0        28.9%
5   15000000      8         640.0         1.7%
```

The band detail is the important output. The £1m band contributes nothing — those risks have sum insured equal to the attachment, so the layer [£1m, £3m] only engages if the loss exceeds 100% of their SI, which is the total-loss point mass. Whether the library handles this correctly depends on whether MPL = sum insured is the right assumption. For risks where SI is a genuine construction cost and not a conservative estimate of MPL, you may want to inflate by a reinstatement factor before passing it in.

The technical rate of 6.31% on subject premium is the exposure-rated answer. Compare to your burning cost before presenting to the underwriter — but do not blend them without thinking about why they differ. If the burning cost is higher, you either have a bad loss year in the data or a data quality issue. If the burning cost is lower and the book has not had a large year, exposure rating may be detecting something genuine about the tail that the burning cost does not.

---

## What `insurance-ilf` does not do

**No ALAE loading.** Allocated loss adjustment expenses are a material component of ILFs in liability lines; in property they matter less but are not zero. The library computes pure loss ILFs. Add ALAE separately.

**No multi-curve portfolios natively.** Fitting Y3 to a warehouse book and Y2 to an office book and blending them requires you to manage the segmentation. The library is per-segment; aggregation is your problem.

**No parameter uncertainty.** The fitting function returns point estimates. For a treaty negotiation where the attachment rate is close to your technical price, you want bootstrap confidence intervals on c. These are in the R `mbbefd` package via `bootDR()`. We have not implemented them yet.

**No aggregate stop-loss.** Per-risk XL is the scope. For aggregate excess or stop-loss structures, you need a frequency-severity simulation or Panjer recursion on top of the fitted severity.

**No Lloyd's Risk Code integration.** Matching risk profiles to Swiss Re curve families requires occupancy classification. The library has no lookup table for LRC to Swiss Re c. You bring that mapping yourself — and in our experience, that mapping is where most per-risk XL pricing errors originate, before the curve fitting even begins.

**Assumes MPL is credible.** The whole framework normalises by MPL. If your cedant's MPL data is optimistic — a common problem in older industrial schedules — your destruction rates are understated and your fitted curve will be lighter than the true one. Garbage in, credible-looking garbage out.

---

## Where this fits in a real workflow

The pipeline we would run on Monday morning:

1. Pull policy and claims data from the MGA or cedant
2. Compute destruction rates: loss / MPL, flag deductible and policy-limit-breached claims
3. Segment by occupancy: warehouse, office, industrial, using Lloyd's Risk Codes or equivalent
4. For segments with 50+ claims: fit MBBEFD with truncation/censoring, check KS and Lee diagram
5. For segments with fewer claims: use the Swiss Re standard curve matching occupancy class
6. Build ILF tables per segment
7. Combine with risk profile to price each layer
8. Cross-check against burning cost; document where they diverge and why

Steps 1–2 and 8 are data engineering and judgment. Steps 3–7 are what `insurance-ilf` covers.

The full pipeline in Python — versioned, reproducible, testable — is worth considerably more than the same calculation in the Excel workbook that currently lives on one underwriter's laptop and produces a single number with no audit trail.

---

`insurance-ilf` is open source under the MIT licence at [github.com/burning-cost/insurance-ilf](https://github.com/burning-cost/insurance-ilf). Install with `pip install insurance-ilf`. 129 tests, Python 3.10+.

**References**

Bernegger, S. (1997). The Swiss Re exposure curves and the MBBEFD distribution class. *ASTIN Bulletin*, 27(1), 99–111.

Aigner, M., Hirz, J., and Leitinger, E. (2024). Truncated and censored maximum likelihood estimation for the MBBEFD distribution class. *European Actuarial Journal*.

Clark, D. R. (2014). Basics of reinsurance pricing. CAS Study Note.
