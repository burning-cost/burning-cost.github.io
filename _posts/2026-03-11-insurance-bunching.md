---
layout: post
title: "Detecting Threshold Gaming in Insurance Portfolios"
date: 2026-03-11
categories: [libraries, pricing, fraud]
tags: [bunching, threshold-gaming, NCD, adverse-selection, mileage-fraud, public-economics, Saez, Kleven, FCA-Consumer-Duty, insurance-bunching, python, WLS, bootstrap, BH-correction]
description: "Policyholders game rating thresholds. Mileage declarations cluster just below 10,000. Claimed ages spike just above 25. Declared sums bunch at £100k, £200k, £300k. insurance-bunching is the first Python implementation of bunching estimators — adapted from public economics (Saez 2010, Kleven 2016) to detect exactly this."
---

Public economists have known since Saez (2010) that people bunch at tax thresholds. When a taxpayer crosses an income bracket boundary, their marginal tax rate jumps, and the distribution of taxable income develops a spike just below that boundary. The density is not continuous at the kink: there is excess mass on one side, a deficit on the other, and the ratio of excess to counterfactual tells you how strongly people are responding.

The statistical machinery to measure this — polynomial counterfactual density estimation, iterative correction for the deficit region, bootstrap standard errors — is mature in public economics. Diamond and Saez used it. Kleven formalised it in a 2016 handbook chapter that has become the field's reference implementation.

Insurance creates identical incentive structures. The specific thresholds differ — NCD levels rather than tax brackets, mileage bands rather than pension contribution limits, sum-insured breakpoints rather than NICs thresholds — but the underlying mechanics are the same. A policyholder facing a discrete premium change at a threshold has an incentive to manipulate their position relative to it. In aggregate, that manipulation produces exactly the density spike that bunching estimators detect.

Nobody has applied this to insurance. The R package `bunching` (Mavrokonstantis 2022) is designed for public economics data. Stata has `bunchit`. Python has nothing in any field.

[`insurance-bunching`](https://github.com/burning-cost/insurance-bunching) is the first Python bunching estimator, and the first application of bunching methodology to insurance pricing. 120 tests, MIT-licensed, PyPI.

```bash
uv add insurance-bunching
```

---

## Where the thresholds are

UK motor insurance gives policyholders several strong threshold incentives, all of which create detectable bunching signals.

**Mileage bands.** Insurers price on declared annual mileage in bands: 0–5k, 5–10k, 10–15k, 15–20k, 20k+. A driver who covers 10,200 miles per year faces a premium increase at the 10,000 boundary. If they declare 9,800, they save the premium difference. Across a large motor book, this produces excess mass in the distribution of declared mileage just below each band boundary. The deficit appears just above, where honest declarers at 10,100 should be but are not.

**Age boundaries.** UK motor rating factors are steep at young ages — typically a sharp drop somewhere around 24–25. A policyholder who will turn 25 next month has an incentive to misstate their age. In aggregate, declared-age distributions develop a spike just above each rating band boundary where premiums fall sharply.

**Sum insured.** Household policies show round-number clustering at £100k, £150k, £200k, £300k. This is partly cognitive — round numbers are natural reference points — and partly strategic: policyholders declaring just above a pricing threshold to qualify for a cheaper band. The declared sum insured distribution is not smooth.

**NCD levels.** UK NCD creates the most complex incentive. The De Pril (1978) and Lemaire (1995) theoretical work on "hunger for bonus" shows that policyholders near maximum NCD rationally suppress small claims, since the premium increase from NCD loss exceeds the claim value. In the claims severity distribution, this creates a notch: claims are suppressed below the break-even amount and the density spikes just above. v1.0 handles continuous running variables; discrete NCD levels (number of claim-free years) follow in v1.1.

**Deductible selection.** Adverse selection in deductible choice (Einav et al., QJE 2010) is well established: high-risk individuals select lower deductibles because their expected claim frequency makes low deductibles more valuable. Excess mass at popular deductible levels (£250, £500, £1,000) beyond what continuous risk heterogeneity implies is a bunching signal. Einav et al. used price variation instruments; bunching density estimation is a complementary detection method that does not require exogenous price variation.

---

## The estimator

The core estimator follows Kleven (2016) with insurance-specific modifications.

**Counterfactual density.** The observed density of the running variable (mileage, declared age, sum insured) has excess mass near the threshold. The counterfactual is what the density would look like absent the threshold incentive — a smooth polynomial fit to the data excluding a window around the threshold.

```python
from insurance_bunching import BunchingEstimator
import numpy as np

# Declared annual mileage from a motor book
# Running variable: declared_mileage (continuous)
# Threshold: 10,000 miles

estimator = BunchingEstimator(
    z=declared_mileage,
    zstar=10_000,
    bw=200,                  # bin width (miles)
    poly_order=7,            # counterfactual polynomial degree
    excl_below=500,          # exclusion window below threshold (miles)
    excl_above=1_000,        # exclusion window above threshold (miles)
    mode='notch',            # discrete premium jump at threshold
    n_boot=500,              # bootstrap iterations for SE
)

result = estimator.fit()

print(f"Bunching mass B: {result.B_hat:.3f}")
print(f"Standard error: {result.B_se:.3f}")
print(f"p-value: {result.p_value:.4f}")
print(f"Excess policyholders: {result.excess_n:.0f}")
```

**Iterative correction.** The exclusion window — the bins immediately around the threshold that are excluded from counterfactual fitting — contains both the excess mass and the missing mass. The missing mass (the deficit of honest policyholders who should appear just above the threshold but have shifted down) biases the polynomial fit if you use the raw exclusion region. The estimator corrects iteratively: fit the counterfactual, estimate the missing mass, redistribute it proportionally to bins above the exclusion window, refit. This converges in three to five iterations for typical insurance data and is standard practice in the bunching literature.

**Exposure-weighted WLS.** Insurance bins are not equal-sized. A mileage bin of 9,800–10,000 contains more policies than a bin of 3,800–4,000 simply because more people report round numbers near pricing thresholds. The polynomial fit uses WLS weighted by earned exposure in each bin, so that thin cells have less influence on the counterfactual.

**Kink vs notch.** A kink is a change in the rate of premium increase — the derivative changes at the threshold. A notch is a discrete jump in premium level — the threshold itself triggers a step change. Mileage bands create notches: crossing 10,000 triggers a discrete premium increase. Sum-insured brackets may create kinks where the price per £k of cover changes. The estimator handles both; the counterfactual specification differs.

---

## Policy-level data: ExposureWeightedBunching

Most insurance pricing teams work with policy-level DataFrames, not pre-binned density data. `ExposureWeightedBunching` is a thin wrapper that handles the aggregation.

```python
import pandas as pd
from insurance_bunching import ExposureWeightedBunching

# policy-level DataFrame
# columns: declared_mileage, earned_years, vehicle_age, ncd_level, ...

ewb = ExposureWeightedBunching(
    df=motor_policies,
    running_var='declared_mileage',
    exposure_col='earned_years',
    zstar=10_000,
    bw=200,
    excl_below=500,
    excl_above=1_000,
    mode='notch',
)

result = ewb.fit()
result.plot(title='Declared mileage bunching at 10,000-mile threshold')
```

The exposure weighting matters. A policy with 0.5 earned years contributes half the weight of a full-year policy to each bin. Without this, the density estimate is biased towards policies that appear multiple times in the data (annual renewals for multi-year policyholders).

---

## Scanning multiple thresholds: MultiThresholdScanner

A motor book has several mileage thresholds simultaneously. Testing each separately inflates false discovery rates — if you run five tests at α = 0.05, you expect 0.25 false positives even with no bunching anywhere.

`MultiThresholdScanner` applies Benjamini-Hochberg FDR correction across all tested thresholds.

```python
from insurance_bunching import MultiThresholdScanner

scanner = MultiThresholdScanner(
    z=declared_mileage,
    exposure=earned_years,
    thresholds=[5_000, 10_000, 15_000, 20_000],
    bw=200,
    excl_below=500,
    excl_above=1_000,
    mode='notch',
    fdr_level=0.05,
)

results = scanner.scan()
print(results)
# Returns a polars DataFrame:
#    threshold  B_hat  B_se   p_value  p_adjusted  significant
#         5000   0.18  0.04    0.0001      0.0004         True
#        10000   0.31  0.05    0.0000      0.0001         True
#        15000   0.09  0.06    0.1340      0.2680        False
#        20000   0.12  0.07    0.0870      0.1740        False
```

In this example, the 5k and 10k thresholds show significant bunching; the 15k and 20k thresholds do not (perhaps because the premium differential is smaller there, or the threshold is less salient to policyholders). The BH correction means this conclusion accounts for the multiple testing problem.

---

## FCA Consumer Duty evidence: BunchingReport

FCA Consumer Duty requires insurers to demonstrate that pricing outcomes are fair and that the firm has monitored for practices contrary to customer interests. A portfolio where mileage declarations bunch at a threshold — because the incentive to understate is strong — is not necessarily the insurer's fault, but it is a risk factor: the book is mispriced if the declared mileage distribution differs systematically from true mileage.

`BunchingReport` generates a self-contained HTML report suitable for regulatory evidence packs.

```python
from insurance_bunching import BunchingReport

report = BunchingReport(
    scanner_results=results,          # MultiThresholdScanner output
    portfolio_name='UK Motor 2025 Q4',
    regulatory_context='FCA Consumer Duty — mileage declaration integrity',
    output_path='bunching_report_q4.html',
)

report.generate()
```

The report includes: density plots with counterfactual overlays for each threshold, B-hat estimates with confidence intervals, BH-adjusted p-values, and a summary table with pass/fail against the configured FDR level. All charts are base64-embedded PNGs — the HTML file is fully self-contained, with no CDN dependencies that break in regulatory archives three years from now.

---

## What B-hat means in practice

The bunching mass B-hat is the ratio of excess mass to counterfactual mass in the bunching region: `B = (observed - counterfactual) / counterfactual`. A value of 0.30 means 30% more policies than expected are clustering at the threshold.

For mileage declarations, a B-hat of 0.30 at the 10,000-mile threshold means your book has 30% more policies declaring 9,500–10,000 miles than a smooth density would predict, with a corresponding deficit above 10,000. If the premium differential at this threshold is £150/year and the average mileage misstatement is 1,000 miles, you can translate this into an approximate leakage figure per 1,000 policies.

For sum insured, a large B-hat at £200k in household insurance suggests systematic underinsurance: customers are declaring below their true rebuild cost to avoid the premium increase at the £200k boundary. This is a fair value concern as well as a pricing accuracy concern.

For age declarations, a B-hat at the age 25 boundary tells you your declared-age distribution has been manipulated. The demographic pattern of policyholders just above 25 — far more than the population distribution would predict — is a fraud signal.

None of this requires matching declared data to ground truth. The bunching test works on the observable distribution alone. That is the technique's key property: it detects gaming from aggregate density patterns without needing to observe the true value.

---

## What the method does not do

The estimator requires a continuous running variable. NCD level (number of claim-free years) is discrete; standard bunching estimation does not apply directly because the "density" is a mass function over integer values. We defer discrete running variables to v1.1.

The iterative correction redistributes missing mass uniformly above the exclusion window. This is the simple choice; proportional redistribution (proportional to the counterfactual density) is more principled but not materially different for typical insurance data with reasonable bin widths.

Significant bunching at a threshold tells you there is gaming or adverse selection. It does not tell you which policyholders are gaming, by how much, or whether the gaming is fraudulent versus rational. Those are separate questions. For individual-level detection, you need something like `insurance-deploy` monitoring rules or `insurance-monitoring` drift alerts. The bunching test is a portfolio-level signal.

---

## This is new

To be direct: this is the first Python bunching estimator anywhere. The existing implementations are Mavrokonstantis (2022) `bunching` for R and Cattaneo et al. `bunching` for Stata. Neither applies to insurance data. Neither handles exposure weighting, multi-threshold FDR correction, or regulatory report generation.

The technique is not new — Saez (2010) is now 16 years old. The application is. The "hunger for bonus" theoretical literature on NCD (De Pril 1978; Lemaire 1995; Artis et al. 2002) models optimal claiming strategies analytically. It has never used bunching density estimation to detect manipulation empirically. These are separate literatures that have not been connected before.

We think there is real signal in UK motor and household data. If your pricing team has not looked at the distribution of declared mileage around 10,000 and 15,000, you should.

---

**[insurance-bunching on GitHub](https://github.com/burning-cost/insurance-bunching)** — 120 tests, MIT-licensed, PyPI.

---

## See also

- [Your champion/challenger test has no audit trail](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/) — `insurance-deploy` for structured model deployment: once you have identified and corrected for threshold gaming, getting the updated model into production with sign-off and rollback.
