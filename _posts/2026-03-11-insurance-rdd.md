---
layout: post
title: "Does the Risk Actually Drop at 25? Using Regression Discontinuity to Test Your Age Threshold"
date: 2026-03-11
categories: [libraries, pricing, causal-inference]
tags: [RDD, regression-discontinuity, causal-inference, age-25, NCD, territory, postcode, FCA, Consumer-Duty, Poisson, GLM, geographic-RDD, insurance-rdd, python]
description: "UK motor insurers charge under-25s roughly three times the premium of 25-30 drivers. The risk discontinuity at age 25 is assumed, not demonstrated. insurance-rdd brings Regression Discontinuity Design to Python for the first time with insurance-specific defaults: exposure weighting, Poisson outcomes, geographic territory boundaries, and FCA Consumer Duty output."
---

UK motor insurers charge under-25s approximately three times the premium of 25-30 drivers. The ABI's own data confirms the scale of that cliff. What it does not confirm — and what pricing teams almost never formally test — is whether the observed claims risk drops by the same factor at age 25, or whether the pricing premium far exceeds the causal risk change.

The distinction matters. If your tariff relativity at the age-25 boundary is 3.0 but the causal rate ratio is 1.6, you are overcharging drivers just below 25 relative to their actual risk contribution. That is a Consumer Duty exposure. But you do not know whether you are in that position unless you have run the test.

[`insurance-rdd`](https://github.com/burning-cost/insurance-rdd) provides that test. It brings Regression Discontinuity Design — the standard econometric method for causal identification at thresholds — to Python for the first time with insurance-specific defaults: exposure weighting, Poisson and Gamma outcome families, geographic territory boundary analysis, and FCA Consumer Duty formatted output.

```bash
uv add insurance-rdd
```

---

## What RDD actually identifies

Regression Discontinuity Design exploits a simple fact: if treatment assignment switches at a threshold and nothing else changes discontinuously at that exact point, then any discontinuity in outcomes must be caused by the threshold crossing.

For the age-25 boundary: driver age is externally verified by DVLA records and cannot be manipulated. The premium treatment switches at the cutoff. If claim frequency drops at age 25 — after controlling for everything else that varies smoothly with age — that drop is causal. If it does not drop, or drops by less than the premium, the pricing relativity is not justified by observed risk.

The formal assumption is continuity of potential outcomes at the cutoff: absent the threshold treatment, the outcome trend would be smooth through the cutoff. For driver age, this is highly credible. For NCD level, where policyholders actively suppress claims near the maximum step boundary, it requires explicit caveats — and the library handles both cases.

The estimate is a Local Average Treatment Effect: the causal effect at the cutoff, for drivers right at the threshold. It is not an average over the full portfolio. That is a feature, not a limitation. The cutoff boundary is precisely where your tariff decision is most contestable.

---

## The Python gap

The core estimation methodology for RDD — the Cattaneo-Calonico-Titiunik (CCT) local polynomial estimator with bias-corrected robust inference — already exists in Python via the `rdrobust` package. We do not reimplement it.

What `rdrobust` does not provide:

- **Exposure weighting**: Insurance policies have variable exposure (45-day MTA, cancelled mid-term). Claim frequency is Y/t, not Y. Standard packages treat a 45-day policy identically to a 365-day policy.
- **Poisson and Gamma outcomes**: Claim counts are not Gaussian. Local polynomial regression on log-transformed counts is biased. The correct approach is local Poisson GLM with log link and log-exposure offset: a separate likelihood, not a transformation.
- **Geographic RDD**: Territory boundary analysis requires computing signed distance from each postcode centroid to the boundary line, then running RDD on that distance. SpatialRDD implements this in R (Lehner 2020). No Python equivalent exists.
- **Insurance presets**: Practical defaults for the age-25 donut radius (3 months, because age is reported in integer years and heaps at round numbers), the NCD manipulation warning, the FCA framing.

`insurance-rdd` fills these gaps while calling `rdrobust` for the core estimation.

---

## Using it: age-25 claim frequency

```python
from insurance_rdd import InsuranceRD, presets

rd = InsuranceRD(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,               # 25 years * 12 months
    data=df,
    outcome_type='poisson',
    exposure='exposure_years',
    preset=presets.AGE_25,    # loads donut=3 months, bandwidth defaults, FCA context
)
result = rd.fit()
print(result.summary())
```

The `AGE_25` preset does several things automatically. It sets a donut radius of 3 months — excluding observations between 297 and 303 months of age — because age reported in integer years creates a spike in the density at exactly 300 months that would otherwise distort the discontinuity estimate. It recommends bandwidth selection via `mserd` (mean-squared error optimal, separate bandwidths each side). And it carries the FCA context string for use in regulatory output.

The result gives you the log rate ratio `tau_bc` (bias-corrected), standard error, 95% confidence interval, p-value, and effective sample sizes on each side. The output you actually care about is one method call away:

```python
rr = result.rate_ratio()
# {'rate_ratio': 0.71, 'ci_lower': 0.58, 'ci_upper': 0.87, 'tau': -0.34, ...}
```

A rate ratio of 0.71 means claim frequency drops by 29% at age 25. If your tariff relativity at the same boundary implies a 67% premium reduction — i.e., you charge twice as much below 25 as above — you have a gap to explain.

---

## Methodologically correct Poisson outcomes

`InsuranceRD` with `outcome_type='poisson'` uses rdrobust's weighted OLS as an approximation. For sparse data near the threshold — common for age-specific cells at the boundary — the Gaussian assumption breaks down. `PoissonRD` implements the methodologically correct version: local polynomial Poisson regression with log-exposure offset.

```python
from insurance_rdd import PoissonRD

rd = PoissonRD(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,
    exposure='exposure_years',
    data=df,
    bandwidth=24.0,    # months; or let it select via cross-validation
    poly_order=1,
    n_boot=500,
)
result = rd.fit()
print(result.rate_ratio())
```

The local polynomial here is the Poisson score function — fitted separately on each side of the cutoff, with a kernel-weighted likelihood objective via `scipy.optimize`. The treatment effect `tau` is the difference in intercepts on the log scale; `exp(tau)` is the rate ratio at the boundary. This maps directly to the log-link GLM relativity framework pricing actuaries already use. The output is a number you can put next to your tariff factor on the same scale.

Bootstrap CIs are used throughout in v0.1.0. The analytical bias correction for non-Gaussian local regression requires the full CCT derivation adapted to the GLM score, which is non-trivial and is deferred to v0.2.

---

## Validity tests

RDD identification can fail in two ways: if the running variable is manipulable (people sort to one side of the cutoff), or if other factors jump discontinuously at the cutoff alongside the treatment.

Each test is a standalone class:

```python
from insurance_rdd import DensityTest, CovariateBalance, PlaceboTest

# McCrary density test — does the density of driver ages spike at 300 months?
density = DensityTest(
    running_var='driver_age_months',
    cutoff=300,
    data=df,
).fit()
print(density.summary())
# Density Test (rddensity / Cattaneo-Jansson-Ma 2018)
#   p-value: 0.41 — not significant (no manipulation concern)
# (Expected: DOB is externally verified, cannot be gamed)

# Covariate balance — do vehicle group, region jump at the cutoff?
balance = CovariateBalance(
    covariates=['vehicle_group', 'region'],
    running_var='driver_age_months',
    cutoff=300,
    data=df,
).fit()
print(balance.summary())
# vehicle_group:  tau=-0.02, p=0.71 — balanced
# region:         tau=0.03,  p=0.63 — balanced

# Placebo tests at false cutoffs (AGE_25 preset supplies these: 264, 282, 318, 336 months)
placebo = PlaceboTest(
    outcome='claim_count',
    running_var='driver_age_months',
    cutoff=300,
    data=df,
    placebo_cutoffs=[264.0, 282.0, 318.0, 336.0],
    outcome_type='poisson',
    exposure='exposure_years',
).fit()
print(placebo.summary())
# No significant effects at any placebo cutoff
```

The McCrary test will pass cleanly for driver age (date of birth is externally verified and biologically impossible to manipulate). For NCD level at the maximum step, the test will fail. Artis et al. (2002) documented claim withholding to preserve NCD, and the library knows this. The `NCD_MAX` preset flags the expected density failure and instructs you to interpret the estimate as a lower bound on the true causal effect, for non-manipulators only.

---

## NCD: multi-cutoff RDD

NCD has five step transitions in UK motor (0→1, 1→2, 2→3, 3→4, 4→5), each corresponding to a premium discount boundary. Rather than running five separate analyses, `MultiCutoffRD` pools them with inverse-variance weighting.

```python
from insurance_rdd import MultiCutoffRD

mc_rd = MultiCutoffRD(
    outcome='claim_count',
    running_var='ncd_level',
    cutoffs=[1, 2, 3, 4, 5],
    outcome_type='poisson',
    exposure='exposure_years',
    data=df,
    discrete=True,    # integer NCD levels — uses rdlocrand randomisation inference
)
mc_result = mc_rd.fit()

print(mc_result.pooled_effect())
# Pooled rate ratio: 0.94 (95% CI: 0.88, 1.01) — not significant at 5%

print(mc_result.cutoff_effects())
# NCD 0→1:  rate ratio 0.92, p=0.12
# NCD 1→2:  rate ratio 0.96, p=0.38
# NCD 2→3:  rate ratio 0.91, p=0.09
# NCD 3→4:  rate ratio 0.97, p=0.52
# NCD 4→5:  rate ratio 0.93, p=0.14 [density failure — manipulation expected]
```

Our prior is that NCD level has weak causal effects on subsequent claims frequency, once you compare policies right at each step boundary. NCD captures selection rather than a moral hazard effect of the discount itself. If the pooled effect comes back at 0.94 and not significant, your NCD pricing factor is valid from an adverse selection standpoint (lower NCD genuinely predicts higher risk), but the mechanism is not causal in the way a naive reading of the relativity implies. That distinction matters when you are explaining the factor under Consumer Duty: you are pricing observable risk correlation, not a causal intervention.

---

## Geographic RDD: territory boundary analysis

The novel piece. No Python library implements geographic RDD. SpatialRDD does it in R (Lehner 2020, following Keele and Titiunik 2015), and there is nothing equivalent on PyPI.

The methodology converts a 2D spatial problem into a 1D RDD by computing signed distance from each policy to the territory boundary: negative for territory A, positive for territory B. The cutoff is distance zero — the boundary itself. Policies right on either side of the boundary are otherwise similar; any discontinuity in claims frequency at the boundary is attributable to territory assignment rather than the gradual spatial variation in risk that exists further from the line.

```python
from insurance_rdd import GeographicRD

geo_rd = GeographicRD(
    outcome='claim_count',
    treatment_col='territory_band',      # 0 = territory A, 1 = territory B
    data=df,
    boundary_file='territory_AB.geojson',
    lat_col='lat',
    lon_col='lon',
    outcome_type='poisson',
    exposure='exposure_years',
    border_segment_fes=True,   # segment FEs control for heterogeneity along boundary
)
geo_result = geo_rd.fit()
print(geo_result.summary())
```

The `boundary_file` is any geopandas-readable format — GeoJSON, Shapefile, WKT. The library computes geodetic distances from each policy centroid to the nearest boundary point, assigns sign based on territory membership, and then runs standard RDD on those distances. Border segment fixed effects divide the boundary into 10 segments (configurable) and add segment indicators as covariates, controlling for the fact that different parts of a boundary line may cross rivers, motorways, or urban-rural transitions that correlate with risk independently of territory assignment.

The result tells you whether, for policies right at the boundary, being in territory B rather than territory A causes a measurable claims difference. If it does not, and your territory factors are substantially different either side of the line, that is a pricing question worth answering before the FCA asks it.

Install the geo dependency:

```bash
uv add "insurance-rdd[geo]"
```

---

## FCA Consumer Duty output

FCA PS22/9 (Consumer Duty) Sections 8.8-8.12 require firms to demonstrate that pricing factors are risk-reflective. RDD provides exactly the evidence required: a formal causal estimate of the claims risk discontinuity at the threshold, comparable directly to the tariff relativity applied.

The quickest path is `result.regulatory_report()` directly on the `RDResult`:

```python
print(result.regulatory_report(
    tariff_relativity=1.85,           # what your pricing model applies at this boundary
    threshold_name="Age 25 (motor)",
))
```

For a full document combining the RD estimate, density test, and balance results, use `ThresholdReport`:

```python
from insurance_rdd import ThresholdReport, ThresholdReportData

report_data = ThresholdReportData(
    rd_result=result,
    density_result=density,
    balance_result=balance,
    tariff_relativity=1.85,
    threshold_name="Age 25 (motor)",
)
print(ThresholdReport(report_data).markdown())
```

The output is a Markdown section with the causal rate ratio, the tariff relativity, and a verdict: CONSISTENT, OVER-PRICED, or UNDER-PRICED. OVER-PRICED is flagged as a potential Consumer Duty concern. The report includes the formal statistical framing — bandwidth, effective sample size, CCT bias correction — that you would need in an actuarial sign-off document.

We are explicit about what this is not: it is not a full actuarial review of a rating factor. It is one causal diagnostic, local to the threshold, for policies right at the boundary. A pricing team would combine this with portfolio-wide GLM analysis, market benchmarking, and claims data review. What RDD uniquely provides is the causal identification. A GLM relativity alone cannot claim that.

---

## Where it sits

The insurance-rdd library addresses a specific question: at a rating threshold where your tariff changes discontinuously, does the underlying claims risk change by the same amount? This is a question pricing teams should be asking routinely and almost never do formally.

The complementary tool is [`insurance-bunching`](https://github.com/burning-cost/insurance-bunching) — which asks not whether the threshold causes a risk change, but whether policyholders are bunching to game it. Bunching says: is the distribution of the running variable distorted at the threshold? RDD says: does crossing the threshold cause the outcome to change? The two are different questions with different implications. If you see bunching at NCD step 4 but no causal claims effect at that step, policyholders are optimising premium rather than risk. That is a different regulatory exposure than if bunching and risk both jump at the same point.

Run them together on the same threshold.

---

**[insurance-rdd on GitHub](https://github.com/burning-cost/insurance-rdd)** — MIT-licensed, PyPI. The first Python implementation of geographic RDD and the first RDD library with insurance-specific exposure, Poisson, and FCA Consumer Duty output.

---


---

**Related reading:**
- [When exp(beta) Lies: Confounding in GLM Rating Factors](/2026/03/05/your-rating-factor-might-be-confounded/) — the broader confounding problem in GLM rating factors; RDD is the tool when there is a sharp threshold in the running variable
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/02/25/causal-inference-for-insurance-pricing/) — DML as the alternative causal method when no threshold discontinuity exists
- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/) — SDID for evaluating rate changes when RDD is not applicable; the prospective complement to RDD's retrospective diagnosis
