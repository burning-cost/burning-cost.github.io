---
layout: post
title: "Why Generic Synthetic Data Fails Actuarial Fidelity Tests"
date: 2026-03-09
categories: [techniques, libraries]
tags: [synthetic-data, insurance-synthetic, vine-copula, exposure, fidelity, FCA, Consumer-Duty, privacy, python, motor]
description: "Generic synthetic data tools — SDV, CTGAN, TVAE — produce portfolios that look plausible column-by-column and break down as soon as you run a pricing model on them. Exposure weighting, Poisson frequency semantics, and tail behaviour are all wrong. insurance-synthetic fixes this with vine copulas, AIC-based marginal selection, and a three-layer fidelity report that actually tests whether your synthetic data is fit for actuarial use."
---

If you have tried to use SDV or CTGAN to generate synthetic insurance data, you have probably noticed something: it looks fine until it doesn't. Marginal distributions seem reasonable. The age column doesn't produce negative numbers. Run a frequency model on the output, though, and the estimated claim rate doesn't match the real portfolio. More precisely, it doesn't match *for any given exposure level*, because SDV treated exposure as just another numeric column and generated it independently of the claim count column sitting next to it.

This is not a bug in SDV. It is a genuine limitation: generic synthetic data tools don't have a concept of exposure. They have columns. Insurance pricing data has actuarial structure.

We built [`insurance-synthetic`](https://github.com/burning-cost/insurance-synthetic) to close that gap.

---

## What goes wrong with generic tools

The core problem is not technical sophistication. CTGAN is genuinely clever, and TABSYN (Amazon Science, ICLR 2024) produces impressive statistical fidelity on benchmark tabular datasets. The problem is that none of the generic tools model the semantics of an insurance portfolio.

**Exposure handling.** A UK motor book has policies with exposure anywhere from 0.01 to 1.0 policy years: mid-term cancellations, new business incepted partway through the observation window, policies still active at the reporting date. The claim count for a policy with 0.1 years' exposure should be drawn from Poisson(λ × 0.1), not from the same marginal as a full-year policy. When SDV fits a joint distribution across your columns and samples from it, it has no mechanism to enforce this. The frequency/exposure relationship degrades. A synthetic portfolio that looks balanced in aggregate will have annualised claim rates that vary wildly across exposure bands — exactly backwards from how the real data behaves.

**Zero-inflated severity.** Insurance losses are zero-inflated: most policies have no claims, and those that do produce right-skewed severity. CTGAN's mode-specific normalisation was designed for multimodal continuous distributions. It handles skewness, but it systematically underestimates the heaviest tails. The 99th percentile of simulated claim amounts tends to sit materially below the real 99th percentile. For a frequency model this is cosmetic. For pricing work (where your technical premium depends on expected large losses), it is a calibration error.

**Constraint violations.** A 2025 CAS paper testing amputation-imputation approaches against CTGAN on the French freMTPL2 dataset found that CTGAN generated drivers under 18 years old. This is not a rounding artefact; it is what happens when a neural network trained on continuous representations has no hard constraint on driver age. In the UK, a policy with a 16-year-old main driver is impossible. Synthetic portfolios containing such records are useless for model validation.

**No actuarial fidelity metrics.** SDV's built-in quality report computes column-wise KS statistics and pairwise correlation metrics. These are fine as far as they go. They tell you nothing about whether your synthetic data preserves TVaR at the 99th percentile, whether a GLM trained on synthetic produces the same rating factor relativities as one trained on real data, or whether the frequency/severity relationship survives synthesis. These are the questions pricing actuaries actually care about.

The academic literature is similarly limited. The published work on insurance synthetic data (Kuo (2019) on freMTPL2, Côté and Hartman (2021) in *Variance*, the CAS-funded benchmarking) focuses on French motor TPL. There is no published Python tool with actuarial fidelity metrics, exposure semantics, and UK-specific structure. SynthETIC (Avanzi et al., 2021) is excellent for reserving simulation but it is R-only and process-based: it simulates claims from a specified process rather than learning from your real data.

---

## The vine copula choice

The first design question was what dependency model to use. The options are: neural (CTGAN, TVAE, TabDDPM, TABSYN) or vine copula.

We chose vine copulas. The reasons are not primarily performance: TABSYN, the current state-of-the-art as of early 2024, reduces column-wise distribution error by 86% relative to CTGAN in benchmark comparisons. The reasons are actuarial.

Vine copulas are already part of the actuarial mental model. Capital actuaries have been using Gaussian and Clayton copulas in Solvency II internal models for over a decade. A vine copula structure is interpretable: you can read off which pairs of rating factors have tail dependence, show it to a peer reviewer, and explain what assumption you are making. A CTGAN model is a neural network with conditional batch normalisation; you cannot audit what dependency structure it has learned.

More practically: vine copulas separate marginal fitting from dependency modelling. We fit the best marginal for each column independently, then model the copula on the probability-integral-transformed data. This gives us a clean place to plug in insurance-specific marginals (Poisson for frequency, LogNormal or Gamma for severity) without fighting the neural architecture. A Gaussian copula would have been faster to implement; we use an R-vine (implemented via `pyvinecopulib`, a Python binding to the C++ `vinecopulib`, v0.7.5, October 2025) because it can capture tail dependence that a Gaussian copula misses. The young driver + high vehicle group + zero NCD combination is more dangerous than the product of the three marginal risks suggests. Tail dependence, not correlation, captures this.

---

## How insurance-synthetic works

Install it:

```bash
uv add insurance-synthetic

# With TSTR fidelity testing (requires CatBoost):
uv add "insurance-synthetic[fidelity]"
```

The minimal workflow is three steps:

```python
import polars as pl
from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport

synth = InsuranceSynthesizer()
synth.fit(real_df, exposure_col="exposure", frequency_col="claim_count")
synthetic_df = synth.generate(50_000)

report = SyntheticFidelityReport(real_df, synthetic_df)
print(report.to_markdown())
```

### Marginal fitting

`fit()` runs AIC-based selection over Gamma, LogNormal, Normal, Beta, Exponential, and Weibull for continuous columns; Poisson and Negative Binomial for discrete integer columns. String and categorical columns get empirical categorical marginals.

The frequency column (`claim_count` in the example above) gets fitted as a marginal for the copula, but generation is handled differently. The library stores the overall exposure-adjusted rate λ = total_claims / total_exposure at fit time. When you call `generate()`, the copula samples an exposure value from the exposure marginal, and the frequency column is then drawn from Poisson(λ × exposure). This is the actuarially correct procedure: it enforces the frequency/exposure relationship that a generic synthetic tool will break.

You can see what the fitter selected with `synth.summary()`:

```
InsuranceSynthesizer — fitted model summary
==================================================

Method: vine
Frequency column: 'claim_count' (rate lambda = 0.0731 claims/year)
Exposure column: 'exposure'

Fitted marginals:
  Column                    Kind         Family                      AIC
  -------------------------------------------------------------------------
  driver_age                continuous   beta                     312445.2
  vehicle_age               continuous   gamma                    289332.1
  vehicle_group             discrete     poisson                  191204.7
  region                    categorical  Categorical(12 levels)         —
  ncd_years                 discrete     nbinom                   204891.3
  cover_type                categorical  Categorical(3 levels)          —
  payment_method            categorical  Categorical(3 levels)          —
  annual_mileage            continuous   lognorm                  421038.4
  exposure                  continuous   beta                     -18442.1
  claim_count               discrete     poisson                   44201.8
  claim_amount              continuous   gamma                    378291.3
```

### Dependency modelling

After PIT-transforming every column to uniform [0,1], `fit()` fits an R-vine copula on the resulting matrix. Discrete columns get jitter applied across the CDF step before the vine is fitted, which is the standard treatment for fitting copulas to discrete data. Categorical columns are integer-encoded and jittered similarly.

The vine copula selects the best bivariate copula family for each pair: Gaussian, Clayton, Gumbel, Frank, or independence, by default. Clayton captures lower-tail dependence (correlated poor risks); Gumbel captures upper-tail (correlated large losses). A Gaussian copula would miss both.

For high-dimensional data (many rating factors), you can truncate the vine:

```python
synth = InsuranceSynthesizer(trunc_lvl=3, n_threads=4)
```

Truncating at level 3 means the vine only fits pairwise copulas in the first three trees and assumes independence at higher levels. This is a good speed/quality tradeoff for 15+ columns, and it is the standard approach in practice.

### Constraint enforcement

SDV generates drivers under 18. insurance-synthetic lets you specify hard constraints:

```python
schema = uk_motor_schema()

synthetic_df = synth.generate(
    50_000,
    constraints=schema["constraints"],
)
```

The UK motor schema defines:

```python
{
    "driver_age": (17, 90),
    "vehicle_age": (0, 25),
    "vehicle_group": (1, 50),
    "ncd_years": (0, 25),
    "annual_mileage": (1000, 50000),
    "exposure": (0.01, 1.0),
    "claim_count": (0, 10),
    "claim_amount": (0.0, 500_000.0),
}
```

Constraint violations are resampled iteratively from the vine copula until the pool of valid rows is large enough. In practice, on a well-fitted model with reasonable constraints, fewer than 1% of rows need resampling.

There is also a pre-built employer's liability schema (`uk_employer_liability_schema`) with payroll as the exposure base, a different structure from motor but the same API.

---

## Three-layer fidelity reporting

Generating synthetic data without measuring whether it is any good is not useful. `SyntheticFidelityReport` tests fidelity at three levels.

### Layer 1: marginal fidelity

Column-by-column KS statistics and Wasserstein distances, with both real and synthetic means and standard deviations. This is the same layer SDV reports.

```python
report = SyntheticFidelityReport(
    real_df,
    synthetic_df,
    exposure_col="exposure",
    target_col="claim_count",
)

print(report.marginal_report())
```

A KS statistic below 0.05 is excellent. Wasserstein normalised by the real column standard deviation below 0.10 is good. If you are seeing Wasserstein above 0.20 on a column, either the marginal family selection went wrong for that column or the vine copula is distorting it through a poor dependency assumption.

### Layer 2: actuarial fidelity

This is the layer that generic tools don't provide.

**TVaR ratio.** Tail Value at Risk at the 99th percentile, real vs synthetic:

```python
ratio = report.tvar_ratio(col="claim_amount", percentile=0.99)
# Target: ratio ≈ 1.0
# < 1.0: synthetic underestimates tail severity
# > 1.0: synthetic overestimates it
```

A ratio of 0.75 means your synthetic claim_amount distribution produces a 99th percentile conditional mean 25% below the real portfolio. This is the number that tells you whether your severity model will be miscalibrated if trained on the synthetic data.

**Exposure-weighted KS.** Standard KS weights every policy equally. But pricing teams care more about full-year policies (exposure = 1.0) than mid-term ones. The exposure-weighted KS resamples both distributions proportional to exposure before computing the statistic:

```python
ew_ks = report.exposure_weighted_ks(col="driver_age")
```

If your exposure-weighted KS on driver age is materially worse than the unweighted version, the synthesiser is doing a poorer job on the full-year policies, the ones that drive your premium volume.

### Layer 3: predictive fidelity (TSTR)

The most demanding test. Train CatBoost on the synthetic data, evaluate on held-out real data. Compare to the Gini of a model trained on real data:

```python
gini_gap = report.tstr_score(test_fraction=0.2, catboost_iterations=200)
# Closer to 0 is better
# > 0.05 (5 Gini points): the synthetic data is losing meaningful signal
```

We use Gini rather than RMSE because Gini measures rank ordering (the discriminating power of the model), which is what a pricing model's frequency score is actually being used for. A synthetic portfolio that produces a model with a 0.02 Gini gap on a held-out real test set is providing enough signal to be useful for model validation and development work.

The full `to_markdown()` report pulls all three layers into a single document suitable for a Confluence page or a peer review pack.

---

## What you can actually use this for

### Model validation without data sharing

Suppose you want to validate a new vehicle group relativities approach with an external consultant, or you want to share your book with a reinsurer for pricing benchmarking. You cannot hand over the real policyholder data. You can hand over a synthetic portfolio that passes the three-layer fidelity report: the consultant gets a realistic dataset to work with, you get the validation, and you have not shared any personal data.

The TVaR ratio tells you whether the extreme loss behaviour (the part the reinsurer will care most about) is preserved. If the TVaR ratio is 0.90 or above, the synthetic data is telling a story about tail risk that is close to the real one.

### FCA Consumer Duty fairness testing

FCA Consumer Duty (PS22/9, effective July 2023) requires firms to demonstrate fair outcomes across customer groups, including those with protected characteristics. The 2024 multi-firm FCA insurance review found most firms were not monitoring outcomes adequately across customer segments.

The specific problem for synthetic data: if you want to test whether your pricing model produces materially different outcomes for otherwise identical customers who differ only on a protected characteristic, you need a realistic portfolio to run the test on. A synthetic portfolio that does not preserve the correlations between protected characteristics (age, gender) and rating factors (vehicle choice, mileage, region) will produce misleading counterfactuals.

```python
synth = InsuranceSynthesizer()
synth.fit(real_df, exposure_col="exposure", frequency_col="claim_count")

# Generate a large counterfactual portfolio for fairness testing
test_portfolio = synth.generate(100_000, constraints=schema["constraints"])

# The vine copula preserves the correlation structure between
# age, vehicle_group, annual_mileage, and region — so the
# counterfactual customers are internally consistent
```

This is the simple counterfactual approach: it generates plausible customers drawn from the fitted joint distribution. Causal counterfactuals (flip one variable, resample all correlated variables from their conditional distribution) require a structural causal model, which is a research-grade extension we have not implemented in v1. But for basic segment representation and fairness monitoring, the vine copula approach gives you a realistic synthetic pool to draw from.

### Stress testing

You can generate portfolios with deliberately skewed compositions to test how the pricing model behaves:

```python
# High-risk synthetic book: young drivers, high vehicle groups
synth_high_risk = synth.generate(
    20_000,
    constraints={
        "driver_age": (17, 25),
        "vehicle_group": (30, 50),
        "ncd_years": (0, 3),
    },
)

# Does the model price this segment consistently?
```

The vine copula means this is not just a random sample of young drivers: it samples from the full joint distribution of all columns conditional on the constraint. A 23-year-old in the constrained sample will have a realistic joint distribution of vehicle age, annual mileage, region, and payment method — not an implausible combination that a simple filter on age would produce.

### Thin data augmentation

Small books, niche affinity schemes, or high-value segments can have too few policies for stable model development. Synthetic data augmentation is a way to supplement without inflating confidence in the model: mix real and synthetic data for training, evaluate only on real data. The TSTR Gini gap gives you the calibration you need to know how much signal the synthetic data is adding versus noise.

We would caution here: research comparing 1:1 and 1:10 synthetic expansion consistently finds utility degradation at 1:10 ratios. If you are synthetically expanding a thin book by 10x and expecting the resulting model to behave as if trained on 10x the real data, you will be disappointed. A 1:2 or 1:3 expansion is a reasonable working assumption.

---

## Honest limitations

The vine copula handles high-cardinality categoricals badly. `region` with 12 levels is fine. An occupation code column with 300 levels requires discretisation or jittering that can introduce artefacts: the categorical encoding maps each level to a CDF step, and the jitter across steps can smear levels together in the copula space. We document this limitation. It is not unique to insurance-synthetic; it applies to any copula-based approach.

Heavy-tail preservation is hard. Even with tail-dependent copula families (Clayton, Gumbel), the 99.9th percentile of claim severity tends to be underestimated. The TVaR ratio at the 99th percentile can be 0.90+; at the 99.9th percentile it will typically be lower. For capital modelling purposes, where you care about the 99.5th percentile VaR, synthetic data from any tool should be validated carefully against the real distribution's extremes.

We do not provide differential privacy guarantees in v1. The library does not copy real records: the vine copula samples new points from a learned distribution, so Distance to Closest Record is non-zero by construction. But formal differential privacy (PrivBayes-style, with a provable epsilon bound) is a v2 consideration.

---

## Getting started

```bash
uv add insurance-synthetic

# With fidelity testing:
uv add "insurance-synthetic[fidelity]"
```

Source and tests on [GitHub](https://github.com/burning-cost/insurance-synthetic).

The best starting point is `synth.summary()` after fitting: look at which marginal family was selected per column and whether the AIC differences between families are large (confident selection) or small (the data doesn't strongly prefer one family over another, and the choice matters less). Then run `SyntheticFidelityReport.marginal_report()` and look at the columns with the worst Wasserstein distance. If it is the claim amount column, check whether your severity data needs pre-filtering for large outliers before fitting.

The library is at v0.1.0. API stability will improve as we use it in anger on real UK motor books. If you hit edge cases, particularly on high-cardinality categorical columns or unusual exposure distributions, we want to know.

The generic tools are not going to add Tweedie marginals or exposure-weighted KS statistics any time soon. DataCebo's SDV roadmap is finance and healthcare. The actuarial gap is real, and it is ours to fill.

---

**Related articles from Burning Cost:**
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/13/insurance-validation/)
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Why Your Cross-Validation is Lying to You](/2026/02/23/why-your-cross-validation-is-lying-to-you/)
