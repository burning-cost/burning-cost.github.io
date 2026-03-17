---
layout: post
title: "Your Stress Test Portfolio Was Made by Resampling"
date: 2029-02-15
categories: [techniques]
tags: [synthetic-data, vine-copula, stress-testing, capital-modelling, tail-dependence, insurance-synthetic, python, uk-motor]
description: "Naive bootstrap resampling destroys the tail dependence structure that capital models depend on. Vine copula synthetic generation preserves it. We show exactly what resampling gets wrong and when it matters for regulatory stress tests."
---

When a capital actuary needs a stressed portfolio to test their model, the standard move is to resample. Take the real book, draw with replacement, maybe tilt the weights towards bad years or high-risk segments, and call it a stress scenario. This is fast, requires no infrastructure, and produces something that looks like real data because it is real data.

It is also wrong in exactly the way that matters most for capital.

The problem is tail dependence. Resampling from your own book cannot produce joint tail events you have not already observed. If your book has never simultaneously experienced a cluster of young drivers with high vehicle groups and zero NCD in a single bad year, resampling will never produce that scenario. Vine copula synthesis can, because it learns the joint dependence structure from the data and samples from it, including the tails the sample did not visit.

This post is about when that matters and how to do something about it.

---

## Three ways to get a stress portfolio wrong

Before reaching for `insurance-synthetic`, it is worth being precise about what each common approach actually does to the dependence structure.

**Bootstrap resampling.** You sample rows with replacement, possibly with tilted weights. The dependence structure of the synthetic portfolio matches the real portfolio row by row — because you are copying rows. What it cannot do is extrapolate beyond the observed joint distribution. The 99.5th percentile VaR scenario you care about for Solvency II — not the bad year you have seen, but the bad year you have not seen — requires joint realisations the resample cannot generate.

**Independent marginals.** You fit each column separately and sample them independently. The marginal distributions are fine. The dependence is gone entirely. Your synthetic young drivers have NCD distributions drawn independently from all young drivers in the book, rather than from the conditional distribution of young drivers. The resulting portfolio has around 7x more young-driver-with-high-NCD combinations than the real data — biologically and actuarially impossible combinations that will mislead any model trained or tested on it. From our benchmarks on an 8,000-policy UK motor book: 2.30% impossible young-driver/high-NCD combinations under independent sampling, versus 0.32% in the real data.

**Gaussian copula.** You model the dependence with a multivariate normal copula: better than independence, interpretable, fast. The Gaussian copula has no tail dependence in the technical sense. In the tail, joint extreme events become asymptotically independent under a Gaussian copula regardless of the correlation parameter. For a capital model operating at the 99.5th percentile, this is precisely the regime where a Gaussian copula fails you. The right-tail dependence between vehicle group and claim severity — the correlation that makes a bad book worse than the sum of its parts — gets underestimated.

The vine copula approach in `insurance-synthetic` uses Clayton and Gumbel bivariate families where the data supports them. Clayton captures lower-tail dependence (poor risks clustering together), Gumbel captures upper-tail (large claims co-occurring). `pyvinecopulib` selects the best bivariate family for each pair automatically.

---

## What the benchmark shows

We fitted an R-vine on an 8,000-policy UK motor seed portfolio with a known data-generating process: `rho(driver_age, ncd_years) = +0.502`, `rho(ncd_years, vehicle_group) = -0.338`. Then generated 8,000 synthetic policies using vine copula synthesis and naive independent sampling. Both methods target the same marginal distributions.

The Frobenius norm on the Spearman correlation matrix — the summary distance between the synthetic and real correlation structures — was 0.315 for the vine versus 0.880 for naive sampling. Vine is 64% better at preserving the overall correlation structure.

More specifically:

| Pair | Real | Vine | Naive |
|------|------|------|-------|
| driver_age vs ncd_years | +0.502 | +0.400 | +0.001 |
| ncd_years vs vehicle_group | -0.338 | -0.174 | -0.003 |
| exposure vs claim_count | +0.120 | +0.098 | +0.120 |

Naive sampling destroys the age/NCD relationship entirely. A synthetic portfolio where driver age and NCD years have correlation +0.001 is not a stressed version of your book — it is a portfolio from a parallel universe where years of driving experience provide no information about claims history.

The vine does not fully recover the strong age/NCD correlation (+0.400 vs +0.502 true). This is partly a sample-size effect — 8k rows with discrete integer columns is a hard case — and partly a discretisation artefact from applying a continuous copula to integer-valued columns. On continuous risk factors the vine performs closer to the seed correlations. We document this limitation rather than hide it.

---

## The tail severity problem

The TVaR ratio at the 99th percentile on `claim_count` was 1.59 for the vine versus 1.01 for naive sampling. The vine over-estimates tail risk by 59% on this dataset.

That sounds like a failure. It is not exactly that.

The naive result of 1.01 is coincidentally near-perfect on this single benchmark, but it is correct for the wrong reason. Naive sampling has no joint tail structure, so its apparent TVaR fidelity reflects marginal preservation, not any understanding of when large claims cluster. Apply the naive portfolio to a scenario where you need to understand which segments drive the tail — which combinations of risk factors co-occur in the worst losses — and the naive approach has nothing to offer. Its TVaR ratio looks right in aggregate; its tail composition is wrong.

The vine over-estimates. That is a calibration issue worth monitoring. `SyntheticFidelityReport.tvar_ratio()` gives you the number directly:

```python
ratio = report.tvar_ratio(col="claim_count", percentile=0.99)
# vine on this benchmark: 1.59
# naive on this benchmark: 1.01
# target for capital model stress testing: ≈ 1.0
```

For capital modelling work, you want to know the TVaR ratio before relying on the synthetic portfolio for any scenario where tail severity is the output of interest. If it is 1.59, you calibrate accordingly. If it is 0.85, your stress test is understating tail risk. Either way, the fidelity report tells you where you stand.

---

## Generating a stress portfolio

The most direct use is generating a portfolio with a specific risk composition:

```python
from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport, uk_motor_schema
import polars as pl

schema = uk_motor_schema()

synth = InsuranceSynthesizer(random_state=42)
synth.fit(
    real_df,
    exposure_col="exposure",
    frequency_col="claim_count",
    severity_col="claim_amount",
)

# Stressed high-risk book: young drivers, high vehicle groups, minimal NCD
stress_portfolio = synth.generate(
    50_000,
    constraints={
        "driver_age": (17, 28),
        "vehicle_group": (35, 50),
        "ncd_years": (0, 2),
        "exposure": (0.01, 1.0),
    },
)
```

The vine copula means the constraint is applied to the joint distribution, not to independent marginals. A 24-year-old driver in this stressed portfolio has realistic annual mileage, vehicle age, region, and payment method — sampled from the conditional distribution given the constraint. An independent-marginals approach would produce a 24-year-old with mileage and region drawn unconditionally, giving you implausible combinations (a high-mileage 24-year-old in rural Scotland with a vehicle group 48 might be plausible; a zero-mileage one is not).

For regulatory stress testing, this matters. Your capital model's diversification benefit — the reduction in capital requirement because risks don't all move together — needs to be tested with portfolios where the correlations are realistic. A naive resampled stressed portfolio may show more diversification than your real book has in stress because the resampling has not preserved the dependence structure at all.

---

## Validating the stress portfolio before using it

Do not use a synthetic stress portfolio for capital purposes without running the fidelity report.

```python
report = SyntheticFidelityReport(
    real_df,
    stress_portfolio,
    exposure_col="exposure",
    target_col="claim_count",
)

# Check the three layers
print(report.marginal_report())       # KS and Wasserstein per column
print(report.correlation_report())    # Spearman matrix comparison
tvar = report.tvar_ratio(col="claim_amount", percentile=0.99)
print(f"TVaR ratio (claim_amount, 99th pct): {tvar:.3f}")
```

The `correlation_report()` output will show you which pairwise correlations the vine has preserved well and which it has not. For capital modelling, the pairs you care about most are the risk factor combinations that drive the worst joint outcomes. If `vehicle_group vs claim_amount` has poor correlation preservation, your stressed portfolio is not capturing the vehicle-group-dependent severity that makes high-group segments so costly.

The full markdown report via `report.to_markdown()` is suitable for a model validation pack or a PRA internal model review.

---

## Scenario generation rather than just stress testing

Beyond simple risk composition constraints, the vine copula structure supports a second use: generating scenarios where a specific correlation shifts.

This is not a feature `insurance-synthetic` exposes directly via API in the current version. But the raw mechanism is accessible:

```python
# Access the fitted vine copula parameters
params = synth.get_params()
vine = params["vine_model"]  # pyvinecopulib RVineStructure

# Inspect which bivariate family was selected for each pair
synth.summary()
```

The `summary()` output shows the bivariate families selected for the first few vine trees. If the vehicle group / claim amount pair was fitted with a Gumbel copula (upper-tail dependence), you can see the fitted tau and understand quantitatively how strongly large claims cluster among high vehicle group policies. This is the kind of auditable structure you can present in an internal model peer review — something a neural synthetic approach cannot offer.

For Solvency II internal model purposes, being able to show the dependency assumptions in your stress scenarios, explain why those assumptions are appropriate for the UK motor book, and demonstrate that the synthetic portfolio faithfully reflects those assumptions, is half the validation argument.

---

## Where this breaks down

**Heavy tails beyond the 99th percentile.** Even with Clayton and Gumbel families, the vine copula will typically underestimate the 99.9th percentile of claim severity. The TVaR ratio at the 99th percentile might be reasonable (0.90+); at the 99.9th percentile it will typically be worse. For capital work targeting the 99.5th percentile VaR, validate carefully at the tail before relying on synthetic data as your primary scenario generator. It is a useful input to your stress testing toolkit, not a replacement for expert scenario judgement.

**Small books.** At 8,000 policies the vine recovers age/NCD correlation at +0.400 versus +0.502 true — a meaningful attenuation. At 3,000 policies the vine is working harder and the attenuation is worse. If your book is thin and you are generating stressed scenarios for a segment that has perhaps 1,000 observed policies, the vine's correlation estimates for that segment are noisy. The fidelity report will tell you this via wide Wasserstein distances and poor Spearman preservation.

**Time-varying dependence.** The vine is fitted on a static snapshot. If the dependence structure in your book shifts between soft and hard market conditions — as it does — a vine fitted on the past three years captures the average dependence, not the peaked tail dependence characteristic of a hard market. You might consider fitting separate vines on recent hard-market data and using the resulting stressed correlation structure as an input to your scenario generation.

---

## Getting started

```bash
pip install insurance-synthetic

# For the TSTR fidelity test (CatBoost required):
pip install "insurance-synthetic[fidelity]"
```

Source and full API: [github.com/burning-cost/insurance-synthetic](https://github.com/burning-cost/insurance-synthetic).

The fastest path to a useful stress portfolio: fit on your real book, generate 50,000 rows with constraints reflecting the stressed composition you care about, run `SyntheticFidelityReport`, check the TVaR ratio and the Spearman preservation on the pairs that matter for your capital model. If the TVaR ratio is above 0.85 and the correlation preservation on your key pairs is reasonable, you have a stress portfolio that is materially better than anything bootstrap resampling can produce. The bar is not perfection — the bar is whether the joint tail structure is close enough to be useful for the capital question at hand.

Resampling gets you a stressed composition. It does not get you stressed dependencies. For capital modelling, the latter is what you actually need.

---

**Related articles from Burning Cost:**
- [Why Generic Synthetic Data Fails Actuarial Fidelity Tests](/2026/03/09/insurance-synthetic/) — the earlier post on this library covering marginal fitting, exposure semantics, and the three-layer fidelity framework
- [Reserve Range with Conformal Guarantees](/2028/07/15/reserve-range-conformal-guarantee/) — quantile methods for capital-adjacent uncertainty quantification
- [Your Severity Model Assumes the Wrong Variance Structure](/2027/08/15/your-severity-model-assumes-the-wrong-variance/) — tail severity modelling before you get to the synthetic data step
