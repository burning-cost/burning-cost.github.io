---
layout: post
title: "Your Stress Test Portfolio Was Made by Resampling"
date: 2026-03-19
author: Burning Cost
categories: [pricing, actuarial, libraries]
description: "Resampling your real portfolio to generate stress test scenarios destroys the joint tail structure that capital models depend on."
canonical_url: "https://burning-cost.github.io/2026/03/19/stress-test-resampling-vine-copula/"
tags: [vine-copula, synthetic-data, stress-testing, capital-modelling, solvency-ii, tail-dependence, tvar, internal-model, insurance-synthetic, python, resampling, diversification]
---

There is a step in most capital model validation workflows that nobody talks about much. You need a stress test portfolio: a collection of synthetic policies that looks like your real book but is either larger, shifted towards high-risk exposure, or configured to probe a specific tail scenario. You cannot use your real data — it's too small, or it contains live policyholder records, or the peer review team is external.

So you resample it. You draw rows from your portfolio with replacement, maybe apply some perturbations, and feed the result to your internal model. The model produces a Solvency II 99.5th percentile VaR. You present it to the independent validator. Nobody asks how the stress portfolio was generated.

The problem: resampling destroys tail dependence. And tail dependence is precisely what drives the number your capital model produces.

---

## Three methods that look reasonable and are not

**Bootstrap resampling** draws rows with replacement from your real portfolio. It preserves marginal distributions almost exactly — the KS statistic for each individual column is near zero, because you are literally sampling from the empirical distribution. The joint distribution is also preserved in the body. But bootstrap resampling cannot extrapolate beyond the joint support of your observed data. Your 99.5th percentile scenario requires combinations of risk factors that co-occur more severely than anything in your seed portfolio. A bootstrap sample has never seen those combinations and cannot generate them. The tail you care about for Solvency II is a tail the bootstrap has structurally excluded.

**Independent marginals** fits each column separately and draws independently. This is even more common than bootstrapping because it is easy to implement: fit a Gamma to claim severity, fit a NegBin to claim count, draw from each independently. The marginals are right. The joint distribution is rubbish. Our benchmark on an 8,000-policy UK motor portfolio shows a Spearman correlation of +0.001 (real: +0.502) for driver age versus NCD years. The correlation is not attenuated — it is gone. The result is 2.30% impossible combinations (young drivers with high NCD) against 0.32% in the real data: seven times the rate of physically implausible policies. A capital model run on this portfolio will produce a diversification credit it has not earned, because the risk factors are pretending to be independent when they are not.

**Gaussian copula** captures marginal correlation but imposes asymptotic independence in the tails. In the body of the joint distribution, a Gaussian copula is often adequate. In the tail — where Solvency II lives — it fails structurally. Two risk factors with Gaussian dependence become independent as you condition on increasingly extreme values. The bivariate Clayton copula, by contrast, has lower tail dependence: extreme values co-occur more than the marginal rates predict. Gumbel has upper tail dependence. The insurance joint tail is not Gaussian, and a capital model built on a Gaussian copula stress portfolio will systematically underestimate joint extremes.

---

## What vine copulas do differently

An R-vine factorises a multivariate copula into a sequence of bivariate copulas, one per edge in a sequence of trees. Each bivariate copula can be selected from a family set: Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, and their rotations. The algorithm (via [pyvinecopulib](https://github.com/vinecopulib/pyvinecopulib)) selects the best family for each pair by AIC. For pairs with tail dependence — young drivers and zero NCD, high vehicle group and high claim frequency — it will select Clayton or Gumbel rather than Gaussian. The tail structure is then explicitly encoded in the fitted vine.

`insurance-synthetic` wraps this into a portfolio-level API:

```python
from insurance_synthetic import InsuranceSynthesizer, SyntheticFidelityReport, uk_motor_schema

# Fit on your real or seed portfolio
synth = InsuranceSynthesizer(method='vine', random_state=42)
synth.fit(
    real_df,
    exposure_col='exposure',
    frequency_col='claim_count',
    severity_col='claim_amount',
)

# Inspect the vine structure — which families were selected for each pair
synth.summary()
```

The `summary()` output shows which bivariate family was selected at each tree level. If you see Clayton at the driver_age / ncd_years edge, the vine has captured lower tail dependence for that pair — precisely the relationship that drives extreme scenarios. This is auditable: you can show the peer review team exactly what dependence structure the stress portfolio encodes, and why.

---

## Generating a high-risk stress scenario

For a capital stress test you typically want a portfolio tilted towards high-risk exposure — young drivers, high vehicle groups, adverse weather regions — while remaining physically plausible. The `constraints` argument handles this:

```python
schema = uk_motor_schema()

# Generate 100,000-policy stress portfolio
# Constrain to higher-risk exposure profile
stress_df = synth.generate(
    100_000,
    constraints={
        'driver_age':    (17, 35),   # young drivers only
        'vehicle_group': (12, 20),   # high group vehicles
        'ncd_years':     (0, 3),     # limited bonus
        'exposure':      (0.5, 1.0), # near-full year exposures
    },
)
```

The vine copula generates samples from the full joint distribution and then rejection-samples within the constraints. Because the joint tail structure is preserved, the high-risk combinations that emerge reflect genuine co-occurrence patterns from the fitted model, not independent draws that happen to fall in the constrained region. The 99.5th percentile of this portfolio is a number your capital model can use.

---

## Validating the stress portfolio before you use it

Before feeding a synthetic portfolio to your internal model, you need to know whether it has preserved what matters. `SyntheticFidelityReport` runs three checks:

```python
report = SyntheticFidelityReport(
    real_df, stress_df,
    exposure_col='exposure',
    target_col='claim_count',
)

# Correlation structure
corr = report.correlation_report()
print(corr)
# Shows Spearman Frobenius norm vs real, and pairwise comparison

# Tail risk
tvar = report.tvar_ratio('claim_count', percentile=0.99)
print(f"TVaR ratio (claim_count, 99th pct): {tvar:.3f}")
# Target: close to 1.0 for like-for-like; intentionally > 1.0 for stress

# Full markdown report
print(report.to_markdown())
```

For a stress scenario, you are deliberately generating a portfolio with higher tail risk than the seed. The `tvar_ratio` will be above 1.0 and that is correct. What you want is the `correlation_report` to show low Frobenius norm — the relationship between risk factors should look like your real book, even as the marginal risk level is elevated.

---

## Benchmark numbers

These are the published benchmark results from the `insurance-synthetic` README, run post P0 fixes on Databricks serverless against an 8,000-policy UK motor portfolio with known DGP (ρ(driver_age, ncd_years) = +0.502, ρ(ncd_years, vehicle_group) = -0.338).

**Spearman Frobenius norm vs real:** vine = 0.315, naive = 0.880. The vine preserves correlation structure 64% better than naive independent sampling. The naive method destroys the age/NCD relationship almost entirely (Spearman +0.001 vs true +0.502).

**TVaR ratio at 99th percentile (claim_count):** vine = 1.59, naive = 1.01. The vine over-estimates tail risk by 59% on this dataset. The naive result of 1.01 looks right, but for the wrong reason: naive sampling has no joint tail structure, so it produces roughly the right univariate TVaR by chance while generating a joint distribution that is independent. A capital model run on the naive portfolio will compute diversification credits that do not reflect the real dependence between risk factors.

**Impossible combinations (young driver, high NCD):** real = 0.32%, vine = 0.26%, naive = 2.30%. The vine suppresses physically implausible policies through its correlation structure. Naive sampling generates seven times as many impossible rows as the real portfolio.

**TSTR Gini gap:** vine = 0.0006, naive = 0.0016. A CatBoost frequency model trained on either synthetic dataset generalises nearly as well as one trained on real data — the vine is not introducing spurious patterns that would confuse a pricing model.

---

## The capital modelling application

The Solvency II standard formula is notoriously insensitive to the internal correlation structure of a portfolio — it uses fixed correlation matrices between lines of business, not between individual risk factors. If you are running an internal model, though, the 99.5th percentile VaR is highly sensitive to joint tail behaviour. A capital model that treats risk factors as independent will overestimate diversification credit and understate required capital.

The practical workflow for internal model peer review:

1. Fit `InsuranceSynthesizer` on your real portfolio. Audit the vine structure with `summary()` — confirm that Clayton or Gumbel families are selected at the high-dependence edges.

2. Generate a stress portfolio with the risk profile your capital model is designed to probe. Use constraints to represent the exposure profile of the stress scenario.

3. Run `SyntheticFidelityReport`. Check Frobenius norm and confirm the correlation structure has been preserved. Document the TVaR ratio as an explicit measure of how much the stress portfolio has elevated tail risk vs the seed.

4. Feed to your internal model and record the diversification credit. The vine-based portfolio will typically produce a lower diversification credit than a naive-sampled portfolio, because the risk factors are correlated in the tail. That lower credit is correct.

For independent validators reviewing an internal model, the vine summary is a directly auditable artefact: it shows the bivariate copula family at each tree level, the fitted parameters, and the tail dependence coefficient implied by those parameters. You can present this to a Lloyd's model change panel or a Solvency II internal model approval team as evidence that the stress test portfolio captures genuine tail co-occurrence rather than independent marginals dressed up as stress.

---

## Limitations

**99.9th percentile underestimation.** The vine over-estimates tail risk at the 99th percentile by 59% (TVaR ratio 1.59), but for very extreme percentiles — 99.9th and beyond — the vine tends to underestimate. The fitted bivariate families are calibrated on the body of the joint distribution; extreme quantile extrapolation depends on the tail index of the selected family, which may not match the true tail behaviour of your book. Do not use this tool for 1-in-1000 catastrophe scenarios without further validation.

**Small book attenuation.** The vine is fitted on a seed portfolio. With fewer than roughly 5,000 policies, the correlation estimates become noisy and the fitted vine attenuates strong correlations — on our 8,000-policy benchmark, the age/NCD Spearman drops from 0.502 to 0.400 even in the vine case. For small books, generate more seed data from a synthetic portfolio with a known DGP (see [insurance-datasets](https://github.com/burning-cost/insurance-datasets)) before fitting the vine.

**Time-varying dependence.** The vine is a static model. It captures the average dependence structure over the period your seed portfolio was written. If your book has been shifting — perhaps because new distribution channels attract a different risk profile — the fitted vine reflects the historical mix, not the current composition. For a capital model run annually, this is usually acceptable. For a stress test that is supposed to represent an adverse future state, you should check whether the dependence structure of recent vintages differs from the historical average.

---

`insurance-synthetic` is open source under MIT at [github.com/burning-cost/insurance-synthetic](https://github.com/burning-cost/insurance-synthetic). Requires Python 3.10+. Install with `uv add insurance-synthetic`. The vine family audit is in `InsuranceSynthesizer.summary()`.

---

**Related posts:**
- [Your Synthetic Data Doesn't Know What Exposure Is](/2026/03/09/insurance-synthetic/) — the basics: exposure semantics, marginal fitting, TSTR fidelity, and why SDV and CTGAN produce portfolios that break the moment you run a pricing model on them
- [Your Rating Table Smoothing Is Wrong](/2026/03/18/your-rating-table-smoothing-is-wrong/) — another case where a naive approximation gives structurally incorrect results in the tail
- [Your Model Drift Alert Is Too Late](/2026/03/18/your-model-drift-alert-is-too-late/) — complementary monitoring: once you have a stress portfolio and a capital model, this is how you detect when the real portfolio has drifted away from the stress scenario assumptions
