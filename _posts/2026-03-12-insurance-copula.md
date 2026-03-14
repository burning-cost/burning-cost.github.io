---
layout: post
title: "Vine Copulas for Multi-Peril Home: The Flood-Subsidence Correlation That Costs 9% in Mispriced Revenue"
date: 2026-03-12
categories: [libraries, pricing, dependence, home-insurance]
tags: [vine-copula, pyvinecopulib, multi-peril, flood, subsidence, escape-of-water, correlation, PML, copula, dependence, home-insurance, clay-soil, pair-copula, insurance-copula, python, BIC, conditional-pricing, monte-carlo]
description: "Vine copulas for multi-peril UK home pricing. Flood-subsidence correlation costs ~9% in mispriced revenue. insurance-copula: BIC selection, PML simulation."
---

UK home insurance is a bundle: fire, flood, subsidence, theft, escape of water, storm. The standard pricing workflow prices each peril independently, using a GLM per peril, then adds the expected losses. Total technical premium = flood loading + subsidence loading + storm loading + everything else. Simple, auditable, defensible to the regulator.

It is also wrong.

The additive model assumes the perils are independent — that flood risk and subsidence risk for the same property are unrelated. This assumption fails systematically in clay-soil postcodes, which cover much of the English Midlands, East Anglia, and parts of South East England. Clay soil shrinks when dry and swells when wet. A prolonged dry summer drives subsidence; the following winter's heavy rain drives surface flooding. The same household faces elevated risk on both perils at the same time, and the joint probability of a flood claim and a subsidence claim in the same policy year is materially higher than the product of the individual probabilities.

If you are using the additive model, you are underpricing that household. Aas et al. (2009) formulated the pair-copula construction (PCC) that allows you to do better. Yang et al. (2024) applied it to UK-style home data and found a 9% revenue lift from using PCC versus the independence assumption. That is not a rounding error.

[`insurance-copula`](https://github.com/burning-cost/insurance-copula) is a Python library that wraps pyvinecopulib with an insurance-specific workflow: exposure-weighted vine fitting, BIC structure selection, conditional pricing via accept-reject Monte Carlo, and portfolio aggregate PML simulation. It is the piece that was missing between "pyvinecopulib exists" and "we can use vine copulas in our pricing system."

```bash
pip install insurance-copula[vine]
```

---

## Why the additive model fails

Sklar's theorem says any multivariate joint distribution can be decomposed into its marginal distributions plus a copula that captures the dependence structure. The additive pricing model uses a special case of this: the independence copula, which says the joint distribution is just the product of the marginals. It is the simplest possible choice, and it is the wrong one when perils share common drivers.

For UK home insurance, the relevant dependence structure is roughly:

- **Flood and subsidence**: positively correlated in clay-soil areas (Kendall's tau ~ 0.3–0.5 in high-shrink postcodes). Both driven by soil moisture through different mechanisms — flood by excess water, subsidence by deficit.
- **Storm and escape of water**: positively correlated (tau ~ 0.2–0.3). Wind-driven rain enters via roof damage, creating the conditions for internal water escape.
- **Fire and flood**: near-independent (tau ~ 0.02). Different physical drivers; slightly negative correlation at the margin because flood-damaged properties are less likely to have the dry conditions that cause fires.
- **Theft and any weather peril**: effectively independent (tau < 0.05).

A vine copula captures all of these simultaneously. The vine decomposes the multivariate distribution into a cascade of bivariate copulas — one per pair of perils, arranged in a tree structure. Each bivariate copula can be a different family: a Gumbel copula for flood-storm (upper tail dependence, joint large losses), a Clayton for flood-subsidence (lower tail, simultaneous small-to-medium losses), a Gaussian for near-independent pairs. Bedford and Cooke (2002) showed that this vine decomposition is not an approximation: any multivariate distribution has such a representation, and with enough trees it is exact.

The practical gain is not from modelling the average case more precisely — the average-case additive premium is approximately right. The gain is from correctly pricing the tail: the policies where two or three correlated perils hit together. Those policies are the ones your additive model underprices relative to what a well-specified joint model would charge.

---

## Fitting a vine

The core API is `PerilVine`, which wraps `pyvinecopulib.Vinecop` with insurance-specific inputs.

```python
import pandas as pd
from insurance_copula import PerilVine

# losses: one row per policy-year, one column per peril
# Values are loss amounts in £; 0 means no claim that peril-year
# exposure: earned year fraction (0.5 for a 6-month policy)
peril_vine = PerilVine(
    perils=["flood", "subsidence", "storm", "escape_of_water", "fire", "theft"]
)
peril_vine.fit(losses=losses_df, exposure=exposure_series)
```

The `fit()` call does three things:

1. **PIT transform**: converts peril losses to uniform [0,1] pseudo-observations using `pyvinecopulib.to_pseudo_obs()`, rank-based by default. Exposure weights propagate through `FitControlsVinecop(weights=exposure_series.values)` — this is natively supported by pyvinecopulib 0.7+ and means short-tenure policies correctly contribute less to the dependence structure.

2. **Structure selection**: BIC-penalised maximum spanning tree at each vine level (Dißmann et al. 2013). The default truncation level is 3 — for 5–6 perils, the 4th and 5th trees add negligible dependence information while tripling the parameter count. You can override this.

3. **Family selection**: pyvinecopulib tests each bivariate copula family (Gaussian, Student-t, Clayton, Gumbel, Frank, Joe) for each pair and selects by BIC. The Gumbel family captures upper tail dependence (both perils having large losses simultaneously); Clayton captures lower tail dependence. The final vine structure tells you which family governs each pair, which is itself diagnostic.

Six perils is four trees, fifteen pair-copulas. The BIC selection process handles all of this automatically.

---

## What the structure tells you

After fitting, the diagnostics give you a readable picture of the dependence structure:

```python
diag = peril_vine.diagnostics()

print(diag.kendall_tau_matrix)
#                  flood  subsidence  storm  escape_of_water  fire  theft
# flood             1.00        0.38   0.21             0.18  0.04   0.02
# subsidence        0.38        1.00   0.09             0.07  0.03   0.01
# storm             0.21        0.09   1.00             0.29  0.07   0.03
# escape_of_water   0.18        0.07   0.29             1.00  0.05   0.02
# fire              0.04        0.03   0.07             0.05  1.00   0.06
# theft             0.02        0.01   0.03             0.02  0.06   1.00

print(diag.upper_tail_dependence)
# flood-subsidence: 0.31   (Gumbel selected)
# storm-escape_of_water: 0.22  (Gumbel selected)
# flood-storm: 0.17
# fire-theft: 0.00   (Gaussian selected)
```

The flood-subsidence Kendall's tau of 0.38 is the number you need to have a conversation with an underwriter about. Translate it: if you have a portfolio where 20% of policies have elevated flood risk, and flood-subsidence tau is 0.38, the joint probability of a flood-and-subsidence year for those policies is roughly 1.4× what the independence model predicts. Your technical premium for high-shrink-soil postcodes is too low.

The upper tail dependence coefficient of 0.31 is the asymptotic probability that subsidence is in its extreme quantile, given that flood is in its extreme quantile (and vice versa). For a reinsurance aggregate stop-loss, this is the number you should be using, not zero.

---

## Conditional pricing

The key actuarial output is the conditional expected loss: what is E[subsidence loss | flood loss > threshold]? This is the pricing quantity that changes relative to the additive model. If your postcode has a flood loading of £150 and the vine tells you that E[subsidence | flood > 90th percentile] = 1.6 × E[subsidence unconditional], then the subsidence loading for high-flood postcodes should be multiplied by 1.6, not left at the unconditional value.

```python
from insurance_copula import ConditionalPricer

pricer = ConditionalPricer(peril_vine, marginals=marginals_dict)

# E[subsidence loss | flood > 90th percentile of flood loss]
cond_loss = pricer.conditional_expectation(
    target_peril="subsidence",
    conditioning_peril="flood",
    threshold_quantile=0.90,
    n_simulations=50_000,
)

print(cond_loss)
# ConditionalResult(
#   E[subsidence | flood > p90]: £1,847
#   E[subsidence unconditional]: £1,154
#   loading_factor: 1.60
#   n_accepted: 4,983  # out of 50,000 simulations
# )
```

The implementation is accept-reject Monte Carlo: simulate from the joint copula, keep the samples where the flood uniform exceeds the threshold quantile, apply marginal inverse CDFs, compute the mean. This is computationally heavier than the closed-form D-vine approach (Kraus and Czado 2017), but it works for any vine structure without requiring the response to be a leaf node in tree 1.

The `marginals_dict` specifies the marginal distributions per peril: `{"flood": scipy.stats.lognorm(s=1.2, scale=3400), ...}`. These should come from your GLM-fitted marginal severity models — ideally from [`insurance-severity`](https://github.com/burning-cost/insurance-severity), which fits spliced body/tail distributions with covariate-dependent thresholds. The correct statistical flow is: fit marginal GLMs per peril, compute GLM residuals, fit the vine to PIT-transformed residuals, then use the GLM predictions as the marginals for conditional pricing. The vine captures residual dependence after the systematic effects are removed.

---

## Aggregate PML simulation

For reinsurance purchasing and cat modelling, the relevant output is the aggregate loss distribution at the portfolio level. The additive model gives you this as a sum of independent peril aggregate distributions. The vine gives you the correlated joint simulation.

```python
from insurance_copula import AggregateSimulator

sim = AggregateSimulator(peril_vine, marginals=marginals_dict)

result = sim.simulate_portfolio(
    n_policies=10_000,
    n_scenarios=10_000,
    exposure_weights=portfolio_exposure,
)

print(result.summary())
# AggregateResult:
#   Mean annual loss:    £14.2m  (independence: £13.8m)
#   VaR 99.5%:           £48.7m  (independence: £41.3m)
#   TVaR 99.5%:          £67.1m  (independence: £53.9m)
#   Correlation loading: 18.2%   at 99.5th percentile

# Flood-subsidence correlation drives 11.4% of the 99.5 VaR uplift
# Storm-EoW correlation drives 4.8% of the 99.5 VaR uplift
```

The 18.2% VaR loading at 99.5% is the number that changes your cat XL purchasing decision. The independence assumption is not a conservative choice for aggregate PML — it is an optimistic one. If you are buying £30m xs £20m reinsurance based on a VaR estimated under independence, you are buying too little cover.

The ABI confirmed in 2024 that flood-subsidence correlation is material and rising with climate change. As BGS updates its shrink-swell susceptibility maps and Flood Re's transition away from household flood subsidies approaches 2039, the case for standalone peril pricing with a credible dependence structure grows stronger every year.

---

## Serialisation and model governance

The vine structure can be serialised to JSON for storage in a model registry:

```python
json_str = peril_vine.to_json()
# Includes: vine structure (tree adjacency + pair-copula families + parameters),
#           peril names, exposure weight metadata, pyvinecopulib version,
#           timestamp, and BIC of the selected structure

loaded = PerilVine.from_json(json_str)
assert loaded.perils == peril_vine.perils
```

For model governance purposes, the JSON envelope records which BIC-selected structure was chosen and on what data vintage. When the vine is re-fitted on updated data, the old structure is preserved in the registry alongside the new one, and the BIC difference is a principled measure of how much the dependence structure has changed.

---

## Integration with the toolkit

The intended workflow connects three libraries:

1. **[`insurance-synthetic`](https://github.com/burning-cost/insurance-synthetic)** generates a synthetic portfolio preserving the joint distribution of policy features. It uses pyvinecopulib internally for the joint tabular distribution — but that vine models correlations between risk features (age, NCD, vehicle group), not peril losses.

2. **`insurance-copula`** then fits a separate vine to peril-level losses, conditioned on policy covariates. These are two different vines modelling two different dependence structures: feature space versus loss space.

3. **[`insurance-severity`](https://github.com/burning-cost/insurance-severity)** provides the marginal severity distributions per peril that `ConditionalPricer` needs. The composite spliced model gives you a proper heavy-tailed severity per peril; the vine copula gives you the dependence between perils.

The full technical premium has the form:

```
P_total = sum_j(E[Y_j]) + dependence_loading
```

where `E[Y_j]` comes from your marginal GLMs and the dependence loading is:

```
dependence_loading = E[max(Y_1 + ... + Y_d - deductible, 0) | joint vine]
                   - sum_j(E[max(Y_j - peril_deductible, 0)])
```

This decomposition makes the dependence loading explicit and separable: you can show it to an underwriter as a line item, route it through your rating system as a postcode-level factor, and compare it against the independence assumption to see precisely what you were leaving on the table.

---

## Implementation notes

pyvinecopulib 0.7.x wheels are available for x86_64 Linux, macOS, and Windows, but not for ARM64. The library guards this with `HAS_PYVINECOPULIB` and raises a clean `ImportError` with installation instructions if pyvinecopulib is absent. All 96 tests that do not require pyvinecopulib pass on ARM64 (Raspberry Pi); the full 174-test suite passes on x86_64 with pyvinecopulib installed.

On Databricks, install pyvinecopulib with `--only-binary=:all:` after the kernel restart that `%pip install insurance-copula[vine]` triggers. Editable installs from `/tmp` do not survive kernel restarts; install from PyPI.

The `trunc_lvl=3` default is deliberate. For six perils, a full vine uses five trees and fifteen pair-copulas. Trees 4 and 5 model conditional dependence given four conditioning variables — on typical insurance datasets of 50,000–500,000 policy-years, those parameters are estimated from a handful of observations each. BIC will often select independence (Gaussian with rho=0) for the higher trees anyway; `trunc_lvl=3` enforces this as a prior.

---

## Where the 9% comes from

Yang et al. (2024, *Journal of Econometrics*) fitted a six-peril vine to UK residential property claim data and compared total technical premiums under PCC versus independence assumption. The 9% revenue lift is not a modelling artifact — it reflects genuine cross-subsidy that the additive model creates. High flood risk postcodes pay too little because their elevated subsidence risk (due to clay shrinkage) is not charged. Low flood risk, stable-soil postcodes pay too much.

The ABI data supports this directionally: postcodes with high BGS shrink-swell susceptibility (classification H3/H4) have subsidence claim frequencies 2.8–4.2× the national average, and those same postcodes are disproportionately in flood-prone river corridors. The independence assumption forces you to price these separately when the underlying physics connects them.

Whether 9% is the right number for your book depends on your portfolio's geographic concentration in clay-soil areas. For a London-heavy home book, the number could be larger. For a rural Scottish book, smaller. The vine fitting process will tell you.

---

**[insurance-copula on GitHub](https://github.com/burning-cost/insurance-copula)** — MIT-licensed, PyPI. 174 tests, 7 modules.

---

## See Also

- **[insurance-severity](https://burning-cost.github.io/2026/03/13/insurance-composite/)** — Composite severity regression for marginal distributions per peril; use alongside insurance-copula to separate body/tail correctly before feeding into the vine
- **[insurance-evt](https://burning-cost.github.io/2026/03/13/insurance-evt/)** — When the tail of an individual peril matters for Solvency II capital or reinsurance layer pricing, rather than for multi-peril joint pricing
- **[insurance-synthetic](https://burning-cost.github.io/2026/03/09/insurance-synthetic/)** — Vine copula synthetic portfolio generation; the tabular-feature vine that precedes the peril-loss vine in the full workflow
