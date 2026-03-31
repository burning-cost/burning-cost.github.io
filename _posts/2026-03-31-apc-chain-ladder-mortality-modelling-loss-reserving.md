---
layout: post
title: "Chain Ladder Through the Lens of Mortality Modelling: APC Decomposition for Loss Reserving"
date: 2026-03-31
categories: [reserving, methods]
tags: [chain-ladder, APC, age-period-cohort, mortality-modelling, Lee-Carter, reserving, IBNR, superimposed-inflation, clmplus, R]
description: "Pittarello, Hiabu, and Villegas (NAAJ 2025) showed that chain ladder is the age-only special case of an Age-Period-Cohort model borrowed from demography. We explain what this means, when it matters, and whether UK actuaries should care."
---

Chain ladder has a peculiar relationship with the rest of statistics. Every reserving actuary knows it, every regulator accepts it, and almost nobody can explain precisely what distribution it assumes or what estimator it corresponds to. It just works — and because it works, it has resisted formal unification with the broader statistical literature for decades.

A 2025 paper in the *North American Actuarial Journal* by Pittarello, Hiabu, and Villegas closes that gap in an unexpected direction. Chain ladder, it turns out, is mathematically equivalent to the age-only special case of an Age-Period-Cohort (APC) model — the same class of model that demographers use to decompose mortality rates into generation effects, calendar-year trends, and age-specific hazards. The paper ([arXiv:2301.03858](https://arxiv.org/abs/2301.03858)) extends this to add period and cohort effects on top of chain ladder, replicate it exactly, and provide a principled way to separate calendar-year shocks from structural development patterns.

This post explains the framework, works through the identifiability problem that makes APC models awkward, and gives an honest assessment of whether any of this matters for UK general insurance reserving in practice.

---

## The problem chain ladder cannot see

Chain ladder assumes that the development pattern — the sequence of factors that transforms each diagonal into the next — is stable across accident years and calendar years. In a mature, well-behaved line like motor own damage, that assumption is close enough to true that it does not cause serious problems.

Motor bodily injury is not that line. Since 2012, UK motor BI reserving has absorbed LASPO (reducing recoverable costs in 2013), a series of Ogden rate changes (dropping to -0.75% in 2017, then +0.25% in July 2019, then reset to +0.5% in December 2024), whiplash reforms under the Civil Liability Act 2018 that came into force in May 2021, and a COVID-induced claims freeze followed by a post-COVID surge in large PI claims. Every one of these is a calendar-year shock: it hits every accident year's open claims simultaneously at the moment it arrives on a specific diagonal.

Chain ladder has no vocabulary for this. An Ogden rate change in 2017 shows up as a step change in the development factor from the 2016-to-2017 diagonal — and the method dutifully averages that anomalous factor into the weighted mean with everything else, distorting the development pattern for all future projections.

The standard workaround is manual: an actuary excludes a diagonal, applies an explicit inflation adjustment, or weights development factors to down-weight the affected period. These interventions work but they are ad hoc. They leave no audit trail in the model structure, they vary between practitioners, and they compound when multiple calendar-year shocks have occurred in a triangle.

APC offers a systematic alternative: decompose the triangle explicitly into three components, estimate them simultaneously, and let the data tell you how large each effect is.

---

## The APC framework: age, period, cohort

The key move in Pittarello et al. is not to model claim amounts — it is to model *development rates*, which the paper treats as reversed-time hazard rates. This reformulation connects directly to mortality modelling.

For accident year *k* at development age *j*, define:

- Incremental payments: *X*_{kj}
- Exposure: *E*_{kj} = prior cumulative payments + ½ × current increment (the standard actuarial half-year assumption)
- Development rate: μ_{kj} = *X*_{kj} / *E*_{kj}

The development rate μ_{kj} is the proportion of remaining outstanding claims paid in period *j*. It behaves exactly like a mortality hazard rate in reversed development time — hence the connection to demography.

The APC model then says:

```
log(μ_{kj}) = a_j  +  c_{k+j}  +  g_k
              age     period      cohort
```

where:
- **a_j** captures the structural development pattern: how fast claims pay out at each development age regardless of which accident year they belong to
- **c_{k+j}** captures calendar-year effects: shocks that hit all open claims in the same calendar year *k+j* (Ogden changes, court award inflation, COVID)
- **g_k** captures accident-year effects: cohort-specific deviations in ultimate severity or reporting patterns for claims arising in year *k*

The bijection between development rates and chain-ladder factors is exact:

```
f_{kj} = (2 + μ_{kj}) / (2 - μ_{kj})
```

Under the age-only model — *c* and *g* terms set to zero — maximum likelihood estimation under a Poisson likelihood reproduces the volume-weighted chain-ladder development factors exactly. Not approximately. Not asymptotically. Algebraically exact. This is the central theoretical result of Hiabu (2017, *Scandinavian Actuarial Journal*), operationalised as a GLM by Pittarello et al.

The practical consequence is that you can add period and cohort effects incrementally on top of chain ladder and measure exactly how much of the triangle's variance they explain.

---

## The identifiability problem

APC models have a well-known algebraic headache. The three effects are not independently identified because age + period - cohort = constant. You can shift parameters between the three components along a two-parameter family of transformations without changing the fitted values:

```
a_j  →  a_j + φ₁ - j·φ₂
c_s  →  c_s + s·φ₂
g_k  →  g_k - φ₁ - k·φ₂
```

for any φ₁, φ₂. Individual effect estimates are therefore constraint-dependent and cannot be interpreted in isolation.

Pittarello et al. follow the `StMoMo` mortality modelling convention and impose three constraints:

1. Period effects sum to zero: Σ c_s = 0
2. Cohort effects sum to zero: Σ g_k = 0
3. Cohort effects have no linear trend: Σ k·g_k = 0

These three constraints remove the three degrees of freedom and produce a unique set of parameter estimates. Crucially, the fitted values μ_{kj} and the resulting reserve estimates are invariant to the choice of constraints — so the reserves are identified even though the individual effects are not.

The practical implication: do not interpret the magnitude of individual period or cohort parameters as if they were the "true" effect of, say, the 2017 Ogden change. Rely on predicted reserves and on model comparison across APC sub-models (a, ac, ap, apc).

The weak point is extrapolation. For an age-period (ap) model, future period effects beyond the observed triangle must be forecast — typically via a random walk with drift. That drift parameter is estimated from the constrained identified parameters, so forecast uncertainty compounds identification uncertainty. For short time series (ten calendar diagonals), the confidence interval on extrapolated period effects is wide enough to matter.

---

## Connection to existing methods

**Chain ladder** is the *a*-only model. Exact equivalence under Poisson likelihood, as above.

**England and Verrall (2002) ODP GLM** also replicates chain ladder but uses a different parameterisation: E[*X*_{kj}] = x × α_k × β_j, where α_k are row (origin-year) intercept parameters. Pittarello's hazard-rate formulation eliminates α_k because exposure *E*_{kj} absorbs the row structure. The practical difference is fewer parameters — roughly half as many — which matters when your triangle is only 10×10.

**Barnett-Zehnwirth PTF** is the most direct competitor. PTF (published in 2000, implemented in `chainladder-python` as `BarnettZehnwirth`) models log-incremental claim amounts with separate accident, development, and calendar trends via OLS on a lognormal assumption. The APC approach models log development rates with Poisson likelihood. Both handle all three trend dimensions. We think the hazard-rate parameterisation is theoretically cleaner — it has a direct actuarial interpretation and the chain-ladder bijection is a free result — but PTF has a longer track record in UK actuarial practice and the numeric differences in reserve estimates are usually small.

**Bornhuetter-Ferguson** is not addressed by APC. BF introduces an a priori expected loss ratio as an anchor for immature accident years; APC has no natural equivalent. For the most recent accident years in a long-tailed triangle, BF-style credibility blending is arguably more important than calendar-period decomposition, so treating APC and BF as substitutes is wrong. They address different problems.

---

## What a Python implementation would look like

There is no Python implementation of the hazard-rate APC model. The R `clmplus` package (v1.0.1 on CRAN, by Pittarello) does it correctly by wrapping `StMoMo`. The Jonas Harnau `apc` Python package ([PyPI: apc](https://pypi.org/project/apc/)) implements APC on claim amounts with overdispersed Poisson, which is closer to the England-Verrall ODP parameterisation than to Pittarello's hazard-rate approach.

A native Python implementation using `statsmodels` would look like this:

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from itertools import product

def triangle_to_hazard_rates(cum_triangle, eta=0.5):
    """
    Convert a cumulative payment triangle to development rates (hazard rates).
    Returns (mu, exposure) DataFrames indexed by (accident_year, dev_age).
    """
    inc = cum_triangle.diff(axis=1)
    inc.iloc[:, 0] = cum_triangle.iloc[:, 0]  # first column = cumulative

    # Exposure: prior cumulative + eta * current increment
    prior_cum = cum_triangle.shift(1, axis=1).fillna(0)
    exposure = prior_cum + eta * inc

    # Development rates: only where exposure > 0
    mu = inc / exposure.replace(0, np.nan)
    return mu, exposure


def build_apc_design_matrix(mu, exposure, model='ap'):
    """
    Build design matrix for APC GLM on upper triangle cells.
    model: 'a', 'ac', 'ap', or 'apc'
    Returns (X, y, weights) ready for statsmodels Poisson GLM.
    """
    n_origin, n_dev = mu.shape
    rows = []

    for k, j in product(range(n_origin), range(n_dev)):
        # upper triangle only (observed cells)
        if k + j < n_origin and not np.isnan(mu.iloc[k, j]):
            s = k + j  # calendar period (diagonal index)
            rows.append({
                'mu': mu.iloc[k, j],
                'exposure': exposure.iloc[k, j],
                'age': j,
                'period': s,
                'cohort': k,
            })

    df = pd.DataFrame(rows)

    # Development (age) dummies — always included
    age_dummies = pd.get_dummies(df['age'], prefix='age', drop_first=False)

    # Build design matrix based on chosen sub-model
    parts = [age_dummies]
    if 'p' in model:
        period_dummies = pd.get_dummies(df['period'], prefix='per', drop_first=True)
        parts.append(period_dummies)
    if 'c' in model:
        cohort_dummies = pd.get_dummies(df['cohort'], prefix='coh', drop_first=True)
        parts.append(cohort_dummies)

    X = pd.concat(parts, axis=1).astype(float)
    # Note: identification constraints (zero-sum, zero linear trend on
    # period/cohort) should be applied here via linear constraint matrix.
    # Omitted for brevity — use patsy or manual column transformations.

    y = df['mu'] * df['exposure']   # incremental payments as response
    weights = df['exposure']

    return X, y, weights, df


def fit_apc_chain_ladder(cum_triangle, model='ap', eta=0.5):
    """
    Fit APC chain ladder. Returns fitted GLM and the data frame.
    """
    mu, exposure = triangle_to_hazard_rates(cum_triangle, eta=eta)
    X, y, weights, df = build_apc_design_matrix(mu, exposure, model=model)

    # Poisson GLM with log link, exposure as offset
    glm = sm.GLM(
        y,
        X,
        family=sm.families.Poisson(link=sm.families.links.Log()),
        exposure=weights,    # offset = log(exposure)
    ).fit()

    return glm, df, mu, exposure
```

The missing piece in this sketch is (a) the identification constraints, which require projecting the cohort and period dummy columns onto a constrained subspace, and (b) the extrapolation step for future diagonals, which needs a short ARIMA or random walk fitted to the estimated period effects. That extrapolation is where `clmplus` does the heavy lifting via `StMoMo`. Implementing it in Python is around 850 lines of careful work — not intractable, but not a weekend project.

For practical use today, the `apc` package handles period and cohort decomposition on claim amounts with overdispersed Poisson and is worth trying before writing anything from scratch.

---

## UK GI practical value: where it helps and where it does not

**Motor BI (long-tail, high calendar-year sensitivity) — genuinely useful.** The age-period model is the natural framework for systematically absorbing Ogden changes, whiplash reforms, and claims inflation into a period-effect term rather than ad-hoc diagonal exclusions. This is equivalent to what PTF does, but the hazard-rate parameterisation is cleaner and the chain-ladder connection is exact. If your motor BI triangle is 15×15 or larger, the ap model is worth running alongside the standard chain ladder.

**Home subsidence — wrong tool.** Subsidence claims cluster in drought years (1976, 1995, 2018, 2022). That is a cohort effect in the accident-year dimension, not a period effect — the age-cohort (ac) model is more appropriate than age-period. But subsidence cohorts are so different from each other (a 1976 drought cohort behaves nothing like a 2022 one) that APC's assumption of smoothly extrapolatable ARIMA cohort effects is wrong. Explicit catastrophe-year adjustments are more honest.

**EL/PL liability — moderate value.** Disease cohorts (asbestos exposure decade in mesothelioma portfolios) are genuine cohort effects in the APC sense. Calendar-year effects from court award inflation and increasing legal costs are real period effects. If the triangle is large enough, APC is a reasonable framework. In practice, industrial disease portfolios are usually handled with purpose-built models that incorporate legislative exposure data, not generic reserving methods.

**Short-tail lines (home buildings, motor own damage) — not worth it.** Development typically completes in two to three years. A 5×5 triangle does not have enough cells to estimate age, period, and cohort effects simultaneously.

---

## The overfitting problem is real

A standard 10×10 run-off triangle has 55 observed cells. The parameter budget for each sub-model is:

| Model | Parameters (after constraints) | Cells per parameter |
|-------|-------------------------------|---------------------|
| Age (chain ladder) | 10 | 5.5 |
| Age-Cohort | 19 | 2.9 |
| Age-Period | 28 | 2.0 |
| Full APC | 37 | 1.5 |

The full APC model on a 10×10 triangle is near-saturated. It will fit the observed triangle almost perfectly — and that is precisely the problem. Extrapolation from a near-saturated model is unreliable because the fitted effects absorb noise as signal.

Pittarello et al. run cross-validation on 30 public triangles and find that simpler sub-models — ap or ac — outperform the full APC for triangles of typical size. The full model wins only when the triangle is large enough that all three effect dimensions are genuinely identified. In UK personal lines, where 10×15 is a generous triangle, the full APC model is marginal at best.

The compounding problem is extrapolation uncertainty. Period effects beyond the observed triangle are forecast via a random walk fitted to perhaps 15 calendar diagonals. The confidence interval on that forecast is wide. Bootstrap reserve distributions from the full APC model are uncomfortably wide — more honest than Mack, but possibly too honest for practical use in a reserve sign-off where the actuary needs to defend a central estimate.

---

## Honest assessment

The Pittarello/Hiabu/Villegas paper is the best theoretical work on chain ladder in a decade. The reversed-hazard connection to mortality modelling is mathematically tight, not a loose analogy. Reducing the ODP parameter count by eliminating origin-year intercepts is a genuine practical improvement. The exact algebraic equivalence to chain ladder under the age-only model closes a gap that has been annoying for years.

But the practical gains over Barnett-Zehnwirth PTF — which already handles all three trend dimensions and has been available in `chainladder-python` for years — are incremental rather than transformative. PTF models log-amounts via OLS; APC models log-hazard-rates via Poisson GLM. For most triangles, the reserve numbers will be similar. The theoretical elegance of the hazard-rate formulation does not change the data.

The R `clmplus` package is the only production-ready implementation. There is no Python equivalent and writing one properly is not trivial. Until a Python port appears, UK actuaries who want to decompose calendar-year effects from their development patterns are better served by running `BarnettZehnwirth` in `chainladder-python` alongside their standard chain ladder, and using the period trend parameter as a diagnostic for superimposed inflation.

Most UK personal lines triangles are too short to benefit from the full APC model. Motor BI is the one line where the age-period sub-model is genuinely worth trying — particularly now, with the December 2024 Ogden reset creating another identifiable diagonal shock. For everything else, chain ladder with careful diagonal selection remains the more defensible method.

---

## Further reading

- Pittarello, G., Hiabu, M., Villegas, A.M. (2025). "Replicating and Extending Chain-Ladder via an Age-Period-Cohort Structure on the Claim Development in a Run-Off Triangle." *North American Actuarial Journal*. [arXiv:2301.03858](https://arxiv.org/abs/2301.03858)
- Hiabu, M. (2017). "On the relationship between classical chain ladder and granular reserving." *Scandinavian Actuarial Journal*, 2017(8), 708–729.
- England, P.D., Verrall, R.J. (2002). "Stochastic Claims Reserving in General Insurance." *British Actuarial Journal*, 8(3), 443–518.
- Barnett, G., Zehnwirth, B. (2000). "Best Estimates for Reserves." *PCAS*, 87, 245–321.
- `clmplus` R package: [github.com/gpitt71/clmplus](https://github.com/gpitt71/clmplus)
- `apc` Python package (Harnau): [pypi.org/project/apc](https://pypi.org/project/apc/)
