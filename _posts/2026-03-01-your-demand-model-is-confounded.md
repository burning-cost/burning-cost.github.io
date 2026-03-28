---
layout: post
title: "Double Machine Learning for Insurance Price Elasticity"
date: 2026-03-01
categories: [techniques]
tags: [demand, elasticity, DML, double-machine-learning, conversion, retention, FCA, GIPP, PS21-11, ENBP, insurance-optimise, python, motor, catboost, polars]
description: "Double Machine Learning fixes biased price elasticity in insurance quote data. insurance-optimise: conversion, retention, elasticity, demand curves, FCA GIPP."
---

Your renewal pricing model has a price elasticity coefficient. Your team uses it. It is wrong.

Not wrong as in: someone made an arithmetic error. Wrong as in: the quantity it estimates is not the quantity you need for sound pricing decisions. It measures the association between price change and lapse, and it embeds a confound that naive regression cannot remove. In observational insurance data, this confound can materially overstate true price sensitivity. A renewal optimiser built on an overstated elasticity gives back margin to customers who would have renewed anyway.

We built [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) to fix this. This post explains the problem in detail and shows how the library addresses it.

---

## The confound, precisely

Here is what your renewal data actually contains.

Each row is a renewing policy. You observe the renewal price, the price change from last year, and whether the customer renewed. You regress renewal indicator on price change and call the coefficient an elasticity.

The problem: the price change was not randomly assigned. Your pricing model set it, using risk factors. Higher-risk customers received larger premium increases because your actuarial models said they cost more. Higher-risk customers also lapse at higher rates, for reasons that are not entirely about price. They have more claims. They attract fewer alternative insurers willing to cover them. Their financial behaviour is correlated with their claims behaviour in ways your model does not fully capture.

So your regression has conflated two effects: the genuine causal effect of price on renewal probability, and the correlation between high-risk status and lapse propensity. The naive coefficient is a mixture of both. You cannot tell from the regression output how large each component is.

The standard fix is to include technical premium as a control variable. This is better than nothing. But it only removes the variation explained by your risk model, and your risk model is not perfect. Any residual risk variation that correlates with both the price change and the renewal decision will still bias the coefficient. If your pricing model misses a material risk factor - and all of them do - the confound remains.

The same structure appears in conversion modelling. PCW business is the clearest case. Your technical premium for a given risk is high, so you quote high, so you convert poorly on that risk class. A naive regression of conversion on quoted price absorbs the risk-driven price variation and the risk-driven demand variation. The coefficient overstates price sensitivity for the high-risk segments most affected by this problem.

---

## Why this matters commercially

A renewal optimiser built on an overstated elasticity will systematically give back too much margin.

Take a concrete example. Suppose the true causal elasticity is -0.023: a 1 percentage-point premium increase causes a 2.3 percentage-point reduction in renewal probability. Your naive GLM estimate is -0.047. Guelman and Guillén (2014) demonstrated this type of confounding using propensity score methods on automobile insurance renewal data, finding that the causal estimate was substantially lower in absolute magnitude than the naive association. If the optimiser thinks P(renewal) falls twice as fast with price as it actually does, it will undercut more aggressively than it should. On a renewal book of 200,000 policies, using the wrong elasticity can represent millions of pounds in unnecessary margin concession annually.

Post-PS21/11, this matters even more. The FCA's GIPP remedies (PS21/11, effective January 2022) banned price-walking: you cannot charge renewing customers more than the equivalent new business price through the same channel. The direction of the optimisation is now one-sided. You can discount but you cannot surcharge. Every percentage point of discount you apply unnecessarily is pure margin loss with no regulatory upside.

---

## The Double Machine Learning fix

DML isolates the causal effect by partialling out confounders from both the treatment and the outcome using flexible ML models. For the full mathematical procedure and theoretical guarantees, see [DML for Insurance: Practical Benchmarks and Pitfalls](/2026/03/09/dml-insurance-benchmarks/).

The practical upshot: the residualised treatment `D_tilde` is the part of the price variation not explained by risk characteristics - the exogenous variation from seasonal rate changes, portfolio rebalancing, and manual underwriting adjustments. Regressing residualised outcomes on residualised treatment isolates the causal price effect and produces a valid confidence interval.

---

## The full pipeline: conversion, retention, elasticity, demand curve

`insurance-optimise` implements this as four connected components. For the full `ConversionModel` and `RetentionModel` setup. What follows here is the causal estimation layer that sits on top of those base models.

### Debiased elasticity estimation

The conversion and retention models give you naive marginal effects. The `ElasticityEstimator` gives you the causal estimate:

```python
from insurance_optimise.demand import ElasticityEstimator

est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area",
                  "channel", "month"],
    n_folds=5,
    heterogeneous=False,  # global elasticity; see below for segment-level
)

est.fit(df_quotes)
print(est.summary())
```

```
Price Elasticity (DML)
  Treatment: log_price_ratio
  Outcome:   converted
  Estimate:  -0.312
  Std Error:  0.021
  95% CI:    (-0.353, -0.271)
  N:          187,432
```

A 10% increase in the price-to-technical-premium ratio reduces conversion probability by approximately 3.1 percentage points at the average conversion rate, with a confidence interval of (-3.5pp, -2.7pp). This is precise enough to use in an optimiser.

Compare this with `model.marginal_effect()` from the ConversionModel. If the DML estimate and the naive estimate are within 10% of each other, confounding is not material for your portfolio and your GLM was giving you adequate guidance. If they diverge materially - which is common on PCW conversion data - you have been pricing with a biased elasticity.

For segment-level elasticity, set `heterogeneous=True`. This uses CausalForestDML from Microsoft Research's `econml` library under the hood:

```python
est = ElasticityEstimator(
    outcome_col="converted",
    treatment_col="log_price_ratio",
    feature_cols=["age_band", "vehicle_group", "ncd_years", "area", "channel"],
    n_folds=5,
    heterogeneous=True,
)

est.fit(df_quotes)
per_customer_elasticity = est.effect(df_quotes)
```

The heterogeneous result will consistently show higher elasticity for young drivers and PCW customers, and lower elasticity for high-NCD, long-tenure customers. A single portfolio-average elasticity loses this heterogeneity, and any optimiser working from one number will misdirect discount budget.

The DML approach has practical data requirements. Minimum 50,000 quotes with meaningful price variation. Technical premium must have been stored at quote time, not recalculated retrospectively. Rate reviews generate the exogenous price variation DML needs to work - if you applied a uniform commercial loading across the whole book for an extended period, there is little variation left after residualisation and your confidence intervals will be wide.

### Demand curves and pricing

Once you have an elasticity estimate, `DemandCurve` converts it into a callable price-to-probability function:

```python
from insurance_optimise.demand import DemandCurve, OptimalPrice

curve = DemandCurve(
    elasticity=est.elasticity_,
    base_price=1.0,             # price_ratio of 1.0 = at technical premium
    base_prob=0.18,             # current book average conversion rate
    functional_form="semi_log",
)

# Conversion probability at different price levels relative to technical
prices, probs = curve.evaluate(price_range=(0.9, 1.3), n_points=5)
print(probs)
# [0.217, 0.180, 0.149, 0.124, 0.102]
```

For profit-maximising price per risk:

```python
opt = OptimalPrice(
    demand_curve=curve,
    expected_loss=650.0,
    fixed_expense=45.0,
    enbp=None,  # new business: no constraint
)

result = opt.optimise()
print(f"Optimal price: £{result.optimal_price:.0f}")
print(f"Expected conversion: {result.conversion_prob:.1%}")
```

For renewal pricing, the ENBP ceiling is hard. The `OptimalPrice` solver will never return a price above it:

```python
opt = OptimalPrice(
    demand_curve=renewal_curve,
    expected_loss=650.0,
    fixed_expense=30.0,
    enbp=710.0,  # new business price via same channel
)
```

For portfolio-level optimisation with factor-structure constraints, `insurance-optimise` feeds its demand curves into `insurance-optimise`. The demand library's job is to supply the price-to-probability functions; the rate optimiser's job is to find the factor-level adjustments that satisfy LR, volume, and ENBP constraints simultaneously.

---

## FCA compliance: the ENBP checker

PS21/11 is direct on this: renewal prices must not exceed the Equivalent New Business Price for the same risk profile through the same channel. The rule has been in force since January 2022. The FCA's evaluation paper from July 2025 (EP25/2) confirmed price-walking has been substantially eliminated, and also confirmed that multi-firm reviews are ongoing - so the audit question is live.

The ENBP calculation is channel-specific. A customer who originally came via Confused.com has an ENBP calculated from your Confused.com new business price for that risk, not your direct price. If you quote differently across channels, the calculation must reflect that. Cash-equivalent incentives offered to new customers (cashback via PCW, first-month-free) must be reflected in the ENBP - the effective new business price is net of those incentives.

```python
from insurance_optimise.demand.compliance import ENBPChecker

checker = ENBPChecker(
    nb_price_col="nb_price",
    renewal_price_col="renewal_price",
    channel_col="channel",
    tolerance=0.0,  # strict: renewal must be <= NB price
)

report = checker.check(df_renewals)
print(report.n_breaches, "policies with renewal_price > ENBP")
print(report.breach_detail[["policy_id", "channel", "renewal_price",
                             "nb_price", "breach_amount"]])
```

Run this check before any renewal batch. If you have violations, fix them before renewal invitations go out. The FCA's multi-firm reviews have found implementation failures in ENBP calculation methodology - usually because new business prices were pulled from the wrong channel or because cashback incentives were not netted off correctly. The `ENBPChecker` is a systematic audit against these failure modes.

PS21/11 does not prohibit demand modelling. It constrains what the optimiser objective can do with the demand model output. For renewals, you are now optimising in the space where `renewal_price <= ENBP` - you can discount but not surcharge. This is actually a simpler optimisation problem than the pre-2022 world: it is "who do we discount and by how much?" rather than "who do we discount and who do we surcharge?". Demand modelling is still valuable because identifying which customers will lapse without a discount is exactly what survival-based retention models do.

---

## What DML cannot fix

Three limitations to be explicit about.

**Data quality.** DML residualises on observed confounders. If technical premium was recalculated retroactively after a model refit, your `log_price_ratio` is wrong and the elasticity estimate is biased in ways DML cannot recover. Store technical premium at quote time, not at analysis time.

**Unobserved confounders.** If you ran a targeted marketing campaign in Q3 2024 while also applying rate changes in the same period, and campaign exposure is not in your dataset, the campaign effect and price effect are confounded. DML handles observed confounders; it cannot handle unobserved ones. Run `est.sensitivity_analysis()` before taking elasticity results to a pricing committee - it tells you how large unobserved confounding would need to be to overturn the conclusion.

**Near-zero treatment variation.** DML identifies elasticity from the variation in `log_price_ratio` after removing the part explained by confounders. If your commercial loading decisions have been very uniform across the book, the residualised treatment has near-zero variance and your confidence intervals will be too wide to be actionable. The fix is genuine exogenous variation: rate review timing effects, manual underwriting adjustments that break from the standard tariff, or competitor market shocks if you have competitor price data as an instrument.

---

## Where to start

```bash
# Core library
uv add insurance-optimise

# With DML elasticity estimation (requires doubleml, econml):
uv add "insurance-optimise[dml]"

# With survival-based retention models (requires lifelines):
uv add "insurance-optimise[survival]"

# Everything:
uv add "insurance-optimise[dml,survival]"
```

The recommended starting point is the confounding bias check. Fit a `ConversionModel` on a year of PCW quote data, compute `model.marginal_effect()`, then run `ElasticityEstimator` on the same data and compare `est.summary()` against the naive marginal effect. The difference is the confounding bias in your current elasticity assumption.

On motor PCW datasets, the DML estimate is commonly lower in absolute magnitude than the naive estimate. The naive estimate includes risk-driven demand variation; the DML estimate strips it out. If your renewal pricing strategy was built on the naive number, you may have been discounting more than necessary.

Commercial platforms - Akur8, Earnix, Radar - implement versions of this methodology in their demand modules. The methodology is not proprietary. The `insurance-optimise` library is the same maths in an auditable Python package with no vendor lock-in, a clean sklearn-compatible API, and a data structure that your existing Polars and CatBoost workflow already understands.

Source and issue tracker on [GitHub](https://github.com/burning-cost/insurance-optimise).


---

## See also

- [DML for Insurance: Practical Benchmarks and Pitfalls](/2026/03/09/dml-insurance-benchmarks/) - benchmark results across simulated and real insurance datasets; where DML outperforms OLS and where it does not
- [Continuous Treatment Causal Inference for Insurance Pricing](/2026/03/12/insurance-autodml/) - the `insurance-causal` library that implements the DML pipeline described here, with automatic nuisance model selection
- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/15/causal-price-elasticity-tutorial/) - why the confounding problem in renewal pricing is structural, not a data quality issue
- [Your Elasticity Estimate Is Biased and You Already Know Why](/2026/03/15/causal-price-elasticity-tutorial/) - a step-by-step tutorial applying DML to a UK renewal book, with all the data engineering and validation steps
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)  -  benchmark: DML reduces elasticity bias from −4.2 to −2.1 on confounded motor data; when it works and when it does not
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)  -  benchmark: the PortfolioOptimiser achieves the same GWP target with £4-8k higher expected profit on 2,000-policy books
