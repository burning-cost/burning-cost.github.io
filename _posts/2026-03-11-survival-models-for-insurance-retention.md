---
layout: post
title: "Survival Models for Insurance Retention"
date: 2026-03-11
categories: [techniques, libraries]
tags: [survival, retention, CLV, lifelines, cure-models, Consumer-Duty, GIPP, python, motor, home]
description: "Survival models for UK personal lines retention: cure models, survival-adjusted CLV, actuarial lapse tables, MLflow deployment. What lifelines does not do."
---

The standard retention model for UK personal lines is a logistic GLM. Target variable: did the customer renew at their next anniversary, 1 or 0. Features: NCD level, tenure, premium change, payment method. Fit it, validate on a holdout set, read off renewal probabilities. Done.

It works, after a fashion. The problems emerge when you try to use it for anything beyond the next renewal.

The logistic model gives you P(renewal at anniversary t+1). It does not give you P(still active at year t+2 | renewed at t+1), or at t+3, or t+5. If you want a customer lifetime value figure (which post-PS21/11 you increasingly need), you have to chain together multiple logistic models, each conditioned on survival to that point. That chain breaks down quickly because it ignores censoring: the mid-term policies still active at your observation cutoff are not discards, they are incomplete observations that carry information about the tenure distribution. A logistic model that drops them, or treats them as non-renewers, is misspecified.

There is a second problem specific to insurance that clinical survival software does not handle: a meaningful fraction of your policyholders effectively never lapse. The high-NCD customer paying by direct debit who has been with the same insurer since 2011 is not meaningfully "at risk" in the same sense as a PCW switcher on their first renewal. If you fit a standard Weibull or Cox model on the full book, the survival curve will converge asymptotically towards zero. It should plateau instead. The plateau is the never-lapse subgroup.

Both problems now have regulatory weight. Consumer Duty (PS22/9, effective July 2023) requires insurers to demonstrate fair value across the customer lifecycle, not just at point of sale. The FCA's September 2024 Good and Poor Practice guidance makes explicit that firms are expected to use data analytics to understand how long customers benefit from coverage. A single-year renewal probability is not that analysis.

We built [`insurance-survival`](https://github.com/burning-cost/insurance-survival) to handle both problems properly.

---

## What lifelines does not do for insurance

[lifelines](https://lifelines.readthedocs.io/) is an excellent general-purpose survival library. We use it. `insurance-survival` calls it for the standard models. The gaps are specific to insurance:

**Covariate-adjusted cure models.** `lifelines.MixtureCureFitter` exists, but it is univariate: it fits a single cure fraction for the whole population. Insurance data needs a cure fraction that varies by covariate: a high-NCD direct debit customer has a different never-lapse probability than a PCW shopper on NCD 0. The cure logistic model captures this.

**Customer lifetime value.** No Python library integrates survival probabilities with premium and loss schedules to produce per-policy CLV. This is the calculation Consumer Duty requires: can you document that a loyalty discount is CLV-justified? The output needs to be audit-ready, not just a number.

**Actuarial output format.** Pricing actuaries expect qx/px/lx tables. Survival software produces survival curves. Bridging them is mechanical but sufficiently tedious that people avoid it, which is why lapse tables in practice are often computed outside the model.

**MLflow deployment.** lifelines has no native MLflow flavour. You cannot register a `WeibullAFTFitter` in the Model Registry without writing a custom pyfunc wrapper. If your data science platform uses MLflow - increasingly common in UK insurance - this matters.

---

## The data problem first

Before fitting anything, you need your policy data in survival format: one row per observation interval, with `start`, `stop`, and `event` columns. Insurance transaction data is not in this format, and the transformation is not trivial.

Three complications that bite you if you ignore them:

**MTAs.** A mid-term adjustment updates a policy's covariates without constituting a lapse. NCD level, vehicle change, address change: all of these produce transactions in your policy admin system. The survival model needs to split the observation interval at each MTA and carry the updated covariates forward. It must not treat the MTA as an event.

**Fractional first-year exposure.** A policy incepted on 1 July contributes 0.5 years to the first observation period. If you assign it a full year's exposure it will look artificially long-tenured in the model.

**Left truncation.** Policies already in force at the start of your study window are conditionally observed: you only see them because they had not yet lapsed. Dropping them loses the long-tenure policies entirely, which biases the survival curve downwards. Including them without accounting for the truncation also biases the model.

`ExposureTransformer` handles all of this:

```python
import polars as pl
from datetime import date
from insurance_survival import ExposureTransformer

# transactions: policy_id, transaction_date, transaction_type,
#               inception_date, expiry_date, ncd_level, annual_premium, channel
transformer = ExposureTransformer(
    observation_cutoff=date(2025, 12, 31),
    time_scale="policy_year",    # renewal cliff at integers
    exposure_basis="earned",
)

survival_df = transformer.fit_transform(transactions)

print(transformer.summary())
# {'n_policies': 50000, 'event_rate': 0.31, 'median_duration': 2.4,
#  'censoring_rate': 0.69, 'left_truncated_count': 8312, 'mta_count': 14201}
```

The output is start/stop format, directly compatible with `lifelines.CoxTimeVaryingFitter` and all `insurance-survival` models. The `event_rate` of 0.31 means 31% of policies were observed to lapse within the study window. The `censoring_rate` of 0.69 (the majority) would be silently discarded by a logistic model. They are the mid-term policies still active at the observation cutoff, and they carry real information about tenure.

---

## The cure model

Fit a `WeibullMixtureCureFitter` and the survival function becomes:

```
S(t|x) = π(x) + (1 − π(x)) × S_u(t|x)
```

`π(x)` is the cure fraction, the never-lapse probability, modelled as a logistic function of covariates. `S_u(t|x)` is Weibull AFT survival for the subgroup who are genuinely at risk. The two-component structure means the survival curve can plateau correctly for customers who are likely to be permanent.

The R equivalent is `flexsurvcure::flexsurvcure(mixture=TRUE, dist="weibull")`. In Python, nothing comparable existed with covariate adjustment until now.

Parameters are estimated by EM initialisation followed by joint L-BFGS-B optimisation. EM handles the fundamental identification problem: censored observations are ambiguous, as they might be cured (never-lapsers) or simply not-yet-lapsed. The EM algorithm treats the cure indicator for censored observations as a latent variable and iterates to convergence, giving L-BFGS-B a sensible starting point for the joint maximisation.

```python
from insurance_survival import WeibullMixtureCureFitter

fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_level", "channel_direct", "payment_dd"],
    uncured_covariates=["ncd_level", "annual_premium_scaled"],
    penalizer=0.01,
)

fitter.fit(survival_df, duration_col="stop", event_col="event")

print(fitter.summary())
```

The `summary()` output separates the cure logistic coefficients from the Weibull AFT coefficients, with standard errors and confidence intervals on both. The NCD coefficient in the cure model tells you how much each additional NCD step changes the log-odds of being a never-lapper. The NCD coefficient in the Weibull model tells you how the time-to-lapse distribution shifts for the at-risk subgroup.

To predict cure probability for a new cohort:

```python
cure_probs = fitter.predict_cure(new_policies)
# Series of π(x) per policy, range [0, 1]
```

The high-NCD, direct-debit, long-tenure segment will cluster near 0.6–0.7 in a typical motor book. The PCW-sourced, NCD-0, card-payment segment will cluster near 0.15–0.20. That spread is the distribution of commercial risk in your retention book, and it has always been there - it just was not visible in a single-year logistic model.

---

## Survival-adjusted CLV

Once you have a fitted survival model, `SurvivalCLV` integrates it with premium and loss schedules over a planning horizon:

```
CLV(x) = Σ_{t=1}^{T} S(t|x(t)) × (P_t − C_t) × (1+r)^{−t}
```

The `x(t)` notation matters. NCD level is not static; it advances year by year via the UK motor NCD Markov chain (claim: two steps back; no claim: one step up). A customer on NCD 3 today will have a different NCD distribution in year 3 depending on how many claims they make, and that changes both their lapse probability and their premium. `SurvivalCLV` handles this via exact Markov chain marginalisation - no simulation required for this discrete, finite state space.

```python
from insurance_survival import SurvivalCLV

clv_model = SurvivalCLV(
    survival_model=fitter,
    horizon=5,
    discount_rate=0.05,
)

results = clv_model.predict(
    policies,
    premium_col="annual_premium",
    loss_col="expected_loss",
)
# Columns: policy_id, clv, survival_integral, cure_prob, s_yr1 .. s_yr5
```

The `survival_integral` is expected tenure, the sum of annual survival probabilities. A customer with `survival_integral = 3.2` is expected to stay for 3.2 years on average, which belongs in a fair value assessment alongside the headline CLV figure.

For discount targeting:

```python
sensitivity = clv_model.discount_sensitivity(
    policies,
    discount_amounts=[25.0, 50.0, 75.0, 100.0],
)
# Columns: policy_id, discount_amount, clv_with_discount,
#          clv_without_discount, incremental_clv, discount_justified
```

The `discount_justified` column is a boolean: is the CLV with the discount greater than or equal to the CLV without? For a loyalty discount of £50, a customer on the margin of lapsing will flip this to True. A customer who is very likely to stay regardless will have `discount_justified=False` for any discount: you are giving money away for nothing.

This is the Consumer Duty output. Consumer Duty requires that loyalty discounts are demonstrably fair-value-positive for the customers who receive them. An explicit `discount_justified` column with the calculation behind it is what a fair value file should contain. It is not a compliance document by itself, but it is the analysis that a compliance document should be built on.

---

## Actuarial lapse tables

CLV output is for pricing teams. Actuarial lapse tables are for capital modellers, reserving actuaries, and everyone who needs to put a qx/px/lx structure in front of a peer review committee. `LapseTable` bridges the two:

```python
from insurance_survival import LapseTable

table = LapseTable(survival_model=fitter, radix=10_000)

df = table.generate(
    covariate_profile={"ncd_level": 3, "channel_direct": 1, "payment_dd": 1}
)
print(df)
```

```
shape: (7, 6)
┌──────┬───────┬──────┬───────┬───────┬───────┐
│ year │ lx    │ dx   │ qx    │ px    │ Tx    │
╞══════╪═══════╪══════╪═══════╪═══════╪═══════╡
│ 1    │ 10000 │ 2300 │ 0.230 │ 0.770 │ 2.84  │
│ 2    │  7700 │ 1540 │ 0.200 │ 0.800 │ 2.41  │
│ 3    │  6160 │ 1109 │ 0.180 │ 0.820 │ 2.11  │
│ 4    │  5051 │ 858  │ 0.170 │ 0.830 │ 1.87  │
│ 5    │  4193 │ 671  │ 0.160 │ 0.840 │ 1.67  │
│ 6    │  3522 │ 528  │ 0.150 │ 0.850 │ 1.49  │
│ 7    │  2994 │ 0    │ 0.000 │ 1.000 │ 1.00  │
└──────┴───────┴──────┴───────┴───────┴───────┘
```

The `Tx` column is curtate expected future lifetime from year x, the number an actuary uses to calculate expected tenure conditional on having survived to that point. It decreases as tenure increases, as you would expect: a customer who has survived to year 5 has a shorter expected remaining tenure than one who just incepted, simply because the cure model's plateau means the remaining population is increasingly concentrated in the never-lapse subgroup.

You can generate tables for multiple segments and compare:

```python
profiles = pl.DataFrame([
    {"ncd_level": 0, "channel_direct": 0, "payment_dd": 0, "segment": "PCW NCD0"},
    {"ncd_level": 5, "channel_direct": 1, "payment_dd": 1, "segment": "Direct NCD5 DD"},
])

comparison = table.generate(profiles, by="segment")
```

The qx at year 1 for the PCW NCD0 segment will typically be 35–40% for a competitive motor book. For the Direct NCD5 DD segment it will be 10–15%. That ratio is the core of your retention commercial risk - and it is directly quantified, by segment, in a format an actuary can sign off.

---

## Databricks and MLflow deployment

Databricks is increasingly adopted by UK insurance data science teams, and lifelines is not natively registered in MLflow's model flavours, which creates friction when you want to promote a fitted survival model from an experiment to the Model Registry and serve it in a pipeline.

`LifelinesMLflowWrapper` resolves this with a pyfunc wrapper:

```python
import mlflow
from insurance_survival import LifelinesMLflowWrapper

# After fitting your model
with mlflow.start_run():
    wrapper = LifelinesMLflowWrapper(fitter)
    mlflow.pyfunc.log_model(
        artifact_path="survival_model",
        python_model=wrapper,
        registered_model_name="motor_retention_v3",
    )
```

Once registered, the model can be loaded with `mlflow.pyfunc.load_model()` and called from a batch scoring notebook or a Delta Live Tables pipeline. The prediction method returns the same output as `predict_survival_function()` - a DataFrame of survival probabilities at requested time points - so existing downstream code does not need to change.

On Databricks specifically, the recommended pattern is to run `ExposureTransformer` as a Delta Live Tables pipeline that consumes policy transaction events from a Bronze Delta table, assembles the start/stop survival format incrementally, and writes to a Silver table. The cure model and CLV calculations sit in a Gold layer notebook triggered on each rate change cycle. The MLflow wrapper means the fitted model is versioned, staged (Staging → Production), and auditable. When the FCA asks which model version produced a particular CLV estimate, you have a complete lineage.

---

## Connecting to rate optimisation

The CLV output from `SurvivalCLV` is the natural input to CLV-based discount targeting in [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise). The pattern is:

1. Fit the cure model on the retention book.
2. Run `SurvivalCLV.discount_sensitivity()` with a range of discount amounts.
3. Pass the `discount_justified` segmentation to `insurance-optimise` as the set of policies eligible for a loyalty discount.
4. Let `insurance-optimise` find the factor adjustments that hit the LR target while maximising the discount budget allocated to CLV-positive policies.

Without the CLV analysis, a rate optimiser will spread discounts across the renewal book as though all retained customers have equal value. They do not. The PCW-sourced NCD-0 customer retained by a £50 discount has a low cure probability and a short expected tenure. The direct NCD-5 customer who receives the same discount has a much higher expected lifetime value.

---

## Getting started

```bash
uv add insurance-survival

# With optional extras:
uv add "insurance-survival[mlflow,plot,excel]"
```

Source and tests on [GitHub](https://github.com/burning-cost/insurance-survival). 106 tests, all passing.

The minimum viable starting point: run `ExposureTransformer` on your policy transaction data and look at `transformer.summary()`. The `event_rate` and `censoring_rate` tell you immediately whether you have a censoring problem that a logistic model is hiding. If `censoring_rate > 0.5` (and on most live books it will be), the logistic model is making implicit assumptions about those censored observations that it has no right to make.

After that: fit `WeibullMixtureCureFitter` with NCD and channel as cure covariates and call `predict_cure()` on a slice of your current renewal book. The distribution of cure probabilities will tell you where your retention commercial risk actually sits, at a per-policy level, for the first time.

The R survival community has had covariate-adjusted cure models in `flexsurvcure` for years. Python did not. Now it does.

- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/)
- [Your Lapse Model Ignores Cure](/2026/03/11/insurance-cure/) - the subpopulation of customers who were never going to leave: why mixing them into a standard Kaplan-Meier estimate biases your projected retention rate upward
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) - the downstream model that consumes survival-derived CLV scores to allocate the retention discount budget
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) - once the survival model is in production, the Murphy decomposition tells you when its calibration has drifted enough to warrant intervention
