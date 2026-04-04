---
layout: post
title: "Survival Analysis for Insurance Lapse Modelling in Python: A Complete Tutorial"
date: 2026-04-04
categories: [tutorials, retention, python]
tags: [survival-analysis, lapse-modelling, python, insurance-survival, cure-models, kaplan-meier, CLV, consumer-duty, weibull, lifelines, polars, uk-motor, tutorial]
description: "A hands-on Python tutorial for insurance pricing analysts on survival analysis and lapse modelling. Covers Kaplan-Meier, Weibull AFT, mixture cure models, customer lifetime value, and the insurance-survival library."
seo_title: "Survival Analysis Insurance Python: Lapse Modelling Tutorial with Cure Models and CLV"
author: Burning Cost
---

Most lapse models in UK personal lines are logistic regressions. They are also wrong in a specific, predictable way that compounds when you try to use them for anything beyond a one-year renewal probability.

This tutorial is for pricing analysts who want to do lapse modelling properly in Python. We use the [`insurance-survival`](https://github.com/burning-cost/insurance-survival) library throughout. By the end you will have: a Kaplan-Meier baseline, a fitted mixture cure model that separates structural non-lapsers from the at-risk book, per-policy CLV estimates, and a scored lapse table in actuarial qx/px format.

We start with why the logistic model fails, work through the survival analysis properly, and finish with the cure-rate model and CLV calculation that Consumer Duty increasingly requires.

---

## Installation

```bash
uv add insurance-survival
```

Or with pip:

```bash
pip install insurance-survival
```

The library requires `polars >= 1.0`, `lifelines >= 0.27`, `numpy >= 1.24`, and `scipy >= 1.11`. Optional extras for MLflow deployment and Excel export:

```bash
pip install "insurance-survival[mlflow,excel]"
```

---

## 1. Why the logistic lapse model is wrong

The logistic model treats lapse modelling as a binary classification problem: each policy renews or it does not. The model produces P(lapse at next anniversary). That is useful. The problems emerge when you look at what it does with the data.

**Problem one: it discards mid-term censored policies.** Any policy still active at your data extract date has an unknown lapse outcome. A logistic model either drops these rows (throwing away information about long-tenure policyholders, who are systematically over-represented in the censored group) or treats them as non-lapsers (treating incomplete observations as completed ones). Both choices misspecify the model. Survival analysis exists precisely to handle censored observations correctly.

**Problem two: the "not lapsed yet" group is not homogeneous.** On a typical UK motor book, a meaningful fraction of policyholders — somewhere between 20% and 40% depending on channel mix and NCD distribution — are structural non-lapsers. Direct debit payers with eight or more years' NCD who have been with the same insurer since 2008. They are not "not lapsed yet." They are effectively never going to lapse at any realistic premium. A logistic model (and a standard Cox model) cannot represent this structure. They assume every policyholder is susceptible and will eventually lapse given enough time. Fitting them on a book with a genuine immune subgroup produces survival curves that converge asymptotically toward zero when they should plateau.

**Problem three: you cannot get CLV from it.** Consumer Duty (PS22/9, effective July 2023) requires insurers to demonstrate fair value across the customer lifecycle. The FCA's September 2024 good and poor practice review makes clear that a single-year renewal probability does not constitute that analysis. You need expected tenure and expected net contribution over a multi-year horizon. That requires a proper survival model, not a chained sequence of one-year logistic predictions.

---

## 2. The survival analysis setup

### From policy transactions to start-stop format

Raw policy data arrives as a transaction table. `ExposureTransformer` converts it to the start-stop format that survival models expect: one row per policy, with `start` (typically zero), `stop` (duration in days or years at which the policy was last observed), and `event` (1 if lapsed, 0 if censored).

```python
import numpy as np
import polars as pl
from datetime import date, timedelta
from insurance_survival import ExposureTransformer

rng = np.random.default_rng(42)
n = 2_000

# Synthetic UK motor portfolio: 2021-2024 inception dates
inception_dates = [
    date(2021, 1, 1) + timedelta(days=int(d))
    for d in rng.integers(0, 1095, n)
]
expiry_dates = [d + timedelta(days=365) for d in inception_dates]

# 30% structural non-lapsers (immune subgroup); 70% susceptible
# Of susceptibles, ~45% lapse within the observation window
immune = rng.uniform(size=n) < 0.30
susceptible_lapse = (~immune) & (rng.uniform(size=n) < 0.45)

transaction_types = []
transaction_dates = []
for i in range(n):
    if susceptible_lapse[i]:
        transaction_types.append("cancellation")
        # Lapse time: Weibull distributed, peak around 18-24 months
        lapse_days = int(np.clip(rng.weibull(1.2) * 600, 30, 1400))
        transaction_dates.append(inception_dates[i] + timedelta(days=lapse_days))
    else:
        transaction_types.append("nonrenewal")
        transaction_dates.append(expiry_dates[i])

ncd_years      = rng.integers(0, 9, n).astype(float)
channel_direct = rng.choice([0, 1], size=n).astype(float)
annual_premium = rng.uniform(300, 1_200, n)
age            = rng.integers(25, 75, n).astype(float)

transactions = pl.DataFrame({
    "policy_id":        np.arange(1, n + 1),
    "transaction_date": transaction_dates,
    "transaction_type": transaction_types,
    "inception_date":   inception_dates,
    "expiry_date":      expiry_dates,
    "ncd_years":        ncd_years,
    "channel_direct":   channel_direct,
    "annual_premium":   annual_premium,
    "age":              age,
})

transformer = ExposureTransformer(observation_cutoff=date(2025, 12, 31))
survival_df = transformer.fit_transform(transactions)

print(survival_df.shape)
print(survival_df.select(["stop", "event"]).describe())
```

The `stop` column is duration in years from inception. `event = 1` means the policy lapsed; `event = 0` means it was still active at the observation cutoff (censored). On a well-constructed retention dataset you should see around 50-70% censoring — if you see less, either your observation window is very long or your book has structural problems.

---

## 3. Kaplan-Meier: the baseline

Before fitting any parametric model, look at the Kaplan-Meier curve. It is non-parametric, makes no distributional assumption, and will tell you something important about your data: whether the survival function levels off.

```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

kmf = KaplanMeierFitter()
kmf.fit(
    survival_df["stop"].to_numpy(),
    event_observed=survival_df["event"].to_numpy(),
    label="All policies",
)

# Stratify by channel
kmf_dd = KaplanMeierFitter()
kmf_ag = KaplanMeierFitter()

direct_mask = survival_df["channel_direct"].to_numpy() == 1
kmf_dd.fit(
    survival_df["stop"].filter(pl.Series(direct_mask)).to_numpy(),
    event_observed=survival_df["event"].filter(pl.Series(direct_mask)).to_numpy(),
    label="Direct (DD)",
)
kmf_ag.fit(
    survival_df["stop"].filter(pl.Series(~direct_mask)).to_numpy(),
    event_observed=survival_df["event"].filter(pl.Series(~direct_mask)).to_numpy(),
    label="Aggregator",
)

fig, ax = plt.subplots(figsize=(9, 5))
kmf.plot_survival_function(ax=ax, ci_show=True)
kmf_dd.plot_survival_function(ax=ax, ci_show=False)
kmf_ag.plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel("Policy tenure (years)")
ax.set_ylabel("Retention probability S(t)")
ax.set_title("Kaplan-Meier: retention by acquisition channel")
ax.axhline(0.30, ls="--", color="grey", alpha=0.6, label="Expected cure fraction")
plt.tight_layout()
plt.savefig("km_by_channel.png", dpi=150)
```

What you are looking for: does the KM curve flatten out above zero and stay there? On a direct debit-heavy book you will typically see the curve plateau at 0.25-0.35 and stop declining, even with several years of follow-up. That plateau is the immune subgroup — it is visible in the raw data before you fit anything.

If your curve drives toward zero with no plateau, you either have a very price-competitive book with almost no structural loyalty, or your observation window is not long enough to detect the plateau. On aggregator-dominated books this is genuinely common: the plateau can be as low as 5-10%.

The KM curve is also the tool for checking whether you have sufficient follow-up to estimate a cure fraction reliably. The [`sufficient_followup_test`](https://github.com/burning-cost/insurance-survival) diagnostic in `insurance_survival.cure.diagnostics` formalises this check.

### What Kaplan-Meier cannot do

KM is non-parametric, which makes it trustworthy for the observed window but useless for extrapolation. At the last observed event time, KM has no information about what happens next — it cannot tell you whether the survival function continues to decline slowly or has genuinely plateaued. It also has no covariates: it cannot tell you whether a 25-year-old aggregator customer with one year's NCD has the same retention prospects as a 55-year-old direct debit customer with seven years' NCD. For those questions, you need a model.

---

## 4. Lapse modelling with a Cox proportional hazards model

Cox PH is the standard starting point for covariate-adjusted survival analysis. It estimates the hazard ratio associated with each covariate without specifying the baseline hazard — a practical advantage because you do not need to commit to a distribution.

```python
from lifelines import CoxPHFitter

cox_df = survival_df.to_pandas()

cox = CoxPHFitter()
cox.fit(
    cox_df[["stop", "event", "ncd_years", "channel_direct", "age", "annual_premium"]],
    duration_col="stop",
    event_col="event",
)
cox.print_summary()
```

Expected output pattern (on synthetic data structured as above):

- `ncd_years`: negative coefficient — more NCD experience, lower lapse hazard. Correct actuarial direction.
- `channel_direct`: negative coefficient — direct payers have lower lapse hazard than aggregator customers.
- `annual_premium`: positive coefficient — higher premium, slightly higher lapse risk.
- `age`: negative coefficient — older policyholders tend to be more tenured and more loyal.

Cox PH is the right model when you need within-sample lapse scores for the current renewal cycle. If you are building a one-year renewal probability model for use in the next repricing exercise, Cox PH is competitive with everything else and fits in a few seconds on 100,000 policies.

It is the wrong model when you need long-run extrapolation or CLV. Its survival function must eventually reach zero — it has no mechanism for representing a genuinely immune subgroup. On a book with a 30% structural cure fraction, a Cox model will underestimate five-year retention by roughly 10-15 percentage points and systematically undervalue CLV by £40-80 per policy. At portfolio scale — say 50,000 policies — that is a £2m-£4m misvaluation of the renewal book.

---

## 5. Mixture cure models: separating structural non-lapsers from the at-risk book

This is the central insight for survival analysis in insurance lapse modelling: **some policyholders will never lapse**. Not "haven't lapsed yet." Never. Standard survival models cannot represent this. Mixture cure models are built for it.

The mixture cure model assumes the population consists of two subgroups:

- **Susceptibles** (probability `1 - π`): policyholders who will eventually lapse, with some latency distribution governing when.
- **Immunes** (probability `π`): structural non-lapsers. Their observations are censored — they have not lapsed, and they never will.

The overall survival function is:

```
S_pop(t) = π + (1 - π) × S_susceptible(t)
```

As `t → ∞`, S_susceptible(t) → 0, so S_pop(t) → π. The survival curve plateaus at the cure fraction rather than reaching zero.

`WeibullMixtureCureFitter` in `insurance-survival` estimates both components simultaneously: a logistic regression on `π` (which policyholders are immune?) and a Weibull AFT model on the susceptible latency distribution (when do at-risk policyholders lapse?).

### Fitting the cure model

```python
from insurance_survival import WeibullMixtureCureFitter

fitter = WeibullMixtureCureFitter(
    cure_covariates=["ncd_years", "channel_direct"],   # logistic model on π
    uncured_covariates=["ncd_years", "age"],            # Weibull AFT on latency
)
fitter.fit(survival_df, duration_col="stop", event_col="event")

print(fitter.summary())
```

The summary table shows two sets of parameters:

1. **Incidence model** (logistic on π): the coefficients on `ncd_years` and `channel_direct`. The library parameterises the cure fraction directly as `sigmoid(intercept + x'γ)` — so a **positive** coefficient on a covariate increases P(immune). We expect both `ncd_years` and `channel_direct` to have positive coefficients: more NCD years and direct debit both increase the probability of being a structural non-lapser.

2. **Latency model** (Weibull AFT on time-to-lapse for susceptibles): the coefficients on `ncd_years` and `age`. A positive coefficient in AFT parameterisation means longer time-to-lapse (lower lapse hazard).

### Cure fraction estimates

The cure model's most operationally useful output is per-policyholder cure probability: P(immune | observed covariates). This is the parameter for retention campaign exclusion — policyholders with high cure probability should be excluded from retention spend because they are not at risk regardless of what you do.

```python
# Per-policyholder cure probability (P(immune))
cure_probs = fitter.predict_cure_fraction(survival_df)
print(f"Portfolio mean cure fraction: {cure_probs.mean():.3f}")
print(f"Top decile cure probability: {cure_probs.quantile(0.9):.3f}")
```

On the synthetic data constructed above, the model should recover the 30% cure fraction to within 1-2 percentage points. The incidence coefficients should have the correct signs: `ncd_years` positive (more NCD → more likely immune), `channel_direct` positive (direct debit → more likely immune).

### The full mixture cure suite

The `insurance_survival.cure` subpackage provides four mixture cure model variants. The choice depends on your distributional assumptions:

```python
from insurance_survival.cure import WeibullMixtureCure, LogNormalMixtureCure, CoxMixtureCure
from insurance_survival.cure.simulate import simulate_motor_panel
from insurance_survival.cure.diagnostics import sufficient_followup_test

# Simulate a larger panel for model comparison
df_cure = simulate_motor_panel(n_policies=5_000, cure_fraction=0.35, seed=42)

# Always run the sufficient follow-up test before trusting cure fraction estimates
qn = sufficient_followup_test(df_cure["tenure_months"], df_cure["claimed"])
print(qn.summary())
```

The sufficient follow-up test is not optional. If your observation window does not extend well beyond the median latency for susceptibles, the EM algorithm cannot reliably separate the immune fraction from long-tenure censored susceptibles. The test tells you whether your data support a cure fraction estimate.

```python
# WeibullMixtureCure: parametric, fast, most interpretable
model_w = WeibullMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years + age",
    n_em_starts=5,     # multiple random starts; the EM is non-convex
)
model_w.fit(df_cure, duration_col="tenure_months", event_col="claimed")

cure_scores = model_w.predict_cure_fraction(df_cure)

# LogNormalMixtureCure: better for non-monotone hazard (home, pet)
model_ln = LogNormalMixtureCure(
    incidence_formula="ncd_years + age",
    latency_formula="ncd_years",
)
model_ln.fit(df_cure, duration_col="tenure_months", event_col="claimed")

# CoxMixtureCure: semiparametric baseline, most flexible
model_cox = CoxMixtureCure(
    incidence_formula="ncd_years + age + vehicle_age",
    latency_formula="ncd_years",
)
model_cox.fit(df_cure, duration_col="tenure_months", event_col="claimed")
```

For motor personal lines, `WeibullMixtureCure` is the usual starting point. The Weibull family fits most lapse latency distributions well (Weibull shape > 1 implies increasing hazard over time — policyholders who survived multiple renewals have increasing rather than decreasing lapse risk, which is not always true but is a reasonable default). For home insurance, where lapses can occur after long dormant periods, `LogNormalMixtureCure` often fits better because the log-normal allows non-monotone hazard.

---

## 6. Competing risks: when lapse and cancellation are different events

Not all departures from a policy are the same. A policyholder who does not renew at anniversary is a lapse. A policyholder who cancels mid-term is a different event with different economics, different regulatory implications, and different risk characteristics.

If you model time-to-departure with a single event indicator, you are implicitly assuming that lapse and mid-term cancellation are competing events and you need only care about the time to the first one. That is the right framing, but the wrong statistical method. Standard survival models treat competing events as independent — in practice, the policyholders most likely to cancel mid-term are not a random draw from the book, and ignoring the competing risk produces biased estimates of both cause-specific incidence.

The correct tool is Fine-Gray subdistribution hazard regression:

```python
import pandas as pd
from insurance_survival.competing_risks import FineGrayFitter, AalenJohansenFitter

# Event codes: 0 = censored, 1 = non-renewal at anniversary, 2 = mid-term cancellation
rng2 = np.random.default_rng(99)
n_cr = 3_000
T_cr = np.clip(rng2.exponential(2.5, n_cr), 0.05, 8.0)
E_cr = rng2.choice([0, 1, 2], size=n_cr, p=[0.38, 0.37, 0.25])
ncd_cr  = rng2.integers(0, 9, n_cr).astype(float)
age_cr  = rng2.integers(25, 70, n_cr).astype(float)
prem_cr = rng2.uniform(300, 1_200, n_cr)

df_cr = pd.DataFrame({
    "T": T_cr, "E": E_cr,
    "ncd_years": ncd_cr, "age": age_cr, "annual_premium": prem_cr,
})

# Non-renewal at anniversary: event of interest
fg_renewal = FineGrayFitter()
fg_renewal.fit(df_cr, duration_col="T", event_col="E", event_of_interest=1)
print(fg_renewal.summary)

# Cumulative incidence function at 1, 2, 3, 4, 5 years
cif_renewal = fg_renewal.predict_cumulative_incidence(
    df_cr.head(100), times=[1, 2, 3, 4, 5]
)

# Non-parametric CIF via Aalen-Johansen
aj = AalenJohansenFitter()
aj.fit(df_cr, duration_col="T", event_col="E")
aj.plot()
```

The Aalen-Johansen estimator is the non-parametric analogue of Kaplan-Meier for competing risks. If you plot a standard KM curve for each cause separately, you will overestimate the cumulative incidence of each cause (because KM treats departures due to other causes as censored, which over-weights the remaining observations). Aalen-Johansen correctly accounts for the competing structure.

For full details on competing risks in insurance, see our [competing risks calibration post](/2026/03/31/competing-risks-calibration-survival-model-validation/).

---

## 7. The actuarial lapse table

Pricing actuaries work in qx/px format. `LapseTable` converts a fitted survival model into a standard actuarial table, with one row per period:

```python
from insurance_survival import LapseTable

lapse_table = LapseTable.from_survival_model(
    fitter,                    # the WeibullMixtureCureFitter from Section 5
    max_duration=10,           # 10-year horizon
    time_unit="years",
)

print(lapse_table.to_polars())
```

Output columns:

| Column | Definition |
|--------|-----------|
| `t` | Policy year |
| `S_t` | S(t): probability of surviving to start of year t |
| `q_t` | Lapse rate in year t: P(lapse in year t \| active at start of year t) |
| `p_t` | Retention rate: 1 - q_t |
| `l_t` | Lives at start of year t (per 1,000 new policyholders) |
| `T_x` | Curtate future lifetime in years |

The `q_t` column is the annual lapse rate most actuaries work with in renewal pricing models. The advantage of deriving it from a survival model rather than computing it as raw observed lapse rates is that the survival model handles censoring correctly and produces smooth, extrapolatable estimates rather than noisy year-by-year observations.

```python
# Produce separate tables by channel: direct vs aggregator
# Predict survival curve for representative direct-debit customer
direct_profile = pl.DataFrame({
    "policy_id": [1],
    "ncd_years": [6.0],
    "channel_direct": [1.0],
    "age": [45.0],
    "annual_premium": [650.0],
})

agg_profile = pl.DataFrame({
    "policy_id": [2],
    "ncd_years": [1.0],
    "channel_direct": [0.0],
    "age": [32.0],
    "annual_premium": [850.0],
})

# Survival curves for each profile
t_years = np.linspace(0, 5, 100)
s_direct = fitter.predict_survival_function(direct_profile, times=t_years)
s_agg    = fitter.predict_survival_function(agg_profile,    times=t_years)

print(f"5-year retention, direct DD customer (6 NCD): {s_direct.iloc[-1, 0]:.3f}")
print(f"5-year retention, aggregator customer (1 NCD): {s_agg.iloc[-1, 0]:.3f}")
```

The separation between these two profiles is the quantitative basis for channel-differentiated renewal pricing: not just that they have different renewal probabilities in year one, but that their entire expected tenure distributions diverge substantially by year three and beyond.

---

## 8. Customer lifetime value with SurvivalCLV

The `SurvivalCLV` class integrates the survival model with premium and loss schedules to produce per-policy CLV. This is the calculation Consumer Duty requires: not a single-year renewal probability, but expected net contribution over the customer's lifetime, discounted at an appropriate rate.

```python
from insurance_survival import SurvivalCLV

policies = pl.DataFrame({
    "policy_id":      np.arange(1, n + 1),
    "annual_premium": annual_premium,
    "expected_loss":  annual_premium * rng.uniform(0.40, 0.75, n),
    "ncd_years":      ncd_years,
    "channel_direct": channel_direct,
    "age":            age,
})

clv_model = SurvivalCLV(
    survival_model=fitter,
    horizon=7,             # 7-year CLV horizon
    discount_rate=0.05,    # 5% discount rate
)
clv_results = clv_model.predict(
    policies,
    premium_col="annual_premium",
    loss_col="expected_loss",
)

print(clv_results.select([
    "policy_id", "annual_premium", "clv", "cure_prob", "expected_tenure"
]).head(10))

print(f"\nPortfolio mean CLV:       £{clv_results['clv'].mean():.0f}")
print(f"Portfolio mean cure prob:  {clv_results['cure_prob'].mean():.3f}")
print(f"Portfolio mean tenure est: {clv_results['expected_tenure'].mean():.1f} years")
```

The `cure_prob` column is the immune probability for each policy — the most actionable output for retention campaign exclusion. Policies with `cure_prob > 0.7` should not be targeted in retention campaigns: they are not going to lapse regardless, so retention spend on them is waste.

### The CLV bias in Cox PH and Kaplan-Meier

The benchmark in the [`insurance-survival` README](https://github.com/burning-cost/insurance-survival) quantifies the CLV underestimation from using Cox PH or Kaplan-Meier on a book with a 30% cure fraction: approximately £40-80 per policy at a five-year horizon, with a £600 annual premium and a 5% discount rate. At 50,000 policies, that is a £2m-£4m misvaluation of the renewal book.

The bias is upward within the observation window (KM over-weights retained policies during the window) and downward in extrapolation beyond it (KM must eventually reach zero; the cure model plateaus). For CLV models with a horizon that extends beyond your retention data history — which is always the case for new customers — the cure model produces meaningfully better estimates.

For a worked CLV calculation in a Consumer Duty context, see [CLV, ratemaking, and technical premium under Consumer Duty](/2026/04/02/clv-ratemaking-technical-premium-consumer-duty/).

---

## 9. Model evaluation: proper scoring rules for censored outcomes

Standard accuracy metrics do not work for censored survival data. You cannot compute RMSE on a dataset where 60% of outcomes are "we don't know yet." The right tools are proper scoring rules adapted for censored data.

The `CensoredForecastEvaluator` in `insurance_survival.evaluation` implements threshold-weighted CRPS (twCRPS), based on Taggart, Loveday & Louis (arXiv:2603.14835):

```python
from scipy.stats import weibull_min
from insurance_survival.evaluation import CensoredForecastEvaluator

# Hold-out evaluation on synthetic data
n_eval = 500
tau = 5.0    # 5-year evaluation horizon

T_true = weibull_min.rvs(c=1.2, scale=3.0, size=n_eval, random_state=7)
T_obs  = np.minimum(T_true, tau)
event  = (T_true <= tau).astype(int)

# True model survival functions vs a naive exponential
true_surv  = [lambda t: weibull_min.sf(t, c=1.2, scale=3.0)] * n_eval
naive_surv = [lambda t: np.exp(-t / 3.0)] * n_eval

ev = CensoredForecastEvaluator(tau=tau, warn=False)
comparison = ev.compare({
    "Weibull(1.2, 3yr)": (true_surv,  T_obs, event),
    "Exponential":        (naive_surv, T_obs, event),
})
print(comparison)
```

Lower twCRPS is better. The Weibull model should score lower than the exponential approximation because the exponential misspecifies the shape of the hazard function (it assumes constant hazard; Weibull with shape 1.2 implies mildly increasing hazard over time).

Use this to compare cure models against standard Cox and Weibull alternatives on your own hold-out data. The proper scoring rule is the right metric for regulatory model validation documentation under Solvency II Article 120-126 and PRA CP6/24 requirements for insurance model governance.

---

## 10. MLflow deployment

Survival models deployed to production need the same MLflow lifecycle management as any other pricing model: versioning, registry, A/B test tracking.

`LifelinesMLflowWrapper` provides a pyfunc wrapper that registers any lifelines model (and by extension `WeibullMixtureCureFitter`) in the MLflow Model Registry:

```python
import mlflow
from insurance_survival import LifelinesMLflowWrapper

# Log and register the fitted cure model
with mlflow.start_run(run_name="cure_model_motor_2026q1"):
    mlflow.log_params({
        "cure_covariates": str(fitter.cure_covariates),
        "uncured_covariates": str(fitter.uncured_covariates),
        "n_policies": n,
        "observation_cutoff": "2025-12-31",
    })
    mlflow.log_metrics({
        "mean_cure_fraction": float(cure_probs.mean()),
    })

    wrapper = LifelinesMLflowWrapper(fitter)
    mlflow.pyfunc.log_model(
        artifact_path="lapse_model",
        python_model=wrapper,
        registered_model_name="motor_lapse_cure_model",
    )
```

For monitoring whether the deployed lapse model drifts post-deployment — population stability index, A/E ratio tracking, KS statistics on lapse rate predictions — see [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) and our [model monitoring practitioner guide](/2026/04/04/insurance-model-monitoring-python-practitioner-guide/).

---

## 11. When to use which model

A summary of the decision tree for survival analysis in insurance lapse modelling:

| Use case | Recommended model |
|----------|-------------------|
| One-year renewal probability only | Logistic GLM or Cox PH |
| Multi-year survival curve without covariates | Kaplan-Meier |
| Covariate-adjusted within-sample lapse scores | Cox PH |
| Long-run retention, CLV horizon > data history | `WeibullMixtureCure` |
| Retention campaign exclusion (immune scoring) | `WeibullMixtureCure` |
| Separate lapse vs mid-term cancellation | `FineGrayFitter` |
| Non-monotone hazard (home, pet) | `LogNormalMixtureCure` |
| Actuarial qx/px table output | `LapseTable` |
| Consumer Duty CLV documentation | `SurvivalCLV` |

The single most common mistake in insurance lapse modelling in Python is using a logistic regression or Cox PH on a book where the KM curve shows a visible plateau, and then deriving CLV from survival curves that drive toward zero. Run the KM first. If it levels off above zero, use a cure model.

---

## References

- Taggart, R., Loveday, N. & Louis, T. (2025). Threshold-weighted CRPS for censored survival forecasts. arXiv:2603.14835.
- Fine, J.P. & Gray, R.J. (1999). A proportional hazards model for the subdistribution of a competing risk. *Journal of the American Statistical Association*, 94(446), 496-509.
- Farewell, V.T. (1982). The use of mixture models for the analysis of survival data with long-term survivors. *Biometrics*, 38(4), 1041-1046.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer. (For the connection between frailty models and credibility — see `insurance_survival.recurrent`.)

Library: [github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival) — PyPI: [`insurance-survival`](https://pypi.org/project/insurance-survival/)

---

*Related posts:*
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/) — the case against logistic lapse models and the gaps lifelines leaves
- [Mixture Cure Models for Retention Pricing](/2026/03/13/cure-models-lapse-retention/) — deeper treatment of the immune subgroup structure and when it appears
- [Competing Risks: Calibration and Survival Model Validation](/2026/03/31/competing-risks-calibration-survival-model-validation/) — Fine-Gray CIF validation and Aalen-Johansen benchmarking
- [CLV, Ratemaking and Technical Premium under Consumer Duty](/2026/04/02/clv-ratemaking-technical-premium-consumer-duty/) — CLV calculation in a Consumer Duty fair value context
- [Credibility Theory in Python: Buhlmann-Straub Tutorial](/2026/04/04/credibility-theory-python-buhlmann-straub-tutorial/) — the Bühlmann-Straub connection: gamma frailty models and credibility theory arrive at the same result
- [Rate Change, Lapse and Causal Inference](/2026/03/10/rate-change-lapse-evaluation-causal-inference/) — disentangling price elasticity from structural loyalty using causal methods
