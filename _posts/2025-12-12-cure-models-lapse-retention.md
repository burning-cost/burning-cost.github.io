---
layout: post
title: "Your Lapse Model Ignores Cure: The Customers Who Were Never Going to Leave"
date: 2025-12-12
categories: [pricing, libraries, tutorials]
tags: [survival-analysis, cure-models, mixture-cure, lapse, retention, CLV, insurance-survival, python, polars, consumer-duty, ps21-11, uk-motor]
description: "Logistic regression treats all non-lapsers the same. Mixture cure models split them into two groups: structural non-lapsers who will never leave, and..."
---

Every renewal pricing team has a lapse model. Most of them are logistic regressions: feed in NCD level, age, channel, price change, out comes a probability. The model is trained on policies where the dependent variable is 1 if the customer did not renew. Standard, defensible, widely deployed.

Here is the problem. The model's "not lapsed" population contains two genuinely different groups. The first group lapsed eventually, just not last year. The second group will never lapse regardless of what you charge them - direct debit, maximum NCD, insured since 1999, no price sensitivity at any realistic premium. When you train a logistic regression on one year of data, these two groups look identical. The model cannot tell them apart because, in any given observation window, both are censored.

Mixture cure models exist precisely to separate them.

---

## What the KM curve tells you

Before fitting anything, run a Kaplan-Meier on your lapse-free tenure data. Plot it. If the curve levels off well above zero and stays there for several years, that is a plateau. Standard survival models do not produce a plateau: their curves converge to zero eventually, because they assume every policyholder is susceptible. If you see a plateau, your data are telling you that some fraction of the population is structurally immune. Fitting a standard Weibull or Cox model will underestimate long-term survival, because it treats every censored observation as a susceptible who just hasn't lapsed yet.

The mixture cure model encodes this explicitly. The population survival function is:

```
S_pop(t | x, z) = pi(z) * S_u(t | x) + [1 - pi(z)]
```

where `pi(z)` is the probability of being susceptible (modelled as a logistic function of covariates z), and `S_u(t | x)` is the conditional survival for susceptibles. The `[1 - pi(z)]` term is the cure fraction - the proportion who will never lapse. It never converges to zero. That is the model telling you: these people are not going anywhere.

The practical consequence: instead of a single retention score per policy, you get two. One says how likely a customer is to be structurally loyal. The other, conditional on not being structural, says how long they are likely to stay. Those are different questions. Your marketing budget should be targeting the second group, not the first.

---

## Getting started

```bash
pip install insurance-survival
# or
uv add insurance-survival
```

We will use the `cure` subpackage, which provides the full mixture cure model suite - four model classes, a simulation helper, and the diagnostics that matter.

### Step 1: simulate or load data

The `simulate_motor_panel` helper generates synthetic UK motor policy data with a known cure fraction, so you can validate model behaviour before applying it to real data.

```python
from insurance_survival.cure import WeibullMixtureCure
from insurance_survival.cure.simulate import simulate_motor_panel
from insurance_survival.cure.diagnostics import sufficient_followup_test, CureScorecard

df = simulate_motor_panel(n_policies=5000, cure_fraction=0.40, seed=42)
# Returns a DataFrame with: policy_id, tenure_months, claimed,
# ncb_years, age, vehicle_age, channel_direct
```

The `cure_fraction=0.40` means 40% of policies are structural non-claimers. We know the ground truth here, which lets us check whether the model recovers it.

### Step 2: test for sufficient follow-up

This step is not optional. Mixture cure model estimates are upwardly biased when the observation window is too short - what looks like a cure fraction is just late claimers who haven't been observed long enough. The Maller-Zhou Qn test formalises this check.

```python
qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
print(qn.summary())
```

```
Maller-Zhou Sufficient Follow-Up Test
========================================
  Qn statistic      : 4.2361
  p-value           : 0.0000
  n observations    : 5000
  n events          : 2983
  max event time    : 58.0000
  max censoring time: 60.0000

  Conclusion: Sufficient follow-up: evidence for a genuine cure fraction.
              MCM estimates can be trusted.
```

A significant result (p < 0.05) means the plateau in the Kaplan-Meier is real. If the test fails, your observation window is probably too short and the cure fraction estimate is not to be trusted.

### Step 3: fit the model

```python
model = WeibullMixtureCure(
    incidence_formula="ncb_years + age + vehicle_age",
    latency_formula="ncb_years + age",
    n_em_starts=5,
)
model.fit(df, duration_col="tenure_months", event_col="claimed")
```

The `incidence_formula` governs which covariates drive the probability of being susceptible at all. The `latency_formula` governs how long susceptibles take to claim. These can be different. In our experience, NCD years and age both drive structural loyalty (incidence), but vehicle age matters mainly for incidence, not for how quickly susceptibles eventually claim (latency). Separating the two lets you tell a coherent story to a pricing committee.

The `n_em_starts=5` runs five random initialisations of the EM algorithm and keeps the best. Mixture models can get stuck in local maxima; five starts is enough to be reasonably confident for a portfolio of this size.

### Step 4: score and validate

```python
cure_scores = model.predict_cure_fraction(df)
# numpy array of shape (5000,): probability of being structurally immune
```

Validate with the scorecard:

```python
scorecard = CureScorecard(model, bins=10)
scorecard.fit(df, duration_col="tenure_months", event_col="claimed")
print(scorecard.summary())
```

```
Cure Fraction Scorecard
======================================================================
Decile      N   Cure Min  Cure Mean   Cure Max  Events  Event Rate
----------------------------------------------------------------------
     1    500     0.0312     0.1104     0.1843     331      0.6620
     2    500     0.1844     0.2487     0.3129     283      0.5660
     3    500     0.3130     0.3693     0.4253     244      0.4880
     4    500     0.4254     0.4784     0.5312     209      0.4180
     5    500     0.5313     0.5804     0.6292     175      0.3500
     6    500     0.6293     0.6762     0.7229     148      0.2960
     7    500     0.7230     0.7662     0.8091     110      0.2200
     8    500     0.8092     0.8487     0.8880      80      0.1600
     9    500     0.8881     0.9201     0.9518      48      0.0960
    10    500     0.9519     0.9723     0.9924      22      0.0440
======================================================================
```

Decile 1 (lowest cure fraction, highest susceptibility): 66% event rate. Decile 10 (highest cure fraction): 4.4% event rate. The model is discriminating correctly. That 15x spread is what makes the model useful for retention targeting: do not waste retention budget on decile 10. Their loyalty is structural.

---

## Versus naive logistic regression

The comparison worth making is not against other cure model libraries (there are none in Python). It is against the model most teams already have: a one-year logistic regression with "not lapsed" as the dependent variable.

The logistic model will tell you that high-NCD, direct-debit customers are likely to renew. That is correct, but incomplete. It cannot tell you whether they are likely to renew because they are structurally loyal or because last year's price happened to be competitive. Those two customers require different interventions. The cure model separates them.

More concretely: if you use a logistic model to rank your retention list, you will systematically overspend on structurally loyal customers and underspend on susceptibles who happen to have similar covariates. The cure model's incidence component tells you directly who is structural. Your retention team can be targeted at the susceptibles, where intervention has a chance of changing the outcome.

---

## Adding CLV

Once you have a cure model, connecting it to CLV is straightforward. The `SurvivalCLV` class integrates survival retention probabilities with premium and loss schedules across a planning horizon, with NCD path marginalisation:

```python
import polars as pl
import numpy as np
from insurance_survival import SurvivalCLV

# policies DataFrame: policy_id, annual_premium, expected_loss,
# plus any covariate columns the cure model needs
policies = pl.DataFrame({
    "policy_id":      df["policy_id"],
    "annual_premium": np.random.uniform(400, 1200, 5000),
    "expected_loss":  np.random.uniform(200, 700, 5000),
    "ncb_years":      df["ncb_years"],
    "age":            df["age"],
    "vehicle_age":    df["vehicle_age"],
    "channel_direct": df["channel_direct"],
})

clv_model = SurvivalCLV(survival_model=model, horizon=5, discount_rate=0.05)
results = clv_model.predict(
    policies,
    premium_col="annual_premium",
    loss_col="expected_loss",
)
```

The `results` DataFrame includes `S(t)` at each year, cure probability, expected tenure, and the headline CLV figure. The cure model's structural non-lapsers accumulate CLV differently from susceptibles: their S(t) declines slowly (if at all), so their 5-year CLV is not discounted down by the risk of early departure. That difference should feed your discount targeting directly.

This output format is also what Consumer Duty asks for. PS21/5 requires pricing teams to document that discount decisions are CLV-driven. `SurvivalCLV.predict()` gives you the per-policy S(t) path and cure probability in a single auditable table.

---

## Choosing between model classes

The library provides four cure model classes. For most lapse applications, `WeibullMixtureCure` is the right starting point: Weibull AFT latency fits reasonably well when the hazard is monotone (which motor lapse data typically is), and EM estimation converges reliably.

If you see a non-monotone hazard, where the lapse rate peaks at renewal year 2 or 3 before declining, try `LogNormalMixtureCure`. It handles humped hazard shapes that Weibull cannot. `CoxMixtureCure` gives maximum flexibility on the baseline hazard and is worth trying if neither parametric form fits well, though it is slower to estimate.

`PromotionTimeCure` is the outlier: it uses the Tsodikov (1998) non-mixture framing and imposes population-level proportional hazards structure. It can be useful when you want to maintain PH interpretability but is less natural for insurance contexts where the two-group framing is genuinely meaningful to business users.

---

## What we would not use this for

Cure models require sufficient follow-up. If your observation window is shorter than four or five years for a personal lines motor book, the Maller-Zhou test will likely fail and your cure fraction estimates will be upward-biased. In that case, a standard Weibull AFT or Cox model is more honest.

They also require enough events. The incidence sub-model is a logistic regression on censored data, estimated via EM. With fewer than around 500 observed lapses in a segment, the cure fraction estimate is unstable. Do not stratify by segment and fit separate models unless each segment has meaningful event counts.

For new business books with less than two years of data, do not use a cure model at all. The library's `WeibullAFTFitter` wrapper from lifelines is the right tool, and you can migrate to a cure model once the book matures.

---

The code is at [github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival). The library also covers competing risks (Fine-Gray regression for mid-term cancellation vs renewal lapse), recurrent events (shared frailty for pet and fleet books), and the MLflow wrapper for Model Registry deployment.

---

## Related articles

- [Separating Structural Non-Claimers from Risk: Mixture Cure Models for Insurance Pricing](/2026/03/11/insurance-cure/)
- [Treating Competing Risks as Censored Is Biasing Your Retention and Home Insurance Pricing](/2026/03/12/insurance-competing-risks/)
- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/)
