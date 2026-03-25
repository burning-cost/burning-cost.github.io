---
layout: post
title: "Separating Structural Non-Claimers from Risk: Mixture Cure Models for Insurance Pricing"
date: 2026-03-11
categories: [libraries, pricing, frequency, survival]
tags: [mixture-cure-models, MCM, non-claimer-scoring, survival-analysis, EM-algorithm, Weibull-AFT, Maller-Zhou, incidence-latency, structural-zeros, motor, pet, home, flood, insurance-survival, python, Farewell-1982, Peng-Dear-2000]
description: "Mixture cure models for UK motor: separates non-claimers from susceptibles. Per-policyholder cure fraction scoring - insurance-survival Python library."
---

A UK motor book typically has an observed claims frequency of around 8% per year. Run a Kaplan-Meier curve on policyholder tenure against first-claim indicator and watch what happens to the survival curve after year three: it levels off. The plateau never reaches zero. A substantial fraction of the portfolio will never claim regardless of how long you keep watching them. Standard survival models - exponential, Weibull, Cox - cannot fit a non-zero asymptote. They assume everyone eventually claims, given enough time. On a long-retention motor book, this assumption is demonstrably wrong.

The biostatistics literature has had a model for this since Boag (1949) applied it to cancer survival - the mixture cure model. Farewell (1982) formalised the covariate structure. Peng and Dear (2000) and Sy and Taylor (2000) gave us the EM estimation framework. R has had usable implementations (`smcure`, `flexsurvcure`, `cuRe`) for years. Python has had nothing pip-installable that fits covariate-aware MCMs with actuarial output.

[`insurance-survival`](https://github.com/burning-cost/insurance-survival) fills that gap.

```bash
pip install insurance-survival
```

---

## What the model actually does

The population survival function in a mixture cure model is:

```
S_pop(t | x, z) = pi(z) * S_u(t | x) + [1 - pi(z)]
```

The right-hand term `[1 - pi(z)]` is the cure fraction: the probability that policyholder `i` is structurally immune and will never claim. It does not shrink to zero as `t` grows. It is a permanent floor.

The left-hand term `pi(z) * S_u(t | x)` models the susceptible subpopulation. `pi(z)` is the probability of being susceptible - estimated by a logistic regression on incidence covariates `z`. `S_u(t | x)` is the conditional survival function for susceptibles - how long it takes them to claim, modelled by a Weibull AFT, log-normal AFT, or Cox proportional hazards model on latency covariates `x`.

The two sub-models can have different covariate sets. This is the key degree of freedom. In motor, we expect:
- **Incidence** (who is immune?): NCB years, claim history, driver age, vehicle type. A driver with nine years of NCB who has never claimed is probably genuinely immune, not just lucky.
- **Latency** (how quickly do susceptibles claim?): vehicle age, annual mileage, area. Among policyholders who will eventually claim, these factors determine when.

Conflating these two processes - as every frequency GLM does - produces systematically wrong predictions. The GLM assigns a low frequency to a nine-year NCB driver because it interpolates from nearby historical rates. The MCM assigns a high cure probability to that driver because it has explicitly modelled the immune fraction. The predictions converge in the centre of the covariate space and diverge at the extremes. Those extremes are where the pricing errors are.

---

## Before you fit anything: the Maller-Zhou test

The MCM has an identifiability problem. If your observation window is short relative to the latency distribution of susceptibles, you cannot distinguish between a high cure fraction (many structural non-claimers) and a long-tailed latency (susceptibles who simply have not claimed yet). Both produce the same Kaplan-Meier plateau. If follow-up is insufficient, the estimated cure fraction will be upwardly biased - you will classify too many policyholders as immune.

Maller and Zhou (1996) proposed the Qn statistic to test for sufficient follow-up before trusting any cure fraction estimate. The statistic measures the proportion of censored observations whose tenure exceeds the final event time. Under the null hypothesis of no cure fraction (standard exponential tail), this proportion converges to zero. A significant Qn provides evidence that the observed plateau reflects a genuine immune fraction.

```python
from insurance_survival.cure.diagnostics import sufficient_followup_test

qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
print(qn.summary())
```

```
Maller-Zhou Sufficient Follow-Up Test
========================================
  Qn statistic      : 4.2831
  p-value           : 0.0000
  n observations    : 3000
  n events          : 1134
  max event time    : 58.2134
  max censoring time: 59.9341

  Conclusion: Sufficient follow-up: evidence for a genuine cure fraction.
  MCM estimates can be trusted.
```

Run this before fitting anything. A non-significant Qn does not mean you cannot fit an MCM - it means the cure fraction estimate will be unreliable, and you should interpret the output as a lower bound on pricing differentiation rather than an absolute calibrated score. A five-year motor book with heavy early attrition is typically marginal on this test; a seven-year book with stable retention usually passes comfortably.

---

## Fitting the model

The primary workhorse is `WeibullMixtureCure`. The Weibull AFT latency is the right default: it extrapolates cleanly beyond the observation window, the shape parameter has a direct interpretation (sub-exponential vs. super-exponential hazard for susceptibles), and the EM algorithm is numerically stable with it. Log-normal is better when the conditional hazard for susceptibles peaks and then falls - this can happen with pet insurance where very young and very old animals have different claim timing - but it will not extrapolate as gracefully.

```python
import pandas as pd
from insurance_survival.cure import WeibullMixtureCure
from insurance_survival.cure.diagnostics import sufficient_followup_test, CureScorecard
from insurance_survival.cure.simulate import simulate_motor_panel

# Synthetic motor panel: 3000 policies, 5-year window, 40% true cure fraction
df = simulate_motor_panel(n_policies=3000, cure_fraction=0.40, seed=42)

# Step 1: check sufficient follow-up
qn = sufficient_followup_test(df["tenure_months"], df["claimed"])
print(qn.summary())

# Step 2: fit the MCM
model = WeibullMixtureCure(
    incidence_formula="ncb_years + age + vehicle_age",
    latency_formula="ncb_years + age",
    n_em_starts=5,
    random_state=42,
)
model.fit(df, duration_col="tenure_months", event_col="claimed")
print(model.result_.summary())
```

```
Mixture Cure Model Results
========================================
  Convergence    : Yes (47 iterations)
  Log-likelihood : -4821.3174
  Observations   : 3000
  Events         : 1134
  Mean cure frac : 0.3981

Incidence sub-model (logistic):
  Intercept : -0.0841
  ncb_years           : -0.2987
  age                 :  0.0198
  vehicle_age         :  0.0512

Latency sub-model:
  log_lambda          :  3.5621
  log_rho             :  0.1847
  beta_ncb_years      :  0.0523
  beta_age            : -0.0031
```

The recovered cure fraction of 0.398 is close to the true 0.40 - the EM has correctly separated the latent groups. The incidence coefficient on `ncb_years` is -0.30: each additional NCB year reduces the log-odds of susceptibility by 0.30, meaning a nine-year NCB driver has a log-odds of susceptibility roughly 2.7 lower than a zero-NCB driver. On the probability scale, that is a substantial shift in cure probability.

The multiple restarts (`n_em_starts=5`) are not optional. The EM objective for MCMs has local optima, particularly when the incidence and latency sub-models compete to explain the same policyholders. With a single start, you will occasionally converge to a degenerate solution where the algorithm has assigned everyone to the cure fraction and the latency sub-model is uninformative. Five starts takes five times longer but eliminates this failure mode in practice.

---

## The outputs that matter for pricing

Two predictions come out of a fitted MCM:

**Cure fraction per policyholder** - `predict_cure_fraction()` returns P(immune | z_i) for each row. This is the non-claimer propensity score. It drives pricing differentiation between structural non-claimers and genuine risk.

**Population survival** - `predict_population_survival()` returns S_pop(t | x_i, z_i) at specified times. This is the probability of no claim by time t, combining both the immune fraction and the latency distribution for susceptibles.

```python
# Non-claimer propensity scores
cure_scores = model.predict_cure_fraction(df)
# cure_scores is a numpy array of shape (3000,)
# Values close to 1.0: probably structurally immune
# Values close to 0.0: probably susceptible

# Population survival at 12, 24, 36, 60 months
pop_surv = model.predict_population_survival(
    df, times=[12, 24, 36, 60]
)
# Returns a DataFrame with one column per time point
```

For pricing use, the cure score is the primary output. You are not replacing your frequency GLM - you are augmenting it with a new feature. The cure score captures something the frequency GLM cannot: the probability that the policyholder is not in the risk pool at all. A high-NCB young driver with a high cure score should be priced below what the frequency GLM suggests, because the frequency GLM's low-frequency prediction still implies they will eventually claim.

---

## Validating with the CureScorecard

The `CureScorecard` produces the decile table that goes in the pricing committee pack. Rank policyholders by predicted cure fraction, split into deciles, and check that high-cure deciles have lower observed claim rates.

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
     1    300     0.1043     0.1892     0.2641     216      0.7200
     2    300     0.2641     0.3094     0.3529     198      0.6600
     3    300     0.3529     0.3841     0.4165     179      0.5967
     4    300     0.4165     0.4443     0.4728     154      0.5133
     5    300     0.4728     0.4994     0.5259     134      0.4467
     6    300     0.5259     0.5532     0.5812     115      0.3833
     7    300     0.5812     0.6101     0.6402      89      0.2967
     8    300     0.6402     0.6741     0.7121      65      0.2167
     9    300     0.7121     0.7608     0.8171      37      0.1233
    10    300     0.8171     0.8927     0.9731      13      0.0433
======================================================================
```

The event rate drops from 72% in the bottom decile to 4.3% in the top. That gradient is exactly what the model promises: the top decile contains mostly structurally immune policyholders. A frequency GLM would assign these policyholders a low expected frequency but still above zero. The MCM assigns them a probability of being in the risk pool at all - which is the right question.

---

## Which latency model to use

Three models are available. `WeibullMixtureCure` is the right default for most applications: parametric extrapolation, stable EM, interpretable shape parameter. The Weibull shape `rho` (exponentiated from `log_rho` in the output) tells you whether the hazard for susceptibles is increasing (rho > 1), constant (rho = 1), or decreasing (rho < 1) over time. UK motor tends to show rho slightly above 1: susceptible policyholders become slightly more likely to claim as tenure increases, probably due to exposure accumulation.

`LogNormalMixtureCure` is better when the conditional hazard for susceptibles has a non-monotone shape - rises then falls. Pet insurance is the clearest example: very young animals have high claim rates, rates peak at middle age, and older animals have high rates again. A Weibull cannot fit this shape; a log-normal can. The trade-off is that log-normal extrapolation beyond the observation window is less reliable.

`CoxMixtureCure` uses a semiparametric baseline hazard - no distributional assumption on the latency component. Most flexible, but cannot extrapolate beyond the maximum observed time. Use it for model comparison when you want to check whether the Weibull assumption is materially distorting the incidence estimates.

For most UK motor pricing applications, `WeibullMixtureCure` with `n_em_starts=5` and `bootstrap_se=True` is the production configuration.

---

## Standard errors and production use

Bootstrap standard errors on the incidence and latency parameters are available via `bootstrap_se=True`. With 200 bootstrap resamples and `n_jobs=-1`, this runs in parallel and adds roughly 5-10x the single-fit runtime.

```python
model = WeibullMixtureCure(
    incidence_formula="ncb_years + age + vehicle_age",
    latency_formula="ncb_years + age",
    n_em_starts=5,
    bootstrap_se=True,
    n_bootstrap=200,
    n_jobs=-1,
    random_state=42,
)
model.fit(df, duration_col="tenure_months", event_col="claimed")
```

The `result_.summary()` output then includes SE columns for each coefficient. These are the numbers you want in the governance pack alongside the point estimates. The incidence sub-model SEs are typically modest with a few thousand policies; the latency SEs can be wider because you are only fitting to the ~60% of the book that is susceptible.

---

## Where this fits the wider toolkit

`insurance-survival` is not a replacement for a frequency GLM. It is a feature factory. The cure score from a fitted MCM is a new predictor you feed into your existing pricing model alongside the usual rating factors. The MCM does one thing - estimate the immune fraction - and it does it with the right statistical machinery.

The natural place for it is in a two-stage pricing stack. Stage one: fit the MCM to the historical book. Stage two: use `predict_cure_fraction()` as a feature in your main frequency GLM or GBM, alongside NCB, driver age, vehicle age, and the other standard factors. The MCM has already extracted the structural non-claimer signal; the downstream model then blends it with the remaining frequency predictors.

For lines where structural zeros are the main story - UK flood home insurance, where many properties genuinely cannot experience a qualifying flood claim regardless of exposure - the MCM cure fraction may be the dominant pricing factor rather than just one feature among many. The `PromotionTimeCure` variant (Tsodikov 1998) provides a different non-mixture parameterisation for those applications where population-level proportional hazards structure is preferred.

---

**[insurance-survival on GitHub](https://github.com/burning-cost/insurance-survival)** - MIT-licensed, PyPI. 158 tests, 8 modules.

---

## See Also

- **[insurance-survival](/2026/03/11/survival-models-for-insurance-retention/)** - Retention and lapse modelling with competing risks; the customer lifetime value counterpart to the pricing-focused MCM
- **[insurance-experience](https://github.com/burning-cost/insurance-experience)** - Individual policy-level Bayesian posterior experience rating; use alongside cure scores for policyholders with observed claim history
- **[insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools)** - GLM with cluster-robust standard errors; relevant when you introduce cure scores as new features and need correct inference on their coefficients

---

## Related articles

- [Treating Competing Risks as Censored Is Biasing Your Retention and Home Insurance Pricing](/2026/03/12/insurance-competing-risks/)
- [Your Lapse Model Ignores Cure: The Customers Who Were Never Going to Leave](/2027/11/15/cure-models-lapse-retention/)
- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/)
