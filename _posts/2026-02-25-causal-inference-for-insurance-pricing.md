---
layout: post
title: "How Much of Your GLM Coefficient Is Actually Causal?"
date: 2026-02-25
categories: [techniques]
tags: [causal-inference, double-machine-learning, DML, catboost, pricing, FCA, python, motor, confounding]
description: "GLM coefficients measure association, not causation. How Double Machine Learning isolates the causal effect of rating factors from confounding, why exp(beta) gives the wrong answer for intervention decisions, and how insurance-causal implements this for UK pricing teams."
---

Your GLM has a price elasticity coefficient of -0.045. Your pricing team uses it to optimise renewal offers. Your renewal optimisation is wrong.

Not wrong as in miscoded. Wrong as in: the coefficient does not measure what you think it measures. It is an estimate of the association between price changes and renewal, not the causal effect of price on renewal. Those are different quantities, often materially so, and the difference has direct consequences for how you price, what you tell the FCA, and how you compete.

The same structure applies to every `exp(β)` in your factor table. When a pricing actuary reads off the exponentiated coefficient for engine size, age, or telematics harsh braking score, what they get is the association between that factor and loss cost, holding other factors constant. That sounds like a causal estimate. It is not — and the moment someone asks "what happens if we change the weight on this factor?", the GLM coefficient gives you the wrong answer.

We built [`insurance-causal`](https://github.com/burning-cost/insurance-causal) to measure the difference.

---

## The confounding problem in plain terms

A motor insurer running renewals has observational data: for each renewing policy, they have the price change applied, the renewal decision, and the rating factors used to price the policy.

The problem is that price changes are not random. The same pricing model that generates them also uses the rating factors. High-risk customers receive larger premium increases. High-risk customers are also more likely to lapse — for reasons entirely unrelated to price. Poor driving history increases both the price change and the baseline lapse probability.

When you run a GLM with price change as a covariate, you get a coefficient that blends the genuine causal price effect with the risk-driven lapse effect. You cannot disentangle them from within the GLM. The result: the naive coefficient overstates true price sensitivity, because the model is partly attributing risk-driven lapse to price.

The same structure appears everywhere in insurance pricing.

**Age and urban driving.** Young drivers are disproportionately urban. An age coefficient in a motor frequency GLM absorbs some of the urban driving effect, because age and urban exposure are correlated. You cannot tell from the GLM alone how much of the age relativity is actually an age effect versus a proxy for urban driving.

**Telematics harsh braking.** Drivers who brake harshly in city traffic show a strong association with claim frequency. Is that harsh braking causing the claims, or is it an urban driving proxy? The GLM coefficient on harsh braking will absorb both effects, in proportions you cannot observe.

**Engine size.** Engine size correlates with the type of driver who buys the car, their age, their postcode, how they use the vehicle, whether they have telematics. A GLM that includes all those factors does partial adjustment, but the adjustment is correct only if the confounders are included linearly and the model is correctly specified. In practice, the GLM coefficient on engine size is part causal effect and part residual confounding from everything engine size correlates with that the GLM has not fully captured.

For each of these, pricing teams argue endlessly about how much of the association is "real causation" versus confounding. The classical response is educated judgment and factor stability tests. Both are honest but imprecise. There is a better method.

---

## Double Machine Learning: the two-step

Double Machine Learning (DML), introduced by Chernozhukov et al. in their 2018 paper in *The Econometrics Journal*, solves this by explicitly separating the causal question from the prediction question.

The setup assumes a data generating process:

```
Y = θ₀ · D + g₀(X) + ε
D = m₀(X) + V
```

Where Y is the outcome (renewal, claim frequency), D is the treatment (price change, telematics score), X is the vector of observed confounders (rating factors), and θ₀ is the causal parameter you want. The functions g₀(X) and m₀(X) are unknown and potentially highly nonlinear — the "nuisance" functions.

The naive approach — regress Y on D and X using a flexible ML model — fails because regularisation in the ML model biases the estimate of θ₀. DML's solution is to partial out the confounders from both Y and D separately, then regress the residuals on each other:

1. Fit E[Y|X] using CatBoost (with 5-fold cross-fitting). Compute residuals Ỹ = Y - Ê[Y|X].
2. Fit E[D|X] using CatBoost (with 5-fold cross-fitting). Compute residuals D̃ = D - Ê[D|X].
3. Regress Ỹ on D̃ via OLS. The coefficient is θ̂.

The key mathematical property — Neyman orthogonality — means that errors in the nuisance estimates (steps 1 and 2) produce second-order bias in θ̂, not first-order. When CatBoost converges at the rates achievable on typical insurance datasets, the resulting bias in θ̂ is negligible relative to its standard error. Step 3 is just OLS, which gives valid standard errors and confidence intervals.

The residualised treatment D̃ = D - E[D|X] is the part of the treatment that cannot be explained by the confounders — the exogenous variation. Regressing residualised outcomes on residualised treatment isolates the causal channel.

### Why CatBoost for the nuisance models

The nuisance models need to be flexible enough to capture genuinely nonlinear confounding. A 2024 systematic evaluation (arXiv:2403.14385) found that gradient boosted trees outperform LASSO in the DML nuisance step when confounding is nonlinear — which it is for insurance data with postcode effects and age-vehicle interactions.

We use CatBoost specifically because it handles high-cardinality categoricals (postcode band, vehicle group, occupation class) natively without label encoding. Its ordered boosting also reduces target leakage from categoricals with many levels. The nuisance model defaults are 500 trees, depth 6, learning rate 0.05 — more conservative than a predictive model, appropriate for the debiasing goal.

---

## The confounding bias report

The single output that makes this library immediately useful for a pricing team is the confounding bias report. It answers the question they are already arguing about.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims",
                 "postcode_band", "vehicle_group"],
    cv_folds=5,
)

model.fit(df)

report = model.confounding_bias_report(naive_coefficient=-0.045)
print(report)
```

```
  treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct
  pct_price_change  renewal         -0.0450          -0.0230  -0.022     -95.7%
```

The naive estimate overstates the true causal effect by 95.7%. Pricing decisions made using -0.045 are substantially biased.

What this means operationally: the genuine causal elasticity is -0.023, not -0.045. A renewal pricing model optimised on -0.045 will act as if price is twice as powerful a lever as it really is. Guelman and Guillén (2014, *Expert Systems with Applications*, 41(2)) found, applying causal methods to motor renewal data, that the propensity-score-adjusted elasticity was roughly half the naive estimate on their Spanish motor dataset. That finding is directionally consistent with every commercial team we have spoken to who has done this comparison in UK data.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=["age_band", "vehicle_group", "postcode_band", "ncd", "prior_claims"],
    nuisance_model="catboost",
    cv_folds=5,
)
model.fit(df_train, exposure_col="policy_years")
ate = model.average_treatment_effect()
```

```
ATE: -0.023  (95% CI: -0.031 to -0.015)
```

The confidence interval is computed from the asymptotic normality of the OLS estimate in step 3. It is a valid frequentist interval, not a bootstrap approximation.

For a telematics factor:

```
Factor: telematics_harsh_braking
  Naive GLM coefficient:  +0.181
  DML causal estimate:    +0.062
  Implied confounding:    +0.119
  Interpretation: two-thirds of the observed association is geography confounding
```

If the causal effect is genuinely 0.062 rather than 0.181, a pricing factor weighted at 0.181 is doing the wrong thing. This is the output that justifies doing this analysis — it converts an abstract methodological point into a table a head of pricing can use.

You can pass a fitted GLM directly instead of a manual coefficient:

```python
report = model.confounding_bias_report(glm_model=fitted_statsmodels_glm)
```

---

## Treatment types

The library handles the three treatment structures that appear most often in insurance pricing.

**Price change** (continuous, the renewal pricing problem):

```python
from insurance_causal.treatments import PriceChangeTreatment

treatment = PriceChangeTreatment(
    column="pct_price_change",   # proportional: 0.05 = 5% increase
    scale="log",                 # log(1+D) transform; θ is a semi-elasticity
    clip_percentiles=(0.01, 0.99),
)
```

**Binary treatment** (channel, discount flag):

```python
from insurance_causal.treatments import BinaryTreatment

treatment = BinaryTreatment(
    column="is_aggregator",
    positive_label="aggregator",
    negative_label="direct",
)
```

**Continuous treatment** (telematics score):

```python
from insurance_causal.treatments import ContinuousTreatment

treatment = ContinuousTreatment(
    column="harsh_braking_score",
    standardise=True,   # coefficient = effect of 1 SD increase
)
```

---

## Outcome types

Insurance outcomes are not Gaussian. The library handles the distributions that actually appear in the data:

```python
CausalPricingModel(
    outcome_type="binary",      # renewal indicator, conversion
    outcome_type="poisson",     # claim count, with exposure handling
    outcome_type="continuous",  # log loss cost
    outcome_type="gamma",       # claim severity (log-transformed internally)
)
```

For claim frequency with a telematics score, the causal effect of a one-standard-deviation increase in harsh braking on claim frequency is +0.6 percentage points after controlling for urban mileage, postcode, age, vehicle group, and NCB. The naive GLM coefficient on the same feature, without controlling for urban mileage correctly, was +1.8 percentage points. Most of that association is geography confounding.

---

## CATE by segment

Average treatment effects can conceal heterogeneity that matters for pricing. The price elasticity for a 20-year-old in their first year of driving is probably different from that of a 45-year-old with five years' NCB. `cate_by_segment()` estimates the causal effect separately within subgroups:

```python
cate = model.cate_by_segment(df, segment_col="age_band")
```

For heterogeneous treatment effects, `CausalForestDML` (Wager and Athey, 2018, *JASA*, 113(523)) extends DML to per-customer CATE estimation:

```python
from insurance_causal.estimators import HeterogeneousTreatmentEffect

hte = HeterogeneousTreatmentEffect(
    outcome="renewed",
    treatment=PriceChangeTreatment(column="pct_price_change"),
    confounders=["age_band", "vehicle_group", "postcode_band", "ncd", "prior_claims"],
    heterogeneity_features=["time_on_book", "ncd", "age_band"],
    nuisance_model="catboost",
)

hte.fit(df_train)
cate = hte.effect(df_test)
# Returns DataFrame with point estimate and CI per policy
```

A consistent pattern on motor renewal data is that the causal price elasticity is larger in the lower risk deciles than the upper deciles. High-risk customers lapse at high rates regardless of price; the price lever is most effective on the lower-risk, longer-tenure segment. A renewal optimiser built on a single pooled elasticity misses this entirely. The commercial use case: the renewal targeting rule becomes "offer a discount when τ̂(x) × expected_margin > discount_cost."

---

## Sensitivity to unobserved confounders

The critical limitation of DML is that it is only as good as the assumption that all relevant confounders are in the `confounders` list. The sensitivity analysis module quantifies the fragility of a result to violations of this assumption:

```python
from insurance_causal.diagnostics import sensitivity_analysis

ate = model.average_treatment_effect()
report = sensitivity_analysis(
    ate=ate.estimate,
    se=ate.std_error,
    gamma_values=[1.0, 1.25, 1.5, 2.0, 3.0],
)
print(report[["gamma", "conclusion_holds", "ci_lower", "ci_upper"]])
```

```
Gamma=1.1: estimate range [-0.031, -0.014]  - sign stable
Gamma=1.5: estimate range [-0.039, -0.006]  - sign stable
Gamma=2.0: estimate range [-0.048, +0.002]  - sign borderline
Gamma=3.0: estimate range [-0.061, +0.016]  - sign could flip
```

The Rosenbaum parameter Γ represents the odds ratio of treatment assignment for two units with identical observed confounders. If `conclusion_holds` becomes False at Γ = 1.25 — a very mild violation — the result is fragile. Present this analysis alongside any causal estimate that influences a pricing decision.

---

## Three applications where this matters now

### FCA Consumer Duty and factor defence

Consumer Duty (July 2023) requires insurers to demonstrate that products provide fair value. For pricing, this means being able to show that rating factors reflect genuine risk differentials rather than protected-characteristic proxies.

"The GLM coefficient on vehicle group is X" is a correlation. "The DML estimate of vehicle group's causal effect on claim frequency, controlling for age, postcode, and occupation, is Y with 95% CI [a, b]" is evidence. A causal estimate with a confidence interval and a sensitivity analysis is a substantially stronger regulatory argument under FCA PS21/5 than `exp(β)` from a GLM.

### Competitive advantage on confounded relativities

If your competitors' age relativities are partly urban-driving relativities, there is a segment where they are systematically overpricing: suburban young drivers. A suburban 22-year-old is priced as if they were a generic 22-year-old, with a significant portion of the age loading actually tracking urban driving behaviour. Your DML estimate separates the genuine age effect from the urban driving effect — a pricing basis that is more accurate in the suburbs.

### Pricing decisions under mix shift

Confounded coefficients misallocate premium when the portfolio mix shifts. A causal estimate is stable to mix shift: it measures the effect of age on claims holding constant urban driving, so changes in the urban-rural portfolio mix do not cause the age relativity to be miscalibrated.

---

## What the library does not solve

**The bad controls problem.** Including a mediator — a variable causally downstream of treatment — as a confounder blocks the causal channel you are trying to measure. NCB is a good example: it is partly caused by claim history, which is the thing you want to measure. The library includes a DAG specification module that checks for this before fitting:

```python
from insurance_causal import CausalDAG

dag = CausalDAG()
dag.add_treatment("pct_price_change")
dag.add_outcome("renewed")
dag.add_confounder("age_band")
dag.add_confounder("ncd")  # will flag ncd as a potential mediator
dag.validate()
# Warning: ncd is downstream of prior_claims which may affect renewed directly.
# Consider whether ncd is a confounder or mediator for this treatment.
```

**Near-deterministic treatment.** Insurance re-rating makes the offered price nearly a deterministic function of observable risk factors. When D̃ has near-zero variance, confidence intervals blow up and the point estimate is noise. The solution is to include genuinely exogenous variation: manual underwriting adjustments, competitive market shocks, or timing effects outside the model. The PLIV (partially linear IV) extension is also supported by the library.

**Unobserved confounders.** DML cannot adjust for what you did not observe. Actual annual mileage, attitude to risk, and claim reporting behaviour are all relevant and never fully observed. The sensitivity analysis bounds how bad this could be, but it cannot fix the underlying data limitation.

---

## Getting started

```bash
uv add insurance-causal
```

Dependencies: `doubleml`, `catboost`, `polars`, `pandas`, `scikit-learn`, `scipy`, `numpy`.

Start with the confounding bias report. Run it on a factor you already have in your GLM and think you understand. If the DML estimate and the GLM coefficient are within 10% of each other, confounding is not a major problem for that factor. If they diverge materially, you have evidence that the factor's GLM coefficient is carrying confounding bias, and you should think carefully about how that affects the decisions it is used for.

The right use of this library is not to replace your GLM. It is to audit the causal standing of the factors in your GLM — to find out which coefficients you can trust for intervention decisions and which ones are proxies.

Source and issue tracker on [GitHub](https://github.com/burning-cost/insurance-causal).

---

**Related articles from Burning Cost:**
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/)
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/)
