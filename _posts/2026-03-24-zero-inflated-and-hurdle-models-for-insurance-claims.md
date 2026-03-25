---
layout: post
title: "Zero-Inflated and Hurdle Models for Insurance Claims"
date: 2026-03-24
categories: [pricing, techniques, tutorials]
tags: [zero-inflation, hurdle-models, tweedie, frequency, poisson, negative-binomial, catboost, insurance-distributional, specialty-lines, pet-insurance, tutorial]
description: "Standard Tweedie GLMs handle zeros implicitly. When that implicit handling breaks — specialty lines, niche segments, specific peril models — you need ZIP or hurdle models. Here is when, why, and how."
---

The standard workhorse of insurance pricing is the Tweedie GLM. You feed it aggregate claim costs, it handles the zeros automatically, and you move on. For most personal lines motor and home books, this is entirely sensible. Tweedie's compound Poisson-Gamma structure has a probability mass at zero baked in — the zero occurs when the Poisson count is zero, with probability exp(-lambda).

The problem is that this implicit zero-handling is fixed by the mean. Once you specify E[Y|x], the Tweedie model's zero probability is fully determined. For a Tweedie with power p=1.5 and a mean of £200, the zero probability is a particular number. You have no free parameter to adjust it. If your data have substantially more zeros than this implicit structure predicts, the model is misspecified — and you usually will not know, because aggregate Tweedie deviance is not a sensitive test for excess zeros.

In specialty lines, niche segments, and specific peril models, the zero problem is real and the mismatch is large.

---

## Why zeros matter more than you might think

Consider a marine cargo book where 85% of policies have zero claims in a year. A standard Tweedie fitted to this data will try to accommodate that zero mass through the mean — it will push lambda down. The result is that the non-zero predictions are compressed: the model cannot simultaneously have very low mean and a realistic severity distribution for the claims that do occur.

Or consider a telematics sub-segment: young drivers, urban, night-time driving, sub-50 exposures per cell. Some cells have zero claims not because the underlying risk is low, but because the exposure is thin and claim events are rare. A standard Tweedie will assign a low mean to these cells. When similar risks do claim, the model's severity estimate is off because it was fitted partly on these sparse zero cells.

The same pattern appears in:

- **Pet insurance**: most pets never claim in a policy year. The structural zero rate (pet never interacts with a vet) and the sampling zero rate (short exposure, no opportunity to claim) are mixed together.
- **Travel insurance**: most trips are uneventful. Cancellation claims in particular have strong seasonality and event clustering that inflate zeros outside peak periods.
- **Breakdown cover**: a newly serviced vehicle may have a structural zero probability that is genuinely different from an older vehicle's process-zero probability.
- **Fraud indicators**: modelling binary or count fraud indicators often has structural zero rates driven by scheme-level rather than policy-level processes.
- **Specific peril models**: a theft-only model on a book where theft rate is 0.3% has 99.7% zeros. No Tweedie can handle this gracefully.

The distinction that matters throughout this post is between **structural zeros** (the risk could never have generated a positive outcome in the observation window, for reasons not captured by your covariates) and **sampling zeros** (the risk could have claimed but did not). Tweedie conflates these. Zero-inflated and hurdle models separate them.

---

## Tweedie's implicit zero handling

For a compound Poisson-Gamma Tweedie model with power p in (1, 2):

```
Y ~ Tweedie(mu, phi, p)
P(Y = 0) = exp(-mu^(2-p) / (phi * (2-p)))
```

The zero probability is entirely determined by mu, phi, and p. In practice p is fixed in advance (1.5 for most motor books) and phi is estimated as a scalar. So the zero probability is purely a function of the mean prediction. This is the constraint that breaks when zero inflation is structural.

You can test whether Tweedie is adequate with a simple diagnostic: compute `P_tweedie(Y=0 | mu_hat, phi_hat, p)` for each observation and compare the mean to the observed zero proportion. If the observed zero rate is, say, 72% and the model-implied zero rate is 45%, you have a structural zero problem that Tweedie cannot solve by re-estimating mu.

---

## The two families: zero-inflated and hurdle

Both families add an explicit zero-generating mechanism on top of a count or continuous model. They differ in one fundamental way.

**Zero-inflated models** say: zeros come from two sources. Either the unit is a structural zero (with probability pi), or it comes from the count process (and happened to be zero). A zero-inflated Poisson (ZIP) has:

```
P(Y = 0 | x) = pi(x) + (1 - pi(x)) * exp(-lambda(x))
P(Y = k | x) = (1 - pi(x)) * Poisson(k; lambda(x))    for k > 0
E[Y | x] = (1 - pi(x)) * lambda(x)
```

The pi(x) term is modelled separately — you get a second prediction surface for the structural zero probability. A risk with high pi is fundamentally different from a risk with low pi but low lambda, even if both have the same expected claim count.

**Hurdle models** say: first determine whether Y is zero. If it is not zero, draw from a truncated positive distribution. There is no mixture of zero sources — a zero is always a zero, and non-zeros come from a single positive-support distribution. A Poisson hurdle looks like:

```
P(Y = 0 | x) = p0(x)                       [logistic model]
P(Y = k | x) = (1 - p0(x)) * Poisson+(k)   for k > 0, where Poisson+ is zero-truncated
```

The practical difference: in a ZIP model, a structural zero unit and a sampling-zero unit share the same positive-outcome process. In a hurdle model, the zero/non-zero split is entirely separate from the count distribution. For insurance frequency modelling, the ZIP is usually more interpretable — the "would claim if anything happened" risk shares a frequency process with the "actually claimed" risks. For severity or aggregate cost, hurdle models are natural: you first decide if there is a claim, then model the cost conditional on there being one.

Neither approach is universally better. We use ZIP for frequency models where we want to model the structural zero probability explicitly. We use hurdle (separate logistic + positive distribution) for pure cost models where the no-claim/claim split has a clear business interpretation.

---

## Zero-inflated Poisson in practice with `insurance-distributional`

The [`insurance-distributional`](https://github.com/burning-cost/insurance-distributional) library implements `ZIPGBM`: a gradient-boosted ZIP that jointly estimates lambda(x) and pi(x) using coordinate descent, following the So & Valdez (2024, arXiv 2406.16206) ASTIN Best Paper approach.

```bash
uv add insurance-distributional
```

Here is the full workflow on a synthetic pet insurance frequency dataset:

```python
import numpy as np
from insurance_distributional import ZIPGBM

rng = np.random.default_rng(2026)
n = 8_000

# Covariates: pet age, species (0=cat, 1=dog), breed risk score, policy tenure
X = np.column_stack([
    rng.integers(0, 15, n).astype(float),   # pet_age
    rng.integers(0, 2, n).astype(float),    # species
    rng.uniform(0, 1, n),                   # breed_risk
    rng.integers(1, 8, n).astype(float),    # tenure_years
])

# True DGP: 35% structural zeros (indoor cats, specific breeds)
# Non-zero claims: Poisson(0.25)
pi_true = np.where(X[:, 1] == 0, 0.45, 0.25)  # cats have higher structural zero rate
lam_true = 0.2 + 0.05 * X[:, 2]               # breed risk affects lambda
y = np.where(
    rng.random(n) < pi_true,
    0,
    rng.poisson(lam_true)
)

print(f"Observed zero rate: {(y == 0).mean():.1%}")
# Observed zero rate: 68.4%

exposure = np.ones(n)  # annual policies

# Fit ZIP GBM
model = ZIPGBM(n_cycles=2, random_state=42)
model.fit(X, y, exposure=exposure)

pred = model.predict(X, exposure=exposure)

print(f"Mean pi (zero-inflation prob): {pred.pi.mean():.3f}")   # ~0.35
print(f"Mean lambda:                   {model.predict_lambda(X, exposure).mean():.3f}")  # ~0.22
print(f"Mean observable E[Y|x]:        {pred.mean.mean():.3f}")  # ~0.14

# The pi surface shows which segments are structurally different
# Compare cats vs dogs
cat_mask = X[:, 1] == 0
print(f"Mean pi for cats: {pred.pi[cat_mask].mean():.3f}")   # higher
print(f"Mean pi for dogs: {pred.pi[~cat_mask].mean():.3f}")  # lower
```

The `pred.pi` array is what you cannot get from a standard Tweedie. It tells you, for each risk, the estimated probability that the zero-claim outcome is structural rather than just lucky. A policy in a segment with pi=0.7 is categorically different from one with pi=0.05, even if their expected claim rates look similar.

For overdispersed frequency data — fleet motor, commercial property with multiple sub-locations — replace `ZIPGBM` with `NegBinomialGBM`:

```python
from insurance_distributional import NegBinomialGBM

# Fleet motor: overdispersed claim counts
nb_model = NegBinomialGBM(model_r=False, random_state=42)
nb_model.fit(X_fleet, y_fleet, exposure=exposure_fleet)

nb_pred = nb_model.predict(X_fleet, exposure=exposure_fleet)
print(f"Mean E[Y|x]: {nb_pred.mean.mean():.3f}")
print(f"Overdispersion r: {nb_pred.r.mean():.2f}")  # small r = high overdispersion
```

---

## Diagnostics: is my zero rate actually a problem?

Before you reach for a ZIP model, check whether you actually have a structural zero problem. Two diagnostics we use routinely:

**1. Rootogram.** Plot the expected frequencies under the fitted model against the observed frequencies for each count value (0, 1, 2, ...). A hanging rootogram (Kleiber & Zeileis 2016) makes zero inflation visually obvious: the fitted Poisson bar will be substantially shorter than the observed bar at zero, and compensatingly longer at one or two.

**2. Vuong test.** Fit both the standard Poisson GLM and the ZIP model. The Vuong test (Vuong 1989) compares the per-observation log-likelihoods of two non-nested models without a common null. A strongly positive test statistic means ZIP fits better than Poisson; strongly negative means the reverse. In our experience, Vuong is most useful for confirming a ZIP diagnosis, not for discovering one — by the time the test is significant, the rootogram already shows it clearly.

**3. Simple observed vs model-implied zero rate.** Compute `P(Y=0 | mu_hat, phi_hat, p)` from your fitted Tweedie, average across the portfolio, and compare to the observed zero rate. A gap of more than 5 percentage points in a personal lines book is a strong signal. In specialty lines, gaps of 20-30pp are common.

If none of these diagnostics shows a problem, standard Tweedie is fine. We would not complicate a UK motor personal lines model with ZIP unless we were working on a very specific sub-segment (e.g., a glass-only peril model on a book with structural zero rates from direct-replacement schemes).

---

## When to use which model

| Situation | Our recommendation |
|-----------|-------------------|
| Standard UK motor / home aggregate | Tweedie GLM or TweedieGBM. Zero handling is adequate. |
| Pet/travel/breakdown frequency | `ZIPGBM`. Structural zeros are real and predictable. |
| Fleet motor frequency | `NegBinomialGBM`. Overdispersion matters; structural zeros less so. |
| Specific peril (theft-only, glass-only) | ZIP or hurdle with logistic zero model. Zero rate 95%+. |
| Specialty lines (marine, aviation, energy) | Hurdle model: separate logistic + Gamma severity. |
| Sub-50-exposure segments, thin data | Use ZIP but keep the pi model simple (fewer features). Overfitting risk on pi is real. |
| Fraud indicator modelling | ZIP or zero-inflated Negative Binomial depending on count vs binary target. |

The standard Tweedie is not broken. It is the right tool for most aggregate cost models on large personal lines books, and we use it constantly. The issue is knowing when its implicit zero-handling assumption is violated — and having a tool ready when it is.

---

## A note on the hurdle model

The library does not yet implement a hurdle model as a single estimator. The standard approach is to compose one:

```python
from sklearn.linear_model import LogisticRegression
from insurance_distributional import GammaGBM
import numpy as np

# Stage 1: zero vs non-zero
hurdle_zero = LogisticRegression(C=1.0)
hurdle_zero.fit(X_train, (y_train > 0).astype(int))

# Stage 2: severity conditional on positive outcome
mask_pos = y_train > 0
severity_model = GammaGBM(random_state=42)
severity_model.fit(X_train[mask_pos], y_train[mask_pos])

# Combined prediction
p_nonzero = hurdle_zero.predict_proba(X_test)[:, 1]
sev_pred = severity_model.predict(X_test).mean

pure_premium = p_nonzero * sev_pred
```

This is the correct structure for a specialty lines aggregate model where the zero/non-zero decision and the severity amount are driven by genuinely separate processes — a cargo policy either has a loss event or it does not, and the cost of the loss event is largely independent of whether it occurred. This is structurally different from the ZIP assumption, where the non-zero outcomes and the structural zeros come from the same underlying risk population.

---

## References

- So & Valdez (2024). *Zero-Inflated Tweedie Boosted Trees with CatBoost for Insurance Loss Analytics*. arXiv 2406.16206. ASTIN Best Paper 2024.
- Vuong, Q.H. (1989). *Likelihood Ratio Tests for Model Selection and Non-Nested Hypotheses*. Econometrica 57(2):307-333.
- Kleiber, C. & Zeileis, A. (2016). *Visualizing Count Data Regressions Using Rootograms*. The American Statistician 70(3):296-303.
- Lambert, D. (1992). *Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing*. Technometrics 34(1):1-14.
