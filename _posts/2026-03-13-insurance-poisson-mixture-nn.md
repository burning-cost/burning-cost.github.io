---
layout: post
title: "Not All Zero-Claimers Are Equal"
date: 2026-03-13
categories: [frequency-modelling, neural-networks, telematics]
tags: [poisson-mixture, zero-inflation, structural-zeros, telematics, python]
description: "ZIP models treat all zero-claimers identically. The PM-DNN decomposes them into structural zeros (will never claim) and stochastic zeros (got lucky this year), and shows a 14x pricing difference between the two types."
---

Two policyholders. Both 28, both driving a 2021 hatchback in the same postcode. Neither claimed last year.

One drives 500 miles per year - a city dweller who keeps the car for the occasional weekend. The other drives 15,000 miles a year across the UK for work.

Your frequency GLM gives them the same predicted claim rate. Your ZIP model makes it marginally better, but not by much. Both get the same NCD treatment. One of them is almost certainly undercharged.

The problem is not that your model lacks data. It is that it does not ask the right question. The right question is not "did they claim last year?" It is "are they the kind of driver who *could* claim?"

---

## The two types of zero

Yip & Yau (2005, IME) introduced the distinction clearly for insurance: there are two fundamentally different reasons a policyholder records zero claims.

**Structural zeros** are policyholders who will not claim regardless of exposure. The low-mileage city driver barely puts the car on a road where an accident is possible. The second car that sits on the driveway from November to March. The student car that does 400 miles a term. Zero claims is the only actuarially correct outcome for them.

**Stochastic zeros** are policyholders who face genuine claim risk but happened not to trigger it this period. The 15,000-miles-per-year sales rep had a Poisson process running all year; it generated a zero this time. Over a long enough window, they will claim. Poisson(lambda) generates zero with probability exp(-lambda), but for high lambda that probability gets small fast.

The standard ZIP model - the tool most pricing actuaries reach for when zero-inflation is flagged - cannot separate these two types. Its formulation is:

```
P(Y=0 | ZIP) = q + (1-q)*exp(-lambda)
```

The parameter `q` blends both sources. Some of it comes from structural zeros; the remainder is stochastic zeros absorbed into the mixture. ZIP estimates a single `q` across the portfolio. There is no individual-level decomposition. You cannot ask "for *this* policyholder, what is the probability they are a structural zero?"

---

## What the PM-DNN does differently

The NAAJ 2025 paper - "Poisson Mixture Deep Learning Neural Network Models for the Prediction of Drivers' Claims with Excessive Zero Claims Using Telematics Data" (DOI: 10.1080/10920277.2025.2570289) - separates the two components properly.

The model has two Poisson components:

- Component 0 (safe group): Poisson rate `lambda_0(x)`, constrained to be near zero. These are structural zeros.
- Component 1 (risky group): Poisson rate `lambda_1(x)`, constrained to exceed `lambda_0`. These are stochastic zeros and actual claimers.

The mixture probability `pi(x)` - the probability any individual policyholder belongs to the risky group - is estimated by a neural network that takes their full covariate vector as input. Including telematics.

The full PMF:

```
P(Y=0 | x) = (1 - pi(x)) * exp(-lambda_0(x)) + pi(x) * exp(-lambda_1(x))
P(Y=k | x) = (1 - pi(x)) * Poisson(k; lambda_0(x)) + pi(x) * Poisson(k; lambda_1(x))
```

When `lambda_0 -> 0`, this collapses to ZIP. The PM model lets the data determine how degenerate the safe component actually is, rather than forcing it to zero. That flexibility is the key advance.

ZIP is a special case of PM-DNN. Fitting PM-DNN and finding `lambda_0` near zero everywhere tells you ZIP was fine for your portfolio. But in a telematics motor book, that will not be what you find.

---

## pi(x) as an individual pricing signal

The output `pi(x)` - the posterior probability that policyholder `x` belongs to the risky group - is directly usable as a pricing signal. The expected claim frequency is:

```
E[freq] = pi(x) * lambda_1(x) + (1 - pi(x)) * lambda_0(x)
```

Now run the numbers for our two identical-demographic zero-claimers.

**500 miles/year driver (telematics):** `pi = 0.05`, `lambda_0 = 0.001`, `lambda_1 = 0.15`
Expected frequency = 0.05 * 0.15 + 0.95 * 0.001 = **0.0085 claims/year**

**15,000 miles/year driver (telematics):** `pi = 0.80`, `lambda_0 = 0.001`, `lambda_1 = 0.15`
Expected frequency = 0.80 * 0.15 + 0.20 * 0.001 = **0.120 claims/year**

That is a 14x difference in predicted frequency between two policyholders with identical claim histories and identical demographics. The difference comes entirely from mileage and driving behaviour entering `pi(x)`. ZIP would price them the same.

The PM-DNN paper reports outperforming Poisson GLM, Poisson DNN, ZIP GLM, ZIP DNN, and ZINB GLM on both NLL and Gini metrics.

---

## The NCD problem

This decomposition matters for more than rate-at-inception. It changes how you think about NCD.

Under standard NCD logic, both policyholders above would receive the same discount for a clean year. Maximum NCD rewards a clean claims year. It does not ask whether that clean year was structurally guaranteed or statistically lucky.

A structural zero with `pi = 0.05` genuinely will not claim next year either. They deserve the discount. A stochastic zero with `pi = 0.80` and `lambda_1 = 0.15` is a pricing risk. They had a 14.5% chance of claiming last year. Their single clean year should not be treated as evidence of low risk - it is the most likely outcome for their risk level, not evidence that their risk level is low.

The pi score gives you a principled basis for NCD adjustment. A policyholder's expected frequency under PM-DNN already accounts for their true risk classification. The NCD factor does not need to do the heavy lifting of distinguishing the lucky from the genuinely low-risk - the model does it.

---

## UK telematics motor: the strongest application

Structural zeros are hardest to identify without behavioural data. Demographic proxies are weak. But with telematics, mileage and driving behaviour enter `pi(x)` directly - and they are among the strongest predictors of structural zero status.

UK telematics market context: the market is growing at 18.9% CAGR through 2034 (GMI, March 2025). Smartphone-based UBI is the fastest-growing segment, replacing black boxes for young drivers at the majority of UK personal lines insurers. That means behavioural telematics data is increasingly standard, not premium.

Natural structural zero candidates in a UK telematics motor book:

- Parent-insured vehicles for university students (low mileage most of the year, high over holidays)
- City flat-dwellers who own a car primarily for access to parents' homes at weekends
- Second cars in multi-vehicle households used seasonally
- Young drivers whose parents installed telematics for the discount but who rarely use the car independently

For these policyholders, `pi` will be low even if their demographics look average. The PM model identifies them from their telematics behaviour, not their age or occupation.

Where PM-DNN is marginal: standard personal motor without telematics. Without a mileage or behaviour covariate entering `pi(x)`, the structural zero signal is weak. The two lambda components will tend to converge and the model degrades toward ZIP. If you do not have telematics data, ZIP DNN is probably sufficient.

---

## The library

[`insurance-poisson-mixture-nn`](https://github.com/burning-cost/insurance-poisson-mixture-nn) is a PyTorch implementation of the PM-DNN. There is no public code from the NAAJ 2025 paper; the authors used Keras/TensorFlow and did not release it.

The implementation handles the two main technical problems in the paper:

**Component ordering.** Without constraints, `lambda_0` and `lambda_1` can swap during training and `pi(x)` becomes unidentified. We use softplus reparameterisation: `lambda_1 = lambda_0 + softplus(raw)`, which guarantees ordering algebraically throughout training.

**Numerical stability.** Naive NLL computation produces NaN on real insurance data with many near-zero lambdas. The loss is computed via log-sum-exp.

```python
from insurance_poisson_mixture_nn import PoissonMixtureNN, PoissonMixtureTrainer

model = PoissonMixtureNN(n_features=12, hidden_dims=[64, 32, 16])
trainer = PoissonMixtureTrainer(model, lr=1e-3, patience=10)
trainer.fit(X_train, y_train, exposure_train)

# Posterior at-risk probability per policy
pi = model.predict_pi(X_test)

# Structural vs stochastic zero classification
labels = model.classify_zero(X_test, threshold=0.5)
```

The library also ships a comparison module that benchmarks PM-DNN against Poisson GLM, Poisson DNN, and ZIP DNN on the same data, returning NLL, Gini, and calibration metrics in a single DataFrame. This replicates the paper's results table structure.

```bash
uv add insurance-poisson-mixture-nn
```

---

## A note on the regulatory angle

FCA Consumer Duty (PRIN 2A, July 2023) creates an obligation to price with accuracy and to avoid systematic overcharging of lower-risk customers. Structural zeros charged on a blended ZIP probability are being overcharged relative to their actual risk. That is not an academic concern - it is the kind of systematic cross-subsidy that PRIN 2A.4 (fair value) is designed to prevent.

The `pi(x)` score provides an auditable, covariate-conditioned basis for pricing segmentation. It is estimable, interpretable in terms of the underlying model, and documentable in a model card. That is more defensible under Consumer Duty scrutiny than "we applied a ZIP adjustment to the frequency GLM."

One caveat worth stating plainly: if `pi(x)` acts as a proxy for a protected characteristic - mileage correlates with gender in UK portfolios even post the 2012 ECJ ruling - there is indirect discrimination exposure. The library includes a fairness diagnostics module for this reason. Check it before production deployment.

---

The paper: Poisson Mixture Deep Learning Neural Network Models for the Prediction of Drivers' Claims with Excessive Zero Claims Using Telematics Data, NAAJ 2025. DOI: 10.1080/10920277.2025.2570289

The precursor: Boucher, Denuit & Guillen (2007), NAAJ 11(4):110-131 - the foundational zero-inflated count model comparison for motor insurance.
