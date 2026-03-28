---
layout: post
title: "Tweedie vs Frequency-Severity Split: When to Use Which"
date: 2026-03-24
author: Burning Cost
categories: [modelling, tutorials]
tags: [tweedie, frequency-severity, GLM, compound-poisson-gamma, sarmanov, copula, insurance-frequency-severity, pure-premium, motor-insurance, uk-insurance]
description: "The Tweedie GLM and frequency-severity split both model pure premium. They are not interchangeable. Here is how to decide which one you actually need."
---

Every UK motor pricing actuary eventually faces the same question. You are building a pure premium model. Two approaches are on the table: a single Tweedie GLM, or separate frequency and severity models multiplied together. Both are defensible. Both are used in production by major carriers. The question is which one is appropriate for your situation - and the answer is not "it depends" left at that.

We will explain the technical relationship between the two, the conditions under which they diverge, and when the frequency-severity split requires a third model to handle the dependence between N and S.

---

## The relationship between Tweedie and compound Poisson-Gamma

A Tweedie GLM with power parameter p between 1 and 2 is, mathematically, a compound Poisson-Gamma distribution. If you write the true data-generating process as:

```
Y = S_1 + S_2 + ... + S_N
```

where N ~ Poisson(lambda) and S_i ~ Gamma(alpha, beta) independently, then Y has a Tweedie distribution with power parameter p = (alpha + 2) / (alpha + 1). The Tweedie model is fitting lambda and the gamma parameters simultaneously, constrained to a single linear predictor.

The frequency-severity split is the same underlying structure, but with two separate linear predictors: one for E[N|x] and one for E[S|x]. You then combine them as:

```
E[Y|x] = E[N|x] * E[S|x]
```

This combination assumes N and S are independent given x. The Tweedie model assumes the same thing - it is baked into the distributional form. Both models share this independence assumption. The difference is in how they use your covariates.

---

## When the Tweedie GLM is appropriate

The Tweedie model's key constraint is that the power parameter p is a single value for the entire portfolio. This p encodes the ratio of claim count variance to mean, and the gamma shape for severity. In a Tweedie GLM, every policy has the same underlying severity distribution shape - only the scale changes via the linear predictor.

This is fine when:

**Your portfolio is genuinely homogeneous.** Personal lines home insurance, straightforward personal motor where severity is dominated by a few stable vehicle damage categories. One p fits reasonably well across the book.

**You care about aggregate premium accuracy more than segment-level accuracy.** The Tweedie model minimises Tweedie deviance across the whole portfolio. It may trade off accuracy on specific segments (young drivers with high-severity vehicles) against aggregate fit.

**You have limited data.** Fitting one model is more stable than fitting two and multiplying. On a book of 50,000 policies with 3,000 claims, the Tweedie model will often be more reliable than a severity GLM fitted on those 3,000 claim observations.

**The frequency and severity drivers are the same.** If vehicle age, driver age, and postcode affect both frequency and severity in the same direction, a single linear predictor may capture enough. If the drivers are different - which they often are - you are forcing a compromise.

The typical UK personal lines motor Tweedie GLM returns a p estimate around 1.5-1.6. Using the formula p = (alpha+2)/(alpha+1), p=1.5 corresponds to a Gamma shape of 1.0 (the exponential distribution) and p=1.6 corresponds to a shape of 0.67. These are thin-tailed severity distributions. If your expected severity has a shape more like 2-3  -  typical of mixed bodily injury and property damage books  -  the corresponding p would be closer to 1.25-1.33. If your estimated p is outside [1.2, 1.8], question whether the compound Poisson-Gamma assumption is appropriate at all.

---

## When the frequency-severity split is necessary

**Different covariate effects.** In UK motor, young male drivers have high frequency and moderate severity. High-value vehicle owners have moderate frequency and high severity. These are genuinely different relationships, and compressing them into a single linear predictor loses information. The frequency model says "this driver claims a lot"; the severity model says "when they claim, the amount is X". You want both.

**Reserving integration.** If your pricing team shares data with reserving, separate frequency and severity estimates are the natural unit of exchange. The reserving team models IBNR on claim counts; the severity estimates feed into IBNER. A Tweedie output does not decompose cleanly into these.

**Subsidy analysis.** Regulators and pricing committees increasingly ask about cross-subsidy between segments. "What fraction of this segment's pure premium comes from high frequency vs high severity?" is a legitimate question. A Tweedie model cannot answer it.

**Non-standard severity distributions.** If your severity has a heavy tail requiring a lognormal, Pareto, or spliced distribution, a Tweedie model cannot accommodate this. You are constrained to gamma severity. The frequency-severity split lets you choose any severity distribution that fits your data.

---

## The independence assumption and when it breaks down

Both approaches assume N and S are independent given the rating factors. In UK motor, this assumption is almost certainly wrong, and the direction of the bias is predictable.

The No Claims Discount structure suppresses borderline claims. A policyholder at 65% NCD knows that reporting a £600 scratch will cost them more in future premiums than the claim is worth. They absorb it. The policyholders who do claim tend to have more severe incidents. This creates a systematic negative correlation between N and S that the rating factors do not fully explain - conditional on everything your model knows about the policyholder, higher-frequency risks still tend to have lower average severity per claim.

Vernic, Bolancé, and Alemany (2022) quantified this in Spanish auto data: the mispricing from ignoring the dependence amounts to EUR 5-55+ per policyholder. The directional effect in UK motor is the same; the magnitude depends on how aggressive your NCD structure is and what fraction of borderline claims are being suppressed.

The Tweedie model has no mechanism to capture this. Its independence assumption is structural. The standard frequency-severity split inherits the same problem: E[N * S] = E[N] * E[S] only if N and S are independent.

---

## Measuring the dependence before choosing a method

Before you commit to either approach, run the dependence test:

```python
from insurance_frequency_severity import DependenceTest

# policies_df must contain only positive-claim observations
# n_col: claim count, s_col: average severity per claim
test = DependenceTest(n_permutations=1000)
result = test.fit(
    n=policies_df["claim_count"],
    s=policies_df["avg_severity"],
)
result.summary()
```

This runs three tests: Kendall tau, Spearman rho, and a likelihood ratio test for omega=0 in the Sarmanov copula model. If all three fail to reject independence (p > 0.05), use the standard frequency-severity split. The Tweedie model is also fine in this case, and your choice reduces to the other factors above.

If the test rejects independence - which it typically will on UK motor books above 10,000 policies - neither the Tweedie model nor the standard frequency-severity split is correct. You need to model the dependence explicitly.

---

## Handling dependence: the Sarmanov correction

The `insurance-frequency-severity` library fits a Sarmanov copula on top of your existing frequency and severity GLMs. It does not refit the marginals - it estimates the dependence parameter omega on top of whatever GLMs you already have. The correction is per-policy and multiplicative:

```python
from insurance_frequency_severity import JointFreqSev, DependenceTest

# Assume you already have fitted GLMs
# freq_glm: NegativeBinomial or Poisson, fitted via statsmodels
# sev_glm: Gamma, fitted via statsmodels on claim-level data

# Step 1: Test for dependence
test = DependenceTest(n_permutations=1000)
result = test.fit(
    n=train_df["claim_count"],
    s=train_df["avg_severity"],
)
print(result.summary())

# Step 2: If dependence is significant, fit the joint model
model = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm)
model.fit(train_df, n_col="claim_count", s_col="avg_severity")

print(f"Omega: {model.omega_:.3f}  ({model.omega_ci_[0]:.3f}, {model.omega_ci_[1]:.3f})")
print(f"Spearman rho (N,S): {model.rho_:.3f}")

# Step 3: Apply correction factors at scoring time
corrections = model.premium_correction(X_new)
# corrections is a pandas DataFrame with columns: correction_factor, premium_joint, etc.
# corrected_premium = freq_prediction * sev_prediction * corrections["correction_factor"].to_numpy()

```

The omega parameter is the Sarmanov dependence parameter. A negative omega means high-frequency risks have lower conditional severity - consistent with the NCD suppression story. On a synthetic UK motor book with planted negative dependence (omega = -1.5), the library recovers omega within about 20% on 20,000+ policies.

The correction factors are analytical, not simulated. Scoring is as fast as calling `.predict()` twice on your existing GLMs and multiplying by a vector. There is no MCMC, no sampling, no additional scoring infrastructure.

---

## Practical decision framework

Here is how we approach the choice for a UK personal lines motor book:

**Step 1: Run the dependence test.** If independence holds (p > 0.05 across all three tests), you have flexibility. Proceed to step 2. If dependence is significant, you need either the Sarmanov correction on top of a frequency-severity split, or you need to accept that your Tweedie model is misspecified. The Tweedie model with significant N-S dependence is not "approximately right" - it systematically underprices high-frequency segments relative to low-frequency ones.

**Step 2: Check whether your frequency and severity drivers differ.** If the partial effects of your main rating factors (vehicle group, driver age, NCD, area) point in different directions for frequency vs severity, use the split. If they are broadly consistent, a Tweedie model captures the main signal.

**Step 3: Check your data volume.** Below 5,000 policies and 500 claims, the severity GLM is thin. A Tweedie model with empirical p estimation is often more stable. Above 20,000 policies and 2,000+ claims, the split pays off.

**Step 4: Consider downstream use.** If the output feeds reserving, subsidy analysis, or segment-level underwriting, the split is worth the additional complexity regardless of statistical tests.

In practice, most UK commercial motor books and any affinity scheme with 3+ years of data should use the frequency-severity split with a dependence test. Most personal lines home books, and personal motor books below 30,000 policies, can use a Tweedie model and accept a modest cost in accuracy.

---

## What the Tweedie model does not tell you

One underappreciated limitation: the Tweedie GLM's p parameter is estimated as a single scalar. On a mixed book - some commercial, some personal, urban and rural, young and old - there is no single p that describes all segments correctly. The model averages over them.

If you split your book and estimate p separately for, say, London urban policies versus rural policies, you will typically find p differs by 0.1-0.2 across segments. This is not sampling noise. It reflects genuinely different severity distributions. The Tweedie model is fitting a compromise p and accepting bias in every segment to minimise total deviance.

The frequency-severity split avoids this by using the severity GLM's functional form (typically gamma, sometimes lognormal) and letting the shape parameter vary implicitly via the linear predictor. It is a more flexible model at the cost of more complexity and more data requirements.

Neither model is universally right. The decision rule above will steer you correctly in the majority of cases. When in doubt, fit both, run a Gini and double-lift validation on held-out data, and let the out-of-sample performance decide.

---

## Summary

| Question | Tweedie | Frequency-Severity |
|---|---|---|
| Single p assumption reasonable? | Yes | Not required |
| Frequency and severity have different drivers? | No | Yes |
| N-S dependence present? | Not handled | Correctable via Sarmanov |
| Data volume below 500 claims? | Preferred | Risky |
| Downstream use requires separate N and S? | No | Yes |
| Regulatory subsidy analysis needed? | Difficult | Straightforward |

For the dependence testing and Sarmanov correction, see [`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity). The `DependenceTest` class should be standard practice before finalising any pure premium modelling approach on a UK motor book.

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/)  -  if neither Tweedie nor frequency-severity gives you per-risk conditional variance, distributional GBMs model the dispersion parameter as a function of covariates
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)  -  distribution-free prediction intervals that work around the parametric assumptions of both modelling approaches
