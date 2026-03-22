---
layout: post
title: "Sarmanov Copula for Frequency-Severity Dependence: The Independence Assumption UK Motor Violates"
date: 2026-03-08
categories: [pricing, libraries, tutorials]
tags: [frequency-severity, sarmanov-copula, dependence, IFM, NCD, UK-motor, pure-premium, copula, insurance-frequency-severity, python, GLM, poisson, gamma, negative-binomial, tutorial]
description: "Your frequency GLM and severity GLM are both correct. Multiplying them is not. How to test and correct for the dependence your pricing model ignores."
---

Every UK motor pricing team runs two GLMs. A Poisson or Negative Binomial for claim frequency. A Gamma or Lognormal for claim severity. Pure premium = E[N|x] times E[S|x]. This is the convention, and it is a mathematically reasonable convention provided one condition holds: that claim count and claim severity are independent, conditional on the rating factors.

That condition does not hold in UK motor. It almost certainly does not hold in your book.

The reason is the No Claims Discount structure. Policyholders with borderline attritional claims -- the £600 windscreen, the £900 bumper scratch -- know the NCD threshold. Reporting a small claim resets or reduces the discount. A policyholder with one prior claim is less likely to report a second near-miss than one with a clean record. The consequence is that policyholders who do make multiple claims tend to make claims that are worth making: larger claims, more clearly above the NCD write-off threshold. The correlation between claim count and average severity is negative. High-count policyholders have, on average, lower severity per claim than the independence model predicts.

Vernic, Bolancé, and Alemany (2022) quantified this on a Spanish auto book at EUR 5-55 per policyholder depending on risk tier. The directional effect in UK motor is the same; the magnitude depends on how aggressive your NCD structure is.

The independence model overstates the pure premium for high-frequency policyholders. It understates it for low-frequency policyholders. The mismatch is systematic, not random noise.

[`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity) gives you a practical way to measure and correct this.

```bash
uv add insurance-frequency-severity
```

---

## Test for dependence before doing anything else

Do not fit a copula model until you know whether there is something to correct. The library has a `DependenceTest` that runs Kendall's tau and Spearman's rho on the paired (N, S) data for claiming policyholders, with permutation p-values.

```python
import numpy as np
import statsmodels.api as sm
from insurance_frequency_severity import DependenceTest

# Load your data -- one row per policy, NaN severity for zero-claim policies
# claim_count: integer claim count per policy
# avg_severity: mean claim cost for claiming policies, NaN otherwise
claims_mask = claim_count > 0

test = DependenceTest(n_permutations=1000)
test.fit(
    n=claim_count[claims_mask],
    s=avg_severity[claims_mask],
)
print(test.summary())
```

A realistic UK motor result looks like this:

```
DependenceTest Summary
======================
Kendall tau:    -0.083   (p = 0.002, permutation)
Spearman rho:   -0.117   (p < 0.001, permutation)
n_pairs:         3,241

Interpretation: Negative dependence detected. Independence assumption
likely overstates pure premium for high-frequency risks.
```

A Spearman rho of -0.12 is not dramatic -- this is not a correlation of -0.6. It is enough to produce a measurable pricing error at portfolio scale on a book of 50,000+ policies. Smaller books will show wider confidence intervals and the test may not reach significance. On books under 10,000 policies, use `ConditionalFreqSev` (described below) rather than the Sarmanov approach.

---

## Why the Sarmanov copula, not a standard copula

The standard Gaussian copula approach has a technical problem for mixed discrete-continuous distributions. Sklar's theorem says any joint distribution decomposes into its marginals plus a copula. But the theorem is not unique when one margin is discrete. The probability integral transform F_N(N) is not uniform when N is an integer -- it is a step function. The standard copula approach approximates around this, and the approximation introduces bias in the dependence parameter estimate.

The Sarmanov bivariate family sidesteps this entirely. Rather than separating the dependence into a copula layer, it works directly with the joint distribution:

```
f(n, s) = f_N(n) x f_S(s) x [1 + omega x phi_1(n; mu_N) x phi_2(s; mu_S)]
```

where phi_1 and phi_2 are bounded kernel functions with zero mean under their respective marginals. When omega = 0 this is exactly the independence model. When omega < 0, joint observations (high n, high s) and (low n, low s) are less likely than independence predicts -- which is what we see in UK motor. No probability integral transform, no approximation. The count margin stays discrete throughout.

The IFM (Inference Functions for Margins) estimator is the practical workhouse here:

1. Fit your frequency GLM as normal. Get E[N|x_i] for each policy.
2. Fit your severity GLM as normal. Get E[S|x_i] for each claiming policy.
3. Plug both into `JointFreqSev`. The library estimates omega by profile likelihood over the claiming-policy observations.

Steps 1 and 2 are unchanged from your existing pipeline. You are not replacing your GLMs. You are adding a single parameter that captures the dependence between them.

---

## Fitting the model

```python
import statsmodels.api as sm
from insurance_frequency_severity import JointFreqSev, DependenceTest

# Fit your marginal GLMs as you normally would
X_const = sm.add_constant(X_features)

freq_glm = sm.GLM(
    claim_count,
    X_const,
    family=sm.families.NegativeBinomial(alpha=0.8),
).fit()

sev_glm = sm.GLM(
    avg_severity[claims_mask],
    X_const[claims_mask],
    family=sm.families.Gamma(link=sm.families.links.log()),
).fit()

# Fit the joint model -- accepts your existing fitted GLM objects
model = JointFreqSev(
    freq_glm=freq_glm,
    sev_glm=sev_glm,
    copula="sarmanov",
)
model.fit(
    claims_df,
    n_col="claim_count",
    s_col="avg_severity",
)

print(model.dependence_summary())
```

```
JointFreqSev Dependence Summary
================================
Copula:         Sarmanov
omega:          -0.41
95% CI:         (-0.61, -0.21)
Spearman rho:   -0.11
AIC (joint):    48,203.1
AIC (indep):    48,251.4
DELTA_AIC:      48.3 in favour of Sarmanov

The independence model is rejected. Negative omega indicates
high-count policyholders have lower-than-expected severity.
```

The delta AIC of 48 is decisive. You are not overfitting one extra parameter. You are finding a genuine signal. A delta AIC above 10 warrants applying the correction; below 4 the evidence is ambiguous and you can present both.

---

## Getting the correction factors

The key output is `premium_correction()`. This returns a DataFrame with one row per policy showing the corrected pure premium alongside the independence model's estimate.

```python
corrections = model.premium_correction()

print(corrections[["mu_n", "mu_s", "correction_factor", "premium_joint"]].describe())
```

```
         mu_n      mu_s  correction_factor  premium_joint
count  50000.0   50000.0           50000.0        50000.0
mean    0.0823   1247.3             0.962          98.6
std     0.0391    312.1             0.031          41.2
min     0.0121    601.2             0.893          16.8
25%     0.0541    981.4             0.943          65.3
50%     0.0798   1214.2             0.961          96.4
75%     0.1074   1498.6             0.980         130.2
max     0.2341   2841.1             1.021         214.7
```

The correction factor is E[N x S] / (E[N] x E[S]). The portfolio average here is 0.962, meaning the independence model overstates the pure premium by 3.8% on average. That is not uniform: the 25th percentile is 0.943, meaning a quarter of the book is being overstated by more than 5.7%.

The tail is where the commercial impact concentrates. High-frequency policyholders -- the top decile by mu_n -- have lower average correction factors than the rest of the book. The independence model prices them most aggressively wrong.

The correction factor is analytic, not simulated. Scoring at policy level costs nothing extra at runtime.

---

## Small books: ConditionalFreqSev

Stable omega estimation requires roughly 20,000 policyholder-years and at least 2,000 claims. On smaller books the confidence interval on omega will be wide and the correction factor will be unreliable.

For books under 10,000 policies, use the Garrido conditional approach instead. This adds claim count N as a covariate in the severity GLM -- no copula, just one extra GLM parameter -- which is estimable on much thinner data.

```python
from insurance_frequency_severity import ConditionalFreqSev

cond_model = ConditionalFreqSev(freq_glm=freq_glm, sev_glm_base=sev_glm)
cond_model.fit(claims_df, n_col="claim_count", s_col="avg_severity")

corrections = cond_model.premium_correction()
print(corrections["correction_factor"].describe())
```

The correction factor here uses exp(gamma x E[N|x]) where gamma is the severity-on-count regression coefficient. It is less flexible than the Sarmanov form but stable on 500 claims, where omega estimation would produce noise.

---

## Comparing copula families

The library also fits a Gaussian copula (with the PIT approximation) and FGM for comparison. Run `compare_copulas` to see which family the AIC prefers on your data:

```python
from insurance_frequency_severity import compare_copulas

comparison = compare_copulas(
    n=claim_count,
    s=avg_severity,
    freq_glm=freq_glm,
    sev_glm=sev_glm,
)
print(comparison)
```

```
     copula      omega_or_rho      aic        bic
0  sarmanov            -0.41  48203.1    48210.8
1  gaussian            -0.13  48231.6    48239.3
2       fgm            -0.09  48247.2    48254.9
3   indep.              0.00  48251.4    48251.4
```

Sarmanov wins on AIC for most UK motor data because it handles the discrete frequency margin correctly. The Gaussian rho of -0.13 is informative for communicating the result to a pricing review -- rho is a familiar quantity -- but you should fit and price with the Sarmanov form.

---

## What this is not

This is not a replacement for your frequency GLM or severity GLM. Both models fit exactly as before. The Sarmanov stage is a one-parameter correction to the pure premium calculation. Your GLM coefficients, goodness-of-fit diagnostics, and peer review documentation are unchanged.

This is also not the same as the neural shared-trunk approach in [`insurance-dependent-fs`](https://github.com/burning-cost/insurance-dependent-fs). That library trains a single network with a shared encoder trunk and two output heads (Poisson frequency, Gamma severity), which captures more complex dependence structures but requires several thousand claiming policies to train reliably and produces a black-box that is harder to audit. The Sarmanov approach adds one interpretable parameter (omega) to your existing GLM pipeline. For most UK motor books that need to demonstrate the pricing logic to a committee or to the FCA, one interpretable parameter is the right answer.

Use the neural approach when you have 50,000+ claiming observations and genuinely complex dependence structure. Use the Sarmanov approach when you want to fix the independence assumption in your existing two-part GLM without rebuilding the pipeline.

---

## Before you apply the correction

Three checks before putting this in production.

First, run the `DependenceTest` and confirm the p-value warrants action. A Spearman rho of -0.03 with p = 0.4 does not need a copula. The independence model is defensible there.

Second, check the delta AIC from `model.dependence_summary()`. Delta AIC below 4 means the evidence is ambiguous; present both estimates to your pricing committee and let them decide whether the governance overhead of the correction is justified.

Third, check that the correction factor distribution is sensible. Values outside 0.85-1.15 on a standard UK motor book are a warning sign that something has gone wrong in the omega estimation -- possibly a data quality issue in how claiming-policy pairs are constructed.

---

## We think the independence assumption is a free lunch that most teams are leaving on the table

Copula modelling for frequency-severity dependence has been in the actuarial literature since Garrido, Genest, and Schulz (2016). The reason most UK pricing teams have not adopted it is not scepticism about the theory. It is that the implementation path through standard statistical software was painful: getting a discrete-continuous Sarmanov distribution fitted with IFM estimation into a production pricing pipeline required bespoke code that most actuarial teams do not have bandwidth to write.

`insurance-frequency-severity` makes this a two-line addition to your existing GLM pipeline. The correction is analytic, so runtime scoring is unaffected. The output is an interpretable omega parameter that you can defend in a committee paper. The test suite gives you the diagnostic to confirm whether the correction is warranted on your data before you commit to it.

On a 50,000-policy UK motor book with typical NCD structure, a 3-5% pure premium correction concentrated in the high-frequency tail is real money. It is also a systematic bias rather than noise, which means it will show up in your A/E ratios if you look for it.

---

`insurance-frequency-severity` is open source under MIT at [github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). Install with `uv add insurance-frequency-severity`. Requires Python 3.10+, statsmodels, NumPy, and SciPy.

- [Frequency-Severity Dependence in UK Motor: A Shared-Trunk Neural Architecture](/2026/03/13/insurance-dependent-fs/) -- the neural two-part model for teams with larger datasets and more complex dependence structures
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/03/04/per-risk-volatility-scoring-with-distributional-gbms/) -- when you want the full predictive distribution, not just the corrected mean
- [Vine Copulas for Multi-Peril Home](/2026/03/12/insurance-copula/) -- the right copula framework when the dependence is between perils rather than between frequency and severity
