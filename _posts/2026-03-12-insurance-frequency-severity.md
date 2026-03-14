---
layout: post
title: "Two GLMs, One Unjustified Assumption"
date: 2026-03-12
categories: [libraries]
tags: [frequency-severity, Sarmanov, copula, GLM, IFM, motor, pricing, dependence, independence-test, premium-correction]
description: "Every personal lines pricing model makes the same assumption: frequency and severity are independent, conditional on rating factors. Nobody tests this. insurance-frequency-severity does — Sarmanov copula, IFM estimation, analytical premium correction, Garrido conditional approach."
post_number: 74
---

Every personal lines pricing model in the UK makes the same assumption. Two GLMs — one for frequency, one for severity — fitted separately, then multiplied together. Pure premium = E[N|x] × E[S|x]. It is clean, tractable, and taught in every actuarial pricing course.

It also assumes that claim frequency and claim severity are independent, conditional on the rating factors. Nobody tests this. We have spoken to pricing teams at several UK insurers and the answer is almost always the same: "we know it's probably not exactly right, but we don't have a way to measure it."

Now there is.

We built `insurance-frequency-severity` to quantify the frequency-severity dependence in UK personal lines, correct premiums for it, and test whether your data supports the independence assumption at all.

## Why independence is almost certainly wrong

The academic evidence is striking in its consistency. Every published study that has looked at this — US auto, Singapore auto, Spanish auto, French auto, US property — finds the same thing: negative frequency-severity correlation. Policyholders who have more claims tend to have smaller claims on average.

The mechanism in UK motor is particularly well understood. No-claims discount is the largest single rating factor in most UK motor books. Policyholders are acutely aware of this. Small at-fault claims — a scraped bumper, a minor rear-end — often go unreported because the premium impact of losing NCD exceeds the claim amount. What gets reported is disproportionately large: the total write-offs, the serious injuries, the claims where suppression is not economically rational. So policyholders with high claim frequency tend to be people who genuinely cannot suppress (high-risk drivers with serial bad luck), and their individual claim amounts look smaller because the selection is different from the suppressed population.

The standard two-GLM framework treats this as if it were not happening.

Vernic, Bolancé, and Alemany (2022, *Insurance: Mathematics and Economics*) quantified the effect on 99,972 Spanish auto policyholders using a Sarmanov bivariate model. Ignoring frequency-severity dependence understated the risk premium by €5.69 to €12.51+ per policyholder, depending on risk profile. On a book of 100,000 policyholders, that is between €570,000 and €1.25 million+ in systematic underpricing — on a Spanish book at 2022 premium levels. The directional implication for UK motor is clear even if the exact magnitude is not portable. This is not a rounding error — it is a structural bias that compounds across every renewal cycle.

## Why Sarmanov, not Gaussian copula

Most pricing actuaries who have heard of copulas think of Gaussian copulas. They are well understood, implemented everywhere, and give you full Spearman rho coverage of [−1, 1].

The problem is discrete margins. Sklar's theorem — the foundation of copula theory — guarantees that the copula representation of a joint distribution is unique only when both margins are continuous. Claim count N is discrete. When you force a Gaussian copula onto discrete-continuous data, you need a probability integral transform (PIT) approximation at the mass points. This introduces identification noise: the model is slightly misspecified by construction, and the copula parameter absorbs some of that misspecification.

Sarmanov is different. It is defined directly on the product space of the marginal distributions, with no need for an intermediate PIT step. The joint density is:

```
f(n, s) = f_N(n) × f_S(s) × [1 + ω × φ_N(n) × φ_S(s)]
```

where φ_N and φ_S are bounded kernel functions (we use Laplace kernels). When ω = 0, you recover independence exactly. The Spearman rho range with Laplace kernels is [−3/4, 3/4] — substantially wider than FGM's [−1/3, 1/3], which cannot model even moderate dependence in practice.

This is also the copula family that Lee and Shi (2019, *IME*) used on Singapore auto data, that Vernic et al. (2022) used on Spanish auto, and that Blier-Wong (2026, arXiv) analyses for aggregate loss moments. There is a body of literature behind it specifically in the insurance frequency-severity setting.

## What we built

```bash
pip install insurance-frequency-severity
```

The library has four modules: copula classes, joint models, diagnostics, and reporting.

**Start by testing whether independence holds at all:**

```python
from insurance_frequency_severity import DependenceTest

test = DependenceTest(n_claims=df["claim_count"], severity=df["avg_severity"])
result = test.fit()
print(result.summary())
# Kendall tau: -0.073, p-value: 0.002
# Spearman rho: -0.108, p-value: 0.001
# Permutation p-value: 0.003
# Conclusion: reject H0 of independence
```

If you cannot reject independence, stop here. The correction will be small and the estimation uncertainty high. The library tells you this.

**Fit the joint model against your existing GLMs:**

```python
from insurance_frequency_severity import JointFreqSev

# freq_glm and sev_glm are your existing statsmodels GLM results
joint = JointFreqSev(freq_glm=freq_glm, sev_glm=sev_glm, copula="sarmanov")
result = joint.fit(df)

print(result.omega)          # -0.41
print(result.omega_ci_95)    # (-0.61, -0.22)
print(result.aic)
```

The key design decision here is that we accept your fitted GLMs rather than re-estimating the marginals. Pricing teams have their own model pipelines; we do not want to interfere with rating factor selection, regularisation choices, or validation splits. IFM (inference functions for margins) treats the marginal parameters as fixed and estimates ω alone via profile likelihood.

**Get per-policy premium corrections:**

```python
corrections = result.premium_correction(df)
# Returns a Series of multiplicative factors, one per policy
# 1.0 = no correction; <1.0 = overpriced under independence; >1.0 = underpriced

print(corrections.describe())
# mean    0.982
# min     0.951
# max     1.014
```

The correction is analytical — it uses MGF derivatives, not Monte Carlo. For Sarmanov with Laplace kernels, the correction factor has a closed form. This matters in production: you can apply it to a full book of policies in milliseconds.

**Compare against the Gaussian and FGM alternatives:**

```python
from insurance_frequency_severity import compare_copulas

table = compare_copulas(freq_glm, sev_glm, df)
print(table)
#              AIC       BIC    omega   spearman_rho
# sarmanov  -12847.3  -12839.1  -0.41         -0.31
# gaussian  -12841.2  -12833.0  -0.38         -0.28
# fgm       -12803.7  -12795.5  -0.33         -0.11
```

**Or use the simpler Garrido (2016) approach if you prefer no copulas:**

```python
from insurance_frequency_severity import ConditionalFreqSev

cond = ConditionalFreqSev(freq_glm=freq_glm)
cond.fit(df)
# Refits severity GLM with N as covariate
# Under Poisson + log-link: premium correction = exp(gamma * E[N|x])
print(cond.gamma_coef)   # -0.18 (negative = fewer claims -> larger individual claim)
```

Garrido's method requires no copula — it just adds claim count as a covariate in the severity GLM and reads off a correction. It is the lowest-friction entry point. If the gamma coefficient is not significantly different from zero, you have more evidence for independence.

## Honest limitations

**Sample size.** The library warns if you have fewer than 1,000 policies or fewer than 500 claims. Estimating ω on small samples produces wide confidence intervals. The correction factors will be imprecise. On a large affinity book this is fine; on a small commercial lines segment, treat the output as directional not operational.

**Post-GIPP dynamics.** The Financial Conduct Authority's general insurance pricing practices rules (effective January 2022) ended most forms of price walking and changed how NCD interacts with renewal pricing. It is plausible that NCD claim suppression behaviour has shifted as a result. We do not have post-GIPP empirical data on UK frequency-severity correlations. The Spanish and Singaporean estimates remain the best available benchmarks, but UK-specific validation matters.

**GLM interface.** The library currently works with `statsmodels` GLM result objects. Scikit-learn pipelines and EMBLEM/Radar exports are not yet supported. If your pricing is done entirely in vendor tooling, you will need to extract the fitted parameters manually.

**IFM is not full MLE.** Treating marginal parameters as fixed is efficient but not jointly optimal. If your marginals are misspecified — wrong link function, missing interactions — the ω estimate absorbs some of that misspecification. The diagnostics module's Rosenblatt PIT test (`CopulaGOF`) will catch gross copula misspecification, but not marginal misspecification.

## What to actually do with this

Run `DependenceTest` first. If independence is not rejected at the 5% level, put the library away and come back with more data.

If independence is rejected, fit `JointFreqSev` with Sarmanov, run `compare_copulas()`, and look at the correction histogram from `JointModelReport`. If the mean correction is within ±1% and the tails are within ±3%, the effect is real but operationally small. If corrections are running to ±5% or wider, you have a pricing bias worth addressing.

The two places this is most likely to matter in UK personal lines: motor (via NCD suppression) and home contents (via excess suppression, which operates through a similar mechanism). Commercial lines have more complex reporting dynamics and the direction of correlation is less predictable.

The independence assumption has been sitting unchecked at the centre of UK personal lines pricing for as long as two-part GLMs have been the standard. It is not always consequential. But the tools to find out whether it matters for your book now exist, take an afternoon to run, and cost nothing.

---

**install:** `pip install insurance-frequency-severity`
**source:** [github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity)
**key refs:** Garrido, Genest & Schulz (2016 IME) · Lee & Shi (2019 IME) · Vernic, Bolancé & Alemany (2022 IME) · Blier-Wong (2026 arXiv)

---

**Related reading:**
- [Frequency-Severity Dependence in UK Motor: A Shared-Trunk Neural Architecture](/2026/03/13/insurance-dependent-fs/) — when frequency and severity are correlated and independence is the wrong assumption to bake into the architecture
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — the alternative: model the full distribution jointly rather than splitting into two GLMs
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — distribution-free coverage guarantees on the pure premium, regardless of whether your frequency-severity factorisation is correct
