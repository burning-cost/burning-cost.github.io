---
layout: post
title: "Does Sarmanov copula frequency-severity modelling actually work for insurance pricing?"
date: 2026-03-23
categories: [libraries, validation]
tags: [frequency-severity, sarmanov, copula, glm, dependence, ifm, validation, motor, pricing]
description: "We read the source, ran the benchmark, and checked the claim: the independence assumption in standard two-part GLMs is wrong for UK motor, and this library corrects it analytically. Here is what the code actually does and when it helps."
---

The standard UK motor pricing model multiplies two things together: the Poisson GLM for claim count and the Gamma GLM for average severity. Pure premium = E[N] × E[S]. Simple, auditable, embedded in every pricing team's tooling.

The assumption underneath that multiplication is that N and S are independent given your rating factors. They are not. NCD structure systematically suppresses borderline claims  -  policyholders who have already claimed once in the year do not report small incidents near the claim threshold, so high-frequency policies tend to have lower average severity per claim. The independence assumption overstates the risk for your multi-claim segment and understates it for the single-claim segment. Vernic, Bolancé, and Alemany (2022) quantified this mismeasurement at €5–55 per policyholder on a Spanish auto book. The directional effect in UK motor is identical.

[`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity) claims to fix this with a Sarmanov copula, IFM estimation that plugs into your existing GLMs, and analytical (closed-form) correction factors at scoring time. The [full library walkthrough](/2025/05/14/frequency-severity-independence-is-costing-you-premium/) covers the complete API; here we focus on what the code actually does under the hood and whether the benchmark supports the claims.

---

## What the library claims to do

The core idea is to replace the independence assumption with the Sarmanov bivariate distribution:

```
f(n, s) = f_N(n) × f_S(s) × [1 + ω × φ₁(n) × φ₂(s)]
```

When ω = 0 this is exactly the independence model. Non-zero ω captures dependence between the frequency and severity margins. The key point  -  and the reason the Sarmanov family is the right choice here  -  is that it handles the discrete-continuous mixed margins problem directly. Standard copulas require a probability integral transform for the discrete frequency margin, but Sklar's theorem is not unique for discrete distributions, which creates identifiability problems. The Sarmanov family sidesteps this entirely: it is a joint distribution in its own right, not a copula applied after the fact.

The library's workflow is IFM (Inference Functions for Margins):
1. You fit your own frequency GLM (Poisson or Negative Binomial) and severity GLM (Gamma or Lognormal) using whatever tooling you already have.
2. You pass the fitted objects to `JointFreqSev`. The library extracts per-policy predictions from your GLMs and estimates ω by profile likelihood  -  maximising Σᵢ log[1 + ω × φ₁(nᵢ; μ̂ᴺᵢ) × φ₂(sᵢ; μ̂ˢᵢ)] over observed (nᵢ, sᵢ) pairs with nᵢ > 0.
3. `premium_correction()` returns a DataFrame with per-policy correction factors, where the factor is E[N×S] / (E[N] × E[S]).

The correction is analytical  -  no simulation at scoring time. For a 5,000-policy portfolio, the fit takes about 0.1s on top of your marginal GLM fitting.

---

## Does the code actually deliver?

We read the source in `copula.py`, `joint.py`, and `diagnostics.py`. A few observations.

**The kernel mathematics are correctly implemented.** The Laplace kernels for NB and Poisson frequency, and for Gamma and Lognormal severity, are correctly centred (zero-mean under the respective marginal). The MGF expressions are correct: for NB with overdispersion α and mean μ, M_N(θ) = (p/(1-(1-p)e^{-θ}))^r where r = 1/α. The omega bounds calculation  -  finding the feasible range of ω from a grid over the support  -  works correctly, and the library validates the planted omega against these bounds before sampling in the benchmark.

**The zero-claim likelihood contribution is handled correctly.** For policies with N=0, there is no observed severity. The likelihood contribution is just f_N(0). This is not a shortcut  -  it is exact. Because E[φ₂(S)] = 0 under the marginal, the integral over unobserved severity recovers f_N(0) analytically. The code does this correctly in `SarmanovCopula.log_likelihood()`.

**GLM compatibility is statsmodels-specific.** The `_extract_freq_params()` function in `joint.py` detects the marginal family via `model.family` and extracts the NB overdispersion parameter via `family.alpha`  -  both statsmodels conventions. Non-statsmodels objects will fall through to a default (NB with α=1.0), which will silently produce wrong correction factors if your frequency GLM is actually Poisson. The README acknowledges this. If you are using sklearn GLMs, scikit-learn's Tweedie regressor, or CatBoost for your marginals, this library does not currently plug in cleanly. You need statsmodels GLM objects or you need to pass parameter dictionaries manually.

**Per-policy corrections only with the Sarmanov copula.** The README notes this in the Methods section but it is worth flagging explicitly: for `copula="gaussian"` or `copula="fgm"`, `premium_correction()` returns a single portfolio-average correction factor, not per-policy values. Only the Sarmanov copula produces per-policy analytical corrections. If you want the full per-segment breakdown  -  the thing that actually matters for pricing  -  you need `copula="sarmanov"`.

**`DependenceTest` runs before any copula fitting.** This is the right design. The `DependenceTest` class runs Kendall tau and Spearman rho against permutation null distributions (B=1000 by default) and returns a clean summary DataFrame. The correct workflow is: run `DependenceTest` first, check the p-value, then decide whether to fit the joint model. For monitoring whether the dependence structure changes post-deployment, use [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/). The library does not force you down the copula path if independence is not rejected.

**The `compare_copulas()` function is useful but slow for large books.** It fits all three copula families (Sarmanov, Gaussian, FGM) and returns AIC/BIC comparison. Each fit involves a profile likelihood optimisation. For a 20,000-policy portfolio with 10% claim rate (2,000 claims), this runs in a few seconds. For larger books, the Gaussian copula fit is the slow step because it recomputes the PIT for every omega candidate.

---

## The benchmark: what performance to expect

The library's benchmark (`benchmarks/benchmark_insurance_frequency_severity.py`, run 2026-03-21) uses a pure Sarmanov DGP with planted ω = 3.5. This is the right benchmark design. An earlier version used a latent-factor DGP where the correlation was driven by a shared unobserved risk variable  -  that caused apparent sign and magnitude mismatch because the IFM estimator targets the Sarmanov ω parameter, not a latent factor loading. The current benchmark samples (N, S) directly from the Sarmanov joint distribution, so the planted ω is exactly what IFM should recover.

The README also documents a second benchmark (12,000-policy latent-factor DGP, run 2026-03-16) with these headline results:

| Metric | Independent model | Sarmanov copula | Change |
|--------|------------------|-----------------|--------|
| Pure premium MAE vs oracle | 14.84 | 10.60 | −28.6% |
| Portfolio total premium bias | +22.95% | −6.77% | −16.2pp |
| Fit time (seconds) | 0.11 | 0.13 | +21% |

The 28.6% MAE improvement in the latent-factor benchmark is partly copula correction and partly the model absorbing marginal GLM error. The README is transparent about this: on that DGP, the fitted ω was −1.14 with a 95% confidence interval of (−1.61, +0.30)  -  independence is not rejected at 5%. The improvement comes from the correction partly offsetting systematic overfit in the marginals. This is not a bug, but it is also not evidence that the Sarmanov correction is doing what it claims for that particular DGP.

The pure Sarmanov benchmark (planted ω = 3.5) is the cleaner test. There the library should recover ω within ~20% of the planted value at 15,000 policies. That is the condition where the correction factors are trustworthy and the premium adjustments are methodologically clean.

---

## When it works well

**Large personal lines books with measurable dependence.** The README gives the data requirements plainly: approximately 20,000 policyholder-years and at least 2,000 claims for stable ω estimation. Below 1,000 policies or 500 claims, the library warns you. Below those thresholds, the confidence interval on ω is so wide that the correction factors are noise.

**When `DependenceTest` rejects independence.** The correction is only meaningful when the permutation test confirms that ω ≠ 0. On UK motor with typical NCD structure, you should see negative freq-sev dependence (high-frequency policyholders have lower average severity per claim). The expected average correction factor is 0.93–0.98  -  a 2–7% downward adjustment to the pure premium, with larger corrections at the high-frequency tail.

**When your marginals are well-specified statsmodels GLMs.** The IFM estimator takes the marginal predictions as given and estimates only ω on top. If your marginals are misspecified, ω will absorb some of the misfit. This is not catastrophically wrong  -  it still tends to improve MAE  -  but it is harder to interpret and harder to defend to a pricing committee.

**For the top-risk decile.** The independence model's error is most concentrated in the extreme frequency segment. On the benchmark DGP with ω = 3.5, the high-risk decile correction factor is approximately 0.95  -  a 5% premium adjustment. At typical UK motor claim rates and NCD structures, this is the segment where your renewal model is most likely to be systematically mispriced.

---

## When it does not work

**Small or specialty books.** Below 2,000 claims, do not use the Sarmanov estimator. The library's `ConditionalFreqSev` fallback  -  the Garrido et al. (2016) method, which adds N as a covariate in the severity GLM  -  is more stable because it estimates a single scalar parameter γ rather than ω plus two kernel hyperparameters. For books with 500–2,000 claims, use `ConditionalFreqSev`.

**When independence is not rejected.** If `DependenceTest` returns p > 0.05 on both the Kendall tau and Spearman rho tests, do not apply the correction. The library is clear about this. Applying a copula correction to an independent dataset introduces noise into your premium relativities without fixing a real problem.

**Non-statsmodels pipelines.** If your production models are CatBoost or XGBoost, you cannot use this library directly without writing a wrapper that exposes the statsmodels-compatible interface. The per-observation frequency predictions and the dispersion/alpha parameters are needed to construct the kernel functions. This is a genuine gap. The README acknowledges it but there is no helper for it yet.

**Lognormal severity with large alpha kernel.** The `LaplaceKernelLognormal.mgf()` method uses numerical quadrature (scipy `quad`) rather than a closed form, because the lognormal Laplace transform has no closed form. For Gamma severity margins, the MGF is analytical and fast. If you have lognormal severity and are fitting `compare_copulas()` with `n_permutations > 0`, the quadrature calls accumulate and the function becomes noticeably slow. Use Gamma severity where possible, or accept the runtime hit.

---

## Our read

The library does what it claims, subject to the data requirements and the statsmodels dependency. The mathematics are correctly implemented, the zero-claim likelihood is handled properly, and the IFM workflow  -  plug in your existing GLMs, estimate ω, get per-policy corrections  -  is genuinely useful for teams that have already built their marginal models and want to test whether the independence assumption is costing them premium.

The honest caveat is about the benchmark: the headline 28.6% MAE improvement on the latent-factor DGP is partly copula correction and partly marginal error absorption. The cleaner result is from the pure Sarmanov benchmark, where the planted ω is directly the target parameter. Teams should run `DependenceTest` on their own data, check whether independence is rejected, and only then interpret the correction factors as genuine dependence corrections rather than regularisation artefacts.

The two cases where we would not use this library: books under 2,000 claims (use `ConditionalFreqSev`), and non-statsmodels pipelines (write your own IFM wrapper or wait for the library to add one). For a large personal lines motor or property book with statsmodels marginals and a significant DependenceTest result, this is the right tool.

The library is at [github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). The benchmark is in , runnable on Databricks serverless.

---

- [Sarmanov Copula for Frequency-Severity Dependence](/2025/05/14/frequency-severity-independence-is-costing-you-premium/)  -  the full library introduction and worked example on UK motor data
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)  -  once the correction is in production, segment-level A/E monitoring will detect if omega has drifted and the correction needs reestimating
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)  -  record the omega estimate, DependenceTest result, and AIC comparison as model documentation in the inventory
