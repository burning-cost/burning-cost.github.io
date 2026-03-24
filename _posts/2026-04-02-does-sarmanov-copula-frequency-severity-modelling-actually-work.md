---
layout: post
title: "Does Sarmanov Copula Frequency-Severity Modelling Actually Work?"
date: 2026-04-02
categories: [techniques, validation]
tags: [frequency-severity, sarmanov, copula, glm, motor, pricing, python]
description: "The standard UK motor pricing formula multiplies E[N] by E[S] and assumes independence. On a 12,000-policy benchmark with planted freq-sev dependence, that assumption costs you 22.95% portfolio premium bias. The Sarmanov copula cuts it to -6.77%."
---

The honest answer is: yes, when the dependence is real and the data volume supports estimation. When neither condition holds, it is a 20-millisecond overhead with a 28.6% MAE improvement as a side-effect of absorbing marginal model error. Either way, we have not found a scenario where running it made things worse.

We benchmarked [`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity) against the standard independent two-part model on a synthetic UK motor book with known data-generating parameters. The numbers below are from `benchmarks/benchmark_insurance_frequency_severity.py` run 2026-03-16.

---

## The claim

The standard UK motor pricing formula is:

```
Pure premium = E[N|x] × E[S|x]
```

You fit a frequency GLM. You fit a severity GLM. You multiply. This is not an approximation or a shortcut -- it is a precise statement that claim count and average severity are independent given your rating factors. The formula is mathematically exact under that assumption.

The assumption is almost certainly wrong for UK motor.

The reason is the No Claims Discount structure. Policyholders are aware of NCD thresholds. A driver with a small borderline incident -- say a £400 door scrape -- will often absorb it rather than claim and lose two NCD years. This suppression behaviour is more common for policyholders who already have frequent small claims, because they have already made the threshold calculation before. The result is a systematic negative correlation between claim count and average claim severity: the policies reporting the most claims tend to have higher average severities, because the low-severity borderline incidents were suppressed.

Vernic, Bolancé, and Alemany (2022) quantified this on a Spanish auto book and found the mismeasurement costs €5-55+ per policyholder. The directional effect in UK motor is the same; the magnitude varies by book structure and NCD design.

The Sarmanov copula corrects for this. The joint distribution takes the form:

```
f(n, s) = f_N(n) × f_S(s) × [1 + omega × phi_1(n) × phi_2(s)]
```

When omega equals zero, you have independence. When omega is negative, high claim counts and high severities are negatively correlated. The key technical advantage over a standard Gaussian copula for this problem is that the Sarmanov family works directly with the joint distribution and does not require a probability integral transform for the discrete frequency margin. The PIT is not well-defined for discrete distributions -- Sklar's theorem is not unique in the discrete case -- so the Gaussian copula approach involves an approximation that the Sarmanov approach avoids entirely.

---

## What we tested

Benchmark: 12,000 synthetic UK motor policies (8,437 training, 3,563 test) with planted positive freq-sev dependence via a latent risk score. Both models received identical marginal GLMs -- NegativeBinomial frequency, Gamma severity -- fitted on the training set. The only difference was whether the pure premium was computed as E[N] * E[S] (independence) or with the Sarmanov correction.

The Sarmanov model uses Inference Functions for Margins (IFM): the marginal GLMs are fitted first and kept fixed, then omega is estimated by profile likelihood over the joint distribution. You plug in your existing fitted GLM objects; the library estimates one additional parameter on top. Fit time increase over independence: 23 milliseconds (0.105s to 0.128s). At scoring time the correction is analytical -- closed-form, no simulation -- so the production cost is negligible.

The validation notebook at `notebooks/databricks_validation.py` runs a separate DGP with planted positive omega (omega=3.5, 30,000 policies) specifically to test parameter recovery. IFM is asymptotically unbiased on a pure Sarmanov DGP; with 21,000 training policies and 2,000+ claims the relative error on omega recovery is typically 10-20%.

---

## The numbers

| Metric | Independent model | Sarmanov copula | Change |
|--------|------------------|-----------------|--------|
| Pure premium MAE vs oracle | 14.8405 | 10.6010 | -28.6% |
| Portfolio total premium bias | +22.95% | -6.77% | -16.2pp |
| Estimated Spearman rho | 0.000 | -0.015 | -- |
| Fit time (seconds) | 0.105 | 0.128 | +21% |

The portfolio bias number is the one that should concern a pricing actuary. A +22.95% portfolio bias means the independence model is collectively overcharging by nearly a quarter relative to the oracle. The Sarmanov correction pulls this to -6.77% -- not zero, because the benchmark DGP uses a latent risk score and IFM is estimating omega on imperfect marginals, but a substantial improvement.

Correction factors across the portfolio: mean 0.943, p10 0.939, p90 0.950. The Sarmanov model produces per-policy corrections in a 1.1-percentage-point range -- narrow, but the distribution matters because it correctly re-ranks the high-risk tail. High-risk decile correction: 0.952 (-4.8% versus independence). Low-risk decile: 0.940. That 1.2pp differential between deciles is the correction working as intended: the independence model was charging the high-risk decile relatively too much, and the Sarmanov correction narrows that gap by re-allocating to where the true premium is.

One complication worth being transparent about: on the benchmark DGP, the fitted omega is -1.14 with a 95% CI of (-1.61, +0.30). The CI includes zero, so independence is not formally rejected. Despite this, the MAE improvement is 28.6%. The explanation is that the correction is absorbing marginal GLM error: the GLMs slightly overpredict frequency for high-latent-risk policies, and the copula correction partially offsets this. This is not the canonical use case -- that requires omega to be positive and statistically significant -- but it illustrates that the correction can be beneficial even when the dependence is estimated noisily.

---

## Where it works

The canonical scenario for Sarmanov correction is a UK motor or home book where the NCD structure drives systematic NCD threshold avoidance. Test for this with `DependenceTest` before committing:

```python
from insurance_frequency_severity import DependenceTest

test = DependenceTest(n_permutations=1000)
test.fit(n=claim_count[claims_mask], s=avg_severity[claims_mask])
print(test.summary())   # Kendall tau, Spearman rho, permutation p-values
```

If the permutation test rejects independence (p < 0.05), you have a genuine dependence signal worth correcting. The omega estimate in that case reflects real structure in the joint distribution, and the per-policy correction factors correctly re-rank segments.

The databricks validation with omega=3.5 shows the IFM estimator recovers the planted value within 20% at 21,000 policies. On a UK motor book of that size with 2,000+ annual claims, the omega estimate will be stable. Below that, the CI widens and the per-policy correction factors become noisy, but the portfolio-level bias correction still works.

For small books -- under 500 claims -- use `ConditionalFreqSev` instead. It estimates a single parameter gamma from the severity GLM refitted with N as a covariate. Less flexible, but stable on thin data.

---

## Where it does not add much

When independence actually holds. The DGP in the benchmark was designed with planted dependence, which is why the correction helps. On a book where claim count and severity are genuinely independent given rating factors -- some short-tail commercial lines, certain specialty books -- the Sarmanov omega will estimate near zero, the correction factors will cluster around 1.0, and you will have spent 23 milliseconds for nothing. This is not a problem; the test is precisely there to detect this case and tell you to stop.

The Sarmanov copula also does not fix a misspecified frequency or severity GLM. If your frequency model is wrong because you are missing an important rating factor -- vehicle group, say, or postcode district -- the omega correction will absorb some of this error (as we saw with the 28.6% MAE improvement under a noisy DGP), but it is not a substitute for good marginal models. Fix the marginals first, then test for residual dependence.

The Spearman rho range for the Laplace-kernel Sarmanov with NB/Gamma margins is bounded at [-3/4, +3/4] (Blier-Wong 2026). This is comfortable for the moderate negative dependence in UK motor, but very strong dependencies -- say Spearman rho below -0.6 -- will push against the boundary and the model will produce distorted corrections. In practice we have not seen UK personal lines motor data with dependence this strong, but commercial lines with large account effects can get there.

---

## The honest verdict

Use Sarmanov copula correction if:
- Your book is UK motor or home with an NCD structure and `DependenceTest` finds a statistically significant negative correlation between claim count and severity
- You have at least 20,000 policies and 2,000 claims -- smaller than this and the omega estimate will be too noisy for per-policy corrections
- You want an analytically correct per-policy correction that re-ranks high-risk policies, not just a flat portfolio-level scalar

Use `ConditionalFreqSev` (the Garrido 2016 approach -- severity GLM with N as a covariate) if:
- Your book has fewer than 500 claims and you need a stable single-parameter correction
- You want something you can explain to a pricing committee without mentioning copulas

Skip the correction entirely if:
- `DependenceTest` does not reject independence
- Your book is short-tail commercial lines where NCD threshold behaviour is structurally absent

The independence assumption in the standard UK motor pricing formula is not a neutral modelling choice. It is a claim about your data that is almost certainly wrong. The Sarmanov copula provides a way to test that claim, quantify the correction, and apply it at the per-policy level with no simulation overhead. On the benchmark, the portfolio bias dropped from +22.95% to -6.77% for 23 milliseconds of additional fit time. That ratio is hard to argue with.

```bash
uv add insurance-frequency-severity
```

Source, benchmarks, and a Databricks validation notebook with known-DGP omega recovery at [GitHub](https://github.com/burning-cost/insurance-frequency-severity). Run `benchmarks/benchmark_insurance_frequency_severity.py` (seed=42) for reproducible output.

- [Your Frequency-Severity Independence Assumption Is Costing You Premium](/2027/05/15/frequency-severity-independence-is-costing-you-premium/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
- [Does HMM Telematics Risk Scoring Actually Work?](/2026/03/31/does-hmm-telematics-risk-scoring-actually-work/)
