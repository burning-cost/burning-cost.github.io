---
layout: post
title: "Does Sarmanov Copula Frequency-Severity Modelling Actually Work?"
date: 2026-03-28
categories: [techniques, validation]
tags: [frequency-severity, sarmanov, copula, glm, motor, pricing, python]
description: "The standard UK motor pricing formula multiplies E[N] by E[S] and assumes independence. On a 15,000-policy benchmark with planted omega=3.5, that assumption understates portfolio premium by 7-9%. The Sarmanov copula cuts this to 1-3%."
---

The honest answer is: yes, when the dependence is real and the data volume supports estimation. The IFM estimator recovers planted omega within 10% on 15,000 policies with correctly specified marginals, reduces portfolio-level bias from ~8% to ~2%, and applies a 10-15% upward correction to the highest-risk decile that the independence model ignores entirely.

We benchmarked [`insurance-frequency-severity`](/insurance-frequency-severity/) against the standard independent two-part model on a synthetic UK motor book with known data-generating parameters. The numbers below are from `benchmarks/benchmark_insurance_frequency_severity.py` — a pure Sarmanov DGP where the planted omega equals the population parameter the IFM estimator targets directly.

---

## The claim

The standard UK motor pricing formula is:

```
Pure premium = E[N|x] × E[S|x]
```

You fit a frequency GLM. You fit a severity GLM. You multiply. This is not an approximation or a shortcut -- it is a precise statement that claim count and average severity are independent given your rating factors. The formula is mathematically exact under that assumption.

The assumption breaks down in two opposite directions depending on the line of business.

For personal motor with a multi-step NCD structure, policyholders absorb small borderline claims rather than lose discount years. The policies reporting the most claims tend to have higher average severities, because the low-severity incidents were suppressed. This creates negative freq-sev correlation.

For fleet and commercial motor, the causation runs the other way: the same risk profile that generates frequent claims -- young drivers, high mileage, urban routes -- also generates more serious ones. High claim count and high average severity co-occur in the same high-risk policyholders. This creates positive freq-sev correlation.

Vernic, Bolancé, and Alemany (2022) quantified the personal lines effect on a Spanish auto book and found the mismeasurement costs €5-55+ per policyholder. The sign and magnitude vary by book structure and NCD design.

The Sarmanov copula handles both directions. The joint distribution takes the form:

```
f(n, s) = f_N(n) × f_S(s) × [1 + omega × phi_1(n) × phi_2(s)]
```

When omega equals zero, you have independence. When omega is positive, high claim counts and high severities are positively correlated. When omega is negative, they are negatively correlated. The key technical advantage over a standard Gaussian copula for this problem is that the Sarmanov family works directly with the joint distribution and does not require a probability integral transform for the discrete frequency margin. The PIT is not well-defined for discrete distributions -- Sklar's theorem is not unique in the discrete case -- so the Gaussian copula approach involves an approximation that the Sarmanov approach avoids entirely.

---

## What we tested

Benchmark: 15,000 synthetic UK motor policies (70/30 train/test split) generated from a **pure Sarmanov DGP** with planted omega = 3.5 (positive frequency-severity dependence, representative of fleet and commercial motor where high-frequency policyholders also make larger claims). Both models received identical marginal GLMs -- Poisson frequency, Gamma severity -- fitted on the training set. The only difference was whether the pure premium was computed as E[N] * E[S] (independence) or with the Sarmanov correction.

Using a pure Sarmanov DGP matters. On a latent-factor DGP, the Sarmanov omega estimated by IFM does not correspond to any single planted parameter in the data-generating process, so you cannot evaluate parameter recovery. The pure Sarmanov DGP ensures the planted omega is exactly what IFM is designed to estimate.

The Sarmanov model uses Inference Functions for Margins (IFM): the marginal GLMs are fitted first and kept fixed, then omega is estimated by profile likelihood over the joint distribution. You plug in your existing fitted GLM objects; the library estimates one additional parameter on top. At scoring time the correction is analytical -- closed-form, no simulation -- so the production cost is negligible.

The oracle pure premium for each policy is E[N*S|x] under the planted Sarmanov DGP, computed via 200-draw Monte Carlo simulation per policy.

---

## The numbers

| Metric | Independence (standard two-part) | Sarmanov copula (JointFreqSev) |
|--------|------------------|-----------------|
| Omega recovery | 0 (assumed) | ~3.3-3.7 (planted 3.5, <10% error) |
| Overall MAE vs oracle | Baseline | ~15-25% lower |
| Total premium bias vs oracle | ~-7 to -9% (systematic undercharge) | ~-1 to -3% |
| Top decile correction factor | 1.00 (no correction) | ~1.10-1.15 |
| Bottom decile correction factor | 1.00 (no correction) | ~0.98-1.01 |
| Spearman rho (detected) | 0 assumed | ~0.25-0.35 |

The portfolio bias number is the one that should concern a pricing actuary. A -7 to -9% portfolio bias means the independence model is collectively *undercharging* relative to the oracle -- the independence assumption misses the premium loading needed to cover the positive freq-sev covariance in the highest-risk policies. The Sarmanov correction pulls this to -1 to -3%.

The top decile correction factor of ~1.10-1.15 is where the financial exposure sits. The policies in the highest-risk decile -- those with predicted frequency in the top 10% -- tend to also make larger claims. The independence model ignores this covariance and underprices them by 10-15%. The Sarmanov correction re-ranks the portfolio accordingly.

Omega recovery within 10% of the planted value confirms the IFM estimator is well-specified for this DGP. With correctly specified Poisson and Gamma marginals, the library finds the true population parameter reliably at 10,000 training policies.

---

## Where it works

The canonical scenario for Sarmanov correction is a fleet or commercial motor book where high-frequency and high-severity policyholders coincide systematically. Test for this with `DependenceTest` before committing:

```python
from insurance_frequency_severity import DependenceTest

test = DependenceTest(n_permutations=1000)
test.fit(n=claim_count[claims_mask], s=avg_severity[claims_mask])
print(test.summary())   # Kendall tau, Spearman rho, permutation p-values
```

If the permutation test rejects independence (p < 0.05), you have a genuine dependence signal worth correcting. The omega estimate in that case reflects real structure in the joint distribution, and the per-policy correction factors correctly re-rank segments.

The benchmark with omega=3.5 shows the IFM estimator recovers the planted value within 10% at 10,000 training policies with 2,000+ claims. On a UK motor book of that size, the omega estimate will be stable. Below that, the CI widens and the per-policy correction factors become noisy, but the portfolio-level bias correction still works.

For small books -- under 500 claims -- use `ConditionalFreqSev` instead. It estimates a single parameter from the severity GLM refitted with N as a covariate. Less flexible, but stable on thin data.

---

## Where it does not add much

When independence actually holds. On a book where claim count and severity are genuinely independent given rating factors -- some short-tail commercial lines, certain specialty books -- the Sarmanov omega will estimate near zero, the correction factors will cluster around 1.0, and you will have achieved nothing. The test is precisely there to detect this case.

The Sarmanov copula also does not fix a misspecified frequency or severity GLM. If your frequency model is wrong because you are missing an important rating factor -- vehicle group, say, or postcode district -- the omega correction will absorb some of this error, but it is not a substitute for good marginal models. Fix the marginals first, then test for residual dependence.

The Spearman rho range for the Laplace-kernel Sarmanov with NB/Gamma margins is bounded at [-3/4, +3/4] (Blier-Wong 2026). This is comfortable for the moderate dependence seen in most motor books, but very strong dependencies -- Spearman rho above 0.6 or below -0.6 -- will push against the boundary. Commercial lines with large account effects can get there, and the FGM copula (bounded at [-1/3, +1/3]) will fail even earlier. Run `compare_copulas()` if you are near these boundaries to check whether any family fits adequately.

---

## The honest verdict

Use Sarmanov copula correction if:
- Your book is fleet or commercial motor where `DependenceTest` finds a statistically significant positive correlation between claim count and severity, or personal motor where NCD suppression creates a statistically significant negative correlation
- You have at least 10,000 training policies and 2,000 claims -- smaller than this and the omega estimate will be too noisy for per-policy corrections
- You want an analytically correct per-policy correction that re-ranks high-risk policies, not just a flat portfolio-level scalar

Use `ConditionalFreqSev` (the Garrido, Genest & Schulz (2016) approach -- severity GLM with N as a covariate) if:
- Your book has fewer than 500 claims and you need a stable single-parameter correction
- You want something you can explain to a pricing committee without mentioning copulas

Skip the correction entirely if:
- `DependenceTest` does not reject independence
- Your book is short-tail commercial lines where neither NCD suppression nor fleet risk concentration applies

The independence assumption in the standard UK motor pricing formula is not a neutral modelling choice. It is a claim about your data that needs to be tested. The Sarmanov copula provides a way to test that claim, quantify the correction, and apply it at the per-policy level with no simulation overhead. On the benchmark with planted omega=3.5, portfolio bias dropped from ~8% to ~2% and the top-decile correction factor reached 1.10-1.15 -- the segment where mispricing matters most.

```bash
uv add insurance-frequency-severity
```

Source, benchmarks, and a Databricks validation notebook with known-DGP omega recovery at [GitHub](https://github.com/burning-cost/insurance-frequency-severity). Run `benchmarks/benchmark_insurance_frequency_severity.py` for reproducible output.

- [Your Frequency-Severity Independence Assumption Is Costing You Premium](/2025/05/14/frequency-severity-independence-is-costing-you-premium/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
- [Does HMM Telematics Risk Scoring Actually Work?](/2026/03/31/does-hmm-telematics-risk-scoring-actually-work/)
