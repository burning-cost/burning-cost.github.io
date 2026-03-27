---
layout: post
title: "Fairness Auditing When You Don't Have Sensitive Attributes"
date: 2026-03-20
author: Burning Cost
categories: [compliance, fairness, libraries]
description: "Most UK insurers don't hold ethnicity, religion, or disability status. Consumer Duty still requires evidence of fair outcomes. PrivatizedFairnessAudit solves this with local differential privacy."
tags: [FCA, Consumer-Duty, fairness, proxy-discrimination, insurance-fairness, PrivatizedFairnessAudit, differential-privacy, LDP, randomised-response, anchor-point, Equality-Act, python, uk-insurance, v0.3.8]
---

The standard fair value audit assumes you can observe the protected characteristic. You have gender on the policy record. You have date of birth. You can segment by protected group and measure whether outcomes differ.

Most UK motor and home insurers do not hold ethnicity, religion, or disability status. The Equality Act 2010 does not require collection of these attributes, and until recently there was no practical reason to collect them. GDPR adds collection-minimisation pressure in the other direction. So most pricing teams, if asked to audit their model for ethnicity-related discrimination, would answer honestly that they do not know the ethnicity of their policyholders and cannot therefore measure whether outcomes differ by ethnicity.

This is a genuine problem. It does not make the Consumer Duty obligation go away. PRIN 2A.4.6 requires firms to monitor outcomes across groups with protected characteristics, not just the protected characteristics they happen to hold data for. The expectation is that firms take reasonable steps to assess fair value even where direct data is not available.

There are two practical approaches. The first, which `FairnessAudit` already supports, uses proxy variables: ONS Census 2021 LSOA-level ethnicity estimates, for instance, can be joined to postcode data to produce an approximate ethnicity attribution. This is an estimate rather than an observation, and it inherits all the limitations of area-level attribution for individuals. But it is better than nothing, and it is the approach most firms are currently taking where they address the problem at all.

The second approach, added in `insurance-fairness` v0.3.8, is more principled. `PrivatizedFairnessAudit` implements the MPTP-LDP protocol from Zhang, Liu and Shi (2025). The idea is to collect a small opt-in sample of noisy protected attributes via randomised response, then correct for the noise mathematically to recover population-level disparity estimates. The privacy guarantee is cryptographic rather than policy-based, which matters both for GDPR purposes and for the credibility of the audit evidence.

---

## The proxy gap problem

To understand why the data gap matters, consider what an FCA supervision team would want to see in a fair value assessment for home insurance.

They would want evidence that the pricing model does not produce systematically worse outcomes for customers in protected characteristic groups under the Equality Act 2010. For gender and age, this is straightforward: most home insurers hold these attributes and can produce segmented A/E ratios directly. For ethnicity, religion, and disability, the data typically does not exist on the policy record.

The proxy-based approach fills part of this gap. If you have postcode, you can estimate ethnicity concentration via Census 2021 data. If you run `FairnessAudit` with postcode as a protected proxy, you get an analysis of whether premium outcomes differ systematically across postcode bands with different ethnic compositions. This is what Citizens Advice did in their 2022 analysis that identified a £280/year ethnicity penalty in UK motor insurance. It is legitimate methodology and it is auditable. Its limitation is that area-level attribution is noisy at the individual level. A high-BME postcode contains people of all ethnicities.

Randomised response solves this differently. Rather than imputing the protected attribute from proxy variables, it asks a small sample of policyholders directly but uses a privacy mechanism that makes no individual response identifiable. The mathematics then corrects for the noise in the aggregate.

---

## How randomised response works

The mechanics of randomised response are straightforward. Each participant in the opt-in sample is asked to report their protected characteristic (say, ethnicity, encoded as a binary or K-class label). But before responding, they flip a coin privately. If the coin comes up heads, they report their true attribute. If it comes up tails, they report a randomly chosen attribute from the available classes. The interviewer cannot tell, for any individual response, whether it reflects the true attribute or a random substitution.

The noise rate is controlled by the LDP privacy budget epsilon. Higher epsilon means more of the true signal is preserved; lower epsilon means stronger privacy protection. The mathematical correction inverts the noise matrix to recover the population-level prevalences and, more usefully, the group-level outcome disparities.

This is not new methodology. Randomised response dates to Warner (1965). What Zhang, Liu and Shi (2025) add is a complete statistical framework for discrimination-free insurance pricing under LDP, including the correction matrices (Lemma 4.2), a generalisation bound for the corrected estimator (Theorem 4.3), and a procedure for estimating the noise rate when it is unknown (Procedure 4.5, the anchor-point estimator).

---

## Using PrivatizedFairnessAudit

The class takes the noisy attribute `S` (the randomised responses from your opt-in sample) alongside the feature matrix `X` and outcome `Y`. You need to specify the noise rate, either directly via `pi` (the correct-response probability) or via `epsilon` (the LDP privacy budget). If you do not know the noise rate because you are estimating it from the data, pass `X_anchor` and the class uses Procedure 4.5 to estimate it.

```python
uv add insurance-fairness
```

```python
import numpy as np
from insurance_fairness import PrivatizedFairnessAudit

rng = np.random.default_rng(42)
n = 800

# Synthetic UK home insurance portfolio
# X: feature matrix (postcode band, build year, sum insured band, ncd proxy)
X = rng.standard_normal((n, 4))

# True ethnicity: binary, 30% group 1 (not observed)
true_group = rng.choice([0, 1], size=n, p=[0.70, 0.30])

# Inject a disparity: group 1 pays ~12% more on average
base_premium = np.exp(4.0 + 0.15 * X[:, 0] - 0.08 * X[:, 1])
premium = base_premium * (1 + 0.12 * true_group)
claim_amount = base_premium * rng.lognormal(0, 0.5, n)

# Randomised response with epsilon=1.5 (privacy budget)
# epsilon=1.5 -> pi = exp(1.5) / (1 + exp(1.5)) ~ 0.818
epsilon = 1.5
exp_e = np.exp(epsilon)
pi = exp_e / (1 + exp_e)  # ~0.818

S = np.where(
    rng.random(n) < pi,
    true_group,
    1 - true_group,  # flipped
)

# Fit the audit
audit = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=epsilon,
    reference_distribution="uniform",  # uniform P* for UK fairness requirement
    loss="gaussian",
    nuisance_backend="sklearn",
    random_state=42,
)
audit.fit(X, claim_amount, S, exposure=np.ones(n))

result = audit.audit_report()
print(f"Noise-corrected group prevalences: {result.p_corrected}")
print(f"Generalisation bound (95%): {result.bound_95:.4f}")
print(f"Negative weight fraction: {result.negative_weight_frac:.3f}")

# Fair (discrimination-free) premiums on the same portfolio
fair_premiums = audit.predict_fair_premium(X)
```

The `fair_premiums` array contains the discrimination-free predictions: `h*(X) = sum_k f_k(X) * P*(D=k)`. With `reference_distribution="uniform"`, the reference distribution P* assigns equal weight to each group, breaking the indirect discrimination that would occur if the population group imbalance were preserved.

The `negative_weight_frac` field is a quality check. When the LDP noise is too heavy relative to the sample size, the correction matrices produce negative weights for some observations before clipping. A fraction above 0.05 means the noise is too strong for reliable correction: either increase epsilon (weaker privacy, more signal) or collect a larger sample.

---

## What sample size you actually need

The generalisation bound from Theorem 4.3 in Zhang et al. scales as `O(C1 * K^2 / sqrt(n))`, where `C1` is the noise amplification factor determined by epsilon and K. Heavier noise (lower epsilon) means larger C1, which means the bound only tightens slowly with additional data.

For binary attributes (K=2) with epsilon=1.5, the library's `minimum_n_recommended()` method gives a practical answer:

```python
n_min = audit.minimum_n_recommended(delta=0.05, target_bound=0.05)
print(f"Minimum recommended n: {n_min}")
# Approximately 500-700 for epsilon=1.5, K=2
```

The paper (Zhang et al.) suggests 500+ for binary attributes at typical LDP parameters. For K=3 or K=4 groups (for instance, a four-category ethnicity classification), you need more. The bound scales with K squared, so moving from binary to four groups roughly quadruples the sample size requirement, everything else equal.

This means the approach is not appropriate for tiny books. If your opt-in sample is 150 policyholders, the correction is available but the bound will be wide, and the disparity estimates should be treated as directional rather than precise. Report them with explicit uncertainty; do not present them as definitive evidence in either direction.

One important check before relying on the anchor-point estimator: the `anchor_quality` field in the audit result. This is the maximum predicted probability from the anchor classifier in the highest-confidence group region. If it falls below 0.90, the library issues a warning. A low anchor quality means no covariate region achieves near-certain group identification, and the estimated noise rate will be biased. In practice this means the opt-in sample's covariate distribution needs to contain some sub-populations with very high group concentration for the anchor estimator to work well.

---

## When to use which approach

For most firms, the proxy-based approach via `FairnessAudit` is the right starting point. It requires no data collection beyond what you already have (postcode, plus ONS Census 2021 for ethnicity attribution), it runs in minutes on a standard laptop, and it produces group-level A/E analysis and proxy detection across all rating factors. The documentation trail is straightforward.

`PrivatizedFairnessAudit` becomes relevant when proxy attribution is inadequate or when you want to collect first-party evidence with a formal privacy guarantee. The contexts where this matters: large retail books where the postcode-level attribution is too coarse; situations where a regulator or audit function has explicitly questioned the area-level proxy methodology; or cases where the protected characteristic in question (disability status, religion) does not have a reliable postcode-level proxy.

The LDP approach also produces something the proxy approach cannot: a discrimination-free corrected pricing model, not just an audit flag. The output `fair_premiums` is a set of predictions that satisfy the theoretical non-discrimination condition — an approach that complements the causal debiasing available via [double machine learning in insurance-causal](/2026/03/23/does-dml-causal-inference-work-insurance-pricing/), which addresses confounding from observed risk factors rather than proxy discrimination — not merely an estimate of whether the current model discriminates. If the audit finds a disparity and you want to correct it mathematically rather than via tariff surgery, `PrivatizedFairnessAudit` provides that path.

---

## Limitations

Three things this approach does not solve.

First, opt-in bias. Policyholders who agree to disclose their ethnicity or disability status are not a random sample. The corrected disparity estimates reflect the experience of the opt-in sample, which may differ systematically from the full book. Monitor opt-in rates by segment (you can track these as a custom feature in [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)); a very low opt-in rate among a specific group makes the estimates for that group unreliable.

Second, regulatory status. Consumer Duty requires firms to take reasonable steps to assess fair outcomes for all protected characteristic groups. The LDP approach satisfies the spirit of this requirement, but the FCA has not issued specific guidance on acceptable methodology for the case where attributes are not directly observed. The approach is methodologically sound and privacy-preserving, but you should document it carefully — the [insurance-governance model inventory](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) is the natural place to record which fairness tests were run, what the results were, and who signed off — and frame it as part of a broader fair value assessment rather than as a standalone compliance answer.

Third, noise versus signal. With epsilon values below 1.0 (strong privacy), the noise amplification factor C1 becomes large and estimates are unreliable even with substantial sample sizes. We do not recommend deploying `PrivatizedFairnessAudit` with epsilon below 1.0 unless the sample is very large.

---

`insurance-fairness` v0.3.8 is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness) and on [PyPI](https://pypi.org/project/insurance-fairness/). The `PrivatizedFairnessAudit` class and the full MPTP-LDP implementation are available in this release. Python 3.10+.

---

**Related posts:**
- [The FCA Is Investigating Home and Travel Insurers: Is Your Pricing Model Defensible?](/2026/03/19/the-fca-is-investigating-home-and-travel-insurers/) -- the regulatory context and what a defensible Consumer Duty fair value assessment requires
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) -- the detection methodology when you do have proxy data: mutual information, CatBoost R-squared, SHAP decomposition
- [Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA](/2026/03/10/insurance-fairness-ot/) -- WassersteinCorrector for discrimination-free premiums when you have the attribute directly
