---
layout: post
title: "LDP Optimal Mechanism for Privatised Fairness Auditing in Insurance Pricing"
date: 2026-04-01
categories: [fairness, machine-learning]
tags: [local-differential-privacy, fairness, privatized-audit, LDP, randomised-response, optimal-mechanism, epsilon, insurance-fairness, Equality-Act-2010, ECJ-Test-Achats, GDPR-Article-9, Ghoukasian-Asoodeh-2025, Zhang-Liu-Shi-2025, protected-characteristics, insurance-pricing, UK-insurance]
description: "k-ary randomised response applies symmetric noise — wasteful and fairness-suboptimal. The Ghoukasian-Asoodeh optimal mechanism is asymmetric: minority groups get a higher correct-response probability, majority groups absorb more noise. This gives strictly better fairness per epsilon budget. v1.1.1 ships OptimalLDPMechanism, LDPEpsilonAdvisor, and mechanism='optimal' in PrivatizedFairnessAudit."
author: Burning Cost
---

UK insurers have not been allowed to use gender as a direct rating factor since December 2012 — that is the effect of the ECJ Test-Achats ruling (Case C-236/09), transposed into domestic law via the Equality Act 2010 (Amendment) Regulations. Most motor books stopped collecting it at point of sale years ago. What replaced it was a collection of proxy variables: vehicle type, NCD history, telematics patterns, occupation. The proxies carry the gender signal; the attribute itself is absent.

This creates a compliance gap that standard fairness tooling cannot close. FairLearn and AIF360 both require you to observe the protected attribute at training time. If you do not hold gender data, you cannot train group-specific models, cannot compute demographic parity, cannot run any of the Lindholm-style conditional fairness checks we have written about before. The tools that the ML fairness community has built assume the attribute is available. In UK insurance, it often is not.

Local differential privacy (LDP) solves this. The idea, formalised for insurance by Zhang, Liu & Shi (arXiv:2504.11775), is that a policyholder or a trusted third party can report a noised version of the protected attribute — a privatised proxy $S$ — without revealing the true attribute $D$ to the insurer. The insurer receives only $S$. They can then train fairness-corrected models on $S$ using a correction matrix that undoes the known noise structure. The true attribute is never processed.

Version 1.1.1 of `insurance-fairness` adds the optimal LDP mechanism from Ghoukasian & Asoodeh (arXiv:2511.16377), an epsilon advisor, and a `mechanism='optimal'` parameter in `PrivatizedFairnessAudit`. This post explains what was built and why it matters.

---

## The standard mechanism and its waste

The standard k-ary randomised response (k-RR) is the obvious LDP mechanism. A respondent in group $j$ reports their true group with probability $\pi$ and reports each other group with probability $(1-\pi)/(K-1)$:

$$\pi = \frac{e^\varepsilon}{K - 1 + e^\varepsilon}$$

So $\pi$ is above $1/K$ (the pure-noise floor) and below 1 (perfect accuracy). The parameter $\varepsilon$ controls where on this range you sit: larger $\varepsilon$ means less noise and better utility.

The mechanism is symmetric. Every group gets the same $\pi$. This is convenient — one parameter governs everything — but it is also wasteful. Data unfairness is not symmetric. If group 0 is a small minority (say, 15% of the population) and group 1 is the majority (85%), then noise that mislabels group 0 members distorts the minority signal proportionally more. The majority signal is more robust to noise simply because of population mass.

Ghoukasian & Asoodeh (2025) make this precise. They define data unfairness $\Delta(D_Q)$ — the total variation distance between the noised marginal distribution and the true one, under perturbation matrix $Q$ — and ask: for a fixed $\varepsilon$ budget, what $Q$ minimises $\Delta(D_Q)$?

The answer is asymmetric noise.

---

## The optimal mechanism

For a binary attribute ($K = 2$), Theorem 2 of Ghoukasian & Asoodeh gives a closed form. Call the two groups 0 and 1, with true prevalences $p_0$ and $p_1$. The optimal mechanism is:

When group 0 is the minority ($p_0 < p_1$):
- Group 0 correct-response probability: $p^* = 1 - e^{-\varepsilon}/2$
- Group 1 correct-response probability: $q^* = 1/2$

When group 1 is the minority:
- $p^* = 1/2$, $q^* = 1 - e^{-\varepsilon}/2$

The minority group gets preferential noise allocation. At $\varepsilon = 1$, group 0 in the minority case gets $p^* = 1 - e^{-1}/2 \approx 0.816$; the majority gets $q^* = 0.5$. In symmetric k-RR, both groups get $\pi = e^1/(1+e^1) \approx 0.731$. The optimal mechanism is giving the minority group a 8.5 percentage point improvement in correct-response rate while still satisfying the $\varepsilon$-LDP constraint.

The empirical consequence is substantial. On the LSAC dataset used in the paper, the optimal binary mechanism achieves roughly a 50% reduction in both statistical parity and equalised odds gaps relative to symmetric k-RR at the same $\varepsilon$.

For $K > 2$ groups, the closed form does not exist. The mechanism is instead found by solving a linear fractional program: minimise $\Delta(D_Q)$ subject to the standard column-wise $\varepsilon$-LDP constraints ($Q_{ji} / Q_{j'i} \leq e^\varepsilon$ for all $j \neq j'$, all $i$) and row-stochasticity of $Q$. We use a direct LP reformulation with auxiliary variables to handle the L1 objective, solved via scipy's HiGHS backend.

---

## What was built

Three components shipped in v1.1.1.

### OptimalLDPMechanism

```python
from insurance_fairness.optimal_ldp import OptimalLDPMechanism

# Binary case — closed form
mech = OptimalLDPMechanism(
    epsilon=1.0,
    k=2,
    group_prevalences=np.array([0.15, 0.85]),  # 15% minority
)

# Apply to true labels to generate privatised S
S_private = mech.privatise(true_labels, rng=np.random.default_rng(42))

# Inspect the perturbation matrix
Q = mech.perturbation_matrix
# Q[j, i] = P(report i | true group j)
# Diagonal entries are correct-response probabilities — asymmetric for K=2

# TV distance between noised and true marginals
print(mech.unfairness_bound())   # lower is better
```

For $K > 2$, the LP solver runs automatically. If it fails (numerically degenerate), the mechanism falls back to symmetric k-RR with a warning:

```python
# Multi-valued: e.g. four ethnicity proxy categories
mech = OptimalLDPMechanism(
    epsilon=2.0,
    k=4,
    group_prevalences=np.array([0.60, 0.20, 0.12, 0.08]),
)
Q = mech.perturbation_matrix  # 4×4, solved via LP
```

A note on the LP result for $K \geq 3$: `unfairness_bound()` often returns exactly 0.0 at moderate $\varepsilon$. This is not a numerical error — the LP can find mechanisms that perfectly preserve the marginal distribution while still satisfying LDP. It is a mathematical property of the LP feasible set for $K \geq 3$; the binary case cannot achieve this because the feasible set is smaller.

### LDPEpsilonAdvisor

The central practical question for any LDP deployment is: what $\varepsilon$ should we use? The answer depends on dataset size, number of groups, and how much statistical noise you can tolerate. The noise amplification factor $C_1$ quantifies the cost:

$$C_1 = \frac{\pi + K - 2}{K\pi - 1}$$

This factor multiplies the generalisation bound from Theorem 4.3 of Zhang et al. At $K = 2$:
- $\varepsilon = 0.5$: $C_1 \approx 2.54$ — 154% bound inflation vs. non-private
- $\varepsilon = 1.0$: $C_1 \approx 1.58$ — 58% inflation
- $\varepsilon = 2.0$: $C_1 \approx 1.16$ — 16% inflation
- $\varepsilon = 5.0$: $C_1 \approx 1.01$ — effectively no inflation

```python
from insurance_fairness.optimal_ldp import LDPEpsilonAdvisor

advisor = LDPEpsilonAdvisor(
    n_samples=50_000,        # your training set size
    k=2,
    target_bound_inflation=0.20,   # tolerate 20% inflation of the bound
)

rec = advisor.recommend()
print(rec["rationale"])
# "With K=2 groups and n=50,000 samples, epsilon=1.73 gives pi=0.8499
# (correct-response rate), C1=1.200 (20.0% bound inflation vs. non-private).
# Generalisation bound ~ 0.0028."

# Or sweep over epsilon to see the trade-off
df = advisor.sweep()
# Returns polars DataFrame: epsilon, pi, C1, gen_bound, bound_inflation_pct
```

For a 50,000-policy training set with binary protected attribute, $\varepsilon \approx 1.7$ achieves 20% bound inflation. That is the break-even point where the LDP correction costs you roughly one-fifth of a standard deviation of generalisation bound — in our view, a reasonable privacy price for a regulatory evidence trail.

### mechanism='optimal' in PrivatizedFairnessAudit

The `PrivatizedFairnessAudit` class — which implements the full MPTP-LDP protocol from Zhang et al. — now accepts `mechanism='optimal'`:

```python
from insurance_fairness import PrivatizedFairnessAudit

audit = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=1.5,
    mechanism="optimal",          # asymmetric noise allocation
    reference_distribution="uniform",   # gender-neutral: equal P* for both groups
    loss="poisson",
    nuisance_backend="catboost",
)

audit.fit(X=X_train, Y=claim_frequency, S=S_privatised)

# Fair premiums: h*(X) = sum_k f_k(X) * P*(D=k)
# With uniform P*, every risk is priced as if drawn from a 50/50 population
fair_prices = audit.predict_fair_premium(X_test)

result = audit.audit_report()
print(result.mechanism)           # "optimal"
print(result.bound_95)            # generalisation bound at delta=0.05
print(result.negative_weight_frac)  # should be < 0.05 for reliable correction
```

The `mechanism='optimal'` parameter requires `epsilon=` and is incompatible with `pi=` directly — the optimal mechanism derives asymmetric per-group noise from $\varepsilon$, so a single scalar $\pi$ does not describe it. After fitting, the full perturbation matrix is available at `audit.perturbation_matrix_`.

The correction matrices (`audit_report()` and `correction_matrices()`) for the optimal mechanism are derived by inverting $Q$ numerically rather than using the closed-form $T^{-1}$ that applies to symmetric k-RR. This is exact for well-conditioned $Q$ and falls back to pseudoinverse with a warning if $Q$ is near-singular.

---

## The regulatory context

The architecture here is not theoretical preparation. It is the response to a specific legal gap that has existed since 2012 and is expanding.

**Gender (2012 onwards).** UK motor insurers cannot use or hold gender for pricing. The LDP framework allows a fairness audit against gender without the insurer ever processing the raw attribute. A policyholder or verification service can generate $S$ under a known $\varepsilon$ protocol; the insurer receives only the noised label and runs the MPTP-LDP audit. The `reference_distribution='uniform'` option prices every policy as if drawn from a 50/50 population — mechanically satisfying the Equality Act requirement for group-invariant pricing.

**Ethnicity (emerging).** An FCA Research Note published in December 2025 analysed six million motor policies covering more than 50% of the UK market. After full risk adjustment, there is a £28 per year unexplained residual in Luton — 43% minority ethnic population, mean premium £627. The FCA's current position is that this residual is likely unmeasured risk variables rather than direct discrimination. But pricing teams are on notice. If the residual persists through future review cycles, the regulatory pressure toward ethnicity data restrictions analogous to the gender ban will intensify.

**GDPR Article 9.** Ethnicity, health, religious beliefs, and biometric data are all special category data under UK GDPR and DPA 2018. Collecting and processing them for pricing requires an Article 9(2) basis — typically "substantial public interest" with a written policy. Many insurers hold none of this data. LDP offers a potential pathway: if $S$ is generated by the policyholder under a known randomisation protocol, the insurer may not be "processing" the raw special category attribute (because no individual's true attribute is recoverable from $S$ alone). This is legally contested. It is, however, the direction of travel in academic and regulatory discussion.

**Data (Use and Access) Act 2025.** Fully implements UK GDPR Article 22: human review required for solely automated decisions with significant effects. Insurance pricing is in scope. Removing the direct protected-attribute pathway from the pricing model — which is what the MPTP-LDP protocol does — reduces the UK DAUA exposure by eliminating the most obvious automated-decision trigger.

We are not suggesting the LDP architecture fully resolves any of these compliance issues. What it does is create an evidential infrastructure. When the FCA asks "how do you demonstrate that your pricing model does not produce systematically worse outcomes for ethnically diverse postcodes?", `PrivatizedFairnessAudit.audit_report()` produces the statistical bound and group-model evidence that can anchor that answer.

---

## What the generalisation bound actually tells you

The bound from Theorem 4.3 of Zhang et al. is worth understanding before citing it in an evidence pack.

The bound guarantees that the excess risk of the LDP-corrected model $h^*$ relative to the oracle model trained on true attributes is at most:

$$K \cdot \sqrt{\frac{d_{VC} + \ln(2/\delta)}{2n}} \cdot 2 \cdot C_1 \cdot K$$

where $d_{VC}$ is the VC dimension of the function class, $n$ is training set size, and $C_1$ is the noise amplification factor. This is an $O(C_1 K^2 / \sqrt{n})$ bound.

Two things to notice. First, the bound degrades quadratically in $K$ — more groups are genuinely harder. Going from binary gender to four ethnicity categories costs a factor of four in the bound ($(4/2)^2$). You need four times as much data, or accept a worse bound, when you add groups. Second, the bound is an excess risk bound, not a bias bound — it quantifies how far the LDP-corrected model can be from an oracle, not how far it is from a perfectly fair model. The oracle itself may not be perfectly fair; the bound says nothing about that.

`audit.statistical_bound(delta=0.05)` computes this for the fitted audit. `audit.minimum_n_recommended(target_bound=0.05)` inverts it to give the minimum training set size needed for a target bound quality.

---

## Honest limitations

The Zhang et al. paper uses a US health insurance dataset with 1,338 observations and a synthetic dataset with 5,000. These are small by UK motor standards. The generalisation bound analysis is solid but the empirical validation is limited — particularly, there is no comparison to FairLearn or AIF360 under equivalent privacy constraints, and no UK-specific dataset.

The correction matrix inversion can produce negative sample weights. This happens when the LDP noise is so heavy that the matrix $T^{-1}$ has entries large enough to make some reweighted observations negative after clipping. If `result.negative_weight_frac` exceeds 0.05, the correction is unreliable — you are clipping more than 5% of (observation, group) pairs to zero weight, which introduces bias in the group-specific models. At $\varepsilon = 0.5$ and $K = 2$, $C_1 \approx 2.54$; in our testing on datasets below 10,000 observations, negative weight fractions above the 5% threshold are common at this epsilon level. The practical minimum for reliable correction is $\varepsilon \approx 1.0$ for binary attributes and $\varepsilon \approx 1.5$ for four groups.

The anchor-point estimator (Procedure 4.5 in the paper) is the alternative when $\varepsilon$ is not known — you estimate the noise rate from covariate regions where group membership is near-certain. This requires covariate subsets where $P(D = j^* | X) \approx 1$: fleet vehicles for commercial indicator, very young age buckets for a binary age-band variable. If no such anchor exists in your data, `anchor_quality_` will be below 0.90 and the library raises a warning. The estimate will be biased.

---

## Installation

```bash
uv add "insurance-fairness>=1.1.1"
```

`OptimalLDPMechanism` and `LDPEpsilonAdvisor` live in `insurance_fairness.optimal_ldp`. The `mechanism='optimal'` parameter is in `PrivatizedFairnessAudit` in `insurance_fairness.privatized_audit`. Source: [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).

The two papers: Zhang, Liu & Shi (2025) at arXiv:2504.11775, and Ghoukasian & Asoodeh (2025) at arXiv:2511.16377.

---

## Related posts

- [Discrimination-Free Insurance Pricing: The Lindholm Approach](/2026/03/31/discrimination-free-pricing-privatised-attributes-insurance/) — the oracle approach that LDP enables when the attribute is unavailable
- [Sequential Optimal Transport for Multi-Attribute Fairness](/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — for when you have proxy attribute data and need joint fairness across multiple characteristics
- [Proxy Discrimination in Motor Pricing: What FCA PS25/21 Requires](/2026/03/28/fca-ps2521-proxy-discrimination-motor-pricing/) — the regulatory framing
