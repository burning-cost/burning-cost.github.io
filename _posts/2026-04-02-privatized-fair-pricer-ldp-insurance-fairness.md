---
layout: post
title: "Pricing Fairly Without Seeing the Data: Local Differential Privacy for Insurance"
date: 2026-04-02
categories: [techniques, fairness]
tags: [fairness, differential-privacy, ldp, consumer-duty, fca, equality-act, insurance-fairness, arXiv-2504-11775, Zhang, Liu, Shi, Lindholm, categorical, randomized-response, discrimination-free]
description: "insurance-fairness v1.2.0 adds PrivatizedFairPricer: discrimination-free pricing when the sensitive attribute is privatised via local differential privacy. Based on Zhang, Liu & Shi (arXiv:2504.11775). We explain the mechanism, where UK regulation is heading, and why the sample size implications will sting."
math: true
author: burning-cost
---

The FCA does not currently require insurers to implement local differential privacy. It does require insurers to demonstrate, under Consumer Duty, that their pricing does not systematically disadvantage groups sharing protected characteristics. These two facts are about to create an uncomfortable gap.

Here is the problem. Ethnicity data is a special category under UK GDPR Article 9. If you want to audit whether your postcode factor is proxying for ethnicity — and Citizens Advice estimated a £280/year ethnicity penalty in UK motor insurance in 2022 — you need to collect ethnicity data, process it, and store it. The ICO is clear: special category data processing requires an explicit lawful basis. Most insurers do not have one for ethnicity data in pricing. The result: you cannot collect the data to run the fairness audit you are legally expected to demonstrate.

Local differential privacy (LDP) is the technical answer to this problem. Zhang, Liu, and Shi (arXiv:2504.11775, published April 2025) formalise how to do discrimination-free pricing when the sensitive attribute is never seen in clean form. `insurance-fairness` v1.2.0 implements their approach as `PrivatizedFairPricer`.

---

## What LDP actually does here

The mechanism Zhang et al. use is **randomised response** (k-RR), not the Laplace or Gaussian noise that most actuaries think of when they hear "differential privacy". Laplace and Gaussian mechanisms work on continuous values — adding calibrated noise to a numerical input. k-RR is for categorical attributes: gender, ethnicity, religion. The policyholder (or a trusted third party) answers the question "what is your protected group?" with a correct answer probability of $\pi = \exp(\varepsilon) / (|D| - 1 + \exp(\varepsilon))$ and a random wrong answer with probability $1 - \pi$.

At $\varepsilon = 2.0$ with $|D| = 2$ groups (binary attribute), $\pi \approx 0.88$. The insurer receives a noisy group label. The ground truth is never stored. The insurer can still fit group-specific models because the randomisation scheme is known, and correction matrices $T^{-1}$ and $\Pi^{-1}$ can debias the training loss.

The pricing formula that emerges is identical to Lindholm et al. (2022): the discrimination-free premium marginalises over a reference distribution $P^*$:

$$h^*(X) = \sum_k f_k(X) \cdot P^*(D = k)$$

What is new in Zhang et al. is that the group-specific models $f_k$ are fitted using **privatised** labels, not the true ones. The paper proves (Theorem 4.3) that the excess risk of this estimator over the oracle is bounded by a factor $C_1$, which amplifies with noise:

- At $\varepsilon = 2.0$: $C_1 \approx 1.07$ — near-oracle performance
- At $\varepsilon = 1.0$: $C_1 \approx 1.16$ — 16% excess risk amplification
- At $\varepsilon = 0.5$: $C_1 \approx 1.58$ — needs 2.5× the data for the same bound
- At $\varepsilon = 0.1$: $C_1 \approx 2.54$ — needs 6.5× the data

This is the number that will sting in practice.

---

## The API

`PrivatizedFairPricer` is a sklearn-compatible estimator. The key distinction from the existing `PrivatizedFairnessAudit` in the library is the API contract: this is a pricer, not an audit tool. It fits, predicts, and drops into a pipeline.

```python
import numpy as np
from insurance_fairness import PrivatizedFairPricer

# Policyholders respond to their own group question
# via randomised response (epsilon-LDP guaranteed)
pricer = PrivatizedFairPricer(
    epsilon=2.0,               # LDP budget — pi ≈ 0.88
    n_groups=2,                # binary protected attribute
    base_estimator='poisson_glm',
    reference_distribution='uniform',  # P*(D=k) = 0.5 for each group
)

# S: privatised group labels (already noised by policyholders)
# X: non-sensitive features (age, vehicle type, driving history)
# y: observed claims (pure premium or frequency)
pricer.fit(X_train, y_train, S_train, exposure=exposure_train)

# Fair premium — P*(D) marginalised, never conditions on group
fair_premium = pricer.predict(X_new)

# Check the statistical bound (95% confidence)
bound = pricer.statistical_bound(delta=0.05)
print(f"Excess risk bound: {bound:.4f}")

# How much data do you need for a given precision?
n_needed = pricer.minimum_sample_size(delta=0.05, target_bound=0.05)
print(f"Minimum n for bound <= 0.05: {n_needed:,}")

# Correction diagnostics
summary = pricer.correction_summary()
print(f"pi = {summary['pi']:.3f}, C1 = {summary['C1']:.3f}")
print(f"Negative weight fraction: {summary['negative_weight_frac']:.2%}")
```

If you do not yet know $\varepsilon$ but have covariate regions where group membership is nearly certain — anchor points where $P(D = k | X) \approx 1$ — Procedure 4.5 from the paper estimates $\pi$ empirically:

```python
pricer = PrivatizedFairPricer(n_groups=2, base_estimator='poisson_glm')
pricer.fit(X_train, y_train, S_train, X_anchor=X_anchor)
```

For gender, young male drivers (<22) and older female drivers (>65) might serve as approximate anchor populations. For ethnicity in UK motor data, no clean anchor population exists — this is a real limitation we discuss below.

---

## Where UK regulation is heading

The FCA does not require LDP. Colorado SB21-169, in force since January 2023, prohibits insurers from using race, colour, national origin, religion, sex, sexual orientation, disability, or gender identity in pricing — and, crucially, prohibits using **proxy variables** for those characteristics. Colorado has enforcement teeth; the UK has principles.

But the direction of travel is clear. The FCA's December 2025 Research Note on motor insurance pricing and local area ethnicity found a £28 residual premium differential in high-minority-concentration postcodes after full risk adjustment, and noted that firms should be able to demonstrate their pricing does not produce differential outcomes by protected characteristic. The FCA did not impose a data-collection requirement, but the implicit challenge is the same: if you believe the residual reflects risk, demonstrate it.

LDP is one answer to that impasse. The policyholder discloses, under formal privacy guarantees, a noisy version of their protected characteristic. The insurer never sees the true label, satisfies UK GDPR Article 9 data minimisation under Article 5(1)(c), and can still audit and correct for group-based disparities. The ICO has not published guidance on LDP in insurance specifically; the framework is novel enough that firms using it would be setting a precedent.

---

## The limitations you should know before deploying this

**Categorical attributes only.** k-RR is designed for categorical protected variables. Age, postcode deprivation quintile, and any continuous proxy do not fit. The class has `mechanism='laplace'` and `mechanism='gaussian'` stubs that raise `NotImplementedError` — they exist to mark where future extensions would go, not as implemented paths.

**Negative weights.** When $\varepsilon < 1.0$ and $|D| = 2$, a meaningful fraction of the correction weights $\Pi^{-1}$ go negative. Clipping those weights introduces bias. The class warns at a 5% negative weight fraction threshold. At $\varepsilon = 0.5$ this threshold is routinely exceeded.

**The trusted third party assumption.** The formal privacy guarantee in Theorem 4.3 requires a trusted third party (TTP) who receives the privatised labels and does not collude with the insurer. In a single-party deployment — the insurer handling the whole process — the privacy is formal in the documentation but not guaranteed in practice. A policyholder who uses the library to implement their own k-RR step before submission is the intended architecture.

**Anchor point quality.** Estimating $\pi$ from anchor populations requires covariate regions where group membership is nearly certain. For ethnicity in UK motor data, such regions do not exist. The anchor-point path should be treated with caution outside gender (where age extremes may provide partial separation).

**Sample size.** At $\varepsilon = 1.0$ with 80% confidence and a target bound of 0.05, `minimum_sample_size()` returns numbers in the hundreds of thousands for a two-group problem. Most UK personal lines books have enough data. Specialty lines may not.

**Reference distribution changes the portfolio premium.** Marginalising with a uniform $P^*(D = k)$ eliminates the correlation between price and group membership, but it shifts the expected portfolio premium. You need to reprice the book for actuarial balance after applying the correction.

---

## Why this matters now even if you are not deploying it

The Zhang et al. paper establishes the theoretical framework. The `PrivatizedFairPricer` implementation makes it experimentable. You do not need to deploy LDP collection to use this — you can run the pricer on synthetic privatised data to understand the sample size implications, the C1 amplification at different epsilon values, and the premium distribution shift from the reference distribution choice.

That is useful planning work. If the FCA moves from "we are not aware of a standard way to do this" to "we expect firms to demonstrate capability", firms that have already run these numbers are six months ahead of firms that have not.

Colorado SB21-169 passed in 2021. The EU AI Act's fairness provisions came into force in 2024. UK insurance is not immune to the direction that is already established internationally. The gap between what Consumer Duty says and what it will eventually require is closing.

---

## References

Zhang, Q., Liu, S., and Shi, C. (2025). 'Discrimination-Free Insurance Pricing with Privatized Sensitive Attributes.' arXiv:2504.11775v2.

Lindholm, M., Richman, R., Tsanakas, A., and Wüthrich, M. V. (2022). 'Discrimination-Free Insurance Pricing.' *ASTIN Bulletin* 52(1), 55–89.

Citizens Advice (2022). 'Paying over the odds: ethnicity and car insurance pricing.' Citizens Advice Research Report.

FCA (2025). Research Note: Motor insurance pricing and local area ethnicity in England and Wales. December 2025. Available at fca.org.uk/publications/research-notes/motor-insurance-pricing-local-area-ethnicity-england-wales

---

## Related posts

- [Your Pricing Model Might Be Discriminating](https://burning-cost.github.io/2026/03/03/your-pricing-model-might-be-discriminating/) — proxy detection and the FCA Consumer Duty evidence problem
- [insurance-fairness](https://github.com/burning-cost/insurance-fairness) — the full library, v1.2.0
