---
layout: post
title: "Building an FCA Ethnicity Fairness Evidence Pack Without Holding Ethnicity Data"
date: 2026-04-03
categories: [fairness, regulation]
tags: [fairness, local-differential-privacy, randomised-response, fca, consumer-duty, ep25-2, equality-act, gdpr-article-9, ethnicity, proxy-discrimination, insurance-fairness, PrivatizedFairnessAudit, PrivatizedFairPricer, discrimination-free-pricing, zhang-liu-shi-2025, lindholm-2022, correction-matrices, python, uk-insurance]
description: "An FCA Research Note (December 2025) found a £28 unexplained ethnicity residual across six million motor policies. Your pricing team cannot measure it because you do not hold ethnicity. Local differential privacy via randomised response gives you a formal mechanism to collect noised ethnicity labels and build a defensible evidence trail — without processing special category data in clean form."
math: true
author: burning-cost
---

An FCA Research Note published in December 2025 — "Motor insurance pricing and local area ethnicity in England and Wales" — analysed six million motor insurance policies, covering more than half the UK market by insurer. After risk-adjusting for all observable covariates, it found a £28 residual annual premium differential in Luton (where 43% of residents are from minority ethnic backgrounds), on a total mean premium of £627. The FCA's characterisation was cautious — the residual is "likely attributable to unmeasured risk variables" rather than direct discrimination. But that framing contains an implicit challenge: if you believe the residual reflects risk, demonstrate it. Show us the fairness test.

Most UK motor pricing teams cannot run that test. They do not hold individual ethnicity data. The Equality Act 2010 does not require them to collect it, GDPR Article 9 creates friction around collecting it, and until recently there was no practical framework for auditing ethnicity fairness without the attribute.

This post describes what that framework looks like in practice, using Zhang, Liu & Shi (arXiv:2504.11775, 2025) and the `insurance-fairness` library. We are not describing a deployed production system. We are describing what a pricing team should build as a governance artefact — something you can hand to an FCA supervision team if the FCA's ethnicity monitoring work turns into a formal data request.

---

## What you are actually trying to demonstrate

The Consumer Duty (PRIN 2A) requires that pricing delivers fair value to all customers. The Equality Act 2010 section 19 prohibits indirect discrimination: applying a provision, criterion, or practice that puts a protected group at a particular disadvantage. For motor insurance, postcode rating is the obvious candidate. The FCA's December 2025 Research Note measured a £28/year residual in Luton for policies in high-minority-concentration postcodes after full risk adjustment.

The standard response to a fairness question is: run a group-level A/E test, compare observed outcomes against expected for each demographic group, report the ratio. This requires individual group labels. If you do not have ethnicity on your policy records, you cannot run this test.

The proxy-based alternative — joining ONS Census 2021 LSOA ethnicity estimates to postcode and treating area composition as an individual attribute — is the approach most firms have taken where they have addressed this at all. It is the methodology behind the Citizens Advice £280/year figure from 2022 and the FCA's own December 2025 Research Note analysis. It is useful and it is auditable. Its limitation is that it conflates area with individual. A postcode in a high-BME area contains people of all ethnicities.

What local differential privacy (LDP) provides is a path to first-party evidence: an individual-level fairness test with formal statistical guarantees, obtained without the insurer ever processing clean ethnicity data.

---

## The mechanism in one paragraph

The mechanism the Zhang-Liu-Shi paper uses is **randomised response**, the Warner (1965) protocol. Each participant answers the protected-attribute question — "which group do you belong to?" — but applies noise locally before responding. With probability $\pi$ they report truthfully; with probability $1 - \pi$ they report a random alternative. For a binary attribute with privacy parameter $\varepsilon$:

$$\pi = \frac{e^\varepsilon}{1 + e^\varepsilon}$$

At $\varepsilon = 1.0$, $\pi \approx 0.731$: roughly three in four responses are truthful. The insurer receives a noised label $S$, never the true attribute $D$. This is $\varepsilon$-local differential privacy: no individual response can be decoded. The formal guarantee is that $P(S = s \mid D = d) / P(S = s \mid D = d') \leq e^\varepsilon$ for all $d \neq d'$.

The paper's Lemma 4.2 proves that group-specific models $f_k$ can be estimated from the noised $S$ using correction matrices $T^{-1}$ and $\Pi^{-1}$ derived from the known noise rate. These weights are applied during training. The resulting discrimination-free premium is:

$$h^*(X) = \sum_k f_k(X) \cdot P^*(D = k)$$

where $P^*$ is a fixed reference distribution chosen before fitting — not inferred from the training data. With uniform $P^*$ (equal weight to each group), the premium is independent of the estimated group composition of the portfolio. The postcode signal that used to proxy for ethnicity is absorbed into the $f_k$ models; the averaging over $P^*$ prevents it from affecting the final premium through the group composition channel.

The coverage of the theory is in our [earlier post on the Zhang-Liu-Shi mechanism](/2026/03/31/discrimination-free-pricing-privatised-attributes-insurance/). This post is about what you build.

---

## The noise amplification cost

Before describing the implementation, the numbers you need to understand.

The noise amplification factor $C_1$ quantifies how much statistical precision the LDP correction costs relative to training on clean attribute labels:

$$C_1 = \frac{\pi + K - 2}{K\pi - 1}$$

For $K = 2$ (binary protected attribute):

| $\varepsilon$ | $\pi$ | $C_1$ | Relative sample requirement |
|:---:|:---:|:---:|:---:|
| 2.0 | 0.881 | 1.16 | 1.0× baseline |
| 1.0 | 0.731 | 1.58 | 1.9× |
| 0.5 | 0.622 | 2.54 | 4.8× |
| 0.1 | 0.525 | 10.5 | 82× |

The practical floor is $\varepsilon \geq 1.0$. Below 0.5 the correction matrices produce negative sample weights for a material fraction of observations — the library clips these to zero and the corrected estimate becomes biased. Above $\varepsilon = 2.0$ the privacy guarantee weakens to the point where the formal LDP argument is less compelling (at $\varepsilon = 2.0$, an observer can still distinguish true-group responses with likelihood ratio $e^2 \approx 7.4$).

For most UK personal lines motor books — call it 100,000 training policies — operating at $\varepsilon = 1.5$ is achievable: $C_1 \approx 1.30$, meaning you need 66% more data than if you had clean labels, and at 100,000 policies the generalisation bound is still well inside a practically useful range.

---

## The four-step evidence pack

Here is what you actually build.

### Step 1: Collect privatised labels from a volunteer sample

You need an opt-in sample of policyholders willing to answer an ethnicity question under the LDP protocol. This is not a full-portfolio collection — for the purpose of an evidential fairness audit, you need roughly:

```python
from insurance_fairness import PrivatizedFairnessAudit
import numpy as np

# Minimum n for a given quality target
# K=2 binary attribute, epsilon=1.5, 95% confidence, target bound 0.05
audit_sizing = PrivatizedFairnessAudit(n_groups=2, epsilon=1.5)
# After fitting: audit_sizing.minimum_n_recommended(delta=0.05, target_bound=0.05)
# Returns numbers in the hundreds of thousands for these parameters
```

At $\varepsilon = 1.5$ with $K = 2$ groups, the sample requirement for a bound of 0.05 at 95% confidence runs into the hundreds of thousands. The LDP noise correction demands far more data than intuition suggests: the statistical amplification factor at $\varepsilon = 1.5$ ($C_1 \approx 1.30$) means you need roughly 66% more labelled observations than a clean-label audit — but the sample sizes required for useful statistical bounds remain in the hundreds of thousands. For most UK motor books, this means the method is viable only at portfolio scale, not via a small opt-in exercise.

The policyholder themselves applies the k-RR noise before submission. This means the insurer never processes the clean response. In practice this requires either: a client-side implementation in the customer portal (a few lines of JavaScript), or a trusted third party that holds the clean responses and returns only the noised $S$ values to the insurer.

Whether the noised response $S$ constitutes special category data processing under GDPR Article 9 is legally contested. The ICO's current position is that LDP-protected data is not automatically anonymised. We recommend legal advice before collection and document the lawful basis explicitly — likely Article 9(2)(b) (substantial public interest: demonstrating fairness under the Equality Act) with a written policy under Schedule 1 DPA 2018.

### Step 2: Fit the LDP-corrected model

```python
from insurance_fairness import PrivatizedFairnessAudit

audit = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=1.5,           # pi ≈ 0.818; agreed with TTP or self-administered
    reference_distribution="uniform",  # P*(D=k) = 0.5 — group-neutral pricing
    loss="poisson",        # claim frequency model
    nuisance_backend="catboost",
    random_state=42,
)

# X: non-sensitive features (postcode band, vehicle class, NCD, occupation)
# y: claim frequency or pure premium
# S: the privatised (noised) ethnicity labels from your opt-in sample
# exposure: policy years
audit.fit(X_sample, y_sample, S_sample, exposure=exposure_sample)
```

The `audit.fit()` call trains $K$ group-specific models on the full sample with correction-matrix-derived sample weights, then marginalises them over the uniform reference distribution.

Inspect the correction quality immediately after fitting:

```python
mats = audit.correction_matrices()
print(f"pi estimate: {mats['pi']:.3f}")      # should match your epsilon input
print(f"C1: {mats['C1']:.3f}")               # noise amplification factor
print(f"Negative weight fraction: {audit.negative_weight_frac_:.2%}")
# Red flag if above 5%
```

### Step 3: Compare fair premiums against current premiums

The discriminatory exposure is the gap between what your current model charges and what the discrimination-free model $h^*$ would charge, conditional on $X$:

```python
# Fair premiums on the full portfolio
h_star = audit.predict_fair_premium(X_full)

# Current model premiums (your live pricing model)
h_current = current_model.predict(X_full)

# Ratio — how much does the current model deviate from the fair benchmark?
premium_ratio = h_current / h_star

# Segment by postcode ethnicity band (ONS Census proxy — still useful here as a diagnostic)
# High-minority postcodes where ratio > 1 indicate potential proxy overcharge
import pandas as pd
df = pd.DataFrame({
    'h_current': h_current,
    'h_star': h_star,
    'ratio': premium_ratio,
    'postcode_bme_decile': postcode_bme_decile,  # from ONS Census 2021 join
})
print(df.groupby('postcode_bme_decile')['ratio'].agg(['mean', 'std', 'count']))
```

This gives you two things. First, the population-level answer to the FCA's question: do your premiums systematically diverge from the discrimination-free benchmark in high-BME postcodes? If the ratio is near 1.0 across deciles, you have quantitative evidence of fairness. If the ratio climbs in high-BME deciles, you have identified the proxy and can quantify its scale.

Second, it tells you whether the postcode factor in your current model encodes something the discrimination-free model would not price — which is exactly the question the FCA's ethnicity pricing research is asking.

### Step 4: Produce the governance artefact

The evidence pack for the model risk register should contain:

- The privacy protocol used (k-RR, $\varepsilon = 1.5$, policyholder-side application)
- Sample size, opt-in rate, and demographic representativeness checks
- Correction quality diagnostics: $C_1$, negative weight fraction, estimated $\pi$
- The generalisation bound from Theorem 4.3: $\hat{R}^{\text{LDP}} \leq R(f^*) + \text{[bound]}$
- The comparison of $h^*$ against current premiums by protected-characteristic segment
- A clear statement of what the uniform $P^*$ reference distribution means: every policy is priced as if drawn from a population with equal ethnic group proportions

```python
result = audit.audit_report()
bound = audit.statistical_bound(delta=0.05)

print(f"Generalisation bound (95% confidence): {bound:.5f}")
print(f"Corrected group prevalences: {result.p_corrected}")
print(f"Negative weight fraction: {result.negative_weight_frac:.2%}")
```

The `audit_report()` method returns a `PrivatizedAuditResult` dataclass. Its fields are the natural content of a fairness audit section in a Consumer Duty assessment.

---

## The anchor-point route if you have no S at all

The above assumes you have collected even a small opt-in sample via k-RR. If you have neither ethnicity labels nor a collection exercise, Procedure 4.5 in Zhang et al. offers an estimation route using anchor covariates.

The idea: if there exist covariate regions where group membership is near-certain — postcodes with >90% population from a single ethnicity group, or very specific occupation/vehicle combinations — a classifier trained on these anchor points can estimate the noise rate $\pi$ from the covariate-predicted group probabilities.

```python
audit = PrivatizedFairnessAudit(
    n_groups=2,
    # epsilon not specified — will estimate pi from anchor points
    reference_distribution="uniform",
    loss="poisson",
    nuisance_backend="catboost",
)
# X_anchor: the submatrix of X in your anchor regions
# S_proxy: group labels derived from postcode (ONS Census 2021 attribution)
audit.fit(X_sample, y_sample, S_proxy, X_anchor=X_anchor)

mats = audit.correction_matrices()
aq = mats.get('anchor_quality')
if aq is not None:
    print(f"Anchor quality: {aq:.3f}")
else:
    print("Anchor quality: not available")
# Below 0.90: unreliable estimate, treat results as directional only
```

For UK motor insurance, postcode anchor quality for ethnicity is typically 0.70–0.85 at district level. Unit postcodes in Leicester, Bradford, Haringey, or Slough can achieve quality approaching 0.90 for some categories. Nationally, it is weaker. If anchor quality falls below 0.85, run a sensitivity analysis across the plausible range of $\pi \in [0.70, 0.85]$ and report the spread of the premium ratio comparison rather than a single point estimate.

This approach is more fragile than direct k-RR collection, but it is the available path if you have no consent mechanism. It is also more defensible than a purely area-level proxy analysis because it uses a statistical model of individual group probability rather than treating area composition as an individual attribute.

---

## What this does not solve

Be precise about the boundaries.

**It is not a GDPR silver bullet.** Whether noised $S$ constitutes Article 9 processing is contested. Whether the anchor-point approach — where the insurer estimates group probabilities from covariates and never holds an S at all — avoids Article 9 processing is more clearly in scope of standard model governance. The anchor-point route has lower legal risk but weaker statistical guarantees.

**It is not an Equality Act defence by itself.** Demonstrating that your discrimination-free benchmark $h^*$ and your current premium are close is evidence, not proof. A persistent ratio above 1.0 in high-BME postcodes still requires explanation — either the risk differential is genuine (and you need to demonstrate that with actuarial evidence) or it is not (and you need to correct it).

**The negative-weight warning matters.** If more than 5% of correction weights go negative before clipping, the algebraic correction is no longer exact and the bound understates the true uncertainty. At $\varepsilon < 1.0$, this is routine for small samples. Do not cite the theoretical bound from a fit with elevated negative weight fraction as if it were valid; report it alongside the diagnostic.

**Uniform $P^*$ shifts the portfolio premium.** Marginalising over equal group proportions rather than the actual portfolio mix changes the expected portfolio premium. After applying the discrimination-free correction, recalibrate your premium loadings before deployment.

---

## Where this fits in the regulatory timeline

The FCA Research Note does not require action today. The FCA explicitly said the £28 residual is "likely" explained by unmeasured risk rather than discrimination. But:

- The Research Note reviewed six million policies — more than 50% of the UK motor market — and the residual persists after full risk adjustment
- The FCA said it will continue to monitor pricing outcomes by ethnicity and postcode composition
- Consumer Duty Outcome 4 (Price and Value) and PRIN 2A require firms to be able to demonstrate fair outcomes, not merely assert them
- The Colorado SB21-169 proxy-discrimination prohibition (in force January 2023) and the EU AI Act fairness provisions show the direction of international regulatory travel

The gap between "likely explained by risk" and "demonstrated to be explained by risk" is exactly the evidence that LDP-based fairness audit closes. Building this capability now — a documented collection protocol, an LDP-corrected model, and a comparison of $h^*$ against current premiums — takes a few weeks of work. Running it in response to a direct supervision enquiry under time pressure is substantially harder.

Firms that have already run these numbers will be explaining a methodology. Firms that have not will be explaining why they do not have one.

---

## References

Zhang, T., Liu, S. and Shi, P. (2025). Discrimination-Free Insurance Pricing with Privatized Sensitive Attributes. arXiv:2504.11775 (v2, July 2025). University of Wisconsin-Madison and UNC-Chapel Hill.

Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2022). Discrimination-Free Insurance Pricing. *ASTIN Bulletin*, 52(1), 55–89.

Warner, S.L. (1965). Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias. *Journal of the American Statistical Association*, 60(309), 63–69.

FCA (2025). Research Note: Motor insurance pricing and local area ethnicity in England and Wales. December 2025. Available at fca.org.uk/publications/research-notes/motor-insurance-pricing-local-area-ethnicity-england-wales

Citizens Advice (2022). Paying Over the Odds: Ethnicity and Car Insurance Pricing.

Equality Act 2010. Section 19 (indirect discrimination) and Section 29 (services and public functions).

UK GDPR Article 9 (as retained in UK law by the European Union (Withdrawal) Act 2018 and DPA 2018).

---

## Related posts

- [Discrimination-Free Pricing When You Don't Have the Protected Attribute](/2026/03/31/discrimination-free-pricing-privatised-attributes-insurance/) — the theory: correction matrices, LRTW connection, UK post-hoc audit approach
- [Pricing Fairly Without Seeing the Data: Local Differential Privacy for Insurance](/2026/04/02/privatized-fair-pricer-ldp-insurance-fairness/) — the `PrivatizedFairPricer` sklearn API
- [LDP Optimal Mechanism for Privatised Fairness Auditing](/2026/04/01/ldp-optimal-mechanism-privatised-fairness-insurance-pricing/) — asymmetric noise allocation for minority groups
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — `PrivatizedFairnessAudit` walkthrough
