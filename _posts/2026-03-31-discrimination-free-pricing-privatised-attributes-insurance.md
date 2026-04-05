---
layout: post
title: "Discrimination-Free Pricing When You Don't Have the Protected Attribute"
date: 2026-03-31
categories: [fairness]
tags: [fairness, gender, GIC, Test-Achats, local-differential-privacy, LDP, discrimination-free-pricing, LRTW, correction-matrices, insurance-fairness, PrivatizedFairnessAudit, Equality-Act, FCA, proxy-discrimination, python, math]
description: "Zhang, Liu and Shi (arXiv:2504.11775, 2025) extend discrimination-free pricing to the case where you only have a noisy privatised version of the protected attribute. The correction is algebraically exact. The UK application — auditing post-2012 gender proxy discrimination without gender data — is immediately practical."
math: true
---

The standard academic solution to proxy discrimination in insurance pricing assumes you have the protected attribute at training time. You observe gender (or ethnicity, or disability status) for each policyholder, you train group-specific risk models, and you average them over a fixed reference distribution that does not vary by covariate profile. The result is a discrimination-free premium in the Lindholm-Richman-Tsanakas-Wüthrich (2022) sense: every policyholder is priced as if drawn from a population with equal group proportions, regardless of what their postcode or occupation might imply about their likely group membership.

The UK insurance market has a specific problem with this approach: since December 2012, UK insurers cannot use gender in motor and household pricing. Many have removed gender from their training datasets entirely. The attribute is not privatised — it is absent. You cannot train LRTW group-specific models if you do not have the group labels.

A paper from April 2025 — Tianhe Zhang, Suhan Liu and Peng Shi, arXiv:2504.11775, from Wisconsin-Madison and UNC-Chapel Hill — extends the LRTW framework to exactly this case. The result is already in [`insurance-fairness` v0.3.8](https://github.com/insurance-ai/insurance-fairness) as `PrivatizedFairnessAudit`. This post explains the mechanism and what it is actually useful for in practice.

---

## How gender disappeared from UK motor pricing

The backstory is worth getting right, because it shapes what "unavailable" means.

In March 2011, the European Court of Justice ruled in *Test-Achats* (Case C-236/09, ECJ, 1 March 2011) that Article 5(2) of the Gender Directive — the provision allowing member states to grant insurers a permanent derogation to use actuarially justified gender data in pricing — was incompatible with the EU Charter of Fundamental Rights. The derogation had no time limit. A permanent exception to a fundamental equality principle is self-defeating, the court reasoned. The provision was struck down with effect from 21 December 2012.

The UK implemented this via the Equality Act 2010 (Amendment) Regulations 2012 (SI 2012/2992), which repealed the insurance exception in paragraph 22 of Schedule 3 to the Equality Act 2010. The UK is now post-Brexit and not bound by ECJ precedent, but Parliament has not acted to restore gender pricing. The Equality Act prohibition remains.

What remained lawful after December 2012 is worth noting: individual risk factors that happen to correlate with gender — mileage, occupation, engine capacity, vehicle type — are still permitted, provided actuarially justified. Which is, of course, exactly the proxy discrimination problem. A model that has never seen gender can still effectively price by gender if it uses features whose distribution differs materially between male and female drivers. Young males in high-cc vehicles are not penalised for being male; they are penalised for being young, for driving a powerful car, and for covering high mileage — but those three features collectively explain most of the gender gap in motor claims. The LRTW 2022 paper formalised why this constitutes proxy discrimination. The Zhang-Liu-Shi paper provides the audit mechanism to measure it without gender data.

The ethnicity case is structurally similar but worse. UK P&C insurers do not collect individual ethnicity. The FCA's December 2025 Research Note on motor insurance pricing and local area ethnicity identified a £307 raw annual premium gap between high-minority-concentration and white-majority postcodes for motor insurance, with a residual £28 gap unexplained after risk adjustment. No insurer can demonstrate this residual is zero because no insurer holds individual ethnicity data against which to run the test.

---

## The mechanism: noised labels and correction matrices

The core insight in Zhang-Liu-Shi is that you do not need to observe the true attribute D at training time. You need a noised version S that satisfies a known noise model. Then you can correct for the noise algebraically.

The noise model the paper uses is randomised response, the Warner (1965) mechanism. Each individual observes their own D and reports S, where:

$$\mathbb{P}(S = s \mid D = d) = \begin{cases} \pi & \text{if } s = d \\ \bar{\pi} & \text{if } s \neq d \end{cases}$$

For a binary attribute ($K = 2$) and privacy parameter $\varepsilon$:

$$\pi = \frac{e^\varepsilon}{1 + e^\varepsilon}, \qquad \bar{\pi} = \frac{1}{1 + e^\varepsilon}$$

This satisfies $\varepsilon$-local differential privacy: an observer who sees S cannot distinguish whether D = 0 or D = 1 with probability better than $e^\varepsilon$. The "local" part matters — each respondent applies the noise to their own data before any central collection. The insurer receives S, not D. The LDP guarantee holds even if the insurer is the adversary.

The standard LRTW training objective — the one that requires individual D labels — is:

$$R(f_1, \ldots, f_K) = \sum_k \mathbb{E}_{Y,X \mid D=k}\!\left[L(f_k(X), Y)\right] \cdot \mathbb{P}(D = k)$$

Lemma 4.2 in Zhang-Liu-Shi shows this can be rewritten using only S. The LDP transition matrix $T$ has entries $T_{sd} = \mathbb{P}(S=s \mid D=d)$. Its inverse $T^{-1}$ has:

$$T^{-1}_{ii} = \frac{\pi + K - 2}{K\pi - 1}, \qquad T^{-1}_{ij} = \frac{\pi - 1}{K\pi - 1} \quad (i \neq j)$$

The corrected objective replaces expectations over $D = k$ groups with reweighted expectations over $S = j$ groups, using $T^{-1}$ to invert the noise. In practice this amounts to: train each group model $f_k$ on the full dataset, with sample weight $\Pi^{-1}_{k, S_i}$ for observation $i$. The weights up-weight observations whose noised label makes them likely to be in group $k$, and down-weight the rest. Some weights are negative — this is expected and is handled by clipping to zero with a diagnostic warning when more than 5% of observations are affected.

The fair premium is then:

$$h^*(X) = \sum_k f_k(X) \cdot \mathbb{P}^*(D = k)$$

where $\mathbb{P}^*$ is a fixed reference distribution chosen before training. For UK gender-neutral motor pricing, uniform $\mathbb{P}^*(D = k) = 1/2$ is the right choice: it prices every individual as if drawn from an equal-gender population. The correlation between X and D in the training data is irrelevant — the weights are fixed, not covariate-conditional.

The correction is algebraically exact at the population level. The approximation enters only in the sample-based estimation. Theorem 4.3 gives the finite-sample bound:

$$\hat{R}^{\text{LDP}}(f) \leq R(f^*) + O\!\left(C_1 \cdot K^2 / \sqrt{n}\right)$$

where $C_1 = (\pi + K - 2)/(K\pi - 1)$ is the noise amplification factor. For $K = 2$, $\varepsilon = 1$ (so $\pi \approx 0.73$), $C_1 \approx 1.73$: about 73% additional statistical uncertainty from the noise correction, relative to training directly on D. For a UK motor portfolio of 100,000 policies, this is manageable.

The amplification becomes punishing as $\varepsilon \to 0$ (strong privacy, $\pi \to 1/K$, $C_1 \to \infty$) and vanishes as $\varepsilon \to \infty$ (no privacy, $\pi \to 1$, $C_1 \to 1$). The practical range for insurance applications is $\varepsilon \in [1, 2]$: meaningful formal privacy guarantees with acceptable accuracy cost on portfolios of $n > 10{,}000$.

---

## What to do when you have no S at all

The LDP framework requires someone to have collected S via randomised response. Most UK insurers do not have this — they have no individual-level gender data at all post-2012, and never had individual ethnicity data.

Procedure 4.5 in the paper addresses this with anchor-point estimation. The idea: if there is some covariate region where $\mathbb{P}(D = k \mid X = x) \approx 1$ — an "anchor" region where group membership is near-certain from observed characteristics — then $\mathbb{P}(S = k \mid X = x) \approx \pi$ in that region. Train a classifier to predict S from X (using imputed or inferred group labels — census-matched postcodes, surname analysis, or historical pre-ban gender records), find the maximum predicted probability, and recover $\pi$.

For binary gender in UK motor: young males in high-cc vehicles are a near-anchor for the male group, but the mid-age family-car population is ambiguous. Anchor quality — defined as the maximum predicted $\mathbb{P}(S = k \mid X)$ across the portfolio — will typically be in the 0.70–0.85 range. Below 0.90, the implementation raises a warning. The resulting pi estimate is imprecise, and the Theorem 4.3 bound understates the true uncertainty.

For ethnicity from postcode: high-concentration areas like Leicester (Asian population >40%) or Haringey can give anchor quality ~0.85. Nationally, the estimate is weak. This is the honest limitation.

---

## Using PrivatizedFairnessAudit

The implementation in `insurance-fairness` v0.3.8 covers the main use cases:

```python
from insurance_fairness import PrivatizedFairnessAudit

# Known epsilon from a trusted third party (TTP)
audit = PrivatizedFairnessAudit(
    n_groups=2,
    epsilon=2.0,                       # pi = exp(2)/(1+exp(2)) = 0.880
    reference_distribution="uniform",  # P*(D=k) = 0.5 for each group
    loss="poisson",                    # claim frequency
    nuisance_backend="catboost",
)
audit.fit(X_train, y_train, S_train)
h_star = audit.predict_fair_premium(X_new)
report = audit.audit_report()          # PrivatizedAuditResult dataclass

# pi unknown — estimate from anchor covariates
audit = PrivatizedFairnessAudit(n_groups=2)
audit.fit(X_train, y_train, S_train, X_anchor=X_train)

# Diagnostics
print(audit.statistical_bound(delta=0.05))
print(audit.minimum_n_recommended())
mats = audit.correction_matrices()    # T_inv, Pi_inv, C1, pi
```

The `correction_matrices()` call is the useful diagnostic: it exposes $T^{-1}$, $\Pi^{-1}$, $C_1$, and the estimated $\pi$. If $C_1$ is large — say above 2.5 — the weights are volatile and the per-policy fair premium estimates will have wide uncertainty bands.

The `reference_distribution="uniform"` option is the right choice for UK gender-neutral pricing. Switching to `"empirical"` uses the noise-corrected marginal $\mathbb{P}(D = k)$ estimated from S — appropriate for actuarial applications where you want to preserve portfolio-level expected loss, but less defensible as a fairness guarantee because the marginal embeds the actual portfolio gender split rather than a neutral target.

---

## The UK post-hoc audit approach

The near-term value in the UK market is not deploying a live LDP-mediated pricing system — no regulatory framework yet supports the required trusted-third-party architecture. The value is as a post-hoc audit tool.

A UK insurer with historical pre-ban gender data can run this analysis:

1. Draw a sample of historical policies with true gender D available (pre-December 2012 data).
2. Simulate privatised labels S by applying the randomised response mechanism at a chosen $\varepsilon$ — this converts the available D into the format the method expects.
3. Fit the current post-2012 pricing model (no gender) and the discrimination-free model $h^*$ to the same data.
4. Compare $h^*(X)$ against the current premium to quantify the residual gender proxy.

This answers the question insurers currently cannot answer cleanly: not "does our model correlate with gender?" but "what would our prices look like if we actively neutralised the gender proxy?" The gap between the current premium and $h^*$ is the measurable proxy discrimination exposure.

For ethnicity, where no historical individual data exists, the anchor-point path is the only option — with the caveat that anchor quality below 0.85 makes the correction estimates unreliable. The post-hoc audit in this case should be accompanied by explicit uncertainty quantification on $\pi$, which the current implementation's `statistical_bound()` method provides (with the caveat that it uses the Theorem 4.3 formula even for the unknown-pi case, understating the true bound).

---

## The architecture the paper actually describes

The Zhang-Liu-Shi paper proposes a two-party architecture that is more ambitious than the audit use case. The insurer transmits transformed features $(X_\text{tilde}, Y)$ to a trusted third party (TTP). The TTP holds privatised attributes $S_i$ collected directly from policyholders via randomised response, trains the correction-matrix-weighted group models $f_k$, and returns the fair premium predictions $h^*(X_\text{tilde})$ to the insurer. The insurer never sees D; the TTP never sees the final premium. Privacy is formally guaranteed in both directions.

For UK motor insurance this would require: a GDPR Article 9 legal basis for the TTP to collect gender or ethnicity from policyholders; a designated TTP (regulator, industry bureau, or auditor); and insurers willing to share claims data commercially. None of these conditions currently holds. The closest analogy would be the FCA mandating a centralised industry bureau — comparable in function to the Motor Insurers' Bureau — with statutory powers to process sensitive attribute data for discrimination testing. This is a plausible regulatory direction given EP25/2 and the Consumer Duty fair value obligations, but it is hypothetical as of today.

The implementation correctly documents that it supports only single-party operation. The multi-party architecture is absent, and this is the right choice for a library that can actually be deployed.

---

## Limitations worth stating plainly

**Negative weights.** When $\varepsilon$ is small (strong privacy), $C_1$ grows and a meaningful fraction of training observations receive negative correction weights. The current implementation clips these to zero and warns if the fraction exceeds 5%. Clipping introduces bias — the correction is no longer algebraically exact. For $\varepsilon < 0.5$ on any realistically sized UK portfolio, the negative-weight fraction makes the correction unreliable.

**Theorem 4.6 understatement.** When $\pi$ is estimated via anchor points rather than known from a TTP disclosure, the correct statistical bound is Theorem 4.6 in the paper, which includes an additional penalty term $\tilde{\varepsilon}$ capturing uncertainty in the $\pi$ estimate. The current implementation uses Theorem 4.3 for both cases. The reported `statistical_bound()` value understates the true uncertainty in the unknown-$\pi$ case. We have raised this as a known gap in the implementation notes.

**Anchor quality in practice.** Gender from UK motor covariates typically achieves anchor quality 0.70–0.85. Ethnicity nationally achieves lower. Below 0.90, the pi estimate and hence the correction matrices are imprecise. The fair premium $h^*$ inherits this imprecision. A sensitivity analysis over the plausible range of $\pi$ — running the audit at $\pi \in \{0.70, 0.75, 0.80, 0.85\}$ and examining the spread of $h^*(X)$ — is better practice than treating a single estimated $\pi$ as exact.

**GDPR status of S.** Does receiving a privatised label S — D passed through randomised response — constitute processing of special category data under GDPR Article 9? The answer is legally contested. The ICO's March 2025 anonymisation guidance does not classify LDP-protected data as automatically anonymised. Until the ICO provides explicit guidance, there is legal risk in treating S as freely processable. For audit purposes using synthetic S generated from historical D that the insurer already holds, this risk is lower.

**Multi-K scaling.** For $K > 2$ — ethnicity coded to five groups, say — the noise amplification $C_1$ at $\varepsilon = 1$ is approximately 3.7, and the statistical bound scales $O(K^2)$. Sample requirements become demanding. The practical limit for this correction approach is $K = 2$ or $K = 3$ on UK portfolio sizes.

---

## How it relates to the broader LRTW framework

The distinction between Zhang-Liu-Shi and LRTW 2022 is precise. LRTW defines the target — $h^*(X) = \sum_k f_k(X) \cdot \mathbb{P}^*(D = k)$ — and shows how to train $f_k$ when individual D labels are available. Zhang-Liu-Shi prove that the same target can be reached when you have only the noised S, by applying correction matrices to the training objective. They adopt the LRTW definition wholesale. The contribution is the privatised-label training procedure and its statistical guarantees.

If you have individual D labels at training time (historical pre-ban data, or volunteered self-report with consent), use the existing `DiscriminationFreePrice` class via `insurance_fairness.optimal_transport`. If you have noised S from a TTP or can simulate S from historical D, use `PrivatizedFairnessAudit`. If you have neither and want to attempt anchor-point estimation from proxy covariates, use `PrivatizedFairnessAudit` with `X_anchor` set — but treat the results as indicative rather than certified.

The LRTW 2026 sensitivity-based paper (EJOR, January 2026) provides the scalar PD metric that measures how far any pricing model — including $h^*$ — sits from the admissible set. Running the PD metric on the output of `PrivatizedFairnessAudit` is the correct post-fit check: it tells you whether the fair premium you produced is actually admissible, or whether the correction was imprecise enough to leave residual proxy discrimination. We [covered the PD metric here](/2026/03/31/sensitivity-based-proxy-discrimination-insurance-pricing/).

---

## Reference

Zhang, T., Liu, S. and Shi, P. (2025). Discrimination-Free Insurance Pricing with Privatized Sensitive Attributes. arXiv:2504.11775. Submitted April 16, 2025; revised July 14, 2025. Wisconsin-Madison and UNC-Chapel Hill.

Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2022). Discrimination-Free Insurance Pricing. *ASTIN Bulletin*, 52(1), 55–89.

Warner, S.L. (1965). Randomized Response: A Survey Technique for Eliminating Evasive Answer Bias. *Journal of the American Statistical Association*, 60(309), 63–69.

*Association Belge des Consommateurs Test-Achats ASBL v Conseil des Ministres*, Case C-236/09. ECJ, 1 March 2011.

Equality Act 2010 (Amendment) Regulations 2012, SI 2012/2992. In force 21 December 2012.
