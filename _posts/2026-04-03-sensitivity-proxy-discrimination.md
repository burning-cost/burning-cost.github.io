---
layout: post
title: "Per-Policyholder Proxy Discrimination Scores: From Research Metric to Consumer Duty Evidence"
date: 2026-04-03
categories: [fairness, consumer-duty]
tags: [proxy-discrimination, LRTW, Consumer-Duty, FCA, Equality-Act, insurance-fairness-diag, per-policyholder, delta-PD, fair-value, Shapley, Owen-2014, EJOR-2026]
description: "The Lindholm-Richman-Tsanakas-Wüthrich EJOR 2026 paper gives pricing teams something they did not previously have: an instance-level proxy discrimination score for every policy in the book, derived without needing to observe the protected attribute for individual policyholders. This post is about what to do with those scores."
author: Burning Cost
math: true
---

We covered the PD metric itself in [a post from 31 March](/2026/03/31/sensitivity-based-proxy-discrimination-insurance-pricing/). The maths, the constrained QP, the Shapley attribution, the gap in `insurance-fairness-diag`. If you have not read it, start there.

This post is about the part of the Lindholm, Richman, Tsanakas and Wüthrich paper that gets less attention but is arguably more useful for a pricing team doing Consumer Duty work: the per-policyholder proxy discrimination score, $\delta_\text{PD}(x)$. Specifically, what it actually measures, how to translate it into terms that a pricing committee can act on, and how it addresses the specific evidential gap that Consumer Duty creates.

---

## The evidential gap Consumer Duty creates

Consumer Duty does not tell pricing teams what to measure or how. What it does do — through PRIN 2A.3 and the fair value requirement — is create an obligation to monitor outcomes for groups of customers sharing protected characteristics, and to demonstrate that pricing produces fair value across those groups.

The standard response is distributional analysis: compute the mean premium for each D group, compare distributions, run a Kolmogorov-Smirnov test. This is fine for demographic parity in the aggregate. It is not sufficient for the question Consumer Duty is actually asking.

The Equality Act Section 19 test for indirect discrimination requires a particular provision, criterion or practice (PCP) that *disadvantages* individuals sharing a protected characteristic. In insurance pricing, the PCP is the rating structure — the specific combination of factors and interactions that produces each individual's premium. A finding that "average premiums for ethnicity group D=2 are 4.3% higher than for group D=1" is a distributional observation. It does not identify which policies are disadvantaged by the rating structure, or by how much, or which rating factors are responsible.

$\delta_\text{PD}(x)$ does.

---

## What the score is

Recall the PD metric from Definition 4 of the LRTW paper. The constrained regression finds the closest admissible price — the price from the nearest point in the set of proxy-discrimination-free pricing functionals:

$$\pi^*(x) = c^* + \sum_{d \in \mathcal{D}} \mu(x, d)\, v^*_d$$

where $v^*$ are the optimal simplex weights (the same across all x) and $c^*$ is the optimal intercept. The discrimination residual for an individual with covariate profile x is:

$$\delta_\text{PD}(x) = \pi(x) - \pi^*(x) = \Lambda(x)$$

This is Definition 9, Equation 16 of the LRTW paper. It is the per-policyholder proxy discrimination score.

Three things matter about it.

**First: it is signed.** Positive $\delta_\text{PD}(x)$ means the actual premium exceeds the nearest discrimination-free benchmark — the policyholder is *overpriced* relative to what an admissible model would charge. Negative means underpriced. In a portfolio with material proxy discrimination, some groups will be systematically positive and others systematically negative. The global PD metric is the variance of $\delta_\text{PD}$ normalised by the variance of $\pi(X)$.

**Second: it is a function of X only.** You do not need to know D for a specific policyholder to compute their $\delta_\text{PD}$ score. The score is derived from the constrained regression of $\pi(X)$ on the group-specific best-estimate prices $\{\mu(X, d) : d \in \mathcal{D}\}$. Those models are estimated at portfolio level using the observed D values in the training data. But once $v^*$ and $c^*$ are determined, $\delta_\text{PD}(x)$ is evaluated from x alone.

This matters enormously for UK personal lines. Most insurers do not hold ethnicity data for individual policyholders. You can estimate PD and compute $\delta_\text{PD}$ for the whole book without needing per-policy D records. The portfolio-level D observations (which you may have, or may be able to derive from census linkage by postcode) are sufficient to estimate the $\mu(x, d)$ models.

**Third: it is in the same currency as the premium.** The score is the absolute premium difference between the fitted price and the nearest discrimination-free benchmark, in pounds (or as a relativities proportion). It is not a normalised index, a percentile rank, or a correlation coefficient. It is the number you put in front of a pricing committee or an FCA supervisor.

---

## Translating to monetary terms

The PD global metric is dimensionless — it normalises by $\text{Var}(\pi(X))$. A value of 0.003 (the LRTW case study result, n = 165,511 policies) requires translation to be meaningful in a fair value context.

The natural translation is the mean absolute discrimination overcharge:

$$\bar{\delta} = \mathbb{E}[\delta_\text{PD}(X)^+] = \mathbb{E}[\max(\Lambda(X), 0)]$$

This is the expected overcharge for policies receiving a positive score. For a book where the total premium is £800 and PD = 0.003, the implied $\bar{\delta}$ is of order £5–15 per policy per year for the affected segment — depending on the premium variance and the distribution of Lambda. That is a number a committee can evaluate against a fair value judgement.

The converse — $\mathbb{E}[\delta_\text{PD}(X)^-]$ — is the expected undercharge for the offsetting group. If proxy discrimination is zero-sum within the portfolio (the overcharges and undercharges net out at portfolio level, as they must since $\mathbb{E}[\Lambda] = 0$ by construction), then every pound of discriminatory overcharge in one segment has a corresponding undercharge elsewhere.

The implication is uncomfortable: a model can be actuarially balanced in aggregate and simultaneously be systematically overcharging one demographic segment and undercharging another. The A/E ratio does not see this. Claims frequency monitoring does not see this. The composite loss ratio does not see this. $\delta_\text{PD}$ does.

---

## What the FCA December 2024 research note is actually asking for

The FCA's December 2024 research note on bias in supervised machine learning is not a regulatory requirement. It does not prescribe a methodology. It does not set thresholds. But it signals where the FCA's thinking is heading, and the signal is fairly clear.

Three statements in the note map directly to what LRTW provides.

"A feature which causes bias must be associated with a demographic characteristic... but this is not sufficient for contributing to bias as it must also be included in and important to a model's predictions." This rules out correlation-only proxy detection. The FCA is looking for evidence that a factor both associates with D *and* transmits that association into prices. The PD metric does exactly this through the constrained regression: a factor that correlates with ethnicity but has no effect on price variance contributes nothing to PD.

"Identify which variables most influence the racial bias of a credit risk model." This is the demand for Shapley factor attribution. The LRTW Owen 2014 Shapley decomposition produces $\text{PD}^\text{sh}_i$ values that sum to the global PD metric. "Postcode accounts for 0.0009 of the 0.003 total PD" is the form of answer the research note is pointing towards.

"Associations in aggregated data may not exist at individual level." This is the Simpson's Paradox observation, and it is the direct motivation for $\delta_\text{PD}(x)$ as an instance-level score. If a group-level analysis shows higher average premiums for D=2, but the per-policyholder scores show that the high-premium D=2 policies are genuinely high-risk (legitimate variation) whilst the low-premium D=2 policies are paying below the discrimination-free benchmark, aggregate analysis has misidentified the direction of the problem. $\delta_\text{PD}$ resolves this.

None of this means the FCA will require firms to use the LRTW framework. As of April 2026, there is no prescribed methodology and no materiality threshold. But it is the strongest academic answer to the regulatory question. If a firm is using correlation-only proxy detection for a Consumer Duty fair value assessment, the gap between what they are doing and what the FCA research note implies is significant.

---

## The local demographic unfairness score: what it adds

There is a second local measure in the LRTW paper — the local demographic unfairness score $\delta_\text{UF}(x, d)$, Definition 8. This requires knowing D for individual policyholders (or at least the subgroup structure), because it uses output optimal transport to map the premium distribution for group d onto a demographic-parity target distribution:

$$\delta_\text{UF}(x, d) = \pi(x) - \tilde{\pi}(x, d)$$

where $\tilde{\pi}(x, d) = G^{-1}(G_d(\pi(x)))$ maps the premium to the target distribution by quantile. Unlike $\delta_\text{PD}$, this measure enforces distributional parity rather than proximity to the admissible set. For binary D where group D=0 has a stochastically lower premium distribution than D=1, every D=1 policyholder will have $\delta_\text{UF} \geq 0$ (overcharged relative to the parity benchmark) and every D=0 policyholder will have $\delta_\text{UF} \leq 0$.

The two measures answer different questions.

$\delta_\text{UF}(x, d)$ asks: given that this person is in demographic group d, how much are they overcharged relative to a world with demographic parity in the premium distribution? This is a demographic parity question. It can assign an "overcharge" to a policyholder even when the price difference is entirely explained by legitimate risk variation — if group D=1 genuinely has higher claims risk, moving to demographic parity requires cross-subsidising from D=0 to D=1.

$\delta_\text{PD}(x)$ asks: given this person's covariate profile x, how far is their premium from the nearest price that avoids proxy discrimination? This is a proxy discrimination question. It is non-zero only when the pricing structure cannot be explained by x-independent weighting of the group-specific risk models.

Under Consumer Duty and Equality Act indirect discrimination analysis, the relevant measure is $\delta_\text{PD}$, not $\delta_\text{UF}$. Proxy discrimination is a structural property of the rating system. Demographic disparity — which $\delta_\text{UF}$ measures — may be entirely legitimate if it reflects genuine risk differences between groups. The Equality Act permits pricing that has a disparate impact when there is a proportionate means of achieving a legitimate aim. Actuarial risk pricing is generally that aim. $\delta_\text{UF}$ blends legitimate risk variation with discriminatory effects; $\delta_\text{PD}$ isolates the discriminatory component.

We think $\delta_\text{UF}$ is the right metric if your question is "are premium outcomes equitable across groups?" We think $\delta_\text{PD}$ is the right metric if your question is "is the rating structure itself discriminatory?" For regulatory compliance under UK law, the second question is the more legally relevant one.

---

## What a Consumer Duty fair value file should contain

We are not going to tell you what the FCA expects — they have not told us. But here is what a well-constructed proxy discrimination section of a Consumer Duty fair value assessment looks like using the LRTW framework.

**1. Global PD metric with bootstrap confidence interval.** A single number, e.g. PD = 0.0031 (95% CI: 0.0024–0.0038). The interval matters: for a UK personal lines book with n = 30,000 policies, PD = 0.003 may not be statistically distinguishable from zero. The confidence interval tells you whether you have a finding or noise.

**2. Shapley attribution to rating factors.** A ranked table of $\text{PD}^\text{sh}_i$ values, summing to PD. Highlight any factor with $|\text{PD}^\text{sh}_i| / \text{PD} > 10\%$ as material. Identify any factors with negative attributions — these are attenuating proxy discrimination, and removing them might make things worse.

**3. Per-policyholder score distribution.** The distribution of $\delta_\text{PD}(x)$ across the book. Mean should be near zero (by construction). Median, 5th and 95th percentiles, and the proportion of policies with $|\delta_\text{PD}(x)| > \varepsilon$ for a materiality threshold $\varepsilon$ (e.g., 5% of mean premium).

**4. Segment-level analysis.** Cross-tabulate mean $\delta_\text{PD}$ by the segments most associated with protected characteristics: postcode deprivation decile, occupation group, age band. You are looking for systematic patterns — not random noise around zero, but consistent overcharging in one segment and undercharging in another.

**5. Monetary translation.** Mean overcharge per policy in the positively-affected segment, expressed in pounds. A statement of the form: "the pricing structure produces an estimated average overcharge of £X per year for the [segment], relative to the nearest proxy-discrimination-free benchmark, compared to an average undercharge of £Y for the [contrasting segment]."

**6. Actuarial justification for high-attribution factors.** For any factor contributing more than 10% of PD, a brief explanation of why the factor is in the model and what risk-based evidence supports its inclusion. This is the proportionate means test under the Equality Act: not "we included postcode because it predicts claims" but "here is the claims frequency data by postcode, here is the correlation with the non-protected risk drivers, and here is the business case for including it despite its proxy-discrimination attribution."

---

## The implementation gap, briefly

The existing `insurance-fairness-diag` library computes a D_proxy metric equal to $\sqrt{\text{UF}}$ — the square root of the demographic unfairness Sobol index. This is documented in `_admissible.py`. It is not PD. The two converge only when X and D are independent, which is precisely the condition that makes proxy discrimination impossible.

The `_local.py` module does compute local scores, but against the group-mean approximation, not the constrained QP. The per-policyholder scores it outputs are correspondingly approximate.

The correct implementation requires: (1) best-estimate prices $\mu(x, d)$ for all D values in the training data; (2) the constrained SLSQP regression from Definition 4; and (3) Lambda = $\pi(X) - c^* - \sum_d \mu(X,d) v^*_d$ as the per-policyholder score. The code for step (2) is in the [31 March post](/2026/03/31/sensitivity-based-proxy-discrimination-insurance-pricing/). Steps (1) and (3) follow directly.

The unit test is Example 3 from the paper: D ∈ {0,1}, X ~ U(0,1), $\mathbb{P}(D=1|X) = X$, $\mu(X,D) = 1/2 + X + D$. Analytical solution: PD = 0.25, and $\delta_\text{PD}(x) = -1/4 + x/2$ (linear in x, zero at the median). If your implementation returns this, the rest follows.

---

## What PD does not tell you

High PD is a finding, not a verdict. A postcode variable that contributes 30% of total PD might be there because postcode captures genuine claims risk variation — flood zones, urban theft rates, commuter density — that the model cannot fully explain with less geographically-granular factors. The Equality Act permits this if the insurer can show a proportionate means of achieving a legitimate aim.

PD does not have a causal interpretation. A large $\delta_\text{PD}(x)$ for a policyholder in a high-deprivation postcode could mean the model is proxy-discriminating on ethnicity. It could also mean the model is correctly capturing risk variation that happens to correlate with deprivation. The PD framework identifies that the price structure cannot be explained by x-independent weighting of the group-specific risk models. The actuary has to explain why.

The causal question — does the X–D correlation in the pricing structure reflect legitimate risk or discriminatory effect — requires separate analysis. Causal fairness methods (the Cote 2025 framework we have [covered separately](/2026/03/26/proxy-race-distortion-fairness-audits/)) can help if you can specify a defensible causal DAG. For most UK personal lines books, that is not straightforward.

Our position: run PD as the primary measurement. Flag any factor where $\text{PD}^\text{sh}_i / \text{PD} > 10\%$ for review. Build the actuarial justification for those factors. Document the $\delta_\text{PD}$ distribution and its monetary translation. That is a defensible Consumer Duty fair value assessment for proxy discrimination. It is not a guarantee of regulatory compliance — the FCA has not specified what compliance looks like — but it answers the right questions.

---

## Reference

Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2026). Sensitivity-based measures of discrimination in insurance pricing. *European Journal of Operational Research*. DOI: [10.1016/j.ejor.2026.01.021](https://doi.org/10.1016/j.ejor.2026.01.021). Open access: [City Research Online](https://openaccess.city.ac.uk/id/eprint/36642/).

FCA (2024). Research Note: A Literature Review on Bias in Supervised Machine Learning for Financial Services. 11 December 2024. [fca.org.uk](https://www.fca.org.uk/publications/research-notes/research-note-literature-review-bias-supervised-machine-learning)
