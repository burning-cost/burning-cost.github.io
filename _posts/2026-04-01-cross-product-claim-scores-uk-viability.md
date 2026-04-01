---
layout: post
title: "Cross-Product Claim Scores: Do They Work in the UK?"
date: 2026-04-01
categories: [techniques, pricing]
tags: [credibility, BMS, claim-scores, multi-product, GAM, splines, Verschuren, Boucher, Consumer-Duty, NCD, direct-writers, motor, home, UK-pricing, insurance-credibility, a-posteriori-rating, aggregator-market, churn]
description: "Verschuren (2021) showed that a Dutch insurer's home claim history predicts motor risk, and vice versa. The framework is technically clean. The UK structural context — aggregator market, high churn, Consumer Duty — makes it viable for a narrow set of writers, and only just."
math: true
author: burning-cost
---

If a policyholder has had two home insurance claims in three years, is their motor risk higher than their postcode and vehicle suggest? Verschuren (2021) studied exactly this question on a Dutch P&C portfolio and concluded: yes, and meaningfully so. The cross-product signal — using home claim history to sharpen motor pricing and vice versa — improved risk classification enough to be described as "considerably more profitable" than single-product a posteriori rating.

The paper is technically clean. The Dutch evidence is real. And yet we think it is, right now, of limited practical relevance to most UK personal lines pricing teams. This post explains the framework, the evidence, and the specific reasons why UK application is constrained — and where it is not.

---

## The problem the paper solves

Standard GLM pricing is a priori: it uses characteristics known at quote time — age, vehicle, postcode, NCD level. These are fixed at inception and updated only at renewal. As a policyholder accumulates history with an insurer, that history becomes progressively informative, but incorporating it is non-trivial. You cannot simply add a "claims in last N years" flag without grappling with how to weight recent versus distant claims and how to structure the updating recursion.

The Bonus-Malus-panel (BM-panel) framework, developed by Boucher and Inoussa (2014, ASTIN 44(3)) and refined by Boucher and Pigeon (2018), solved this for the single-product case. Rather than a step-function NCD level, you maintain a continuous claim score — a recursively updated sufficient statistic of the policyholder's claim history, with geometric decay so recent claims dominate. This score enters the GLM as a feature, estimated non-parametrically.

Verschuren (2021) takes this one step further: if you have the same customer's motor and home claim history, why limit the signal to a single product? The multi-product extension feeds each product's claim score into the other product's GLM, capturing the shared risk information that cross-product experience contains.

---

## The mathematical structure

Let policyholder $i$ hold product $j \in \{1, 2\}$ (motor and home). Let $Y_{ijt}$ be the claim count for product $j$ in year $t$. The dynamic claim score $S_{ijt}$ is defined recursively — it is a weighted accumulation of past claims on product $j$, with exponential decay weighting recent claims more heavily. This is directly analogous to the Ahn et al. (2023) Poisson-gamma state-space recursion that our [`insurance-credibility`](/insurance-credibility/) library implements for the single-product case.

The multi-product GLM is:

$$\log \mathbb{E}[Y_{ijt}] = x_{ijt}' \beta_j + g_j(S_{ijt}) + \sum_{k \neq j} h_{jk}(S_{ikt})$$

where $g_j(\cdot)$ is the non-linear effect of the policyholder's own-product claim score, and $h_{jk}(\cdot)$ is the non-linear effect of their claim score on product $k$. Both are estimated as natural cubic splines within a GAM framework — penalised IRLS, computationally standard.

The splines matter. The cross-product relationship is not linear: a policyholder with zero home claims is unremarkable; one home claim is a modest signal; two home claims in three years is a materially stronger signal for motor risk. A linear term would miss this non-linearity in the tails, exactly where the signal is most useful.

---

## What the Dutch data showed

Verschuren's empirical work used a matched motor and home portfolio from a Dutch P&C insurer. The full numerical results are behind the Cambridge Core paywall, but the findings are clear in direction: cross-product claim scores significantly improve risk classification on both products. The Gini index improvement over a baseline GLM was described as considerable. The framework was reported as considerably more profitable than single-product a posteriori rating.

We are not going to dress this up as stronger evidence than it is. The paper does not disclose Gini uplift numbers in the abstract, the Dutch portfolio details are opaque, and there is no independent replication on UK data. What we have is a technically well-constructed paper from a credible journal (ASTIN Bulletin), published in January 2021, showing that cross-product signals work in one European direct-writer context.

The mechanism is plausible enough that we do not need to explain it away: some policyholders are systematically more claim-prone across all their insurance. If you have two years of home claim data and three years of motor data on the same person, the combined signal is sharper than either in isolation. That is what Verschuren demonstrates.

---

## Why the UK is harder

### 1. The aggregator market breaks the household link

Approximately 55–60% of UK private motor policies are sourced through price comparison websites: GoCompare, Compare The Market, MoneySuperMarket, and Confused.com. For aggregator-sourced business, the insurer has no prior relationship with the customer and holds no cross-product data. The household link that makes Verschuren's framework work simply does not exist at inception for the majority of new motor business.

This is not a solvable data problem — it is structural. An insurer quoting via an aggregator does not know whether the applicant has a home policy elsewhere, let alone what that policy's claim history looks like. Cross-product claim scores are irrelevant for this channel.

### 2. High churn degrades claim score quality

UK motor churn runs at roughly 25% annually — among the highest in Europe, largely because the aggregator channel makes switching frictionless. Verschuren's recursive claim score requires several years of policy history at the same insurer to become meaningful. A policyholder who has been with a motor insurer for 14 months has a thin claim score; if they also moved their home policy two years ago, the matched cross-product panel is even shorter.

The Dutch market — where the paper's data originates — almost certainly has lower structural churn given a different distribution architecture. The evidence for cross-product signals is from a market where those signals can accumulate. In the UK, for much of the market, they cannot.

### 3. Consumer Duty (FCA PS22/9) adds governance overhead

Using a customer's home claim history to adjust their motor renewal premium is a pricing linkage that requires explicit documentation under Consumer Duty. The fair value obligation (PRIN 12) means the firm must be able to justify to a regulator — and, if asked, to the customer — why a home claim three years ago is reflected in a higher motor quote today.

This is not insurmountable. The causal mechanism is defensible: claim propensity is a real risk characteristic. But it requires pricing governance work that does not currently exist at most firms, and it creates a communication burden at renewal if customers query the basis. Insurers that have struggled to explain existing rating factors to Consumer Duty reviewers will find cross-product pricing linkage a difficult conversation.

### 4. NCD rigidity

The UK motor NCD framework — the 0%, 20%, 30%, 40%, 50%, 65% step system — is embedded in policy terms, reinsurance structures, and customer expectations. Verschuren's continuous claim score does not replace NCD; it operates alongside it. But adding a parallel a posteriori adjustment mechanism alongside an existing one adds pricing complexity without removing the existing machinery. The two-score system is not impossible to explain, but it makes an already complex renewal pricing narrative harder.

---

## Where it is actually viable

The barriers above apply to most of the UK market. They do not apply equally to everyone.

**Direct writers with matched motor and home books** sit in a different position. Direct Line Group writes motor, home, pet, and travel with strong household matching. Aviva writes broadly across personal lines. At these firms, for customers who have held both motor and home for three or more years, the household-matched multi-product panel exists and is actionable. The churn problem is less severe for retained customers — by definition, the policyholders with the longest cross-product history are the ones who have not churned.

For a direct writer's renewal pricing on retained customers, cross-product claim scores are a real option. We estimate this is perhaps 15–20% of the UK personal lines market by volume — the retained direct-writer book. Not trivial, but far from the full market.

**Underwriting referral signals** are a lower-regulatory-bar application. Rather than directly adjusting the premium, a cross-product claim score could flag risks for underwriter review without changing the formal pricing basis. If a customer's home claim history generates a signal that warrants manual review of their motor renewal, that is a risk management decision rather than a pricing change. Consumer Duty scrutiny is lower; data governance requirements are similar but the fair value documentation is simpler. This could be deployed by a wider range of writers than full cross-product pricing.

---

## How this connects to existing dynamic credibility

Our [`insurance-credibility`](/insurance-credibility/) library currently implements the univariate Ahn et al. (2023) dynamic Poisson-gamma state-space model in `experience/dynamic.py`. The recursion is:

$$\alpha_{t|t} = \alpha_t + Y_t, \quad \beta_{t|t} = \beta_t + \mu \cdot e_t$$

with decay parameters $(p, q)$ fitted by empirical Bayes MLE on the Negative Binomial marginal likelihood. This is the single-product version of the claim-score updating that Verschuren's framework builds on.

The multi-product extension is not currently implemented. Building it would require a `CustomerHistory` data structure linking `PolicyHistory` objects across products by customer key, a `ClaimScoreComputer` for the BM-panel recursion across products, and a `MultiProductScoreModel` with GAM spline fitting for own-product and cross-product effects. Estimated at 300–400 lines of new code — straightforward in principle, scipy cubic spline basis, no additional library dependency — but requiring a new data layer that only has value if the upstream portfolio data already links policies across product lines.

We have not built this. The data layer requirement is the constraint: the module is only useful if you bring it household-matched multi-product panel data, and most UK insurers do not have that in a form that is ready to hand to a Python library.

---

## Should a UK pricing team actually do this?

For a pricing actuary at an aggregator-reliant general insurer: not now. The prerequisite data does not exist for the majority of your book, and the regulatory work to justify cross-product pricing linkage is not worth it for a marginal Gini improvement on a subset of renewals.

For a pricing actuary at a direct writer — Direct Line Group, Aviva, LV= — with a retained motor and home book and a functioning customer key that links policies across products: this is worth a scoping exercise. The Verschuren framework is not exotic; the GAM infrastructure is standard; the data question is whether you can assemble a clean multi-year matched panel. If you can, the Dutch evidence suggests the signal is there.

For any team interested in an underwriting referral application: lower bar, broader applicability, worth a proof of concept. The cross-product signal does not need to change the premium to add value — it can inform a referral decision that a human underwriter then makes.

What we would not do is treat a 2021 paper on Dutch data as strong evidence about UK personal lines profitability. The mechanism is sound. The data context is different. A proper UK validation on a matched direct-writer book would be required before putting cross-product claim scores into a pricing model at renewal.

---

## The bottom line

Verschuren (2021) demonstrates something real: customers who claim more on one product tend to claim more on others, and that cross-product signal can be captured systematically in a recursive claim score framework with GAM-estimated non-linear effects. The Dutch evidence is positive.

The UK context does not make this impossible. It makes it a direct-writer problem, applicable to retained multi-product customers, where Consumer Duty compliance is documented and a customer key links the data. That is perhaps 15–20% of the market, not 80%.

For the majority of UK personal lines writers, the aggregator market, annual churn, and data governance constraints mean this framework sits on the shelf. Worth knowing about. Not worth building yet — unless you are one of the few firms where the data already exists.

---

## The paper

Verschuren, R.M. (2021). 'Predictive Claim Scores for Dynamic Multi-Product Risk Classification in Insurance.' *ASTIN Bulletin*, 51(1), 1–25. DOI: [10.1017/asb.2020.34](https://doi.org/10.1017/asb.2020.34). arXiv: [1909.02403](https://arxiv.org/abs/1909.02403).

---

## Related posts

- [Dynamic Credibility in Insurance Pricing: The insurance-credibility Library](/techniques/pricing/2025/11/18/dynamic-credibility-insurance-credibility/) — the single-product dynamic claim score foundation this extends
- [Gamma Decay and Claims Inflation Persistence](/techniques/pricing/2026/03/31/gamma-decay-claims-inflation-persistence/) — the inflation persistence mechanism in our credibility framework
