---
layout: post
title: "The PCW Endogeneity Problem: Why Your Conversion Model Is Biased"
date: 2026-03-26
categories: [pricing, demand-modelling, causal-inference]
tags: [pcw, endogeneity, causal-dag, instrumental-variables, dml, elasticity, fca-ep25-2, uk-motor, conversion-model, logistic-regression]
description: "Most UK insurers fit a logistic regression on PCW quote data and call it a demand model. It is biased in at least three distinct ways. Here is the causal structure that explains why."
---

Most UK motor insurers have a demand model. Ask what it is and the answer will usually be: logistic regression, conversion as the outcome, price and risk features as predictors, sometimes rank thrown in as a control. The team ran it on a year's worth of PCW quote data, it fits reasonably well, and it feeds the pricing optimiser.

We think that model is wrong — not wrong in the sense of imprecise, but wrong in the sense that its price coefficient cannot be interpreted as a demand elasticity, and the direction of bias is not predictable from the data alone. This post explains the causal structure that produces the bias.

This is post 1 of two. Post 2 will show how to fix it using double machine learning on quote data, with Python code.

---

## The standard approach

The typical specification looks like this:

```
logit(P(convert)) = β₀ + β₁·log(price) + β₂·rank + β₃·X + ε
```

where X is a vector of risk features (NCD, vehicle group, area, age), rank is the insurer's position on the PCW results page, and price is the quoted premium. β₁ is treated as the log-elasticity of conversion with respect to price.

The intuition is clear: we have prices, we have outcomes, we have controls — regress away and read the coefficient. The problem is that this model conflates prediction with causal identification, and the conflation matters enormously when the model's output is used to optimise prices.

---

## The causal DAG

Here is the causal structure as we understand it:

```
              U (unobserved risk)
             / \
            v   v
           TP   Y (conversion)
           |
           v
    CL --> P (price) --> R (rank) --> Y
                |
                |       C (competitor prices) --> R --> Y
                |                |
                +--- M (market risk trends) ---+
```

Nodes:
- **U** — unobserved customer risk factors (driving behaviour, actual garage quality, etc.)
- **TP** — technical premium (your risk model's output)
- **CL** — commercial loading (the percentage uplift applied at rate review)
- **P** — quoted price (P = TP × CL)
- **R** — rank position on the PCW results page
- **C** — competitor prices
- **M** — market-wide cost trends (claims inflation, repair costs, accident rates)
- **Y** — conversion outcome (0/1)

The causal arrows matter. Let us trace the three problems in turn.

---

## Problem 1: Price is endogenous (the standard bias)

Price is not assigned randomly. It is set by your risk model applied to the customer's features. Those features are noisy proxies for true risk U. U has a direct path to conversion Y — risk-seeking customers may be more price-sensitive, or may be more likely to shop on price rather than brand. Formally:

- U → TP → P (your price is correlated with unobserved risk)
- U → Y (unobserved risk affects conversion directly)

This is the textbook endogeneity problem. OLS/logistic regression on observed data gives:

```
E[β̂₁] = β₁ + (bias from U)
```

The direction of the bias depends on the sign of the U → P and U → Y correlations. High-risk customers get higher prices (positive U → P). If high-risk customers are also more price-sensitive, they are more likely to convert at any given price level (negative or positive U → Y depending on how you think about it). The bias term is not zero, and you cannot sign it without strong prior assumptions.

The fix is to find variation in price that is uncorrelated with U. The most credible instrument available to UK insurers is **commercial loading variation across quarterly rate reviews**. When the pricing team applies a 5% loading increase in Q3 2024, every customer who quoted in that window got a higher price than an observationally identical customer who quoted in Q2. The loading is applied uniformly across segments; individual risk profiles did not change. This within-segment price variation is orthogonal to U, and it shifts P — the definition of a valid instrument.

This requires storing technical premium at quote time, not just the final commercial price. Most insurers do not do this routinely. If you do not have it, you need to reconstruct it from rate tables, which is painful but usually possible.

---

## Problem 2: The rank mediation trap

This is the less obvious failure, and we have not seen it explicitly named in the published literature.

When an analyst includes rank as a control variable, the reasoning sounds sensible: "rank determines visibility on the page, so I should control for it to isolate the pure price effect." That reasoning is wrong for causal estimation.

Rank is caused by price. Rank is not an independent confounder — it is a **mediator**: price → rank → conversion. Including a mediator as a covariate in a regression blocks the causal pathway you are trying to measure. Part of the price effect on conversion *operates through* rank (cheaper price → better rank → higher conversion). If you condition on rank, you are estimating the effect of price *holding rank fixed* — which is not a quantity that corresponds to any real-world decision.

Put it concretely: if you cut your price by 5%, your rank improves, and conversions increase. Some of that increase is direct (consumers on the results page notice you are cheap) and some is indirect (you moved from rank 4 to rank 2, so more consumers see you at all). The mediated path through rank is part of the policy-relevant price effect. Controlling for rank strips it out.

What the model with rank as a control is actually estimating is something like: "what would happen to conversions if I cut my price but somehow kept my rank constant?" That question has no operational answer. You cannot set price and rank independently.

The correct approach depends on what question you are asking:

- If you want the **total effect of price on conversion** (including rank), do not include rank as a predictor. Use IV for endogeneity.
- If you want the **direct effect of price on conversion** (excluding the rank pathway) — for instance, to separate willingness-to-pay from position effects — you need a mediation analysis design, not naive conditioning.

For pricing optimisation, you almost always want the total effect. Logistic regression with rank as a control gives you neither.

---

## Problem 3: Competitor price simultaneity

Including competitor prices as features — which the better models do — introduces a second endogeneity problem. Competitor prices are not exogenous signals; they are set by competitors' risk models applied to the same customer. More importantly, competitors' prices are driven by the same market-wide cost trends M (claims inflation, repair cost indices, weather events) that also affect your conversion rate through channels other than relative price.

A quarter where used car prices spike (Cazana Q2 2022, for example) will see all insurers raise prices. Your conversion rate changes partly because your relative rank changes, and partly because consumers are reacting to the media coverage about insurance costs. If you include a competitor's price as a raw feature, you are not controlling cleanly for either channel.

Hausman-style instruments address this: use drivers of competitor costs that are orthogonal to individual risk — regional claims inflation indices, or another competitor's pricing movements in a different geographic market. If Admiral raises prices in Wales due to a regional weather event, that shifts the rank structure for Welsh customers without being relevant to a customer in Yorkshire. This cross-competitor, cross-region variation gives you identification of the competitor price effect.

This is routinely done in the empirical industrial organisation literature (the BLP approach to demand estimation), but we have not seen a UK insurer implement it. The data requirements are substantial — you need competitor price panels from Consumer Intelligence or eBenchmarkers going back several rate review cycles.

---

## What the biased elasticity numbers look like

The combination of these three problems produces conversion models that underestimate price sensitivity at some ranks, overestimate it at others, and potentially get the sign wrong in thin-data segments. The rank mediation trap specifically tends to artificially attenuate the price coefficient — because rank, when included as a predictor, absorbs much of the price variation that is actually driving conversions.

Narayanan and Kalyanam (2015) found exactly this pattern in search advertising, a structurally analogous setting: RDD estimates of rank effects were substantially smaller than mean comparisons, because naive regressions were conflating the position effect with selection effects. The PCW setting is not identical, but the geometry of the problem is the same.

We think the practical consequence for UK insurers is that naive demand models systematically overestimate the conversion uplift from moving from rank 3 to rank 1 (because the rank coefficient was estimated with reverse causation baked in), and underestimate the raw price elasticity (because the rank variable absorbed part of the price effect during estimation). Those two errors push in offsetting directions in some pricing decisions, which may be why the models "work" in backtests while being causally wrong.

---

## The regulatory angle is real

FCA Evaluation Paper 25/2 (July 2025) flagged explicitly that elasticity-based pricing requires controls to prevent charging less price-sensitive customers higher prices, including as a proxy for protected characteristics. The paper noted that 1 in 4 motor consumers switched insurer in 2024, up from 1 in 5 in 2023, and that the PCW channel now accounts for 66% of motor new business (up from 60% pre-GIPP remedies).

The regulator is looking at how elasticity estimates are constructed and what they correlate with. A biased demand model that produces segment-level CATE estimates correlated with demographic proxies (NCD band, area code, vehicle group are all proxy-correlated with age and income) creates regulatory exposure that the team probably does not realise exists. You cannot audit a demand model for proxy discrimination if you do not first know whether the elasticity estimates are correct.

---

## What this does not solve

Knowing the DAG is not enough to estimate the model. A few genuine hard problems remain:

**The instrument strength problem.** Commercial loading variation is a credible instrument only if there are enough rate review periods with meaningful loading changes, and if the loading changes are not correlated with market conditions that simultaneously affect consumer behaviour (if you always raise loadings when claims inflation is high, the instrument is weak). The first-stage F-statistic rule-of-thumb of 10 is a floor, not a comfort.

**Quote selection bias.** Your data only contains customers who received a quote. In segments where you are chronically uncompetitive, you either do not appear on the PCW at all, or you appear at ranks 6–10 where consumer selection into clicking is different. The Heckman selection correction for this requires an exclusion restriction — something that affects whether you quote but not conversion given a quote — and we have not identified a clean one yet.

**Rank as both outcome and mediator.** The correct design for disentangling the direct and mediated paths requires either a separate RDD design at rank cutoffs (Narayanan and Kalyanam's approach applied to insurance rank boundaries) or a structural mediation model. Ronayne's (2021) equilibrium model of PCW pricing suggests the rank effect itself is partly artifactual — a consequence of commission pass-through raising all prices equally, leaving rank as the primary selection mechanism. If that is right, the mediated path is very large.

These are unresolved. Post 2 will implement what we can implement given real data constraints, and be explicit about what identification assumptions are doing the work.

---

## The bottom line

If your pricing optimiser uses price elasticities from a logistic regression with rank as a covariate, it is optimising against a biased demand surface. The rank mediation trap specifically inflates the apparent importance of rank relative to absolute price, which pushes optimisers toward aggressive top-of-table pricing strategies that may not generate the conversion uplift the model predicts.

The fix requires instrumental variables — and the instruments exist. Commercial loading variation across rate reviews is the most accessible. Competitor regional variation is harder to implement but adds substantial identification power.

Post 2 shows the implementation using `doubleml` and our `insurance-causal` library.

---

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the `insurance-optimise` library that implements the DML fix described in this post, with full renewal pricing pipeline
- [Estimating PCW Conversion Elasticity with Double Machine Learning](/2026/03/26/estimating-pcw-conversion-elasticity-with-double-machine-learning/) — part 2: the Python implementation of the IV-DML pipeline

---

- FCA: [Evaluation Paper 25/2 (July 2025)](https://www.fca.org.uk/publications/corporate-documents/evaluation-paper-25-2-general-insurance-pricing-practices-remedies)
- Narayanan & Kalyanam (2015), *Marketing Science* 34(3):388–407 — RDD for position effects in search advertising
- Ronayne (2021), *International Economic Review* 62(3):1081–1110 — PCW welfare paradox
- Schultz et al. (2023), arXiv:2312.15282 — causal forecasting for pricing using DML
- Muijsson (2022), *Expert Systems with Applications* — continuous DML for insurance renewal elasticity
- Treetanthiploet et al. (2023), arXiv:2308.06935 — RL pricing on PCWs
