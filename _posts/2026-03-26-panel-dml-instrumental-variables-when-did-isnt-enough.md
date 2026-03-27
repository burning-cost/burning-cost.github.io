---
layout: post
title: "Panel DML with Instrumental Variables: When DiD Isn't Enough"
date: 2026-03-26
categories: [pricing, causal-inference, rate-change]
tags: [double-machine-learning, instrumental-variables, panel-data, fixed-effects, rate-change-evaluation, endogeneity, weak-instruments, dml, econometrics, causal-inference]
description: "A new paper combines panel fixed effects, double machine learning, and instrumental variables. The headline result is not the estimator — it's that ML covariate adjustment frequently reveals instruments are weaker than 2SLS suggested."
---

There is a class of pricing question that difference-in-differences cannot answer cleanly. DiD handles confounding from observed and unobserved time-invariant characteristics. It does not handle the case where the treatment itself — the rate change — was endogenous to something unobserved that was also shifting at the same time.

That is the panel IV problem. A new paper, Baiardi, Clarke, Naghi and Polselli (arXiv:2603.20464, March 2026), addresses it by combining three things: panel fixed effects via first-differencing, Neyman-orthogonal DML moment conditions, and instrumental variables. The result is Panel IV DML — or, in the R package they've built alongside it, `xtivdml`.

We are not building a Python implementation. We will explain why not, and what pricing actuaries should actually take from this paper.

---

## The problem: endogenous rate changes

Our existing rate change evaluation tools — DiD estimators with DML controls — rest on a conditional parallel trends assumption. Given observed risk characteristics X, the untreated counterfactual trend of the treated segment would have matched the control segment. That assumption is plausible when the rate change was applied uniformly across a segment for administrative or commercial reasons.

It breaks down when the rate change was driven by the segment's own recent claims experience. A segment that had a bad loss year gets a large rate increase. That same segment may have had an underlying risk shift — a change in policyholder mix, a fraud pattern beginning to resolve, a broker channel change — that was both causing the claims spike and driving the re-pricing decision. The rate change and the unobserved risk shift are correlated. The DiD estimate of rate change impact on lapse or loss ratio is biased.

Panel IV DML fixes this by introducing an instrument Z: something that influenced how large the rate change was, but did not directly affect the outcome through any channel other than the rate change itself.

---

## What the estimator does

The paper's setup is a static panel:

```
Y_it = θ · D_it + g_0(X_it) + α_i + ε_it
D_it = r_0(X_it) + m_i + V_it
```

Y is the outcome (loss ratio, lapse rate, claim frequency), D is the treatment (rate change applied), X are observed controls, and α_i, m_i are individual fixed effects. The fixed effects are eliminated by first-differencing — not demeaning, which matters for the cross-fitting step.

Three nuisance functions are estimated using ML (Lasso, gradient boosting, or neural networks, each cross-fitted using block-k-fold to preserve within-unit serial correlation):

- **l₀**: outcome on controls (partial out X from Y)
- **m₀**: instrument on controls (partial out X from Z)
- **r₀**: treatment on controls (partial out X from D)

The residuals feed into a Neyman-orthogonal score that produces a root-n consistent, asymptotically normal estimate of θ, the treatment effect, despite the high-dimensional nonlinear controls.

The innovation over standard DoubleML — which handles selection-on-observables — is the IV component. When D is endogenous even after conditioning on X, you need Z to identify θ. The paper formalises this with a Neyman-orthogonal moment condition (Proposition 3.1) and derives weak instrument diagnostics adapted for the DML setting: an F-statistic F^DML and an Anderson-Rubin confidence set AR(θ₀)^DML.

---

## The most important finding is about instrument weakness

The empirical applications in the paper use shift-share instruments — Tabellini (2020) on US immigration and economic outcomes, and Moriconi (2019, 2022) on similar questions. Shift-share IVs are a popular and often defensible instrument choice in regional economics.

Here is what happened when the authors applied Panel IV DML to these datasets:

- **Tabellini (2020)**: instrument relevance actually improved with flexible ML adjustment. But the economic outcome showed no statistically significant effect, where naive 2SLS had found a positive one.
- **Moriconi (2019, 2022)**: the Anderson-Rubin confidence sets became unbounded. The instruments, which looked adequate under 2SLS, failed the weak instrument test once ML was used to partial out the confounders properly.

Two of three applications lost their instruments. Not because the instruments were obviously bad — they had reasonable F-statistics under conventional 2SLS — but because flexible covariate adjustment absorbed variation that naive linear controls had left in the instrument-treatment relationship.

This is the paper's genuinely important contribution. It is not that Panel IV DML gives you better estimates when instruments are strong. It is that it tells you when your instruments were never strong enough to begin with, and 2SLS was flattering them by using linear first-stage controls.

---

## Why valid instruments are nearly impossible to find in insurance pricing

The candidate instrument list for a UK pricing application looks reasonable until you stress-test each one:

**Group-level average rate changes (shift-share style)**: the market-wide component of a rate change is external to any individual segment's risk evolution. In principle, if a pricing model applies a broad market rate change where segment-level deviations are driven by technical re-pricing, the broad component could instrument for the total rate change. In practice, the broad component is correlated with broad-market claims trends that affect all segments simultaneously — not exogenous.

**Reinsurance cost shocks**: external treaty price changes that forced rate action in specific lines are plausibly exogenous to individual policyholders. But they are line-level, not segment-level, and the timing is usually too coarse to identify within-line treatment variation. If your motor XL treaty repriced in January and you raised motor rates in Q1, every motor segment got treated — you have no control group within the instrument's variation.

**Regulatory floor changes**: FCA or PRA minimum rate requirements for specific segments are genuinely external. But these are rare events, affect specific segments, and the identification comes from a handful of observations. Asymptotic theory does not apply.

**Competitor rate index**: external market rate movement used as justification for your own rate increase could instrument for the commercial element of your rate decision. But competitor rate indices are themselves endogenous to the same market-wide loss trends that drove your claims — the exclusion restriction fails.

The deeper problem is structural: in insurance, re-pricing IS the primary response mechanism to observed risk. The technical premium output is derived from claims data. Rate changes and risk shifts are not just correlated in practice — they are causally linked by design. The construction of the instrument is almost always contaminated by this.

---

## What data structure the method needs

For completeness, here is what a valid application would require:

- **Panel structure**: multiple observations per risk segment (or policyholder) over at least two periods. The paper's applications had N≈180 geographic units over several years.
- **Treatment D_it**: the rate change applied to segment i at time t.
- **Outcome Y_it**: loss ratio, lapse rate, or claim frequency at segment-period level.
- **Controls X_it**: observed risk factors — age, vehicle group, NCD, area.
- **Instrument Z_it**: something that predicts D_it but is independent of contemporaneous unobserved risk shifts in segment i.

The additive separability assumption (Assumption 2.6 in the paper) is also non-trivial. It requires that fixed effects enter additively in outcome, instrument, and treatment equations. If you have interactions between fixed effects and time-varying covariates — which insurance rating structures essentially always have — the assumption is violated.

---

## The software situation

The companion R package `xtdml` is on CRAN (v0.1.12, published 2026-03-13) and implements panel DML without IV — partially linear panel regression with fixed effects, using the mlr3 ecosystem for nuisance learners. It is production-ready.

The IV extension, `xtivdml`, is on GitHub only, with one commit, no CRAN release, and no stars. It is the authors' prototype. Do not build a production workflow on it yet.

In Python, DoubleML has added panel support (`DoubleMLPLPR` for partially linear panel regression, added 2025) and panel DiD (`DoubleMLDIDMulti`, Callaway-Sant'Anna style). Panel IV DML is not implemented in any Python package as of March 2026. Building it from scratch would require the block-k-fold cross-fitting, the Neyman-orthogonal score matrix algebra, and the Anderson-Rubin test — approximately 800–1,200 lines of validated econometric code, and that is before the instrument problem is solved.

We will not be adding this to [`insurance-causal`](/insurance-causal/) now. We will reassess when `xtivdml` stabilises on CRAN, probably in late 2026.

---

## When this method is actually appropriate

Panel IV DML is the right tool when two conditions hold simultaneously:

1. You believe the rate change was endogenous to unobserved contemporaneous risk shifts — not just confounded by observed characteristics that DiD controls can handle, but endogenous in a way that requires an instrument to break.

2. You have a credible external instrument that passes the exclusion restriction: it affected the magnitude of the rate change but had no direct effect on the outcome through any other channel.

In UK motor and home personal lines pricing, both conditions are rarely met together. The method is more plausible for commercial lines where a large corporate client's rate negotiation involves factors (broker relationships, treaty structures, cross-line considerations) that are genuinely external to that client's risk evolution. It is also relevant for cross-insurer academic studies where market-level instruments exist.

For individual insurer pricing teams evaluating their own rate changes, our existing DiD approach with DML controls remains the appropriate tool. It handles the selection-on-observables problem. If you genuinely believe there is residual endogeneity that observables cannot address, Panel IV DML is the methodologically correct next step — but finding the instrument is the hard part, and no software solves that.

---

## What to take from this paper

The practical message is not "add IV to your DML pipeline." The practical message is: if you are using 2SLS or IV-style methods to evaluate rate change impacts, you may be over-estimating instrument strength. ML covariate adjustment is not conservative — it aggressively absorbs variation that linear first-stage models leave on the table, and that variation may be what was making your F-statistic look respectable.

Before trusting any IV estimate in a pricing evaluation, we would run the DML-based weak instrument test from this paper. Not because Panel IV DML is the estimation method you will use, but because the diagnostic is more honest than the standard Staiger-Stock threshold applied to a linear first stage.

The paper is at [arXiv:2603.20464](https://arxiv.org/abs/2603.20464). The R package is [xtdml on CRAN](https://cran.r-project.org/package=xtdml) for the non-IV panel DML case.

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the `insurance-optimise` library that implements the cross-sectional DML pipeline; Panel IV DML is the extension for multi-period panel data
- [The PCW Endogeneity Problem: Why Your Conversion Model Is Biased](/2026/03/26/the-pcw-endogeneity-problem-why-your-conversion-model-is-biased/) — the causal structure that motivates instrumental variable estimation in insurance demand models
