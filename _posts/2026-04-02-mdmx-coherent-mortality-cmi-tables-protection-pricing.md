---
layout: post
title: "Tensors for Mortality: What MDMx Does That Lee-Carter Can't, and Why CMI Still Wins for Now"
date: 2026-04-02
author: burning-cost
categories: [research]
tags: [mortality, life, protection, cmi, lee-carter, tucker-decomposition, tensor, kalman, sex-coherence, forecasting, fca, solvency-ii, arXiv-2603-20518, actuarial]
description: "Samuel Clark's MDMx (arXiv:2603.20518) applies Tucker tensor decomposition to HMD mortality data across 50 countries, producing coherent forecasts, disruption detection, and model life tables from one unified framework. For UK protection pricing, it is a genuine benchmark tool but not a CMI replacement."
permalink: /2026/04/02/mdmx-coherent-mortality-cmi-tables-protection-pricing/
math: true
---

Samuel Clark at Ohio State has produced a paper — arXiv:2603.20518, submitted March 2026 — that organises all Human Mortality Database period life tables as a four-dimensional tensor and then decomposes that tensor to do four distinct mortality tasks: construct model life tables, fit incomplete life table data, forecast future mortality, and detect exceptional years like pandemics or wars. All four from one decomposition. He calls the framework MDMx.

The maths is clean. The structural sex coherence is genuinely useful. The disruption detection via Bayes factors is the most original piece. We think UK protection pricing teams should read this paper — but should go in with a precise sense of what it does and does not replace.

---

## What Tucker decomposition adds over Lee-Carter

Lee-Carter (1992) is the workhorse mortality forecasting model. In its standard form it decomposes a matrix of log death rates $\log m_{x,t}$ as:

$$\log m_{x,t} = a_x + b_x \kappa_t + \varepsilon_{x,t}$$

The age effect $b_x$ is fixed. The single time index $\kappa_t$ is projected forward with a random walk with drift. The entire forecast is driven by where $\kappa_t$ goes, with the age pattern of improvement locked to $b_x$. This is a fundamental limitation: in reality, the age pattern of mortality improvement shifts over time. Rectangularisation of the survival curve — compression of deaths into a narrower age window — means that younger ages improve faster in some periods, older ages in others. A fixed $b_x$ cannot capture this.

Tucker decomposition (Tucker, 1966) is the N-way generalisation of matrix SVD. For Clark's four-way tensor $\mathbf{M} \in \mathbb{R}^{S \times A \times C \times T}$ (sex, age, country, calendar year):

$$\mathbf{M} \approx \mathbf{G} \times_1 \mathbf{U}_S \times_2 \mathbf{U}_A \times_3 \mathbf{U}_C \times_4 \mathbf{U}_T$$

The core tensor $\mathbf{G}$ captures interactions between all four modes simultaneously. The factor matrices $\mathbf{U}_S, \mathbf{U}_A, \mathbf{U}_C, \mathbf{U}_T$ are orthonormal. Clark selects ranks $(r_1, r_2, r_3, r_4) = (2, 4, 8, 7)$ — retaining 99.99% of variance — which compresses roughly 2.9 million observed values into around 50,000 parameters.

The age rotation problem that plagues Lee-Carter dissolves: with four age factors evolving jointly, the age pattern of improvement is free to rotate as the time components evolve. Sex coherence is structural rather than imposed: the two-dimensional sex factor matrix means male and female mortality trajectories share the same underlying factor structure, so forecasts cannot diverge in biologically implausible ways. Lee-Carter models fit independently by sex can — and occasionally do — produce forecasts where the sex mortality gap narrows to zero or reverses, which is actuarially embarrassing and impossible in MDMx.

One technical choice worth noting: Clark uses logit($q_x$) as the response, not log death rates. At extreme ages — 85+ — log death rates can become unbounded in a way that causes numerical instability. The logit of the probability of death is bounded above by zero, which makes the decomposition better behaved in the tail ages that matter most for annuity and protection pricing.

---

## The lineage question: is this actually new?

We should be honest about this. Tucker decomposition was first applied to mortality data by Russolillo, Giordano and Haberman in 2011 (*Scandinavian Actuarial Journal* 2011(2):96–117). Their paper used a three-way decomposition across age, year, and country, projected the time component forward with ARIMA, and framed it explicitly as a Lee-Carter extension. That was fifteen years ago.

Clark's novelty relative to Russolillo is real but incremental: adding the sex mode explicitly, using logit($q_x$) instead of log rates, unifying four tasks in one framework, replacing ARIMA projection with a hierarchical Kalman filter, and adding Bayes-factor disruption detection. These are meaningful contributions, not cosmetic ones. The Bayes-factor disruption detection in particular is original — it uses a Laplace-approximated Bayes factor to identify 'exceptional' calendar years in the fit, rather than applying ad hoc multipliers for wars and pandemics.

We score the novelty at 3/5. Reviewers in formal demography will note the Russolillo lineage. That does not diminish the paper's value; it contextualises it accurately.

---

## The forecasting accuracy numbers

Clark reports rolling-origin cross-validation at six origins across a 15-year forecast horizon. The headline comparison:

| Model | $e_0$ MAE (years) | Sex-gap MAE (years) |
|---|---|---|
| MDMx | 1.44 | 0.60 |
| Hyndman-Ullah (2007) | 1.44 | 0.84 |
| Lee-Carter | 1.73 | 1.11 |

MDMx and Hyndman-Ullah are essentially indistinguishable on life expectancy accuracy. MDMx wins on sex-gap preservation, which is where the structural coherence pays off. Coverage of nominal 95% prediction intervals is 93.7% — reasonably calibrated.

Hyndman-Ullah (functional data analysis applied to mortality) has been the main competitor to Lee-Carter for multi-population forecasting for nearly two decades. It also improves on the age-rotation problem, via a different route. The honest reading of Clark's accuracy results is: MDMx does as well as the best existing method, with better sex coherence and a richer structural framework. It does not dramatically outperform.

---

## The FCA hook: legacy mortality assumptions

The FCA's pure protection market study interim report (MS24/1.4, January 2026) flagged that some firms carry legacy mortality assumptions that have not been updated to capture recent mortality improvements. This is specific enough to quote: the FCA's analysis found cases where reinsurance pricing embeds mortality assumptions that are stale, and had to unwind reinsurance profit figures and replace them with consistent claims amounts. The final report is due Q3 2026.

This is a live regulatory pressure for UK protection insurers. It is not a pressure that MDMx directly resolves — MDMx is a population-level forecasting tool trained on HMD data, not a CMI_2025 upgrade — but it creates a context where demonstrating that your mortality assumptions are benchmarked against international trends becomes a governance argument, not just a theoretical one.

MDMx could serve that benchmarking function. If UK assured-lives mortality is diverging from the international cluster that England and Wales historically sits within, the tensor decomposition would reveal that: the country factor loadings change, and the gap between the country's actual trajectory and its projected trajectory widens before the Bayes-factor disruption flag fires. That is useful diagnostic information.

What it cannot do is tell you whether your CMI_2025 long-term improvement rate assumption of 1.5% per annum for males aged 65 is appropriate for your particular assured lives portfolio. That requires insurer-specific experience data, Whittaker-Henderson graduation of your own A/E ratios, and judgment about how your distribution channel affects the socioeconomic profile of your book. MDMx has no input into any of that.

---

## What UK protection pricing teams can actually use this for

We think there are three narrow but genuine applications:

**Benchmarking improvement assumptions.** The MDMx forecasts for England and Wales can be compared against CMI_2025 implied improvements at each age. If the international tensor model suggests, say, that UK male mortality at ages 70–80 is tracking on a trajectory consistent with a 1.2% long-term improvement, and your CMI parameterisation implies 1.8%, you now have a quantified challenge to that assumption. You may have good reasons to diverge; you should be able to articulate them.

**Disruption detection for COVID adjustment.** UK protection insurers are still carrying COVID-period mortality experience in their base data and deciding how to weight it. The MDMx Bayes-factor disruption detection provides a principled way to identify that 2020–2021 were statistically exceptional in the HMD data, with a formal evidence measure rather than an actuarial judgment call. That is the kind of documented, transparent methodology the FCA's governance concerns point toward.

**Reinsurer and multi-market work.** For Lloyd's syndicates writing life or trade credit business in emerging markets, or for reinsurers pricing developing-country mortality risk, the MDMx model life table capability is directly useful. The summary indicator approach — mapping two observed mortality indicators to a complete age schedule — is designed exactly for populations with sparse data.

For standard UK term assurance or critical illness pricing from a UK composite: CMI tables remain the baseline, the CMI improvement model remains the projection tool, and Whittaker-Henderson graduation of insurer experience remains the A/E adjustment methodology. MDMx does not touch any of these. The infrastructure is regulatory (CMI is embedded in Lloyd's and reinsurance treaty structures), commercial (reinsurance treaties reference CMI basis), and practical (CMI updates annually with a practitioner governance process that MDMx cannot replicate).

---

## The missing package

Clark's prior work — SVD-Comp, a related model — is on CRAN as `SVDMx`. MDMx has no published R or Python package as of the paper's submission. There is no code to clone and run against your own data.

This matters for practitioners rather than academics. The Tucker decomposition itself is straightforward: `tensorly` v0.7+ in Python implements `tucker` with one function call. The domain wiring — HMD data parsing, logit($q_x$) transform, the three-stage fitting algorithm, the disruption detection with Laplace approximation, the hierarchical Kalman forecaster — is a substantial implementation project. We estimate a demonstration notebook at a day's work; a production-quality library at a week or more.

We are watching for a package release. If Clark publishes MDMx in R or Python, the calculus changes and a library post becomes more valuable than this one.

---

## Our position

MDMx is a well-constructed mortality forecasting framework with specific advantages over Lee-Carter that matter actuarially: structural sex coherence, age-pattern flexibility, and disruption detection with a formal evidence measure. The novelty claim is partly overstated — the Tucker decomposition lineage in mortality is 15 years old — but the unified four-task framework and the Kalman forecasting with hierarchical drift are real contributions.

For UK protection pricing teams, it is a benchmark tool. Use it to challenge your CMI improvement assumptions with an international reference, and to document your COVID-period adjustment decisions with something more principled than expert judgment. Do not expect it to replace any component of the CMI-based workflow, because that workflow is embedded in regulatory, reinsurance, and commercial infrastructure that will not move on the basis of a single arXiv preprint.

The FCA scrutiny is coming anyway. Having an internationally benchmarked mortality framework you can point to, when the Q3 2026 final report lands, is a better position than not having one.

---

*arXiv:2603.20518 — "Multi-dimensional Mortality (MDMx)" — Samuel J. Clark, Ohio State University. Submitted 20 March 2026, revised 25 March 2026. No public package at time of writing; prior SVDMx is on CRAN.*
