---
layout: post
title: "EVT Meets Machine Learning: Three Papers Worth Reading for Severity Modellers"
date: 2026-03-25
categories: [techniques, severity]
tags: [evt, extreme-value-theory, machine-learning, severity, gpd, heavy-tails, insurance-severity, bayesian-nonparametric, uk-motor, uk-home]
description: "Three recent papers on EVT and ML — from generalisation bounds for tail learning to Bayesian nonparametric splicing — and what they actually imply for UK severity models."
---

Large losses are the part of motor and home insurance that keeps actuaries awake. A single flood claim, a catastrophic injury award, a fire that takes out an entire block — these observations sit in the far tail of the severity distribution, they are relatively rare, and they drive a disproportionate share of profit volatility. Traditional frequency-severity models handle them with a degree of hand-waving: fit a Pareto or GPD above some chosen threshold, check the mean excess plot, argue about whether the tail index looks reasonable, move on. It works well enough most of the time.

But three papers published in the last twelve months suggest the toolbox is expanding in interesting directions. The advances come from different starting points — statistical learning theory, applied EVT for insurance, and Bayesian nonparametrics — and they are not yet connected to each other in any unified framework. What they share is a common ambition: to let data (and covariates) do more of the work in the tail, rather than relying on practitioner judgement at every step.

## Why the tail is hard

Before getting to the papers, it is worth being precise about the problem. Extreme value theory rests on the Generalised Pareto Distribution (GPD): for losses above a threshold $u$, under mild regularity conditions, the excess distribution converges to a GPD as $u \to \infty$. The GPD has two parameters — a shape $\xi$ (tail heaviness) and a scale $\sigma$ — and the shape parameter is the one that actuaries argue about. For UK motor bodily injury, $\xi$ typically sits between 0.3 and 0.7 depending on the book and the threshold. For property lines it tends to be lower. Getting $\xi$ wrong by 0.1 has material consequences for any ILF above three times the basic limit.

The practical difficulty is that tail estimation is inherently data-sparse. If you have 50,000 motor claims and apply a threshold at the 95th percentile, you are fitting a GPD to 2,500 observations. That sounds comfortable until you remember that the 99th percentile estimate is driven by the top 500, and the 99.9th by the top 50. At that point the confidence intervals on $\xi$ are wide enough to drive a bus through, and the classic actuarial response — pooling years, or borrowing from industry data — introduces its own biases.

Machine learning enters here not to replace EVT but to address a related problem: the tail parameters are not constant across the book. A claim from a high-powered sports car, a claim involving a young driver on a motorway, a claim under a commercial vehicle policy — these sit in the same severity distribution in the aggregate, but there is no particular reason to believe they share the same tail behaviour. Conditional EVT, where $\xi$ and $\sigma$ are modelled as functions of covariates, is the right framework. Getting it right is the challenge.

## Paper 1: Clémençon and Sabourin on generalisation bounds for tail learning (arXiv:2504.06984)

Stephan Clémençon and Anne Sabourin submitted "Weak Signals and Heavy Tails: Machine-learning meets Extreme Value Theory" in April 2025 (revised June 2025). It is a survey-style paper that brings multivariate EVT and statistical learning theory into a common non-parametric, non-asymptotic framework.

The core contribution is establishing generalisation bounds for algorithms learning from tail data. In ordinary machine learning, generalisation bounds tell you how much worse your model will perform on new data than on training data. In the tail, this question is harder because the relevant data is rare by definition — you cannot expect to see many examples from the far right of the distribution. Clémençon and Sabourin derive exponential maximal deviation inequalities tailored specifically to low-probability regions, and concentration results for stochastic processes that describe extreme observations.

The applications they examine include classification and regression, anomaly detection, model selection via cross-validation, and high-dimensional Lasso adapted for extreme-value covariates. The paper is mathematical but the practical message is not obscure: when you fit a conditional EVT model (say, a neural network that outputs $\xi$ and $\sigma$ as functions of risk features), you can now ask a meaningful question about how many tail observations you need before the model's predictions at extreme quantiles are reliable. The answer, as one would expect, is "more than you think" — but the framework gives you the tools to quantify this rather than guessing.

For UK motor severity work, this is relevant to any attempt to model ILFs by risk segment. The moment you start conditioning the tail index on covariates, you are implicitly making a claim about what can be learned from tail data. Clémençon and Sabourin's framework is one of the first to put rigorous non-asymptotic bounds around this — which matters when you are trying to justify a model to a reserving actuary or an internal model validation team.

The paper does not provide a ready-to-use Python implementation. It is theoretical foundations work. But the theory is the part that has been missing.

## Paper 2: Albrecher and Beirlant on EVT for the insurance industry (arXiv:2511.22272)

Hansjoerg Albrecher and Jan Beirlant's "Statistics of Extremes for the Insurance Industry" (November 2025, forthcoming in the Chapman & Hall Handbook of Statistics of Extremes) is a different kind of paper — a practical survey aimed explicitly at insurance modellers, written by two of the field's leading EVT researchers.

The paper covers three practical complications that standard textbook EVT ignores: truncation, censoring, and tempering.

**Truncation** occurs when your data excludes large losses. Reinsurance treaties, policy limits, and data sharing agreements all create truncation. Albrecher and Beirlant show that standard GPD fitting on truncated data underestimates the tail index — you see a lighter tail than is actually there because the heaviest observations have been removed. Their proposed correction tests for truncation using a modified Pareto QQ-plot, then jointly estimates the truncation point and extreme quantiles. Their example uses German flood loss data, where truncation effects only become visible in the top 15 observations.

**Censoring** under random right-censoring applies directly to IBNR: large claims are still developing, so the observed amount is an underestimate of the final settlement. The paper applies a censoring-corrected Hill estimator, dividing the standard estimator by the proportion of non-censored data among the top-$k$ observations. They identify a structural problem that any motor reserving actuary will recognise: larger claims have longer development times, so they are more likely to be censored. This violates the independence assumptions in the asymptotic theory, and they note it openly rather than papering over it.

**Tempering** is the most interesting contribution for UK property lines. Pure Pareto tails extend to infinity, which is physically unreasonable for most property classes. The Weibull-tempered Pareto survival function $S(x) = x^{-\alpha} e^{-(\beta x)^\tau}$ interpolates between power-law and exponential decay. Applied to Norwegian fire insurance data, this produces $\hat{\alpha} = 1.199$, $\hat{\tau} = 0.702$ at the optimal threshold $\hat{k} = 4920$. The tail is heavy but bounded in practice — which matches how we think about UK domestic property.

The paper also discusses spliced Erlang-Pareto models, using the EM algorithm for fitting and AIC/BIC for model selection. The result — "good fit with $\hat{\xi} = 0.67$" on liability data using three Erlang body components — is directly relevant to the kind of split-distribution severity models that UK pricing teams build for motor BI.

This paper is worth printing out and reading slowly. It is the most practically grounded of the three.

## Paper 3: Nieto-Barajas on Bayesian nonparametric mixtures for heavy tails (arXiv:2602.07228)

Luis E. Nieto-Barajas submitted "Modelling heavy tail data with Bayesian nonparametric mixtures" in February 2026. The core idea is that the standard EVT approach — fit to exceedances above a threshold, discard the bulk of the data — wastes information. The bulk of the distribution is informative about the threshold, about which model family is appropriate, and about the body-tail transition.

Nieto-Barajas proposes two mixture models: one for the body (below the threshold) and one for the tail (above it). Both use mixtures of shifted gamma-gamma distributions with normalised stable processes as mixing distributions. One parameter in the tail component governs tail behaviour directly, allowing the posterior distribution of that parameter to tell you how much of the data supports a heavy-tail interpretation. This is a genuinely useful diagnostic — rather than testing whether the tail is heavy (a binary question), you get a probability distribution over the degree of heaviness.

The computational method is MCMC with adaptive Metropolis-Hastings, which means this is not something you fit on 100,000 policies overnight. But for large-loss analysis on a specific commercial or specialty book — say, a motor fleet account with a multi-year claims history — the Bayesian posterior is exactly what you want. You can propagate tail uncertainty into ILF estimates and get credible intervals that reflect genuine epistemic uncertainty rather than asymptotic approximations that assume you have more data than you do.

The paper applies the method to simulated data and a real dataset. Results show that the MCMC approach correctly identifies the proportion of observations supporting heavy-tail behaviour, and that the method outperforms threshold-exceedance approaches on smaller samples — precisely the regime that matters for specialty insurance.

## What this means for UK motor and home severity models

None of these papers is a plug-and-play solution. They are advances in the underlying methodology, and the path from methodology to production pricing model is always longer than it looks.

That said, three practical implications stand out.

**Conditional EVT needs theoretical justification before deployment.** It is technically straightforward to train a gradient-boosted model that outputs $\xi$ and $\sigma$ as functions of vehicle features or property characteristics. Several teams are already doing this. What Clémençon and Sabourin show is that the sample sizes required for these estimates to be reliable are much larger than intuition suggests — particularly when you are estimating at the 99th or 99.9th percentile conditional on rare feature combinations. If you have a motor book with 200,000 earned cars, you probably do not have enough tail observations by risk segment to trust segment-level tail indices. This is an uncomfortable result, but it is better to know it than not.

**Censoring and truncation corrections should be standard practice, not optional extras.** The Albrecher-Beirlant paper makes a compelling case that ignoring truncation or censoring in tail estimation introduces systematic bias. For UK motor BI, where large claims take 5-10 years to develop and are frequently capped at reinsurance layers, both effects are material. The corrected Hill estimator is not mathematically complex — it is the standard estimator divided by a single adjustment factor — but it requires knowing which of your top-$k$ claims are still open or subject to limits.

**The threshold selection problem is not solved, but Bayesian approaches reduce the arbitrariness.** The choice of threshold $u$ is one of the most consequential and least principled decisions in any EVT analysis. Move it up and your tail estimates are noisier but less contaminated by body-distribution effects. Move it down and you have more data but risk GPD misspecification. The Nieto-Barajas approach sidesteps this somewhat by jointly modelling the body and tail — the posterior over tail-heaviness implicitly incorporates threshold uncertainty. This is not a complete solution, but it is more honest than picking a threshold by eyeballing a mean excess plot.

## Connection to our insurance-severity library

Our [insurance-severity](https://github.com/burning-cost/insurance-severity) package already implements several of the techniques discussed in these papers. The `TruncatedGPD` class fits a GPD with a truncated maximum likelihood correction — for the practical context on why policy limits create truncation bias in standard GPD fits, see [your GPD is lying because your claims are truncated](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) — directly addressing the truncation bias that Albrecher and Beirlant document. The `CensoredHillEstimator` applies the censoring correction from the same paper (their reference is cited in the module docstring). The `WeibullTemperedPareto` implements the tempered Pareto survival function $S(x) = x^{-\alpha} e^{-\lambda x^\tau}$ for property lines where pure Pareto tails are implausible.

On the composite modelling side, `LognormalBurrComposite` and `CompositeSeverityRegressor` handle the body-tail split with automatic threshold selection via profile likelihood — closer in spirit to the Nieto-Barajas approach than to standard POT.

What the library does not yet have is a conditional EVT regressor that properly accounts for the sample-size constraints that Clémençon and Sabourin identify. That is work in progress. For now, the honest recommendation is: use composite regression for conditional severity modelling (it is more stable with typical UK book sizes), and reserve conditional GPD fitting for cases where you have enough tail data by segment to trust the estimates.

## A note on what "EVT + ML" actually means in practice

There is a tendency in the literature — and in some conference presentations — to treat "we used a neural network to estimate the tail index" as an inherently superior approach to "we ran a regression on the log of excess losses." Whether it is superior depends almost entirely on the data volume available in the tail. For a large personal lines book (500,000+ policies, multiple years), conditional neural tail models may well outperform fixed-$\xi$ approaches. For most commercial lines portfolios, or for any book where you are trying to be precise about segment-level ILFs, the theoretical bounds from Clémençon and Sabourin suggest you should be cautious about how much conditioning you attempt.

The Albrecher-Beirlant paper implicitly makes the same point: the regime where fancy ML methods add the most value is also the regime where the data is richest. In the tails of insurance loss distributions, richness is precisely what you do not have.

---

*The papers discussed here are:*
- *[arXiv:2504.06984](https://arxiv.org/abs/2504.06984) — Clémençon, Sabourin. "Weak Signals and Heavy Tails: Machine-learning meets Extreme Value Theory." June 2025.*
- *[arXiv:2511.22272](https://arxiv.org/abs/2511.22272) — Albrecher, Beirlant. "Statistics of Extremes for the Insurance Industry." November 2025.*
- *[arXiv:2602.07228](https://arxiv.org/abs/2602.07228) — Nieto-Barajas. "Modelling heavy tail data with Bayesian nonparametric mixtures." February 2026.*
