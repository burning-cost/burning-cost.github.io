---
layout: post
title: "Causal Effects at Extreme Quantiles: The TIEE Estimator"
date: 2026-03-26
categories: [research, causal-inference, extreme-value-theory]
tags: [causal-inference, extreme-value-theory, gpd, quantile-treatment-effects, ipw, reinsurance, flood, climate, causal-evst, tiee, arXiv]
description: "Li and Castro-Camilo (arXiv:2603.23309, March 2026) unify inverse probability weighting and extreme value extrapolation in a single estimating equation. Here is what it does, where it genuinely helps insurance, and where it does not."
---

Most causal inference asks: does this treatment shift the mean? DML, double-robust estimators, causal forests — all of them are, at bottom, about average treatment effects or average conditional treatment effects. For insurance, the mean is often the wrong question. The mean claim is not what drives reinsurance cost. It is not what determines whether a risk is insurable. It is not what moves the needle in a Solvency II stress test.

The question we actually want to answer is: does this binary risk factor causally shift the tail of the loss distribution, after adjusting for confounders? Does 'post-2000 climate regime' cause higher 99th percentile flood losses, or is that confounded by changes in land use and reporting? Does driving a vehicle over ten years old causally elevate the 99th percentile of third-party bodily injury, or is that correlation driven by age and postcode?

These are hard questions. They combine two independently hard problems: causal identification (dealing with confounding) and extreme quantile estimation (dealing with data sparsity in the tail). The standard approach is to solve them sequentially — first fit a causal model, then look at the tail residuals. The problem is that sequential approaches do not have good statistical properties at extreme quantiles. Standard quantile treatment effect estimators are asymptotically normal at intermediate quantiles but their confidence intervals blow up as tau approaches 1. You can show this happens, but there has not been a clean unified solution.

Li and Castro-Camilo (arXiv:2603.23309, March 2026, Glasgow Statistics) propose one. They call it the Tail-Calibrated Inverse Estimating Equation — TIEE.

---

## What TIEE does

The target estimand is the quantile treatment effect at level tau:

```
delta(tau) = Q_{Y(1)}(tau) - Q_{Y(0)}(tau)
```

where Y(d) is the potential outcome under treatment arm d, and tau is close to 1 — say 0.99 or 0.999. This is a marginal QTE: it compares the tau-quantile of the full treated potential outcome distribution against the full control potential outcome distribution. It is *not* a conditional effect given covariates X. Keep that distinction in mind; it matters for what TIEE can and cannot do.

The estimator solves a signal equation E[S_d(W, theta)] = 0, where the signal integrates information across *all* quantile levels from 0 to 1 — not just the tail. In the bulk of the distribution, the signal uses standard inverse probability weighting: each observation is reweighted by 1/pi_d(X), where pi_d(X) is the propensity score (probability of treatment arm d given covariates X). Above a threshold — the GPD tail region — the signal replaces the raw quantile with the GPD extrapolation:

```
Q_tail(p) = u + (sigma_d / xi_d) * [((1-p_u)/(1-p))^xi_d - 1]
```

The GPD is not a post-hoc step bolted onto an existing estimator. It is embedded inside the signal function itself. Solving the estimating equation yields theta, the extreme quantile, directly. The propensity score is estimated by logistic regression; the GPD parameters are estimated from observations above threshold u, which the paper sets at the (1 - n^{-0.65})-th quantile. The integral over all quantile levels is discretized with a grid of K = 800 points (for n = 1,000) and solved as a convex L1 optimisation problem — direct root-finding is unstable because the indicator functions in the signal are non-smooth.

The key theoretical result is that theta_hat achieves sqrt(n) convergence even at extreme quantiles. Standard EVT estimators (Causal Hill, for instance) achieve only sqrt(k) convergence, where k is the number of tail observations — much slower. TIEE borrows efficiency from the bulk of the distribution, which is why the integrated signal matters. The asymptotic variance has the standard semiparametric sandwich form, with full correction for propensity score estimation uncertainty.

---

## How bad is the competition

The simulation results make the case clearly. At n = 1,000 with a very extreme tail level (1 - tau ≈ 1/6,900, which corresponds to roughly a 1-in-6,900 quantile):

| Method | MSE | Relative to TIEE |
|---|---|---|
| TIEE | 544 | 1× |
| Zhang-Firpo QTE | 6,191 | 11× worse |
| Causal Hill | 86,214 | 158× worse |

Zhang-Firpo is the standard IPW-based quantile treatment effect estimator from Firpo (2007, Econometrica). It works at moderate quantiles and fails badly at extremes because it has no EVT extrapolation. Causal Hill estimates the tail index separately per treatment arm — it has the right tail model but the wrong identification structure, and requires choosing a tail threshold separately for each arm without any integration across the rest of the distribution. Coverage at 90% nominal: TIEE achieves it; Zhang-Firpo systematically under-covers.

The simulation also tests light-tailed distributions (Normal, Exponential, Weibull). TIEE remains competitive even there — the GPD tail model does not hurt when tails are lighter than expected.

---

## Three insurance applications worth taking seriously

### UK flood: climate attribution

The paper's empirical application is Austrian precipitation extremes — does the post-industrial climate regime shift extreme precipitation quantiles after conditioning on circulation patterns? This translates directly to UK flood pricing. The question becomes: does 'post-2000 climate regime' (D = 1 for post-2000 flood years) causally shift the 99th percentile of flood loss, after adjusting for catchment characteristics, antecedent soil moisture, and North Atlantic Oscillation index?

This is actuarially useful. The answer feeds into climate change uplift factors for Flood Re and excess flood pricing. The confounders matter: catchment urbanisation has accelerated since 2000, river management has changed, LIDAR-based flood mapping has improved the reported loss data. A naive before/after comparison conflates the climate signal with all of these. TIEE gives a causal estimate with a credible confidence interval.

This is the application we would actually try to build. The data exists in EA river gauge and insurance industry loss records. The binary treatment is well-defined. The tail is genuinely heavy (precipitation extremes are in the Fréchet domain of attraction). The EVA Society's R packages give the GPD fitting machinery; the propensity score is logistic regression on catchment and circulation variables.

### Vehicle age and extreme TPBI

Does driving a vehicle over ten years old (D = 1) causally shift the 99th percentile of third-party bodily injury severity, after adjusting for policyholder age, postcode, and annual mileage?

The answer matters for per-risk excess of loss pricing by vehicle age band. TPBI severity has heavy tails — catastrophic injury claims are Pareto-like. An older vehicle fleet has worse passive safety (no lane assist, no automatic emergency braking), which should mechanically elevate the extreme tail, not just the mean. If you can show that delta(0.99) is positive and well-estimated, you have actuarial justification for differentiated per-risk XL charges by vehicle age.

The limitation here is that vehicle age as a binary treatment (old vs new, with some threshold) is a simplification. Real tariff work wants a continuous vehicle age variable. TIEE as specified handles binary D only. You could test multiple thresholds, but you cannot fit a smooth dose-response curve.

### Non-standard construction and extreme property claims

Does timber frame or thatched construction (D = 1) causally shift the 99th percentile of buildings claims relative to standard brick, after adjusting for property age, postcode flood and subsidence zone, and sum insured?

UK non-standard construction data is thin. If your book has 3,000 timber-frame properties, the far tail of claims will have perhaps 30 to 50 observations — enough for GPD fitting at moderate extremity, but marginal at 1-in-6,900 levels. This application is more realistic at the 99th or 99.5th percentile than at truly extreme levels.

---

## What TIEE cannot do

We find the limitations section more useful than the headline results, so here it is in full.

**Conditional effects at the tail.** TIEE estimates the marginal QTE — the shift in the full distribution's tau-quantile under treatment versus control. It does not answer 'given policyholder X with profile x, by how much does treatment shift their personal tail?' That is a conditional average treatment effect at the tail, sometimes called CATE-at-the-tail, and it requires a different tool. Our EQRN implementation (insurance-eqrn, covering Pasche and Chavez-Demoulin 2022) does this for the covariate-conditional extreme quantile. For the causal identification side of the TIEE setup — estimating propensity scores and treatment effects — see also [causal price elasticity for UK renewal pricing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) for the DML framework that underlies this approach. TIEE and EQRN answer genuinely different questions.

**Continuous treatments.** D must be binary. Deductible levels, rebuild cost bands, vehicle power — all continuous in practice. An IEE framework for continuous treatments exists in the econometrics literature, but TIEE specifically is binary. If your question requires a dose-response curve, TIEE is not the tool.

**Excess layer pricing.** XL pricing needs E[max(0, Y - retention)], the expected layer loss cost, not the retention-level quantile. You cannot get this directly from TIEE output — you can approximately reconstruct it from the GPD fit, but it is roundabout. If you want to fit GPD to severity directly and price a layer, you do not need TIEE; you need a GPD fitted to treaty loss data, which is a solved problem.

**Small books.** GPD fitting requires enough exceedances above the threshold u. At n^{0.65}/n threshold fraction and n = 500 claims, you have roughly 80 tail observations. That is adequate for sigma and xi estimation at moderate extremity but marginal for the 1-in-6,900 quantile range the paper demonstrates. For specialist commercial lines — offshore energy, marine hull — TIEE at its most extreme range is not credible.

**No implementation exists.** The GitHub repository (github.com/MengranLi-git/TIEE) is empty as of March 2026. No R package, no Python package. Implementing TIEE requires propensity score estimation, GPD fitting with threshold selection, an L1 convex optimisation over a K = 800+ grid, and a sandwich variance estimator. Feasible in Python using scipy, cvxpy, and our insurance-evt GPD tooling — but 400 to 600 lines of careful code, not a weekend project.

---

## Our read

TIEE is a genuine methodological advance on a real problem. The simulation results are convincing — an 11× improvement over Zhang-Firpo and 158× over Causal Hill at extreme quantiles is not noise. The theoretical contribution (sqrt(n) convergence at extreme quantiles by integrating information across all levels) is the right insight; it is surprising it took this long.

The insurance value is real but specific. Climate attribution for weather perils — UK flood in particular — is the application we would prioritise. The question is well-posed, the data is available at national scale, and the actuarial downstream use (climate uplift factors, Flood Re pricing) is concrete. The vehicle age and construction type applications are defensible but secondary.

We are watching this one rather than building it. The paper is from March 2026 with an empty GitHub repo. When an implementation appears — ideally Python, but R is fine as a starting point — we will revisit. The build score on our pipeline framework is 13/20, below our build threshold of 16/20, primarily on demand (niche estimand) and implementation feasibility (no code to wrap).

If you are working on climate attribution for flood pricing and want to discuss whether TIEE fits your specific question, get in touch.
