---
layout: post
title: "Beyond Weibull: When Does the Shape of Your Hazard Function Actually Matter?"
date: 2026-04-02
categories: [techniques, survival, research]
tags: [survival-analysis, hazard-function, ode, weibull, lapse, motor-insurance, bayesian, non-monotone, bathtub-hazard, arXiv-2602.06322, arXiv-2512.16336, survmode]
description: "A new paper models hazard functions as solutions to nonlinear ODEs, producing shapes no standard parametric family can match. The maths is genuinely interesting. The absence of a covariate model means it is not yet a pricing tool. We explain the framework, where it matters for UK insurance, and why the paper to watch is SurvMODE."
math: true
author: burning-cost
---

The Weibull distribution is the workhorse of survival analysis in UK insurance pricing. It fits motor lapse, pet retention, claims settlement timing — almost anything with a duration outcome. It has two parameters, a shape and a scale. The shape parameter governs whether the hazard is increasing (shape > 1), decreasing (shape < 1), or flat exponential (shape = 1). That is the entire range of hazard shapes available to you.

For most applications, that is sufficient. But there is a class of insurance duration problems where Weibull's monotone hazard is structurally wrong, and fitting one anyway produces residuals that politely tell you something is missing. UK motor lapse is one of them.

A paper published on arXiv in February 2026 — Liyanage, Hridoy, and Mostafa (arXiv:2602.06322) — proposes modelling hazard functions as solutions to nonlinear second-order ODE systems. The resulting shapes are rich: damped oscillatory, logistic-growth-bounded, exponentially-coupled. They can model bathtub hazards, hazards with a mid-life hump, hazards that saturate at a ceiling. Things Weibull cannot do.

We think it is an interesting piece of mathematics in search of a regression framework. Here is what it does, why it matters (within limits), and what you would actually need for a pricing team to use it.

---

## Why motor lapse has a non-monotone hazard

Motor lapse follows a pattern that anyone working in UK personal lines renewal pricing will recognise. Year 1 lapse rates are high — new policyholders switch aggressively, price-comparison site behaviour is at its peak, and the initial renewal uplift is large. Lapse rates fall through years 2–4 as the retained cohort becomes increasingly inert. Then they rise again around years 5–7 as inertia weakens, loyalty discounts erode, and comparison-site retargeting catches up.

That is a bathtub. Weibull cannot fit a bathtub. You can approximate it by mixing two Weibull components (a Weibull mixture model), which adds parameters and complexity. Or you can take a step back and ask: what functional form naturally produces the shape I'm actually seeing?

The damped oscillatory ODE does. The second-order system:

$$h''(t) + \alpha h'(t) + \beta h(t) = \gamma$$

has equilibrium $h^* = \gamma/\beta$. The discriminant $\Delta = \alpha^2 - 4\beta$ determines behaviour: underdamped ($\Delta < 0$) produces oscillation around the equilibrium, critical damping ($\Delta = 0$) produces the fastest decay without oscillation, overdamped ($\Delta > 0$) decays monotonically. For motor lapse, an underdamped solution — oscillating around a long-run lapse rate equilibrium — is mechanistically sensible.

The paper also introduces a logistic-damped model:

$$h''(t) + \eta h'(t) = r h(t)\left(1 - \frac{h(t)}{K}\right)$$

where $K$ is a carrying capacity and $r$ is a growth rate. This is a logistic equation on the hazard — the hazard grows rapidly at first but saturates at $K$ as the portfolio approaches some steady-state lapse rate. For a book approaching market equilibrium, this shape has a plausible interpretation.

---

## What the paper actually shows

The methodology is Bayesian: t-walk MCMC with 110,000 iterations, 10,000 burn-in, thinned to 20,000 posterior samples. Inference uses scipy's solve_ivp (RK45) to evaluate the likelihood at each sample. There is no neural network here — this is classical numerical ODE integration wrapped in a Bayesian sampler. That matters for reproducibility and explainability.

The simulation validation in Table 1 is credible. Parameter recovery improves cleanly as $n$ increases from 200 to 5,000; $n = 1{,}000$ is adequate for most parameters. A UK motor book with 50,000+ policies per year is well above this threshold.

The BIC improvements over Weibull on synthetic data are substantial. On sinusoidal data at $n = 2{,}000$: sinusoidal ODE BIC 8,450 versus Weibull BIC 8,536 — a difference above 80 BIC units, which is not marginal. On the population dynamics dataset: ODE BIC 12,074 versus Weibull 12,184. These are the conditions under which the ODE models are supposed to win, so this is not surprising, but it is good to see the margin quantified.

The real-data application is gastric cancer ($n = 90$). The sinusoidal ODE achieves AIC 251.29 versus Weibull 254.87. At $n = 90$, BIC penalises the extra parameters heavily and the advantage largely disappears. The paper is honest about this.

---

## The problem: there is no covariate model

Here is where it stops being a pricing tool.

The models fitted in this paper are univariate. You estimate five ODE parameters ($\alpha$, $\beta$, $\gamma$, $h_0$, $v_0$ for the damped oscillator) on the pooled data. Every policyholder in the dataset gets the same hazard function. There is no formula interface, no NCD, no age, no premium relativity, no region effect.

A survival model without regression is not a lapse model for pricing — it is a distributional fitting exercise. The question for a motor pricing team is not "does our pooled lapse hazard look non-monotone?" It is "how does lapse hazard vary with NCD level, tenure, channel, and renewal premium change?" Without covariates, the ODE framework tells you the aggregate shape; it cannot tell you who is at risk.

The extension that would make this usable is straightforward to describe and non-trivial to implement: each ODE parameter gets its own linear predictor. Log links preserve positivity:

$$\log \alpha = X^\top \beta_\alpha, \quad \log \beta = X^\top \beta_\beta, \quad \gamma = \exp(X^\top \beta_\gamma)$$

With $p$ covariates and five ODE parameters, you have $5p + 5$ free parameters. At $p = 10$ covariates, that is 55 parameters to identify. The paper already reports fragile recovery at $n = 200$ for the no-covariate case with five parameters. Adding regression without strong regularisation is going to create identifiability problems at any realistic sample size.

This is not an insurmountable problem. It is a paper that hasn't been written yet.

---

## The paper to watch: SurvMODE

The paper that does add covariate regression to ODE hazard models is arXiv:2512.16336 — Rubio and Christen, "SurvMODE: Survival Model with ODE-Driven Estimation", submitted December 2025. This is the natural sequel to the Christen and Rubio 2023 linear ODE framework that the current paper extends into the nonlinear case. SurvMODE adds distributional regression: each ODE shape parameter is linked to a covariate vector via a transformation.

The problem is that SurvMODE's primary implementation is in Julia. There is no pip-installable Python version as of April 2026. The Python ecosystem for ODE survival analysis more broadly is thin: `andreschristen/ODESurv` on GitHub covers the original linear framework in Python, but has not been updated to the nonlinear or regression extensions.

If SurvMODE acquires a Python implementation — or if someone ports the covariate regression machinery to Python against the existing `scipy.integrate.solve_ivp` infrastructure — that would be worth building. The combination of a principled nonlinear hazard shape with a proper regression framework is genuinely competitive with Weibull AFT and piecewise-exponential models for the non-monotone cases in UK insurance.

---

## When does the hazard shape actually matter?

The honest answer is: in a narrow set of UK insurance problems. Here is where we think it matters and where it doesn't.

**Motor lapse, years 1–7**: The bathtub shape is real. Most books show it. Weibull mixture models handle it clumsily; the damped oscillator handles it naturally. The value here depends on whether the non-monotone shape varies meaningfully with covariates — if the shape is homogeneous across the book, you might capture it adequately with a time-stratified Cox model. The real test would be on a book large enough ($n \geq 5{,}000$ per segment) to estimate ODE parameters reliably within covariate strata.

**Home and pet lapse**: Similar pattern, shallower. The non-monotone hazard is detectable but smaller in magnitude. Cure models — the fraction who never lapse — explain much of the apparent non-monotonicity in these lines and are already well-supported in Python. ODE hazard models add marginal value here.

**Claims settlement timing**: Settlement hazard often peaks sharply then has a long tail, which an underdamped ODE could model. In practice, competing risks (open/closed/reopened) and frailty for complex claims explain more of the heterogeneity than the marginal shape refinement. The case for ODE hazard models in claims settlement is weaker than in lapse.

**Life and critical illness**: Mortality hazard is famously non-monotone — the Bathtub curve, infant mortality through the Makeham humps to senescent mortality. The Gompertz-Makeham family handles this analytically; ODE models provide a more flexible alternative but would compete against a literature with decades of established parametric forms. UK life actuarial practice has regulatory constraints that make novel hazard forms harder to justify.

---

## What we'd do with this

The paper (arXiv:2602.06322) is available open access. The R code uses base `optim`, `nlminb`, and the `t-walk` MCMC package. For a team that wants to explore non-monotone hazard fitting on their own lapse data without covariates — perhaps as a diagnostic, to quantify how much Weibull is misspecifying the shape — the framework is accessible.

The BIC comparison against Weibull on your own data is the useful exercise. If the ODE model wins by a large margin on pooled data, that is evidence of genuine non-monotonicity and an argument for returning to this when SurvMODE has a Python implementation.

For production use, we would wait. SurvMODE (arXiv:2512.16336) is the version that matters for pricing. When it has a Python implementation with a formula interface, that is when we build.

The maths in arXiv:2602.06322 is clean and the framework is principled. But a hazard model you cannot condition on covariates is a research contribution, not a pricing tool.
