---
layout: post
title: "Probabilistic Gradient Boosting for Insurance Pricing — Beyond Point Predictions"
description: "XGBoostLSS, LightGBMLSS, NGBoost, and PGBM can all output a full conditional distribution rather than a point prediction. The Chevalier & Côté benchmark (EAJ 2025) tested 11 algorithms on 5 insurance datasets. Here is what actually works, what the distribution support gaps are, and what to use when."
date: 2026-03-28
categories: [pricing, machine-learning]
tags: [XGBoostLSS, LightGBMLSS, NGBoost, PGBM, probabilistic-regression, distributional-regression, Tweedie, ZI-Tweedie, insurance-distributional, Chevalier-Cote-2025, GBMLSS, frequency-severity, calibration, uncertainty-quantification, GAMLSS, CatBoost, Solvency-II, conformal-prediction]
---

Most GBM pricing models return a point prediction: the expected claim frequency, the expected severity, the expected pure premium. That is usually sufficient when the decision is "set the base rate." It is not sufficient when the question is "how confident are we in this rate?" or "what is the 95th percentile loss for this risk?" or "does this risk belong on the facultative reinsurance programme?"

For those questions you need a full conditional distribution — not just E[Y|X] but f(Y|X). Probabilistic gradient boosting methods do exactly that. They are gradient boosted trees that output the parameters of a chosen distribution rather than a single number.

The field has produced half a dozen competing approaches since 2019. The authoritative head-to-head is Chevalier & Côté (European Actuarial Journal, August 2025, arXiv:2412.14916), which tested 11 algorithms on 5 insurance datasets. Their conclusions are clear enough to act on. So is the distribution support matrix, which reveals why no single tool covers every insurance pricing use case.

---

## Why distributions matter in insurance pricing

The point prediction problem is a proxy for something deeper. When you fit a Gamma GLM to severity, you are implicitly asserting that the conditional distribution of claim cost is Gamma. If you use XGBoost with a Gamma deviance loss, you are fitting the same conditional mean under the same distributional assumption — but you have discarded the Gamma scale parameter entirely. You cannot recover a confidence interval, a percentile, or a premium loading without making additional assumptions you have never validated.

Three practical situations where this becomes acute:

**Capital and reinsurance.** Solvency II SCR calculations require the 99.5th percentile of the aggregate loss distribution. If your pricing model only outputs expected losses, you have no direct link from individual risk pricing to portfolio capital. A model that outputs the full conditional distribution for each risk gives you a simulation-ready input. The CoV per risk feeds straight into the capital allocation without a separate parametric fitting step.

**Pricing uncertainty and minimum premium.** A risk where f(Y|X) has a CV of 0.8 is fundamentally different from one with a CV of 0.2, even if their expected values are identical. The second risk has a higher probability of a large loss relative to the premium. For low-frequency, high-severity commercial lines, ignoring this distributional spread understates the technical minimum premium.

**Model validation.** If your model only outputs a mean, you can check the lift curve. If it outputs a distribution, you can check whether the stated 90% interval actually covers 90% of outcomes. Chevalier & Côté found that point models produce confidence intervals that are "too narrow" for severity data. That is not a calibration problem; it is a fundamental limitation of single-number outputs.

---

## Current probabilistic GBM options

There are three structurally different approaches to getting a full conditional distribution from a gradient boosted tree.

### Multi-parameter distributional boosting (GBMLSS)

For a K-parameter distribution — Gamma has 2 (mean and scale), NegBinomial has 2 (mean and dispersion), ZI-Poisson has 2 (Poisson rate and zero-inflation probability) — you grow K separate tree ensembles simultaneously. At each boosting step, the gradient and Hessian of the negative log-likelihood are computed for each parameter via automatic differentiation, and each parameter ensemble gets one new tree. The result is K separate gradient boosted models, one per distributional parameter, jointly trained on the full likelihood.

This approach is implemented in **XGBoostLSS** (März, arXiv:1907.03178, v0.6.1 December 2025) and **LightGBMLSS** (same author, v0.6.x December 2025). Both use PyTorch autograd to derive the gradients — which means PyTorch, Pyro, and Optuna arrive as dependencies. For practitioners used to the lean XGBoost + LightGBM install, this is a heavier stack than expected.

**cyc-GBM** (Delong, Lindholm, Zakrisson, SSRN 2023) takes a variant approach: rather than updating all K parameter trees in parallel each round, it cycles through them in sequence, per-parameter. The theoretical advantage is that each parameter can have its own learning rate and tree depth. In practice, it never progressed beyond a research codebase and the GitHub repository has been dormant since July 2023. We would not put this in a production model.

### Natural gradient boosting (NGBoost)

NGBoost (Duan et al., Stanford ML Group, ICML 2020, arXiv:1910.03225) addresses a legitimate theoretical concern with GBMLSS: the standard gradient of the log-likelihood with respect to distributional parameters is not parameterisation-invariant. If you reparametrise a Normal from (mu, sigma) to (mu, log-sigma), you get different gradient directions. The natural gradient — the standard gradient premultiplied by the inverse Fisher information matrix — is invariant under reparametrisation.

In practice, the theoretical elegance carries a computational cost. NGBoost uses sklearn regression trees as base learners rather than a native GBDT engine. Sklearn trees are slower by a meaningful factor. And despite being actively maintained (v0.5.10 released March 2026), the distribution coverage for insurance is limited: Normal, LogNormal, Exponential, Poisson, StudentT. No Gamma. No NegBinomial. No zero-inflated variants.

### Variance-based uncertainty (PGBM)

PGBM (Sprangers, Schelter, de Rijke, KDD 2021, arXiv:2106.01682) takes a fundamentally different and more indirect approach. It uses a modified HistGradientBoosting (LightGBM-based) and tracks the sample variance of leaf predictions across trees. After training, it fits a distribution to the per-sample (mean, variance) pair by method of moments. You can get a Tweedie fit this way — the Tweedie loss is available in the underlying LGBM — but you are fitting the distribution to estimated moments rather than maximising the target distribution's likelihood directly.

This works for generic uncertainty quantification. For insurance distributions where you need to estimate two structurally distinct parameters — say, the zero-inflation probability and the Poisson rate in a ZIP model — method-of-moments fitting from a single (mean, variance) pair is not sufficient. You need distinct gradient signals for each parameter.

---

## What the benchmark says

Chevalier & Côté (EAJ 2025, arXiv:2412.14916) is the closest thing the field has to a definitive insurance benchmark for probabilistic GBM. Eleven algorithms, five public insurance datasets (including BelgianMTPL and FreMTPL variants), measured on four dimensions: computational efficiency, predictive accuracy (Poisson/Gamma/LogNormal deviance), calibration (coverage at 50%, 75%, 95%), and model adequacy (portfolio balance, tariff structure).

The findings we think matter most for UK pricing teams:

**On algorithm selection for accuracy:** When the data is relatively homogeneous, all algorithms achieve similar deviance. The choice of distribution matters more than the choice of algorithm. Where differences appear is with high-cardinality categorical variables — vehicle make, occupation, postcode sector — where CatBoost's ordered target statistics give a measurable edge. For a UK motor book with 2,000+ vehicle makes and 9,000 postcode sectors, that is not a hypothetical advantage.

**On probabilistic methods specifically:** Point models produce confidence intervals that are "too narrow" for severity data. XGBoostLSS provides the best coverage of stated confidence intervals among the probabilistic methods. NGBoost also performs adequately on lognormal severity coverage. cyc-GBM and NGBoost lose on average rank in three of four summary tables — cyc-GBM additionally has portfolio balance problems.

**On speed:** LightGBM is the fastest algorithm overall, with essentially no accuracy penalty. XGBoostLSS is the fastest among the probabilistic methods. NGBoost is meaningfully slower due to the sklearn tree back-end.

**On EGBM:** The benchmark validates InterpretML's Explainable Boosting Machine as "competitive performance, no trade-off" against GBM. For pricing models facing direct regulatory scrutiny, this is worth noting — EGBM's shape functions are literally the model, which makes the PRA explainability conversation much simpler. We are monitoring whether IFoA working parties start recommending EBM over GBM+SHAP.

**What the benchmark does not cover:** No Tweedie or compound Poisson-Gamma modelling. No zero-inflated distributions. No CatBoost as a distributional model (only as a point predictor). No actuarial-specific outputs such as layer expected values or CRPS diagnostics.

---

## Distribution support: the gap table

The single most practically important fact about the current ecosystem is this: no library supports all the distributions UK insurance pricing teams actually need. The gap is not minor.

| Method | Poisson | NegBin | Gamma | Tweedie | ZI-Poisson | ZI-NegBin | ZA-Gamma | ZI-Tweedie | Exposure offset |
|--------|---------|--------|-------|---------|------------|-----------|----------|------------|-----------------|
| XGBoostLSS | Y | Y | Y | **NO** | Y | Y | Y | **NO** | Unclear |
| LightGBMLSS | Y | Y | Y | **NO** | Y | Y | Y | **NO** | Unclear |
| NGBoost | Y | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** |
| PGBM | indirect | indirect | indirect | indirect | **NO** | **NO** | **NO** | **NO** | Via LGBM |
| cyc-GBM | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** | **NO** |
| insurance-distributional | **NO** | Y | Y | Y | Y | **NO** | **NO** | **GAP** | **YES** |

Three things stand out.

First, XGBoostLSS and LightGBMLSS have 26 distributions but Tweedie is not among them. This is not an oversight — the compound Poisson-Gamma parameterisation does not map cleanly onto the GBMLSS framework. Tweedie is characterised by a power parameter p that moves continuously between the Poisson (p=1) and Gamma (p=2) cases. Fitting p as a separate distributional parameter alongside mu and phi turns out to be numerically unstable. So the most commonly used loss function in insurance GBM pricing is unavailable in the two most capable distributional GBM libraries.

Second, NGBoost's distribution coverage is thin. This is a constraint of the natural gradient framework — adding a new distribution requires deriving the Fisher information matrix analytically. There is no autograd path here.

Third, our [insurance-distributional](https://github.com/open-insurance/insurance-distributional) library fills the CatBoost gap: TweedieGBM, GammaGBM, NegBinomialGBM, and ZIPGBM all with a first-class `exposure=` argument to `fit()`. The exposure offset question — handled awkwardly in XGBoostLSS via `base_score` and not at all in NGBoost — is a proper actuarial API. But insurance-distributional has its own gap, which brings us to the most important missing piece in the ecosystem.

---

## The ZI-Tweedie gap

So & Valdez (ASTIN Best Paper 2024, arXiv:2406.16206, published Applied Soft Computing 2025) introduced zero-inflated Tweedie gradient boosting with CatBoost as the backend. The motivation is direct: for most personal lines portfolios, a non-trivial fraction of policies generate zero claims in any given period — not because their expected frequency is zero, but because claim frequency is low enough that the observed period produces no event. Standard Tweedie handles this in expectation but does not model the excess zeros explicitly.

The ZI-Tweedie formulation is:

P(Y=0) = q + (1-q) × P(Y_Tweedie = 0)

f(Y=y | Y>0) = (1-q) × f_Tweedie(y)

The paper presents two parameterisations. Scenario 1 uses separate CatBoost ensembles for logit(q) and log(mu) in coordinate descent — structurally similar to GBMLSS but with CatBoost trees. Scenario 2 is more elegant: logit(q) is constrained to equal -γ × (log(E) + W_T(x)), making q a functional of mu. This reduces the parameter space to a single scalar γ and a single set of trees for mu. Simpler, fewer parameters, and in practice it performs comparably to Scenario 1.

So & Deng (NAAJ 2025, doi:10.1080/10920277.2025.2454460) extended this further, comparing XGBoost, LightGBM, and CatBoost backends explicitly for ZI-Tweedie tasks.

No pip-installable library implements any of this. insurance-distributional cites So & Valdez as its primary reference but does not yet implement ZI-Tweedie — implementing Scenario 2 as `ZeroInflatedTweedieGBM` is the single highest-priority gap in the open-source ecosystem. It would differentiate insurance-distributional from XGBoostLSS and LightGBMLSS on the distributions that matter most to UK motor and home pricing.

---

## What we recommend

For a UK pricing team building distributional GBM models today:

**For frequency and severity modelling (not combined), with non-CatBoost data:** Use XGBoostLSS or LightGBMLSS. LightGBMLSS will be faster on most datasets. Accept the PyTorch/Pyro dependency overhead. You get Gamma, Poisson, NegBinomial, ZI-Poisson, ZA-Gamma — everything except Tweedie. Use the benchmark finding that XGBoostLSS gives the best confidence interval coverage as justification for the approach internally.

**For high-cardinality categoricals (motor, commercial lines):** CatBoost point models with insurance-distributional for distributional layers. The Chevalier & Côté finding on CatBoost's categorical advantage is consistent with what we see on UK motor data.

**For pure premium (frequency × severity combined):** insurance-distributional with TweedieGBM is the only pip-installable option with a proper exposure offset API. It is CatBoost-backed, which handles the categorical variables well.

**For zero-inflated aggregate claims:** Wait for ZI-Tweedie implementation, or implement Scenario 2 from So & Valdez directly using the paper's appendix — it is not a large implementation lift. The framework in insurance-distributional makes this a tractable addition.

**For regulatory explainability requirements:** Consider EGBM (InterpretML) as a complement. The Chevalier & Côté benchmark shows no accuracy trade-off, and the interpretability case to the PRA is considerably cleaner.

What we would avoid: NGBoost for insurance applications given the narrow distribution support. cyc-GBM in any production context — it is abandoned research code. PGBM where you need to estimate structurally distinct distributional parameters, because the method-of-moments back-end is not suited to that task.

The honest summary is that the probabilistic GBM ecosystem is more fragmented than the marketing suggests. XGBoostLSS/LightGBMLSS are the best single choice for probabilistic methods, but they have a Tweedie-shaped hole in them. The paper that won the ASTIN Best Paper award in 2024 addresses exactly that hole and nobody has published the code. We will fix that.
