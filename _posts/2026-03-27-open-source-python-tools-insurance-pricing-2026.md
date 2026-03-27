---
layout: post
title: "Open-Source Python Tools for Insurance Pricing: What's Actually Available in 2026"
date: 2026-03-27
featured: true
categories: [tools, pricing]
tags: [python, open-source, insurance-pricing, scikit-learn, catboost, statsmodels, chainladder, actuarial, glm, gbm, pymc, survey]
description: "A definitive survey of open-source Python tools for insurance pricing in 2026. General-purpose ML libraries, specialist actuarial packages, the Burning Cost stack, and honest gaps. The post a pricing actuary bookmarks."
author: Burning Cost
---

Every few months someone asks in an actuarial Slack channel: "what Python libraries should I actually be using for pricing?" The answers are a mix of "scikit-learn obviously" and "I built something in pandas" and occasionally a GitHub link that goes nowhere.

This is our attempt at a proper answer. We have surveyed every meaningful open-source Python tool for insurance pricing as of March 2026. We cover general-purpose ML libraries pricing teams actually use, the specialist actuarial ecosystem, our own libraries where they fill real gaps, and — critically — what is still missing.

We will be honest about where commercial tooling is genuinely better. The goal is to be useful, not promotional.

---

## General-purpose ML libraries

These are the tools every UK pricing team is already using. We will not re-explain what a random forest is. Instead, we will note what they do and do not do for the specific workflow of GI pricing.

### scikit-learn

The baseline. The `TweedieRegressor` (added in v0.23, now mature) handles the Tweedie compound Poisson-Gamma distribution that sits at the heart of personal lines pricing — a log link, Tweedie power parameter, with the same exposure-as-offset pattern you want. `GammaRegressor` and `PoissonRegressor` are there for frequency-severity split modelling.

What it does not give you: factor tables, A/E diagnostics, rating band collapsing, exposure weighting in any meaningful sense beyond `sample_weight`. You can get a fitted Tweedie GLM out of scikit-learn in 15 lines. Turning that into a set of auditable rating relativities that a pricing committee can review takes considerably more work, and you are writing that yourself.

The `Pipeline` and `ColumnTransformer` API work well for encoding; `cross_val_score` with appropriate stratification is reasonable for out-of-time splits. The tooling is not the problem. What is missing is the actuarial interpretation layer on top of it.

**Honest verdict:** The best starting point for anyone building a pricing model from scratch. Not a pricing-complete solution.

### CatBoost, XGBoost, LightGBM

The GBM stack. In UK personal lines, CatBoost has earned a strong position because it handles categorical variables natively (critical for vehicle codes, occupation, postcode sectors) without target encoding leakage, and its `CatBoostRegressor` with `loss_function='Tweedie'` fits directly on the right response distribution.

XGBoost's `reg:tweedie` and LightGBM's `tweedie` objective work similarly. All three expose `scale_pos_weight` and `sample_weight` — you need the latter for exposure-weighted fitting.

The interpretability problem is acute: none of them produce factor tables. TreeSHAP (SHAP library, v0.44+) solves the attribution question but does not produce the multiplicative relativities that rating engines expect and actuarial committees review. That translation step is non-trivial and the reason [shap-relativities](https://github.com/burning-cost/shap-relativities) exists.

**Honest verdict:** CatBoost is our preferred GBM for most pricing tasks. The tooling around factor extraction is where you need additional work.

### statsmodels

The most underappreciated library in the actuarial Python stack. `statsmodels.genmod.generalized_linear_model.GLM` with `families.Tweedie()`, `Poisson()`, or `Gamma()` gives you a proper maximum likelihood GLM with standard errors, deviance residuals, and a `summary()` that shows something an actuary can interpret rather than just a loss curve.

The offset argument for exposure is clean: `glm = smf.glm(formula, data, family=Tweedie(...), offset=np.log(data.exposure))`. Factor tables from a fitted GLM are just `np.exp(glm_result.params)`. This is the workflow that Emblem wraps commercially.

The `formula` API (via patsy) handles factor encoding, interaction terms, and spline bases. Smoothing via penalised regression (ridge) is in there but clunky.

`statsmodels` is notably absent from most ML blog posts because it does not do gradient boosting. For GLM-native actuarial work, it is the right tool.

**Honest verdict:** Use `statsmodels` for any GLM where you need proper statistical inference — coefficient CIs, LRT tests, deviance. Do not use it as a drop-in for GBM.

### glum

[glum](https://github.com/Quantco/glum) (Quantco, v2.x) deserves a specific mention because it fills a real gap: a fast, numerically stable GLM with Lasso/Ridge/ElasticNet regularisation, a scikit-learn-compatible API, and proper handling of Tweedie/Poisson/Gamma families. On large books (2M+ policies), `statsmodels` GLM fitting becomes slow; glum is materially faster through sparse matrix arithmetic and a coordinate descent solver.

It is not as widely known in UK pricing teams as it should be. If you are fitting large GLMs with regularisation — either for variable selection or shrinkage of sparse factor levels — `glum` is worth knowing.

**Honest verdict:** Recommended over statsmodels when you have large data or need regularisation. Still does not produce factor tables in actuarial format.

### PyMC

For Bayesian hierarchical models — credibility weighting, thin-data portfolios, spatial territory rating — PyMC v5 is the primary Python tool. The NUTS sampler handles the non-conjugate posteriors that Bühlmann-Straub does not. `BoundedNormal`, log-normal priors on rate factors, partial pooling across geographies: the full hierarchical GLM workflow is possible.

The practical barrier is computational. A well-specified Bayesian hierarchical model on a 500k-policy book takes hours on a single machine. Variational inference (`pymc.fit()`) is faster but trades accuracy for speed in ways that can mislead. The tooling is powerful and the results are genuinely useful for thin-data problems — commercial lines, new products, geographic credibility — but it requires significant statistical expertise to specify and interpret correctly.

**Honest verdict:** Right tool for hierarchical credibility and thin-data problems. Steep learning curve. Not a replacement for frequentist GLMs on large personal lines books.

### SciPy

Used for distribution fitting (`scipy.stats`) — Gamma, Lognormal, Pareto, Weibull for severity; Poisson and negative binomial for frequency. `scipy.optimize` is the solver layer under most constrained optimisation. `scipy.stats.ks_2samp` and `anderson_ksamp` appear in portfolio shift detection. More infrastructure than tool — you use it without thinking about it.

---

## Specialist actuarial packages

### chainladder-python

[chainladder-python](https://github.com/casact/chainladder-python) is the strongest specialist actuarial Python package. It implements the full reserving toolkit: chain ladder, Bornhuetter-Ferguson, Cape Cod, Clark LDF curves, Mack variance, bootstrap IBNR distributions, and the Munich chain ladder. The API is clean, it works with development triangles in a way that statsmodels never could, and the community (CAS) is active.

It covers claims reserving. It does not cover pricing — no frequency/severity GLMs, no rating relativities, no exposure rating. A reserving actuary building a Python toolkit should start here. A pricing actuary building a Python toolkit should know it exists and stop there.

**Honest verdict:** Best-in-class for reserving. Zero overlap with pricing workflow.

### lifelib

[lifelib](https://github.com/fumitoh/lifelib) covers life insurance: whole life, term life, savings products, with-profit endowments, economic scenario generators for IFRS 17. The underlying engine (modelx) is designed for actuarial projection models with audit trails.

GI pricing has no use for it. Mention it because it demonstrates that the life actuarial space has invested in tooling in a way that GI pricing has not.

### pyliferisk

[pyliferisk](https://github.com/franciscogarate/pyliferisk) computes actuarial commutation functions from life tables — annuity present values, life expectancy, A/E ratios against standard mortality tables. Single-file library. Used by life actuaries for exam calculations and pricing formulae. No connection to GI pricing.

### GEMAct

[GEMAct](https://github.com/gpitt71/gemact-code) handles aggregate loss distributions and reinsurance pricing: collective risk model, Panjer recursion, copula-based aggregate distributions, stop-loss pricing. Around 30 GitHub stars as of early 2026, but genuine technical depth — it is doing real actuarial mathematics.

It covers reinsurance and aggregate risk, not primary GI pricing frequency/severity modelling. A reinsurance pricing actuary should know about it.

### insurancerating (R, not Python)

The closest thing in any open-source language to a proper GI pricing workflow library. [insurancerating](https://github.com/MHaringa/insurancerating) handles GLM fitting, A/E ratio visualisation, factor tables, premium relativities — the core pricing workflow — in R. It has 50+ GitHub stars and active maintenance.

No Python equivalent exists. This is the gap the Burning Cost stack partially addresses, though we are building upward from specialist tools rather than providing the full workflow in a single package.

### actuarial-science notebooks and tutorials

There are a handful of GitHub repositories — DeutscheAktuarvereinigung/claim_frequency, open-source-modelling/insurance_python, a scatter of university course repos — that look like libraries but are Jupyter notebooks. They are tutorials, not installable tools. Do not mistake them for maintained libraries. We checked in March 2026: none are on PyPI, most have zero recent activity.

---

## The Burning Cost stack

We built these because the gap above is real. UK GI pricing teams using Python are assembling tools from general-purpose ML libraries that were not designed for actuarial work, then writing a lot of bespoke code around them. Each library below addresses a specific failure mode we encountered or observed repeatedly.

We should be honest upfront: these are relatively new libraries (most published 2025–2026), the download numbers are modest (roughly 250–1,050 installs/month per package as of March 2026, and the distribution is suspiciously uniform — likely mirror traffic alongside real users), and some are more mature than others. Where there are known limitations, we say so.

### [insurance-fairness](https://github.com/burning-cost/insurance-fairness) — proxy discrimination auditing

The problem: postcode is a legitimate risk variable, but it correlates with ethnicity. FCA Consumer Duty and EP25/2 require you to demonstrate it is not operating as a proxy. Spearman correlation misses non-linear categorical relationships entirely.

The library runs CatBoost proxy R² and mutual information scoring per protected characteristic, and produces an evidence pack structured for Consumer Duty sign-off.

`pip install insurance-fairness`

### [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) — model drift detection

The problem: aggregate A/E ratios look stable, but drift is concentrated in young drivers and cancels at portfolio level. By the time the aggregate deteriorates, you are already mispriced.

The library computes exposure-weighted PSI/CSI per segment, segmented A/E with IBNR adjustment, Gini z-test, and a formal recalibrate-vs-refit decision rule. The `PITMonitor` (e-process martingale for calibration drift) runs at 3% false positive rate versus 46% for repeated Hosmer-Lemeshow. `InterpretableDriftDetector` uses BH-corrected feature-level attribution to identify which rating factors are driving drift.

`pip install insurance-monitoring`

### [insurance-conformal](https://github.com/burning-cost/insurance-conformal) — prediction intervals

The problem: standard conformal prediction achieves aggregate coverage but systematically undercovers the highest-risk decile — the segment that matters most for SCR and reinsurance cost.

Locally-weighted non-conformity scores adapt interval width to local variance. Distribution-free, finite-sample coverage. Solvency II SCR bounds included.

`pip install insurance-conformal`

### [shap-relativities](https://github.com/burning-cost/shap-relativities) — GBM-to-factor-table extraction

The problem: your CatBoost model beats the production GLM by 3 Gini points. The rating engine needs multiplicative factor tables. There is no `exp(β)` in CatBoost.

The library extracts TreeSHAP values, exposure-weights them per rating band, and produces the factor table format that pricing committees and rating engines expect, with reconstruction R² to validate fidelity.

`pip install shap-relativities`

### [insurance-causal](https://github.com/burning-cost/insurance-causal) — deconfounding and price elasticity

The problem: vehicle value looks significant in the GLM, but it correlates with distribution channel. You cannot tell whether it is risk signal or channel confounding, and ordinary regression cannot separate the two.

Double machine learning residualises both outcome and treatment on confounders using CatBoost nuisance models, then estimates the causal effect in the residuals. `causal_forest` extends this to segment-level heterogeneous treatment effects (GATES/CLAN/RATE). Known limitation: over-partialling at n < 50k — the nuisance models absorb signal they should not. We document this in the README.

`pip install insurance-causal`

### [insurance-optimise](https://github.com/burning-cost/insurance-optimise) — constrained rate optimisation

The problem: rate change recommendations are done in spreadsheets where constraints interact and the solution is not optimal. You have a technical price, a loss ratio target, movement caps, and FCA ENBP constraints. A spreadsheet solver cannot handle all of these simultaneously with any guarantee of optimality.

SLSQP with analytical Jacobians finds the optimal rate changes across these constraints. `ParetoFrontier` (v0.4.1) makes the profit/retention/fairness trade-off explicit — single-objective optimisation produces a premium disparity ratio of 1.168 in the benchmark; the Pareto surface is the defensible alternative.

`pip install insurance-optimise`

### [insurance-governance](https://github.com/burning-cost/insurance-governance) — PRA SS1/23 validation reports

The problem: model validation reports are produced manually in PowerPoint.

The library runs bootstrap Gini CI, Poisson A/E CI, double-lift charts, and a renewal cohort test, structured to what a PRA SS1/23 review expects. HTML and JSON output. The benchmark case demonstrates that manual checklists miss miscalibration concentrated in young drivers (age < 30) that the automated suite catches via Hosmer-Lemeshow (p < 0.0001).

`pip install insurance-governance`

### [insurance-severity](https://github.com/burning-cost/insurance-severity) — spliced distributions and EVT

The problem: a single Gamma GLM fits attritional claims but fails at the tail. Large losses follow a Pareto distribution with different physics; the same covariate effects do not apply.

Spliced body-tail models with covariate-dependent thresholds, composite Lognormal-GPD for heavy tails, Deep Regression Networks for non-parametric severity, EQRN extreme quantile neural networks. ILF tables and TVaR per risk. The EVT module corrects for policy limit truncation, which naïve GPD ignores.

`pip install insurance-severity`

### [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) — causal rate change evaluation

The problem: you put through a rate increase in Q3 and conversion dropped. You cannot separate the rate change effect from market conditions because you have no control group and the before/after comparison is confounded by market inflation.

Synthetic difference-in-differences constructs a synthetic control from unaffected segments to isolate the rate change effect. HonestDiD sensitivity analysis for violations of parallel trends. FCA evidence pack output.

`pip install insurance-causal-policy`

### [insurance-quantile](https://github.com/burning-cost/insurance-quantile) — tail risk quantile regression

The problem: your mean model gives no handle on the upper tail. Large loss loading and ILF curves require quantile estimates at the 90th and 99th percentile per risk.

Quantile and expectile GBMs at multiple levels simultaneously, producing per-risk TVaR and ILF tables in actuarial format. Known limitation: lognormal parametric models win at small n — the GBM advantage materialises at larger portfolios where the non-linear covariate effects on the tail are estimable.

`pip install insurance-quantile`

---

## What is still missing

Being honest about gaps matters more than overstating coverage.

**The core pricing workflow in a single package.** `insurancerating` in R provides GLM fitting, A/E diagnostics, factor tables, and relativities as an integrated workflow. No Python equivalent exists. The Burning Cost stack provides specialist tools around the edges of this workflow, not the workflow itself. Assembling a complete Python pricing workflow still requires significant glue code.

**Rating engine integration.** Radar and Emblem cannot be automated from Python in any open-source way. Radar's Python integration (September 2024) lets you call Python models at runtime, but the gap between a fitted model and a populated rating engine is still largely manual. There is no open-source rating engine for Python that replicates what Radar and Emblem do for production pricing.

**PCW simulation.** Price comparison website dynamics — competitor price distributions, rank-response curves, conversion by position — are a core part of UK personal lines strategy. There is no open-source Python tool for this. Teams build bespoke models internally, and the input data (competitor quotes) requires commercial aggregator access that open-source tooling cannot abstract away.

**Claims triage and fraud.** The ML stack for claims triage is general-purpose: XGBoost/LightGBM on claims characteristics, survival models for settlement timing, network analysis for fraud rings. None of this is packaged specifically for insurance, and the data requirements (claims history, linkage to policy) are too heterogeneous for a general library. The space is dominated by commercial vendors (FRISS, Shift Technology, Verisk).

**Telematics feature engineering at scale.** Raw telematics data (trip-level GPS, accelerometer events) to pricing features is an unsolved pipeline problem in open-source. [insurance-telematics](https://github.com/burning-cost/insurance-telematics) covers the modelling end (HMM risk scoring, trip features to GLM relativities), but the ingestion-and-feature-engineering pipeline from raw telemetry is a different and harder problem.

**Demand and elasticity modelling.** Price elasticity modelling for renewal retention is genuinely hard to do correctly — it requires handling confounding (rate increases correlate with claims history), censoring (lapsed policies), and simultaneity (your price and the market price move together). [insurance-causal](https://github.com/burning-cost/insurance-causal) addresses the causal identification part. The full demand modelling workflow (elasticity curves by segment, retention surface, optimal renewal price) is not packaged anywhere.

---

## Commercial alternatives

The honest version of this section: commercial tools are genuinely better for several tasks.

**Radar (Applied Systems) and Emblem (Verisk)** are the dominant UK personal lines pricing platforms. They handle the full rating engine workflow — factor tables, rate testing, override management, sign-off trails, regulatory evidence — in a way that no open-source tooling approaches. Emblem's GLM fitting with exposure rating, credibility, and A/E diagnostics in a single integrated environment is significantly more productive than assembling the same from statsmodels and pandas. Radar's rating engine is production-grade in a way that a Python script is not.

If you are running a UK personal lines pricing function and you do not already have Radar or Emblem, that is probably the right investment before any open-source Python tooling. Open-source complements these tools — for the modelling that happens before the rating engine, for the regulatory evidence the rating engine cannot produce, for the causal inference that the rating engine was never designed to do.

**Earnix** provides a combined pricing and optimisation platform with real-time pricing capability. The commercial case is strongest where you need real-time rating model updates and direct integration with the policy administration system. Open-source cannot match this.

**Willis Towers Watson (ICT, ResQ)** covers reserving at the enterprise level in ways chainladder-python does not. For a large carrier doing complex commercial lines IBNR, a commercial reserving system is the right tool.

**Guidewire** is a policy administration platform, not a pricing tool. It appears in these conversations because it can consume pricing model outputs, but pricing in Guidewire is typically done via rule engines rather than statistical models.

The honest framing: commercial platforms win on workflow integration, audit trails, regulatory sign-off, and production stability. Open-source tools win on methodological flexibility, cost, and the ability to do things commercial platforms were not designed for — causal inference, distributional prediction, Bayesian credibility, proxy discrimination testing. The right architecture usually involves both.

---

## Where to start

If you are setting up a Python pricing workflow from nothing:

1. **GLM baseline:** `statsmodels` or `glum` for Tweedie/Poisson/Gamma GLMs. Get comfortable with the formula API and understand what deviance and A/E ratios are telling you before moving to GBMs.

2. **GBM layer:** CatBoost for most pricing tasks (categorical handling, Tweedie objective, native SHAP). XGBoost or LightGBM as alternatives.

3. **Interpretation:** [shap-relativities](https://github.com/burning-cost/shap-relativities) to extract factor tables from GBMs. The SHAP library directly for feature attribution.

4. **Validation:** [insurance-governance](https://github.com/burning-cost/insurance-governance) for structured validation reports. [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) for ongoing drift detection.

5. **Regulatory:** [insurance-fairness](https://github.com/burning-cost/insurance-fairness) if you are in UK personal lines and need Consumer Duty proxy evidence.

6. **Specialist needs:** [insurance-causal](https://github.com/burning-cost/insurance-causal) for deconfounding, [insurance-severity](https://github.com/burning-cost/insurance-severity) for tail modelling, [insurance-optimise](https://github.com/burning-cost/insurance-optimise) for constrained rate changes, [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) for rate change evaluation.

All ten flagship libraries are documented at [burning-cost.github.io/top-10/](/top-10/) with specific benchmark results, known failure modes, and honest performance claims.

If this list is incomplete or wrong, [we want to know](https://burning-cost.github.io/work-with-us/).
