---
layout: page
title: "Open-Source Python Libraries for Insurance Pricing"
description: "25 production-ready Python libraries for UK personal lines pricing. From SHAP relativities and Bayesian credibility to causal inference and regulatory compliance."
permalink: /tools/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-datasets",
      "description": "Synthetic UK motor data with a known data-generating process, for testing and teaching.",
      "codeRepository": "https://github.com/burning-cost/insurance-datasets",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-cv",
      "description": "Temporal walk-forward cross-validation with IBNR buffers and sklearn-compatible scorers.",
      "codeRepository": "https://github.com/burning-cost/insurance-cv",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-synthetic",
      "description": "Vine copula synthetic portfolio generation preserving multivariate dependence structure.",
      "codeRepository": "https://github.com/burning-cost/insurance-synthetic",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "shap-relativities",
      "description": "Multiplicative rating factor tables from CatBoost models via SHAP, in the same format as exp(beta) from a GLM.",
      "codeRepository": "https://github.com/burning-cost/shap-relativities",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-interactions",
      "description": "Automated GLM interaction detection using CANN, NID, and SHAP-based methods.",
      "codeRepository": "https://github.com/burning-cost/insurance-interactions",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-anam",
      "description": "Actuarial neural additive model in PyTorch: interpretable deep learning for pricing.",
      "codeRepository": "https://github.com/burning-cost/insurance-anam",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "bayesian-pricing",
      "description": "Hierarchical Bayesian models for thin-data pricing segments using PyMC 5.",
      "codeRepository": "https://github.com/burning-cost/bayesian-pricing",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-multilevel",
      "description": "CatBoost combined with REML random effects for high-cardinality categorical groups.",
      "codeRepository": "https://github.com/burning-cost/insurance-multilevel",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "credibility",
      "description": "Buhlmann-Straub credibility in Python with mixed-model equivalence checks.",
      "codeRepository": "https://github.com/burning-cost/credibility",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "experience-rating",
      "description": "NCD and bonus-malus systems for UK motor insurance, including claiming threshold optimisation.",
      "codeRepository": "https://github.com/burning-cost/experience-rating",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal",
      "description": "Distribution-free prediction intervals for insurance GBMs with finite-sample coverage guarantees.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-quantile",
      "description": "Quantile and expectile GBMs for tail risk, TVaR, and increased limit factors.",
      "codeRepository": "https://github.com/burning-cost/insurance-quantile",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-distributional",
      "description": "Distributional GBMs with Tweedie, Gamma, ZIP, and negative binomial objectives.",
      "codeRepository": "https://github.com/burning-cost/insurance-distributional",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-causal",
      "description": "Causal inference via double machine learning for deconfounding rating factors.",
      "codeRepository": "https://github.com/burning-cost/insurance-causal",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-elasticity",
      "description": "Causal price elasticity estimation via CausalForestDML and DR-Learner.",
      "codeRepository": "https://github.com/burning-cost/insurance-elasticity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-causal-policy",
      "description": "Synthetic difference-in-differences for causal rate change evaluation and FCA evidence packs.",
      "codeRepository": "https://github.com/burning-cost/insurance-causal-policy",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-demand",
      "description": "Conversion, retention, and DML price elasticity modelling integrated with rate optimisation.",
      "codeRepository": "https://github.com/burning-cost/insurance-demand",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "rate-optimiser",
      "description": "Constrained rate change optimisation with efficient frontier between loss ratio target and movement cap constraints.",
      "codeRepository": "https://github.com/burning-cost/rate-optimiser",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-optimise",
      "description": "SLSQP portfolio rate optimisation with analytical Jacobians for large factor spaces.",
      "codeRepository": "https://github.com/burning-cost/insurance-optimise",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-fairness",
      "description": "Proxy discrimination auditing and FCA Consumer Duty documentation support.",
      "codeRepository": "https://github.com/burning-cost/insurance-fairness",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-validation",
      "description": "Structured PRA SS1/23 model validation reports covering nine required sections, output as HTML and JSON.",
      "codeRepository": "https://github.com/burning-cost/insurance-validation",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-monitoring",
      "description": "Exposure-weighted PSI/CSI, actual-vs-expected ratios, and Gini drift z-tests for deployed models.",
      "codeRepository": "https://github.com/burning-cost/insurance-monitoring",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-mrm",
      "description": "Model risk management: ModelCard, ModelInventory, and GovernanceReport generation.",
      "codeRepository": "https://github.com/burning-cost/insurance-mrm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-deploy",
      "description": "Champion/challenger framework with shadow mode, rollback, and full audit trail.",
      "codeRepository": "https://github.com/burning-cost/insurance-deploy",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-spatial",
      "description": "BYM2 spatial models for postcode-level territory ratemaking, borrowing strength from neighbours.",
      "codeRepository": "https://github.com/burning-cost/insurance-spatial",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-survival",
      "description": "Cure models, customer lifetime value, lapse tables, and MLflow wrapper for retention modelling.",
      "codeRepository": "https://github.com/burning-cost/insurance-survival",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-whittaker",
      "description": "Whittaker-Henderson smoothing for experience rating tables with REML lambda selection and Bayesian CIs.",
      "codeRepository": "https://github.com/burning-cost/insurance-whittaker",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-ebm",
      "description": "Insurance workflow layer for interpretML EBMs: relativity tables, monotonicity editing, GLM comparison, actuarial diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-ebm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-ilf",
      "description": "Increased limits factor curves from severity distributions with bootstrap confidence intervals.",
      "codeRepository": "https://github.com/burning-cost/insurance-ilf",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    }
  ]
}
</script>

Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. These libraries are the practical output of that research — each one solving a specific problem in the pricing workflow, built to run on Databricks, tested against actuarial standards. All 29 libraries are on PyPI and MIT-licensed.

The 10 most useful notebooks are collected in the [Databricks Notebook Archive](/notebooks/). Download the full set as a zip and import directly into Databricks — no cluster setup needed beyond the `%pip install` in the first cell.

---

## Data & Validation

**[insurance-datasets](https://github.com/burning-cost/insurance-datasets)**
Synthetic UK motor data with a known data-generating process, for testing pricing pipelines and teaching without touching real customer data.
`uv add insurance-datasets`

**[insurance-cv](https://github.com/burning-cost/insurance-cv)**
Temporal walk-forward cross-validation with IBNR buffers and sklearn-compatible scorers. Standard k-fold cross-validation gives you optimistic estimates on insurance data because it leaks future claims into training.
`uv add insurance-cv`
&rarr; [Why your cross-validation is lying to you](https://burning-cost.github.io/2026/02/23/why-your-cross-validation-is-lying-to-you/)

**[insurance-synthetic](https://github.com/burning-cost/insurance-synthetic)**
Vine copula synthetic portfolio generation that preserves the multivariate dependence structure of your actual book, for scenario testing and model benchmarking.
`uv add insurance-synthetic`
&rarr; [Generating realistic synthetic insurance portfolios](https://burning-cost.github.io/2026/03/09/insurance-synthetic/)

**[insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)**
Whittaker-Henderson smoothing for experience rating tables. 1D, 2D, and Poisson variants with REML/GCV/AIC lambda selection and Bayesian confidence intervals. Borrows strength from neighbours to smooth noisy age curves, NCD scales, and vehicle group relativities. Pure NumPy/SciPy.
`uv add insurance-whittaker`

---

## Model Building

**[shap-relativities](https://github.com/burning-cost/shap-relativities)**
Multiplicative rating factor tables from CatBoost models via SHAP, in the same format as exp(beta) from a GLM. Exports to Excel and Radar-compatible CSV.
`uv add shap-relativities`
&rarr; [Extracting rating relativities from GBMs with SHAP](https://burning-cost.github.io/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)

**[insurance-interactions](https://github.com/burning-cost/insurance-interactions)**
Automated interaction detection using CANN, NID, and SHAP-based methods. Finds the interaction effects your GLM missed and quantifies their materiality.
`uv add insurance-interactions`
&rarr; [Finding the interactions your GLM missed](https://burning-cost.github.io/2026/02/27/finding-the-interactions-your-glm-missed/)

**[insurance-anam](https://github.com/burning-cost/insurance-anam)**
Actuarial neural additive model in PyTorch: interpretable deep learning for pricing with per-feature shape functions a pricing committee can inspect.
`uv add insurance-anam`
&rarr; [Your interpretable model isn't interpretable enough](https://burning-cost.github.io/2026/03/17/your-interpretable-model-isnt-interpretable-enough/)

**[insurance-ebm](https://github.com/burning-cost/insurance-ebm)**
Insurance workflow layer on top of interpretML's ExplainableBoostingMachine. RelativitiesTable, MonotonicityEditor, GLMComparison, and actuarial diagnostics (Gini, double-lift, calibration). EBMs are GAMs fitted by gradient boosting - the shape functions are the model, not a post-hoc summary.
`uv add insurance-ebm[interpret]`
&rarr; [EBMs for insurance pricing](https://burning-cost.github.io/2026/03/09/explainable-boosting-machines-for-insurance-pricing/)

**[bayesian-pricing](https://github.com/burning-cost/bayesian-pricing)**
Hierarchical Bayesian models for thin-data pricing segments using PyMC 5. When a segment has 200 policies and 3 claims, full posterior inference tells you more than a point estimate.
`uv add bayesian-pricing`
&rarr; [Bayesian hierarchical models for thin-data pricing](https://burning-cost.github.io/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/)

**[insurance-multilevel](https://github.com/burning-cost/insurance-multilevel)**
CatBoost combined with REML random effects for high-cardinality categorical groups such as brokers and schemes. Partial pooling without leaving the gradient boosting framework.
`uv add insurance-multilevel`
&rarr; [Your broker adjustments are guesswork](https://burning-cost.github.io/2026/03/15/your-broker-adjustments-are-guesswork/)

**[credibility](https://github.com/burning-cost/credibility)**
Buhlmann-Straub credibility in Python with mixed-model equivalence checks. Includes NCD factor stabilisation and blending a new model with incumbent rates in thin cells.
`uv add credibility`
&rarr; [Buhlmann-Straub credibility in Python](https://burning-cost.github.io/2026/02/19/buhlmann-straub-credibility-in-python/)

**[experience-rating](https://github.com/burning-cost/experience-rating)**
NCD and bonus-malus systems for UK motor insurance. NCD as a Markov chain, stationary distributions, and the non-obvious claiming threshold result: optimal thresholds peak at 20% NCD, not 65%.
`uv add experience-rating`
&rarr; [Experience rating: NCD and bonus-malus systems](https://burning-cost.github.io/2026/02/27/experience-rating-ncd-bonus-malus/)

---

## Interpretation & Risk

**[insurance-conformal](https://github.com/burning-cost/insurance-conformal)**
Distribution-free prediction intervals for insurance GBMs with finite-sample coverage guarantees. Variance-weighted non-conformity scores produce intervals roughly 30% narrower than the naive approach.
`uv add insurance-conformal`
&rarr; [Conformal prediction intervals for insurance pricing](https://burning-cost.github.io/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)

**[insurance-quantile](https://github.com/burning-cost/insurance-quantile)**
Quantile and expectile GBMs for tail risk, TVaR, and increased limit factors. When the mean is not the right risk measure.
`uv add insurance-quantile`
&rarr; [Quantile GBMs for insurance tail risk](https://burning-cost.github.io/2026/03/07/insurance-quantile/)

**[insurance-distributional](https://github.com/burning-cost/insurance-distributional)**
Distributional GBMs with Tweedie, Gamma, ZIP, and negative binomial objectives. Fit the full conditional distribution, not just the conditional mean.
`uv add insurance-distributional`
&rarr; [Distributional GBMs for insurance pricing](https://burning-cost.github.io/2026/03/05/insurance-distributional/)

**[insurance-ilf](https://github.com/burning-cost/insurance-ilf)**
Increased limits factor curves from severity distributions. Mixed Exponential, Lognormal-Pareto, and empirical methods with bootstrap confidence intervals. For pricing layers and excess-of-loss attachments.
`uv add insurance-ilf`

---

## Causal Inference

**[insurance-causal](https://github.com/burning-cost/insurance-causal)**
Causal inference via double machine learning for deconfounding rating factors. Separates the causal effect of a rating factor from the confounding driven by correlated features.
`uv add insurance-causal`
&rarr; [Causal inference for insurance pricing](https://burning-cost.github.io/2026/02/25/causal-inference-for-insurance-pricing/)

**[insurance-elasticity](https://github.com/burning-cost/insurance-elasticity)**
Causal price elasticity estimation via CausalForestDML and DR-Learner. Heterogeneous treatment effects on conversion and retention, corrected for selection bias.
`uv add insurance-elasticity`
&rarr; [Your rating factor might be confounded](https://burning-cost.github.io/2026/03/05/your-rating-factor-might-be-confounded/)

**[insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy)**
Synthetic difference-in-differences for causal rate change evaluation. Produces the counterfactual portfolio and FCA evidence pack you need to demonstrate a rate change worked.
`uv add insurance-causal-policy`
&rarr; [Your rate change didn't prove anything](https://burning-cost.github.io/2026/03/19/your-rate-change-didnt-prove-anything/)

**[insurance-demand](https://github.com/burning-cost/insurance-demand)**
Conversion, retention, and DML price elasticity modelling integrated with rate optimisation. Demand curves feed directly into constrained optimisation.
`uv add insurance-demand`
&rarr; [Demand modelling for insurance pricing](https://burning-cost.github.io/2026/02/25/demand-modelling-for-insurance-pricing/)

---

## Optimisation

**[rate-optimiser](https://github.com/burning-cost/rate-optimiser)**
Constrained rate change optimisation with an efficient frontier between loss ratio target and movement cap constraints. Linear programming formulation with shadow price reporting and FCA GIPP (PS21/5) constraints.
`uv add rate-optimiser`
&rarr; [Constrained rate optimisation and the efficient frontier](https://burning-cost.github.io/2026/02/21/constrained-rate-optimisation-efficient-frontier/)

**[insurance-optimise](https://github.com/burning-cost/insurance-optimise)**
SLSQP portfolio rate optimisation with analytical Jacobians for large factor spaces. When the number of factors makes linear programming too slow.
`uv add insurance-optimise`
&rarr; [Portfolio rate optimisation with insurance-optimise](https://burning-cost.github.io/2026/03/07/insurance-optimise/)

---

## Compliance & Governance

**[insurance-fairness](https://github.com/burning-cost/insurance-fairness)**
Proxy discrimination auditing and FCA Consumer Duty documentation support. Quantifies indirect discrimination risk through protected characteristic proxies in rating factors.
`uv add insurance-fairness`
&rarr; [Your pricing model might be discriminating](https://burning-cost.github.io/2026/03/03/your-pricing-model-might-be-discriminating/)

**[insurance-validation](https://github.com/burning-cost/insurance-validation)**
Structured PRA SS1/23 model validation reports covering nine required sections, output as HTML and JSON. Designed around the actual regulatory standard, not a generic model card.
`uv add insurance-validation`
&rarr; [Model validation reports for PRA SS1/23](https://burning-cost.github.io/2026/03/13/insurance-validation/)

**[insurance-calibration](https://github.com/burning-cost/insurance-calibration)**
Balance property testing, auto-calibration curves, Murphy score decomposition. Detect and correct systematic bias in predicted frequencies and severities before deployment.
`uv add insurance-calibration`

**[insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)**
Exposure-weighted PSI/CSI, actual-vs-expected ratios, and Gini drift z-tests for deployed models. Scheduled monitoring with alert thresholds, not one-off validation.
`uv add insurance-monitoring`
&rarr; [Your pricing model is drifting](https://burning-cost.github.io/2026/03/03/your-pricing-model-is-drifting/)

**[insurance-mrm](https://github.com/burning-cost/insurance-mrm)**
Model risk management: ModelCard, ModelInventory, and GovernanceReport generation. Replaces the spreadsheet model risk register with a structured, versioned artefact.
`uv add insurance-mrm`
&rarr; [Your model risk register is a spreadsheet](https://burning-cost.github.io/2026/03/19/your-model-risk-register-is-a-spreadsheet/)

**[insurance-deploy](https://github.com/burning-cost/insurance-deploy)**
Champion/challenger framework with shadow mode, rollback, and full audit trail. Structured deployment for pricing models that need a sign-off process.
`uv add insurance-deploy`
&rarr; [Your champion/challenger test has no audit trail](https://burning-cost.github.io/2026/03/15/your-champion-challenger-test-has-no-audit-trail/)

---

## Spatial

**[insurance-spatial](https://github.com/burning-cost/insurance-spatial)**
BYM2 spatial models for postcode-level territory ratemaking, borrowing strength from neighbouring areas. Handles the sparse-data problem in granular geographic segmentation.
`uv add insurance-spatial`
&rarr; [Spatial territory ratemaking with BYM2](https://burning-cost.github.io/2026/03/09/spatial-territory-ratemaking-bym2/)

---

## Retention

**[insurance-survival](https://github.com/burning-cost/insurance-survival)**
Cure models, customer lifetime value, lapse tables, and MLflow wrapper for retention modelling. Treats lapse as a competing risk rather than a binary outcome.
`uv add insurance-survival`
&rarr; [Survival models for insurance retention](https://burning-cost.github.io/2026/03/11/survival-models-for-insurance-retention/)

---

All libraries are MIT-licensed and available on [PyPI](https://pypi.org/search/?q=insurance-) and [GitHub](https://github.com/burning-cost). Each ships with a Databricks notebook demo and synthetic data. If something is missing from this list, [get in touch](mailto:pricing.frontier@gmail.com).

Looking for runnable notebooks? The [Databricks Notebook Archive](/notebooks/) collects 10 curated demos you can import directly.

