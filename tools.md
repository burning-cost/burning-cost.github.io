---
layout: page
title: "Open-Source Python Libraries for Insurance Pricing"
description: "47 production-ready Python libraries for UK personal lines pricing. From SHAP relativities and Bayesian credibility to causal inference and regulatory compliance."
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
      "name": "insurance-experience",
      "description": "Individual policy-level Bayesian posterior experience rating. Four model tiers: static Bühlmann-Straub, dynamic Poisson-gamma state-space (Ahn/Jeong/Lu/Wüthrich 2023), surrogate IS-based posteriors (Calcetero/Badescu/Lin 2024), and deep attention credibility (Wüthrich 2024). Multiplicative credibility factor slots into existing GLM rating engines.",
      "codeRepository": "https://github.com/burning-cost/insurance-experience",
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
      "name": "insurance-uplift",
      "description": "Heterogeneous treatment effects for UK personal lines retention targeting. RetentionPanel, CausalForestDML CATE estimation, Qini evaluation, four-customer taxonomy (Persuadable/Sure Thing/Lost Cause/Do Not Disturb), PolicyTree, ENBP constraint, Consumer Duty fairness audit.",
      "codeRepository": "https://github.com/burning-cost/insurance-uplift",
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
      "name": "insurance-calibration",
      "description": "Balance property testing, auto-calibration curves, and Murphy score decomposition for insurance frequency and severity models.",
      "codeRepository": "https://github.com/burning-cost/insurance-calibration",
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
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-nested-glm",
      "description": "Nested GLM with neural network entity embeddings and spatially constrained territory clustering for insurance ratemaking.",
      "codeRepository": "https://github.com/burning-cost/insurance-nested-glm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-distributional-glm",
      "description": "GAMLSS for Python: model ALL distribution parameters as functions of covariates. Seven families, RS algorithm with backtracking. Pure NumPy/SciPy.",
      "codeRepository": "https://github.com/burning-cost/insurance-distributional-glm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-dispersion",
      "description": "Double GLM (DGLM) for joint modelling of mean and dispersion in insurance pricing. Per-observation phi_i, alternating IRLS, REML correction. Six families. Pure NumPy/SciPy.",
      "codeRepository": "https://github.com/burning-cost/insurance-dispersion",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-drn",
      "description": "Distributional Refinement Networks for insurance pricing. Refines any baseline GLM or GBM distribution bin by bin, per covariate. Vectorised CDF/quantile/CRPS. PyTorch. JBCE loss.",
      "codeRepository": "https://github.com/burning-cost/insurance-drn",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-fairness-ot",
      "description": "Discrimination-free pricing via optimal transport. Lindholm marginalisation, causal path decomposition, Wasserstein barycenter for multi-attribute fairness, FCA EP25/2 compliance.",
      "codeRepository": "https://github.com/burning-cost/insurance-fairness-ot",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-fairness-diag",
      "description": "Proxy discrimination diagnostics. D_proxy scalar (LRTW 2026), Owen (2014) Shapley attribution of discrimination to individual rating factors, per-policyholder vulnerability scores, HTML/JSON audit reports for FCA EP25/2 compliance.",
      "codeRepository": "https://github.com/burning-cost/insurance-fairness-diag",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-trend",
      "description": "Loss cost trend analysis with structural break detection, ONS API integration, and bootstrap confidence intervals. Frequency, severity, and combined loss cost fitters.",
      "codeRepository": "https://github.com/burning-cost/insurance-trend",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-telematics",
      "description": "End-to-end telematics pricing pipeline: 1Hz GPS/accelerometer ingestion, continuous-time HMM driving state classification, Bühlmann-Straub credibility aggregation, and Poisson GLM integration. Includes TripSimulator for synthetic fleet generation.",
      "codeRepository": "https://github.com/burning-cost/insurance-telematics",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-glm-cluster",
      "description": "GLM factor level clustering via R2VF algorithm. Collapses high-cardinality categoricals (vehicle make, occupation) into pricing bands using ridge-regularised ranking and fused lasso fusion. BIC lambda selection, min-exposure enforcement, monotonicity constraints. Poisson, Gamma, Tweedie.",
      "codeRepository": "https://github.com/burning-cost/insurance-glm-cluster",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-nowcast",
      "description": "ML-EM nowcasting for claims reporting delays — covariate-conditioned completion factors and IBNR counts by risk segment.",
      "codeRepository": "https://github.com/burning-cost/insurance-nowcast",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-composite",
      "description": "Composite severity regression — spliced body/tail distributions with covariate-dependent thresholds, ILF estimation, and TVaR. 106 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-composite",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-sensitivity",
      "description": "Shapley effects for rating factor variance decomposition — Song (2016) random permutation estimator, Rabitti-Tzougas (2025) CLH subsampling, exposure weighting, correlated categoricals. Answers which rating factor drives the most premium variance.",
      "codeRepository": "https://github.com/burning-cost/insurance-sensitivity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-dispersion",
      "description": "Double GLM (DGLM) for joint modelling of mean and dispersion in non-life insurance pricing. Adds a separate regression for the dispersion parameter so volatility varies by risk segment, not just the mean.",
      "codeRepository": "https://github.com/burning-cost/insurance-dispersion",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-experience",
      "description": "Individual Bayesian a posteriori experience rating — four model tiers (static Bühlmann-Straub, dynamic Poisson-Gamma state-space, surrogate GBM with IS posteriors, deep attention), policy-level posterior credibility factors with balance property.",
      "codeRepository": "https://github.com/burning-cost/insurance-experience",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-credibility-transformer",
      "description": "Credibility Transformer — CLS token attention as Bühlmann-Straub credibility, ICL zero-shot pricing, base/deep/ICL variants (85 tests)",
      "codeRepository": "https://github.com/burning-cost/insurance-credibility-transformer",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-uplift",
      "description": "HTE uplift modelling for retention — CausalForestDML, Qini/AUUC, four customer taxonomy, PolicyTree, ENBP constraint, Consumer Duty fairness audit (127 tests).",
      "codeRepository": "https://github.com/burning-cost/insurance-uplift",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-bunching",
      "description": "Bunching estimators for insurance threshold gaming — exposure-weighted density discontinuity, kink/notch modes, FCA report (120 tests).",
      "codeRepository": "https://github.com/burning-cost/insurance-bunching",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-changepoint",
      "description": "Bayesian change-point detection for UK insurance pricing — exposure-weighted Poisson-Gamma BOCPD, PELT with bootstrap CIs, UK regulatory event priors, Consumer Duty evidence pack (157 tests).",
      "codeRepository": "https://github.com/burning-cost/insurance-changepoint",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    }
  ]
}
</script>

Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. These libraries are the practical output of that research — each one solving a specific problem in the pricing workflow, built to run on Databricks, tested against actuarial standards. All 49 libraries are on PyPI and MIT-licensed.

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
&rarr; [Whittaker-Henderson smoothing for insurance pricing](https://burning-cost.github.io/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/)

---

## Model Building

**[insurance-trend](https://github.com/burning-cost/insurance-trend)**
Loss cost trend analysis with structural break detection. Frequency/severity/loss cost fitters, ONS API integration for index deflation, superimposed inflation decomposition, and 1,000-replicate bootstrap confidence intervals. Regime-aware trend selection with ruptures.
`uv add insurance-trend`
&rarr; [Trend selection is not actuarial judgment](https://burning-cost.github.io/2026/03/13/insurance-trend/)

**[shap-relativities](https://github.com/burning-cost/shap-relativities)**
Multiplicative rating factor tables from CatBoost models via SHAP, in the same format as exp(beta) from a GLM. Exports to Excel and Radar-compatible CSV.
`uv add shap-relativities`
&rarr; [Extracting rating relativities from GBMs with SHAP](https://burning-cost.github.io/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)

**[insurance-interactions](https://github.com/burning-cost/insurance-interactions)**
Automated interaction detection using CANN, NID, and SHAP-based methods. Finds the interaction effects your GLM missed and quantifies their materiality.
`uv add insurance-interactions`
&rarr; [Finding the interactions your GLM missed](https://burning-cost.github.io/2026/02/27/finding-the-interactions-your-glm-missed/)

**[insurance-glm-cluster](https://github.com/burning-cost/insurance-glm-cluster)**
GLM factor level clustering via R2VF (Ben Dror, arXiv:2503.01521). Collapses high-cardinality categoricals — 500 vehicle makes to 20 pricing bands, 350 occupation codes to 30 groups — using ridge-regularised ranking followed by fused lasso fusion. BIC lambda selection, min-exposure enforcement, monotonicity constraints. Poisson, Gamma, Tweedie with log(exposure) offset.
`uv add insurance-glm-cluster`
[![PyPI](https://img.shields.io/pypi/v/insurance-glm-cluster)](https://pypi.org/project/insurance-glm-cluster/)
&rarr; [500 Vehicle Makes, One Afternoon, Zero Reproducibility](https://burning-cost.github.io/2026/03/10/insurance-glm-cluster/)

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

**[insurance-experience](https://github.com/burning-cost/insurance-experience)**
Individual policy-level Bayesian posterior experience rating. Four model tiers: static Bühlmann-Straub, dynamic Poisson-gamma state-space with seniority weighting (Ahn/Jeong/Lu/Wüthrich 2023), IS-based surrogate Bayesian posteriors for non-conjugate models (Calcetero/Badescu/Lin 2024), and deep attention credibility (Wüthrich 2024). All four produce a multiplicative credibility factor; the balance property holds at portfolio level.
`uv add insurance-experience`
[![PyPI](https://img.shields.io/pypi/v/insurance-experience)](https://pypi.org/project/insurance-experience/)
&rarr; [Individual Experience Rating Beyond NCD: From Bühlmann-Straub to Neural Credibility](https://burning-cost.github.io/2026/03/11/insurance-experience/)

**[insurance-credibility-transformer](https://github.com/burning-cost/insurance-credibility-transformer)**
Credibility Transformer — CLS token attention as Bühlmann-Straub credibility, ICL zero-shot pricing, base/deep/ICL variants (85 tests). The CLS token's self-attention weight IS the credibility weight — mathematically identical to Bühlmann-Straub, not an analogy. 1,746 parameters, beats CAFTT (27K parameters). ICL extension enables zero-shot pricing for unseen categorical levels (new vehicle models, new regions) without retraining.
`pip install insurance-credibility-transformer`
&rarr; [The Attention Head That Is Also a Credibility Weight](https://burning-cost.github.io/2026/03/11/credibility-transformer/)

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

**[insurance-drn](https://github.com/burning-cost/insurance-drn)**
Distributional Refinement Networks for insurance pricing. Wraps any GLM or GBM baseline with a neural network that refines the predictive distribution bin by bin, per covariate. Vectorised CDF, quantile, CRPS, and Expected Shortfall. JBCE loss. PyTorch. Preserves GLM calibration exactly when the network has nothing useful to add.
`uv add insurance-drn`
&rarr; [GLMs Predict Means. DRN Predicts Everything Else.](https://burning-cost.github.io/2026/03/10/distributional-refinement-network-insurance/)

**[insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm)**
GAMLSS (Generalised Additive Models for Location, Scale and Shape) for Python. Model ALL distribution parameters as functions of covariates — mean, dispersion, shape, zero-inflation probability. Seven families: Gamma, LogNormal, InverseGaussian, Tweedie, Poisson, NegativeBinomial, ZIP. RS algorithm with backtracking. Pure NumPy/SciPy.
`uv add insurance-distributional-glm`
&rarr; [GAMLSS in Python, finally](https://burning-cost.github.io/2026/03/10/insurance-distributional-glm/)

**[insurance-dispersion](https://github.com/burning-cost/insurance-dispersion)**
Double GLM (DGLM) for joint modelling of mean AND dispersion in insurance pricing. Every observation gets its own phi_i — enabling risk-adequate margin loading, Solvency II variance at policy level, and Tweedie segments with independent frequency-severity movement. Six families, REML correction, overdispersion LRT, factor tables. Pure NumPy/SciPy.
`uv add insurance-dispersion`
&rarr; [Double GLM for Insurance: Every Risk Gets Its Own Dispersion](https://burning-cost.github.io/2026/03/11/insurance-dispersion/)

**[insurance-ilf](https://github.com/burning-cost/insurance-ilf)**
Increased limits factor curves from severity distributions. Mixed Exponential, Lognormal-Pareto, and empirical methods with bootstrap confidence intervals. For pricing layers and excess-of-loss attachments.
`uv add insurance-ilf`

**[insurance-sensitivity](https://github.com/burning-cost/insurance-sensitivity)**
Shapley effects for rating factor variance decomposition. Which rating factor drives the most premium variance in your book? Song et al. (2016) random permutation estimator with Rabitti-Tzougas (2025) CLH subsampling for large datasets, exposure-weighted variance throughout, and a fitted-model interface that accepts any GLM or GBM. 112 tests.
`uv add insurance-sensitivity`
&rarr; [Shapley Effects for Rating Factor Variance Decomposition](https://burning-cost.github.io/2026/03/22/insurance-sensitivity/)

**[insurance-composite](https://github.com/burning-cost/insurance-composite)**
Composite severity regression — spliced body/tail distributions with covariate-dependent thresholds, ILF estimation, and TVaR. The body and tail of a severity distribution obey different physics; this fits them separately while letting the threshold vary by covariate. 106 tests.
`uv add insurance-composite`
[![PyPI](https://img.shields.io/pypi/v/insurance-composite)](https://pypi.org/project/insurance-composite/)
&rarr; [Composite Severity Regression: Fitting Body and Tail Separately](https://burning-cost.github.io/2026/03/23/insurance-composite/)

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


**[insurance-uplift](https://github.com/burning-cost/insurance-uplift)**
Uplift modelling for UK personal lines retention targeting. Estimates per-customer treatment effects (CATE) on renewal probability via CausalForestDML, classifies the portfolio into Persuadables, Sure Things, Lost Causes, and Do Not Disturbs (Guelman et al.), evaluates with Qini curves and AUUC, builds optimal targeting rules via PolicyTree, clips recommendations to ENBP (ICOBS 6B.2), and audits CATE distribution across vulnerability proxies for Consumer Duty. 127 tests.
`uv add insurance-uplift`
[![PyPI](https://img.shields.io/pypi/v/insurance-uplift)](https://pypi.org/project/insurance-uplift/)
&rarr; [Your Retention Campaign Has No Targeting Rule](https://burning-cost.github.io/2026/03/11/insurance-uplift/)
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

**[insurance-fairness-ot](https://github.com/burning-cost/insurance-fairness-ot)**
Optimal transport discrimination-free pricing. Computes corrected premiums via Lindholm (2022) marginalisation, Côté-Genest-Abdallah (2025) causal path decomposition, and Wasserstein barycenter for the multi-attribute case. Built for FCA EP25/2 compliance and the Equality Act Section 19 proportionate justification test. PyPI pending.
`uv add insurance-fairness-ot`
&rarr; [Discrimination-Free Pricing with Optimal Transport](https://burning-cost.github.io/2026/03/10/insurance-fairness-ot/)

**[insurance-fairness-diag](https://github.com/burning-cost/insurance-fairness-diag)**
Proxy discrimination diagnostics. Quantifies how much proxy discrimination a fitted model contains (D_proxy scalar, LRTW 2026) and which rating factors are responsible (Owen 2014 Shapley effects). Per-policyholder vulnerability scores (Côté-Charpentier 2025). HTML/JSON audit reports for FCA EP25/2 and Consumer Duty evidence. The diagnostic layer before you decide whether to correct.
`uv add insurance-fairness-diag`
&rarr; [Your Pricing Model Is Discriminating. Here's Which Factor Is Doing It.](https://burning-cost.github.io/2026/03/10/insurance-fairness-diag/)

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

**[insurance-nested-glm](https://github.com/burning-cost/insurance-nested-glm)**
Nested GLM with neural network entity embeddings for high-cardinality categoricals (vehicle make/model, postcode sector) and spatially constrained territory clustering via SKATER. The final model is a standard interpretable GLM with multiplicative relativities. Based on Wang, Shi, Cao (NAAJ 2025).
`uv add insurance-nested-glm`

---

## Telematics

**[insurance-telematics](https://github.com/burning-cost/insurance-telematics)**
End-to-end pipeline from raw 1Hz GPS/accelerometer data to GLM-compatible risk scores. TripSimulator generates synthetic fleets (Ornstein-Uhlenbeck speed processes, Dirichlet regime mixtures) so teams without raw data can build and test immediately. DrivingStateHMM and ContinuousTimeHMM classify driving into cautious/normal/aggressive states. Bühlmann-Straub credibility aggregation to driver level. Poisson GLM pipeline.
`uv add insurance-telematics`
&rarr; [HMM-Based Telematics Risk Scoring for Insurance Pricing](https://burning-cost.github.io/2026/03/10/insurance-telematics/)

---

## Retention

**[insurance-survival](https://github.com/burning-cost/insurance-survival)**
Cure models, customer lifetime value, lapse tables, and MLflow wrapper for retention modelling. Treats lapse as a competing risk rather than a binary outcome.
`uv add insurance-survival`
&rarr; [Survival models for insurance retention](https://burning-cost.github.io/2026/03/11/survival-models-for-insurance-retention/)

**[insurance-uplift](https://github.com/burning-cost/insurance-uplift)**
HTE uplift modelling for retention. CausalForestDML heterogeneous treatment effects identify which customers are genuinely persuadable — positive responders — rather than treating the whole book as equally retention-sensitive. Qini coefficient and AUUC for model evaluation, four-quadrant customer taxonomy (persuadables, sure things, lost causes, sleeping dogs), PolicyTree for interpretable uplift rules, expected net benefit per policyholder (ENBP) profit constraint, and a Consumer Duty fairness audit checking that retention effort does not disadvantage vulnerable customers. 127 tests.
`uv add insurance-uplift`
&rarr; [Uplift Modelling for Insurance Retention: Finding Who to Save](https://burning-cost.github.io/2026/03/11/insurance-uplift/)

---

## Pricing / Reserving Bridge

**[insurance-nowcast](https://github.com/burning-cost/insurance-nowcast)**
ML-EM nowcasting for claims reporting delays. The last 6–24 months of your accident year data is partially developed. Standard practice applies aggregate development factors from the reserving triangle — factors that do not condition on your actual risk mix. insurance-nowcast implements the Wilsens/Antonio/Claeskens ML-EM algorithm to produce completion factors by risk segment, and nowcasted IBNR counts that a frequency GLM can treat as fully developed data.
`uv add insurance-nowcast`
&rarr; [Your Most Recent Experience Data Is Wrong, and Aggregate Factors Won't Fix It](https://burning-cost.github.io/2026/03/23/insurance-nowcast/)

**[insurance-bunching](https://github.com/burning-cost/insurance-bunching)**
Bunching estimators for detecting threshold gaming in insurance portfolios. Adapted from public economics (Saez 2010, Kleven 2016) — the first Python implementation for insurance. Detects excess mass at NCD thresholds, deductible boundaries, and sum-insured round numbers. Iterative polynomial counterfactual density, WLS exposure weighting, bootstrap standard errors, kink/notch modes, Benjamini-Hochberg FDR correction for multi-threshold scanning, and Jinja2 HTML/JSON reports.
`uv add insurance-bunching`
[![PyPI](https://img.shields.io/pypi/v/insurance-bunching)](https://pypi.org/project/insurance-bunching/)
&rarr; [Detecting Threshold Gaming in Insurance Portfolios](https://burning-cost.github.io/2026/03/25/insurance-bunching/)

**[insurance-changepoint](https://github.com/burning-cost/insurance-changepoint)**
Bayesian change-point detection for UK insurance pricing time series. Online BOCPD (Adams & MacKay 2007) with exposure-weighted Poisson-Gamma conjugate for claim frequency — no existing Python package handles insurance exposure in BOCPD. PELT with bootstrap CIs for retrospective analysis. Ten UK regulatory event priors (Ogden, COVID, Whiplash Reform, GIPP, storms) encoded as hazard function multipliers. Consumer Duty (PRIN 2A.9) evidence pack generation for FCA compliance.
`uv add insurance-changepoint`
[![PyPI](https://img.shields.io/pypi/v/insurance-changepoint)](https://pypi.org/project/insurance-changepoint/)
&rarr; [When Did Your Loss Ratio Actually Change?](https://burning-cost.github.io/2026/03/25/insurance-changepoint/)

---


All libraries are MIT-licensed and available on [PyPI](https://pypi.org/search/?q=insurance-) and [GitHub](https://github.com/burning-cost). Each ships with a Databricks notebook demo and synthetic data. If something is missing from this list, [get in touch](mailto:pricing.frontier@gmail.com).

Looking for runnable notebooks? The [Databricks Notebook Archive](/notebooks/) collects 10 curated demos you can import directly.

