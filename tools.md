---
layout: page
title: "Open-Source Python Libraries for Insurance Pricing"
description: "75 production-ready Python libraries for UK personal lines pricing. From SHAP relativities and Bayesian credibility to causal inference and regulatory compliance."
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
      "description": "Distribution-free prediction intervals for insurance GBMs. v0.2 adds locally-weighted intervals (~24% narrower via CatBoost spread model), model-free conformal (Hong 2025), and SCRReport for Solvency II 99.5% upper bounds with coverage validation tables.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal-risk",
      "description": "Conformal Risk Control (Angelopoulos et al. ICLR 2024) for UK insurance pricing. Controls expected monetary loss directly. PremiumSufficiencyController bounds E[shortfall/premium] <= alpha. SelectiveRiskController (SCRC-I, arXiv:2512.12844) bounds expected loss on accepted underwriting book with DKW correction. 121 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-risk",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-multivariate-conformal",
      "description": "Joint multi-output conformal prediction for insurance models. Fan & Sesia (arXiv:2512.15383) coordinate-wise standardization gives P(frequency in interval AND severity in interval) >= 1-alpha. LWC method 20-35% more efficient than Bonferroni. SolvencyCapitalEstimator for Solvency II SCR at 99.5%. 198 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-multivariate-conformal",
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
      "name": "insurance-counterfactual-sets",
      "description": "Lei & Candès 2021 weighted conformal inference for individual ENBP counterfactual prediction sets. Finite-sample valid harm assessment at the policyholder level for ICOBS 6B compliance.",
      "codeRepository": "https://github.com/burning-cost/insurance-counterfactual-sets",
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
      "name": "insurance-dro",
      "description": "Distributionally robust rate optimisation using Wasserstein ambiguity sets. Produces a price-of-robustness curve for pricing committee papers.",
      "codeRepository": "https://github.com/burning-cost/insurance-dro",
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
      "name": "insurance-bunching",
      "description": "Bunching estimators for insurance threshold gaming detection. Adapted from Saez (2010) and Kleven (2016) public economics methodology to detect mileage declaration fraud, adverse deductible selection, and sum-insured rounding. BH FDR-corrected multi-threshold scanning, FCA Consumer Duty HTML reports.",
      "codeRepository": "https://github.com/burning-cost/insurance-bunching",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
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
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-cure",
      "description": "Mixture cure models for insurance non-claimer scoring. WeibullMCM, LogNormalMCM, CoxMCM with EM algorithm. Maller-Zhou sufficient follow-up test and CureScorecard decile validation.",
      "codeRepository": "https://github.com/burning-cost/insurance-cure",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-competing-risks",
      "description": "Fine-Gray subdistribution hazard regression for competing risks in insurance. AalenJohansenFitter, FineGrayFitter with IPCW, Gray's test, competing-risks Brier score and C-index. For retention lapse/MTC/NTU modelling and home insurance peril CIF estimation.",
      "codeRepository": "https://github.com/burning-cost/insurance-competing-risks",
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
      "name": "insurance-mediation",
      "description": "Causal mediation analysis for FCA proxy discrimination compliance. Decomposes postcode price differentials into legitimate (IMD-mediated) and residual (direct) effects via CDE/NDE/NIE estimators. Imai rho sensitivity analysis, E-values, FCA-ready HTML audit reports. Poisson, Gamma, Tweedie GLM outcome models.",
      "codeRepository": "https://github.com/burning-cost/insurance-mediation",
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
      "name": "insurance-gas",
      "description": "GAS (Generalised Autoregressive Score) models for dynamic insurance pricing. Poisson frequency, Gamma/LogNormal severity, Beta loss ratio, and NegBin variants. Exposure-weighted MLE, trend_index output, panel data support. First maintained Python GAS library.",
      "codeRepository": "https://github.com/burning-cost/insurance-gas",
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
      "name": "insurance-evt",
      "description": "Extreme Value Theory for catastrophic insurance claim severity. GPD/GEV fitting with censored MLE for open TPBI claims, profile likelihood CIs, return levels, ExcessGPD layer pricing, and Solvency II 1-in-200 reporting. First Python ExcessGPD implementation.",
      "codeRepository": "https://github.com/burning-cost/insurance-evt",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-eqrn",
      "description": "Extreme Quantile Regression Neural Networks for covariate-dependent GPD tail modelling. First Python implementation of Pasche & Engelke (2024, AoAS). PyTorch GPDNet estimates xi(x) and sigma(x) per risk profile for per-risk XL pricing, TPBI segment VaR, and Solvency II internal models. 119 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-eqrn",
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
      "name": "insurance-copula",
      "description": "Vine copulas for multi-peril home insurance pricing. PerilVine wraps pyvinecopulib with exposure-weighted fitting, BIC structure selection, conditional pricing via accept-reject Monte Carlo, and aggregate PML simulation. 174 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-copula",
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
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal-fraud",
      "description": "Conformal p-values for claims fraud detection with FDR-controlled SIU referral. Mondrian stratification by claim type, integrative conformal using SIU case files (Lemos et al. 2024), Fisher combination for IFB consortium detection. 133 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-fraud",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-evt",
      "description": "Extreme Value Theory for catastrophic claim severity. GPD/GEV MLE with profile likelihood CIs, censored MLE for large loss thresholds, ExcessGPD reinsurance layer pricing, Solvency II 1-in-200 VaR/ES. 144 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-evt",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-reconcile",
      "description": "Hierarchical forecast reconciliation for insurance pricing hierarchies. MinTrace with earned premium weighting, loss cost / loss ratio transform, frequency x severity log-space reconciliation, peril tree DSL, coherence diagnostics. 160 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-reconcile",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-tabpfn",
      "description": "TabPFN v2 and TabICLv2 foundation model wrapper for thin-data insurance pricing. In-context learning with no gradient descent, split conformal intervals, GLM benchmark, PDP relativities, committee report with five mandatory limitations (72 tests).",
      "codeRepository": "https://github.com/burning-cost/insurance-tabpfn",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-transfer",
      "description": "Transfer learning for thin-segment insurance pricing. Tian-Feng two-step GLM transfer (first Python implementation, adds Poisson/Gamma), CatBoost source-as-offset GBM transfer, and CANN pre-train/fine-tune. MMD covariate shift test and negative transfer diagnostic included — run both before and after fitting.",
      "codeRepository": "https://github.com/burning-cost/insurance-transfer",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-jlm",
      "description": "Joint longitudinal-survival models for insurance telematics repricing. Wulfsohn-Tsiatis SREM with EM-GHQ estimation, DynamicPredictor for mid-term risk updates as new telematics readings arrive, attenuation-corrected association parameter. 120 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-jlm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-nflow",
      "description": "Normalizing flows for insurance claim severity. Neural Spline Flow with Student-t CDF tail transform (Hickling-Prangle ICML 2025), conditional severity by rating factor, ILF curves, TVaR, reinsurance layer pricing. No parametric family assumption. 200 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-nflow",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-online",
      "description": "Multi-armed bandit algorithms for GIPP-compliant loading optimisation on direct insurance channels. Thompson Sampling, UCB1, and LinUCB with ENBP constraints and FCA PS21/5 documentation.",
      "codeRepository": "https://github.com/burning-cost/insurance-online",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal-ts",
      "description": "Conformal prediction for non-exchangeable claims time series. ACI, EnbPI, SPCI, Conformal PID, MSCP multi-step intervals, Poisson/NB nonconformity scores, exposure weighting.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-ts",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-recurrent",
      "description": "Shared frailty models for recurrent insurance claims. AndersenGillFrailty (gamma/lognormal), PWPModel, JointFrailtyModel for informative censoring, NelsonAalenFrailty. Frailty posterior is the Buehlmann-Straub credibility premium.",
      "codeRepository": "https://github.com/burning-cost/insurance-recurrent",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-rdd",
      "description": "Regression Discontinuity Design for insurance rating thresholds. Sharp/fuzzy RD via rdrobust, PoissonRD/GammaRD with bootstrap CIs, geographic territory boundary analysis, McCrary density test, FCA Consumer Duty threshold reports. 194 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-rdd",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-garch",
      "description": "GARCH volatility models for claims inflation. Exposure-weighted ClaimsInflationModel wrapping arch. GJR-GARCH for asymmetric shocks. Bootstrap CIs for trend projections. First insurance-native GARCH Python implementation. 92 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-garch",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-frequency-severity",
      "description": "Sarmanov copula joint frequency-severity modelling for UK personal lines. IFM estimation, analytical premium correction, dependence tests, JointModelReport. Challenges the independence assumption in the standard two-model GLM pricing framework. 80+ tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-frequency-severity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-tmle",
      "description": "TMLE (Targeted Maximum Likelihood Estimation) for insurance causal inference. First Python library with Poisson TMLE and exposure offsets. Doubly robust: valid if either outcome or propensity model is correctly specified. SuperLearner, CVTMLE, heterogeneous CATE. 6 modules.",
      "codeRepository": "https://github.com/burning-cost/insurance-tmle",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-bcf",
      "description": "Bayesian Causal Forests for heterogeneous rate change elasticity. stochtree BCFModel wrapper with RIC correction, propensity diagnostics, segment CATE with credible intervals, FCA EP25/2 BCFAuditReport. 149 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-bcf",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-recourse",
      "description": "Algorithmic recourse for FCA Consumer Duty in UK personal lines insurance. DiCE/alibi/FOCUS backends with insurance-native actionability constraints, causal propagation, and RecourseReport audit hashing. SHAP tells you why. Recourse tells you what to change. 177 tests.",
      "codeRepository": "https://github.com/burning-cost/insurance-recourse",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    }
  ]
}
</script>

Burning Cost is on the forefront of machine learning and data science research in UK personal lines insurance. These libraries are the practical output of that research — each one solving a specific problem in the pricing workflow, built to run on Databricks, tested against actuarial standards. All 74 libraries are on PyPI and MIT-licensed (PyPI pending for insurance-fairness-ot).

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

**[insurance-gas](https://github.com/burning-cost/insurance-gas)**
GAS (Generalised Autoregressive Score) models for dynamic insurance pricing — the first maintained Python implementation. Exposure-weighted Poisson frequency, Gamma/log-normal severity, Beta loss ratio, and negative binomial variants. GAS(1,1) MLE via L-BFGS-B, `trend_index` in the development factor format, panel data across rating cells, and formal diagnostics (PIT, Ljung-Box). Sits between changepoint detection and static trend projection: continuous adaptive estimation updated at each period.
`uv add insurance-gas`
&rarr; [Your Trend Estimate Has No Likelihood](https://burning-cost.github.io/2026/03/25/insurance-gas/)

**[insurance-garch](https://github.com/burning-cost/insurance-garch)**
GARCH volatility models for claims inflation time series. The 2021-2024 motor/home inflation episode produced time-varying variance that a static trend projection cannot represent. ClaimsInflationModel wraps the arch library with exposure weighting, GJR-GARCH for asymmetric volatility shocks, bootstrap confidence intervals for trend projections, and a VolatilityReport in the format pricing committees expect. First Python implementation with insurance-native defaults. 92 tests.
`pip install insurance-garch`
&rarr; [Your Trend Line Has No Error Bars](https://burning-cost.github.io/2026/03/12/insurance-garch/)

**[insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity)**
Sarmanov copula joint frequency-severity modelling. Every personal lines pricing model assumes frequency and severity are independent, conditional on rating factors. Insurance-frequency-severity tests that assumption and corrects for it. SarmanovCopula with Poisson/NB and Gamma/Lognormal margins, IFM estimation, analytical premium correction factor, Kendall tau and Spearman rho dependence tests, and a JointModelReport. 80+ tests.
`pip install insurance-frequency-severity`
&rarr; [Two GLMs, One Unjustified Assumption](https://burning-cost.github.io/2026/03/12/insurance-frequency-severity/)

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

**[insurance-recurrent](https://github.com/burning-cost/insurance-recurrent)**
Shared frailty models for recurrent insurance claims. The frailty posterior mean is the Bühlmann-Straub credibility premium, connecting counting process survival analysis to classical actuarial credibility theory. AndersenGillFrailty (gamma and lognormal, EM algorithm), PWPModel (Prentice-Williams-Peterson, stratified by event number), JointFrailtyModel for informative censoring when high-risk policyholders lapse, and NelsonAalenFrailty for non-parametric heterogeneity estimation. Use cases: fleet motor (no NCD), pet insurance (breed propensity), home (subsidence repeat claims).
`pip install insurance-recurrent`
&rarr; [Shared Frailty for Repeat Claimers: When the Poisson GLM Gets the Wrong Answer](https://burning-cost.github.io/2026/03/12/insurance-recurrent/)


**[insurance-credibility-transformer](https://github.com/burning-cost/insurance-credibility-transformer)**
Credibility Transformer — CLS token attention as Bühlmann-Straub credibility, ICL zero-shot pricing, base/deep/ICL variants (85 tests). The CLS token's self-attention weight IS the credibility weight — mathematically identical to Bühlmann-Straub, not an analogy. 1,746 parameters, beats CAFTT (27K parameters). ICL extension enables zero-shot pricing for unseen categorical levels (new vehicle models, new regions) without retraining.
`pip install insurance-credibility-transformer`
&rarr; [The Attention Head That Is Also a Credibility Weight](https://burning-cost.github.io/2026/03/11/credibility-transformer/)

**[insurance-tabpfn](https://github.com/burning-cost/insurance-tabpfn)**
TabPFN v2 and TabICLv2 foundation model wrappers for thin-data insurance pricing. When a segment has under 2,000 policies and GLM MLE is unstable, in-context learning from pretraining on millions of synthetic datasets outperforms tuned XGBoost by 3–8 Gini points. GLMBenchmark (Gini/deviance/RMSE/double-lift), split conformal prediction intervals, PDP-based RelativitiesExtractor, and CommitteeReport with five mandatory disclosure fields. Not a GLM replacement — a thin-segment specialist. 72 tests.
`uv add insurance-tabpfn`
&rarr; [Your Thin Segment Has 200 Policies. Your GLM Has No Idea What to Do With Them.](https://burning-cost.github.io/2026/03/25/insurance-tabpfn/)
**[insurance-transfer](https://github.com/burning-cost/insurance-transfer)**
Transfer learning for thin-segment insurance pricing. Tian-Feng two-step GLM transfer (first Python implementation with Poisson and Gamma families for frequency/severity), CatBoost source-as-offset GBM transfer, and CANN pre-train/fine-tune. `CovariateShiftTest` (MMD, mixed-kernel for continuous and categorical) and `NegativeTransferDiagnostic` (deviance comparison on held-out target data) are mandatory workflow steps, not optional extras.
`uv add insurance-transfer`
&rarr; [Borrowing Experience You Don't Have](https://burning-cost.github.io/2026/03/12/borrowing-experience-you-dont-have/)

---

## Interpretation & Risk

**[insurance-conformal](https://github.com/burning-cost/insurance-conformal)**
Distribution-free prediction intervals with finite-sample coverage guarantees. v0.2 adds locally-weighted intervals (~24% narrower, CatBoost spread model), model-free conformal (Hong 2025, no point predictor needed), SCRReport for Solvency II 99.5% upper bounds with multi-alpha coverage validation tables, and ERT diagnostics for conditional coverage by rating segment. 186 tests.
`uv add "insurance-conformal[catboost]"`
&rarr; [Distribution-free solvency capital from conformal prediction](https://burning-cost.github.io/2026/03/11/conformal-scr-solvency/) · [Why your prediction intervals are lying to you](https://burning-cost.github.io/2026/03/06/why-your-prediction-intervals-are-lying-to-you/)

**[insurance-conformal-risk](https://github.com/burning-cost/insurance-conformal-risk)**
Conformal Risk Control (Angelopoulos et al. ICLR 2024) for UK insurance pricing. Controls expected monetary loss directly — not coverage probability. PremiumSufficiencyController finds the smallest loading factor λ* guaranteeing E[max(claim - λ·premium, 0)/premium] ≤ α. IntervalWidthController minimises interval width within a risk budget. SelectiveRiskController (SCRC-I, arXiv:2512.12844) bounds expected loss on an accepted underwriting book with DKW correction on selection rate. 121 tests.
`pip install insurance-conformal-risk`
&rarr; [Coverage Is the Wrong Guarantee for Pricing Actuaries](https://burning-cost.github.io/2026/03/25/insurance-conformal-risk/)

**[insurance-multivariate-conformal](https://github.com/burning-cost/insurance-multivariate-conformal)**
Joint multi-output conformal prediction for insurance pricing models. Separate frequency and severity GLMs give marginal prediction intervals — no joint coverage guarantee. insurance-multivariate-conformal implements Fan & Sesia (arXiv:2512.15383) coordinate-wise standardization to produce a single prediction set where P(frequency &isin; interval AND severity &isin; interval) &ge; 1&minus;&alpha;. LWC method is 20&ndash;35% more efficient than Bonferroni. SolvencyCapitalEstimator constructs one-sided joint upper bounds for Solvency II SCR at 99.5% with finite-sample guarantee. 198 tests.
`pip install insurance-multivariate-conformal`
&rarr; [Frequency and Severity Are Two Outputs. You Have One Prediction Interval.](https://burning-cost.github.io/2026/03/26/insurance-multivariate-conformal/)

**[insurance-conformal-fraud](https://github.com/burning-cost/insurance-conformal-fraud)**
Conformal p-values for claims fraud detection with provable FDR-controlled SIU referral. Wraps any sklearn anomaly detector. Mondrian stratification by claim type (TPBI, AD, Theft), integrative conformal from SIU case files (Lemos et al. 2024, JRSS-B), Fisher combination for IFB consortium detection without sharing claim data. 133 tests.
`pip install insurance-conformal-fraud`
&rarr; [Your SIU Referral Threshold Is Arbitrary. Here Is How to Fix It.](https://burning-cost.github.io/2026/03/11/insurance-conformal-fraud/)

**[insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts)**
Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, Conformal PID, MSCP multi-step intervals, Poisson/NB nonconformity scores, exposure weighting. Standard split conformal assumes exchangeability; claims frequency data violates it. All five sequential methods adapt the coverage threshold online as distribution shift occurs. 150 tests.


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

**[insurance-nflow](https://github.com/burning-cost/insurance-nflow)**
Normalizing flows for insurance claim severity modelling. UK motor bodily injury severity is bimodal — soft-tissue claims cluster around GBP 3–5k, catastrophic injury follows a power law from GBP 100k upward. A lognormal cannot represent this shape. SeverityFlow wraps zuko's Neural Spline Flow with a Student-t CDF tail transform (Hickling & Prangle, ICML 2025) that learns heavy-tail weight from data rather than assuming it. ConditionalSeverityFlow conditions the full distribution on rating factors, capturing shape changes across segments that a mean-only GLM misses. ILF curves, TVaR, reinsurance layer pricing from flow samples. Benchmark comparisons against lognormal, gamma, and composite models. 200 tests.
`pip install insurance-nflow`
[![PyPI](https://img.shields.io/pypi/v/insurance-nflow)](https://pypi.org/project/insurance-nflow/)
&rarr; [Beyond Lognormal: Normalizing Flows for Insurance Severity Modelling](https://burning-cost.github.io/2026/03/12/insurance-nflow/)

**[insurance-evt](https://github.com/burning-cost/insurance-evt)**
Extreme Value Theory for catastrophic claim severity. GPD fitting with censored MLE for open TPBI claims (Poudyal & Brazauskas 2023), profile likelihood confidence intervals, return levels for Solvency II SCR, and the first Python implementation of ExcessGPD for XL layer pure premium pricing. GEV for block maxima (subsidence, portfolio catastrophe). Five UK peril presets: TPBI, flood, subsidence, large fire, storm.
`uv add insurance-evt`
&rarr; [Extreme Value Theory for UK Motor Large Loss Pricing](https://burning-cost.github.io/2026/03/25/insurance-evt/)

**[insurance-eqrn](https://github.com/burning-cost/insurance-eqrn)**
Extreme Quantile Regression Neural Networks for covariate-dependent GPD tail modelling. Standard EVT gives you one shape parameter for the whole portfolio. EQRN gives you xi(x) and sigma(x) per risk profile — the first Python implementation of Pasche & Engelke (2024, Annals of Applied Statistics). PyTorch GPDNet with orthogonal reparameterisation for stable training, K-fold out-of-fold intermediate quantile estimation via LightGBM, conditional VaR and TVaR at arbitrary levels, per-risk XL layer pricing. Applications: motor TPBI by driver profile, commercial property large loss by construction class, per-risk XL burn cost, Solvency II segment-level 99.5th percentile. 119 tests.
`pip install insurance-eqrn`
&rarr; [Your EVT Model Has One Tail. EQRN Has One Per Risk.](https://burning-cost.github.io/2026/03/12/insurance-eqrn/)

**[insurance-copula](https://github.com/burning-cost/insurance-copula)**
Vine copulas for multi-peril home insurance pricing. UK home insurers price perils additively — the independence assumption costs around 9% in mispriced revenue in clay-soil postcodes where flood and subsidence are correlated. PerilVine wraps pyvinecopulib with exposure-weighted fitting, BIC structure selection, conditional pricing via accept-reject Monte Carlo (E[Y|X>threshold]), and aggregate PML simulation. 174 tests, JSON serialisation for model governance.
`pip install insurance-copula[vine]`
[\![PyPI](https://img.shields.io/pypi/v/insurance-copula)](https://pypi.org/project/insurance-copula/)
&rarr; [Your Multi-Peril Home Pricing Is Wrong: The Flood-Subsidence Correlation You Are Ignoring](https://burning-cost.github.io/2026/03/12/insurance-copula/)

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

**[insurance-counterfactual-sets](https://github.com/burning-cost/insurance-counterfactual-sets)**
Lei & Candès 2021 weighted conformal inference for individual ENBP counterfactual prediction sets. Finite-sample valid harm assessment at the policyholder level: for each renewing customer, constructs a prediction set for their claims under new-business pricing. FCA EP25/2 attestation pack output. CQR variant for heteroskedastic claims. Jin, Ren & Candès 2023 (PNAS) sensitivity analysis with per-policyholder Gamma values.
`pip install insurance-counterfactual-sets`
&rarr; [The ENBP Counterfactual Problem Has a Statistical Answer](https://burning-cost.github.io/2026/03/11/conformal-counterfactual-sets/)

**[insurance-demand](https://github.com/burning-cost/insurance-demand)**
Conversion, retention, and DML price elasticity modelling integrated with rate optimisation. Demand curves feed directly into constrained optimisation.
`uv add insurance-demand`
&rarr; [Demand modelling for insurance pricing](https://burning-cost.github.io/2026/02/25/demand-modelling-for-insurance-pricing/)


**[insurance-uplift](https://github.com/burning-cost/insurance-uplift)**
Uplift modelling for UK personal lines retention targeting. Estimates per-customer treatment effects (CATE) on renewal probability via CausalForestDML, classifies the portfolio into Persuadables, Sure Things, Lost Causes, and Do Not Disturbs (Guelman et al.), evaluates with Qini curves and AUUC, builds optimal targeting rules via PolicyTree, clips recommendations to ENBP (ICOBS 6B.2), and audits CATE distribution across vulnerability proxies for Consumer Duty. 127 tests.
`uv add insurance-uplift`
[![PyPI](https://img.shields.io/pypi/v/insurance-uplift)](https://pypi.org/project/insurance-uplift/)
&rarr; [Your Retention Campaign Has No Targeting Rule](https://burning-cost.github.io/2026/03/11/insurance-uplift/)

**[insurance-rdd](https://github.com/burning-cost/insurance-rdd)**
Regression Discontinuity Design for insurance rating thresholds. UK motor insurers assume a claims discontinuity at age 25 — insurance-rdd tests whether the evidence supports the tariff. Sharp and fuzzy RD via rdrobust, PoissonRD and GammaRD with exposure weighting and bootstrap CIs, multicutoff analysis for NCD steps, McCrary density test for threshold manipulation, geographic territory boundary RDD. FCA Consumer Duty threshold reports with formal LATE estimates. 194 tests.
`pip install insurance-rdd`
&rarr; [Your Age-25 Premium Cliff Has No Causal Evidence](https://burning-cost.github.io/2026/03/11/insurance-rdd/)

**[insurance-tmle](https://github.com/burning-cost/insurance-tmle)**
TMLE (Targeted Maximum Likelihood Estimation) for insurance causal inference. Doubly robust: gives valid causal estimates if either your outcome model or your propensity model is correctly specified. For insurance pricing, where the treatment mechanism (competitor quotes, PCW ranking, risk appetite rules) is structurally unobservable, this is the practical advantage over DML. First Python implementation with Poisson TMLE, exposure offsets, and Poisson-specific efficient influence function. SuperLearner ensemble, CVTMLE, stratified heterogeneous CATE, DMLvsTMLE comparison. 6 modules.
`pip install insurance-tmle`
&rarr; [Your Propensity Model Is Wrong. TMLE Doesn't Mind.](https://burning-cost.github.io/2026/03/12/insurance-tmle/)

**[insurance-bcf](https://github.com/burning-cost/insurance-bcf)**
Bayesian Causal Forests for heterogeneous rate change elasticity. The GLM gives you an average lapse elasticity. BCF gives you per-policy posterior CATE with credible intervals. BayesianCausalForest wraps stochtree 0.4.0 BCFModel with RIC correction, positivity diagnostics, and probit link for binary outcomes. ElasticityEstimator aggregates segment CATE with credible intervals, partial dependence, and optimal rate adjustment. BCFAuditReport outputs FCA EP25/2 HTML with protected characteristic moderation check. 149 tests.
`pip install insurance-bcf`
&rarr; [Your Rate Change Analysis Is Averaging Over the Wrong Thing](https://burning-cost.github.io/2026/03/12/insurance-bcf/)

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

**[insurance-dro](https://github.com/burning-cost/insurance-dro)**
Distributionally robust rate optimisation using Wasserstein ambiguity sets. Instead of optimising against a point-estimate demand model, optimises against the worst-case distribution within a calibrated ball around your empirical samples. Produces a price-of-robustness curve for pricing committee papers. CVXPY + CLARABEL backend, FCA ENBP as hard constraint, 137 tests.
`pip install insurance-dro`
&rarr; [Your Rate Optimiser Has No Safety Margin](https://burning-cost.github.io/2026/03/25/insurance-dro/)

**[insurance-online](https://github.com/burning-cost/insurance-online)**
Multi-armed bandit algorithms for GIPP-compliant loading optimisation. Thompson Sampling, UCB1, and LinUCB apply to the discretionary loading band — the slice of pricing that every UK insurer operates but almost none has documented as an optimised decision. ENBP constraint prevents loadings that would breach FCA PS21/5 price walking limits. FCA Consumer Duty audit report included. Beats static A/B testing by routing quotes to superior arms while learning: after 500 quotes with arm B at 8% conversion and arm A at 12%, you are no longer allocating 50/50.
`pip install insurance-online`
&rarr; [Bandit Algorithms for GIPP-Compliant Price Experimentation](https://burning-cost.github.io/2026/03/12/insurance-online/)

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

**[insurance-mediation](https://github.com/burning-cost/insurance-mediation)**
Causal mediation analysis for FCA proxy discrimination compliance. Decomposes the total causal effect of postcode (or any protected characteristic proxy) into the component mediated by a legitimate risk factor (e.g., IMD deprivation score) and the residual direct effect. CDE, NDE, NIE estimators with Poisson/Gamma/Tweedie GLM outcome models. Imai et al. rho sensitivity analysis, E-value robustness, FCA-ready HTML audit reports with Section 19 proportionality statements.
`uv add insurance-mediation`
&rarr; [Your Postcode Premium Isn't Discriminatory. Probably. Can You Prove It?](https://burning-cost.github.io/2026/03/11/insurance-mediation/)

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

**[insurance-recourse](https://github.com/burning-cost/insurance-recourse)**
Algorithmic recourse for FCA Consumer Duty in UK personal lines insurance. SHAP attribution tells a policyholder which features drove their premium. That is not the same as telling them what they can change. RecourseGenerator wraps DiCE, alibi, and FOCUS backends with insurance-native ActionabilityGraph constraints (immutable features, causal propagation between correlated factors), InsuranceCostFunction for effort-weighted recommendations, and RecourseReport with FCA Consumer Duty JSON/HTML and SHA-256 audit hash. 177 tests.
`pip install insurance-recourse`
&rarr; [SHAP Tells You Why Your Premium Is High. It Doesn't Tell You What to Do About It.](https://burning-cost.github.io/2026/03/26/insurance-recourse/)

**[insurance-deploy](https://github.com/burning-cost/insurance-deploy)**
Champion/challenger framework with shadow mode, rollback, and full audit trail. Structured deployment for pricing models that need a sign-off process.
`uv add insurance-deploy`
&rarr; [Your champion/challenger test has no audit trail](https://burning-cost.github.io/2026/03/15/your-champion-challenger-test-has-no-audit-trail/)

**[insurance-bunching](https://github.com/burning-cost/insurance-bunching)**
Bunching estimators for threshold gaming detection. Adapted from public economics (Saez 2010; Kleven 2016) to detect mileage declaration fraud, adverse deductible selection, sum-insured rounding, and age misreporting. BH FDR-controlled multi-threshold scanning. Self-contained HTML reports for FCA Consumer Duty evidence. First Python bunching estimator in any field.
`uv add insurance-bunching`
&rarr; [Detecting Threshold Gaming in Insurance Portfolios](https://burning-cost.github.io/2026/03/11/insurance-bunching/)


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

**[insurance-jlm](https://github.com/burning-cost/insurance-jlm)**
Joint longitudinal-survival models for telematics mid-term repricing. The Wulfsohn-Tsiatis (1997) shared random effects model links individual telematics score trajectories to claim hazard. Two drivers at the same score but different trajectories — one improving, one declining — present different risks; the standard GLM cannot tell them apart. EM-GHQ estimation corrects Berkson attenuation bias from score measurement error. DynamicPredictor updates P(claim | history) at each monthly reading without refitting. 120 tests.
`pip install insurance-jlm`
&rarr; [The Telematics Score That Forgets Where It's Been](https://burning-cost.github.io/2026/03/12/insurance-jlm/)

**[insurance-poisson-mixture-nn](https://github.com/burning-cost/insurance-poisson-mixture-nn)**
Neural Poisson mixture model for structural vs stochastic zero decomposition in motor claim frequency. Two-component Poisson mixture with end-to-end gradient training — EM fails for neural networks with large parameter counts. Separate sub-networks for pi(x) (at-risk probability), lambda_0 (safe group rate), lambda_1 (risky group rate). Softplus reparameterisation enforces lambda_0 < lambda_1 without projection. Log-sum-exp stable NLL loss. predict_pi() per policy and classify_zero() structural/stochastic labelling. 14x pricing difference between identical-demographic telematics zero-claimers. Based on NAAJ 2025 PM-DNN paper (DOI: 10.1080/10920277.2025.2570289). 110 tests.
`uv add insurance-poisson-mixture-nn`
&rarr; [Not All Zero-Claimers Are Equal](https://burning-cost.github.io/2026/03/27/insurance-poisson-mixture-nn/)


---

## Retention

**[insurance-survival](https://github.com/burning-cost/insurance-survival)**
Cure models, customer lifetime value, lapse tables, and MLflow wrapper for retention modelling. Treats lapse as a competing risk rather than a binary outcome.
`uv add insurance-survival`
&rarr; [Survival models for insurance retention](https://burning-cost.github.io/2026/03/11/survival-models-for-insurance-retention/)

**[insurance-cure](https://github.com/burning-cost/insurance-cure)**
Mixture cure models for insurance non-claimer scoring. Separates structurally immune policyholders from susceptibles using a two-component mixture: logistic incidence sub-model (cure fraction) and survival latency (time-to-claim for susceptibles). WeibullMCM, LogNormalMCM, and CoxMCM for parametric and semiparametric latency. Maller-Zhou Qn test for sufficient follow-up before fitting. CureScorecard decile validation. First Python implementation with covariate-aware MCMs. 158 tests.
`uv add insurance-cure`
&rarr; [Separating Structural Non-Claimers from Risk: Mixture Cure Models for Insurance Pricing](https://burning-cost.github.io/2026/03/11/insurance-cure/)

**[insurance-competing-risks](https://github.com/burning-cost/insurance-competing-risks)**
Fine-Gray subdistribution hazard regression for competing risks in insurance. Treats lapse, MTC, and NTU as distinct competing events rather than censoring — gives calibrated CIF estimates directly usable in technical premium calculations. AalenJohansenFitter for non-parametric CIF, FineGrayFitter with IPCW for covariate regression, Gray's test for CIF equality, competing-risks Brier score and C-index. Covers retention (lapse/MTC/NTU) and home insurance peril-level first-claim modelling (fire/EoW/flood/subsidence). 150 tests.
`uv add insurance-competing-risks`
&rarr; [Treating Competing Risks as Censored Is Biasing Your Retention and Home Insurance Pricing](https://burning-cost.github.io/2026/03/12/insurance-competing-risks/)

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

**[insurance-reconcile](https://github.com/burning-cost/insurance-reconcile)**
Hierarchical forecast reconciliation for insurance pricing hierarchies. MinTrace (Wickramasuriya et al. 2019 JASA) with earned premium weighting — the economically correct weight, not structural count. Loss cost / loss ratio transform via claims amounts (so the EP-weighted aggregation constraint is satisfied). Frequency×severity log-space reconciliation. Peril tree and geographic hierarchy DSL. Coherence diagnostics pre-reconciliation, attribution report post-reconciliation. 160 tests, pure NumPy GLS.
`pip install 'insurance-reconcile[mintrace]'`
&rarr; [Your Peril Forecasts Don't Add Up](https://burning-cost.github.io/2026/03/11/your-peril-forecasts-dont-add-up/)

---


All libraries are MIT-licensed and available on [PyPI](https://pypi.org/search/?q=insurance-) and [GitHub](https://github.com/burning-cost). Each ships with a Databricks notebook demo and synthetic data. If something is missing from this list, [get in touch](mailto:pricing.frontier@gmail.com).

Looking for runnable notebooks? The [Databricks Notebook Archive](/notebooks/) collects 10 curated demos you can import directly.

