---
layout: tools
title: "Open-Source Python Libraries for Insurance Pricing"
description: "Open-source Python libraries for UK non-life pricing — from SHAP relativities and Bayesian credibility to causal inference, conformal prediction, and FCA Consumer Duty compliance."
permalink: /tools/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "SoftwareSourceCode",
      "name": "shap-relativities",
      "description": "SHAP-based rating relativities from GBM models — extract GLM-style multiplicative factors from CatBoost.",
      "codeRepository": "https://github.com/burning-cost/shap-relativities",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-glm-tools",
      "description": "GLM tooling for insurance pricing — nested GLM embeddings, R2VF factor level clustering, territory banding, SKATER.",
      "codeRepository": "https://github.com/burning-cost/insurance-glm-tools",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-gam",
      "description": "Interpretable GAM models for insurance pricing — EBM tariffs, Actuarial NAM, Pairwise Interaction Networks, exact Shapley values.",
      "codeRepository": "https://github.com/burning-cost/insurance-gam",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-interactions",
      "description": "Automated GLM interaction detection using CANN, NID scoring, and SHAP interaction values.",
      "codeRepository": "https://github.com/burning-cost/insurance-interactions",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-distributional",
      "description": "Distributional GBMs with Tweedie, Gamma, ZIP, and negative binomial objectives — per-risk volatility scoring.",
      "codeRepository": "https://github.com/burning-cost/insurance-distributional",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-distributional-glm",
      "description": "GAMLSS for Python — model all distribution parameters (mean, dispersion, shape) as functions of covariates. Seven families.",
      "codeRepository": "https://github.com/burning-cost/insurance-distributional-glm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-dispersion",
      "description": "Double GLM for joint mean-dispersion modelling — alternating IRLS, REML, Gamma/Tweedie/Poisson families.",
      "codeRepository": "https://github.com/burning-cost/insurance-dispersion",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-frequency-severity",
      "description": "Sarmanov copula joint frequency-severity — analytical premium correction, IFM estimation, Garrido conditional.",
      "codeRepository": "https://github.com/burning-cost/insurance-frequency-severity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-scmoe",
      "description": "Spatially Clustered Mixture of Experts for joint frequency-severity insurance pricing (NAAJ 2025).",
      "codeRepository": "https://github.com/burning-cost/insurance-scmoe",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-poisson-mixture-nn",
      "description": "Neural Poisson mixture for structural zero-claimers — PM-DNN, reparameterised constraint, pi(x) at-risk score.",
      "codeRepository": "https://github.com/burning-cost/insurance-poisson-mixture-nn",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-zit-dglm",
      "description": "Zero-Inflated Tweedie Double GLM — three-head CatBoost EM for mean, dispersion, and zero-inflation jointly.",
      "codeRepository": "https://github.com/burning-cost/insurance-zit-dglm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-severity",
      "description": "Severity modelling toolkit for UK non-life insurance.",
      "codeRepository": "https://github.com/burning-cost/insurance-severity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-evt",
      "description": "Extreme Value Theory for catastrophic claim severity — GPD/GEV, censored MLE, ExcessGPD reinsurance pricing, Solvency II 1-in-200.",
      "codeRepository": "https://github.com/burning-cost/insurance-evt",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-ilf",
      "description": "MBBEFD exposure curves, Swiss Re families, ILF tables, and per-risk XL pricing.",
      "codeRepository": "https://github.com/burning-cost/insurance-ilf",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-nflow",
      "description": "Normalising flows for severity modelling — NSF with Student-t tail transform, TVaR, ILF, conditional rating factor flow.",
      "codeRepository": "https://github.com/burning-cost/insurance-nflow",
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
      "name": "insurance-copula",
      "description": "Copula models for insurance pricing — D-vine temporal dependence, two-part occurrence/severity.",
      "codeRepository": "https://github.com/burning-cost/insurance-copula",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "bayesian-pricing",
      "description": "Hierarchical Bayesian models for insurance pricing — PyMC 5, thin-data segments, credibility factors.",
      "codeRepository": "https://github.com/burning-cost/bayesian-pricing",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-credibility",
      "description": "Credibility models for UK non-life pricing: Bühlmann-Straub and Bayesian experience rating.",
      "codeRepository": "https://github.com/burning-cost/insurance-credibility",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-credibility-transformer",
      "description": "Credibility Transformer — CLS token attention as Bühlmann-Straub credibility, ICL zero-shot pricing.",
      "codeRepository": "https://github.com/burning-cost/insurance-credibility-transformer",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-multilevel",
      "description": "Two-stage CatBoost + REML random effects for high-cardinality group factors — Bühlmann-Straub credibility, ICC diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-multilevel",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-experience",
      "description": "Individual policy Bayesian posterior experience rating — static Bühlmann-Straub, dynamic state-space, surrogate IS posteriors, and deep attention credibility.",
      "codeRepository": "https://github.com/burning-cost/insurance-experience",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-thin-data",
      "description": "Pricing techniques for low-volume segments where standard GLM fitting is unreliable.",
      "codeRepository": "https://github.com/burning-cost/insurance-thin-data",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-whittaker",
      "description": "Whittaker-Henderson 1D/2D smoothing with REML lambda selection, Bayesian CIs, and Poisson PIRLS.",
      "codeRepository": "https://github.com/burning-cost/insurance-whittaker",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-spatial",
      "description": "BYM2 spatial territory ratemaking for insurance pricing — PyMC 5 ICAR, adjacency matrices, Moran's I diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-spatial",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-causal",
      "description": "Causal inference for insurance pricing via double machine learning — CatBoost nuisance models, confounding bias reports.",
      "codeRepository": "https://github.com/burning-cost/insurance-causal",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-causal-policy",
      "description": "Synthetic difference-in-differences for causal rate change evaluation — event study, HonestDiD sensitivity, FCA evidence pack.",
      "codeRepository": "https://github.com/burning-cost/insurance-causal-policy",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-bcf",
      "description": "Bayesian Causal Forests — stochtree BCF wrapper, segment CATE estimation, ElasticityEstimator, FCA EP25/2 audit report.",
      "codeRepository": "https://github.com/burning-cost/insurance-bcf",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-tmle",
      "description": "Targeted Maximum Likelihood Estimation — doubly robust causal inference, Poisson TMLE with exposure offset, SuperLearner, CV-TMLE.",
      "codeRepository": "https://github.com/burning-cost/insurance-tmle",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-rdd",
      "description": "Regression discontinuity for rating thresholds — sharp/fuzzy RD, Poisson/Gamma outcomes, geographic RDD, McCrary test, FCA reports.",
      "codeRepository": "https://github.com/burning-cost/insurance-rdd",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-bunching",
      "description": "Bunching estimators for rating threshold gaming — mileage declaration fraud, adverse deductible selection, BH FDR-corrected scanning.",
      "codeRepository": "https://github.com/burning-cost/insurance-bunching",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-counterfactual-sets",
      "description": "Weighted conformal ITE for individual counterfactuals — Lei & Candès 2021, sensitivity analysis, FCA ENBP harm reports.",
      "codeRepository": "https://github.com/burning-cost/insurance-counterfactual-sets",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-covariate-shift",
      "description": "Density ratio correction for book shifts — CatBoost/RuLSIF/KLIEP, LR-QR conformal, FCA SUP 15.3 diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-covariate-shift",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-uplift",
      "description": "Heterogeneous treatment effects for retention targeting — CausalForestDML CATE, Qini/AUUC, PolicyTree, ENBP constraint, Consumer Duty fairness audit.",
      "codeRepository": "https://github.com/burning-cost/insurance-uplift",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-fairness",
      "description": "Proxy discrimination auditing for insurance pricing — FCA EP25/2, Consumer Duty, bias metrics.",
      "codeRepository": "https://github.com/burning-cost/insurance-fairness",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-mediation",
      "description": "Causal mediation for FCA proxy discrimination — CDE/NDE/NIE decomposition, Poisson/Gamma/Tweedie GLMs, Imai sensitivity, FCA HTML report.",
      "codeRepository": "https://github.com/burning-cost/insurance-mediation",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-recourse",
      "description": "Algorithmic recourse for FCA Consumer Duty in UK personal lines insurance pricing.",
      "codeRepository": "https://github.com/burning-cost/insurance-recourse",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal",
      "description": "Distribution-free prediction intervals for insurance GBM and GLM models — locally-weighted Pearson residuals, Solvency II SCR bounds.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal-fraud",
      "description": "Conformal anomaly detection for claims fraud — BH FDR control, integrative conformal p-values, IFB Fisher combination, Mondrian stratification.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-fraud",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-conformal-ts",
      "description": "Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, MSCP, Poisson/NB scores.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-ts",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-sensitivity",
      "description": "Global sensitivity analysis — Song 2016 Shapley effects, Rabitti-Tzougas 2025 CLH subsampling, exposure-weighted variance decomposition.",
      "codeRepository": "https://github.com/burning-cost/insurance-sensitivity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-optimise",
      "description": "Constrained portfolio rate optimisation — SLSQP with analytical Jacobians, FCA ENBP constraints, efficient frontier, JSON audit trail.",
      "codeRepository": "https://github.com/burning-cost/insurance-optimise",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-dro",
      "description": "Distributionally robust rate optimisation — Wasserstein/KL/chi2 ambiguity sets, CVXPY+CLARABEL, price-of-robustness frontier.",
      "codeRepository": "https://github.com/burning-cost/insurance-dro",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-online",
      "description": "Bandit algorithms for GIPP-compliant price experimentation — UCB1, Thompson Sampling, LinUCB, ENBP constraints, FCA audit trail.",
      "codeRepository": "https://github.com/burning-cost/insurance-online",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-cv",
      "description": "Temporal walk-forward cross-validation for insurance pricing — respects policy time structure, IBNR buffers, sklearn-compatible scorers.",
      "codeRepository": "https://github.com/burning-cost/insurance-cv",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-deploy",
      "description": "Champion/challenger pricing framework — shadow mode, SHA-256 routing, SQLite quote log, bootstrap LR test, ENBP audit.",
      "codeRepository": "https://github.com/burning-cost/insurance-deploy",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-monitoring",
      "description": "Model drift detection for deployed pricing — exposure-weighted PSI/CSI, actual-vs-expected ratios, Gini drift z-test.",
      "codeRepository": "https://github.com/burning-cost/insurance-monitoring",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-governance",
      "description": "Model governance for insurance pricing — PRA SS1/23 validation reports, model risk management, risk tier scoring.",
      "codeRepository": "https://github.com/burning-cost/insurance-governance",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-reconcile",
      "description": "Hierarchical forecast reconciliation — PremiumWeightedMinTrace, LossRatioReconciler, FreqSevReconciler, InsuranceHierarchy DSL.",
      "codeRepository": "https://github.com/burning-cost/insurance-reconcile",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-trend",
      "description": "Loss cost trend analysis — frequency/severity decomposition, ONS index integration, structural break detection.",
      "codeRepository": "https://github.com/burning-cost/insurance-trend",
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
      "name": "insurance-garch",
      "description": "GARCH volatility for claims inflation — GJR-GARCH, BoE fan charts, Christoffersen backtest.",
      "codeRepository": "https://github.com/burning-cost/insurance-garch",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-dynamics",
      "description": "Dynamic insurance pricing — GAS score-driven filters, Bayesian changepoint detection (BOCPD/PELT).",
      "codeRepository": "https://github.com/burning-cost/insurance-dynamics",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-survival",
      "description": "Survival models for insurance — cure models, customer lifetime value, lapse tables, MLflow wrapper.",
      "codeRepository": "https://github.com/burning-cost/insurance-survival",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-jlm",
      "description": "Joint longitudinal-survival models — Wulfsohn-Tsiatis SREM, EM+GHQ, DynamicPredictor for telematics mid-term repricing.",
      "codeRepository": "https://github.com/burning-cost/insurance-jlm",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-telematics",
      "description": "HMM-based driving state classification and GLM-compatible risk scoring from raw telematics trip data.",
      "codeRepository": "https://github.com/burning-cost/insurance-telematics",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-lda-risk",
      "description": "LDA probabilistic risk profiling — latent risk archetypes, portfolio mix drift, book-transfer segmentation.",
      "codeRepository": "https://github.com/burning-cost/insurance-lda-risk",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-synthetic",
      "description": "Vine copula synthetic portfolio generation — exposure-aware, preserves multivariate dependence, fidelity reporting.",
      "codeRepository": "https://github.com/burning-cost/insurance-synthetic",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    }
  ]
}
</script>

<div class="tools-intro">
  <p>30+ open-source Python libraries covering the full pricing stack — from GLM tooling and severity modelling through to causal inference, conformal prediction, and FCA regulatory compliance.</p>
  <p>All libraries are MIT-licensed, installable via pip, and built for Python 3.10+. The GitHub organisation is <a href="https://github.com/burning-cost">burning-cost</a>.</p>
</div>

---

## Essential Libraries

If you're new here, these ten are the best place to start. They solve problems most UK pricing teams hit first, have clear documentation, and work without needing to understand the full stack first.

<div class="essential-grid" id="essential-grid">

<div class="essential-card" data-name="shap-relativities" data-desc="Extract multiplicative rating factor tables from CatBoost models using SHAP values">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></div>
  <div class="essential-card-desc">Extract multiplicative rating factor tables from CatBoost models. Same output format as exp(&beta;) from a GLM: factor tables, confidence intervals, exposure weighting.</div>
  <div class="essential-card-tag">Model interpretation</div>
</div>

<div class="essential-card" data-name="insurance-cv" data-desc="Temporally correct cross-validation for insurance models with IBNR buffers">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></div>
  <div class="essential-card-desc">Walk-forward cross-validation that respects policy time structure. IBNR buffers prevent immature periods from contaminating validation folds. sklearn-compatible.</div>
  <div class="essential-card-tag">Validation</div>
</div>

<div class="essential-card" data-name="insurance-monitoring" data-desc="Model drift detection PSI CSI actual vs expected Gini z-test">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></div>
  <div class="essential-card-desc">Three-layer monitoring for deployed pricing models. Exposure-weighted PSI/CSI, segmented A/E ratios with IBNR adjustment, Gini z-test to distinguish recalibration from refit signals.</div>
  <div class="essential-card-tag">Monitoring</div>
</div>

<div class="essential-card" data-name="insurance-governance" data-desc="Model validation reports PRA SS1/23 Gini calibration double-lift">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></div>
  <div class="essential-card-desc">PRA SS1/23-compliant model validation reports. Bootstrap Gini CI, Poisson A/E CI, double-lift charts, renewal cohort test. HTML/JSON output for model governance committees.</div>
  <div class="essential-card-tag">Governance</div>
</div>

<div class="essential-card" data-name="credibility" data-desc="Buhlmann-Straub credibility weighting Python CatBoost Polars">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/credibility" target="_blank">credibility</a></div>
  <div class="essential-card-desc">Bühlmann-Straub credibility in Python with CatBoost and Polars. Practical for capping thin segments, stabilising NCD factors, and blending a new model with an incumbent rate.</div>
  <div class="essential-card-tag">Credibility</div>
</div>

<div class="essential-card" data-name="insurance-interactions" data-desc="Automated GLM interaction detection CANN NID SHAP interaction values">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-interactions" target="_blank">insurance-interactions</a></div>
  <div class="essential-card-desc">Detect, quantify, and present interaction effects that a main-effects-only GLM misses. CANN neural test, NID scoring, and SHAP interaction values.</div>
  <div class="essential-card-tag">Model building</div>
</div>

<div class="essential-card" data-name="insurance-causal" data-desc="Double machine learning causal inference deconfounding rating factors">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></div>
  <div class="essential-card-desc">Double machine learning for deconfounding rating factors. Separates genuine risk signal from correlated association — relevant wherever factors correlate with distribution channel or policyholder behaviour.</div>
  <div class="essential-card-tag">Causal inference</div>
</div>

<div class="essential-card" data-name="insurance-fairness" data-desc="Proxy discrimination auditing FCA EP25/2 Consumer Duty bias metrics">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></div>
  <div class="essential-card-desc">Proxy discrimination auditing for insurance pricing. FCA EP25/2-aligned, Consumer Duty compliant. Quantifies indirect discrimination risk from rating variables correlated with protected characteristics.</div>
  <div class="essential-card-tag">Regulatory</div>
</div>

<div class="essential-card" data-name="insurance-conformal" data-desc="Distribution-free prediction intervals GBM GLM Solvency II SCR">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a></div>
  <div class="essential-card-desc">Distribution-free prediction intervals with finite-sample coverage guarantees. Variance-weighted non-conformity scores produce tighter intervals. Solvency II SCR bounds included.</div>
  <div class="essential-card-tag">Uncertainty</div>
</div>

<div class="essential-card" data-name="insurance-optimise" data-desc="Constrained portfolio rate optimisation SLSQP FCA ENBP efficient frontier">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-optimise" target="_blank">insurance-optimise</a></div>
  <div class="essential-card-desc">Constrained portfolio rate optimisation via SLSQP. Enforces FCA ENBP constraints, movement caps per segment. Outputs the efficient frontier as a rate change recommendation with a JSON audit trail.</div>
  <div class="essential-card-tag">Optimisation</div>
</div>

</div>

---

## Search and Filter

<div class="tools-search-row">
  <input type="search" id="tools-search" class="tools-search-box" placeholder="Search libraries by name or description&hellip;" autocomplete="off">
  <div class="tools-filter-btns" id="cat-filters">
    <button class="cat-btn active" data-cat="all">All</button>
    <button class="cat-btn" data-cat="model-building">Model Building</button>
    <button class="cat-btn" data-cat="validation-monitoring">Validation &amp; Monitoring</button>
    <button class="cat-btn" data-cat="causal-inference">Causal Inference</button>
    <button class="cat-btn" data-cat="fairness-regulation">Fairness &amp; Regulation</button>
    <button class="cat-btn" data-cat="credibility-thin-data">Credibility &amp; Thin Data</button>
    <button class="cat-btn" data-cat="distributional-tail">Distributional &amp; Tail Risk</button>
    <button class="cat-btn" data-cat="optimisation-strategy">Optimisation &amp; Strategy</button>
    <button class="cat-btn" data-cat="time-series-trends">Time Series &amp; Trends</button>
    <button class="cat-btn" data-cat="other">Other</button>
  </div>
  <div class="tools-search-count" id="tools-search-count"></div>
</div>

<div id="tools-no-results" class="tools-no-results" style="display:none">No libraries match that search.</div>

---

## Model Building

Tools for building and interpreting pricing models.

<div class="tool-group" data-cat="model-building">

| Library | What it does |
|---|---|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | SHAP-based rating relativities from GBM models — extract GLM-style multiplicative factors from CatBoost |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tooling — nested GLM embeddings, R2VF factor level clustering, territory banding, SKATER |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | Interpretable GAMs — EBM tariffs, Actuarial NAM, Pairwise Interaction Networks, exact Shapley values |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection using CANN, NID scoring, and SHAP interaction values |
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | Sarmanov copula joint frequency-severity — analytical premium correction, IFM estimation |
| [insurance-scmoe](https://github.com/burning-cost/insurance-scmoe) | Spatially Clustered Mixture of Experts for joint frequency-severity pricing (NAAJ 2025) |
| [insurance-poisson-mixture-nn](https://github.com/burning-cost/insurance-poisson-mixture-nn) | Neural Poisson mixture for structural zero-claimers — PM-DNN, pi(x) at-risk score |
| [insurance-copula](https://github.com/burning-cost/insurance-copula) | Copula models — D-vine temporal dependence, two-part occurrence/severity |
| [insurance-sensitivity](https://github.com/burning-cost/insurance-sensitivity) | Global sensitivity analysis — Song 2016 Shapley effects, exposure-weighted variance decomposition |
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking — PyMC 5 ICAR, adjacency matrices, Moran's I diagnostics |

</div>

---

## Distributional & Tail Risk

Beyond point estimates: full distribution modelling and tail risk quantification.

<div class="tool-group" data-cat="distributional-tail">

| Library | What it does |
|---|---|
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Distributional GBMs — Tweedie, Gamma, ZIP, NegBin objectives with per-risk volatility scoring |
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | GAMLSS for Python — model all distribution parameters as functions of covariates, seven families |
| [insurance-dispersion](https://github.com/burning-cost/insurance-dispersion) | Double GLM for joint mean-dispersion modelling — alternating IRLS, REML, actuarial factor tables |
| [insurance-zit-dglm](https://github.com/burning-cost/insurance-zit-dglm) | Zero-Inflated Tweedie Double GLM — three-head CatBoost EM for mean, dispersion, and zero-inflation |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile and expectile GBMs for tail risk, TVaR, and increased limit factors |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Severity modelling toolkit for UK non-life insurance |
| [insurance-evt](https://github.com/burning-cost/insurance-evt) | Extreme Value Theory for catastrophic claim severity — GPD/GEV, censored MLE, ExcessGPD reinsurance, Solvency II 1-in-200 |
| [insurance-ilf](https://github.com/burning-cost/insurance-ilf) | MBBEFD exposure curves, Swiss Re families, ILF tables, per-risk XL pricing |
| [insurance-nflow](https://github.com/burning-cost/insurance-nflow) | Normalising flows for severity — NSF with Student-t tail transform, TVaR, ILF, conditional factor flow |

</div>

---

## Credibility & Thin Data

When you don't have enough data to trust a standard GLM.

<div class="tool-group" data-cat="credibility-thin-data">

| Library | What it does |
|---|---|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | Credibility models for UK non-life pricing: Bühlmann-Straub and Bayesian experience rating |
| [insurance-credibility-transformer](https://github.com/burning-cost/insurance-credibility-transformer) | Credibility Transformer — CLS token attention as Bühlmann-Straub credibility, ICL zero-shot pricing |
| [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | Two-stage CatBoost + REML random effects for high-cardinality group factors — ICC diagnostics |
| [insurance-experience](https://github.com/burning-cost/insurance-experience) | Individual policy Bayesian posterior experience rating — static, dynamic, surrogate, and deep attention credibility |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data pricing segments using PyMC 5 |
| [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) | Pricing techniques for low-volume segments where standard GLM fitting is unreliable |
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | Whittaker-Henderson 1D/2D smoothing with REML lambda selection, Bayesian CIs, Poisson PIRLS |

</div>

---

## Causal Inference

Separating what causes what from what correlates with what.

<div class="tool-group" data-cat="causal-inference">

| Library | What it does |
|---|---|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | Double machine learning for deconfounding rating factors — CatBoost nuisance models, confounding bias reports |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | Synthetic difference-in-differences for causal rate change evaluation — event study, HonestDiD sensitivity |
| [insurance-bcf](https://github.com/burning-cost/insurance-bcf) | Bayesian Causal Forests — stochtree BCF wrapper, segment CATE estimation, FCA EP25/2 audit report |
| [insurance-tmle](https://github.com/burning-cost/insurance-tmle) | Targeted Maximum Likelihood Estimation — doubly robust causal inference, Poisson TMLE with exposure offset |
| [insurance-rdd](https://github.com/burning-cost/insurance-rdd) | Regression discontinuity for rating thresholds — sharp/fuzzy RD, Poisson/Gamma outcomes, geographic RDD |
| [insurance-bunching](https://github.com/burning-cost/insurance-bunching) | Bunching estimators for rating threshold gaming — mileage fraud detection, adverse selection, BH FDR scanning |
| [insurance-counterfactual-sets](https://github.com/burning-cost/insurance-counterfactual-sets) | Weighted conformal ITE for individual counterfactuals — Lei & Candès 2021, FCA ENBP harm reports |
| [insurance-uplift](https://github.com/burning-cost/insurance-uplift) | Heterogeneous treatment effects for retention — CausalForestDML CATE, Qini/AUUC, PolicyTree, ENBP constraint |

</div>

---

## Fairness & Regulation

Proxy discrimination, Consumer Duty, and FCA evidence packs.

<div class="tool-group" data-cat="fairness-regulation">

| Library | What it does |
|---|---|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | Proxy discrimination auditing — FCA EP25/2, Consumer Duty, bias metrics |
| [insurance-mediation](https://github.com/burning-cost/insurance-mediation) | Causal mediation for FCA proxy discrimination — CDE/NDE/NIE decomposition, Imai sensitivity, FCA HTML report |
| [insurance-recourse](https://github.com/burning-cost/insurance-recourse) | Algorithmic recourse for FCA Consumer Duty in UK personal lines insurance |
| [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift) | Density ratio correction for book shifts — CatBoost/RuLSIF/KLIEP, LR-QR conformal, FCA SUP 15.3 diagnostics |

</div>

---

## Validation & Monitoring

Prediction intervals, conformal methods, and post-deployment monitoring.

<div class="tool-group" data-cat="validation-monitoring">

| Library | What it does |
|---|---|
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal walk-forward cross-validation — respects policy time structure, IBNR buffers, sklearn-compatible scorers |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | Distribution-free prediction intervals for insurance GBMs — locally-weighted Pearson residuals, Solvency II SCR bounds |
| [insurance-conformal-fraud](https://github.com/burning-cost/insurance-conformal-fraud) | Conformal anomaly detection for claims fraud — BH FDR control, integrative conformal p-values, Mondrian stratification |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, Poisson/NB scores |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger pricing framework — shadow mode, SHA-256 routing, SQLite quote log, bootstrap LR test |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — exposure-weighted PSI/CSI, actual-vs-expected ratios, Gini drift z-test |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | Model governance — PRA SS1/23 validation reports, model risk management, risk tier scoring |

</div>

---

## Optimisation & Pricing Strategy

From rate change recommendations to live price experimentation.

<div class="tool-group" data-cat="optimisation-strategy">

| Library | What it does |
|---|---|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained portfolio rate optimisation — SLSQP with analytical Jacobians, FCA ENBP constraints, efficient frontier |
| [insurance-dro](https://github.com/burning-cost/insurance-dro) | Distributionally robust rate optimisation — Wasserstein/KL/chi2 ambiguity sets, price-of-robustness frontier |
| [insurance-online](https://github.com/burning-cost/insurance-online) | Bandit algorithms for GIPP-compliant price experimentation — UCB1, Thompson Sampling, LinUCB, ENBP constraints |
| [insurance-reconcile](https://github.com/burning-cost/insurance-reconcile) | Hierarchical forecast reconciliation — PremiumWeightedMinTrace, LossRatioReconciler, FreqSevReconciler |

</div>

---

## Time Series & Trends

Inflation, reporting delays, and dynamic pricing.

<div class="tool-group" data-cat="time-series-trends">

| Library | What it does |
|---|---|
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Loss cost trend analysis — frequency/severity decomposition, ONS index integration, structural break detection |
| [insurance-nowcast](https://github.com/burning-cost/insurance-nowcast) | ML-EM nowcasting for claims reporting delays — covariate-conditioned completion factors, IBNR by risk segment |
| [insurance-garch](https://github.com/burning-cost/insurance-garch) | GARCH volatility for claims inflation — GJR-GARCH, BoE fan charts, Christoffersen backtest |
| [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | Dynamic pricing models — GAS score-driven filters, Bayesian changepoint detection (BOCPD/PELT) |

</div>

---

## Other Libraries

<div class="tool-group" data-cat="other">

| Library | What it does |
|---|---|
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Survival models — cure models, customer lifetime value, lapse tables, MLflow wrapper |
| [insurance-jlm](https://github.com/burning-cost/insurance-jlm) | Joint longitudinal-survival models — Wulfsohn-Tsiatis SREM, EM+GHQ, DynamicPredictor for mid-term repricing |
| [insurance-telematics](https://github.com/burning-cost/insurance-telematics) | HMM-based driving state classification and GLM-compatible risk scoring from raw telematics trip data |
| [insurance-lda-risk](https://github.com/burning-cost/insurance-lda-risk) | LDA probabilistic risk profiling — latent risk archetypes, portfolio mix drift, book-transfer segmentation |
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Vine copula synthetic portfolio generation — exposure-aware, preserves multivariate dependence |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK motor data with a known data-generating process, for testing and teaching |

</div>

<script>
(function() {
  var searchBox = document.getElementById('tools-search');
  var catBtns = document.querySelectorAll('.cat-btn');
  var toolGroups = document.querySelectorAll('.tool-group');
  var essentialCards = document.querySelectorAll('.essential-card');
  var noResults = document.getElementById('tools-no-results');
  var countEl = document.getElementById('tools-search-count');
  var activecat = 'all';

  // Build flat list of all table rows for searching
  function allRows() {
    var rows = [];
    toolGroups.forEach(function(group) {
      group.querySelectorAll('tbody tr').forEach(function(row) {
        rows.push({ el: row, group: group });
      });
    });
    return rows;
  }

  function applyFilters() {
    var q = searchBox ? searchBox.value.trim().toLowerCase() : '';
    var cat = activecat;
    var shown = 0;

    toolGroups.forEach(function(group) {
      var groupCat = group.getAttribute('data-cat') || '';
      var catMatch = cat === 'all' || groupCat === cat;

      if (!q) {
        // No search: just show/hide groups by category
        group.style.display = catMatch ? '' : 'none';
        if (catMatch) {
          group.querySelectorAll('tbody tr').forEach(function(r) { r.style.display = ''; shown++; });
        }
      } else {
        // With search: filter rows regardless of category (unless category filter active)
        var anyVisible = false;
        group.querySelectorAll('tbody tr').forEach(function(row) {
          var text = row.textContent.toLowerCase();
          var matchSearch = text.indexOf(q) > -1;
          var matchCat = cat === 'all' || groupCat === cat;
          var visible = matchSearch && matchCat;
          row.style.display = visible ? '' : 'none';
          if (visible) { anyVisible = true; shown++; }
        });
        group.style.display = anyVisible ? '' : 'none';
      }
    });

    // Essential cards: filter by search only (always shown on category filter)
    essentialCards.forEach(function(card) {
      if (!q) {
        card.style.display = '';
      } else {
        var name = (card.getAttribute('data-name') || '').toLowerCase();
        var desc = (card.getAttribute('data-desc') || '').toLowerCase();
        card.style.display = (name.indexOf(q) > -1 || desc.indexOf(q) > -1) ? '' : 'none';
      }
    });

    if (countEl && q) {
      countEl.textContent = shown + ' librar' + (shown === 1 ? 'y' : 'ies') + ' match';
    } else if (countEl) {
      countEl.textContent = '';
    }

    if (noResults) {
      noResults.style.display = (shown === 0 && q) ? 'block' : 'none';
    }
  }

  if (searchBox) {
    searchBox.addEventListener('input', applyFilters);
  }

  catBtns.forEach(function(btn) {
    btn.addEventListener('click', function() {
      catBtns.forEach(function(b) { b.classList.remove('active'); });
      this.classList.add('active');
      activecat = this.getAttribute('data-cat');
      applyFilters();
    });
  });
})();
</script>
