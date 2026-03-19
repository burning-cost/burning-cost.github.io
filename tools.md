---
layout: tools
title: "Open-Source Python Libraries for Insurance Pricing"
description: "Open-source Python libraries for UK non-life pricing — 10 flagship tools for proxy discrimination auditing, PRA SS1/23 validation, DML causal inference, conformal prediction, Whittaker-Henderson smoothing, GAMLSS, and Bühlmann-Straub credibility."
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
      "name": "insurance-severity",
      "description": "Spliced Pareto/Gamma severity, Deep Regression Networks, composite Lognormal-GPD, and EQRN extreme quantile neural networks for large loss modelling.",
      "codeRepository": "https://github.com/burning-cost/insurance-severity",
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
      "name": "insurance-multilevel",
      "description": "Two-stage CatBoost + REML random effects for high-cardinality group factors — Bühlmann-Straub credibility, ICC diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-multilevel",
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
      "name": "insurance-covariate-shift",
      "description": "Density ratio correction for book shifts — CatBoost/RuLSIF/KLIEP, LR-QR conformal, FCA SUP 15.3 diagnostics.",
      "codeRepository": "https://github.com/burning-cost/insurance-covariate-shift",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-fairness",
      "description": "Proxy discrimination auditing for insurance pricing — FCA Consumer Duty, Equality Act 2010, bias metrics.",
      "codeRepository": "https://github.com/burning-cost/insurance-fairness",
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
      "name": "insurance-conformal-ts",
      "description": "Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, MSCP, Poisson/NB scores.",
      "codeRepository": "https://github.com/burning-cost/insurance-conformal-ts",
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
      "name": "insurance-trend",
      "description": "Loss cost trend analysis — frequency/severity decomposition, ONS index integration, structural break detection.",
      "codeRepository": "https://github.com/burning-cost/insurance-trend",
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
      "name": "insurance-telematics",
      "description": "HMM-based driving state classification and GLM-compatible risk scoring from raw telematics trip data.",
      "codeRepository": "https://github.com/burning-cost/insurance-telematics",
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
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-datasets",
      "description": "Synthetic UK motor portfolio with known DGP parameters — validate that your model recovers true relativities before using real data.",
      "codeRepository": "https://github.com/burning-cost/insurance-datasets",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "experience-rating",
      "description": "Individual policy Bayesian posterior experience rating — static Bühlmann-Straub, dynamic state-space, surrogate IS posteriors, and deep attention credibility.",
      "codeRepository": "https://github.com/burning-cost/experience-rating",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-demand",
      "description": "Demand modelling for insurance pricing — conversion and retention elasticity, price response curves.",
      "codeRepository": "https://github.com/burning-cost/insurance-demand",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    },
    {
      "@type": "SoftwareSourceCode",
      "name": "insurance-elasticity",
      "description": "Price elasticity estimation for insurance — CausalForestDML, DR-Learner, and Automatic Debiased ML.",
      "codeRepository": "https://github.com/burning-cost/insurance-elasticity",
      "programmingLanguage": "Python",
      "license": "https://opensource.org/licenses/MIT"
    }
  ]
}
</script>

<div class="tools-intro">
  <p>37 open-source Python libraries covering the full pricing stack. Ten of them we consider genuinely differentiated — tools that address hard problems in UK pricing where no adequate Python solution existed before. The full portfolio is below the flagship section.</p>
  <p>All libraries are MIT-licensed, installable via pip, and built for Python 3.10+. The GitHub organisation is <a href="https://github.com/burning-cost">burning-cost</a>.</p>
</div>

---

## Flagship Libraries

Ten libraries for the problems that matter most. Each addresses a specific hard problem — regulatory compliance, causal estimation, uncertainty quantification, smoothing, telematics scoring — where practitioners were previously doing things by hand or not doing them at all.

<div class="essential-grid" id="essential-grid">

<div class="essential-card" data-name="insurance-fairness" data-desc="Proxy discrimination auditing FCA Consumer Duty Equality Act 2010 bias metrics protected characteristics">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></div>
  <div class="essential-card-desc">Proxy discrimination auditing aligned to FCA Consumer Duty. Quantifies indirect discrimination risk from rating variables correlated with protected characteristics. Equality Act 2010 proportionality documentation built in.</div>
  <div class="essential-card-tag">FCA Consumer Duty</div>
</div>

<div class="essential-card" data-name="insurance-governance" data-desc="PRA SS1/23 model validation MRM Gini calibration double-lift governance report">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></div>
  <div class="essential-card-desc">PRA SS1/23-compliant model validation reports. Bootstrap Gini CI, Poisson A/E CI, double-lift charts, renewal cohort test. HTML/JSON output structured for model risk committees and PRA review.</div>
  <div class="essential-card-tag">PRA SS1/23 &middot; MRM</div>
</div>

<div class="essential-card" data-name="insurance-causal" data-desc="Double machine learning DML causal inference deconfounding rating factors channel bias">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></div>
  <div class="essential-card-desc">Double machine learning for deconfounding rating factors. Standard GLM coefficients are biased wherever rating variables correlate with distribution channel or policyholder selection. DML removes that bias without a structural model.</div>
  <div class="essential-card-tag">Causal inference &middot; DML</div>
</div>

<div class="essential-card" data-name="insurance-whittaker" data-desc="Whittaker-Henderson smoothing REML GCV lambda selection Bayesian CI age curves NCD">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-whittaker" target="_blank">insurance-whittaker</a></div>
  <div class="essential-card-desc">Whittaker-Henderson smoothing for experience rating tables. 1D, 2D, and Poisson variants with REML lambda selection and Bayesian credible intervals. Smooths age curves, NCD scales, and vehicle group relativities without parametric assumptions.</div>
  <div class="essential-card-tag">Non-parametric smoothing &middot; REML</div>
</div>

<div class="essential-card" data-name="insurance-telematics" data-desc="HMM telematics driving state classification risk score UBI GLM GPS accelerometer">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-telematics" target="_blank">insurance-telematics</a></div>
  <div class="essential-card-desc">End-to-end pipeline from raw 1Hz GPS/accelerometer data to GLM-compatible risk scores. HMM driving state classification (cautious/normal/aggressive), Bühlmann-Straub credibility aggregation to driver level, Poisson GLM integration.</div>
  <div class="essential-card-tag">Telematics &middot; HMM &middot; UBI</div>
</div>

<div class="essential-card" data-name="insurance-conformal" data-desc="conformal prediction distribution-free prediction intervals coverage guarantee GBM Solvency II SCR">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a></div>
  <div class="essential-card-desc">Distribution-free prediction intervals with finite-sample coverage guarantees. Variance-weighted non-conformity scores produce tighter intervals. Beyond point estimates: principled uncertainty for every risk score. Solvency II SCR bounds included.</div>
  <div class="essential-card-tag">Conformal prediction &middot; Solvency II</div>
</div>

<div class="essential-card" data-name="insurance-credibility" data-desc="Buhlmann-Straub credibility Python thin segments NCD factors mixed model">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-credibility" target="_blank">insurance-credibility</a></div>
  <div class="essential-card-desc">Bühlmann-Straub credibility in Python with mixed-model equivalence checks. Caps thin segments, stabilises NCD factors, blends a new model with an incumbent rate. The actuarial answer to the thin data problem.</div>
  <div class="essential-card-tag">Bühlmann-Straub &middot; Thin data</div>
</div>

<div class="essential-card" data-name="insurance-frequency-severity" data-desc="Sarmanov copula joint frequency severity modelling GLM marginals IFM premium correction dependence">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-frequency-severity" target="_blank">insurance-frequency-severity</a></div>
  <div class="essential-card-desc">Sarmanov copula joint frequency-severity modelling with GLM marginals. IFM estimation, analytical premium correction, Garrido conditional severity, dependence tests. Tests the independence assumption every pricing model makes.</div>
  <div class="essential-card-tag">Sarmanov copula &middot; Joint modelling</div>
</div>

<div class="essential-card" data-name="insurance-gam" data-desc="EBM NAM interpretable GAM neural additive model factor table deep learning insurance">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-gam" target="_blank">insurance-gam</a></div>
  <div class="essential-card-desc">EBM and Neural Additive Model for interpretable deep learning in insurance pricing. Shape functions per rating factor give the transparency of a GLM with the predictive power of a neural network. Factor table output for pricing committee review.</div>
  <div class="essential-card-tag">EBM &middot; NAM &middot; Interpretable ML</div>
</div>

<div class="essential-card" data-name="insurance-distributional-glm" data-desc="GAMLSS distributional GLM mean dispersion shape zero inflation RS algorithm seven families">
  <div class="essential-card-name"><a href="https://github.com/burning-cost/insurance-distributional-glm" target="_blank">insurance-distributional-glm</a></div>
  <div class="essential-card-desc">GAMLSS for Python: model all distribution parameters as functions of covariates — mean, dispersion, shape, zero-inflation. Seven families, RS algorithm with backtracking. The standard GLM models the mean. This models the whole distribution.</div>
  <div class="essential-card-tag">GAMLSS &middot; Beyond point estimates</div>
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

| Library | What it does | Install |
|---|---|---|
| [shap-relativities](https://github.com/burning-cost/shap-relativities) | SHAP-based rating relativities from GBM models — extract GLM-style multiplicative factors from CatBoost<br>_Benchmark: +2.85pp Gini lift over GLM; 9.4% mean relativity error (honest trade-off vs GLM's 4.5%)_ | `pip install shap-relativities` |
| [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools) | GLM tooling — nested GLM embeddings, R2VF factor level clustering, territory banding, SKATER<br>_Benchmark: R2VF fused lasso clustering vs manual quintile banding benchmarked_ | `pip install insurance-glm-tools` |
| [insurance-gam](https://github.com/burning-cost/insurance-gam) | **Flagship.** Interpretable GAMs — EBM tariffs, Actuarial NAM, Pairwise Interaction Networks, exact Shapley values<br>_Benchmark: EBM/NAM interpretable tariffs with exact Shapley values benchmarked on synthetic motor data_ | `pip install insurance-gam` |
| [insurance-interactions](https://github.com/burning-cost/insurance-interactions) | Automated GLM interaction detection using CANN, NID scoring, and SHAP interaction values<br>_Benchmark: Production defaults recover both planted interactions; compact config less reliable_ | `pip install insurance-interactions` |
| [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) | **Flagship.** Sarmanov copula joint frequency-severity — analytical premium correction, IFM estimation<br>_Benchmark: Sarmanov copula joint modelling with analytical premium correction benchmarked_ | `pip install insurance-frequency-severity` |
| [insurance-spatial](https://github.com/burning-cost/insurance-spatial) | BYM2 spatial territory ratemaking — PyMC 5 ICAR, adjacency matrices, Moran's I diagnostics<br>_Benchmark: BYM2 vs raw vs manual banding benchmarked on synthetic territory data_ | `pip install insurance-spatial` |
| [insurance-distill](https://github.com/burning-cost/insurance-distill) | GBM-to-GLM distillation — fits a surrogate Poisson/Gamma GLM to CatBoost predictions, exports multiplicative factor tables for Radar/Emblem | `pip install insurance-distill` |

</div>

---

## Distributional & Tail Risk

Beyond point estimates: full distribution modelling and tail risk quantification.

<div class="tool-group" data-cat="distributional-tail">

| Library | What it does | Install |
|---|---|---|
| [insurance-distributional](https://github.com/burning-cost/insurance-distributional) | Distributional GBMs — Tweedie, Gamma, ZIP, NegBin objectives with per-risk volatility scoring<br>_Benchmark: GammaGBM +1.5% log-likelihood; prediction intervals calibrated_ | `pip install insurance-distributional` |
| [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm) | **Flagship.** GAMLSS for Python — model all distribution parameters as functions of covariates, seven families<br>_Benchmark: GAMLSS sigma correlation 0.998 vs 0.000 for constant-phi_ | `pip install insurance-distributional-glm` |
| [insurance-dispersion](https://github.com/burning-cost/insurance-dispersion) | Double GLM for joint mean-dispersion modelling — alternating IRLS, REML, actuarial factor tables<br>_Benchmark: Double GLM captures heteroscedasticity that constant-phi misses_ | `pip install insurance-dispersion` |
| [insurance-quantile](https://github.com/burning-cost/insurance-quantile) | Quantile and expectile GBMs for tail risk, TVaR, and increased limit factors<br>_Benchmark: GBM lower TVaR bias on heavy tails; lognormal wins on pinball at small n_ | `pip install insurance-quantile` |
| [insurance-severity](https://github.com/burning-cost/insurance-severity) | Spliced Pareto/Gamma severity, Deep Regression Networks, composite Lognormal-GPD, and EQRN extreme quantile neural networks for large loss modelling<br>_Benchmark: Composite reduces tail error 5.6% vs single lognormal_ | `pip install insurance-severity` |

</div>

---

## Credibility & Thin Data

When you don't have enough data to trust a standard GLM.

<div class="tool-group" data-cat="credibility-thin-data">

| Library | What it does | Install |
|---|---|---|
| [insurance-credibility](https://github.com/burning-cost/insurance-credibility) | **Flagship.** Credibility models for UK non-life pricing: Bühlmann-Straub and Bayesian experience rating<br>_Benchmark: 6.8% MAE improvement on thin schemes vs raw experience_ | `pip install insurance-credibility` |
| [insurance-multilevel](https://github.com/burning-cost/insurance-multilevel) | Two-stage CatBoost + REML random effects for high-cardinality group factors — ICC diagnostics<br>_Benchmark: 15.9% gamma deviance reduction; thin-group MAPE 63.6% vs one-hot 66.1%_ | `pip install insurance-multilevel` |
| [experience-rating](https://github.com/burning-cost/experience-rating) | Individual policy Bayesian posterior experience rating — static, dynamic, surrogate, and deep attention credibility | `pip install experience-rating` |
| [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing) | Hierarchical Bayesian models for thin-data pricing segments using PyMC 5 | `pip install bayesian-pricing` |
| [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data) | Pricing techniques for low-volume segments where standard GLM fitting is unreliable<br>_Benchmark: GLMTransfer narrows bootstrap 90% CI widths 30–60% vs standalone GLM_ | `pip install insurance-thin-data` |
| [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) | **Flagship.** Whittaker-Henderson 1D/2D smoothing with REML lambda selection, Bayesian CIs, Poisson PIRLS<br>_Benchmark: 57.2% MSE reduction vs raw rates_ | `pip install insurance-whittaker` |

</div>

---

## Causal Inference

Separating what causes what from what correlates with what.

<div class="tool-group" data-cat="causal-inference">

| Library | What it does | Install |
|---|---|---|
| [insurance-causal](https://github.com/burning-cost/insurance-causal) | **Flagship.** Double machine learning for deconfounding rating factors — CatBoost nuisance models, confounding bias reports<br>_Benchmark: DML removes nonlinear confounding bias at scale (n≥50k); honest: over-partials at small n_ | `pip install insurance-causal` |
| [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) | Synthetic difference-in-differences for causal rate change evaluation — event study, HonestDiD sensitivity<br>_Benchmark: SDID 98% CI coverage; naive before-after biased +3.8pp by market inflation_ | `pip install insurance-causal-policy` |
| [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) | Price elasticity estimation — CausalForestDML, DR-Learner, and Automatic Debiased ML<br>_Benchmark: DML 47.6% GATE RMSE improvement over OLS; valid CIs_ | `pip install insurance-elasticity` |

</div>

---

## Fairness & Regulation

Proxy discrimination, Consumer Duty, and FCA evidence packs.

<div class="tool-group" data-cat="fairness-regulation">

| Library | What it does | Install |
|---|---|---|
| [insurance-fairness](https://github.com/burning-cost/insurance-fairness) | **Flagship.** Proxy discrimination auditing — FCA Consumer Duty, Equality Act 2010, bias metrics<br>_Benchmark: Proxy R²=0.78 catches postcode proxy that Spearman (r=0.06) misses entirely_ | `pip install insurance-fairness` |
| [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift) | Density ratio correction for book shifts — CatBoost/RuLSIF/KLIEP, LR-QR conformal, FCA SUP 15.3 diagnostics<br>_Benchmark: Density ratio correction for portfolio composition shift benchmarked_ | `pip install insurance-covariate-shift` |

</div>

---

## Validation & Monitoring

Prediction intervals, conformal methods, and post-deployment monitoring.

<div class="tool-group" data-cat="validation-monitoring">

| Library | What it does | Install |
|---|---|---|
| [insurance-cv](https://github.com/burning-cost/insurance-cv) | Temporal walk-forward cross-validation — respects policy time structure, IBNR buffers, sklearn-compatible scorers<br>_Benchmark: Walk-forward catches 10.5% optimism that k-fold hides_ | `pip install insurance-cv` |
| [insurance-conformal](https://github.com/burning-cost/insurance-conformal) | **Flagship.** Distribution-free prediction intervals for insurance GBMs — locally-weighted Pearson residuals, Solvency II SCR bounds<br>_Benchmark: 90.1% marginal coverage on Tweedie frequency data_ | `pip install insurance-conformal` |
| [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) | Conformal prediction for non-exchangeable claims time series — ACI, EnbPI, SPCI, Poisson/NB scores<br>_Benchmark: Sequential conformal vs static prediction intervals on claims time series benchmarked_ | `pip install insurance-conformal-ts` |
| [insurance-deploy](https://github.com/burning-cost/insurance-deploy) | Champion/challenger pricing framework — shadow mode, SHA-256 routing, SQLite quote log, bootstrap LR test<br>_Benchmark: Champion/challenger routing, shadow mode, quote logging benchmarked_ | `pip install insurance-deploy` |
| [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) | Model drift detection — exposure-weighted PSI/CSI, actual-vs-expected ratios, Gini drift z-test<br>_Benchmark: Exposure-weighted PSI/CSI vs aggregate A/E benchmarked_ | `pip install insurance-monitoring` |
| [insurance-governance](https://github.com/burning-cost/insurance-governance) | **Flagship.** Model governance — PRA SS1/23 validation reports, model risk management, risk tier scoring<br>_Benchmark: Automated suite catches age-band miscalibration manual checklists miss_ | `pip install insurance-governance` |

</div>

---

## Optimisation & Pricing Strategy

From rate change recommendations to live price experimentation.

<div class="tool-group" data-cat="optimisation-strategy">

| Library | What it does | Install |
|---|---|---|
| [insurance-optimise](https://github.com/burning-cost/insurance-optimise) | Constrained portfolio rate optimisation — SLSQP with analytical Jacobians, FCA ENBP constraints, efficient frontier<br>_Benchmark: Demand-curve pricing +143.8% profit lift over flat loading_ | `pip install insurance-optimise` |
| [insurance-demand](https://github.com/burning-cost/insurance-demand) | Demand modelling — conversion and retention elasticity, price response curves<br>_Benchmark: Recovers true price elasticity (-2.09 vs true -2.0); naive model off by 5x_ | `pip install insurance-demand` |

</div>

---

## Time Series & Trends

Inflation, reporting delays, and dynamic pricing.

<div class="tool-group" data-cat="time-series-trends">

| Library | What it does | Install |
|---|---|---|
| [insurance-trend](https://github.com/burning-cost/insurance-trend) | Loss cost trend analysis — frequency/severity decomposition, ONS index integration, structural break detection<br>_Benchmark: 3.93pp MAPE improvement over naive OLS trend_ | `pip install insurance-trend` |
| [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics) | Dynamic pricing models — GAS score-driven filters, Bayesian changepoint detection (BOCPD/PELT)<br>_Benchmark: GAS Poisson +13% MAE improvement over GLM trend_ | `pip install insurance-dynamics` |

</div>

---

## Other Libraries

<div class="tool-group" data-cat="other">

| Library | What it does | Install |
|---|---|---|
| [insurance-survival](https://github.com/burning-cost/insurance-survival) | Survival models — cure models, customer lifetime value, lapse tables, MLflow wrapper<br>_Benchmark: Cure model recovers 34.1% cure fraction (true 35.0%); KM/Cox extrapolate to zero_ | `pip install insurance-survival` |
| [insurance-telematics](https://github.com/burning-cost/insurance-telematics) | **Flagship.** HMM-based driving state classification and GLM-compatible risk scoring from raw telematics trip data<br>_Benchmark: HMM state features 3–8pp Gini improvement over raw trip averages_ | `pip install insurance-telematics` |
| [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic) | Vine copula synthetic portfolio generation — exposure-aware, preserves multivariate dependence<br>_Benchmark: Copula 64% better correlation preservation vs naive_ | `pip install insurance-synthetic` |
| [insurance-datasets](https://github.com/burning-cost/insurance-datasets) | Synthetic UK motor portfolio with known DGP parameters — validate that your model recovers true relativities before using real data<br>_Benchmark: GLM parameter recovery RMSE 0.069; OVB demo shows 24% NCD inflation when age omitted_ | `pip install insurance-datasets` |

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
