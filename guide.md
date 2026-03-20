---
layout: page
title: "Which library do I need?"
description: "A problem-oriented decision guide: map a pricing challenge to the right Burning Cost library. Organised by workflow stage — data preparation through to deployment and monitoring."
permalink: /guide/
---

<style>
.guide-intro {
  font-size: 1.05rem;
  color: #2a2d3e;
  max-width: 680px;
  line-height: 1.7;
  margin-bottom: 2.5rem;
}

/* Quick reference table */
.qr-section {
  margin-bottom: 3rem;
}

.qr-section h2 {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f1117;
  letter-spacing: -0.02em;
  margin-bottom: 0.4rem;
}

.qr-sub {
  font-size: 0.9rem;
  color: #5a6278;
  margin-bottom: 1.25rem;
}

.qr-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.qr-table thead tr {
  background: #f4f6fb;
  border-bottom: 2px solid #e2e6f0;
}

.qr-table thead th {
  padding: 0.65rem 0.85rem;
  text-align: left;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #5a6278;
}

.qr-table tbody tr {
  border-bottom: 1px solid #e2e6f0;
}

.qr-table tbody tr:hover {
  background: #f9fafc;
}

.qr-table td {
  padding: 0.6rem 0.85rem;
  vertical-align: top;
  line-height: 1.5;
  color: #2a2d3e;
}

.qr-table td:first-child {
  font-style: italic;
  color: #4a4f66;
}

.qr-table td code {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  color: #4f7ef8;
  background: rgba(79,126,248,0.08);
  padding: 0.1rem 0.35rem;
  border-radius: 4px;
  white-space: nowrap;
}

/* Stage sections */
.guide-stage {
  margin-bottom: 3rem;
}

.guide-stage-header {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.25rem;
  padding-bottom: 0.75rem;
  border-bottom: 2px solid #e2e6f0;
}

.guide-stage-num {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  height: 2rem;
  background: #4f7ef8;
  color: #fff;
  font-size: 0.8rem;
  font-weight: 700;
  border-radius: 50%;
  flex-shrink: 0;
}

.guide-stage-title {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f1117;
  letter-spacing: -0.02em;
  margin: 0;
}

.guide-stage-desc {
  font-size: 0.88rem;
  color: #5a6278;
  margin: 0.35rem 0 0 2.75rem;
  line-height: 1.6;
}

/* Library cards */
.lib-entries {
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.lib-entry {
  border: 1px solid #e2e6f0;
  border-radius: 10px;
  padding: 1.1rem 1.25rem;
  background: #fff;
}

.lib-entry-top {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.5rem;
  flex-wrap: wrap;
}

.lib-entry-name {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.87rem;
  font-weight: 700;
}

.lib-entry-name a {
  color: #4f7ef8;
  text-decoration: none;
}

.lib-entry-name a:hover {
  text-decoration: underline;
}

.lib-complexity {
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 0.2rem 0.55rem;
  border-radius: 100px;
  white-space: nowrap;
  flex-shrink: 0;
}

.complexity-drop-in {
  background: #d4f8e8;
  color: #1a6b44;
  border: 1px solid #a8ecd0;
}

.complexity-setup {
  background: #fff8e1;
  color: #7a5a00;
  border: 1px solid #ffe09e;
}

.complexity-advanced {
  background: #fde8e8;
  color: #8b1a1a;
  border: 1px solid #f5b8b8;
}

.lib-entry-problem {
  font-size: 0.88rem;
  color: #4a4f66;
  line-height: 1.6;
  margin-bottom: 0.5rem;
}

.lib-entry-output {
  font-size: 0.8rem;
  color: #5a6278;
}

.lib-entry-output strong {
  font-weight: 600;
  color: #0f1117;
}

/* Complexity legend */
.complexity-legend {
  display: flex;
  gap: 1rem;
  font-size: 0.78rem;
  color: #5a6278;
  margin-bottom: 2rem;
  flex-wrap: wrap;
}

.complexity-legend-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

@media (max-width: 640px) {
  .qr-table td:first-child {
    min-width: 160px;
  }
  .guide-stage-desc {
    margin-left: 0;
    margin-top: 0.5rem;
  }
  .guide-stage-header {
    flex-wrap: wrap;
  }
  .lib-entry-top {
    flex-direction: column;
    gap: 0.5rem;
  }
}
</style>

<p class="guide-intro">Not every problem needs the same tool. This guide maps common pricing challenges to specific libraries, organised by where in the workflow you hit them. If you know what you are trying to do, find it here. If you are not sure where to start, try <a href="/getting-started/">Getting Started</a> first.</p>

---

## Quick reference

<div class="qr-section">
<p class="qr-sub">Find the library that matches the task. Click any library name to go to its GitHub repo.</p>

<table class="qr-table">
<thead>
<tr><th>I want to&hellip;</th><th>Library</th></tr>
</thead>
<tbody>
<tr><td>extract GLM-style factors from a GBM</td><td><a href="https://github.com/burning-cost/shap-relativities"><code>shap-relativities</code></a></td></tr>
<tr><td>run cross-validation without leaking future claims</td><td><a href="https://github.com/burning-cost/insurance-cv"><code>insurance-cv</code></a></td></tr>
<tr><td>smooth noisy one-way curves</td><td><a href="https://github.com/burning-cost/insurance-whittaker"><code>insurance-whittaker</code></a></td></tr>
<tr><td>get spatial territory factors</td><td><a href="https://github.com/burning-cost/insurance-spatial"><code>insurance-spatial</code></a></td></tr>
<tr><td>detect if my model has drifted</td><td><a href="https://github.com/burning-cost/insurance-monitoring"><code>insurance-monitoring</code></a></td></tr>
<tr><td>prove my pricing is not discriminatory</td><td><a href="https://github.com/burning-cost/insurance-fairness"><code>insurance-fairness</code></a></td></tr>
<tr><td>generate a PRA SS1/23 validation report</td><td><a href="https://github.com/burning-cost/insurance-governance"><code>insurance-governance</code></a></td></tr>
<tr><td>get prediction intervals on my GBM</td><td><a href="https://github.com/burning-cost/insurance-conformal"><code>insurance-conformal</code></a></td></tr>
<tr><td>model frequency and severity jointly</td><td><a href="https://github.com/burning-cost/insurance-frequency-severity"><code>insurance-frequency-severity</code></a></td></tr>
<tr><td>price a segment with 200 policies</td><td><a href="https://github.com/burning-cost/insurance-thin-data"><code>insurance-thin-data</code></a></td></tr>
<tr><td>deconfound a rating factor from channel correlation</td><td><a href="https://github.com/burning-cost/insurance-causal"><code>insurance-causal</code></a></td></tr>
<tr><td>measure whether a rate change worked</td><td><a href="https://github.com/burning-cost/insurance-causal-policy"><code>insurance-causal-policy</code></a></td></tr>
<tr><td>set rates with movement caps and a loss ratio target</td><td><a href="https://github.com/burning-cost/insurance-optimise"><code>insurance-optimise</code></a></td></tr>
<tr><td>run a champion/challenger experiment</td><td><a href="https://github.com/burning-cost/insurance-deploy"><code>insurance-deploy</code></a></td></tr>
<tr><td>model tail risk beyond the mean</td><td><a href="https://github.com/burning-cost/insurance-quantile"><code>insurance-quantile</code></a></td></tr>
<tr><td>cluster vehicle groups or occupation codes into bands</td><td><a href="https://github.com/burning-cost/insurance-glm-tools"><code>insurance-glm-tools</code></a></td></tr>
<tr><td>model dispersion as well as the mean</td><td><a href="https://github.com/burning-cost/insurance-distributional-glm"><code>insurance-distributional-glm</code></a></td></tr>
<tr><td>build a broker or fleet-adjusted model</td><td><a href="https://github.com/burning-cost/insurance-multilevel"><code>insurance-multilevel</code></a></td></tr>
<tr><td>apply individual experience rating to a policy</td><td><a href="https://github.com/burning-cost/insurance-credibility"><code>insurance-credibility</code></a> (includes experience rating)</td></tr>
<tr><td>generate synthetic training data</td><td><a href="https://github.com/burning-cost/insurance-synthetic"><code>insurance-synthetic</code></a></td></tr>
<tr><td>score telematics trips as GLM-compatible risk</td><td><a href="https://github.com/burning-cost/insurance-telematics"><code>insurance-telematics</code></a></td></tr>
<tr><td>estimate price elasticity causally</td><td><a href="https://github.com/burning-cost/insurance-causal"><code>insurance-causal</code></a> (includes elasticity)</td></tr>
<tr><td>correct for a shift in my book mix</td><td><a href="https://github.com/burning-cost/insurance-covariate-shift"><code>insurance-covariate-shift</code></a></td></tr>
</tbody>
</table>
</div>

---

<div class="complexity-legend">
  <div class="complexity-legend-item"><span class="lib-complexity complexity-drop-in">Drop-in</span> Works like sklearn. No domain configuration needed.</div>
  <div class="complexity-legend-item"><span class="lib-complexity complexity-setup">Needs setup</span> Requires adjacency matrices, priors, or portfolio structure.</div>
  <div class="complexity-legend-item"><span class="lib-complexity complexity-advanced">Advanced</span> Expects familiarity with the statistical method.</div>
</div>

---

## Stage 1: Data preparation

<div class="guide-stage">
<div class="guide-stage-header">
  <div class="guide-stage-num">1</div>
  <h2 class="guide-stage-title">Data preparation</h2>
</div>
<p class="guide-stage-desc">Before any modelling: synthetic data for testing, representative training sets, and correcting for the fact that your historical book is not your target book.</p>

<div class="lib-entries">

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-datasets" target="_blank">insurance-datasets</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">You need real-looking motor data to test a method but cannot share your book externally — or you want a benchmark with a known data-generating process to validate whether your model is even recovering the right signal.</p>
  <div class="lib-entry-output"><strong>Output:</strong> synthetic UK motor portfolio DataFrames with known frequency, severity, and interaction structure</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-synthetic" target="_blank">insurance-synthetic</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your model development team cannot touch production data, but the synthetic data you have does not preserve the multivariate dependence structure of the real book — correlations between vehicle age, driver age, and claim frequency are wrong.</p>
  <div class="lib-entry-output"><strong>Output:</strong> vine copula synthetic portfolio with exposure-aware Poisson marginals and a TSTR Gini fidelity report</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-covariate-shift" target="_blank">insurance-covariate-shift</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Your book mix has shifted — you wrote more young drivers last year through a new aggregator — and the model fitted on historical data is mispriced for today's portfolio because it was never trained on this mix.</p>
  <div class="lib-entry-output"><strong>Output:</strong> importance weights for reweighting training data, LR-QR conformal prediction intervals under shift, FCA SUP 15.3 diagnostic report</div>
</div>

</div>
</div>

---

## Stage 2: Core modelling

<div class="guide-stage">
<div class="guide-stage-header">
  <div class="guide-stage-num">2</div>
  <h2 class="guide-stage-title">Core modelling</h2>
</div>
<p class="guide-stage-desc">Building and validating the technical price — from temporally correct cross-validation through to distributional and thin-data methods.</p>

<div class="lib-entries">

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your GBM outperforms the production GLM on every holdout metric, but the actuarial team and your rating engine both need multiplicative factor tables — and exp(&beta;) from CatBoost is not a thing.</p>
  <div class="lib-entry-output"><strong>Output:</strong> multiplicative factor tables per rating variable with confidence intervals, exposure weighting, and reconstruction R² validation</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">You used random k-fold CV and your held-out Poisson deviance looks fine, but the model was trained on claims that had not yet developed — IBNR from the last six months is in both your training and validation sets.</p>
  <div class="lib-entry-output"><strong>Output:</strong> walk-forward split indices with configurable IBNR buffer, sklearn-compatible scorers including Poisson and Gamma deviance</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-interactions" target="_blank">insurance-interactions</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your main-effects GLM has unexplained residual structure — one-ways look fine but double-lifts between age and vehicle group show a gap the model cannot explain, and you suspect an interaction you have not captured.</p>
  <div class="lib-entry-output"><strong>Output:</strong> ranked list of significant interactions via CANN neural test, NID scores, and SHAP interaction decomposition</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-distributional" target="_blank">insurance-distributional</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your point-estimate GBM tells you the expected loss cost per risk, but you have no way to distinguish a low-risk policy with stable claims from a high-risk policy that just happened to be quiet last year — you need per-risk volatility.</p>
  <div class="lib-entry-output"><strong>Output:</strong> full predictive distribution per risk (Tweedie, Gamma, ZIP, NegBin), per-risk volatility score, mean and variance separately</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-distributional-glm" target="_blank">insurance-distributional-glm</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Your overdispersion test is significant and the dispersion parameter phi varies by segment — fleet policies have much higher variance than personal lines — but a standard GLM assumes one scalar phi for the whole portfolio.</p>
  <div class="lib-entry-output"><strong>Output:</strong> GAMLSS model where mean, dispersion, and shape are all functions of covariates; factor tables for each distribution parameter</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-dispersion" target="_blank">insurance-dispersion</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You want to model dispersion as a function of rating factors without the full complexity of GAMLSS — a simpler double GLM where the mean model and the dispersion model alternate until convergence.</p>
  <div class="lib-entry-output"><strong>Output:</strong> joint mean-dispersion GLM, overdispersion LRT, actuarial factor tables for both mean and dispersion submodels</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-gam" target="_blank">insurance-gam</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You need interpretability that a GLM factor table gives you, but you also need the predictive power of a neural network — and you want exact Shapley values rather than approximate SHAP for sign-off.</p>
  <div class="lib-entry-output"><strong>Output:</strong> EBM tariff or actuarial NAM with shape functions per rating factor, exact Shapley values, pairwise interaction network</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-glm-tools" target="_blank">insurance-glm-tools</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your vehicle make variable has 500 levels, your occupation code has 350 — the committee wants pricing bands, not a 500-row factor table, and you need a defensible method for collapsing them that respects the underlying risk signal.</p>
  <div class="lib-entry-output"><strong>Output:</strong> GLM-compatible pricing bands via R2VF algorithm, ridge ranking for nominals, fused lasso for ordered levels, BIC lambda selection</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-whittaker" target="_blank">insurance-whittaker</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your one-way driver age curve is spiky — the raw relativities jump between adjacent age bands because your data is thin in some cells — and you are eye-balling a smooth curve through it rather than doing this formally.</p>
  <div class="lib-entry-output"><strong>Output:</strong> smoothed relativities table with REML-selected smoothing parameter, Bayesian confidence intervals, 1D or 2D support</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-multilevel" target="_blank">insurance-multilevel</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You have broker, scheme, or fleet as a grouping factor with hundreds of levels — some with thousands of policies, some with ten — and you need credibility-weighted group adjustments rather than either ignoring the grouping or overfitting to it.</p>
  <div class="lib-entry-output"><strong>Output:</strong> CatBoost GBM with REML random effects for group factors, Bühlmann-Straub credibility weights, ICC diagnostics per group</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-frequency-severity" target="_blank">insurance-frequency-severity</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Your frequency and severity models are fitted independently, but high-frequency risks also tend to have higher severity — you are ignoring a dependence structure that is inflating the premium in some segments and deflating it in others.</p>
  <div class="lib-entry-output"><strong>Output:</strong> joint Sarmanov copula model with GLM marginals, analytical premium correction for frequency-severity dependence, IFM estimation</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-severity" target="_blank">insurance-severity</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your severity model uses a single Gamma GLM across all loss sizes, but the body and tail of your loss distribution follow completely different physics — attritional claims behave like Gamma, large losses follow a Pareto tail, and one model cannot fit both.</p>
  <div class="lib-entry-output"><strong>Output:</strong> spliced severity model with covariate-dependent threshold, separate body and tail submodels, ILF tables, TVaR per risk</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-quantile" target="_blank">insurance-quantile</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You need to price for large loss loading or produce an increased limits factor table, but your mean model gives you no handle on the upper tail — you need quantile estimates at the 90th and 99th percentile per risk, not just the expected value.</p>
  <div class="lib-entry-output"><strong>Output:</strong> quantile GBM predictions at multiple levels per risk, TVaR, ILF tables, exceedance probability curves</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-thin-data" target="_blank">insurance-thin-data</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">A segment has 200 policies and your GLM estimates are unstable — the standard errors are wide, the coefficient on vehicle age has flipped sign, and a Bayesian approach requires you to specify priors you do not have strong views on.</p>
  <div class="lib-entry-output"><strong>Output:</strong> TabPFN or TabICLv2 predictions with GLM benchmark comparison, PDP relativities in standard factor table format, CommitteeReport</div>
</div>

</div>
</div>

---

## Stage 3: Specialised techniques

<div class="guide-stage">
<div class="guide-stage-header">
  <div class="guide-stage-num">3</div>
  <h2 class="guide-stage-title">Specialised techniques</h2>
</div>
<p class="guide-stage-desc">Territory pricing, telematics, causal inference, experience rating, and credibility — problems that need more than a standard GBM.</p>

<div class="lib-entries">

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-spatial" target="_blank">insurance-spatial</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Your territory factors are eye-balled from a heat map or computed naively from raw one-ways — thin postcode cells get volatile relativities, adjacent postcodes have no relationship to each other, and the factors do not borrow strength from neighbouring areas.</p>
  <div class="lib-entry-output"><strong>Output:</strong> BYM2 spatial relativities per postcode with credibility-based smoothing from neighbours, Moran's I spatial autocorrelation diagnostics, PyMC 5 posteriors</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/bayesian-pricing" target="_blank">bayesian-pricing</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">You have a specialist segment — classic car, high-net-worth, niche occupation — with enough data to price but not enough for a standard GLM to separate signal from noise, and you want to incorporate prior actuarial knowledge formally rather than ad hoc.</p>
  <div class="lib-entry-output"><strong>Output:</strong> PyMC 5 hierarchical Bayesian model with partial pooling, credibility factor output in standard actuarial format, posterior predictive distributions</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Your vehicle value factor looks significant in the GLM, but vehicle value correlates strongly with distribution channel — direct customers buy cheaper cars — and you cannot tell whether it is genuine risk signal or channel confounding.</p>
  <div class="lib-entry-output"><strong>Output:</strong> deconfounded coefficient estimates via double machine learning, confounding bias report, CatBoost nuisance models</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-causal-policy" target="_blank">insurance-causal-policy</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">You put through a rate increase in Q3 and conversion dropped — but you cannot tell how much of that drop was the rate change versus market conditions, because you have no control group and the pre/post comparison is confounded by seasonal effects.</p>
  <div class="lib-entry-output"><strong>Output:</strong> causal rate change impact estimate via synthetic difference-in-differences, event study chart, HonestDiD sensitivity, FCA evidence pack</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-telematics" target="_blank">insurance-telematics</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">You have raw 1Hz GPS and accelerometer data from a UBI product and need to turn it into a GLM-compatible risk score — but you have no principled method for aggregating trip-level events into driver-level risk that accounts for the mix of journey types.</p>
  <div class="lib-entry-output"><strong>Output:</strong> HMM-classified driving states per trip, Bühlmann-Straub aggregated driver risk score, Poisson GLM-compatible risk variable</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-dynamics" target="_blank">insurance-dynamics</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">Claims frequency has been drifting upward for eighteen months and you need to know whether it is a sustained trend, a structural break, or noise — and you need the answer expressed as a trend index in development factor format for the reserving team.</p>
  <div class="lib-entry-output"><strong>Output:</strong> GAS score-driven trend index per rating cell, BOCPD/PELT changepoint detection, trend factor in development factor format</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-credibility" target="_blank">insurance-credibility</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your NCD schedule is fixed at five years old and the discount scales were set judgementally — you want to fit actuarially correct credibility weights from your own data and incorporate dynamic claim history into an individual policy posterior. Experience rating is available via <code>insurance_credibility.experience</code>.</p>
  <div class="lib-entry-output"><strong>Output:</strong> credibility-weighted NCD factors, Bühlmann-Straub static or dynamic state-space individual policy posterior, deep attention credibility model</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-survival" target="_blank">insurance-survival</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You want to model lapse risk and customer lifetime value, but standard logistic regression ignores the time structure — policies renewing at different durations have different lapse hazards, and a cross-sectional model is missing that entirely.</p>
  <div class="lib-entry-output"><strong>Output:</strong> survival model with cure rate, CLV estimates per policy, lapse table by duration, MLflow-wrapped for production</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></span>
    <span class="lib-complexity complexity-advanced">Advanced</span>
  </div>
  <p class="lib-entry-problem">You need a price elasticity estimate for rate change planning, but conversion data is observational — the prices customers saw were set by a model, not randomly assigned — and a naive regression of conversion on price is picking up selection effects. Elasticity estimation is available via <code>insurance_causal.elasticity</code>.</p>
  <div class="lib-entry-output"><strong>Output:</strong> causal price elasticity estimates per segment via CausalForestDML or DR-Learner, ENBP-constrained rate optimiser, regulatory audit trail</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-optimise" target="_blank">insurance-optimise</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your technical price model and your commercial rate are disconnected — the pricing committee sets rate change by segment without knowing the conversion and retention response, and FCA GIPP requires you to consider demand effects formally. Demand modelling is available via <code>insurance_optimise.demand</code>.</p>
  <div class="lib-entry-output"><strong>Output:</strong> demand model linking conversion and retention to price, price response curves per segment, FCA GIPP-compliant rate optimisation output</div>
</div>

</div>
</div>

---

## Stage 4: Fairness &amp; compliance

<div class="guide-stage">
<div class="guide-stage-header">
  <div class="guide-stage-num">4</div>
  <h2 class="guide-stage-title">Fairness &amp; compliance</h2>
</div>
<p class="guide-stage-desc">Proxy discrimination auditing, Consumer Duty documentation, and PRA model governance — what the FCA and PRA can ask for, answered before they ask.</p>

<div class="lib-entries">

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">Your pricing actuary says postcode is a legitimate risk variable, but the compliance team is worried it correlates with ethnicity — and under Consumer Duty you need to quantify the indirect discrimination risk, not just assert there is none.</p>
  <div class="lib-entry-output"><strong>Output:</strong> Consumer Duty-aligned proxy discrimination audit, disparate impact metrics, fairness-accuracy trade-off analysis, Consumer Duty evidence pack</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your model governance committee needs a validation report and you are producing it manually in PowerPoint — bootstrap Gini CI, Poisson A/E CI, double-lift charts, and a renewal cohort test, all formatted to what a model risk function expects.</p>
  <div class="lib-entry-output"><strong>Output:</strong> PRA SS1/23-compliant validation report in HTML and JSON, ModelCard, ModelInventory, GovernanceReport, risk tier scoring</div>
</div>

</div>
</div>

---

## Stage 5: Deployment &amp; monitoring

<div class="guide-stage">
<div class="guide-stage-header">
  <div class="guide-stage-num">5</div>
  <h2 class="guide-stage-title">Deployment &amp; monitoring</h2>
</div>
<p class="guide-stage-desc">Getting a model to production and keeping it honest once it is there.</p>

<div class="lib-entries">

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-deploy" target="_blank">insurance-deploy</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You want to run a new model in shadow mode before cutting over, but you have no infrastructure for routing quotes deterministically, logging both prices, and eventually running a statistically valid comparison to declare a winner.</p>
  <div class="lib-entry-output"><strong>Output:</strong> champion/challenger routing framework with SHA-256 deterministic routing, SQLite quote log, bootstrap likelihood ratio test, ICOBS 6B.2 audit trail</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your model has been live for nine months and A/E ratios are creeping up — but you do not know whether to recalibrate the intercept, refit with new data, or whether it is just IBNR noise in the most recent quarter.</p>
  <div class="lib-entry-output"><strong>Output:</strong> exposure-weighted PSI/CSI, segmented A/E ratios with IBNR adjustment, Gini z-test with a formal recalibrate-vs-refit decision rule</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a></span>
    <span class="lib-complexity complexity-drop-in">Drop-in</span>
  </div>
  <p class="lib-entry-problem">Your GBM gives a point estimate per risk but Solvency II internal models need uncertainty bounds — and the standard bootstrap approach is slow and the distributional assumptions are wrong for insurance losses.</p>
  <div class="lib-entry-output"><strong>Output:</strong> distribution-free prediction intervals with finite-sample coverage guarantee, locally-weighted Pearson residual non-conformity scores, Solvency II SCR bounds</div>
</div>

<div class="lib-entry">
  <div class="lib-entry-top">
    <span class="lib-entry-name"><a href="https://github.com/burning-cost/insurance-optimise" target="_blank">insurance-optimise</a></span>
    <span class="lib-complexity complexity-setup">Needs setup</span>
  </div>
  <p class="lib-entry-problem">You have a technical price per segment, a loss ratio target, and maximum movement caps — and the rate change recommendation is currently done in a spreadsheet where the constraints interact and the solution is not optimal.</p>
  <div class="lib-entry-output"><strong>Output:</strong> SLSQP-optimised rate changes per segment, efficient frontier between loss ratio improvement and movement constraints, JSON audit trail for FCA ENBP</div>
</div>

</div>
</div>

---

<div class="gs-next" style="margin-top:2.5rem;padding-top:2rem;border-top:1px solid #e2e6f0;">
  <h2 style="font-size:1.1rem;font-weight:700;color:#0f1117;margin-bottom:1rem;">Still not sure where to start?</h2>
  <div class="gs-cta-row" style="display:flex;gap:0.75rem;flex-wrap:wrap;">
    <a href="/getting-started/" class="btn-primary-sm" style="display:inline-block;padding:0.55rem 1.1rem;border-radius:8px;font-size:0.875rem;font-weight:600;text-decoration:none;background:#4f7ef8;color:#fff;border:1px solid #4f7ef8;">Getting started paths</a>
    <a href="/tools/" class="btn-outline-sm" style="display:inline-block;padding:0.55rem 1.1rem;border-radius:8px;font-size:0.875rem;font-weight:600;text-decoration:none;background:#fff;color:#4f7ef8;border:1px solid #c6d4f9;">Browse all libraries</a>
    <a href="https://github.com/burning-cost" target="_blank" class="btn-outline-sm" style="display:inline-block;padding:0.55rem 1.1rem;border-radius:8px;font-size:0.875rem;font-weight:600;text-decoration:none;background:#fff;color:#4f7ef8;border:1px solid #c6d4f9;">View on GitHub</a>
  </div>
</div>
