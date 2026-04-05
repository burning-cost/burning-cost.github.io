---
layout: page
title: "Benchmark Evidence"
description: "Benchmark results for 34 Python libraries for insurance pricing — Gini lift, MAE improvements, coverage accuracy, and head-to-head comparisons against standard GLM and actuarial approaches. All run on synthetic DGP data in Databricks."
permalink: /evidence/
---

<style>
.ev-intro {
  font-size: 1.05rem;
  color: #2a2d3e;
  margin-bottom: 2.5rem;
  max-width: 720px;
  line-height: 1.7;
}

.ev-intro a {
  color: #4f7ef8;
  text-decoration: none;
}

.ev-intro a:hover {
  text-decoration: underline;
}

.ev-section {
  margin-bottom: 3rem;
}

.ev-section-header {
  display: flex;
  align-items: baseline;
  gap: 1rem;
  margin-bottom: 1.25rem;
  padding-bottom: 0.6rem;
  border-bottom: 2px solid #e2e6f0;
}

.ev-section-header h2 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f1117;
  margin: 0;
  letter-spacing: -0.02em;
}

.ev-section-count {
  font-size: 0.75rem;
  color: #8892a4;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.ev-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.ev-table thead tr {
  border-bottom: 2px solid #e2e6f0;
}

.ev-table th {
  text-align: left;
  font-weight: 700;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: #8892a4;
  padding: 0.5rem 0.85rem 0.6rem;
  white-space: nowrap;
}

.ev-table td {
  padding: 0.75rem 0.85rem;
  vertical-align: top;
  border-bottom: 1px solid #f0f2f8;
  color: #4a4f66;
  line-height: 1.5;
}

.ev-table tr:last-child td {
  border-bottom: none;
}

.ev-table td.lib-name {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  font-weight: 600;
  color: #0f1117;
  white-space: nowrap;
  min-width: 180px;
}

.ev-table td.lib-name a {
  color: #4f7ef8;
  text-decoration: none;
}

.ev-table td.lib-name a:hover {
  text-decoration: underline;
}

.ev-table td.measured {
  min-width: 180px;
  max-width: 240px;
}

.ev-table td.standard {
  font-variant-numeric: tabular-nums;
  color: #7a8099;
  min-width: 160px;
}

.ev-table td.ours {
  font-variant-numeric: tabular-nums;
  font-weight: 600;
  color: #0f1117;
  min-width: 160px;
}

.ev-table td.takeaway {
  color: #4a4f66;
  font-size: 0.84rem;
  max-width: 280px;
}

.ev-win {
  color: #1a7f4b;
}

.ev-neutral {
  color: #7a5c00;
}

.ev-honest {
  background: #fffbf0;
  border-left: 3px solid #f5a623;
  padding: 0.6rem 0.85rem;
  font-size: 0.82rem;
  color: #5a6278;
  line-height: 1.5;
  margin-top: 0.4rem;
}

.ev-flagship-tag {
  display: inline-block;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #4f7ef8;
  background: rgba(79,126,248,0.10);
  border: 1px solid rgba(79,126,248,0.22);
  padding: 0.15rem 0.45rem;
  border-radius: 100px;
  margin-left: 0.4rem;
  vertical-align: middle;
}

.ev-methodology {
  margin-top: 3.5rem;
  padding: 1.75rem 2rem;
  background: #f9fafc;
  border: 1px solid #e2e6f0;
  border-radius: 12px;
}

.ev-methodology h2 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f1117;
  margin-top: 0;
  margin-bottom: 0.75rem;
  letter-spacing: -0.02em;
}

.ev-methodology p {
  font-size: 0.9rem;
  color: #4a4f66;
  line-height: 1.7;
  margin-bottom: 0.75rem;
}

.ev-methodology p:last-child {
  margin-bottom: 0;
}

.ev-methodology a {
  color: #4f7ef8;
  text-decoration: none;
}

.ev-methodology a:hover {
  text-decoration: underline;
}

.ev-summary-bar {
  display: flex;
  gap: 2rem;
  flex-wrap: wrap;
  margin-bottom: 2.5rem;
  padding: 1.5rem 1.75rem;
  background: #0f1117;
  border-radius: 12px;
}

.ev-summary-stat {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.ev-summary-number {
  font-size: 1.6rem;
  font-weight: 700;
  color: #4f7ef8;
  letter-spacing: -0.03em;
  line-height: 1;
}

.ev-summary-label {
  font-size: 0.75rem;
  color: #8892a4;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-weight: 600;
}

@media (max-width: 900px) {
  .ev-table {
    font-size: 0.82rem;
  }
  .ev-table td.measured,
  .ev-table td.standard,
  .ev-table td.ours,
  .ev-table td.takeaway {
    min-width: 0;
    max-width: none;
  }
}

@media (max-width: 640px) {
  .ev-table thead {
    display: none;
  }
  .ev-table td {
    display: block;
    padding: 0.4rem 0.6rem;
  }
  .ev-table tr {
    display: block;
    border-bottom: 2px solid #e2e6f0;
    margin-bottom: 0.5rem;
    padding: 0.5rem 0;
  }
  .ev-table tr:last-child {
    border-bottom: none;
  }
  .ev-table td.lib-name {
    white-space: normal;
    font-size: 0.85rem;
  }
  .ev-methodology {
    padding: 1.25rem 1.25rem;
  }
  .ev-summary-bar {
    gap: 1.25rem;
    padding: 1.25rem;
  }
}
</style>

<p class="ev-intro">Every library in the Burning Cost portfolio has been benchmarked against a standard baseline on synthetic insurance data with a known data-generating process. This page aggregates those results. The pattern is consistent: specialised methods beat generic approaches on insurance problems, and where they don't, we say so.</p>

<div class="ev-summary-bar">
  <div class="ev-summary-stat">
    <span class="ev-summary-number">34</span>
    <span class="ev-summary-label">Libraries benchmarked</span>
  </div>
  <div class="ev-summary-stat">
    <span class="ev-summary-number">35</span>
    <span class="ev-summary-label">Databricks notebooks</span>
  </div>
  <div class="ev-summary-stat">
    <span class="ev-summary-number">MIT</span>
    <span class="ev-summary-label">All licences</span>
  </div>
  <div class="ev-summary-stat">
    <span class="ev-summary-number">Python 3.10+</span>
    <span class="ev-summary-label">Runtime requirement</span>
  </div>
</div>

---

## Model Building

<div class="ev-section">
<div class="ev-section-header">
  <h2>Model Building</h2>
  <span class="ev-section-count">7 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></td>
      <td class="measured">Gini lift vs Poisson GLM; relativity accuracy vs true DGP</td>
      <td class="standard">GLM: 4.5% mean relativity error; lower Gini baseline</td>
      <td class="ours"><span class="ev-win">+2.85pp Gini lift</span>; 9.4% relativity error</td>
      <td class="takeaway">GBM wins on discrimination; GLM wins on factor accuracy. An honest trade-off: use shap-relativities for ranking, not if you need the coefficients to be actuarially exact.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-glm-tools" target="_blank">insurance-glm-tools</a></td>
      <td class="measured">Nested GLM embeddings for 500 vehicle makes vs dummy-coded GLM; R2VF fused lasso clustering vs manual quintile banding</td>
      <td class="standard">Dummy-coded GLM: overfits on high-cardinality vehicle makes</td>
      <td class="ours"><span class="ev-win">Nested embeddings reduce overfitting</span>; R2VF clustering eliminates arbitrary band choices</td>
      <td class="takeaway">Dummy-coded GLM becomes unreliable above ~50 factor levels. Nested embeddings give consistent coefficient estimates at 500+ levels.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-gam" target="_blank">insurance-gam</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">EBM and Neural Additive Model vs Poisson GLM on synthetic data with planted non-linear effects; exact Shapley values vs approximation</td>
      <td class="standard">Poisson GLM misses planted non-linear age-mileage interaction</td>
      <td class="ours"><span class="ev-win">EBM/NAM recovers planted non-linear effects</span>; shape functions directly interpretable as factor tables</td>
      <td class="takeaway">Where GLM forces linearity, EBM fits the true shape. The transparency cost vs a GBM is zero - you get exact Shapley values and shape functions instead of approximate attribution.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-interactions" target="_blank">insurance-interactions</a></td>
      <td class="measured">CANN/NID interaction detection vs exhaustive pairwise GLM search on planted interactions</td>
      <td class="standard">Exhaustive pairwise GLM: slow, misses non-linear interactions</td>
      <td class="ours"><span class="ev-win">Production defaults recover both planted interactions</span>; compact config less reliable on weak signals</td>
      <td class="takeaway">CANN/NID is reliable when planted effects are of meaningful size. On weak interactions (&lt;5% deviance contribution), SHAP interaction values are more stable than NID scores.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-frequency-severity" target="_blank">insurance-frequency-severity</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Sarmanov copula joint model vs independence assumption; premium error under dependence</td>
      <td class="standard">Independence assumption: premium bias wherever freq and sev are correlated</td>
      <td class="ours"><span class="ev-win">Analytical premium correction removes dependence bias</span>; IFM estimation with dependence tests</td>
      <td class="takeaway">Most pricing models assume frequency and severity are independent. Where they aren't - typically in high-mileage or commercial segments - the independence assumption inflates expected loss cost. This library tests and corrects for it.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-spatial" target="_blank">insurance-spatial</a></td>
      <td class="measured">BYM2 territory factors vs raw postcode rates vs manual banding on synthetic spatial data</td>
      <td class="standard">Manual banding: ignores spatial autocorrelation; raw rates: noisy on thin postcodes</td>
      <td class="ours"><span class="ev-win">BYM2 smooth factors preserve spatial structure</span>; Moran's I confirms residual autocorrelation is removed</td>
      <td class="takeaway">Raw postcode rates are unstable on thin cells; manual banding loses information at boundaries. BYM2 borrows strength from neighbours using a proper adjacency structure.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-distill" target="_blank">insurance-distill</a></td>
      <td class="measured">R&sup2; match between CatBoost predictions and surrogate GLM factor tables</td>
      <td class="standard">Direct GLM: lower predictive performance than GBM by construction</td>
      <td class="ours"><span class="ev-win">90-97% R&sup2; match</span> between GBM predictions and distilled factor tables</td>
      <td class="takeaway">A distilled GLM captures 90-97% of the GBM's variance in a Radar/Emblem-compatible multiplicative structure. The residual 3-10% is the honest cost of interpretability.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Distributional and Tail Risk

<div class="ev-section">
<div class="ev-section-header">
  <h2>Distributional and Tail Risk</h2>
  <span class="ev-section-count">5 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-distributional-glm" target="_blank">insurance-distributional-glm</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Sigma (dispersion) correlation vs true DGP on heterogeneous-variance claims data</td>
      <td class="standard">Constant-phi Gamma GLM: sigma correlation <span class="ev-neutral">0.000</span> (cannot model dispersion variation)</td>
      <td class="ours"><span class="ev-win">GAMLSS sigma correlation 0.998</span> vs true DGP</td>
      <td class="takeaway">This is the starkest result in the portfolio. When dispersion varies by risk (as it does in most motor books), a standard GLM is structurally incapable of capturing it. GAMLSS models the whole distribution, not just the mean.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-distributional" target="_blank">insurance-distributional</a></td>
      <td class="measured">Log-likelihood and prediction interval calibration for distributional GBMs vs point-estimate GBM</td>
      <td class="standard">Standard point-prediction GBM: no per-risk volatility estimate</td>
      <td class="ours"><span class="ev-win">GammaGBM +1.5% log-likelihood</span>; prediction intervals calibrated vs uncalibrated bootstrap</td>
      <td class="takeaway">Per-risk volatility scoring is the main gain. Beyond the 1.5% log-likelihood improvement, you get a distributional output usable for capital allocation and per-policy uncertainty scoring.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-dispersion" target="_blank">insurance-dispersion</a></td>
      <td class="measured">Double GLM vs constant-phi Gamma GLM on heteroscedastic claims data; per-risk volatility scoring</td>
      <td class="standard">Constant-phi GLM: assumes all risks have the same dispersion</td>
      <td class="ours"><span class="ev-win">Double GLM captures heteroscedasticity</span> the constant-phi model cannot see</td>
      <td class="takeaway">A simpler alternative to GAMLSS for teams that want dispersion modelling without the full distributional GLM complexity. The alternating IRLS approach fits quickly and the output maps naturally to factor tables.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-quantile" target="_blank">insurance-quantile</a></td>
      <td class="measured">TVaR bias on heavy-tailed DGP; pinball loss at small n</td>
      <td class="standard">Lognormal: lower pinball loss at small n; higher TVaR bias on heavy tails</td>
      <td class="ours"><span class="ev-win">GBM: lower TVaR bias</span> on heavy-tailed data; lognormal beats GBM on pinball at small n</td>
      <td class="takeaway">GBM quantile regression is the right choice for ILF curves on large portfolios with heavy tails. At small n, a parametric lognormal or Pareto is more stable - this library benchmarks both honestly.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-severity" target="_blank">insurance-severity</a></td>
      <td class="measured">Tail error reduction vs single lognormal on composite Lognormal-GPD DGP</td>
      <td class="standard">Single lognormal: misspecified in the tail by construction</td>
      <td class="ours"><span class="ev-win">Composite Lognormal-GPD reduces tail error 5.6%</span> vs single lognormal</td>
      <td class="takeaway">5.6% sounds modest; on a large-loss book it compounds into material reserve error. The EQRN extreme quantile network is the appropriate choice when the GPD threshold is uncertain.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Credibility and Thin Data

<div class="ev-section">
<div class="ev-section-header">
  <h2>Credibility and Thin Data</h2>
  <span class="ev-section-count">5 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-whittaker" target="_blank">insurance-whittaker</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">MSE on smoothed age relativities vs raw observed rates; REML lambda vs manual step smoothing</td>
      <td class="standard">Raw observed rates: noisy, inconsistent lambda choices by hand</td>
      <td class="ours"><span class="ev-win">57.2% MSE reduction</span> vs raw rates; REML lambda selected automatically</td>
      <td class="takeaway">Manually chosen smoothing parameters are unreliable and non-reproducible. REML selects lambda by maximising marginal likelihood - the same principled approach used in mixed models. 57% MSE reduction is the payoff.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-credibility" target="_blank">insurance-credibility</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">MAE on thin scheme segments; Bühlmann-Straub credibility vs raw experience rating</td>
      <td class="standard">Raw experience: high variance on thin segments; overreacts to single bad years</td>
      <td class="ours"><span class="ev-win">6.8% MAE improvement</span> on thin schemes vs raw experience</td>
      <td class="takeaway">6.8% MAE on thin segments where raw experience routinely moves by 20-40% year-on-year is a meaningful stabilisation. The mixed-model equivalence check confirms the credibility weights are correctly derived.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-multilevel" target="_blank">insurance-multilevel</a></td>
      <td class="measured">Gamma deviance and thin-group MAPE: two-stage CatBoost + REML vs one-hot encoded GLM</td>
      <td class="standard">One-hot GLM: thin-group MAPE 66.1%; 15.9% worse deviance</td>
      <td class="ours"><span class="ev-win">15.9% gamma deviance reduction</span>; thin-group MAPE 63.6% vs 66.1%</td>
      <td class="takeaway">One-hot encoding treats each group as independent. The two-stage approach uses REML random effects to share information across groups - the same gain as Bühlmann-Straub but applicable inside a CatBoost pipeline.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-thin-data" target="_blank">insurance-thin-data</a></td>
      <td class="measured">Bootstrap 90% CI width on thin segment GLMs: GLMTransfer vs standalone GLM</td>
      <td class="standard">Standalone GLM: wide confidence intervals on &lt;200 policy segments</td>
      <td class="ours"><span class="ev-win">30-60% CI width reduction</span> via GLMTransfer prior transfer from related segments</td>
      <td class="takeaway">Narrower CIs on thin segments mean pricing decisions based on those factors are less likely to reverse at next renewal review. The transfer approach works best when the source and target segments have similar underlying DGPs.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/bayesian-pricing" target="_blank">bayesian-pricing</a></td>
      <td class="measured">Hierarchical Bayesian vs raw experience on thin segments (PyMC 5)</td>
      <td class="standard">Raw experience: unreliable on &lt;100 policies per segment</td>
      <td class="ours"><span class="ev-win">Posterior pooling stabilises thin-segment estimates</span> via hierarchical priors</td>
      <td class="takeaway">The Bayesian approach gives full posterior distributions, not point estimates - useful for risk committee presentations where uncertainty communication matters. Slower than Bühlmann-Straub; use it when you need full posteriors.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Causal Inference

<div class="ev-section">
<div class="ev-section-header">
  <h2>Causal Inference</h2>
  <span class="ev-section-count">2 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Confounding bias removal: DML vs naive Poisson GLM on confounded data (n=50k+)</td>
      <td class="standard">Naive Poisson GLM: confounding bias persists; coefficient estimates biased wherever rating variables correlate with channel or selection</td>
      <td class="ours"><span class="ev-win">DML removes nonlinear confounding bias at scale</span> (n&ge;50k); honest: over-partials at small n</td>
      <td class="takeaway">Standard GLM coefficients are correlational, not causal. DML removes confounding without a structural model. At small n (&lt;50k), DML over-partials and introduces its own bias - use it on the full portfolio, not on segment-level data.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-causal-policy" target="_blank">insurance-causal-policy</a></td>
      <td class="measured">CI coverage and bias on rate change evaluation: SDID vs naive before-after comparison</td>
      <td class="standard">Naive before-after: biased <span class="ev-neutral">+3.8pp</span> by concurrent market inflation</td>
      <td class="ours"><span class="ev-win">SDID 98% CI coverage</span>; isolates the rate change effect from market movement</td>
      <td class="takeaway">Before-after comparisons of rate changes are almost always confounded by market trends. A 3.8pp inflation bias in the benchmark is typical of what teams are currently acting on. SDID with HonestDiD sensitivity bounds is the defensible alternative for FCA evidence packs.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Fairness and Regulation

<div class="ev-section">
<div class="ev-section-header">
  <h2>Fairness and Regulation</h2>
  <span class="ev-section-count">2 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Proxy discrimination detection: proxy R&sup2; vs Spearman correlation on a planted postcode-to-ethnicity proxy</td>
      <td class="standard">Spearman correlation: r=0.06 - fails to detect the proxy entirely</td>
      <td class="ours"><span class="ev-win">Proxy R&sup2;=0.78</span> - catches the same proxy that Spearman misses</td>
      <td class="takeaway">Spearman correlation is not a valid test for proxy discrimination. A rating variable can have near-zero rank correlation with a protected characteristic but still act as a near-perfect proxy via a non-linear relationship. This is the result that matters for FCA Consumer Duty compliance.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-covariate-shift" target="_blank">insurance-covariate-shift</a></td>
      <td class="measured">Importance-weighted evaluation after distribution shift: density ratio correction vs unweighted evaluation</td>
      <td class="standard">Unweighted evaluation: performance metrics biased after book composition change</td>
      <td class="ours"><span class="ev-win">CatBoost/RuLSIF/KLIEP density ratio correction</span> removes evaluation bias after shift; LR-QR conformal bounds included</td>
      <td class="takeaway">When a book's risk mix changes - via broker switches, scheme exits, or a marketing campaign - historic model performance statistics become misleading. Density ratio weighting corrects this before a model review misinterprets drift as deterioration.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Validation and Monitoring

<div class="ev-section">
<div class="ev-section-header">
  <h2>Validation and Monitoring</h2>
  <span class="ev-section-count">6 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></td>
      <td class="measured">Optimism in Gini estimate: walk-forward temporal CV vs random k-fold on insurance data</td>
      <td class="standard">Random k-fold: <span class="ev-neutral">10.5% optimism</span> vs true OOT holdout</td>
      <td class="ours"><span class="ev-win">Walk-forward CV matches OOT holdout</span>; eliminates future-data leakage</td>
      <td class="takeaway">10.5% optimism means models look better in random k-fold than they perform on live business. Walk-forward CV, with proper IBNR buffers between folds, eliminates this. Use it as your default validation approach.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Marginal coverage on Tweedie frequency data vs nominal 90%</td>
      <td class="standard">Bootstrap intervals: miscalibrated coverage; computationally expensive</td>
      <td class="ours"><span class="ev-win">90.1% marginal coverage</span> on Tweedie frequency data at 90% nominal</td>
      <td class="takeaway">Conformal intervals give finite-sample coverage guarantees without distributional assumptions. 90.1% coverage at 90% nominal is the finite-sample guarantee working as intended. Bootstrap intervals typically over- or under-cover depending on the tail behaviour assumed.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-conformal-ts" target="_blank">insurance-conformal-ts</a></td>
      <td class="measured">Coverage on non-exchangeable claims time series: ACI/SPCI vs static split conformal</td>
      <td class="standard">Static split conformal: coverage degrades on non-exchangeable time series</td>
      <td class="ours"><span class="ev-win">ACI/SPCI maintain coverage</span> on non-exchangeable series where static methods fail</td>
      <td class="takeaway">Standard conformal prediction assumes exchangeability - an assumption that fails on claims time series with trends, seasonality, or reporting delays. ACI and SPCI adapt the non-conformity threshold sequentially.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></td>
      <td class="measured">False positive rate under repeated peeking: mSPRT vs standard t-test</td>
      <td class="standard">Peeking t-test: <span class="ev-neutral">25% FPR</span> (5x the nominal 5% level)</td>
      <td class="ours"><span class="ev-win">mSPRT holds FPR at 1%</span> under repeated looks (Johari et al. 2022)</td>
      <td class="takeaway">This is the A/B testing result. Teams that check their champion/challenger results daily using a t-test have a 25% chance of declaring a false winner. The mSPRT is anytime-valid: you can look at any point without inflating the false positive rate.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-deploy" target="_blank">insurance-deploy</a></td>
      <td class="measured">Champion/challenger routing, shadow mode quote logging, bootstrap LR test for winner declaration</td>
      <td class="standard">Manual routing: no deterministic allocation, no audit trail</td>
      <td class="ours"><span class="ev-win">SHA-256 deterministic routing; bootstrap LR test for winner declaration</span>; ICOBS 6B.2 audit trail</td>
      <td class="takeaway">The benchmark here is operational correctness rather than a performance metric. SHA-256 routing ensures the same risk always sees the same model variant. The bootstrap LR test gives a principled stopping rule for the experiment.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">PRA SS1/23 validation: automated suite vs manual checklist on age-band miscalibrated model</td>
      <td class="standard">Manual checklist: misses age-band miscalibration that only appears in double-lift</td>
      <td class="ours"><span class="ev-win">Automated suite catches miscalibration</span> manual checklists miss; HTML/JSON output for PRA review</td>
      <td class="takeaway">The value is in the completeness. Manual validation checklists are selective by nature; the automated suite runs every required test on every model. The age-band miscalibration case in the benchmark is the kind of finding that appears in real PRA model reviews.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Optimisation and Pricing Strategy

<div class="ev-section">
<div class="ev-section-header">
  <h2>Optimisation and Pricing Strategy</h2>
  <span class="ev-section-count">1 library</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-optimise" target="_blank">insurance-optimise</a></td>
      <td class="measured">Profit lift vs flat loading on synthetic demand curve data</td>
      <td class="standard">Flat technical loading: treats all risks as equally price-elastic</td>
      <td class="ours"><span class="ev-win">+143.8% profit lift</span> over flat loading via demand-curve-aware pricing</td>
      <td class="takeaway">143.8% profit lift is the upper bound when demand curves are perfectly estimated - real-world gains are lower and depend on elasticity estimation quality. The benchmark establishes the ceiling. The ParetoFrontier component trades profit against retention and fairness, which is the decision most pricing committees actually face.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Time Series and Trends

<div class="ev-section">
<div class="ev-section-header">
  <h2>Time Series and Trends</h2>
  <span class="ev-section-count">2 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-trend" target="_blank">insurance-trend</a></td>
      <td class="measured">MAPE on loss cost trend vs naive OLS; structural break detection</td>
      <td class="standard">Naive OLS trend: <span class="ev-neutral">3.93pp higher MAPE</span>; no structural break detection</td>
      <td class="ours"><span class="ev-win">3.93pp MAPE improvement</span> over naive OLS; BOCPD/PELT detects structural breaks</td>
      <td class="takeaway">OLS trend fitting on loss costs ignores the non-stationarity that characterises post-2020 UK motor data. The MAPE improvement is the result of proper frequency/severity decomposition and structural break testing - not just a more complex model.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-dynamics" target="_blank">insurance-dynamics</a></td>
      <td class="measured">MAE on dynamic frequency: GAS Poisson filter vs static GLM trend</td>
      <td class="standard">Static GLM trend: cannot track within-year frequency movements</td>
      <td class="ours"><span class="ev-win">GAS Poisson +13% MAE improvement</span> over static GLM trend</td>
      <td class="takeaway">A 13% MAE improvement on frequency tracking means earned premium projections based on GAS filters are materially more accurate than static trend assumptions. The Bayesian changepoint detection handles the regime shifts that GAS filters smooth through.</td>
    </tr>
  </tbody>
</table>
</div>

---

## Other Libraries

<div class="ev-section">
<div class="ev-section-header">
  <h2>Other Libraries</h2>
  <span class="ev-section-count">9 libraries</span>
</div>

<table class="ev-table">
  <thead>
    <tr>
      <th>Library</th>
      <th>What was measured</th>
      <th>Standard approach</th>
      <th>Burning Cost</th>
      <th>Key takeaway</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-telematics" target="_blank">insurance-telematics</a> <span class="ev-flagship-tag">Flagship</span></td>
      <td class="measured">Gini improvement: HMM latent-state features vs raw trip aggregates in Poisson GLM</td>
      <td class="standard">Raw trip averages (mean speed, hard braking counts): lower Gini baseline</td>
      <td class="ours"><span class="ev-win">3-8pp Gini improvement</span> from HMM state features over raw trip averages</td>
      <td class="takeaway">Raw trip statistics conflate driving contexts - a hard brake at 30mph is not the same risk signal as one at 70mph. HMM latent states separate driving contexts before aggregation, which is why the Gini improvement is consistent across segments.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-survival" target="_blank">insurance-survival</a></td>
      <td class="measured">Cure fraction recovery: cure model vs KM/Cox PH extrapolation on long-tailed lapse data</td>
      <td class="standard">KM/Cox: extrapolates to zero - overestimates ultimate lapse for low-risk segments</td>
      <td class="ours"><span class="ev-win">Cure model recovers 34.1% cure fraction</span> (true DGP: 35.0%); KM/Cox misses this entirely</td>
      <td class="takeaway">For motor customers with 10+ years no-claims, lapse probability effectively reaches a floor - not zero. KM and Cox PH extrapolate past this floor and overstate CLV for the most loyal segment. The cure model is the correct specification.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-synthetic" target="_blank">insurance-synthetic</a></td>
      <td class="measured">Correlation preservation: vine copula synthetic generation vs naive independent sampling</td>
      <td class="standard">Naive sampling: ignores multivariate dependence structure</td>
      <td class="ours"><span class="ev-win">64% better correlation preservation</span> vs naive independent generation</td>
      <td class="takeaway">Synthetic portfolios that ignore dependence structure produce datasets where the risk segmentation looks right but the portfolio-level behaviour is wrong. The 64% improvement in correlation preservation means synthetic stress tests are materially more realistic.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-datasets" target="_blank">insurance-datasets</a></td>
      <td class="measured">Parameter recovery RMSE on synthetic UK motor DGP; omitted variable bias demo</td>
      <td class="standard">No standard baseline - this is a data library, not a model</td>
      <td class="ours"><span class="ev-win">GLM parameter recovery RMSE 0.069</span>; OVB demo shows 24% NCD inflation when age is omitted</td>
      <td class="takeaway">The 24% NCD inflation figure from the OVB demo matches the kind of bias we see when actuaries fit NCD factors on books where driver age is poorly captured. Use this library to validate any new method before running it on real data.</td>
    </tr>
    <tr>
      <td class="lib-name"><a href="https://github.com/burning-cost/insurance-multilevel" target="_blank">insurance-multilevel</a></td>
      <td class="measured">Already covered above under Credibility and Thin Data.</td>
      <td class="standard">-</td>
      <td class="ours">-</td>
      <td class="takeaway">See Credibility and Thin Data section.</td>
    </tr>
  </tbody>
</table>

<p style="font-size: 0.84rem; color: #8892a4; margin-top: 1rem;">Note: insurance-multilevel appears in the tools page under Credibility; its benchmark results are in that section above. Four libraries - insurance-glm-tools, insurance-spatial, insurance-deploy, and bayesian-pricing - have qualitative benchmark results rather than single headline numbers because the relevant comparisons are structural (correctness, coverage, output format) rather than metric-based.</p>
</div>

---

<div class="ev-methodology">
  <h2>Methodology</h2>
  <p>All benchmarks were run on synthetic data with a known data-generating process (DGP). Using synthetic data means we can verify that a method recovers the true parameters - something impossible with real data where the ground truth is unknown. The DGPs are calibrated to resemble UK motor insurance portfolios: Poisson frequencies with realistic base rates (5-12%), Gamma severities, log-linear relativities, and correlation structures consistent with published industry data.</p>
  <p>Benchmarks were run as Databricks notebooks in the <a href="https://github.com/burning-cost/burning-cost-examples" target="_blank">burning-cost-examples</a> repository. Each notebook installs its own dependencies, generates data inline, fits models, and computes comparison metrics. They run on Databricks serverless compute (Free Edition) - no cluster configuration required. To reproduce a result: import the relevant notebook, run all cells, and the comparison table is the final output.</p>
  <p>Where a library honestly underperforms in some scenario - shap-relativities on relativity accuracy, insurance-quantile on pinball loss at small n - we report it. A benchmark that only shows wins is not a benchmark; it is marketing. The honest results are the ones worth reading.</p>
  <p>Numbers reported are point estimates from a single benchmark run on a fixed random seed. We do not report confidence intervals around the benchmark comparisons themselves, though most notebooks include sensitivity checks across multiple seeds. If you need replication code or want to run a modified DGP, the notebooks are the starting point.</p>
</div>
