---
layout: page
title: "Getting Started"
description: "Where to begin with Burning Cost depends on where you're coming from. Three entry paths: pricing actuary, data scientist, and technical team lead."
permalink: /getting-started/
---

<style>
.gs-intro {
  font-size: 1.05rem;
  color: #2a2d3e;
  margin-bottom: 2.5rem;
  max-width: 680px;
  line-height: 1.7;
}

.gs-path {
  border: 1px solid #e2e6f0;
  border-radius: 12px;
  padding: 1.75rem 2rem;
  margin-bottom: 2rem;
  background: #f9fafc;
}

.gs-path-label {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #4f7ef8;
  background: rgba(79,126,248,0.10);
  border: 1px solid rgba(79,126,248,0.22);
  padding: 0.2rem 0.6rem;
  border-radius: 100px;
  margin-bottom: 0.75rem;
}

.gs-path h2 {
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f1117;
  margin-bottom: 0.5rem;
  letter-spacing: -0.02em;
}

.gs-path-context {
  font-size: 0.9rem;
  color: #5a6278;
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.gs-lib-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.gs-lib-item {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 1rem;
  align-items: baseline;
  padding: 0.7rem 0.85rem;
  background: #fff;
  border: 1px solid #e2e6f0;
  border-radius: 8px;
}

.gs-lib-name {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.82rem;
  font-weight: 600;
  color: #0f1117;
}

.gs-lib-name a {
  color: #4f7ef8;
  text-decoration: none;
}

.gs-lib-name a:hover {
  text-decoration: underline;
}

.gs-lib-desc {
  font-size: 0.875rem;
  color: #4a4f66;
  line-height: 1.5;
}

.gs-next {
  margin-top: 2.5rem;
  padding-top: 2rem;
  border-top: 1px solid #e2e6f0;
}

.gs-next h2 {
  font-size: 1.1rem;
  font-weight: 700;
  color: #0f1117;
  margin-bottom: 1rem;
}

.gs-cta-row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.gs-cta-row a {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.15s, border-color 0.15s;
}

.btn-primary-sm {
  background: #4f7ef8;
  color: #fff;
  border: 1px solid #4f7ef8;
}

.btn-primary-sm:hover {
  background: #3d6de6;
  border-color: #3d6de6;
  text-decoration: none;
}

.btn-outline-sm {
  background: #fff;
  color: #4f7ef8;
  border: 1px solid #c6d4f9;
}

.btn-outline-sm:hover {
  background: rgba(79,126,248,0.06);
  border-color: #4f7ef8;
  text-decoration: none;
}

.gs-examples {
  border: 1px solid #e2e6f0;
  border-radius: 12px;
  padding: 1.75rem 2rem;
  margin-bottom: 2rem;
  background: #f9fafc;
}

.gs-examples h2 {
  font-size: 1.25rem;
  font-weight: 700;
  color: #0f1117;
  margin-top: 0;
  margin-bottom: 0.5rem;
  letter-spacing: -0.02em;
}

.gs-examples-intro {
  font-size: 0.9rem;
  color: #5a6278;
  margin-bottom: 1.5rem;
  line-height: 1.6;
}

.gs-examples-intro a {
  color: #4f7ef8;
  text-decoration: none;
}

.gs-examples-intro a:hover {
  text-decoration: underline;
}

.gs-examples-note {
  font-size: 0.82rem;
  color: #5a6278;
  margin-top: 1.25rem;
  line-height: 1.6;
  font-style: italic;
}

.gs-example-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.gs-example-table thead tr {
  border-bottom: 2px solid #e2e6f0;
}

.gs-example-table th {
  text-align: left;
  font-weight: 700;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #8892a4;
  padding: 0.5rem 0.75rem 0.6rem;
}

.gs-example-table td {
  padding: 0.65rem 0.75rem;
  vertical-align: top;
  border-bottom: 1px solid #e2e6f0;
  color: #4a4f66;
  line-height: 1.5;
}

.gs-example-table tr:last-child td {
  border-bottom: none;
}

.gs-example-table td:first-child {
  font-weight: 600;
  color: #0f1117;
  white-space: nowrap;
  padding-right: 1.25rem;
}

.gs-example-table a {
  color: #4f7ef8;
  text-decoration: none;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
}

.gs-example-table a:hover {
  text-decoration: underline;
}

@media (max-width: 640px) {
  .gs-lib-item {
    grid-template-columns: 1fr;
    gap: 0.25rem;
  }
  .gs-path {
    padding: 1.25rem 1.25rem;
  }
  .gs-examples {
    padding: 1.25rem 1.25rem;
  }
  .gs-example-table td:first-child {
    white-space: normal;
  }
}
</style>

<p class="gs-intro">Most people coming to Burning Cost have one of three starting points. Pick the path closest to yours — each one recommends three to five libraries that solve problems you will encounter first, in an order that makes sense.</p>

<div class="gs-path">
  <div class="gs-path-label">Path 1</div>
  <h2>Pricing actuary moving to Python</h2>
  <p class="gs-path-context">You know the techniques: GLMs, factor tables, A/E monitoring, credibility. What you need are Python equivalents that produce output in the same formats you already use, and that handle the insurance-specific details (IBNR buffers, exposure weighting, renewal cohort structure) that generic ML libraries ignore.</p>
  <ul class="gs-lib-list">
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></span>
      <span class="gs-lib-desc">Turns a CatBoost GBM into a multiplicative factor table — same output format as exp(&beta;) from Emblem or Radar, with confidence intervals and exposure weighting. This is where most actuaries start.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></span>
      <span class="gs-lib-desc">Walk-forward cross-validation with IBNR buffers. Stops future experience from leaking into training folds. Produces Poisson and Gamma deviance scores, not just RMSE.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></span>
      <span class="gs-lib-desc">Three-layer post-deployment monitoring: exposure-weighted PSI/CSI, segmented A/E ratios with IBNR adjustment, and a Gini z-test that tells you whether to recalibrate or refit.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></span>
      <span class="gs-lib-desc">PRA SS1/23-aligned model validation reports. Bootstrap Gini CI, Poisson A/E CI, double-lift charts, renewal cohort test. HTML/JSON output for model risk committees.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-credibility" target="_blank">insurance-credibility</a></span>
      <span class="gs-lib-desc">Bühlmann-Straub credibility in Python. Practical for capping thin segments, stabilising NCD factors, and blending a new model with an incumbent rate.</span>
    </li>
  </ul>
</div>

<div class="gs-path">
  <div class="gs-path-label">Path 2</div>
  <h2>Data scientist joining an insurance pricing team</h2>
  <p class="gs-path-context">You have the ML fundamentals. What you are missing is the insurance context: why you cannot use k-fold CV, what IBNR means for your validation, how proxy discrimination is tested under FCA guidance, and why your coefficient estimates might be confounded. These libraries encode that context.</p>
  <ul class="gs-lib-list">
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-datasets" target="_blank">insurance-datasets</a></span>
      <span class="gs-lib-desc">Synthetic UK motor portfolio data with a known data-generating process. Use it to validate your methods before touching real data.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></span>
      <span class="gs-lib-desc">Temporally correct cross-validation. Random folds are wrong for insurance data. This explains why and fixes it.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></span>
      <span class="gs-lib-desc">Proxy discrimination auditing aligned with FCA EP25/2 and Consumer Duty. Quantifies indirect discrimination risk from rating variables correlated with protected characteristics.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></span>
      <span class="gs-lib-desc">Double machine learning for deconfounding rating factors. If your rating variables correlate with distribution channel or policyholder behaviour, standard GLM coefficients are biased.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></span>
      <span class="gs-lib-desc">Produces outputs that the actuarial side of the team will recognise. Bridge between a model that lives in Python and a committee that wants a factor table.</span>
    </li>
  </ul>
</div>

<div class="gs-path">
  <div class="gs-path-label">Path 3</div>
  <h2>Technical pricing team lead evaluating what to adopt</h2>
  <p class="gs-path-context">You need to know what is production-ready, what the regulatory exposure is, and how to move models through sign-off without creating a maintenance burden. These libraries have actuarial tests, clear scope, and outputs a pricing committee or auditor can follow.</p>
  <ul class="gs-lib-list">
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-deploy" target="_blank">insurance-deploy</a></span>
      <span class="gs-lib-desc">Champion/challenger framework with shadow mode, SHA-256 deterministic routing, SQLite quote log, and a bootstrap likelihood ratio test to declare a winner. ICOBS 6B.2 audit trail included.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></span>
      <span class="gs-lib-desc">PRA SS1/23 model validation reports in HTML and JSON. Covers the tests a model risk function will ask for, in a format they can file.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></span>
      <span class="gs-lib-desc">Post-deployment drift monitoring with a clear decision rule: recalibrate vs. refit. Reduces the judgement calls that stall model review cycles.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></span>
      <span class="gs-lib-desc">Proxy discrimination audit that produces an evidence pack for Consumer Duty and FCA supervisory review. Quantifies risk before sign-off, not after a regulatory question.</span>
    </li>
    <li class="gs-lib-item">
      <span class="gs-lib-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a></span>
      <span class="gs-lib-desc">Distribution-free prediction intervals with finite-sample coverage guarantees. Relevant wherever a model needs a principled uncertainty bound for Solvency II or internal capital.</span>
    </li>
  </ul>
</div>

<div class="gs-examples">
  <h2>Worked Examples</h2>
  <p class="gs-examples-intro">The <a href="https://github.com/burning-cost/burning-cost-examples" target="_blank">burning-cost-examples</a> repo contains full worked examples that go beyond the quick-start.</p>
  <table class="gs-example-table">
    <thead>
      <tr>
        <th>Example</th>
        <th>What it covers</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Causal Rate Change Evaluation</td>
        <td>SDID workflow for measuring the causal impact of a rate change</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/examples/causal_rate_change_evaluation.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>Price Elasticity Optimisation</td>
        <td>DML elasticity estimation and ENBP-constrained renewal pricing</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/examples/price_elasticity_optimisation.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>Conformal Prediction Intervals</td>
        <td>Distribution-free prediction intervals for Tweedie models</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/examples/conformal_prediction_intervals.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>Champion/Challenger Deployment</td>
        <td>Shadow mode, quote logging, bootstrap LR test, ENBP audit</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/examples/champion_challenger_deployment.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>Model Drift Monitoring</td>
        <td>Exposure-weighted PSI/CSI, Gini drift z-test, quarterly governance</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/examples/model_drift_monitoring.py" target="_blank">view</a></td>
      </tr>
    </tbody>
  </table>
  <p class="gs-examples-note">Each example generates synthetic data inline — no external files needed. Install the relevant library and run.</p>
</div>

<div class="gs-next">
  <h2>Ready to go deeper?</h2>
  <div class="gs-cta-row">
    <a href="/tools/" class="btn-primary-sm">Browse all libraries</a>
    <a href="/blog/" class="btn-outline-sm">Read the articles</a>
    <a href="https://github.com/burning-cost" target="_blank" class="btn-outline-sm">View on GitHub</a>
  </div>
</div>
