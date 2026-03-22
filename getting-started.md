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

/* Quick install cheat sheet */
.gs-install {
  margin-bottom: 3rem;
}

.gs-install h2 {
  font-size: 1.2rem;
  font-weight: 700;
  color: #0f1117;
  margin-bottom: 0.4rem;
  letter-spacing: -0.02em;
}

.gs-install-sub {
  font-size: 0.9rem;
  color: #5a6278;
  margin-bottom: 1.25rem;
}

.gs-install-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 0.75rem;
}

.gs-install-card {
  background: #fff;
  border: 1px solid #e2e6f0;
  border-radius: 10px;
  padding: 1rem 1.15rem;
}

.gs-install-card-name {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  color: #4f7ef8;
  margin-bottom: 0.5rem;
}

.gs-install-card-name a {
  color: #4f7ef8;
  text-decoration: none;
}

.gs-install-card-name a:hover {
  text-decoration: underline;
}

.gs-install-cmds {
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
}

.gs-install-cmd {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.78rem;
  background: #f4f6fb;
  border: 1px solid #e2e6f0;
  border-radius: 5px;
  padding: 0.3rem 0.6rem;
  color: #1a1d2e;
  white-space: nowrap;
  overflow-x: auto;
}

.gs-install-cmd .cmd-prefix {
  color: #8892b0;
  user-select: none;
}

/* First 5 minutes section */
.gs-five-min {
  margin-bottom: 3rem;
  border: 1px solid #e2e6f0;
  border-radius: 12px;
  overflow: hidden;
}

.gs-five-min-header {
  padding: 1.25rem 1.75rem;
  background: #f9fafc;
  border-bottom: 1px solid #e2e6f0;
}

.gs-five-min-header h2 {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f1117;
  margin: 0 0 0.3rem 0;
  letter-spacing: -0.02em;
}

.gs-five-min-header p {
  font-size: 0.88rem;
  color: #5a6278;
  margin: 0;
  line-height: 1.5;
}

.gs-five-min-body {
  padding: 1.5rem 1.75rem;
  background: #fff;
}

.gs-five-min-code {
  background: #0f1117;
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  line-height: 1.7;
  color: #cdd6f4;
  overflow-x: auto;
  white-space: pre;
}

.gs-five-min-code .c-kw  { color: #89b4fa; }  /* keyword */
.gs-five-min-code .c-cls { color: #a6e3a1; }  /* class */
.gs-five-min-code .c-fn  { color: #89dceb; }  /* function */
.gs-five-min-code .c-str { color: #f9e2af; }  /* string */
.gs-five-min-code .c-num { color: #fab387; }  /* number */
.gs-five-min-code .c-cm  { color: #6c7086; font-style: italic; } /* comment */

.gs-five-min-note {
  margin-top: 1rem;
  font-size: 0.85rem;
  color: #5a6278;
  line-height: 1.6;
}

/* Worked examples section */
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
  .gs-install-grid {
    grid-template-columns: 1fr;
  }
  .gs-five-min-body {
    padding: 1.25rem;
  }
  .gs-five-min-header {
    padding: 1rem 1.25rem;
  }
  .gs-examples {
    padding: 1.25rem 1.25rem;
  }
  .gs-example-table td:first-child {
    white-space: normal;
  }
}

.gs-next-steps {
  margin-top: 1.25rem;
  padding: 1.1rem 1.25rem;
  background: #eef2ff;
  border: 1px solid #c6d4f9;
  border-radius: 8px;
}

.gs-next-steps-label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #4f7ef8;
  margin-bottom: 0.6rem;
}

.gs-next-steps-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.gs-next-steps-list li {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  font-size: 0.875rem;
  color: #2a2d3e;
  line-height: 1.5;
}

.gs-next-steps-list li::before {
  content: "";
  display: inline-block;
  width: 5px;
  height: 5px;
  background: #4f7ef8;
  border-radius: 50%;
  flex-shrink: 0;
  position: relative;
  top: -1px;
}

.gs-next-steps-list a {
  color: #4f7ef8;
  text-decoration: none;
  font-weight: 600;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
}

.gs-next-steps-list a:hover {
  text-decoration: underline;
}

.gs-next-steps-list .nb-desc {
  color: #5a6278;
}
</style>

<p class="gs-intro">Most people coming to Burning Cost have one of three starting points. Pick the path closest to yours — each one recommends three to five libraries that solve problems you will encounter first, in an order that makes sense.</p>

<div style="background: #eef2ff; border: 1px solid #c6d4f9; border-left: 4px solid #4f7ef8; border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 2rem; font-size: 0.92rem; color: #2a2d3e; line-height: 1.65;">
  <strong>Want to see all our libraries working together?</strong> The <a href="/pipeline/" style="color: #4f7ef8; font-weight: 600;">complete motor pricing pipeline</a> walks through a full production workflow — data with a known DGP, GBM training, SHAP factor extraction, GLM distillation, conformal prediction intervals, fairness audit, drift monitoring, and governance pack — in a single runnable script.
</div>

<!-- Quick install cheat sheet -->
<div class="gs-install">
  <h2>Quick install</h2>
  <p class="gs-install-sub">The six libraries most teams reach for first. Copy the command that matches your setup.</p>
  <div class="gs-install-grid">

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install shap-relativities</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add shap-relativities</div>
      </div>
    </div>

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/insurance-cv" target="_blank">insurance-cv</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install insurance-cv</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add insurance-cv</div>
      </div>
    </div>

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install insurance-fairness</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add insurance-fairness</div>
      </div>
    </div>

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install insurance-monitoring</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add insurance-monitoring</div>
      </div>
    </div>

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/insurance-distributional" target="_blank">insurance-distributional</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install insurance-distributional</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add insurance-distributional</div>
      </div>
    </div>

    <div class="gs-install-card">
      <div class="gs-install-card-name"><a href="https://github.com/burning-cost/insurance-glm-tools" target="_blank">insurance-glm-tools</a></div>
      <div class="gs-install-cmds">
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>pip install insurance-glm-tools</div>
        <div class="gs-install-cmd"><span class="cmd-prefix">$ </span>uv add insurance-glm-tools</div>
      </div>
    </div>

  </div>
</div>

<!-- Your first 5 minutes -->
<div class="gs-five-min">
  <div class="gs-five-min-header">
    <h2>Your first 5 minutes</h2>
    <p>A minimal working example using shap-relativities. Fits a CatBoost model on synthetic motor data and extracts a multiplicative factor table — the same output format as exp(&beta;) from a GLM. No data download required.</p>
  </div>
  <div class="gs-five-min-body">
    <div class="gs-five-min-code"><span class="c-kw">import</span> numpy <span class="c-kw">as</span> np
<span class="c-kw">from</span> <span class="c-cls">catboost</span> <span class="c-kw">import</span> <span class="c-cls">CatBoostRegressor</span>
<span class="c-kw">from</span> <span class="c-cls">shap_relativities</span> <span class="c-kw">import</span> <span class="c-cls">SHAPRelativities</span>

<span class="c-cm"># Synthetic motor portfolio: vehicle_age, driver_age, annual_mileage</span>
rng = np.<span class="c-fn">random</span>.<span class="c-cls">default_rng</span>(<span class="c-num">42</span>)
X = np.<span class="c-fn">column_stack</span>([
    rng.<span class="c-fn">integers</span>(<span class="c-num">0</span>, <span class="c-num">15</span>, <span class="c-num">2000</span>),   <span class="c-cm"># vehicle_age</span>
    rng.<span class="c-fn">integers</span>(<span class="c-num">18</span>, <span class="c-num">80</span>, <span class="c-num">2000</span>),   <span class="c-cm"># driver_age</span>
    rng.<span class="c-fn">integers</span>(<span class="c-num">5000</span>, <span class="c-num">30000</span>, <span class="c-num">2000</span>),  <span class="c-cm"># annual_mileage</span>
])
y = <span class="c-num">0.08</span> + <span class="c-num">0.005</span> * X[:, <span class="c-num">0</span>] - <span class="c-num">0.001</span> * X[:, <span class="c-num">1</span>] + rng.<span class="c-fn">exponential</span>(<span class="c-num">0.02</span>, <span class="c-num">2000</span>)

model = <span class="c-cls">CatBoostRegressor</span>(iterations=<span class="c-num">200</span>, verbose=<span class="c-num">0</span>)
model.<span class="c-fn">fit</span>(X, y)

sr = <span class="c-cls">SHAPRelativities</span>(model, feature_names=[<span class="c-str">"vehicle_age"</span>, <span class="c-str">"driver_age"</span>, <span class="c-str">"annual_mileage"</span>])
factors = sr.<span class="c-fn">fit_transform</span>(X)

<span class="c-fn">print</span>(factors[<span class="c-str">"vehicle_age"</span>].head())
<span class="c-cm">#    level  relativity  ci_lower  ci_upper</span>
<span class="c-cm">#    0      1.000       0.981     1.019</span>
<span class="c-cm">#    1      1.043       1.028     1.058</span>
<span class="c-fn">print</span>(<span class="c-str">f"Reconstruction R² = {sr.reconstruction_r2:.4f}"</span>)</div>
    <p class="gs-five-min-note">The <code>factors</code> dict maps each feature name to a DataFrame with one row per level. <code>relativity</code> is the multiplicative factor — the same structure as a GLM output from Emblem or Radar. <code>reconstruction_r2</code> tells you how much of the model's variance the factor table explains; above 0.95 is production-usable.</p>
  </div>
</div>

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

<div class="gs-next-steps">
  <div class="gs-next-steps-label">Your next 3 notebooks</div>
  <ul class="gs-next-steps-list">
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py" target="_blank">shap-relativities</a> <span class="nb-desc">— CatBoost relativities vs GLM vs true DGP on synthetic UK motor data. The factor table output format matches Emblem and Radar imports.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/temporal_cross_validation.py" target="_blank">insurance-cv</a> <span class="nb-desc">— Random CV vs temporal CV vs true out-of-time holdout. Shows how much your current CV approach is flattering your model.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distill_demo.py" target="_blank">insurance-distill</a> <span class="nb-desc">— GBM-to-GLM distillation, surrogate factor tables for rating engines. The notebook to read before your next "how do we get it into Radar?" conversation.</span></li>
  </ul>
</div>
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
      <span class="gs-lib-desc">Proxy discrimination auditing aligned with FCA Consumer Duty and Equality Act 2010. Quantifies indirect discrimination risk from rating variables correlated with protected characteristics.</span>
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

<div class="gs-next-steps">
  <div class="gs-next-steps-label">Your next 3 notebooks</div>
  <ul class="gs-next-steps-list">
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/conformal_prediction_intervals.py" target="_blank">insurance-conformal</a> <span class="nb-desc">— Tweedie conformal intervals vs bootstrap on 50k motor policies. Distribution-free coverage guarantees that don't require a correctly specified model.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_causal_demo.py" target="_blank">insurance-causal</a> <span class="nb-desc">— DML causal effect vs naive Poisson GLM on confounded data. Shows how much bias your current elasticity estimates contain.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py" target="_blank">insurance-fairness</a> <span class="nb-desc">— Proxy discrimination audit aligned to FCA Consumer Duty. Runs the Lindholm correction and produces the evidence pack the FCA expects to see.</span></li>
  </ul>
</div>
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

<div class="gs-next-steps">
  <div class="gs-next-steps-label">Your next 3 notebooks</div>
  <ul class="gs-next-steps-list">
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_governance_demo.py" target="_blank">insurance-governance</a> <span class="nb-desc">— PRA SS1/23 validation workflow, model risk tiering, HTML report output. The format a model risk committee or PRA supervisor can act on.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py" target="_blank">insurance-monitoring</a> <span class="nb-desc">— Exposure-weighted PSI/CSI, segmented A/E ratios, Gini drift z-test. Gives you a defensible decision rule for recalibrate vs refit.</span></li>
    <li><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/champion_challenger_deployment.py" target="_blank">insurance-deploy</a> <span class="nb-desc">— Shadow mode, quote logging, bootstrap likelihood ratio test, ICOBS 6B.2 audit trail. End-to-end champion/challenger with a clear winner declaration method.</span></li>
  </ul>
</div>
</div>

<!-- Worked examples -->
<div class="gs-examples">
  <h2>Worked Examples</h2>
  <p class="gs-examples-intro">The <a href="https://github.com/burning-cost/burning-cost-examples" target="_blank">burning-cost-examples</a> repo contains 47 Databricks notebooks covering the full ecosystem. (These are in the burning-cost-examples GitHub repo — the <a href="/notebooks/">Notebooks</a> page on this site hosts a curated subset.) Each one installs its own dependencies, generates synthetic data, fits models, and benchmarks against a standard actuarial baseline. Browse the <a href="https://github.com/burning-cost/burning-cost-examples/tree/main/notebooks" target="_blank">notebooks/ directory</a> or pick from the table below — sorted by library name.</p>
  <table class="gs-example-table">
    <thead>
      <tr>
        <th>Library</th>
        <th>What it shows</th>
        <th></th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>bayesian-pricing</td>
        <td>Hierarchical Bayesian vs raw experience on thin segments</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/bayesian_pricing_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-causal</td>
        <td>DML causal effect vs naive Poisson GLM on confounded data</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_causal_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-causal-policy</td>
        <td>SDID rate change evaluation with event study and HonestDiD</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/causal_rate_change_evaluation.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-conformal</td>
        <td>Tweedie conformal intervals vs bootstrap on 50k motor</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/conformal_prediction_intervals.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-conformal-ts</td>
        <td>ACI/SPCI vs split conformal on non-exchangeable time series</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_conformal_ts_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-covariate-shift</td>
        <td>Importance-weighted evaluation after distribution shift</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_covariate_shift_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-credibility</td>
        <td>Bühlmann-Straub credibility vs raw experience on 30 segments</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_credibility_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-cv</td>
        <td>Random CV vs temporal CV vs true OOT holdout</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/temporal_cross_validation.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-datasets</td>
        <td>Synthetic UK motor portfolio with known DGP, parameter recovery</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_datasets_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-deploy</td>
        <td>Shadow mode, quote logging, bootstrap LR test, ENBP audit</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/champion_challenger_deployment.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-dispersion</td>
        <td>DGLM vs constant-phi Gamma GLM, per-risk volatility scoring</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_dispersion_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-distributional</td>
        <td>Distributional GBM (TweedieGBM) vs standard point predictions</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distributional_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-distributional-glm</td>
        <td>GAMLSS vs standard Gamma GLM on heterogeneous-variance data</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distributional_glm_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-distill</td>
        <td>GBM-to-GLM distillation, surrogate factor tables for rating engines</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_distill_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-dynamics</td>
        <td>GAS Poisson filter vs static GLM, BOCPD changepoint detection</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_dynamics_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-fairness</td>
        <td>Proxy discrimination audit, bias metrics, Lindholm correction</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/fairness_audit_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-frequency-severity</td>
        <td>Sarmanov copula joint freq-sev vs independence assumption</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_frequency_severity_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-gam</td>
        <td>EBM/ANAM vs Poisson GLM with planted non-linear effects</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_gam_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-glm-tools</td>
        <td>Nested GLM embeddings for 500 vehicle makes vs dummy-coded GLM</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_glm_tools_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-governance</td>
        <td>PRA SS1/23 validation workflow, MRM risk tiering, HTML report</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_governance_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-interactions</td>
        <td>CANN/NID interaction detection vs exhaustive pairwise GLM search</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_interactions_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-monitoring</td>
        <td>Exposure-weighted PSI/CSI, A/E ratios, Gini drift z-test</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/monitoring_drift_detection.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-multilevel</td>
        <td>CatBoost + REML random effects vs one-hot encoding</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_multilevel_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-optimise</td>
        <td>SLSQP constrained optimisation, efficient frontier, FCA audit</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_optimise_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-quantile</td>
        <td>CatBoost quantile regression vs lognormal, TVaR, ILF curves</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_quantile_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-severity</td>
        <td>Spliced Lognormal-GPD + DRN vs Gamma GLM, tail quantiles</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_severity_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-spatial</td>
        <td>BYM2 territory factors vs postcode grouping, Moran's I</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/spatial_territory_ratemaking.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-survival</td>
        <td>Cure models vs KM/Cox PH, CLV bias by cure band</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_survival_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-synthetic</td>
        <td>Vine copula generation, fidelity report, TSTR benchmarks</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/synthetic_portfolio_generation.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-telematics</td>
        <td>HMM latent-state features vs raw trip aggregates</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_telematics_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-thin-data</td>
        <td>GLMTransfer + TabPFN vs raw GLM on thin segments</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_thin_data_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-trend</td>
        <td>Automated trend selection vs naive OLS, structural breaks</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_trend_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>insurance-whittaker</td>
        <td>W-H smoothing with REML lambda vs manual step smoothing</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/insurance_whittaker_demo.py" target="_blank">view</a></td>
      </tr>
      <tr>
        <td>shap-relativities</td>
        <td>CatBoost relativities vs GLM vs true DGP on synthetic motor</td>
        <td><a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/shap_relativities_demo.py" target="_blank">view</a></td>
      </tr>
    </tbody>
  </table>
  <p class="gs-examples-note" style="font-style: normal;"><strong>How to run:</strong> Import any notebook into Databricks via <code>databricks workspace import notebooks/&lt;name&gt;.py /Workspace/Users/you@example.com/&lt;name&gt; --language PYTHON --overwrite</code>, or drag-and-drop the <code>.py</code> file in the Databricks UI. Notebooks use <code>%pip install</code> cells and run on Databricks Free Edition serverless compute — no cluster setup needed.</p>
  <p class="gs-examples-note">Each notebook generates synthetic data inline — no external files needed. Install the relevant library and run.</p>
</div>

<div class="gs-next">
  <h2>Ready to go deeper?</h2>
  <div class="gs-cta-row">
    <a href="/tools/" class="btn-primary-sm">Browse all libraries</a>
    <a href="/guide/" class="btn-outline-sm">Which library do I need?</a>
    <a href="/blog/" class="btn-outline-sm">Read the articles</a>
    <a href="https://github.com/burning-cost" target="_blank" class="btn-outline-sm">View on GitHub</a>
  </div>
</div>
