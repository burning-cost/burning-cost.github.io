---
layout: page
title: "Top 10 Libraries — Where to Start"
description: "Ten libraries for the hard problems in UK personal lines pricing. FCA proxy auditing, PRA model validation, DML causal inference, conformal intervals, constrained rate optimisation, and more. Start here."
permalink: /top-10/
---

<style>

/* ── Intro strip ──────────────────────────────────────────────── */

.t10-intro {
  font-size: 1.05rem;
  color: #2a2d3e;
  max-width: 680px;
  line-height: 1.75;
  margin-bottom: 0.5rem;
}

.t10-intro a { color: #4f7ef8; }

/* ── Library cards ────────────────────────────────────────────── */

.t10-list {
  list-style: none;
  padding: 0;
  margin: 2.5rem 0 0;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.t10-card {
  border: 1px solid #e2e6f0;
  border-radius: 12px;
  padding: 1.5rem 1.75rem;
  background: #fff;
  margin-bottom: 1.25rem;
  transition: border-color 0.15s, box-shadow 0.15s;
  position: relative;
}

.t10-card:hover {
  border-color: #c6d4f9;
  box-shadow: 0 2px 12px rgba(79,126,248,0.07);
}

/* Rank badge */
.t10-rank {
  position: absolute;
  top: 1.5rem;
  right: 1.5rem;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #8b92a8;
  background: #f4f6fb;
  border: 1px solid #e2e6f0;
  padding: 0.2rem 0.55rem;
  border-radius: 100px;
}

/* Name row */
.t10-name-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-bottom: 0.25rem;
}

.t10-name {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 1rem;
  font-weight: 700;
}

.t10-name a {
  color: #4f7ef8;
  text-decoration: none;
}

.t10-name a:hover { text-decoration: underline; }

.t10-tag {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 0.2rem 0.6rem;
  border-radius: 100px;
  white-space: nowrap;
}

.tag-regulatory {
  background: #fde8e8;
  color: #8b1a1a;
  border: 1px solid #f5b8b8;
}

.tag-monitoring {
  background: #fff8e1;
  color: #7a5a00;
  border: 1px solid #ffe09e;
}

.tag-uncertainty {
  background: #e8f0fe;
  color: #1a3a8b;
  border: 1px solid #b8cafc;
}

.tag-modelling {
  background: #d4f8e8;
  color: #1a6b44;
  border: 1px solid #a8ecd0;
}

.tag-causal {
  background: #f3e8ff;
  color: #5b1a8b;
  border: 1px solid #d9b8fc;
}

.tag-optimisation {
  background: #fff3e0;
  color: #7a4a00;
  border: 1px solid #ffd099;
}

/* One-liner */
.t10-oneliner {
  font-size: 0.88rem;
  color: #5a6278;
  margin-bottom: 0.9rem;
  font-style: italic;
}

/* Problem statement */
.t10-problem {
  font-size: 0.95rem;
  color: #2d3148;
  line-height: 1.7;
  margin-bottom: 1rem;
}

/* Install + links row */
.t10-footer {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.t10-install {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.8rem;
  background: #0d1117;
  color: #79c0ff;
  padding: 0.3rem 0.75rem;
  border-radius: 6px;
  white-space: nowrap;
}

.t10-link {
  font-size: 0.8rem;
  font-weight: 600;
  color: #4f7ef8;
  text-decoration: none;
  padding: 0.3rem 0.65rem;
  border: 1px solid #c6d4f9;
  border-radius: 6px;
  transition: background 0.15s;
  white-space: nowrap;
}

.t10-link:hover {
  background: rgba(79,126,248,0.07);
  text-decoration: none;
}

/* Benchmark callout */
.t10-benchmark {
  font-size: 0.8rem;
  color: #5a6278;
  background: #f7f8fc;
  border-left: 3px solid #4f7ef8;
  border-radius: 0 6px 6px 0;
  padding: 0.5rem 0.85rem;
  margin: 0.85rem 0 1rem;
  line-height: 1.55;
}

.t10-benchmark strong { color: #2d3148; font-weight: 700; }

/* ── Bottom nav ───────────────────────────────────────────────── */

.t10-bottom-nav {
  margin-top: 3rem;
  padding-top: 2rem;
  border-top: 1px solid #e2e6f0;
}

.t10-bottom-nav h2 {
  font-size: 1rem;
  font-weight: 700;
  color: #0f1117;
  margin-bottom: 1rem;
}

.t10-cta-row {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.btn-primary-sm {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
  background: #4f7ef8;
  color: #fff;
  border: 1px solid #4f7ef8;
}

.btn-primary-sm:hover { background: #6690f9; text-decoration: none; }

.btn-outline-sm {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
  background: #fff;
  color: #4f7ef8;
  border: 1px solid #c6d4f9;
}

.btn-outline-sm:hover { background: rgba(79,126,248,0.05); text-decoration: none; }

@media (max-width: 640px) {
  .t10-card { padding: 1.25rem; }
  .t10-rank { position: static; display: inline-block; margin-bottom: 0.5rem; }
  .t10-footer { flex-direction: column; align-items: flex-start; }
}

</style>

<p class="t10-intro">Forty-plus libraries, ten that matter most. These are the ones we would reach for first — the tools that address genuinely hard problems in UK personal lines pricing where nothing adequate existed before. Start here, then use the <a href="/guide/">problem guide</a> to find everything else.</p>

<ul class="t10-list">

<!-- 1. insurance-fairness -->
<li class="t10-card">
  <span class="t10-rank"># 1</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-fairness" target="_blank">insurance-fairness</a></span>
    <span class="t10-tag tag-regulatory">FCA &middot; Consumer Duty</span>
  </div>
  <p class="t10-oneliner">Proxy discrimination auditing — FCA Consumer Duty, Equality Act 2010, EP25/2</p>
  <p class="t10-problem">Your pricing actuary says postcode is a legitimate risk variable. Your compliance function needs to demonstrate it does not operate as an ethnic proxy under Consumer Duty and EP25/2. Spearman correlation misses non-linear categorical relationships entirely. This library runs CatBoost proxy R² and mutual information scoring to catch what rank correlation cannot see — and produces the Consumer Duty evidence pack for sign-off.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> Postcode proxy R²&nbsp;=&nbsp;0.777 (RED); manual Spearman&nbsp;r&nbsp;=&nbsp;0.064 — missed entirely. Detection time: 0.5s.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-fairness</code>
    <a href="https://github.com/burning-cost/insurance-fairness" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-fairness--proxy-discrimination-detection" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 2. insurance-monitoring -->
<li class="t10-card">
  <span class="t10-rank"># 2</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-monitoring" target="_blank">insurance-monitoring</a></span>
    <span class="t10-tag tag-monitoring">Monitoring &middot; Drift</span>
  </div>
  <p class="t10-oneliner">Model drift detection — exposure-weighted PSI/CSI, A/E ratios, Gini drift, sequential testing</p>
  <p class="t10-problem">A/E ratios are creeping but aggregate figures look fine — because errors in young drivers and vehicle age cancel at portfolio level. This library disaggregates: exposure-weighted PSI/CSI per segment, segmented A/E with IBNR adjustment, and a Gini z-test with a formal recalibrate-vs-refit decision rule. v0.7.0 adds <code>PITMonitor</code> (e-process martingale for calibration drift — 3% FPR vs 46% for repeated Hosmer-Lemeshow) and <code>InterpretableDriftDetector</code> (BH-corrected feature-level attribution so you know which rating factors are driving the drift, not just that drift exists).</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> Manual aggregate A/E verdict: INVESTIGATE. MonitoringReport verdict: REFIT — because calibration drift concentrated in vehicle_age &lt; 3 cancels at portfolio level. PITMonitor FPR ~3% vs repeated H-L 46%.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-monitoring</code>
    <a href="https://github.com/burning-cost/insurance-monitoring" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-monitoring--in-production-drift-detection" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 3. insurance-conformal -->
<li class="t10-card">
  <span class="t10-rank"># 3</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-conformal" target="_blank">insurance-conformal</a></span>
    <span class="t10-tag tag-uncertainty">Uncertainty &middot; Solvency II</span>
  </div>
  <p class="t10-oneliner">Conformal prediction intervals for Tweedie/Poisson GBMs — distribution-free, finite-sample coverage</p>
  <p class="t10-problem">Standard conformal prediction meets aggregate coverage but systematically undercovers the highest-risk decile — the segment that drives SCR and reinsurance cost. Locally-weighted non-conformity scores adapt interval width to local variance, producing calibrated bounds in every risk segment. No distributional assumptions. Solvency II SCR bounds included.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> Standard conformal: 87.9% worst-decile coverage (misses 90% target). Locally-weighted conformal: 90%+ in every decile, 11.7% narrower than parametric.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-conformal</code>
    <a href="https://github.com/burning-cost/insurance-conformal" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-conformal--conformal-prediction-intervals" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 4. shap-relativities -->
<li class="t10-card">
  <span class="t10-rank"># 4</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/shap-relativities" target="_blank">shap-relativities</a></span>
    <span class="t10-tag tag-modelling">GBM &middot; Interpretability</span>
  </div>
  <p class="t10-oneliner">SHAP-based rating relativities from GBM models — extract GLM-style multiplicative factor tables</p>
  <p class="t10-problem">Your GBM beats the production GLM on every holdout metric. The rating engine and the actuarial committee both need multiplicative factor tables. There is no <code>exp(β)</code> in CatBoost. This library extracts TreeSHAP values, exposure-weights them per rating band, and produces the factor table format that pricing committees and rating engines expect — with reconstruction R² to validate fidelity.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> +2.85pp Gini lift over direct GLM. NCD=5 relativity error 4.47% (GLM: 9.44%). Conviction factor recovered at 1.57× within confidence interval.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install shap-relativities</code>
    <a href="https://github.com/burning-cost/shap-relativities" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#shap-relativities--shap-rating-relativities-from-gbms" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 5. insurance-causal -->
<li class="t10-card">
  <span class="t10-rank"># 5</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-causal" target="_blank">insurance-causal</a></span>
    <span class="t10-tag tag-causal">Causal Inference &middot; DML</span>
  </div>
  <p class="t10-oneliner">Double machine learning for deconfounding rating factors and causal price elasticity</p>
  <p class="t10-problem">Your vehicle value factor looks significant in the GLM, but vehicle value correlates with distribution channel — direct customers buy cheaper cars. You cannot tell whether it is genuine risk signal or channel confounding, and ordinary regression cannot separate the two. Double machine learning residualises both outcome and treatment on confounders using CatBoost nuisance models, then estimates the causal effect in the residuals. Produces a confounding bias report alongside the deconfounded coefficients. <code>insurance_causal.causal_forest</code> extends to segment-level heterogeneous treatment effects (GATES/CLAN/RATE) for portfolio-level price response analysis.</p>
  <div class="t10-benchmark"><strong>Honest benchmark:</strong> DML wins at n&nbsp;&ge;&nbsp;50,000 with large treatment effects and compounding GLM misspecification. At n&nbsp;=&nbsp;5,000 it over-partials — see the README for conditions.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-causal</code>
    <a href="https://github.com/burning-cost/insurance-causal" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-causal--dml-causal-effect-estimation" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 6. insurance-optimise -->
<li class="t10-card">
  <span class="t10-rank"># 6</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-optimise" target="_blank">insurance-optimise</a></span>
    <span class="t10-tag tag-optimisation">Optimisation &middot; FCA ENBP</span>
  </div>
  <p class="t10-oneliner">Constrained portfolio rate optimisation — SLSQP, FCA ENBP compliance, Pareto front</p>
  <p class="t10-problem">You have a technical price per segment, a loss ratio target, and movement caps. The rate change recommendation is still done in a spreadsheet where the constraints interact and the solution is not optimal. SLSQP with analytical Jacobians finds the optimal rate changes while respecting FCA ENBP constraints and retention floors simultaneously. v0.4.1 adds <code>ParetoFrontier</code>: single-objective optimisation is blind to fairness costs (premium disparity ratio 1.168 in the benchmark); the Pareto surface makes the profit/retention/fairness trade-off explicit and defensible.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> 3–8% profit uplift over flat rate change on a 2,000-renewal portfolio. ParetoFrontier: 4 non-dominated solutions on the 3-objective surface; TOPSIS selection picks the balanced operating point.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-optimise</code>
    <a href="https://github.com/burning-cost/insurance-optimise" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-optimise--constrained-portfolio-optimisation" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 7. insurance-governance -->
<li class="t10-card">
  <span class="t10-rank"># 7</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-governance" target="_blank">insurance-governance</a></span>
    <span class="t10-tag tag-regulatory">PRA SS1/23 &middot; MRM</span>
  </div>
  <p class="t10-oneliner">Model governance — PRA SS1/23 validation reports, risk tier scoring</p>
  <p class="t10-problem">Your model governance committee needs a validation report. You are producing it manually in PowerPoint. The automated suite runs bootstrap Gini CI, Poisson A/E CI, double-lift charts, and a renewal cohort test — structured to what a model risk function and PRA review expect. HTML and JSON output. The benchmark case shows manual checklists miss miscalibration concentrated in young drivers (age &lt; 30) that the automated suite catches via Hosmer-Lemeshow (p&nbsp;&lt;&nbsp;0.0001).</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> Manual checklist: flags global A/E only. Automated suite: catches age-band miscalibration, PSI shift, and Poisson CI on A/E. Overhead: 1.2s vs 0.09s — acceptable for a sign-off workflow.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-governance</code>
    <a href="https://github.com/burning-cost/insurance-governance" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-governance--pra-model-validation" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 8. insurance-severity -->
<li class="t10-card">
  <span class="t10-rank"># 8</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-severity" target="_blank">insurance-severity</a></span>
    <span class="t10-tag tag-modelling">Severity &middot; EVT</span>
  </div>
  <p class="t10-oneliner">Spliced distributions, EVT, Deep Regression Networks, composite Lognormal-GPD</p>
  <p class="t10-problem">A single Gamma GLM fits attritional claims adequately but fails structurally at the tail — large losses follow a Pareto distribution with completely different physics. This library provides spliced body-tail models with covariate-dependent thresholds, composite Lognormal-GPD for heavy tails, Deep Regression Networks for non-parametric severity, and EQRN extreme quantile neural networks. ILF tables and TVaR per risk are included. The EVT module corrects for policy limit truncation, which naive GPD ignores.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> Composite model: 5.6% tail error reduction vs single lognormal. Heavy-tail benchmark (Pareto &alpha;&nbsp;=&nbsp;1.5): 15–20pp Q99 error reduction vs Gamma GLM. TruncatedGPD vs naive GPD: shape bias 0.006 vs 0.035.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-severity</code>
    <a href="https://github.com/burning-cost/insurance-severity" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-severity--evt-tail-modelling" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 9. insurance-causal-policy -->
<li class="t10-card">
  <span class="t10-rank"># 9</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-causal-policy" target="_blank">insurance-causal-policy</a></span>
    <span class="t10-tag tag-causal">Causal Inference &middot; SDID</span>
  </div>
  <p class="t10-oneliner">SDID + doubly robust SC for causal rate change evaluation — HonestDiD sensitivity, FCA evidence pack</p>
  <p class="t10-problem">You put through a rate increase in Q3 and conversion dropped. You cannot tell how much of that drop was the rate change versus market conditions, because you have no control group and the before/after comparison is confounded by market inflation. Synthetic difference-in-differences constructs a synthetic control from unaffected segments to isolate the rate change effect. Produces event study charts, HonestDiD sensitivity analysis for violations of parallel trends, and an FCA evidence pack.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> SDID: near-zero bias, ~93–95% CI coverage. Naive before-after: ~2pp upward bias (4 periods &times; 0.5pp market inflation absorbed into the estimate).</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-causal-policy</code>
    <a href="https://github.com/burning-cost/insurance-causal-policy" class="t10-link" target="_blank">GitHub</a>
    <a href="/benchmarks/#insurance-causal-policy--synthetic-did--sdid-for-pricing-interventions" class="t10-link">Benchmark</a>
  </div>
</li>

<!-- 10. insurance-quantile -->
<li class="t10-card">
  <span class="t10-rank"># 10</span>
  <div class="t10-name-row">
    <span class="t10-name"><a href="https://github.com/burning-cost/insurance-quantile" target="_blank">insurance-quantile</a></span>
    <span class="t10-tag tag-modelling">Tail Risk &middot; ILF</span>
  </div>
  <p class="t10-oneliner">Tail risk quantile and expectile regression — TVaR, increased limit factors, exceedance curves</p>
  <p class="t10-problem">Your mean model gives you no handle on the upper tail. Large loss loading and ILF curves require quantile estimates at the 90th and 99th percentile per risk, not just the expected value. This library fits quantile and expectile GBMs at multiple levels simultaneously, producing per-risk TVaR and ILF tables in standard actuarial format. The GBM approach captures non-linear covariate effects on the tail that parametric EVT methods miss in moderate-sized portfolios.</p>
  <div class="t10-benchmark"><strong>Benchmark:</strong> GBM lower TVaR bias on heavy tails vs lognormal parametric. Lognormal wins on pinball loss at small n — use the parametric path below ~10,000 policies, GBM above it.</div>
  <div class="t10-footer">
    <code class="t10-install">pip install insurance-quantile</code>
    <a href="https://github.com/burning-cost/insurance-quantile" class="t10-link" target="_blank">GitHub</a>
  </div>
</li>

</ul>

<div class="t10-bottom-nav">
  <h2>All 34 libraries across the full pricing stack</h2>
  <div class="t10-cta-row">
    <a href="/tools/" class="btn-primary-sm">Browse all libraries</a>
    <a href="/guide/" class="btn-outline-sm">Problem &rarr; library guide</a>
    <a href="/getting-started/" class="btn-outline-sm">Getting started paths</a>
    <a href="/benchmarks/" class="btn-outline-sm">All benchmarks</a>
  </div>
</div>
