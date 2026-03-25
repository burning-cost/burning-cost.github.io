---
layout: default
title: "The Stack"
permalink: /stack/
---

<style>
.stack-stage {
  margin: 2rem 0;
}
.stage-label {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6b7280;
  margin-bottom: 0.25rem;
}
.lib-row {
  margin: 0.5rem 0;
}
.lib-row a {
  font-weight: 600;
}
.workflow-line {
  font-size: 1rem;
  font-weight: 600;
  margin: 0.75rem 0;
  color: #374151;
}
.arrow {
  color: #9ca3af;
  margin: 0 0.4rem;
}
</style>

Ten libraries. One workflow. Here is what to reach for at each stage.

---

## Workflow overview

<div class="workflow-line">
  Data prep
  <span class="arrow">→</span>
  Smoothing
  <span class="arrow">→</span>
  Modelling
  <span class="arrow">→</span>
  Validation
  <span class="arrow">→</span>
  Deployment
  <span class="arrow">→</span>
  Monitoring
</div>

---

## Data prep

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-datasets">insurance-datasets</a> — Synthetic UK motor data with a known data-generating process. Use it to validate your pipeline before touching production data, or to benchmark a new method against something with ground truth.
</div>
</div>

---

## Smoothing

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-whittaker">insurance-whittaker</a> — Whittaker-Henderson penalised smoothing with automatic REML lambda selection and Bayesian credible intervals. Replaces the Excel five-point moving average for experience rating curves.
</div>
</div>

---

## Modelling

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-gam">insurance-gam</a> — Interpretable tariff models: Explainable Boosting Machines, Actuarial NAM, and PIN. Shape functions a pricing actuary can read; exact Shapley values for relativities.
</div>
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-causal">insurance-causal</a> — Double Machine Learning for causal inference. Estimates price elasticity, renewal CATEs, and telematics treatment effects from observational data, with valid frequentist confidence intervals.
</div>
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-credibility">insurance-credibility</a> — Bühlmann-Straub group credibility and Bayesian individual experience rating. Finds the optimal blend between a scheme's own history and the portfolio average.
</div>
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-frequency-severity">insurance-frequency-severity</a> — Sarmanov bivariate model for correlated frequency and severity. Fits on top of your existing statsmodels GLMs; returns per-policy correction factors.
</div>
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-telematics">insurance-telematics</a> — HMM-based trip scoring into latent driving regimes, credibility-weighted to driver level, producing GLM-ready features you can explain to the FCA.
</div>
</div>

---

## Validation

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-conformal">insurance-conformal</a> — Distribution-free prediction intervals for Tweedie and Poisson models. Finite-sample coverage guarantee, 13–14% narrower than parametric baselines on heterogeneous motor books.
</div>
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-fairness">insurance-fairness</a> — Proxy discrimination audit for FCA Consumer Duty. Detects which rating factors act as protected-characteristic proxies; generates a sign-off-ready Markdown report with regulatory cross-references.
</div>
</div>

---

## Deployment

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-optimise">insurance-optimise</a> — Constrained portfolio rate optimisation. Enforces FCA PS21/5 ENBP ceilings, loss ratio targets, retention floors, and rate-change caps simultaneously; produces an auditable JSON pricing record.
</div>
</div>

---

## Monitoring

<div class="stack-stage">
<div class="lib-row">
<a href="https://github.com/burning-cost/insurance-monitoring">insurance-monitoring</a> — Model drift detection across three axes: covariate shift (PSI/CSI per feature), calibration drift (A/E with Murphy decomposition), and discrimination decay (Gini drift z-test). Returns a structured RECALIBRATE / REFIT / NO_ACTION recommendation.
</div>
</div>

---

## Common workflows

**Motor pricing refresh**

[insurance-datasets](https://github.com/burning-cost/insurance-datasets) → [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker) → [insurance-gam](https://github.com/burning-cost/insurance-gam) → [insurance-conformal](https://github.com/burning-cost/insurance-conformal) → [insurance-fairness](https://github.com/burning-cost/insurance-fairness) → [insurance-optimise](https://github.com/burning-cost/insurance-optimise) → [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

Start with synthetic data to validate the pipeline, smooth experience curves, fit an interpretable tariff, attach distribution-free prediction intervals, audit for proxy discrimination, optimise rates under FCA constraints, then monitor post-deployment.

**Renewal pricing with causal elasticity**

[insurance-causal](https://github.com/burning-cost/insurance-causal) → [insurance-optimise](https://github.com/burning-cost/insurance-optimise) → [insurance-fairness](https://github.com/burning-cost/insurance-fairness) → [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

Estimate causal price sensitivity from your existing renewal book (correcting the confounding that a raw GLM carries), feed CATE-segmented elasticities into the constrained optimiser, audit the resulting differentials for Consumer Duty compliance, and track the deployed rates.

**Telematics product build**

[insurance-telematics](https://github.com/burning-cost/insurance-telematics) → [insurance-causal](https://github.com/burning-cost/insurance-causal) → [insurance-gam](https://github.com/burning-cost/insurance-gam) → [insurance-conformal](https://github.com/burning-cost/insurance-conformal)

Score raw trip data through an HMM into auditable driver-level features, use DML to estimate the causal effect of driving style on claims (separating it from correlated demographics), fit an interpretable tariff with shape functions, and quantify per-policy uncertainty before deployment.

**Scheme and fleet pricing**

[insurance-credibility](https://github.com/burning-cost/insurance-credibility) → [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity) → [insurance-optimise](https://github.com/burning-cost/insurance-optimise)

Blend thin scheme experience with portfolio rates using Bühlmann-Straub credibility, apply a Sarmanov correction for frequency-severity dependence driven by NCD structure, then optimise the scheme terms under commercial and regulatory constraints.
