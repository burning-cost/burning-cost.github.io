---
layout: page
title: "Insurance Pricing in Python: A Practitioner's Course"
description: "12-module written training course covering GLMs, GBMs, SHAP relativities, conformal prediction, causal elasticity, spatial ratemaking, and rate optimisation. Python notebooks. £97."
permalink: /course/
---

<style>
.course-hero {
  max-width: 720px;
  margin-bottom: 2.5rem;
}

.course-hero p {
  font-size: 1.1rem;
  line-height: 1.75;
  color: #2a2d3e;
}

.module-list {
  border-left: 3px solid #e2e6f0;
  padding-left: 1.5rem;
  margin: 2rem 0;
}

.module-list li {
  margin-bottom: 0.75rem;
  line-height: 1.6;
}

.module-list strong {
  color: #1a1d2e;
}

.format-box {
  background: #f7f8fc;
  border: 1px solid #e2e6f0;
  border-radius: 8px;
  padding: 1.5rem 2rem;
  margin: 2rem 0;
}

.format-box ul {
  margin: 0.5rem 0 0 0;
}

.cta-block {
  background: #1a1d2e;
  color: #fff;
  border-radius: 8px;
  padding: 2rem 2.5rem;
  margin: 2.5rem 0;
  text-align: center;
}

.cta-block p {
  color: #c8ccdc;
  margin-bottom: 1.25rem;
}

.cta-button {
  display: inline-block;
  background: #f5a623;
  color: #1a1d2e;
  font-weight: 700;
  font-size: 1.1rem;
  padding: 0.85rem 2.25rem;
  border-radius: 6px;
  text-decoration: none;
}

.cta-button:hover {
  background: #e09510;
  color: #1a1d2e;
}

.refund-note {
  font-size: 0.9rem;
  color: #888;
  margin-top: 1rem;
}

.preview-posts {
  display: grid;
  gap: 1.25rem;
  margin: 1.5rem 0;
}

.preview-post {
  border: 1px solid #e2e6f0;
  border-radius: 6px;
  padding: 1rem 1.25rem;
}

.preview-post a {
  font-weight: 600;
  text-decoration: none;
}

.preview-post p {
  margin: 0.3rem 0 0 0;
  font-size: 0.92rem;
  color: #555;
}
</style>

<div class="course-hero">
<p>If you can fit a GLM in Excel but want to stop there, this is not the right course. If you are a pricing actuary or data scientist who needs to take a proper Python pricing stack into production — GLMs, GBMs, conformal intervals, causal elasticity, spatial territory models — this is twelve modules of worked code written by people who have done it on real UK books.</p>
</div>

<div class="cta-block">
  <p>Written course · 12 modules · Python notebooks · £97 one-off payment</p>
  <a class="cta-button" href="mailto:pricing.frontier@gmail.com?subject=Insurance%20Pricing%20Course%20—%20Purchase&body=I'd%20like%20to%20purchase%20the%20Insurance%20Pricing%20in%20Python%20course%20(£97).%20Please%20send%20payment%20details.">Buy — £97</a>
  <p class="refund-note">No questions asked refund within 30 days. Email us and we refund in full.</p>
</div>

---

## What you get

A written course: twelve self-contained modules delivered as Markdown tutorials with Python throughout. Each module ships with a detailed tutorial covering the theory, the implementation, and the failure modes — not just the happy path.

<div class="format-box">
<strong>Every module includes:</strong>
<ul>
  <li>Markdown tutorial — readable offline, no login required</li>
  <li>Jupyter notebooks (<code>.ipynb</code>) — run locally or on Databricks</li>
  <li>Exercises with worked solutions</li>
  <li>Bundled library wheels — install without a PyPI connection if your environment requires it</li>
</ul>
</div>

No video. No subscription. No content dripped out week by week. You get the full zip on purchase and work through it at your own pace.

---

## Who it is for

**Pricing actuaries** who know GLM theory but want to work in Python rather than Emblem or R — or who are building on top of a GBM and need to get results into a form the regulator and rating engine will accept.

**Data scientists joining insurance teams** who understand ML but have not yet encountered exposure offsets, temporal cross-validation, FCA ENBP constraints, or what a rating factor table is supposed to look like.

**Technical pricing managers** who need to understand, challenge, and sign off what their team is building — without necessarily running the code themselves.

You need working Python knowledge. You do not need actuarial qualifications or prior insurance pricing experience.

---

## The 12 modules

<ol class="module-list">
  <li><strong>Databricks setup</strong> — workspace, clusters, Unity Catalog, and cluster libraries for pricing workloads; the environment the rest of the course runs in</li>
  <li><strong>GLMs in Python</strong> — statsmodels Tweedie GLM with exposure offset, factor diagnostics, and lift charts; everything Emblem does, in code</li>
  <li><strong>GBMs with CatBoost</strong> — frequency/severity gradient boosting on freMTPL2, temporal walk-forward cross-validation, hyperparameter search</li>
  <li><strong>SHAP relativities</strong> — extract GLM-style multiplicative factor tables from a CatBoost model; the format your rating engine and regulator expect</li>
  <li><strong>Conformal prediction intervals</strong> — distribution-free per-risk uncertainty bounds with coverage guarantees that hold without distributional assumptions</li>
  <li><strong>Credibility and Bayesian methods</strong> — Bühlmann-Straub in Python, PyMC hierarchical models for thin-data segments, mixed-model equivalence</li>
  <li><strong>Rate optimisation</strong> — constrained SLSQP to hit a loss ratio target while respecting movement caps; FCA ENBP enforcement; efficient frontier between LR and volume</li>
  <li><strong>End-to-end pipeline</strong> — raw tabular data to filed rates; MLflow tracking, model registry, and a signed-off audit trail in a single Databricks workflow</li>
  <li><strong>Demand elasticity</strong> — Double ML causal estimation of own-price elasticity for renewal pricing; deconfounding with CausalForestDML</li>
  <li><strong>Interaction detection</strong> — SHAP interaction values, dependence plots, and partial dependence to find what your GLM missed</li>
  <li><strong>Model monitoring and drift detection</strong> — exposure-weighted PSI/CSI, A/E drift ratios, sequential testing for early warnings, PIT calibration checks; when to retrain and when to leave it alone</li>
  <li><strong>Spatial territory rating</strong> — geospatial smoothing with BYM2, Voronoi banding from postcode centroids, borough-level risk surfaces</li>
</ol>

---

## Free previews

These posts on the blog cover the same ground as three of the course modules, so you can get a sense of the depth and style before buying.

<div class="preview-posts">
  <div class="preview-post">
    <a href="/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/">Extracting Rating Relativities from GBMs with SHAP</a>
    <p>Covers the same ground as Module 4 — SHAP decomposition, multiplicative factor tables, confidence intervals. The course module adds exercises and a worked pipeline from raw CatBoost model to Radar-importable file.</p>
  </div>
  <div class="preview-post">
    <a href="/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/">Conformal Prediction Intervals for Insurance Pricing Models</a>
    <p>The theoretical foundation for Module 5. The course extends this with heteroscedasticity handling, temporal exchangeability, and a worked calibration exercise on a real frequency/severity split.</p>
  </div>
  <div class="preview-post">
    <a href="/2026/02/21/constrained-rate-optimisation-efficient-frontier/">Constrained Rate Optimisation and the Efficient Frontier</a>
    <p>The rate optimisation post that underpins Module 7. The course module adds the demand elasticity coupling, FCA ENBP enforcement code, and a full scenario comparison exercise.</p>
  </div>
</div>

---

## Questions

Ask on [GitHub Discussions](https://github.com/orgs/burning-cost/discussions) or email [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com).

<div class="cta-block">
  <p>Written course · 12 modules · Python notebooks · £97 one-off payment</p>
  <a class="cta-button" href="mailto:pricing.frontier@gmail.com?subject=Insurance%20Pricing%20Course%20—%20Purchase&body=I'd%20like%20to%20purchase%20the%20Insurance%20Pricing%20in%20Python%20course%20(£97).%20Please%20send%20payment%20details.">Buy — £97</a>
  <p class="refund-note">No questions asked refund within 30 days. Email us and we refund in full.</p>
</div>
