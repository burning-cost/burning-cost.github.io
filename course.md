---
layout: page
title: "Insurance Pricing in Python"
description: "Twelve modules. Real UK pricing problems. Working code you can take into production."
---

<style>
  /* Course page overrides — extends page.html prose styles */

  .course-hero-meta {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
  }

  .course-price-badge {
    display: inline-block;
    background: #4f7ef8;
    color: #fff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    padding: 0.45rem 1rem;
    border-radius: 6px;
    letter-spacing: -0.02em;
  }

  .course-meta-pill {
    display: inline-block;
    background: rgba(79,126,248,0.10);
    color: #4f7ef8;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0.7rem;
    border-radius: 4px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }

  .course-cta-block {
    background: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 10px;
    padding: 2rem 2rem 1.75rem;
    margin: 2.5rem 0;
    text-align: center;
  }

  .course-cta-block p {
    color: #8b92a8 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0 !important;
    margin-top: 0.75rem;
  }

  .btn-buy {
    display: inline-block;
    background: #4f7ef8;
    color: #fff !important;
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.8rem 2rem;
    border-radius: 7px;
    text-decoration: none !important;
    letter-spacing: -0.01em;
    transition: background 0.15s;
  }

  .btn-buy:hover {
    background: #6690f9;
    text-decoration: none !important;
  }

  .module-list {
    list-style: none !important;
    padding-left: 0 !important;
    margin-bottom: 0 !important;
    counter-reset: module-counter;
  }

  .module-list li {
    counter-increment: module-counter;
    display: grid;
    grid-template-columns: 2rem 1fr;
    gap: 0 0.75rem;
    align-items: baseline;
    padding: 0.85rem 0;
    border-bottom: 1px solid #e4e7ef;
    margin-bottom: 0 !important;
  }

  .module-list li:last-child {
    border-bottom: none;
  }

  .module-list li::before {
    content: counter(module-counter, decimal-leading-zero);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    color: #4f7ef8;
    padding-top: 0.15rem;
  }

  .module-list strong {
    color: #1a1d2e;
    font-weight: 700;
    font-size: 0.95rem;
    display: block;
    line-height: 1.4;
    margin-bottom: 0.2rem;
  }

  .module-list .module-desc {
    color: #5a6278;
    font-size: 0.875rem;
    line-height: 1.5;
  }

  .module-entry {
    grid-column: 2;
  }

  .includes-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin: 1.5rem 0 2rem;
  }

  @media (max-width: 600px) {
    .includes-grid { grid-template-columns: 1fr; }
  }

  .include-card {
    background: #f7f8fc;
    border: 1px solid #e4e7ef;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
  }

  .include-card-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #1a1d2e;
    margin-bottom: 0.35rem;
  }

  .include-card-desc {
    font-size: 0.85rem;
    color: #5a6278;
    line-height: 1.5;
  }

  .preview-posts {
    list-style: none !important;
    padding-left: 0 !important;
  }

  .preview-posts li {
    border-left: 3px solid #4f7ef8;
    padding: 0.6rem 1rem;
    margin-bottom: 0.75rem !important;
    background: #f7f8fc;
    border-radius: 0 6px 6px 0;
  }

  .preview-posts a {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1a1d2e;
    text-decoration: none;
  }

  .preview-posts a:hover {
    color: #4f7ef8;
    text-decoration: none;
  }

  .preview-posts .preview-desc {
    font-size: 0.83rem;
    color: #5a6278;
    margin-top: 0.2rem;
    line-height: 1.45;
    display: block;
  }

  .refund-note {
    font-size: 0.875rem !important;
    color: #5a6278 !important;
    border-left: none !important;
    background: none !important;
    text-align: center;
    margin-top: 0.5rem;
    line-height: 1.6;
  }
</style>

<div class="course-hero-meta">
  <span class="course-price-badge">£97</span>
  <span class="course-meta-pill">One-time payment</span>
  <span class="course-meta-pill">12 modules</span>
  <span class="course-meta-pill">No subscription</span>
</div>

---

If you can fit a GLM in Excel but want to stop there, this course is not for you.

If you are a pricing actuary or data scientist who needs to take a proper Python pricing stack into production — GLMs, GBMs, conformal intervals, causal elasticity, spatial territory models — this is twelve modules of worked code written by people who have done it on real UK books.

---

## Who it is for

- **Pricing actuaries** who know GLM theory but want to work in Python rather than Emblem or R
- **Data scientists joining insurance teams** who understand ML but have not encountered exposure offsets, temporal cross-validation, or FCA ENBP constraints before
- **Technical pricing managers** who need to understand and challenge what their team is building

You need working Python knowledge. You do not need actuarial qualifications or prior insurance pricing experience.

---

## The 12 modules

<ul class="module-list">
  <li>
    <div class="module-entry">
      <strong>Databricks setup</strong>
      <span class="module-desc">Workspace, clusters, Unity Catalog, and cluster libraries for pricing workloads. The environment the rest of the course runs in.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>GLMs in Python</strong>
      <span class="module-desc">statsmodels Tweedie GLM with exposure offset, factor diagnostics, and lift charts. Everything Emblem does, in code.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>GBMs with CatBoost</strong>
      <span class="module-desc">Frequency/severity gradient boosting on freMTPL2, temporal walk-forward cross-validation, hyperparameter search.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>SHAP relativities</strong>
      <span class="module-desc">Extract GLM-style multiplicative factor tables from a CatBoost model. The format your rating engine and regulator expect.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Conformal prediction intervals</strong>
      <span class="module-desc">Distribution-free per-risk uncertainty bounds. Coverage guarantees that hold without distributional assumptions.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Credibility and Bayesian methods</strong>
      <span class="module-desc">Bühlmann-Straub in Python, PyMC hierarchical models for thin-data segments, mixed-model equivalence.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Rate optimisation</strong>
      <span class="module-desc">Constrained SLSQP to hit a loss ratio target while respecting movement caps. FCA ENBP enforcement. Efficient frontier between LR and volume.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>End-to-end pipeline</strong>
      <span class="module-desc">Raw tabular data to filed rates. MLflow tracking, model registry, and a signed-off audit trail in a single Databricks workflow.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Demand elasticity</strong>
      <span class="module-desc">Double ML causal estimation of own-price elasticity for renewal pricing. Deconfounding with CausalForestDML.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Interaction detection</strong>
      <span class="module-desc">SHAP interaction values, SHAP dependence, and partial dependence plots to find what your GLM missed.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Exposure curves</strong>
      <span class="module-desc">Increased limits factors, ILF credibility, and Lee diagram construction for commercial and home severity modelling.</span>
    </div>
  </li>
  <li>
    <div class="module-entry">
      <strong>Spatial territory rating</strong>
      <span class="module-desc">Geospatial smoothing with BYM2, Voronoi banding from postcode centroids, borough-level risk surfaces.</span>
    </div>
  </li>
</ul>

---

## What you receive

A zip file containing all twelve modules. No drip-feed, no access portal, no subscription.

<div class="includes-grid">
  <div class="include-card">
    <div class="include-card-title">Markdown tutorials</div>
    <div class="include-card-desc">Theory, implementation, and failure modes for each topic. Written for practitioners, not students.</div>
  </div>
  <div class="include-card">
    <div class="include-card-title">Jupyter notebooks</div>
    <div class="include-card-desc">Working <code>.ipynb</code> files. Run locally or on Databricks. Real data, real outputs.</div>
  </div>
  <div class="include-card">
    <div class="include-card-title">Exercises with solutions</div>
    <div class="include-card-desc">Problems that test the concepts, with worked solutions included.</div>
  </div>
  <div class="include-card">
    <div class="include-card-title">Bundled wheels</div>
    <div class="include-card-desc">Install the main dependencies without a PyPI connection. Useful if your Databricks environment is locked down.</div>
  </div>
</div>

No video. No subscription. You get everything at once.

---

## Free previews

These posts cover material from three of the modules:

<ul class="preview-posts">
  <li>
    <a href="/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/">Extracting Rating Relativities from GBMs with SHAP</a>
    <span class="preview-desc">A worked example of the SHAP relativities approach from Module 4 — pulling multiplicative factor tables out of a CatBoost model.</span>
  </li>
  <li>
    <a href="/2026/01/17/causal-price-elasticity-tutorial/">Your Elasticity Estimate Is Biased and You Already Know Why</a>
    <span class="preview-desc">The Double ML causal framework from Module 9 — deconfounding own-price elasticity for renewal pricing under GIPP constraints.</span>
  </li>
  <li>
    <a href="/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/">Conformal Prediction Intervals for Insurance Pricing Models</a>
    <span class="preview-desc">Coverage guarantees without distributional assumptions — the theory and implementation behind Module 5.</span>
  </li>
</ul>

---

<div class="course-cta-block">
  <a href="#" class="btn-buy">Buy Now — £97</a>
  <p class="refund-note">One-time payment. No subscription. 30-day no-questions refund policy — email the address on your receipt.</p>
</div>
