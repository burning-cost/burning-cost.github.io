---
layout: home
---

<div class="bc-hero">
<div class="bc-hero-label">Open source - UK insurance pricing</div>
<h1>The GBM works.<br>The problem is everything around it.</h1>
<p class="bc-hero-sub">13 Python libraries covering the full pricing workflow - from temporally-correct cross-validation to constrained rate optimisation. Built for teams that already know what a GLM is.</p>
<p class="bc-hero-who">Practitioner-written. Actuarially tested. Designed to produce outputs a pricing committee will actually accept.</p>
<div class="bc-cta-row">
  <a class="bc-btn bc-btn-primary" href="/course/">Training course</a>
  <a class="bc-btn" href="https://github.com/burningcost" target="_blank">GitHub</a>
</div>
</div>

<div class="bc-stats">
  <div class="bc-stat">
    <span class="bc-stat-number">13</span>
    <span class="bc-stat-label">Python libraries</span>
  </div>
  <div class="bc-stat">
    <span class="bc-stat-number">600+</span>
    <span class="bc-stat-label">Actuarial tests</span>
  </div>
  <div class="bc-stat">
    <span class="bc-stat-number">14</span>
    <span class="bc-stat-label">Technical posts</span>
  </div>
  <div class="bc-stat">
    <span class="bc-stat-number">8</span>
    <span class="bc-stat-label">Course modules</span>
  </div>
</div>

<div class="bc-problem">
<p>Most UK pricing teams have adopted GBMs but are still taking GLM outputs to production. The GBM sits on a server outperforming the production model, but its outputs are not in a form that a rating engine, regulator, or pricing committee can work with. The missing piece is not technical skill - it is the tooling that bridges the two.</p>
<p>That is what we build here. Each library solves one specific problem in the pricing workflow, ships with actuarial tests, and produces outputs in formats that pricing teams already recognise.</p>
</div>

<div class="bc-featured">
<p class="bc-section-title">Start here</p>
<div class="bc-featured-grid">

  <div class="bc-featured-card">
    <div class="bc-featured-tag">End-to-end workflow</div>
    <h3><a href="/2026/03/06/from-gbm-to-radar-databricks-workflow.html">From GBM to Radar: A Complete Databricks Workflow for Pricing Actuaries</a></h3>
    <p>Data ingestion, walk-forward CV, CatBoost training, SHAP relativities, conformal intervals, and rate optimisation - one notebook, one pipeline, ready to adapt.</p>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Core technique</div>
    <h3><a href="/2026/03/05/extracting-rating-relativities-from-gbms-with-shap.html">Extracting Rating Relativities from GBMs with SHAP</a></h3>
    <p>How to turn SHAP values into multiplicative factor tables in the format a GLM reviewer expects - with confidence intervals, exposure weighting, and reconstruction validation.</p>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Validation</div>
    <h3><a href="/2026/03/06/why-your-cross-validation-is-lying-to-you.html">Why Your Cross-Validation is Lying to You</a></h3>
    <p>Standard k-fold CV is wrong for insurance data. Walk-forward splits with IBNR buffers are not optional - they are the difference between an honest and a flattering Gini.</p>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Commercial pricing</div>
    <h3><a href="/2026/03/06/constrained-rate-optimisation-efficient-frontier.html">Constrained Rate Optimisation and the Efficient Frontier</a></h3>
    <p>Formulating rate changes as a linear programme that respects movement caps, target loss ratio, and cross-subsidy constraints simultaneously - and why most teams do not.</p>
  </div>

</div>
</div>

<div class="bc-course">
<div class="bc-course-label">Training course</div>
<h2>Modern Insurance Pricing with Python and Databricks</h2>
<p class="bc-course-desc">Eight modules written for pricing actuaries and analysts at UK personal lines insurers. Every module covers a real pricing problem - not a generic data science tutorial adapted for insurance.</p>
<div class="bc-course-modules">Databricks for pricing teams &middot; GLMs in Python (the bridge from Emblem) &middot; GBMs for insurance &middot; SHAP relativities &middot; Conformal prediction intervals &middot; Credibility and Bayesian pricing &middot; Constrained rate optimisation &middot; End-to-end pipeline capstone</div>
<div class="bc-course-pricing">
  <div class="bc-price-item">
    <span class="bc-price-amount">&pound;295</span>
    <span class="bc-price-label">MVP bundle</span>
  </div>
  <div class="bc-price-item">
    <span class="bc-price-amount">&pound;495</span>
    <span class="bc-price-label">Full course</span>
  </div>
  <div class="bc-price-item">
    <span class="bc-price-amount">&pound;79</span>
    <span class="bc-price-label">Per module</span>
  </div>
</div>
<div class="bc-cta-row">
  <a class="bc-btn" href="/course/">See the full curriculum</a>
</div>
</div>

<div class="bc-repos">
<p class="bc-section-title">Libraries</p>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Model interpretation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/shap-relativities" target="_blank">shap-relativities</a>
    <p>Extract multiplicative rating factor tables from CatBoost models using SHAP values. Factor tables, confidence intervals, exposure weighting, reconstruction validation.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Validation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-cv" target="_blank">insurance-cv</a>
    <p>Temporally-correct cross-validation for insurance pricing models. Walk-forward splits with configurable IBNR buffers, Poisson and Gamma deviance scorers, sklearn-compatible API.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-conformal" target="_blank">insurance-conformal</a>
    <p>Distribution-free prediction intervals for insurance GBMs. Implements the variance-weighted non-conformity score from Manna et al. (2025) - roughly 30% narrower than the naive approach with identical coverage guarantees.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Techniques</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/credibility" target="_blank">credibility</a>
    <p>Buhlmann-Straub credibility in Python, with mixed-model equivalence checks. Practical for capping thin segments, stabilising NCD factors, and blending a new model with an incumbent rate.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/bayesian-pricing" target="_blank">bayesian-pricing</a>
    <p>Hierarchical Bayesian models for thin-data pricing segments. Partial pooling across risk groups, with credibility factor output in a format that maps back to traditional actuarial review.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-interactions" target="_blank">insurance-interactions</a>
    <p>Detecting, quantifying, and presenting interaction effects in insurance pricing models - the effects a main-effects-only GLM cannot see.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-causal" target="_blank">insurance-causal</a>
    <p>Causal inference methods for insurance pricing. Separating genuine risk signal from confounded association - relevant wherever rating factors are correlated with distribution channel or policyholder behaviour.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-spatial" target="_blank">insurance-spatial</a>
    <p>Spatial territory ratemaking using BYM2 models. Geographically smoothed relativities that borrow strength across adjacent areas - useful for postcode-level home and motor models with thin data.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Commercial</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/rate-optimiser" target="_blank">rate-optimiser</a>
    <p>Constrained rate change optimisation for UK personal lines. Formulates the efficient frontier between loss ratio improvement and movement cap constraints as a linear programme.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-demand" target="_blank">insurance-demand</a>
    <p>Demand and conversion modelling for insurance pricing. Price elasticity curves, own-price and cross-price effects, integration with rate optimisation.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Compliance</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-fairness" target="_blank">insurance-fairness</a>
    <p>Proxy discrimination detection for insurance pricing models. Measures of disparate impact, fairness-accuracy trade-off analysis, FCA Consumer Duty documentation support.</p>
  </li>
</ul>
</div>

</div>

<div class="bc-posts">
<p class="bc-section-title">Recent posts</p>
<ul class="bc-post-list">
{% for post in site.posts limit:8 %}
  <li class="bc-post-item">
    <span class="bc-post-date">{{ post.date | date: "%d %b %Y" }}</span>
    <a href="{{ post.url }}">{{ post.title }}</a>
  </li>
{% endfor %}
</ul>
</div>

<div class="bc-newsletter">
  <div class="bc-newsletter-copy">
    <h3>Stay up to date</h3>
    <p>New libraries and posts go out by email. No digest spam - only when something substantive ships.</p>
  </div>
  <a class="bc-btn bc-btn-primary" href="mailto:pricing.frontier@gmail.com?subject=Subscribe%3A%20Burning%20Cost%20updates">Join the mailing list</a>
</div>
