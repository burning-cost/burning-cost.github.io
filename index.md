---
layout: home
---

<div class="bc-hero">
<div class="bc-hero-inner">
<div class="bc-hero-label">Open source - UK insurance pricing</div>
<h1>The GBM works.<br>The problem is everything around it.</h1>
<p class="bc-hero-sub">13 Python libraries covering the full pricing workflow - from temporally-correct cross-validation to constrained rate optimisation. Written for teams that already know what a GLM is and are tired of rebuilding the same infrastructure from scratch.</p>
<div class="bc-cta-row">
  <a class="bc-btn bc-btn-primary" href="/course/">Training course</a>
  <a class="bc-btn bc-btn-ghost" href="https://github.com/burningcost" target="_blank">GitHub</a>
</div>
</div>
</div>

<div class="bc-stats-strip">
  <div class="bc-stat">
    <span class="bc-stat-number">13</span>
    <span class="bc-stat-label">libraries</span>
  </div>
  <div class="bc-stat-divider"></div>
  <div class="bc-stat">
    <span class="bc-stat-number">600+</span>
    <span class="bc-stat-label">actuarial tests</span>
  </div>
  <div class="bc-stat-divider"></div>
  <div class="bc-stat">
    <span class="bc-stat-number">8</span>
    <span class="bc-stat-label">training modules</span>
  </div>
  <div class="bc-stat-divider"></div>
  <div class="bc-stat">
    <span class="bc-stat-number">14</span>
    <span class="bc-stat-label">technical posts</span>
  </div>
</div>

<div class="bc-problem">
<p>Most UK pricing teams are running GBMs in development and GLMs in production. The GBM outperforms, but its outputs - raw SHAP values, unconstrained predictions, no factor tables - are not in a form a rating engine, regulator, or pricing committee will accept. The missing piece is not technical skill. It is tooling.</p>
<p class="bc-problem-solution">Each library here solves one specific gap in that pipeline. All ship with actuarial tests and produce outputs in formats pricing teams already recognise.</p>
</div>

<div class="bc-featured">
<p class="bc-section-title">Start here</p>
<div class="bc-featured-grid">

  <div class="bc-featured-card">
    <div class="bc-featured-tag">End-to-end workflow</div>
    <h3><a href="/2026/03/06/from-gbm-to-radar-databricks-workflow.html">From GBM to Radar: A Complete Databricks Workflow</a></h3>
    <p>The full pipeline in one place: ingestion, walk-forward CV, CatBoost, SHAP relativities, conformal intervals, rate optimisation. Adapt it directly.</p>
    <div class="bc-card-arrow">Read &rarr;</div>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Core technique</div>
    <h3><a href="/2026/03/05/extracting-rating-relativities-from-gbms-with-shap.html">Extracting Rating Relativities from GBMs with SHAP</a></h3>
    <p>SHAP values are not factor tables. This shows you how to get from one to the other - with confidence intervals, exposure weighting, and a reconstruction test your pricing committee can check.</p>
    <div class="bc-card-arrow">Read &rarr;</div>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Validation</div>
    <h3><a href="/2026/03/06/why-your-cross-validation-is-lying-to-you.html">Why Your Cross-Validation is Lying to You</a></h3>
    <p>Standard k-fold gives you a flattering Gini. Walk-forward splits with IBNR buffers give you an honest one. The difference is not subtle.</p>
    <div class="bc-card-arrow">Read &rarr;</div>
  </div>

  <div class="bc-featured-card">
    <div class="bc-featured-tag">Commercial pricing</div>
    <h3><a href="/2026/03/06/constrained-rate-optimisation-efficient-frontier.html">Constrained Rate Optimisation and the Efficient Frontier</a></h3>
    <p>If you are optimising rates with movement caps in a spreadsheet, you are probably leaving money on the table. This formulates it as a linear programme and shows you what the frontier looks like.</p>
    <div class="bc-card-arrow">Read &rarr;</div>
  </div>

</div>
</div>

<div class="bc-course">
<div class="bc-course-inner">
<div class="bc-course-label">Training course</div>
<h2>Modern Insurance Pricing with Python and Databricks</h2>
<p class="bc-course-desc">Eight modules written specifically for pricing actuaries and analysts at UK personal lines insurers. Each one covers a real pricing problem - not a generic data science tutorial with an insurance example bolted on at the end.</p>
<div class="bc-course-modules">Databricks for pricing teams &middot; GLMs in Python (the bridge from Emblem) &middot; GBMs for insurance &middot; SHAP relativities &middot; Conformal prediction intervals &middot; Credibility and Bayesian pricing &middot; Constrained rate optimisation &middot; End-to-end pipeline capstone</div>
<div class="bc-course-pricing">
  <div class="bc-price-item">
    <span class="bc-price-amount">&pound;295</span>
    <span class="bc-price-label">MVP bundle</span>
  </div>
  <div class="bc-price-item bc-price-item--featured">
    <span class="bc-price-amount">&pound;495</span>
    <span class="bc-price-label">Full course</span>
  </div>
  <div class="bc-price-item">
    <span class="bc-price-amount">&pound;79</span>
    <span class="bc-price-label">Per module</span>
  </div>
</div>
<div class="bc-cta-row">
  <a class="bc-btn bc-btn-primary" href="/course/">See the full curriculum</a>
</div>
</div>
</div>

<div class="bc-repos">
<p class="bc-section-title">Libraries</p>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Model interpretation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/shap-relativities" target="_blank">shap-relativities</a>
    <p>Converts CatBoost SHAP values into multiplicative factor tables - the format a pricing committee expects. Includes confidence intervals, exposure weighting, and reconstruction validation so you can defend the output.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Validation</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-cv" target="_blank">insurance-cv</a>
    <p>Temporally-correct cross-validation that respects development lags. Walk-forward splits with configurable IBNR buffers prevent your Gini from lying to you. Sklearn-compatible, so it drops into existing pipelines.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-conformal" target="_blank">insurance-conformal</a>
    <p>Distribution-free prediction intervals for insurance GBMs. The variance-weighted non-conformity score from Manna et al. (2025) gives roughly 30% narrower intervals than the naive approach - with identical coverage guarantees.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Techniques</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/credibility" target="_blank">credibility</a>
    <p>Buhlmann-Straub credibility in Python, with mixed-model equivalence checks. Useful for capping thin segments, stabilising NCD factors, and blending a new model with an incumbent rate at review.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/bayesian-pricing" target="_blank">bayesian-pricing</a>
    <p>Hierarchical Bayesian models for segments where you do not have enough data to fit a factor directly. Partial pooling across risk groups, with credibility factor output that maps back to traditional actuarial review formats.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-interactions" target="_blank">insurance-interactions</a>
    <p>Finds and quantifies the interaction effects that a main-effects-only GLM cannot see. Useful for understanding where your GBM is doing something your GLM cannot replicate.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-causal" target="_blank">insurance-causal</a>
    <p>Separates genuine risk signal from confounded association. Relevant wherever rating factors correlate with distribution channel or policyholder behaviour - which is most books of business.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-spatial" target="_blank">insurance-spatial</a>
    <p>Geographically smoothed postcode relativities using BYM2 models. Borrows strength across adjacent areas to stabilise thin-data postcodes in home and motor models.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Commercial</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/rate-optimiser" target="_blank">rate-optimiser</a>
    <p>Optimises rate changes across rating cells subject to movement caps, target loss ratio, and cross-subsidy constraints simultaneously - as a linear programme, not a spreadsheet iteration. Produces the efficient frontier so you can see the trade-off.</p>
  </li>
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-demand" target="_blank">insurance-demand</a>
    <p>Price elasticity and conversion modelling for pricing decisions. Own-price and cross-price effects, with direct integration into the rate optimiser so demand response feeds back into the objective.</p>
  </li>
</ul>
</div>

<div class="bc-repo-group">
<div class="bc-repo-group-label">Compliance</div>
<ul class="bc-repo-list">
  <li class="bc-repo-item">
    <a href="https://github.com/burningcost/insurance-fairness" target="_blank">insurance-fairness</a>
    <p>Proxy discrimination detection and fairness-accuracy trade-off analysis for pricing models. Produces documentation in the format FCA Consumer Duty review requires, not just a p-value.</p>
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
  <div class="bc-newsletter-inner">
    <div class="bc-newsletter-copy">
      <h3>New work, by email</h3>
      <p>When a new library ships or a post worth reading goes up, we send one email. No roundups, no digest spam - only when something substantive is ready.</p>
    </div>
    <a class="bc-btn bc-btn-primary" href="mailto:pricing.frontier@gmail.com?subject=Subscribe%3A%20Burning%20Cost%20updates">Subscribe</a>
  </div>
</div>
