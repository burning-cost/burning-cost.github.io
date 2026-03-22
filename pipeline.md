---
layout: page
title: "A Complete Motor Pricing Pipeline in Python"
description: "From raw data to governance sign-off using seven Burning Cost libraries. Step-by-step walkthrough of a production motor pricing workflow — data with a known DGP, GBM training, SHAP factor extraction, GLM distillation, conformal prediction intervals, fairness audit, drift monitoring, and PRA SS1/23 governance pack."
permalink: /pipeline/
---

<style>
.pipeline-intro {
  font-size: 1.05rem;
  color: #2a2d3e;
  max-width: 720px;
  line-height: 1.7;
  margin-bottom: 2.5rem;
}

.pipeline-install {
  background: #0f1117;
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 2.5rem;
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.82rem;
  color: #cdd6f4;
  overflow-x: auto;
  white-space: pre;
  line-height: 1.7;
}

.pipeline-toc {
  background: #f9fafc;
  border: 1px solid #e2e6f0;
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 3rem;
}

.pipeline-toc h2 {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8892a4;
  margin: 0 0 0.75rem 0;
}

.pipeline-toc ol {
  margin: 0;
  padding-left: 1.25rem;
}

.pipeline-toc li {
  font-size: 0.88rem;
  color: #4a4f66;
  line-height: 1.8;
}

.pipeline-toc a {
  color: #4f7ef8;
  text-decoration: none;
}

.pipeline-toc a:hover {
  text-decoration: underline;
}

.pipeline-step {
  margin-bottom: 3.5rem;
}

.pipeline-step-label {
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
  margin-bottom: 0.6rem;
}

.pipeline-step h2 {
  font-size: 1.3rem;
  font-weight: 700;
  color: #0f1117;
  letter-spacing: -0.02em;
  margin-top: 0;
  margin-bottom: 0.75rem;
}

.pipeline-step p {
  font-size: 0.95rem;
  color: #2a2d3e;
  line-height: 1.75;
  max-width: 720px;
  margin-bottom: 1.1rem;
}

.pipeline-step p code,
.pipeline-step li code {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.83em;
  background: #f0f2f8;
  border: 1px solid #e2e6f0;
  border-radius: 4px;
  padding: 0.1em 0.35em;
  color: #0f1117;
}

.pipeline-step ul,
.pipeline-step ol {
  font-size: 0.92rem;
  color: #2a2d3e;
  line-height: 1.8;
  max-width: 680px;
  padding-left: 1.4rem;
  margin-bottom: 1rem;
}

.pipeline-divider {
  border: none;
  border-top: 1px solid #e2e6f0;
  margin: 3rem 0;
}

.pipeline-summary {
  background: #f9fafc;
  border: 1px solid #e2e6f0;
  border-radius: 12px;
  padding: 1.75rem 2rem;
  margin-bottom: 2rem;
}

.pipeline-summary h2 {
  font-size: 1.15rem;
  font-weight: 700;
  color: #0f1117;
  letter-spacing: -0.02em;
  margin-top: 0;
  margin-bottom: 1rem;
}

.pipeline-summary ul {
  margin: 0;
  padding-left: 0;
  list-style: none;
}

.pipeline-summary li {
  font-size: 0.9rem;
  color: #4a4f66;
  line-height: 1.6;
  padding: 0.45rem 0;
  border-bottom: 1px solid #e2e6f0;
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 0.75rem;
}

.pipeline-summary li:last-child {
  border-bottom: none;
}

.pipeline-summary li code {
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  font-size: 0.82em;
  font-weight: 600;
  color: #4f7ef8;
}

.pipeline-cta {
  margin-top: 2.5rem;
  padding-top: 2rem;
  border-top: 1px solid #e2e6f0;
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.pipeline-cta a {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  border-radius: 8px;
  font-size: 0.875rem;
  font-weight: 600;
  text-decoration: none;
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

@media (max-width: 640px) {
  .pipeline-summary li {
    grid-template-columns: 1fr;
    gap: 0.1rem;
  }
  .pipeline-install {
    font-size: 0.73rem;
  }
}
</style>

<p class="pipeline-intro">Most pricing tutorials show you one thing: how to fit a GLM, or how to train a GBM, or how to calculate a Gini coefficient. This is not that. This is the full workflow — from raw data to governance sign-off — using seven libraries that were built to work together. If you are a pricing actuary who wants to see what a properly instrumented model pipeline looks like before you commit to building one yourself, this is the page.</p>

<p class="pipeline-intro" style="margin-top: -1.5rem;">The full runnable script is <a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/end_to_end_demo.py">end-to-end-demo.py</a>. It runs on Databricks Free Edition, single-node Python, no Spark. Runtime is roughly 3–5 minutes.</p>

<div class="pipeline-install">pip install insurance-datasets "shap-relativities[ml]" "insurance-distill[catboost]" \
    "insurance-conformal[catboost]" insurance-fairness \
    insurance-monitoring insurance-governance</div>

<div class="pipeline-toc">
  <h2>Contents</h2>
  <ol>
    <li><a href="#step-1">Data you can actually test against</a></li>
    <li><a href="#step-2">Rating factors from a GBM</a></li>
    <li><a href="#step-3">A GLM the rating engine can load</a></li>
    <li><a href="#step-4">Prediction intervals that actually hold</a></li>
    <li><a href="#step-5">Fairness audit before anything goes to production</a></li>
    <li><a href="#step-6">Monitoring that catches drift before the loss ratio does</a></li>
    <li><a href="#step-7">Governance sign-off</a></li>
  </ol>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-1">
  <div class="pipeline-step-label">Step 1</div>
  <h2>Data you can actually test against</h2>

  <p>The hardest problem in developing pricing models is that you never know if your implementation is right. Real policyholder data has unknown true coefficients. You can fit a GLM and get plausible-looking numbers, but you cannot tell whether those numbers are correct or whether they are absorbing some omitted variable.</p>

  <p><code>insurance-datasets</code> gives you synthetic UK motor data where the true parameters are published. The frequency model is Poisson with a log-linear predictor; the exact coefficients are in <code>MOTOR_TRUE_FREQ_PARAMS</code>. If your model recovers <code>ncd_years = -0.12</code> and you fitted it on 50,000 policies and got <code>-0.125</code>, that is correct. If you got <code>-0.18</code>, something is wrong with your specification.</p>

```python
from insurance_datasets import load_motor, MOTOR_TRUE_FREQ_PARAMS

df = load_motor(n_policies=50_000, seed=42)
print(MOTOR_TRUE_FREQ_PARAMS)
# {'intercept': -3.2, 'vehicle_group': 0.025, 'ncd_years': -0.12, ...}
```

  <p>The split is temporal, not random. Train on 2019–2022, validate on 2023. Insurance data is not exchangeable across time — validating on a random shuffle inflates your Gini and you will not find out until the model goes live.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-2">
  <div class="pipeline-step-label">Step 2</div>
  <h2>Rating factors from a GBM</h2>

  <p>The standard complaint about GBMs in pricing is that you cannot get the factor table out. The regulator wants <code>exp(beta)</code> — a number you can defend in a rate filing. The GBM gives you a black box.</p>

  <p><code>shap-relativities</code> closes that gap. For a Poisson model with log link, SHAP values are additive in log space. That means you can compute the exposure-weighted mean SHAP value for each level of each feature and exponentiate the difference — which is exactly the same operation as computing <code>exp(beta_k - beta_ref)</code> from a GLM coefficient.</p>

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(
    model=gbm,
    X=X_train,
    exposure=train["exposure"],
    categorical_features=["area_code", "ncd_years", "has_convictions", "vehicle_group"],
)
sr.fit()
rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0, "vehicle_group": 1},
)
```

  <p>The output is one row per feature level, with columns for relativity, confidence interval lower, and upper. The same format your rating engine expects for a factor table import.</p>

  <p>Always call <code>sr.validate()</code> before trusting the output. The reconstruction check verifies that the SHAP decomposition is numerically exact. If it fails, you have a mismatch between the model objective and the SHAP output type.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-3">
  <div class="pipeline-step-label">Step 3</div>
  <h2>A GLM the rating engine can load</h2>

  <p>SHAP relativities are descriptive. You can inspect them and export them as a CSV, but you cannot load a GBM into Radar or Emblem. If your rating engine requires a multiplicative GLM, you need distillation.</p>

  <p><code>insurance-distill</code> fits a Poisson GLM using the GBM's predictions as the target rather than the raw claims. This matters: fitting on GBM pseudo-predictions eliminates the noise from individual claim events. The GBM has already smoothed that away. The result is a surrogate GLM that retains 90–97% of the GBM's Gini coefficient on typical UK motor books.</p>

```python
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=gbm, X_train=X_train, y_train=train["claim_count"].to_numpy(),
    exposure=train["exposure"].to_numpy(), family="poisson",
)
surrogate.fit(max_bins=10, method_overrides={"ncd_years": "isotonic"})

report = surrogate.report()
print(report.metrics.summary())
```

  <p>The isotonic override for <code>ncd_years</code> is deliberate. NCD discount should be monotone — more years of no claims means lower frequency, without exception. Isotonic regression enforces that constraint directly. The tree-based default would find the statistically optimal bins but might produce a non-monotone step at NCD=4 if the data happens to support one. That is not a result you can defend to a pricing committee.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-4">
  <div class="pipeline-step-label">Step 4</div>
  <h2>Prediction intervals that actually hold</h2>

  <p>Point estimates are not enough for pricing decisions involving large limits, reinsurance attachment points, or capital allocation. You need uncertainty quantification. The question is whether your intervals actually achieve their stated coverage levels.</p>

  <p>Parametric intervals for insurance data typically fail in the high-risk tail. The assumption is that all risks have the same coefficient of variation. High-risk policies — young drivers, high vehicle groups — have genuinely higher dispersion than the parametric model predicts. The result: the model overcaters low-risk policies (unnecessarily wide intervals) and barely meets the coverage target for top-decile risks.</p>

  <p>Conformal prediction is distribution-free. The coverage guarantee holds regardless of the true data distribution and regardless of model misspecification. The <code>pearson_weighted</code> score accounts for the <code>Var(Y) ~ mu</code> relationship in Poisson data, which is why intervals are 13% narrower than the parametric alternative without sacrificing coverage.</p>

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=gbm, nonconformity="pearson_weighted",
    distribution="tweedie", tweedie_power=1.0,
)
cp.calibrate(X_cal.to_pandas(), cal_mask["claim_count"].to_numpy())
intervals = cp.predict_interval(X_val.to_pandas(), alpha=0.10)
```

  <p>Two things matter for getting the coverage guarantee to hold. First, the calibration set must not overlap with training — we use the 2022 accident year, held out from the GBM training. Second, calibrate on data that is temporally close to the test period. Conformal prediction requires exchangeability, and a 2019 calibration set is not exchangeable with a 2023 test set after claims inflation.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-5">
  <div class="pipeline-step-label">Step 5</div>
  <h2>Fairness audit before anything goes to production</h2>

  <p>The FCA's Consumer Duty (PS22/9) requires firms to evidence fair value by customer group. The FCA's 2024 thematic review found most insurers' fair value assessments were, in their words, "high-level summaries with little substance." Six Consumer Duty investigations followed.</p>

  <p>The mechanism creating fair value failures in motor is proxy discrimination. Area band — which you almost certainly have in your model — correlates with ethnicity because urban postcodes are more diverse. You are not modelling ethnicity. But if area band is systematically correlated with ethnicity, and area band drives price, the Equality Act Section 19 prohibition on indirect discrimination is a live concern regardless of your intent.</p>

  <p><code>insurance-fairness</code> checks whether this is happening in your specific book:</p>

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=gbm,
    data=val_with_preds,
    protected_cols=["area_band"],
    prediction_col="predicted_rate",
    outcome_col="incurred",
    exposure_col="exposure",
    factor_cols=["area_band", "ncd_years", "has_convictions", "vehicle_group"],
    model_name="Motor Frequency GBM v1.0",
    run_proxy_detection=True,
)
report = audit.run()
report.to_markdown("fairness_audit.md")
```

  <p>The primary test is calibration by group. If the model's A/E ratio is 1.0 for urban and rural areas at every predicted pricing level, any premium differences reflect genuine risk differences and are defensible under the proportionality test. The proxy detection module flags factors where a CatBoost model can predict the protected characteristic with R-squared above 0.05 — the threshold where it is worth investigating further. It also runs mutual information scores to catch non-linear relationships that R-squared misses.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-6">
  <div class="pipeline-step-label">Step 6</div>
  <h2>Monitoring that catches drift before the loss ratio does</h2>

  <p>Deployed models go stale. The standard approach is to track aggregate A/E — total actual claims divided by total expected — and investigate if it moves outside a band. The problem is that errors cancel. A model that is 15% cheap on under-25s and 15% expensive on over-65s reads 1.00 at portfolio level and nobody raises an alarm.</p>

  <p><code>insurance-monitoring</code> monitors the features, not just the headline number. PSI per rating factor detects distributional shift before it shows up in A/E. The Gini drift z-test — implemented from arXiv:2510.04556 — answers the question that A/E cannot: has the model's ranking degraded? This is the difference between a recalibration (hours of work) and a refit (weeks of work).</p>

```python
from insurance_monitoring import MonitoringReport

monitoring = MonitoringReport(
    reference_actual=act_ref, reference_predicted=pred_ref,
    current_actual=act_cur, current_predicted=pred_cur,
    feature_df_reference=feat_ref, feature_df_current=feat_cur,
    features=["driver_age", "vehicle_group", "ncd_years", "area_code"],
    murphy_distribution="poisson",
)
print(monitoring.recommendation)
# 'NO_ACTION' | 'RECALIBRATE' | 'REFIT' | 'INVESTIGATE'
```

  <p>The Murphy decomposition sharpens the RECALIBRATE/REFIT decision. It decomposes miscalibration into a global component (fixed by multiplying all predictions by A/E — a recalibration) and a local component (requires rebuilding the model). If local MCB exceeds global MCB, the ranking is broken and you need a refit. A/E alone cannot tell you this.</p>
</div>

<hr class="pipeline-divider">

<div class="pipeline-step" id="step-7">
  <div class="pipeline-step-label">Step 7</div>
  <h2>Governance sign-off</h2>

  <p>Before the model goes to production, you need documented validation evidence and a risk tier assessment. <code>insurance-governance</code> automates both.</p>

  <p><code>ModelValidationReport</code> runs the standard test suite — Gini with bootstrap confidence interval, lift chart, A/E by predicted decile with Poisson CI, Hosmer-Lemeshow goodness-of-fit, PSI — and produces a self-contained HTML report. Each test returns a <code>TestResult</code> with a pass/fail flag, severity level, and a human-readable detail string. The report is printable as a PDF and goes directly into the model validation pack.</p>

  <p><code>RiskTierScorer</code> assigns a risk tier from objective criteria: GWP impacted, model complexity, deployment status, regulatory use, external data, customer-facing flag. The 0–100 composite score has documented rules for every point. This removes the subjectivity from MRC presentations — you are not arguing about whether the model is "complex", you are presenting a score with documented methodology.</p>

```python
from insurance_governance import (
    ModelValidationReport, ValidationModelCard,
    MRMModelCard, RiskTierScorer, GovernanceReport,
)

val_report = ModelValidationReport(
    model_card=ValidationModelCard(name="Motor Frequency GBM v1.0", ...),
    y_val=act_cur, y_pred_val=pred_cur, exposure_val=val["exposure"].to_numpy(),
)
val_report.generate("validation_report.html")

tier = RiskTierScorer().score(
    gwp_impacted=85_000_000, model_complexity="high",
    deployment_status="champion", customer_facing=True,
)
GovernanceReport(card=MRMModelCard(...), tier=tier).save_html("mrm_pack.html")
```
</div>

<hr class="pipeline-divider">

<div class="pipeline-summary">
  <h2>What this pipeline gives you</h2>
  <p style="font-size: 0.92rem; color: #4a4f66; margin-bottom: 1.25rem; line-height: 1.7;">Seven libraries, one workflow. Each one solves a specific problem that the others do not:</p>
  <ul>
    <li><code>insurance-datasets</code> <span>Known-DGP test environment so you can verify implementations rather than assume they work.</span></li>
    <li><code>shap-relativities</code> <span>Makes GBM predictions interpretable in the format pricing teams and regulators expect.</span></li>
    <li><code>insurance-distill</code> <span>Converts the GBM into something a rating engine can actually load.</span></li>
    <li><code>insurance-conformal</code> <span>Quantifies uncertainty without parametric assumptions that fail on heterogeneous motor books.</span></li>
    <li><code>insurance-fairness</code> <span>Produces the documented evidence trail that FCA Consumer Duty requires.</span></li>
    <li><code>insurance-monitoring</code> <span>Catches the segment-level drift that aggregate A/E misses.</span></li>
    <li><code>insurance-governance</code> <span>Automates the test suite and governance pack that sign-off requires.</span></li>
  </ul>
</div>

<div class="pipeline-cta">
  <a href="https://github.com/burning-cost/burning-cost-examples/blob/main/notebooks/end_to_end_demo.py" target="_blank" class="btn-primary-sm">View full demo script</a>
  <a href="/tools/" class="btn-outline-sm">Browse all libraries</a>
  <a href="/getting-started/" class="btn-outline-sm">Getting started</a>
  <a href="/guide/" class="btn-outline-sm">Which library do I need?</a>
</div>
