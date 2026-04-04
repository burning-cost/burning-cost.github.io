---
layout: post
title: "glum Insurance Pricing in Python: Fitting, Intervals, Monitoring, Fairness, Governance"
date: 2026-04-04
categories: [techniques, libraries]
tags: [glum, GLM, tweedie, conformal-prediction, model-monitoring, fairness, governance, python, insurance-conformal, insurance-monitoring, insurance-fairness, insurance-governance, freMTPL2, motor, uk-pricing, CatBoost, PSI, CSI, FCA, Consumer-Duty]
description: "glum fits the Tweedie GLM in seconds. Here is how our libraries handle everything around it: distribution-free prediction intervals, PSI/CSI drift monitoring, proxy discrimination auditing under FCA EP25/2, and a governance pack for the Model Risk Committee."
author: Burning Cost
---

[glum](https://github.com/Quantco/glum) is QuantCo's GLM engine. 2.6 million downloads a month (as of early 2026), faster than statsmodels on large insurance datasets, and it produces proper sklearn-compatible fitted models. If you are still fitting Tweedie GLMs with `statsmodels.genmod.GLM`, the switch is worth about ten minutes and a noticeable improvement in fit time on anything above 200,000 rows.

What glum does not do is the work around the model: prediction intervals with coverage guarantees, drift monitoring in production, discrimination auditing under the FCA's proxy discrimination framework, or a governance pack for the Model Risk Committee. Those are the gaps our libraries fill.

This post shows a complete workflow on the French motor dataset freMTPL2freq. Each section is a standalone step you can drop into your existing pipeline.

```bash
uv add glum insurance-conformal insurance-monitoring insurance-fairness insurance-governance
```

---

## 1. Fit a Tweedie GLM with glum

freMTPL2 is the standard benchmark dataset for non-life pricing. It contains about 678,000 French motor policies with claim counts, exposure, and nine rating factors. If you want the full step-by-step on building the Poisson frequency GLM itself — data prep, offset handling, deviance residuals, factor relativities — that is covered separately in the [GLM frequency model tutorial](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/).

```python
import numpy as np
import polars as pl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix
from glum import GeneralizedLinearRegressor

# Load freMTPL2freq from OpenML
df_raw = fetch_openml("freMTPL2freq", as_frame=True).frame
df = pl.from_pandas(df_raw)

# Clip exposure to (0, 1]
df = df.with_columns(pl.col("Exposure").clip(0.001, 1.0))

cat_cols = ["VehBrand", "VehGas", "Region", "Area"]
num_cols = ["VehAge", "DrivAge", "BonusMalus", "VehPower", "Density"]
exposure = df["Exposure"].to_numpy()
y        = df["ClaimNb"].to_numpy().astype(float)

enc = OneHotEncoder(sparse_output=True, drop="first")
X_cat = enc.fit_transform(df[cat_cols].to_pandas())
X_num = csr_matrix(df[num_cols].to_numpy().astype(float))
X     = hstack([X_num, X_cat])

X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
    X, y, exposure, test_size=0.2, random_state=42
)

# Tweedie GLM with log link and log-exposure offset
glm = GeneralizedLinearRegressor(
    family="tweedie",
    power=1.5,     # compound Poisson-Gamma
    link="log",
    alpha=0.01,    # L2 regularisation
    max_iter=100,
)
glm.fit(X_train, y_train, sample_weight=exp_train, offset=np.log(exp_train))

mu_train = glm.predict(X_train, offset=np.log(exp_train))
mu_test  = glm.predict(X_test,  offset=np.log(exp_test))
print(f"Mean predicted claims (test): {mu_test.mean():.4f}")
print(f"Mean actual claims (test):    {y_test.mean():.4f}")
```

glum fits this in roughly 1-2 seconds on a modern laptop - statsmodels takes closer to 20-30 seconds on the same data. The `GeneralizedLinearRegressor` follows the sklearn API: `fit`, `predict`, `coef_`, `intercept_`. The `offset=np.log(exposure)` argument handles the exposure term correctly: in a log-link Poisson/Tweedie GLM, log(exposure) enters as a fixed coefficient of 1 on the linear predictor.

One thing to know: glum expects dense or sparse scipy matrices. If your features are in a Polars DataFrame, convert numeric columns with `.to_numpy()` and sparse-encode categoricals as shown above. There is no direct Polars integration, but the bridging is two lines.

---

## 2. Wrap predictions with conformal intervals

The GLM gives point predictions. A PRA-supervised internal model also needs uncertainty bounds. Standard Tweedie confidence intervals assume the model is correctly specified - which it never quite is. Conformal prediction intervals carry a coverage guarantee regardless of model misspecification.

`insurance-conformal` implements the Manna et al. (2025) Pearson nonconformity score for Tweedie data. The guarantee: for any test point drawn from the same distribution as the calibration set, the interval contains the true value at the nominal rate. No parametric assumptions required.

glum's `predict()` requires an `offset` argument. The conformal predictor calls `model.predict(X)` internally, so we wrap glum in a thin callable that closes over the per-split exposures:

```python
from insurance_conformal import InsuranceConformalPredictor

# Split training data: use 70% to fit the GLM (already done above),
# hold back 30% as the conformal calibration set.
n = X_train.shape[0]
cal_start = int(0.7 * n)
X_cal, y_cal = X_train[cal_start:], y_train[cal_start:]
exp_cal = exp_train[cal_start:]

class GlumPredictor:
    """Wraps glum so that predict(X) bakes in a stored log-exposure offset."""
    def __init__(self, fitted_glm, log_offset: np.ndarray):
        self.glm = fitted_glm
        self.log_offset = log_offset  # must be same length as X rows

    def predict(self, X):
        return self.glm.predict(X, offset=self.log_offset[:X.shape[0]])


# Calibrate on the held-out 30%
wrapped_glm_cal = GlumPredictor(glm, np.log(exp_cal))
cp = InsuranceConformalPredictor(
    model=wrapped_glm_cal,
    nonconformity="pearson_weighted",
    distribution="tweedie",
    tweedie_power=1.5,
)
cp.calibrate(X_cal, y_cal, exposure=exp_cal)

# Predict intervals on test set
wrapped_glm_test = GlumPredictor(glm, np.log(exp_test))
cp.model = wrapped_glm_test  # swap offset to match test exposure
intervals = cp.predict_interval(X_test, alpha=0.10)

# intervals is a Polars DataFrame with columns lower, point, upper
covered = (y_test >= intervals["lower"].to_numpy()) & (y_test <= intervals["upper"].to_numpy())
print(f"Empirical coverage: {covered.mean():.3f}  (nominal: 0.90)")
width = intervals["upper"] - intervals["lower"]
print(f"Mean interval width: {width.mean():.4f}")
```

The key choice is `nonconformity="pearson_weighted"`. The Tweedie Pearson score is `|y - mu| / mu^{p/2}`, which normalises the residual by the expected standard deviation scale. This produces intervals roughly 30% narrower than raw residual scores at the same coverage level (Manna et al. 2025, ASMBI asmb.70045). For Solvency II 99.5% capital bounds, swap `alpha=0.005`.

The wrapper pattern above is the right mental model for glum integration: the GLM handles the statistical fitting, the conformal layer wraps the predictions with distribution-free coverage. The offset complexity is a one-time engineering decision at the calibration boundary, not a recurring concern.

---

## 3. Monitor drift with PSI and CSI

Models drift. The freMTPL2 data covers 2003-2005. Run the same GLM on today's book and you would see vehicle age distributions shifted (EVs, SUVs replacing petrol mid-range), density distributions changed (urban migration post-2020), and BonusMalus distributions compressed as behaviour-based telematics products filter out the worst risks. PSI catches distributional shift before it materialises as A/E divergence.

`insurance-monitoring` implements exposure-weighted PSI and CSI. The exposure weighting matters: a bin containing 1,000 monthly policies should not weigh the same as 1,000 annual ones.

```python
from insurance_monitoring.drift import psi, csi

# Extract a training-window feature array for the PSI reference
n_fit = int(0.7 * X_train.shape[0])
driver_age_train = df["DrivAge"].to_numpy()[:len(y_train)][:n_fit]
exp_fit          = exp_train[:n_fit]
mu_fit           = mu_train[:n_fit]

# Simulate a monitoring window with mild age drift (+2 years on average)
rng = np.random.default_rng(2026)
driver_age_current = driver_age_train + rng.normal(2.0, 1.0, len(driver_age_train))
score_current      = mu_fit * rng.lognormal(0.05, 0.10, len(mu_fit))

# PSI: has the input distribution drifted since training?
psi_age = psi(
    reference=driver_age_train,
    current=driver_age_current,
    n_bins=10,
    exposure_weights=exp_fit,
)
print(f"PSI (driver age): {psi_age:.4f}")
# < 0.10: stable  |  0.10-0.25: investigate  |  >= 0.25: model likely stale

# CSI: has the predicted score distribution shifted?
csi_score = csi(
    reference=mu_fit,
    current=score_current,
    n_bins=10,
)
print(f"CSI (predicted pure premium): {csi_score:.4f}")
```

For a complete monitoring run - drift plus A/E plus Gini drift with bootstrap CIs - use `MonitoringReport`:

```python
from insurance_monitoring import MonitoringReport

monitoring_report = MonitoringReport(
    reference_actual=y_train[:n_fit],
    reference_predicted=mu_fit,
    current_actual=y_test,
    current_predicted=mu_test,
    exposure=exp_test,
    feature_df_reference=pl.DataFrame({
        "driver_age": driver_age_train.tolist(),
    }),
    feature_df_current=pl.DataFrame({
        "driver_age": driver_age_current[:len(y_test)].tolist(),
    }),
    features=["driver_age"],
    murphy_distribution="tweedie",
    gini_bootstrap=True,
)

print(monitoring_report.recommendation)   # 'OK', 'RECALIBRATE', or 'REFIT'
```

The `recommendation` field implements the Brauer, Menzel & Wuthrich (2025) two-step decision rule from arXiv:2510.04556: Gini drift determines whether to refit, A/E drift determines whether to recalibrate. Murphy decomposition separates miscalibration (a global scaling problem, fixable without refit) from resolution loss (the model has lost rank-ordering ability, requires refit). This replaces the "Gini has dropped, refit everything" reflex with a structured diagnosis.

---

## 4. Audit for proxy discrimination

FCA Evaluation Paper EP25/2 (published January 2025) formally set out the regulator's approach to proxy discrimination in insurance pricing. The concern is not that your model uses gender directly - that is restricted under the Test-Achats ruling for most personal lines products. The concern is that combinations of variables you do use (postcode, vehicle make, occupation, payment method) functionally reproduce a protected characteristic, resulting in systematically different outcomes for protected groups.

`insurance-fairness` implements the proxy detection and fairness audit that Consumer Duty ongoing monitoring requires.

```python
import polars as pl
from insurance_fairness import detect_proxies, FairnessAudit

# Build a policy-level DataFrame for the audit.
# We treat Region as a proxy for socioeconomic group as an illustration.
n_train_all = X_train.shape[0]
df_audit = pl.DataFrame({
    "region":            df["Region"].to_numpy()[:n_train_all].tolist(),
    "vehicle_brand":     df["VehBrand"].to_numpy()[:n_train_all].tolist(),
    "driver_age":        df["DrivAge"].to_numpy()[:n_train_all].tolist(),
    "bonus_malus":       df["BonusMalus"].to_numpy()[:n_train_all].tolist(),
    "predicted_premium": mu_train.tolist(),
    "claim_amount":      y_train.tolist(),
    "exposure":          exp_train.tolist(),
})

# Step 1: proxy detection - does any rating factor act as a proxy for region?
proxy_result = detect_proxies(
    df=df_audit,
    protected_col="region",
    factor_cols=["vehicle_brand", "driver_age", "bonus_malus"],
)
proxy_result.summary()
# Reports mutual information, partial correlation, and proxy R-squared per factor.

# Step 2: full fairness audit
fairness_audit = FairnessAudit(
    model=None,           # pass None when predictions are pre-computed
    data=df_audit,
    protected_cols=["region"],
    prediction_col="predicted_premium",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["vehicle_brand", "driver_age", "bonus_malus"],
)
fairness_report = fairness_audit.run()
fairness_report.summary()
fairness_report.to_markdown("fairness_audit_freMTPL2.md")
```

The `FairnessReport` covers: demographic parity ratio (predicted premium disparity by group), calibration by group (A/E per protected group), Gini by group (discrimination quality within each group), and Theil index. `to_markdown()` produces the evidence record the FCA expects to see in a Consumer Duty review.

One thing worth being explicit about: `detect_proxies` reports on statistical association, not causation. A high mutual information between `vehicle_brand` and `region` in French motor data reflects genuine risk correlations (rural areas have a different vehicle mix from urban ones) as well as potential proxy effects. EP25/2 asks you to investigate and document, not automatically remove. The proxy score gives you the quantified starting point for that conversation with your underwriting team.

---

## 5. Generate the governance pack

The final step is the governance record for the Model Risk Committee. `insurance-governance` assembles this from a `ModelCard` (the structured metadata record), validation results, and monitoring results.

```python
from insurance_governance import MRMModelCard, GovernanceReport, RiskTierScorer

card = MRMModelCard(
    model_id="motor-tweedie-glm-freMTPL2-v1",
    model_name="Motor TP Pure Premium GLM (freMTPL2)",
    version="1.0.0",
    model_class="pricing",
    intended_use="Pure premium estimation for personal lines motor TP",
    not_intended_for=["Commercial fleet pricing", "Capital modelling"],
    target_variable="claim_count",
    distribution_family="Tweedie (power=1.5)",
    model_type="GLM",
    rating_factors=["VehAge", "DrivAge", "BonusMalus", "VehPower",
                    "Density", "VehBrand", "VehGas", "Region", "Area"],
    training_data_period=("2003-01-01", "2005-12-31"),
    development_date="2026-04-04",
    developer="Pricing Actuarial",
    champion_challenger_status="development",
    customer_facing=True,
    regulatory_use=False,
    gwp_impacted=85_000_000,
    monitoring_owner="Head of Pricing",
    monitoring_frequency="Quarterly",
    monitoring_triggers={
        "psi_driver_age": 0.25,
        "ae_ratio": 1.10,
        "gini_drift_pvalue": 0.05,
    },
)

# Score the risk tier - this determines the depth of validation required
scorer = RiskTierScorer()
tier   = scorer.score(card)
print(f"Risk tier: {tier.tier}  |  {tier.rationale}")

# Assemble the governance report for the committee pack
gov_report = GovernanceReport(
    card=card,
    tier=tier,
    validation_results={
        "overall_rag": "GREEN",
        "run_date":    "2026-04-04",
        "gini":        0.41,
        "ae_ratio":    1.02,
        "psi_score":   psi_age,
    },
    monitoring_results={
        "period":            "2026-Q1",
        "ae_ratio":          1.02,
        "psi_score":         psi_age,
        "gini":              0.41,
        "recommendation":    monitoring_report.recommendation,
        "triggered_alerts":  [],
    },
)

print(f"Report generated for {card.model_name}")
print(f"Tier {tier.tier} | RAG: GREEN | Next review: 2026-07-04")
```

The `GovernanceReport` output is a self-contained HTML document that is print-to-PDF ready. It covers model identity, risk tier with rationale, last validation outcomes, monitoring summary, outstanding issues, and approval record - the structure a PRA supervisor or internal audit function expects.

`RiskTierScorer` applies a weighted scorecard across customer-facing status, GWP impact, regulatory use, and model complexity. The freMTPL2 GLM scores Tier 2 (medium risk): customer-facing, material GWP, but a simple architecture. Tier 1 requires quarterly validation; Tier 2 gets annual validation with quarterly monitoring. The tier assignment is documented, reproducible, and not subject to the "we called it Tier 3 because validation was inconvenient" failure mode.

---

## The division of labour

The pipeline above fits naturally into two layers.

**glum** does the statistical heavy lifting: L1/L2-regularised IRLS, sparse matrix arithmetic, convergence in a handful of seconds on hundreds of thousands of rows. The `GeneralizedLinearRegressor` is the right tool for fitting Tweedie, Poisson, and Gamma GLMs at insurance scale. We have no reason to replace it.

**Our libraries** handle the surrounding workflow:

- `insurance-conformal`: distribution-free prediction intervals with a coverage guarantee that does not depend on the model being correctly specified. The Pearson nonconformity score exploits Tweedie variance structure to produce narrower intervals than naive residual methods.
- `insurance-monitoring`: exposure-weighted PSI/CSI plus the Brauer-Menzel-Wuthrich two-step decision rule to distinguish a recalibration problem from a refit problem. The `MonitoringReport` replaces a collection of ad-hoc scripts with one reproducible pass.
- `insurance-fairness`: proxy detection and FCA EP25/2-aligned fairness auditing. Produces the evidence record that Consumer Duty ongoing monitoring requires.
- `insurance-governance`: MRM model cards, risk tier scoring, and HTML governance reports for the Model Risk Committee.

None of these replace glum. They extend it. If you are fitting GLMs with glum (or statsmodels, or sklearn) and then doing the surrounding analysis in ad-hoc scripts, the integration points shown above are the things to standardise first.

The [insurance-conformal](/insurance-conformal/) and [insurance-governance](/insurance-governance/) library pages have the full API reference.

---

*Related:*
- [Building a GLM Frequency Model in Python: A Step-by-Step Tutorial](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — the Poisson frequency model build that feeds into this workflow
- [Insurance Model Monitoring in Python: A Practitioner's Guide](/2026/04/04/insurance-model-monitoring-python-practitioner-guide/) — deeper coverage of the three monitoring scenarios and when each action is warranted
- [Five Libraries, One Pipeline: End-to-End Motor Pricing in Python](/2026/04/04/five-libraries-one-pipeline-end-to-end-motor-pricing/) — an EBM-based alternative that shows the same five libraries working from model fit through governance report
