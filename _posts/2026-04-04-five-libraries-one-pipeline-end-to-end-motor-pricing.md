---
layout: post
title: "Five Libraries, One Pipeline: End-to-End Motor Pricing in Python"
date: 2026-04-04
featured: true
categories: [techniques]
tags: [insurance-gam, insurance-conformal, insurance-monitoring, insurance-fairness, insurance-governance, motor, ebm, python, freMTPL2, pipeline, ecosystem]
description: "A single freMTPL2 motor pipeline running through insurance-gam, insurance-conformal, insurance-monitoring, insurance-fairness, and insurance-governance. No other open-source ecosystem does all five."
author: Burning Cost
---

Every piece of pricing tooling solves one problem well. Emblem does GLM fitting. Radar does GLM fitting with a rating engine attached. Akur8 builds transparent ML inside its own SaaS environment. Python's scikit-learn gives you models but nothing above the prediction layer. The standard open-source motor pricing workflow in Python is: fit a CatBoost model, produce a Lorenz curve, call it done.

That leaves a lot unfinished. You have a point estimate with no uncertainty bounds. No drift detection when the portfolio shifts. No audit trail for proxy discrimination. No governance pack for the model risk committee. These are not optional extras for a UK insurer — Consumer Duty, Solvency II Article 120, and the FCA's expectations around fair value and proxy discrimination all bite. The tools to address them just did not exist in a coherent Python ecosystem.

We built five libraries to change that. This post shows them working together on a single motor pricing pipeline, from model training to governance report, using the `freMTPL2` dataset throughout. freMTPL2 is French, not UK, but the workflow transfers directly to a UK personal lines motor book.

The five libraries are `insurance-gam`, `insurance-conformal`, `insurance-monitoring`, `insurance-fairness`, and `insurance-governance`. Install them:

```bash
uv add "insurance-gam[ebm]" insurance-conformal insurance-monitoring insurance-fairness insurance-governance
```

---

## The data

freMTPL2 is a French third-party liability dataset with around 678,000 policies. It has exactly the rating variables a UK actuarial audience would recognise: driver age (`DrivAge`), vehicle age (`VehAge`), vehicle power (`VehPower`), region (`Region`), density (`Density`), and earned exposure in years (`Exposure`). Claims count (`ClaimNb`) with exposure gives you a frequency model.

```python
import numpy as np
import polars as pl
from sklearn.datasets import fetch_openml

raw = fetch_openml(name="freMTPL2freq", as_frame=True)
df = pl.from_pandas(raw.data).with_columns([
    pl.col("ClaimNb").cast(pl.Int32),
    pl.col("Exposure").cast(pl.Float64),
])

# Non-overlapping 60/20/20 split.
# Temporal ordering would be correct in production; freMTPL2 has no date field,
# so we use a deterministic permutation.
n = len(df)
idx = np.random.default_rng(42).permutation(n)
train = df[idx[:int(0.6 * n)]]
cal   = df[idx[int(0.6 * n):int(0.8 * n)]]
test  = df[idx[int(0.8 * n):]]
```

We hold back the calibration set for conformal calibration (Step 2). The test set never touches any fitting or calibration step.

---

## Step 1: train an EBM frequency tariff with `insurance-gam`

An EBM (Explainable Boosting Machine) is a generalised additive model with pairwise interaction terms. The shape functions are fitted by coordinate-wise boosting, one feature at a time, which prevents the overfitting that kills standard GAMs on large datasets. The key actuarial property: every rating factor contributes an additive term to the log-rate, and you can read the shape function directly as a relativity curve. The model is not a black box.

We use `InsuranceEBM`, which wraps interpretML's `ExplainableBoostingRegressor` with actuarial extras: exposure weighting, balance checks, and a relativities table.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

features = ["DrivAge", "VehAge", "VehPower", "VehGas", "Region", "Density"]

ebm = InsuranceEBM(
    feature_names=features,
    objective="poisson",
    interactions=3,       # up to 3 pairwise interaction terms
    max_bins=256,
)
ebm.fit(
    train[features].to_pandas(),
    train["ClaimNb"].to_pandas(),
    sample_weight=train["Exposure"].to_pandas(),
)

# Inspect the DrivAge shape function
rel = RelativitiesTable(ebm)
print(rel.to_polars("DrivAge"))
```

The output is a Polars DataFrame with bin boundaries and multiplicative relativities — what you would hand to a rating team to load into a system. A 20-year-old driver might show a relativity of 2.1x relative to the base; a 45-year-old shows 0.85x. These numbers come from the data, not from actuarial judgement applied to binned counts, but they are expressed in a form that actuaries can audit, challenge, and adjust using `MonotonicityEditor` if the business needs a smooth age curve.

One thing to be honest about: the EBM above uses freMTPL2, which is a real dataset, but it is French. The implied relativities will not match a UK motor book. The pipeline structure is what matters. If you want a GLM-first approach using the same dataset — with step-by-step coverage of exposure offsets, deviance residuals, and factor relativity extraction — the [GLM frequency model tutorial](/blog/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) is the right starting point.

---

## Step 2: add distribution-free prediction intervals with `insurance-conformal`

The EBM gives us a point estimate — expected claim frequency per policy year. For a standard motor risk sitting in a dense region of the feature space, that estimate is reasonably stable. For an unusual risk — very young driver, high-powered vehicle, rural region — the model is extrapolating, and the point estimate carries real uncertainty. Without quantifying that uncertainty, you cannot tell the two apart at pricing time.

Conformal prediction adds coverage-guaranteed intervals without assuming anything about the error distribution. The `TweedieConformPredictor` is the right entry point for Poisson frequency data: it handles exposure weighting and offers four non-conformity score types.

```python
from insurance_conformal import TweedieConformPredictor

cp = TweedieConformPredictor(
    model=ebm,
    tweedie_power=1.0,         # Poisson
    score="pearson_weighted",  # (y - yhat) / sqrt(yhat * exposure)
)
cp.calibrate(
    cal[features].to_pandas(),
    cal["ClaimNb"].to_pandas(),
    exposure=cal["Exposure"].to_pandas(),
)

intervals = cp.predict_interval(
    test[features].to_pandas(),
    alpha=0.10,                # 90% marginal coverage
    exposure=test["Exposure"].to_pandas(),
)
# intervals["lower"], intervals["upper"] per policy
```

The coverage guarantee is marginal: across the whole test set, at least 90% of true values fall inside the interval. It is not conditional on any covariate. For a rigorous check of whether the intervals are narrower in the data-rich centre and wider at the edges (which they should be for good heteroscedastic coverage), use `ConditionalCoverageERT`:

```python
from insurance_conformal import ConditionalCoverageERT

ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=5)
result = ert.evaluate(
    test[features].to_pandas(),
    intervals["lower"],
    intervals["upper"],
    test["ClaimNb"].to_pandas(),
    alpha=0.10,
)
print(result)
# ERTResult(ert=0.018, CI=[0.004, 0.033], p=0.041)
```

An ERT above 0.05 with p < 0.05 means the intervals are conditionally thin in some region — worth investigating before deploying. The subgroup breakdown by feature is a one-liner:

```python
subgroups = ert.subgroup_coverage(
    test[features].to_pandas(),
    intervals["lower"],
    intervals["upper"],
    test["ClaimNb"].to_pandas(),
    alpha=0.10,
    feature_names=features,
)
```

---

## Step 3: set up production drift detection with `insurance-monitoring`

The model is fitted on historical data. The moment it goes live, the portfolio composition starts to drift. New vehicle models come to market. Underwriting appetite shifts. A competitor changes their pricing and selectively wins the risks you did not want. None of this is exotic — it happens to every UK motor book.

`insurance-monitoring` gives us three layers of detection. PSI for feature distribution shift. A/E ratio monitoring for calibration. Gini drift testing for discrimination stability. We set a baseline on the test set and would update it with live data monthly:

```python
from insurance_monitoring import MonitoringReport
from insurance_monitoring.drift import psi

train_pd = train[features].to_pandas()
test_pd  = test[features].to_pandas()

# PSI on DrivAge — typical early-warning check
psi_age = psi(
    reference=train_pd["DrivAge"].values,
    current=test_pd["DrivAge"].values,
    bins=10,
)
print(f"DrivAge PSI: {psi_age:.4f}")
# PSI < 0.10: stable. 0.10–0.25: monitor. > 0.25: investigate.

# Full monitoring report — A/E, Gini, PSI in one call
y_pred_test = ebm.predict(test_pd)
report = MonitoringReport(
    y_true=test["ClaimNb"].to_numpy(),
    y_pred=y_pred_test,
    exposure=test["Exposure"].to_numpy(),
    X_train=train_pd,
    X_monitor=test_pd,
)
print(report.summary())
```

The `MonitoringReport.summary()` produces a traffic-light output: GREEN / AMBER / RED for each check. The [insurance model monitoring practitioner's guide](/blog/2026/04/04/insurance-model-monitoring-python-practitioner-guide/) covers the three-scenario decision logic in detail: when to redeploy, recalibrate, or refit. If the Gini drops more than two standard deviations from the training estimate (the default alpha=0.32 threshold from Brauer, Menzel & Wüthrich 2025), the report recommends RECALIBRATE or REFIT.

For production use, you schedule this monthly. The `PITMonitor` (anytime-valid calibration test) gives you the same protection without the repeated-testing inflation that monthly Hosmer-Lemeshow checks suffer. It works at the individual policy level: for each new claim observation, you compute the Poisson CDF at the observed count, then feed that PIT value to the monitor:

```python
from insurance_monitoring import PITMonitor
from scipy.stats import poisson

monitor = PITMonitor(alpha=0.05, rng=42)

# Feed policies as they are observed in production
for row in live_policies:
    mu = row.exposure * ebm.predict([[*row.features]])[0]
    pit_value = float(poisson.cdf(row.claim_count, mu))
    alarm = monitor.update(pit_value)
    if alarm:
        print(f"Calibration shift at step {alarm.time}")
        break
```

This guarantees P(ever alarm | model calibrated) <= 0.05 across any monitoring horizon — you can look at it every month forever without inflating the false alarm rate.

---

## Step 4: audit for proxy discrimination with `insurance-fairness`

Driver age is a legitimate rating factor for UK motor. The loss data supports it and Equality Act 2010 Schedule 3 paragraph 25 provides a specific exception for age-based motor insurance pricing where the use is actuarially justified. This matters because `DrivAge` is also a strong proxy for characteristics that are protected without a pricing exception — and the FCA's guidance on proxy discrimination (EP25/2) requires firms to understand the proxy structure of their rating plan.

We run `IndirectDiscriminationAudit` to quantify how much of the age signal passes through the other features versus how much is a direct age effect. This does not say the model is discriminatory — it says where to look:

```python
from insurance_fairness import IndirectDiscriminationAudit, detect_proxies

# First: understand the proxy structure
proxy_result = detect_proxies(
    df=test.to_pandas(),
    protected_col="DrivAge",
    factor_cols=["VehPower", "Region", "Density", "VehAge", "VehGas"],
)
proxy_result.summary()
# Prints mutual information scores: which features correlate most with age

# Then: quantify indirect discrimination
audit = IndirectDiscriminationAudit(
    protected_attr="DrivAge",
    proxy_features=["VehPower", "Region", "Density"],
    exposure_col="Exposure",
)
result = audit.fit(
    train.to_pandas()[features + ["Exposure"]],
    train["ClaimNb"].to_numpy(),
    test.to_pandas()[features + ["Exposure"]],
    test["ClaimNb"].to_numpy(),
)
print(f"Proxy vulnerability: {result.proxy_vulnerability:.4f}")
# 0.0 = age effect fully explained by proxies (unaware model == aware model)
# High = age adds discrimination beyond what the proxies capture
print(result.segment_report)
```

The proxy vulnerability score (from Côté, Côté & Charpentier 2025) is the mean absolute difference between the aware model (which uses `DrivAge` directly) and the unaware model (which uses only the correlated proxies). A score near zero means the proxies are already doing the age discrimination — removing `DrivAge` from the feature set would not materially change pricing. A high score means `DrivAge` contributes genuinely independent discrimination.

For an insurer where age IS used, and used with Schedule 3 justification, this output documents the actuarial basis. For a product where you have chosen NOT to use a protected characteristic, the audit verifies the proxies are not smuggling it back in.

We are deliberately not prescribing a threshold here. What counts as acceptable proxy vulnerability is a regulatory and commercial judgement that belongs with the underwriting team and their legal counsel, not with a Python library.

---

## Step 5: generate a PRA-style validation report with `insurance-governance`

The final step packages the whole pipeline into a defensible validation document. `ModelValidationReport` runs the statistical tests — Gini with bootstrap CI, A/E ratio, Hosmer-Lemeshow, PSI — and writes an HTML report with RAG status on each test:

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

card = ValidationModelCard(
    name="Motor TPPD Frequency — EBM v1.0",
    version="1.0.0",
    purpose="Predict claim frequency for private motor. freMTPL2 DGP.",
    methodology="Explainable Boosting Machine (Poisson), insurance-gam v0.5",
    target="ClaimNb",
    features=features,
    limitations=[
        "Fitted on freMTPL2 (French). Not calibrated to UK experience.",
        "No telematics features.",
        "EBM interactions capped at pairwise — no three-way terms.",
    ],
    owner="Pricing Team",
)

val_report = ModelValidationReport(
    model_card=card,
    y_val=test["ClaimNb"].to_numpy(),
    y_pred_val=y_pred_test,
    exposure_val=test["Exposure"].to_numpy(),
    y_train=train["ClaimNb"].to_numpy(),
    y_pred_train=ebm.predict(train_pd),
)

val_report.generate("motor_freq_validation.html")
val_report.to_json("motor_freq_validation.json")
```

The HTML report has a documented pass/fail outcome per test: Gini with bootstrap 95% CI, A/E ratio with Poisson confidence interval, Hosmer-Lemeshow statistic and p-value, PSI per feature. The JSON sidecar is machine-readable for ingestion into an MRM system.

One regulatory note: `insurance-governance` aligns its statistical tests with PRA model risk management principles. PRA SS1/23 formally applies to banks; the insurance equivalent is PRA CP6/24. The library does not care which regime you operate in — the statistical tests are the same either way. What matters is that you have quantitative, auditable, reproducible pass/fail results rather than a narrative.

For the model risk committee, add a `GovernanceReport`:

```python
from insurance_governance import MRMModelCard, RiskTierScorer, GovernanceReport, Assumption

mrm_card = MRMModelCard(
    model_id="motor-freq-ebm-v1",
    model_name="Motor TPPD Frequency EBM",
    version="1.0.0",
    model_class="pricing",
    intended_use="Frequency pricing for private motor. Not for commercial fleet.",
    assumptions=[
        Assumption(
            description="Claim frequency stationarity since 2023",
            risk="MEDIUM",
            mitigation="Monthly A/E monitoring via PITMonitor",
        ),
        Assumption(
            description="freMTPL2 DGP is representative of book risk profile",
            risk="HIGH",
            mitigation="Replace with UK portfolio data before production deployment",
        ),
    ],
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=85_000_000,
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)

gov_report = GovernanceReport(card=mrm_card, tier=tier)
gov_report.save_html("motor_freq_mrm_pack.html")
```

The risk tier score (`tier.tier`) drives the model oversight requirements: a Tier 1 model needs annual independent validation; Tier 2 is biennial. The `GovernanceReport` HTML includes the model card, tier rationale, assumption register, and links to the validation evidence.

---

## What this looks like end-to-end

Five libraries, roughly 120 lines of substantive code (not counting data loading), and you have:

1. An interpretable EBM tariff with readable relativity curves per rating factor
2. Distribution-free 90% prediction intervals per policy with conditional coverage diagnostics
3. Monthly PSI, A/E, and Gini drift monitoring with traffic-light output
4. A quantified proxy discrimination audit with documentation for regulatory evidence packs
5. A governance report with RAG-status validation tests and a model risk tier

Emblem gives you step 1 for a GLM. Radar gives you step 1 plus a rating engine. No other open-source Python ecosystem gives you steps 2 through 5 in a form that integrates with step 1's output. That is what we are building.

The code in this post shows the real APIs with realistic parameters. For a worked Jupyter notebook against the actual freMTPL2 dataset, the individual library repositories have notebooks in their `notebooks/` directories. Pull requests for the pipeline notebook into [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples) are welcome.

---

*Related:*
- [Building a GLM Frequency Model in Python](/blog/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — the Poisson GLM equivalent of Step 1, with deviance residuals and relativity extraction
- [Actuarial Model Validation in Python](/blog/2026/04/04/actuarial-model-validation-python/) — deeper treatment of the Step 5 governance layer: the five-test suite, MRM model cards, and CI/CD integration
- [Conformal Prediction for Insurance Python: A Frequency-Severity Tutorial](/blog/2026/04/04/conformal-prediction-insurance-python/) — Step 2 in detail for two-stage models, including the calibration subtlety that breaks naive implementations
