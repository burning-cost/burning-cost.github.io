---
layout: post
title: "Building a Modern Insurance Pricing Pipeline in Python"
date: 2026-03-12
description: "Complete UK insurance pricing pipeline in Python: CatBoost GLM distillation, causal inference, FCA fairness auditing, rate optimisation, PRA SS1/23 governance."
tags: [pricing, python, pipeline, tutorial, databricks]
---

*This post is the vendor-agnostic pipeline overview: decisions and tooling that apply regardless of compute environment. For a complete worked example on Databricks specifically - Unity Catalog, MLflow tracking, scheduled Jobs, and Radar export - see [From GBM to Radar: A Complete Databricks Workflow](/2026/02/21/from-gbm-to-radar-databricks-workflow/).*


The UK pricing team's standard toolkit has been largely unchanged for two decades: Emblem or Radar for GLMs, Excel for scenario management, SAS or bespoke R scripts for data manipulation, and periodic exports to whatever catastrophe model the reinsurer mandates. It works. But it has structural limits that become more painful as model complexity increases.

This post is a complete walkthrough of what a modern Python-based pricing pipeline looks like: from raw claims data to deployed model with an audit trail. We write about tools we use, but the goal is to explain the decisions so that you can implement the same workflow with whatever stack suits your team. A pricing actuary should be able to read this and know what to build, even if they never touch our libraries.

---

## The traditional workflow and where it strains

Emblem and Radar are good at what they were designed to do: multiplicative GLMs with factor tables, clean relativity displays, and structured interaction testing. The workflow they enforce is sensible: minimum bias, GLM fit, one-way analyses, lifted Gini plots. Generations of UK pricing actuaries learned to price on these tools, and they are not wrong.

The strains appear at the edges.

**Model complexity.** A Tweedie GLM with 30 rating factors and manually specified interactions is roughly where the Emblem workflow gets unwieldy: interaction specification is manual, and it is not designed for the feature volumes that GBMs handle natively. Moving GBM output back into Radar is a manual, error-prone step.

**Validation.** Temporal cross-validation (splitting on policy inception date, not randomly, to avoid IBNR contamination) is not built into Emblem. Without it, a single held-out period is the default, which overstates model performance on stable books and understates it on books with seasonal claims patterns.

**Uncertainty.** Point estimates come out of every model. Confidence intervals on GLM coefficients assume the model is correctly specified. For a Tweedie GBM on motor claims, that assumption is false in ways that are hard to quantify with standard tools.

**Reproducibility.** An Emblem project file contains the model. The steps that produced the data it was trained on - the extract logic, the outlier removals, the exposure calculations: these typically live in a separate SAS or SQL script, version-controlled separately (or not at all).

**Governance.** PRA SS1/23 is currently scoped to banks and building societies, but its model risk management principles are widely expected to extend to insurance. An Excel model register and a Word validation template are unlikely to satisfy a serious review.

None of this means you should throw away Radar. Radar remains the dominant tariff management system in UK personal lines, and the pipeline described here is designed to feed it, not replace it. But the computation that produces the factor tables can be made substantially more rigorous.

---

## What a modern pipeline looks like

We are going to walk through each stage. For each one, we will show the minimal working code alongside the actuarial reasoning. The examples use synthetic motor data throughout.

---

### 1. Data preparation and synthetic data for testing

Production data is confidential, IBNR-contaminated, and expensive to iterate on. The first practical problem in any pipeline is having a dataset you can test code against before touching live claims.

Generic synthetic data tools - SDV, CTGAN, TVAE - produce data that looks reasonable column by column and breaks down as soon as you run a frequency model on it. The frequency/severity decomposition fails because these tools do not understand that claim counts are Poisson with an exposure offset, that claim severity is conditional on a claim occurring, or that the joint distribution of rating factors matters for the multiplicative structure of a GLM.

A better approach uses vine copulas to model the joint distribution of rating factors, then samples frequency and severity from distributions consistent with the underlying actuarial structure:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 50_000

# Actuarially structured synthetic motor data:
# exposure ~ Uniform(0.1, 1.0), claim_count ~ Poisson(mu * exposure)
exposure = rng.uniform(0.1, 1.0, size=n)
age_band = rng.choice(["17-21","22-25","26-35","36-50","51-65","66+"], n)
vehicle_group = rng.choice([f"G{i}" for i in range(1, 17)], n)
mu = 0.08 * np.where(age_band == "17-21", 2.5, 1.0)  # known relativities
claim_count = rng.poisson(mu * exposure)

df = pd.DataFrame({
    "exposure": exposure, "claim_count": claim_count,
    "driver_age_band": age_band, "vehicle_group": vehicle_group,
})
# For a full structured motor portfolio with calibrated vine copulas:
#   uv add insurance-datasets  # github.com/burning-cost/insurance-datasets
#   from insurance_datasets import load_motor; df = load_motor(n_policies=50_000, seed=42)
```

The output is a DataFrame with `exposure`, `claim_count`, `claim_cost`, and a set of rating factors whose joint distribution respects the vine copula structure. You can run a Tweedie GLM on it and get sensible coefficients. That matters for testing downstream pipeline steps without needing production data.

---

### 2. Temporal cross-validation

Standard k-fold CV is wrong for insurance pricing. It randomly assigns policies to folds, which means development-year data for a 2023 policy can appear in the training fold while the same policy's inception data appears in the test fold. This leaks IBNR information into the training set. The test score is optimistic.

Walk-forward validation fixes this by respecting the temporal ordering of data. Train on 2020–2022, test on 2023. Then train on 2020–2023, test on 2024. Average the test scores. The model never sees data from after its training cutoff.

```python
from insurance_cv import InsuranceCV, walk_forward_split
from insurance_cv import split_summary

# walk_forward_split creates a list of TemporalSplit objects with an
# IBNR buffer between each training cutoff and test window start
cv_splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=24,
    test_months=6,
    ibnr_buffer_months=3,   # IBNR buffer between train cutoff and test start
    step_months=6,
)

cv = InsuranceCV(splits=cv_splits, df=df)
print(split_summary(cv_splits, df, date_col="inception_date"))

# Use sklearn cross_val_score with a Gini scorer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from insurance_monitoring import gini_coefficient

gini_scorer = make_scorer(gini_coefficient, needs_proba=False)
scores = cross_val_score(tweedie_gbm, X, y, cv=cv, scoring=gini_scorer,
                         fit_params={"sample_weight": df["exposure"].values})
```

The `gap_months=3` parameter enforces an IBNR buffer: policies incepting in the three months before the test window are excluded from training. For long-tail lines you would set this higher: 12 to 18 months for casualty.

The Gini coefficient you get out of this is a conservative, honest estimate of model discrimination. It will be lower than k-fold CV. That is correct.

---

### 3. Feature engineering and model building

A UK personal lines pricing pipeline typically involves at least two sub-models: frequency (claim count, Poisson) and severity (average claim cost given a claim, Gamma). The product gives the pure premium, which feeds the technical price.

**GLMs** remain the default for tariff production. They are transparent, regulatorily familiar, and directly exportable to Radar. For a Poisson frequency model in Python:

```python
import statsmodels.api as sm

glm_freq = sm.GLM(
    endog=df["claim_count"],
    exog=sm.add_constant(X_encoded),
    family=sm.families.Poisson(link=sm.families.links.Log()),
    exposure=df["exposure"],
).fit()

relativities = np.exp(glm_freq.params)  # multiplicative rating factors
```

**GBMs** capture non-linear relationships and implicit interactions that a manually specified GLM misses. CatBoost handles categorical features natively, which eliminates most of the encoding pipeline:

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.7",
    learning_rate=0.05,
    depth=6,
    iterations=500,
    cat_features=cat_cols,
    eval_metric="Tweedie",
    random_seed=42,
)
model.fit(
    X_train, y_train,
    sample_weight=exposure_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False,
)
```

**Neural additive models** are worth knowing about, though they are not yet widely production-deployed in UK insurance. The ANAM architecture (Laub, Pho, Wong, arXiv:2509.08467) constrains a neural network so that the output is a sum of univariate neural networks, one per feature, fitted jointly. This gives you near-GLM interpretability with near-GBM flexibility. The key constraint for insurance is that the individual shape functions can be inspected and challenged, which GLM relativities permit and standard GBMs do not.

---

### 4. Model interpretation: SHAP relativities and interaction detection

A GBM that you cannot explain to a head of pricing is commercially useless regardless of its Gini. The standard approach is SHAP values: for each prediction, SHAP decomposes the model output into the contribution of each feature, summing to the total prediction (in log space for multiplicative models).

For pricing, what you actually want is not raw SHAP values but multiplicative relativities in the same format as `exp(beta)` from a GLM, indexed so that the base level of each factor equals 1.0, exposure-weighted, with confidence intervals from bootstrap resampling:

```python
from shap_relativities import SHAPRelativities

extractor = SHAPRelativities(
    model=catboost_model,
    X=X_train,
    exposure=exposure_train,
    # annualise_exposure=True by default — rates are per policy year
)
relativities = extractor.fit()

# Output: dict of DataFrames, one per feature
# Each DataFrame has columns: level, relativity, ci_lower, ci_upper, exposure
print(relativities["vehicle_group"].head())
```

The reconstruction check matters: sum the SHAP relativities across all features for every policy and verify they reconstruct the model prediction to within floating-point tolerance. If they do not, something has gone wrong in the extraction.

**Interaction detection** is a separate step. A GLM with no interaction terms can miss material rate inadequacy in segments where two factors combine non-linearly. The classic example is young age combined with high-performance vehicle. The CANN+NID pipeline (combined actuarial neural network + neural interaction detection) automates the search:

```python
from insurance_interactions import test_interactions

# test_interactions ranks all candidate pairs by NID score and runs LR tests
candidates = test_interactions(
    X_train, y_train, exposure=exposure_train, family="poisson", n_top=20
)
# Returns a DataFrame ranked by NID score with LR test p-values (Bonferroni-corrected)
```

The output is a ranked list of feature pairs. You then decide which to add as explicit interaction terms in the GLM. It is a decision aid, not an automated model builder.

---

### 5. Uncertainty quantification

Point estimates are insufficient for several reasons that go beyond regulatory preference. A risk that sits in a sparse corner of the feature space (an unusual vehicle type, a rare postcode) has wider genuine uncertainty than a standard risk with thousands of similar policies. A tariff that charges the same relativity regardless of prediction confidence is systematically underpricing that uncertainty.

**Conformal prediction** gives distribution-free coverage guarantees. For a properly calibrated conformal predictor, at least 90% of true values will fall inside the 90% prediction interval. This is not a promise conditional on model assumptions; it is a finite-sample guarantee regardless of whether the model is correctly specified:

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=catboost_model,
    nonconformity="pearson_weighted",   # respects heteroscedasticity
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, coverage=0.90)

lower, upper = cp.predict_interval(X_new)
```

**Distributional models** go further: rather than a point estimate plus an interval, they predict the full probability distribution of the outcome for each risk. This is useful for large commercial risks where the tail of the loss distribution matters for pricing, not just the mean:

```python
from insurance_distributional import GammaGBM

# GammaGBM predicts the full Gamma distribution per risk
# — mean, variance, and any percentile
dgbm = GammaGBM()
dgbm.fit(X_train, y_train, sample_weight=exposure_train)

# Returns a DistributionalPrediction with .mean(), .std(), .quantile()
pred = dgbm.predict(X_new)
mean_prediction = pred.mean()
p95 = pred.quantile(0.95)
var_loaded_price = mean_prediction * (1 + safety_loading * pred.std() / mean_prediction)
```

**Quantile regression** targets specific percentiles directly - useful for setting excess points or pricing first-loss layers:

```python
from insurance_quantile import QuantileGBM

q90 = QuantileGBM(quantiles=[0.90])
q90.fit(X_train, y_train, sample_weight=exposure_train)

# predict() returns a DataFrame with one column per quantile
p90_loss = q90.predict(X_new)[0.90]   # 90th percentile of loss for each risk
```

---

### 6. Causal inference and price elasticity

This is where most pricing pipelines have a genuine gap. A standard demand model estimates the association between price changes and renewal. Causal inference estimates the effect - what would happen to renewal if you changed price, holding everything else fixed.

The distinction matters because pricing decisions are non-random. Policies that received a large rate increase in the previous year tend to have specific claim characteristics, agent profiles, and behavioural histories. A naive regression of renewal on price change conflates the price effect with all of these selection effects.

Double Machine Learning (DML) addresses this by partialling out the confounding in two stages: first, predict the treatment (price change) and the outcome (renewal) separately from the confounders, then regress the residuals of each on the other. The resulting coefficient is a consistent estimate of the average treatment effect under assumptions that are weaker than standard regression.

```python
from insurance_causal import CausalPricingModel

dml = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment="log_price_change",
    confounders=list(X_renewal.columns),   # rating factors, history
    cv_folds=5,
    random_state=42,
)
result = dml.fit(df, exposure_col="exposure")

ate = result.average_treatment_effect()
print(f"Elasticity (ATE): {ate.ate:.3f}")
print(f"95% CI: ({ate.ci_lower:.3f}, {ate.ci_upper:.3f})")
```

For FCA Consumer Duty purposes, the causal elasticity estimate is more defensible than a correlation-based one if you are asked to demonstrate that your renewal pricing respects the ENBP principle. An association-based coefficient systematically overstates the true price sensitivity of customers who were selected for large rate increases on actuarial grounds.

---

### 7. Rate optimisation

Given a technical price and a demand model, rate optimisation finds the set of factor adjustments that maximises a business objective (portfolio profit, volume, Sharpe ratio on underwriting income) subject to constraints. The constraints are where regulation lives.

FCA PS21/11 imposes equivalent new business pricing (ENBP): a renewing customer cannot be charged more than a new customer with equivalent risk characteristics would pay. ICOBS 6B translates this into a hard constraint on the renewal-to-new ratio by rating cell. PRA Solvency II capital requirements create an implicit constraint on the loss ratio distribution.

A linear programming formulation handles all of these simultaneously:

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

constraints = ConstraintConfig(
    lr_max=0.74,              # maximum acceptable loss ratio
    enbp_buffer=0.0,          # strict ENBP compliance
    retention_min=0.80,
)

optimiser = PortfolioOptimiser(
    technical_price=df["technical_price"].values,
    expected_loss_cost=df["expected_loss"].values,
    p_demand=df["p_renewal"].values,
    elasticity=df["elasticity"].values,
    renewal_flag=df["is_renewal"].values,
    enbp=df["nb_equivalent_price"].values,
    constraints=constraints,
)

result = optimiser.optimise()
frontier = optimiser.optimise_scenarios()   # efficient frontier
```

The efficient frontier, the set of (LR, volume) outcomes achievable given the constraints, is the output your board actually needs. A single optimised scenario is a point on that frontier. Knowing the full frontier tells you the regulatory cost of ENBP compliance in volume terms, and the profitability cost of a one-point tightening in the LR constraint.

---

### 8. Model validation

PRA SS1/23 is currently formal guidance for banks and building societies on model risk management. Its requirements for model validation documentation (independent validation, outcome testing, sensitivity analysis, assumption documentation) are the correct standard for insurance pricing models regardless of regulatory status.

A validation report that will survive scrutiny needs to cover: discriminatory power (lifted Gini, by segment), calibration (A/E by decile and by rating factor), stability (Gini over time, PSI on feature distributions), sensitivity (coefficient stability under data perturbation), and a clear statement of model limitations and mitigants.

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

card = ValidationModelCard(
    name="Motor Frequency v3",
    methodology="CatBoost with Poisson objective",
    target="claim_count",
)

validator = ModelValidationReport(
    model_card=card,
    y_val=y_test,
    y_pred_val=production_model.predict(X_test),
    exposure_val=exposure_test,
)

validator.run()
validator.generate("validation_report_motor_freq_v3.html")
```

The report should be independently reproducible: anyone with access to the data and the model artefact should be able to rerun the validation and get the same numbers. This means the validation script itself needs to be version-controlled alongside the model.

---

### 9. Drift monitoring

A model validated in Q1 may not be valid in Q4. Feature distributions shift: the vehicle mix changes, the postcode distributions shift, a large commercial account rolls off. Calibration drifts as the aggregate A/E moves, with segmented A/E moving in ways the aggregate masks. Discrimination degrades as the Gini falls when model factors become stale relative to the claims environment.

A three-layer monitoring framework covers these at different frequencies:

- **Feature drift (monthly):** Population Stability Index (PSI) on each rating factor distribution. PSI > 0.25 on any major feature triggers investigation.
- **Calibration monitoring (quarterly):** Segmented A/E by decile, by vehicle group, by region. Aggregate A/E in range can mask material segment-level deterioration.
- **Discrimination testing (semi-annual):** Formal Gini test against the baseline. A 3-point Gini degradation on a 50,000-policy portfolio is statistically significant at conventional levels.

```python
from insurance_monitoring import MonitoringReport, psi

# PSI per feature: flag anything > 0.25
for col in rating_factors:
    psi_val = psi(X_baseline[col], X_current[col])
    if psi_val > 0.25:
        print(f"AMBER: {col} PSI={psi_val:.3f}")

# Full monitoring report: calibration, Gini drift, A/E by segment
report = MonitoringReport(
    reference_actual=y_baseline,
    reference_predicted=model.predict(X_baseline),
    current_actual=y_current,
    current_predicted=model.predict(X_current),
    exposure=exposure_current,
)
print(report.recommendation())     # GREEN / AMBER / RED
```

The output should feed a dashboard your pricing manager and actuarial function see regularly - not a notebook you run when something goes wrong.

---

### 10. Deployment: champion/challenger with an audit trail

Champion/challenger testing is the standard mechanism for evaluating whether a new model version should replace the production model. The challenger model is routed a fraction of live quotes (typically 5–15%) and its loss ratio is compared to the champion's on matched business.

The FCA's record-keeping requirements under ICOBS 6B.2.51R require that renewal pricing decisions are auditable: you need to be able to demonstrate, for any given renewal quote, which model version produced it, what inputs were used, and what price was offered. Without this, a thematic review is likely to find gaps.

A proper implementation requires deterministic routing (the same policy always goes to the same model version, based on a hash of a stable policy identifier), quote logging with enough fidelity to reconstruct the decision, and a formal statistical test comparing outcomes.

```python
from insurance_deploy import Experiment, ModelRegistry, QuoteLogger, ModelComparison, KPITracker

registry = ModelRegistry(path="/data/model_registry.db")
champion_v = registry.register("motor_freq_v3", champion_model, tags={"status": "champion"})
challenger_v = registry.register("motor_freq_v4", challenger_model, tags={"status": "challenger"})

# Deterministic routing: the same policy_id always maps to the same model
experiment = Experiment(
    name="v3_vs_v4",
    champion=champion_v,
    challenger=challenger_v,
    challenger_pct=0.10,
    mode="live",
)

# Quote logging for ICOBS 6B.2.51R audit trail
logger = QuoteLogger(path="/data/quote_log.db")

# At quote time:
model_version = experiment.route(policy_id)    # deterministic routing
# ... generate quote_data ...
logger.log_quote(quote_data)

# After sufficient exposure: bootstrap LR comparison
tracker = KPITracker()
comp = ModelComparison(tracker=tracker)
test_result = comp.bootstrap_lr_test(realised_outcomes, n_bootstrap=10_000)
```

The bootstrap LR test gives you a power-adjusted comparison: before you run the test, you should know how much exposure you need to detect a given LR difference at a given confidence level. Running a champion/challenger test without a power calculation is common. It means you are often making model promotion decisions on noise.

---

### 11. Governance: model risk management

Model risk management for pricing means having a live inventory of all production models, their risk tier (based on materiality and complexity), their last validation date, their outstanding issues, and a clear escalation path when issues are identified.

PRA SS1/23 defines a structured model lifecycle: development, validation, approval, ongoing monitoring, retirement. Each stage has documentation requirements. The documentation needs to be findable, not in someone's personal OneDrive, and version-controlled alongside the model artefacts.

```python
from insurance_governance.mrm import ModelInventory, ModelCard, RiskTierScorer

inventory = ModelInventory(path="/data/model_registry.db")

card = ModelCard(
    model_id="motor_freq_v3",
    model_name="Motor Frequency GBM",
    version="2026-Q1",
    model_type="CatBoost Tweedie",
    developer="Pricing Actuarial",
    approved_by=["Chief Actuary"],
    approval_date="2026-01-15",
    next_review_date="2026-07-15",
)
inventory.register(card)

# Risk tier: score against materiality and complexity dimensions
tier = RiskTierScorer().score(card)
print(f"Tier {tier.tier}: {tier.tier_label} (score {tier.score:.0f}/100)")

# Board-level inventory summary
df_summary = inventory.summary()
```

Fairness auditing sits alongside this. The FCA's Consumer Duty requires that pricing outcomes do not systematically disadvantage customers with protected characteristics as a proxy. The practical implementation is a structured audit against protected and proxy-protected features, with a documented rationale for any disparate impact that is present:

```python
from insurance_fairness import FairnessAudit

# FCA Consumer Duty: check for proxy discrimination by protected characteristics
audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["age_band", "postcode_deprivation_decile"],
    factor_cols=["vehicle_age", "payment_method"],
    exposure_col="exposure",
)
audit.run().save_report("fairness_audit_motor_2026_Q1.md")
```

---

## What this looks like on Databricks

For teams on Databricks, the pipeline above maps naturally onto the platform's components.

**Unity Catalog** is the right place for claims and exposure Delta tables. Table versioning via Delta time-travel gives you a reproducible training dataset: `SELECT * FROM motor.claims.policy_history VERSION AS OF 42` returns the exact snapshot used to train a specific model version. This is the simplest route to training data reproducibility.

**MLflow** tracks model artefacts, parameters, and metrics for every training run. The key discipline is logging enough to reproduce the run: not just the final Gini, but the training data version, the hyperparameters, the CV fold boundaries, and the validation report path.

```python
import mlflow

with mlflow.start_run(run_name="motor_freq_v3_2026Q1"):
    mlflow.log_params({
        "model_type": "CatBoost",
        "loss_function": "Poisson",
        "training_data_version": 42,
        "cv_splits": 4,
        "gap_months": 3,
    })
    mlflow.log_metric("temporal_gini", results["mean_gini"])
    mlflow.catboost.log_model(model, artifact_path="model")
    mlflow.log_artifact("validation_report_motor_freq_v3.pdf")
```

**Jobs** run the pipeline on a schedule. A well-structured Databricks Job for monthly pricing refresh has tasks for: data extract and validation, feature engineering, model training (with CV), SHAP extraction, validation report generation, and a conditional task that promotes the challenger to champion if the test passes. The conditional promotion step is important: it removes humans from the loop for routine refreshes while preserving human oversight for material changes.

**Feature Store** handles feature sharing across models. The vehicle risk score that feeds the frequency model should be the same feature value that feeds the severity model, the demand model, and the optimiser. Computing it once and registering it in Feature Store eliminates a class of subtle discrepancies.

---

## Where this is heading

Two directions are worth watching.

**Regulatory formalisation.** The PRA's consultation on extending SS1/23 principles to insurance is expected to produce formal guidance within the next two to three years. Pricing teams that have built structured model inventories and reproducible validation pipelines will be well-positioned. Those that have not will be building the documentation under time pressure, which is when errors happen.

**Interpretable-by-design models.** The actuarial neural additive model (ANAM) architecture points toward a future where you do not need to choose between a GLM (interpretable, limited) and a GBM (powerful, opaque). A model that is simultaneously as powerful as a GBM and as interpretable as a GLM, with shape functions you can challenge and override, would change the economics of the SHAP-relativity extraction step significantly. The architecture exists; the production implementations are early.

The efficient frontier of interpretability versus predictive power is shifting. The workflow described in this post (build a GBM for maximum performance, extract SHAP relativities for interpretability, export to Radar for tariff management) is the right approach today. It may not be the right approach in five years.

---

## The pipeline in summary

The complete workflow, in order:

1. Synthetic data for safe iteration during development (`insurance-synthetic`)
2. Temporal CV to get honest discrimination estimates (`insurance-cv`)
3. GLM for the tariff, GBM for the technical price, ANAM if your team is ready
4. SHAP relativities to make the GBM explainable (`shap-relativities`)
5. Interaction detection to find what the GLM missed (`insurance-interactions`)
6. Conformal intervals for prediction uncertainty (`insurance-conformal`)
7. Calibration testing - balance test, auto-calibration, Murphy decomposition (`insurance-monitoring`)
8. Distributional models and quantile regression for tail risk (`insurance-distributional`, `insurance-quantile`)
9. Causal elasticity for defensible demand modelling (`insurance-causal`)
10. Constrained optimisation and the efficient frontier (`insurance-optimise`)
11. Formal validation report for PRA SS1/23 (`insurance-governance`)
12. Drift monitoring with a three-layer framework (`insurance-monitoring`)
13. Champion/challenger with SHA-256 routing and quote logging (`insurance-deploy`)
14. Model inventory and fairness audit for Consumer Duty (`insurance-governance`)

None of these steps is optional if you are pricing personal lines in the UK in 2026. Some can be implemented more lightly than others. But skipping the causal inference step means your demand model is producing confident estimates of the wrong quantity. Skipping the audit trail means you are exposed on ICOBS 6B. Skipping the drift monitoring means your model may be deteriorating for months before anyone notices.

The tools exist to do all of this properly. The bottleneck is almost never the software.


- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/): how `shap-relativities` works in detail, including the reconstruction check and exposure weighting
- [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/): temporal leakage and IBNR contamination in standard CV, and the walk-forward fix
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/): distribution-free coverage guarantees and handling heteroscedasticity in GBMs
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/): balance test, auto-calibration, and Murphy decomposition for the recalibrate-vs-refit decision
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/03/01/your-demand-model-is-confounded/): Double Machine Learning for price elasticity estimation
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/): LP formulation for ENBP-compliant rate changes
- [Champion/Challenger Testing with ICOBS 6B.2.51R Compliance](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/): ICOBS 6B.2.51R requirements and what a proper audit trail looks like
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/): what PRA SS1/23-level validation documentation actually requires
