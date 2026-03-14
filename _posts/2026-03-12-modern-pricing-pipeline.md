---
layout: post
title: "Building a Modern Insurance Pricing Pipeline in Python"
date: 2026-03-12
description: "A complete walkthrough of the modern pricing workflow — from data preparation through model deployment — using open-source Python tools."
tags: [pricing, python, pipeline, tutorial, databricks]
---

*This post is vendor-agnostic: the decisions behind a modern Python pricing pipeline — from raw claims data to deployed model — using open-source tools that work on any infrastructure. If you are running on Databricks and want a Databricks-specific implementation with Unity Catalog, MLflow, and Radar export, read [From GBM to Radar: A Complete Databricks Workflow](/2026/02/21/from-gbm-to-radar-databricks-workflow/).*


The UK pricing team's standard toolkit has been largely unchanged for two decades: Emblem or Radar for GLMs, Excel for scenario management, SAS or bespoke R scripts for data manipulation, and periodic exports to whatever catastrophe model the reinsurer mandates. It works. But it has structural limits that become more painful as model complexity increases.

This post is a complete walkthrough of what a modern Python-based pricing pipeline looks like: from raw claims data to deployed model with an audit trail. We write about tools we use, but the goal is to explain the decisions so that you can implement the same workflow with whatever stack suits your team. A pricing actuary should be able to read this and know what to build, even if they never touch our libraries.

---

## The traditional workflow and where it strains

Emblem and Radar are good at what they were designed to do: multiplicative GLMs with factor tables, clean relativity displays, and structured interaction testing. The workflow they enforce is sensible: minimum bias, GLM fit, one-way analyses, lifted Gini plots. Generations of UK pricing actuaries learned to price on these tools, and they are not wrong.

The strains appear at the edges.

**Model complexity.** A Tweedie GLM with 30 rating factors and manually specified interactions is roughly where the Emblem workflow gets unwieldy: interaction specification is manual, and it is not designed for the feature volumes that GBMs handle natively. Moving GBM output back into Radar is a manual, error-prone step.

**Validation.** Temporal cross-validation (splitting on policy inception date, not randomly, to avoid IBNR contamination) is not built into Emblem. Without it, a single held-out period is the default, which overstates model performance on stable books and understates it on books with seasonal claims patterns.

**Uncertainty.** Point estimates come out of every model. Confidence intervals on GLM coefficients assume the model is correctly specified. For a Tweedie GBM on motor claims, that assumption is false in ways that are hard to quantify with standard tools.

**Reproducibility.** An Emblem project file contains the model. The steps that produced the data it was trained on — the extract logic, the outlier removals, the exposure calculations: these typically live in a separate SAS or SQL script, version-controlled separately (or not at all).

**Governance.** PRA SS1/23 is currently scoped to banks and building societies, but its model risk management principles are widely expected to extend to insurance. An Excel model register and a Word validation template are unlikely to satisfy a serious review.

None of this means you should throw away Radar. Radar remains the dominant tariff management system in UK personal lines, and the pipeline described here is designed to feed it, not replace it. But the computation that produces the factor tables can be made substantially more rigorous.

---

## What a modern pipeline looks like

We are going to walk through each stage. For each one, we will show the minimal working code alongside the actuarial reasoning. The examples use synthetic motor data throughout.

---

### 1. Data preparation and synthetic data for testing

Production data is confidential, IBNR-contaminated, and expensive to iterate on. The first practical problem in any pipeline is having a dataset you can test code against before touching live claims.

Generic synthetic data tools — SDV, CTGAN, TVAE — produce data that looks reasonable column by column and breaks down as soon as you run a frequency model on it. The frequency/severity decomposition fails because these tools do not understand that claim counts are Poisson with an exposure offset, that claim severity is conditional on a claim occurring, or that the joint distribution of rating factors matters for the multiplicative structure of a GLM.

A better approach uses vine copulas to model the joint distribution of rating factors, then samples frequency and severity from distributions consistent with the underlying actuarial structure:

```python
from insurance_synthetic import SyntheticPortfolio

portfolio = SyntheticPortfolio(
    n_policies=50_000,
    frequency_mean=0.08,          # 8 claims per 100 policy years
    severity_lognormal_mu=7.5,    # ~£1,800 mean severity
    severity_lognormal_sigma=1.2,
    copula_family="vine",
    random_state=42,
)
df = portfolio.generate()
```

The output is a DataFrame with `exposure`, `claim_count`, `claim_cost`, and a set of rating factors whose joint distribution respects the vine copula structure. You can run a Tweedie GLM on it and get sensible coefficients. That matters for testing downstream pipeline steps without needing production data.

---

### 2. Temporal cross-validation

Standard k-fold CV is wrong for insurance pricing. It randomly assigns policies to folds, which means development-year data for a 2023 policy can appear in the training fold while the same policy's inception data appears in the test fold. This leaks IBNR information into the training set. The test score is optimistic.

Walk-forward validation fixes this by respecting the temporal ordering of data. Train on 2020–2022, test on 2023. Then train on 2020–2023, test on 2024. Average the test scores. The model never sees data from after its training cutoff.

```python
from insurance_cv import TemporalSplit, cross_val_gini

splitter = TemporalSplit(
    date_col="inception_date",
    min_train_months=24,
    test_months=6,
    gap_months=3,          # IBNR buffer between train cutoff and test start
    n_splits=4,
)

results = cross_val_gini(
    model=tweedie_gbm,
    X=X,
    y=y,
    groups=df["inception_date"],
    cv=splitter,
    weight=df["exposure"],
)
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
    link="log",              # multiplicative model
    n_bootstrap=200,
    confidence_level=0.95,
)
relativities = extractor.fit()

# Output: dict of DataFrames, one per feature
# Each DataFrame has columns: level, relativity, ci_lower, ci_upper, exposure
print(relativities["vehicle_group"].head())
```

The reconstruction check matters: sum the SHAP relativities across all features for every policy and verify they reconstruct the model prediction to within floating-point tolerance. If they do not, something has gone wrong in the extraction.

**Interaction detection** is a separate step. A GLM with no interaction terms can miss material rate inadequacy in segments where two factors combine non-linearly. The classic example is young age combined with high-performance vehicle. The CANN+NID pipeline (combined actuarial neural network + neural interaction detection) automates the search:

```python
from insurance_interactions import InteractionDetector

detector = InteractionDetector(
    base_model=glm_freq,
    X=X_train,
    y=y_train,
    exposure=exposure_train,
    threshold=0.01,          # flag interactions explaining >1% of deviance
)
candidates = detector.fit().top_interactions(n=10)
```

The output is a ranked list of feature pairs. You then decide which to add as explicit interaction terms in the GLM. It is a decision aid, not an automated model builder.

---

### 5. Uncertainty quantification

Point estimates are insufficient for several reasons that go beyond regulatory preference. A risk that sits in a sparse corner of the feature space (an unusual vehicle type, a rare postcode) has wider genuine uncertainty than a standard risk with thousands of similar policies. A tariff that charges the same relativity regardless of prediction confidence is systematically underpricing that uncertainty.

**Conformal prediction** gives distribution-free coverage guarantees. For a properly calibrated conformal predictor, at least 90% of true values will fall inside the 90% prediction interval. This is not a promise conditional on model assumptions; it is a finite-sample guarantee regardless of whether the model is correctly specified:

```python
from insurance_conformal import ConformalPricing

cp = ConformalPricing(
    model=catboost_model,
    coverage=0.90,
    heteroscedastic=True,    # accounts for variance varying across the portfolio
)
cp.calibrate(X_cal, y_cal, exposure_cal)

lower, upper = cp.predict_interval(X_new, exposure_new)
```

**Distributional models** go further: rather than a point estimate plus an interval, they predict the full probability distribution of the outcome for each risk. This is useful for large commercial risks where the tail of the loss distribution matters for pricing, not just the mean:

```python
from insurance_distributional import DistributionalGBM

dgbm = DistributionalGBM(
    distribution="LogNormal",
    n_estimators=300,
    learning_rate=0.05,
)
dgbm.fit(X_train, y_train, sample_weight=exposure_train)

# Returns (mu, sigma) for each prediction
mu, sigma = dgbm.predict(X_new)
mean_prediction = np.exp(mu + 0.5 * sigma**2)
var_loaded_price = np.exp(mu + 0.5 * sigma**2) * (1 + safety_loading * sigma)
```

**Quantile regression** targets specific percentiles directly — useful for setting excess points or pricing first-loss layers:

```python
from insurance_quantile import QuantileRegressor

q90 = QuantileRegressor(quantile=0.90, n_estimators=200)
q90.fit(X_train, y_train, sample_weight=exposure_train)

p90_loss = q90.predict(X_new)  # 90th percentile of loss for each risk
```

---

### 6. Causal inference and price elasticity

This is where most pricing pipelines have a genuine gap. A standard demand model estimates the association between price changes and renewal. Causal inference estimates the effect — what would happen to renewal if you changed price, holding everything else fixed.

The distinction matters because pricing decisions are non-random. Policies that received a large rate increase in the previous year tend to have specific claim characteristics, agent profiles, and behavioural histories. A naive regression of renewal on price change conflates the price effect with all of these selection effects.

Double Machine Learning (DML) addresses this by partialling out the confounding in two stages: first, predict the treatment (price change) and the outcome (renewal) separately from the confounders, then regress the residuals of each on the other. The resulting coefficient is a consistent estimate of the average treatment effect under assumptions that are weaker than standard regression.

```python
from insurance_causal import DoubleMLElasticity

dml = DoubleMLElasticity(
    outcome_model=CatBoostRegressor(iterations=200),
    treatment_model=CatBoostRegressor(iterations=200),
    n_folds=5,
    random_state=42,
)
dml.fit(
    X=X_renewal,                    # confounders: rating factors, history
    treatment=df["log_price_change"],
    outcome=df["renewed"],
    sample_weight=df["exposure"],
)

print(f"Elasticity: {dml.elasticity_:.3f}")
print(f"95% CI: ({dml.ci_lower_:.3f}, {dml.ci_upper_:.3f})")
```

For FCA Consumer Duty purposes, the causal elasticity estimate is more defensible than a correlation-based one if you are asked to demonstrate that your renewal pricing respects the ENBP principle. An association-based coefficient systematically overstates the true price sensitivity of customers who were selected for large rate increases on actuarial grounds.

---

### 7. Rate optimisation

Given a technical price and a demand model, rate optimisation finds the set of factor adjustments that maximises a business objective (portfolio profit, volume, Sharpe ratio on underwriting income) subject to constraints. The constraints are where regulation lives.

FCA PS21/5 imposes equivalent new business pricing (ENBP): a renewing customer cannot be charged more than a new customer with equivalent risk characteristics would pay. ICOBS 6B translates this into a hard constraint on the renewal-to-new ratio by rating cell. PRA Solvency II capital requirements create an implicit constraint on the loss ratio distribution.

A linear programming formulation handles all of these simultaneously:

```python
from insurance_optimise import RateOptimiser, ENBPConstraint, LossRatioConstraint

optimiser = RateOptimiser(
    technical_price=df["technical_price"],
    demand_model=dml_demand,
    exposure=df["exposure"],
)

optimiser.add_constraint(ENBPConstraint(
    renewal_flag=df["is_renewal"],
    new_business_price=df["nb_equivalent_price"],
    tolerance=0.0,       # strict ENBP compliance
))

optimiser.add_constraint(LossRatioConstraint(
    target_lr=0.72,
    tolerance=0.02,
))

result = optimiser.optimise(objective="profit")
efficient_frontier = optimiser.efficient_frontier(n_points=50)
```

The efficient frontier, the set of (LR, volume) outcomes achievable given the constraints, is the output your board actually needs. A single optimised scenario is a point on that frontier. Knowing the full frontier tells you the regulatory cost of ENBP compliance in volume terms, and the profitability cost of a one-point tightening in the LR constraint.

---

### 8. Model validation

PRA SS1/23 is currently formal guidance for banks and building societies on model risk management. Its requirements for model validation documentation (independent validation, outcome testing, sensitivity analysis, assumption documentation) are the correct standard for insurance pricing models regardless of regulatory status.

A validation report that will survive scrutiny needs to cover: discriminatory power (lifted Gini, by segment), calibration (A/E by decile and by rating factor), stability (Gini over time, PSI on feature distributions), sensitivity (coefficient stability under data perturbation), and a clear statement of model limitations and mitigants.

```python
from insurance_governance.validation import PricingModelValidator

validator = PricingModelValidator(
    model=production_model,
    X_test=X_test,
    y_test=y_test,
    exposure_test=exposure_test,
    model_id="motor_freq_v3",
    model_version="2026-Q1",
    validation_date="2026-03-12",
)

report = validator.run(
    checks=["gini", "calibration", "lift", "sensitivity", "feature_importance"],
    regulatory_standard="PRA_SS123",
)
report.to_pdf("validation_report_motor_freq_v3.pdf")
report.to_html("validation_report_motor_freq_v3.html")
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
from insurance_monitoring import DriftMonitor

monitor = DriftMonitor(
    reference_data=X_baseline,
    reference_date="2025-Q3",
    psi_threshold=0.25,
    gini_degradation_alert=3.0,
)

status = monitor.check(
    current_data=X_current,
    current_predictions=model.predict(X_current),
    current_actuals=y_current,
    current_exposure=exposure_current,
)
print(status.summary())     # GREEN / AMBER / RED by layer
```

The output should feed a dashboard your pricing manager and actuarial function see regularly — not a notebook you run when something goes wrong.

---

### 10. Deployment: champion/challenger with an audit trail

Champion/challenger testing is the standard mechanism for evaluating whether a new model version should replace the production model. The challenger model is routed a fraction of live quotes (typically 5–15%) and its loss ratio is compared to the champion's on matched business.

The FCA's record-keeping requirements under ICOBS 6B.2.51R require that renewal pricing decisions are auditable: you need to be able to demonstrate, for any given renewal quote, which model version produced it, what inputs were used, and what price was offered. Without this, a thematic review is likely to find gaps.

A proper implementation requires deterministic routing (the same policy always goes to the same model version, based on a hash of a stable policy identifier), quote logging with enough fidelity to reconstruct the decision, and a formal statistical test comparing outcomes.

```python
from insurance_deploy import ChampionChallenger, QuoteLogger

cc = ChampionChallenger(
    champion=champion_model,
    challenger=challenger_model,
    challenger_fraction=0.10,
    routing_key="policy_id",       # SHA-256 hash for deterministic routing
    random_seed=42,
)

logger = QuoteLogger(
    db_path="/data/quote_log.db",
    log_inputs=True,               # logs X alongside predictions
    log_model_hash=True,           # SHA-256 of model artefact
)

# At quote time:
quote = cc.predict(X_quote, policy_id=policy_id)
logger.log(quote)

# After sufficient exposure:
test_result = cc.test_outcomes(
    outcomes_df=realised_outcomes,
    method="bootstrap_lr",
    n_bootstrap=10_000,
)
print(test_result.report())        # ICOBS 6B.2.51R compatible output
```

The bootstrap LR test gives you a power-adjusted comparison: before you run the test, you should know how much exposure you need to detect a given LR difference at a given confidence level. Running a champion/challenger test without a power calculation is common. It means you are often making model promotion decisions on noise.

---

### 11. Governance: model risk management

Model risk management for pricing means having a live inventory of all production models, their risk tier (based on materiality and complexity), their last validation date, their outstanding issues, and a clear escalation path when issues are identified.

PRA SS1/23 defines a structured model lifecycle: development, validation, approval, ongoing monitoring, retirement. Each stage has documentation requirements. The documentation needs to be findable, not in someone's personal OneDrive, and version-controlled alongside the model artefacts.

```python
from insurance_governance.mrm import ModelInventory, ModelRiskTier

inventory = ModelInventory(db_path="/data/model_registry.db")

inventory.register(
    model_id="motor_freq_v3",
    description="Motor frequency GBM, personal lines, CatBoost Tweedie",
    risk_tier=ModelRiskTier.TIER_2,      # material, moderate complexity
    owner="Pricing Actuarial",
    approved_by="Chief Actuary",
    approval_date="2026-01-15",
    next_review_date="2026-07-15",
    validation_report_path="/models/motor_freq_v3/validation_report.pdf",
)

# Generate board-level summary
report = inventory.exec_summary(as_of="2026-03-12")
report.to_pdf("model_risk_exec_summary_Q1_2026.pdf")
```

Fairness auditing sits alongside this. The FCA's Consumer Duty requires that pricing outcomes do not systematically disadvantage customers with protected characteristics as a proxy. The practical implementation is a structured audit against protected and proxy-protected features, with a documented rationale for any disparate impact that is present:

```python
from insurance_governance.mrm import FairnessAuditor

auditor = FairnessAuditor(
    protected_features=["age_band", "postcode_deprivation_decile"],
    proxy_features=["vehicle_age", "payment_method"],
)
audit = auditor.run(
    predictions=model.predict(X),
    actuals=y,
    metadata=df[['age_band', 'postcode_deprivation_decile', 'vehicle_age', 'payment_method']],
)
audit.to_report("fairness_audit_motor_2026_Q1.pdf")
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
7. Calibration testing — balance test, auto-calibration, Murphy decomposition (`insurance-monitoring`)
8. Distributional models and quantile regression for tail risk (`insurance-distributional`, `insurance-quantile`)
9. Causal elasticity for defensible demand modelling (`insurance-causal`)
10. Constrained optimisation and the efficient frontier (`insurance-optimise`)
11. Formal validation report for PRA SS1/23 (`insurance-governance`)
12. Drift monitoring with a three-layer framework (`insurance-monitoring`)
13. Champion/challenger with SHA-256 routing and quote logging (`insurance-deploy`)
14. Model inventory and fairness audit for Consumer Duty (`insurance-governance`)

None of these steps is optional if you are pricing personal lines in the UK in 2026. Some can be implemented more lightly than others. But skipping the causal inference step means your demand model is producing confident estimates of the wrong quantity. Skipping the audit trail means you are exposed on ICOBS 6B. Skipping the drift monitoring means your model may be deteriorating for months before anyone notices.

The tools exist to do all of this properly. The bottleneck is almost never the software.

---

## Related articles

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/): how `shap-relativities` works in detail, including the reconstruction check and exposure weighting
- [Why Your Cross-Validation is Lying to You](/2026/02/23/why-your-cross-validation-is-lying-to-you/): temporal leakage and IBNR contamination in standard CV, and the walk-forward fix
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/): distribution-free coverage guarantees and handling heteroscedasticity in GBMs
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/): balance test, auto-calibration, and Murphy decomposition for the recalibrate-vs-refit decision
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/02/25/causal-inference-for-insurance-pricing/): Double Machine Learning for price elasticity estimation
- [Constrained Rate Optimisation and the Efficient Frontier](/2026/02/21/constrained-rate-optimisation-efficient-frontier/): LP formulation for ENBP-compliant rate changes
- [Champion/Challenger Testing with ICOBS 6B.2.51R Compliance](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/): ICOBS 6B.2.51R requirements and what a proper audit trail looks like
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/): what PRA SS1/23-level validation documentation actually requires
