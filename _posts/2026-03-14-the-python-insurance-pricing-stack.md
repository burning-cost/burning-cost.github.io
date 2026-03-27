---
layout: post
title: "The Python Insurance Pricing Stack: 35 Libraries for Everything Emblem Can't Do"
date: 2026-03-14
categories: [guides]
tags: [python, insurance-pricing, machine-learning, glm, gbm, pricing-tools]
description: "35 open-source Python libraries for UK insurance pricing: GBM-to-GLM distillation, causal inference, FCA fairness auditing, rate optimisation, PRA SS1/23."
---
> **Update (March 2026):** `insurance-demand` has been merged into [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise). Install `insurance-optimise` and use `insurance_optimise.demand` for the same functionality.

Every UK pricing team has the same gap. The GBM outperforms the GLM by 4-6 Gini points on the holdout set. Everyone knows it. Nobody in production is running it, because nobody can explain it, govern it, or get it into Radar in a form the head of pricing will sign off on. Emblem and ReMetrica are excellent tools for managing multiplicative rating factor structures. They were not built for gradient boosting, causal inference, or conformal prediction intervals.

This stack was. It is 35 Python libraries, built specifically for UK personal lines pricing workflows. They are not wrappers around generic ML tools with "insurance" added to the name. Each one solves a specific problem that appeared repeatedly across actual pricing projects: the factor table a regulator wanted that a SHAP notebook could not produce, the cross-validation that was inflating GBM performance by 7 Gini points due to temporal leakage, the rate change that the model said was working but which was confounded from the start.

Below is the full stack, organised by where in the pricing workflow each library sits. If you only implement one section, start with the Core Stack. Everything else is additive.

---

## The Core Stack

Five libraries that cover the most common failure modes. A pricing team that uses all five is better positioned than most UK insurers.

### shap-relativities - GBM → GLM factor tables

Your GBM is sitting in a notebook because you cannot get a multiplicative factor table out of it. `shap-relativities` closes that gap. It uses SHAP additive decomposition to extract exposure-weighted, confidence-interval-equipped relativities in `exp(β)` format from any CatBoost model - the format your pricing committee and Radar both understand.

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(model=catboost_model, X=X_train, exposure=exposure)
tables = sr.extract_relativities(
    categorical_features=["driver_age_band", "vehicle_group", "area"]
)
tables["driver_age_band"].write_csv("age_relativities.csv")
```

The full methodology - Shapley efficiency axiom, log-space decomposition, confidence intervals via bootstrap - is covered in [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/).

[github.com/burning-cost/shap-relativities](https://github.com/burning-cost/shap-relativities)

---

### insurance-cv - temporal cross-validation

Standard k-fold CV is wrong for insurance data. Policies from 2020 land in the same folds as policies from 2023. Claims from the test fold have less development than claims in training. The IBNR structure poisons both. The result is a CV metric that is optimistic by 5-8 Gini points on UK motor data, and a model tuned against phantom signal.

`insurance-cv` implements walk-forward validation with configurable development buffers and IBNR guards.

```python
from insurance_cv import InsuranceCV, walk_forward_split
from sklearn.model_selection import cross_val_score

cv_splits = walk_forward_split(
    df,
    date_col="inception_date",
    min_train_months=24,
    test_months=12,
    ibnr_buffer_months=6,
    step_months=12,
)
cv = InsuranceCV(splits=cv_splits, df=df)
scores = cross_val_score(model, X, y, cv=cv)
```

The full diagnosis of how k-fold breaks on insurance data is in [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/).

[github.com/burning-cost/insurance-cv](https://github.com/burning-cost/insurance-cv)

---

### insurance-monitoring - drift detection

A pricing model is not complete when it goes live. It starts degrading from day one. `insurance-monitoring` tracks population stability (PSI), feature drift (KS/Cramér's V), and modelled-to-actual ratios across your rating factors on a configurable cadence. It writes structured alerts your existing monitoring infrastructure can ingest.

```python
from insurance_monitoring import psi

# PSI per feature: flag factors with PSI > 0.2
drifted = {
    col: psi(train_df[col].values, live_df[col].values)
    for col in rating_factors
    if psi(train_df[col].values, live_df[col].values) > 0.2
}
print(f"Drifted factors: {list(drifted.keys())}")
```

[insurance-monitoring](/insurance-monitoring/)

---

### insurance-fairness - FCA proxy discrimination

The Citizens Advice research (2022) found a £280/year ethnicity penalty in UK motor insurance in postcodes where more than 50% of residents are people of colour - £213m per year across the market. The mechanism is postcode as a rating factor correlating with ethnicity. The FCA Consumer Duty (live July 2023) and Equality Act 2010 Section 19 mean this is not a theoretical risk.

`insurance-fairness` runs the full proxy discrimination audit: mutual information between rating factors and protected characteristics, counterfactual fairness tests, and a structured Markdown report with explicit FCA regulatory mapping you can drop into a pricing committee pack.

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=catboost_model,
    data=policy_df,
    protected_cols=["gender"],
    factor_cols=["postcode_district", "vehicle_age", "ncd_years"],
    exposure_col="exposure",
)
audit.run().save_report("fairness_audit_2026Q1.md")
```

[insurance-fairness](/insurance-fairness/)

---

### insurance-datasets - synthetic test data

Every code example in this stack uses `insurance-datasets`. It generates realistic synthetic UK personal lines portfolios with configurable size, rating factor distributions, and known ground truth parameters. You can test your code before your data is available and verify your implementations produce the right answer.

```python
# uv add insurance-datasets  # github.com/burning-cost/insurance-datasets
# from insurance_datasets import load_motor
# df = load_motor(n_policies=50_000, seed=42)

# Alternatively, generate structured synthetic data directly:
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 50_000
df = pd.DataFrame({
    "driver_age_band": rng.choice(["17-21","22-25","26-35","36-50","51+"], n),
    "vehicle_group": rng.choice([f"G{i}" for i in range(1,17)], n),
    "area": rng.choice([f"R{i}" for i in range(1,13)], n),
    "ncd_years": rng.integers(0, 9, n),
    "exposure": rng.uniform(0.1, 1.0, n),
    "claim_count": rng.poisson(0.08, n),
    "pure_premium": rng.exponential(150, n),
})
```

[github.com/burning-cost/insurance-datasets](https://github.com/burning-cost/insurance-datasets)

---

## Model Building

Libraries for specific modelling challenges that the standard GLM/GBM toolchain does not cover well.

### bayesian-pricing - thin segments

A representative UK motor rating model has roughly 4.5 million theoretical cells. A mid-sized insurer with one million policies covers perhaps 3% of them with more than 30 observations. The thin-data problem is worse at interaction level: "age 17-21 × ABI group 40+ × London" may have 8 claims across five years. That is not enough to estimate frequency to within 30%.

`bayesian-pricing` uses partial pooling via PyMC hierarchical models. Thin segments borrow strength from related segments. The degree of borrowing is determined by the data, not a hyperparameter you tune by hand.

```python
from bayesian_pricing import HierarchicalFrequency

model = HierarchicalFrequency(
    group_cols=["driver_age_band", "vehicle_group"],
    exposure_col="exposure",
)
model.fit(df)
print(model.relativities())  # posterior means + 95% credible intervals
```

For the full partial-pooling methodology, see [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/).

[github.com/burning-cost/bayesian-pricing](https://github.com/burning-cost/bayesian-pricing)

---

### insurance-spatial - territory banding

Territory banding by eye from a modelled loss cost map is the most common approach in UK personal lines. It is also demonstrably wrong: human-drawn bands ignore spatial autocorrelation, produce bands whose internal variance is higher than it needs to be, and have no principled criterion for how many bands to use.

`insurance-spatial` uses BYM2 spatial models (Riebler et al. 2016) to estimate geographically smoothed loss cost surfaces that respect postcode district adjacency structure, then applies graph-based clustering to derive rate bands with minimum within-band variance.

```python
from insurance_spatial import BYM2Model, build_grid_adjacency

# BYM2Model estimates spatially smoothed loss costs respecting adjacency
spatial = BYM2Model()
spatial.fit(
    y=df["claim_count"].values,
    E=df["exposure"].values,
    adjacency=build_grid_adjacency(df["postcode_district"]),
)
# derive_bands would be a downstream clustering step
```

See Your Territory Banding is Wrong for what the BYM2 model finds that choropleth maps miss.

[github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial)

---

### insurance-interactions - GLM interaction detection

Adding interaction terms to a GLM is guesswork without a systematic search. A portfolio with 20 rating factors has 190 possible pairwise interactions. `insurance-interactions` uses a three-stage pipeline: a Combined Actuarial Neural Network (CANN) trained on GLM residuals, Neural Interaction Detection (NID) scoring from the CANN weight matrices to rank all candidate pairs, and GLM likelihood-ratio testing on the top-K shortlist. The result is a ranked interaction table with deviance improvement, AIC/BIC, and Bonferroni-corrected p-values - not a list of guesses.

```python
from insurance_interactions import build_glm_with_interactions, test_interactions

# test_interactions ranks candidate pairs by NID score and runs LR tests
candidates = test_interactions(
    X_train, y_train, exposure=exposure, family="poisson", n_top=20
)
print(candidates)  # ranked by NID score with LR test results and Bonferroni-corrected p-values

# build_glm_with_interactions adds the significant pairs to a new GLM
glm_with_ints, glm_df = build_glm_with_interactions(
    X_train, y_train, exposure=exposure,
    interaction_pairs=candidates.head(5)[["feature_a", "feature_b"]].to_dicts(),
)
```

See [Finding the Interactions Your GLM Missed](/2026/02/27/finding-the-interactions-your-glm-missed/) for why the standard approach misses most of the signal.

[github.com/burning-cost/insurance-interactions](https://github.com/burning-cost/insurance-interactions)

---

### insurance-glm-tools - nested GLMs and clustering

Two specific GLM problems that statsmodels does not solve. First: nested GLMs, where you want to segment the portfolio into clusters and fit separate GLMs per cluster, with the cluster structure itself estimated from data rather than defined by hand. Second: GLM coefficient smoothing via Whittaker-Henderson, for when raw GLM factor relativities oscillate implausibly across ordered bands (age, vehicle age, NCD).

```python
from insurance_glm_tools import FactorClusterer, NestedGLMPipeline

# FactorClusterer groups sparse factor levels into defensible composite levels
# lambda_ controls regularisation: 'bic' selects it automatically
clusterer = FactorClusterer(family="poisson", lambda_="bic")
clusterer.fit(X_train["vehicle_group"], y_train, exposure=exposure)
level_map = clusterer.level_map  # mapping from original levels to clusters

# NestedGLMPipeline fits segment-specific GLMs with data-driven cluster structure
pipe = NestedGLMPipeline(family="poisson")
pipe.fit(X_train, y_train, exposure=exposure)
```

[github.com/burning-cost/insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools)

---

### insurance-gam - interpretable boosting machines

Explainable Boosting Machines (EBMs, Lou et al. 2013) are additive models with interaction terms that are fully interpretable by construction. Each feature contribution is a learned shape function. For a pricing actuary, EBMs sit between a GLM (fully interpretable, constrained functional form) and a GBM (maximum lift, black box). On UK motor data, EBMs typically close 60-70% of the Gini gap between GLM and GBM while remaining directly explainable without post-hoc tools.

```python
# insurance-gam provides EBM tooling via the interpret package integration
# The package wraps EBM setup for insurance Poisson/Tweedie objectives
# uv add insurance-gam  # github.com/burning-cost/insurance-gam
from interpret.glassbox import ExplainableBoostingRegressor

ebm = ExplainableBoostingRegressor(interactions=5, objective="poisson")
ebm.fit(X_train, y_train, sample_weight=exposure)
ebm.explain_local(X_train[:5])  # shows per-prediction shape contributions
```

[github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)

---

### insurance-frequency-severity - frequency/severity dependence

The standard two-model GLM multiplies `E[N|x]` by `E[S|x]` and assumes independence. The assumption is wrong in UK motor. The NCD structure suppresses borderline claims. Policyholders near the NCD threshold do not report small incidents. The result is a systematic negative dependence between claim count and average severity.

Vernic, Bolancé and Alemany (2022) found this mismeasurement runs at €5-55 per policyholder on a Spanish auto book. The UK direction is the same. `insurance-frequency-severity` fits a Sarmanov copula over your existing frequency and severity GLMs, estimates the dependence parameter, and corrects the joint prediction.

```python
from insurance_frequency_severity import JointFreqSev, DependenceTest

test = DependenceTest()
test.fit(n=claims_df["claim_count"], s=claims_df["avg_severity"])
print(test.summary())  # omega, p-value, directional interpretation

model = JointFreqSev(freq_glm=my_nb_glm, sev_glm=my_gamma_glm, copula="sarmanov")
model.fit(claims_df)
```

[github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity)

---

### insurance-distributional - variance modelling

A point estimate of pure premium is not enough when you are setting capital requirements or evaluating risk-adequate pricing for volatile segments. `insurance-distributional` fits distributional regression models - GAMLSS-style, where location, scale, and shape parameters are all functions of rating factors - to give you a full predictive distribution per policy, not just a conditional mean.

```python
from insurance_distributional import TweedieGBM

# TweedieGBM predicts the full Tweedie distribution per risk
dglm = TweedieGBM()
dglm.fit(X_train, y_train, sample_weight=exposure_train)
pred = dglm.predict(X_test)
# pred.mean() is the point estimate; pred.quantile(0.95) is the 95th percentile
```

[github.com/burning-cost/insurance-distributional](https://github.com/burning-cost/insurance-distributional)

---

## Validation and Governance

What the PRA expects, implemented as runnable code.

### insurance-governance - PRA SS1/23 and model risk management

PRA Supervisory Statement SS1/23 (Model Risk Management Principles) and FCA TR24/2 require documented, independent validation with traceable audit trails. In practice most UK insurers have a collection of ad-hoc scripts, Excel workbooks, and email threads. `insurance-governance` turns that into a structured, reproducible HTML report with a JSON sidecar for model risk system ingestion. It also implements the model risk tiering framework: a 0-100 composite risk score maps each model to Tier 1/2/3, determining what governance is required.

```python
from insurance_governance import ModelValidationReport, RiskTierScorer, ValidationModelCard

card = ValidationModelCard(
    name="Motor Frequency v3.2",
    methodology="CatBoost with Poisson objective",
    target="claim_count",
)
report = ModelValidationReport(model_card=card, y_val=y_val, y_pred_val=y_pred)
report.generate("validation_report_2026Q1.html")  # Gini, PSI, lift charts, regulatory refs

tier = RiskTierScorer().score(mrm_card)
print(f"Model tier: {tier.tier} (score: {tier.score}/100)")
```

[github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance)

---

### insurance-calibration - balance property

A well-specified pricing model should satisfy the balance property: the sum of predicted premiums should equal the sum of actual losses in aggregate, and within every rating factor band. Failures of calibration within bands reveal systematic mispricing by segment that aggregate metrics hide. `insurance-calibration` runs the full suite of calibration tests - actual-to-expected by factor, Hosmer-Lemeshow bins, calibration curves - and flags the bands where your model is consistently above or below one.

```python
from insurance_monitoring import CalibrationReport, check_balance, check_auto_calibration, murphy_decomposition

# check_balance: does E[predicted] == E[actual] per segment?
bal = check_balance(y_val, y_pred, exposure=exposure_val)
# CalibrationReport bundles balance, auto-calibration and Murphy decomposition
report = CalibrationReport(
    balance=bal,
    auto_calibration=check_auto_calibration(y_val, y_pred, exposure=exposure_val),
    murphy=murphy_decomposition(y_val, y_pred, distribution="poisson"),
    distribution="poisson",
    n_policies=len(y_val),
    total_exposure=float(exposure_val.sum()),
)
print(report.verdict())
```

[github.com/burning-cost/insurance-calibration](https://github.com/burning-cost/insurance-calibration)

---

### insurance-conformal - prediction intervals

Your Tweedie GBM gives point estimates. A pricing actuary needs to know the uncertainty around those estimates - not as a parametric confidence interval that depends on distributional assumptions, but as a guarantee: this interval will contain the actual loss at least 90% of the time, for any data distribution.

Conformal prediction provides that guarantee. The implementation detail that matters for insurance: the standard approach uses raw absolute residuals as non-conformity scores, which treats a £1 error on a £100 risk identically to a £1 error on a £10,000 risk. `insurance-conformal` uses Pearson-weighted residuals that respect the heteroscedasticity of insurance data, producing around 30% narrower intervals with identical coverage guarantees.

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(
    model=catboost_model,
    nonconformity="pearson_weighted",
    distribution="tweedie",
)
cp.calibrate(X_cal, y_cal, coverage=0.90)
lower, upper = cp.predict_interval(X_test)
```

The full methodology, based on Manna et al. (2025), is in [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/).

[github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

---

## Causal Inference

Rating factors are not causes. They are correlates. For most pricing purposes, that is good enough. For three specific decisions, it is not: evaluating whether a rating factor's effect is genuine or proxied through a confounder, measuring price elasticity for renewal pricing, and proving a rate change actually worked.

### insurance-causal - double machine learning

Double Machine Learning (Chernozhukov et al. 2018) estimates the causal effect of a treatment - a rating factor, a price change, a telematics score - controlling for high-dimensional confounders using flexible ML models, while preserving valid frequentist inference on the causal parameter. For a GLM coefficient interpretation question ("is this engine size effect genuine or residual confounding from driver type?"), DML gives you a clean answer.

```python
from insurance_causal import CausalPricingModel

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="count",
    treatment="engine_size_cc",
    confounders=["driver_age_band", "vehicle_group", "area", "ncd_years"],
)
result = model.fit(df, exposure_col="exposure")
print(result.ate, result.ci_95)  # average treatment effect + confidence interval
```

For the mechanism and a worked motor example, see Your Rating Factor Might Be Confounded.

[github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal)

---

### insurance-causal-policy - rate change evaluation

You applied a 15% rate increase to a segment six months ago. The loss ratio on that segment has improved. Did the rate change work? Or did the rate increase attract better risks (adverse selection reversing), or did claims inflation happen to cool in that period? The naive before/after loss ratio comparison conflates all three.

`insurance-causal-policy` implements difference-in-differences and synthetic control designs for insurance rate changes, using adjacent unexposed segments as controls and adjusting for exposure-period mix changes.

```python
from insurance_causal_policy import SDIDEstimator, PolicyPanelBuilder

# Build a Polars panel DataFrame from policy-level data
panel = PolicyPanelBuilder().build(df, unit_col="policy_segment", period_col="period",
                                   outcome_col="loss_ratio", treatment_col="is_treated")

# SDIDEstimator: synthetic difference-in-differences for insurance rate change evaluation
sdid = SDIDEstimator(panel, outcome="loss_ratio")
result = sdid.fit()   # returns SDIDResult directly
print(result.att, result.se)  # average treatment effect on treated
```

[github.com/burning-cost/insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy)

---

### insurance-uplift - retention heterogeneous treatment effects

*Note: the `insurance-uplift` repository has been archived and is no longer maintained. The approach is documented here for reference.*

Not every customer responds equally to a price change. A 5% premium increase may have near-zero effect on a long-tenure, direct-debit paying customer and a large effect on a price-comparison website customer who shopped aggressively last year. Treating the whole book as one segment leaves retention profit on the table.

The archived library estimated heterogeneous treatment effects on renewal probability using causal forests (Wager and Athey 2018), segmented the book by response profile, and constructed the optimal individualised retention intervention. The same causal forest machinery is available via `insurance-optimise` for the elasticity estimation use case.

---

## Optimisation and Deployment

Getting from a technical price to an implemented rate change.

### insurance-optimise - rate optimisation

The standard pricing cycle applies rate changes manually: technical model produces adequate prices, demand model produces elasticity estimates, someone reconciles the two in a spreadsheet while watching the loss ratio. This manual process is not optimising anything. It is approximating a joint optimisation problem with sequential heuristics.

`insurance-optimise` solves it directly: maximise expected profit subject to the FCA PS21/11 ENBP constraint, a loss ratio ceiling, a retention floor, GWP bounds, and individual rate change guardrails. Analytical Jacobians make this fast enough for N=10,000 policies in seconds.

```python
from insurance_optimise import PortfolioOptimiser, ConstraintConfig

constraints = ConstraintConfig(
    lr_max=0.70, retention_min=0.85, max_rate_change=0.20, enbp_buffer=0.01,
)
opt = PortfolioOptimiser(
    technical_price=df["tc"].values,
    expected_loss_cost=df["expected_loss"].values,
    p_demand=df["p_renewal"].values,
    elasticity=df["elasticity"].values,
    enbp=df["enbp"].values,
    constraints=constraints,
)
result = opt.optimise()
print(result.optimal_multipliers)  # per-policy price multipliers
```

For the full treatment of the constrained rate optimisation problem and the efficient frontier interpretation, see Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement.

[github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

### insurance-optimise - price elasticity (formerly insurance-demand)

Before you can run `insurance-optimise`, you need elasticity estimates. `insurance-optimise` fits price elasticity models on renewal and new business data, with the confounding correction that naive logistic regression omits: risk factors drive both the price charged and the renewal decision, so OLS conflates the causal price effect with risk-driven selection.

The library wraps CausalForestDML for the causal component, with treatment variation diagnostics that flag the near-deterministic price problem before you fit. If your pricing grid leaves no residual variation in price after conditioning on risk factors, the elasticity estimate is noise.

```python
from insurance_optimise import ElasticityEstimator

elast = ElasticityEstimator(
    feature_cols=["age", "ncd_years", "vehicle_group", "area"],
    treatment_col="log_price_ratio",
    outcome_col="renewed",
)
elast.fit(df)
df["elasticity"] = elast.effect(df)  # individual-level semi-elasticities
```

[github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

### insurance-deploy - champion/challenger

A pricing model without a structured champion/challenger framework has no way to prove it outperforms its predecessor. `insurance-deploy` implements the full champion/challenger infrastructure: traffic splitting (by volume or stratified by risk segment), a hold-out audit trail for FCA review, lift calculation between the challenger GBM and the champion GLM on the same live population, and a model registry with version control for pricing committee sign-off.

```python
from insurance_deploy import Experiment, ModelRegistry, ModelVersion

registry = ModelRegistry(path="/data/model_registry.db")
champ_v = registry.register("glm_v3", glm_model)
challgr_v = registry.register("catboost_v1", catboost_model)

# Experiment routes quotes deterministically by policy_id hash
cc = Experiment(
    name="glm_vs_catboost",
    champion=champ_v,
    challenger=challgr_v,
    challenger_pct=0.10,
    mode="live",
)
df["cohort"] = df["policy_id"].map(lambda pid: "challenger"
                                   if cc.route(pid) == challgr_v else "champion")
```

[github.com/burning-cost/insurance-deploy](https://github.com/burning-cost/insurance-deploy)

---

## Advanced and Specialist

The remaining libraries cover problems that are real but less universal. Brief descriptions:

**Credibility and experience rating**

- [insurance-credibility](https://github.com/burning-cost/insurance-credibility): Bühlmann-Straub group credibility for scheme and fleet pricing, plus Bayesian experience rating at individual policy level. When a fleet scheme has three years of loss history and you need the optimal weight between own experience and market rate, this is the calculation. [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/).

- [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data): When you have under 5,000 policies and no related book to borrow from, TabPFN and TabICLv2 foundation models work on small datasets without overfitting. Merged from `insurance-tabpfn` and `insurance-transfer`.

**Loss development and trend**

- [insurance-trend](https://github.com/burning-cost/insurance-trend): Log-linear trend fitting for frequency and severity, with structural break detection (COVID lockdown, Ogden rate changes), ONS index integration for severity deflation, and piecewise refit on detected breaks. UK motor claims inflation ran 34% from 2019 to 2023 versus CPI of 21%. That is a 13-point superimposed component that CPI alone will not capture.

- [insurance-dynamics](https://github.com/burning-cost/insurance-dynamics): Bayesian change-point detection for pricing time series. When you need to know whether a shift in your loss ratio is structural or noise, before you take rate.

**Fairness and equity**

- [insurance-fairness](https://github.com/burning-cost/insurance-fairness): Optimal transport-based discrimination-free pricing using the Lindholm-Richman-Tsanakas-Wüthrich causal path decomposition. Decomposes premium differences between groups into the component justified by risk and the component that cannot be justified.

- [insurance-fair-longterm](https://github.com/burning-cost/insurance-fair-longterm): Multi-state fairness for long-term insurance products. First Python implementation of the arXiv:2602.04791 framework (February 2026).

**Severity and large loss**

- [insurance-nflow](https://github.com/burning-cost/insurance-nflow) (archived): Normalising flows for severity distribution modelling. When a Pareto or Lognormal tail does not fit your large loss data and you need a more flexible parametric form.

- [insurance-quantile](https://github.com/burning-cost/insurance-quantile): Extreme quantile regression neural networks. For when you care about the 99th percentile of the severity distribution, not just the mean. That matters for excess of loss treaty pricing and capital allocation.

**Survival and lapse**

- [insurance-survival](https://github.com/burning-cost/insurance-survival): Survival analysis for insurance pricing. Merges mixture cure models (covariate-adjusted, not univariate), competing risks via Fine-Gray regression, shared frailty for recurrent events, and survival-adjusted customer lifetime value. lifelines is excellent; these are the gaps it leaves. [Experience Rating, NCD, and Bonus-Malus Systems](/2026/02/27/experience-rating-ncd-bonus-malus/).

**Telematics**

- [insurance-telematics](https://github.com/burning-cost/insurance-telematics): Raw 1Hz GPS/accelerometer trip data to GLM-ready driver risk scores. Hidden Markov Model driving state classification (cautious/normal/aggressive), Bühlmann-Straub credibility weighting to driver level, Poisson frequency GLM output. Implements Jiang and Shi (2024, NAAJ).

**Robustness and portability**

- [insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift): Density ratio correction when deploying a model trained on one portfolio to a different book distribution: via a new channel, a transferred portfolio, or a book that has shifted materially since the model was trained.

- [insurance-dro](https://github.com/burning-cost/insurance-dro): Distributionally robust optimisation for premium setting under model uncertainty. Sets premia that remain adequate under worst-case distributional perturbations within a Wasserstein ball around the empirical distribution.

- [insurance-sensitivity](https://github.com/burning-cost/insurance-sensitivity): Global sensitivity analysis via Sobol indices and Shapley effects. Measures which input features drive variance in model outputs, for model governance documentation and variable selection decisions.

**Multivariate and joint**

- [insurance-conformal](https://github.com/burning-cost/insurance-conformal) (`insurance_conformal.multivariate`): Joint conformal prediction intervals for multi-output models. Use it when you need simultaneous coverage for frequency and severity, not just marginal intervals for each. The multivariate functionality is part of `insurance-conformal` v0.4.0+.

- [insurance-copula](https://github.com/burning-cost/insurance-copula) (archived): D-vine temporal dependence and two-part occurrence/severity copula models for cases where the Sarmanov structure in `insurance-frequency-severity` is insufficient.

---

## Where to Start

If you are installing for the first time, the order matters:

1. `insurance-datasets` - you need synthetic data to run anything else
2. `insurance-cv` - fix your CV methodology before tuning any model
3. `shap-relativities` - if you have a GBM, extract the factor tables first
4. `insurance-fairness` - run the proxy discrimination audit before the model goes to pricing committee
5. `insurance-governance` - generate the PRA SS1/23 validation report in parallel with the model build, not after

Everything else is additive depending on your specific problem. Thin segments? Add `bayesian-pricing`. Territory banding? Add `insurance-spatial`. Going to production with a challenger? Add `insurance-deploy`.

All 35 libraries are MIT-licensed, installable via `pip` or `uv`, and maintained under [github.com/burning-cost](https://github.com/burning-cost). Each ships with synthetic data generators so you can run the examples without needing a live portfolio.
