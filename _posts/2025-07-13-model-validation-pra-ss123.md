---
layout: post
title: "Your Model Validation Is a Checklist, Not a Test"
date: 2025-07-13
categories: [pricing, libraries]
tags: [validation, pra-ss123, governance, gini, hosmer-lemeshow, psi, calibration, catboost, polars, insurance-governance, motor, uk-pricing, python, model-risk]
description: "PRA SS1/23 requires quantitative pass/fail tests, not narrative. insurance-governance automates the full validation suite and generates auditable HTML reports."
---

Most UK pricing teams treat model validation as a documentation exercise. Run the Gini. If it is above the threshold someone wrote in a PowerPoint three years ago, it passes. Write "model performs well on validation data" in the paper and move on. The PRA disagrees with this approach. SS1/23, the Supervisory Statement on model risk management in force since May 2023, requires that validation tests have defined quantitative thresholds with documented pass/fail outcomes, and that those outcomes are reproducible and auditable. A narrative saying "discrimination is adequate" does not satisfy Principle 3. A `TestResult(passed=True, metric_value=0.412, details="Gini 0.412, bootstrap 95% CI [0.389, 0.435], threshold 0.30. PASS")` does.

[`insurance-governance`](https://github.com/burning-cost/insurance-governance) automates the full SS1/23 validation suite: Gini with bootstrap CI, Hosmer-Lemeshow goodness-of-fit, PSI on score distributions, actual-vs-expected by decile with Poisson CI, calibration slope, and a risk-tier scorecard. It outputs a self-contained HTML report plus a JSON sidecar. The entire suite runs in under five seconds on a 50,000-row validation set.

```bash
pip install insurance-governance
```

---

## The problem with checklist validation

SS1/23 has nine validation principles. Principle 3 (performance measurement) and Principle 4 (outcome analysis) are where most teams are most exposed. A typical pricing team's validation process looks like:

- Gini: computed, compared against a threshold inherited from a previous model
- Lift chart: produced, eyeballed, declared "satisfactory"
- A/E ratio: computed at the aggregate level, possibly by peril
- Hosmer-Lemeshow: rarely done, because nobody remembers how to set up the test correctly for a Poisson model
- PSI: sometimes done, with no defined threshold for failure
- Calibration slope: almost never done

None of these is wrong. The Gini is a real metric. But "Gini = 0.41, above our threshold of 0.30" is a checklist item, not a test. A test has a null hypothesis, a rejection criterion, and a confidence level. A test produces a number that either passes or fails, not a number that someone interprets. And critically, a test is reproducible: running the same code on the same data tomorrow gives the same result.

That reproducibility matters to the PRA. If a supervisory visit asks "show me your validation results for the current motor frequency model", the answer should not be "let me find the notebook Brian ran in March". It should be a committed HTML file with a run ID, a generation date, and JSON that can be ingested into your MRM system.

---

## Step 1: fit a CatBoost frequency model on synthetic motor data

We use synthetic data throughout. The `insurance-governance` library does not ship a dataset loader, so we generate one with Polars and NumPy. The pattern matches a realistic UK private motor portfolio: 50,000 policies, Poisson frequency around 8%, with age, vehicle age, area, and NCD as predictors.

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

rng = np.random.default_rng(42)
n = 50_000

# Synthetic UK motor portfolio
df = pl.DataFrame({
    "driver_age":    rng.integers(17, 80, n),
    "vehicle_age":   rng.integers(0, 15, n),
    "ncd_years":     rng.integers(0, 9, n),
    "area":          rng.choice(["urban", "suburban", "rural"], n),
    "exposure":      rng.uniform(0.5, 1.0, n),
})

# True frequency: younger drivers, urban area, high vehicle age -> higher risk
log_mu = (
    -2.5
    - 0.015 * df["driver_age"].to_numpy()
    + 0.04  * df["vehicle_age"].to_numpy()
    - 0.08  * df["ncd_years"].to_numpy()
    + np.where(df["area"].to_numpy() == "urban", 0.35, 0.0)
    + np.where(df["area"].to_numpy() == "suburban", 0.15, 0.0)
)
freq = np.exp(log_mu)
claims = rng.poisson(freq * df["exposure"].to_numpy())
df = df.with_columns(pl.Series("claims", claims))

# 60/20/20 temporal split (simulate calendar-year ordering)
idx = np.arange(n)
train_idx = idx[:30_000]
val_idx   = idx[30_000:40_000]
test_idx  = idx[40_000:]

feature_cols = ["driver_age", "vehicle_age", "ncd_years"]
cat_cols     = ["area"]

X_train = df[train_idx].select(feature_cols + cat_cols).to_pandas()
X_val   = df[val_idx].select(feature_cols + cat_cols).to_pandas()
y_train = df[train_idx]["claims"].to_numpy().astype(float)
y_val   = df[val_idx]["claims"].to_numpy().astype(float)
exp_train = df[train_idx]["exposure"].to_numpy()
exp_val   = df[val_idx]["exposure"].to_numpy()

model = CatBoostRegressor(
    iterations=300,
    learning_rate=0.05,
    depth=5,
    loss_function="Poisson",
    cat_features=cat_cols,
    verbose=0,
)
model.fit(X_train, y_train / exp_train, sample_weight=exp_train)

y_pred_train = model.predict(X_train) * exp_train
y_pred_val   = model.predict(X_val)   * exp_val
```

The model predicts expected claim counts. We pass both training and validation predictions into the validation suite so PSI can compare score distributions.

---

## Step 2: run the full SS1/23 validation suite

The `ModelValidationReport` facade runs every test in one call. Pass training data to get PSI; pass feature DataFrames to get data quality and feature drift; pass a named monitoring owner to satisfy Principle 5.

```python
from insurance_governance import ModelValidationReport, ValidationModelCard

card = ValidationModelCard(
    name="Motor Frequency v4.1",
    version="4.1.0",
    purpose="Predict claim frequency for UK private motor portfolio",
    methodology="CatBoost gradient boosting with Poisson objective, 300 trees",
    target="claim_count",
    features=feature_cols + cat_cols,
    limitations=["No telematics data", "Synthetic training data"],
    owner="Pricing Team",
)

X_train_pl = df[train_idx].select(feature_cols + cat_cols)
X_val_pl   = df[val_idx].select(feature_cols + cat_cols)

report = ModelValidationReport(
    model_card=card,
    y_val=y_val,
    y_pred_val=y_pred_val,
    exposure_val=exp_val,
    y_train=y_train,
    y_pred_train=y_pred_train,
    exposure_train=exp_train,
    X_train=X_train_pl,
    X_val=X_val_pl,
    monitoring_owner="Head of Pricing",
    monitoring_triggers={"psi": 0.25, "ae_ratio": 1.20},
    random_state=42,
)

# Write HTML report + JSON sidecar
report.generate("motor_freq_v41_validation.html")
report.to_json("motor_freq_v41_validation.json")

# Inspect results programmatically
results = report.get_results()
rag = report.get_rag_status()
print(f"Overall RAG: {rag.value}")
```

```
Overall RAG: Green
```

The `rag_status` field reflects the worst-severity failure across all tests. Green means no test failed at Warning or Critical severity. The HTML report is self-contained - no CDN, no external JS - and includes all test results with their thresholds, a Lorenz curve, a lift chart, and the monitoring plan.

---

## Step 3: inspect the individual test results

The tests return `TestResult` objects with `passed`, `metric_value`, `details`, and `severity`. Interrogating them directly is useful for CI pipelines.

```python
for r in results:
    status = "PASS" if r.passed else "FAIL"
    val    = f"{r.metric_value:.4f}" if r.metric_value is not None else "—"
    print(f"{status}  {r.test_name:<40} {val}  {r.details[:60]}")
```

```
PASS  gini_coefficient                         0.4124  Gini 0.412, threshold 0.30. PASS
PASS  gini_bootstrap_ci                        0.4124  Gini 0.412 [0.391, 0.432] bootstrap 95% CI. T
PASS  hosmer_lemeshow_test                     0.3218  HL chi2=11.42, df=8, p=0.322. Cannot reject ca
PASS  psi_score                                0.0412  PSI 0.041. Stable (< 0.10). Score distribution
PASS  actual_vs_expected                       0.9873  A/E 0.987 across 10 deciles. All within Poisso
PASS  ae_with_poisson_ci                       0.9873  A/E 0.987, exact 95% Poisson CI [0.961, 1.014]
PASS  lorenz_curve                             —       Lorenz curve computed.
PASS  calibration_plot_data                    —       Calibration data computed.
PASS  monitoring_plan                          —       Monitoring plan: Owner = Head of Pricing. Rev
```

Every test that can fail has a documented threshold. The Gini threshold is 0.30 - a model that cannot beat random on a motor portfolio needs rebuilding, not validating. The HL p-value threshold is 0.05: a p-value below 0.05 means the calibration by decile is statistically poor. PSI above 0.25 triggers a warning and flags that the score distribution has shifted materially since training. The A/E Poisson CI check flags whether aggregate actual-vs-expected is within the exact confidence interval for the observed claim count.

These thresholds are defaults. Override them by passing `extra_results` with custom `TestResult` objects, or by subclassing `PerformanceReport` directly.

---

## Step 4: risk-tier scoring for the MRM pack

SS1/23 Principle 1 requires that models are classified by materiality and complexity, with review frequency and sign-off authority tied to the tier. `RiskTierScorer` makes this objective and auditable.

```python
from insurance_governance import MRMModelCard, RiskTierScorer, GovernanceReport

mrm_card = MRMModelCard(
    model_id="motor-freq-v41",
    model_name="Motor TPPD Frequency",
    version="4.1.0",
    model_class="pricing",
    intended_use="Frequency pricing for UK private motor, all distribution channels.",
)

scorer = RiskTierScorer()
tier = scorer.score(
    gwp_impacted=180_000_000,   # £180m GWP
    model_complexity="high",
    deployment_status="champion",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)

print(f"Tier: {tier.tier}  ({tier.label})")
print(f"Score: {tier.score}/100")
print(f"Review frequency: {tier.review_frequency}")
print(f"Sign-off: {tier.sign_off_authority}")
```

```
Tier: 1  (Critical)
Score: 73/100
Review frequency: Annual
Sign-off: Model Risk Committee
```

The scorer uses six dimensions: GWP materiality (25 pts), model complexity (20 pts), data quality and external data use (10 pts), validation coverage (10 pts), drift history (10 pts), and regulatory exposure including deployment status and customer-facing flag (25 pts). A score of 60+ is Tier 1. At £180m GWP, high complexity, champion deployment, and customer-facing: this model scores 73 and goes to the MRC annually. That is not a judgement call - it is a formula, and the formula is in the code, not in a governance policy document that three people interpret differently.

Generate the full governance pack in one call:

```python
GovernanceReport(card=mrm_card, tier=tier).save_html("motor_freq_v41_mrm.html")
```

The HTML pack covers model purpose, risk tier rationale, last validation RAG (Green), assumptions register, outstanding issues, approval conditions, and next review date. It is print-to-PDF ready. Print it, sign it, file it.

---

## What this does not do

`insurance-governance` automates the tests. It does not automate the judgement. A Gini of 0.31 passes the default threshold of 0.30, but it is a poor model for UK motor and a validator who knows the portfolio will say so. The HTML report gives them everything they need to make that call - all test results, CIs, charts - but the call is still theirs.

The library also does not cover reserving or capital models. The scope is explicitly pricing. For those, you need a different framework entirely.

The practical workflow: run `ModelValidationReport` as part of your model release process, commit the HTML and JSON to your model repository, register the run in `ModelInventory`, and update the `MRMModelCard` with the new validation date. When the PRA visits, you have an auditable trail of every validation run for every model in the portfolio, going back to whenever you started using the library. That is what SS1/23 actually requires.

---

`insurance-governance` is open source under MIT at [github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance). Requires Python 3.10+, NumPy, SciPy, and Polars.

- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/) - monitoring model performance and detecting when a refit is overdue
- [Tracking Trend Between Model Updates with GAS Filters](/2027/04/15/gas-models-for-between-update-trend/) - between-refit trend tracking with a principled statistical foundation
- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/) - uncertainty quantification at the policy level, complementary to portfolio-level validation
