---
layout: post
title: "Blending GLMs and GBMs for UK Pricing: Cross-Validated Weights, Not a Choice Between Them"
date: 2026-02-28
featured: true
categories: [pricing, techniques]
tags: [GLM, GBM, model-blending, stacking, catboost, insurance-cv, shap-relativities, insurance-governance, pra-ss123, python, temporal-cv]
description: "When to use gradient boosting vs GLM for insurance pricing: the case for blending, cross-validated weights, PRA interpretability, and the conditions where each approach wins."
seo_title: "When to Use Gradient Boosting vs GLM for Insurance Pricing"
---

Most UK pricing teams frame the GLM-vs-GBM question as a choice: run the GLM because it's defensible, or run the GBM and spend the rest of your life arguing with compliance. We think this is the wrong frame entirely.

The GLM and the GBM are measuring different things. The GLM is a constrained, interpretable approximation of your rating structure. The GBM is an unconstrained predictor of claim frequency or severity. Neither is the truth. Both contain signal the other is missing. A blend of the two, done with care, outperforms either on holdout data while keeping enough GLM structure to satisfy a PRA model validation.

This post is about how to do that blend in production, using actual code.

---

## When to use gradient boosting vs GLM: the direct answer

If you are looking for a quick rule:

- **Use a GLM** when your book is under 50,000 policies, you need a factor table for a rating engine (Emblem, Radar), or a regulator/validator will ask you to explain each variable's effect. GLMs are also right when interpretability is the primary constraint and you can accept a Gini 3-5 points below the GBM ceiling.
- **Use a GBM** (CatBoost, XGBoost) when you have enough data that variance is under control (typically 200k+ policies with adequate claims), you are comfortable with SHAP-based factor extraction for the governance conversation, and you want the full non-linear and interaction signal.
- **Blend both** — a 30/70 or 40/60 GLM/GBM mix — when you want most of the GBM's lift while retaining a factor table for the pricing committee. This is the dominant pattern at UK tier-1 insurers running production models in 2024-2025.

The rest of this post explains how to execute the blend correctly: honest cross-validation, CV-derived blend weights, SHAP factor extraction, and model governance registration.

---

## Why blending works at all

The starting intuition is bias-variance. A GLM is high-bias, low-variance: it imposes an additive log-linear structure that is almost certainly wrong, but that structure stabilises coefficient estimates and prevents the model from fitting noise. A GBM is low-bias, high-variance: it can fit interactions and nonlinearities the GLM misses, but on small-to-medium insurance books the variance is real and the holdout performance can be misleading (especially if your cross-validation has temporal leakage - more on that below).

When you blend predictions from a high-bias and a low-variance model, you get a middle ground that is often better than either. This is not a theoretical curiosity. In our experience on UK motor and home books, a 30/70 or 40/60 GLM/GBM blend typically captures 60-80% of the GBM's incremental Gini lift over the pure GLM, while retaining enough GLM structure that the interpretability argument is manageable.

The question is not whether to blend. It is how to pick the weights honestly.

---

## Getting the cross-validation right first

Before fitting any blend weights, you need an honest estimate of each model's out-of-sample performance. For insurance data this means temporal walk-forward validation, not k-fold.

Standard k-fold randomly assigns policies to folds. On a motor book where claims develop over 12-36 months, this leaks future development information into training folds and makes every model look better than it is. We've seen this inflate apparent GBM Gini scores by 4-6 points - enough to change decisions.

`insurance-cv` handles this correctly:

```python
from insurance_cv import walk_forward_split

# Generate walk-forward temporal splits with a 3-month IBNR buffer
splits = walk_forward_split(
    df=df,
    date_col="inception_date",
    min_train_months=24,        # require at least 2 years of history
    test_months=3,
    ibnr_buffer_months=3,       # respect 3-month reporting lag
)

for split in splits:
    train_idx, test_idx = split.get_indices(df)
    train, test = df[train_idx], df[test_idx]
    # fit GLM on train, predict on test
    # fit GBM on train, predict on test
```

The `ibnr_buffer_months=3` parameter is important. It creates a buffer between the end of each training window and the start of the test window, preventing IBNR-contaminated claims from appearing in both. Without this gap, a policy that incepted near the fold boundary appears in training with partially-developed claims, and in the test set with those same claims at a slightly earlier development stage. The model learns the development pattern as if it were a rating factor.

---

## Fitting blend weights from CV predictions

Once you have out-of-sample predictions from both models across all CV folds, fit the blend weight with a constrained optimisation on the combined holdout set:

```python
import polars as pl
import numpy as np
from scipy.optimize import minimize_scalar

# oof_preds has columns: policy_id, glm_pred, gbm_pred, actual, exposure
oof = pl.read_parquet("oof_predictions.parquet")

def poisson_deviance(alpha: float) -> float:
    blend = alpha * oof["glm_pred"] + (1 - alpha) * oof["gbm_pred"]
    # weighted Poisson deviance
    mu = blend.to_numpy()
    y  = oof["actual"].to_numpy()
    w  = oof["exposure"].to_numpy()
    return float(np.sum(w * (mu - y * np.log(mu))))

result = minimize_scalar(poisson_deviance, bounds=(0.0, 1.0), method="bounded")
alpha_opt = result.x

print(f"Optimal GLM weight: {alpha_opt:.3f}")
print(f"GBM weight:         {1 - alpha_opt:.3f}")
```

The constrained search over `[0, 1]` is deliberate. Unconstrained stacking (learn arbitrary weights, allow negatives) will overfit the blend weights to the CV folds and produce results that do not generalise. For a two-model blend, the bounded scalar search is both fast and defensible.

On a typical UK motor book we see `alpha_opt` settling between 0.25 and 0.45. If it comes out above 0.6, the GBM is not adding much and you should ask whether it is worth the governance overhead. If it comes out below 0.1, you probably have a temporal leakage problem in the GBM CV that is making the GBM look better than it is.

---

## Practical production setup

In production, the blend is a prediction-time operation. Both models score every policy; the blend weight is applied to the log-predictions before exponentiating:

```python
import catboost as cb
import statsmodels.iolib.smpickle as smpickle

# Load trained models
gbm = cb.CatBoostRegressor()
gbm.load_model("models/gbm_freq_2026q1.cbm")

glm_result = smpickle.load_pickle("models/glm_freq_2026q1.pkl")

alpha = 0.35  # GLM weight from CV

def score_blend(df: pl.DataFrame) -> pl.Series:
    glm_log_pred = glm_result.predict(df.to_pandas())
    gbm_log_pred = gbm.predict(df.to_pandas())

    # blend in log space, then exponentiate
    blended_log = alpha * np.log(glm_log_pred) + (1 - alpha) * np.log(gbm_log_pred)
    return pl.Series("blended_freq", np.exp(blended_log))
```

Note that we blend in log space, not prediction space. Both the GLM and CatBoost Poisson model produce log-linear predictions, so blending the log-predictions is equivalent to taking a geometric mean weighted by `alpha`. Blending in prediction space (the arithmetic mean) is also common but produces slightly different results for the same `alpha` - it assigns relatively more weight to high predictions. For claim frequency, we prefer the geometric mean.

---

## Extracting interpretable factors from the blend

Here is the part the PRA cares about.

A pure GBM prediction cannot be decomposed into a factor table. A blended prediction that is 35% GLM and 65% GBM can. The GLM component is fully decomposed by construction. The GBM component can be decomposed via SHAP, using `shap-relativities`:

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(
    model=gbm,
    X=X_train,
    exposure=exposure_array,
)
sr.fit()

# Get multiplicative relativities from the GBM component
# Note: features to include are controlled via X at construction time
gbm_factors = sr.extract_relativities(
    base_levels={"driver_age_band": "30-39", "ncd_years": "5"},
    ci_level=0.95,
)
```

`shap-relativities` decomposes the GBM's predictions into per-factor multiplicative relativities using SHAP's efficiency axiom: the values sum (in log space) to the model output, so the relativities multiply to the model prediction. The output is exposure-weighted, has confidence intervals, and passes a reconstruction check verifying the SHAP values actually reproduce the model's predictions.

You can then present the blend to a model validation committee as: "The blend is 35% of the GLM factor table plus 65% of the GBM factor table (extracted via SHAP). Here are both tables side by side. Here is where they materially disagree. Here is why we believe the GBM estimates are credible."

That is a conversation most pricing committees can have. "The GBM predicts higher risk for this segment" with no factor table to inspect is not.

---

## Validation via insurance-governance

The blend is a new model for model risk management purposes. It goes in the model inventory with its own `run_id` and a validation report.

`insurance-governance` runs the standard battery - Gini CI, actual/expected, Hosmer-Lemeshow calibration, PSI on input distributions, double-lift - against both the pure GLM and the blend, so the committee can see the incremental lift and what it cost in complexity:

```python
from insurance_governance import ModelValidationReport, ValidationModelCard, ModelInventory, MRMModelCard, RiskTierScorer

# Build a model card for the blend (MRM inventory record)
card = MRMModelCard(
    model_id="motor-freq-blend-2026q1",
    model_name="Motor Frequency GLM-GBM Blend 2026 Q1",
    version="1.0.0",
    intended_use="GLM-GBM blend for UK motor frequency pricing",
    model_type="Poisson GLM (35%) + CatBoost (65%) blend, CV-derived weights",
    developer="Pricing Team",
    customer_facing=True,
)

# Score risk tier  -  RiskTierScorer takes explicit model attributes, not a card object
scorer = RiskTierScorer()
tier_result = scorer.score(
    gwp_impacted=25_000_000,
    model_complexity="high",
    deployment_status="challenger",
    regulatory_use=False,
    external_data=False,
    customer_facing=True,
)

# Register in the model inventory (JSON file, version-controlled)
inv = ModelInventory("governance/model_inventory.json")
inv.register(card, tier_result)

# Run validation report against blend predictions
val_card = ValidationModelCard(
    name=card.model_name,
    version=card.version,
    purpose=card.intended_use,
    methodology=card.model_type,
    target="claim_freq",
    features=FEATURES,
    owner=card.developer,
)
report = ModelValidationReport(
    model_card=val_card,
    y_val=df_holdout["actual"].to_numpy(),
    y_pred_val=df_holdout["blend_pred"].to_numpy(),
    exposure_val=df_holdout["exposure"].to_numpy(),
    y_train=df_train["actual"].to_numpy(),
    y_pred_train=df_train["blend_pred_train"].to_numpy(),
    exposure_train=df_train["exposure"].to_numpy(),
    X_val=df_holdout.select(FEATURES).to_pandas(),
    X_train=df_train.select(FEATURES).to_pandas(),
)
report.to_json("validation/motor_freq_blend_2026q1.json")
```

`RiskTierScorer` maps the model to a materiality tier aligned with PRA SS1/23. A blend that is rate-setting for a significant book is almost certainly Tier 2. The generated report includes the RAG status the PRA executive pack needs, cross-referenced to the validation metrics, with the `run_id` linking the validation output to the inventory record.

---

## When blending helps and when it doesn't

Blending helps most when:

- The GBM captures interactions the GLM misses, but the GBM CV performance is inflated because your book is too small to trust pure GBM variance. The blend takes the lift, dampens the noise.
- You have a segment where the GLM fits poorly (high residuals, systematic A/E drift) and the GBM has enough data to do better. The blend improves that segment without requiring a full model rewrite.
- A regulator or internal model validation team requires a multiplicative factor table. The blend with SHAP decomposition gives you one.

Blending does not help when:

- The GBM's holdout performance is genuinely better, not leakage-inflated, and you have enough data that its variance is controlled. In that case, distil the GBM directly into a GLM with `shap-relativities` and run that in production. The blend is a halfway house; distillation is the end state.
- The GLM and GBM strongly disagree on a particular segment and you do not know which is right. A blend will produce a number, but the number is not meaningful - it is the average of two models that are telling you contradictory things. Stop, investigate the disagreement, and resolve it before blending.
- The GBM was trained on the same time period as the GLM with no holdout discipline. You will not know the optimal `alpha` because you have no honest OOS predictions to optimise against.

---

## The PRA interpretability question

PRA SS1/23 does not ban GBMs. It requires that material models have documented validation, clear ownership, a risk-tiered governance process, and that model outputs can be explained to senior management and, if asked, to the regulator.

"Explained" does not mean "reconstructed from a hand-drawn factor table." It means understood: why is this risk segment priced higher? What factors drive it? What assumptions underlie the model? Are those assumptions monitored?

A blend with GLM base and SHAP-extracted GBM factors satisfies this in a way a pure GBM does not. The GLM component gives you the factor table your rating engine, Emblem, and your pricing committee all understand. The GBM component gives you the lift. The blend weight is a documented, CV-derived parameter. The SHAP decomposition gives you a factor-level diagnosis when the blend moves.

This is not a trick to dress up a GBM in GLM clothing and hope the regulator doesn't notice. The blend genuinely behaves more like a GLM than a pure GBM: at `alpha = 0.35`, 35% of every prediction is driven by the fully-transparent GLM factor table. The interpretability is real, not cosmetic.

---

## Our recommendation

Start with the pure GLM. Fit the GBM. Run `insurance-cv` walk-forward validation on both. If the honest GBM lift - post-leakage-correction - is more than 2 Gini points, the blend is probably worth the governance overhead. Fit the blend weight from OOS predictions. Extract GBM factors with `shap-relativities`. Register the blend in `insurance-governance` before it goes near a rate change.

If the honest GBM lift is less than 2 points, do not blend. GLM complexity is invisible to pricing committees; GBM governance overhead is not. The blend only earns its keep when the lift is real.

- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/)
- [Why k-Fold CV Is Wrong for Insurance](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/)
- [One Package, One Install: PRA SS1/23 Validation and MRM Governance Unified](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
