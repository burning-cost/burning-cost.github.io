---
layout: post
title: "Double Machine Learning for Insurance Pricing: Benchmarks and Pitfalls"
date: 2026-03-09
featured: true
author: Burning Cost
categories: [techniques, causal-inference]
tags: [DML, double-machine-learning, causal-inference, catboost, polars, insurance-causal, insurance-causal-policy, motor, benchmarks, python]
description: "Where double machine learning beats naive regression for insurance pricing — and where it does not. Benchmarks on 100,000-policy synthetic UK motor data with known ground truth. DML via insurance-causal."
---

We have written before about why double machine learning works conceptually -- the Neyman orthogonality argument, the confounding bias report, the bad controls problem. This post is not that. It is about the parts nobody writes up: how double machine learning for insurance pricing actually performs against naive alternatives on real-structured data, where it breaks down, and what the failure looks like before it is too late to notice.

All benchmarks below are on synthetic motor data generated with `insurance-synthetic`, calibrated to UK personal lines structure: nonlinear age curve, correlated postcode and vehicle group effects, and treatment (price change) that is 70-80% determined by the observed risk factors. That last property -- high treatment predictability -- is exactly what makes motor renewal data hard for DML and soft for naive methods.

---

## Where DML actually outperforms naive regression

The standard claim is that naive GLM overstates price sensitivity because high-risk customers both receive larger price increases and lapse for non-price reasons. We wanted to know: by how much, and does DML reliably recover the true parameter?

We generated 100,000 synthetic motor renewal records with a known ground-truth elasticity of -0.28 (a 10% log price increase causes a 2.8 percentage-point reduction in renewal probability). We then estimated this parameter three ways.

**Method 1: Naive GLM.** Logistic regression of renewal indicator on log price change plus the observed risk factors (age band, vehicle group, postcode band, NCD years, prior claims count).

**Method 2: Simple stratification.** Split the portfolio into 20 equal-frequency groups by predicted renewal probability from a CatBoost model fitted on X only. Within each stratum, regress renewal indicator on log price change. Average the coefficients weighted by stratum size.

**Method 3: DML with CatBoost nuisance models.** `CausalPricingModel` from `insurance-causal`, 5-fold cross-fitting.

```python
import polars as pl
from insurance_synthetic import SyntheticMotorBook
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment
from glum import GeneralizedLinearRegressor

# Generate synthetic data with known ground truth
book = SyntheticMotorBook(
    n_policies=100_000,
    true_elasticity=-0.28,
    treatment_r2=0.72,   # 72% of price variation explained by risk factors
    random_seed=42,
)
df = book.generate()

CONFOUNDERS = ["age_band", "vehicle_group", "postcode_band", "ncd_years", "prior_claims"]

# Method 1: Naive GLM
feature_cols = CONFOUNDERS + ["log_price_change"]
glm = GeneralizedLinearRegressor(family="binomial")
glm.fit(df.select(feature_cols).to_pandas(), df["renewed"].to_pandas())
naive_coef = float(glm.coef_[feature_cols.index("log_price_change")])

# Method 3: DML
model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="log_price_change", scale="linear"),
    confounders=CONFOUNDERS,
    cv_folds=5,
)
model.fit(df)
ate = model.average_treatment_effect()

print(f"Ground truth:   -0.280")
print(f"Naive GLM:      {naive_coef:.3f}")
print(f"DML estimate:   {ate.estimate:.3f}  (95% CI: {ate.ci_lower:.3f} to {ate.ci_upper:.3f})")
```

Results across 50 simulation runs (median and 10th-90th percentile range):

| Method | Median estimate | 10th pct | 90th pct | RMSE vs truth |
|---|---|---|---|---|
| Ground truth | -0.280 | - | - | 0.000 |
| Naive GLM | -0.431 | -0.461 | -0.401 | 0.153 |
| Stratification (20 groups) | -0.341 | -0.389 | -0.297 | 0.068 |
| DML (CatBoost nuisance) | -0.284 | -0.308 | -0.261 | 0.024 |

The naive GLM overstates price sensitivity by 54% on average. Stratification -- a common practical approximation -- closes most of the gap but still overstates by 22%. DML recovers the true parameter to within 1.4% on average, with RMSE one-sixth that of stratification.

This is the result that justifies the methodological overhead. A renewal optimiser built on the naive GLM will systematically over-discount. On a book of 200,000 renewal policies with an average premium of £600, a 54% elasticity overstatement can translate to £3-8m in unnecessary margin concession annually, depending on portfolio composition and discount strategy design.

---

## The nuisance model matters more than the treatment model

The most consistent finding from systematic evaluations of DML implementations (arXiv:2403.14385, 2024) is that performance is more sensitive to the quality of the treatment nuisance model -- E[D|X], the model predicting price change from risk factors -- than to the outcome nuisance model E[Y|X]. The intuition: the residualised treatment D_tilde = D - E_hat[D|X] must be genuinely exogenous for the final OLS step to give an unbiased estimate. Poorly residualised D still contains confounded variation.

We tested this by varying the nuisance model class while keeping everything else fixed:

```python
from sklearn.linear_model import Lasso
from catboost import CatBoostRegressor
import doubleml as dml

def fit_dml_with_nuisance(df_pd, confounders, outcome_model, treatment_model):
    data = dml.DoubleMLData(
        df_pd[confounders + ["log_price_change", "renewed"]],
        y_col="renewed",
        d_cols="log_price_change",
        x_cols=confounders,
    )
    plr = dml.DoubleMLPLR(data, ml_l=outcome_model, ml_m=treatment_model, n_folds=5)
    plr.fit()
    return float(plr.coef[0]), float(plr.se[0])

cb = lambda: CatBoostRegressor(iterations=500, depth=6, verbose=0)
lasso = lambda: Lasso(alpha=0.01)

configs = {
    "Lasso (Y) + Lasso (D)":       (lasso(), lasso()),
    "CatBoost (Y) + Lasso (D)":    (cb(), lasso()),
    "Lasso (Y) + CatBoost (D)":    (lasso(), cb()),
    "CatBoost (Y) + CatBoost (D)": (cb(), cb()),
}
for label, (ml_y, ml_d) in configs.items():
    coef, se = fit_dml_with_nuisance(df.to_pandas(), CONFOUNDERS, ml_y, ml_d)
    print(f"{label:42s}  coef={coef:.3f}  se={se:.3f}  bias={coef - (-0.280):.3f}")
```

Results (single run, n=100,000, truth = -0.280):

| Nuisance configuration | Estimate | SE | Bias |
|---|---|---|---|
| Lasso (Y) + Lasso (D) | -0.389 | 0.031 | -0.109 |
| CatBoost (Y) + Lasso (D) | -0.361 | 0.028 | -0.081 |
| Lasso (Y) + CatBoost (D) | -0.297 | 0.019 | -0.017 |
| CatBoost (Y) + CatBoost (D) | -0.284 | 0.017 | -0.004 |

Improving only the outcome nuisance model (row 2) reduces bias from 0.109 to 0.081: a 26% improvement. Improving only the treatment nuisance model (row 3) reduces bias from 0.109 to 0.017: an 84% improvement. The treatment model dominates by a factor of three.

The practical implication: if you are constrained by compute or interpretability to use a simpler model for one of the two nuisance steps, use CatBoost for the treatment model and accept a weaker outcome model. The default configuration in `insurance-causal` uses CatBoost for both, which is the right default for a library -- but if you are hitting performance limits, the outcome model is where you can afford to economise.

After fitting, always inspect the nuisance model diagnostics before reviewing results:

```python
from insurance_causal.diagnostics import nuisance_model_summary

summary = nuisance_model_summary(model)
print(summary)
# {
#   'outcome_r2': 0.341,
#   'treatment_r2': 0.718,
#   'treatment_residual_variance': 0.0031,
# }
```

`treatment_r2` is the most important number. At 0.718, roughly 72% of price variation is explained by risk factors -- D_tilde captures only the remaining 28% as exogenous variation. If `treatment_r2` exceeds 0.95, the library emits a warning. At that level, the estimate is identified from very little genuine exogenous variation and the SE will reflect it.

---

## Common failure modes

### Near-deterministic treatment

In UK motor renewal, the pricing model is the primary source of price variation, which means `treatment_r2` often sits between 0.70 and 0.90. This is workable. When it reaches 0.90-0.95, confidence intervals widen noticeably and the estimate becomes unstable across CV folds.

The diagnostic:

```python
stats = model.treatment_overlap_stats()
raw_sd = stats["treatment_sd"]
residual_sd = stats["residual_sd"]
exogenous_fraction = residual_sd / raw_sd
print(f"Exogenous fraction: {exogenous_fraction:.1%}")
```

If the exogenous fraction falls below 10%, the DML estimate is unreliable. The fix is not more data -- it is identifying genuinely exogenous price variation. Quarterly commercial loading reviews that apply uniformly to all policies in a rating band create within-risk-band variation in price that is approximately exogenous: policies renewing the month before a 3% commercial uplift are on the old rate, identical policies renewing after are on the new one. This timing effect is the primary source of clean identification in UK motor portfolios, and it is available in any insurer's renewals data if you join on the rate review calendar.

### Low overlap in continuous treatment

For continuous D, the analogue of propensity score overlap failure is that D_tilde has low density in parts of its range -- few observations with exogenous price variation of a given size. Extreme values of D_tilde then exert high leverage on the final OLS slope.

Practical check: after fitting, plot D_tilde against Y_tilde. The relationship should be roughly linear with observations distributed reasonably evenly across the D_tilde range. Obvious leverage points (isolated observations with |D_tilde| > 3 SD) should be inspected and excluded if they represent data errors rather than genuine price variation. `insurance-causal` clips the treatment at the 1st and 99th percentiles by default; narrow this to the 2nd-98th percentile if you see leverage issues.

### High dimensionality in confounders

Adding more confounders is appealing -- more variables, better control for confounding. Two problems: CatBoost with 50+ features can overfit the nuisance models within each cross-fitting fold, producing poorly calibrated residuals; and some additional variables may be bad controls (mediators or colliders) that block genuine causal channels.

Our practice: start with 6-10 well-understood confounders. Confirm that adding further features does not materially change the estimate. If it does, you have a specification problem to diagnose rather than a signal to average over. If the estimate shifts by more than 15% when you add a variable, understand why before using it as a confounder.

---

## When not to use DML

**When treatment variance is near-zero.** A uniform 5% renewal increase across the book leaves no price variation to identify elasticity from. D_tilde will be near-zero for everyone. Use synthetic difference-in-differences with the prior year as control if available, or accept that this data cannot estimate elasticity.

**When the question is predictive, not causal.** A model identifying which customers will lapse for retention call triage does not need DML. Prediction of lapse probability is well served by CatBoost directly. DML is for causal questions: what would happen if we changed the price? The SE from DML is wider than from a well-tuned GBM, which matters for prediction accuracy but is exactly the right property for causal inference.

**When n is below 10,000.** With 5-fold cross-fitting, each training fold has 8,000 observations. Enough for a GBM on a modest feature set, but the CI on theta will be wide. On small books, stratification with a carefully chosen stratum variable gives more useful guidance. For heterogeneous treatment effects with `insurance-causal`'s `RenewalElasticityEstimator`, the minimum is closer to 20,000-30,000 per segment -- causal forests fitted on fewer observations produce CATEs that effectively average to the portfolio ATE anyway.

---

## Practical checklist

Work through this before running DML on real portfolio data.

**Data preparation**

- Technical premium stored at quote date, not recalculated retrospectively. Recalculation corrupts the treatment variable.
- Price change computed as log(offer_price / last_year_price), not raw pounds or percentage. Log specification aligns with the semi-elasticity interpretation.
- Outcome verified as a clean binary indicator (0/1 renewal), not days-to-lapse or revenue.
- All confounders are pre-treatment variables: risk factors known before the price change was set.
- NCD handled carefully. It is a confounder (affects current price) but also a potential mediator (caused by prior claims, which also drive renewal behaviour). Use NCD band at policy start, not at renewal date.

**Treatment variance check**

- Compute `treatment_overlap_stats()` before reviewing results. Exogenous fraction above 10%. Treatment R-squared below 0.90.
- If `treatment_r2 > 0.90`: identify whether the dataset contains rate review timing effects, manual override records, or competitor price data that could serve as instruments.

**Nuisance model quality**

- `nuisance_model_summary()` after fitting. Treatment nuisance R-squared in the range 0.5-0.90.
- Compare nuisance R-squared across cross-fitting folds. A range above 0.15 suggests instability -- regularise the CatBoost model more aggressively (`depth=4`, `l2_leaf_reg=5`).
- Plot D_tilde against Y_tilde. Check the linearity assumption of the partially linear model.

**Validity checks**

- Run `confounding_bias_report()` from `insurance_causal.diagnostics` to quantify sensitivity to unobserved confounding. `sensitivity_analysis()` has been removed from the library pending a redesign; use the confounding bias report and manual robustness checks instead.
- Run `confounding_bias_report()` and document the naive vs DML comparison. Agreement within 10% means confounding is not material. Divergence above 30% needs explanation before the DML estimate enters a pricing model.
- Check for bad controls: does including NCD change the estimate materially? If it does, consider whether NCD is acting as a mediator in this analysis.

**Heterogeneous effects**

- For segment-level elasticities, use `RenewalElasticityEstimator` with `cate_model="causal_forest"` from `insurance-causal` rather than running separate `CausalPricingModel` instances per segment. The causal forest shares information across segments.
- Check that the ATE from the causal forest (`est.ate()`) is consistent with the ATE from `CausalPricingModel`. Divergence above 15% is a specification inconsistency to diagnose.
- Do not use segment-level CATE estimates for segments with fewer than 2,000 observations.

**Presentation**

- Always report the CI alongside the point estimate.
- Report the sensitivity analysis result: at what gamma does the conclusion flip? This is the single most useful number for a pricing committee that wants to know how much to trust the estimate.
- Report the bias percentage from `confounding_bias_report()`. "The naive GLM overstated elasticity by 54%" is more useful to a head of pricing than any coefficient table.

---

## The honest assessment

DML is not a general improvement on GLMs for insurance pricing. It is a tool for a specific task: recovering causal effect estimates from observational data where treatment assignment is correlated with unmodelled outcome predictors. That task matters a lot for renewal pricing and telematics factor selection; it matters much less for pure predictive work.

The benchmark above -- 6x lower RMSE than naive GLM on synthetic data with treatment R-squared of 0.72 -- is genuine and repeatable. It is also a best case: the synthetic data satisfies the conditional ignorability assumption exactly, and we knew the ground truth. On real data, unobserved confounders mean the true bias is unknown. The sensitivity analysis is how you characterise that uncertainty honestly rather than ignoring it.

The practical bar for using DML in pricing is not "does it perform better than the GLM in a simulation?" It is: do we have enough exogenous treatment variation, are our confounders well-specified, and are we prepared to run and report the sensitivity analysis? If the answer to any of those is uncertain, invest in the diagnostics first. A DML estimate with a wide CI or fragile sensitivity analysis is not better than a naive GLM -- it is just slower to produce and harder to explain.

`insurance-causal` and `insurance-causal-policy` are both on GitHub and installable via `uv add`. Start with the confounding bias report on data you already understand. If the DML estimate and the GLM coefficient agree within 10%, confounding is not material and your GLM was adequate. If they diverge materially, you have the starting point for the conversation about which number to use and why.


---

## See also

- [Double Machine Learning for Insurance Price Elasticity: Why Your Demand Model Is Confounded](/2026/03/01/your-demand-model-is-confounded/) - the motivation and theoretical setup: why OLS elasticity estimates are biased in a formula-rated book
- [Continuous Treatment Causal Inference for Insurance Pricing](/2026/03/12/insurance-autodml/) - the production library; the benchmarks here test its underlying algorithms
- [DML Works at 1,000 Policies Now. Here Is What Changed.](/2026/03/17/dml-small-samples-adaptive-regularisation/) - adaptive regularisation that extends DML to the thin-segment case, addressing the sample-size limitations noted in these benchmarks
