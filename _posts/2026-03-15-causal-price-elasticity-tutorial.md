---
layout: post
title: "Your Elasticity Estimate Is Biased and You Already Know Why"
date: 2026-03-15
categories: [techniques]
tags: [causal-inference, price-elasticity, dml, causal-forest, catboost, renewal-pricing, enbp, icobs-6b2, fca, gipp, heterogeneous-treatment-effects, dr-learner, insurance-elasticity, uk-motor, polars, python, tutorial]
description: "OLS elasticity in formula-rated books is contaminated by your own risk model. insurance-elasticity fixes this with CausalForestDML and CatBoost nuisance."
---
> **Update (March 2026):** `insurance-elasticity` has been merged into [`insurance-causal`](https://github.com/burning-cost/insurance-causal). Install `insurance-causal` and use `insurance_causal.elasticity` for the same functionality.

Every pricing team we have talked to has an elasticity model. Most of them are logistic regressions with price change as a regressor, confounders included, holdout Gini looking acceptable. And nearly all of them are wrong in the same direction: they overstate price sensitivity, which means the optimiser gives back too much discount to customers who would have renewed anyway.

The cause is not a coding error. It is a structural feature of how insurance pricing works, and it cannot be fixed by adding more covariates to the regression.

`insurance-elasticity` wraps CausalForestDML from EconML with insurance-specific handling for binary outcomes, GIPP structural constraints, and the near-deterministic treatment problem that plagues formula-rated books. This tutorial covers the full workflow in seven steps.

```bash
uv add insurance-elasticity
# or
uv add insurance-elasticity
```

---

## Step 1: Why OLS elasticity is wrong in a formula-rated book

The confounding structure is precise. Your risk model produces a technical premium `c_i` for each customer. At renewal, pricing applies loadings and commercial adjustments to produce an offer price `p_i`. A customer with a worsening risk profile — an at-fault claim, a birthday moving them into a higher age band — receives a larger premium increase. That same customer is also more likely to lapse, independently of price, because their circumstances have changed.

Formally: `price_change_i = f(risk_factors_i) + noise`. The risk factors are confounders — they affect both the treatment (price change) and the outcome (renewal). OLS on `P(renew) ~ price_change + confounders` does not solve this, even with a rich confounder set. The issue is not omitted variables in the regression; it is that the treatment itself is a function of the confounders in the training data. When you include both, the regression cannot separate the causal effect of price from the risk-driven component of the price change.

The standard illustration. Suppose high-risk customers (NCB 0, recent claim) receive on average +15% at renewal and have a 65% baseline renewal rate. Low-risk customers (NCB 5+, no claims) receive on average +4% and have an 88% baseline renewal rate. OLS sees: bigger price increases correlate with lower renewal rates. That is true. But it is mostly the risk-driven correlation, not the price causal effect. The OLS slope on price change is picking up `E[risk | price_change]`, not `causal_effect(price_change)`.

The fix: partial out the risk model from both the renewal outcome and the price change, then regress the residuals. This is the partially linear regression model (Robinson 1988, Econometrica 56(4): 931-954), estimated via Double Machine Learning (Chernozhukov et al. 2018, Econometrics Journal 21(1): C1-C68). The residualised price change `D_tilde = price_change - E[price_change | X]` is the component of your pricing decisions that was not determined by the risk model — commercial rate adjustments, timing effects, manual overrides, competitor rate movements that affected all customers equally. That is the variation DML uses to identify the causal effect.

---

## Step 2: Data and CausalForestDML setup

The library ships `make_renewal_data()` for reproducible examples. It generates synthetic UK motor renewal portfolios with a known data-generating process, including heterogeneous true elasticity by NCD band.

```python
import polars as pl
from insurance_elasticity.data import make_renewal_data
from insurance_elasticity import RenewalElasticityEstimator

df = make_renewal_data(n=50_000, random_state=42)
# df has columns: driver_age, ncd_years, vehicle_group, region,
# log_price_change, renewed, true_elasticity (DGP ground truth)

print(df.shape)
# (50000, 12)

print(df["log_price_change"].describe())
# mean: 0.072 (7.2% average increase)
# std:  0.089
```

Define your confounders — everything that entered the risk model and the commercial pricing decision. For a UK motor portfolio this is a minimum set; real implementations add more:

```python
confounders = [
    "driver_age",
    "ncd_years",
    "vehicle_group",
    "region",
    "vehicle_age",
    "occupation_band",
    "channel",          # PCW vs direct — must be here, not optional
    "months_since_claim",
]
```

Now fit the estimator. `cate_model="causal_forest"` uses CausalForestDML with honest estimation (separate subsamples for splitting and leaf estimation). The `binary_outcome=True` flag switches the outcome nuisance model to a calibrated CatBoost classifier, which gives better-calibrated residuals than regression on a 0/1 outcome.

```python
est = RenewalElasticityEstimator(
    cate_model="causal_forest",
    n_estimators=500,
    n_folds=5,
    catboost_iterations=800,
    binary_outcome=True,
    log_price=True,      # treatment = log(offer/last_year), theta = elasticity
    random_state=42,
)

est.fit(
    df,
    outcome="renewed",
    treatment="log_price_change",
    confounders=confounders,
)
```

The `n_folds=5` cross-fitting is not optional. Without it, the nuisance models overfit to their own training data, introducing a regularisation bias into the final `theta` estimate that does not vanish with sample size. Five folds is the standard; three is acceptable when data is scarce.

The fitting time on 50,000 rows with 800-iteration CatBoost nuisance models is roughly 4-6 minutes on a single core. The bottleneck is the treatment nuisance model (predicting price change from risk factors), not the forest.

---

## Step 3: Heterogeneous treatment effects by NCD band and vehicle group

The average treatment effect is the number most people ask for first. It is useful for sanity-checking the model and for regulatory reporting. It is the wrong number for pricing decisions.

```python
ate_point, ate_lb, ate_ub = est.ate()
print(f"ATE: {ate_point:.3f} ({ate_lb:.3f}, {ate_ub:.3f})")
# ATE: -2.14 (-2.31, -1.97)
# A 10% price increase reduces renewal probability by ~21.4 percentage points
# on the linear probability model scale
```

Cross-check against the DGP ground truth in the synthetic data:

```python
print(f"True ATE: {df['true_elasticity'].mean():.3f}")
# True ATE: -2.09
```

Group average treatment effects (GATEs) are where it gets useful. The library's `.gate()` returns a Polars DataFrame with point estimates, confidence intervals, and sample counts per group:

```python
gate_ncd = est.gate(df, by="ncd_years")
print(gate_ncd)
# ┌───────────┬───────────┬───────────┬───────────┬───────┐
# │ ncd_years │ gate      │ gate_lb   │ gate_ub   │ n     │
# ╞═══════════╪═══════════╪═══════════╪═══════════╪═══════╡
# │ 0         │ -1.41     │ -1.68     │ -1.14     │ 4821  │
# │ 1         │ -1.83     │ -2.05     │ -1.61     │ 6203  │
# │ 2         │ -2.09     │ -2.28     │ -1.90     │ 6891  │
# │ 3         │ -2.31     │ -2.49     │ -2.13     │ 7104  │
# │ 4         │ -2.58     │ -2.79     │ -2.37     │ 8230  │
# │ 5+        │ -2.88     │ -3.04     │ -2.72     │ 16751 │
# └───────────┴───────────┴───────────┴───────────┴───────┘
```

This is the pattern you should expect: long-tenure customers with full NCB are more price-sensitive. They have been shopping the market for years, know what comparable cover costs, and their loyalty is not infinite. Customers at NCB 0 — often post-claim — are less price-sensitive not because they are happy to pay more, but because their alternatives are expensive and they have limited shopping power.

Do the same for vehicle group:

```python
gate_veh = est.gate(df, by="vehicle_group")
```

The output here tends to show performance vehicles (group 40+) as less elastic than small city cars (group 1-10). The mechanism: high-group vehicles are harder to insure competitively, fewer alternative quotes, and the customer has less market power. This is actuarially intuitive and the DML estimate is capturing it correctly — not because we told it to, but because the heterogeneity is in the data.

Per-customer CATEs from the forest:

```python
cate_values = est.cate(df)           # numpy array, shape (50000,)
df = df.with_columns(
    pl.Series("cate", cate_values)
)
```

The CATE distribution matters for the optimiser in step 5. If the distribution is narrow, a single ATE is adequate for segmented pricing. If it is wide — say, a 10th-to-90th percentile range of more than 1.5 — you have material heterogeneity and should use the per-customer estimates.

---

## Step 4: DR-Learner as a robustness check

CausalForestDML is our primary estimator. It is not infallible: it requires the overlap assumption (discussed in step 6), it uses a specific forest structure, and the honest estimation approach has finite-sample properties that depend on tree depth. Before trusting the GATEs, run the DR-Learner as an independent check.

The DR-Learner (Kennedy 2023, Annals of Statistics 51(2): 958-981) constructs pseudo-outcomes via double-robustness and then regresses them on covariates. It is consistent if either the outcome model or the treatment model is correctly specified — not both. The insurance-elasticity library wraps it with the same interface:

```python
est_dr = RenewalElasticityEstimator(
    cate_model="dr_learner",
    n_folds=5,
    catboost_iterations=800,
    binary_outcome=True,
    log_price=True,
    random_state=42,
)
est_dr.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)

ate_dr, lb_dr, ub_dr = est_dr.ate()
print(f"DR-Learner ATE: {ate_dr:.3f} ({lb_dr:.3f}, {ub_dr:.3f})")
# DR-Learner ATE: -2.08 (-2.27, -1.89)
```

Agreement between CausalForestDML (`-2.14`) and DR-Learner (`-2.08`) within overlapping confidence intervals is the robustness check passing. If they disagree by more than 0.3 on an elasticity centred at -2, investigate: the usual culprit is poor overlap in a segment, which you will see in the diagnostics.

The DR-Learner GATEs:

```python
gate_ncd_dr = est_dr.gate(df, by="ncd_years")
# Compare against gate_ncd from the forest — confirm direction and approximate magnitudes match
```

We do not recommend choosing between the two estimators based on which gives a better-looking answer. Use the forest as primary; use the DR-Learner to confirm. If they disagree, treat both with caution until you have understood why.

---

## Step 5: ENBP-constrained renewal pricing optimisation

The ICOBS 6B.2 constraint (FCA PS21/5, effective January 2022) is a hard ceiling: the renewal offer cannot exceed the Equivalent New Business Price through the same channel. The library's optimiser takes ENBP as a pre-computed column — it is not in the business of running your new-business model. You provide it.

```python
from insurance_elasticity.optimise import RenewalPricingOptimiser

# Assume df has 'tech_prem' (technical premium), 'enbp' (pre-computed),
# 'last_year_price' columns
optimiser = RenewalPricingOptimiser(
    elasticity_model=est,
    technical_premium_col="tech_prem",
    enbp_col="enbp",
    floor_loading=1.0,       # do not offer below technical premium
)
```

For a profit-maximising objective subject to the ENBP constraint:

```python
result = optimiser.optimise(df, objective="profit")

print(result.select([
    "policy_id",
    "tech_prem",
    "enbp",
    "last_year_price",
    "optimal_price",
    "cate",
    "enbp_binding",          # True if the optimal price was capped at ENBP
]).head(10))
```

For a retention-targeted run (85% target renewal rate across the book):

```python
result_ret = optimiser.optimise(
    df,
    objective="retention",
    target_retention=0.85,
)
print(f"Estimated renewal rate: {result_ret['predicted_renewal_rate'].mean():.3f}")
print(f"Average optimal price / tech prem: {(result_ret['optimal_price'] / result_ret['tech_prem']).mean():.3f}")
```

The optimiser solves policy-by-policy using the per-customer CATE from the forest. The contribution-maximising price for customer `i` is:

```
p*_i = c_i - 1 / (cate_i * (1/p_bar_i))
```

where `p_bar_i` is the baseline renewal probability at last year's price. The ENBP cap is enforced after solving — if `p*_i > ENBP_i`, the offer is set to `ENBP_i`. The `enbp_binding` flag records which customers hit the ceiling.

A practical note on the kink. When ENBP is binding for a large fraction of the book — typically for customers previously priced above ENBP who were repriced down to ENBP at GIPP implementation — the demand curve has a structural break at ENBP. The DML estimate of elasticity near the ENBP level is less reliable because the price variation around that point is constrained. We flag this in the diagnostics.

---

## Step 6: Diagnostics — overlap, balance, variation_fraction

Three diagnostics before you trust any of the above.

**Treatment variation.** DML identification requires variation in `D_tilde = price_change - E[price_change | X]`. In a formula-rated book, the price change is often nearly deterministic: the R² of a good risk model on price change can be 0.85 or higher, leaving very little residual variation to identify the causal effect. The `variation_fraction` is `Var(D_tilde) / Var(D)`.

```python
diag = est.diagnostics()

print(f"Treatment variation fraction: {diag.variation_fraction:.3f}")
# Treatment variation fraction: 0.18
# 18% of price variation is unexplained by risk factors — adequate but not generous

if diag.variation_fraction < 0.10:
    print("WARNING: weak treatment — DML estimates will be noisy")
    print("Consider restricting to periods with exogenous rate movements")
```

The 10% threshold is not a formal test statistic — it is a rule of thumb. Below 10%, the standard errors on ATE will be large and GATEs will be unreliable. The correct response is to find more exogenous variation, not to proceed regardless.

**Overlap.** Every risk profile in the data must have had some realistic chance of receiving a range of different prices (the positivity assumption). CausalForest's `min_samples_leaf` implicitly handles some of this, but you should check directly:

```python
print(diag.overlap_summary())
# Propensity score summary for treatment overlap:
#   10th pct: 0.23 (OK)
#   90th pct: 0.81 (OK)
#   Fraction with propensity < 0.05: 0.031
#   Fraction with propensity > 0.95: 0.018
#
# VERDICT: Adequate overlap. 4.9% of policies are near-deterministic treatment
#          assignments. CATEs for these policies should be interpreted cautiously.
```

Policies with near-deterministic price assignments are typically those subject to underwriting rules — high-risk postcodes, young drivers in high-group vehicles where pricing discretion is limited. The CATEs for these policies are unreliable. Flag them; do not use them as input to the optimiser.

**Balance.** After cross-fitting, the correlation between the residualised treatment `D_tilde` and the confounders should be near zero. If it is not, either the treatment model is misspecified or CatBoost is not capturing the true treatment assignment mechanism.

```python
balance = diag.balance_table()
print(balance.filter(pl.col("abs_correlation") > 0.05))
# If this returns rows, investigate those confounders
```

The full diagnostic plot:

```python
diag.plot(
    save_path="./elasticity_diagnostics.png",
    include=["treatment_histogram", "overlap", "balance", "cate_distribution"],
)
```

---

## Step 7: ICOBS 6B.2 evidence trail for renewal pricing

The FCA's ICOBS 6B.2 requires that firms can demonstrate, for any renewal offer, that the price did not exceed the ENBP. It does not require you to document your demand model methodology in the same level of detail — but Consumer Duty (PS22/9) and SS1/23 together create a strong expectation that pricing models, including elasticity models used to set discounts, are documented and their limitations understood.

The library generates a compliance-facing summary. This is a factual record, not a legal opinion. It goes in the pricing governance pack.

```python
audit = optimiser.enbp_audit(result)
print(audit.summary())
```

```
ENBP COMPLIANCE AUDIT
Run date: 2028-03-15
Policies processed: 50,000

Policies where optimal_price > enbp: 0 (0.0%) — COMPLIANT
Policies where optimal_price = enbp (ceiling binding): 12,847 (25.7%)
Policies where optimal_price < enbp (discount applied): 37,153 (74.3%)

Average discount from ENBP: 4.1%
Average discount from last_year_price: 2.8%

Elasticity model: RenewalElasticityEstimator(cate_model=causal_forest,
                  n_folds=5, catboost_iterations=800, binary_outcome=True)
ATE: -2.14 (95% CI: -2.31, -1.97)
Treatment variation fraction: 0.18
Overlap verdict: ADEQUATE (4.9% near-deterministic)
```

What to add in the governance narrative — the text the audit report does not write for you:

- What was the source of the exogenous price variation used to identify the causal effect? Rate review periods, manual override campaigns, PCW repricing events that affected all customers in a segment identically.
- What is the estimation period, and does it span the GIPP implementation date? Post-January 2022 data only; the demand structure before and after the dual pricing ban is different.
- What are the known limitations? Near-deterministic treatment in underwriting-constrained segments. CATEs for NCB 0 customers with recent claims are wider confidence intervals than for the NCB 5+ majority.
- How will the model be monitored? Recommended: quarterly recalibration of ATE, comparison of predicted vs actual renewal rates by NCD band, and retesting overlap diagnostics when book mix shifts.

The FCA has not, as of March 2028, issued specific guidance on DML-based demand models. FCA EP25/2 (2025) reviewed GIPP compliance broadly and found firms largely compliant with the ENBP cap; the next wave of scrutiny will be on the pricing logic used to set discounts, not just the ENBP ceiling itself. Having a documented causal model, with known limitations and a monitoring cadence, is a better position than an OLS model with no documented identification strategy.

---

`insurance-elasticity` is open source under MIT at [github.com/burning-cost/insurance-elasticity](https://github.com/burning-cost/insurance-elasticity). Requires Python 3.10+, CatBoost 1.2+, EconML 0.16+, and Polars 0.20+.

- [Your Model Was Trained on Last Year's Book](/2026/03/15/covariate-shift-detection-book-mix-changes/) — when book mix shifts, your elasticity data shifts with it; density ratio correction before refitting
- [Model Validation Is a Checklist, Not a Test](/2027/09/15/model-validation-pra-ss123/) — what the PRA's SS1/23 expects from model documentation and the actuarial sign-off process
- [Channel Mix Drift Your Model Didn't Notice](/2027/02/15/channel-mix-drift-your-model-didnt-notice/) — PCW customers are more elastic; a model trained on a channel mix you no longer have will overstate overall elasticity

---

## See also

- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) — the theoretical foundations this tutorial builds on
- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) — the structural diagnosis: why OLS is confounded in a formula-rated book, before the DML fix
- [Continuous Treatment Causal Inference for Insurance Pricing](/2026/03/12/insurance-autodml/) — the `insurance-causal` library used throughout this tutorial
- [DML Works at 1,000 Policies Now. Here Is What Changed.](/2026/03/17/dml-small-samples-adaptive-regularisation/) — thin segment extensions; applies the approach here to books where individual segments have fewer than 5,000 renewals
