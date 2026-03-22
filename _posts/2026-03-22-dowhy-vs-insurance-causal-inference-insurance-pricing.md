---
layout: post
title: "DoWhy vs insurance-causal: Which Causal Inference Library Should Insurers Use?"
date: 2026-03-22
author: Burning Cost
categories: [causal-inference, libraries, comparisons, pricing]
description: "DoWhy is the most rigorous general-purpose causal inference library in Python — DAG specification, formal identification, refutation tests. It was not built for insurance pricing. Here is what that difference costs you."
canonical_url: "https://burning-cost.github.io/2026/03/22/dowhy-vs-insurance-causal-inference-insurance-pricing/"
tags: [DoWhy, insurance-causal, DML, double-machine-learning, DAG, causal-graph, refutation, price-elasticity, Poisson, exposure, renewal, causal-inference, python, uk-insurance, dowhy-insurance-pricing]
---

If you search "causal inference Python", DoWhy is near the top. It deserves to be. Originally from Microsoft Research and now stewarded by the PyWhy community, it has roughly 7,900 GitHub stars and takes the theory seriously: you must specify a causal graph, identify the estimand from that graph, estimate the effect, and then refute your assumptions. That four-step discipline separates it from libraries that hand you an estimator and hope for the best.

For general causal reasoning — economic policy analysis, clinical trial analysis, A/B test augmentation, root cause investigation — DoWhy is an excellent choice. We recommend it without reservation in those contexts.

For insurance pricing, specifically estimating causal effects in renewal portfolios with exposure-weighted claim outcomes and high-cardinality rating factors, it leaves you to do the hard parts yourself.

This post explains what those hard parts are, shows the code difference, and is honest about where DoWhy genuinely wins.

```bash
uv add insurance-causal
```

---

## What DoWhy does well

The four-step framework is genuinely good practice, and DoWhy enforces it:

**1. Explicit causal graph.** You write down your assumptions before touching data. The DAG encodes which variables are confounders, which are instruments, which are mediators. If your graph is wrong, you find out in step 2 — not after fitting a model.

**2. Formal identification.** DoWhy uses graph algorithms to determine whether the causal effect you want is identifiable from the data you have, given the graph you specified. It tells you which identification strategy applies: backdoor adjustment, front-door criterion, instrumental variables. If the effect is not identifiable, it says so.

**3. Multiple estimation methods.** Once the estimand is identified, DoWhy can estimate it via propensity weighting, regression adjustment, matching, IV, or (through EconML integration) DML and causal forests.

**4. Refutation tests.** This is DoWhy's strongest feature and the one most pricing actuaries do not use. The refutation API runs a suite of tests: random common cause (adds a spurious covariate — the estimate should not change), placebo treatment (replaces treatment with a random variable — estimate should go to zero), data subsample (check stability across random subsamples), dummy outcome (replace outcome with noise — estimate should be zero). These tests will catch specification errors that confidence intervals will not.

```python
import dowhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment="price_change_pct",
    outcome="renewed",
    common_causes=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.linear_regression",
)

# The refutation tests — this is what DoWhy does that almost nothing else does
refute_random = model.refute_estimate(
    identified_estimand, estimate, method_name="random_common_cause"
)
refute_placebo = model.refute_estimate(
    identified_estimand, estimate, method_name="placebo_treatment_refuter"
)
refute_subset = model.refute_estimate(
    identified_estimand, estimate, method_name="data_subset_refuter", subset_fraction=0.8
)
print(refute_random, refute_placebo, refute_subset)
```

This is the right way to structure a causal analysis. The graph-first discipline, the formal identification check, the refutation suite — these are things a pricing team should be doing, and DoWhy makes them easier to do properly.

---

## Where DoWhy stops and the insurance-specific problems begin

### Problem 1: no exposure handling

This is fundamental. Every insurance outcome of interest — claim frequency, claim count, loss ratio — is a rate, not a level. A policy in-force for three months contributes 0.25 years of earned exposure; an annual policy contributes 1.0. When you estimate the causal effect of a price change on claim frequency, you need to weight observations by earned exposure and use a Poisson log-likelihood, not squared error.

DoWhy has no exposure parameter. Its estimation methods — including the DML back-end via EconML integration — treat all observations as equally weighted in the standard squared-error sense. You can work around this by passing a carefully constructed custom estimator, but:

1. The documentation does not explain how to do this for Poisson outcomes
2. The cross-fitting procedure in the DML backend does not automatically preserve the exposure offset
3. Getting this wrong produces estimates that are confounded by exposure variation, silently

`insurance-causal` handles it in one argument:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    exposure_col="earned_years",          # offset applied correctly throughout
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
    cv_folds=5,
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
```

The exposure column becomes a log-offset in the CatBoost nuisance model for `E[Y|X]`. The DML score is computed in Poisson log-likelihood space. The coefficient is interpretable as a multiplicative rate effect, consistent with how a pricing GLM coefficient works. A DoWhy analysis that passes raw claim counts without handling exposure will produce a number. That number will be contaminated by exposure variation.

### Problem 2: categorical features

UK motor data contains dozens of high-cardinality categoricals: ABI group (50+ levels), occupation code (200+ levels), payment method, region, broker channel. DoWhy's estimation methods rely on scikit-learn models by default, which require these to be pre-encoded. One-hot encoding a 200-level occupation variable inflates dimensionality, degrades nuisance model accuracy, and degrades the DML causal estimate along with it — because DML's bias correction is only as good as the nuisance models.

`insurance-causal` uses CatBoost for all nuisance models. CatBoost handles high-cardinality categoricals natively via ordered target statistics. No encoding step. On UK insurance data with postcode, occupation, and ABI group, the nuisance models fit substantially better, which translates directly to a less biased causal estimate.

```python
# DoWhy with DML backend — encoding is your problem
from dowhy.causal_estimators.econml_estimator import EconMLEstimator
from econml.dml import LinearDML
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

# 200-level occupation → 200 dummies → noise in the nuisance step
model_y = Pipeline([
    ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ("reg", GradientBoostingRegressor()),
])
```

```python
# insurance-causal — categoricals handled natively
model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=PriceChangeTreatment(column="pct_price_change", scale="log"),
    confounders=feature_cols,
    categorical_cols=["abi_group", "occupation", "region", "payment_method"],
    # No encoding step. CatBoost handles all levels natively.
)
```

### Problem 3: the confounding bias report

The most commercially useful output from any causal pricing analysis is not the ATE estimate in isolation — it is the comparison between the naive GLM coefficient and the causal estimate, with a quantification of how much confounding matters. A pricing actuary using a GLM for price sensitivity estimation for five years will ask: "how wrong have we been, and by how much?"

DoWhy will give you a causal estimate. It will not tell you how it compares to the correlation you have been using, or how sensitive the conclusion is to unobserved confounding.

`insurance-causal` produces this as a first-class output:

```python
report = model.confounding_bias_report(naive_coefficient=-0.045)
print(report)
```

```
  treatment         outcome  naive_estimate  causal_estimate    bias  bias_pct
  pct_price_change  renewal         -0.0450          -0.0230  -0.022     -95.7%
```

In the synthetic benchmark from the README (seed=42, n=50,000, run on Databricks serverless 2026-03-19), the naive GLM estimate of price sensitivity is −0.045. The causal estimate is −0.023. The naive estimate is roughly double the true causal effect. The confounding mechanism is standard in renewal portfolios: high-risk customers receive larger price increases, and those customers have lower baseline renewal rates independently of price. The naive regression attributes some of that risk-driven lapse to price sensitivity.

DoWhy will compute the same DML estimate if you configure it correctly. It will not produce this report.

### Problem 4: the DAG requirement is a burden in production

This is DoWhy's theoretical strength and its practical constraint. Specifying a full causal DAG over 30 rating variables, a price treatment, a renewal outcome, and potential instrumental variables requires actuarial judgement about the causal structure of your portfolio. That judgement is valuable — it forces you to think carefully about mediators vs confounders, which matters enormously (including a mediator in the confounder set produces the "bad controls" bias).

But in production, pricing teams need to run DML analyses across multiple portfolios, lines of business, and treatment definitions. Re-specifying the full DAG for each analysis is brittle: it changes when new rating factors are introduced, when the book profile shifts, when the treatment definition changes.

`insurance-causal` takes a different design choice: you specify confounders explicitly and trust the DML partial-linear model to handle the regression adjustment. You lose the formal identifiability check that DoWhy provides. You gain a workflow that scales to regular production analysis without requiring DAG maintenance.

The trade-off is real. For a one-off strategic causal analysis ("does telematics driving score cause claims, or just proxy urban driving?"), DoWhy's structured approach is worth the overhead. For a quarterly renewal elasticity refresh across motor, home, and van lines — use `insurance-causal`.

---

## Side-by-side code comparison: the same question, both libraries

**The question:** what is the causal effect of a 10% renewal price increase on lapse probability, controlling for risk quality?

**DoWhy approach:**

```python
import dowhy
from dowhy import CausalModel

# You must specify the graph. This is the right discipline — but it is work.
causal_graph = """
digraph {
    ncb_years -> pct_price_change;
    ncb_years -> renewed;
    driver_age -> pct_price_change;
    driver_age -> renewed;
    prior_claims -> pct_price_change;
    prior_claims -> renewed;
    vehicle_age -> pct_price_change;
    vehicle_age -> renewed;
    pct_price_change -> renewed;
}
"""

model = CausalModel(
    data=df.to_pandas(),
    treatment="pct_price_change",
    outcome="renewed",
    graph=causal_graph,
)

identified_estimand = model.identify_effect()
# Uses backdoor adjustment — correct given the graph above

# DML estimate via EconML
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.econml.dml.DML",
    method_params={
        "init_params": {
            "model_y": GradientBoostingRegressor(),
            "model_t": GradientBoostingRegressor(),
        },
        "fit_params": {},
    },
)
print(f"ATE: {estimate.value:.4f}")

# Run refutation tests
refute = model.refute_estimate(
    identified_estimand, estimate, method_name="random_common_cause"
)
print(refute)
# If the estimate shifts substantially, the specification is fragile.
```

Notes on what is missing: no exposure handling for the Poisson outcome, no CatBoost nuisance models, no comparison to the naive estimate. You would need to add all of these.

**insurance-causal approach:**

```python
import polars as pl
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

# No DAG required. You nominate confounders; DML handles the rest.
model = CausalPricingModel(
    outcome="renewed",
    outcome_type="binary",
    treatment=PriceChangeTreatment(
        column="pct_price_change",
        scale="log",
    ),
    confounders=["age_band", "ncb_years", "vehicle_age", "prior_claims"],
    cv_folds=5,
)

model.fit(df)

ate = model.average_treatment_effect()
print(ate)
# Average Treatment Effect
#   Estimate:  -0.0231
#   Std Error: 0.0089
#   95% CI:    (-0.0406, -0.0057)
#   p-value:   0.0092

# The confounding report — how much did the GLM overstate sensitivity?
report = model.confounding_bias_report(naive_coefficient=-0.045)
print(report)

# Note: sensitivity_analysis() in insurance_causal.diagnostics is being
# redesigned (the previous Rosenbaum formula had no statistical basis).
# For now, use confounding_bias_report() to assess confounding magnitude,
# or SelectionCorrectedElasticity.sensitivity_bounds() for IPW-based bounds.
```

---

## Comparison table

| | DoWhy | insurance-causal |
|---|---|---|
| Primary use case | General causal reasoning, any domain | Insurance pricing causal questions |
| GitHub stars | ~7,900 (March 2026) | Small, focused |
| DAG specification | Required — forces discipline | Not required — you nominate confounders |
| Formal identification check | Yes — a genuine strength | No — assumes backdoor adjustment |
| Refutation tests | Yes — placebo, random cause, subsample | `confounding_bias_report()` for qualitative assessment; formal sensitivity analysis planned |
| Estimation methods | Regression adjustment, IPW, IV, DML (via EconML), matching | DML with CatBoost nuisance; causal forests |
| Exposure/offset handling | No | Yes — Poisson/Gamma throughout |
| Categorical features | Manual encoding required | CatBoost native |
| Confounding bias report | No | Yes — naive vs causal, sensitivity bounds |
| GATES / CLAN / RATE inference | No | Yes — `HeterogeneousInference` |
| Renewal elasticity module | No | Yes — `RenewalElasticityEstimator` with ENBP constraint |
| Continuous treatment (Riesz) | No | Yes — `autodml.PremiumElasticity` |
| Dual selection bias | No | Yes — `DualSelectionDML` |
| Production scaling | Requires DAG maintenance | Designed for regular refresh |
| Documentation | Excellent — tutorials, papers, API reference | Focused on UK pricing use cases |

---

## When to use DoWhy

Use DoWhy when the causal structure of your problem is complex and worth specifying explicitly, and when you want the formal refutation apparatus to test your assumptions. Specific cases:

- **Root cause analysis**: "Why did motor loss ratio increase in Q3?" — the DAG framework is right for multi-variable causal attribution across marketing, claims, and pricing variables.
- **Research**: one-off strategic analyses where you want to document your causal assumptions formally and run the full refutation suite.
- **Novel treatment types**: DoWhy supports instrumental variables, front-door identification, and non-standard estimands that `insurance-causal` does not.
- **Non-insurance domains**: if your team is doing causal inference across healthcare, operations, and fraud as well as pricing, DoWhy is the unifying framework.
- **Academic credibility**: if the analysis will be reviewed externally or published, the DoWhy four-step framework documents assumptions in a way that stands up to scrutiny.

The refutation tests specifically are something we think pricing teams should run more often. DoWhy makes them easy. We have seen analyses where the random common cause test revealed an unstable estimate that confidence intervals had not flagged — the kind of specification fragility that would otherwise surface as an embarrassing rate movement post-model.

## When to use insurance-causal

Use `insurance-causal` when you need production-grade causal estimation in a UK insurance pricing context:

- **Renewal elasticity**: quarterly estimation of price sensitivity across motor or home book segments, corrected for risk-driven selection. The `RenewalElasticityEstimator` produces an elasticity surface by ABI group and NCD band, ready for a pricing optimiser, in under 10 minutes on a standard Databricks cluster.
- **Telematics causal attribution**: is the harsh braking score causing lower claims, or is it proxying urban driving? `CausalPricingModel` with a `ContinuousTreatment` handles the continuous treatment with CatBoost confounders. DoWhy can do this too, but you will spend an afternoon wiring up the exposure and categorical handling.
- **Heterogeneous elasticity**: which segments have meaningfully different price sensitivities? `HeterogeneousElasticityEstimator` with the BLP/GATES/RATE inference suite gives you segment-level effects with formal tests for whether the heterogeneity is real. This directly answers the FCA PS21/11 question: are you applying the same pricing logic to materially different customer groups?
- **Confounding bias audit**: you want to quantify how much your existing GLM elasticity estimates are biased by risk selection. The confounding bias report is a one-call summary that a pricing committee can read without a causal inference primer.
- **Regular production refresh**: no DAG to maintain, no graph to update when a new rating factor joins the model. Nominate confounders; fit; report.

---

## Our view

DoWhy is the more theoretically rigorous library. The four-step framework — model, identify, estimate, refute — is the right discipline for causal analysis, and the refutation suite catches problems that confidence intervals miss. If you are doing causal work outside insurance pricing, DoWhy is where we would start.

The gap for insurance pricing is the same gap we see with EconML: neither library was built with exposure-weighted rate outcomes, high-cardinality insurance categoricals, or the specific dual-selection structure of renewal portfolios in mind. Those omissions are not failures of the libraries — they are the result of being built for general use. The work of adapting them to insurance is significant and error-prone, and the errors (particularly on exposure handling) are silent.

`insurance-causal` is not more theoretically sophisticated than DoWhy. It is more specifically fitted to the problem a UK pricing actuary actually has. The two are complementary: use DoWhy when you need formal causal reasoning with explicit graph assumptions and a thorough refutation audit; use `insurance-causal` when you need to run DML-based elasticity estimation in production with the right Poisson likelihood, native CatBoost categoricals, and an output that makes sense to a pricing committee.

---

`insurance-causal` is at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal). Python 3.10+. Polars-native.

---

**Related posts:**
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) — the same analysis for EconML: stronger on heterogeneous effects, same gap on insurance-specific structure
- [DML for Insurance: Practical Benchmarks and Pitfalls](/2026/03/09/dml-insurance-benchmarks/) — foundational post on why DML beats naive GLM for causal price elasticity
- [Your demand model is confounded](/2026/03/01/your-demand-model-is-confounded/) — the problem this library solves, explained without library comparisons

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/22/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) — proxy discrimination auditing
- [EquiPy vs insurance-fairness](/2026/03/22/equipy-vs-insurance-fairness/) — optimal transport fairness
- [MAPIE vs insurance-conformal](/2026/03/22/mapie-vs-insurance-conformal-prediction-intervals/) — conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/22/econml-vs-insurance-causal-inference-pricing/) — causal inference for pricing
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) — model monitoring
- [NannyML vs insurance-monitoring](/2026/03/22/nannyml-vs-insurance-monitoring-drift-detection-insurance/) — drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/22/alibi-detect-vs-insurance-monitoring-drift-detection/) — statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) — distributional regression
