---
layout: post
title: "AutoML for Insurance Pricing — Does It Actually Work?"
date: 2026-03-28
categories: [tutorials, pricing]
tags: [automl, h2o, flaml, autogluon, insurance-pricing, poisson, tweedie, exposure, freMTPL2, frequency-modelling, python, tutorial]
description: "H2O, FLAML, and AutoGluon are genuinely useful tools. None of them handle the log(exposure) offset that makes insurance frequency modelling work. Here is an honest account of where they are, what they can do today, and where not to trust them."
seo_title: "AutoML for Insurance Pricing: H2O, FLAML, AutoGluon vs GLM/GBM on freMTPL2"
---

The pitch for AutoML in insurance pricing is appealing: less time tuning hyperparameters, faster iteration across model types, reproducible pipelines, and maybe the odd algorithmic surprise that a team of actuaries would not have found manually. All of that is real. But if you take H2O AutoML, FLAML, or AutoGluon and point them at a motor frequency dataset, you will get output that looks plausible and is statistically wrong in a way that matters for pricing.

The reason is a single missing parameter: `offset_column`. None of the three expose it at the AutoML interface level. For insurance frequency modelling, this is not a minor inconvenience — it breaks the statistical foundation of the model.

This post is about what AutoML frameworks actually do, where they genuinely help, and where they currently fail. The honest answer to the headline question is: not yet for frequency models, yes for several other use cases, and the literature is thinner than you might expect.

---

## What pricing teams actually want from AutoML

Before getting to the problems, it is worth being precise about what practitioners are asking for. The three things we hear most often:

**Less tuning, more iteration.** A typical GBM hyperparameter search on freMTPL2 (677,991 French motor policies) with Optuna runs 200 trials over several hours. AutoML frameworks wrap that loop. The question is not whether they tune better than a careful actuary with Optuna — they probably do not — but whether they produce good-enough results in a fraction of the time, so that the team can test more modelling ideas per sprint.

**Reproducibility.** Bespoke HPO scripts accumulate technical debt. AutoML frameworks give you a logged, versioned run with a consistent interface. This matters for model governance and FCA audit trails.

**Algorithm breadth.** An actuary building a frequency model will reach for LightGBM with a Poisson objective and reasonable defaults. They are unlikely to manually test CatBoost, XGBoost, a neural network, and three ensemble variants in the same experiment. AutoML does.

None of these wants are unreasonable. The problem is that the frameworks were designed for general tabular regression and classification tasks, not for the specific statistical requirements of actuarial frequency models.

---

## The three contenders

### H2O AutoML (v3.46)

H2O is the most mature of the three and, for insurance purposes, the most capable — though that is not saying a great deal. The AutoML run trains a fixed GLM, three pre-specified XGBoost variants, five GBMs, a deep neural net, an Extremely Randomised Forest, and random grids of GBM/XGBoost/DNN, finishing with stacked ensembles.

What H2O does well: it exposes `monotone_constraints` at the AutoML level, so you can enforce business rules like "higher BonusMalus implies higher predicted frequency." It has built-in target encoding for high-cardinality categoricals. The JVM backend handles large datasets (677k rows is not a problem). Individual H2O algorithms — `H2OGBMEstimator`, `H2OXGBoostEstimator`, `H2OGeneralisedLinearEstimator` — do support Poisson distribution and `offset_column`.

The problem: `H2OAutoML.train()` accepts only `x`, `y`, `training_frame`, `validation_frame`, `leaderboard_frame`, `blending_frame`, `fold_column`, and `weights_column`. There is no `distribution` parameter. There is no `offset_column`. The moment you want to use the AutoML wrapper, you lose exposure offset support entirely.

### FLAML (v2.5, Microsoft)

FLAML is designed for low-resource environments and runs fast. Its estimator list — LightGBM, XGBoost, Random Forest, Extra Trees, HistGBM, CatBoost, KNN — is solid. The framework is highly extensible via custom estimator subclasses, and it documents the pattern for plugging in custom objectives explicitly.

What FLAML does well: it is lightweight, it handles tabular data efficiently, and LightGBM's native categorical handling is genuinely good. If you are willing to write a custom `LGBMEstimator` subclass that passes `objective='poisson'` and `init_score=log(exposure)` to the underlying training call, you can approximate what you need.

What you cannot do without that subclass: use Poisson deviance as the optimisation metric, pass an exposure offset, or apply monotonicity constraints. Out of the box, FLAML minimises RMSE. For an insurance frequency model, that is the wrong loss function.

### AutoGluon-Tabular (v1.5, AWS)

AutoGluon dominates the AutoML Benchmark (AMLB) 2024-2025 results, with a statistically significant advantage over all other AutoML systems in Nemenyi post-hoc tests across all time constraints. The model zoo in v1.5 is impressive: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, TabNet, FastAI, EBM, TabPFN-2.5, TabDPT, RealTabPFN-2, TabM, RealMLP, Mitra, and stacked ensembles.

The addition of EBM (Explainable Boosting Machine) in v1.5 is worth noting — EBMs are inherently interpretable and support monotonicity constraints, which matters for UK regulatory submissions.

What AutoGluon does not have: Poisson or Tweedie problem types, Poisson deviance as an evaluation metric, or `offset_column` support at any level of the API. The problem types are binary, multiclass, regression, and quantile. There is no documented workaround for exposure offsets — not even a bad one. AutoGluon will produce regression output using RMSE, and it will look like it is working.

---

## The four hard problems that make insurance different

### 1. Exposure offset — the fundamental blocker

A motor frequency model predicts claims per unit of exposure (vehicle-years). A policy written for six months contributes 0.5 exposure-years. A policy written for a full year contributes 1.0. The standard GLM formulation is:

```
log(E[ClaimNb]) = log(Exposure) + β₀ + β₁x₁ + ... + βₚxₚ
```

The `log(Exposure)` term is an offset — it forces the coefficient on exposure to exactly 1.0. This is not a modelling preference; it is a statistical requirement for the model to produce claim frequency predictions that are comparable across policies with different durations.

If you include exposure as a raw feature, the model is free to learn any relationship between exposure and claim count. It might learn something close to the correct one, or it might not. More importantly, it will absorb part of the expected signal into the exposure coefficient and leave less signal for the rating factors you actually care about. The resulting relativities will be subtly wrong in ways that are difficult to detect without an explicit A/E (actual-to-expected) ratio check.

None of H2O AutoML, FLAML, or AutoGluon expose `offset_column` at the AutoML interface level. This is not a bug — it reflects the fact that offsets are an actuarial concept, not a standard ML concept. But it makes all three frameworks unsuitable for frequency modelling without meaningful custom engineering.

### 2. Poisson/Tweedie loss functions — partial support only

The correct loss function for claim count data is Poisson deviance (or, for combined frequency-severity, Tweedie). Minimising RMSE on count data over-penalises large residuals and will systematically produce models that are well-calibrated for average policies and poorly calibrated for high-risk segments — the exact opposite of what pricing wants.

H2O supports Poisson and Tweedie at the individual algorithm level but not through the AutoML wrapper. FLAML requires a custom subclass. AutoGluon has no mechanism.

### 3. High-cardinality categoricals — manageable

freMTPL2 has Region (22 categories) and VehBrand (~11 categories). These are not difficult. Real UK motor data has postcode district (2,700+ categories) and ABI vehicle group (1,200+ categories). H2O has built-in target encoding in the AutoML preprocessing pipeline. AutoGluon v1.5 added TabPrep-LightGBM with target mean encoding and feature crossing. FLAML delegates to LightGBM's native categorical handling, which is adequate. PyCaret one-hot encodes by default — do not use PyCaret on real UK pricing data.

### 4. Regulatory interpretability

FCA Consumer Duty and ABI principles require that rating factors can be explained. SHAP values are accepted practice. All three frameworks produce SHAP values from their GBM/XGBoost models, though in FLAML they are not surfaced automatically. AutoGluon's EBM models are natively interpretable without post-hoc explanation.

What none of them do automatically: produce a relativity table — the multiplicative factors by rating band that actuaries and regulators actually work with. That post-processing step is bespoke in every implementation we have seen.

---

## The exposure offset problem in detail

Here is what happens in practice when you feed freMTPL2 to H2O AutoML without handling exposure correctly.

The dataset has 677,991 policies with Exposure values ranging from 0.003 to 1.0 (fractional years). ClaimNb is 0 for the vast majority of policies. A model trained to minimise RMSE on ClaimNb will learn that:

- Short-exposure policies almost always have ClaimNb = 0
- Long-exposure policies sometimes have ClaimNb > 0
- Exposure is therefore a strong predictor of ClaimNb

This is correct in the wrong way. The model learns that exposure predicts claim count, which it does — but the relationship it learns is exposure-dependent rather than frequency-based. When you score this model on new policies, predictions are unstable for policies with unusual durations. A three-month new-business quote will have a different predicted frequency than the same risk quoted for 12 months, even after dividing by exposure to convert back. The A/E ratio will drift.

A proper Poisson model with log(exposure) offset does not learn a relationship between exposure and claim count — it constrains that relationship to be exactly 1.0 and learns the residual signal from the rating factors. The predictions are claim frequencies (claims per vehicle-year), portable across any exposure duration.

The workaround — modelling claim frequency rate (ClaimNb/Exposure) as a regression target — is imperfect. It abandons Poisson distributional assumptions, over-weights short-exposure policies in RMSE minimisation (a short-exposure policy with 0 claims has a small residual; a long-exposure policy with 0 claims has a larger residual, even though both contain the same information), and produces different calibration behaviour. It is better than nothing, but it is not the same as a proper offset.

---

## What Dong & Quan (2024) actually found

The only published paper benchmarking AutoML on freMTPL2 with Poisson deviance is Dong & Quan (arXiv:2408.14331, August 2024). The results are useful but the methodology has a critical flaw.

The paper builds a custom insurance-specific AutoML pipeline — not H2O, FLAML, or AutoGluon — and evaluates it on freMTPL2freq, Wisconsin LGPIF, and Australian auto insurance data. Using Poisson deviance as the metric, they report:

| Evaluation budget (G) | Test deviance |
|---|---|
| 8 | 0.3689 |
| 32 | 0.3250 |
| 64 | 0.3122 |
| 128 | 0.3034 |
| 256 | 0.3009 |
| 512 | 0.3020 |
| 1024 | 0.3114 |

The Wüthrich (2019) GLM baseline on freMTPL2 is 0.3149. So the claim is that 64+ pipeline evaluations beats the benchmark GLM, and 256 evaluations achieves near-optimal performance.

The problem: Table 1 of the paper lists Exposure as a numerical feature. It is not treated as a log-offset — it is passed into the pipeline as a predictor on equal footing with VehAge and BonusMalus. This is methodologically wrong for frequency modelling. The model cannot distinguish between "this policy has high exposure because it was written for a full year" and "this policy has high exposure because it is a high-risk vehicle." The Gini coefficient on the test set is therefore measuring something different from what a properly-specified frequency model would measure.

We do not know how much this affects the reported deviance numbers. It may be that the AutoML pipeline partially learns the correct offset relationship from the data. But the comparison against the Wüthrich GLM baseline (which does use a proper offset) is not on equal terms.

There is also no comparison against a tuned GBM. A LightGBM with Poisson objective, proper exposure offset via init\_score, and 200 Optuna trials typically achieves 0.295-0.305 Poisson deviance on freMTPL2. The Dong & Quan result at G=256 (0.3009) would not beat that. The paper does not include this comparison, which makes the headline claim that AutoML "beats the GLM" less meaningful than it appears — GLMs with splines and properly handled exposure already outperform the Wüthrich baseline, and GBMs outperform those.

---

## When AutoML does work for pricing

We want to be clear that the above critique is specific to frequency models. There are legitimate use cases where AutoML adds real value:

**Severity models.** Claim severity is a continuous regression target without an exposure offset requirement. Tweedie or gamma distributions are standard, but a well-calibrated RMSE regression is often competitive. AutoML frameworks are considerably more useful here — point them at severity data and they will produce competitive models with minimal engineering. AutoGluon in particular is strong for this task given its AMLB benchmark standing.

**Non-frequency targets.** Lapse propensity, renewal pricing, MTA uplift, fraud score — these are classification or regression tasks where Poisson/exposure concerns do not apply. All three frameworks are appropriate out of the box.

**Feature selection.** AutoML leaderboards tell you which algorithms and feature combinations perform well. Even if you do not use the AutoML output directly in production, running H2O AutoML or AutoGluon as a feature importance screen before building your manual GBM pipeline can surface unexpected interactions and reveal which variables are genuinely carrying signal. This is low-cost and frequently useful.

**Rapid prototyping.** A new dataset, an unfamiliar line of business, a proof-of-concept for a new rating variable — AutoML gives you a reasonable baseline in 30 minutes. If the baseline is weak, you know the signal is weak. If it is strong, you have a starting point for a properly-specified model.

---

## Our recommended approach: AutoML for exploration, manual for production

For a frequency model going into production, our view is:

Use AutoGluon or H2O AutoML as an exploratory tool. Run it on your training data for 30-60 minutes. Inspect the leaderboard for which algorithms dominate, which features rank highest in SHAP importance, and what performance ceiling the data seems to support. Treat the AutoML output as a well-informed prior for your manual model design.

Then build the production model manually. For frequency:

1. LightGBM with `objective='poisson'` and `init_score=log(exposure)` as the GBM backbone
2. Optuna for HPO with Poisson deviance as the objective, 200+ trials
3. Explicit monotonicity constraints where the business requires them
4. SHAP-based relativity tables for regulatory submission

This approach takes longer than AutoML alone, but it produces a model that is statistically correct, calibrated, and explainable. The AutoML phase saves time in the exploratory step; the manual phase ensures the production model meets actuarial standards.

If you want to use AutoML further in the pipeline, H2O gives you the best starting point — `monotone_constraints` at the AutoML level is a genuine advantage, and the individual H2O algorithms do support offsets when you bypass the AutoML wrapper. A hybrid approach where you use AutoML for architecture selection and then retrain the winning architecture with proper offset handling is practical.

---

## What needs to change

The gap is not subtle. Adding `offset_column` to `H2OAutoML.train()` is presumably not a large engineering lift for the H2O team — individual algorithms already support it. The same addition to FLAML's `AutoML.fit()` would require surfacing it from the custom estimator interface. AutoGluon would need a new problem type or a more flexible custom metric/loss system.

None of this has happened yet because insurance is a niche, and the actuarial community has not, collectively, filed the GitHub issues and made the case loudly enough. That is a gap in our professional engagement with open-source tooling, not a fundamental limitation of the frameworks.

Until it changes, AutoML for insurance frequency modelling requires workarounds that undermine the point of AutoML. Use it where it works. For frequency models, build it properly.

---

*freMTPL2 data from the CASdatasets R package (Dutang & Charpentier). Dong & Quan (2024): arXiv:2408.14331. Wüthrich (2019) GLM baseline: 0.3149 mean Poisson deviance on freMTPL2freq. H2O AutoML v3.46, FLAML v2.5, AutoGluon v1.5.*
