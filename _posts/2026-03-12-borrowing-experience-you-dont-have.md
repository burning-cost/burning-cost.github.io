---
layout: post
title: "Borrowing Experience You Don't Have"
date: 2026-03-12
categories: [libraries, pricing, transfer-learning]
tags: [transfer-learning, GLMTransfer, GBMTransfer, CANNTransfer, thin-data, MMD, negative-transfer, CatBoost, Poisson, Gamma, CANN, Tian-Feng, python, insurance-transfer]
description: "Transfer learning for thin-segment insurance pricing. The Tian-Feng two-step GLM algorithm, CatBoost source-as-offset GBM transfer, and CANN fine-tuning — with mandatory MMD covariate shift testing and a negative transfer diagnostic that tells you when to stop."
---

Every pricing actuary has faced this problem. A segment exists — young drivers, exotic pets, a brand-new telematics portfolio — where the exposure is thin enough that fitting a standalone model produces something embarrassing. The confidence intervals are wide, the parameters are unstable, and you are essentially guessing. The standard response is to credibility-blend with the main book, which works up to a point, but credibility weighting is a blunt instrument when what you actually need is a model that borrows structure, not just a scalar adjustment.

Transfer learning is the more principled answer. [`insurance-transfer`](https://github.com/burning-cost/insurance-transfer) implements three approaches drawn from the academic literature, wrapped in a consistent API.

```bash
uv add insurance-transfer
```

---

## The problem with thin data, precisely stated

A segment with 300 policies and 18 claims has about as much information content as it sounds. Fit a GLM with eight rating factors and you will get coefficient standard errors that span the meaningful range of the factor. Your frequency estimate for a 19-year-old male with three months of driving history will be dominated by the prior you implicitly chose when you decided which data to include, not by the data itself.

The instinct is to pool. Include everyone aged 17–25 to get better frequency estimates for the 17–25 segment. That is transfer learning by another name, and it usually helps — but it also introduces systematic bias. If the young driver segment differs structurally from the main book (different vehicle mix, different annual mileage distribution, different claim reporting behaviour), then naively pooling will push your young driver coefficients toward the main book values even where the difference is real.

The question is not whether to borrow from a larger source dataset. The question is how to borrow in a way that preserves genuine differences while stabilising estimates where the thin segment genuinely agrees with the main book.

---

## Three methods

### 1. GLMTransfer: Tian and Feng's two-step

Tian and Feng (JASA, 2023, 118(544), 2684–2697) formalised a two-step transfer procedure for high-dimensional GLMs. The paper's innovation is the debiasing step.

Step one: pool source and target data and fit a regularised GLM (l1 penalty) to get an initial coefficient vector. This is the transfer step — the large source dataset stabilises your estimates.

Step two: fit a second model using target data only, where the response variable is the residual from step one. This debiasing step estimates delta — the difference between the pooled coefficients and what the target data actually supports. The final coefficients are the pooled estimate plus the debiased correction.

The effect is that where the target data is consistent with the source, the debiasing correction is near zero and you retain the stability of the pooled estimate. Where the target genuinely differs, the debiasing correction pulls the coefficient away from the source. It is Bayesian in spirit — the source acts as a prior, and the target data updates it — but it is implemented as a two-stage frequentist estimator, which makes it easier to reason about and to audit.

`GLMTransfer` is the first Python implementation. The R package `glmtrans` (on CRAN) covers the Gaussian and binomial families; this library adds Poisson and Gamma for frequency and severity modelling.

```python
from insurance_transfer import GLMTransfer

# source = main book, target = young driver segment
model = GLMTransfer(family="poisson", alpha=0.1)
model.fit(X_source, y_source, X_target, y_target, exposure_source, exposure_target)

# coefficients show where young drivers differ from main book
print(model.delta_)          # the debiasing correction
print(model.coef_)           # final coefficients (pooled + delta)
predict = model.predict(X_new, exposure_new)
```

The `alpha` parameter controls the l1 penalty in step one. Tune it via cross-validation on the target data; the library provides a `GLMTransferCV` wrapper that does this.

### 2. GBMTransfer: source as offset

Gradient boosted machines are not naturally amenable to the Tian-Feng approach — you cannot easily decompose a GBM fit into a pooled component and a debiasing correction. But there is a simpler pattern that works well in practice: pre-train on the source data, then use those predictions as a log-offset when fitting on the target.

The intuition: if your source model gives you a reasonable estimate of the baseline risk for any given set of covariates, then the target model only needs to learn the residual. With a thin target segment, learning residuals is easier than learning the full risk structure from scratch.

This is something sophisticated actuaries already do informally — "start from the main book rate and fit adjustments for the young driver segment" — but `GBMTransfer` formalises it with CatBoost and makes the offset construction explicit.

```python
from insurance_transfer import GBMTransfer

# pre-train on main book
source_model = GBMTransfer()
source_model.fit_source(X_source, y_source, exposure_source)

# fine-tune on young driver segment — source predictions become log-offset
source_model.fit_target(X_target, y_target, exposure_target)
predict = source_model.predict(X_new, exposure_new)

# or: bring your own pre-trained CatBoost model
transfer = GBMTransfer(source_model=pretrained_catboost)
transfer.fit_target(X_target, y_target, exposure_target)
```

The offset approach constrains the target model: it cannot wander far from the source predictions without strong evidence. With 300 policies, that constraint is usually a feature.

### 3. CANNTransfer: pre-train and fine-tune

Combined Actuarial Neural Networks (CANN, Schelldorfer and Wüthrich 2019) pair a GLM embedding with a neural network that learns residuals. `CANNTransfer` extends this to the transfer setting: pre-train the CANN on source data, freeze the main network layers, then fine-tune only the output layer on target data.

This is closest in spirit to how transfer learning works in NLP — a large language model is pre-trained on a broad corpus, then fine-tuned on a small task-specific dataset. The difference is that insurance pricing has domain structure (the GLM embedding) that anchors the fine-tuning and keeps the model from drifting into implausible territory.

```python
from insurance_transfer import CANNTransfer

model = CANNTransfer(hidden_layers=[64, 32], freeze_after="layer_2")
model.fit_source(X_source, y_source, exposure_source)
model.fine_tune(X_target, y_target, exposure_target, epochs=50, lr=1e-4)
predict = model.predict(X_new)
```

`CANNTransfer` works best when the source and target covariates are structurally similar — same features, different distribution. If the target segment uses a genuinely different feature set, you will need to rebuild the pre-training setup rather than fine-tune.

---

## The diagnostics matter as much as the methods

Transfer learning can go wrong. If the source and target distributions are too different, pooling information from the source will hurt more than it helps — a failure mode called negative transfer.

`insurance-transfer` includes two diagnostic tools that we consider mandatory parts of any transfer workflow.

**CovariateShiftTest** uses a mixed-kernel Maximum Mean Discrepancy (MMD) test to measure how different the source and target covariate distributions are. The kernel is RBF for continuous variables and indicator-based for categorical — this handles the typical insurance feature mix without pre-processing. A high MMD statistic is a warning: the source data may be misleading.

```python
from insurance_transfer import CovariateShiftTest

test = CovariateShiftTest(categorical_cols=["vehicle_group", "area"])
result = test.fit(X_source, X_target)
print(result.mmd_statistic)   # higher = more distribution shift
print(result.p_value)         # test against permutation null
```

**NegativeTransferDiagnostic** does the direct comparison: fit both a transfer model and a target-only model, then compare Poisson deviance on held-out target data. The NTG metric (negative transfer gain) is the deviance difference: negative means the transfer model is better, positive means target-only wins. If it is positive, use the target-only model.

```python
from insurance_transfer import NegativeTransferDiagnostic

diag = NegativeTransferDiagnostic(transfer_model=glm_transfer, baseline_family="poisson")
result = diag.evaluate(X_target_test, y_target_test, exposure_test)
print(result.ntg)             # deviance(transfer) - deviance(target_only)
if result.negative_transfer:
    print("Use target-only model")
```

This is not optional. We have seen cases — particularly in specialty lines where the source portfolio is commercial and the target is personal — where transfer learning actively degraded predictions. The diagnostic catches it.

---

## TransferPipeline: putting it together

```python
from insurance_transfer import TransferPipeline, GLMTransfer

pipeline = TransferPipeline(
    method=GLMTransfer(family="poisson"),
    shift_threshold=0.05,        # p-value threshold for shift test
    diagnose=True,
    categorical_cols=["vehicle_group", "area", "nc_bonus"],
)

result = pipeline.fit(
    X_source, y_source, exposure_source,
    X_target, y_target, exposure_target,
)

print(result.shift_detected)    # did MMD flag a distribution shift?
print(result.method_used)       # which method was actually fitted
print(result.negative_transfer) # did transfer help or hurt?
predict = result.model.predict(X_new, exposure_new)
```

The pipeline runs the shift test first. If the shift is large enough to be concerning, it logs a warning but continues — you may still want to transfer, but with eyes open. After fitting, it runs the negative transfer diagnostic automatically.

---

## Four use cases where this matters

**Young drivers.** 17–24-year-olds represent about 7% of UK licence holders but account for around 22% of road fatalities (Department for Transport, 2023 data). Insurers who price this segment well have a competitive advantage; insurers who price it badly lose money. The segment is large enough to be commercially significant, small enough that standalone modelling is uncomfortable. Transfer from the main book stabilises frequency and severity estimates while the debiasing step preserves the genuine (and substantial) risk differences.

**Rare breed pets.** A mid-size pet insurer might have 40,000 dog policies but only 30 French Bulldogs in a specific age and health status cell. Frequency and severity estimates for that cell from a standalone model are essentially useless. Transfer from the broader dog portfolio, with MMD confirming the covariate distributions are not wildly different, gives you something defensible.

**New entrant underwriting.** An insurer launching into a line of business without historical data has no credible option for standalone pricing. Industry data (where available) or public datasets can serve as source material for transfer. The negative transfer diagnostic will tell you whether the industry experience is predictive of your portfolio or misleading.

**Telematics bootstrap.** A traditional motor book transitioning to telematics has no claims history on the telematics population. Transfer from the traditional book, using conventional rating factors as shared covariates, bootstraps the telematics model until there is enough observed experience to stand alone.

---

## When to use which method

**GLMTransfer** is our default for frequency and severity modelling. The Tian-Feng algorithm has the strongest theoretical grounding, the debiasing step handles genuine structural differences cleanly, and the two-step structure gives you something auditable at each stage. If you are pricing in a regulated environment where you need to explain coefficient movements to the FCA, GLMTransfer gives you the most defensible story.

**GBMTransfer** is the right choice when you have already invested in CatBoost models for the main book and want to extend them to a thin segment without rebuilding. The offset pattern is simple enough to explain to non-technical stakeholders ("we start from the main book rate and fit adjustments") and robust in practice. It is not as theoretically elegant as GLMTransfer but it is fast and it works.

**CANNTransfer** is for teams already using neural network architectures for pricing — typically telematics or behaviour-based products where the feature space is richer than conventional rating factors support. It requires more careful hyperparameter management and is harder to audit, so we would not use it where regulatory explainability is the primary concern.

Run the CovariateShiftTest before any of them. If the MMD statistic is very high, treat the transfer result with scepticism regardless of which method you use. Run the NegativeTransferDiagnostic after any of them. If it flags negative transfer, use the target-only model.

---

## A note on the literature

Loke and Bauer (NAAJ, 2025, doi:10.1080/10920277.2024.2365615) survey transfer learning applications in the actuarial domain and provide the theoretical scaffolding for why these methods are appropriate for insurance pricing. The paper is behind a paywall; if you have access via an actuarial institute membership, it is worth reading alongside Tian and Feng.

Tian and Feng (JASA, 2023, 118(544), 2684–2697) is the core reference for `GLMTransfer`. The arXiv preprint (arXiv:2105.14328) is publicly available and covers the two-step algorithm in detail, including the convergence properties of the debiasing estimator under high-dimensional conditions.

---

**[insurance-transfer on GitHub](https://github.com/burning-cost/insurance-transfer)** — MIT-licensed, PyPI. For the segments your main book has never seen.

---

**Related articles from Burning Cost:**
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/)
- [Your Thin Segment Has 200 Policies. Your GLM Has No Idea What to Do With Them.](/2026/03/25/insurance-tabpfn/)
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [HMM-Based Telematics Risk Scoring for Insurance Pricing](/2026/03/21/insurance-telematics/)
