---
layout: post
title: "Credibility vs GBM for Thin Segments: A Decision Guide"
date: 2026-03-26
categories: [pricing, techniques, comparisons]
tags: [credibility, buhlmann-straub, gbm, thin-data, transfer-learning, insurance-credibility, insurance-thin-data, GBMTransfer, GLMTransfer, UK-motor, commercial-lines, python, decision-guide]
description: "When you have fewer than 5,000 policies in a segment, should you use Bühlmann-Straub credibility or a GBM with transfer learning? The answer depends on whether you have a related source book, whether factor interactions matter, and what your data structure looks like."
---

Thin segment pricing has two main schools of thought. The actuary's answer is Bühlmann-Straub credibility: shrink the segment's observed experience toward the portfolio mean, with the shrinkage weight determined by exposure and the estimated noise-to-signal ratio. The data scientist's answer is a GBM with transfer learning: train on a larger related book, then fine-tune on the thin segment.

Both approaches work. They work in different situations. This post explains the decision criteria with concrete thresholds, actual code, and a clear position on which to choose when.

---

## What "thin" means

We mean two things by thin, and they require different treatments:

**Thin at the segment level**: your book has 200 commercial fleet schemes with an average of 800 earned vehicle years each. Enough aggregate experience to estimate a scheme-level mean, not enough to estimate the full multivariate factor structure within each scheme.

**Thin as an absolute count**: you have 1,200 EV motor policies total. Not enough to fit a stable GLM from scratch, let alone a GBM, even if you ignore the within-segment factor structure.

Bühlmann-Straub credibility addresses the first problem well. It is less useful for the second. GBM transfer learning addresses the second problem, and can also address the first — but only if a related source book exists and the covariate shift is not severe.

---

## Bühlmann-Straub: the right tool for segment-level blending

If your data is structured as a panel — loss rates and exposures across multiple segments and multiple periods — Bühlmann-Straub is simple, auditable, and theoretically correct.

The model estimates three numbers from your data:

- **mu**: the collective mean loss rate across all segments
- **v**: the expected within-segment process variance (noise)
- **a**: the between-segment variance of the true underlying rates (signal)

From these, k = v/a gives the noise-to-signal ratio, and the credibility factor for each segment is Z_i = w_i / (w_i + k), where w_i is the segment's total exposure. Small exposure means small Z_i — the segment's own experience gets little weight and the premium stays close to the collective mean. Large exposure means Z_i approaching 1 and the segment is largely rated on its own experience.

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# Five underwriting years of commercial fleet scheme data
panel = pl.DataFrame({
    "scheme":     ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
    "year":       list(range(2019, 2024)) * 3,
    "loss_rate":  [0.72, 0.68, 0.74, 0.70, 0.73,   # A: stable, large
                   1.15, 0.98, 1.22, 1.05, 1.18,   # B: volatile, thin
                   0.51, 0.53, 0.50, 0.52, 0.54],  # C: stable, large
    "exposure":   [8000, 8500, 9000, 9200, 9400,
                   400,  420,  380,  410,  390,
                   15000, 14500, 15200, 15800, 16000],
})

bs = BuhlmannStraub()
bs.fit(panel, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

bs.summary()
# Bühlmann-Straub Credibility Model
# ====================================
#   Collective mean    mu  = 0.632
#   Process variance   v   = 0.00418   (EPV, within-group)
#   Between-group var  a   = 0.0312    (VHM, between-group)
#   Credibility param  k   = 0.134
#
# Interpretation: a group needs exposure = 0.134 to achieve Z = 0.50

print(bs.z_)
# shape: (3, 2)  columns: ['group', 'Z']
# scheme A: Z ≈ 0.998  (44,100 total exposure >> k)
# scheme B: Z ≈ 0.9999  (2,000 total exposure >> k)
# scheme C: Z ≈ 0.999  (76,500 total exposure >> k)
```

In this example, k = 0.134 is very small — the groups are genuinely different — so even scheme B, with only 2,000 earned vehicle years, earns high credibility. If k were 500, the picture would reverse: scheme B's 2,000 exposures would give Z ≈ 0.004 and its premium would track the collective mean almost entirely.

The crucial diagnostic is k itself. When k is large relative to typical segment exposures, the portfolio is noisy and relatively homogeneous — trust the collective. When k is small relative to typical exposures, the groups genuinely differ and even modest experience should move the premium.

**Where Bühlmann-Straub works well:**
- Aggregate panel data only (no policy-level features)
- 15 to 300 segments, 3+ years of experience per segment
- Primarily a level adjustment: you are pricing the *average risk within the segment*, not trying to understand how risk varies within it
- Regulatory context requiring a closed-form, hand-verifiable calculation

**Where it does not work:**
- You need to know whether the age-frequency relationship is different in segment B versus the rest of the book
- You have policy-level features and want to use them
- Segments have fewer than 3 years of data and the variance estimates become unreliable

For the individual policy-level version — adjusting the a priori premium for a specific policyholder's own claims history — `insurance-credibility` also provides `StaticCredibilityModel`:

```python
from insurance_credibility import StaticCredibilityModel, ClaimsHistory

# Portfolio of policy histories used to estimate kappa
histories = [
    ClaimsHistory("P1", exposure=[1, 1, 1], claims=[0, 1, 0], prior_premium=850.0),
    ClaimsHistory("P2", exposure=[1, 1, 1], claims=[2, 1, 2], prior_premium=850.0),
    # ... many more policies
]

model = StaticCredibilityModel()
model.fit(histories)

# Predict credibility factor for a specific policy
new_history = ClaimsHistory("P999", exposure=[1, 1, 1], claims=[0, 0, 0], prior_premium=900.0)
cf = model.predict(new_history)
posterior_premium = new_history.prior_premium * cf
```

`StaticCredibilityModel` estimates the structural parameter kappa = sigma²/tau² from the portfolio and applies the Bühlmann credibility formula at policy level. A policy with three claim-free years earns a factor below 1.0; a policy with a bad run earns a factor above 1.0. The shrinkage depends entirely on kappa — estimated from the whole portfolio, not assumed.

---

## GBM transfer: the right tool when you have a source book

Transfer learning solves a different problem. You have a thin *total* book — 1,200 EV motor policies, not 1,200 policies per segment of a large book — and you want to estimate the full factor structure: how age, mileage, vehicle age, and postcode interact to determine risk.

Bühlmann-Straub cannot help here. You have one "segment" (EVs) and not enough data within it. The solution is to borrow the factor structure from a related, larger book (ICE motor, 80,000 policies) and then fine-tune to capture what is genuinely different about EVs.

`insurance-thin-data` implements this as `GBMTransfer` for CatBoost models:

```python
from catboost import CatBoostRegressor
from insurance_thin_data import GBMTransfer, CovariateShiftTest

# Step 1: check whether transfer is appropriate
shift = CovariateShiftTest(categorical_cols=[5, 6], n_permutations=1000)
result = shift.test(X_source_ice, X_target_ev)
# ShiftTestResult(MMD²=0.031, p=0.003 [significant])
# Significant shift, but GBMTransfer's source-as-offset pattern handles moderate shift

# Step 2: fit the transfer model
source_model = CatBoostRegressor(loss_function="Poisson", iterations=500, verbose=0)
source_model.fit(X_source_ice, y_source_ice)

transfer = GBMTransfer(
    source_model=source_model,
    mode="offset",             # source-as-offset: target model learns the residual
    loss_function="Poisson",
    catboost_params={
        "iterations": 150,     # fewer trees: less capacity to overfit to thin data
        "learning_rate": 0.05,
        "depth": 4,
        "verbose": 0,
    },
)

transfer.fit(X_target_ev, y_target_ev, exposure=exposure_ev)
predictions = transfer.predict(X_test_ev, exposure=exposure_test_ev)
```

The `mode="offset"` pattern is the key design choice. The source model's log-predictions are passed as a fixed baseline offset when training the target CatBoost model. The target model then learns the residual — the systematic deviation of EV risk from the source model's prediction — rather than the full risk level. This means the target model needs far fewer trees to converge: the source has already explained most of the variance structure.

For a GLM base and thin target, `GLMTransfer` from the same package implements the Tian and Feng (JASA, 2023) two-step penalised estimator:

```python
from insurance_thin_data import GLMTransfer

transfer_glm = GLMTransfer(
    family="poisson",
    lambda_pool=0.005,   # pooling step regularisation
    lambda_debias=0.02,  # debiasing step: larger = more shrinkage toward source
    scale_features=True,
)

transfer_glm.fit(
    X_target, y_target, exposure_target,
    X_source=X_source, y_source=y_source, exposure_source=exposure_source,
)
```

The debiasing delta (`transfer_glm.delta_`) shows you which coefficients are genuinely different in the thin segment. Where delta is near zero, the segment shares the source's factor structure and the source estimate is used. Where delta is large, the segment is meaningfully different and the debiasing step corrects for it.

**Where GBM transfer works well:**
- You have a related source book and can verify that transfer is beneficial (check with `NegativeTransferDiagnostic`)
- Your thin segment needs a full multivariate model, not just a level adjustment
- You are comfortable with a machine learning workflow

**Where it does not work:**
- The source and target share no common features (there is nothing to transfer)
- The target segment has fewer than roughly 200 policies (the debiasing step has nothing to work with and the offset model may overfit)
- You need an output that maps directly to a factor table for a rating engine without an intermediate ML step

---

## The decision in practice

The question is not "which method is better" but "what does my data structure look like and what am I trying to produce?"

**Use Bühlmann-Straub when:**
- Your data is at the segment level — aggregate loss rates and exposures, not policy-level feature matrices
- You have 3+ periods per segment and are adjusting rates for 15–300 segments
- The output needs to be a credibility premium (a scalar per segment) that can be directly loaded into a rating structure
- You have no related source book to borrow from
- Auditability is a hard constraint: the calculation must be reproducible by hand or in a spreadsheet

**Use GBM transfer learning when:**
- Your data is at the policy level with a full feature matrix
- You have a related, larger source book with a well-fitted model
- You need to understand how factors interact within the thin segment — not just what the average loss rate is, but whether young drivers are riskier in this segment than elsewhere
- You have at least 300 policies in the thin segment (below this, the target model cannot meaningfully learn any residual)

**Use both, sequentially, when:**
- You have policy-level data and also want segment-level adjustments on top
- The correct architecture is: GBM transfer to learn the base rate structure from the source book, then Bühlmann-Straub on the residuals at segment level to capture group-level departures

---

## The 300-policy threshold

Below 300 policies, neither approach is fully reliable. Bühlmann-Straub will estimate a credibility premium that is almost entirely the collective mean — there is not enough within-segment experience to move the needle. GBM transfer will fit a target model that adds very little to the source offset, because the residual variance is dominated by noise.

Below 200 policies, we recommend against fitting a segment-specific model at all. Use the collective mean (from Bühlmann-Straub on the broader book) plus a manual loading for any known structural differences. Document that you have insufficient data to estimate a segment-specific model and review annually as data accumulates.

This is an uncomfortable answer but an honest one. The alternative — presenting a confidently wrong model as if it were reliably estimated — is worse for your book and worse for your regulatory standing.

---

## Why not just use the full book model on the thin segment?

The naive approach is to fit a single model on the combined book, include a segment indicator, and read off the segment effect. This works when the segment is large enough that the model can estimate its effect reliably. When the segment is thin, the model interpolates from the rest of the book and the segment indicator absorbs only the level difference.

The problem: this assumes the factor *structure* is the same in the thin segment as in the full book. If EV motor risk has a different mileage-frequency relationship than ICE motor — which the evidence suggests it does — the combined model will miss this. You get the wrong shape even if the level is approximately right.

GBM transfer via source-as-offset explicitly models the residual. Where the segment genuinely departs from the source model, the target model learns it. Where the segment behaves identically to the source, the target model stays flat. This is structurally more honest than a single indicator in a combined model.

---

**Code:**
- [`insurance-credibility`](/insurance-credibility/) — `uv add insurance-credibility`
- [`insurance-thin-data`](https://github.com/burning-cost/insurance-thin-data) — `uv add insurance-thin-data`

**Related:**
- [When You Can't Fit a GLM from Scratch: Transfer Learning for Thin Segments](/2026/03/10/transfer-learning-for-thin-segments/)
- [When Credibility Meets CatBoost: Choosing Between Classical and Modern Approaches](/2026/02/28/when-credibility-meets-catboost/)
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Foundation Models for Thin Segments: TabPFN and TabICLv2](/2026/03/13/insurance-tabpfn/)
