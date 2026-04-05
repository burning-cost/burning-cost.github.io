---
layout: post
title: "ZeroInflatedTweedieGBM: The So & Valdez (2024) Implementation"
date: 2026-03-28
categories: [techniques, libraries]
tags: [zero-inflation, tweedie, catboost, gbm, insurance-distributional, so-valdez, pure-premium, contents-insurance, astin, severity, frequency-severity, python]
description: "The first pip-installable implementation we know of for So & Valdez (2024) Scenario 2: a two-stage CatBoost model that separates structural zero probability from Tweedie severity. When standard Tweedie's implicit zero-handling is too crude, this is the fix."
---

The [previous post on zero-inflated models](/2026/03/24/zero-inflated-and-hurdle-models-for-insurance-claims/) covered the count side: ZIP for frequency, the structural-vs-sampling-zero distinction, and when Tweedie's implicit zero handling breaks down. That post deliberately sidestepped the continuous aggregate cost version.

This post covers the continuous version. Specifically: `ZeroInflatedTweedieGBM`, the first pip-installable implementation of Scenario 2 that we are aware of from So & Valdez (*Boosted trees for zero-inflated counts with an offset for insurance ratemaking*, Applied Soft Computing, arXiv 2406.16206). If you are modelling aggregate losses — not claim counts — on a book where 85–95% of policies have zero outgo, this is the model that fits the problem.

---

## What standard Tweedie gets wrong on zero-heavy books

The Tweedie GLM's zero probability is fully determined by the mean:

```
P(Y = 0) = exp(-mu^(2-p) / (phi * (2-p)))
```

Once you fix p (typically 1.5 for motor) and estimate phi, the proportion of zeros implied by your model is a direct function of the mean prediction. There is no free parameter to adjust it independently. The model cannot simultaneously produce a high zero rate and a realistic severity distribution for the policies that do claim, because both come from the same mean parameter.

For a UK contents book where 90% of policies have zero claims, this constraint is binding. The model pushes the mean down to accommodate the zero mass, and the non-zero severity estimates get compressed as a result. You will not see this in aggregate Tweedie deviance — it is not a sensitive test for this kind of misspecification. But you will see it in actual-vs-expected by claim size band: the model will underpredict the tail of non-zero claims.

The standard fix in personal lines is the frequency-severity split: separate Poisson frequency and Gamma severity models. This works well and is defensible. But it separates by model architecture, not by the statistical problem. Severity is still estimated on the same underlying risk population, and the two models must be calibrated to multiply to the right pure premium. ZI-Tweedie takes a different route.

---

## Scenario 2: a two-stage model for aggregate loss

So & Valdez (2024) propose two scenarios for zero-inflated Tweedie GBMs. Scenario 1 is a joint maximum likelihood approach over the full ZI-Tweedie likelihood — theoretically cleaner but computationally harder to implement. Scenario 2, which we implement, is the two-stage version:

**Stage 1 — Zero probability model:**
Train a binary classifier on all observations, predicting P(Y = 0 | x). The label is 1 if the policy had zero outgo, 0 otherwise.

**Stage 2 — Tweedie severity model:**
Train a Tweedie regressor on the non-zero observations only, estimating E[Y | Y > 0, x].

**Combined prediction:**

```
E[Y | x] = (1 - π̂(x)) × μ̂_Tweedie(x)
```

This is a Cragg (1971) two-part model applied to the Tweedie family. It is not a new idea. The contribution of So & Valdez is the gradient boosting formulation with CatBoost as the boosting engine, the treatment of the log-exposure offset in Stage 2, and the empirical validation that this framing outperforms a standard TweedieGBM on the datasets that justify using it.

The key architectural decision in our implementation: Stage 1 uses CatBoost's `CrossEntropy` loss on all n observations. Stage 2 uses CatBoost's `Tweedie:variance_power=p` loss on the n_nonzero observations only, with `log(exposure)` as a baseline offset. The zero classifier and the severity model are independently trained — there is no joint gradient signal between stages.

This is the honest version of the model. Scenario 1 would have a joint gradient, but implementing a custom Tweedie ZI loss in CatBoost requires substantially more engineering and produces marginal gains on most real datasets.

---

## Using `ZeroInflatedTweedieGBM`

```bash
uv add insurance-distributional
```

The minimal workflow:

```python
import numpy as np
from insurance_distributional import ZeroInflatedTweedieGBM

rng = np.random.default_rng(2026)
n = 5_000

# UK contents portfolio: vehicle age, sum insured band, region risk score,
# occupancy type (0=owner-occupied, 1=rented), alarm grade
X = np.column_stack([
    rng.integers(0, 30, n).astype(float),   # property_age
    rng.uniform(50_000, 500_000, n),        # sum_insured
    rng.uniform(0.5, 2.0, n),              # region_risk
    rng.integers(0, 2, n).astype(float),   # rented
    rng.integers(1, 5, n).astype(float),   # alarm_grade
])

# True DGP: 88% structural zeros (most contents policies never claim)
pi_true = 0.88
is_zero = rng.random(n) < pi_true
# Non-zero severity: Gamma-ish, mean around £1,200
y = np.where(is_zero, 0.0, rng.gamma(shape=1.5, scale=800, size=n))

print(f"Zero rate: {(y == 0).mean():.1%}")
# Zero rate: 87.8%

model = ZeroInflatedTweedieGBM(power=1.5)
model.fit(X, y)
```

The model trains silently — set `catboost_params_zero={'verbose': 100}` or `catboost_params_severity={'verbose': 100}` if you want iteration logs.

---

## Prediction: the combined mean and its components

`predict()` returns the combined mean E[Y | x]. But the value of this model is in the components:

```python
components = model.predict_components(X)
# Returns dict with keys: zero_prob, severity_mean, combined_mean

pi_hat   = components["zero_prob"]       # P(Y = 0 | x) from Stage 1
mu_hat   = components["severity_mean"]   # E[Y | Y > 0, x] from Stage 2
combined = components["combined_mean"]   # (1 - pi_hat) * mu_hat
```

`pi_hat` is what a standard Tweedie cannot give you. Two policies can sit at the same combined expected loss but for completely different reasons:

- Policy A: `pi_hat = 0.95`, `mu_hat = £2,000` — structural near-zero, but high severity if anything happens
- Policy B: `pi_hat = 0.60`, `mu_hat = £250` — frequent-ish but low severity per event

Both produce a combined mean of roughly £100. A Tweedie GBM treats them identically. ZI-Tweedie does not. For underwriter referrals, large-loss loading, and IFRS 17 risk adjustment, Policy A and Policy B are genuinely different risks.

---

## Exposure

Pass exposure to `fit()` and `predict()`. It enters Stage 2 as a log-offset, not Stage 1:

```python
exposure = rng.uniform(0.5, 1.0, n)  # partial-year policies

model = ZeroInflatedTweedieGBM(power=1.5)
model.fit(X, y, exposure=exposure)

# At prediction time, pass the new exposure
pred = model.predict(X_test, exposure=exposure_test)
components = model.predict_components(X_test, exposure=exposure_test)
```

Stage 1 (zero classifier) does not receive exposure. The zero probability P(Y = 0 | x) is a property of the risk, not of the observation window — or at least that is the modelling assumption. If you have strong prior reason to believe exposure materially affects whether a policy claims at all (e.g., very short mid-term policies), pass log(exposure) as a feature to the zero stage explicitly.

Stage 2 applies `log(exposure)` as a fixed offset: `log E[Y | Y > 0, x] = log(exposure) + f(x)`. This is the standard actuarial treatment — a policy on risk for six months contributes half the expected outgo of the same risk on risk for twelve months.

---

## Separate hyperparameters for each stage

Both stages accept CatBoost parameter overrides:

```python
model = ZeroInflatedTweedieGBM(
    power=1.5,
    catboost_params_zero={
        "iterations": 500,
        "learning_rate": 0.03,
        "depth": 5,
    },
    catboost_params_severity={
        "iterations": 600,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 5.0,
    },
)
```

The defaults are conservative: 300 iterations, learning rate 0.05, depth 6, L2 regularisation 3.0. For production use, tune them separately. The two stages can have very different optimal tree depth — the zero classifier often benefits from shallower trees than the severity model, particularly on books where the structural zero probability is a simpler function of the covariates than the severity surface.

Categorical features are passed to both stages:

```python
model = ZeroInflatedTweedieGBM(
    power=1.5,
    cat_features=[3, 4],  # column indices of categorical features
)
```

---

## Scoring

Two scoring methods for different purposes:

**Tweedie deviance** (`score()`): evaluates the combined mean against observed outcomes using the standard Tweedie deviance formula. This is comparable to what you would report for a standard TweedieGBM — lower is better.

```python
deviance = model.score(X_test, y_test, exposure=exposure_test)
```

**ZI log-score** (`log_score()`): separates the two-stage contribution. For zero observations it rewards the classifier; for positive observations it rewards both the classifier (did it correctly predict non-zero?) and the severity model.

```python
ll = model.log_score(X_test, y_test, exposure=exposure_test)
# Lower is better. Useful for comparing ZI-Tweedie against standard Tweedie.
```

When comparing `ZeroInflatedTweedieGBM` against `TweedieGBM` from the same library, the Tweedie deviance is the right head-to-head metric. Both return valid Tweedie deviance scores — the comparison is fair.

---

## When to use it over alternatives

This is the three-way choice that comes up on any zero-heavy aggregate loss problem:

**Standard TweedieGBM.** Use this when the zero rate is moderate (below ~75%), the Tweedie-implied zero probability diagnostic is not flagging a large gap, and you do not need separate interpretability on the zero-probability surface. For most UK motor books, this is still the right choice. The computational overhead of the two-stage model is only worth it when the misspecification is demonstrable.

**Frequency × Severity (separate Poisson + Gamma GBMs).** Use this when you want full business interpretability — separate claim frequency relativities for underwriting and separate severity relativities for large-loss analytics. The frequency-severity split gives you a clean story: "your frequency went up 8%, severity was flat." ZI-Tweedie does not decompose into frequency and severity in the traditional actuarial sense; it decomposes into zero-probability and conditional severity, which is a different decomposition. The Poisson frequency model can also accept claim counts as its target, which is more informative than the binary zero/non-zero label that Stage 1 of ZI-Tweedie uses.

**ZeroInflatedTweedieGBM.** Use this when: (a) the zero rate is above ~80%, (b) the Tweedie deviance diagnostic shows systematic misspecification on the zero mass, and (c) you want a single model that handles both stages in consistent GBM formulation with shared exposure offset treatment. Contents insurance, pet insurance, travel cancellation, breakdown cover, specific peril models. Books where a separate frequency model would be poorly behaved because claim counts are overwhelmingly 0 or 1.

A concrete diagnostic before you decide: compute the Tweedie-implied zero rate from your fitted standard model and compare it to the observed zero rate. A gap of more than 10 percentage points on a personal lines book is a clear signal. In that regime, ZI-Tweedie's Stage 1 will be estimating a non-trivial zero-probability surface that the standard model cannot represent.

---

## The paper

So & Valdez (2024) addressed a real problem that every pricing team with a zero-heavy book has, and did it addressed a real problem that every pricing team with a zero-heavy book has, and it addressed it rigorously. The paper covers Scenario 1 (joint ZI-Tweedie likelihood with gradient boosting) and Scenario 2 (the two-stage model here). Scenario 1 produces marginally better likelihood scores on the paper's datasets. Scenario 2 is substantially easier to implement cleanly, is more stable on small non-zero subsets, and produces diagnostics that practitioners can act on.

We implement Scenario 2. We think this is the right choice for a library aimed at production use. If you need Scenario 1's joint gradient, the paper gives you enough to implement it on top of a custom CatBoost objective — the mathematical formulation is clear. We are not aware of another pip-installable Python implementation of either scenario.

The [distributional GBMs overview post](/2026/03/05/insurance-distributional/) gives the broader context on where this sits in the `insurance-distributional` library alongside `TweedieGBM`, `ZIPGBM`, `GammaGBM`, and the dispersion models.

---

## References

- So, B. & Valdez, E.A. (2024). 'Boosted trees for zero-inflated counts with an offset for insurance ratemaking.' *Applied Soft Computing*. DOI: [10.1016/j.asoc.2025.113226](https://doi.org/10.1016/j.asoc.2025.113226). arXiv: [2406.16206](https://arxiv.org/abs/2406.16206).
- Cragg, J.G. (1971). 'Some statistical models for limited dependent variables with application to the demand for durable goods.' *Econometrica* 39(5):829–844.
- Duan, N., Manning, W.G., Morris, C.N. & Newhouse, J.P. (1983). 'A comparison of alternative models for the demand for medical care.' *Journal of Business & Economic Statistics* 1(2):115–126.

---

Related posts:
- [Zero-Inflated and Hurdle Models for Insurance Claims](/2026/03/24/zero-inflated-and-hurdle-models-for-insurance-claims/) — the count-model version: ZIP frequency, ZINB, diagnostics
- [Tweedie vs Frequency-Severity Split: When to Use Which](/2026/03/24/tweedie-vs-frequency-severity-when-to-use-which/) — the alternative to ZI-Tweedie for zero-heavy aggregate cost
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — `insurance-distributional` library overview
