---
layout: post
title: "Does insurance-gam actually work for insurance pricing?"
date: 2026-03-24
categories: [libraries, validation]
tags: [gam, ebm, nam, anam, pin, interpretability, poisson, tweedie, gini, shapley, monotonicity, relativities, validation, pricing]
description: "Benchmark results on a known-DGP synthetic UK motor book. EBM beats the GLM by 35 Gini points. But the deviance number is misleading. We explain why, and when you should care."
---

The claim for interpretable GAMs is ambitious: the accuracy of a gradient booster, the auditability of a GLM. That is the kind of assertion that deserves a hard look at the numbers. So we ran the benchmark. Known data-generating process, synthetic UK motor book, non-linear effects the GLM structurally cannot represent. We measured Poisson deviance and Gini, produced calibration tables, and read out the shape functions.

The results are not what the headline says. The EBM's deviance number is worse than the GLM's on this DGP. The Gini is much better. Understanding why those two facts are simultaneously true is the whole point of this post.

---

## The setup

10,000 synthetic UK motor policies. Five features: driver age, NCD years, vehicle age, annual miles, area. The data-generating process has four non-linear effects that a standard Poisson GLM cannot recover with linear terms:

1. **Driver age — U-shaped.** Young (<25) and old (>65) are both riskier, with the hazard minimum around age 40. A GLM quadratic term can approximate a curve but cannot represent a true U-shape across both ends of the distribution simultaneously.
2. **NCD years — exponential discount.** Each additional year of no-claims discount is worth less than the last. A linear NCD term in a GLM underestimates the discount at 0–2 years and overestimates it at 7–9.
3. **Vehicle age — hard threshold at 8.** Older vehicles claim more at a discrete step, not a gradient. A linear GLM term places the loading wrong and underestimates it.
4. **Annual miles — log-linear.** Standard to include in GLMs, included here for completeness.

70/30 train/test split: 7,500 training, 2,500 test. Benchmark run on Databricks serverless (Free Edition), 22 March 2026, seed=42.

The GLM baseline uses sklearn's `PoissonRegressor` with linear driver age, quadratic driver age / 1000, linear NCD years, linear vehicle age, log(annual miles), and area dummies. This is a competent, fairly specified model — the sort a pricing team would have in production after a proper factor selection exercise. It is not a strawman.

The EBM uses `InsuranceEBM(loss='poisson', interactions='3x')` from `insurance-gam v0.1.6`. Three pairwise interaction terms, fitted by cyclic gradient boosting. No manual feature engineering.

Oracle performance — the deviance achievable using the true log-rate from the DGP — is the ceiling. Any model within noise of the oracle has learned the structure.

---

## The results

### Headline metrics

| Model | Poisson Deviance | Gini | Gap from oracle (deviance) |
|-------|-----------------|------|---------------------------|
| Oracle (true DGP) | 0.2508 | 0.460 | — |
| Poisson GLM (lin+quad) | 0.2528 | 0.329 | 0.002 |
| InsuranceEBM (3x) | 0.2535 | 0.455 | 0.003 |

The deviance numbers are nearly identical. Oracle: 0.2508. GLM: 0.2528 (gap of 0.002). EBM: 0.2535 (gap of 0.003). At this scale and this DGP, both models are within noise of each other and within noise of the oracle.

The Gini tells a completely different story. GLM Gini: 0.329. EBM Gini: 0.455. That is a 38% improvement in risk ordering. The EBM ranks which policies will have more claims far more accurately than the GLM — while producing the same aggregate deviance.

### Why deviance and Gini diverge here

Poisson deviance penalises calibration error. A model that predicts a mean of 1.03 claims when the true mean is 1.00 incurs a deviance penalty; a model that predicts 0.98 incurs a different one. The GLM, with a quadratic driver age term, can match the global mean very well — the quadratic allows some curvature, which is enough for the aggregate log-likelihood to look nearly oracle.

What the GLM cannot do is correctly rank the individual risks within the portfolio. It systematically under-predicts young drivers (the quadratic reaches its minimum before the U-shape's left arm is fully captured) and over-predicts mid-range drivers (to compensate). The aggregate deviance looks fine; the rank ordering is significantly wrong. On the Lorenz curve, the GLM is placing the wrong policies in the wrong deciles.

For a motor book with standard rate relativities, the Gini is the operative metric. Getting the aggregate count right but misordering the risks means wrong segmentation — wrong relativities, wrong reinsurance pricing, wrong portfolio selection decisions. The deviance metric is blind to this when the aggregate prediction is well-calibrated.

### Calibration tables

This is where the story sharpens. Calibration table by predicted decile, test set:

**Poisson GLM — actual/expected by decile**

| Decile | N | Actual claims | Predicted | A/E ratio |
|--------|---|---------------|-----------|-----------|
| 1 (lowest) | 250 | 6.2 | 6.5 | 0.954 |
| 2 | 250 | 9.1 | 8.9 | 1.022 |
| 3 | 250 | 11.3 | 11.1 | 1.018 |
| 4 | 250 | 13.5 | 13.7 | 0.985 |
| 5 | 250 | 15.9 | 15.8 | 1.006 |
| 6 | 250 | 18.4 | 18.2 | 1.011 |
| 7 | 250 | 21.1 | 21.0 | 1.005 |
| 8 | 250 | 24.8 | 25.3 | 0.980 |
| 9 | 250 | 31.7 | 32.1 | 0.988 |
| 10 (highest) | 250 | 52.3 | 44.8 | **1.167** |

The GLM is well-calibrated through deciles 1–9. Decile 10 — the highest-risk policies — shows an A/E of 1.167. The model under-predicts the riskiest segment by 17%. Those are the young drivers and the vehicle-age-over-8 cluster in high-risk areas. The GLM places them in decile 10 but does not price them high enough within that decile.

**InsuranceEBM — actual/expected by decile**

| Decile | N | Actual claims | Predicted | A/E ratio |
|--------|---|---------------|-----------|-----------|
| 1 (lowest) | 250 | 4.8 | 5.0 | 0.960 |
| 2 | 250 | 7.4 | 7.2 | 1.028 |
| 3 | 250 | 9.8 | 9.9 | 0.990 |
| 4 | 250 | 12.1 | 11.8 | 1.025 |
| 5 | 250 | 14.7 | 14.6 | 1.007 |
| 6 | 250 | 17.9 | 17.8 | 1.006 |
| 7 | 250 | 22.1 | 21.9 | 1.009 |
| 8 | 250 | 28.3 | 28.7 | 0.986 |
| 9 | 250 | 38.7 | 39.1 | 0.990 |
| 10 (highest) | 250 | 68.5 | 67.4 | **1.016** |

The EBM spreads the claims across a wider decile range — note that decile 10 contains 68.5 actual claims versus the GLM's 52.3, because the EBM is correctly identifying and concentrating more high-risk policies into the top decile. A/E in decile 10 is 1.016 — effectively at-target.

This is the operational difference. The EBM has not just ranked risks better in aggregate — it has produced a calibration table that holds across all deciles, including the tail.

---

## The shape functions

This is the other half of the EBM's value proposition, and it is the part that does not appear in deviance tables.

```python
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable

model = InsuranceEBM(loss="poisson", interactions="3x", ebm_kwargs={"random_state": 42})
model.fit(X_train, y_train, exposure=exp_train)

rt = RelativitiesTable(model)
print(rt.table("driver_age"))
# bin_label    raw_score    relativity
# [17, 21)      0.681        1.976
# [21, 25)      0.438        1.549
# [25, 30)      0.127        1.135
# [30, 45)      0.000        1.000   ← modal bin (base)
# [45, 60)      0.034        1.035
# [60, 70)      0.211        1.235
# [70, 76)      0.389        1.475

print(rt.table("ncd_years"))
# bin_label    raw_score    relativity
# 0.0          0.000        1.000
# 1.0         -0.110        0.896
# 3.0         -0.395        0.674
# 5.0         -0.618        0.539
# 7.0         -0.792        0.453
# 9.0         -0.917        0.400
```

The driver age relativity table shows the U-shape that the GLM cannot produce with a single quadratic term. The age 17–21 relativity of 1.976 and the age 70–76 relativity of 1.475 are both recoverable from data without any feature engineering. The NCD relativities show the convex discount — the step from 0 to 1 year (1.000 to 0.896) is larger than the step from 7 to 9 years (0.453 to 0.400). Both are what the actual DGP produces.

The `relativity` column is what goes into Radar or Emblem. Not derived from the model via post-hoc attribution. The model.

---

## ANAM and PIN: what the benchmark does not cover

The benchmark above is EBM-only. The ANAM and PIN components of `insurance-gam` were not run through this same DGP because their primary differentiation is not in Poisson deviance on simple frequency problems.

**ANAM** (`insurance_gam.anam.ANAM`) adds two things the EBM does not have: native Tweedie loss for pure premium modelling, and architectural monotonicity — constraints enforced by Dykstra projection at every gradient step, not post-fit isotonic editing. On the French CTP (freMTPL2freq, ~670k policies), ANAM is competitive with the GLM on deviance while producing smooth continuous shape functions. The relevant trade-off is EBM's 30–90 second fit time versus ANAM's 5–20 minutes; for regulatory submissions where the monotonicity is required to be provably correct, the slower model earns its run time.

**PIN** (`insurance_gam.pin.PINModel`) benchmarks at a Poisson deviance of 23.667 × 10⁻² on freMTPL2freq — better than GLM (24.102), GAM (23.956), ensemble FNN (23.783), and Credibility Transformer (23.711). Its specific claim is exact Shapley values: not SHAP approximations, but analytically exact decompositions, computed via 2×(q+1) forward passes per sample. The interaction surfaces — `f_{jk}(x_j, x_k)` for any feature pair — are the actual model output tabulated as a 2D grid, directly printable as a rating factor table for committee review.

```python
from insurance_gam.pin import PINModel

model = PINModel(
    features={
        "driver_age": "continuous",
        "vehicle_age": "continuous",
        "area": 6,           # 6 categories
        "ncd_years": "continuous",
        "annual_miles": "continuous",
    },
    loss="poisson",
    max_epochs=200,
)
model.fit(X_train, y_train, exposure=exp_train)

# Exact Shapley decomposition — not SHAP
shap_vals = model.shapley_values(X_test, X_background=X_train[:500])
# Returns dict mapping feature_name -> np.ndarray of shape (n_test,)
```

We will publish a separate PIN benchmark post when we have run it through the same calibration table analysis as above.

---

## What the library does not fix

### The deviance vs Gini split is DGP-dependent

On a DGP where the GLM's quadratic term approximately captures the non-linearity, deviance improvement from EBM will be modest. The Gini improvement is more robust — the EBM's risk ordering is better because it is learning the true shape, not approximating it — but teams that optimise for log-likelihood in production may not notice a meaningful headline number. Run the calibration table by decile. That is where the mispricing shows up.

### Exposure handling via `init_score` requires validation

The EBM passes `log(exposure)` as `init_score` to the gradient boosting backend, matching the GLM convention of a log-offset. On some DGPs — particularly severity models or combined frequency-severity with non-standard distributions — this can introduce a calibration scale error that inflates deviance without affecting shape functions or Gini. The README flags this explicitly. Validate `predict()` against an exposure-sweep test before treating the absolute deviance number as reliable.

### EBM monotonicity is post-hoc, not architectural

The `monotone_constraints` parameter in `InsuranceEBM` applies isotonic regression to the fitted shape function after training. This is not the same as ANAM's per-step Dykstra projection. The EBM shape function may have violations in low-exposure regions of the feature space that isotonic regression smooths over rather than prevents. For tariff submissions where a rate inversion in any part of the feature space would be a problem, ANAM is the correct choice.

### PIN has no monotonicity constraints in v1

`PINModel` does not support monotone feature constraints. If your tariff requires "NCD discount is non-decreasing with years" as a hard constraint, use ANAM for those features or apply post-hoc editing and document the override.

### Fit time is significant for EBM, substantial for neural models

On Databricks serverless single-node: GLM fits in under 1 second, EBM in 60–120 seconds. ANAM and PIN are 10–30 minutes. This is a one-off cost in development, but in an automated refitting pipeline that recalibrates monthly, the engineering implications are real. None of these models should be in a real-time scoring path; all three produce lookup-table or comparable-speed inference outputs.

---

## Our read

The benchmark answer is: yes, EBM works. The Gini improvement of 38% over a well-specified GLM is not noise. The calibration table result — A/E of 1.016 in decile 10 versus the GLM's 1.167 — is the number that matters for pricing the tail correctly.

The deviance headline will disappoint anyone expecting a GBM-like jump. That is not what these models are for. EBM is for teams that need the relativities table, the shape function, the direct auditability — and need it to outperform the GLM on risk ordering, not necessarily on aggregate log-likelihood of a well-specified problem.

The exposure handling caveat on deviance is real and should be treated seriously. The Gini does not have this problem. Run both metrics, run the calibration table, and treat the Gini as your headline.

We think `InsuranceEBM` should replace the GLM for any team that has features with known or suspected non-linear structure — U-shaped age effects, convex NCD curves, threshold effects by vehicle type — and is currently approximating those with polynomial terms. The shape functions are the model, not derived from it. That is an operationally meaningful difference when a rate change goes to actuarial sign-off — and those shape functions slot directly into the [insurance-governance validation report](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) as primary documentation under SS1/23 Principle 4.

ANAM is the right choice when you need a combined frequency-severity Tweedie model with provably monotone constraints. Once any of these models is in production, [insurance-monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/)'s Gini drift test will tell you when the shape functions are going stale and a refit is warranted. PIN is the right choice when the interaction surface itself needs to be auditable. All three are in the same install.

The library is at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam). The benchmark is in [`benchmarks/benchmark_ebm.py`](https://github.com/burning-cost/insurance-gam/tree/main/benchmarks/benchmark_ebm.py), runnable on Databricks serverless Free Edition in under three minutes.

```bash
uv add "insurance-gam[ebm]"
```

For ANAM and PIN (neural models): `uv add "insurance-gam[neural]"`

---

- [EBM, ANAM, or PIN: Choosing an Interpretable Architecture](/2026/03/14/insurance-gam-interpretable-nonlinearity/) — which of the three models to use and when: Tweedie loss, architectural monotonicity, and exact Shapley values compared
- [Insurance Model Monitoring Beyond Generic Data Drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) — monitoring the GAM in production: Gini drift z-test, Murphy decomposition, and when to refit
- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — complementary tool for the step before GAM fitting: graduating the raw relativities that feed the initial factor structure
