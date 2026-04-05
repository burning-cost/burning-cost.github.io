---
layout: post
title: "Quantile Premium Principle: What the NAAJ 2025 Benchmark Gets Right (and What It Missed)"
date: 2026-04-03
categories: [pricing, techniques, libraries]
tags: [quantile-regression, ratemaking, two-part-model, qpp, catboost, qrnn, severity, motor, uk-pricing, insurance-quantile, naaj, python]
description: "Zanzouri et al. (NAAJ 2025) benchmark four ML severity models inside the QPP framework. The tau adjustment is elegant. CatBoost was missing from the comparison. Our library already handles both."
published: true
---

Zanzouri, Kacem and Belkacem published a benchmark of ML-based quantile ratemaking in the North American Actuarial Journal this year (DOI: 10.1080/10920277.2025.2503744). They compare four severity models — Quantile Regression Forest, Gradient Boosting, XGBoost, and Neural Net — inside the Quantile Premium Principle framework that Heras, Moreno and Vilar-Zanon established in the *Scandinavian Actuarial Journal* in 2018. The headline result is that QRNN wins on the AutoClaims dataset.

The QPP framework itself is genuinely useful for pricing actuaries. The paper's benchmark is a serviceable literature addition. But CatBoost MultiQuantile — the strongest GBM candidate for this task — was not included. And [`insurance-quantile`](/insurance-quantile/) already implements the entire pipeline.

This post explains the tau adjustment (the clever part), shows what the library does with it, and notes where the paper's conclusions need hedging.

---

## The QPP formula

The Quantile Premium Principle solves a practical problem: given a target aggregate quantile level for a policy's total annual loss, what severity quantile should your model be asked to predict?

For a policy with no-claim probability `p_i = Pr(N_i = 0 | x_i)`, and a target aggregate quantile level `tau` (say, 0.90), the adjusted severity quantile is:

```
tau_i = (tau - p_i) / (1 - p_i)
```

This is valid only when `p_i < tau` — i.e., where the target aggregate level is not already met by the no-claim probability alone.

The intuition: if 80% of your motor own damage policies claim nothing, then to sit at the 90th percentile of the *aggregate* loss for those policies, your severity model only needs to produce the *50th percentile* of conditional severity. The no-claim probability already accounts for the lower 80% of the aggregate distribution. The severity quantile covers the remaining gap between `p_i` and `tau` — hence `(tau - p_i) / (1 - p_i)`.

For a UK motor OD book with `p_i = 0.80` and `tau = 0.90`:

```
tau_i = (0.90 - 0.80) / (1 - 0.80) = 0.50
```

You are asking the severity model for the median. That is a well-estimated quantity even with moderate data. Compare this to directly predicting the 90th percentile of aggregate loss — which requires precise tail estimation from sparser data.

For a higher-risk policy with `p_i = 0.60`, the same `tau = 0.90` gives `tau_i = 0.75`. You are asking for the 75th percentile of severity. The severity model is asked to work harder in proportion to how much tail risk is concentrated in severity rather than frequency. This is the correct risk-sensitive behaviour.

---

## The loaded premium

Once you have `tau_i` per policy, the QPP loaded premium is:

```
P_i = gamma * Q_{tau_i}(x_i) + (1 - gamma) * E[S_i | x_i]
```

where `gamma` is the safety loading weight (typically calibrated to the book's target loss ratio), `Q_{tau_i}(x_i)` is the severity quantile prediction, and `E[S_i | x_i]` is the expected aggregate loss (frequency × expected severity).

The per-policy safety loading is then:

```
Loading_i = gamma * (Q_{tau_i}(x_i) - E[S_i | x_i])
```

This loading is risk-specific. A policy with high frequency volatility and a heavy severity tail produces a larger loading than a policy with the same expected cost but better-behaved losses. The flat-scalar approach — "add 10% to technical premium" — produces neither of these. The QPP loading is directly defensible under the FCA's Consumer Duty Price and Value outcome, because you can trace every pound of loading back to the policy's own risk characteristics.

---

## What insurance-quantile provides

`TwoPartQuantilePremium` implements the full pipeline. The frequency model (any sklearn-compatible classifier returning `Pr(N = 0 | x)`) is passed at construction. The severity quantile model is a `QuantileGBM`, fitted on non-zero claims only.

```python
from insurance_quantile import QuantileGBM, TwoPartQuantilePremium

# Step 1: severity model on non-zero claims
severity_model = QuantileGBM(
    quantiles=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
    iterations=500,
    learning_rate=0.05,
    depth=6,
)
severity_model.fit(X_severity, y_severity)

# Step 2: two-part QPP model
qpp = TwoPartQuantilePremium(
    frequency_model=fitted_logistic,   # any classifier: predict_proba -> Pr(N=0)
    severity_model=severity_model,
    tau=0.90,                          # target aggregate quantile level
    gamma=0.30,                        # safety loading weight
)

# Step 3: per-policy premium and loading
result = qpp.predict_premium(X_pricing, mean_loss=expected_loss_tweedie)

print(result.to_frame().head())
# columns: adjusted_tau, severity_quantile, premium, safety_loading
```

The `adjusted_tau` column is `(tau - p_i) / (1 - p_i)` per policy. The `safety_loading` column is the additive supplement over expected loss. Both are directly auditable.

The library's severity interpolation is vectorised piecewise-linear — it interpolates across the trained quantile grid rather than fitting a separate model per quantile level. This is more rigorous than the paper's description of their implementation, which does not address the interpolation method explicitly.

---

## What the paper missed: CatBoost

Zanzouri et al. test QRF, QRGB, QRXGBoost, and QRNN. CatBoost with `MultiQuantile` loss was not included.

This matters for three reasons.

**Single model, all quantiles.** CatBoost MultiQuantile fits all quantile levels in one pass, with shared feature representations learned simultaneously across levels. XGBoost in quantile mode fits a separate model per quantile level — that is five training runs for five quantiles, with no information sharing across levels. For tabular insurance data, the joint representation in CatBoost generally produces more coherent quantile estimates.

**Quantile crossing.** Without monotonicity enforcement, GBM quantile models can produce crossings at inference: `q_0.9 < q_0.95` for individual risks. The paper does not mention whether crossing correction was applied to QRGB or QRXGBoost. Without it, the piecewise-linear interpolation at Step 3 of the QPP pipeline can produce non-monotone severity quantile estimates. This may have artificially degraded the gradient boosting results relative to QRNN, which has a natural ordering via the pinball loss objective. `QuantileGBM` applies isotonic regression correction by default.

**Data scale.** The AutoClaims dataset is 6,773 closed claims — modest by the standards of UK motor books, which routinely run to hundreds of thousands of policies. At this scale, QRNN is competing on its strongest ground: it has enough data to converge but not so much that the GBM's structural efficiency advantage dominates. On a UK book with 200,000+ non-zero severity observations, we would expect CatBoost to match or beat QRNN in pinball loss. The paper's headline result should not be read as a general endorsement of neural networks over gradient boosting for insurance severity.

---

## UK application

The QPP framework is well-suited to UK motor OD. Typical no-claim probabilities of 0.75–0.85 imply adjusted severity quantiles in the range 0.33–0.67 for a tau of 0.90 — body-of-distribution estimates that are stable even with moderate data.

For motor BI and liability, we recommend expectile mode (`use_expectile=True` in `QuantileGBM`) rather than quantile mode. Expectiles are coherent — they satisfy subadditivity, which matters for capital allocation and Solvency UK internal model work. The QPP quantile is elicitable (the pinball loss is a strictly proper scoring rule) but fails coherence. For heavy-tailed lines where the capital charge is driven by the tail beyond the 99th percentile, that distinction is not academic.

For extreme tail estimation — the 99th percentile and above, relevant for Solvency UK's 99.5% SCR calculation — `EQRNModel` (Pasche & Engelke 2024, *Annals of Applied Statistics*, DOI: 10.1214/24-AOAS1907) provides a GPD-based neural network tail model that can be plugged into the same `TwoPartQuantilePremium` pipeline.

One data note: the AutoClaims dataset includes `GENDER` as a feature. Under the Equality Act 2010 as amended by SI 2012/2992 — implementing the EU Gender Directive following the Test-Achats ruling (Case C-236/09) — gender cannot be used for insurance pricing. This is not a comment on the paper's methodology — the QPP framework itself is unaffected — but any UK implementation should exclude gender from the feature set from the outset.

---

## Limitations of the QPP framework

The formula `tau_i = (tau - p_i) / (1 - p_i)` assumes binary frequency: zero or one claim. For multi-claim policies — standard in UK commercial fleet, employer's liability, and trade credit — the aggregate distribution is compound, and the mapping from frequency parameters to an adjusted severity quantile requires either simulation or Panjer recursion. The paper does not address this. Neither does Heras (2018). It is an open gap.

The QPP also assumes the frequency and severity models are independent conditional on features. In practice, claim occurrence and claim severity can be positively correlated through unobserved risk factors — the same underlying recklessness that drives claim frequency may drive severity. The paper does not test for this. Any production implementation should run a check of residual correlation between frequency model errors and severity model errors on held-out data.

---

## What to take from the paper

Zanzouri et al. (NAAJ 2025) provide the first systematic head-to-head ML benchmark within the QPP framework. The prior literature (Heras 2018: linear QR; Laporta 2024: QRNN on health data) each tested one model family. The benchmark is a useful extension of the evidence base, even if CatBoost was absent and the dataset is small.

The tau adjustment formula is the core insight and it is correct. The implementation in `insurance-quantile` is production-ready — it handles the interpolation, the crossing correction, the loading decomposition, and the calibration diagnostics. If you are pricing UK motor own damage and want a risk-specific safety loading that you can defend to a sign-off committee and the FCA, QPP is the right framework.

```bash
uv add insurance-quantile
```

Full documentation at [`insurance-quantile`](/insurance-quantile/).

---

**Related:**
- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/) — the base severity modelling library this post extends
- [Conformal Prediction for Solvency II SCR Validation](/2026/03/26/conformal-prediction-for-solvency-ii-scr-validation/) — distribution-free coverage guarantees on quantile predictions for capital purposes
