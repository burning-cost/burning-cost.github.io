---
layout: post
title: "Uncertainty for Free: Prediction Intervals on Your GBM Pricing Model Without Retraining"
date: 2026-04-01
categories: [techniques, pricing]
tags: [conformal-prediction, GBT, GBM, XGBoost, LightGBM, CatBoost, prediction-intervals, uncertainty-quantification, heteroscedastic, Tweedie, local-conformal, LoBoostCP, insurance-conformal, arXiv-2602-22432, Santos, motor-pricing, python]
description: "LoBoostCP in insurance-conformal v1.0.0 implements Santos et al. arXiv:2602.22432 — local conformal prediction that uses the leaf structure of your existing GBT to calibrate prediction intervals. No retraining, no auxiliary model, and the intervals are heteroscedastic by construction because different leaf neighbourhoods have different calibration sets."
math: true
author: burning-cost
---

Every UK motor and property pricing team running a gradient boosted tree model faces the same gap: the model produces point predictions, but pricing decisions depend on uncertainty. How wide is the plausible range around a £2,400 expected loss estimate? Is the interval narrower for a well-represented risk profile and wider for a marginal one? Without that, you are presenting regulators and pricing committees with false precision.

The standard fixes all require extra work. Quantile regression needs a separate model (or three, for lower/median/upper). Jackknife+ needs to retain individual tree predictions from training. Standard split conformal prediction gives you one global threshold — marginal coverage, not local. None of these use the information already sitting in your fitted model.

`insurance-conformal` v1.0.0 ships `LoBoostCP`, which does.

---

## The idea: your trees already define neighbourhoods

A gradient boosted tree ensemble assigns every observation to a leaf in every tree. Two observations assigned to the same leaf in a given tree are, in that tree's view, interchangeable — they satisfy the same sequence of split conditions. Observations that share leaf assignments across *many* trees are genuinely similar across many learned feature interactions simultaneously.

Santos, Izbicki, Stern, and Saad-Roy (arXiv:2602.22432) formalise this. For a test point $x$, define its local calibration set as the subset of calibration observations that share at least a fraction $\rho$ of leaf assignments with $x$:

$$\mathcal{N}_\rho(x) = \left\{ i \in \mathcal{I}_{\text{cal}} : \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}\{\ell_t(x) = \ell_t(x_i)\} \geq \rho \right\}$$

where $T$ is the number of trees and $\ell_t(x)$ is the leaf index assigned to $x$ by tree $t$. Then calibrate a conformal threshold on $\mathcal{N}_\rho(x)$ rather than on the full calibration set.

The intervals are local because the calibration is local. If risk profiles in the neighbourhood of $x$ have small residuals, the interval is narrow. If they have large residuals — as they will for marginal risks near rating boundaries — the interval is wide. This is heteroscedasticity without fitting a variance model.

---

## Why overlap fraction, not exact leaf matching

The natural first instinct is exact matching: only calibration points that share *all* leaf assignments with $x$. This collapses immediately in practice.

With 200 trees in a LightGBM model and each tree having, say, 32 leaves, the probability that a random calibration point matches on all 200 trees is negligible. Even with 10,000 calibration observations you will routinely find zero exact matches for test points in sparsely populated feature regions — which are precisely the high-risk, marginal policies where intervals matter most.

Overlap fraction is the resolution. Setting `overlap_frac=0.5` (the default) means a calibration point qualifies for $\mathcal{N}_\rho(x)$ if it matches on at least half the trees. This is still a strong locality condition — matching on 100 of 200 trees means the observation has traversed the same feature-space partitions as $x$ along a substantial fraction of the learned splits — but it gives workable neighbourhood sizes. Santos et al. show in their paper that moderate $\rho$ values (0.4–0.6) give the best empirical tradeoff between locality and sample size.

---

## min_samples and the marginal coverage fallback

For truly unusual risk profiles — risks near the edge of the training distribution — the local neighbourhood may shrink below a useful size even at `overlap_frac=0.5`. `LoBoostCP` handles this via `min_samples` (default 30).

If $|\mathcal{N}_\rho(x)| < \text{min\_samples}$, the class falls back to the full global calibration set and issues a `UserWarning`. This is conservative — the interval reverts to a marginal conformal interval — but it preserves the marginal coverage guarantee. A warning for a specific test point tells you the model has genuinely limited local context for that risk; that is useful information in itself, not a failure.

The choice of 30 is motivated by finite-sample conformal theory. At $\alpha = 0.1$ (90% coverage), the finite-sample correction to the ideal quantile is roughly $\pm 1/\sqrt{n_{\text{cal}}}$, which at $n = 30$ gives approximately ±0.18. With fewer points, the quantile estimate is too noisy to improve meaningfully on the global calibration.

---

## Score types: absolute and normalised

Two nonconformity scores are available via `score_type`:

**Absolute**: $s_i = |y_i - \hat{y}_i|$. Standard choice for frequency models where residuals do not scale with the mean. Appropriate for Poisson claim count models.

**Normalised**: $s_i = |y_i - \hat{y}_i| / \hat{y}_i$. For Tweedie and Gamma severity models, residuals scale with the predicted mean. A high-severity risk with $\hat{y} = £10{,}000$ will have larger absolute residuals than a low-severity risk with $\hat{y} = £500$, even after good calibration. The normalised score accounts for this — equivalent to working in relative rather than absolute error space. The resulting intervals are wider in absolute terms for high-severity risks, which is correct.

---

## Backend support

`LoBoostCP` dispatches leaf retrieval to whichever GBT backend you are using:

| Backend | Call |
|---|---|
| CatBoost | `model.calc_leaf_indexes(data)` |
| XGBoost | `model.predict(data, pred_leaf=True)` |
| LightGBM | `model.predict(data, pred_leaf=True)` |
| sklearn `GradientBoostingRegressor` | `model.apply(data)` |

All return an `(n_samples, n_trees)` array of `int32` leaf indices. The dispatch is handled by `_get_leaf_indices()` and requires no changes to how you fitted the model — LoBoostCP wraps it.

---

## API

```python
from insurance_conformal import LoBoostCP

cp = LoBoostCP(model, score_type="normalized", overlap_frac=0.5)
cp.fit(X_cal, y_cal)
results = cp.predict(X_test, alpha=0.1)
```

`results` is a dict (or structured output) containing `lower`, `upper`, and `point_pred`. The `predict()` call takes one `alpha` argument — the miscoverage rate — so `alpha=0.1` gives 90% prediction intervals.

`fit()` extracts leaf assignments for the calibration set and stores them; `predict()` extracts leaf assignments for the test set, computes local neighbourhood sizes, calibrates per-point thresholds, and returns intervals.

For the Tweedie severity use case on a UK home model, the call we recommend is `score_type="normalized"` with `overlap_frac=0.5` and `min_samples=30`. For a frequency model, `score_type="absolute"` is correct. The `alpha` passed to `predict()` can be varied without re-running `fit()` — the calibration data is already stored.

---

## Memory and runtime

The inner loop in `predict()` is $O(n_{\text{test}} \times n_{\text{cal}} \times n_{\text{trees}})$ comparisons via numpy broadcasting. For a typical UK motor portfolio — say $n_{\text{cal}} = 10{,}000$, $n_{\text{trees}} = 200$, $n_{\text{test}} = 1{,}000$ — this is 2 billion operations, which runs in a few seconds on CPU with numpy vectorisation. For batch scoring of full portfolios ($n_{\text{test}} = 500{,}000$), you would chunk the test set.

This is not a real-time quote path tool. It is a validation, pricing review, and risk selection tool: run it offline on portfolios, renewal books, or model validation datasets.

---

## What this gives you in practice

For a UK personal motor pricing review, `LoBoostCP` answers questions that point predictions cannot:

- Which rating cells have wide intervals? Those are cells where the model is uncertain, often corresponding to thin data segments — young drivers, high-value vehicles, unusual occupation codes.
- For a renewal price change: what is the 90% interval around the predicted loss cost increase? If the lower bound implies a 3% increase and the upper bound implies a 12% increase, the renewal pricing decision is different from a case where the interval is £2,400 ± £150.
- For Solvency II model validation: demonstrating that prediction intervals achieve their nominal coverage across rating cells is a stronger validation claim than RMSE on a holdout set. The coverage report gives you exactly that evidence.

The intervals are locally calibrated, not globally. A global 90% coverage rate is consistent with systematic undercoverage in high-risk cells. Local calibration detects that.

---

## Relationship to other methods in insurance-conformal

`LoBoostCP` sits alongside the other calibration approaches in the library:

| Method | Locality mechanism | Requires retraining? |
|---|---|---|
| Split conformal | None — global threshold | No |
| Mondrian (group) | Pre-specified groups | No |
| ShapeAdaptiveCP (MOPI) | Minimax over rating cells | No |
| LoBoostCP | Leaf overlap from fitted GBT | No |

The key difference from ShapeAdaptiveCP is that LoBoostCP derives locality directly from the model's internal structure rather than from externally specified rating groups. If your GBT has learned that rural postcodes with high-value vehicles and young drivers form a coherent risk segment, LoBoostCP will find that neighbourhood automatically — you do not need to specify it. ShapeAdaptiveCP, by contrast, requires you to define the groups over which you want coverage guarantees.

For exploratory uncertainty analysis, LoBoostCP. For targeted coverage guarantees by regulatory or actuarial category (age band, vehicle class), ShapeAdaptiveCP.

---

## Getting it

`LoBoostCP` is exported from `insurance_conformal` in v1.0.0:

```python
from insurance_conformal import LoBoostCP
```

Source: `src/insurance_conformal/loboost.py`. Tests: `tests/test_loboost.py` (21 tests). The locality definition and theoretical guarantees are in Santos, Izbicki, Stern, and Saad-Roy (arXiv:2602.22432); the build decisions are documented in our KB entries 5184 and 5185.

---

## The paper

Santos, Bernardo Augusto Rodrigues dos, Rafael Izbicki, Rafael Bassi Stern, and Ryan Saad-Roy. "LoBoostCP: Local Conformal Prediction via Gradient Boosting." arXiv:2602.22432 [stat.ML]. February 2026.

---

## Related posts

- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/2026/04/01/shape-adaptive-conformal-prediction/) — MOPI calibration for group-conditional coverage with masked protected characteristics
- [CRPS-Optimal Conformal Binning: When Actuarial Scoring Drives the Interval](/2026/04/01/crps-optimal-conformal-binning-prediction-intervals-insurance/) — calibrating bin thresholds to minimise CRPS rather than achieve coverage
- [Conformal Prediction for Insurance Pricing: Intervals, Risk Control, and the Practical Toolkit](/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — when conformal prediction works, when it does not, and what calibration data you actually need
- [Conditional Coverage in Conformal Prediction: Model Selection with CVI](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — checking whether your conformal guarantees hold conditionally, not just marginally
- [Conformal Prediction vs Bootstrap Intervals for Insurance Pricing](/2026/03/26/conformal-prediction-vs-bootstrap-intervals-insurance-pricing/) — a direct comparison with the main alternative
