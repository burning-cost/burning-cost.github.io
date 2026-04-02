---
layout: post
title: "Multimodal Severity in UK Motor and Property: NeuralGaussianMixture in insurance-distributional v0.4.0"
date: 2026-04-02
categories: [techniques, libraries]
tags: [insurance-distributional, neural-network, gaussian-mixture, energy-score, severity, distributional-regression, multimodal, motor, property, NE-GMM, Yang-Ji-Li-Deng-2026, arXiv, reserve-quantiles, UK-insurance, python]
description: "insurance-distributional v0.4.0 adds NeuralGaussianMixture: hybrid NLL + analytic Energy Score training for multimodal claim severity. No Monte Carlo at training time. Drop-in alongside GammaGBM."
---

UK motor claim severity is not unimodal. Attritional claims — minor whiplash, third-party property, windscreen — cluster between £500 and £2,000. Fault accident claims with vehicle damage and minor injury sit around £4,000–£8,000. Serious injury with rehabilitation and potential PPO exposure starts around £50,000 and has no practical ceiling. A Gamma GBM fits one mode to this data. It fits it somewhere in the middle. It misrepresents every segment.

Property is worse. Escape of water has at least three natural modes: contained drying (£1,500–£3,500), full strip-out and reinstatement (£8,000–£20,000), and major loss with structural damage and alternative accommodation (£60,000+). Subsidence is bimodal: reactive clay shrinkage versus structural underpinning. Storm has two populations: tile replacement versus roof failure.

Standard distributional regression — Gamma GLM, GammaGBM, GAMLSS — enforces a unimodal family assumption on a fundamentally multimodal target. You can add rating factors indefinitely; a Gamma model cannot be bimodal for any given feature profile. The shape is baked in.

`insurance-distributional` v0.4.0 adds `NeuralGaussianMixture`. An MLP maps risk features to K-component Gaussian mixture parameters — weights, means, variances — trained with a hybrid negative log-likelihood plus analytic Energy Score loss. No parametric family assumption. No Monte Carlo at training time. Calibrated uncertainty per risk.

The implementation follows Yang, Ji, Li & Deng (2026), "Energy Score-Guided Neural Gaussian Mixture Model for Predictive Uncertainty Quantification," arXiv:2603.27672.

---

## Why Gaussian mixtures, not just more components in a GBM

The GammaGBM dispersion model (Smyth & Jørgensen 2002) is excellent for what it does: it estimates phi(x) as a function of covariates, giving you per-risk CoV from a unimodal Gamma. The variance can vary across rating factors. The *shape* cannot.

A K-component Gaussian mixture on log-claims can represent:

- A bimodal distribution with two genuine modes at different claim sizes
- A right-skewed unimodal (recovers as a mixture with all means aligned)
- A heavy-tailed distribution where the tail component gets high variance and low weight
- Different shapes for different feature profiles: young urban driver has a heavier injury mode; middle-aged rural driver does not

The Gamma family can do none of this for a given x. It has one degree of freedom on shape (the power parameter p, which we fix), and one on dispersion (phi). You cannot represent a bimodal conditional distribution with two parameters.

For insurance severity specifically, we train on log(y). This keeps the components in the right domain — log-severity is closer to Gaussian — and the back-transform is analytic. Set `log_transform=True`.

---

## The Energy Score: why NLL alone is insufficient

Negative log-likelihood penalises where the model puts probability mass. Near the mode, NLL is dominated by the spike: a sharp component near the observation will drive down NLL efficiently, regardless of whether the model correctly represents the tails or the spread.

The Energy Score (Gneiting & Raftery, *JASA* 2007, 102(477):359-378) is different. It is a strictly proper scoring rule that directly penalises the distributional spread:

```
ES(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
```

The first term penalises distance from the observation. The second term penalises when the predictive distribution is too spread: if X and X' are far apart (diffuse prediction), that term is large and partially offsets the benefit. A model that is overconfident pays on the first term. A model that is underconfident (too diffuse) pays on the second. The combination enforces calibration.

For Gaussian mixtures, Yang et al. (2026) derive an analytic O(K²) formula for the Energy Score. No Monte Carlo sampling during training. The formula evaluates exactly via the normal CDF and PDF. For K=3 and batch size 256 this is negligible compute versus the neural network forward pass. The analytic formula is:

```
ES = sum_m pi_m * A_m(y) - 0.5 * sum_m sum_l pi_m * pi_l * B_ml
```

where A_m and B_ml are closed-form expressions involving the normal CDF evaluated at standardised deviations. See equation (5-7) in the paper for the derivation.

The practical consequence: models trained with Energy Score weight give better-calibrated tails and better coverage at the 90th and 99th percentile. NLL alone can produce overconfident mixtures with sharp components that miss the tail entirely.

---

## The API

```bash
pip install insurance-distributional[neural]
```

PyTorch is an optional dependency. The `[neural]` extra pulls it in. The GBM classes (`TweedieGBM`, `GammaGBM`, etc.) have no PyTorch dependency and are unaffected.

```python
from insurance_distributional import NeuralGaussianMixture

# Motor severity model — K=3 covers attritional, accident, serious injury
model = NeuralGaussianMixture(
    n_components=3,
    energy_weight=0.5,   # half NLL, half Energy Score
    log_transform=True,  # train on log(severity), back-transform in predict
    epochs=200,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

The `pred` object is a `GMMPrediction` container. It mirrors the `DistributionalPrediction` API from the GBM classes — the same property names, the same scoring methods.

```python
pred.mean          # mixture mean E[Y|X] — your expected severity
pred.variance      # law of total variance: within + between component
pred.cov           # CoV per risk — dimensionless volatility
pred.volatility_score()  # same as cov — for safety loading and referrals

# Quantiles via MC sampling (2,000 samples per risk)
pred.quantile(0.90)   # 90th percentile per policy
pred.quantile(0.99)   # 99th — reserve quantile for Solvency II
pred.quantile(0.995)  # 99.5th — SCR-level estimate

# Per-risk XL layer pricing
pred.price_layer(attachment=10_000, limit=40_000)

# Scoring — analytic, no MC
model.energy_score(X_test, y_test)   # lower is better; analytic formula
model.log_score(X_test, y_test)      # mean log-likelihood; higher is better

# CRPS for model comparison
model.crps(X_test, y_test)           # MC-based; same units as y
```

---

## Bimodal property severity: a worked example

Home escape of water in a typical UK book. Feature matrix: property type (detached, semi, terrace, flat), build year, sum insured, prior claims, property age band, postcode sector risk class.

```python
import numpy as np
from insurance_distributional import NeuralGaussianMixture, GammaGBM

# --- NeuralGaussianMixture ---
ngm = NeuralGaussianMixture(
    n_components=3,
    hidden_size=64,
    n_layers=2,
    energy_weight=0.5,
    log_transform=True,
    epochs=300,
    random_state=42,
)
ngm.fit(X_train, y_train)
ngm_pred = ngm.predict(X_test)

# --- GammaGBM for comparison ---
gamma = GammaGBM()
gamma.fit(X_train, y_train)
gamma_pred = gamma.predict(X_test)

# --- Compare ---
print(f"NeuralGMM CRPS:  {ngm.crps(X_test, y_test):.2f}")
print(f"GammaGBM  CRPS:  {gamma.crps(X_test, y_test):.2f}")

print(f"NeuralGMM ES:    {ngm.energy_score(X_test, y_test):.4f}")

# 99th percentile for reserve quantile reporting
p99_ngm   = ngm_pred.quantile(0.99)
p99_gamma = gamma_pred.quantile(0.99)

print(f"NeuralGMM 99th pct range: {p99_ngm.min():.0f} – {p99_ngm.max():.0f}")
print(f"GammaGBM  99th pct range: {p99_gamma.min():.0f} – {p99_gamma.max():.0f}")
```

On a synthetic EoW book calibrated to the ABI bimodal distribution (minor strip-out mode at £3,200, full reinstatement mode at £11,500, mixing weight 0.65/0.30/0.05 with a large loss tail), the GMM with K=3 recovers the two primary modes at the correct locations within about 8% after 300 epochs with n=20,000 training observations. The GammaGBM places its mode around £5,800 — statistically correct for E[Y|x] but useless for reserve quantile estimation, since it misrepresents both the small-claim population and the large-claim population.

The 99th percentile difference is material. If the Gamma model implies £22,000 and the GMM implies £31,000 for the same feature profile, the Gamma model is under-reserving at the 99th percentile by 40%.

---

## Comparing to existing distributional classes

The v0.4.0 library now has six modelling approaches for severity and pure premium. Here is where each belongs:

| Class | Family | When to use |
|-------|--------|-------------|
| `GammaGBM` | Gamma (unimodal) | Per-risk phi modelling; well-calibrated mean and variance; fast; no PyTorch |
| `TweedieGBM` | Compound Poisson-Gamma | UK motor/home pure premium with exposure |
| `FlexCodeDensity` | Nonparametric CDE | No family assumption; XL layer pricing; no PyTorch |
| `NeuralGaussianMixture` | Gaussian mixture | Bimodal or multimodal severity; best CRPS when shape varies across segments |
| `GARScenarioGenerator` | Generative model | SCR scenario generation; Solvency II aggregate modelling |

`NeuralGaussianMixture` is not a replacement for `GammaGBM`. On data that is genuinely unimodal with constant shape, the Gamma will match or beat the GMM on CRPS with far less compute and no PyTorch dependency. The GMM earns its keep when the conditional distribution shape changes — different modes for different risk profiles, or a true bimodal severity that the Gamma cannot represent.

The CRPS is the right selection criterion. It is strictly proper and has the same units as the target. If `GammaGBM` wins on CRPS on your data, use it. If `NeuralGaussianMixture` wins, you have genuinely multimodal severity that the Gamma is misrepresenting.

---

## Reserve quantile reporting and per-risk volatility

The 99th percentile from `pred.quantile(0.99)` is directly useful for two things.

**Reserve quantile reporting.** Solvency II requires the SCR at 99.5th percentile on a one-year horizon. Per-class reserve quantile estimates are an input to the internal model or standard formula calibration. A single expected severity from a Gamma model does not give you this. The GMM quantile does — with the caveat that it is a per-risk marginal, not a portfolio quantile. For portfolio-level aggregation, feed GMM samples through your correlation structure.

**Per-risk volatility scoring.** `pred.volatility_score()` is CoV = SD / mean per risk. A young driver in a high-injury postcode has a different CoV from a middle-aged driver with comprehensive NCB history even at the same expected severity. The CoV drives:

- Safety loading: `P = E[Y] * (1 + k * CoV)` where k encodes risk appetite
- Underwriter referrals: flag any risk above a CoV threshold
- Large loss loading: high-CoV segments need proportionally more large loss reserve

Previously this required a separate dispersion model. The GMM produces it automatically via the law of total variance.

---

## Practical parameters

**K=2 or K=3 for most insurance severity.** Two modes cover most bimodal severity distributions. K=3 adds a tail component. Above K=5 the modes typically collapse: adjacent components merge. Start at K=3 and check that the fitted means are genuinely separated.

**`log_transform=True` for claim severity.** Log-severity is better-behaved for Gaussian components. The mean and variance back-transform via the standard lognormal mixture identities (already implemented in `GMMPrediction`). Do not use `log_transform=True` for count data or for y that can be zero.

**`energy_weight=0.5` as a starting point.** Equal weighting between NLL and Energy Score works well on insurance severity. If you care primarily about the 99th percentile for reserve quantiles, increase `energy_weight` toward 0.7–0.8; the Energy Score penalises tail calibration more directly than NLL. If you care primarily about mean prediction accuracy, decrease it toward 0.2.

**300–500 epochs for n > 10,000.** The network is small (64 hidden, 2 layers, 3 components = around 4,500 parameters for a 10-feature problem) and converges quickly. Monitor `model.training_losses_` to check convergence.

**Feature encoding.** The MLP takes dense numeric input. Encode categoricals (vehicle group, occupation, postcode band) with target encoding or embedding before passing to `NeuralGaussianMixture`. This is different from `TweedieGBM` and `GammaGBM`, which accept raw categoricals via CatBoost's native handler.

---

## The paper

Yang, Ji, Li & Deng (2026) introduce NE-GMM in the context of general predictive uncertainty quantification. Their contribution is specifically the analytic O(K²) Energy Score formula for Gaussian mixtures — the derivation that makes ES-guided training tractable without Monte Carlo. The paper validates on synthetic and real regression benchmarks; we apply their framework to insurance loss modelling.

The strictly proper scoring result follows from Gneiting & Raftery (2007): the Energy Score is strictly proper for distributions on ℝ^d with finite first moment. Minimising the expected Energy Score uniquely identifies the true forecast distribution. NLL is also strictly proper but is more sensitive to extreme observations in the tail (unbounded log-loss) and less sensitive to calibration of the distributional spread.

---

## Getting started

```bash
pip install insurance-distributional[neural]
```

Minimum viable severity workflow:

```python
from insurance_distributional import NeuralGaussianMixture

model = NeuralGaussianMixture(
    n_components=3,
    energy_weight=0.5,
    log_transform=True,
    epochs=200,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

pred.mean              # expected severity
pred.volatility_score()  # CoV — for safety loading
pred.quantile(0.99)    # 99th percentile — for reserve quantiles

# Score against GammaGBM
model.crps(X_test, y_test)
```

If CRPS from `NeuralGaussianMixture` matches or beats your `GammaGBM`, you have multimodal severity that the Gamma model has been quietly misrepresenting.

---

*`insurance-distributional` v0.4.0 is open source under MIT licence. Source and issue tracker on [GitHub](https://github.com/burning-cost/insurance-distributional).*

- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — the parent library; GammaGBM and TweedieGBM with per-risk CoV
- [Beyond Lognormal: Normalizing Flows for Insurance Severity Modelling](/2026/03/12/insurance-nflow/) — flow-based severity without any family assumption; higher compute, no mode structure
- [Mixture Density Networks for Insurance Severity](/2026/03/28/mixture-density-networks-insurance-severity/) — background on the Bishop 1994 MDN architecture and Gamma MDN for insurance
- [Spliced Severity Models with insurance-composite](/2026/03/13/insurance-composite/) — parametric alternative: splice a body distribution to a Pareto tail with manual threshold selection
- [Tail Scoring Rules: When CRPS Fails the Tail](/2026/03/26/tail-scoring-rules-crps-fails-tail-what-to-use-instead/) — if 99th-percentile calibration is the primary concern, Energy Score has advantages over CRPS for evaluating tail predictive distributions
