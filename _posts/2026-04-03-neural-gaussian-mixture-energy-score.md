---
layout: post
title: "Validating a Mixture Severity Model: When NE-GMM Earns Its Keep and When GammaGBM Still Wins"
date: 2026-04-03
categories: [techniques, pricing, validation]
tags: [NE-GMM, gaussian-mixture, energy-score, severity-modelling, model-validation, CRPS, PIT, GammaGBM, insurance-distributional, arXiv-2603.27672, motor-insurance, bodily-injury, UK-insurance, distributional-regression, proper-scoring-rules, layer-pricing, IFRS17, reinsurance, python, Yang-Ji-Li-Deng-2026]
description: "NeuralGaussianMixture is now in insurance-distributional v0.4.0. The question is not whether it can fit bimodal severity — it can. The question is whether your data actually needs it. Here is the validation workflow to find out."
author: burning-cost
math: true
---

We released `NeuralGaussianMixture` in `insurance-distributional` v0.4.0, and [we have explained why bimodal severity matters for UK motor BI quantile pricing](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) and [what the NE-GMM paper actually contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/). That is the motivation. This post is about what you do after you install it.

The question a pricing team faces is not whether NeuralGaussianMixture is theoretically superior to GammaGBM on bimodal data — it is. The question is whether their specific severity dataset is bimodal enough, and their sample large enough, for the additional model complexity to produce better out-of-sample predictions. The answer is not always yes.

Running NE-GMM without first checking whether your severity data warrants it will cost you compute, a PyTorch dependency, and potentially a harder-to-explain model — for no improvement in predictive accuracy. This post walks through the validation workflow for deciding whether to use it, how to evaluate it once you do, and how to interpret the outputs in a regulatory context.

---

## Step 1: Check whether your severity is actually multimodal

Before fitting anything, look at the empirical marginal distribution of your severity data. Not the model-predicted distribution — the raw histogram of claim amounts, on a log scale.

```python
import numpy as np
import matplotlib.pyplot as plt

# y_sev: array of positive claim amounts, after removing zeros
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Linear scale — heavy right skew will compress everything into one bar
axes[0].hist(y_sev, bins=100)
axes[0].set_title("Severity (linear scale)")

# Log scale — bimodal structure visible if present
axes[1].hist(np.log(y_sev), bins=80)
axes[1].set_title("log(Severity) — look for two humps")

plt.tight_layout()
```

For UK motor bodily injury, the log-severity histogram will show a bimodal pattern: a peak around log(1,500) = 7.3 (attritional soft tissue) and a secondary peak around log(50,000) = 10.8 (serious injury). The trough between them, around log(8,000) = 8.9, is visually clear.

If your log-severity histogram shows a single clean peak with no secondary mode, stop here. Your severity is unimodal. `GammaGBM` with dispersion modelling is the right tool. `NeuralGaussianMixture` will fit something — it will not fit something better.

---

## Step 2: Fit both models and compare on CRPS

If the histogram shows bimodal structure, fit both models on your training set and evaluate on a held-out test set. The right comparison metric is CRPS.

```python
from insurance_distributional import NeuralGaussianMixture, GammaGBM

# Fit GammaGBM — standard distributional baseline
gamma = GammaGBM(n_estimators=500)
gamma.fit(X_sev_train, y_sev_train)

# Fit NeuralGaussianMixture — three components for motor BI
ngm = NeuralGaussianMixture(
    n_components=3,       # attritional / fault accident / serious injury
    energy_weight=0.5,    # paper's recommended default
    log_transform=True,   # essential for claim severity spanning 3 orders of magnitude
    epochs=300,
    random_state=42,
)
ngm.fit(X_sev_train, y_sev_train)

# Compare on held-out test set
gamma_crps = gamma.crps(X_sev_test, y_sev_test)
ngm_crps   = ngm.crps(X_sev_test, y_sev_test)

print(f"GammaGBM CRPS:         £{gamma_crps:.2f}")
print(f"NeuralGaussianMixture CRPS: £{ngm_crps:.2f}")
print(f"Improvement:               {100*(gamma_crps - ngm_crps)/gamma_crps:.1f}%")
```

Decision rule: if `NeuralGaussianMixture` improves CRPS by more than 3–5%, the multimodal structure in your data is large enough to be commercially significant. Less than that, and the Gamma model's parsimony wins.

Why CRPS and not mean absolute error? CRPS is strictly proper: it penalises both miscalibrated means and misrepresented distributional shape. A model that gets the mean right but misrepresents the bimodal structure — a Gamma model fitted to bimodal data — loses on CRPS, even though it would look fine on MAE. CRPS is the right yardstick for a severity model used for quantile pricing and XL layer calculations.

---

## Step 3: Check the Energy Score as a calibration diagnostic

CRPS evaluates the entire predictive distribution. The Energy Score (ES) specifically checks whether the model's distributional forecasts are calibrated — not just accurate in mean. A model with good CRPS but poor ES is getting the right answer for the wrong reasons.

```python
# Energy Score: analytic for NeuralGaussianMixture, MC-based for GammaGBM
ngm_es = ngm.energy_score(X_sev_test, y_sev_test)
print(f"NeuralGaussianMixture ES: {ngm_es:.4f}")
```

The absolute value of the Energy Score is less informative than its trend during training. If you have access to a validation curve during training:

```python
ngm_verbose = NeuralGaussianMixture(
    n_components=3,
    energy_weight=0.5,
    log_transform=True,
    epochs=300,
    verbose=True,  # prints NLL and ES per 50 epochs
    random_state=42,
)
ngm_verbose.fit(X_sev_train, y_sev_train)
```

Two warning signs to look for in the training output:

**NLL decreasing but ES flat or increasing.** The model is trading calibration for sharpness. It is learning to be confident about the wrong distribution. Increase `energy_weight` toward 0.7–0.8 and refit.

**ES decreasing but NLL rising sharply.** The model is prioritising calibration at the cost of predictive accuracy. Decrease `energy_weight` toward 0.3 and refit.

The 0.5 default holds for most severity datasets, but motor BI with a very small serious injury sub-population (below 8%) sometimes needs a higher `energy_weight` because the serious injury component is harder to separate and the NLL pushes the model toward collapsing it into the attritional mode.

---

## Step 4: PIT histogram — checking calibration visually

The probability integral transform (PIT) histogram is the sharpest visual check for distributional calibration. A well-calibrated model produces a uniform PIT histogram. Systematic humps or dips reveal calibration failures.

```python
# Compute PIT values: P(Y_pred <= y_obs) for each test observation
# For a mixture model, use Monte Carlo CDF evaluation

n_mc = 10_000
pit_values = np.zeros(len(y_sev_test))

for i in range(len(y_sev_test)):
    pred_i = ngm.predict(X_sev_test[i:i+1])
    # Sample from predictive distribution for this risk
    samples = pred_i.sample(n_mc, seed=42)[0]
    pit_values[i] = (samples <= y_sev_test[i]).mean()

plt.figure(figsize=(8, 4))
plt.hist(pit_values, bins=30, density=True, alpha=0.7)
plt.axhline(1.0, color='red', linestyle='--', label='Uniform (ideal)')
plt.xlabel("PIT value")
plt.ylabel("Density")
plt.title("PIT histogram — NeuralGaussianMixture severity")
plt.legend()
```

Common failure patterns and their causes:

**U-shaped PIT (humps at 0 and 1).** The model is overconfident — predictive distributions are too narrow. Increase `n_components` or `energy_weight`. More likely, your feature engineering is missing important covariates that drive severity.

**Hump at 0.5.** The model is underdispersed in the middle of the distribution — too much mass around the attritional mode. Often fixed by enabling `log_transform=True` if you have not already.

**Right-side hump (excess mass near PIT=1).** The model underestimates large losses. The serious injury tail component has too low a weight or too low a variance. Increase `n_components` to 4 or 5 if you have enough serious injury observations.

A flat PIT histogram is not proof the model is right — it is evidence it is not systematically wrong in the ways the PIT can detect. Pair it with CRPS.

---

## Step 5: Component diagnostics — check the mixture is not collapsed

Even with Energy Score training, it is worth checking that the model has actually learned a genuine multimodal structure rather than a near-unimodal mixture. This is the post-fit mode collapse check.

```python
pred = ngm.predict(X_sev_test)

# Component weights: should not have one component dominating
# pred.weights: (n_test, K)
mean_weights = pred.weights.mean(axis=0)
print(f"Mean component weights: {mean_weights}")
# Healthy: e.g. [0.78, 0.14, 0.08]
# Collapsed: e.g. [0.94, 0.04, 0.02]

# Component means: should span the bimodal range
# pred.means: (n_test, K) — sorted by mean ascending in GMMPrediction
mean_of_means = pred.means.mean(axis=0)
print(f"Mean component means (£): {np.exp(mean_of_means):.0f}")
# Healthy: [~£1,500, ~£8,000, ~£60,000] for motor BI
# Collapsed: [£12,000, £13,000, £13,500] — all components piled near the mean
```

If the mean weights are something like [0.94, 0.04, 0.02], the model has collapsed despite the Energy Score term. This can happen when `energy_weight` is too low (below 0.3) or when the serious injury sub-population is very small — fewer than 5% of training observations — and the model cannot cleanly separate it. In that case:

1. Increase `energy_weight` to 0.7
2. Oversample the serious injury sub-population during training
3. If neither works, accept that `GammaGBM` is the right tool for this dataset — the bimodal structure is not cleanly identifiable from the available data

---

## The XL pricing case: where the model difference is commercially visible

The validation steps above are about model quality in abstract. The concrete commercial case for getting this right is XL reinsurance layer pricing.

```python
# Expected loss in a £200k xs £50k XL layer — per risk
pred = ngm.predict(X_sev_test)
xl_el_ngm = pred.price_layer(attachment=50_000, limit=200_000)

pred_gamma = gamma.predict(X_sev_test)
xl_el_gamma = pred_gamma.price_layer(attachment=50_000, limit=200_000)

# Ratio: NE-GMM / GammaGBM
ratio = xl_el_ngm / xl_el_gamma

print(f"Median XL EL ratio (NE-GMM / GammaGBM): {np.median(ratio):.2f}")
print(f"90th pct XL EL ratio:                    {np.quantile(ratio, 0.9):.2f}")
```

For genuinely bimodal UK motor BI severity, the median ratio will be around 1.3–1.5. The GammaGBM is underestimating expected layer loss by 30–50% across the book. The 90th percentile ratio is higher still: for risks where the model assigns higher weight to the serious injury component, the discrepancy compounds.

This is not a model sensitivity exercise. The reinsurer's actuary is computing the same calculation with their own severity view. If your cedant model underestimates layer EL and the reinsurer's model does not, you will disagree systematically on rate adequacy — and the reinsurer will be right.

---

## Regulatory interpretation: IFRS 17 and Solvency II

`NeuralGaussianMixture` outputs per-risk distributional forecasts. For regulatory purposes, the relevant outputs are:

**IFRS 17 risk adjustment.** The risk adjustment under the general measurement model requires the insurer to hold capital reflecting the uncertainty in the fulfilment cash flows — specifically, the distribution of possible outcomes around the central estimate. `pred.variance` gives you the per-risk conditional variance from the mixture. For an IFRS 17 confidence-level approach, `pred.quantile(0.75)` gives you the 75th percentile reserve estimate per risk.

**Solvency II per-risk quantile for internal models.** Firms using internal models for reserve risk or premium risk will need per-risk 99th percentile severity estimates. `pred.quantile(0.99)` provides this. For the one-year VaR interpretation, these per-risk estimates feed into the aggregation layer — they are not portfolio numbers.

**Note on PRA SS1/23 model risk principles.** PRA SS1/23 (model risk management) principles apply to internal models. A `NeuralGaussianMixture` component embedded in a severity module of an internal model will need to be documented, validated, and tested for robustness to input changes. The validation workflow above — CRPS comparison, PIT histogram, component diagnostics — constitutes the quantitative validation evidence the PRA expects under SS1/23 principles. Governance documentation should record the decision to use a mixture model (bimodal distribution evidenced by histogram and CRPS test), the component interpretation (attritional / fault accident / serious injury), and the tuning choices made for `energy_weight` and `n_components`.

---

## When not to use it

We want to be direct about the cases where `GammaGBM` remains the right choice.

**Unimodal severity.** Windscreen claims, minor property claims below a contents deductible, vehicle recovery charges — these are narrow, right-skewed distributions that a Gamma handles well. There is no bimodal structure to capture.

**Small datasets.** Below roughly 10,000 severity observations, the neural network has limited signal to separate mixture components from noise. The Gamma model's parametric assumptions become an asset: they impose structure when data is sparse. On 3,000 observations, `GammaGBM` will almost always beat `NeuralGaussianMixture` on CRPS.

**Frequency-only pricing.** If your ratemaking goal is pure premium via frequency × severity and you are pricing to the mean, the Gamma model's mean is accurate even when bimodal — the mean of a mixture is the weighted mean of the component means. The bimodal model only earns its keep when the distributional shape beyond the mean matters: quantile pricing, layer pricing, risk adjustment.

**Infrastructure constraints.** PyTorch is a new dependency if your current stack is scikit-learn and XGBoost. The `[neural]` extra is opt-in and does not affect GBM classes. But deploying a PyTorch model to production infrastructure requires a different environment and carries additional serialisation and version management overhead. If your team does not have this in place, the operational cost may outweigh the modelling gain for anything short of a material reinsurance renegotiation.

---

## Summary decision flowchart

```
Is your severity log-histogram visibly bimodal?
  No  → Use GammaGBM. Done.
  Yes ↓

Do you have ≥ 10,000 severity observations?
  No  → Use GammaGBM. The neural model will not outperform on sparse data.
  Yes ↓

Does your use case require quantile estimates, XL layer pricing,
IFRS 17 risk adjustment, or Solvency II per-risk VaR?
  No  → Use GammaGBM. Bimodality does not affect the mean.
  Yes ↓

Fit NeuralGaussianMixture(n_components=3, energy_weight=0.5, log_transform=True).
Compare CRPS on held-out test set.

  NE-GMM CRPS > 5% better  → Use NeuralGaussianMixture. Material improvement.
  NE-GMM CRPS 2–5% better  → Judgement call. Consider operational overhead.
  NE-GMM CRPS < 2% better  → Use GammaGBM. Bimodal structure is not identifiable
                               from your data with current features.
```

---

## Reference

Yang, Ji, Li & Deng (2026). "Energy Score-Guided Neural Gaussian Mixture Model for Predictive Uncertainty Quantification." arXiv:2603.27672.

```bash
pip install "insurance-distributional[neural]>=0.4.0"
```

---

*Related posts:*
- [Two-Hump Distributions: Why Your Severity Model Gets the 95th Percentile Wrong](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) — the case for bimodal severity modelling on UK motor BI data
- [Energy Score-Guided Mixture Models: What the NE-GMM Paper Actually Contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/) — the paper assessment: what is new, what the ablations show, where the gaps are
- [Multimodal Severity: NeuralGaussianMixture in insurance-distributional v0.4.0](/2026/04/02/neuralgaussianmixture-insurance-distributional-v040/) — the full API reference and worked property example
- [Tail Scoring Rules: When CRPS Fails the Tail](/2026/03/26/tail-scoring-rules-crps-fails-tail-what-to-use-instead/) — why Energy Score sometimes beats CRPS for evaluating tail distributional predictions
- [Reserve Models Have a Calibration Problem Nobody Is Testing](/2026/04/03/score-decomposition-model-validation/) — score decomposition tests with HAC-robust p-values for quantile reserve model validation
