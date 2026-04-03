---
layout: post
title: "NeuralGaussianMixture vs the insurance-distributional Stack: When to Use What"
date: 2026-04-03
categories: [techniques, comparisons, pricing]
tags: [NE-GMM, gaussian-mixture, energy-score, severity-modelling, insurance-distributional, GammaGBM, TweedieGBM, ZIPGBM, FlexCodeDensity, distributional-regression, motor-insurance, property-insurance, bodily-injury, XL-reinsurance, IFRS17, production-readiness, arXiv-2603.27672, Yang-Ji-Li-Deng-2026, UK-insurance, python]
description: "insurance-distributional now has five distributional model classes. NeuralGaussianMixture is the newest and the most demanding. A routing guide: which model for which problem, and what it actually takes to put NeuralGaussianMixture in production."
author: burning-cost
---

`insurance-distributional` v0.4.0 now ships five model classes: `GammaGBM`, `TweedieGBM`, `ZIPGBM`, `FlexCodeDensity`, and the new `NeuralGaussianMixture`. That range prompts an obvious question from a pricing team that has just installed the library: which one?

The short version is that `NeuralGaussianMixture` is the right choice for a specific, narrow situation — and the wrong choice for most situations. Understanding where that line falls is more useful than knowing the method in detail.

We have already written about [why UK motor BI severity is structurally bimodal](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/), [what the NE-GMM paper actually contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/), [why NLL training fails for mixture models](/2026/04/03/ne-gmm-loss-function-why-nll-fails-mixture-training/), and [how to validate the model once fitted](/2026/04/03/neural-gaussian-mixture-energy-score/). This post is the one that sits before all of those: which distributional model should you reach for, and why?

---

## What the five classes actually do

**`TweedieGBM`** is the general-purpose pure premium model. It handles the combined frequency-severity process in one shot via the Tweedie compound Poisson family. You give it exposure-weighted policies and total claims as the target. It is the starting point for any new pricing exercise and the right default for overall pure premium estimation.

**`GammaGBM`** is the severity-only model for positive losses. You filter to claims-only records, give it claim amounts, and it estimates a conditional Gamma distribution — mean and dispersion — per risk. The Gamma family is unimodal by construction. It is the correct choice when your severity is a single right-skewed hump. This describes the majority of attritional insurance lines: windscreen, minor property, vehicle recovery, short-duration health.

**`ZIPGBM`** handles zero-inflated counts. It is not a severity model; it models claim frequency when there is an excess of zero-claim policies beyond what a standard Poisson or negative binomial predicts. Use this when your claim count data has a structural zero class — telematics policies with long zero-mileage periods, fleet policies where individual vehicles may generate no claims in a policy year, lapse-heavy SME liability books.

**`FlexCodeDensity`** is the non-parametric distributional model. It estimates the full conditional density f(y|x) without assuming any parametric family. It makes no shape assumptions and can represent multimodality, heavy tails, or any distributional feature. The cost is interpretability — you get a density curve, not a parametric object with named components — and a larger computational footprint per prediction.

**`NeuralGaussianMixture`** is the mixture model. An MLP maps risk features to K Gaussian mixture parameters — weights, means, variances — trained with the hybrid NLL + analytic Energy Score loss from Yang et al. (arXiv:2603.27672). It sits between `GammaGBM` (too constrained to represent multimodality) and `FlexCodeDensity` (flexible but non-parametric). The mixture structure makes the components interpretable: for UK motor BI, component 0 is the attritional population, component 2 is serious injury. The Energy Score training prevents mode collapse, which is the failure mode that makes standard MDN training impractical on insurance data.

---

## The routing logic

The decision is sequential. Answer each question in order and stop when you have a routing.

**Is your target a pure premium (combined frequency and severity)?**

Use `TweedieGBM`. This is not a severity model; it models the combined process. You are not in this decision tree.

**Is your target a count with structural excess zeros?**

Use `ZIPGBM`.

**Is your target claim severity — positive loss amounts on claims-only records?**

Continue.

**Is your log-severity histogram visibly bimodal?**

This requires looking at the data. Plot `np.log(y_sev)` as a histogram with 60–80 bins. A unimodal distribution produces one hump. A bimodal distribution produces two humps with a visible trough. If you cannot tell by inspection, use `scipy.stats.gaussian_kde` to smooth the density and look for two local maxima.

If the answer is no — one hump, possibly right-skewed, but one hump — use `GammaGBM`. Full stop. `NeuralGaussianMixture` will not outperform it on unimodal severity. It adds PyTorch infrastructure and compute cost for no modelling benefit.

If the answer is yes: continue.

**Do you have at least 10,000 severity observations in the training set?**

Below roughly 10,000 observations, the neural network has insufficient signal to reliably separate mixture components from noise. The Gamma model's parametric structure pools information across all observations and becomes an asset when data is sparse. The mixture model needs enough observations in the minority component — typically the serious injury population, at 10–15% of BI claims — to calibrate that component's parameters reliably. 15% of 10,000 is 1,500 serious injury observations, which is close to the minimum for stable component estimation.

If the answer is no: use `GammaGBM`, potentially with `FlexCodeDensity` as a diagnostic tool to characterise the bimodal structure without trying to parametrise it.

**Does your use case require the distributional shape above the mean?**

If you are pricing to mean pure premium — frequency × severity for a standard P&L budget — the mean of a bimodal distribution is the weighted mean of the components. A `GammaGBM` that gets the mean right, even if it misrepresents the shape, is adequate for mean pricing. The bimodal model earns its keep only when the shape above the mean matters commercially.

The cases where it matters:

- **XL reinsurance layer pricing.** Expected loss in a layer £X xs £Y requires integrating the severity density above £Y. A unimodal model misrepresents the density in the tail. A bimodal model that correctly places 13% of the probability mass in a high-mean serious injury component will price a £200k xs £50k layer 30–50% higher than a `GammaGBM` — and that higher price is correct.
- **IFRS 17 risk adjustment.** The risk adjustment under the general measurement model should reflect the full distribution of outcomes. The variance of a bimodal distribution is larger than the variance of the fitted Gamma even when means are equal. Using the Gamma variance understates the required risk adjustment for BI reserves.
- **Solvency II per-risk tail estimates.** Internal models using per-risk 99th percentile severity need the full distribution. The Gamma P99 for bimodal UK motor BI data is understated by 40–60% in the serious injury segment.
- **Quantile pricing.** P75 or P90 severity as a conservative anchor for large-risk pricing. That anchor is wrong if the severity model is unimodal and the data is not.

If your use case is purely mean pure premium with no layer pricing, risk adjustment, or quantile pricing requirement, use `GammaGBM`.

If your use case includes any of the above and you have cleared the bimodality and sample size gates: use `NeuralGaussianMixture`.

**Can your infrastructure support PyTorch?**

This is not a modelling question, but it belongs in the decision. A method that cannot be deployed is not a practical method.

`GammaGBM` runs on scikit-learn-compatible infrastructure. Any Python environment that can install XGBoost can run it. `NeuralGaussianMixture` requires PyTorch: `pip install insurance-distributional[neural]`. If your scoring environment is a Docker container, a Lambda function, or a managed ML platform, adding PyTorch is non-trivial — it is a large dependency, version-sensitive, and requires GPU-aware runtime configuration if you are training at scale (inference on CPU is fast, but training is not).

If PyTorch is not available within the project timescale, use `FlexCodeDensity`. It is non-parametric, requires no PyTorch, and will represent the bimodal structure correctly — at the cost of non-interpretable components and slower prediction evaluation.

---

## Where `FlexCodeDensity` fits relative to `NeuralGaussianMixture`

`FlexCodeDensity` is more flexible: it makes no parametric assumption about the conditional density shape, including no Gaussian component assumption. For home escape of water with three modes — attritional repair, full strip-out and reinstatement, structural loss with decant — it will characterise all three modes without you specifying K=3. It also handles unusual severity shapes that Gaussian components represent poorly: strongly skewed component distributions, asymmetric tails.

`NeuralGaussianMixture` is more interpretable. You can inspect component weights, means, and variances per risk and say "this risk profile has a 22% weight on the serious injury component." That is a number you can explain to an underwriter, a reinsurance partner, or a regulator. `FlexCodeDensity` gives you a density curve; `NeuralGaussianMixture` gives you named populations.

`NeuralGaussianMixture` is also faster to evaluate per prediction once trained. The mixture density has closed-form expressions for mean, variance, quantiles, and layer pricing. `FlexCodeDensity` requires numerical integration for all of these.

Our view: for UK motor BI specifically — where the bimodal structure is well-characterised and the K=3 component interpretation maps to recognisable claim populations (attritional, fault accident, serious injury) — `NeuralGaussianMixture` is the better tool. For home severity, or for lines where the distributional shape is genuinely unknown, `FlexCodeDensity` is the correct default until you have enough data to justify a parametric mixture structure.

---

## The production readiness question

The routing logic above assumes you have already made the infrastructure and governance decisions. A frank production readiness assessment requires addressing those directly.

**Model governance under PRA CP6/24.** The PRA's insurance model risk guidance (CP6/24, 2024) applies to internal model components. A `NeuralGaussianMixture` embedded in a severity module needs documented motivation (why is this line's severity bimodal? — log-histogram evidence), quantitative validation (CRPS comparison against `GammaGBM`, PIT calibration, component weight diagnostics), and documented tuning decisions (`n_components=3`, `energy_weight=0.5`, `log_transform=True` with validation evidence for each). The neural network component adds one governance requirement that a GBM does not: the training process is stochastic, so validation evidence needs to demonstrate stability across random seeds, not just for a single fit.

**Reproducibility.** PyTorch training with a fixed `random_state` is reproducible on the same hardware. Across different hardware — GPU vs CPU, different BLAS implementations — numerical differences in floating-point arithmetic can produce different training trajectories. For a regulatory submission, fixing the runtime environment (Docker image, PyTorch version, CPU/GPU specification) is necessary to ensure the model produces the same output on refit. This is standard practice for any neural network in a regulated context, but it is a new requirement for teams whose current stack is pure GBM.

**Serialisation.** `NeuralGaussianMixture` serialises via `pickle`, matching the convention of `GammaGBM`. Ensure the deserialisation environment has a matching PyTorch version — PyTorch's serialised state dict format is not always forward-compatible across major versions.

**Training latency.** On a CPU with a 30,000-observation severity dataset and `epochs=300`, a `NeuralGaussianMixture` with `n_components=3` and `hidden_size=64` trains in roughly 4–8 minutes. On a GPU, under a minute. For a quarterly pricing model rebuild, either is acceptable. For a monthly rebuild or any requirement to retrain mid-cycle, GPU infrastructure is warranted.

---

## A note on the Tweedie/Gamma choice within the stack

One question that arises: should you use `TweedieGBM` with a power parameter near 1.5 rather than a separate frequency model and `GammaGBM`?

`TweedieGBM` is the correct choice when you want a single model for pure premium and the frequency/severity decomposition is not required for any downstream use. It is more robust on sparse lines (few claims per policy) and requires less feature engineering for the frequency/severity split.

Separate frequency and severity models — `TweedieGBM(power=1.0)` or `NegBinomialGBM` for frequency, `GammaGBM` or `NeuralGaussianMixture` for severity — are the correct choice when the severity model output is used independently: XL pricing, reserving at claim level, IFRS 17 fulfilment cash flows at risk segment. In these cases the severity model needs to be meaningful on its own.

`NeuralGaussianMixture` is always a severity-only model. It has no exposure offset. The frequency-severity pipeline is:

```python
from insurance_distributional import TweedieGBM, NeuralGaussianMixture

# Frequency component — all policies
freq = TweedieGBM(power=1.0)
freq.fit(X, y_count, exposure=exposure)

# Severity component — claims only
sev = NeuralGaussianMixture(n_components=3, log_transform=True)
sev.fit(X_claims, y_severity)

# Mean pure premium = predicted frequency × predicted severity mean
pp = freq.predict(X_new).mean * sev.predict(X_new).mean

# Distributional questions: quantile, layer pricing
sev_pred = sev.predict(X_new)
p95_severity = sev_pred.quantile(0.95)
xl_el = sev_pred.price_layer(attachment=50_000, limit=200_000)
```

---

## Summary

| Use case | Recommended model | Reason |
|----------|-------------------|--------|
| Pure premium, combined process | `TweedieGBM` | Single model, no split required |
| Count with structural excess zeros | `ZIPGBM` | Zero-inflation handled explicitly |
| Unimodal severity | `GammaGBM` | Parametric, fast, no bimodal signal to capture |
| Bimodal severity, < 10k observations | `GammaGBM` | Neural model underfits; parametric pooling wins |
| Bimodal severity, mean pricing only | `GammaGBM` | Mean of mixture ≈ Gamma mean; shape does not matter |
| Bimodal severity, unknown shape | `FlexCodeDensity` | Non-parametric; no parametric assumption required |
| Bimodal severity, quantile/layer/IFRS 17, ≥ 10k obs | `NeuralGaussianMixture` | Correct bimodal representation; analytic quantiles |

`NeuralGaussianMixture` earns its keep in a specific, commercially material slot: XL layer pricing, IFRS 17 risk adjustment, or Solvency II per-risk tail estimates on lines where severity is structurally bimodal and the dataset is large enough. For UK motor BI and some home perils, that slot is real and the financial consequences of using a unimodal model are significant. For most other severity problems, `GammaGBM` is simpler, faster, and equally accurate.

```bash
pip install "insurance-distributional[neural]>=0.4.0"
```

---

## Reference

Yang, Ji, Li & Deng (2026). "Energy Score-Guided Neural Gaussian Mixture Model for Predictive Uncertainty Quantification." arXiv:2603.27672.

---

*Related posts:*
- [Two-Hump Distributions: Why Your Severity Model Gets the 95th Percentile Wrong](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) — quantifying the GammaGBM understatement at P90/P95 on bimodal BI data
- [Energy Score-Guided Mixture Models: What the NE-GMM Paper Actually Contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/) — paper assessment: what is new, what the ablations show, where the gaps are
- [Why NLL Fails to Train Mixture Severity Models](/2026/04/03/ne-gmm-loss-function-why-nll-fails-mixture-training/) — the mode collapse mechanism and how the Energy Score term changes the loss surface
- [Validating a Mixture Severity Model: When NE-GMM Earns Its Keep and When GammaGBM Still Wins](/2026/04/03/neural-gaussian-mixture-energy-score/) — the step-by-step validation workflow: log-histogram, CRPS comparison, PIT, component diagnostics
- [Multimodal Severity in insurance-distributional v0.4.0](/2026/04/02/neuralgaussianmixture-insurance-distributional-v040/) — full API reference and worked property example
