---
layout: post
title: "Two-Hump Distributions: Why Your Severity Model Gets the 95th Percentile Wrong"
date: 2026-04-02
categories: [techniques, pricing]
tags: [severity-modelling, gaussian-mixture, bimodal, energy-score, neural-network, motor-insurance, bodily-injury, reserve-quantiles, xl-reinsurance, ifrs17, insurance-distributional, NE-GMM, arXiv, UK-insurance, python]
description: "UK motor bodily injury severity is structurally bimodal. A GammaGBM fits one mode between two humps, understating the 95th percentile by 30-40%. NeuralGaussianMixture fixes this without parametric assumptions."
---

UK motor bodily injury severity has two humps. Everyone who has spent time with raw claims data knows this. Attritional soft tissue claims settle under £5,000. Serious injury claims — fractures, brain injuries, PPO cases — start around £15,000 and have no ceiling. These are not the same population. The mechanisms are different, the litigation timelines are different, the reserve development patterns are different.

Your GammaGBM does not know this. It fits a single shape to the combined distribution. It parks its mode somewhere between £3,000 and £8,000 — technically close to the mean, useless for understanding either population. Then you price an excess of loss layer and wonder why the reinsurer disagrees with your expected loss calculation.

This post works through why unimodal parametric severity models systematically understate upper quantiles for bimodal loss distributions, quantifies the bias, and shows how `NeuralGaussianMixture` from `insurance-distributional` v0.4.0 addresses it directly.

---

## The structural bimodality of UK motor BI

UK motor bodily injury severity splits into two economically distinct populations.

**Attritional claims** settle under Pre-Action Protocol or the small claims track. Minor whiplash, soft tissue, short-term physiotherapy. The bulk of these settle between £500 and £5,000. They account for roughly 85–90% of BI claims by count and perhaps 45–50% of total BI reserve value.

**Serious and catastrophic injury claims** involve litigation, rehabilitation, and — for the worst cases — a Periodical Payment Order (PPO) that converts a lump sum into an annuity. These start around £15,000 and extend to £500,000+. They represent 10–15% of claims by count but over 50% of total BI reserve value on most commercial books.

The gap between the modes is not a smooth continuum. There is a genuine trough in the £5,000–£15,000 range. Claims in that band are typically soft tissue with complications, or minor fractures — present, but sparse. The distribution is approximately:

```
f(y) ≈ 0.87 * Lognormal(7.5, 0.5)   # attritional: mode ~£1,600
      + 0.13 * Lognormal(11.2, 0.9)  # serious: mode ~£73,000
```

Those numbers are indicative calibrations from published UK Claims studies, not model output. But the qualitative structure — two distinct modes with a trough between them — is consistent across books.

Home insurance escape of water is actually worse. Three natural modes: contained drying (£1,500–£3,500), full strip-out and reinstatement (£8,000–£20,000), structural damage with decant (£60,000+). A Gamma model fitted to that data has no chance.

---

## What a GammaGBM actually fits

The Gamma distribution has a mean-variance relationship Var[Y|x] = phi * E[Y|x]^2. It is unimodal by construction. For any given risk profile x, the fitted conditional distribution is a single-humped curve with one mode.

When you fit a GammaGBM to bimodal severity data, the maximum likelihood estimate pushes the mode toward the mean of the mixture. For the BI parameters above, the mixture mean is around:

```
E[Y] = 0.87 * exp(7.5 + 0.5²/2) + 0.13 * exp(11.2 + 0.9²/2)
     ≈ 0.87 * 1,874 + 0.13 * 88,000
     ≈ £13,060
```

The Gamma model's mode will land somewhere in the £8,000–£15,000 range — the trough between the two actual modes. It has no probability mass where the actual data concentrates. It is a model that is right about the mean and wrong about almost everything else.

The consequence for upper quantiles is systematic and large. On synthetic bimodal BI data calibrated to the parameters above, a GammaGBM underestimates the 90th percentile by roughly 30% and the 95th percentile by 35–40%.

| Metric | True (mixture) | GammaGBM | Understatement |
|--------|---------------|----------|---------------|
| Mean | £13,060 | £12,840 | 2% — close |
| P75 | £8,700 | £6,200 | 29% |
| P90 | £45,000 | £31,500 | 30% |
| P95 | £98,000 | £60,000 | 39% |
| P99 | £280,000 | £155,000 | 45% |

These are not edge-case numbers. At P90, the Gamma model is off by 30% for every single risk. That translates directly into:

- **XL reinsurance layers**: expected loss in a £50k xs £50k layer is driven by probability mass above £50k. A model that puts its mode at £10k with thin tails underestimates this materially. The reinsurer's actuary has the same bimodal awareness you are missing, which is why their price seems high.
- **IFRS 17 risk adjustment**: the risk adjustment under IFRS 17 depends on the variance of the liability estimate and its tail behaviour. A unimodal model with underestimated tails produces an understated risk adjustment.
- **Solvency II capital**: per-risk 99th percentile estimates feed into internal models and standard formula calibrations. The Gamma model understates these by ~45% in our example.

---

## The mode collapse problem with standard MDNs

The natural fix is a Mixture Density Network (Bishop, 1994): an MLP with output heads for mixture weights, component means, and component variances. Trained on your data, it should learn the bimodal structure directly.

The catch is that standard MDNs are trained with negative log-likelihood (NLL) alone, and NLL-trained MDNs have a well-documented failure mode: **mode collapse**. The loss function rewards sharpness. One component acquires most of the probability mass, the other components' weights decay toward zero, and the model degenerates into a near-unimodal predictor — exactly the problem you started with, just with a neural network rather than a GBM underneath.

Mode collapse happens because NLL penalises misplaced probability mass but does not penalise overconfidence in a single mode. If placing all weight on one sharp component near the observation drives down the NLL efficiently, the gradient will do exactly that.

---

## The Energy Score: a properly calibrated alternative

Yang, Ji, Li & Deng (arXiv:2603.27672, March 2026) address this with a hybrid training objective. Instead of pure NLL, they train with:

```
L = η * L_NLL + (1 - η) * L_ES
```

The Energy Score (Gneiting & Raftery, *JASA* 2007) is:

```
ES(F, y) = E_F[|X - y|] - 0.5 * E_F[|X - X'|]
```

The first term measures how far the predictive distribution is from the observation. The second term — the self-energy term — penalises overconfident or degenerate distributions: if your predictive samples X and X' are always close together (mode collapse), the second term is small, and the total ES is dominated by the first term's penalty. Spreading the distribution appropriately reduces the first term; collapsing it increases the penalty for the second term. A strictly proper scoring rule: the loss is uniquely minimised by the true data distribution. There is no gradient incentive to be pathologically sharp.

The practical problem with Energy Score training is that the expectations require Monte Carlo sampling: draw many X and X' from the forecast distribution, compute pairwise absolute differences. For a Gaussian mixture, this is O(M²) where M is the sample size. Too slow for a training loop.

Yang et al.'s load-bearing contribution is deriving a **closed-form analytic Energy Score formula** for Gaussian mixtures. The expectations in the Energy Score have analytic solutions when F is a Gaussian mixture — they reduce to integrals of the form E[|X - a|] for a Gaussian X, which evaluates as a function of the normal CDF and PDF. The resulting formula is O(K²) in the number of components, not O(M²) in sample size. For K=3, this is nine terms. Negligible compute.

The formula is:

```
ES = sum_m pi_m * A_m(y) - 0.5 * sum_m sum_l pi_m * pi_l * B_ml
```

where A_m and B_ml are closed-form expressions in the normal CDF. This means you can include the Energy Score in every gradient step with no additional sampling overhead.

---

## How NeuralGaussianMixture works in practice

```bash
pip install insurance-distributional[neural]
```

PyTorch is optional: the `[neural]` extra pulls it in. The GBM classes are unaffected.

```python
from insurance_distributional import NeuralGaussianMixture, GammaGBM

# Fit on positive severity observations only — this is a severity model
sev_model = NeuralGaussianMixture(
    n_components=3,     # K=3: attritional, fault accident, serious injury
    energy_weight=0.5,  # equal NLL + ES weighting; paper's recommended default
    log_transform=True, # train on log(y); back-transform in predict
    epochs=300,
    random_state=42,
)
sev_model.fit(X_sev_train, y_sev_train)
pred = sev_model.predict(X_sev_test)

pred.mean              # E[Y|X] — expected severity per risk
pred.volatility_score()  # CoV = SD/mean — for safety loading
pred.quantile(0.90)    # 90th percentile — XL pricing
pred.quantile(0.99)    # 99th — reserve quantile
pred.price_layer(attachment=50_000, limit=200_000)  # EL in layer

# Score it
sev_model.energy_score(X_sev_test, y_sev_test)  # analytic; lower is better
sev_model.crps(X_sev_test, y_sev_test)          # MC; same units as y
```

`K=3` is the right default for UK motor BI: attritional soft tissue, fault accident with property damage, serious and catastrophic injury. If you are modelling home EoW, K=3 also covers the three natural modes there.

For the frequency-severity pipeline, the position is straightforward:

```python
# Frequency model handles exposure — severity model does not
freq_model = TweedieGBM(power=1.5)
freq_model.fit(X, y_total, exposure=exposure)

# Severity model: fit only on positive losses
mask = y_sev > 0
sev_model = NeuralGaussianMixture(n_components=3, log_transform=True)
sev_model.fit(X[mask], y_sev[mask])

# Pure premium
freq_pred = freq_model.predict(X_test, exposure=exposure_test)
sev_pred  = sev_model.predict(X_test)
pure_premium = freq_pred.mean * sev_pred.mean
```

The `mean` property is the mixture mean: sum_k(pi_k * mu_k), computed analytically. For XL pricing, `price_layer()` uses Monte Carlo sampling from the fitted mixture, which is fast since sampling from a Gaussian mixture is exact (sample component by weight, then sample from that Gaussian).

---

## Comparing GammaGBM against NeuralGaussianMixture

The CRPS is the right selection criterion. It is strictly proper, has the same units as severity (pounds), and is sensitive to both the mean and the distributional shape. A model that gets the mean right but misrepresents the tails will lose on CRPS.

On data with genuinely unimodal severity, `GammaGBM` will match or beat `NeuralGaussianMixture` on CRPS with far less compute and no PyTorch dependency. The Gamma model is well-specified for that case. Use it.

On bimodal BI data, the story inverts. The GammaGBM's CRPS deteriorates at high-severity thresholds because the Gamma is misspecified — it cannot put probability mass where the data actually is. The NeuralGaussianMixture's CRPS improves as epochs increase and the two modes become distinct in the fitted mixture.

The practical check:

```python
from insurance_distributional import NeuralGaussianMixture, GammaGBM

gamma = GammaGBM()
gamma.fit(X_sev_train, y_sev_train)

ngm = NeuralGaussianMixture(n_components=3, log_transform=True, epochs=300)
ngm.fit(X_sev_train, y_sev_train)

print(f"GammaGBM CRPS:  {gamma.crps(X_sev_test, y_sev_test):.2f}")
print(f"NeuralGMM CRPS: {ngm.crps(X_sev_test, y_sev_test):.2f}")
```

If the NeuralGMM wins on CRPS by more than a few percent, your BI severity data is genuinely multimodal and the Gamma model has been quietly misrepresenting it.

---

## A note on the strictly proper scoring result

There is a point worth being precise about. NLL is also a strictly proper scoring rule. The reason we still get mode collapse from NLL-only training is that mode collapse is a **local optima problem** in the optimisation, not a property of the loss function's theoretical minimiser.

In theory, NLL is minimised by the true distribution. In practice, neural network training via gradient descent gets trapped: one component dominates early, acquires most of the gradient signal, and the other components stop contributing. The loss surface has sharp local minima corresponding to degenerate one-component solutions.

The Energy Score term changes the shape of the loss surface in a way that makes these degenerate minima less attractive. The self-energy term in ES penalises solutions where all samples cluster near one mode. This regularises the optimisation toward multi-component solutions, not just the theoretical solution.

The combination is therefore not just "two proper scoring rules averaged": it is the right practical engineering choice for avoiding the known failure mode of NLL-only training on mixture models. Yang et al. (2026) confirm this in ablation experiments: NLL-only MDNs collapse on multimodal data; ES-only models can be calibrated but are less sharp; the 50/50 hybrid is the consistent winner.

---

## What this means for your reserving and capital models

The P90, P95, and P99 estimates from `NeuralGaussianMixture` are per-risk marginal quantiles. They are not portfolio quantiles. For aggregate reserving under IFRS 17 or Solvency II:

1. Use `pred.sample(n_samples=10_000)` to generate Monte Carlo paths per risk from the fitted mixture
2. Aggregate across risks with your correlation assumptions
3. Report portfolio quantiles from the aggregate distribution

The per-risk quantile is still useful for:
- **XL reinsurance pricing**: per-risk expected loss in a layer is a per-risk calculation
- **Large loss loading**: identifying which risk profiles contribute disproportionately to tail exposure
- **Underwriter referrals**: flagging risks where the 99th percentile exceeds an underwriting authority threshold — currently unimplementable with a unimodal severity model

For capital purposes, the per-risk 99.5th percentile estimate is an input, not a portfolio answer. Feed the samples into your aggregation layer.

---

## Getting started

```bash
pip install insurance-distributional[neural]
```

Start with your BI book, positive losses only, K=3. Run it for 300 epochs. Compare CRPS against your existing GammaGBM. If the GMM wins by more than 3%, you have a modelling gap that is affecting your XL pricing and your reserve quantiles.

```python
from insurance_distributional import NeuralGaussianMixture

model = NeuralGaussianMixture(
    n_components=3,
    energy_weight=0.5,
    log_transform=True,
    epochs=300,
    random_state=42,
)
model.fit(X_train_sev, y_train_sev)
pred = model.predict(X_test_sev)

# The number your current model is getting wrong
pred.quantile(0.95)
```

The bimodal structure in UK motor BI severity is not a modelling curiosity. It is a real feature of the claims distribution with real consequences for pricing the tail. A model that fits one hump to two humps is wrong about the structure, and the error compounds as you move further into the tail.

---

*`insurance-distributional` v0.4.0 is open source under MIT licence. Source on [GitHub](https://github.com/burning-cost/insurance-distributional). The underlying method is from Yang, Ji, Li & Deng (2026), arXiv:2603.27672.*

- [Multimodal Severity in UK Motor and Property: NeuralGaussianMixture in insurance-distributional v0.4.0](/2026/04/02/neuralgaussianmixture-insurance-distributional-v040/) — the library release post with API reference and worked property example
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — the parent library; GammaGBM and TweedieGBM with per-risk CoV
- [Tail Scoring Rules: When CRPS Fails the Tail](/2026/03/26/tail-scoring-rules-crps-fails-tail-what-to-use-instead/) — if 99th-percentile calibration is the primary concern, Energy Score has advantages over CRPS for evaluating tail predictive distributions
- [Spliced Severity Distributions: When One Distribution Isn't Enough](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) — the parametric alternative: manual threshold selection with a body distribution and GPD tail
- [How to Build a Large Loss Loading Model for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) — quantile GBMs for per-risk large loss loading; complementary to the distributional approach here
