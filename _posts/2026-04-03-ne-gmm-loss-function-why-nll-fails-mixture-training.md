---
layout: post
title: "Why NLL Fails to Train Mixture Severity Models: The Engineering Fix in NE-GMM"
date: 2026-04-03
categories: [techniques, research]
tags: [NE-GMM, energy-score, gaussian-mixture, mixture-density-networks, mode-collapse, proper-scoring-rules, severity-modelling, motor-insurance, bodily-injury, arXiv-2603.27672, Yang-Ji-Li-Deng-2026, Bishop-1994, Gneiting-Raftery-2007, loss-function, insurance-distributional, python, UK-insurance, distributional-regression, calibration, CRPS]
description: "Negative log-likelihood is a proper scoring rule. So why does NLL training collapse a Mixture Density Network to a single component? The answer is in the loss surface geometry, not the theory. NE-GMM's analytic Energy Score term changes the surface. Here is the mechanism."
author: burning-cost
math: true
---

We have written about [why UK motor BI severity is bimodal](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) and [what the NE-GMM paper contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/). Those posts are about the what and the so-what. This one is about the mechanism. Why does NLL training collapse mixture models in practice when the theory guarantees it should not? What does the Energy Score term do to the training dynamics, specifically? And why is the analytic O(K²) formula the thing that makes this practical rather than interesting?

This is aimed at actuaries comfortable with GLM training — MLE, log-likelihood, gradient descent — who want to understand a claim they will encounter: "our mixture model uses a different loss function and it is strictly proper." That claim is true but incomplete. The incomplete version misses the engineering problem that motivated the fix.

---

## The theory says NLL should work. The practice says otherwise.

Negative log-likelihood is a strictly proper scoring rule. This is the same result that justifies maximum likelihood estimation for GLMs: if the model class is correct, NLL is uniquely minimised by the true data distribution. The Gamma GLM minimises NLL over the space of Gamma distributions; if the true severity is Gamma-distributed, the MLE recovers the true parameters in the large-sample limit.

For a Gaussian mixture, the same theory applies. The NLL for observation y_i given a K-component mixture is:

```
l_i = log( sum_k pi_k * N(y_i; mu_k, sigma2_k) )
```

In principle, gradient descent on the sum of -l_i over the training set should find mixture parameters that approximate the true conditional distribution. In practice, it does not, for a very specific reason.

The NLL loss surface for mixture models has local minima corresponding to **degenerate one-component solutions**. These are configurations where one component carries nearly all the weight (pi_1 ≈ 1, pi_2 ≈ ... ≈ 0) and sits near the training data's mean. They are not the global minimum — that requires a genuine K-component solution — but they are attainable by gradient descent because the gradient early in training points toward them.

The mechanism is this: in early training epochs, the component closest to any given observation contributes most of the log-likelihood for that observation. The gradient for that component's parameters is large and well-directed. The gradients for the other components are small — they are weighted by pi_k, which is close to zero. Those components do not receive useful gradient signal, their parameters drift, and the network learns to ignore them. By the time the dominant component has overfit to the attritional peak, the other components have nothing to offer: any move that increases their weight also increases the penalty for probability mass away from the dominant cluster.

This is mode collapse. It is not a theoretical failure of NLL — it is a practical failure of gradient descent on a loss surface with degenerate local minima.

---

## What the loss surface looks like

For a simple two-component Gaussian mixture with equal variance, the NLL surface as a function of mixing weight pi_1 and the separation between component means looks roughly like this:

- At pi_1 = 1 (one-component degenerate solution): a local minimum with a basin of attraction. The component mean parks near the training data's overall mean. The second component is irrelevant.
- At pi_1 = 0 (other one-component solution): symmetric, same structure.
- At pi_1 = 0.7 or 0.87 (the true bimodal UK motor BI mixing weights): a lower NLL value — this is the global minimum — but to reach it, gradient descent must climb out of the basin around the degenerate solution first.

Whether gradient descent escapes the degenerate basin depends on initialisation, learning rate, and batch structure. On non-insurance benchmark data, it sometimes escapes. On real insurance severity data with the serious injury mode being a small minority (13% of claims), the serious injury component is chronically underweighted in early batches — any given batch is likely to contain very few serious injury observations — and the gradient toward the global multi-component solution is drowned out by the gradient toward the dominant attritional component.

This is why mode collapse is worse on insurance data than on symmetric benchmark datasets. The bimodal structure is asymmetric in a way that makes the degenerate solution particularly stable.

---

## What the Energy Score does to the loss surface

The Energy Score (Gneiting & Raftery, *JASA* 2007, 102:359–378) is:

```
ES(F, y) = E_F[|X − y|] − 0.5 * E_F[|X − X'|]
```

where X and X' are independent draws from F.

The first term is minimised by placing mass near the observation. The second term — the self-energy of F — is large when draws from F are spread out and small when they cluster together.

Think about what happens to these two terms under mode collapse. If F has collapsed to a single sharp component:

- First term: draws from F cluster near the component mean; the expected distance to y depends on where y falls relative to that mean.
- **Second term: draws from F cluster together; the expected distance between X and X' is small.** The self-energy term is minimised in exactly the wrong way — it contributes almost nothing to the total loss, leaving the first term to dominate.

Now think about what happens with a two-component model that has correctly separated the attritional and serious injury modes. Draws from F sometimes come from the attritional component (£1,500 range) and sometimes from the serious injury component (£50,000 range). The self-energy E[|X − X'|] is large — draws from different components are far apart. This keeps the second term meaningful, and the negative sign means a large self-energy **reduces the loss** for a correctly spread distribution.

The Energy Score therefore penalises mode collapse that NLL does not, by penalising the self-energy collapse that accompanies degenerate one-component solutions. The gradient of the ES term with respect to the component separation and mixing weights points away from degenerate solutions, even in early training epochs when the NLL gradient does not.

The hybrid objective:

```
L = 0.5 * L_NLL + 0.5 * L_ES
```

inherits NLL's sharpness incentive (place mass near observations) and adds ES's spread incentive (keep the self-energy from collapsing). The Yang et al. (2026) ablations confirm this: NLL-only MDNs collapse on multimodal data; ES-only models are well-calibrated but less sharp; the 50/50 hybrid avoids collapse and matches sharpness.

---

## Why ES training was impractical before the analytic formula

The Energy Score requires evaluating E_F[|X − y|] and E_F[|X − X'|]. For an arbitrary distribution F, these require Monte Carlo approximation: draw M samples from F, compute M absolute differences for the first term and M² pairwise absolute differences for the second. For M=1000 samples per observation and a batch of 256 observations, the second term alone requires 256 million absolute difference computations per gradient step. Training becomes infeasible within any reasonable time budget.

The Yang et al. contribution is showing that both expectations have closed-form analytic solutions when F is a Gaussian mixture. The key identity is the folded-normal expectation: for X ~ N(μ, σ²),

```
E[|X − a|] = (μ − a)(2Φ((μ − a)/σ) − 1) + 2σ φ((μ − a)/σ)
```

where Φ is the standard normal CDF and φ the standard normal PDF. This is a well-known result from truncated normal theory. Applying it to a K-component Gaussian mixture:

**First term** (E_F[|X − y|]): sum over k components of pi_k times the folded-normal expectation for component k. O(K) per observation.

**Second term** (E_F[|X − X'|]): X − X' where X and X' are independent draws from F. If X comes from component m and X' from component l (with probability pi_m * pi_l), then X − X' ~ N(μ_m − μ_l, σ²_m + σ²_l). The same folded-normal identity applies. Sum over K² component pairs. O(K²) per observation.

For K=3, this is 3 + 9 = 12 terms per observation. The full analytic Energy Score formula adds roughly three lines of vectorised PyTorch to the forward pass — slower than pure NLL by a constant factor, not by orders of magnitude.

This is the load-bearing technical contribution. The concept of using proper scoring rules to regularise mixture training was available before this paper. What was not available was an efficient implementation. The analytic formula changes ES-guided mixture training from a research curiosity to a practical method.

---

## The implementation in PyTorch

The analytic formula translates directly into vectorised operations. The full implementation for a batch of n observations with K components:

```python
import math
import torch

def analytic_energy_score(pi, mu, sigma2, y):
    """
    pi:     (n, K) — mixture weights (sum to 1 along dim 1)
    mu:     (n, K) — component means
    sigma2: (n, K) — component variances (positive)
    y:      (n,)   — observed targets
    Returns: scalar mean Energy Score (lower is better)
    """
    sigma = torch.sqrt(sigma2)
    y_exp = y.unsqueeze(1)              # (n, 1) for broadcasting

    # Folded-normal E[|X_k - y|] for each component k
    delta   = mu - y_exp               # (n, K)
    std_d   = delta / sigma            # (n, K)
    SQRT2PI = math.sqrt(2 * math.pi)
    phi_    = torch.exp(-0.5 * std_d**2) / SQRT2PI
    Phi_    = 0.5 * (1 + torch.erf(std_d / math.sqrt(2)))
    A       = delta * (2*Phi_ - 1) + 2*sigma * phi_  # (n, K)

    # Folded-normal E[|X_m - X_l|] for all (m, l) pairs
    mu_i    = mu.unsqueeze(2)           # (n, K, 1)
    mu_j    = mu.unsqueeze(1)           # (n, 1, K)
    s2_i    = sigma2.unsqueeze(2)
    s2_j    = sigma2.unsqueeze(1)
    d_ij    = mu_i - mu_j              # (n, K, K)
    sv_ij   = torch.sqrt(s2_i + s2_j)
    std_ij  = d_ij / sv_ij
    phi_ij  = torch.exp(-0.5 * std_ij**2) / SQRT2PI
    Phi_ij  = 0.5 * (1 + torch.erf(std_ij / math.sqrt(2)))
    B       = d_ij*(2*Phi_ij - 1) + 2*sv_ij * phi_ij  # (n, K, K)

    term1   = (pi * A).sum(dim=1)                    # (n,)
    pi_pi   = pi.unsqueeze(2) * pi.unsqueeze(1)      # (n, K, K)
    term2   = 0.5 * (pi_pi * B).sum(dim=[1, 2])      # (n,)

    return (term1 - term2).mean()
```

The K=3 case generates tensors of shape (n, 3, 3) — nine component pairs per observation. For n=256 batch size, the B matrix is 256×3×3 = 2,304 elements. The full forward pass for this term is microseconds.

The hybrid loss then is:

```python
nll_loss = -torch.logsumexp(log_pi + log_comp, dim=1).mean()
es_loss  = analytic_energy_score(pi, mu, sigma2, y)
loss     = energy_weight * nll_loss + (1 - energy_weight) * es_loss
```

One point that matters for numerical stability: `log_pi + log_comp` is computed for the NLL term using log-sum-exp to avoid underflow when mixture weights are small. The Energy Score term works directly in probability space, not log space, which is numerically fine for sigma2 away from zero. The `softplus(x) + eps` activation on the variance head ensures sigma2 >= eps = 1e-6 throughout training.

---

## Verifying the formula: analytic versus Monte Carlo

The analytic formula should match Monte Carlo estimation to within noise. A useful sanity check during implementation:

```python
from insurance_distributional import NeuralGaussianMixture
import numpy as np

# Fit on bimodal synthetic data
rng = np.random.default_rng(42)
n = 2000
y = np.where(rng.binomial(1, 0.13, n),
             rng.lognormal(11.2, 0.9, n),   # serious injury: mode ~£73k
             rng.lognormal(7.5,  0.5, n))   # attritional: mode ~£1,600
X = rng.standard_normal((n, 4))

model = NeuralGaussianMixture(n_components=3, energy_weight=0.5,
                               log_transform=True, epochs=200, random_state=42)
model.fit(X, y)

# analytic Energy Score
es_analytic = model.energy_score(X, y)

# Monte Carlo Energy Score (independent implementation)
pred = model.predict(X)
mc_samples = pred.sample(n_samples=5000, seed=0)   # (n, 5000)
mc_samples_2 = pred.sample(n_samples=5000, seed=1)
term1_mc = np.abs(mc_samples - y[:, None]).mean()
term2_mc = 0.5 * np.abs(mc_samples - mc_samples_2).mean()
es_mc = term1_mc - term2_mc

print(f"Analytic ES: {es_analytic:.4f}")
print(f"Monte Carlo ES: {es_mc:.4f}")
print(f"Relative difference: {abs(es_analytic - es_mc) / abs(es_mc):.3%}")
# Expected: < 2% for 5000 MC samples
```

If the analytic and MC results disagree by more than a few percent, there is a bug in the analytic formula — most likely a sign error in the self-energy term or a missing factor in the folded-normal identity. The formula is not complicated, but the signs matter.

---

## What this changes for severity model selection

The honest case for NE-GMM versus a GammaGBM is not "NE-GMM is always better." It is "NE-GMM is better when your severity is genuinely bimodal, and the bimodal structure has practical consequences for your pricing."

The GammaGBM's failure mode is systematic and predictable: it underestimates upper quantiles for bimodal severity. On synthetic data calibrated to published UK motor BI distributions (87% attritional, 13% serious injury), the understatement at P95 is around 35–40%. This flows directly into XL layer pricing — the expected loss in a £200k xs £50k layer is driven by probability mass above £50k, which the Gamma model underestimates — and into IFRS 17 risk adjustment calculations that depend on distributional spread and tail behaviour.

The NE-GMM's failure mode, when it fails, is different: on unimodal or weakly bimodal data, or with fewer than around 10,000 severity observations, the mixture structure either does not materialise cleanly or adds variance without reducing bias. A CRPS comparison on held-out data resolves this empirically.

The decision tree is short:

1. Is your log-severity histogram visibly bimodal?
2. Do you have at least 10,000 severity observations?
3. Does your use case require quantile estimates, XL layer pricing, IFRS 17 risk adjustment, or Solvency II per-risk VaR?

If the answer to all three is yes, run the comparison. If NE-GMM improves CRPS by more than 3–5% on your held-out data, the bimodal structure in your severity is material and worth modelling explicitly. If it does not, the Gamma model is adequate for your data.

---

## The gap that remains

Yang et al. validate on synthetic and UCI benchmark datasets. There is no insurance severity dataset in the paper. This is not a failure of the paper — it is a general ML paper, not an actuarial paper — but it means the published benchmark results are not direct evidence of performance on UK motor BI or property severity.

The calibration argument is sound. The analytic formula is correct. What is not yet in the published literature is a controlled comparison on, say, freMTPL or a comparable UK motor dataset, showing that NE-GMM outperforms both Delong et al.'s Gamma MDN and a well-specified GammaGBM on held-out CRPS.

We plan to run this comparison as part of the `insurance-distributional` v0.4.0 validation work and publish the results. Until then, the right approach is to run the comparison on your own data, using CRPS as the criterion, before deploying the model.

---

## Getting started

```bash
pip install "insurance-distributional[neural]>=0.4.0"
```

PyTorch is opt-in. The GBM classes are unaffected.

```python
from insurance_distributional import NeuralGaussianMixture

# Start here for UK motor BI severity
model = NeuralGaussianMixture(
    n_components=3,       # attritional / fault / serious injury
    energy_weight=0.5,    # 50/50 NLL + ES; paper's recommended default
    log_transform=True,   # essential for severity spanning 3 orders of magnitude
    epochs=300,
    random_state=42,
)
model.fit(X_sev_train, y_sev_train)

# Check the mixture has not collapsed
pred = model.predict(X_sev_test)
print(pred.weights.mean(axis=0))
# Healthy: e.g. [0.78, 0.14, 0.08]
# Collapsed: e.g. [0.93, 0.05, 0.02] — increase energy_weight to 0.7 and refit
```

The component weight check is important. Energy Score training prevents most mode collapse, but on datasets where the minority mode is very small (below 5% of observations), the gradient toward a degenerate solution can still win. Check the mean weights after fitting. If one component dominates above 0.90, increase `energy_weight` toward 0.7 and refit.

---

Yang, Ji, Li & Deng (2026). "Energy Score-Guided Neural Gaussian Mixture Model for Predictive Uncertainty Quantification." arXiv:2603.27672.

---

*Related posts:*
- [Two-Hump Distributions: Why Your Severity Model Gets the 95th Percentile Wrong](/2026/04/02/two-hump-distributions-severity-model-95th-percentile/) — the applied case: quantifying GammaGBM understatement at P90/P95, with API examples
- [Energy Score-Guided Mixture Models: What the NE-GMM Paper Actually Contributes](/2026/04/02/ne-gmm-energy-score-neural-mixture-insurance-severity/) — the paper assessment: what is new, what the ablations show, where the gaps are
- [Validating a Mixture Severity Model: When NE-GMM Earns Its Keep](/2026/04/03/neural-gaussian-mixture-energy-score/) — the validation workflow: CRPS comparison, PIT histogram, component diagnostics, XL layer pricing checks
- [Tail Scoring Rules: When CRPS Fails the Tail](/2026/03/26/tail-scoring-rules-crps-fails-tail-what-to-use-instead/) — Gneiting & Raftery (2007) in depth: when Energy Score is preferable to CRPS for evaluating tail distributional predictions
