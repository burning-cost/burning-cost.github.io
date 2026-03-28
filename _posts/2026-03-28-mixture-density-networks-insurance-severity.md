---
layout: post
title: "Mixture Density Networks for Insurance Severity"
description: "Gamma GLMs fit a single mode to severity data that often has two or three. Mixture Density Networks output the full conditional distribution — mixing weights, component means, and scales all as functions of rating factors. Here is what they are, where they beat the alternatives, and how to build one in PyTorch."
date: 2026-03-28
categories: [pricing, severity-modelling]
tags: [MDN, mixture-density-networks, severity, escape-of-water, Bishop-1994, Gamma-GLM, PyTorch, lognormal-mixture, spliced-distributions, CANN, GAMLSS, normalizing-flows, LRMoE, UK-home-insurance, distributional-regression]
---

Here is a situation most UK home insurance pricing teams have been in. You run a Gamma GLM on your escape of water severity data. The model fits adequately by Gamma deviance. The lift curve looks fine. You plot the actual severity distribution against what the model implies and you see the problem: the empirical distribution has two peaks — one around £3,000 (minor strip-out, drying equipment, limited reinstatement) and another around £10,000 (full strip-out plus complete reinstatement, possibly alternative accommodation) — and your fitted Gamma sits somewhere in the middle of both, describing neither particularly well.

This is not a data quality issue. It is not a missing variable issue. It is a structural limitation of single-family distributional models on genuinely multimodal severity data. The Gamma family cannot be bimodal. You can add all the rating factors you like; you are still fitting one mode.

Mixture Density Networks (MDNs) were designed for exactly this situation. They let a neural network output the parameters of a mixture distribution rather than a point estimate, so the model can represent "60% probability this is a small claim around £3k, 30% it is a full reinstatement around £10k, 10% it is a large loss above £50k" — and those mixing weights, means, and scales are all functions of the rating factors.

---

## Why severity is often multimodal

Escape of water is the dominant UK home peril by volume — the ABI estimates UK insurers pay around £1.8 million per day in EoW claims, roughly 28–30% of all home claims by count. The claim process has distinct phases, each with its own cost distribution.

Phase 1 is emergency containment: find the leak, isolate the supply, stop further damage. This costs roughly £200–800 and has low variance across properties.

Phase 2 is strip-out and drying: remove saturated materials (plasterboard, carpets, joinery), install commercial drying equipment for three days to three weeks. Typical range is £1,500–£6,000 for a standard property, but drying specifications vary substantially by assessor, contractor framework, and drying conditions.

Phase 3 is reinstatement: redecoration, new flooring, kitchen and bathroom replacement if affected. This is where cost variance explodes. A straightforward 1970s semi runs to £3,000–£8,000. A period property with bespoke joinery, original features, or high-specification finishes runs to £20,000–£40,000.

Phase 4 is major loss: structural damage, extended drying beyond the standard protocol, full alternative accommodation for three to six months, possible subsidence. These claims sit above £50,000 and are uncommon but material to the book.

The empirical severity distribution is therefore a mixture. Many claims resolve at Phase 2 only (early detection, contained damage, tenant in a flat who called quickly). A large fraction escalate to full Phase 3 (landlord property, slow discovery, older construction, high-spec kitchen). A tail escalates to Phase 4. A Gamma GLM will fit a single right-skewed mode somewhere around £5,000–£7,000 and will systematically underprice the Phase 3–4 claims while overpricing the contained drying claims.

The same argument applies to subsidence — minor shrinkage versus structural underpinning is a bimodal distribution — and to storm claims (minor tile replacement versus structural roof failure). EoW is just the most common and the most commercially important case.

---

## The Bishop 1994 architecture

Bishop's 1994 technical report from Aston University (NCRG/94/004) sets out the MDN construction cleanly. It is worth understanding the original before looking at the actuarial extensions.

A standard neural network minimising squared loss produces the conditional mean E[y | x]. For multimodal data this is actively harmful: if the true conditional distribution has two modes at £3k and £10k, the conditional mean is somewhere around £5k and the network will learn to predict £5k for that feature vector. The prediction is statistically correct but useless for pricing individual policies or estimating limited payment functions.

Bishop's insight is that the network should instead output the *parameters* of a mixture distribution. For a K-component Gaussian mixture:

- **K mixing weights** π_k(x): produced by a softmax output layer so they sum to one
- **K component means** μ_k(x): unconstrained linear outputs
- **K component scales** σ_k(x): exp or softplus outputs, constrained positive

Total output neurons: 3K. The loss is the negative log-likelihood of the mixture:

```
L = -Σ_i log[ Σ_k π_k(x_i) · N(y_i; μ_k(x_i), σ_k(x_i)²) ]
```

Everything is differentiable. You backpropagate through the mixture NLL exactly as you would through any other loss. No EM required for the Gaussian case.

For insurance severity, Gaussian components are the wrong choice because claim sizes are strictly positive and right-skewed. The fix is immediate: model log(y) with Gaussian components, which is equivalent to a lognormal mixture on the original scale. This requires no architectural change — transform your severity target to log-space before training and transform back for inference. We prefer this to the Gamma MDN in most practical settings because lognormal mixtures are analytically familiar to actuaries and simple to implement in PyTorch with `torch.distributions.MixtureSameFamily`.

---

## The Delong–Lindholm–Wüthrich Gamma MDN

The actuarial literature's primary reference on MDN severity modelling is Delong, Lindholm, and Wüthrich (2021), 'Gamma Mixture Density Networks and their application to modelling insurance claim amounts', *Insurance: Mathematics and Economics*, available as SSRN 3705225. It replaces Gaussian components with Gamma(α_k, β_k) — more theoretically consistent with the actuarial tradition of Gamma severity models.

Direct maximisation of the Gamma mixture NLL is numerically harder than the Gaussian case. The paper develops two EM algorithm variants: EM Network Boosting (update mixing weights via gradient boosting iterations) and EM Forward Network (single network updated via forward passes using the EM responsibilities as pseudo-weights). The EM Forward Network is preferred in their experiments. On the freMTPL dataset (the standard French motor benchmark), the Gamma MDN with three components outperforms a Gamma GLM on held-out log-likelihood and Gini coefficient.

The code is in R only (github.com/LukaszDelong/GammaMDN). There is no published Python port. The EM approach also means training time is substantial: their recommended configuration — 20 neurons per layer, three layers, 300 EM iterations of 25 epochs each — runs around 13 minutes on freMTPL. That is not prohibitive, but it is not the sub-minute iteration cycle you get from direct NLL maximisation of a lognormal mixture in PyTorch.

One practical finding from the paper worth noting: when the true data-generating process is lognormal rather than Gamma-mixture (as in our synthetic EoW benchmark below), you need approximately seven Gamma components to approximate the lognormal tails adequately. The Gamma family has an exponential right tail; the lognormal has a heavier tail. If your EoW severity has a material large-loss component, Gamma components require more components to fit it than lognormal components do.

---

## Competing approaches: an honest comparison

Before committing to MDN, you should understand what you are trading against.

**Spliced lognormal + GPD.** The current industry standard for sophisticated UK home severity modelling. Fit a lognormal below a threshold (say £30,000) and a Generalised Pareto Distribution above it. The GPD has theoretical backing from extreme value theory — Balkema–de Haan and Pickands showed that the conditional distribution of exceedances above a high threshold converges to GPD. The tail is therefore not an arbitrary parametric assumption; it follows from first principles.

The problem is that spliced models address the tail problem, not the body problem. They still fit a single lognormal to everything below the threshold, and that single lognormal cannot be bimodal. The EoW multimodality sits at £3k–£15k — well inside any reasonable £30k threshold. A spliced model also typically treats the threshold as fixed and chosen empirically rather than as a function of covariates, though Fissler, Merz, and Wüthrich (2023, arXiv:2112.03075) address this with their deep composite regression model, which places the splice point at a covariate-dependent quantile.

**CANN.** The Schelldorfer–Wüthrich (2019) Combined Actuarial Neural Network feeds a GLM prediction as a skip connection into a neural network, which then adjusts the GLM offset. CANN is genuinely excellent for pricing teams that need regulatory defensibility — the model starts at the GLM, and the neural adjustment is auditable against actuarial intuition. Holvoet, Antonio, and Henckaerts (2025, *North American Actuarial Journal*, arXiv:2310.12671) found CANN outperforms plain feed-forward neural networks on severity benchmarks.

But CANN outputs a conditional mean, not a distribution. For limit and layer pricing, limited payment function curves, and reserve uncertainty quantification, you need the full distribution. CANN cannot give you P(loss > £50k | covariates) without additional assumptions. MDN gives you this directly.

**GAMLSS / distributional regression.** GAMLSS (Rigby and Stasinopoulos 2005) models all distribution parameters — location, scale, shape — as smooth functions of covariates. The R package is mature and covers dozens of distributional families. Neural extensions exist: NAMLSS (Thielmann et al., accepted AISTATS 2024) uses neural additive model sub-networks for each parameter, preserving some interpretability. The Avanzi group's DRN (arXiv:2406.00998) approximates the conditional CDF non-parametrically using a piecewise-constant step function.

GAMLSS beats MDN on interpretability — you can read off the covariate effect on scale and shape separately. MDN beats GAMLSS when the distributional family is genuinely unknown, because K-component mixtures are universal density approximators as K grows. The practical tradeoff: if you are confident your severity is broadly lognormal with some tail distortion, GAMLSS. If you suspect genuine multimodality, MDN.

**LRMoE.** The Logit-weighted Reduced Mixture of Experts framework from Fung, Badescu, and Lin (University of Toronto, 2019–2021) uses softmax linear gating functions — not a full neural network — to mix expert distributions. The gating functions are interpretable: actuaries can read the covariate effects on which latent claim-type a policy falls into. The R and Julia packages (LRMoE.jl, Tseung et al. 2021) handle censoring and truncation natively, which matters for excess-of-loss reinsurance data. The limitation is the linear gating: complex feature interactions — postcode × property_age × construction_type — are outside what a linear softmax can capture.

**Normalising flows.** Flows express the conditional density as an invertible transformation of a simple base distribution. RealNVP, MAF, and similar architectures can approximate any continuous density arbitrarily well and can handle complex multimodality. The insurance literature has not embraced them seriously: they are opaque, the training is more finicky than MDN, and the heavy discrete mass at zero for gross incurred amounts (partial claim payments, zeros in individual development) creates structural difficulties. We consider normalising flows research territory rather than a practical pricing tool for 2026.

---

## Practical configuration

**Number of components.** Delong et al.'s empirical finding: three components suffices when the true distribution is Gamma-mixture; seven are needed to approximate a lognormal with Gamma components. For lognormal MDN (Gaussian components on log-scale), three components maps naturally to the EoW structure — drying-only, full reinstatement, major loss. Start at three. Treat K as a hyperparameter and select via held-out negative log-likelihood, not BIC (which assumes fixed-parameter models; neural networks are neither). Above ten components you are almost certainly overfitting.

**Kernel choice.** For positive severity data, do not use Gaussian components on the untransformed scale. Work in log-space. Log-transform your claims target before training; your Gaussian components then correspond to lognormal components on the original scale. This is equivalent to the lognormal MDN and requires no architectural change. The alternative — Gamma components with the Delong et al. EM training — is theoretically elegant but harder to implement in Python, slower to train, and requires more components for heavy-tailed data.

**Training stability.** MDN training has several known failure modes. NaN loss is the most common early failure: a component collapses to near-zero variance, generating an enormous log-likelihood contribution for the sample nearest its mean. Fix with a variance lower bound: use `softplus(output) + 1e-5` rather than `exp(output)` for scale parameters. Always use the log-sum-exp trick for the mixture NLL rather than computing the sum directly — floating point underflow will otherwise occur for low-probability components.

Mode collapse — one component acquires mixing weight near 1 while others decay — is the second failure mode. Fix with entropy regularisation on the mixing weights during warmup (add −λ Σ π_k log π_k to the loss for the first few epochs with λ gradually decaying to zero), or initialise component means with k-means++ on the training data to ensure initial spread.

Batch normalisation before the MDN head destabilises training. Avoid it, or if you need normalisation elsewhere in the network, stop it before the final layer feeding the mixture parameters.

Batch size matters more for MDN than for standard regression: small batches give noisy NLL gradients because the mixture NLL is more sensitive to individual outliers than squared loss. Use at least 256; on a 50k-claim dataset, 512 is reasonable.

**Calibration testing.** Point prediction metrics (Gamma deviance, RMSE) tell you almost nothing about the distributional fit. Use the probability integral transform: compute the CDF of each fitted mixture at each observed claim; if calibrated, this produces a Uniform(0,1) distribution. A KS test will tell you whether it departs significantly from uniform. CRPS (continuous ranked probability score) is the preferred single-number metric — it is a proper scoring rule, it averages well across the distribution, and it has a closed form for Gaussian mixtures (Gneiting and Raftery 2007). For Gamma mixtures you need numerical integration or simulation-based CRPS. Also compute the limited payment function at key limits (£5k, £10k, £25k, £50k) and compare to empirical LPF — this is the metric that matters for reinsurance pricing.

---

## Python tooling

The honest situation in early 2026 is that there is no pip-installable Python package with insurance-specific MDN support. The R package from Delong et al. is the only actuarial-specific implementation.

For practical Python implementation, TensorFlow Probability (TFP) has the most complete built-in support via `tfp.layers.MixtureSameFamily`. For a lognormal MDN in PyTorch, `torch.distributions.MixtureSameFamily` with `torch.distributions.Normal` components (on log-transformed targets) requires about 50 lines of code:

```python
import torch
import torch.nn as nn
import torch.distributions as D

class LognormalMDN(nn.Module):
    def __init__(self, in_features, K=3, hidden=64):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.pi_head  = nn.Linear(hidden, K)          # logits → softmax
        self.mu_head  = nn.Linear(hidden, K)          # unconstrained
        self.sig_head = nn.Linear(hidden, K)          # → softplus + eps

    def distribution(self, x):
        h = self.net(x)
        pi  = torch.softmax(self.pi_head(h), dim=-1)
        mu  = self.mu_head(h)
        sig = torch.nn.functional.softplus(self.sig_head(h)) + 1e-4
        mix  = D.Categorical(probs=pi)
        comp = D.Normal(mu, sig)
        return D.MixtureSameFamily(mix, comp)

    def forward(self, x, y_log):
        # y_log = log(claim_amount): train on log-transformed target
        return -self.distribution(x).log_prob(y_log).mean()
```

Initialise `sig_head` bias positively (e.g. 0.5) to avoid NaN in the first epoch. Initialise `mu_head` bias to spread components across the training target range — k-means++ on the log-transformed claims is worth the extra setup for larger datasets.

For reserving rather than pricing, Al-Mudafer, Avanzi, Taylor, and Wong (2021, *Insurance: Mathematics and Economics* 2022, arXiv:2108.07924) implemented a ResNet-style MDN (ResMDN) with a GLM skip connection analogous to CANN. If you need MDN for triangle-based reserve uncertainty, their code at casact/rp-bnn-claims is the starting point.

There is no pip-installable Python package that provides Gamma MDN components, insurance-specific evaluation metrics (Gini coefficient of lift, LPF curves), GLM initialisation, or censoring support. This is a gap worth filling.

---

## A proposed benchmark

The MDN literature has a credibility problem: the papers establish that MDN can outperform a Gamma GLM, but no published study has done a head-to-head comparison of MDN versus spliced lognormal+GPD versus CANN versus GAMLSS on the same dataset with a consistent evaluation protocol.

Holvoet et al. (2025), the most comprehensive actuarial neural network benchmark we are aware of, compared GLM, GBT, FFNN, and CANN on severity data but omitted MDN entirely.

We are building the comparison. The design:

**Dataset 1: freMTPL2sev.** 26,000 French motor severity claims from the CASdatasets package, the community-standard benchmark. The severity distribution is unimodal and right-skewed. We do not expect MDN to outperform Gamma GLM dramatically here — this is the control.

**Dataset 2: synthetic EoW with known ground truth.** We generate 50,000 observations from a true three-component lognormal mixture where the mixing weights are functions of covariates:

```python
# true data-generating process (simplified)
# π_3 (tail component) driven by is_listed and property_age
# μ_k (component means) driven by sum_insured and region_cost_index
```

The tail component mixing weight is a function of `is_listed` and `property_age_std`. A Gamma GLM cannot recover this covariate dependence — it collapses it into a mean shift. The MDN, if correctly specified, should recover the true π_k(x) functions. This is the test that matters.

**Competitors:** Gamma GLM, lognormal GLM, Gamma GAM (PyGAM), CANN with Gamma response, spliced lognormal+GPD at the empirical 90th percentile, 3-component lognormal MDN, 5-component lognormal MDN (over-parameterised variant), Gamma MDN (Delong et al. approach).

**Evaluation:** held-out NLL (primary), CRPS, PIT uniformity (KS test), limited payment function accuracy at £5k / £10k / £25k / £50k. For the synthetic dataset: recovery of mixing weight functions against ground truth.

We expect the primary finding to be: on unimodal data, MDN is competitive but not transformative. On genuinely multimodal data — which UK home EoW severity is — the LPF accuracy and distributional calibration of MDN is materially better, particularly for the tail component. That is the practical case for adopting the approach.

The benchmark code is in development. We will publish the full results and code when the comparison is complete.

---

## Where this fits in the pricing workflow

One question pricing actuaries will ask is: when should I use MDN rather than what I already have?

If your severity is genuinely unimodal and your primary concern is tail pricing, a spliced lognormal+GPD is probably sufficient and gives you better EVT-grounded tail behaviour than an MDN with Gaussian/lognormal components. Add the EVT tail and move on.

If you need a full conditional distribution for LPF curves, reinsurance layer pricing, or reserve uncertainty, and your peril has multimodal severity characteristics (EoW, subsidence, storm, flood), an MDN adds real value over a CANN or GLM. The lognormal MDN in PyTorch is a manageable implementation, the training is stable with the precautions above, and calibration testing is tractable.

If you need interpretability for Solvency II or FCA purposes, consider the CANN-MDN hybrid: use a GLM as a skip connection to the MDN head so the GLM part is fully auditable and the MDN adds distributional flexibility on top. No published paper has done this for severity as of early 2026, but the architecture is straightforward. The ResMDN from Al-Mudafer et al. is conceptually adjacent (it applies to reserving rather than rating, but the skip-connection design is identical).

The gap in the Python ecosystem — no insurance-grade MDN library with Gamma kernels, GLM initialisation, and proper calibration tooling — is real. Until it is filled, implementation requires writing your own. That is not a barrier for a team comfortable with PyTorch, but it is a reason to wait for the benchmark results before committing to production deployment.
