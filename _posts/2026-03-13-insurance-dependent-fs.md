---
layout: post
title: "Your Frequency × Severity Model Is Multiplying the Wrong Things"
date: 2026-03-13
author: Burning Cost
categories: [pricing, machine-learning, libraries]
tags: [frequency-severity, dependence, neural-networks, two-part-model, shared-trunk, multi-task-learning, GLM, Poisson, Gamma, pure-premium, python, insurance-dependent-fs]
description: "Standard two-part models multiply frequency and severity as if they were independent. They are not. We built insurance-dependent-fs — a shared-trunk neural model with explicit dependence testing — to price the interaction correctly."
---

Your pricing model multiplies frequency by severity. It does this because that is the actuarial convention, and the convention is mathematically justified if frequency and severity are independent conditional on the covariates. In most UK personal lines books, they are not.

The problem is not that actuaries are unaware of this. It is that the standard toolkit makes it hard to do anything about it. The copula approach is theoretically clean but computationally painful for mixed discrete-continuous distributions. Encoding the dependence parametrically via a single scalar works, but only captures the simplest possible dependence structure. And most off-the-shelf neural two-part models — including the NeurFS architecture from arXiv:2106.10770 — fit two separate networks and then link them with a scalar correction. That is better than independence, but it still leaves structure on the table.

We built [`insurance-dependent-fs`](https://github.com/burning-cost/insurance-frequency-severity) to do this properly. The core architecture is a shared encoder trunk — a single set of neural network weights that learns features informative for both frequency and severity simultaneously. The dependence is not a correction applied after training. It is encoded in the representation the model learns.

---

## Why independence is wrong

The textbook pure premium formula is:

```
E[Z | x] = E[N | x] × E[Y | x]
```

This holds if and only if N and Y are independent conditional on x. The empirical evidence in motor insurance consistently shows they are not.

In UK motor, high-frequency policyholders — younger drivers, urban postcodes, higher NCD erosion rates — tend to produce lower average claim severity. There are two overlapping mechanisms. First, the UK's no-claims discount system is steep (up to 65% discount) and creates strong incentives to suppress borderline claims in the £200–£500 range. Policyholders who claim frequently are making a deliberate choice to claim: their unreported claim frequency is even higher, but the reported claims they bring to you are the ones that cleared the suppression threshold. For an accident-prone driver, that threshold is lower. Second, there is genuine omitted variable structure: risk appetite correlates with both frequency and severity in ways that your covariate vector does not fully capture.

The Civil Liability Act 2021 and the OIC portal's £4,215 whiplash cap strengthened this negative correlation further for motor liability. The highest-frequency claim category in motor was capped at a lower average severity. Post-2021, if your frequency model is good and your severity model is good but you multiply them assuming independence, you will systematically overprice high-frequency risks and underprice low-frequency risks.

The magnitude is not trivial. From the French MTPL data used in the NeurFS paper (arXiv:2106.10770), fitting the dependence correction explicitly gives a factor of exp{γ(1 + λ)} where γ ≈ −0.15 and λ is the Poisson rate. For an average risk with λ = 0.08, that is a 2.6% correction. For a high-frequency risk with λ = 0.30, it is 4%. For the tails of the distribution — young male urban drivers with prior claims history — you are in cross-subsidy territory.

---

## Three ways to model the dependence, and why two of them fall short

There are three distinct families of approach in the published literature.

**Family 1: Copula linking.** Fit marginal frequency and severity models, then link them via a bivariate copula. Our existing [`insurance-frequency-severity`](https://github.com/burning-cost/insurance-frequency-severity) library does this with the Sarmanov bivariate distribution. The Sarmanov approach is analytically elegant and produces a closed-form pure premium correction: E[N·Y] = E[N]·E[Y] + ω·E[N·φ₁(N)]·E[Y·φ₂(Y)]. The limitation is that the dependence parameter ω is a single scalar linking the marginal distributions — it captures the average association between claim counts and severities, but the dependence is homogeneous across the covariate space. It cannot capture a segment where the frequency-severity correlation differs from the portfolio average. Use it for small datasets where interpretability is paramount; it is the right tool for that setting.

**Family 2: N as covariate.** Garrido, Genest, and Schulz (Insurance: Mathematics and Economics, 70, 2016) proposed the simplest possible fix: include the claim count N as a covariate in the severity model. The severity regression becomes:

```
log(E[Y | x, N]) = S(x) + γ·N
```

The parameter γ has a clean interpretation: a one-unit increase in realised claims multiplies expected severity by e^γ. For UK motor, γ is negative. The NeurFS paper (arXiv:2106.10770, revised March 2025) extended this to neural networks by training two separate networks — one for frequency, one for severity — with N fed into the severity head. Peng Shi and Kun Shi (Variance, 17(1), 2024) added sparse group lasso regularisation on top of the same architecture.

The limitation is that the dependence structure is parametrically constrained to γ·N. The model cannot learn that the frequency-severity relationship is different for young urban drivers versus rural retired drivers. Everything is captured by a single scalar, which is a strong modelling assumption.

**Family 3: Shared latent representation.** This is what `insurance-dependent-fs` implements. A single encoder network learns a latent representation h(x) ∈ ℝ^d from the covariate vector. Both the frequency head and severity head decode from h(x). Gradients from the Poisson loss and the Gamma loss flow through the same trunk simultaneously:

```
∂L/∂W_trunk = ∂ℓ_freq/∂h · ∂h/∂W_trunk + ∂ℓ_sev/∂h · ∂h/∂W_trunk
```

The trunk therefore learns features that are jointly informative for frequency and severity. The dependence is not a scalar correction — it is encoded in the representation. This is multi-task learning applied to the two-part model, and it is the genuinely novel thing the library does.

No published insurance paper implements this. The closest is the NeurFS architecture, which uses separate networks with N as covariate. The shared trunk approach has no insurance-specific published precedent that we can find.

---

## The architecture

The core PyTorch model has three components: a shared encoder, a frequency head, and a severity head.

```
x ∈ ℝ^p → Encoder (FC + BatchNorm + ELU) → h ∈ ℝ^d_latent
                                              ↓                    ↓
                                    FreqHead → log(λ)    SevHead → log(μ)
```

The frequency head adds a log-exposure offset and produces the Poisson rate. The severity head produces the Gamma mean. Optionally, the explicit γ·N term from the GGS approach can be added to the severity head, giving the model both implicit latent dependence and the analytically tractable scalar correction.

The joint loss is the negative sum of the Poisson log-likelihood (over all policies) and the Gamma log-likelihood (over policies with positive claims only). Both losses backpropagate through the shared trunk. This is the computational step that makes the library require PyTorch — you cannot do this in statsmodels or with gradient boosting, because the shared weight update requires automatic differentiation through both heads simultaneously.

---

## Using the library

```python
from insurance_dependent_fs import DependentFSModel, SharedTrunkConfig, DependentFSDiagnostics

config = SharedTrunkConfig(hidden_dims=[64, 32], latent_dim=16)
model = DependentFSModel(config=config, use_gamma=True)
model.fit(X_train, n_claims=n_train, avg_severity=s_train, exposure=e_train)
```

The `use_gamma=True` flag adds the explicit GGS-style dependence parameter to the severity head. When `use_gamma=False`, the model relies entirely on the shared trunk for dependence. In practice, we recommend `use_gamma=True` during initial exploration — it gives you a testable scalar alongside the implicit representation-based dependence.

Predicting pure premiums:

```python
pp = model.predict_pure_premium(X_test, exposure=e_test)
```

Without the explicit γ term, pure premiums are computed by Monte Carlo: sample N from Poisson(λ), sample Y from Gamma(μ, φ), return the mean of N·Y over n_mc samples. With the explicit γ term, a semi-analytical correction via the Poisson moment generating function is available and is the default.

Extracting the shared latent representation — useful for downstream diagnostics or as feature input to another model:

```python
h = model.latent_repr(X_test)  # shape (n, latent_dim)
```

---

## Diagnostics: testing whether the dependence is real

The thing we most wanted to avoid was shipping a more complex model with no way to test whether the complexity was justified. `DependentFSDiagnostics` addresses this directly.

```python
diag = DependentFSDiagnostics(model)
diag.dependence_test()   # LRT for H₀: γ = 0
diag.vs_independent()    # deviance comparison against two separate GLMs
```

The `dependence_test()` method runs a likelihood ratio test for γ = 0 against γ ≠ 0. If you cannot reject γ = 0 at your chosen threshold, the explicit dependence correction is not warranted and you should use the standard two-part model. This is not a theoretical nicety — it is a model selection tool. For some perils and some portfolios, frequency and severity are genuinely approximately independent conditional on the covariates, and the right answer is to not use this library.

The `vs_independent()` method fits two separate GLMs — Poisson for frequency, Gamma for severity — and compares their combined deviance against the joint model. If the shared trunk does not outperform two independent GLMs, the representation learning is not adding value on that dataset.

These tests mean you can take the library to a model validation conversation and say: we tested whether the dependence structure was real, here is the LRT statistic, here is the comparison against independent baselines. That is a different conversation from "we used a more complex model because it felt right."

---

## When to use this library versus `insurance-frequency-severity`

Use `insurance-frequency-severity` (Sarmanov) when:
- Dataset is under 50,000 policies
- Interpretability and regulatory auditability are the primary concerns
- A single dependence scalar is a sufficient approximation
- You need closed-form analytical output

Use `insurance-dependent-fs` when:
- Dataset is 100,000 policies or more (PyTorch training loop amortises)
- Nonlinear feature interactions are expected (telematics, behavioural data)
- You want the representation to discover dependence structure you have not specified in advance
- You are comfortable explaining multi-task learning to a model validator

The two libraries are not competitors. They implement different statistical philosophies — parametric bivariate GLM versus neural multi-task learning. They answer different questions about the same problem.

---

## The UK pricing context

The evidence for frequency-severity dependence in UK motor is strong and, post-2021, arguably stronger than it was. The whiplash cap means the average severity for the highest-frequency claim type in motor liability has a hard upper bound that did not exist before. Any model trained on pre-2021 data and not updated should be treated with scepticism, both for the severity level and for the frequency-severity relationship.

Home insurance is more complex. Escape of water is the UK peril where we most expect positive frequency-severity correlation: high-frequency EoW incidents (repeated minor plumbing failures in a property) tend to coincide with higher severity when a major failure occurs. The `insurance-dependent-fs` architecture does not presuppose the direction or shape of the dependence — it learns it from data — which is the right property for a peril where the dependence structure is less settled than in motor.

We are not aware of a published UK-specific study that quantifies the premium impact of ignoring frequency-severity dependence. The French MTPL evidence suggests 2–4% mispricing for average to high-frequency risks. Given the similarities between French and UK motor data-generating processes (both MTPL regimes, both NCD systems), that is a reasonable starting estimate for the order of magnitude, not a number to put in a board paper.

---

## Installation and dependencies

```bash
uv add insurance-dependent-fs
```

Python 3.10 or later. Dependencies: PyTorch, NumPy, SciPy, scikit-learn, Pandas. Matplotlib is optional, used only for diagnostic plots.

The library ships two benchmark datasets: the French MTPL dataset (the standard actuarial neural network benchmark) and a synthetic dataset with known dependence parameter γ = −0.15. The synthetic dataset is the right place to start — fit the model, check that the estimated γ̂ is close to −0.15, and verify that `dependence_test()` rejects γ = 0. If the library cannot recover the known γ on synthetic data, it is not ready for your real book.

Source at [github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity). 2,150 lines of code across model, training, wrapper, premium estimation, diagnostics, and data modules. 170+ tests.

---

The standard two-part model is not wrong in principle. It is wrong in application to the datasets most UK pricing actuaries actually have. Independence is a strong assumption. It is empirically violated in motor, probably in home EoW, and possibly in health protection. The question is not whether to account for the dependence. The question is which model of the dependence is appropriate for your data and your risk.

Start with the diagnostic. If `dependence_test()` does not reject γ = 0 and `vs_independent()` shows no deviance improvement, the standard model is fine. If it does, you have a quantified, testable justification for a more sophisticated approach.

---

**Related articles from Burning Cost:**
- [Sarmanov Copulas for Frequency-Severity Dependence](/insurance-frequency-severity/)
- [Your Tweedie Model Doesn't Know About Strategic Non-Claimers](/2026/03/30/insurance-zit-dglm/)
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/02/25/causal-inference-for-insurance-pricing/)
- [Actuarial Neural Additive Models: Exact Interpretability with Tweedie Loss](/2026/03/17/your-interpretable-model-isnt-interpretable-enough/)
