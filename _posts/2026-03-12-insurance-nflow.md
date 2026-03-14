---
layout: post
title: "Beyond Lognormal: Normalizing Flows for Insurance Severity Modelling"
date: 2026-03-12
categories: [libraries]
tags: [normalizing-flows, severity, neural-spline-flow, tail-transform, student-t, ILF, TVaR, reinsurance, zuko, Hickling-Prangle, motor-BI, Python, insurance-nflow]
description: "UK motor bodily injury severity is bimodal. A lognormal cannot represent this shape. insurance-nflow drops the family assumption entirely — Neural Spline Flows with a Student-t tail transform, built on zuko, with an actuarial API producing TVaR, ILF curves, and reinsurance layer costs."
post_number: 72
---

Every severity model we build is a bet. A lognormal is a bet that log-claims are roughly Gaussian. A gamma GLM is a bet that the coefficient of variation is constant across rating groups. A Pareto is a bet that the body does not matter and only the tail does.

For UK motor bodily injury, every one of those bets loses money in the same place: the transition between soft-tissue claims and catastrophic injury. BI severity is bimodal. Soft-tissue claims — whiplash, minor orthopaedic — cluster around GBP 3,000–5,000 after the Civil Liability Act 2018 reforms compressed the lower end of that distribution. Catastrophic injury claims — spinal cord damage, traumatic brain injury, fatalities — follow a power law from GBP 100,000 upward, with no natural ceiling. A lognormal cannot represent this shape. It smooths over the valley between the two modes, misrepresents both peaks, and in the tail it decays too fast by an order of magnitude.

`insurance-nflow` drops the family assumption entirely.

---

## What normalizing flows actually do

A normalizing flow is an invertible neural network that transforms a simple base distribution — typically a standard Gaussian — into an arbitrarily complex target distribution. The key property is tractable likelihood: because the transformation is invertible and the Jacobian is computable, we know the exact log-probability of any observation under the model. No approximations, no variational bounds. This means we can compare flows against lognormal or gamma by test log-likelihood on a held-out set, and the comparison is honest.

The architecture we use is the Neural Spline Flow (NSF, Durkan et al., NeurIPS 2019). NSF uses monotone rational quadratic splines as its coupling layers, which means it can represent non-Gaussian shapes — including bimodality — with depth-6 to depth-8 networks. It is faster to train than autoregressive variants (MAF, IAF) because the coupling architecture parallelises across the batch, and faster still to sample from, which matters when you are drawing a million scenario losses for a catastrophe simulation.

For positive-only claim data, we log-transform first. This is not a modelling assumption; it is a change of variable. `log(severity)` lives on the real line, where the flow operates, and the Jacobian correction is analytic. Set `n_transforms=0` and no tail layer, and `SeverityFlow` is exactly a lognormal. Every layer of depth we add relaxes that assumption incrementally.

The tail problem requires one more step. Standard NSF has a Gaussian base distribution, which means log-probabilities decay quadratically in the tail — too thin for Pareto-tailed claims. Hickling and Prangle (ICML 2025, PMLR 267:23155-23178) solve this with a Tail Transform Flow: a final bijection that maps the Gaussian base to a Generalised Pareto tail, with learnable shape parameters lambda+ and lambda-. The statistical justification is EVT: the tail transform maps the Gaussian to the Fréchet domain of attraction, with GPD tails whose shape parameters are identifiable from the data.

The ICML paper used a complementary error function (erfc) construction for this transform. That implementation has sign-preservation failures when the tail parameter lambda < 1 — exactly the heavy-tail regime relevant to motor BI catastrophic injury. We replaced this with Student-t CDF warping: T(u) = F_t^{-1}(Phi(u); nu), parameterised as lambda = 1/nu. Mathematically equivalent in tail behaviour, provably monotone, and numerically stable across the entire parameter range. The Hill double-bootstrap estimator pre-estimates lambda from the upper tail of training data before fixing it for the main training run; Hickling's paper shows this "fix" variant outperforms joint training on heavy-tailed targets.

---

## The API

The library is built on [zuko](https://github.com/probabilists/zuko) (v1.6.0, March 2026 — the only actively maintained Python normalizing flow library), wrapped behind an actuarial interface that hides PyTorch entirely. All inputs and outputs are numpy arrays.

### Unconditional severity

```python
import numpy as np
from insurance_nflow import SeverityFlow

claims = np.array([...])  # positive GBP amounts

flow = SeverityFlow(
    n_transforms=6,
    tail_transform=True,   # Student-t CDF warping tail layer
    tail_mode='fix',       # Hill double-bootstrap pre-estimates tail params
    max_epochs=200,
    patience=20,
)
result = flow.fit(claims)

print(result)
# SeverityFlowResult(val_logL=-9.2341, AIC=1234567.8, n_params=97412, lambda+=0.612, epochs=143)

print(flow.tvar(0.99))        # TVaR(99%) in GBP
print(flow.quantile(0.995))   # VaR(99.5%) in GBP
```

The `lambda+=0.612` in the result is the upper tail index — equivalently, a Student-t with nu = 1/0.612 ≈ 1.63 degrees of freedom. A lognormal would have lambda = 0 (Gaussian tail). A Pareto(alpha=1.5) — representative of UK motor catastrophic BI — gives lambda ≈ 0.67. The flow is estimating tail weight from the data, not assuming it.

### ILF curve from flow samples

```python
ilf = flow.ilf(
    limits=[50_000, 100_000, 250_000, 500_000, 1_000_000],
    basic_limit=50_000,
)
# {50000.0: 1.0, 100000.0: 1.31, 250000.0: 1.72, 500000.0: 2.18, 1000000.0: 2.61}
```

The ILF curve is derived from half a million flow samples, with no parametric assumption about the tail. A lognormal ILF curve will be conservative on the right for BI severity; a Pareto ILF curve will be too aggressive across the full range. The flow gives you the shape the data actually implies.

### Conditional severity with rating factors

This is the use case that matters for pricing. The standard GLM captures the mean shift across rating groups; it cannot capture changes in distribution shape. A young driver in London and a mid-age driver in the North West may have the same expected severity under a GLM, but the London young driver's distribution has a heavier right tail — more catastrophic injury claims per thousand, not just higher frequency. A conditional flow sees this difference.

```python
from insurance_nflow import ConditionalSeverityFlow

claims = np.array([...])    # shape (N,)
context = np.array([...])   # shape (N, n_factors): age_band, vehicle_group, region, ncb_years, postcode_band

flow = ConditionalSeverityFlow(
    context_features=5,
    n_transforms=6,
    tail_transform=True,
)
result = flow.fit(claims, context=context, exposure_weights=exposure)

# TVaR(99%) for two risk profiles
young_london = np.array([[1, 8, 1, 3, 0]])   # young, performance vehicle, London
mid_north    = np.array([[3, 5, 3, 5, 5]])   # mid-age, standard vehicle, North West

print(flow.conditional_tvar(young_london, 0.99))
print(flow.conditional_tvar(mid_north,   0.99))
```

The rating factors are embedded via a context encoder network that feeds into each coupling layer's parameter network — this is zuko's native conditional API, `flow(context).log_prob(x)`. The context encoder learns a compact representation of the rating factor space; the coupling layers use it to modulate the spline knots per observation.

### Reinsurance layer pricing

```python
# Expected cost to 200k xs 50k per-occurrence XL layer
layer_cost = flow.reinsurance_layer(
    attachment=50_000,
    limit=200_000,
    context=portfolio_factors,
    n_samples=500_000,
)
```

Five hundred thousand samples from the flow takes a few seconds on CPU (coupling architecture parallelises). This replaces the traditional workflow of fitting a composite lognormal-Pareto model with a manually chosen threshold, a fragile EM algorithm, and the implicit assumption that the lognormal body is correct. It also replaces the need to decide where the body ends and the tail begins — the flow models both jointly.

### Benchmarking against parametric families

```python
from insurance_nflow.diagnostics import fit_parametric_benchmarks, model_comparison_table

ll_benchmarks, k_benchmarks = fit_parametric_benchmarks(claims)
ll_benchmarks["flow"] = float(result.train_log_likelihood * len(claims))
k_benchmarks["flow"] = result.n_parameters

table = model_comparison_table(claims, ll_benchmarks, k_benchmarks)
# Sorted by test log-likelihood per observation.
# AIC column is reported with a note: flow has ~97k params vs 2 for lognormal.
```

The comparison is designed to be honest about where the flow wins and where it does not. On the synthetic bimodal UK motor BI generator included in the library, the flow achieves roughly 0.3 nats per observation better than lognormal at n = 50,000 claims. On smoother severity distributions, the gap narrows; below n = 10,000, the flow may not have enough data to outperform on held-out likelihood.

---

## What we actually built

The library has 200 tests across six modules, all passing on Python 3.11 with current torch and zuko.

**tail.py** implements the Student-t CDF warping bijection. Round-trip accuracy is ~1e-3 across the support; the Jacobian is verified both analytically and numerically against autograd. The Hill double-bootstrap estimator initialises lambda from the upper tail before the tail layer parameters are fixed for main training.

**flows.py** wraps zuko's NSF with the TTF tail layer and log-transform mixin. The log-transform Jacobian is applied analytically rather than through autograd — autograd on log-transforms accumulates numerical errors in float32 that compound across coupling layers.

**severity.py** provides `SeverityFlow` and `ConditionalSeverityFlow` with exposure-weighted NLL. Weights are normalised to mean 1.0 before training; the effective sample size n_eff = (Σw)² / Σw² is reported in results for correct AIC/BIC calculation, since standard AIC/BIC assume unweighted i.i.d. observations.

**actuarial.py** is pure numpy: TVaR, ILF, LEV, and reinsurance layer decomposition, each tested against analytic closed forms for Pareto and lognormal. These functions work on any array of positive samples — flow-generated or otherwise. If you have a Monte Carlo sample from any model and want actuarial outputs, this module is useful independently.

**data.py** generates synthetic bimodal UK motor BI data: a mixture of lognormal (soft-tissue mode, mean GBP ~5k, weight 0.85) and Pareto (catastrophic mode, tail index ~1.5, weight 0.15). The DGP has known theoretical TVaR and ILF values used throughout the test suite.

**diagnostics.py** produces PIT histograms, QQ plots, tail index comparison, and the AIC/BIC comparison table across families.

---

## Limitations, stated plainly

**No truncation or censoring in v0.1.0.** Property claims data often has left-truncation at the deductible: claims below the threshold are not reported. Adjusting the NLL for truncation requires evaluating the flow CDF at each observation's truncation point. Normalizing flows do not have analytic CDFs — the CDF requires numerical integration over the change-of-variables formula, which roughly doubles training time. This is technically feasible and planned for v0.2.0. For now, the library is appropriate for motor BI (minimal truncation) and not for excess-of-loss property books.

**CPU training is slow for large books.** One hundred thousand claims, 200 epochs, depth-6 NSF takes around 45–50 minutes on a modern CPU. A 500,000-claim book would take 4–6 hours. The standard mitigation: subsample to 100,000 for model selection, validate on the full held-out set. GPU training is opt-in via `device='cuda'` and brings 100k claims down to under 5 minutes on an A100.

**The AIC comparison is not clean.** A depth-6 NSF with a 64-wide context encoder has around 97,000 parameters. A lognormal has 2. AIC will always prefer the lognormal on a small dataset — this is correct behaviour, not a bug. We report AIC for completeness, we caveat it in the output, and we recommend test log-likelihood per observation as the primary comparison metric.

**Regulatory acceptance is an open question.** FCA PS21/5 requires pricing models to be explainable and auditable. A flow with 97,000 parameters satisfies tractable likelihood — you can always compute the exact probability of any claim — but does not produce interpretable coefficients. We think this is a genuine constraint for regulated rate filings. We do not think it prevents teams from using flows for internal pricing analysis, scenario generation, reinsurance placement, and capital modelling.

**PyTorch is a dependency.** Insurance pricing teams run sklearn, pandas, scipy, xgboost, lightgbm. PyTorch is not standard kit. The CPU-only install is around 900MB. The library API hides all PyTorch from the user, but the installation is the team's problem to manage.

---

## Where this sits in the severity modelling hierarchy

The Burning Cost portfolio now covers four severity modelling approaches, each appropriate at a different point on the interpretability/flexibility trade-off:

**Parametric GLM** (Gamma or lognormal): interpretable coefficients, fast to fit, FCA-friendly. Wrong distribution shape for bimodal data. Use when sample size is small or regulatory explainability is the binding constraint.

**insurance-gamlss**: penalised splines for distribution parameters (mu, sigma, nu, tau of a flexible family). More flexible than GLM, still parametric, still interpretable. Use when the family assumption is approximately right but the shape parameters vary by rating factor.

**insurance-severity**: distributional regression network — neural network for all distribution parameters, still parametric family. Use when GLM linearity is the binding constraint but you can still commit to a family.

**insurance-nflow**: no family assumption. Full conditional distribution learned nonparametrically from data. Use when the family is wrong — bimodal severity, shape changes across rating factors, catastrophic tail that lognormal misrepresents.

The flow is at the top of that hierarchy in terms of flexibility and at the bottom in terms of interpretability and speed. It is the right choice when the alternative is being wrong about the tail by a factor of two.

---

## Installation

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install insurance-nflow
```

Source and issue tracker: [github.com/burning-cost/insurance-nflow](https://github.com/burning-cost/insurance-nflow)

---

## See also

- [Distributional Regression Networks for Insurance Pricing](https://burning-cost.github.io/2026/03/10/distributional-refinement-network-insurance/) — when you need nonlinear rating factor effects but can commit to a parametric family; faster and more interpretable than a flow
- [Spliced Severity Models with insurance-severity](https://burning-cost.github.io/2026/03/13/insurance-composite/) — parametric alternative to flows for heavy-tailed severity; lower compute cost, manual threshold selection required
- [Borrowing Experience You Don't Have: insurance-thin-data](https://burning-cost.github.io/2026/03/12/borrowing-experience-you-dont-have/) — transfer learning for thin-segment pricing; relevant when your BI book lacks enough catastrophic injury claims for stable tail estimation
- [Extreme Value Theory for UK Motor Large Loss Pricing](https://burning-cost.github.io/2026/03/13/insurance-evt/) — if you only care about the tail (XL pricing), EVT targets the tail directly without modelling the body

---

*insurance-nflow v0.1.0 — 200 tests, 6 modules. Built on [zuko](https://github.com/probabilists/zuko) v1.6.0 and the Hickling & Prangle (ICML 2025) tail transform.*
