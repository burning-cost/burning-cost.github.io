---
layout: post
title: "Tab-TRM: Best on the Benchmark, Not the Right Starting Point"
date: 2026-03-31
categories: [machine-learning, pricing]
tags: [Tab-TRM, neural-architecture, Richman, Wuthrich, Padayachy, French-MTPL, Poisson-deviance, PIN, insurance-gam, EBM, linearisation, state-space, GLM, IRLS, boosting, benchmark, interpretability, FCA-governance, UK-motor-pricing]
description: "Tab-TRM sets the French MTPL benchmark at 23.589×10⁻² Poisson deviance, beating PIN ensemble by 0.3%. The linearisation result — Tab-TRM is approximately a state-space model — is the real intellectual contribution. For UK pricing teams: start with EBM, not this."
author: burning-cost
math: true
---

Tab-TRM (Padayachy, Richman, Wüthrich, arXiv:2601.07675, January 2026) is the best-performing published neural architecture on the French MTPL benchmark. It beats the previous best — PIN ensemble, which ships in our [insurance-gam](https://github.com/burning-cost/insurance-gam) library — by 0.078×10⁻² Poisson deviance. It is a genuine contribution from a group that consistently produces actuarially grounded ML.

We are not building it. This post explains why the decision is right, and why the most interesting part of the paper is not the benchmark result at all.

---

## What the architecture actually does

Tab-TRM maintains two learned latent vectors throughout inference: an answer token **a** and a reasoning state **z**, both 28-dimensional. The model tokenises the nine MTPL covariates, then runs a nested recursive update — three inner refinement steps per outer iteration, six outer iterations. Both update functions are zero-hidden-layer affine maps followed by GELU activation. The model's depth comes entirely from recursion: 36 recursive applications in total.

Total trainable parameters: 14,820. For context, the plain-vanilla FNN on this benchmark uses 792 parameters and achieves 23.783×10⁻² ensemble deviance. Tab-TRM uses 19 times more parameters and achieves 23.589×10⁻². That is not a criticism — the parameter efficiency relative to CAFFT (27,133 parameters, 23.726×10⁻²) is genuine.

The full benchmark table on French MTPL:

| Model | Parameters | Ensemble (10-model) |
|---|---|---|
| Poisson GLM | 50 | 24.102 |
| Plain FNN | 792 | 23.783 |
| Credibility Transformer | 1,746 | 23.711 |
| PIN (insurance-gam) | 4,147 | 23.667 |
| **Tab-TRM** | **14,820** | **23.589** |
| Linearised Tab-TRM | ~14K | 23.698 |

The GLM-to-Tab-TRM improvement is 0.513×10⁻² (2.1% relative). That is the gain from adopting neural pricing at all. The PIN-to-Tab-TRM improvement is 0.078×10⁻² (0.3% relative). That is the marginal gain from Tab-TRM over what insurance-gam already provides.

---

## The linearisation result

The benchmark numbers are not what makes this paper interesting. The linearisation result is.

The authors demonstrate quantitatively that Tab-TRM's recursive dynamics are approximately linear. Their evidence:

1. Least-squares surrogates of the update steps achieve R² > 0.9 across recursion steps
2. The fully linearised variant — activations removed — achieves 23.698×10⁻² ensemble deviance, only 0.109×10⁻² worse than the full model
3. The linearised form is exactly a discrete-time state-space: $s(t+1) = A \cdot s(t) + B \cdot \bar{e} + c$, where $\bar{e}$ is the mean feature token

The outer loop, which updates the answer token using accumulated reasoning state, parallels IRLS — at each step the estimate is corrected using information about the current gap. The inner loop, which updates the reasoning state by integrating feature tokens, parallels gradient boosting's stagewise residual correction. This is not a rhetorical parallel: the update equations have the same mathematical structure. Tab-TRM is approximately a learned implementation of what IRLS and boosting are doing.

For a pricing actuary, this matters. It means you can point to the update equations and say "step T in the outer loop is doing approximately what one IRLS iteration does." That is a meaningful regulatory narrative — though it requires tooling that does not yet exist to make it operational.

The linearised variant achieves 23.698×10⁻², which is better than the Credibility Transformer (23.711×10⁻²) and only 0.031 worse than PIN ensemble. It is also the more interpretable model: the state-space transition matrix in principle supports feature importance analysis via the $A^\top B$ structure. The paper does not build this. If the ETH Zurich group packages it with a feature-importance API, the case for building changes.

---

## What insurance-gam already provides

The three models that ship in insurance-gam cover the same benchmark territory with different interpretability profiles:

**EBM** (via InterpretML): shape functions are the model. Every feature's contribution is a curve you can plot and explain to a pricing committee. Hard monotonicity via soft regularisation. The clearest governance path of anything in the library.

**ANAM**: per-feature MLP subnetworks summed in a GAM structure with guaranteed monotonicity via Dykstra projection. Explicit pairwise interaction specification. Factor tables are a direct output.

**PIN**: handles all pairwise feature interactions via a shared interaction network. Exact Shapley values without sampling approximation. PINEnsemble achieves 23.667×10⁻² on French MTPL — the closest published competitor to Tab-TRM.

Every model in insurance-gam produces either per-feature shape curves or exact Shapley values. Tab-TRM produces neither for the full nonlinear model. The linearised variant could produce feature importances, but no implementation exists today.

---

## The honest case against building

The 0.3% deviance improvement over PIN is real. On a £500M GWP motor portfolio it could be financially material in theory. But the evidence base for that claim is thin:

**Single dataset.** French MTPL has nine clean features and a well-understood structure. UK motor books typically have 50–150 features, with sparse experience in many cells. The improvement margin over PIN may not survive at scale, and there is no evidence either way.

**No time-based validation.** All evaluations use random train/test splits. TabReD (ICLR 2025) showed that models with strong temporal structure can reverse benchmark rankings when tested on future data. A motor pricing model trained on 2022 data and tested on 2024 data is a different experiment from a random split.

**No severity model.** Tab-TRM addresses claim frequency only (Poisson). A complete pricing model requires a severity side. Until there is a Tab-TRM Gamma severity variant that matches the frequency architecture, you cannot actually replace a GLM with it.

**No pip-installable package.** There is no `uv add tab-trm`. Implementing from scratch means 1,200–1,500 lines of code, a test suite, and no reference implementation to validate against.

**No interpretability advantage.** The FCA governance path for a model with no factor tables and no Shapley values is unclear. The EBM in insurance-gam has the clearest governance path of any neural architecture we know of.

---

## Where to start if you are not already using neural pricing

If your pricing team is running a Poisson GLM and is considering an upgrade, Tab-TRM is not the right entry point. The 2.1% deviance improvement over a GLM is real, but you will get most of it from EBM with less implementation cost, better interpretability, and a model your pricing committee can actually review.

The right sequence for a UK personal lines pricing team:

1. **Start with EBM** (`insurance_gam.ebm`). Pip-installable, shape functions your committee can sign off, reasonable deviance improvement over GLM. This is the answer for teams that are not yet using neural pricing.

2. **Move to PIN ensemble** if you need the extra precision and are willing to explain Shapley values to governance. 23.667×10⁻² is the best result available in production-ready tooling.

3. **Watch Tab-TRM** for the two things that would change the build decision: a second benchmark dataset that confirms the improvement generalises, and a severity model that completes the pricing pipeline.

Tab-TRM's current benchmark position is genuinely impressive. The linearisation result is the best conceptual contribution from the Richman/Wüthrich group in some time — bridging neural architecture and actuarial estimation theory in a way that is not just rhetorical. But benchmark leadership on a single French dataset, without packaging, without interpretability tooling, and without a severity model, is not sufficient to displace models that are already in production.

---

## Further reading

- [insurance-gam on GitHub](https://github.com/burning-cost/insurance-gam) — EBM, ANAM, and PIN for insurance pricing
- [Tree-like Pairwise Interaction Networks for Insurance Pricing](/2026/03/13/insurance-pin/) — the PIN architecture that Tab-TRM beats by 0.3%
