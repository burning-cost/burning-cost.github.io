---
layout: post
title: "Constrained Neural Networks for Insurance Pricing — Architectural vs Post-Hoc"
seo_title: "Constrained Neural Networks for Insurance Pricing: When Architectural Monotonicity Beats XGBoost's monotone_constraints"
date: 2026-03-31
categories: [neural-networks, techniques]
tags: [monotonic-nn, ICEnet, MonoDense, tensorflow-lattice, XGBoost, CMNN, Consumer-Duty, monotonicity, smoothness, actuarial-constraints, CANN, GLM, insurance-pricing, python, FCA, PRA-SS123, calibration, balance-property]
description: "If you are using XGBoost's monotone_constraints or applying isotonic regression post-hoc, you may already have everything you need. Or you may not. The answer depends on your model type, regulatory context, and whether you care about smoothness as well as monotonicity. We lay out the full decision."
author: burning-cost
---

A recent paper from Seyam & Osman (SOIC 2025) frames monotonicity, calibration, and premium adequacy as "physics-informed" constraints on a Transformer architecture. The framing borrows prestige from the PDE literature — there are no PDEs here, just actuarial rules — but the underlying question is genuinely important: if you are fitting a neural network for insurance pricing, which constraints should be in the architecture itself, and which can be handled afterwards?

This matters because the answer is not the same for all model types, and the regulatory environment has shifted. Consumer Duty, in force since July 2023, creates real pressure to explain non-monotonic pricing relationships. The old answer — "we checked it in validation and it looks fine" — is harder to defend under an outcomes-monitoring regime.

---

## The four constraints that matter for pricing

Before getting to architecture, it helps to be precise about which constraints we are actually discussing. There are four that come up repeatedly in the actuarial literature.

**Non-negativity.** Claims cannot be negative; neither can premiums. This is the easiest one. Every standard insurance NN architecture uses an exp output activation or softplus. It is a solved problem.

**Balance property.** At the portfolio level, expected predicted claims should equal observed claims. More strongly: this should hold locally on sub-cohorts of policies (autocalibration). Wüthrich (2021, *Statistical Theory and Related Fields*) showed that standard SGD with early stopping breaks the global balance property — neural networks can be systematically biased even when the training loss looks fine. The fix is either post-hoc renormalisation (divide all predictions by the ratio of total actuals to total predicted) or the local autocalibration step proposed by Denuit, Charpentier & Trufin (2021, *Insurance: Mathematics and Economics* 101(B): 485-497).

**Monotonicity.** Vehicle age should not produce decreasing risk beyond some point in a way that is actuarially unjustifiable. NCD level should reduce premium monotonically. This is the constraint that requires the most care architecturally, and the one most directly implicated by Consumer Duty.

**Smoothness.** Not usually listed with the other three, but commercially it is as important. A pricing model that produces a sharp spike in premium at age 27 and then drops back is monotone if the overall direction is right, but it is inexplicable to a pricing committee and practically impossible to justify to an intermediary. Smoothness means the rate-change functions are well-behaved — not rough, not oscillatory, not sensitive to small changes in input.

Of these four, non-negativity is trivial, balance/calibration is best handled post-hoc (no architectural surgery needed), monotonicity is where architectural choices matter, and smoothness is almost entirely absent from the current Python landscape except in one published method.

---

## Architectural, soft, and post-hoc: what each actually buys you

The constraint taxonomy is simpler than the literature makes it look.

**Hard architectural constraints** mean the model is *structurally incapable* of violating the constraint. MonoDense (Runje & Shankaranarayana, ICML 2023) and the Deep Lattice Networks underlying tensorflow-lattice (You et al., NeurIPS 2017) achieve this by construction. XGBoost's `monotone_constraints` parameter does it for gradient-boosted trees by forbidding splits that violate the specified direction. If you have a hard architectural constraint, you can say: "This model cannot produce a lower premium for a higher-risk vehicle." Full stop. No caveats about training data coverage or post-processing reliability.

**Soft constraints via loss penalty** train the model to prefer constraint satisfaction but do not guarantee it. ICEnet (Richman & Wüthrich, *Annals of Actuarial Science* 2024, 18: 712-739) uses this approach — adding a Whittaker-Henderson smoothing penalty and a monotonicity penalty to the Poisson deviance. The constraint holds in proportion to how well-tuned the penalty weights λ are. Richman and Wüthrich state this directly: "It is important to note that using the ICEnet method does not guarantee smoothness or monotonicity if the values of the penalties λ are not chosen appropriately." Soft constraints are more flexible and easier to implement on arbitrary architectures. They are not guarantees.

**Post-hoc methods** apply no constraint during training and correct afterwards. Isotonic regression applied to output predictions can enforce monotonicity on the training distribution but makes no guarantee for out-of-distribution inputs. Post-hoc renormalisation for balance is different — the correction is applied to every prediction by the same scaling factor, so it holds everywhere. The fundamental weakness of post-hoc monotonicity is that a new input that falls between training data points may violate the constraint before the post-processing step is reached.

---

## What you can actually install

If you are using XGBoost or LightGBM today, `monotone_constraints` in the booster parameters gives you hard architectural monotonicity for free. Specify direction as +1 or -1 per feature. This is built in, well-tested, and has essentially no performance penalty. If this covers your use case, you do not need anything else for monotonicity.

For neural networks, the landscape as of March 2026:

**`monotonic-nn`** (pip-installable). Implements MonoDense — the CMNN architecture from Runje & Shankaranarayana (ICML 2023). The key insight is that forcing all weights positive, which is the naive monotonicity trick, restricts the network to approximating only convex monotone functions. CMNN constructs two extra activation functions from a standard unsaturated activation — one convex, one concave — and applies them to different subsets of neurons. The result can approximate any continuous monotone function. Per-feature direction is specified through a `monotonicity_indicator` parameter on the `MonoDense` layer. Drop-in replacement for Keras Dense layers. Hard architectural guarantee. Runje shows CMNN has comparable accuracy to unconstrained networks on standard benchmarks, and better accuracy than other monotonic methods.

**`tensorflow-lattice`** (pip-installable, Google-maintained). Implements Deep Lattice Networks. Supports monotonicity, convexity, and pairwise trust constraints. Most production-tested option — Google uses it in ad ranking and loan approval. The constraint is architectural. The downside is TensorFlow dependency, which is a friction point for teams whose Databricks or Sagemaker stacks are PyTorch-first. For teams already on TF/Keras, this is the most mature option.

**MonoKAN** (arXiv:2409.11078, September 2024). KAN (Kolmogorov-Arnold Network) with certified partial monotonicity via cubic Hermite spline derivative conditions. On most benchmarks it outperforms both CMNN and lattice networks. Named insurance and finance as primary applications. No pip package as of March 2026 — research code only.

**ICEnet** (Richman & Wüthrich 2024). Code available in paper supplementary materials. Not on PyPI. The method that handles both smoothness and monotonicity.

---

## ICEnet: the smoothness case

ICEnet is worth understanding even if you do not use it today, because it is the only published method that treats smoothness as a first-class constraint alongside monotonicity.

The mechanism is unusual. Rather than modifying the network architecture, ICEnet augments the training data with pseudo-observations. For each feature j requiring a constraint, pseudo-data is generated by varying x_j across its quantile range while holding all other features fixed — this is exactly the Individual Conditional Expectation (ICE) plot construction. The same network is then applied to both the real data and the pseudo-data. The training loss becomes:

```
L = L₁ + λ_s·L₂ + λ_m·L₃
```

where L₁ is the standard Poisson deviance on real predictions, L₂ is a Whittaker-Henderson third-order difference penalty on the ICE predictions (penalising roughness), and L₃ penalises negative first-differences in the ICE predictions (penalising direction violations). Training data size increases by a factor proportional to the number of constrained features — compute cost scales linearly.

Why does smoothness matter commercially? Because a pricing model that produces smooth, monotone factor curves is one that can be shown to a pricing committee and understood. A motor pricing model where the vehicle-age factor rises sharply at 5 years, dips at 8 years, and rises again at 12 is prima facie suspicious under Consumer Duty even if, mechanically, the overall level is correct. Isotonic regression applied post-hoc can enforce monotonicity but will not smooth out those oscillations. ICEnet handles both in one training step.

Richman and Wüthrich tested ICEnet on the French MTPL benchmark (freMTPL2freq from CASdatasets). The results show smoothness and monotonicity constraints are achievable with negligible accuracy loss on standard insurance frequency data when λ values are appropriately tuned.

---

## The regulatory context

Consumer Duty does not require architectural monotonicity. But it creates a specific evidencing problem for non-monotonic pricing relationships.

The relevant Consumer Duty outcome is fair value: firms must demonstrate that prices are reasonable relative to the benefits provided. A model where premiums fall as a risk factor increases — vehicle value rising through a certain range, say — creates a potential fair value problem because it suggests a cohort of higher-risk customers is being cross-subsidised by lower-risk customers in a way that cannot be justified. The defence "the model found that relationship in the data and we validated it" is weaker under outcomes-monitoring than "this relationship is impossible by construction."

For Consumer Duty outcomes monitoring, an architectural constraint is a model-level property. A post-hoc correction is a pipeline-level property. These are different regulatory claims: the first says the model is incapable of producing the bad outcome; the second says we correct the bad outcome after the model produces it.

PRA SS1/23 on model risk management applies directly to banks, not insurers. But a significant number of large UK general insurers have voluntarily adopted its principles as a governance framework. For those firms, the SS1/23 framing is useful: Principle 5 treats model risk mitigation via architectural constraints (removing a degree of freedom that could produce failures) differently from mitigation via post-training checks (validating that a particular failure mode did not occur in testing). The former is a stronger form of evidence.

For personal lines motor and home — where Consumer Duty exposure is highest — the case for architectural monotonicity on the key rating factors is solid. For commercial lines, the direct Consumer Duty exposure is lower, but governance best practice is pushing in the same direction.

---

## A worked example: MonoDense in practice

Here is the minimum viable implementation of a monotone frequency network using `monotonic-nn`. The architecture mirrors a standard CANN structure, with the prior estimates from a GLM entering as an offset.

```bash
uv add monotonic-nn
```

```python
import numpy as np
import tensorflow as tf
from monotonic_nn.layers import MonoDense

# Features: [log_vehicle_age, log_driver_age, log_vehicle_value,
#            ncd_level, region_effect]
# Monotonicity directions:
#   +1 = increasing (higher vehicle age -> higher risk)
#   -1 = decreasing (higher NCD -> lower risk)
#    0 = unconstrained
monotonicity_indicators = [1, 1, 0, -1, 0]

n_features = len(monotonicity_indicators)
indicator_tensor = tf.constant(monotonicity_indicators, dtype=tf.float32)

inputs = tf.keras.Input(shape=(n_features,), name="features")
glm_offset = tf.keras.Input(shape=(1,), name="glm_log_mu")

# First hidden layer: partial monotonicity applied
x = MonoDense(
    64,
    activation="elu",
    monotonicity_indicator=indicator_tensor,
    name="mono_hidden_1"
)(inputs)

# Second hidden layer: monotonicity propagated through
x = MonoDense(
    32,
    activation="elu",
    monotonicity_indicator=1,  # all monotone after first layer
    name="mono_hidden_2"
)(x)

# Output: log predicted frequency, added to GLM offset
log_nn_adjustment = MonoDense(1, activation=None, monotonicity_indicator=1)(x)
log_mu = tf.keras.layers.Add()([glm_offset, log_nn_adjustment])
frequency = tf.keras.layers.Activation("exponential", name="frequency")(log_mu)

model = tf.keras.Model(
    inputs=[inputs, glm_offset],
    outputs=frequency
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.Poisson()  # Poisson deviance proxy
)
```

A few points on this implementation. The `MonoDense` layer with `monotonicity_indicator` per feature handles partial monotonicity — some features constrained, others not. After the first layer, the constraint is propagated automatically through subsequent layers if `monotonicity_indicator=1` (all-monotone). The GLM offset entry is the standard CANN pattern: the GLM fits the main effects, and the NN learns residual adjustments. This preserves the GLM prior while adding non-linear interactions where the data supports them.

The constraint guarantee here is architectural: this network cannot produce a lower predicted frequency by increasing vehicle age or driver age, or a higher predicted frequency by increasing NCD, regardless of input values. That guarantee is not conditional on the training data or the penalty weight — it is a property of the network structure.

---

## Decision framework

**Start here:** Are you using XGBoost, LightGBM, or a GBDT ensemble? If yes, use `monotone_constraints` in the booster parameters. Hard architectural constraint, no extra libraries, done.

**If you are fitting a neural network:** Do you need smoothness as well as monotonicity? If yes — and for most commercial actuarial use cases, you do — ICEnet is the only published approach that handles both. Get the code from the paper supplementary (Richman & Wüthrich 2024, doi:10.1017/S174849952400006X). It requires penalty parameter tuning via cross-validation or actuarial judgment, but the mechanism is sound.

**If you only need monotonicity and not smoothness:** Use `uv add monotonic-nn` and the MonoDense layer. Hard architectural guarantee, negligible accuracy cost, Keras-compatible, no TF-lattice dependency.

**If you are on a TF stack and need multiple constraint types** (monotonicity + convexity + pairwise trust): use tensorflow-lattice. Most mature production option.

**If you need post-hoc balance/calibration** (separately from monotonicity): apply the Denuit autocalibration step (local GLM applied to NN outputs) or global renormalisation. These do not require retraining and work on any model type.

**The regulatory test:** if a model review asks "can this model produce a lower premium for a higher-risk driver?", you want to answer with a model property, not a validation result. Architectural constraints give you the property. Post-hoc corrections give you the validation result. Under Consumer Duty outcomes monitoring, these are different claims.

---

## What we are not using yet

MonoKAN (arXiv:2409.11078) is the most promising development in this space. KANs replace the fixed activation functions of MLPs with learnable univariate functions on the edges — splines in the original Kolmogorov-Arnold formulation. This makes the learned functions inherently more interpretable: you can visualise the spline for each input, which directly corresponds to the marginal effect curves actuaries already plot. MonoKAN adds certified monotonicity by constraining the first derivative of the Hermite splines to remain positive. On standard monotonicity benchmarks it outperforms CMNN, lattice networks, and certified monotonic networks on the majority of tasks.

The problem is that there is no pip package. Research code is available on GitHub but is not production-grade. When a stable release appears, this becomes the default recommendation for interpretability-constrained environments: you get ICE-plot style interpretability baked into the model architecture, with certified monotonicity, without needing to maintain your own ICEnet training loop.

For now: GBDT with `monotone_constraints`, or MonoDense for pure NN use cases, and ICEnet from the paper supplementary if smoothness matters.

---

The distinction between "the model cannot do this" and "we checked and it did not do this" sounds philosophical. Under Consumer Duty outcomes monitoring, it is not. Architectural constraints are the actuarial equivalent of underwriting rules that cannot be overridden — they remove a whole category of potential failure. Post-hoc corrections are the equivalent of auditing the output: useful, necessary, but a different claim about the model's behaviour. The choice of which one to use should be deliberate, not defaulted into by whatever library you happened to reach for first.

---

*ICEnet: Richman, R. & Wüthrich, M.V. (2024). 'Smoothness and monotonicity constraints for neural networks using ICEnet.' Annals of Actuarial Science, 18: 712-739. [doi:10.1017/S174849952400006X](https://doi.org/10.1017/S174849952400006X)*

*MonoDense / CMNN: Runje, D. & Shankaranarayana, S.M. (2023). 'Constrained Monotonic Neural Networks.' ICML 2023.*

*Seyam, E. & Osman, M.A.M. (2025). 'Physics-Informed Transformer Networks for Multi-Peril Insurance Pricing.' Statistics, Optimization & Information Computing, 15(3): 2105-2132.*

*Deep Lattice Networks: You, S. et al. (2017). 'Deep Lattice Networks and Partial Monotonic Functions.' NeurIPS 2017.*

*MonoKAN: Kalina, M. et al. (2024). 'MonoKAN: Certified Monotonic Kolmogorov-Arnold Network.' arXiv:2409.11078.*

*Balance property: Wüthrich, M.V. (2021). Statistical Theory and Related Fields. (On the balance property breaking under SGD with early stopping.)*

*Autocalibration: Denuit, M., Charpentier, A. & Trufin, J. (2021). 'Autocalibration and Tweedie-dominance for insurance pricing with machine learning.' Insurance: Mathematics and Economics, 101(B): 485-497.*
