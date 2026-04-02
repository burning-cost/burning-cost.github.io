---
layout: post
title: "ANAM's Monotonicity Insight Is Real. The tensorflow_lattice Dependency Is Not."
date: 2026-04-02
categories: [machine-learning, research]
tags: [nam, ebm, interpretable-models, monotonicity, tensorflow-lattice, insurance-gam, actuarial, arXiv-2509.08467, interpretML]
description: "Laub/Pho/Wong's Actuarial Neural Additive Model has a genuine architectural insight in PWLCalibration monotonicity. It also depends on an unmaintained TensorFlow library. EBM is still the right choice for UK pricing teams."
author: burning-cost
---

Neural Additive Models for insurance pricing is not a new idea. The additive architecture — one subnetwork per feature, interpretable shape functions, interactions handled explicitly — has been available via EBM since 2019 and via the canonical NAM paper (Agarwal et al., 2021) since shortly after. Any new paper in this space has to justify why the existing tools are insufficient.

Laub, Pho & Wong (arXiv:2509.08467) make one genuine architectural argument and several weaker ones. The genuine argument is about monotonicity. The implementation buries it inside an unmaintained dependency. Here is an honest account of where that leaves UK pricing teams.

---

## What ANAM actually does

The paper's architecture — which they call ANAM (Actuarial Neural Additive Model) — is a standard NAM with three modifications worth examining:

**1. PWLCalibration for monotone features.** For variables where monotonicity is required — BonusMalus being the obvious example — ANAM does not use a standard MLP subnetwork. It uses `tfl.layers.PWLCalibration` from Google's TensorFlow Lattice library. This is a piecewise-linear calibration layer with keypoints spaced over the feature range; monotonicity is enforced as a convex constraint during training, not as a post-hoc clamp.

**2. Multiplicative exposure offset.** The model multiplies its final output by exposure, matching the GLM convention. This is actuarially correct and avoids the approximation that `sample_weight` makes in EBM.

**3. Pairwise interactions via 2D TF Lattice.** When an interaction involves a monotone variable, ANAM uses a two-dimensional lattice layer rather than a concatenated subnetwork, preserving monotonicity in both dimensions simultaneously.

The benchmark is beMTPL97 (163,212 Belgian MTPL policies). On Poisson NLL, ANAM's best hyperparameter trial achieves 0.3687 (validation) against EBM's 0.3800 (test). NAM itself gets 0.3788 — outperforming EBM, GLM at 0.3803, GBM at 0.3798, and LocalGLMnet at 0.3797. The spread across all non-EBM neural models is roughly 0.0013 NLL. Whether that gap is meaningful in a business context is a separate question to whether it is statistically significant, and the paper does not address it.

We note: the ANAM figure of 0.3687 is the best validation trial across 130 Bayesian hyperparameter search runs, not a single holdout test number. The comparison is not apples-to-apples.

---

## The genuine contribution: PWLCalibration vs weight clamping

Our [insurance-gam](https://pypi.org/project/insurance-gam/) library implements monotonicity via weight clamping: the Dykstra projection method, which forces ReLU network weights to be non-negative in the monotone subnetworks. This is the standard approach in the literature and it works, but it has a known weakness.

Weight clamping requires ReLU activations throughout the monotone subnetwork — the method breaks if you use any other activation. It also only guarantees monotonicity for the specific activation + clamp combination; it is not a structural guarantee independent of the network architecture.

PWLCalibration is architecturally stronger. The keypoints of the piecewise-linear function are trained under an explicit convexity constraint: the differences between adjacent keypoint values are forced to be non-negative (for increasing) or non-positive (for decreasing). This constraint holds regardless of what activation the rest of the network uses. You can prove the monotone layer is monotone by inspecting the trained keypoint weights — there is no "it happened to be monotone on the training distribution" caveat.

For UK pricing under the Equality Act 2010 and FCA Consumer Duty, this distinction matters. When a pricing committee asks whether BonusMalus is guaranteed to rank worse risks higher, "the weight clamp forces this" is a weaker answer than "the piecewise-linear function is structurally increasing, here are the keypoint weights." The latter is a direct, documentable guarantee.

This is ANAM's real contribution. It is not about benchmark NLL numbers; it is about the form of the monotonicity guarantee.

---

## The problem: tensorflow_lattice is unmaintained

TensorFlow Lattice was published by Google Research around 2019-2020. As of April 2026, the [tensorflow-lattice PyPI package](https://pypi.org/project/tensorflow-lattice/) has not had a substantive release since 2023. It requires pinned TensorFlow versions that are incompatible with current Python environments without significant dependency management work. The ANAM paper provides a research notebook, not a pip-installable package with a documented API.

This is not a minor inconvenience. Pricing model infrastructure in UK insurers typically runs on Python 3.11 or 3.12 with current TensorFlow or PyTorch. Introducing a dependency that requires TF version pinning creates:

- A security surface: you cannot update TF without breaking the lattice layer
- A reproducibility risk: the keypoint constraint code is not maintained
- A governance problem: SS1/23 (PRA model risk management) requires documented, testable model components — an unmaintained external library with no current support is difficult to defend in a model validation

The paper's results cannot be reproduced without resolving the TF Lattice dependency issue. We tried. It is not straightforward.

---

## EBM already does most of this. Not all of it.

InterpretML's [EBM](https://interpret.ml/docs/ebm.html) covers the majority of what ANAM offers:

- Additive structure with one shape function per feature: yes
- Automatic interaction detection (FAST/GA2M): yes, and more principled than ANAM's variance-based ensemble screening
- Poisson deviance objective (since InterpretML >= 0.7): yes
- sklearn-compatible API, pip-installable, actively maintained: yes
- Monotonicity constraints: yes, via post-hoc shape function editing

That last point is the gap. EBM's monotonicity is enforced by editing the shape function after fitting — if the fitted shape is non-monotone in the desired direction at any point, it is flattened. This is a reasonable approximation but not a structural guarantee. The same caveat that applies to weight clamping applies here: you are asserting that the fitted function happens to respect the constraint, not that the architecture enforces it.

For most UK pricing applications, EBM's monotonicity approach is sufficient. For situations where you need to demonstrate structural monotonicity to a regulator or pricing committee — particularly for protected characteristics or legally constrained variables — ANAM's PWLCalibration approach is the cleaner answer. The problem is that the implementation is not usable.

---

## What we would do instead

The PWLCalibration insight is replicable in PyTorch in roughly 100 lines of code. A `MonotonicPWLFeatureNetwork` layer — piecewise linear with constrained keypoint differences — could sit alongside the existing monotone subnetworks in insurance-gam as an alternative to weight clamping. This would give the structural guarantee without the TF Lattice dependency.

We have not built this yet because the governance benefit is marginal relative to the implementation cost for most UK pricing teams. Weight clamping is a defensible position in a model validation; it is not the liability some might fear. But if a pricing committee pushes for a formally documentable monotonicity proof rather than an empirical one, a MonotonicPWLFeatureNetwork is the right tool and we would add it.

For now, our recommendation is straightforward: **use EBM**. It is faster to fit than ANAM, has a better API, is actively maintained, and handles the vast majority of UK motor and home pricing requirements. If you need demonstrably structural monotonicity guarantees rather than empirical ones, open an issue on [insurance-gam](https://github.com/insurance-gam/insurance-gam) and we will prioritise the PWLCalibration port to PyTorch.

---

## What is missing from the paper

Three things that would change our assessment:

**Gini coefficients.** The paper's notebook reports Poisson NLL, RMSE, and MAE. It does not report Gini coefficients. For UK motor pricing, Gini is the standard measure of ranking power. NLL improvements in the third decimal place on beMTPL97 do not translate to pricing value without Gini lift evidence.

**A fair GAM comparison.** The paper's GAM baseline uses pyGAM with gridsearch that took over eleven hours to fit. EBM would fit the same data in minutes and produce a better result. The paper compares ANAM to a badly configured competitor.

**Test set ANAM NLL.** The best ANAM result (0.3687) is validation loss from the best hyperparameter trial across 130 Bayesian search runs. The actual held-out test NLL for the best ANAM configuration is not clearly printed in the notebook as a single comparable figure.

These gaps do not invalidate the PWLCalibration insight. They do make the headline "ANAM outperforms EBM in most cases" unverifiable from the public evidence.

**Paper:** [arXiv:2509.08467](https://arxiv.org/abs/2509.08467) | **InterpretML EBM:** [interpret.ml](https://interpret.ml/docs/ebm.html) | **insurance-gam:** [PyPI](https://pypi.org/project/insurance-gam/) | [GitHub](https://github.com/insurance-gam/insurance-gam)
