---
layout: post
title: "Kolmogorov-Arnold Networks for Insurance Pricing: What Actuaries Should Know"
date: 2026-04-16
categories: [techniques, research]
tags: [KAN, neural-networks, interpretability, monotonicity, GLM, pricing, regulatory, splines, symbolic-regression, MonoKAN, KANN, mortality, actuarial]
description: "A new neural architecture published in ASTIN Bulletin 2026 replaces fixed activation functions with learnable splines on edges. This is not another GBM variant. It has a genuinely different interpretability story, certified monotonicity, and a credible path to regulatory compliance — with honest limitations attached."
---

Every GLM in UK non-life pricing encodes a human choice: the log link. The actuary decides that frequency is log-linear in the predictors. Most of the time this is approximately right, and GLM diagnostics do not obviously reject it, so the choice stands. But it is a choice, not a discovery.

Kolmogorov-Arnold Networks (KAN) learn the functional form instead of assuming it. The splines that connect inputs to outputs are trained, not fixed. This is a genuinely different architecture from the MLPs underneath most neural pricing models, and it produces a different kind of interpretability — not post-hoc approximation, but the model itself expressed as curves you can plot and, in simple cases, read as closed-form expressions.

In February 2026, the ASTIN Bulletin published the first actuarial application of KANs: Xu, Badescu and Pesenti's mortality modelling paper, *What KAN mortality say: smooth and interpretable mortality modeling using Kolmogorov-Arnold networks* (ASTIN Bulletin 56(1):32–59, DOI: [10.1017/asb.2025.10079](https://doi.org/10.1017/asb.2025.10079)). There is no published non-life pricing application yet. We think there will be within eighteen months, and pricing actuaries should understand the architecture now rather than when someone asks them to validate one.

---

## The architecture in plain English

A standard MLP has neurons at each node. Each neuron applies a fixed activation — ReLU, sigmoid, or similar — to a weighted sum of its inputs. The learning happens in the weights; the activation shape is fixed by the designer.

KAN inverts this. There are no fixed activations and no scalar weights in the traditional sense. Instead, each *edge* between nodes carries a learnable univariate spline function — a B-spline by default, a Hermite spline in the monotone variant. The "weights" in a KAN are spline coefficients. What the network learns is a collection of curves, one per edge, that compose together to produce the output.

This structure is grounded in the Kolmogorov-Arnold representation theorem, which states that any continuous multivariate function can be expressed as a composition of univariate continuous functions. A two-layer KAN approximates exactly this decomposition.

The consequence for interpretability is direct. After training, you can extract the spline on any edge and plot it: this curve shows you how input feature *j* influences intermediate node *k*. You are not asking a post-hoc explainability method what the model "thinks" — you are reading the model. The splines are the model.

A further step is available in simple, well-trained KANs: symbolic regression. The pykan library ([github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)) includes routines that try to snap each spline to a closed-form expression — logarithm, power law, polynomial, piecewise-linear. If a KAN frequency model's edge from vehicle age to a hidden node snaps cleanly to a log function, the model is telling you that the relationship is log-linear. If it snaps to a fractional power, the model has found something the GLM would have missed. Either way, you get a formula you can put in a model governance document.

The original foundation paper is Liu et al. (2024), *KAN: Kolmogorov-Arnold Networks*, ICLR 2025 ([arXiv:2404.19756](https://arxiv.org/abs/2404.19756)).

---

## Why this is different from GAMs and EBMs

We have covered GAMs, EBMs, and NAMs at length. The honest comparison:

A GAM (whether fitted by penalised splines or cyclic gradient boosting) models the outcome as a sum of univariate smooth functions of each input. The functional form of each smooth is flexible, but the structure is *additive*. Interactions must be added explicitly. An EBM can fit pairwise interactions, but they still compose additively.

KAN is not additive in the same sense. The spline functions sit on edges in a multi-layer graph. The composition of spline layers can represent more complex functional relationships than a sum of univariate smooths, more naturally. A two-layer KAN with a five-node hidden layer and nine inputs has 9×5 + 5×1 = 50 edge-splines, each learning a curve — this is not a GAM with interactions bolted on. It is a different model class.

The comparison to SHAP-based interpretability of GBMs is also worth being precise about. SHAP gives you an *attribution* — how much does feature *j* contribute to this prediction, on average. It is an approximation computed after the model is trained. A KAN spline is not an attribution: it is the actual transformation the model applies. These answer different questions. For model governance, being able to say "here is the curve the model applies to driver age" is stronger than "here is the average contribution of driver age across the portfolio."

---

## The ASTIN Bulletin 2026 result

Xu, Badescu and Pesenti extended the CANN architecture pattern to KAN. CANN (Combined Actuarial Neural Network) works by adding a neural network correction to a classical model's output — GLM skip connection plus neural adjustment. KANN does the same but with a KAN layer instead of an MLP layer. They tested three variants on 34 populations from the Human Mortality Database:

- **KANN[2,1]**: A pure two-layer KAN with no classical model initialisation.
- **KANNLC**: KAN initialised from Lee-Carter model coefficients, then fine-tuned on mortality data.
- **KANNAPC**: KAN initialised from Age-Period-Cohort model coefficients, then fine-tuned.

The initialisation variants — KANNLC and KANNAPC — outperform the MLP-based CANN on most of the 34 populations. The advantage is most visible in the smoothness of recovered age, period, and cohort effects. Mortality surfaces from KANNLC are smooth enough to be directly usable in a reserving context; the MLP-based CANN sometimes produces jagged period effects that require post-processing.

This is a life insurance application. The result cannot be directly imported into non-life frequency modelling. But it establishes several things that matter for pricing actuaries: KANs train stably on actuarial datasets; the CANN initialisation pattern works as well for KAN as it does for MLP; and the interpretability claim — smooth, plottable splines — holds on real data, not just toy examples.

No non-life pricing application using KAN has been published as of April 2026. This is the gap.

---

## MonoKAN and regulatory compliance

The most immediate regulatory application is certified monotonicity. Inchingolo, Ferroni and Bacciu (2024) published MonoKAN ([arXiv:2409.11078](https://arxiv.org/abs/2409.11078)), which replaces B-splines with cubic Hermite splines and constrains the spline derivative values at control points to be non-negative. The paper proves that this construction guarantees monotonicity — not as a penalty, not approximately, but as a mathematical property of the trained model.

Why does this matter for UK non-life pricing specifically?

The FCA's Consumer Duty obligation requires actuaries to be able to explain why a 19-year-old driver pays more than a 35-year-old, and to do so in terms that a reasonably sophisticated regulator can scrutinise. The PRA's January 2026 Dear CEO letter expects firms to demonstrate robust justification for model assumptions — a standard that implicitly follows the principles of PRA SS1/23 (the Model Risk Management supervisory statement written for banks) even though no direct insurance equivalent exists. When a GBM enforces monotonicity via a training flag, the guarantee is structural but not certified: the model *should* be monotone in the specified feature, but the guarantee degrades at feature values far from the training distribution. MonoKAN's guarantee is architectural — the model cannot produce a non-monotone output for the specified factors regardless of input values.

This matters for factors where monotonicity is a regulatory and commercial baseline, not an empirical hypothesis: vehicle age increasing risk, no-claims bonus increasing discount, years of driving experience generally decreasing risk above a threshold. In [our post on constrained EBMs](/2026/04/05/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/), we found that unconstrained EBMs violate young-driver monotonicity in 31% of training runs. A MonoKAN would not: the constraint is baked into the architecture.

The MonoKAN paper benchmarks against monotone MLP baselines (MonoNN, COMET, and others) and outperforms all of them with exact monotonicity. There is no public MonoKAN repository at the time of writing — it is implementable in PyTorch from the paper, but requires some engineering. This is a friction point.

---

## The honest limitations

**Training speed.** KANs train 2–5x slower than MLPs of comparable parameter count on tabular data. Compared to XGBoost or LightGBM, the gap is larger. Poeta et al. (2024), *A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data* ([arXiv:2406.14529](https://arxiv.org/abs/2406.14529)), tested KANs across 31 tabular benchmarks and found that KANs are competitive with MLPs but do not consistently outperform gradient boosted trees. On accuracy alone, XGBoost remains the default for tabular insurance data. KAN does not improve on it.

**Ecosystem maturity.** pykan exists and is pip-installable. It is a research library, not a production-grade framework. The documentation is good for the core paper's use cases — function approximation, symbolic regression on toy examples — and thinner on production concerns: custom loss functions, exposure offsets for Poisson deviance, handling of categorical features. MonoKAN has no public repository. Anyone building a non-life KAN pricing model today is doing engineering work as well as modelling work.

**Symbolic regression reliability.** The ability to snap splines to closed-form expressions is compelling in principle. In practice, it works well on clean, low-dimensional function approximation tasks and degrades on noisy, high-dimensional real datasets. Whether a KAN trained on French MTPL frequency data will produce readable symbolic expressions is an open empirical question. The ASTIN Bulletin paper does not attempt symbolic extraction on the mortality surfaces — it reports the smoothness of the splines, not their closed-form approximations.

**No non-life validation.** The ASTIN Bulletin paper works on life mortality data. Mortality surfaces are relatively smooth, well-studied, and structurally different from non-life claim frequency and severity distributions. We genuinely do not know how KAN performs on Poisson-distributed claim frequency with exposure offsets, sparse high-severity tails, or the categorical feature-heavy structures common in UK motor data. This is not a theoretical concern. It is an empirical gap that needs to be filled with published benchmarks before anyone should put a KAN model into production pricing.

---

## Our assessment

KAN is not a replacement for your GBM. It is not faster, not more accurate on standard tabular benchmarks, and not production-ready out of the box.

What it offers is a genuinely different interpretability story. Not post-hoc attribution — the model expressed as curves. Not soft monotonicity enforcement — mathematical guarantees. Not a parametric link function choice — a learned functional form that might tell you something about your data you did not already know.

The ASTIN Bulletin 2026 paper is a credible proof of concept on actuarial data. MonoKAN is the most immediately applicable piece: if you are building a regulatory-grade pricing model and you need certifiable monotonicity in specified factors, MonoKAN offers something GBMs and standard EBMs do not. The architecture is sound and the paper is peer-reviewed.

We think the first non-life KAN pricing paper will appear within eighteen months. The gap is obvious, the library is available, and the regulatory demand for interpretable, certifiably monotone models is building. When that paper appears, pricing teams that understand the architecture will be able to evaluate it quickly. Teams that encounter it cold will not.

The practical question for a UK pricing team today is whether to invest in exploratory work. Our view: if you have a research budget and a researcher comfortable with PyTorch, a KAN frequency benchmark on your motor data against your existing GBM is worth running. Not to replace the GBM — to understand whether the splines reveal anything useful and whether MonoKAN's certified monotonicity would simplify your model governance process. If the splines do not add insight and the performance is inferior, you have spent a few researcher-weeks and learned something real. If the splines reveal structure the GBM was obscuring, you have found something the published literature has not yet reported.

We are planning to build `insurance-kan` — a pykan wrapper with Poisson frequency loss, exposure offsets, Gamma severity, and MonoKAN integration — once the current pipeline of open-source libraries stabilises. When we do, we will test it on freMTPL2 and report the results honestly, including if KAN turns out to be less useful for non-life pricing than the architecture suggests it should be.

---

## References

- Liu, Z., Wang, Y., Vaidya, S., et al. (2024). 'KAN: Kolmogorov-Arnold Networks.' ICLR 2025. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756).
- Xu, J., Badescu, A. & Pesenti, S. (2026). 'What KAN mortality say: smooth and interpretable mortality modeling using Kolmogorov-Arnold networks.' *ASTIN Bulletin* 56(1):32–59. DOI: [10.1017/asb.2025.10079](https://doi.org/10.1017/asb.2025.10079).
- Inchingolo, R., Ferroni, F. & Bacciu, D. (2024). 'MonoKAN: Certified Monotonic Kolmogorov-Arnold Network.' *Neural Networks*. [arXiv:2409.11078](https://arxiv.org/abs/2409.11078).
- Poeta, E., Giobergia, F., Pastor, E., Cerquitelli, T. & Baralis, E. (2024). 'A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data.' [arXiv:2406.14529](https://arxiv.org/abs/2406.14529).

---

Related posts:
- [Does Monotonicity-Constrained EBM Actually Work for Insurance Pricing?](/2026/04/05/does-monotonicity-constrained-ebm-actually-work-for-insurance-pricing/)
- [The PRA Just Named AI a Supervisory Theme. What That Means for Your Pricing Models.](/2026/04/10/pra-ai-supervisory-theme-pricing-models/)
- [Actuarial Neural Additive Model: What the Paper Actually Does](/2026/03/25/actuarial-neural-additive-model-anam-arxiv-2509-08467/)
