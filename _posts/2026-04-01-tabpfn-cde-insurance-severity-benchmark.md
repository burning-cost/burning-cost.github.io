---
layout: post
title: "TabPFN as a Conditional Density Estimator: What the Benchmark Actually Shows for Severity Pricing"
date: 2026-04-01 09:00:00 +0100
categories: [techniques, pricing]
tags: [TabPFN, TabICL, conditional-density-estimation, severity, CDE, benchmark, distributional, CRPS, calibration, thin-data, foundation-models, arXiv-2603-26611, Izbicki-Rodrigues, insurance-severity, insurance-distributional, DRN, MDN, small-data, heavy-tail]
description: "Izbicki and Rodrigues (arXiv:2603.26611, March 2026) benchmark TabPFN-2.5, RealTabPFN-2.5 and TabICL-Quantiles as conditional density estimators across 39 datasets. The thin-data results are genuinely impressive. The large-n results are not. Here is what that means for severity pricing."
math: true
author: burning-cost
---

TabPFN has been generating considerable hype as a "foundation model for tabular data" — the claim being that a single pre-trained transformer, requiring no gradient descent at your data, can beat tuned tree ensembles on small classification problems. Hollmann et al. showed this for classification in [Nature in January 2025](https://www.nature.com/articles/s41586-024-08328-6). The obvious next question for insurance is whether it extends to regression, and specifically to severity modelling — which is not just predicting a mean but estimating a conditional distribution.

Izbicki and Rodrigues ([arXiv:2603.26611](https://arxiv.org/abs/2603.26611), submitted 27 March 2026) have now run that experiment. Their paper benchmarks three tabular foundation models as conditional density estimators (CDEs) against 10+ classical baselines across 39 datasets at training sizes ranging from n=50 to n=20,000. The results are more nuanced than the TabPFN marketing suggests, and they have direct implications for how you should think about distributional severity tools.

---

## What "conditional density estimation" means for severity

A severity model is, at its core, a conditional density estimator. Given a set of rating factors $x$ for a claim, you want $f(y \mid x)$ — the full distribution of claim amounts, not just the mean. The Gamma GLM gives you $f(y \mid x)$ under a Gamma assumption. The DRN in `insurance-severity` gives you a non-parametric refinement of $f(y \mid x)$ that does not assume Gamma. Both are CDEs.

TabPFN-2.5's approach to CDE is different: it outputs a "bar distribution" — a discrete probability mass over 200 binned intervals spanning the response range. Bin widths are uniform after normalising the response. This is converted to a density by dividing by bin width and linearly interpolating to a 200-point grid. TabICL-Quantiles takes yet another approach: it outputs predictive quantiles directly, which are interpolated to a CDF and then numerically differentiated to a density.

Both are reasonable approximations. Both have failure modes at the tail. More on that below.

---

## The benchmark setup

Izbicki and Rodrigues ran three foundation models:

- **TabPFN-2.5** — the standard model (bar distribution, 200-point grid)
- **RealTabPFN-2.5** — additionally fine-tuned on curated real-world tabular datasets
- **TabICL-Quantiles** — direct quantile outputs, interpolated to density

Against a wide set of baselines including: Gamma GLM (ridge), Lognormal GLM (homo/hetero), Student-t, BART variants, FlexCode-RF (cosine basis expansion on random forest), Mixture Density Networks (MDN), neural spline flows, and a binned-softmax MLP (CatMLP).

Evaluation metrics: CDE loss (a proper scoring rule for densities), log-likelihood, CRPS, PIT-KS calibration statistic, and 90% empirical coverage. Training sizes: n ∈ {50, 500, 1,000, 5,000, 10,000, 20,000} across 39 OpenML regression datasets.

Code and data at [github.com/rizbicki/tabDensityComparisons](https://github.com/rizbicki/tabDensityComparisons).

---

## What the numbers actually say

The headline ranking by average CDE loss (lower rank = better):

| n | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| 50 | RealTabPFN-2.5 (2.2) | TabPFN-2.5 (2.3) | ... |
| 1,000 | RealTabPFN-2.5 (2.1) | TabPFN-2.5 (2.3) | TabICL (2.6) |
| 20,000 | TabPFN-2.5 (2.7) | TabICL (2.9) | RealTabPFN-2.5 (3.4) |

Foundation models achieved the best CDE loss on 82% of datasets at n=50 and 92% at n=1,000. That is a strong result. In the paper's headline case study — photometric redshift estimation from the SDSS galaxy survey — TabPFN trained on n=50,000 outperformed every method trained on the full n=500,000 dataset, across CDE loss, CRPS, and log-likelihood simultaneously. A ten-fold data-efficiency advantage.

For CRPS (the metric most actuaries would recognise), TabICL-Quantiles is particularly competitive: average rank 2.1 at n=1,000, 1.7 at n=20,000.

**The calibration story is different.** PIT-KS rank (how well the predictive density integrates to a uniform probability integral transform):

- At n=50 and n=1,000: TabPFN variants rank ~4.4–4.6. Competitive but not dominant.
- At n ≥ 5,000: TabPFN variants deteriorate to ranks 7–9 of approximately 14 methods.
- MDN achieves the best calibration at larger n.

This is the finding that matters most for insurance pricing. Calibration is not an aesthetic property — an uncalibrated severity model misprices risk systematically. A model that is well-calibrated on aggregate but not conditionally calibrated by risk group will underprice the heavy risks and overprice the light ones.

---

## Why calibration breaks down at scale

The deterioration is structurally predictable. TabPFN uses in-context learning: the model has a Bayesian prior over tabular data generation processes baked in during pre-training. At small n, this prior does most of the work — it is like a very flexible parametric assumption that has learned from millions of synthetic datasets. At large n, your observed data should dominate the prior. But TabPFN's architecture cannot fully update away from its pre-trained prior as n grows — the context window is finite, the prior is baked in, and the bar distribution discretisation becomes the binding constraint.

At n=5,000 with a severity distribution spanning £100 to £1M, the 200-bin grid gives bin widths of roughly £5,000. Quantile accuracy at the 99th percentile depends on getting those tail bins right. A grid that coarse cannot resolve the tail of a lognormal or Pareto-tailed distribution with the precision that claims pricing requires.

MDN, by contrast, learns a parametric mixture directly from data — its calibration improves with n because there is no baked-in prior to fight against. This is why MDN is the right choice at scale. It is also why the MDN implementation in [`insurance-severity`](/insurance-severity/) is the one we reach for when n is in the thousands.

---

## The hard constraints

These are not soft limitations — they are structural blockers for production severity use:

**Row cap.** TabPFN-2.5 is capped at n=50,000 training rows. TabICL-Quantiles caps at n=10,000 under default configuration. UK motor frequency has millions of rows. Even a single line of commercial business might exceed 50,000 claims in a mature book. These models simply cannot train on production data as-is.

**Memory.** Both TabPFN variants ran out of memory (OOM) on the CTSlices dataset (d=384 features) at n=20,000 on an NVIDIA RTX 5070 Ti with 16GB VRAM. UK motor and home feature spaces are typically 30–60 features, well inside the tested range — but it is a warning about what happens as feature dimensionality grows.

**No offset/exposure.** TabPFN has no log(exposure) offset term. This is non-negotiable for frequency models. It is less critical for severity (you are conditioning on a claim having occurred), but it means TabPFN cannot be integrated into a proper freq-sev pipeline without careful feature engineering.

**Tail resolution.** The 200-point bar distribution is inadequate for tail quantile pricing. If your severity spans two orders of magnitude, your grid has ~1% granularity in log-space. That is not enough to price excess-of-loss reinsurance attachment points or to estimate 99th percentile large loss loadings.

**No heavy-tail evaluation.** The paper does not evaluate on Gamma, Lognormal, or Pareto-tailed distributions. The 39 datasets are drawn from general OpenML regression tasks and one astrophysics dataset. Insurance severity is not in the benchmark. We do not know how the 200-bin grid performs on distributions with a £100 modal claim and a £2M 99th percentile.

---

## Where it would genuinely help

Despite all of the above, there is a real use case: **thin-data severity segments as a benchmark tool**.

Consider a specialist commercial lines segment — say, equine mortality, fine art, or a new cyber product — where you have fewer than 200 claims. Your Gamma GLM has converged in the sense that it has fit numbers to parameters, but the MLE standard errors are wide enough to be uninformative. The question is whether your distributional assumptions (Gamma, Lognormal, spliced) are introducing model misspecification on top of the sampling noise.

Running TabPFN-2.5 as a non-parametric CDE alongside your fitted parametric model gives you a calibration check. If TabPFN achieves substantially better CRPS than your fitted Gamma GLM on held-out claims from that segment, the Gamma assumption is probably wrong and you should consider a spliced or DRN model. If they agree, you have some evidence the parametric family is adequate for the data volume you have.

This is the diagnostic role. No fitting, no hyperparameter tuning — just run TabPFN in-context and compare.

```python
from insurance_tabpfn import TabPFNSeverity

# Thin-data segment: 180 commercial property claims
model = TabPFNSeverity()
model.fit(X_train, y_train)  # in-context: no gradient descent

# Evaluate against your fitted Gamma GLM
from insurance_severity import GammaSeverityGLM

glm = GammaSeverityGLM(exposure_col=None)
glm.fit(X_train, y_train)

# Compare CRPS on held-out claims
# If tabpfn_crps << glm_crps: parametric assumption is wrong
# If roughly equal: Gamma adequate for this n
```

We would not ship TabPFN CDE as the production severity model. We would use it to stress-test the parametric assumption on segments too thin to do it any other way.

---

## What to use instead at scale

The paper's own findings point toward MDN for calibration at larger n, and that matches our production experience. The `insurance-severity` library has had an MDN implementation since v0.3, built specifically for insurance with:

- Lognormal mixture components (appropriate for right-skewed severity)
- Log-normal re-parameterisation that prevents numerical instability on large claims
- Isotonic post-hoc recalibration to correct residual calibration error
- Polars API with exposure handling

At n > 2,000, the MDN dominates TabPFN on calibration and is not constrained by a row cap. For n > 10,000, `TweedieGBM` or `GammaGBM` from `insurance-distributional` remain the benchmark — they are not CDEs in the full sense (they produce a parametric distribution given the mean), but for the majority of pricing work where you need the mean and a credible interval, they are faster, interpretable, and deployable on CPU.

For non-parametric density on larger segments, the DRN (Distributional Refinement Network) in `insurance-severity` produces a full conditional density by refining a GLM-predicted mean with a set of residual quantile adjustments, calibrated via isotonic regression. It handles the heavy tail better than TabPFN's 200-bin grid because it can learn unbounded quantile adjustments rather than being constrained to a fixed grid.

The practical toolkit, by segment size:

| Segment size | Tool | Why |
|---|---|---|
| n < 200 | TabPFN-2.5 (diagnostic), Gamma GLM (production) | Zero-training density check against parametric assumption |
| n = 200–2,000 | Gamma/Lognormal GLM + isotonic recalibration | MLE has converged; parametric is reliable if family is right |
| n = 2,000–50,000 | MDN (`insurance-severity`) or DRN | Full non-parametric density, well-calibrated at this scale |
| n > 50,000 | GammaGBM / TweedieGBM (`insurance-distributional`) | Tree model captures interactions; production-ready |

---

## The honest verdict

TabPFN is not a severity pricing tool. Its row cap, tail resolution, calibration degradation at scale, and lack of offset support all make it unsuitable for production use in anything except the thinnest specialist segments — and even there it is a benchmark rather than a model you would rate off.

What the Izbicki and Rodrigues paper does confirm is the genuine data-efficiency advantage at small n. At n=50, TabPFN produces better densities than any classical alternative, including MDN and neural spline flows. That is a real result. It means that for the 180-claim equine mortality segment or the 120-claim fine art portfolio, TabPFN gives you a calibration check that was previously unavailable — you had no non-parametric reference to compare your Gamma GLM against.

That diagnostic role is narrow but genuine. We will keep `insurance-tabpfn` as the entry point for it. For everything else, the production stack is `insurance-distributional` and `insurance-severity`.

---

## The paper

Izbicki, Rafael, and Pedro L. C. Rodrigues. "Benchmarking Tabular Foundation Models for Conditional Density Estimation in Regression." arXiv:2603.26611. Submitted 27 March 2026.

Code: [github.com/rizbicki/tabDensityComparisons](https://github.com/rizbicki/tabDensityComparisons)

---

## Related posts

- [Foundation Models for Thin Segments: TabPFN and TabICLv2 in Insurance Pricing](/2026/03/13/insurance-tabpfn/) — the `insurance-tabpfn` library overview
- [TabPFN vs CatBoost vs GLM on freMTPL2: Why Exposure Offset Matters](/2026/03/28/tabpfn-catboost-glm-fremtpl2-exposure-offset/) — TabPFN failure mode on frequency data without offset
- [GLMs Predict Means. DRN Predicts Everything Else.](/2026/03/10/distributional-refinement-network-insurance/) — the DRN for full conditional density in production
- [insurance-distributional: TweedieGBM, GammaGBM and the Full Distributional Toolkit](/2026/03/05/insurance-distributional/) — the production distributional modelling library
- [Mixture Density Networks for Insurance Severity: When the Gamma Isn't Enough](/2026/03/28/mixture-density-networks-insurance-severity/) — MDN as the right non-parametric CDE at larger n
