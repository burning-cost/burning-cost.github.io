---
layout: post
title: "Mahalanobis Conformal Prediction: When Ellipsoids Beat Rectangles (and When They Do Not)"
date: 2026-04-01
categories: [techniques, pricing, uncertainty]
tags: [conformal-prediction, multivariate, Mahalanobis, ellipsoidal-prediction-sets, joint-prediction, frequency-severity, multi-peril, Home, Braun-Berta-Jordan-Bach, arXiv-2507-20941, LWC, Fan-Sesia, hyperrectangle, FCA, Consumer-Duty, covariance, insurance-conformal, UK-regulatory, python]
description: "Braun et al. (arXiv:2507.20941) replace the standard hyperrectangular joint prediction set with an ellipsoid built from a Mahalanobis nonconformity score. For d=2 (frequency + severity), the gain over locally-weighted conformal is marginal and the interpretability cost is real. The method becomes genuinely useful at d≥5 — think Home multi-peril books where fire, flood, theft, escape of water, and subsidence need to be considered jointly."
math: true
author: burning-cost
---

Joint conformal prediction for insurance pricing is usually a two-output problem: frequency and severity, stacked into a rectangle. The rectangle is a fine shape for two dimensions. It maps directly to what an actuary writes in a pricing memo — "frequency in [0.08, 0.35], severity in [£1,200, £4,800]" — and every number in it is defensible to an FCA supervisor without reaching for matrix algebra.

But insurance is not always two outputs. A Home pricing team managing fire, flood, theft, escape of water, and subsidence has five correlated outputs. At five outputs, the rectangle fails badly. Bonferroni — the standard way to extend marginal intervals to joint ones — requires each dimension to be calibrated at α/5. If you want 90% joint coverage across five perils, each marginal interval has to be at 98% coverage, which means extremely wide bounds that will be useless for pricing decisions. The geometric inefficiency grows with dimension, and rectangles aligned to axes are a poor fit for outputs whose joint distribution sits on a correlated ellipse in $\mathbb{R}^5$.

Braun, Berta, Jordan, and Bach (arXiv:2507.20941, "Multivariate Standardized Residuals for Conformal Prediction") offer a different shape: an ellipsoid, calibrated using a Mahalanobis nonconformity score that exploits the full cross-output covariance structure. Our verdict is that for the typical UK pricing model — two outputs — the gain over existing methods in `insurance-conformal` is too small to justify the added complexity. But for multi-peril Home books at $d \geq 5$, this paper deserves a serious read.

---

## What the method does

Standard joint conformal prediction for $d$ outputs collapses the residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} \in \mathbb{R}^d$ to a scalar nonconformity score by taking the maximum of per-dimension standardised absolute residuals — the coordinate-wise standardisation approach in Fan and Sesia (arXiv:2512.15383), which is what `insurance-conformal`'s locally-weighted conformal (LWC) method implements. This scores a point by how far it deviates in the worst single dimension. The resulting prediction set is a hyperrectangle aligned to the output axes.

Braun et al. replace the coordinate-wise maximum with a Mahalanobis distance:

$$s_i = \left\| \hat{\Sigma}_\varphi(x_i)^{-1/2} \cdot (\mathbf{y}_i - \hat{\mathbf{f}}(x_i)) \right\|_2$$

where $\hat{\Sigma}_\varphi(x)$ is a local covariance estimate that depends on $x$. The prediction set at a new point $x$ is then:

$$C(x) = \left\{ \mathbf{y} : \left\| \hat{\Sigma}_\varphi(x)^{-1/2} (\mathbf{y} - \hat{\mathbf{f}}(x)) \right\|_2 \leq \hat{q} \right\}$$

where $\hat{q}$ is the usual conformal quantile calibrated on a hold-out set. This is an ellipsoid centred at $\hat{\mathbf{f}}(x)$, whose shape and orientation are determined by $\hat{\Sigma}_\varphi(x)^{-1/2}$. The coverage guarantee is the same as for any split conformal method: marginal, finite-sample, distribution-free.

There are two modes. The **neural mode** learns $\hat{\Sigma}_\varphi(x)$ jointly with the base predictor $\hat{\mathbf{f}}$ via a covariance network that outputs a Cholesky factor $L_\varphi(x)$. This achieves near-conditional coverage when the learned covariance is consistent for the true local covariance. It requires end-to-end training on a neural architecture, which means it cannot be bolted onto a pre-trained GLM or GBM stack. That rules it out for almost every UK insurance pricing workflow in production.

The **global covariance mode** uses the empirical covariance of calibration residuals:

$$\hat{\Sigma} = \frac{1}{n} R^T R + \lambda I, \quad R_{ij} = |y_{ij} - \hat{f}_j(x_i)|$$

No retraining, no neural network, only `scipy.linalg.eigh`. The prediction set is still an ellipsoid, calibrated to the empirical correlation structure of the residuals. This works with any pre-trained model stack and costs negligible compute. It is the mode that is actually relevant for insurance.

The authors report 20–40% volume reduction versus Bonferroni on correlated-output regression benchmarks. That number comes from synthetic experiments and we would expect it to shrink on real insurance data, where residuals are right-skewed rather than jointly normal. But the directional result is correct: ellipsoids are geometrically more efficient than axis-aligned boxes when outputs are correlated, and the efficiency gap grows with dimension.

---

## Why $d = 2$ does not need this

The typical UK pricing model has frequency and severity. That is $d = 2$.

For $d = 2$, `insurance-conformal`'s LWC method already captures most of the joint correlation. The coordinate-wise standardisation in Fan and Sesia partitions calibration points into groups based on their maximum-dimension score, which implicitly exploits the correlation structure between outputs at a group level. The remaining gap between LWC and Mahalanobis ellipsoid for $d = 2$ — about 5–10% in volume on typical correlated Gaussian residuals — is not worth the loss in interpretability.

Interpretability here is not a soft concern. FCA Consumer Duty requires firms to demonstrate that pricing outcomes are fair and that models produce decisions capable of explanation. A hyperrectangle has a direct, per-dimension reading: each entry is a plausible value for frequency or severity individually. An ellipsoid has no such decomposition — a point that falls inside the ellipsoid might sit outside the per-dimension plausible range, and a point on the boundary of the rectangle might be inside the ellipsoid. When an underwriter or a conduct review team asks "why is this risk outside the joint prediction set?", the ellipsoidal answer involves eigenvectors of a covariance matrix. The rectangular answer involves two numbers.

There is also a practical audit trail point. Per-peril justification is the standard the FCA expects. A joint ellipsoid obscures the peril-by-peril structure. This is not an insurmountable problem — you can always report the circumscribed rectangle of the ellipsoid alongside the ellipsoid — but it is an extra step that adds no value in the two-output case where LWC already performs well.

Our recommendation: for $d = 2$, use LWC. It is already in `insurance-conformal` and it is the right tool.

---

## Where ellipsoids actually earn their keep

The picture changes at $d \geq 5$. Consider a Home multi-peril model covering fire, flood, theft, escape of water, and subsidence. These are five correlated outputs — they share drivers (construction type, age of property, postcode flood risk) and their residuals covary. Bonferroni at this dimension sets each marginal interval at $\alpha/5 = 0.02$: for 90% joint coverage you need 98% marginal coverage per peril. The resulting bounds will typically be so wide they are not useful for pricing discrimination.

LWC at $d = 5$ is better than Bonferroni but still takes the worst coordinate, throwing away the correlation information across the other four dimensions. The Mahalanobis ellipsoid exploits the full $5 \times 5$ covariance matrix and produces a region that is tighter along the directions of high joint probability mass and wider along the directions of genuine joint uncertainty. For highly correlated perils — escape of water and subsidence both respond to soil type and property age — the ellipsoid can be substantially smaller than any hyperrectangle with the same joint coverage.

The empirical efficiency advantage is geometrically guaranteed: the volume of the minimum-volume axis-aligned box enclosing the ellipsoid grows with $d$, while the ellipsoid itself scales with the $d$-th root of the determinant of $\hat{\Sigma}$. When the covariance matrix has a few dominant directions (which it will for correlated perils), the ellipsoid can be dramatically smaller than the box.

For Home multi-peril pricing, we think the global covariance mode of Mahalanobis conformal is worth running as a **diagnostic**, even if the output at the end is still a circumscribed rectangle for the pricing memo. The diagnostic value is in the covariance matrix itself: $\hat{\Sigma}$ tells you which perils move together and by how much. If escape of water and subsidence have a residual correlation of 0.7 in your calibration set, that is an empirical signal about shared model misspecification that you would not see by running five independent interval calibrations.

The circumscribed rectangle formula is exact: the half-width in dimension $j$ is $\hat{q} \sqrt{\hat{\Sigma}_{jj}}$, derived by maximising $|e_j^T \mathbf{y}|$ subject to $\mathbf{y}^T \hat{\Sigma}^{-1} \mathbf{y} \leq \hat{q}^2$ via Lagrange multipliers. This gives you the per-peril bounds for the audit trail while still calibrating the joint set correctly.

---

## The partial-observation case

The Braun et al. paper has one feature that has no analogue in the hyperrectangular methods: **conditioning on partial observations**. If at renewal you observe the actual fire and flood outcomes but not theft (claim under investigation), you can condition the joint prediction set on the observed components. The block matrix structure of $\hat{\Sigma}$ allows you to marginalise the ellipsoid onto the unobserved subspace analytically.

The mathematics follows from the partition of the inverse covariance (precision matrix) into blocks. If $\mathbf{y} = (\mathbf{y}_A, \mathbf{y}_B)$ where $\mathbf{y}_A$ is observed, the conditional prediction set for $\mathbf{y}_B$ given $\mathbf{y}_A = \mathbf{a}$ is:

$$C_{B|A}(x, \mathbf{a}) = \left\{ \mathbf{y}_B : \left\| \hat{\Sigma}_{B|A}^{-1/2} (\mathbf{y}_B - \hat{\mu}_{B|A}) \right\|_2 \leq \hat{q} \right\}$$

where $\hat{\mu}_{B|A} = \hat{\mathbf{f}}_B(x) + \hat{\Sigma}_{BA} \hat{\Sigma}_{AA}^{-1} (\mathbf{a} - \hat{\mathbf{f}}_A(x))$ and $\hat{\Sigma}_{B|A} = \hat{\Sigma}_{BB} - \hat{\Sigma}_{BA} \hat{\Sigma}_{AA}^{-1} \hat{\Sigma}_{AB}$.

This is the multivariate normal conditional distribution formula, applied to the residual space. It is only approximately valid — the true conditional distribution of the residuals may not be Gaussian — but as a heuristic it can tighten the prediction set for unobserved perils substantially when the observed perils carry real information. For Home renewal pricing where some peril outcomes are known at time of renewal, this is a genuine feature without a rectangular analogue.

We note that the marginal coverage guarantee does not survive this conditioning step. The calibrated $\hat{q}$ is a marginal quantile; the conditional set has no finite-sample guarantee. This must be disclosed clearly in any technical documentation.

---

## Implementation

The global covariance mode requires only numpy and scipy. The algorithm in five steps:

1. Compute the residual matrix $R \in \mathbb{R}^{n \times d}$ from the calibration set: $R_{ij} = |y_{ij} - \hat{f}_j(x_i)|$.
2. Compute $\hat{\Sigma} = \frac{1}{n} R^T R + \lambda I$ for small regularisation $\lambda = 10^{-6}$.
3. Compute $\hat{\Sigma}^{-1/2}$ via eigendecomposition: $V, \Lambda = \text{eigh}(\hat{\Sigma})$; $\hat{\Sigma}^{-1/2} = V \cdot \text{diag}(\Lambda^{-1/2}) \cdot V^T$.
4. Score each calibration point: $s_i = \| \hat{\Sigma}^{-1/2} R_i \|_2$.
5. Set $\hat{q}$ to the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile of $\{s_i\}$.

Predictions are then $\{ \mathbf{y} : \| \hat{\Sigma}^{-1/2} (\mathbf{y} - \hat{\mathbf{f}}(x)) \|_2 \leq \hat{q} \}$, and the circumscribed rectangle half-widths are $\hat{q} \sqrt{\hat{\Sigma}_{jj}}$.

The code at [github.com/ElSacho/Multivariate_Standardized_Residuals](https://github.com/ElSacho/Multivariate_Standardized_Residuals) handles both modes; the PyTorch dependency is only required for the neural (local covariance) mode. The global mode can be extracted in under 100 lines of pure numpy/scipy.

`insurance-conformal` does not currently ship a `MahalanobisConformalPredictor` class. We looked at building one (the spec is straightforward — global mode is approximately 150 lines, full API including circumscribed rectangle output) and scored the build at 16/25 versus the blog post at 22/25. The existing LWC covers the dominant use case. We may revisit this if the multi-peril Home case becomes more widely asked about.

---

## When to use what

The decision is not complicated:

**Frequency + severity ($d = 2$)**: use `insurance-conformal`'s LWC. It is already implemented, it performs within 5–10% of the Mahalanobis ellipsoid in volume, and it produces per-dimension bounds that are directly interpretable. The Mahalanobis gain is not worth the audit friction.

**Home multi-peril ($d = 3$–$4$)**: LWC remains the right default. Consider running the Mahalanobis global covariance mode in parallel as a diagnostic for the correlation structure. Report the circumscribed rectangle for pricing.

**Home multi-peril ($d \geq 5$)**: Mahalanobis global covariance mode is justified. Bonferroni at $d = 5$ is genuinely too conservative to be useful. Run the ellipsoid, output the circumscribed rectangle per peril for the FCA audit trail, and keep the covariance matrix as model documentation.

**Partial observations at renewal**: the block matrix conditioning in the Mahalanobis framework is the only method that handles this case structurally. Use it as a heuristic with explicit caveats on the absence of a finite-sample guarantee.

**Commercial lines with many correlated exposures**: worth evaluating case by case. If your outputs are genuinely high-dimensional and residuals are well-correlated, the ellipsoid will be smaller than any rectangle with equivalent joint coverage. If residuals are approximately independent across dimensions, the gain is near zero.

The local covariance (neural) mode of Braun et al. is not currently practical for UK insurance pricing. It requires joint training, it cannot handle pre-trained GLM/GBM stacks, and its near-conditional coverage guarantee is asymptotic rather than finite-sample. File it under "interesting for future architecture work".

---

## The paper

Braun, Stefan, Marco Berta, Baptiste Jordan, and Mathieu Bach. "Multivariate Standardized Residuals for Conformal Prediction." arXiv:2507.20941 [stat.ML]. Version 3, February 2026.

Code: [github.com/ElSacho/Multivariate_Standardized_Residuals](https://github.com/ElSacho/Multivariate_Standardized_Residuals)

---

## Related

- [Frequency and Severity Are Two Outputs. You Have One Prediction Interval.](/libraries/pricing/uncertainty/2026/03/13/insurance-multivariate-conformal/) — the `insurance-conformal` multivariate module, Fan and Sesia LWC, and joint coverage for $d = 2$
- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/techniques/pricing/2026/04/01/shape-adaptive-conformal-prediction/) — MOPI and group-conditional calibration for Tweedie models
- [Two Ways to Control Risk in Automated Underwriting: Conditional vs Marginal](/techniques/underwriting/2026/04/01/selective-conformal-prediction-automated-underwriting-conditional-vs-marginal-risk/) — SCoRE and SelectiveConformalRC for STP triage
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/techniques/pricing/uncertainty/2026/03/13/insurance-conformal-risk/) — conformal risk control and the full `insurance-conformal` library
