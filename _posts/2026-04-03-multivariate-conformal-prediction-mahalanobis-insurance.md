---
layout: post
title: "Your Joint Prediction Sets Are 20–40% Too Wide"
date: 2026-04-03
categories: [conformal-prediction, insurance-pricing]
tags: [conformal-prediction, multivariate, mahalanobis, ellipsoid, bonferroni, frequency-severity, multi-peril, home-insurance, motor-insurance, insurance-conformal, arXiv-2507.20941, braun, berta, jordan, bach, fan-sesia, arXiv-2512.15383, covariance, prediction-sets, uncertainty-quantification, actuarial]
description: "The Bonferroni correction for joint frequency-severity prediction sets is conservative by construction. Braun et al. (arXiv:2507.20941) show that covariance whitening produces ellipsoidal sets 20–40% tighter by volume. Here is what that means for insurance pricing teams, where it helps, and where it does not."
math: true
author: burning-cost
---

Most pricing teams who have adopted conformal prediction for uncertainty quantification are running joint frequency-severity intervals using the Bonferroni correction, or something close to it. Target 95% joint coverage, run each dimension at 97.5%, produce a rectangle $[\ell_f, u_f] \times [\ell_s, u_s]$, move on. It works. It has a finite-sample guarantee. It is also provably wasteful — sometimes by 20–40% of the prediction region's volume.

Braun, Berta, Jordan and Bach (arXiv:2507.20941, July 2025, updated February 2026) demonstrate the source of the waste and a remedy. The waste comes from ignoring the cross-dimensional correlation structure when computing nonconformity scores. The remedy is to whiten the residuals using the covariance between outputs before scoring. The resulting prediction sets are ellipsoids, not hyperrectangles, and they are tighter wherever your outputs are correlated.

Whether this matters for your team depends almost entirely on one number: the number of outputs you are modelling jointly. For $d = 2$ (frequency and severity), the gain is real but modest, and there are simpler alternatives already available. For $d \geq 5$ — a multi-peril home model covering fire, flood, escape of water, theft, and accidental damage separately — the efficiency argument becomes compelling.

---

## Why Bonferroni wastes coverage

The Bonferroni correction for joint coverage at level $1 - \alpha$ across $d$ outputs allocates $\alpha / d$ miscoverage to each dimension. For $d = 2$ at $\alpha = 0.05$, each dimension is calibrated at 97.5%.

That allocation is correct — jointly covering two uncorrelated, independent outputs at 95% does require 97.5% marginal coverage on each. The problem is that frequency and severity are not independent. In motor insurance they are modestly negatively correlated: books with higher claim frequency tend to have lower average severity (more minor bumps, fewer catastrophic events). In home insurance the correlation structure between perils is more complex but never zero.

When outputs are correlated, the Bonferroni rectangle overcovers. The 97.5% per-dimension intervals cover more joint probability mass than 95% because the two dimensions are not moving independently. The rectangle is padded in the joint space by an amount that depends on the correlation. Bonferroni treats the joint distribution as if it were the product of its marginals. It is not.

The picture is clearest in the $d = 2$ case. If frequency and severity had correlation $\rho = -0.4$, the Bonferroni rectangle at 95% joint coverage would actually achieve around 96.5–97% joint coverage — it is calibrated for the independent case, applied to a correlated case. That surplus coverage corresponds to a region of the joint space that the rectangle includes but a proper joint set would not.

---

## The Mahalanobis remedy

Braun et al.'s nonconformity score for observation $i$ with outputs $y_i \in \mathbb{R}^d$ and point prediction $\hat{y}_i$ is:

$$s_i = \left\| \hat\Sigma^{-1/2} (y_i - \hat{y}_i) \right\|_2$$

where $\hat\Sigma$ is an estimated covariance matrix of the residuals. Whitening the residuals by $\hat\Sigma^{-1/2}$ makes the score invariant to the correlation structure: a residual vector that is large in the direction of high covariance is shrunk relative to one that is large in a direction of low covariance. The resulting score is comparable across calibration observations regardless of their joint residual structure.

The prediction set for a new observation $x$ is the ellipsoid:

$$C(x) = \left\{ y \in \mathbb{R}^d : \left\| \hat\Sigma^{-1/2}(y - \hat{f}(x)) \right\|_2 \leq \hat{q} \right\}$$

where $\hat{q}$ is the standard conformal quantile of calibration scores at level $\lceil (n+1)(1-\alpha) \rceil / n$. The finite-sample marginal coverage guarantee $P(Y \in C(X)) \geq 1 - \alpha$ follows immediately from the exchangeability argument — it is the same guarantee as Bonferroni, just for an ellipsoid rather than a rectangle.

What the ellipsoid gives you that the rectangle cannot is tightness along directions of high correlation. If frequency and severity co-move, the ellipsoid shrinks along the diagonal where they move together and expands only where they diverge. The rectangle is indifferent to this structure.

---

## The covariance estimation question

The original paper trains a Cholesky factor network jointly with the base predictors — a neural network that maps each input $x$ to a local covariance $\hat\Sigma_\phi(x) = L_\phi(x)L_\phi(x)^T$. The resulting ellipsoids vary in shape and orientation across the input space. The theoretical guarantee holds under exchangeability regardless of whether the covariance model is correct; near-conditional coverage (rather than just marginal) holds asymptotically when the covariance model is well-specified.

This is architecturally elegant and largely irrelevant for UK pricing teams. The base predictors — frequency GLM or GBM, severity GLM or GBM — are fitted separately by actuarial teams and handed to the conformal layer fully trained. Joint end-to-end training is not how pricing models are built, validated, or signed off.

What is available without joint training is the simpler version: estimate $\hat\Sigma$ from the calibration residuals as a single global matrix. This is not the paper's full method, but it exploits the same intuition. Compute the $(n \times d)$ residual matrix $R$ on the calibration set, estimate $\hat\Sigma = R^T R / n$ (plus a small ridge term for numerical stability), and whiten accordingly. No PyTorch. No joint training. Compatible with any base predictor.

The global-covariance version captures the average correlation structure across the entire portfolio. It misses risk-level heteroscedasticity — for example, it cannot know that the frequency-severity correlation for young drivers is different from that for mature fleet risks. The paper's Cholesky network captures that; global covariance does not. For homogeneous portfolios, this limitation is minor. For highly stratified books, it matters.

---

## The interpretability problem

Here is what you cannot do with an ellipsoid: read off a frequency bound and a severity bound separately.

The hyperrectangle from Bonferroni or from our `gwc` and `lwc` conformal methods gives you two numbers per dimension — lower and upper — that a pricing actuary can act on directly. "The 95% prediction interval for frequency is 0.08 to 0.35 claims per year, and for severity is £1,200 to £4,800." That is auditable, presentable to a CRO, and can be used directly in a reserving or capital calculation.

An ellipsoid is defined by a shape matrix and a radius. The set $\{y : (y - \hat{y})^T \hat\Sigma^{-1} (y - \hat{y}) \leq \hat{q}^2\}$ does not decompose into per-dimension intervals without losing the tightness gain. You can compute the smallest hyperrectangle that contains the ellipsoid — the circumscribed rectangle — but its half-widths are $\hat{q}\sqrt{\hat\Sigma_{jj}}$ per dimension. That is a valid joint prediction set, but it is wider than the rectangle you would get from Bonferroni (it inherits the ellipsoid's joint coverage without the ellipsoid's efficiency).

Put differently: the efficiency gain from Mahalanobis over Bonferroni exists in the ellipsoidal space. If you immediately project back to a rectangle for actuary consumption, you surrender some of it.

The FCA Consumer Duty framework creates a practical governance pressure here. "Explainable AI" requirements under Consumer Duty and fair value frameworks apply to pricing decisions, and a pricing committee explaining its uncertainty quantification methodology should not need to explain a matrix inverse. "We use a 95% Bonferroni correction, so each dimension is held at 97.5%" is a sentence. "We use a Mahalanobis-weighted nonconformity score whose prediction region is an ellipsoid defined by the eigenstructure of the calibration residual covariance" is not.

---

## Where the gain is real: multi-peril home

At $d = 2$, our existing locally-weighted conformal (`lwc`) method already recovers a substantial fraction of the efficiency available over Bonferroni. LWC uses group-wise coordinate standardisation — it implicitly captures some of the correlation structure through group membership in the calibration set. For a motor portfolio, LWC versus full Mahalanobis is probably a 5–10% additional volume reduction at best, in exchange for interpretability and the circumscribed-rect bridge.

At $d \geq 5$, the arithmetic changes. Consider a home insurance model with five perils jointly: fire, flood, escape of water, theft, and accidental damage. Bonferroni at $\alpha = 0.05$ allocates 1% per dimension — calibrating each dimension at 99%. The overcoverage from ignoring the correlation structure grows as $O(d(d-1)/2)$ pairwise terms are left on the table. The 20–40% volume reduction cited in the paper is most plausible in this regime.

The insurance motivation for multi-peril joint sets is real. A home insurer accumulating weather-exposed flood and escape-of-water risks faces correlation between those two perils that is neither small nor well-modelled by coordinate-wise methods. A joint prediction set that correctly accounts for the flood-escape of water correlation will be tighter and more informative for risk selection and portfolio accumulation management.

We have not run this on a UK multi-peril home dataset, and the paper's empirical claims are on synthetic and unspecified real data. The 20–40% figure should be treated as a rough indicator, not a guarantee. Insurance residuals are skewed — Poisson near zero for frequency, Gamma heavy-tailed for severity — and the empirical covariance estimate can be a poor fit for skewed marginals. The actual volume reduction on a real home portfolio may be smaller.

---

## What we are not building yet

The full Braun et al. implementation — the input-conditional Cholesky network — requires joint training of base predictor and covariance network in PyTorch. That is a fundamental change to how pricing models are trained and validated. The marginal benefit over the global-covariance version is conditional coverage rather than just marginal coverage. That is worth having, but not at the cost of overhauling the training pipeline.

The global covariance version is ~300 lines of NumPy and SciPy, slots alongside the existing `JointConformalPredictor`, and would add a fifth method to `insurance-conformal`'s multivariate subpackage. For $d = 2$ motor, the honest recommendation remains LWC: it is interpretable, hyperrectangular, and already deployed. For a team building a new $d = 5$ home model, the Mahalanobis global mode is worth evaluating against Bonferroni.

One genuinely useful feature of the full Braun et al. framework deserves a separate mention: conditioning on partial observations. Given that severity was £2,500 this year, what is the marginal prediction interval for next year's frequency? The block-matrix structure of $\hat\Sigma$ supports this conditioning analytically. For renewal pricing — where you know the current year's loss experience and want to update predictions for the renewal — this is a concrete application that does not require ellipsoidal sets to be presented to actuaries. The conditioning can be done internally and the output presented as a scalar interval on the dimension of interest.

That capability is not currently in the library. It is the most practically interesting extension from the paper.

---

## The paper

Braun, Berta, Jordan and Bach (2025), "Multivariate Standardized Residuals for Conformal Prediction", arXiv:2507.20941. Braun is at ETH Zürich; Jordan and Bach are at Berkeley and INRIA respectively. The core algorithm is in Section 2; the low-rank approximation for high-dimensional outputs is in Section 3. The missing-output conditioning is in Section 4.

The GitHub repository is [ElSacho/Multivariate_Standardized_Residuals](https://github.com/ElSacho/Multivariate_Standardized_Residuals). It requires PyTorch for the Cholesky network. The global covariance simplification does not appear in the repository.

The earlier Fan & Sesia paper (arXiv:2512.15383), which is the basis for the `gwc` and `lwc` methods already in [insurance-conformal](https://pypi.org/project/insurance-conformal/), uses coordinate-wise standardisation rather than full covariance whitening. Both papers are attempting to achieve near-conditional coverage in the multivariate setting; Braun et al. do it with a richer covariance structure at the cost of a more complex model.

---

## Related

- [Conformal Prediction for Joint Frequency-Severity Models](/conformal-prediction/insurance-pricing/2026/03/20/conformal-prediction-joint-frequency-severity-models/) — the baseline `JointConformalPredictor` using Bonferroni and LWC
- [Your Monitoring Thresholds Are Made Up](/model-monitoring/insurance-pricing/2026/04/03/your-monitoring-thresholds-are-made-up/) — conformal SPC for model monitoring
