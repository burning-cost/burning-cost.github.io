---
layout: post
title: "SPQRx: Semi-Parametric Severity Modelling Without Threshold Selection"
date: 2026-04-01
categories: [techniques, pricing, libraries]
tags: [severity, EVT, GPD, threshold-selection, quantile-regression, splines, neural-network, ILF, excess-of-loss, reinsurance, insurance-severity, arXiv-2504-19994, Majumder-Richards, SPQRx, M-spline, bGPD, heavy-tail, UK-motor, TPBI, python]
description: "Threshold selection is the Achilles' heel of extreme value theory in insurance pricing. Majumder and Richards (arXiv:2504.19994) eliminate it by blending a spline-neural-network bulk density into a GPD tail continuously, using covariate-conditional blending quantiles rather than a fixed threshold. We have shipped this as SPQRxSeverity in insurance-severity v0.3.3."
math: true
author: burning-cost
---

Every pricing actuary who has fitted a GPD to large losses has faced the same problem: where do you set the threshold? Too low and you include body-of-distribution claims in a tail fit that was designed for exceedances. Too high and you have fifteen data points above the threshold, your shape parameter has a standard error of 0.4, and you are extrapolating to the 99.9th percentile from a fit that the data cannot support. The mean residual life plot gives you a range. The stability plot gives you a different range. You pick something in the middle and call it judgement.

This is not a niche failure mode. It affects every EVT application in severity pricing — motor bodily injury, commercial property, marine cargo, professional indemnity. The threshold problem is endemic, and all current approaches, including the tools in `insurance-severity` before v0.3.3, share it.

Majumder and Richards ([arXiv:2504.19994](https://arxiv.org/abs/2504.19994), University of Edinburgh, submitted April 2025) have a solution that does not involve selecting a threshold at all. Their SPQRx method replaces the threshold with a pair of quantile levels that define a blending interval. The GPD parameters are then derived analytically from the fitted bulk density at those quantile levels — and crucially, they vary by covariate. The threshold is not a single number applied to the whole portfolio. It is a function of each risk's characteristics.

We have implemented this in `insurance-severity` v0.3.3 as `SPQRxSeverity`.

```bash
uv add "insurance-severity>=0.3.3"
```

---

## Why threshold selection is genuinely hard

The classical approach to tail modelling — pick a threshold $u$, fit a GPD to exceedances — has a structural problem that no amount of diagnostic plotting resolves. You are choosing a single number that determines whether each claim goes into the bulk model or the tail model. That number is:

1. **Global.** A threshold of £250,000 treats a claim from a high-value commercial property the same as a claim from a standard motor policy. The former might be well into the ordinary bulk of its distribution. The latter might be an extreme tail event. A single threshold cannot accommodate both.

2. **Bias-variance unstable.** Lower threshold = more data above it = lower variance in $\hat\xi$ but higher bias (you are including non-tail observations in a fit that assumes GPD). Higher threshold = less bias but higher variance. The optimum is unknown and data-dependent.

3. **Serially reselected.** Every time you refresh the model — annually for most UK pricing teams, more often for exposed books — you re-run the diagnostics and potentially pick a different threshold. Your GPD parameters jump discontinuously. Explaining a 0.15-unit change in $\xi$ to a Chief Actuary is uncomfortable if the cause is "the stability plot looked different this year."

The tools in `insurance-severity` prior to v0.3.3 all share this problem. `TruncatedGPD` requires the user to pass a threshold. `CensoredHillEstimator` uses a rolling-variance heuristic to select the tail fraction. `CompositeSeverityModel` has three threshold selection modes — fixed, profile likelihood, mode matching — and none of them produce results that are stable across data refreshes. This is not a criticism of the implementations. It is a fundamental limitation of the approach.

---

## What SPQRx does instead

SPQRx (Semi-Parametric Quantile Regression eXtreme) starts from a different architecture. The bulk of the distribution is modelled by an MLP whose outputs are weights over $K$ M-splines (monotone splines that integrate to a proper density). The MLP is covariate-conditional: different weights for different risk characteristics. The tail is a GPD, but not one fitted to exceedances above a threshold. Instead:

1. The user specifies two quantile levels $p_a < p_b$ (defaults: 0.85 and 0.95 in our implementation).
2. For each observation $x$, the fitted bulk model predicts the quantiles $a(x) = Q_{\text{bulk}}(p_a \mid x)$ and $b(x) = Q_{\text{bulk}}(p_b \mid x)$. These define a blending interval $[a(x), b(x)]$.
3. The GPD location $\tilde{u}(x)$ and scale $\tilde{\sigma}(x)$ are solved analytically from the constraint that the blended density is continuous at $a(x)$ and $b(x)$. Only the shape parameter $\xi(x)$ is a free trained parameter.
4. The blending itself uses a Beta-CDF weight function that transitions smoothly from pure bulk to pure GPD across the interval.

The result is a three-regime quantile function:

$$Q_{\text{SPQRx}}(\tau \mid x) = \begin{cases} Q_{\text{bulk}}(\tau \mid x) & \tau < p_a \\ \text{invert blended CDF numerically} & p_a \leq \tau < p_b \\ \tilde{u}(x) + \frac{\tilde{\sigma}(x)}{\xi(x)}\left[\left(\frac{1-\tau}{1-p_b}\right)^{-\xi(x)} - 1\right] & \tau \geq p_b \end{cases}$$

The third regime is GPD extrapolation beyond $p_b$, and it is in closed form. For pricing at the 99th or 99.9th percentile, this is the operative formula, and its inputs $(\tilde{u}(x), \tilde{\sigma}(x), \xi(x))$ are all covariate-conditional.

The critical point: there is no threshold. There are quantile levels. The distinction matters because quantile levels are stable across data refreshes in a way that a threshold amount in pounds is not. If the mean claim severity increases by 15% due to inflation, the 85th percentile of the new distribution is still the 85th percentile. A fixed threshold of £150,000 is no longer in the same part of the distribution.

---

## The implementation

```python
from insurance_severity import SPQRxSeverity
import numpy as np

# X: (n, p) covariate matrix — continuous or encoded categoricals
# y: (n,) strictly positive claim amounts

model = SPQRxSeverity(
    n_splines=25,        # K M-splines in bulk density
    hidden_size=32,      # nodes per hidden layer
    num_hidden_layers=2, # H
    pa=0.85,             # lower blending quantile
    pb=0.95,             # upper blending quantile
    max_epochs=500,
    patience=30,
    device='cpu',
)

model.fit(X_train, y_train)
```

Once fitted, there are four main query methods:

```python
# Covariate-conditional quantile at any level
q99 = model.predict_quantile(X_score, tau=0.99)   # shape (n_score,)
q999 = model.predict_quantile(X_score, tau=0.999)

# Inspect the tail parameters directly
tail = model.tail_params(X_score)
# tail['xi']:          shape parameter per risk
# tail['u_tilde']:     effective GPD location per risk (£)
# tail['sigma_tilde']: GPD scale per risk (£)

# Full distributional object for a batch
dist = model.predict_distribution(X_score)
print(dist.mean())          # (n_score,) expected values
print(dist.quantile(0.995)) # same as predict_quantile but via object
```

The `predict_distribution` method returns an `SPQRxDistribution` object that mirrors the API of `MDNMixture` from earlier versions of the library — the same `.quantile()`, `.cdf()`, `.pdf()`, and `.mean()` interface.

---

## ILF computation

The most direct application to reinsurance pricing is ILF (increased limits factor) calculation. An ILF at limit $L$ relative to basic limit $L_0$ is:

$$\text{ILF}(L, L_0) = \frac{E[\min(Y, L)]}{E[\min(Y, L_0)]} = \frac{\int_0^L S(y) \, dy}{\int_0^{L_0} S(y) \, dy}$$

where $S(y) = 1 - F(y)$ is the survival function. `SPQRxDistribution` computes this directly:

```python
dist = model.predict_distribution(X_score)

# ILF for £2M limit relative to £500k basic limit, per risk
ilf = dist.ilf(L=2_000_000, basic_limit=500_000)

# Covariate-conditional: each risk gets its own ILF
print(ilf.mean())   # portfolio average
print(np.percentile(ilf, [25, 50, 75, 95]))  # dispersion across risks
```

The covariate-conditional GPD shape $\xi(x)$ is what makes this genuinely useful. A portfolio with mixed risk classes will have materially different ILF curves by segment — a high-value commercial property account and a standard SME property account should not share a single GPD shape parameter. With SPQRx, they do not.

---

## Blending sensitivity analysis

The choice of $p_a$ and $p_b$ is the analogue of threshold selection — but it is more tractable because you are choosing quantile levels, not a cash amount, and you can inspect the sensitivity directly:

```python
sensitivity = model.pa_pb_sensitivity(
    X_sample,
    pa_grid=[0.80, 0.85, 0.90],
    pb_grid=[0.90, 0.95, 0.97],
    tau=0.99,
)
# Returns DataFrame: pa, pb, mean_q99, std_q99, mean_xi, std_xi
print(sensitivity)
```

A stable model should show modest variation in the 99th percentile quantile across this grid. If the 99th percentile varies by 30% between $p_a = 0.80$ and $p_a = 0.90$, the bulk spline fit in the 80th–90th percentile range is unstable, which points to insufficient data or a poorly specified feature set rather than a sensitivity to the blending choice.

The paper's simulation results (Table 1 of Majumder and Richards, at $n=1000$ and $n=10000$) show that SPQRx achieves lower tail-calibrated Integrated Wasserstein Distance than both SPQR (without the blended tail) and deep GP regression on synthetic data with $\xi = 0.3$. At $n=10000$ the improvement over SPQR is around 15%. At $n=1000$ it is more variable.

---

## When to use SPQRx

SPQRx is not a replacement for all EVT tooling in the library. It has a specific regime where it dominates:

**Use SPQRx when:**
- You have $n \geq 500$ large losses with observed settlement amounts (not truncated at policy limits)
- You want covariate-conditional tail parameters — different $\xi$ for different risk segments
- You are computing ILF curves or XL pricing and need accuracy at Q95–Q99.9
- You are refreshing the model regularly and want stable parameters across refreshes

**Stay with `TruncatedGPD` or `CompositeSeverityModel` when:**
- You have $n < 500$ losses: the spline-MLP is data-hungry and will overfit
- Your data has significant upper truncation at policy limits: the bGPD framework does not correct for censoring, and this produces downward-biased tail estimates
- You need a scalar $\xi$ with a standard error for a regulatory submission: $\xi(x)$ as a covariate function is harder to defend to Lloyd's or the PRA than $\hat\xi = 0.42 \pm 0.08$
- You are working with UK motor own-damage at normal limits: the tail is light ($\xi$ near zero or negative) and GPD extrapolation adds no value

For UK motor TPBI with five or more years of fully developed claims — typically $n > 2000$ at a national insurer — SPQRx is the right tool. TPBI has a genuine heavy tail ($\xi$ in the range 0.3–0.8 for severe injury cohorts), material covariate heterogeneity (injury type, liability split, vehicle class), and ILF curves that feed directly into reinsurance treaty pricing. All three features of SPQRx — covariate-conditional shape, continuous blending, closed-form GPD extrapolation — are productive there.

---

## Comparison with EQRN

The library's `EQRNModel` (from `insurance-eqrn`) solves a related problem. EQRN is a neural quantile regression model that, like SPQRx, produces covariate-conditional GPD tail parameters. The key difference is in how the tail transition is handled:

- EQRN requires an intermediate quantile estimated by a separate GBM as the effective threshold. The threshold problem is reduced but not eliminated — it is now a quantile level estimated by the GBM rather than a cash amount, but the GBM quantile estimate itself introduces uncertainty.
- SPQRx builds the bulk and tail into a single model trained jointly. There is no separate intermediate model.

For a portfolio where you already have a well-fitted GBM quantile model, EQRN is fast to stack on top of it. For a fresh build, SPQRx's single-model architecture is cleaner. Both are available in the library.

---

## The technical implementation

The M-spline basis in `SPQRxSeverity` is implemented using the de Boor B-spline recursion, converted to M-splines via $M_k = B_k \cdot \text{order} / \text{knot-span}$. I-splines (the integrated M-splines that give the CDF) are computed via 200-point midpoint quadrature per batch. This is not the only implementation choice — `patsy` has an M-spline basis, and a fully analytic I-spline is possible — but the de Boor approach keeps the dependency footprint to numpy and scipy, which are already required.

The bGPD parameter solver (Equations 4–5 of the paper) is vectorised numpy with no root-finding. $\tilde{\sigma}(x)$ and $\tilde{u}(x)$ are derived from the two-point GPD constraint at $(a(x), p_a)$ and $(b(x), p_b)$, given $\xi(x)$. The shape parameter is clipped to $[10^{-4}, 0.5]$ during training — the lower bound prevents numerical instability in the closed-form GPD formula; the upper bound prevents pathological heavy tails during early training epochs.

Quantile inversion in the bulk and blending regimes uses `brentq` from scipy, vectorised over the covariate dimension. For the GPD regime ($\tau \geq p_b$) the formula is closed form and no inversion is required.

The full implementation is approximately 580 lines of code across `network.py`, `spqrx.py`, and `distribution.py` in the `insurance_severity.spqrx` subpackage.

---

## The paper

Majumder, Aritra, and Jennifer L. Richards. "Semi-parametric bulk and tail regression using spline-based neural networks." arXiv:2504.19994 [stat.ME]. April 2025.

The paper is technically careful and the mathematical derivation of the blended GPD parameters (Equations 4–5) is worth reading if you want to understand why the continuous blending works and how it compares to the hard threshold of standard POT approaches. The wildfire application (US wildfire burnt area, 1990–2020) is less directly relevant to insurance than the simulation results, but the tIWD (tail-calibrated Integrated Wasserstein Distance) evaluation metric is worth adopting as a tail scoring rule alongside CRPS for severity model comparison.

---

## Related posts

- [insurance-severity: The Full Severity Modelling Toolkit](https://github.com/burning-cost/insurance-severity) — library overview
- [EQRN: Covariate-Conditional GPD Tail Modelling for XL Pricing](/2026/03/27/eqrn-covariate-conditional-gpd-xl-pricing/) — the alternative tail approach
- [ILF Curves from First Principles: What Your Composite Model Gets Wrong](/2026/02/28/ilf-curves-composite-severity-models/) — ILF methodology and failure modes
- [Mixture Density Networks for Insurance Severity: When the Gamma Isn't Enough](/2026/03/28/mixture-density-networks-insurance-severity/) — MDN as the bulk density alternative
