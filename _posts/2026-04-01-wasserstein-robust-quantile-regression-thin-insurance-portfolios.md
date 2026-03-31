---
layout: post
title: "The 99th Percentile From 200 Claims: Wasserstein Robust Quantile Regression for Thin Portfolios"
date: 2026-04-01
categories: [techniques, pricing]
tags: [quantile-regression, distributional-robustness, Wasserstein, thin-data, reserve-quantile, large-loss-loading, reinsurance, insurance-quantile, arXiv-2603-14991, Zhang-Mao-Wang, closed-form, O-N-half, finite-sample-guarantee, commercial-lines, Solvency-II, python]
description: "Zhang, Mao and Wang (arXiv:2603.14991, March 2026) prove a closed-form equivalent for worst-case quantile regression under Wasserstein distributional uncertainty — a result that previously did not exist. For UK pricing teams estimating 99th-percentile reserve quantiles from 200 historical claims, this is a principled replacement for the 'take the second-largest claim and hope' approach. We have shipped it in insurance-quantile v0.4.0."
math: true
author: burning-cost
---

Take a commercial property segment: 200 historical claims, roughly lognormal severity, and you need the 99th percentile for your large loss loading. The empirical 99th is your second-largest observation. If that observation happens to be a warehouse fire that was genuinely exceptional — or if it is not exceptional but your next renewal season happens to be worse — your loading is either too high or catastrophically too low. CatBoost quantile regression does not help much here: a tree model fitted on 200 points at the 99th percentile is memorising noise.

This is not an exotic problem. It is how UK commercial lines pricing actually works for any segment that is not motor volume. D&O, agriculture, high-net-worth, professional indemnity, pet — most of these have segments where N < 500 per rating cell. The thin-data quantile problem is endemic.

Zhang, Mao and Wang (arXiv:2603.14991, USTC and University of Waterloo, submitted March 2026) provide a mathematically principled answer. They derive the first closed-form solution for worst-case quantile regression under Wasserstein distributional uncertainty. We have implemented their result in `insurance-quantile` v0.4.0 as the `WassersteinRobustQR` class, with two actuarial wrappers: `wdrqr_large_loss_loading` and `wdrqr_reserve_quantile`.

---

## The problem with standard quantile regression at high quantiles

Standard quantile regression — and CatBoost's implementation of it — minimises the empirical check (pinball) loss over your training data. This is sensible asymptotically. When N is large, the empirical distribution concentrates around the true distribution and the empirical minimiser converges to the true population quantile at a rate of $O(N^{-1/2})$.

The issue is the gap between "large" and "200". The empirical 99th percentile on 200 claims has a sampling distribution with very wide tails. The true 99th percentile could plausibly be anywhere from your 196th-largest observation to three times your largest observation, depending on the true underlying severity distribution. Standard QR does not quantify this uncertainty or correct for it — it just finds the best fit to what you observed.

There is a second, subtler problem: deployment shift. Your 200 claims were written over the past five years. Claims inflation, exposure mix, regulatory changes, and fraud patterns mean the distribution at renewal is not the same as the distribution in your training data. The empirical minimiser was optimal for the historical distribution; it may be substantially suboptimal for what you are actually writing.

Wasserstein distributionally robust optimisation (DRO) addresses both problems in one move.

---

## What Wasserstein distributional robustness means

A Wasserstein ball of radius $\varepsilon$ around your empirical distribution $\hat{F}_n$ is the set of all probability distributions reachable from $\hat{F}_n$ by moving probability mass a total distance of at most $\varepsilon$. The "distance" is the cost to transport one unit of probability mass, measured in the feature space.

Formally: $\mathcal{B}_p(\hat{F}_n, \varepsilon) = \{ F : W_p(F, \hat{F}_n) \leq \varepsilon \}$

where $W_p$ is the type-$p$ Wasserstein distance.

For insurance, this is directly interpretable. A ball of radius $\varepsilon = 0.2$ around your historical commercial property severity distribution contains all distributions you could reach by shifting individual claim observations by a total of 0.2 (in standardised feature units). If your five-year severity history concentrated at £50k–£500k, a ball of this size encompasses distributions with somewhat heavier tails, somewhat lighter tails, and somewhat shifted mean — without allowing arbitrarily pathological distributions.

The distributionally robust quantile regression problem is:

$$\min_{\beta, s} \; \max_{F \in \mathcal{B}_p(\hat{F}_n, \varepsilon)} \; \mathbb{E}_F[\rho_\tau(Y - \beta^\top X - s)]$$

where $\rho_\tau(u) = u(\tau - \mathbf{1}(u < 0))$ is the check loss. You minimise expected loss under the worst-case distribution within the ball. The result is robust to any distributional shift smaller than $\varepsilon$.

Prior Wasserstein DRO theory (Esfahani and Kuhn, 2018) required the loss function to be "Lipschitz continuous to power $p$". The check loss is globally Lipschitz but emphatically not Lipschitz to power $p$ for $p > 1$. Standard theory breaks. Zhang et al. develop bespoke duality arguments that exploit the piecewise-linear structure of the check loss to get around this.

---

## The closed-form result

The main theorem (Theorem 1 of Zhang et al.) is that the inner maximisation has a closed-form solution. You do not need to solve a minimax problem. After optimising out the intercept, the slope estimator reduces to penalised quantile regression:

$$\beta^* = \operatorname{argmin}_{\bar\beta} \; \mathbb{E}_{\hat{F}_n}[\rho_\tau(Y - \bar\beta^\top X - \bar s^*)] + c_{\tau,p} \cdot \varepsilon \cdot \|\bar\beta\|_*$$

where $\|\cdot\|_*$ is the dual norm to the feature norm and $c_{\tau,p}$ is a closed-form constant depending only on $\tau$ and $p$.

The intercept then receives an analytic correction:

$$s^* = \bar s^* + \frac{\varepsilon}{q} \left(\tau^q - (1-\tau)^q\right) c_{\tau,p}^{1-q} \|\bar\beta^*\|_*$$

where $q = p/(p-1)$ is the conjugate exponent.

Two structural features of this result drive the practical value.

**The intercept correction is the mechanism that matters.** For $\tau > 0.5$ (which is every high-quantile insurance use case), the correction term is positive — the intercept is pushed upward. This directly addresses the well-known downward bias of empirical high-quantile estimators in small samples. The standard QR intercept systematically underestimates because the empirical distribution does not have enough mass in the tail. The WDRQR correction accounts for this by acknowledging that the true distribution may have heavier tails than observed.

**$p = 1$ gives you nothing.** The $W_1$ case is a special case where both the slope and the intercept are invariant to $\varepsilon$. The distributional robustness is entirely absorbed into the constraint structure without producing any regularisation. Increasing $\varepsilon$ in the $W_1$ case does not change the estimator. This is not a bug — it reflects the fact that $W_1$ robustness is already implicit in standard quantile regression. For genuine distributional robustness with a practical effect, you need $p = 2$.

The uniqueness result is the stranger finding: for $p > 1$, the check loss is the **only** class of convex loss functions that admits this additive Wasserstein regularisation structure. This is not arbitrary. The geometry of the piecewise-linear check loss and the dual Wasserstein constraint happen to align in a way that produces a clean closed form. For any smoother loss (squared error, Huber), the inner max does not have this property.

---

## The finite-sample guarantee

Theorem 3 of Zhang et al. gives the calibration schedule for $\varepsilon$. Under a finite $s$-th moment condition (only $s > 2$ required — no sub-Gaussian assumption, no bounded support), set the radius to:

$$\varepsilon_N(\eta) = c_\alpha \cdot \frac{\log(2N+1)^{1/s}}{\sqrt{N}}$$

With this schedule, the WDRQR estimator achieves its out-of-sample guarantee with probability at least $1 - \eta$ over the training draw.

The guarantee has three properties that matter for insurance:

**Dimension-free.** The $O(N^{-1/2})$ rate does not deteriorate as you add features. Most DRO bounds have factors that grow exponentially in the feature dimension or with $\sqrt{d}$. Zhang et al.'s bound has neither. For insurance models with 50–100 rating factors, this is essential — a bound that degrades with feature count is useless in practice.

**Only $s > 2$ moments required.** Insurance severity is not sub-Gaussian. Motor bodily injury severity, commercial property, and marine cargo all have power-law tails — finite variance is plausible for the bulk of the distribution, but Gaussian tails are not. The $s > 2$ condition is satisfied by lognormal and many Pareto-tailed distributions in the range relevant to most segments (noting that EQRN handles the true extreme tail, where neither condition is comfortable).

**Rate-optimal.** The $O(N^{-1/2})$ rate matches standard quantile regression. Distributional robustness does not cost convergence speed.

What this means in practice for segment sizing:

| Segment size N | $\varepsilon_N$ (at $s=4$, $\eta=0.05$) |
|---|---|
| 50 | $0.38 \cdot c$ |
| 200 | $0.19 \cdot c$ |
| 500 | $0.12 \cdot c$ |
| 2,000 | $0.06 \cdot c$ |
| 10,000 | $0.027 \cdot c$ |

The constant $c$ requires calibration — more on that below. But the shape of the table is interpretable: a segment of 200 claims needs roughly seven times the radius of a segment of 10,000. The ratio $\sqrt{10000}/\sqrt{200} = 7.1$. This is directly presentable to a Chief Actuary or pricing committee: "we are treating our estimate of the 200-claim segment as seven times more uncertain about distributional shift than our estimate of the 10,000-claim segment."

---

## Three insurance use cases

### 1. Reserve quantile for thin commercial lines segments

You need the 99th percentile of aggregate severity for an IBNR reserve in a commercial lines segment with 150 historical claims per year. The regulatory context is Solvency II: your actuarial function report needs a defensible quantile with stated uncertainty.

```python
import polars as pl
import numpy as np
from insurance_quantile import wdrqr_reserve_quantile

# X_train: rating factor matrix (Polars DataFrame, numeric)
# y_train: per-claim severity (Polars Series, £)
result = wdrqr_reserve_quantile(
    X_train=X_train,
    y_train=y_train,
    X_score=X_score,
    tau=0.99,
    eps=None,        # auto-calibrated from Theorem 3 schedule
)

# result["q_0.99"]: per-risk 99th percentile estimates
# result["eps_used"]: the Wasserstein radius actually applied
# result["intercept_correction"]: the upward adjustment to the intercept
print(result.select(["policy_id", "q_0.99", "eps_used", "intercept_correction"]))
```

The `eps=None` path uses the `optimal_eps` method internally: $\varepsilon_N = c_\tau \cdot \log(2N+1)^{1/4} / \sqrt{N}$ with $s=4$ (finite fourth moment). You can override with a domain-specific $\varepsilon$ if you have a view on distributional shift magnitude.

### 2. Large loss loading for commercial property

Your current large loss loading pipeline uses `large_loss_loading` from `insurance-quantile`, which wraps a CatBoost quantile GBM. For segments with N > 2,000, the GBM is the right tool. For the thin segments, WDRQR provides a robust linear estimate as a floor or blend:

```python
from insurance_quantile import wdrqr_large_loss_loading, MeanModelWrapper

# mean_model: any fitted model with predict() returning expected severity
robust_loading = wdrqr_large_loss_loading(
    X_train=X_train,
    y_train=y_train,
    X_score=X_score,
    mean_model=mean_model,
    alpha=0.99,
    eps=0.20,        # explicit Wasserstein radius: "20% of distribution mass could shift"
)
# Returns: Polars Series of per-risk large loss loadings (£)
```

The loading is $Q_{0.99}^{\text{WDRQR}} - \hat{\mu}$ where $\hat{\mu}$ is the mean model prediction. The WDRQR quantile includes the intercept correction that accounts for downward bias in thin-data empirical quantiles.

### 3. Excess-of-loss reinsurance attachment pricing

You are pricing a per-risk XL layer at xs £500k for a commercial property account with four large losses in five years of data, one of which hit £2.1m. The empirical 99.5th percentile is £2.1m — the single largest claim. You do not believe this one observation is a reliable estimate of the true 99.5th percentile.

```python
from insurance_quantile import WassersteinRobustQR
import numpy as np

model = WassersteinRobustQR(tau=0.995, p=2, eps=0.35)
model.fit(X_train_np, y_train_np)

# Inspect the correction
print(f"Intercept correction: £{model.intercept_correction_:,.0f}")
print(f"Effective eps used:   {model.eps_used_:.3f}")

q_995 = model.predict(X_score_np)
```

An $\varepsilon$ of 0.35 corresponds to roughly the uncertainty level at $N \approx 80$ using the Theorem 3 schedule — a reasonable prior for "I have 4 claims above attachment and the true tail could be substantially heavier." The intercept correction will push the quantile estimate upward relative to standard QR, reducing the risk of under-pricing the attachment.

---

## How to calibrate $\varepsilon$

The Theorem 3 schedule gives the *shape* of the radius as a function of $N$, but not the constant $c$. There are three practical approaches:

**Cross-validation.** Split your claims data into calibration and validation sets. Fit WDRQR at a grid of $\varepsilon$ values and select the value that minimises pinball loss on the validation set. This is the most mechanical approach and works well when you have enough data to afford a 20–30% holdout.

**Historical distribution shift.** If you have multi-year data, compute the Wasserstein distance between the severity distribution in year $t$ and year $t+1$ across several year-pairs. The average of these observed distributional shifts gives a direct estimate of the "typical" $\varepsilon$ for your line of business. For UK motor own damage, this would be dominated by claims inflation cycles. For commercial property, it would reflect exposure mix drift and natural catastrophe years.

**Domain expert input.** Present the table above to your Chief Actuary: "We are estimating from 200 claims. How much could the true severity distribution differ from our observed data?" A qualitative answer of "quite a lot — we've had significant inflation and fraud pattern changes" maps to $\varepsilon$ in the 0.15–0.25 range; "pretty stable" maps to 0.05–0.10. This is not precise, but it is more principled than an ad-hoc loading and it is auditable.

The `WassersteinRobustQR` class exposes `optimal_eps(N, s=4.0, eta=0.05)` which implements the Theorem 3 schedule directly. Use it as a default; override when you have a specific view.

---

## When not to use WDRQR

**When N > 5,000 per segment.** CatBoost's quantile GBM (`QuantileGBM`) captures non-linear interactions that a linear model cannot. For standard volume motor segments, the GBM dominates. WDRQR's linear constraint is a cost.

**For extreme quantiles above 0.999.** At the 99.9th percentile you need genuine tail extrapolation, not robustification of the empirical quantile. Use `EQRNModel` from the `eqrn` subpackage, which fits a covariate-dependent GPD to the tail. WDRQR and EQRN are complementary: WDRQR for the 90th–99th percentile in thin segments, EQRN for the 99.9th+ where GPD extrapolation is required.

**For prediction intervals.** WDRQR gives a single point quantile estimate, not an interval with coverage guarantees. If you need distribution-free prediction intervals — for example, for Consumer Duty fair value evidence — use conformalized quantile regression from `insurance-conformal`. The two tools solve different problems: CQR guarantees coverage on new observations; WDRQR guarantees robustness of the quantile estimate to distributional shift.

**For frequency models.** Motor frequency data typically has thousands of observations per segment. Thin-data concerns are much less acute for claim counts than for severity. Standard Poisson GLM or CatBoost frequency model is reliable.

---

## Where this fits in the toolkit

The `insurance-quantile` library now has four distinct quantile estimation approaches, and they do not overlap:

| Tool | Best for | N regime | Guarantee type |
|---|---|---|---|
| `QuantileGBM` | Non-linear severity, large volume | N > 5,000 | Asymptotic consistency |
| `WassersteinRobustQR` | Linear model, thin segments, formal certificate needed | N = 50–2,000 | Finite-sample, robust to $\varepsilon$-shift |
| `EQRNModel` | Extreme quantile extrapolation (Q > 0.999) | N > 500 anchors | GPD tail fit |
| `TwoPartQuantilePremium` | Zero-inflated frequency-severity, aggregate quantile | Any N, but thin frequency is the challenge | Premium decomposition |

The typical workflow for commercial lines: fit `QuantileGBM` on aggregate portfolio, extract the thin segments (by claim count), refit those segments with `WassersteinRobustQR`, blend or take the more conservative estimate. For reinsurance attachment pricing, go directly to `WassersteinRobustQR` with a deliberately conservative $\varepsilon$.

---

## The implementation

`WassersteinRobustQR` is implemented in `_robust.py` (~425 LOC including the Polars API wrappers and tests). The core algorithm uses scipy and numpy — no CVXPY dependency. For $p=2$, the slope optimisation is an L-BFGS-B solve with explicit gradient on the regularised check loss; the intercept correction is computed analytically after the slope converges.

The implementation follows Option A from our research brief: closed-form for $p=2$ without a general convex solver. This keeps the dependency footprint within what `insurance-quantile` already requires. For $p \neq 2$ (which we have no insurance use case for), you would need CVXPY.

The module is available from `insurance-quantile` v0.4.0:

```bash
pip install "insurance-quantile>=0.4.0"
```

```python
from insurance_quantile import (
    WassersteinRobustQR,
    wdrqr_large_loss_loading,
    wdrqr_reserve_quantile,
)
```

---

## The paper

Zhang, Chunxu, Tiantian Mao, and Ruodu Wang. "Wasserstein Distributionally Robust Quantile Regression." arXiv:2603.14991 [math.ST, math.OC]. March 2026.

The paper is cleanly written and the theorem statements are precise. The proof of Theorem 1 uses a Fenchel duality argument specific to the piecewise-linear check loss structure — it is worth reading if you want to understand why $p=1$ is degenerate and why the intercept correction has the specific form it does. Theorem 3 (the finite-sample guarantee) is the practically critical result for insurance.

---

## Related posts

- [insurance-quantile: Tail Risk Quantile and Expectile Regression for Insurance](https://burning-cost.github.io/techniques/pricing/2026/03/07/insurance-quantile/) — the library overview
- [Conformalized Quantile Regression for Insurance Prediction Intervals](https://burning-cost.github.io/techniques/pricing/2026/03/24/conformalised-quantile-regression-insurance-prediction-intervals/) — CQR coverage guarantees on top of any quantile model
- [Transfer Learning for Thin Portfolios: What Works, What Doesn't](https://burning-cost.github.io/machine-learning/pricing/2026/03/31/transfer-learning-thin-portfolios-what-works-what-doesnt/) — borrowing from related portfolios when your segment is too small to model standalone
