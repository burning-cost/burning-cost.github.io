---
layout: post
title: "Offset vs Ratio Exposure in Tweedie GLMs: When It Matters (and When It Doesn't)"
date: 2026-04-03
categories: [techniques, research]
tags: [tweedie, glm, exposure, pure-premium, poisson, frequency-severity, statsmodels, glum, financial-balance]
description: "Boucher & Coulibaly (arXiv:2502.11788) prove that offset and ratio exposure handling are equivalent for Poisson frequency models — but diverge for Tweedie pure-premium GLMs, where the ratio approach guarantees portfolio-level financial balance and the offset approach does not."
---

Every UK pricing actuary has written `offset=np.log(exposure)` without losing a moment's sleep over it. The offset approach — fixing `log(t_i)` as a term in the linear predictor so that `mu_i = t_i * exp(X^T beta)` — is what the textbooks say, what the training courses teach, and what every team's existing pipeline does. It is, in short, unexamined.

Boucher & Coulibaly (arXiv:2502.11788, revised March 2026) gave it a proper examination. Their main result is reassuring for most people: for Poisson frequency models, offset and ratio are provably identical. But for Tweedie pure-premium GLMs — fitting total losses or loss rates directly with `p` strictly between 1 and 2 — the two approaches diverge in a financially material way. The ratio approach guarantees portfolio-level balance. The offset approach does not.

If you model frequency (Poisson) and severity (Gamma) separately, stop reading. Nothing changes. If you fit a single Tweedie model to pure premiums, read on.

---

## The two approaches

The **offset approach** is what most GLM frameworks default to. The response is total losses `y_i`, and `log(t_i)` enters the linear predictor with a coefficient fixed at 1:

```
eta_i = log(t_i) + X_i^T beta
mu_i  = t_i * exp(X_i^T beta)
```

The **ratio approach** normalises the response by exposure and uses exposure as sample weights. The response becomes `z_i = y_i / t_i` (loss rate per unit time), and `t_i` enters as the GLM weight:

```
eta_i  = X_i^T beta
zeta_i = exp(X_i^T beta)
```

with the Tweedie log-likelihood multiplied by `t_i` for each observation.

The score gradients look similar until you examine the weight matrices. For the offset approach:

```
w_i^O = t_i^(2-p)
```

For the ratio approach:

```
w_i^R = t_i
```

When `p` is strictly between 1 and 2 and exposure `t_i` is in `(0, 1]`, we have `t_i^(2-p) >= t_i`, with equality only when `t_i = 1` (full-year policies) or `p = 1` (Poisson). The offset approach puts *more weight* on partial-year observations than their actual exposure share warrants.

---

## Why Poisson is different

For Poisson (`p = 1`), the weight matrices collapse to the same thing:

```
D^O = diag( t_i^(2-1) * exp(X_i^T beta) ) = diag( t_i * exp(X_i^T beta) )
D^R = diag( t_i * exp(X_i^T beta) )
```

`D^O = D^R` exactly. The score equations are identical. Both log-likelihoods reduce to:

```
sum_i t_i * (-exp(X_i^T beta) + z_i * X_i^T beta)   (up to constants)
```

Fitting a Poisson GLM with `log(exposure)` as offset gives exactly the same parameter estimates as fitting a Poisson GLM on `claims/exposure` with `exposure` as sample weights. This is a theorem, not an approximation.

This means: if you use the offset approach in a Poisson frequency model, you are already doing the ratio approach. The parameter estimates, standard errors, and predictions are identical. Nothing in your pipeline needs to change.

---

## Where offset breaks: Tweedie with heterogeneous exposure

The Poisson equivalence breaks as soon as `p > 1`. The mechanism is easiest to see in the homogeneous intercept-only case.

The ratio approach QMLE is:

```
zeta_hat^R = sum(t_i * z_i) / sum(t_i)
```

This is the exposure-weighted average loss rate. Multiply by total exposure to get aggregate premium:

```
sum_i t_i * zeta_hat^R = sum_i t_i * z_i = sum_i y_i
```

Balance is exact by construction. The score equation for the intercept forces `predicted aggregate = actual aggregate losses`. Always.

The offset approach QMLE is:

```
zeta_hat^O = sum(t_i^(2-p) * z_i) / sum(t_i^(2-p))
```

This is a `t_i^(2-p)`-weighted average loss rate. The aggregate premium is:

```
sum_i t_i * zeta_hat^O = [sum(t_i)] * [t_i^(2-p)-weighted mean of z_i]
```

This equals `sum_i y_i` only if the `t_i^(2-p)`-weighted mean of `z_i` equals the `t_i`-weighted mean — which requires loss rates to be uncorrelated with the difference in weighting schemes. That is not a testable assumption you can rely on; it is an implicit requirement that breaks systematically whenever cancellation behaviour and loss experience are connected.

The paper backs this with simulation: across scenarios with `n` from 1,000 to 50,000, the ratio approach achieves closer aggregate balance in roughly 87% of replications.

---

## The UK cancellation problem

This matters in UK personal lines because mid-term cancellation rates are substantial. LexisNexis data from 2024 put cancellation rates at approximately 13% for the top 10 motor insurers and 18% for the rest of the market. A 15% cancellation rate, assuming cancellations are distributed through the year, means average exposure of about 0.5 years for cancelled policies. Cooling-off cancellations (14-day statutory period) produce `t_i ~ 0.04` — these maximally stress the offset approach because `t_i^(1-p)` grows large as `t_i` approaches zero.

The problem compounds when cancellation is correlated with risk. Early adverse selection — a bad-risk customer who claims quickly then cancels — produces exactly the scenario where the offset approach's over-weighting of short-exposure contracts creates predictable aggregate bias. The offset model treats the claim from the 6-week policy as if it represents a more substantial share of the portfolio than it does.

Telematics and pay-as-you-go products are the extreme case. Exposure heterogeneity there spans nearly the full `(0, 1]` range by design.

---

## What to do in practice

**If you use separate Poisson frequency and Gamma severity models:** nothing. The equivalence result covers you at the frequency stage. The offset approach in your Poisson GLM is identical to the ratio approach. Severity with a Gamma GLM and exposure weights is a separate question — check whether your exposure scaling there is intentional.

**If you use a single Tweedie GLM for pure premiums:** switch to the ratio approach. The implementation in glum is straightforward:

```python
from glum import GeneralizedLinearRegressor
import numpy as np

model = GeneralizedLinearRegressor(
    family='tweedie',
    power=1.5,
    link='log'
)
model.fit(
    X,
    y_total / exposure,    # response: loss rate z_i = y_i / t_i
    sample_weight=exposure  # weight: t_i
)

pred_rate  = model.predict(X_new)
pred_total = pred_rate * exposure_new   # scale back to total losses

# Financial balance check — should be near zero
gap = (pred_rate * exposure).sum() / y_total.sum() - 1.0
```

glum already defaults to this pattern in its tutorials; if your team has followed the glum freMTPL2 tutorial, you are likely already using ratio. statsmodels users fitting Tweedie with `offset=np.log(exposure)` are using the offset approach and should check their balance.

Before switching, run the diagnostic on your existing model:

```python
gap_offset = (pred_offset * exposure).sum() / y_total.sum() - 1.0
```

If `|gap_offset|` is below 1% and mid-term cancellations are modest, the practical impact may be negligible. If it exceeds 2% on a book with material cancellations, we would switch.

**There is a trade-off to acknowledge:** the offset approach is asymptotically more efficient — it has smaller parameter variance because its weight matrix dominates the ratio weight matrix element-wise. If individual policy rank-ordering (Gini, Lorenz curve, D-statistic) matters more to you than aggregate balance, the offset approach may perform better on those metrics. The paper's 87% dominance for ratio is specifically about aggregate financial balance, not individual prediction accuracy.

---

## What this paper doesn't resolve

The paper identifies the problem but does not provide a formal diagnostic test for when offset imbalance is statistically significant. Running a balance check on held-out data is the closest available substitute.

It also does not address the interaction with estimated `p`. When you estimate the power parameter from data (as most implementations allow), offset and ratio models may produce different estimates of `p`, which makes direct comparison messier. If your pipeline estimates `p`, check that the comparison is apples-to-apples.

Finally, there is no IFoA or CAS guidance note on this distinction as of today. The Boucher & Coulibaly paper is the only formal actuarial analysis of it. The CAS monograph (Goldburd et al.), Anderson et al. 2007, and the standard UK GLM references all document offset as the default without formal comparison to the ratio alternative.

---

## The bottom line

For most UK pricing teams, who model frequency and severity separately, this paper is confirmation that what you are doing is correct. The offset in your Poisson GLM is not a choice with a better alternative — it is equivalent to the ratio approach by proof.

For teams running single Tweedie pure-premium GLMs — increasingly common as GBMs and neural nets push teams toward end-to-end models — the offset approach carries a hidden assumption about the independence of loss rates and exposure that UK cancellation patterns will violate. The fix is simple: divide the response by exposure, pass exposure as weights, and scale predictions back. The financial balance guarantee is worth the minor implementation change.

---

**Reference:** Jean-Philippe Boucher & Ilias Coulibaly, "Offset vs. Ratio Approaches for Incorporating Exposure in Regression Models for Insurance," arXiv:2502.11788 (revised March 2026). [https://arxiv.org/abs/2502.11788](https://arxiv.org/abs/2502.11788)
