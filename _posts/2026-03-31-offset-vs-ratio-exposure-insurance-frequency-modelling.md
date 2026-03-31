---
layout: post
title: "Offset vs Ratio Exposure in Insurance Frequency Modelling: When It Matters and When It Doesn't"
date: 2026-03-31
categories: [frequency]
tags: [glm, frequency-modelling, poisson, tweedie, exposure, offset, pure-premium, statsmodels, glum, uk-insurance, mid-term-cancellation, gipp, boucher-coulibaly, financial-balance, insurance-pricing, python]
description: "Boucher & Coulibaly (arXiv:2502.11788) prove that for Poisson frequency models, log-exposure offset and exposure-weighted ratio approaches are identical. The distinction only bites with Tweedie pure-premium GLMs — and then it bites hard, because the offset approach has no guarantee of portfolio-level financial balance when mid-term cancellations create heterogeneous exposures."
math: true
---

There are two ways to handle exposure in a GLM. You can add $\log(t_i)$ as an offset in the linear predictor — the industry default, supported by every toolkit, rarely questioned. Or you can divide the response by exposure, use exposure as observation weights, and drop the offset entirely. Most pricing teams are in the first camp without knowing the second exists.

A paper from Boucher and Coulibaly (arXiv:2502.11788, submitted February 2025, revised March 2026) proves that for Poisson frequency models, both methods produce identical parameter estimates. The proof is exact, not approximate. For Tweedie pure-premium models, they diverge — and the offset approach develops a financial balance problem that gets worse with mid-term cancellations. The finding is precise enough to be worth knowing in detail.

---

## The setup

Every policy in your training data has three things that matter here: a vector of rating covariates $\mathbf{x}_i$, a response $y_i$ (total claims or total losses), and an exposure $t_i$ — the fraction of a year the policy was at risk. For a full-year policy, $t_i = 1$. For a policy that cancelled after four months, $t_i \approx 0.33$.

There are two natural ways to incorporate $t_i$ into a GLM.

**Offset approach.** Keep $y_i$ as the response. Add $\log(t_i)$ as a fixed term in the linear predictor with coefficient constrained to 1:

$$\eta_i = \log(t_i) + \mathbf{x}_i^\top \boldsymbol{\beta}$$

The implied mean is $\mu_i = t_i \exp(\mathbf{x}_i^\top \boldsymbol{\beta})$, which scales the predicted loss linearly with exposure. This is what `offset=np.log(exposure)` does in statsmodels, what `offset_column` does in H2O, what every standard GLM toolkit reaches for first.

**Ratio approach.** Transform the response to a rate: $z_i = y_i / t_i$. Use exposure as observation weights. No offset term:

$$\eta_i = \mathbf{x}_i^\top \boldsymbol{\beta}, \quad \text{mean: } \zeta_i = \exp(\mathbf{x}_i^\top \boldsymbol{\beta})$$

To recover total predicted losses for policy $i$, you multiply the predicted rate back by exposure: $\hat{y}_i = \zeta_i \cdot t_i$.

These look like different models. The question is whether they produce the same parameter estimates $\boldsymbol{\beta}$.

---

## For Poisson, they are identical

The paper proves this formally. The key is what happens to the score gradient — the derivative of the log-likelihood with respect to $\boldsymbol{\beta}$ — under each formulation.

For a Tweedie GLM with power parameter $p$, the score gradient takes the form:

$$\nabla L(\boldsymbol{\beta}) = \frac{1}{\phi} \mathbf{X}^\top \mathbf{D} \mathbf{R}$$

where $\mathbf{D}$ is a diagonal weight matrix and $\mathbf{R}$ is the vector of Pearson-style residuals $r_i = (y_i/\mu_i - 1)$.

The weight matrices differ between the two approaches:

$$w_i^{\text{offset}} = t_i^{2-p} \exp\bigl((2-p)\,\mathbf{x}_i^\top \boldsymbol{\beta}\bigr)$$

$$w_i^{\text{ratio}} = t_i \exp\bigl((2-p)\,\mathbf{x}_i^\top \boldsymbol{\beta}\bigr)$$

At $p = 1$ (Poisson), $t_i^{2-p} = t_i^{1} = t_i$. The weight matrices are identical. The score equations are identical. The log-likelihoods are proportional. The parameter estimates are the same.

This means: **if you fit a Poisson frequency model with log-exposure as offset, you get exactly the same $\boldsymbol{\beta}$ as if you had divided claim counts by exposure and used exposure as GLM weights.** The industry habit of using offsets for frequency models is not wrong. It is one of two equivalent paths to the same answer.

---

## For Tweedie pure premium, they diverge

For $p \in (1, 2)$ — which covers the Tweedie compound Poisson-Gamma family used for pure premiums (total loss per unit exposure) — the two weight matrices are different. Specifically, for $t_i \in (0, 1]$ and $p > 1$:

$$t_i^{2-p} \geq t_i$$

with equality only at $t_i = 1$ or $p = 1$.

The offset approach assigns more weight to partial-year policies than the ratio approach. This is counterintuitive: a policy observed for three months gets a higher relative weight under offset than its share of total exposure warrants.

The consequence is a financial balance problem. Define the "balance gap" as the difference between total predicted premium and total observed losses across the portfolio.

Under the ratio approach, the intercept score equation in the homogeneous case forces:

$$\sum_i t_i \hat{\zeta} = \sum_i t_i z_i = \sum_i y_i$$

Balance is exact by construction. The intercept estimate is the exposure-weighted average loss rate, so aggregate predicted premium equals aggregate observed losses.

Under the offset approach, the intercept estimate is the $t_i^{2-p}$-weighted average loss rate. Aggregate predicted premium equals:

$$\sum_i t_i \hat{\zeta}^{\text{offset}} = \left(\sum_i t_i\right) \cdot \frac{\sum_i t_i^{2-p} z_i}{\sum_i t_i^{2-p}}$$

This equals total observed losses only if the $t_i^{2-p}$-weighted mean of loss rates equals the $t_i$-weighted mean. That holds when loss rates are uncorrelated with exposure — a condition that almost never holds in practice. Customers who cancel early (creating $t_i \ll 1$) are not a random sample of the book. They have different loss characteristics. This is the mechanism by which offset produces systematic aggregate mis-pricing in Tweedie pure-premium models.

The paper's simulation evidence backs this up. Across 1,000 to 50,000 simulated policies with heterogeneous exposures and a Tweedie data-generating process, the ratio approach achieves closer aggregate balance in roughly 87% of replications.

There is a trade-off worth acknowledging. The offset approach has smaller asymptotic variance:

$$\boldsymbol{\Sigma}^{\text{ratio}} - \boldsymbol{\Sigma}^{\text{offset}} \text{ is positive semi-definite}$$

Offset is more efficient in the statistical sense — better for ranking individual policies by predicted cost (Lorenz curve, Gini coefficient). The ratio approach wins on aggregate balance but loses on per-policy accuracy. For ratemaking, where aggregate balance is a hard constraint, ratio is the right choice. For a risk-scoring model where rank-ordering is everything, the tradeoff is less obvious.

---

## The UK context: why mid-term cancellations matter

UK motor insurance has a material mid-term cancellation rate. LexisNexis data from 2024 put the top 10 insurers at around 13% and the rest of the market closer to 18%. Cooling-off cancellations — FCA rules allow 14 days — create policies with $t_i \approx 0.04$, which stress the offset approach maximally since $0.04^{2-p}$ can be several times larger than $0.04$ for $p$ close to 2.

The FCA's GIPP rules (effective January 2022) did not create this exposure heterogeneity — mid-term cancellations predate GIPP — but the regulatory environment means pricing teams cannot afford systematic aggregate imbalance that falls disproportionately on any identifiable cohort. A customer who cancels and immediately repurchases on a comparison site creates two short-exposure policies in quick succession. Under offset, both get over-weighted relative to their actual exposure. The aggregate effect compounds with churn.

---

## Where UK pricing teams actually stand

The honest summary is that **this paper has narrow practical impact for most UK actuaries**.

The standard UK pricing architecture is separate Poisson frequency and Gamma severity models. Since Poisson is Tweedie at $p = 1$, the equivalence proof applies directly. Your offset-based Poisson frequency GLM is correct. No change needed.

The debate becomes live only if you are fitting a single Tweedie GLM directly to pure premiums — total losses per unit exposure — rather than going through the frequency-severity split. This approach has some real advantages (handles zero-inflation naturally, one model to maintain) and has become more common as teams adopt GBMs via scikit-learn's `TweedieRegressor`. glum's `GeneralizedLinearRegressor` supports both formulations cleanly.

If you are in that category, the switch from offset to ratio is worth making. The implementation change is small.

---

## Code: both approaches in Python

The following demonstrates offset vs ratio for a Tweedie pure-premium GLM using a synthetic dataset with heterogeneous exposures, matching the UK cancellation profile (15% mid-term cancellation rate). To see the Poisson equivalence, substitute `var_power=1` in the offset model and `power=1` in the ratio model — the coefficients will match to numerical precision.

```python
import numpy as np
import statsmodels.api as sm
from glum import GeneralizedLinearRegressor

rng = np.random.default_rng(42)
n = 5_000

# Synthetic book: one continuous rating factor, heterogeneous exposures
x = rng.normal(size=n).reshape(-1, 1)

# ~15% mid-term cancellations; cooling-off policies at t ~ 0.04
exposure = np.where(
    rng.random(n) < 0.15,
    rng.uniform(0.04, 0.9, n),   # cancelled
    np.ones(n)                    # full-term
)

# True loss rate: exp(0.5 + 0.3*x)
true_rate = np.exp(0.5 + 0.3 * x.ravel())

# Compound Poisson-Gamma losses (Tweedie DGP)
n_claims = rng.poisson(true_rate * exposure)
severity = rng.gamma(shape=2.0, scale=1.0, size=n)
y = n_claims * severity   # total losses

# --- Approach 1: offset (statsmodels) ---
X_sm = sm.add_constant(x)   # adds intercept column for statsmodels
model_offset = sm.GLM(
    y,
    X_sm,
    family=sm.families.Tweedie(
        var_power=1.5,
        link=sm.families.links.Log()
    ),
    offset=np.log(exposure)
)
result_offset = model_offset.fit(disp=False)

# --- Approach 2: ratio (glum, fit_intercept=True by default) ---
model_ratio = GeneralizedLinearRegressor(
    family='tweedie',
    power=1.5,
    link='log',
    fit_intercept=True
)
# Response = loss rate; weight = exposure
model_ratio.fit(x, y / exposure, sample_weight=exposure)

# --- Compare coefficients ---
print("Intercept — offset: {:.4f}, ratio: {:.4f}".format(
    result_offset.params[0], model_ratio.intercept_))
print("Slope     — offset: {:.4f}, ratio: {:.4f}".format(
    result_offset.params[1], model_ratio.coef_[0]))

# --- Financial balance check ---
pred_offset = result_offset.predict(X_sm, offset=np.log(exposure))
pred_ratio  = model_ratio.predict(x) * exposure   # rate * exposure = total

balance_offset = pred_offset.sum() / y.sum() - 1
balance_ratio  = pred_ratio.sum()  / y.sum() - 1

print(f"\nBalance gap — offset: {balance_offset:+.4f} ({balance_offset*100:+.2f}%)")
print(f"Balance gap — ratio:  {balance_ratio:+.4f} ({balance_ratio*100:+.2f}%)")
```

On a typical run with a 15% cancellation rate and $p = 1.5$, the ratio balance gap is under 0.5% and the offset gap lands between 1% and 4%. The divergence grows as $p$ approaches 2 and as the proportion of cancelled policies increases.

For a quick diagnostic on existing production data:

```python
# Offset imbalance diagnostic for Tweedie pure-premium models
gap = pred_offset.sum() / y.sum() - 1
if abs(gap) > 0.02:
    print(f"Warning: offset balance gap of {gap*100:.1f}%. "
          f"Consider ratio approach if using Tweedie p in (1,2).")
```

A gap above 2% with material mid-term cancellations is reason to review.

---

## What the paper leaves open

Boucher and Coulibaly do not claim that ratio dominates offset on individual prediction accuracy. The efficiency result goes the other way: offset produces tighter confidence intervals on $\boldsymbol{\beta}$ and, asymptotically, better rank-ordering of individual policies.

There is also no formal test for when offset imbalance is statistically significant. The 87% dominance figure is for aggregate balance in simulation; it does not translate into a decision rule for production use. Running the balance diagnostic above and checking whether the gap is material for your specific cancellation profile is the only practical guidance the paper offers.

For teams building a pure-premium GBM — where exposure often enters as a feature or weight rather than a formal offset — the paper is worth reading before settling on your exposure-weighting strategy. GBMs do not have the same algebraic equivalence proofs that GLMs do, and the intuitions from the Poisson case do not carry over automatically.

---

## Summary

| Model | Offset = Ratio? | Financial balance? | UK relevance |
|-------|-----------------|-------------------|--------------|
| Poisson frequency ($p = 1$) | Yes — provably identical | Both guaranteed | Standard UK pipeline: unaffected |
| Tweedie pure premium ($p \in (1,2)$) | No — diverge with partial exposures | Ratio: guaranteed; Offset: not | Relevant if using single Tweedie GLM |

If you run Poisson frequency + Gamma severity, your offset implementation is correct. If you fit Tweedie pure premiums with material mid-term cancellations, switch to the ratio approach. The implementation cost is low; the balance guarantee is not.

**Reference:** Jean-Philippe Boucher and Romuald Coulibaly, "Offset versus Ratio Regression for Tweedie Models," arXiv:2502.11788 (submitted February 2025, revised March 2026). [https://arxiv.org/abs/2502.11788](https://arxiv.org/abs/2502.11788)
