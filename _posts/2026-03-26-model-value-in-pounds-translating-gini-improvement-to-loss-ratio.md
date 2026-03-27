---
layout: post
title: "Model Value in Pounds: Translating Gini Improvement to Loss Ratio"
date: 2026-03-26
categories: [pricing, model-governance, business-case]
tags: [loss-ratio, gini, pearson-correlation, price-elasticity, lre, business-value, insurance-monitoring, uk-motor, ps21-5, model-validation, python]
description: "The LRE metric converts Pearson correlation improvement into expected loss ratio change in basis points. Here is how to use it, and where it breaks."
---

Every pricing review ends the same way. The actuaries present a lift chart, a Gini improvement from 0.41 to 0.43, and a log-loss reduction of 3%. The CFO asks: "What is that worth?" The room goes quiet.

The problem is not that the actuaries do not know. It is that the standard validation metrics — Gini coefficient, double-lift chart ratio, normalised Gini index — are dimensionless. They rank model quality but do not translate to pounds. You need an additional step: a relationship between model discrimination and expected portfolio loss ratio.

Evans Hedges (2025), arXiv:2512.03242, provides one. It is the first published closed-form formula connecting a model validation metric to expected loss ratio. We implemented it in [`insurance-monitoring`](/insurance-monitoring/) v0.9.1 as the `business_value` module.

This post explains the formula, shows the calibrate-then-forecast workflow in code, and is honest about where the assumptions break down — particularly under PS21/5.

---

## The formula

Theorem 1 of Evans Hedges (2025) gives the expected portfolio loss ratio as a function of model quality:

```
LR = (1/M) * [(1 + ρ² CV⁻²) / (ρ² (1 + CV⁻²))]^{(2η−1)/2}
```

Four parameters:

- **ρ** — Pearson correlation between the model's predicted loss cost and the true loss cost. Not Gini. Not normalised Gini. Pearson ρ.
- **CV** — coefficient of variation of the true loss cost distribution (not of your predictions).
- **η** — price elasticity of demand. The demand model assumed is c(p) ∝ p^{−η}.
- **M** — pricing margin factor. For a 74% loss ratio target, M = 1/0.74 ≈ 1.351.

At ρ = 1 (perfect model), the ratio inside the power is exactly 1, so LR = 1/M — your target loss ratio. Any ρ < 1 inflates the LR above that. The excess is the Loss Ratio Error:

```
E_LR = [(1 + ρ² CV⁻²) / (ρ² (1 + CV⁻²))]^{(2η−1)/2} − 1
```

The derivation assumes multiplicative log-normal errors: λ̂ = λ · exp(ε), ε ~ N(0, σ²), ε independent of λ. The correlation ρ is then a deterministic function of σ² and CV, which Evans Hedges inverts to express LR entirely in terms of observables.

---

## Why Gini is not a drop-in substitute

If your validation suite reports Gini and not Pearson ρ, you cannot plug Gini directly into the formula. The two metrics measure related but distinct things:

- **Gini** — area under the Lorenz curve: how well does ordered risk ranking separate profitable from unprofitable business?
- **Pearson ρ** — linear correlation between predicted and actual loss cost magnitudes.

For a well-specified model on well-behaved data they will be close. They diverge in the presence of outliers (Pearson ρ is sensitive to them; Gini is not), non-linearity, or segmentation structure. Evans Hedges (2025) cites Frees (2014) on the connection between Gini and correlation but does not provide a conversion formula. We have not found one in the literature.

The practical implication: run both metrics in your validation suite. If you only have Gini, you need to compute ρ before using this framework. On most insurance portfolios this is a one-line calculation given your model predictions and outcomes.

---

## The calibrate workflow

The formula has three parameters you can observe (ρ, CV, M) and one you cannot directly observe (η). The right approach — and the one Evans Hedges recommends — is to reverse-calibrate η from historical data, then use it to forecast the impact of a proposed model change.

The `calibrate_eta` function inverts Theorem 1 using Brent's method: given an observed (LR, ρ, CV, M) from a previous model generation, find the η that makes the formula consistent with what actually happened.

```python
from insurance_monitoring.business_value import calibrate_eta, lre_compare

# Previous model generation: an older GLM
# Motor book targeting 74% LR, but actually ran at 86%
# The gap reflects model mispricing (among other factors)
rho_old_glm = 0.87   # Pearson correlation of the legacy GLM
cv           = 1.8   # CV of true loss costs (UK motor, moderately heterogeneous)
lr_observed  = 0.86  # Observed portfolio LR in that period
margin       = 1.0 / 0.74  # 74% LR target

eta_implied = calibrate_eta(
    rho_observed=rho_old_glm,
    cv=cv,
    lr_observed=lr_observed,
    margin=margin,
)

print(f"Implied η: {eta_implied:.3f}")
# Implied η: 1.185
```

The implied η of 1.185 is consistent with published UK motor elasticity estimates (direct-line, mid-range). Now use that η to evaluate the proposed GBM upgrade:

```python
# Current deployed GBM: rho=0.91
# Candidate upgrade: rho=0.94 on holdout
result = lre_compare(
    rho_old=0.91,
    rho_new=0.94,
    cv=cv,
    eta=eta_implied,
    margin=margin,
)

print(result)
# LREResult(rho 0.910->0.940, delta_lr=-282.6bps [improvement], lr_old=0.8185, lr_new=0.7902)

gwp_gbp = 250_000_000  # £250m GWP
saving  = abs(result.delta_lr) * gwp_gbp
print(f"Expected annual saving: £{saving:,.0f}")
# Expected annual saving: £7,065,556
```

The `LREResult` dataclass carries `rho_old`, `rho_new`, `cv`, `eta`, `margin`, `lr_old`, `lr_new`, `delta_lr`, `delta_lr_bps`, `e_lr_old`, `e_lr_new`. Multiply `delta_lr` by gross written premium to get the monetary figure.

Note: `lr_old=0.8185` is the formula's prediction for the current model at ρ=0.91. We calibrated η from the older model, not from observed data at ρ=0.91, so 0.8185 is a forward projection. Whether the current book is actually running at 82% depends on factors outside the model — market cycle, large losses, mix shift — that the formula cannot see.

---

## Diminishing returns are real

Theorem 3 of Evans Hedges (2025) formalises something actuaries know intuitively: for a fixed absolute improvement in ρ, the loss ratio benefit is monotonically decreasing in the starting ρ. Fixing a bad model is worth more than polishing a good one.

```python
scenarios = [
    (0.70, 0.73, "poor to mediocre"),
    (0.82, 0.85, "mediocre to good"),
    (0.91, 0.94, "good to strong"),
    (0.96, 0.99, "strong to near-perfect"),
]

for rho_old, rho_new, label in scenarios:
    r = lre_compare(rho_old, rho_new, cv=cv, eta=eta_implied, margin=margin)
    print(f"{label:25s}  Δρ=+0.03  ΔLR={r.delta_lr_bps:+.1f} bps")
```

```
poor to mediocre           Δρ=+0.03  ΔLR=-535.0 bps
mediocre to good           Δρ=+0.03  ΔLR=-364.7 bps
good to strong             Δρ=+0.03  ΔLR=-282.6 bps
strong to near-perfect     Δρ=+0.03  ΔLR=-247.7 bps
```

The same three-point ρ improvement is worth roughly twice as much when coming from ρ=0.70 as when coming from ρ=0.96. The gradient steepens significantly at low ρ. This is why the first GBM upgrade over a legacy GLM typically has a larger business case than any subsequent incremental improvement — and why the CFO's question "what is the marginal value of the next model refresh?" has a structurally diminishing answer.

---

## The η estimation problem

The calibrate workflow assumes η is stable across time and model generations. That is an assumption worth examining.

UK motor elasticity estimates vary substantially by channel and segment. Direct-line business tends to have higher elasticity (consumers are actively shopping) than broker-intermediated business (broker manages switching friction). Published estimates for UK motor personal lines range from roughly 0.5 to 2.5. The difference between η = 0.8 and η = 1.8 changes the basis-point estimate by a factor of five:

```python
for eta_test in [0.8, 1.0, 1.2, 1.5, 1.8]:
    r = lre_compare(rho_old=0.91, rho_new=0.94, cv=1.8, eta=eta_test, margin=margin)
    print(f"η={eta_test:.1f}  ΔLR={r.delta_lr_bps:+.1f} bps")
```

```
η=0.8  ΔLR=-118.2 bps
η=1.0  ΔLR=-201.8 bps
η=1.2  ΔLR=-289.5 bps
η=1.5  ΔLR=-428.9 bps
η=1.8  ΔLR=-578.3 bps
```

This is the primary source of uncertainty in any LRE-based business case. If you cannot calibrate η from historical data, treat the output as order-of-magnitude. Present a range across η scenarios rather than a single number.

For UK home insurance, CV is typically higher (5–15 for property contents versus 1.5–4 for motor) and elasticity estimates are lower. Higher CV amplifies the sensitivity to ρ imperfection; lower η dampens the basis-point impact. These effects partially offset, but not predictably.

---

## Where the formula breaks down

**Independence assumption.** The derivation requires ε to be independent of λ — the log-space pricing error must be uncorrelated with the true loss cost. This fails when your model is systematically biased for specific risk segments (heteroskedastic errors in log-space). The paper's own simulation shows MAPE exceeding 75% when independence is violated. If your residual plots show structure against predicted loss cost, the LRE numbers are unreliable.

**PS21/5 and non-proportional pricing.** The formula assumes proportional pricing: p = M · λ̂. Under the FCA's pricing practices rules (PS21/5, effective January 2022), insurers cannot freely price renewals on technical cost. Renewal rates are constrained by the equivalent new business rate, which breaks the clean proportional structure. The demand model embedded in the formula — c(p) ∝ p^{−η} with constant η — does not capture the dual pricing regime. For books with significant renewal volume, the independence violation MAPE of >75% is directly relevant.

**Portfolio size and LLN.** The formula relies on the law of large numbers to equate expected loss ratio with the ratio of expectations. For small portfolios or niche segments, sampling variability will dominate the signal.

**Calibration attribution.** When you observe LR > 1/M historically, the gap has multiple causes: model mispricing, large loss years, mix shift, reserving movements. `calibrate_eta` attributes the entire LR excess to model imperfection. If the observed LR overshoot was partly driven by a cat event or a soft market year, the implied η will be overstated, and the forecast impact of ρ improvement will be overstated in proportion.

None of these mean the metric is useless. They mean you should present LRE estimates with explicit assumption disclosure and a range, not as a point forecast.

---

## What to tell the CFO

The honest version of the business case looks like this:

> "The candidate model improves Pearson correlation from 0.91 to 0.94. Using historical loss ratio data to calibrate the elasticity parameter, we estimate this is worth between 120 and 430 basis points of loss ratio improvement, depending on the elasticity assumption. At £250m GWP, the central estimate is around £7m annually. The range is wide because we are not certain how price-sensitive our customers are, and the formula assumes proportional cost-plus pricing, which our renewal book does not fully satisfy under PS21/5."

That is more honest than "Gini went up 2 points", more informative than "the model is better", and more credible than pretending the sensitivity to η does not exist.

```bash
uv add insurance-monitoring
```

Source: [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

---

- [Your Book Has Shifted and Your Model Doesn't Know](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) — monitoring for covariate drift, which determines whether your calibrated η is still valid
- [Your GBM and GLM Are Not Competitors](/2026/02/28/your-gbm-and-glm-are-not-competitors/) — the mean model whose ρ you are measuring here
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) — when model degradation is large enough that a full refit changes the LRE picture materially
