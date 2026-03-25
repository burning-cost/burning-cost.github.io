---
layout: post
title: "How to Quantify What a Model Improvement Is Worth in Pounds"
date: 2026-03-25
categories: [pricing, model-validation, techniques]
tags: [gini, loss-ratio, model-value, LRE, loss-ratio-error, discrimination, insurance-monitoring, model-validation, business-value, CFO, uk-motor, python, arXiv-2512-03242, hedges-2025]
description: "A 5pp Gini improvement means nothing to a CFO. The Loss Ratio Error framework from arXiv:2512.03242 converts model correlation into expected loss ratio — and from there into pounds. We work through the maths, correct a common misconception about the input metric, and show a Python example using insurance-monitoring."
---

Every UK pricing team has lived through this meeting. The models team presents: "we improved Gini from 42% to 47%." The Head of Pricing nods. The CFO asks: "what does that mean in money?" The models team says something about better risk separation, and the meeting ends without an answer.

This is a genuine gap. The Gini coefficient is the dominant discrimination metric in UK non-life pricing — it measures how well the model ranks risks — but it is dimensionless. A 5-percentage-point Gini improvement on a £300M motor book could be worth £1M in reduced adverse selection. It could also be worth £15M. Without a framework for the conversion, you are guessing, and investments in model improvement are evaluated on intuition rather than expected return.

A paper published in December 2025 (Hedges, arXiv:2512.03242) provides the conversion. It derives a closed-form relationship between model prediction accuracy and expected loss ratio — a quantity that is immediately meaningful to a CFO. This post works through the framework, explains where the assumptions bite, and shows how to use `insurance-monitoring`'s Gini APIs to compute the inputs you need.

---

## The Loss Ratio Error metric

The central result is a formula connecting model quality (measured by the Pearson correlation ρ between predicted and actual losses) to the expected loss ratio:

```
LR = (1/M) × ((1 + ρ²·CV⁻²) / (ρ²·(1 + CV⁻²)))^((2η−1)/2)
```

Where:

- **ρ** is the Pearson correlation between predicted and actual losses on holdout data
- **CV** is the coefficient of variation of true losses (standard deviation divided by mean — typically 1.5–3.5 for UK motor, higher for property)
- **η** is the price elasticity (absolute value of the demand elasticity; typical range 0.8–2.5 for personal lines)
- **M** is the target margin multiplier (1 / target_loss_ratio if you price to a fixed margin)

The **Loss Ratio Error** is defined as:

```
E_LR = LR × M − 1
     = ((1 + ρ²·CV⁻²) / (ρ²·(1 + CV⁻²)))^((2η−1)/2) − 1
```

This is the fractional excess loss ratio attributable to model imperfection. When ρ = 1 (perfect prediction), E_LR = 0. When ρ < 1, E_LR > 0: prediction errors mean you price some risks too low and others too high, and adverse selection does the rest.

---

## A note on ρ versus Gini

Before working the numbers, it is important to be precise about what ρ means here.

The Gini coefficient measures **rank discrimination** at the individual policy level: how consistently does the model assign higher predicted rates to policies that actually claim more? It is a rank-order statistic. A model that produces the right relative ordering everywhere will have high Gini regardless of whether its absolute predictions are accurate.

The Pearson correlation ρ in the LRE framework measures something different: the **linear agreement** between predicted and actual losses across the portfolio. For any deployed insurance model that is not catastrophically miscalibrated, ρ sits in the range 0.88–0.98. Even a weak model with Gini 35% will have ρ near 0.90 if it is roughly calibrated, because the large majority of policies agree closely in aggregate (most have zero claims; the model predicts a low rate; both are low).

This matters practically: **you cannot directly convert Gini to ρ** using a simple formula. Compute ρ directly from your holdout predictions:

```python
import numpy as np

rho = np.corrcoef(y_pred_holdout, y_actual_holdout)[0, 1]
```

For a well-calibrated UK motor frequency model with Gini 42%, a ρ in the range 0.92–0.94 is typical. Improving Gini to 47% typically corresponds to moving ρ to approximately 0.94–0.96. The exact correspondence depends on the distribution of losses and the model's calibration.

---

## Worked example: a £300M UK motor book

Suppose we have a UK motor book with £300M in written premium. The current frequency model has ρ = 0.93, and a retrained CatBoost model achieves ρ = 0.95 (corresponding to a Gini improvement of roughly 4–5pp).

Parameters:
- CV = 2.0 (coefficient of variation of individual losses — realistic for a UK motor frequency model)
- η = 1.2 (price elasticity — moderate, typical for direct personal motor)
- Target loss ratio = 65%

```python
import numpy as np

def loss_ratio_error(rho: float, cv: float, eta: float) -> float:
    """
    Loss Ratio Error from arXiv:2512.03242 (Hedges 2025).

    E_LR = ((1 + rho^2 * CV^-2) / (rho^2 * (1 + CV^-2)))^((2*eta - 1) / 2) - 1

    Parameters
    ----------
    rho : float
        Pearson correlation between predicted and actual losses. Range (0, 1].
    cv : float
        Coefficient of variation of true losses (std / mean). Typically 1.5–3.5.
    eta : float
        Price elasticity (demand elasticity, absolute value). Typically 0.8–2.5.

    Returns
    -------
    float
        Fractional excess loss ratio above the theoretical optimum.
        E_LR = 0 at rho = 1. E_LR > 0 for rho < 1.
    """
    cv_inv2 = 1.0 / cv**2
    numerator = 1.0 + rho**2 * cv_inv2
    denominator = rho**2 * (1.0 + cv_inv2)
    exponent = (2.0 * eta - 1.0) / 2.0
    return (numerator / denominator) ** exponent - 1.0


def expected_loss_ratio(rho: float, cv: float, eta: float, target_lr: float) -> float:
    """Expected loss ratio given model correlation and book parameters."""
    return (1.0 + loss_ratio_error(rho, cv, eta)) * target_lr


# Parameters
cv = 2.0
eta = 1.2
target_lr = 0.65
premium_volume = 300_000_000  # £300M

rho_old = 0.93   # current model (Gini ~42%)
rho_new = 0.95   # new model (Gini ~47%)

lr_old = expected_loss_ratio(rho_old, cv, eta, target_lr)
lr_new = expected_loss_ratio(rho_new, cv, eta, target_lr)
improvement_pounds = (lr_old - lr_new) * premium_volume

print(f"Current model:  ρ={rho_old}, expected LR={lr_old:.2%}")
print(f"New model:      ρ={rho_new}, expected LR={lr_new:.2%}")
print(f"LR improvement: {(lr_old - lr_new)*100:.2f} percentage points")
print(f"Value at £300M: £{improvement_pounds:,.0f}")
```

Output:
```
Current model:  ρ=0.93, expected LR=70.58%
New model:      ρ=0.95, expected LR=68.88%
LR improvement: 1.70 percentage points
Value at £300M: £5,104,000
```

A model improvement that lifts ρ from 0.93 to 0.95 is worth approximately **£5.1M** on a £300M motor book under these parameters. That is the number you bring to the CFO.

---

## Why the parameters matter enormously

The £5.1M figure is a point estimate within a wide range. The three parameters — CV, η, and the ρ estimate itself — are not constants.

**Elasticity (η) is the most sensitive.** The exponent (2η−1)/2 scales the entire calculation. At η = 0.8 (inelastic book — home insurance with limited price comparison), the same ρ improvement is worth £2.1M. At η = 2.0 (highly elastic, PCW-dominated motor), it is worth £11.9M. A 5x range from a single parameter.

```python
# Sensitivity to elasticity
print("Value of rho: 0.93 → 0.95 at different elasticities:")
for eta_val in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
    improvement = (
        expected_loss_ratio(rho_old, cv, eta_val, target_lr)
        - expected_loss_ratio(rho_new, cv, eta_val, target_lr)
    ) * premium_volume
    print(f"  η={eta_val:.1f}: £{improvement:,.0f}")
```

```
  η=0.8: £2,101,000
  η=1.0: £3,573,000
  η=1.2: £5,104,000
  η=1.5: £7,514,000
  η=2.0: £11,852,000
  η=2.5: £16,617,000
```

The paper recommends calibrating η from historical data: fit the implied elasticity that makes the model's predicted loss ratio match observed loss ratios across previous model generations. The paper provides parameter estimation guidance in Section 4.

**Diminishing returns are real and meaningful.** The same ρ improvement of +0.02 is worth substantially more at a lower starting point:

```python
# Diminishing returns: same +0.02 rho improvement at different baselines
print("Value of a +0.02 rho improvement at different starting points:")
for rho_base in [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]:
    rho_top = rho_base + 0.02
    improvement = (
        expected_loss_ratio(rho_base, cv, eta, target_lr)
        - expected_loss_ratio(rho_top, cv, eta, target_lr)
    ) * premium_volume
    print(f"  ρ {rho_base:.2f} → {rho_top:.2f}: £{improvement:,.0f}")
```

```
  ρ 0.80 → 0.82: £7,401,000
  ρ 0.85 → 0.87: £6,376,000
  ρ 0.88 → 0.90: £5,853,000
  ρ 0.90 → 0.92: £5,536,000
  ρ 0.92 → 0.94: £5,243,000
  ρ 0.94 → 0.96: £4,970,000
  ρ 0.96 → 0.98: £4,717,000
```

At ρ = 0.80 the same improvement is 57% more valuable than at ρ = 0.96. If you have a legacy severity model at ρ = 0.83 and a state-of-the-art frequency model at ρ = 0.95, the allocation question answers itself.

---

## Getting the inputs from insurance-monitoring

The LRE framework requires two ingredients: the ρ estimate and confidence intervals on the Gini improvement, to bound the uncertainty in the pound-denominated output.

`insurance-monitoring` provides the Gini toolkit. Start with the holdout Gini and a bootstrap CI:

```python
import numpy as np
from insurance_monitoring.discrimination import gini_coefficient, GiniDriftBootstrapTest

# Holdout data from model comparison experiment
# actual: observed claim frequency (claims / exposure)
# exposure: earned car-years
# pred_old: current model predictions
# pred_new: challenger model predictions

gini_old = gini_coefficient(actual, pred_old, exposure=exposure)
gini_new = gini_coefficient(actual, pred_new, exposure=exposure)

print(f"Current model Gini: {gini_old:.3f}")
print(f"New model Gini:     {gini_new:.3f}")
print(f"Improvement:        {(gini_new - gini_old)*100:.1f}pp")
```

Then verify the improvement is statistically real before making any business case:

```python
# Bootstrap CI on the Gini improvement
# We treat the old model's Gini as the "training" reference
test = GiniDriftBootstrapTest(
    training_gini=gini_old,
    monitor_actual=actual,
    monitor_predicted=pred_new,
    monitor_exposure=exposure,
    n_bootstrap=1000,
    confidence_level=0.95,
    random_state=42,
)
result = test.test()

print(test.summary())
# Gini change: +0.051 (95% CI: +0.038, +0.064)
# z = 7.93, p < 0.001
# Improvement is statistically unambiguous.
```

Once you have the bootstrap CI on the Gini change, compute ρ directly from the holdout data, then propagate the Gini CI bounds through the LRE calculation to get a pound-denominated range:

```python
# Compute rho directly from holdout predictions
rho_old_computed = float(np.corrcoef(pred_old, actual)[0, 1])
rho_new_computed = float(np.corrcoef(pred_new, actual)[0, 1])

# The Gini CI bounds correspond to approximate rho bounds
# For a rough translation: rho uncertainty scales with Gini uncertainty
# Safer: use the point rho estimates and note the Gini CI is your
# primary statistical evidence that the improvement exists at all.

lr_old_est = expected_loss_ratio(rho_old_computed, cv, eta, target_lr)
lr_new_est = expected_loss_ratio(rho_new_computed, cv, eta, target_lr)

improvement_point = (lr_old_est - lr_new_est) * premium_volume

print(f"ρ: {rho_old_computed:.3f} → {rho_new_computed:.3f}")
print(f"LR: {lr_old_est:.2%} → {lr_new_est:.2%}")
print(f"Value estimate: £{improvement_point:,.0f}")
print(f"Gini improvement 95% CI: [{result.ci_change_lower*100:+.1f}pp, {result.ci_change_upper*100:+.1f}pp]")
print(f"Statistical significance: p={result.p_value:.4f}")
```

Report the value estimate alongside the Gini CI. If the CI lower bound corresponds to a marginal improvement, adjust the business case accordingly. For a holdout of 10,000 policies, a 5pp Gini improvement typically has a 95% CI of roughly ±1.5pp — narrow enough to be confident in the direction and approximate magnitude, not narrow enough to treat the central estimate as precise.

---

## Connecting to ongoing monitoring

The LRE framework changes how you think about Gini drift in production.

If your deployed model's Gini drifts from 47% to 44% over a monitoring quarter, you can quantify the cost:

```python
from insurance_monitoring.discrimination import gini_drift_test_onesample

# Training Gini and rho stored at deployment time
stored_training_gini = 0.47
stored_rho = 0.95

# Q1 2026 monitoring data
monitor_result = gini_drift_test_onesample(
    training_gini=stored_training_gini,
    monitor_actual=q1_actual,
    monitor_predicted=q1_predicted,
    monitor_exposure=q1_exposure,
    n_bootstrap=500,
    alpha=0.32,  # one-sigma monitoring threshold (arXiv 2510.04556)
)

if monitor_result.significant:
    # Approximate rho at current Gini by scaling from training values
    gini_ratio = monitor_result.monitor_gini / stored_training_gini
    rho_current = stored_rho * gini_ratio  # rough approximation; compute directly if data available

    lr_deployed = expected_loss_ratio(stored_rho,   cv, eta, target_lr)
    lr_current  = expected_loss_ratio(rho_current,  cv, eta, target_lr)

    drift_cost_annual = (lr_current - lr_deployed) * premium_volume

    print(f"Gini drift: {stored_training_gini:.0%} → {monitor_result.monitor_gini:.0%}")
    print(f"LR impact:  +{(lr_current - lr_deployed)*100:.2f}pp")
    print(f"Annual cost at £300M book: £{drift_cost_annual:,.0f}")
```

This closes the loop: Gini drift is no longer an abstract technical indicator. It is a financial number that belongs in a model refresh conversation.

---

## Where the framework breaks down

We should be honest about the assumptions. The derivation in arXiv:2512.03242 requires:

1. **Log-normal multiplicative error structure**: the model error is ε ~ N(0, σ²), and predicted loss = true_loss × e^ε. Reasonable for GLMs with log link; less defensible for threshold-based systems or models with systematic segment bias.

2. **Error independence**: model prediction errors are uncorrelated with true risk level. This is the hardest assumption. Models trained with insufficient data on rare segments (young drivers, exotic vehicles) have correlated errors — systematically wrong in specific parts of the risk space, not randomly wrong everywhere. The LRE formula will understate the cost of that kind of error.

3. **Constant price elasticity**: the power-law demand model c(p) ∝ p^(−η) ignores non-linearity around price thresholds (the PCW comparison effect at ±5% of market rate) and portfolio composition shifts. For heavily PCW-exposed books, estimate η from causal models — see `insurance-causal`'s `PremiumElasticity` for a DML-based approach that avoids the confounding present in naive elasticity estimates.

4. **ρ estimation**: the formula is sensitive to ρ near 1, so estimation error matters. Compute ρ on a held-out sample of adequate size (10,000+ policies for stable estimates). Cross-validate if the holdout is small.

None of these assumptions are fatal. They mean the output is an order-of-magnitude calculation with explicit parameter uncertainty. A range of £3M–£8M is more useful than false precision of £5,104,231.

---

## What to actually do with this

The framework produces two practically useful outputs:

**For model investment decisions:** run the LRE calculation at the start of a modelling project, not at the end. Estimate the likely ρ improvement from a scope-of-work, plug it in, get a pound range. That range is your budget ceiling. If the modelling spend exceeds the expected value, the business case is not there.

**For monitoring escalation:** instrument the LRE calculation into your Gini monitoring. When `insurance-monitoring` flags a Gini drift, translate it immediately into an expected annual loss impact. A Gini drop that was previously a yellow-flag technical metric becomes a financial number that the Head of Pricing can act on.

The Gini coefficient is a good model metric. "£5M" is a good CFO metric. The LRE framework — with three parameters you can calibrate from your own portfolio history — gives you both in the same sentence.

---

## Libraries used

- [insurance-monitoring](https://burning-cost.github.io/insurance-monitoring): `gini_coefficient`, `GiniDriftBootstrapTest`, `gini_drift_test_onesample`
- [insurance-causal](https://burning-cost.github.io/insurance-causal): `PremiumElasticity` for DML-based elasticity estimation (for calibrating η)

## Reference

Hedges, C.E. (December 2025). "A Theoretical Framework Bridging Model Validation and Loss Ratio in Insurance." [arXiv:2512.03242](https://arxiv.org/abs/2512.03242)

## Related posts

- [Sequential A/B Testing for Insurance Champion/Challenger Experiments](/2026/03/24/sequential-ab-testing-insurance-champion-challenger/) — for verifying that a Gini improvement is statistically real before committing to a model swap
- [Does GBM-to-GLM Distillation Actually Work?](/2026/03/24/does-gbm-glm-distillation-work-insurance-pricing/) — benchmark results that feed directly into the LRE calculation
