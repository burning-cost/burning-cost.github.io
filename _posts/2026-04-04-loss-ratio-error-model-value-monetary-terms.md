---
layout: post
title: "Is Your Model Improvement Worth Building? The Loss Ratio Error Framework"
date: 2026-04-04
categories: [pricing, model-validation, techniques]
tags: [loss-ratio-error, LRE, model-value, pearson-correlation, price-elasticity, model-roi, diminishing-returns, motor-pricing, model-prioritisation, arXiv-2512-03242, lemonade, evans-hedges, uk-motor, gwp, model-development]
description: "C. Evans Hedges (Lemonade, December 2025) derives the first closed-form formula connecting model discrimination to expected loss ratio. LRE translates a correlation improvement into pounds — and shows why improving a weak model is worth more than polishing a strong one."
author: Burning Cost
math: true
---

Every pricing team has a model development backlog. A new telematics feature. A postcode clustering update. A GBM refit with 18 months of additional data. A flood score upgrade. Each of these takes weeks of data engineering, validation, governance, and deployment. The question that gets asked at backlog prioritisation — "is this worth building?" — is almost always answered informally: rough Gini lift estimate, optimistic revenue projection, no formal framework connecting the two.

C. Evans Hedges at Lemonade published a paper in December 2025 (arXiv:2512.03242) that provides the first closed-form answer. The Loss Ratio Error (LRE) metric converts a model correlation improvement directly into an expected change in loss ratio, given three parameters you can estimate from your own portfolio. That is the calculation pricing teams have been doing on the back of an envelope for years. This paper puts it on a proper mathematical footing.

---

## The problem with informal ROI estimates

When a pricing actuary says "the new telematics feature should be worth 2–3 Gini points, which translates to maybe £X on a book this size," they are doing an implicit conversion from model discrimination to monetary value. The conversion is real — a better-discriminating model genuinely produces a better loss ratio — but the mechanics of the conversion depend on things that vary significantly between books: how heterogeneous the risk is, how elastic customers are to price changes, and what margin you are running at.

Two pricing teams with identical Gini improvements on identical-size books can have very different monetary outcomes if one book has a claims CV of 2.0 and the other is at 4.0, or if one operates in a highly price-elastic direct channel and the other writes through captive brokers with less price sensitivity. The informal calculation treats these as constants. They are not.

LRE makes them explicit parameters. That is most of what the paper does: it insists you state your assumptions rather than absorb them into an order-of-magnitude guess.

---

## What the framework derives

The paper models prediction error multiplicatively. The predicted loss cost $\hat{\lambda}$ relates to the true loss cost $\lambda$ as:

$$\hat{\lambda} = \lambda \cdot e^\varepsilon, \quad \varepsilon \sim N(0, \sigma^2), \quad \varepsilon \perp \lambda$$

This is proportional error in log-space: the model is off by a fraction of the true cost, not by a fixed amount. Under proportional pricing ($p = M \hat{\lambda}$, where $M$ is the margin factor) and power-law demand ($c(p) \propto p^{-\eta}$), you can derive the portfolio-level expected loss ratio analytically via the law of large numbers.

The closed-form result (Theorem 1) is:

$$E[\text{LR}] = \frac{1}{M} \left[ \frac{1 + \rho^2 \cdot CV_\lambda^{-2}}{\rho^2 \cdot (1 + CV_\lambda^{-2})} \right]^{(2\eta - 1)/2}$$

where $\rho$ is the Pearson correlation between predicted and actual losses, $CV_\lambda$ is the coefficient of variation of true losses (the portfolio's inherent heterogeneity), $\eta$ is the price elasticity of demand, and $M$ is the pricing margin factor.

Loss Ratio Error (LRE) is then the excess above perfect-model loss ratio:

$$E_\text{LR} = \left[ \frac{1 + \rho^2 \cdot CV_\lambda^{-2}}{\rho^2 \cdot (1 + CV_\lambda^{-2})} \right]^{(2\eta - 1)/2} - 1$$

At $\rho = 1$ (perfect model), $E_\text{LR} = 0$. As $\rho$ falls, $E_\text{LR}$ rises monotonically.

The key thing to understand about $E_\text{LR}$ is what it is measuring. The framework models a competitive market where customers with elastic demand respond to relative prices. When your model is wrong, you systematically overprice good risks (they leave) and underprice bad risks (they stay) — adverse selection. $E_\text{LR}$ is not an additive adjustment to your observed loss ratio; it is a multiplicative factor capturing how much the adverse selection from model imperfection inflates the equilibrium loss ratio above what a perfect model would achieve. The ratio $(1 + E_\text{LR,new}) / (1 + E_\text{LR,old})$ is what you apply to your current claims cost to project the improvement from going from $\rho_\text{old}$ to $\rho_\text{new}$.

One valid parameter range note: LRE produces negative values when $\eta < 0.5$, implying a worse model improves the loss ratio — in practice, $\eta$ below 0.5 indicates the model adds no value over random assignment, and results in that regime should not be used for prioritisation decisions.

---

## What each parameter means in practice

**$\rho$ — Pearson correlation between predicted and actual losses.** This is not the Gini coefficient. The two are related on well-specified models but can diverge substantially with outliers or non-linear relationships. The LRE formula requires Pearson $\rho$; if your validation pipeline reports Gini, you need to compute $\rho$ separately. On UK motor frequency models, $\rho$ is typically in the range 0.3–0.7 depending on feature richness and market segment.

**$CV_\lambda$ — coefficient of variation of true losses.** This captures how spread out the underlying risk is. For motor frequency, where losses are relatively concentrated around a portfolio mean, $CV_\lambda$ might be 1.5–2.5. For motor bodily injury severity, where large claims dominate the distribution, $CV_\lambda$ can reach 5–8. Higher $CV_\lambda$ means more room for model error to affect selection — the penalty for a poor model is larger on a more heterogeneous book.

**$\eta$ — price elasticity of demand.** This is the hardest parameter to estimate and the most consequential one. $\eta$ quantifies how much purchased volume falls for each percentage point increase in price. In a fully competitive direct-to-consumer motor market, $\eta$ might be 1.5–2.5. In a broker-mediated commercial lines book with sticky relationships, $\eta$ might be 0.5–1.0. The FCA's evaluation of the pricing practices remedies (EP25/2, July 2025) found that renewal pricing constraints significantly reduced effective elasticity post-PS21/5, which means post-PS21/5 motor renewal $\eta$ values are lower than pre-PS21/5 estimates would suggest.

**$M$ — the margin factor.** $p = M\hat{\lambda}$ means $M = 1.05$ corresponds to pricing at 105% of technical cost (a 5% margin). This shifts the level of the loss ratio but does not change the shape of how LRE varies with $\rho$. For most UK personal lines, $M$ is in the range 1.02–1.12 depending on product, channel, and cycle position.

---

## The diminishing returns result

The most practically important result in the paper is Theorem 3: for a fixed percentage improvement in $\rho$, the improvement in loss ratio is monotonically decreasing in the starting value of $\rho$.

Put directly: if you improve $\rho$ from 0.3 to 0.35 (a 17% relative improvement), you get substantially more loss ratio benefit than if you improve $\rho$ from 0.7 to 0.75 (a similar relative improvement). The curve is steep at the low end and flat at the high end.

This has an obvious but routinely ignored implication for model prioritisation. Pricing teams typically direct the most development effort at their main models — the frequency and severity GLMs that already have the most features, the most historical data, the most tuning. The marginal Gini point on a mature motor frequency model is worth less than the first Gini point on a thin-data segment model, a new product line, or a model that has been running stale for three years.

The LRE framework quantifies this. If your motor frequency model is at $\rho = 0.65$ and your motorcycle model is at $\rho = 0.35$, the ROI calculation for improving each by five correlation points is not symmetric. The motorcycle model improvement is worth more per Gini point invested — potentially far more, depending on the portfolio parameters.

---

## A worked example: £50m GWP motor book

Take a private motor book. GWP £50m, current claims cost £39m (78% loss ratio). $CV_\lambda = 2.5$, current model $\rho = 0.65$. A proposed model refit is expected to improve $\rho$ to 0.70 based on holdout lift testing.

The monetary calculation is:

$$\text{improvement} = \text{claims cost} \times \left(1 - \frac{1 + E_\text{LR}(\rho=0.70)}{1 + E_\text{LR}(\rho=0.65)}\right)$$

The result depends heavily on $\eta$. Running the formula at three plausible values:

| $\eta$ | Description | $E_\text{LR}(0.65)$ | $E_\text{LR}(0.70)$ | Claims cost reduction (%) | Monetary (£m) |
|--------|-------------|---------------------|---------------------|--------------------------|---------------|
| 0.8 | Sticky book, broker channel | 0.26 | 0.21 | 4.1pp | £1.6m |
| 1.4 | Mid-elasticity, direct | 1.02 | 0.78 | 11.7pp | £4.6m |
| 2.0 | High elasticity, price comparison | 2.22 | 1.61 | 18.7pp | £7.3m |

The range — £1.6m to £7.3m from the same 5-point $\rho$ improvement — is not a sign of a poorly constructed model. It reflects real variation in how much a model improvement translates into improved selection depending on how elastic your customers are. The same improvement on a captive-renewal book (low $\eta$) is genuinely worth less than on a pure price-comparison writer.

The diminishing returns effect shows in the absolute numbers too. Running the same calculation for a book starting at $\rho = 0.75$ improving to $\rho = 0.80$ (same 5-point gain, higher starting point):

| $\eta$ | Claims cost reduction (%) | Monetary (£m) |
|--------|--------------------------|---------------|
| 0.8 | 2.7pp | £1.4m |
| 1.4 | 7.8pp | £3.9m |
| 2.0 | 12.6pp | £6.3m |

The same engineering effort, starting from a stronger model, produces consistently less monetary impact. The motorcycle model example holds: if the underdeveloped model is genuinely weaker, investing there first is more profitable per pound spent.

---

## How to actually use this

The paper recommends a reverse-calibration approach for practical use. You have historical data: an observed loss ratio for a prior model generation, the correlation that model achieved, and the margin you were running at. Plug those three knowns into the LRE formula and solve for the implied $\eta$. That calibrated $\eta$ is your portfolio's effective price elasticity, estimated from your own data rather than assumed from the literature.

Once you have $\eta$ calibrated, you can project forward. A proposed model improvement with an expected $\Delta\rho$ (from lift testing on holdout data) translates into a projected $\Delta E_\text{LR}$, which translates into a £ impact on the portfolio. That number goes into the prioritisation conversation. "This model improvement is projected to reduce the portfolio loss ratio by 7–9 points (depending on elasticity), which at current book size is £3.6m–£4.6m per annum" is a very different conversation from "we expect a 3 Gini point improvement."

Three caveats for UK application that the paper does not address.

**The demand model assumption.** LRE assumes power-law demand: $c(p) \propto p^{-\eta}$. The paper notes that linear demand produces acceptable approximations, but exponential demand breaks the derivation entirely. For UK motor, the academic evidence on demand model shape is mixed. The most defensible practical position is to treat $\eta$ as a sensitivity parameter and compute LRE under a range of plausible values — exactly as the table above does. The directionality of the answer — which model improvement is worth more — is usually stable across the range.

**Proportional pricing assumption.** The derivation requires $p = M\hat{\lambda}$: you price at a fixed multiple of your technical price. Most UK insurers have significant non-technical adjustments — retention pricing, telematics loadings, broker terms, cycle adjustments. These break the proportional pricing assumption to varying degrees. The LRE formula will still give directional guidance, but the numerical precision should not be overstated. Use it to rank model investments, not to set a budget to the nearest thousand pounds.

**PS21/5 renewal constraints.** The pricing practices rules cap the gap between new business and renewal prices. This creates a floor on how much a model improvement can reduce renewal loss ratios, because renewal prices cannot track the improved model as freely as new business prices can. The effective $\eta$ for the renewal book is lower than the new business $\eta$, and the LRE calculation should use separate $\eta$ estimates for each cohort rather than a blended figure.

---

## The simulation validation

The paper runs 625 simulations across 125 parameter combinations (5 replications each, portfolio sizes 40,000–200,000 policies) to check whether the closed-form formula holds under simulated data. Overall median MAPE is 17.6%, mean 30.2%. For $\rho \geq 0.5$, errors are typically below 15%.

The 30% mean MAPE at first looks discouraging. It is worth understanding what drives it. The large errors occur at low $\rho$ values (below 0.5), and particularly when the independence assumption — that prediction errors are uncorrelated with true losses — is violated. When that independence breaks down, MAPE exceeds 75%. The framework is not designed to diagnose models with correlated errors; it assumes you have already dealt with systematic heteroskedasticity in log-space.

For UK motor books with mature models (typically $\rho \geq 0.5$) and an independence check on the error structure, the 15% MAPE figure is the operative one. That means a projected improvement of £4.6m should be read as approximately £3.9m–£5.3m, not as a precise number. This is still substantially more disciplined than the informal "somewhere in the low millions" that currently drives backlog prioritisation.

---

## What this does not replace

LRE quantifies the loss ratio benefit of model discrimination. It does not quantify:

- **The value of improved calibration.** A better-discriminating model that is miscalibrated can actually produce a worse outcome in practice. Calibration and discrimination are separate failure modes — see the Murphy decomposition post.
- **Adverse selection dynamics.** A better-discriminating model attracts better risks, which feeds back into the claim distribution and can change $CV_\lambda$ over time. The paper is a static analysis.
- **Regulatory constraints.** Consumer Duty (PRIN 2A) and proxy discrimination rules (EP25/2) mean some model improvements that would maximise LRE are not available — a protected-characteristic proxy that improves $\rho$ by 0.03 cannot simply be added to the model. The LRE framework is blind to regulatory feasibility.
- **The cost side.** LRE quantifies the revenue benefit numerator. The investment required — data engineering, validation, governance, deployment, monitoring — is the denominator. The paper leaves that to you.

On the frequency-severity extension (Theorem 2): the paper shows LRE generalises to a frequency-severity structure where the total LRE is a product of the frequency and severity LRE terms, each with their own $\rho$ and $CV$. This is directly applicable to the standard UK motor pricing approach. For a combined model, improving frequency discrimination and severity discrimination independently contribute to the total LRE reduction, and you can attribute the projected benefit to each component separately.

---

## Implementation

No PyPI package for LRE exists as of April 2026. The formula is four lines of code:

```python
import numpy as np


def loss_ratio_error(rho: float, cv_lambda: float, eta: float) -> float:
    """
    Compute Loss Ratio Error (Evans Hedges, arXiv:2512.03242, 2025).

    Parameters
    ----------
    rho       : Pearson correlation between predicted and actual losses
    cv_lambda : Coefficient of variation of true losses
    eta       : Price elasticity of demand

    Returns
    -------
    E_LR : Loss ratio error (0 at perfect model, positive otherwise)
    """
    inv_cv2 = cv_lambda ** -2
    bracket = (1 + rho**2 * inv_cv2) / (rho**2 * (1 + inv_cv2))
    exponent = (2 * eta - 1) / 2
    return bracket ** exponent - 1


def lre_monetary_impact(
    rho_old: float,
    rho_new: float,
    cv_lambda: float,
    eta: float,
    current_claims_cost: float,
) -> dict:
    """
    Project monetary impact of a model correlation improvement.

    The LRE ratio (1 + E_LR_new) / (1 + E_LR_old) gives the proportional
    reduction in adverse selection cost. Applied to current claims cost,
    this gives the expected annual saving.

    Parameters
    ----------
    rho_old            : Current model Pearson correlation
    rho_new            : Projected model Pearson correlation
    cv_lambda          : Coefficient of variation of true losses
    eta                : Price elasticity of demand
    current_claims_cost: Current annual claims cost in £

    Returns
    -------
    dict: e_lr_old, e_lr_new, improvement_pp, monetary_gbp
    """
    e_old = loss_ratio_error(rho_old, cv_lambda, eta)
    e_new = loss_ratio_error(rho_new, cv_lambda, eta)
    improvement_frac = 1 - (1 + e_new) / (1 + e_old)
    return {
        "e_lr_old": e_old,
        "e_lr_new": e_new,
        "improvement_pp": improvement_frac * 100,
        "monetary_gbp": improvement_frac * current_claims_cost,
    }
```

For a prioritisation exercise, run this across each backlog item with its projected $\Delta\rho$, a shared set of portfolio parameters, and a plausible range of $\eta$ values. The output is a ranked table of model investments by projected monetary impact, with uncertainty bounds from the $\eta$ range. This is a defensible prioritisation basis; the informal equivalent is not.

---

## The broader point

Pricing teams invest heavily in model development but rarely subject that investment to the same rigour they apply to the models themselves. A feature that would not pass a validation uplift test gets added to the backlog based on optimistic intuition; a model refit that would have a material monetary impact sits undone because the Gini improvement does not look dramatic enough to justify the governance cycle.

LRE does not solve the governance friction. What it does is give the conversation a quantitative anchor. When the question is "is this worth building?", the answer should be a number with stated assumptions that can be challenged and revised, not a direction ("probably yes") based on informal lift estimates.

The paper is a first step — the assumptions are strong, the simulation validation carries real uncertainty, and UK-specific complications around elasticity estimation and PS21/5 constraints mean the numbers should be treated as indicative rather than precise. But "indicative and principled" is substantially better than "informal and unchallenged" for decisions that are typically worth seven figures.

---

## References

- Evans Hedges, C. (2025). "A Theoretical Framework Bridging Model Validation and Loss Ratio in Insurance." Lemonade. [arXiv:2512.03242](https://arxiv.org/abs/2512.03242)
- FCA EP25/2: Evaluation of General Insurance Pricing Practices remedies, July 2025
- FCA PS21/5: General Insurance Pricing Practices, effective 1 January 2022
- Frees, E.W. (2014). "Insurance Ratemaking and a Gini Index." *Journal of Risk and Insurance*, 81(2), 335–366.

Related on Burning Cost:

- [Recalibrate or Refit? The Murphy Decomposition Makes it a Data Question](/2026/02/28/recalibrate-or-refit/) — discrimination and calibration are separate failure modes; LRE measures discrimination value only
- [Calibration Testing That Goes Beyond the Residual Plot](/2026/03/09/insurance-calibration/) — balance property, auto-calibration, Murphy decomposition
- [Three-Layer Drift Detection: What PSI and A/E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) — monitoring after deployment
- [Building a GLM Frequency Model in Python: A Step-by-Step Tutorial](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/) — the baseline GLM whose Pearson correlation feeds directly into the LRE formula
- [Insurance Model Monitoring in Python: A Practitioner's Guide](/2026/04/04/insurance-model-monitoring-python-practitioner-guide/) — the ongoing measurement that detects ρ degradation in production
