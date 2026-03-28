---
layout: post
title: "Your Elasticity Model Has a 3x Bias"
description: "Most UK motor insurers think they know their price elasticity. They are probably wrong by a factor of 3–5, in the direction that makes them systematically mispricing. The evidence, the mechanism, and what actually fixes it."
date: 2026-03-28
categories: [pricing, causal-inference]
tags: [price-elasticity, endogeneity, DML, GIPP, PS21-5, insurance-causal, Woodard-Yi, double-machine-learning, UK-motor, PCW, Bayesian, Kumar, EP25-2]
---

Most UK motor pricing teams have a price elasticity estimate. It lives in a spreadsheet, or embedded in a logistic regression somewhere, and it is almost certainly wrong by a factor of three to five. The direction of the error is the uncomfortable part: you are underestimating how price-sensitive your customers actually are. Your model tells you a 5% rate increase costs you 7–8 points of retention. The true cost may be 20–25 points.

This is not a marginal miscalibration you can paper over with a sensitivity test. A true elasticity of −4 versus an assumed elasticity of −1.5 implies a completely different pricing strategy.

---

## What elasticity means here

Price semi-elasticity, as used in personal lines renewal modelling, is the percentage change in conversion (or retention) for a 1% change in price. An elasticity of −2 means a 1% price increase reduces retention by 2 percentage points. The number that goes into an optimisation model drives every downstream decision: the profit-maximising renewal price, the new business loading, the PCW competitiveness target.

Pricing actuaries in UK motor are familiar with this. The conventional practitioner range for PCW renewal semi-elasticity sits at roughly −1.5 to −3.0. Some teams estimate it more tightly. The number comes from some variant of a logistic regression fit on renewal offer and decision data.

We think that range is wrong. Not wrong in the normal sense of imprecise — wrong because the estimation method has a systematic bias that makes customers look less price-sensitive than they are. The true range, after correcting for that bias, is plausibly −3.0 to −6.0 or beyond.

---

## The endogeneity problem: how OLS deceives you

Price in insurance is not randomly assigned. Premiums are actuarially determined by risk profile: a high-risk customer gets a higher premium because that is what their risk profile warrants. That pricing process creates an endogeneity problem that biases any naive estimate of price sensitivity — and the bias always runs in the same direction: toward zero.

Here is the specific mechanism in a UK motor renewal context.

Your lapse model is approximately:

```
lapse_i = α + β · log(price_i) + γ · X_i + ε_i
```

The problem is that `price_i` is correlated with `ε_i`, the unobserved drivers of lapse propensity. Two channels create this correlation:

**Channel 1: loyal low-risk customers.** Low-risk customers receive lower premiums (actuarially correct). Low-risk customers also tend to be older, more stable, longer-tenured — and genuinely less price-sensitive, for reasons unrelated to price level. They renew not because the price is low but because they do not shop around. OLS sees low price and low lapse propensity together and attributes it partly to the price. The price coefficient gets pulled toward zero.

**Channel 2: voluntary excess selection.** High-risk customers are offered higher premiums. Some of them self-select into higher voluntary excess to reduce the quoted premium. Their higher excess means the effective premium they perceive is lower than the actuarial premium would suggest. This creates a negative correlation between unobserved risk appetite (propensity to take high excess) and the observed premium. OLS confounds this with price sensitivity.

Both channels attenuate the estimated β toward zero. Customers appear less elastic than they are.

---

## Woodard & Yi (2020): the empirical magnitude

The clearest empirical evidence for the scale of this bias comes from Woodard & Yi (2020), 'Estimation of Insurance Deductible Demand Under Endogenous Premium Rates', *Journal of Risk & Insurance* 87(2):477–500.

Their setting: US Federal Crop Insurance programme, where premium rates are set actuarially by government agencies and the price variation is driven by observable risk factors — structurally identical to the endogeneity mechanism in personal lines. They estimated demand elasticity using both naive OLS and a properly specified instrumental variables estimator. The finding: **OLS underestimates price sensitivity by a factor of 3–5**. The IV estimates revealed that farmers were far more responsive to deductible pricing than the naive regression suggested.

We should be explicit about the extrapolation here. US crop insurance is not UK motor. The products differ, the markets differ, and the UK PCW channel has no direct US analogue. We are not claiming the 3–5x figure is portable without adjustment. We are claiming it provides credible empirical evidence — in a setting with the same structural endogeneity mechanism — that this bias can be very large. There is no structural reason UK motor is immune to the same confounding.

If anything, the PCW market amplifies the problem. Around 66% of UK motor new business now transacts through comparison websites (FCA EP25/2, July 2025). PCW-acquired customers are self-selected price shoppers. Their observed price-conversion relationships are dominated by selection effects that standard regression cannot disentangle.

---

## The post-GIPP structural break

Even setting the endogeneity problem aside entirely, any elasticity estimate from pre-2022 renewal data is invalid for post-2022 optimisation. This is a separate issue that compounds the bias problem.

The FCA's General Insurance Pricing Practices rules (PS21/5, effective 1 January 2022 — GIPP) prohibited renewal prices exceeding the Equivalent New Business Price. Before GIPP, loyal customers were routinely overcharged at renewal, sometimes by 20–30% relative to the equivalent new business rate. Those customers renewed anyway — not because they were genuinely insensitive to price, but because the overcharge accumulated gradually and switching friction was high. Their revealed elasticity was artificially suppressed by inertia, not genuine preference.

Post-GIPP, the loyalty penalty mechanism is removed. Prices have converged. The FCA's three-year evaluation (EP25/2, July 2025) confirmed this: motor insurance premiums fell 5.9% in Q1 2022, with an estimated £1.6bn saving over 10 years. Price walking was largely eliminated across the 13 motor and 16 home insurers studied.

The structural change matters for demand modelling. Customers who previously stayed through inertia now face genuine market pricing. The retention book's effective price sensitivity has increased, because the segment held by the absence of information rather than genuine willingness to pay has been exposed to market rates for the first time.

A model trained on 2018–2021 data was calibrated in a market where a significant share of the retention book was price-insensitive for reasons that no longer exist. That model needs replacing — not recalibrating, replacing — on post-2022 data.

EP25/2 documented a further concern: of the 66 firms reviewed, 28 could not demonstrate ENBP compliance with sufficient granularity, and 27 had inadequate documentary evidence that their controls were working. The report explicitly flagged elasticity-based pricing models as a compliance risk: firms must have controls to prevent less price-sensitive customers being charged more for non-cost reasons. That requires understanding which customers are actually less price-sensitive — which requires a causally identified demand model, not a biased OLS estimate.

---

## The PCW rank problem

There is a third source of bias specific to the PCW channel, distinct from the standard endogeneity problem.

On a price comparison website, conversion is not a smooth function of absolute price. It is primarily driven by rank position: rank 1 on a PCW converts at around 20% in UK motor; rank 2 converts substantially less. The rank position is determined by your price relative to competitors — and competitor prices are unobserved.

Standard conversion models handle this by including rank as a control variable. This is wrong for two reasons.

First, rank is a *mediator*, not a confounder. Rank is caused by your price (and by competitor prices). If you control for rank in the regression, you are estimating the effect of price holding rank fixed — which is not the effect of a pricing decision. A pricing decision changes both your price and, through that, your rank. The direct effect of price (conditional on rank) is less than the total effect of price (through all channels including rank). Models that control for rank systematically underestimate the pricing team's actual leverage.

Second, rank itself is endogenous to unobserved factors. Competitor prices respond to market-wide claims cost trends that also affect your pricing. If claims inflation rises, all insurers reprice upward, your rank position is preserved, but the absolute price effect on conversion is confounded by market-wide demand shifts. Rank is not a clean control for competitor effects.

The correct treatment of rank is either as the running variable in a regression discontinuity design — exploiting the near-discontinuity in conversion at the rank 1/rank 2 boundary — or excluded from the primary price regression and handled through a structural model of the conversion-rank relationship. What it should not be is a standard covariate in a logistic regression. Most conversion models do exactly the wrong thing.

---

## What DML actually corrects — and what it does not

Double Machine Learning (Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, & Robins, 2018, *Econometrics Journal* 21(1):C1–C68) addresses the observed confounding problem. The procedure partials out the effect of observable risk factors from both the price variable and the outcome before estimating the price coefficient. Because the nuisance models are fit on separate folds, the approach avoids the regularisation bias that plagues naive high-dimensional regression.

Our [insurance-causal](https://github.com/burning-cost/insurance-causal) library implements this:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(treatment=PriceChangeTreatment())
model.fit(df, outcome="lapsed", treatment="price_change", features=rating_factors)
print(model.ate_)  # average treatment effect, with confidence interval
```

DML handles confounders you can measure and include. On synthetic data with known ground truth, it produces substantially less biased estimates than naive GLM when observable confounding is strong. This is not trivial — correcting for observed confounders through proper double-residualisation is a genuine improvement over fitting a logistic regression with risk factors as covariates.

What DML does not handle: the correlation between price and *unobserved* demand drivers. If there are customer characteristics that predict both price (through the actuarial model) and demand (through propensity to shop) that are not captured in your rating factors, those become part of the residuals and the DML estimate inherits the bias. This is the gap that instrumental variables fills.

Finding a valid instrument in insurance is hard. Useful candidates: broad commercial loading changes applied at quarterly rate reviews (variation in the commercial margin applied uniformly across segments); reinsurance cost events that shifted the technical price across whole portfolios without changing individual risk profiles; regulatory capital events. Each candidate requires careful argument for validity. None are as clean as the Woodard & Yi instruments in crop insurance.

---

## The Bayesian two-stage approach

The most complete solution proposed in the literature for this class of problem is a two-stage Bayesian framework from Kumar, Boluki, Isler, Rauch, and Walczak (arXiv:2205.01875, 2022), applied to airline pricing.

Stage one: fit ML models (including deep networks) to predict both purchase probability and price from observable customer features. Extract the residuals — the variation in price and conversion behaviour unexplained by observables. These residuals are the signal for stage two.

Stage two: fit a Bayesian Dynamic GLM on those residuals to estimate the price-sensitivity parameter β as a full posterior distribution, not a point estimate.

The paper reports estimation error reduced from 25% to 4% relative to direct regression on simulation data. The simulation context is important — these numbers were generated under controlled conditions, and real data will underperform that benchmark. But the direction is right. The two-stage structure exploits the fact that the residuals from well-specified ML models carry less of the observable confounding signal than raw prices and conversion rates do.

More importantly, the Bayesian output is what pricing optimisation actually needs. A point estimate of elasticity — "the semi-elasticity for 30–35 year-olds on PCW is −3.2" — is false precision. The optimiser that takes that number produces a confidence in the recommended price that is not warranted. A full posterior distribution over β enables Thompson sampling: at each pricing decision, draw a parameter realisation from the posterior and set the expected-value-maximising price. This naturally incorporates uncertainty into the pricing strategy, produces exploratory price variation that tightens the posterior over time, and is theoretically proven to achieve near-optimal cumulative performance.

No open-source insurance implementation of the Kumar et al. two-stage approach exists. The insurance-causal library has DML, but not a Bayesian posterior over the demand parameter. This is a documented gap.

---

## What the commercial platforms do not tell you

Akur8 and Earnix are the dominant commercial platforms for UK personal lines price optimisation. Akur8's demand module claims separate static (conversion propensity) and dynamic (price sensitivity) models, with a proprietary "Derivative Lasso" algorithm. Earnix's Price-It platform includes a conversion model connected to the rate optimisation engine.

Neither platform, as far as we can establish from public documentation, applies endogeneity correction to the elasticity estimates. The "Derivative Lasso" algorithm is not publicly described — we cannot rule out that it incorporates IV or DML-style debiasing, but there is no evidence that it does. Both platforms allow pricing actuaries to adjust elasticity assumptions by segment, which is a workaround rather than a solution: if the model is biased, a discretionary adjustment made without knowledge of the bias direction and magnitude is not a correction.

The audit consequence is also unresolved. Consumer Duty, FCA AI principles, and the emerging PRA supervisory expectations for AI in insurance all point toward firms being able to explain and justify their pricing decisions. A black-box elasticity estimate from a commercial platform, derived from an unspecified method on historical data of uncertain quality, does not meet that standard. If the FCA or PRA asks "how was price sensitivity estimated and what is the uncertainty on that estimate?", the answer needs to be technically defensible. "The platform calculated it" is not an answer.

A practical exercise that costs nothing: run your current optimisation at three elasticity values — say, −1.5, −3.0, and −5.0 for PCW motor renewal. If the profit-maximising price shifts materially across that range, you have a decision that is highly sensitive to an uncertain input. That uncertainty should be visible in the pricing decision, not hidden inside the platform output.

---

## Why this makes pre-GIPP models doubly wrong

The two problems compound. Pre-2022 renewal data was generated in a market where the loyalty penalty mechanism artificially suppressed revealed price sensitivity. Models trained on that data have a structural break problem. And those same models used OLS or logistic regression on the available data, embedding the endogeneity bias on top of the structural misspecification.

A team that trained their elasticity model in 2020 and has not rebuilt it on post-2022 data has an estimate with two layers of error stacked in the same direction: the pre-GIPP inertia effect and the OLS attenuation. The combined effect could easily push the true elasticity to three or four times the estimated value.

The practical consequence: if you believe elasticity is −1.5 and set renewal prices accordingly, you are charging more than the profit-maximising price in segments where true elasticity is −4 or beyond. You retain fewer customers than you expected, write business at lower volumes than planned, and when you re-estimate from the resulting data you inherit a biased sample — because the data now reflects both your incorrect pricing decisions and the customers who were lost. The bias is self-reinforcing.

---

## What to do about it

**Step one: estimate whether your current model is plausibly biased.** Run a DML estimation from insurance-causal on a year of post-2022 renewal data. Compare the DML estimate to your existing GLM elasticity. If the DML estimate is materially more negative, you have at least the observed-confounding component of the bias. If DML and GLM agree closely, either your existing model handles observable confounding well (unlikely if it is a simple logistic regression) or the DML is also biased in the same direction (possible if the observable confounders do not explain much of the price variation).

**Step two: find an instrument.** Look at rate review periods where the commercial loading changed uniformly across segments. These create price variation that is mechanically related to the loading decision, not to individual risk quality. Even imperfect instruments — ones that explain only some of the price variation — improve on pure observational estimation. The first-stage F-statistic tells you whether the instrument has enough relevance to be useful.

**Step three: rebuild on post-2022 data.** This is not optional. The pre-GIPP data is structurally invalid. Three years of post-January-2022 renewal observations are now available. Use them.

**Step four: quantify the uncertainty.** Whether through the Kumar et al. two-stage Bayesian structure or a simpler bootstrap of the DML estimates, you need a distribution over the elasticity parameter, not just a point estimate. Pricing decisions made without uncertainty quantification are falsely precise. The Bayesian posterior is the right goal; honest bootstrapped intervals are a reasonable starting point.

---

## What we do not know

The 3–5x bias estimate from Woodard & Yi is not a UK motor fact. It is an empirical finding from a different insurance product in a different country under a different regulatory regime, applied as a structural argument to UK motor. The size of the bias here could be larger or smaller. We do not have access to a large UK motor dataset with a valid instrument and will not pretend to have done that calculation.

The post-GIPP structural break is directionally certain — the mechanism is unambiguous — but the magnitude is empirical. How much of pre-GIPP retention stickiness was genuine price insensitivity versus inertia is not established from public data.

If you have estimated the true causal elasticity on UK motor PCW data with a proper instrument and found something substantially different from the naive OLS estimate — in either direction — we would very much like to know what you found.

---

## References

- Woodard, J.D. & Yi, F. (2020). Estimation of Insurance Deductible Demand Under Endogenous Premium Rates. *Journal of Risk & Insurance*, 87(2), 477–500.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Kumar, A., Boluki, S., Isler, V., Rauch, C., & Walczak, T. (2022). Machine Learning based Framework for Robust Price-Sensitivity Estimation with Application to Airline Pricing. arXiv:2205.01875.
- FCA (2021). General Insurance Pricing Practices: final rules. PS21/5. Financial Conduct Authority, January 2021.
- FCA (2025). Evaluation Paper 25/2: An evaluation of our General Insurance Pricing Practices (GIPP) remedies. Financial Conduct Authority, July 2025.
- Burning Cost [insurance-causal](https://github.com/burning-cost/insurance-causal): Double Machine Learning for causal price elasticity in insurance.
