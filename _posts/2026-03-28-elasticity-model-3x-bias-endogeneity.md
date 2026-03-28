---
layout: post
title: "Your Elasticity Model Has a 3x Bias"
description: "Most UK motor insurers think they know their price elasticity. They are probably wrong by a factor of 3–5, and in the direction that makes their pricing strategy too aggressive. Here is why, and what to do about it."
date: 2026-03-28
categories: [pricing, causal-inference]
tags: [price-elasticity, endogeneity, DML, GIPP, PS21-5, insurance-causal, Woodard-Yi, double-machine-learning, UK-motor, renewal-pricing]
---

Most UK motor pricing teams have a price elasticity estimate. It sits in a spreadsheet or a pricing model somewhere, probably derived from a GLM on renewal data, and it is probably wrong by a factor of three to five. The direction of the error makes your pricing strategy too aggressive: you are underestimating how sensitive customers are to price, so you are charging more than you should and losing more customers than you expect.

This is not a marginal miscalibration. A true elasticity of −4 versus an estimated elasticity of −1.5 is a different pricing strategy entirely.

---

## The endogeneity problem

Price is not randomly assigned. Premium rates are actuarially determined by risk profile, and that creates an endogeneity problem that biases any naive estimate of price sensitivity.

The specific mechanism: actuarially-priced premiums mean that high-risk customers face higher premiums. High-risk customers also tend to buy with higher voluntary excess levels — not because of any optimising response to price, but because they are more willing to self-insure small losses, or because of correlated selection effects. Their higher excess reduces the effective premium they see, creating a negative correlation between unobserved customer characteristics (propensity to buy at high excess) and the observed premium level. The result is that price variation in your data is systematically correlated with unobserved demand characteristics, which contaminates the OLS coefficient.

In a lapse model of the form

```
lapse_i = α + β · price_i + γ · X_i + ε_i
```

this correlation between price and the error term attenuates the estimated β toward zero — making customers appear *less* price-sensitive than they actually are. You observe: high-price customers and low-price customers both have moderate lapse rates (the high-price customers are also higher-excess customers, absorbing some of the price effect). You conclude: price does not move lapse much. You are wrong.

There is a second attenuation mechanism specific to renewal books. Low-risk, loyal customers tend to have both lower premiums and lower lapse propensity. This creates a positive correlation between unobserved loyalty and low price. OLS sees this and partially cancels the price signal: the loyal-and-cheap customers are hard to distinguish from the price-insensitive-and-cheap customers. The elasticity estimate gets pulled toward zero.

Both mechanisms work in the same direction: OLS underestimates the absolute magnitude of price sensitivity. Customers are more elastic than your model thinks.

We have written separately about the causal DAG underlying PCW quote data: [The PCW Endogeneity Problem](/2026/03/26/the-pcw-endogeneity-problem-why-your-conversion-model-is-biased/). This post focuses on the magnitude of the bias and what to do about it.

---

## Woodard & Yi (2020): the empirical magnitude

The clearest empirical evidence for the scale of this bias comes from Woodard & Yi (2020), who studied price elasticity estimation in US crop insurance. The US federal crop insurance programme involves actuarially-determined premiums with a government subsidy structure that creates correlated price variation across crops, farms, and years — a setting structurally similar to the endogeneity problem in personal lines. They estimated price sensitivity using both naive OLS and an instrumental variables approach, and found that OLS underestimates price sensitivity by a factor of 3–5. The naive estimates suggested moderate elasticity; the IV estimates revealed farmers were far more responsive to price.

We should be honest about the extrapolation: US crop insurance is not UK motor. The products are different, the markets are different, the selection mechanisms are different. We are not claiming Woodard & Yi proves UK motor elasticity is biased by exactly 3–5x. We are claiming it provides credible empirical evidence that the endogeneity bias can be large — much larger than most practitioners assume — and that there is no structural reason UK motor should be immune to the same mechanism.

The structural conditions for the bias are present in UK motor: actuarially-priced premiums, risk-correlated voluntary excess choices, loyal low-risk customers creating attenuation in the renewal dataset, and no randomised pricing experiment to identify the true causal effect. If anything, the UK PCW market — where roughly three-quarters of new business is transacted via price comparison — creates stronger selection effects than the managed distribution channels that dominate in the US.

---

## What the practitioner estimate actually tells you

If you have estimated elasticity from your own renewal book, the number is not useless. It tells you the average association between price change and lapse in your historical data, conditional on your rating factors. That is genuinely informative for short-term forecasting in a stable environment. But it is not a causal estimate. It answers "what happened to lapse when price went up?" not "what would happen to lapse if we raised price?".

The distinction matters when you use the elasticity for optimisation. A price optimisation routine that maximises expected profit subject to a lapse model is implicitly treating the elasticity estimate as causal. If the causal elasticity is −4 and you are using −1.5, you will set prices too high, your retention will be worse than the model predicted, and when you re-estimate the elasticity from the resulting data, you will get an even more confused answer because the new data reflects your optimisation decisions.

This is a feedback loop that makes the bias self-perpetuating.

---

## The post-GIPP structural break

Even setting aside endogeneity, any elasticity estimate derived from pre-2022 data is invalid for post-2022 optimisation. The FCA's General Insurance Pricing Practices rules (PS21/5, effective January 2022 — colloquially "GIPP") fundamentally changed the data-generating process.

Before GIPP, loyal customers were systematically overcharged at renewal relative to equivalent new business customers. The overcharge compounded over multiple renewal cycles, sometimes reaching 20–30%. These customers continued to renew anyway, because the friction of switching was high and many did not know they were being overcharged. Their revealed elasticity was artificially low — not because they were genuinely insensitive to price, but because the loyalty penalty happened gradually and they lacked an obvious comparison point.

After GIPP, renewal prices must converge toward equivalent new business prices. The loyalty penalty mechanism is removed. Customers who were previously sticky because they were not actively comparing now face prices closer to market rates. The effective price sensitivity of the retained book has increased, because the segment held by inertia rather than genuine price preference has been exposed to market pricing.

If your elasticity model was estimated on 2018–2021 data, it was estimated in a market where a significant share of the customer base was being retained by the absence of information rather than genuine willingness to pay. That model is structurally wrong for the post-2022 market, independently of the endogeneity problem.

---

## DML as a partial solution

Double Machine Learning (Chernozhukov et al. 2018) handles the observed confounding problem. The idea is to partial out the effect of observable risk factors from both the price variable and the outcome, then estimate the price coefficient from the residuals. Because the nuisance models are fit on a separate fold, the procedure avoids the regularisation bias that plagues naive high-dimensional regression.

Our [insurance-causal](https://github.com/burning-cost/insurance-causal) library implements this:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(treatment=PriceChangeTreatment())
model.fit(df, outcome="lapsed", treatment="price_change", features=rating_factors)
print(model.ate_)  # average treatment effect with confidence interval
```

The DML estimate corrects for confounding by observable rating factors. On synthetic data with known ground truth, it produces less biased estimates than a naive GLM when the confounding is strong. We have been honest in our benchmarks: when treatment variation is largely explained by observed factors and the nuisance models over-partial, DML can be imprecise. It is not a magic fix.

The more important limitation is conceptual: DML handles observed confounders. The endogeneity that Woodard & Yi identified — the correlation between price and unobserved demand characteristics — requires an instrument. If you cannot find a valid instrument (a variable that affects price but is independent of unobserved demand shocks), DML will reduce but not eliminate the bias.

Finding a valid instrument in insurance is hard. Reinsurance cost shocks that affected pricing guidelines but were independent of individual risk quality, regulatory changes that shifted market prices across the board, or exogenous changes in reinsurer appetite are candidates — but each requires careful argumentation for validity, and any instrument must pre-date the post-GIPP structural break to avoid confounding that with the treatment effect.

---

## The two-stage Bayesian approach

A more complete solution, proposed in arXiv:2205.01875, uses a two-stage Bayesian framework. Stage 1 fits ML models to estimate both purchase probability and price from customer features, then extracts residuals — the unexplained variation in price and purchase behaviour after conditioning on observables. Stage 2 fits a Bayesian dynamic GLM on those residuals to estimate the price-sensitivity parameter β.

The paper reports a reduction in estimation error from 25% to 4%. We have not independently replicated this on UK insurance data; the figure comes from the paper's simulation study, and simulation studies are optimistic. But the direction is right: the combination of ML residualisation and Bayesian uncertainty propagation gives you a properly calibrated posterior over β, rather than a point estimate with a confidence interval that assumes away all the specification uncertainty.

This is not currently implemented in insurance-causal. It is on the roadmap.

---

## What commercial platforms do not do

Akur8 and Earnix are the dominant commercial platforms for UK personal lines price optimisation. Both use elasticity inputs derived from conversion and lapse models, and both allow pricing actuaries to adjust the elasticity assumptions. Neither, as far as we can determine from their public documentation, applies endogeneity correction or provides Bayesian uncertainty over the elasticity parameter.

This means the output of the optimisation is conditional on a point estimate that may be 3–5x wrong. The platform will find the optimal price given the assumed elasticity; it will not tell you how sensitive the optimal price is to the elasticity assumption, or flag that the estimate is likely biased.

You can address this manually: run the optimisation at multiple elasticity values (say, −1.5, −3.0, and −5.0 for PCW motor renewal) and compare the resulting strategies. If the optimal price changes materially across the range, you have a decision that is highly sensitive to an uncertain input. That uncertainty should inform how aggressively you implement the model output.

---

## Practical implications

If true elasticity for UK motor PCW renewal is closer to −4 than −1.5, several things follow:

**Rate changes have larger retention effects than expected.** A 5% rate increase that your model says costs you 7.5 percentage points of retention might actually cost 20 percentage points. The profit-maximising price is lower than your model thinks.

**Segment-level elasticity matters more than average elasticity.** High-elasticity segments (young drivers, annual switchers, PCW-acquired customers) may be deeply unprofitable to retain at anything above cost-plus pricing. A model that uses average elasticity will systematically over-retain these customers at a loss.

**The value of causal identification is high.** If you could get a 3–5x more accurate elasticity estimate, the return on that investment through better pricing decisions would be substantial. Building a proper instrumental variables or two-stage Bayesian pipeline is not an academic exercise; it is a pricing accuracy problem.

**Pre-2022 models need replacing.** Not adjusting — replacing. The structural break from GIPP is large enough that recalibration is not sufficient. You need to re-estimate from post-2022 data, with a method that accounts for endogeneity.

---

## What we do not know

We are extrapolating Woodard & Yi to a different market, a different product, and a different regulatory environment. The 3–5x bias figure is an empirical finding from US crop insurance; it is not a proven fact about UK motor. It is informed speculation backed by a theoretical mechanism that applies to UK motor — but the size of the bias here could be larger or smaller.

We do not have access to a large UK motor dataset with a valid instrument. We cannot independently estimate the true causal elasticity and compare it to the naive OLS estimate. If you work at a UK insurer and have done this, we would like to know what you found.

The post-GIPP structural break point is directionally sound — the mechanism is unambiguous — but the magnitude is uncertain. How much of pre-GIPP lapse stickiness was genuine price insensitivity versus inertia is an empirical question we cannot answer from public data.

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Woodard, J.D. & Yi, F. (2020). Endogenous price regulation and price elasticity estimation: Evidence from crop insurance markets. *American Journal of Agricultural Economics*, 102(2), 556–578.
- arXiv:2205.01875 — Two-stage Bayesian dynamic GLM for price sensitivity estimation with ML residualisation.
- FCA PS21/5 (2021). General Insurance Pricing Practices. Financial Conduct Authority.
- Burning Cost [insurance-causal](https://github.com/burning-cost/insurance-causal): Double Machine Learning for causal price elasticity in insurance.
