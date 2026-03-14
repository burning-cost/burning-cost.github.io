---
layout: post
title: "Causal Elasticity Estimation for Renewal Pricing"
date: 2026-03-11
categories: [pricing, causal-inference]
tags: [renewal, elasticity, DML, causal-inference, FCA, ENBP, GIPP, python, motor]
description: "Standard renewal demand models overestimate price sensitivity for bad risks and underestimate it for good ones — because risk drives both premium and lapse. Double Machine Learning removes this structural confound to produce causal elasticity estimates that hold up under FCA Consumer Duty scrutiny."
---

Most UK personal lines insurers have a renewal demand model. Many of them are wrong in the same way, and the wrongness compounds directly into pricing decisions.

The problem is not technical incompetence. It is a structural confound that standard logistic regression cannot fix, no matter how many features you add or how long you spend tuning your hyperparameters.

## The Confound

Here is how renewal pricing data is generated.

A customer comes up for renewal. Your rating engine re-prices them based on current risk factors: claims history, NCB level, age, vehicle, postcode, market rates. If their risk has gone up, the premium goes up. The customer either renews or lapses.

You collect this data and build a demand model: logistic regression of `renewed ~ price_change + risk_features`. You find that customers who received a 10% price increase lapsed at higher rates than customers who received a 5% increase. You call this your price elasticity estimate and use it to optimise renewal offers.

The problem: customers who received large price increases are not a random sample. They received large increases *because* their risk went up. Higher-risk customers also lapse at higher rates — independent of price — because they are more likely to shop around, because they have had recent claims that prompt them to review their insurance, because they are later in the customer lifecycle when switching costs have eroded.

Your logistic regression cannot separate these two effects. When you regress renewal on price change, you are measuring a mixture of genuine price sensitivity and the correlation between risk deterioration and shopping behaviour. The result overestimates how much price increases drive lapse and underestimates how price-sensitive your good-risk, stable customers actually are.

The practical consequence: your demand model tells you that high-risk customers are very price-sensitive, so you discount them. Your good-risk customers look price-inelastic, so you don't. You have it backwards. You are subsidising your worst customers at the expense of your best.

## Why Controlling for Risk Features Does Not Help

The natural response is: "We control for risk factors in the regression." This does not solve the problem.

Controlling for observed risk features removes *measured* confounding. But price changes are also driven by unmeasured or imperfectly measured risk factors: residual claims development, risk selection that the model does not fully capture, underwriting judgment applied during re-rating. The residual correlation between price change and unobserved risk remains.

More fundamentally: even if your rating model is perfectly predictive, if you use the same risk features to predict *both* price change (treatment) and renewal (outcome), you are in a multicollinearity situation where the regression coefficients for price change become very sensitive to specification. The standard errors balloon. The confidence intervals are uninformative. The point estimates are unstable across small dataset changes.

This is not a modelling failure. It is a consequence of the data-generating process. Price in insurance is nearly a deterministic function of risk factors. There is very little natural variation in price *after* conditioning on risk. Standard methods — including OLS, logistic regression, and gradient-boosted demand models — need variation in the treatment (price) to estimate its effect. When that variation is absent, they pick up noise.

## Double Machine Learning

DML isolates the causal effect by partialling out confounders from both the treatment and the outcome using flexible ML models. For the full mathematical procedure and Neyman orthogonality guarantee, see [Causal Inference for Insurance Pricing](/2026/02/25/causal-inference-for-insurance-pricing/).

The key point for renewal pricing: what remains in `D_tilde = D - E[D|X]` after partialling out risk factors is the part of the price change not driven by risk: manual overrides, bulk re-rating decisions, competitive environment effects. This is exogenous variation, and regressing renewal residuals on price residuals gives an approximately unbiased causal estimate.

For heterogeneous elasticity (the question of which customers are more or less price-sensitive), CausalForestDML extends this by fitting a causal forest on the residuals. Rather than a single average elasticity, you get a separate estimate for each customer, with confidence intervals.

## Using the Library

The `insurance-causal` library wraps EconML's DML implementations with insurance-specific defaults: CatBoost nuisance models (which handle categorical features like region, vehicle group, and occupation without encoding), binary outcome correction, and diagnostics tuned for the UK personal lines data structure.

Fitting the average treatment effect:

```python
from insurance_causal.elasticity import RenewalElasticityEstimator

estimator = RenewalElasticityEstimator(cate_model='causal_forest')
estimator.fit(df, outcome='renewed', treatment='log_price_change',
              confounders=['age', 'ncd', 'channel', 'region', 'vehicle_group'])

# Average elasticity
estimator.ate()
# {'estimate': -0.42, 'ci_lower': -0.58, 'ci_upper': -0.26, 'pvalue': 0.001}
```

An estimate of -0.42 means a 1% price increase is associated with a 0.42 percentage point reduction in renewal probability, after removing the confounding from risk factors. The 95% CI of [-0.58, -0.26] excludes zero — the effect is statistically identifiable in this data.

For the heterogeneous elasticity surface:

```python
# Per-customer elasticity
cate_df = estimator.cate(df)

# Group effects by segment
estimator.gate(df, by='channel')
```

The `gate()` method gives Group Average Treatment Effects: the average elasticity within each channel, NCD band, age group, or any other segmentation variable. In practice, PCW customers are reliably the most price-sensitive. Direct customers with five or more years' NCD and a direct debit mandate are typically among the least price-sensitive.

## Heterogeneity Matters for Optimisation

A single average elasticity is not actionable. Renewal pricing operates at the individual policy level. The question is not 'what is our average elasticity?' but 'for this specific customer, if we offer them ENBP minus 2%, how does that change their renewal probability?'

The per-customer elasticity from CausalForestDML feeds directly into the renewal optimisation problem:

```python
from insurance_causal.elasticity import RenewalPricingOptimiser

optimiser = RenewalPricingOptimiser(
    elasticity_model=estimator,
    technical_premium_col='tech_prem',
    enbp_col='enbp',
)
result = optimiser.optimise(df, objective='profit')
audit = optimiser.enbp_audit(df)
```

The optimiser solves, for each policy:

```
max_{p_i}  (p_i − c_i) × P(renew | p_i, X_i)

subject to:
  p_i ≤ ENBP_i
  p_i ≥ c_i × floor_loading
```

The ENBP constraint is one-sided: you cannot price above the equivalent new business price for the same risk through the same channel. You can price below. The optimiser finds the profit-maximising point on that feasible set, using the individual elasticity estimate to compute the renewal probability at each candidate price.

The `enbp_audit()` call produces a compliance report for each policy showing the ENBP used, the offer price relative to ENBP, and whether the constraint was binding. ICOBS 6B.2 requires firms to demonstrate their ENBP methodology. The audit trail is the documentation.

## The Hard Problem: Near-Deterministic Treatment

We should be direct about the biggest limitation of this approach, because most DML tutorials gloss over it.

DML works by exploiting variation in price that is *not explained by risk factors*. In insurance, how much such variation actually exists?

Run the treatment model — the one that predicts log price change from risk features — and look at the R². In a typical UK motor renewal dataset, this is 0.85 to 0.95. Your rating engine explains most of the price change. What is left (the residual `D_tilde`) is a small signal: manual underwriting overrides, competitive environment shifts applied at the book level, timing effects from when in the renewal cycle the customer was priced.

When residual variance is small, DML estimates become noisy. The standard errors on the elasticity estimate widen. The confidence intervals may span the range from "inelastic" to "highly elastic". You have identified the treatment effect, but not precisely.

The library flags this explicitly. If the treatment model R² exceeds 0.90, you will see a warning. If it exceeds 0.95, the estimates should be treated with serious caution regardless of what the confidence intervals say. The asymptotic theory requires meaningful residual variation, and near-zero residual variance violates the effective positivity assumption.

This is not a library design flaw. It is an honest description of what observational data can and cannot tell you.

## What Actually Works for Identification

If observational data is limited, there are genuine quasi-experimental designs available to UK insurers.

**A/B testing.** The gold standard. Randomly assign a subset of renewals to price variants that deviate from the rating model output. Even a 2–3% random price perturbation on 10% of renewals, run for two renewal cycles, gives clean exogenous variation. The library handles A/B data cleanly — the treatment model R² will be low (the perturbation is random by construction), and the elasticity estimate will be precise.

**Bulk re-rating quasi-experiments.** When a book is re-rated (new base rates, new rating factors), some customers see price changes that are mechanically determined by the rating change rather than by their individual risk change. If you can isolate customers who experienced the same rating factor change for non-risk reasons, you have something close to an instrumental variable. The library's diagnostic tools help identify these sub-populations.

**ENBP kink regression.** PS21/5 created a natural experiment. For customers who were previously priced above ENBP (price-walking was common before January 2022), the rule forced a price reduction to exactly ENBP. This generates a discontinuity in the price-change distribution at the ENBP level. Regression kink or discontinuity designs can exploit this to estimate elasticity near the ENBP constraint, precisely the elasticity that matters most for post-GIPP optimisation.

**Panel data within-customer variation.** For customers with multiple renewal cycles, the within-customer price change (net of risk change) provides identification. A customer whose risk profile is stable but who received different price changes in consecutive years has given you a near-controlled comparison. This requires multi-year data and careful risk-change adjustment, but it is genuine exogenous variation.

## Limitations

We want to be clear about what this library does and does not solve.

**It does not create data that does not exist.** If your observational data has near-zero exogenous price variation (treatment R² > 0.90) and you have no quasi-experimental supplement, the estimates will be imprecise. A wide confidence interval is an honest answer.

**It does not handle pre-2022 data correctly by default.** Pre-GIPP data has a fundamentally different demand structure. Price-walking inflated premiums for inertial customers, creating artificially low measured elasticity for long-tenure customers. The library flags data containing pre-2022 renewals and recommends using post-2022 data only.

**Channel mixing produces a fiction.** PCW and direct customers have different elasticity distributions by a large margin. The average of the two populations does not describe either correctly. Always fit channel-stratified models.

**ENBP is taken as a given.** The library takes the ENBP column as input. Computing ENBP correctly (applying the right new business model, the right channel, the right incentive adjustments) is the firm's responsibility and is not inside the library's scope. Wrong ENBP inputs produce wrong optimisation outputs.

**The positivity assumption can fail for extreme risks.** For customers with very constrained pricing (technical premium near or above ENBP, underwriting restrictions), the assumption that they could in principle have received any price in a reasonable range breaks down. The library's diagnostics flag policies where the elasticity estimate should be discarded rather than used.

## The Regulatory Incentive

FCA EP25/2, published in July 2025, confirmed that the GIPP remedies are working in motor — the evaluation estimated £1.6 billion in consumer savings since January 2022. The FCA is watching whether firms are using segment-level elasticity appropriately. A firm that cannot demonstrate how it computed renewal elasticity estimates, or that cannot show the estimates are defensible at a segment level, is exposed under Consumer Duty.

A causal elasticity model with an explicit audit trail (treatment variation diagnostics, CI-bounded estimates, ENBP compliance reports per policy) is not just better pricing. It is a defensible methodology.

## What We Think

Standard demand models produce biased elasticity estimates. We think this is widely understood at a theoretical level in UK actuarial functions and almost never fixed at a practical level.

That is what `insurance-causal` is for: DML estimation with the insurance-specific corrections applied by default, connected directly to an ENBP-constrained optimiser, with diagnostics that tell you honestly when the data cannot support reliable estimates.

If your renewal demand model gives you a single price sensitivity parameter and a confidence interval that your team treats as decoration, the model is flying blind. Usually better data and a better model are both needed, but the better model at least tells you which.

---

*The `insurance-causal` library is open source under MIT licence. The ENBP optimiser, heterogeneous elasticity surface, and compliance audit tools are covered in the [library documentation](#).*

---

**Related articles from Burning Cost:**
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/)
- [Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/)
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/)
