---
layout: post
title: "Credibility-Weighted Broker and Scheme Effects with REML"
date: 2026-03-13
categories: [pricing, libraries]
tags: [multilevel-models, random-effects, credibility, catboost, reml, broker, scheme, high-cardinality, python]
description: "Two-stage CatBoost plus REML random effects for UK insurance broker adjustments. insurance-multilevel - Buhlmann-Straub credibility weighting, not guesswork."
---

There is a specific conversation that happens in most UK personal lines pricing teams, usually during model validation, and usually when someone pulls a double-lift chart broken down by broker.

The chart shows that the model (the GBM you spent three months building and validating) is systematically mispricing risks from five or six brokers. The average log loss ratio residual for Broker A is +18%. For Broker B it is −12%. The model is otherwise well-calibrated. It predicts individual risk factors well. It just cannot handle the fact that different brokers write structurally different business that your features do not fully capture.

The response to this chart is almost always the same: "We'll apply a manual loading." Someone opens a spreadsheet. The loading for Broker A goes to 1.18. The loading for Broker B goes to 0.88. The conversation ends. Everyone feels the problem has been solved.

It has not been solved. It has been named.

## What the spreadsheet does not know

The manual loading is not wrong in direction. Broker A's business genuinely runs hot. But it is wrong in magnitude, and it is wrong in the way that matters most: it does not know how much to trust its own estimate.

Broker A wrote 23 policies last year. Its +18% average log residual is based on 23 data points. How much of that +18% is genuine Broker A effect (their risk appetite, their underwriting judgment, the class of business they bring) and how much is random noise on a thin sample? The spreadsheet cannot tell you. The manual loading implicitly assumes it is all signal. It is not.

The correct answer is a credibility-weighted estimate. If you have 23 policies from Broker A, you should trust Broker A's own experience partially. Exactly how partially depends on the ratio of between-broker variance to within-broker noise, which you can estimate from the data. The Bühlmann (1967) credibility framework has been the actuarial answer to this problem for nearly sixty years. The Bühlmann-Straub (1970) extension handles heterogeneous exposure weights. Neither requires proprietary software.

What previously required proprietary software is integrating credibility estimation with a modern gradient-boosted pricing model. That is the gap [`insurance-multilevel`](https://github.com/burning-cost/insurance-multilevel) fills.

## The two-stage architecture

The library implements a two-stage model. Stage 1 is CatBoost on individual risk factors: age, vehicle, claims history, postcode sector, whatever you have. The critical design choice is that group identifiers (broker ID, scheme ID, territory code) are **excluded** from Stage 1. This is not an oversight.

If you pass broker ID to CatBoost, the GBM partially absorbs the broker signal. For brokers with many policies it will learn something real. For brokers with few policies it will overfit. In both cases, Stage 2, which estimates how much of the residual variance is attributable to broker membership, will see a distorted picture. The identifiable approach is to keep the stages clean: CatBoost sees individual features, REML sees the residuals.

Stage 2 fits a one-way random intercepts model on the log-ratio residuals from Stage 1:

```
r_i = log(y_i / f_hat_i)   (how far off was CatBoost for policy i?)
r_i = mu + b_g(i) + epsilon_i
b_g ~ N(0, tau2)            (broker effect, group-level)
epsilon_i ~ N(0, sigma2)    (within-broker noise)
```

The REML estimator (Restricted Maximum Likelihood, from Patterson and Thompson (1971)) estimates `tau2` (between-broker variance) and `sigma2` (within-broker noise) simultaneously. REML rather than ML because ML systematically underestimates variance components, particularly `tau2`, when the number of groups is small. The difference matters: underestimating `tau2` leads to over-shrinkage, which means your large brokers get insufficient credit for their distinct risk profiles.

From the variance components, the BLUP (Best Linear Unbiased Predictor) for each broker is:

```
b_hat_g = Z_g × (r_bar_g − mu_hat)
Z_g = tau2 / (tau2 + sigma2/n_g)   (Bühlmann credibility weight)
```

When `n_g` is large, `Z_g` approaches 1 and the BLUP converges to the broker's own average residual: full credibility. When `n_g` is small, `Z_g` is close to 0 and the BLUP is shrunk toward the grand mean, which is appropriate scepticism about a thin sample. The final prediction multiplies the CatBoost score by `exp(b_hat_g)`.

This is the multiplicative structure used throughout UK personal lines pricing. The broker factor is a number you multiply by the technical price, exactly like any other rating factor, but derived rather than guessed and with credibility weighting that reflects how much data you have.

## Using the library

```python
from insurance_multilevel import MultilevelPricingModel

model = MultilevelPricingModel(
    catboost_params={"iterations": 500, "loss_function": "Poisson"},
    random_effects=["broker_id", "scheme_id"],
    min_group_size=5,
)

model.fit(X_train, y_train, weights=exposure, group_cols=["broker_id", "scheme_id"])
premiums = model.predict(X_test, group_cols=["broker_id", "scheme_id"])
```

The `group_cols` list accepts multiple levels. The library fits a separate `RandomEffectsEstimator` for each. Broker effects and scheme effects are estimated independently, with their own REML variance components, and the adjustments compose on the log scale. This handles the common structure where policies sit within schemes, schemes sit within brokers, and you want a loading at each level.

The `credibility_summary()` method is where you should spend most of your diagnostic time:

```python
summary = model.credibility_summary()
print(summary)
```

```
shape: (42, 11)
┌────────────┬─────────────────┬────────┬────────────┬─────────┬────────────┬──────────────────┬────────┬────────┬────────┬──────────┐
│ level      ┆ group           ┆ n_obs  ┆ group_mean ┆ blup    ┆ multiplier ┆ credibility_weight ┆ tau2   ┆ sigma2 ┆ k      ┆ eligible │
╞════════════╪═════════════════╪════════╪════════════╪═════════╪════════════╪══════════════════╪════════╪════════╪════════╪══════════╡
│ broker_id  ┆ Broker_A        ┆ 847.0  ┆ 0.183      ┆ 0.171   ┆ 1.186      ┆ 0.934            ┆ 0.032  ┆ 0.214  ┆ 6.69   ┆ true     │
│ broker_id  ┆ Broker_B        ┆ 312.0  ┆ -0.119     ┆ -0.101  ┆ 0.904      ┆ 0.822            ┆ 0.032  ┆ 0.214  ┆ 6.69   ┆ true     │
│ broker_id  ┆ Broker_C        ┆ 23.0   ┆ 0.187      ┆ 0.082   ┆ 1.085      ┆ 0.255            ┆ 0.032  ┆ 0.214  ┆ 6.69   ┆ true     │
│ broker_id  ┆ Broker_D        ┆ 8.0    ┆ -0.215     ┆ -0.030  ┆ 0.970      ┆ 0.100            ┆ 0.032  ┆ 0.214  ┆ 6.69   ┆ true     │
└────────────┴─────────────────┴────────┴────────────┴─────────┴────────────┴──────────────────┴────────┴────────┴────────┴──────────┘
```

Read that table carefully. Broker A has 847 policy-years of exposure and a credibility weight of 0.93. Its +18.6% multiplier is almost entirely trusted. Broker C has 23 policy-years and a raw average of +18.7%, which looks similar to Broker A. But its credibility weight is 0.26. The library applies a multiplier of 1.085, not 1.187. The difference is 10 points of loading on every policy Broker C submits, based on how much you should rationally trust 23 policies of experience.

Broker D's situation is more interesting. Its raw average is −21.5%, apparently very good business. But with only 8 policy-years, the credibility weight is 0.10 and the BLUP is −3.0%. If Broker D's next year brings a large loss, that −21.5% evaporates. The shrinkage is protecting you from that.

## The ICC diagnostic

Before committing to the two-stage structure, check whether group effects actually exist:

```python
from insurance_multilevel import variance_decomposition

vd = variance_decomposition(model.variance_components)
print(vd)
```

```
┌───────────┬───────┬────────┬───────┬────────────┐
│ level     ┆ tau2  ┆ sigma2 ┆ icc   ┆ buhlmann_k │
╞═══════════╪═══════╪════════╪═══════╪════════════╡
│ broker_id ┆ 0.032 ┆ 0.214  ┆ 0.130 ┆ 6.69       │
│ scheme_id ┆ 0.051 ┆ 0.198  ┆ 0.205 ┆ 3.88       │
└───────────┴───────┴────────┴───────┴────────────┘
```

The ICC (Intraclass Correlation Coefficient) is `tau2 / (tau2 + sigma2)`, the proportion of total residual variance explained by group membership, after Stage 1 CatBoost has done its job. An ICC of 0.13 for broker means 13% of the premium variation your CatBoost cannot explain is attributable to which broker placed the risk. That is not noise. That is pricing signal you were previously ignoring.

The scheme-level ICC of 0.21 is higher. Schemes are selected by definition (a fleet scheme, a professional indemnity scheme for solicitors, a property owner scheme) and carry more embedded selection effect than broker identity alone. This is the typical pattern in UK motor and home insurance.

The Bühlmann k is `sigma2 / tau2`. It answers the question: how many policy-years of group experience are equivalent to the prior? At k = 6.69 for broker, a group needs roughly 6.69 policy-years of exposure to reach 50% credibility (`Z = n / (n + k)`). At 200 policy-years, `Z = 0.97`. At 20 policy-years, `Z = 0.75`. These thresholds are estimated from your data, not assumed.

## What about the REML implementation?

We implemented REML from scratch rather than wrapping `statsmodels` or `lme4` (via `rpy2`). The reason is control: insurance pricing models need to work in production environments where R is not available, where `rpy2` adds dependency hell, and where the REML computation needs to compose cleanly with CatBoost's output.

The REML log-likelihood for the one-way random effects model decomposes as:

```
-2 l_REML = Σ_g (n_g - 1) log(sigma2) + SS_within_g/sigma2   (within-group)
           + Σ_g log(v_g)                                       (between-group means)
           + Σ_g (r_bar_g - mu_hat)^2 / v_g                    (group deviations)
           + log(Σ_g 1/v_g)                                     (REML correction)
```

where `v_g = tau2 + sigma2/n_g`. We optimise this via L-BFGS-B with Henderson method-of-moments starting values. During development, we found that the within-group log-determinant term — the `(n_g - 1) log(sigma2)` piece — is easy to omit and catastrophic when omitted: without it, the optimiser has no penalty for inflating `sigma2` to absorb all variance, and `tau2` collapses to zero. The Bühlmann-Straub framework tells you something. The something collapses to nothing. We wrote this up in detail because it is the kind of bug that produces plausible-looking output — the optimiser converges, the estimates look reasonable, everything is just wrong.

## New brokers at prediction time

A new broker submits three quotes. They have no history in your training data. What factor do they get?

```python
# allow_new_groups=True (default): new groups get exp(0) = 1.0
premiums = model.predict(X_new, allow_new_groups=True)
```

New groups receive a BLUP of zero, the grand mean of the random effects distribution. In multiplicative terms, factor 1.0: no adjustment up or down. This is the correct prior given zero data. It is not generous (you are not giving a new broker the benefit of low-risk assumptions) and not punitive (you are not loading them for being unknown). You are pricing their submitted risks on their individual merits, with no group adjustment until you have evidence to form one.

The `min_group_size=5` default means groups with fewer than five policy-years of exposure are also excluded from REML estimation and assigned `Z=0`. You can adjust this:

```python
model = MultilevelPricingModel(
    random_effects=["broker_id"],
    min_group_size=10,  # stricter: need 10 policy-years to enter REML
)
```

Increasing `min_group_size` reduces the number of eligible groups in REML, which affects the variance component estimates. In markets with many thin groups, a higher threshold often produces more stable `tau2` estimates because you are fitting on the groups where you actually have useful signal.

## The data-need diagnostic

The `groups_needing_data()` function answers a question that is useful for capacity planning:

```python
from insurance_multilevel import groups_needing_data

needs = groups_needing_data(summary, target_z=0.8)
print(needs.filter(pl.col("n_additional") > 0).sort("n_additional", descending=True))
```

```
┌───────────┬──────────┬────────┬──────────────────┬──────────┬──────────────┐
│ level     ┆ group    ┆ n_obs  ┆ credibility_weight ┆ n_target ┆ n_additional │
╞═══════════╪══════════╪════════╪══════════════════╪══════════╪══════════════╡
│ broker_id ┆ Broker_C ┆ 23.0   ┆ 0.255            ┆ 26.76    ┆ 3.76         │
│ broker_id ┆ Broker_D ┆ 8.0    ┆ 0.100            ┆ 26.76    ┆ 18.76        │
└───────────┴──────────┴────────┴──────────────────┴──────────┴──────────────┘
```

At these k values, reaching 80% credibility requires approximately 27 policy-years. Broker C needs about 4 more. Broker D needs about 19. This is useful information for a capacity meeting: you know exactly how much volume from each broker changes the confidence you can have in their rating factor. It is not a vague "we need more data." It is a number.

## Where this fits in the pricing stack

The library's position in the stack is deliberate. It sits after technical pricing (GLM or GBM) and before commercial adjustment. The CatBoost stage is your risk model. The REML stage is your group correction. The final premium is their product.

This replaces two things: the manual loading spreadsheet (replaced by BLUP estimates with credibility weights) and the pretence that group effects do not exist (replaced by a formal test via the ICC). It does not replace underwriting judgment about specific brokers; a broker with a known claims culture problem should be managed on that basis, not just through their data. But it does provide a data-driven baseline from which judgment can depart explicitly.

## Limitations

The model assumes the one-way random intercepts structure: each group has a single additive effect on the log-residual. It does not handle crossed effects (broker by territory) in Version 1: the joint `(broker, territory)` combination is not available as a two-dimensional random effect. If your broker effects vary systematically by territory, the library will absorb this variation into the marginal effects, which will be correct on average but wrong at the intersection.

The Gaussian assumption on log-ratio residuals is reasonable for most insurance lines but can fail with heavy-tailed losses. If your Stage 1 CatBoost residuals are highly skewed or leptokurtic (check with `residual_normality_check()` from the diagnostics module), the REML variance component estimates are still consistent, but inference (credibility interval width) will be approximate.

Stage 2 is fitted on training residuals. If the distribution of groups in the training data differs substantially from production (you launched three new broker relationships last year that were not in the training period), the variance component estimates may not generalise well. The new-group fallback handles the prediction problem, but the `tau2` estimate may be understated if production has more group heterogeneity than training did.


Ad-hoc broker loadings are not irrational. They are a rational response to a real signal in the data. The problem is that they are applied at full credibility regardless of how much data the signal is based on, which means they overfit small groups and potentially underfit large ones where the full weight of experience should be trusted.

The Bühlmann-Straub framework has been in actuarial syllabuses since the 1970s. Most UK pricing actuaries have studied it. Very few have it running in their pricing models, because the implementation gap between "credibility theory" and "credibility-weighted broker factors in a GBM pricing model" has historically required either proprietary software or a significant amount of custom code.

`insurance-multilevel` closes that gap. The REML implementation handles unbalanced groups correctly, composes with CatBoost cleanly, and produces a credibility summary that gives you numbers you can put in front of an underwriter and explain. The broker factor for Broker A is 1.186 because they have 847 policy-years of experience at +18.3% residual and a credibility weight of 0.934. That is a defensible number. The 1.18 in the spreadsheet is not.

---

`insurance-multilevel` is open source under the MIT licence at [github.com/burning-cost/insurance-multilevel](https://github.com/burning-cost/insurance-multilevel). Install with `uv add insurance-multilevel`. 61 tests passing. Requires Python 3.10+, CatBoost, Polars, and SciPy.

---

## When to use which credibility approach

This library is the right tool when you need to adjust for group-level residuals from a GBM pricing model — broker, scheme, or territory effects that the risk model cannot capture. The two-stage CatBoost + REML architecture keeps the stages clean and produces credibility-weighted adjustments that compose multiplicatively with the technical price.

For segment-level credibility pricing without a GBM stage — where you want to blend thin segment loss experience with a portfolio prior using explicit EPV/VHM decomposition and K factors — see [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/). For full Bayesian posteriors with multiple crossed rating dimensions and Poisson or Gamma likelihoods, see [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/).

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/)
- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/)
