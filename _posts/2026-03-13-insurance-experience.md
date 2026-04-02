---
layout: post
title: "Individual Experience Rating Beyond NCD: From Bühlmann-Straub to Neural Credibility"
date: 2026-03-13
categories: [libraries, pricing, credibility]
tags: [experience-rating, credibility, Bühlmann-Straub, Bayesian, posterior, Poisson-Gamma, GBM, attention, NCD, FCA, PS21-11, insurance-experience, python, pytorch]
description: "Four-tier experience rating in Python: Buhlmann-Straub, Poisson-Gamma state-space, GBM surrogate, attention credibility. Policy-level multiplicative factors."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-experience` has been consolidated into [`insurance-credibility`](/insurance-credibility/). Install with `uv add insurance-credibility` — individual experience rating models are available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-credibility)
</div>


NCD has a clean actuarial logic behind it: your claims history tells us something about your risk level that the rating factors don't fully capture, and we want to adjust for it. The implementation, though, is not Bayesian posterior inference. It is a contractual mechanism with a transition matrix. The factor you get depends on how many claims you made last year and which NCD level you started at. The transition rules are chosen to be commercially sensible and regulatorily defensible, not to minimise posterior expected loss.

This distinction matters for two reasons. First, FCA PS21/11 tightened the rules on how NCD discounts interact with the fair price obligation - the contractual structure of NCD can produce outcomes that are hard to defend under a posterior-risk framing when a long-term NCD holder at 65% discount is paying materially less than the posterior risk would suggest. Second, the NCD mechanism is only one specific instantiation of what should be a general facility: using the claims history of an individual policyholder to update the prior estimate of their risk level. There is no reason this has to take the form of a discrete transition matrix.

[`insurance-experience`](https://github.com/burning-cost/insurance-credibility) is the actuarially correct alternative. It implements individual Bayesian a posteriori experience rating across four model tiers, producing a multiplicative credibility factor with the balance property that slots directly into Emblem or Radar as a rating factor. 125 tests, MIT-licensed, on PyPI.

```bash
uv add insurance-credibility
```

---

## What the balance property means here

A multiplicative credibility factor has the balance property if, when applied across all policyholders, the portfolio aggregate expected loss cost is unchanged. This is a non-negotiable requirement for any experience rating mechanism used in a pricing context: you cannot systematically shift money from one group to another without the pricing structure adjusting elsewhere. The balance property ensures the experience factor is a redistribution within the portfolio, not a systematic uplift or reduction.

All four model tiers in `insurance-experience` satisfy the balance property by construction. The factor for a given policyholder is their posterior frequency estimate divided by the prior estimate, so the exposure-weighted product across all policyholders returns to 1.0.

---

## The four tiers

### Tier 1: StaticCredibilityModel (Bühlmann-Straub)

The standard actuarial reference point. Bühlmann-Straub credibility is the minimum variance linear estimator of the individual risk premium given the group structure. The credibility weight Z for a policyholder with n exposure periods is:

    Z = n / (n + k)    where k = σ² / τ²

σ² is the within-group variance (process variance) and τ² is the between-group variance (structural parameter). k is the credibility parameter: the more heterogeneous the portfolio, the smaller k and the more quickly credibility builds.

For a new policyholder with no history, Z = 0 and the estimate is the portfolio mean. For a policyholder with 5 years of data at low k, Z approaches 1 and the estimate approaches their individual mean.

The important actuarial fact is that Bühlmann-Straub credibility is equivalent to a mixed linear model with a random intercept. You can fit it with REML, which gives you exact standard errors on the structural parameters rather than moment-based estimates. `StaticCredibilityModel` does this.

```python
from insurance_credibility import StaticCredibilityModel

model = StaticCredibilityModel()
model.fit(
    df,
    policy_col="policy_id",
    period_col="policy_year",
    claims_col="claim_count",
    exposure_col="earned_exposure",
)

factors = model.predict_credibility_factors(df)
# factors["credibility_factor"]: multiplicative, balance property holds
```

The output is a factor table: policy_id, credibility weight Z, prior mean, posterior estimate, multiplicative factor. The factor is what you add as a rating variable in Emblem.

### Tier 2: DynamicPoissonGammaModel

Static credibility assumes the individual risk level is fixed over time - yesterday's experience is as informative as ten years ago. That is wrong for most insurance lines. Vehicle condition changes, drivers improve or deteriorate, business risk profiles shift. The claim frequency you observed from a policyholder in year 1 should receive less weight than year 4 when estimating year 5.

`DynamicPoissonGammaModel` implements a Poisson-gamma state-space model in the tradition of Ahn, Jeong, Lu and Wüthrich's work on dynamic credibility (the specific 2023 paper attribution is unconfirmed; the architecture follows the state-space credibility literature). The individual risk level follows a latent state process that evolves over time. The model has:

- **Observation model**: observed claims ~ Poisson(λ_t × exposure_t), where λ_t is the latent rate in period t
- **State model**: λ_t evolves according to a mean-reverting process - high-claim years are partially discounted in favour of the portfolio mean

The Poisson-gamma conjugacy means the posterior distribution of λ_t given observed claims has a closed form - no MCMC required. The state dynamics introduce seniority weighting: older observations receive exponentially decaying weight. MLE on the negative binomial marginals gives you the structural parameters.

This is the first Python implementation of this model. The R literature has it; nothing on pip does.

```python
from insurance_credibility import DynamicPoissonGammaModel

model = DynamicPoissonGammaModel(decay=0.85)
model.fit(df, policy_col="policy_id", period_col="year",
          claims_col="claims", exposure_col="exposure")

factors = model.predict_credibility_factors(df)
# Seniority-weighted: year 4 gets ~0.85^0 = 1.0, year 1 gets ~0.85^3 = 0.61
```

The `decay` parameter controls how quickly old observations are discounted. 1.0 recovers static Bühlmann-Straub. Values around 0.8–0.9 are typical for annual motor data.

### Tier 3: SurrogateModel

The Poisson-gamma conjugacy that makes `DynamicPoissonGammaModel` tractable breaks down the moment you want covariates in the individual risk model - for example, if you believe the latent risk trajectory depends on vehicle age, or if the prior should be heterogeneous across risk segments.

`SurrogateModel` implements the Calcetero/Badescu/Lin (2024) importance-sampling approach. The idea is to approximate the posterior distribution for non-conjugate models using importance sampling, then correct for the approximation error with a weighted least squares step.

The surrogate fits a GBM on the covariate-adjusted claims history. IS weights are computed to produce a posterior sample consistent with the likelihood of the observed data under the true model. WLS correction adjusts for any remaining bias from the approximation.

This is the tier you want when:
- The prior distribution should depend on risk segment (young drivers have a different prior from mature drivers)
- You have covariate-dependent latent risk evolution
- The Poisson-gamma conjugacy fails (e.g., over-dispersed data with zero-inflation)

```python
from insurance_credibility import SurrogateModel

model = SurrogateModel(n_is_samples=500, wls_correction=True)
model.fit(df, policy_col="policy_id", period_col="year",
          claims_col="claims", exposure_col="exposure",
          covariate_cols=["vehicle_age", "driver_age", "region"])

factors = model.predict_credibility_factors(df)
```

### Tier 4: DeepAttentionModel

The deep attention tier implements Wüthrich (2024) individual experience rating via transformer-style self-attention. Each policyholder's claims history is treated as a sequence, and the attention mechanism learns which observations in the history are most informative for predicting the next period's rate.

This is not interpretable in the same way the other tiers are. You cannot extract a credibility weight Z and show it to a pricing committee. What you get is a predictive posterior that conditions on the full temporal pattern of the claims history - clusters of claims, long claim-free periods, step changes in frequency - in ways that a scalar seniority weighting cannot capture.

Requires `torch`. Optional dependency - if PyTorch is not installed, importing `DeepAttentionModel` raises a clear error message.

```python
from insurance_credibility import DeepAttentionModel

model = DeepAttentionModel(d_model=64, n_heads=4, n_layers=2)
model.fit(df, policy_col="policy_id", period_col="year",
          claims_col="claims", exposure_col="exposure",
          epochs=50, lr=1e-3)

factors = model.predict_credibility_factors(df)
```

The balance property is enforced by normalising the output factors after prediction, not by model architecture. The network produces posterior rate estimates; these are divided by the prior mean and then rescaled to preserve the portfolio total.

---

## Which tier to use

The tiers are not competitors - they are appropriate for different situations.

**StaticCredibilityModel** is the reference. Fit it first. Its structural parameters (k, the between/within variance ratio) tell you how much individual experience information exists in your portfolio. If k is large, individual experience is not very informative and the other tiers will show limited improvement. If k is small, there is genuine individual heterogeneity to exploit.

**DynamicPoissonGammaModel** is the default upgrade. Same assumptions as static, but with temporal discounting. Almost always outperforms static on annual renewable lines where individual risk profiles evolve. The decay parameter adds one degree of freedom; cross-validate on a held-out policy-year to select it.

**SurrogateModel** when the prior is heterogeneous or the claims process is non-Poisson. Use this when your book has meaningful sub-groups (fleet versus personal, young versus mature) with different prior distributions.

**DeepAttentionModel** when you have 5+ years of history per policyholder for a substantial fraction of the portfolio, and when the temporal pattern of claims matters beyond what seniority weighting captures. At fewer than 3 years average history, the attention mechanism does not have enough to work with.

---

## The FCA PS21/11 framing

FCA PS21/11 (August 2021) introduced the General Insurance Pricing Practices (GIPP) rules, banning the practice of price-walking where renewing customers were charged more than equivalent new customers. NCD discounts were specifically scrutinised in the context of dual pricing remediation - long-standing customers receiving NCD-inflated discounts that exceeded what posterior risk would justify. The obligation on firms to price for individual consumer risk sits under Consumer Duty (PS22/9), but PS21/11 created the fair value framework that makes pricing transparency and posterior risk grounding a regulatory expectation.

`insurance-experience` produces factors grounded in posterior risk inference, not contractual history. The distinction is auditable: you can show the regulator that a 65% NCD policyholder's factor of 0.78 reflects a posterior frequency estimate based on their actual claims history, not a table lookup from a transition matrix. This does not replace NCD in the contractual sense - you still need to honour NCD discounts. It gives you a posterior risk signal that runs alongside the contractual mechanism and can be used to identify where the two diverge materially.

---

## Usage: full workflow

```python
from insurance_credibility import (
    StaticCredibilityModel,
    DynamicPoissonGammaModel,
    ExperienceRatingEvaluator,
)

# Fit static and dynamic models
static = StaticCredibilityModel()
static.fit(df, policy_col="policy_id", period_col="year",
           claims_col="claims", exposure_col="exposure")

dynamic = DynamicPoissonGammaModel(decay=0.87)
dynamic.fit(df, policy_col="policy_id", period_col="year",
            claims_col="claims", exposure_col="exposure")

# Compare models on held-out year
evaluator = ExperienceRatingEvaluator()
results = evaluator.compare(
    models={"static": static, "dynamic": dynamic},
    test_df=df_holdout,
    policy_col="policy_id",
    period_col="year",
    claims_col="claims",
    exposure_col="exposure",
)

print(results.summary())
# Model     | Gini   | LogLik    | Balance error
# static    | 0.142  | -12841.3  | 0.0000
# dynamic   | 0.163  | -12779.1  | 0.0000

# Export factors for Emblem/Radar
factors = dynamic.predict_credibility_factors(df_score)
factors.to_csv("experience_factors.csv", index=False)
```

The evaluator produces Gini coefficients (lift of the posterior factor over no experience rating), log-likelihood, and balance error. Balance error should be machine-zero for all tiers - if it is not, there is a bug in the calibration.

---

## What is not in this library

This library does not implement NCD transition matrices or bonus-malus Markov chains. That is [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility). The two libraries are complementary: `insurance-credibility` for the contractual NCD mechanism, `insurance-experience` for the Bayesian posterior that should sit alongside or eventually replace it.

This library does not implement group-level credibility (Bühlmann at broker or scheme level). That is [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility). Again, complementary: `insurance-credibility` for group-level partial pooling, `insurance-experience` for individual-level posterior inference.

---

**[insurance-experience on GitHub](https://github.com/burning-cost/insurance-credibility)** - 125 tests, MIT-licensed, PyPI. Library #43.

---

**Related reading:**
- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/) - the contractual NCD mechanism; insurance-experience extends individual-level posterior rating beyond the NCD scale
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) - group-level credibility for broker, scheme, or territory blending; complements individual experience rating at the portfolio level
- [The Attention Head That Is Also a Credibility Weight](/2026/03/11/credibility-transformer/) - neural credibility weighting as an alternative to the classical Bühlmann posterior; relevant when the credibility structure is complex or non-linear
