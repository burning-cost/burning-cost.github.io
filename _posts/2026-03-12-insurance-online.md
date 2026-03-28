---
layout: post
title: "Bandit Algorithms for GIPP-Compliant Price Experimentation"
date: 2026-03-12
categories: [libraries, pricing, experimentation]
tags: [bandit, Thompson-Sampling, UCB1, LinUCB, GIPP, PS21-11, ICOBS, ENBP, FCA, Consumer-Duty, insurance-online, python, direct-channel, motor, home]
description: "Bandit algorithms for FCA GIPP-compliant price experimentation in UK general insurance. ENBP constraints and compliance reporting built in - insurance-online."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-online` has been consolidated into [`insurance-optimise`](https://pypi.org/project/insurance-optimise/). Install with `pip install insurance-optimise`  -  bandit pricing algorithms are available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-optimise)
</div>


The PRA's Dear Chief Actuary letter on claims inflation gets all the attention. The FCA's GIPP multi-firm review - 28 firms found with insufficient documentation of pricing controls in their 2023-24 cycle - gets treated as a compliance checkbox. What both regulators are circling around, without ever quite saying it directly, is that UK pricing teams operate a significant slice of their pricing on undocumented discretion.

Consider the discretionary loading band. Your GLM produces a technical price. Your pricing team applies a loading: maybe -5% for a segment they think is overshooting, +10% for one they think is underpriced. The band is typically documented as a governance limit (-15% to +15% is common in motor). What is not documented is which loading, within that band, actually maximises the metric you care about. Conversion rate? Lifetime value? Combined ratio? Nobody tested it. The current loading exists because someone set it two years ago and nobody changed it.

This is the problem [`insurance-online`](https://github.com/burning-cost/insurance-online) addresses. It brings multi-armed bandit algorithms to loading optimisation, with regulatory constraints built in from the start.

```bash
uv add insurance-online
```

---

## Why A/B testing fails here

The standard instinct is: run an A/B test. Allocate 50% of quotes to loading A, 50% to loading B, wait, read the results.

Three problems.

First, the annual policy lifecycle. In motor or home insurance, a customer takes a quote, binds, and then the claim signal doesn't fully emerge until well into the policy year. You can observe conversion quickly - did they bind? You cannot observe claims quickly. A proper frequency-informed test on a standard personal lines book needs at least 12 months of exposure on each arm to have a clear view of the loss ratio. That is a long time to be routing half your quotes to what might be the inferior arm.

Second, the waste. A/B testing is symmetric by design. You put equal weight on arms regardless of what you're learning. After 500 quotes, if arm B is showing 8% conversion and arm A is showing 12%, you are still routing 50% of your quotes to arm B. You've sacrificed those quotes learning something you already know.

Third, it doesn't scale. If you want to test five loadings in parallel - your current rate plus -6%, -3%, +3%, +6% - a classical multi-arm A/B test at adequate power requires a sample size that, on a typical mid-tier direct channel, takes 18 months to accumulate. By then the market has moved.

Multi-armed bandits solve all three. They adapt allocation toward better-performing arms as evidence accumulates. They minimise regret - the total loss from routing quotes to suboptimal arms. They scale to many arms without proportionally increasing the required sample size.

---

## The regulatory question

The FCA's GIPP rules (PS21/11, January 2022) created a legitimate concern here. The headline rule - ICOBS 6B.2.51R - says that renewal prices must not exceed the Equivalent New Business Price (ENBP). The mechanism by which a pricing team might use an adaptive allocation algorithm to route customers to higher prices is precisely the kind of thing GIPP was designed to prevent.

So is loading optimisation via bandit algorithms "dynamic pricing"? We think the answer is clearly no, and the FCA's own GIPP Q&A provides the grounding. Q1.12 and Q1.13 of the GIPP Q&A explicitly permit different margins for different new business cohorts. Testing which loading maximises new business conversion - or new business loss ratio - is margin optimisation on new business. It is not renewal repricing. There is no renewal customer in the experiment.

The condition is that renewal customers cannot be systematically routed to higher-priced arms. You must track arm selection by tenure and verify allocation independence independently - a simple chi-square test on your experiment log will do.

The condition also is that controls are documented. The FCA's 2023-24 multi-firm review found 28 firms with insufficient documentation. Experiments should log every arm selection and outcome with a timestamp. The `bandit.summary()` output gives you arm-level conversion rates and allocation counts - document this at each review period. That is the evidence base the FCA is looking for.

---

## How the library works

The core class is `ThompsonBandit`. It takes price levels and a reward model.

```python
from insurance_online import ThompsonBandit, PricingSimulation

# A 3-arm loading experiment on direct motor new business
# Price levels are multipliers applied to the base premium
bandit = ThompsonBandit(
    price_levels=[1.00, 1.03, 0.97],   # base, +3%, -3%
    reward="conversion",
)
```

At quote time, you call `select_price()`, passing the base premium. The bandit returns the selected price level:

```python
# Quote arrives  -  GLM produces technical price
base_premium = 398.0
price_level = bandit.select_price(base_premium)
# Ensure the offered price does not exceed ENBP (FCA PS21/5 obligation)
enbp = 412.0
offer_price = min(price_level * base_premium, enbp)
```

When the outcome is known (typically at bind, or at renewal for a frequency experiment):

```python
bandit.update(price_level, converted=True)   # 1.0 = bind
bandit.update(price_level, converted=False)  # 0.0 = no bind
```

To summarise experiment progress:

```python
print(bandit.summary())
best = bandit.best_price()
```

The HTML report is deliberately conservative in styling. It will print cleanly and survive being emailed to your compliance team as an attachment.

---

## The policies

Three policies cover the practical range of needs.

**Thompson Sampling** is our recommendation for most experiments. It uses conjugate priors: Beta-Bernoulli for conversion rate, Gamma-Poisson for claim frequency. At each selection, it samples from each arm's current posterior distribution and picks the arm with the highest sampled value. Arms with wide posteriors (few observations) get explored; arms with narrow posteriors that are clearly worse get de-prioritised. It is Bayesian in the right sense: uncertainty drives exploration, and exploration reduces as evidence accumulates.

The Beta-Bernoulli update is the simplest. A bind is `alpha += 1`; a no-bind is `beta += 1`. After 200 quotes on an arm that's been converting at 15%, the posterior is approximately Beta(31, 171). Sampling from that distribution rarely produces values above 0.2, so the algorithm won't waste quotes on arms that have clearly lost.

The Gamma-Poisson model is for frequency experiments. You're not trying to maximise conversion. You're trying to identify the loading band that attracts the best-risk business. After observing `n` claims over `e` policy years on an arm, the posterior rate becomes Gamma(alpha_0 + n, beta_0 + e). For frequency, Thompson Sampling picks the arm with the *minimum* sampled rate - the arm that looks like it attracts lower-frequency risk.

**UCB1** (Auer, Cesa-Bianchi & Fischer 2002) is the deterministic alternative. It scores each arm as `mu_hat(a) + alpha * sqrt(2 * ln(t) / n(a))` - empirical mean plus an exploration bonus that shrinks as the arm accumulates observations. It is easier to explain to a compliance team because it is deterministic: given the same data, it always makes the same selection. The default `alpha=1.0` is the theoretically motivated choice.

**Epsilon-greedy** exists as a sanity-check baseline. Before committing to Thompson or UCB, run epsilon-greedy for a few weeks to verify that your reward signal is sensible. If epsilon-greedy cannot identify a best arm after 500 observations, the reward definition needs revisiting. Not the algorithm.

---

## The GIPP constraint in detail

ENBP enforcement is not built into the bandit algorithm itself - it is your responsibility at the point of price selection. The pattern is simple: compute `offered_price = price_level * base_premium`, then cap it at the ENBP before quoting. As shown in the code above, `offer_price = min(price_level * base_premium, enbp)`. This keeps the arm selection and the compliance constraint separate, which makes both easier to audit.

Enforce the ENBP cap consistently. If your quoting engine is not passing the ENBP at quote time - which every GIPP-compliant insurer must be computing anyway - the experiment cannot run safely.

---

## Honest limitations

**The PCW rank problem.** If your direct channel includes an aggregator feed, the loading you apply affects your rank on the aggregator page, and rank affects conversion independently of price. A lower loading might improve conversion not because it's better value but because it moved you from rank 5 to rank 3. Thompson Sampling cannot separate these effects. The reward signal it learns is "what loading produces the highest conversion at this rank?" - but rank is endogenous to the experiment. This is the primary reason we describe `insurance-online` as a direct channel tool. Aggregator channel experiments are a harder problem that requires explicit rank modelling; that is v2 scope.

**Cold start.** In the first few observations on each arm, the posteriors are wide and the algorithm explores roughly uniformly. For a typical personal lines product on a mid-tier direct channel - 200-500 quotes per day - you will have meaningful differentiation between arms after roughly two to three weeks. If your volume is lower than that, the experiment runs slowly.

**Annual policy lifecycle.** Conversion is a fast signal. A customer either binds or doesn't, within hours of the quote. Claim frequency is a slow signal. You need policy years of exposure to observe it reliably. For frequency experiments, update the bandit as claims emerge by calling `bandit.update(price_level, converted=False)` for unclosed policies until claims are confirmed. But you cannot shortcut the fundamental problem: a loading that attracts better-risk business will take 12+ months to confirm at the claim frequency level.

**State is in-process.** `ThompsonBandit` is not thread-safe. In production, use one process per experiment. The `_history` list on each bandit records every selection; serialise it to JSON for checkpointing and restore by replaying updates.

---

## When to use this

Direct channel, new business only. Three situations fit well.

The first is loading band optimisation on an established product. You have a pricing model, a loading band, and you've been applying a fixed loading for a year. You want to know if a different loading converts better without sacrificing loss ratio. Three or five arms, Thompson Sampling, conversion reward model.

The second is new product launch. You have no historical data on what loading the market will accept. Rather than setting a fixed launch loading based on market positioning instinct, you can run a bandit experiment from day one. The algorithm will explore initially and converge as data accumulates.

The third is segment-level loading optimisation using `LinUCBPolicy`. If you believe the optimal loading varies by segment - vehicle group, NCD band, area - the contextual bandit takes a feature vector at selection time and learns a separate linear reward model per arm. This is the right tool for "our loading should be +5% in rural postcodes and -2% in urban postcodes". The contextual policy is more data-hungry than the non-contextual one - you need enough volume per segment to learn the linear weights - but for a product with meaningful volume segmentation it outperforms a single-loading experiment.

---

**[insurance-online on GitHub](https://github.com/burning-cost/insurance-online)** - MIT-licensed, PyPI. Bandit algorithms for UK general insurance pricing, with GIPP constraints and compliance reporting built in.

---

**Related reading:**
- [Constrained Portfolio Rate Optimisation with FCA ENBP Enforcement](/2026/03/07/insurance-optimise/) - the optimisation layer that bandit experiments inform: once you have unbiased loading estimates from online learning, this is how to translate them into portfolio-level rate recommendations
- [Synthetic Difference-in-Differences for Rate Change Evaluation](/2026/03/13/your-rate-change-didnt-prove-anything/) - SDID for evaluating the causal impact of a concluded experiment rather than running a live bandit
- [Double Machine Learning for Insurance Price Elasticity](/2026/03/01/your-demand-model-is-confounded/) - structural elasticity estimation as an alternative to experimental methods when you need a retrospective causal estimate
