---
layout: post
title: "Reinforcement Learning for PCW Insurance Pricing: What Actually Works"
date: 2026-03-28
categories: [pricing, research, strategy]
tags: [reinforcement-learning, contextual-bandit, PCW, margin-optimisation, GIPP, Thompson-Sampling, motor, insurance-optimise]
description: "A practitioner-oriented deep dive on applying reinforcement learning and contextual bandits to PCW margin optimisation for UK personal lines. Two serious papers exist, six hard problems remain unsolved, and there is no production-ready open-source implementation. Here is the honest state of play."
---

UK motor insurance is now a PCW-first market. Around 66% of personal lines motor policies are sold through comparison websites — up from 60% before GIPP — and the six major insurers compete for top-5 positions on every single quote. The margin decision happens at the moment of quote: set the loading too high and you fall off the results page; set it too low and you write business at a loss. Every insurer is solving a version of this problem, and most are solving it with a batch pricing model that runs monthly.

The question we want to answer here is whether reinforcement learning — specifically contextual bandits — can do better. The honest answer, as of early 2026, is: probably yes in principle, definitely not yet in practice. The literature is thin, the open-source tooling is missing, and none of the serious papers address the full set of constraints a UK insurer actually operates under. But the building blocks exist, and the problem structure maps well onto two adjacent fields that are further along.

This post covers: the business problem and why it is a bandit problem; the two papers worth reading and what they actually show; the six hard technical problems that the papers do not solve; what transfers from ad bidding and e-commerce; and what a practical MVP looks like.

We assume familiarity with the endogeneity problem in PCW conversion modelling — if you have not read our posts on [the PCW endogeneity problem]({% post_url 2026-03-26-the-pcw-endogeneity-problem-why-your-conversion-model-is-biased %}) and [DML demand elasticity estimation]({% post_url 2026-03-26-estimating-pcw-conversion-elasticity-with-double-machine-learning %}), start there. The demand model is a prerequisite for the bandit.

---

## The business problem

At each quote event, an insurer observes a customer profile $$x_t$$ (age, NCD, vehicle, area, claims history) and sets a premium $$P_t = a_t \times C(x_t)$$, where $$C(x_t)$$ is the burn cost from the technical pricing model and $$a_t \geq 1$$ is the margin multiplier. The customer converts with probability $$p(y_t = 1 \mid x_t, a_t)$$, which depends on how $$P_t$$ ranks against competitors on the PCW. The insurer observes conversion $$y_t \in \{0, 1\}$$ some days later, but never observes competitor prices directly.

The objective is to maximise cumulative expected profit:

$$\max_\pi \; \mathbb{E}\left[\sum_{t=1}^{T} (a_t - 1) \cdot C(x_t) \cdot y_t\right]$$

subject to: $$a_t \leq a_{\text{ENBP}}(x_t)$$ for renewal customers (GIPP constraint, mandatory since January 2022).

This is a contextual bandit problem. The context is $$x_t$$, the action is $$a_t$$, the reward is margin times conversion, and the key difficulty is that the reward model — conversion probability as a function of $$a_t$$ — is unknown and must be learned from the logged data. The competitor prices that mediate conversion are never directly observed.

The contrast with current practice matters. Most insurers today run `insurance-optimise`-style batch optimisers: fit a demand model to historical quote data, optimise loading by segment, deploy as a rate table, review quarterly. This is gradient-based batch optimisation, not online learning. It cannot adapt between review cycles, and it cannot personalise to individual customers within a segment. An online bandit does both.

---

## Two papers worth reading

The academic literature on RL for PCW insurance pricing is, to put it plainly, small. We surveyed it in March 2026 and found two papers worth treating seriously.

### Treetanthiploet et al. (2023) — arXiv:2308.06935

This is the foundational PCW bandit paper. The authors — from LSE, Edinburgh/ATI, Accenture UK, and esure — formulate the problem as a contextual bandit with action space $$a_t \in [0.7, 1.3]$$, discretised to 600 values. The reward is:

$$r_t = a_t \times P_0(x_t) - b(x_t)$$

where $$P_0$$ is a benchmark premium and $$b$$ is burn cost. Conversion probability depends on normalised rank:

$$z_t = \frac{a_t P_0(x_t) - \overline{\text{Top5}}}{\overline{\text{Top6-10}} - \overline{\text{Top5}}}$$

and the demand function is piecewise quadratic in $$z_t$$, capped at 0.2 — a realistic PCW insight, since the cheapest insurer on a PCW still only converts around 20% of quotes.

The methodological contribution is the hybrid approach. Rather than training from scratch on live data (cold-start problem), they pretrain offline on a synthetic market simulator, then update with live conversions. In benchmarks against seven agents on 7,000 test customers, the hybrid outperforms model-free RL, random agents, and variants with miscalibrated market simulators. Only a perfect-information oracle beats it.

The paper is a genuine advance on the problem. What it does not do: no GIPP constraint; no IPS correction for the historical demand estimate; no competitor dynamics (competitors are modelled as exogenous draws from $$\mathcal{N}(1, 0.3) \times C$$, not as strategic responders); no code released. The synthetic data makes the results hard to generalise.

### Young et al. (2024) — arXiv:2408.00713

This paper from the Alan Turing Institute takes a different angle: not a contextual bandit per customer, but a full MDP that optimises a portfolio towards a target risk mix $$\rho^*$$. The state is $$(s_t, \rho_t, t)$$ — current customer, current portfolio composition, time — and the agent uses backward induction with a novel k-value function:

$$k_\pi(s, \rho, t) = 1 - \frac{V_\pi(\rho \cup \{s\}, t) - V_\pi(\rho, t)}{C(s)}$$

The optimal action at each quote is then $$\argmax_a \, P(y=1 \mid s, a) \times (a - k_\pi)$$ — a margin over a customer-specific hurdle rate that encodes how well this customer fits the portfolio objective.

Results: +7% profit (£6,916 vs £6,444, $$p < 0.0001$$, Cohen's $$d = 0.51$$, 24 trials, $$T = 1{,}000$$ customers per epoch). The code is publicly available at [github.com/EdwardJamesYoung/RL-portfolio-pursuit](https://github.com/EdwardJamesYoung/RL-portfolio-pursuit) under CC BY-NC 4.0 — free for research, not commercial use.

The k-value framework is the most practically useful idea in the insurance RL literature. It makes explicit what every portfolio manager does intuitively: a customer who fits the portfolio is worth more than one who concentrates risk. The limitation is T=1,000 customers per epoch, which is small, and again the market is synthetic. No GIPP constraint.

The other three papers in this space are not worth spending much time on. Rebollo-Monedero et al. (2019) address renewal price adjustment on direct channels with real BBVA data — no PCW, no competitor modelling. The ASTIN Bulletin 2023 paper on premium control is about solvency-aware aggregate rate setting, a different problem entirely. The 2025 SSRN preprint RL-Insure (Islam et al.) has no real data, no PCW context, and introduces metrics it does not define.

---

## Six hard problems the papers leave open

This is the section we would have wanted to read before starting to build. Each problem is tractable, but none has a production-tested solution in the insurance context.

### 1. Competitor price observability

On a PCW, conversion is determined by rank among N competitors. The insurer sees its own price and the binary conversion signal. It never sees what competitors quoted.

Treetanthiploet et al. work around this by estimating the normalised rank $$z_t$$ from a market simulator trained on historical data. The insurer does not observe $$z_t$$ live — it predicts it from its own price and a prior on competitor behaviour. This is defensible as a starting point but breaks down when competitors react strategically to your pricing changes.

The ad tech literature has the same problem. In first-price auctions, losing bids are unobserved. The solution is bid distribution estimation from winning data alone — estimating the competitor price distribution from the subset of quotes where you converted (because you were cheapest). The technique is directly portable to insurance PCW data. The key paper here is arXiv:2402.07363 (2024) on strategically robust learning for first-price auction bidding. No insurance paper has applied this.

### 2. Delayed feedback

The standard bandit assumption is that the reward is observed immediately after the action. In UK motor insurance it is not. A quote is issued today; the customer may accept up to 14 days later; the policy starts some weeks hence; the first claim arrives months or years after that.

Delayed NeuralUCB (arXiv:2504.12086, April 2025) derives a regret bound for this setting:

$$\mathcal{R}(T) = O\!\left(\tilde{d}\sqrt{T \log T} + \tilde{d}^{3/2} D_+ \log(T)^{3/2}\right)$$

where $$D_+$$ is the expected delay. The reassuring result is that delay cost is sub-linear in $$T$$ — with a 14-day conversion window, the delay penalty is small relative to the primary exploration cost.

The more serious problem is claims delay, which runs to 12+ months. Treetanthiploet address this by defining reward as margin times conversion only, ignoring claims entirely. This sidesteps the delay problem but also ignores risk selection. The practical fix is to decouple conversion (immediate proxy reward) from profitability (deferred correction from the technical model), treating them as separate reward channels with different update frequencies.

### 3. GIPP / ENBP constraint

Neither paper implements this, and for a UK motor insurer it is not optional. Since January 2022, renewal premium must not exceed the Equivalent New Business Price. In bandit terms, the action space for renewal customers is clipped:

$$a_t \leq a_{\text{ENBP}}(x_t) \quad \forall \; \text{renewal customers}$$

The mechanical fix — clip the action space before selecting $$a_t$$ — is straightforward. The subtlety is that $$a_{\text{ENBP}}(x_t)$$ is endogenous to your own new business pricing strategy. If the bandit lowers your NB prices in a segment to win more business, it tightens the renewal ceiling for those customers in subsequent years. A naive implementation ignores this feedback loop; a correct one treats the NB and renewal action spaces jointly.

A second subtlety: FCA EP25/2 (published July 2025) confirmed GIPP largely achieved its objectives. Motor premiums fell 5.9% in Q1 2022, home 6.6%, and PCW share rose from 60% to 66% of sales. The post-GIPP market is structurally more competitive on PCW than the pre-2022 market from which most demand models are calibrated. A market simulator trained on pre-2022 data is misspecified for the current environment.

### 4. Adverse selection in the reward function

The standard reward — $$(a_t - 1) \cdot C(x_t) \cdot y_t$$ — penalises underpricing only through the burn cost: if $$a_t \times P_0(x_t) < b(x_t)$$, the reward is negative. But this only captures expected adverse selection, not the variance of loss ratio exposure. A high-risk customer at $$a_t = 1.0$$ may look marginally profitable in expectation but carries significant adverse selection risk if $$C(x_t)$$ is estimated with error.

Young et al. partially address this through portfolio pursuit: the target portfolio $$\rho^*$$ encodes a desired risk mix, and the k-value framework internalises the portfolio cost of adding each customer. This is a more principled approach than burn-cost clipping alone.

A cleaner formulation augments the reward with an adverse selection penalty:

$$r_t = (a_t - 1) \cdot C(x_t) \cdot y_t - \lambda_{\text{AS}} \cdot \mathbb{E}[\text{LR}(x_t, a_t) - 1]_+ \cdot y_t$$

where $$\text{LR}(x_t, a_t)$$ is the expected loss ratio at loading $$a_t$$. This penalises underpricing of high-risk customers independently of the expected margin. No paper implements this.

### 5. Non-stationarity

The PCW market has undergone two major structural shifts since 2020: GIPP in January 2022, and the hard market in 2022–23 when insurers repriced aggressively after inflation-driven claims cost increases. A bandit pretrained on pre-2022 data will be miscalibrated for the current market on both conversion probability and competitor behaviour.

Treetanthiploet et al. assume i.i.d. customer arrivals — no non-stationarity handling. Young et al. have six competing agents that iteratively update strategies, which is better, but still within a single synthetic episode. Neither paper addresses regime changes.

The correct tools are: sliding-window Thompson Sampling (discount older conversion observations); change-point detection on the demand function; periodic full retraining of the market simulator. These are all standard in the non-stationary bandit literature but have not been applied to insurance PCW pricing.

### 6. Algorithmic collusion

If multiple major UK insurers deploy RL pricing agents simultaneously, there is a non-trivial probability of tacit collusion emerging — sustained prices above the competitive equilibrium without any explicit coordination. We covered this in detail in our [earlier post on PCW collusion risk]({% post_url 2026-03-25-algorithmic-pricing-pcw-collusion-risk %}), and the short version is: tabular Q-learning is more prone to collusion than deep RL; PPO appears least prone among algorithms tested (arXiv:2406.02437, 2024; arXiv:2503.11270, 2025); the CMA launched a dynamic pricing investigation in November 2024.

The practical implication for RL design: prefer PPO over DQN; log all pricing decisions with sufficient detail for regulatory audit; build hard price cap constraints that operate independently of the RL output. Thompson Sampling, by its nature, maintains exploration permanently — which disrupts the punishment mechanism that sustains tacit collusion. This is an underappreciated argument in its favour.

---

## What transfers from ad bidding and e-commerce

Two adjacent fields are 5–10 years ahead of insurance on bandit-based price optimisation.

**Real-time bidding** is the closest analogue. In an RTB auction, an advertiser bids $$b_t$$ on an impression (context = user features). If $$b_t$$ exceeds the clearing price, they win; otherwise they lose and never observe the clearing price. This is structurally identical to PCW insurance: margin corresponds to bid, burn cost to cost, conversion to win signal, competitor prices to unobservable competing bids.

The technique that transfers directly is competitor price distribution estimation from winning data. The insurer only observes the prices at which it converted — i.e., the prices at which it was cheapest. From this censored sample, plus aggregate market data from suppliers like Consumer Intelligence or TransUnion, the competitor price distribution can be estimated. This gives a better-calibrated market simulator than Treetanthiploet's assumption of $$\mathcal{N}(1, 0.3)$$ competitor noise.

The important difference: PCW insurance uses the cheapest-wins mechanism, not a second-price auction. There is no bid shading correction required — the insurer pays its own quoted price. This simplifies the reward structure relative to RTB.

**E-commerce markdown pricing** is the second relevant field. Contextual Thompson Sampling (CTS) in production e-commerce (ACM JCDL 2024) consistently outperforms UCB variants in sparse reward settings — and insurance conversion rates (typically 2–8% on PCW) are exactly sparse. This is a direct recommendation: CTS over LinUCB for the policy algorithm, particularly in the early data-sparse phase of deployment.

E-commerce knows all competitor prices (public shelf prices). Insurance does not. This is the key gap that makes every other technique harder to apply.

**Revenue management** (hotels, airlines) is partially relevant for non-stationary demand modelling. The airline literature on booking curve estimation under volatile demand transfers to the problem of handling hard market cycles. What does not transfer: airlines price continuously with no GIPP equivalent and have immediate conversion signals.

**Delayed feedback bandit theory** is directly applicable to the 14-day conversion window. Delayed NeuralUCB's result — that delay cost is sub-linear in $$T$$ — is reassuring. The 14-day window is short enough that a standard Thompson Sampling implementation, treating unresolved quotes as missing data rather than zero rewards, performs close to the theoretical optimum.

---

## The open-source gap

No open-source package exists that combines the four things a PCW insurance bandit needs:

1. A realistic UK PCW market simulator with competitor behaviour, PCW ranking mechanics, and GIPP constraint
2. A contextual bandit policy (Thompson Sampling or LinUCB) for margin setting on that simulator
3. Off-policy evaluation tools using logged quote data — IPS or doubly robust estimators to correct for historical pricing policy selection bias
4. GIPP/ENBP constraint enforcement in the action space

What does exist is a set of useful building blocks. Vowpal Wabbit (Microsoft) is production-grade for contextual bandits and used at scale in ad serving. Open Bandit Pipeline (zr-obp) has the off-policy evaluation components — IPS, DM, and DR estimators — directly applicable to the logged quote data correction problem. Young et al.'s code (github.com/EdwardJamesYoung/RL-portfolio-pursuit) is the only insurance-specific open source, but it is synthetic-market-only and CC BY-NC 4.0. Treetanthiploet released nothing.

The commercial platforms — Earnix Price-It, Akur8, hx Renew — do price optimisation, but based on all available public information their approaches are batch gradient descent with a static demand model, not online bandit learning. The update frequency advantage of a contextual bandit (posterior update after each observed conversion) is the gap that commercial platforms have not closed. Earnix's UK clients include major insurers; their methodology is a black box.

---

## What a practical MVP looks like

Given all of the above, here is what we would actually build.

**Phase 1 (4–6 weeks): Demand model and offline simulation**

The demand model $$\hat{p}(y=1 \mid x, a)$$ is the foundation. This is where `insurance-causal-policy` (our DML demand elasticity library) does the heavy lifting: the DoubleML estimator gives a causally identified price coefficient from logged quote data, using commercial loading variation across rate review periods as an instrument. The key step most teams skip is IPS correction: the historical quotes were generated under an existing pricing policy, so the sample is not a random exploration of the action space. The correction matters.

Initial model: logistic regression with rank effect encoded via estimated normalised rank $$\hat{z}_t$$. The market simulator calibrates the competitor price distribution from the winning-bid censored sample using standard censored regression, not the $$\mathcal{N}(1, 0.3)$$ assumption.

Minimum data requirement: 50,000 PCW quotes with timestamp, customer features, quoted premium, burn cost, approximate rank position (or at minimum: top-5 indicator), and conversion. Monthly market quantiles from Consumer Intelligence or similar to calibrate competitor priors.

**Phase 2 (4–6 weeks): Bandit policy and GIPP constraint**

Thompson Sampling over the demand model. At each quote, sample $$\theta \sim \text{posterior}$$, compute expected reward over the feasible action space, select:

$$a_t = \argmax_{a \in \mathcal{A}(x_t)} \hat{p}_\theta(y=1 \mid x_t, a) \cdot (a - 1) \cdot C(x_t)$$

where $$\mathcal{A}(x_t) = [1.0, a_{\text{ENBP}}(x_t)]$$ for renewals, $$[0.7, 1.3]$$ for new business. GIPP enforcement is a clip on the action space — no algorithmic change required, but the endogenous relationship between NB pricing and renewal constraint needs to be modelled explicitly.

Off-policy evaluation using zr-obp's doubly robust estimator to test the candidate bandit policy against logged data before any live deployment. The baseline for OPE comparison is `insurance-optimise`'s current batch margin output.

**Phase 3 (ongoing): A/B deployment and live updates**

Deploy the bandit on a subset of PCW traffic (20–30%) while the existing pricing system handles the rest. Update the posterior after each resolved quote (conversion observed). The 14-day window means an average 7-day delay to posterior update — small enough to treat with standard delayed-feedback Thompson Sampling (buffer unresolved quotes; update in batch as conversions arrive).

Monitor demand function stability monthly. Fit a change-point detector on the empirical conversion rate by rank bucket. If a structural break is detected (as would have occurred in January 2022 or during the 2022–23 hard market), trigger a full market simulator refit before the next live update cycle.

Log every pricing decision with sufficient granularity for regulatory audit. The CMA dynamic pricing investigation means an insurer deploying an RL agent without decision logs is taking unnecessary regulatory risk.

---

## What we are not claiming

This post is a research review and architecture sketch, not a deployment playbook. The two serious papers in this literature both use synthetic markets and neither has been validated on real PCW data. The hard market 2022–23 was a significant non-stationarity event that no existing model has been tested against. GIPP enforcement, as described here, handles the static constraint but not the dynamic feedback between NB pricing and renewal ceiling.

We are also not claiming that a contextual bandit necessarily outperforms a well-specified batch optimiser in practice. The theoretical advantage — continuous learning, personalisation to individual customers rather than segments, explicit exploration — is real. Whether it survives the operational messiness of a production insurance pricing system (quote volumes, data latency, IT constraints, regulatory approval cycles) is an empirical question nobody has yet answered.

The honest state of play: the problem formulation is right, two decent papers exist, the building blocks are available, and the gap between current practice and what the literature shows is tractable. Whether it is worth closing depends on your PCW volume, your data quality, and your tolerance for the regulatory uncertainty that attaches to any RL-based pricing system in the current FCA environment.

We think it is worth closing. We will write more as we build.
