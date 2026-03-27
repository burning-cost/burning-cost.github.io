---
layout: post
title: "Algorithmic Pricing on PCWs: The Collusion Risk Nobody Is Talking About"
date: 2026-03-25
categories: [pricing, regulation, strategy]
tags: [algorithmic-pricing, PCW, collusion, FCA, game-theory, Q-learning, reinforcement-learning, motor, competition-law, insurance-monitoring, insurance-optimise, market-structure]
description: "arXiv:2504.16592 formalises what pricing teams have been quietly observing for years: autonomous pricing algorithms can converge to supra-competitive prices without any firm ever picking up the phone. Here is what the paper actually shows, and what UK motor pricing teams should do about it."
---

UK motor insurance is distributed over 80% through price comparison websites. Every major insurer has an algorithmic pricing engine. Those two facts, in combination, create a game-theoretic structure that a paper published in April 2025 — Bichler, Durmann, and Oberlechner, arXiv:2504.16592, accepted at *Business & Information Systems Engineering* — analyses with some care. The conclusion is uncomfortable: under plausible conditions, independently-operating learning algorithms can sustain prices above the competitive equilibrium without any explicit coordination.

The FCA, which has been circling PCW market dynamics for some time, is now in a position to take this seriously. We think UK pricing teams that have not yet thought about this will be thinking about it within two years, whether or not they choose to.

---

## What the paper actually shows

The setup is a repeated Bertrand game. Multiple firms set prices simultaneously in each period. The theoretical baseline is the Nash equilibrium of the stage game: a price equal to cost (in symmetric cases), which is individually rational and leaves no collusive rent on the table. The question the paper asks is whether learning algorithms, operating independently, can sustain outcomes above this baseline.

The answer is yes, but with important conditions.

Calvano et al. (2020) — the most cited result in this literature — showed that Q-learning agents with logit demand maintained supra-competitive prices through a mechanism that resembles tacit collusion in repeated games. The algorithms, without any communication, learned to punish deviations: a firm that undercut was met with aggressive price reductions by the other agent, so undercutting was never rewarded. This is Nash reversion as emergent behaviour, not as an explicit strategy.

Bichler et al. do not simply accept this result. They map it more carefully against the game-theoretic conditions for collusion in repeated games. The key tension is between two senses of "equilibrium." Learning algorithms may converge to a correlated equilibrium or an approximate Nash equilibrium of the *repeated* game — which includes cooperative outcomes — without converging to the Nash equilibrium of the *stage* game, which does not. Whether the former counts as collusion in a legal or regulatory sense is genuinely unsettled.

The quantitative evidence the paper cites is striking. Assad et al. (2024) found that margins increased by 28% in local duopoly retail petrol markets in Germany when both firms adopted algorithmic pricing software. This is real-world evidence, not simulation. By 2015, algorithms set prices for roughly one-third of the top 1,600 products on Amazon; by 2018, the average product price changed every ten minutes. The PCW environment — discrete quote events, near-identical products, observable market prices, repeated interactions — is structurally similar to the oligopoly settings where collusive dynamics emerge most reliably.

Not every algorithm produces collusive outcomes. The paper reports that Q-learning with sufficiently large epsilon-greedy exploration (Abada et al., 2024) showed no collusion: enough random exploration disrupts the punishment mechanism. UCB variants sometimes produced correlated exploration that yielded supra-competitive outcomes (Hansen et al., 2021). Deep RL results are inconsistent across specifications. The architecture of the algorithm matters — which means firms have more agency here than the alarming headline suggests, but also more responsibility.

---

## Why PCWs amplify the risk

Standard tacit collusion in a repeated game requires each firm to observe competitors' prices and condition its future behaviour on what it sees. A PCW provides exactly that infrastructure. Rates flow through aggregators in near-real time. An algorithm deployed on GoCompare or Confused.com is operating in an environment where competitor prices are, in effect, observable on a quote-by-quote basis.

The standard legal defence against collusion is that firms never communicated. In a PCW market, they do not need to. The market mechanism itself is the communication channel. Bichler et al. are careful about the legal framing — they note that the European Commission's 2023 guidelines on horizontal agreements explicitly cover explicit arrangements to use the same pricing software but do not cover independent algorithms that converge to the same outcome through separate learning. That gap is where the risk lives.

The second amplifying factor is product homogeneity. UK motor is, for most customers in most segments, functionally identical across providers — comprehensive cover, motor legal, breakdown assist, the same core policy wording. When products are homogeneous and prices are directly comparable, the only dimension of competition is price. That is exactly the market structure where Bertrand equilibria are sharpest and where deviations from competitive equilibrium are most observable and most punishable.

The third factor is market concentration at the aggregator level. Four platforms — Compare the Market, GoCompare, MoneySuperMarket, and Confused.com — control the overwhelming majority of PCW-distributed motor business. That concentration means a small number of pricing systems are interacting repeatedly in a well-defined arena. The theory of algorithmic collusion does not require many players: duopoly results are the clearest, and a four-player game is structurally close enough to a duopoly when each firm has a dominant position in some segment.

---

## The Q-learning mechanism in plain terms

For pricing teams who want the mechanism without the game theory: Q-learning maintains a value table mapping (state, action) pairs to expected rewards. In a pricing context, the state might include current competitor prices (observable on a PCW), own NB volume, and quote conversion rate. The action is the price to quote. The reward is margin per policy, discounted.

After enough iterations, the algorithm learns that quoting below a certain threshold triggers retaliatory cuts from competitors — because competitors' algorithms have learned the same thing from the other side. The result is a price floor that both algorithms enforce without either being programmed to do so. Neither algorithm has a "cooperate" flag. Neither has communicated with the other. But the equilibrium they converge to is cooperative in outcome.

The epsilon parameter matters because randomisation is what breaks the punishment cycle. An algorithm that occasionally quotes randomly cannot be reliably punished — you cannot distinguish deliberate deviation from noise. High exploration rates produce competitive outcomes; low exploration rates (as algorithms mature and exploitation dominates) can produce cooperative ones. This is relevant for UK motor specifically because pricing teams routinely reduce exploration rates on mature books. A well-tuned, well-exploiting algorithm may be exactly the kind that sustains supra-competitive prices.

---

## The regulatory moment

The FCA's 2023 and 2024 work on PCW market dynamics — including its multi-firm requests for information on pricing methodology and its Consumer Duty implementation reviews — has largely focused on customer outcomes at the individual level: are renewal customers being treated fairly, are vulnerable customers being identified, is the ENBP constraint holding? These are PS21/5 and Consumer Duty questions.

The competition dimension is structurally different. It sits with the Competition and Markets Authority rather than the FCA, and the CMA has historically been slow on algorithmic competition cases — partly because the theory is genuinely complex, partly because the existing legal frameworks were built for explicit agreements. The CMA's 2025 work programme includes a strand on algorithmic competition. Bichler et al.'s paper — and the wider literature it synthesises — provides the theoretical scaffolding that regulators need to bring cases. We expect the question of whether independent algorithmic pricing on PCWs constitutes tacit collusion to be live before the CMA within three years.

The practical implication for firms is not that they should stop using algorithmic pricing — that would be commercially self-destructive — but that they should be able to demonstrate that their algorithms are not tuned in ways that produce collusive dynamics. That means documentation of exploration parameters, convergence behaviour, and the degree to which competitor price signals feed into the action space.

---

## What pricing teams should watch for

Bichler et al. identify several observable signatures of algorithmic collusion. Here is our translation for UK motor pricing:

**Convergent pricing across segments.** If your quoted prices are tracking competitor prices closely across risk segments — not just at the market average, but within individual cells — that is a signal worth investigating. Genuine cost-based pricing should produce segment-level divergence because cost structures differ. Convergence suggests the algorithm is responding to market price signals rather than pure cost.

You can detect this with [insurance-monitoring](/insurance-monitoring/). The `MonitoringReport` class computes per-feature distribution shifts using PSI — for a detailed walkthrough of how PSI and Gini drift detection work in practice, see [insurance model monitoring beyond generic drift](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/). If you also pass competitor price relativities as a feature, sustained convergence will show as low PSI between own and market rates — which is the opposite of the drift signal you normally look for. We would add a bespoke correlation monitor: track the rolling correlation between own rate changes and observed market rate changes at segment level. If that correlation is consistently above 0.8 and you cannot attribute it to shared cost drivers, you have a question to answer.

**Price stickiness at round numbers or common thresholds.** Algorithms that have learned soft coordination often converge to salient focal points — round premium levels, common rate multiples. These are not programmed; they emerge from the learning process. Review your quote distribution for unusual clustering at round figures.

**Reduced sensitivity to own cost signals.** In a collusive equilibrium, prices respond more to competitor prices than to own cost changes. Build a simple regression: does your rate change in period *t* predict better from own loss ratio development or from the market average rate change in period *t-1*? If the latter dominates, your algorithm is tracking the market, not your costs.

**Exploration rates declining over time.** Check what your algorithm's effective exploration rate is now versus twelve months ago. If epsilon has been tuned down through standard optimisation cycles and you have never explicitly considered the collusion risk of low exploration, review it now.

On the constraint side: [insurance-optimise](https://burning-cost.github.io/insurance-optimise) enforces technical floor constraints (`ConstraintConfig(technical_floor=True)`) that prevent pricing below cost. But a technical floor does not protect against supra-competitive pricing — the concern here is that prices are too high relative to competition, not too low. The relevant constraint is to ensure your optimiser's objective function is genuinely cost-anchored and that competitor price terms enter only as soft constraints, not as the primary signal.

---

## Our position

The paper's core argument is correct: algorithmic pricing in a repeated PCW market creates the structural conditions for tacit collusion, and the existing competition law framework does not have a clean answer for it.

We do not think most UK motor insurers are deliberately engineering collusive outcomes. We do think that the current generation of pricing algorithms — particularly those that have matured to low exploration rates, that feed on PCW quote data, and that optimise margin subject to volume constraints — may be producing collusive dynamics as an emergent property without anyone inside the firm intending that or noticing it.

The FCA has the consumer-protection side of PCW dynamics under active supervision. The CMA has the competition side on its radar but has not yet brought algorithmic collusion into sharp focus for insurance. That gap will not persist indefinitely. The firms that can demonstrate, credibly, that their pricing algorithms are cost-based rather than market-tracking will be better positioned when regulators close it.

The practical action is not algorithmic redesign. It is monitoring and documentation: know what your exploration parameters are, know whether your rates are tracking costs or competitors, and have an audit trail that shows which. That is the same discipline that good model risk management requires for any other dimension of pricing — it should not require a regulator to ask for it first.

---

**Paper:** Bichler, Durmann, Oberlechner — "Algorithmic Pricing and Algorithmic Collusion", arXiv:2504.16592, accepted *Business & Information Systems Engineering* (2025).

**Related tools:** [insurance-monitoring](https://burning-cost.github.io/insurance-monitoring) — portfolio and model monitoring including PSI, A/E, Gini drift. [insurance-optimise](https://burning-cost.github.io/insurance-optimise) — constrained pricing optimisation with FCA audit trail.
