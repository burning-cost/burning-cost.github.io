---
layout: post
title: "Reinforcement Learning for Individual Claims Reserving: What Avanzi, Richman, and Wüthrich Propose"
date: 2026-03-25
categories: [reserving, machine-learning, research]
tags: [reserving, reinforcement-learning, mdp, individual-claims, micro-level, selection-bias, ibnr, rbns, chain-ladder, neural-reserving, soft-actor-critic, outstanding-claims, actuarial, research]
description: "Avanzi, Richman, and Wüthrich reformulate individual claims reserving as a Markov Decision Process. We explain why it matters, what it actually does, and when a UK reserving actuary would use it."
---

Benjamin Avanzi, Ronald Richman, Bernard Wong, Mario Wüthrich, and colleagues published [arXiv:2601.07637](https://arxiv.org/abs/2601.07637) in January 2026. The paper reformulates individual claims reserving as a claim-level Markov Decision Process in which an agent sequentially updates Outstanding Claim Liability (OCL) estimates over development. The reward function is designed to balance predictive accuracy against reserve stability. The implementation uses Soft Actor-Critic (SAC) — a standard deep RL algorithm — and evaluation runs on two datasets: the CAS loss development data and the SPLICE synthetic insurance dataset.

This post explains why the problem is hard, what the MDP formulation achieves, how it fits the pattern of neural reserving we have been tracking, and where the practical limits are for UK actuaries.

---

## The selection bias problem that chain ladder ignores

Aggregate triangle methods treat claims as a portfolio property. You count paid losses at each development period, project the triangle forward, and the individual claim is never observed. This is computationally convenient and works tolerably well on stable, homogeneous books.

Individual claims reserving breaks this abstraction. Now you observe each claim's development path: FNOL characteristics, handler reserves, interim payments, changes in litigation status, and eventual settlement. The unit of analysis is the claim, not the accident year cohort.

The problem is that at any valuation date, your training data is biased. You have full development histories for settled claims. You have partial histories for open claims. If you train a supervised model on settled claims — predicting ultimate cost from early-period features — you are implicitly assuming that open claims at valuation follow the same distribution as claims that have already settled. They do not.

Claims settle when they are ready to settle. Small, straightforward claims close quickly. Large, disputed, or litigated claims stay open. At any given valuation date, the open portfolio is adversely selected relative to the settled portfolio. A model trained on settled claims will systematically underestimate reserves for open claims.

This is not a subtle effect. On liability lines with long development tails — UK motor bodily injury, EL, public liability — claims open after five or more years are structurally different from claims that settled in year one. If your individual claims model does not account for this, you are measuring a biased sample and applying it to a population that does not match.

Traditional micro-level approaches address this through survival analysis (fitting a model for time-to-settlement) or by explicitly conditioning on claim status. This works but requires specifying the selection mechanism correctly. If your model of the settlement process is wrong, the bias correction is wrong.

---

## What the MDP formulation does differently

The RL framing does something conceptually cleaner. Rather than treating OCL estimation as a prediction problem, it treats it as a sequential decision problem: at each development quarter, the agent observes claim state and chooses an action — how much to revise the OCL estimate. The environment returns a reward based on how accurate the estimate turns out to be and how stable the revision was.

The key insight is that the agent **learns from all claim trajectories**, including open claims at valuation. A supervised approach can only evaluate prediction quality against eventually-settled outcomes. The RL agent receives reward signals during development — it learns from interim accuracy, not just terminal accuracy. Open claims contribute training signal throughout their development, not just once they close.

This sidesteps the selection bias problem in a principled way. The agent never needs to ask "what would this claim have settled for under the settled-claim distribution?" It asks instead "given this claim's current state, what is the OCL estimate that has been rewarded historically in similar states?"

The MDP state at each period includes claim characteristics, payment history, current handler reserve, time in development, and claim status (open or closed). The action space is continuous — the agent proposes an OCL update, not a bucket choice. The reward penalises both inaccuracy (difference between OCL and eventual settlement) and instability (large jumps in the OCL from one period to the next). Reserve volatility is a real cost: it affects the P&L, creates capital strain, and invites regulatory attention. Penalising it directly in the training objective is the right move.

Three additional components in the paper are worth noting:

**Initialisation.** New claim OCL estimates at FNOL must be set before any development is observed. The paper proposes a separate initialisation procedure for new claims rather than using the RL agent, which would have no state to condition on.

**Rolling-settlement tuning.** Hyperparameter selection is done via a rolling-settlement scheme — essentially a time-series cross-validation that respects the chronological structure of claim development. You cannot use random cross-validation here; a claim settled in 2022 must not be used to tune parameters for 2021 valuations.

**Importance weighting for large claims.** Large claims are rare. A flat reward function will produce an agent that performs well on the modal claim and catastrophically on the 1% of claims that drive 30% of IBNR. The paper applies importance weighting to increase the training signal from large claim events. This connects directly to the severity modelling challenge: the tail needs disproportionate attention.

---

## How this fits the neural reserving trend

A companion paper from Richman and Wüthrich — [arXiv:2603.11660](https://arxiv.org/abs/2603.11660), "One-Shot Individual Claims Reserving" — takes a different approach: feedforward networks and LSTMs that predict ultimate cost from claim transaction histories in a single step. That work finds that handler reserve (incurred) substantially outperforms cumulative paid as a primary feature on liability lines, and that one-shot projection-to-ultimate factors at claim level cut Mack chain-ladder RMSEP by around 40% on accident data.

The RL paper is doing something categorically different. It is not predicting ultimate cost from claim features. It is training an agent to produce estimates that are accurate **and stable over time** — an objective that one-shot prediction cannot express. A single-period neural network makes one prediction per claim. The SAC agent makes a prediction at every development period and is trained jointly on accuracy across all of them.

The papers are complementary: one-shot prediction gives you the best estimate of ultimate cost given claim characteristics at a point in time; the RL agent tells you how to update that estimate as the claim develops and new information arrives. Both beat chain ladder on immature claim segments. The RL paper reports strongest results on segments with high proportions of open claims, precisely where selection bias is most acute and where the agent's ability to learn from open claim trajectories matters most.

The broader pattern we are tracking is that micro-level models are now genuinely competitive with aggregate methods at reasonable scale. The data volumes required are realistic for any UK insurer writing 50,000+ claims per year on a liability line. The computational cost of SAC training is not trivial, but it is not prohibitive either.

---

## The severity library connection

Both the selection bias problem and the importance weighting solution connect to the severity modelling work we maintain in `insurance-severity`. The distributional refinement network (DRN) in that library produces full predictive distributions rather than point predictions. Our EVT module in [`insurance-severity`](/insurance-distributional/) addresses the same tail problem from a different angle: for the practical context on why policy limits and truncation distort standard severity fits, see [your GPD is lying because your claims are truncated](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) — which matters when you are evaluating OCL estimates against eventual settlement.

The importance weighting mechanism in the RL paper is essentially a tail-weighted loss. Our EVT module in `insurance-severity` addresses the same problem from a different angle: fitting a generalised Pareto distribution to exceedances above a threshold so that the tail is modelled separately rather than distorted by the body of the distribution. If you are building an individual claims reserving pipeline and large claims are a material concern — which they will be on EL and motor BI — EVT-based initialisation for high-severity FNOL claims combined with the RL agent's sequential updating is a sensible architecture worth exploring.

---

## Practical relevance for UK reserving actuaries

The use case is narrow but real. The approach is relevant when:

1. **You have sufficient individual claim history.** The RL agent trains on individual claim trajectories. You need several years of development history with reasonable completeness. This disqualifies new classes and materially restructured books.

2. **Selection bias is material.** For short-tailed, homogeneous lines — household contents, simple motor own damage — average development patterns are stable and selection effects are small. Chain ladder will be competitive. For long-tailed liability lines with significant litigation, open claims are systematically heavier than settled claims and the selection effect is worth addressing.

3. **Reserve stability is a genuine objective.** If your board, regulator, or investors are sensitive to period-on-period reserve movements, an objective function that explicitly penalises instability is better aligned with real constraints than one that minimises RMSEP alone. Solvency II SCR calculations are affected by reserve volatility; reducing it directly is not just aesthetically satisfying.

4. **You have actuarial resource to implement and validate it.** This is not a drop-in replacement for a reserving triangle. You need to validate the MDP state specification, tune the reward function, and back-test the rolling-settlement scheme. Solvency II Article 121 and PRA SS3/18 validation requirements apply (for an internal model change application under Solvency II). The interpretability challenge is significant — explaining to a reserving committee why the agent revised OCL from £85k to £92k requires more than pointing at a triangle.

The last point is the honest constraint. For most UK non-life teams, the practical path is to use this as a benchmark or sensitivity check alongside chain ladder and Bornhuetter-Ferguson, not as the primary reserve estimate until there is more regulatory familiarity with RL-derived reserves.

---

## Honest limitations

**Data requirements are substantial.** The paper evaluates on SPLICE — a synthetic dataset with 30,000 simulated claims — and on CAS data. Real individual claim data is dirtier, more sparse, and subject to changes in claims handling philosophy that are not captured by the MDP state specification. A change in litigation strategy mid-period will look like non-stationarity in the environment and will degrade agent performance.

**Computational cost.** SAC training with a large replay buffer is GPU-dependent for any realistic claim volume. A UK insurer with 200,000 liability claims in development will need dedicated infrastructure. This is not a laptop job.

**No IBNR.** The paper is RBNS only — reported-but-not-settled claims. IBNR (unreported claims) requires a separate model. Individual claims reserving never fully replaces aggregate methods because IBNR by definition lacks individual claim data. A complete reserving framework still needs a macro-level IBNR estimate alongside the micro-level RBNS model.

**Regulatory acceptance is zero today.** The PRA does not have guidance on RL-derived reserves. Getting this through an internal model change application would require significant groundwork on validation methodology, back-testing evidence, and explainability standards. We expect that standard to shift over the next five years, but we would not present an SAC-derived reserve to a reserving committee without a substantial parallel-run against approved methods first.

**The reward function is a design choice.** The balance between accuracy and stability in the reward function is a hyperparameter with material consequences. Overweighting stability produces a conservative agent that under-responds to deteriorating claims. Overweighting accuracy produces a volatile agent. The rolling-settlement tuning helps, but the choice is not automatic and needs actuarial judgment.

---

## The bottom line

The Avanzi/Richman/Wüthrich RL formulation is technically sound and addresses a real problem — selection bias in individual claims reserving — more cleanly than most supervised alternatives. The sequential learning from open claims, the explicit stability penalty, and the importance weighting for large claims are all the right ideas.

For UK non-life actuaries: treat this as a significant research result worth understanding and prototyping rather than a production-ready tool. The immature claim segments where it works best — long-tail liability lines with high proportions of open claims — are exactly the segments where current reserving uncertainty is highest. That makes the potential value substantial, but also makes the validation burden higher.

The paper is at [arXiv:2601.07637](https://arxiv.org/abs/2601.07637).
