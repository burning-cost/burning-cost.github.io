---
layout: post
title: "Reinforcement Learning for Dynamic Reinsurance Optimisation: What the Research Actually Shows"
date: 2026-03-31
categories: [reinsurance]
tags: [reinforcement-learning, PPO, VAE, reinsurance, XL, quota-share, solvency-ii, SCR, capital-modelling, Dong-Finlay-2025, deep-learning, tail-risk, python, stable-baselines3]
description: "A January 2025 paper by Dong & Finlay (arXiv:2501.06404) combines a Variational Autoencoder with Proximal Policy Optimisation to dynamically adjust excess of loss and quota share treaty parameters. 11.5% better surplus than next-best benchmark (Monte Carlo). Ruin probability zero. VAE tail fit: KS 0.6. We work through what the architecture actually does, where it breaks, and what a real UK implementation would need."
author: burning-cost
---

We covered the static version of this problem in [a previous post](https://burning-cost.github.io/research/2026/03/28/optimal-reinsurance-structuring-python-theory-practice-gap.html) — theory tells you XL layers are optimal under VaR, most pricing teams set retentions by feel, and the Python tooling is thin. The natural follow-up question is whether machine learning can do any better than a grid search over attachment points and limits.

A paper by Stella Dong (UC Davis) and James Finlay (Wharton), submitted to arXiv in January 2025 (arXiv:2501.06404, revised March 2026), tries exactly this. They combine a Variational Autoencoder with a Proximal Policy Optimisation agent to create a system that watches the insurer's surplus, reads the claim history, and dynamically adjusts treaty parameters in real time rather than waiting for annual renewal. The headline results are striking. The deeper you go into the paper, the more complicated the story gets.

---

## Why static optimisation falls short

The classical approach to reinsurance optimisation — which is what most UK insurers actually do, whether manually or with tools like ReMetrica — treats treaty structure as an annual decision. You pick an attachment and limit, sign a treaty, and that structure governs the year. If October turns out catastrophically worse than your return period assumed, you have no mechanism to respond until the next renewal.

This is not just an aesthetic limitation. It creates three specific problems:

**Regime shifts.** A motor book's frequency can move substantially within a year — think the inflation spike in repair costs from 2021 to 2023, where claim severity on UK motor physical damage shifted faster than most annual treaty structures could accommodate. If your retention was set when Thatcham-rated repairs cost £1,800, a regime where they cost £3,200 materially changes your exposed layer.

**Multi-year capital dynamics.** Solvency II SCR is a one-year measure, but the ORSA requires forward-looking stress testing over three to five years. The surplus an insurer carries at the start of year two depends on year one's net losses, which depend on the reinsurance structure in year one. A static annual optimisation ignores this path dependency.

**High-dimensional treaty space.** A real reinsurance programme has multiple XL layers, potentially a quota share, and aggregate protections. The joint parameter space — retentions, limits, and cession rates across five or more layers — is too large for a grid search to explore thoroughly, especially once you allow parameters to vary over time.

Reinforcement learning is the natural framework for this type of problem: sequential decisions, uncertain environment, reward signal tied to multi-period outcomes. The question is whether it can be made to work on real insurance data.

---

## The architecture: what Dong & Finlay actually built

The paper proposes a two-stage system. The first stage learns the claim distribution. The second stage learns the reinsurance policy given that distribution.

### Stage 1: VAE for claim distribution

The Variational Autoencoder takes claim data as input and learns a compressed latent representation from which new claim scenarios can be sampled. The innovation here is the loss function. Standard VAEs minimise reconstruction error plus a KL divergence term that regularises the latent space toward a unit Gaussian:

```
ℒ_VAE = ℒ_rec + β · D_KL(q_φ(z|x) ‖ p(z))
```

The problem with applying a standard VAE to insurance claims is that the reconstruction loss treats all claims equally. A model that fits the bulk of the distribution well but gets the large claims wrong will still score reasonably on standard reconstruction metrics. For reinsurance, this is precisely backwards — the entire economic rationale for XL protection is those large claims. The paper introduces a tail-weighted reconstruction penalty:

```
w(x) = 1 + ω · 𝟙{x > q_τ(X)}
```

Claims above the τ-th quantile (say, the 95th percentile) receive upweighted reconstruction penalty by the scalar ω. The β parameter is annealed upward during training: early epochs prioritise fitting the data, later epochs regularise toward the prior.

### Stage 2: PPO agent for treaty decisions

The reinforcement learning component is a Proximal Policy Optimisation agent. PPO is the current workhorse of continuous-action RL — it is more stable than vanilla policy gradient methods and more sample-efficient than trust region approaches. If you have used Stable-Baselines3 for any RL project, you have used PPO.

The agent's state at each timestep includes: current surplus, observed claim history, and the current treaty parameters across K layers. The action space is the set of adjustments to those parameters — changes to retention rates, attachment points, and detachment points for each XL layer. With K=5 layers and three adjustable parameters per layer, the action space is 15-dimensional.

The reward function is:

```
r(s_t, a_t) = ΔS_t − η · Premium_t − λ_ruin · 𝟙{S_t < 0} − κ · T̂ail_t
```

The primary signal is surplus growth (ΔS_t). The agent is penalised for reinsurance cost (η · Premium_t), for entering negative surplus (λ_ruin), and for tail exposure measured on the simulated loss distribution (κ · T̂ail_t). A terminal bonus rewards survival.

The agent trains against scenarios generated by the VAE rather than historical data directly. This is important: the VAE functions as a simulation engine, giving the PPO agent an essentially unlimited supply of synthetic claim scenarios to learn from.

---

## The results

The paper tests the system on a synthetic 10-year horizon with Lognormal(μ=3.5, σ=1.0) claim severities and Poisson(λ=10) frequency. Initial surplus is $20,000. The budget cap is $150,000.

The main comparison table (Table 4) reports:

| Method | Final Surplus | Ruin Probability | Compute (s) |
|---|---|---|---|
| Dynamic Programming | $12,488 | 0.0% | 7.96 |
| Monte Carlo | $12,803 | 0.0% | 414.27 |
| NSGA-II (genetic algorithm) | $12,467 | 0.0% | 8.52 |
| **VAE + PPO** | **$14,281** | **0.0%** | **7.92** |

An 11.5% surplus improvement over the next-best method (Monte Carlo at $12,803), achieved in less time than dynamic programming. Out-of-sample, the mean surplus is $16,687 with 0% ruin probability.

On the numbers, this looks impressive. We have some reservations.

---

## Where the paper struggles

### The tail problem is fundamental

The VAE's tail-weighted loss function is a reasonable idea, but the KS statistics reported for the fitted model are poor:

- Lognormal claims: KS = 0.59
- Pareto claims: KS = 0.62
- Combined Lognormal-Pareto: KS = 0.44

A KS statistic of 0.6 indicates substantial distributional mismatch. For context, a well-fitted copula model on insurance data would typically achieve KS below 0.1 on validation data. The authors acknowledge this directly — the VAE "systematically underestimates large claims" because the β regularisation compresses extreme values to maintain bulk reconstruction accuracy. This is the fundamental tension in VAEs applied to heavy-tailed data: the latent space has finite capacity, and the KL penalty pushes the model toward smoother, more Gaussian representations.

The consequence the authors also acknowledge: "capital requirements estimated using such a model may be biased downward, potentially leading to solvency shortfalls." A generative model that systematically underestimates the tail is training the PPO agent to be too aggressive. The 11.5% surplus outperformance over Monte Carlo might partly reflect the agent taking risks the VAE fails to simulate accurately.

### Reward function weights are missing

The reward function has four scalar hyperparameters: η, λ_ruin, κ, and ρ. None of their values are reported anywhere in the paper. The experimental results are therefore not reproducible. This is a serious omission.

### Architecture is unspecified

The VAE's layer counts, hidden dimensions, latent dimension, activation functions, and optimiser choice are absent. Two research groups attempting to replicate this would likely build materially different models.

### Real data: none

All results are on synthetic data from a single homogeneous claim stream. The paper's introduction describes the VAE's ability to "learn complex dependencies, including joint tail behaviour, across lines of business" — but no multi-line model is actually built or tested. The "multi-line, multi-year" framing is conceptual only.

### No code, no open-source release

As of March 2026, no public repository exists for this paper or for Dong and Finlay's two related papers on CVaR-constrained RL (arXiv:2504.09396) and ClauseLens (arXiv:2510.08429).

---

## The Solvency II angle

The paper references "99.5% VaR under Solvency II" and "ORSA practices" in passing, but no SCR calculation is actually performed. The ruin constraint in the reward function is a multi-year path probability — the probability of surplus going negative at any point in the 10-year simulation. This is structurally different from the Solvency II SCR, which is the 1-year 99.5% Value at Risk of Basic Own Funds.

For UK insurers post-Brexit, the relevant framework is now Solvency UK (PRA, effective December 2024). The standard formula sigma adjustments for the non-life premium and reserve risk sub-module remain unchanged at the segment level. The critical implication: standard formula SCR is insensitive to the specific attachment and limit on an XL treaty. Adjusting (α, a, b) dynamically does not change the standard formula capital charge.

For internal model firms — which is where this kind of approach would need to sit — the RL policy could plausibly integrate with the net loss simulation. But an internal model with a VAE tail fit of KS=0.6 would not pass PRA model validation. The 99.5th percentile of the net loss distribution is precisely what the validation will scrutinise, and systematic tail underestimation is the specific failure mode that review committees look for.

The most defensible near-term use case for RL in this context is ORSA scenario analysis. ORSA requires multi-year forward-looking stress testing, which is exactly the temporal structure the Dong-Finlay framework addresses. An RL agent trained on real loss data and constrained to achieve ruin probability below a target threshold could generate a richer set of ORSA scenarios than deterministic management actions. But this requires building the real version of the system — real data, proper tail modelling, documented architecture, validated reward weights.

---

## Comparison: ClauseLens and harder constraints

A related paper, ClauseLens (arXiv:2510.08429), takes a different approach to the constraint problem. Where Dong and Finlay implement their ruin constraint as a soft penalty in the reward function (calibrated post-hoc so the trained policy meets the target), ClauseLens uses a Lagrangian dual mechanism:

```
λ_k ← [λ_k + η(d̄_k − ε_k)]₊
```

Dual variables are updated during training via gradient ascent on the constraint violation. This is a genuinely harder constraint: the policy is penalised in proportion to how much it exceeds the risk tolerance during learning, not just at evaluation. For regulatory purposes, Lagrangian constrained RL is considerably more defensible than penalty tuning, because you can demonstrate that the constraint was active during training rather than hoping the soft penalty happened to produce a compliant policy.

ClauseLens uses CVaR as its risk measure rather than a ruin indicator, which also aligns better with standard actuarial risk metrics.

Neither paper has an open-source implementation.

---

## What a deployable UK implementation would actually need

There is no Python package that does end-to-end reinsurance optimisation via RL. The pieces that exist:

- **`gemact`** (GPitt71/gemact-code): The most complete open-source reinsurance computation library. Collective risk models, aggregate distributions via FFT, quota share and XL mechanics. Peer-reviewed in Annals of Actuarial Science (2023). Handles the loss distribution input — not the optimisation.
- **`stable-baselines3`**: Standard PPO implementation. The Dong-Finlay framework would use this for the RL component.
- **`gymnasium`**: Environment wrapper for RL. The Dong-Finlay approach requires a custom `gymnasium.Env` that simulates treaty mechanics and computes rewards.

The missing piece is a Gymnasium environment for reinsurance treaty optimisation. No public implementation exists. Building one requires encoding: the net retained loss formula across K layers, the premium cost as a function of treaty structure and market conditions, the surplus dynamics over time, and a reward function with documented weights.

A minimal stack for someone wanting to experiment:

1. `gemact` or custom simulation → net loss distribution per treaty structure
2. Custom `gymnasium.Env` → treaty mechanics + reward function with explicit η, λ, κ, ρ
3. `stable-baselines3` PPO → policy learning
4. Held-out simulation against real historical losses → validation

Steps 1, 3, and 4 are tractable today. Step 2 is the gap.

---

## Our assessment

The Dong-Finlay paper is a genuine conceptual contribution. The problem setup is right: reinsurance optimisation is multi-period, the state space includes surplus dynamics and claim history, and PPO is a reasonable algorithm for continuous action spaces of this size. The tail-weighted VAE loss is a sensible idea, even if the implementation does not yet deliver adequate tail fit.

But the KS=0.6 tail fit is not a minor technical debt. It is a fundamental problem for the specific application. Reinsurance protection exists to cover the tail. A generative model that systematically smooths the tail trains an agent that will behave incorrectly on the scenarios that matter most. The 11.5% surplus improvement over Monte Carlo is measured on a stylised single-line simulation with reward weights that are never disclosed. That is not a result a UK actuary can take to a board capital committee.

Our view: this is a research direction worth watching, not a recipe to follow. The combination of RL and generative claim simulation is conceptually correct. The specific implementation is not production-ready. A credible UK deployment would need real loss data, a properly specified tail model (EVT augmentation of the VAE, or a separate GPD fit for the tail), publicly documented reward weights, and a regulatory integration that engages with the actual SCR calculation rather than a multi-year ruin surrogate.

On the current evidence, RL beats grid search for reinsurance optimisation only when the action space is genuinely high-dimensional (many layers, multiple treaty types simultaneously), the time horizon is multi-year, and you have enough historical data to train or validate a generative model. For a UK personal lines insurer with a standard two-layer XL programme, grid search remains adequate and considerably more auditable.

---

**Reference:** Dong, S.C. and Finlay, J.R. (2025) "A Hybrid Framework for Reinsurance Optimization: Integrating Generative Models and Reinforcement Learning." arXiv:2501.06404 \[q-fin.RM\], revised March 2026.

**Related:** Dong & Finlay (arXiv:2504.09396) — CVaR-constrained RL for insurance reserving. ClauseLens (arXiv:2510.08429) — Lagrangian-constrained PPO for treaty structuring.
