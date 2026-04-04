---
layout: post
title: "Your NCD Relativities Are Wrong, and the Maths Now Tells You How Wrong"
date: 2026-04-02
author: burning-cost
categories: [pricing, motor, techniques]
tags: [BMS, NCD, bonus-malus, underreporting, hunger-for-bonus, game-theory, Nash-equilibrium, UK-motor, Lemaire, Liang, arXiv-2601.12655, arXiv-2601.07655, GLM, Poisson, frequency-model, motor-pricing, pricing-bias, claim-suppression, PDMP, competitive-equilibrium, actuarial]
description: "Two January 2026 arXiv papers formalise what motor actuaries have always known informally: NCD creates rational incentives to suppress small claims, and the GLM you're using to set NCD relativities is fit on data poisoned by that suppression. Liang et al. (arXiv:2601.12655) now give the Nash equilibrium under competition. The practical implication is a frequency bias that runs to 35% at mid-ladder NCD classes."
math: true
---

Every UK motor pricing team has an NCD ladder. Every UK motor pricing team fits a Poisson GLM with NCD as a rating factor. And almost every UK motor pricing team is using claim frequency data that systematically understates the true risk at high NCD classes — because the policyholders holding those classes are making rational decisions not to claim.

This is not a new observation. Jean Lemaire formalised it in 1977 under the name "La Soif du Bonus" — the hunger for bonus. The dynamic programming is straightforward: if the discounted NPV of future premium increases from a claim exceeds the claim amount, the rational policyholder self-insures. The threshold is higher at better NCD classes because the premium penalty from a step-back is proportionally larger. A 9-year NCD holder dropping to 7 years loses far more in absolute premium terms than a 2-year holder dropping to 0.

What is new — and what two January 2026 arXiv papers establish — is a proper formalisation of what happens when insurer competition is added to this picture, and a rigorous proof that the underreporting behaviour and the competitive premiums jointly converge to a Nash equilibrium. This equilibrium is not the one your GLM finds. The difference is not trivial.

---

## What the GLM is actually measuring

The standard industry workflow: take observed claim counts, fit a Poisson frequency model with NCD as a categorical or ordered factor, read off the NCD relativities, apply them to the base rate. The resulting relativities describe the frequency of *reported* claims by NCD class.

Reported and actual are not the same thing. If the retention threshold at 5-year NCD is approximately £280 — which is roughly where it falls on a £1,000 base premium with standard step-back rules, as we calculate below — and the severity distribution implies around 35% of claims fall below that threshold, then observed frequency at 5-year NCD is approximately 65% of the true underlying frequency. The GLM relativity for 5-year NCD is calibrated from data that understates the risk by a third.

The direction of the bias matters. Underreporting is most severe at mid-ladder NCD classes (3–6 years), where the absolute premium penalty from a step-back is largest. At the top of the ladder (9-year NCD, with discounts already capped at 70%), the additional penalty from a step-back is smaller in absolute terms, so the retention threshold is actually lower. At the bottom (0–1 year NCD), there is little NCD to protect and the threshold approaches zero. The shape is an inverted U: the worst underreporting sits in the middle of the ladder, which is also where the majority of your exposure sits.

The practical consequence: lower-NCD policyholders are cross-subsidising higher-NCD policyholders. The 5-year NCD holder's observed frequency looks like 65% of a new driver's, so they are priced accordingly. Their true risk is closer to 90%. The premium gap is being absorbed somewhere — and it is absorbed by the new-to-insurance drivers who are priced from observed data that has no suppression bias.

---

## Calculating the retention thresholds

The conceptual framework is simple. For a policyholder in NCD class $j$ paying a base premium $B$ with discount $D_j$, one fault claim steps them back to class $j-k$ (typically $k=2$ in the UK). The present value of the penalty over a rebuilding horizon of $T$ years is approximately:

$$\text{PV(penalty)} \approx B \cdot (D_j - D_{j-k}) \cdot \sum_{t=1}^{T} \delta^t$$

The policyholder should suppress the claim if the loss $Y$ satisfies $Y < \text{PV(penalty)}$.

Three illustrative calculations on a £1,000 base premium, standard UK discount schedule, 2-year step-back, and a discount rate of 5%:

| NCD class | Discount | Premium | Step-back to | Post-claim premium | Annual increase | PV(penalty) |
|-----------|----------|---------|--------------|-------------------|-----------------|-------------|
| 9-yr | 70% | £300 | 7-yr (65%) | £350 | £50/yr | ~£93 |
| 5-yr | 65% | £350 | 3-yr (50%) | £500 | £150/yr | ~£278 |
| 3-yr | 50% | £500 | 1-yr (30%) | £700 | £200/yr | ~£370 |

The 3-year and 5-year holders have the highest rational retention thresholds in absolute pound terms. If, say, 35% of claims in those NCD classes fall below £278–370 — plausible for a portfolio with a typical mix of small own-damage and third-party property claims — then observed frequency is running at 65% of true frequency. That is material.

These are simplified single-period calculations. Lemaire's full dynamic programming algorithm, calibrated to the actual ladder structure and the severity distribution, produces somewhat different numbers. The direction is unambiguous.

---

## The new literature: competition makes it worse

Jean Lemaire's analysis assumed a single monopolist insurer. That is an adequate foundation for the behavioural decision, but it does not capture the UK market: motor insurance in the UK is aggressively competitive, price-comparison-driven, and the NCD relativities of one insurer are implicitly constrained by market structure.

Liang, Zhang, Zhou and Zou (arXiv:2601.12655, submitted 19 January 2026) are the first to model strategic underreporting under *oligopolistic competition*. Their paper does two things:

1. For any fixed pair of insurer premiums, it characterises the policyholders' unique optimal reporting strategy — a barrier strategy where a loss is reported if and only if it exceeds a threshold $b^*_{n}$ that depends on the current NCD class and the premium pair.

2. It proves that, under regularity conditions on the loss density, Nash equilibrium premium strategies $(θ^*_1, θ^*_2)$ exist, and characterises how they depend on price sensitivity and brand preference parameters.

The key symmetry result (Remark 3.1 of the paper) is that $b^*_{n,1} = b^*_{n,2}$ — the reporting threshold is the same regardless of which insurer the policyholder is with. What matters is the current NCD class and the weighted-average premium in the market, not which company holds the policy. This considerably simplifies the equilibrium analysis.

The closed-form barrier under the two-class tractable case is:

$$b^*(θ_1, θ_2) = \delta(\kappa - 1) \cdot \left[ θ_1 \cdot \eta(θ_1 - θ_2) + θ_2 \cdot (1 - \eta(θ_1 - θ_2)) \right]$$

where $\delta$ is the discount factor, $\kappa$ is the penalty ratio between premium classes (typically 1.2–1.5 in real BMS), and $\eta$ is the policyholder's switching probability as a function of the premium differential. The threshold is the probability-weighted average of both insurers' Class 1 premiums, scaled by the discounted premium increase from a downgrade. Mechanically: the more competitive the market, the lower the effective premium in the average, and the lower the rational retention threshold. Competition slightly *reduces* suppression at the margin — policyholders have lower premiums to protect.

The Nash equilibrium numerical base case (p₀ = 0.9, Gamma severity with mean around £117, $\kappa = 1.25$, $\delta = 0.97$, price-sensitivity $k_1 = 0.015$) produces equilibrium premiums of approximately £35.83 and £33.45 for the preferred and less-preferred insurer — a 7% gap. Higher price sensitivity compresses the gap and drives both premiums down. Higher brand preference amplifies the gap. These are stylised, but the qualitative direction maps cleanly onto the UK PCW market: brands with higher recognition and retention charge meaningfully more, and the competition premium gap is in the low single digits of percentage points.

---

## Why the naive GLM is not Nash-optimal

The standard UK pricing process — fit GLM, update NCD relativities, reprice — has a feedback loop with no guaranteed convergence property. The premiums set determine the rational retention thresholds. The retention thresholds determine the observed claim frequencies. The observed frequencies calibrate next year's premiums. There is no reason this process converges to the Liang et al. Nash equilibrium. It converges to whatever fixed point the annual refit process finds, which may be a stable equilibrium, an unstable one, or a drift.

The Liang et al. equilibrium requires a three-way fixed point: premiums → reporting barrier → stationary distribution of policyholders across NCD classes → expected profit → best-response premiums. The stationary distribution solves $(I - T^T)\pi = 0$ where $T$ is the Markov transition matrix induced by the optimal barrier. The transition matrix depends on the loss CDF evaluated at $b^*$, which is itself a function of premiums. This is the structure the GLM refit loop does not capture.

The competitive distortion is subtle but important. If one insurer corrects for underreporting bias and another does not, the uncorrecting insurer will underprice high-NCD policyholders relative to their true risk. Those policyholders — who also happen to be the ones best at suppressing claims — will disproportionately end up on the cheaper book. The correcting insurer loses them on price and ends up with an adversely selected residual. The rational response, if you cannot change the market, is not to correct. This is the adverse selection trap that keeps the industry at the biased equilibrium.

The complementary paper, arXiv:2601.07655, adds intra-period dynamics in continuous time using Piecewise Deterministic Markov Processes. Its key result on barrier dynamics: the optimal threshold *decreases monotonically as contract maturity approaches*. Near expiry, even a modest claim is worth reporting — the premium penalty for being demoted will last less time, so the cost is lower. This is a within-year effect that sits on top of the cross-year competition dynamics in Liang et al. Taken together, the two papers bracket the problem: competitive equilibrium on the vertical axis, temporal dynamics on the horizontal.

---

## What a corrected approach looks like in practice

Neither paper requires a full Nash equilibrium computation to be useful. The practical benefit for a UK pricing team comes in four steps, the first three of which require only internal data.

**Step 1: Implement Lemaire's algorithm for your own ladder.** The dynamic programming is around 50 lines of NumPy. Inputs: your discount schedule, step-back rules, average base premium by segment, and your fitted severity distribution. Output: a retention table $\{b^*_n\}$ for each NCD class. This is one day's work.

**Step 2: Estimate the censoring probability per class.** If you have a fitted Gamma severity — which you should, from your existing model — compute $p_n = P(Y > b^*_n)$ for each class. This is a one-liner once you have the threshold table.

**Step 3: Compute corrected frequency estimates.** Divide observed frequency by $p_n$ at each NCD class: $\hat{\lambda}_n = \lambda^{\text{obs}}_n / p_n$. Compare the corrected relativities to your current GLM output. If the corrected relativities at 3–5 year NCD are 10–20% higher than your current ones, the cross-subsidy is real and quantified.

**Step 4 (optional, requires competitor data): Nash equilibrium iteration.** Use market-level competitor premium schedules — available from PCW aggregator feeds — to run the fixed-point iteration in Liang et al. This is the full competitive equilibrium computation. It is not achievable with internal data alone, but Steps 1–3 are.

One data asset is particularly valuable but often overlooked: FNOL-captured claims that were subsequently withdrawn without payment. These are the self-settlements — the policyholder called up, thought about it, and decided not to proceed. Their claim amount (if your FNOL system captures it) directly estimates the retention threshold in practice. High FNOL withdrawal rates at specific NCD classes are the empirical fingerprint of the hunger for bonus effect. If your FNOL withdrawal rate at 5-year NCD is 15% and at 0-year NCD it is 3%, the difference is almost certainly suppression.

---

## The UK regulatory overlay

There is no FCA requirement to correct for underreporting bias in frequency models. The regulator has not issued guidance on this, and Consumer Duty's fair value requirement does not obviously mandate a correction — the pricing methodology is the firm's actuarial responsibility.

That said, there are two places where this matters for regulatory purposes.

Protected NCD is the more immediate one. If the PNCD loading does not reflect the actual statistical reduction in claims liability given a fault incident — because the baseline frequency model used to price it is itself biased — then the product's fair value case rests on inaccurate actuarial foundations. No published FCA guidance addresses this specifically, but it is the kind of gap that emerges in thematic reviews.

GIPP (PS21/5) removed price walking but applies to the total quoted premium. There is no constraint on NCD ladder design per se. However, if your NCD structure systematically overprices low-NCD policyholders to cross-subsidise the high-NCD group, and Consumer Duty requires you to demonstrate fair value to all customer segments, you are implicitly being asked to defend that cross-subsidy. The corrected relativities give you the data to understand whether it exists.

---

## What remains open

The Liang et al. Nash equilibrium is proved only for $N = 2$ NCD classes. The UK standard is 9–10 classes; some insurers extend to 15. The barrier characterisation (that a threshold strategy is uniquely optimal) holds for general $N$, but the equilibrium existence proof uses the two-class tractability. Numerical fixed-point methods would need to fill this gap for real implementation.

No published paper has estimated the hunger-for-bonus thresholds for the UK motor market specifically using UK data. The calibration benchmarks available are Belgian and French, with different ladder structures and premium levels. A UK insurer with access to individual-level NCD history and FNOL data could do this calibration. As far as we know, none has published results.

The telematics interaction is also unmodelled. A telematics device may give the insurer independent evidence of an incident — potentially before the policyholder decides whether to claim. If the insurer knows about the incident, the policyholder cannot suppress it. This fundamentally changes the game-theoretic structure for telematics policyholders, reducing the effective retention threshold toward zero for detectable incidents. The literature has nothing on this.

---

## The bottom line

The GLM NCD relativity for a 5-year NCD class is fit on frequency data that understates the true underlying risk by somewhere between 25% and 40%, depending on your severity distribution and premium levels. This is not a modelling failure — it is a rational response to the incentive structure your NCD ladder creates. The two 2026 papers formalise the mechanism and, for the first time, prove what the competitive equilibrium looks like when policyholders and insurers are both responding optimally.

Steps 1–3 above are achievable with internal data in less than a week of implementation. The corrected relativities will not transform your book overnight — the competitive equilibrium problem ensures that unilateral correction creates adverse selection pressure. But knowing the size of the bias is the first step, and right now, most UK pricing teams do not know it.

---

*Primary papers: Liang, Z., Zhang, J., Zhou, Z., Zou, B. (2026). "Optimal Underreporting and Competitive Equilibrium." arXiv:2601.12655. And: author names unconfirmed (2026). "To report or not to report: Optimal claim reporting in a bonus-malus system." arXiv:2601.07655 (authors not verified at time of publication).*

*Classical foundation: Lemaire, J. (1977). "La Soif du Bonus." ASTIN Bulletin 9(1-2):181–190. The 1995 monograph (Kluwer) remains the definitive treatment.*

---

## Related posts

- [Applying Bonus-Malus to Driving Behaviour, Not Just Claims](/2026/04/01/weekly-dynamic-telematics-bms-bonus-malus-driving-behaviour/) — the telematics extension of BMS, which changes the game-theoretic structure for policyholders with black-box devices
- [Why Your Reinsurer Is Never Offering You a Fair Deal](/2026/04/02/reinsurer-never-fair-deal-game-theory-treaty-pricing/) — the Boonen/Ghossoub result on Stackelberg inefficiency in treaty pricing; same game-theoretic framework, different application
