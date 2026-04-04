---
layout: post
title: "Your Policyholders Are Playing a Game with Your NCD Ladder"
date: 2026-03-31
categories: [techniques, pricing]
tags: [ncd, bms, bonus-malus, underreporting, game-theory, nash-equilibrium, uk-motor, frequency-model, glm, lemaire, claim-suppression, pricing-bias, motor-insurance, personal-lines]
description: "Policyholders with good NCD rationally choose not to report small claims. Your frequency model is trained on that suppressed data. Two January 2026 papers formalise what this means for pricing, and the answer is worse than most teams assume."
math: true
author: burning-cost
---

Your 5-year NCD customer has a small scrape. The repair is £250. They do not call you. They fix it themselves, quietly, and protect their 65% discount.

This is not a rounding error. It is a rational, optimal decision — one that your pricing model has no adjustment for.

The result is that your NCD relativities are estimated from claim data that is systematically missing small losses at high discount levels. The frequency model you rely on is fit on suppressed data. The NCD coefficients absorb the suppression signal and you call it good. Every UK motor pricing team we are aware of does this. Two papers published in January 2026 make clear how wrong it is.

---

## The hunger for bonus

The phenomenon has a name in the actuarial literature: *hunger for bonus* (Jean Lemaire coined it in 1977 — "La Soif du Bonus", ASTIN Bulletin). The concept is simple. Under a bonus-malus system, reporting a fault claim triggers a premium increase that lasts for years. If that future cost exceeds the claim amount, a rational policyholder does not claim.

The calculation is not difficult. Take a 5-year NCD holder paying £350 on a £1,000 base (65% discount). A fault claim steps them back to 3-year NCD (50% discount, £500). That is £150 per year more in premium. Rebuilding to 5-year NCD takes two years. Discounting at ~5%:

```
PV(penalty) ≈ £150 × (0.95 + 0.90) ≈ £278
```

The rational threshold: do not claim for losses below £278. Self-insure anything up to roughly three months of the premium saving.

Work through the same calculation at different points on the UK NCD ladder.

| NCD years | Typical premium (£1,000 base) | Step-back to | Annual increase | Years to rebuild | Rational threshold (approx.) |
|-----------|-------------------------------|--------------|-----------------|------------------|------------------------------|
| 9 | £300 (70%) | 7 yr (65%) | £50 | 2 | ~£93 |
| 5 | £350 (65%) | 3 yr (50%) | £150 | 2 | ~£278 |
| 3 | £500 (50%) | 1 yr (30%) | £200 | 2 | ~£370 |
| 1 | £700 (30%) | 0 yr (0%) | £300 | 1 | ~£285 |

The pattern is not monotone. Thresholds peak in the middle of the ladder — around the 3-to-5-year range — and fall at both extremes. At 9-year NCD the absolute premium difference is small (the 70% cap means a step-back to 65% is only £50/year). At 1-year NCD the annual increase is large but the rebuild time is short — one claim-free year gets you back to 1-year NCD — so the present value of the penalty is lower than at mid-ladder.

The 3-year NCD holder faces the highest rational retention threshold: roughly £370 on a £1,000 base premium. That is an amount large enough to cover most minor own-damage incidents.

Lemaire's full dynamic programming treatment accounts for the stochastic claim history during the rebuilding period and multiple future periods, so the real numbers differ from these back-of-envelope figures. But the direction is clear, and the magnitudes are not trivial. At 5-year NCD, the rational threshold exceeds a quarter of the annual base premium.

---

## What this does to your frequency model

Here is the problem as a pricing actuary should see it.

True claim frequency at NCD class *n* is λ_n. But a policyholder in class *n* reports a claim only if its size exceeds the retention threshold b*_n. So observed frequency is:

```
λ_obs_n = λ_n × P(Y > b*_n)
```

If the severity distribution is Gamma with mean £1,500 and P(Y > £278) ≈ 0.65 for a 5-year NCD holder, then observed frequency is roughly 65% of true frequency. A 35% underestimate.

When you fit a Poisson GLM on reported claims with NCD as a categorical factor, you are fitting on λ_obs, not λ. The NCD coefficients you get back are descriptively accurate — they correctly predict future reported claim counts conditional on current NCD class. But they are structurally wrong as a measure of true risk. The 5-year NCD coefficient is too low because it has been estimated from data where 35% of claims never happened, from the model's perspective.

The cross-subsidy runs from lower-NCD classes (whose observed frequency is closer to their true frequency — a 0-year NCD holder has a rational threshold near £0, because they have nothing to protect) toward the mid-ladder classes where suppression is heaviest. Your current 3-to-5-year NCD customers are likely undercharged relative to their actual risk. Drivers at the extremes — new to insurance or just past a claim — are cross-subsidising them.

The bias compounds another way: because small claims are suppressed, the claims that are reported skew toward larger losses. This inflates your observed mean severity at mid-ladder NCD classes. If your frequency and severity models interact, the severity distortion bleeds back.

---

## The competitive layer: Liang et al. (2026)

This would be a contained, if underappreciated, calibration problem if it happened in isolation. But Liang, Zhang, Zhou, and Zou (arXiv:2601.12655, January 2026) show it is worse than that — because insurer competition turns it into a game with an equilibrium that most pricing teams are not solving for.

Their paper is the first analysis of strategic underreporting under oligopolistic competition. Prior work (including Lemaire's) assumed a single insurer. Liang et al. add a second insurer and a policyholders' switching choice, and show that the system has a Nash equilibrium with three interlocking conditions: each insurer's premiums are profit-maximising given the other's premiums; policyholders' reporting strategy is optimal given both insurers' premiums; and the stationary distribution of policyholders across NCD classes is consistent with the reporting strategy.

The key theorem (Theorem 3.1) establishes that the optimal reporting threshold is insurer-independent — it depends on the policyholder's NCD class and the premium levels, but not on which specific insurer they are with. A 5-year NCD holder faces the same rational threshold at Insurer A as at Insurer B, as long as both charge similar premiums at that class. This simplifies the equilibrium considerably.

In their two-class numerical example (p₀ = 0.9, loss ~ Gamma(1.2, 0.0085), penalty ratio κ = 1.25, discount factor δ = 0.97), the Nash equilibrium has the more attractive insurer charging approximately 7% more than its competitor: θ*₁ ≈ 35.83, θ*₂ ≈ 33.45. The preferred insurer can exploit its competitive position in exactly the way you would expect: brand loyalty creates pricing power, and price-sensitivity governs how far premiums can diverge before the gap closes.

The practical implication is harder to digest. The standard UK workflow — fit GLM on reported claims, update relativities, repeat — does not converge to the Nash equilibrium. It converges to whatever fixed point the iterative refit finds, which is not the same thing. If you are not explicitly solving the three-way fixed point (premiums → reporting threshold → stationary distribution → expected profit → best-response premiums), your NCD rates are not in equilibrium. They are, in Liang et al.'s framework, a locally stable approximation to the wrong answer.

There is also an adverse selection dynamic. If one insurer corrects for the suppression bias and another does not, the correcting insurer prices mid-ladder NCD correctly (higher) while the naive insurer prices it cheap. Mid-ladder customers migrate to the cheaper book. The correcting insurer loses the good-looking business and keeps a portfolio that selected adversely on every other variable. The market dynamics push both insurers toward the naive equilibrium unless correction is adopted broadly.

---

## When the contract nears its end: arXiv:2601.07655

A companion paper from a different author group (arXiv:2601.07655, January 2026) approaches the same problem from a continuous-time angle and adds a result that Liang et al. do not cover: how the reporting threshold changes *over the life of the policy*, not just at steady state.

The model uses Piecewise Deterministic Markov Processes and solves the Hamilton-Jacobi-Bellman system to find the optimal barrier b(t) as a function of remaining contract time. The core result: the retention threshold decreases as the policy approaches its expiry date. Near renewal, even small claims become worth reporting, because the NCD disadvantage lasts less time before the policyholder is repriced anyway. The paper proves the value function is the unique viscosity solution to the HJB system — technically the strongest result available for this class of problem.

The practical implication for UK motor is underappreciated. Mid-year claims have a higher rational threshold than end-of-year claims. If your claims development patterns show a spike in reported small losses in the fourth quarter of policy years, this is part of the explanation — and it is not noise, it is rational behaviour. Standard GLMs fit on policy-year claims data aggregate across policy ages and miss this intra-period variation entirely.

The two papers are complementary. Liang et al. gives you the competitive equilibrium — the right premiums to charge given that policyholders and competitors are both playing strategically. The second paper gives you the within-year barrier dynamics — the right shape of the reporting threshold over the contract term. A complete model would incorporate both.

---

## Why a flat underreporting loading is the wrong fix

The instinctive response, once you accept that reported frequencies are too low at mid-ladder NCD classes, is to apply a uniform upward correction — add 10% (or 15%, or 20%) to your NCD relativity across the board and move on. This is wrong, and it is worth being specific about why.

The suppression rate is not uniform across NCD classes. From the table above: the rational retention threshold is ~£93 at 9-year NCD, ~£278 at 5-year, ~£370 at 3-year, and ~£285 at 1-year. If your severity distribution is Gamma with mean £1,500, the implied reporting probabilities are approximately:

| NCD years | Retention threshold | P(Y > threshold) | Implied suppression |
|-----------|--------------------|--------------------|---------------------|
| 9 | ~£93 | ~0.93 | ~7% |
| 5 | ~£278 | ~0.65 | ~35% |
| 3 | ~£370 | ~0.57 | ~43% |
| 1 | ~£285 | ~0.64 | ~36% |

A flat 20% uplift overcorrects at 9-year NCD (true suppression ~7%), undercorrects at 3-year NCD (true suppression ~43%), and applies an uplift to 1-year NCD where suppression is meaningful but originates from a different position on the penalty gradient.

The pattern is genuinely non-monotone: the highest-NCD class has the lowest suppression, mid-ladder classes have the most, and the distribution of the severity model governs the precise shape. A flat loading applies the same correction to a class with 7% suppression as to one with 43% suppression. That is not a calibration improvement — it is a different shape of the wrong answer.

The correction must be class-by-class, derived from the retention threshold at each class and the relevant severity distribution. The suppression calculation is specific to own-damage versus third-party claims (the penalty calculation and the typical severity differ), to vehicle segment (severity distributions are not homogeneous), and to the insurer's specific NCD ladder. None of that heterogeneity can be captured with a single loading percentage.

---

## What the FCA has and has not said

GIPP (PS21/5, effective January 2022) eliminated price walking — the renewal premium for existing customers must be no higher than the equivalent new business price through the same channel. The FCA's 2025 evaluation paper (EP25/2) confirmed GIPP has held: the new-to-renewal gap has not returned, even as claims costs rose 49% between 2022 and 2024 (expected cost per policy from £92 to £138).

Consumer Duty, effective July 2023, requires firms to deliver fair value. The Price and Value outcome could, in principle, apply to NCD pricing methodology: if your NCD relativities are estimated from suppressed data and you are systematically undercharging mid-ladder customers while overcharging lower-NCD drivers, a fair-value argument is at least arguable. No FCA guidance exists on this point — the regulator has not addressed BMS calibration methodology, and there is no immediate regulatory obligation to correct for underreporting bias.

Protected NCD sits in more active regulatory territory. The FCA has signalled scrutiny of PNCD under Consumer Duty's Consumer Understanding outcome, because the product is routinely misunderstood: consumers believe it protects their total premium, not just the discount percentage. After a fault claim, the NCD percentage is preserved but the base premium can be repriced upward — which can erase much of the apparent protection at first renewal. This is a live compliance question, separate from the pricing calibration issue.

---

## What you can actually do about it

No off-the-shelf package handles the full correction pipeline — not in R, not in Python. The actuar package in R provides the credibility machinery; insurancerating has the GLM workflow with NCD as a factor; Loss Data Analytics (Chapter 12 at openacttexts.github.io) has worked BMS Markov chain examples. But the correction from observed to true frequency, calibrated to your own NCD ladder, has to be built.

We think a working first-pass is achievable in a few days.

**Step 1: Implement Lemaire's algorithm for your own ladder.** This is roughly 50 lines of NumPy. Inputs are your NCD discount percentages, your step-back rules, your average base premium by segment, and a severity distribution. Output is a retention table b*_n for each NCD class. The severity distribution you already have — use the Gamma parameters from your existing severity GLM, stratified by cover type.

**Step 2: Compute the censoring correction per class.** For each NCD class, p_n = P(Y > b*_n) under your severity distribution. Compare λ_obs / p_n to λ_obs. If the correction factors are within 5% across all classes, the bias is not material for your portfolio and you can stop. Our guess is they will not be.

**Step 3: Re-examine your NCD relativities against corrected frequencies.** You are not necessarily re-fitting the full GLM at this stage — you are asking whether the size of the bias is consistent with how much your current 5-year relativities differ from 0-year. If your observed 5-year relativity is 0.60 but the corrected frequency implies 0.68, that is meaningful. You are undercharging that segment by roughly 13%.

Steps 1–3 require nothing beyond internal data — your NCD ladder, your severity model, and your claims system. They quantify the first-order bias before you need to consider competitive dynamics.

**Step 4 (the full solution): Iterate toward Nash equilibrium.** Per Liang et al., premiums and reporting thresholds are jointly determined — each insurer's rates affect what thresholds are rational for policyholders, which affects the stationary distribution of policyholders, which affects expected profit, which feeds back into rate-setting. Solving this properly requires market-level competitor premium data at NCD class resolution. That means PCW aggregator data or an industry data-sharing arrangement — a larger project. But if the step 1–3 analysis shows the bias is large enough to matter commercially, the step 4 project becomes justifiable.

**Look at your FNOL withdrawal data.** Claims that are opened and then withdrawn — the policyholder called to register a claim, then settled it themselves — are your cleanest empirical signal for calibrating b*_n directly. If your claims system captures indicative reserves or loss estimates at FNOL, the size distribution of withdrawals at each NCD class directly estimates the retention threshold without any model assumption. This data almost certainly exists somewhere in your claims system. Most teams have never looked at it this way.

---

## The open questions

Liang et al. prove Nash equilibrium existence for two NCD classes. Most UK ladders have nine or more. The extension to N classes is conjectured but not proved, and numerical fixed-point approaches are needed for practical implementation.

The framework also assumes homogeneous policyholders. In practice, the interaction between a priori risk heterogeneity (the core pricing problem your GLM is trying to solve) and strategic reporting behaviour is unmodelled in the published literature. A high-frequency policyholder has a different reporting threshold calculation from a low-frequency one — their expected future premium is different, their rebuilding path is different. This is the problem Norberg's credibility approach partially addresses, but the underreporting correction has not been integrated into the Bayesian updating framework.

The empirical question is also open: no published estimate of the hunger-for-bonus threshold exists for UK motor using UK data. Abbring, Chiappori, and Zavadil (SSRN, 2008) used Dutch longitudinal data and found evidence consistent with strategic non-reporting, but the Dutch BMS is more punitive than the UK standard and the numbers do not transfer directly.

Finally, telematics complicates the picture in a way the literature has not addressed. A connected car policy may detect an impact event, meaning the policyholder cannot suppress it cleanly. If your telematics portfolio has different NCD distributions from your standard book, your NCD relativities should not be shared across the two.

---

## The bottom line

Policyholders with 3-to-5 years of NCD are rationally self-insuring losses up to several hundred pounds. Your observed claim frequency at those classes understates true frequency by a third or more. You are pricing those customers too cheaply relative to their actual risk, and your lower-NCD customers are funding the difference.

This is not a new finding — Lemaire documented it in 1977 — but the UK industry has not corrected for it, there is no regulatory requirement to do so, and no standard tool exists. Liang et al. (2026) show that the problem is structurally worse in a competitive market: the naive GLM re-fit does not converge to the right equilibrium.

The temptation is to apply a flat loading and call the problem addressed. Do not. The suppression rates vary from ~7% at 9-year NCD to ~43% at 3-year NCD. A flat correction applies the same number to both and creates a different pattern of mispricing. The correction needs to be class-by-class and severity-distribution-aware.

The full correction is not a one-week project. Estimating the size of the bias is. If you have a pricing model and a severity distribution, you can quantify whether your NCD relativities have a material bias problem in a few days of work. We think you should.

---

## Related reading

- [Protected NCD Is a Fair Value Problem Waiting to Happen](/2026/04/01/protected-ncd-consumer-duty-fair-value-pricing/) — how the underreporting literature interacts with PNCD pricing adequacy under Consumer Duty
- [Profit-Fairness Market Equilibria: Why Consumer Duty Audits the Wrong Unit](/2026/03/31/profit-fairness-market-equilibria-consumer-duty/) — the market-level fair value problem that firm-level Consumer Duty compliance cannot capture
