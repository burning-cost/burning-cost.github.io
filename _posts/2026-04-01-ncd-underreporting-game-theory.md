---
layout: post
title: "The Hunger for Bonus: How UK Motor NCD Pricing Gets the Frequency Wrong"
date: 2026-04-01
categories: [pricing, techniques]
tags: [ncd, bonus-malus, bms, underreporting, game-theory, nash-equilibrium, uk-motor, motor-pricing, claim-suppression, frequency-bias, liang-2026, lemaire, poisson-glm, credibility, arXiv-2601.12655, arXiv-2601.07655, pricing-bias, cross-subsidy, fnol, personal-lines]
description: "Two January 2026 arXiv preprints formalise what UK pricing teams have long intuited: observed claim frequency at high-NCD classes understates true frequency by 15–35%, because policyholders rationally suppress small claims. No UK insurer corrects for this. Here is what it costs them, and what to do about it."
math: true
author: burning-cost
---

Every UK motor pricing team fits a Poisson frequency GLM with NCD as a categorical factor. The NCD coefficients come out monotonically declining, broadly consistent with year-on-year observation. Nobody questions them too hard, because the data tells a coherent story: higher NCD, fewer claims.

The data is lying, but in a predictable way. Policyholders with high NCD are not reporting all their claims. They are making a rational financial calculation — the premium cost of claiming exceeds the benefit — and self-insuring small losses. The result: observed claim frequency at high-NCD classes understates true frequency, the GLM absorbs this suppression into the NCD coefficients, and policyholders with high NCD are systematically underpriced relative to their actual risk. Lower-NCD policyholders cross-subsidise them.

Jean Lemaire called this *la soif du bonus* in 1977. The hunger for bonus. It has been understood for fifty years. Two January 2026 arXiv preprints now give it a rigorous game-theoretic foundation — and close the loop between policyholder strategy and insurer equilibrium in a way that makes the practical implications harder to dismiss.

---

## What policyholders actually do

Start with the decision a policyholder faces when they have a small accident — a scraped bumper, a minor third-party ding. Should they claim?

The rational answer is: claim if and only if the claim amount exceeds the present value of the future premium penalty from a step-back on the NCD ladder.

For a 9-year NCD holder on a £1,000 base premium (70% discount, paying £300), a fault claim under standard UK rules steps them back two years to 7-year NCD. If 7-year NCD gives 65% — £350 — the extra cost is £50 per year, sustained for two years while they rebuild. At a discount rate of 5%:

$$\text{PV(penalty)} \approx £50 \times (0.95 + 0.90) \approx £92.50$$

They should not claim for anything below roughly £90–100. Scraped bumpers go unreported.

For a 5-year NCD holder (65% discount, paying £350), a step-back to 3-year (50%, paying £500) costs £150 per year for two years:

$$\text{PV(penalty)} \approx £150 \times (0.95 + 0.90) \approx £277.50$$

They self-insure everything below about £280. A fair amount of minor own damage disappears from the data.

For a 3-year NCD holder (50%, paying £500), stepping back to 1-year (30%, £700) costs £200 per year:

$$\text{PV(penalty)} \approx £200 \times (0.95 + 0.90) \approx £370$$

Below £370, rational not to claim.

These are simplified single-period calculations — Lemaire's full dynamic programming extends the analysis to account for the stochastic claims path during the rebuilding period, which shifts the thresholds somewhat. But the direction is clear, and it contains a non-obvious result: **the highest NCD classes do not have the highest suppression thresholds in absolute terms.** At the 9-year cap, the premium differential between adjacent NCD levels is compressed because many insurers stop adding material discount beyond year 7 or 8. Mid-ladder policyholders (3–6 years NCD) face the steepest absolute step-backs and therefore suppress the largest claims in absolute terms.

---

## Why this poisons the GLM

If the true claim frequency at 5-year NCD is $\lambda$, and policyholders suppress all claims below threshold $b^*_n$, then the probability that any given claim actually gets reported is:

$$p_n = P(Y > b^*_n) = 1 - F_{\text{Gamma}}(b^*_n;\, \alpha, \beta)$$

The observed frequency the GLM fits against is $\lambda^{\text{obs}}_n = \lambda_n \cdot p_n$, not $\lambda_n$.

With a retention threshold of £280 at 5-year NCD and a Gamma severity distribution with mean around £1,500 (reasonable for UK own-damage plus third-party property), $P(Y > £280) \approx 0.65$. The GLM is fitting to 65% of the true frequency at that NCD level — a 35% underestimate. At 9-year NCD with threshold ~£90 and $P(Y > £90) \approx 0.85$, the underestimate is around 15%. At 0-year NCD, where the step-back penalty is negligible and most policyholders are in the first year after a claim, underreporting is minimal — the observed frequency is close to true.

The NCD coefficient in the GLM will therefore understate the true frequency differential between 0-year and high-year NCD policyholders. High-NCD policyholders look cheaper than they are. They are charged accordingly.

---

## The 2026 formalisation

Lemaire (1977) established the optimal barrier strategy for a single policyholder under a given premium schedule. Two January 2026 arXiv preprints extend this in important directions.

**Liang, Zhang, Zhou, Zou (arXiv:2601.12655)** is the first published analysis of strategic underreporting in an *oligopolistic* market. Prior BMS game-theory work assumed a single monopolist; this paper adds insurer competition as a second strategic layer. The core result:

*Theorem 3.1.* For any fixed premium pair $(c^1, c^2)$ offered by two competing insurers, there exists a unique optimal barrier strategy $b^*_{n,i}$ such that a loss $Y$ is reported if and only if $Y > b^*_{n,i}$. Moreover, $b^*_{n,1} = b^*_{n,2}$ — the threshold is **insurer-independent**. It depends on the BMS class and the premium structure, not on which company the policyholder is currently with.

Under a two-class BMS with common penalty ratio $\kappa \in (1,2)$, the barrier has a closed form:

$$b^*(\theta_1, \theta_2) = \delta(\kappa - 1) \cdot \left[\theta_1 \cdot \eta(\theta_1 - \theta_2) + \theta_2 \cdot \left(1 - \eta(\theta_1 - \theta_2)\right)\right]$$

where $\delta$ is the discount factor, $\theta_i$ is the Class 1 premium for insurer $i$, and $\eta(\cdot)$ is the switching probability function. The threshold is the probability-weighted average of the two insurers' base premiums, scaled by the discounted future cost of a class downgrade. If one insurer is dominant ($\theta_1 > \theta_2$ and $\eta$ is concentrated near insurer 1), the threshold tracks insurer 1's premium.

*Theorem 4.2* establishes the existence of a Nash equilibrium premium pair $(\theta^*_1, \theta^*_2)$ under three regularity conditions. The conditions are weak enough that the authors note they hold for the $\kappa \in (1.2, 1.5)$ range characteristic of most real BMS structures.

The Nash equilibrium is a three-way fixed point: premiums determine the reporting barrier, the barrier determines the stationary distribution of policyholders across NCD classes, the stationary distribution determines each insurer's expected profit, and best-response premiums close the loop. The standard GLM pricing cycle has no mechanism for reaching this fixed point. It converges to whatever the data gives it each year, which is not the same thing.

The companion paper **(arXiv:2601.07655)** takes the single-insurer case and adds continuous-time dynamics via Piecewise Deterministic Markov Processes. Its key practical result: the optimal suppression barrier **decreases monotonically as the policy approaches its renewal date**. Near renewal, even small claims become worth reporting — the NCD penalty from switching lasts less calendar time, reducing its NPV. If you observe that FNOL rates tick up in November and December (and many UK insurers do), some fraction of that is seasonal driving conditions, and some fraction is the temporal dynamics of rational reporting behaviour.

---

## The Nash equilibrium no insurer is reaching

The Liang et al. numerical example gives a sense of the magnitudes. Base case: two-class BMS, loss distribution Gamma(1.2, 0.0085), discount rate 3%, penalty ratio $\kappa = 1.25$, price sensitivity $k_1 = 0.015$, brand preference asymmetry $k_2 = 0.8$. Equilibrium: $\theta^*_1 \approx 35.83$, $\theta^*_2 \approx 33.45$ — the preferred insurer charges approximately 7% more. As price sensitivity $k_1$ rises, both premiums fall but the premium gap is non-monotonic, widening then narrowing as competition intensifies.

The more pointed result is *Proposition 4.1*: when $k_2 > 0.5$ (one insurer has a brand preference advantage), that insurer charges more and the market does not converge to equal pricing. The preferred insurer extracts a rent that the barrier mechanism reinforces — policyholders at the preferred insurer set a higher suppression threshold (because their premium and hence the penalty NPV is higher), reducing observed frequency, which in turn makes the preferred insurer's book look cheaper than it is. The brand advantage compounds the pricing signal distortion.

The competitive implication is uncomfortable. Under Liang et al., if Insurer A corrects for underreporting bias and Insurer B does not, Insurer B will attract adverse selection from high-NCD policyholders, who are underpriced by B's naive GLM. Insurer A, correctly pricing them higher, loses them to B. The market dynamic pushes both insurers toward the naive equilibrium unless correction is industry-wide. This is why no UK insurer does it: the first mover is punished with adverse selection, not rewarded.

---

## The cross-subsidy, quantified

To be concrete about who pays whom: suppose the corrected NCD relativities at five NCD levels look like this, compared with the GLM-estimated (underreporting-contaminated) relativities on a 0-year NCD = 1.00 baseline.

| NCD years | GLM relativity | Corrected relativity | Bias |
|-----------|---------------|---------------------|------|
| 0 | 1.00 | 1.00 | — |
| 1–2 | 0.75 | 0.77 | −3% |
| 3–4 | 0.55 | 0.62 | −11% |
| 5–6 | 0.40 | 0.53 | −25% |
| 7–9+ | 0.30 | 0.36 | −17% |

These are illustrative, not empirical. The direction and order of magnitude are consistent with a 35% underestimate at 5-year NCD (as derived above) applied to a typical UK NCD frequency model. The practical read: policyholders at 5–6 years NCD are likely undercharged by around 25% relative to their true risk. Policyholders at 0–2 years NCD pay more than their share.

The bias at 7-year-plus is smaller than at 5-year, because the absolute suppression threshold is lower at the top of the ladder (£90–100 versus £280) — many more claims clear the reporting bar even when suppression is occurring.

---

## Four steps to a corrected model

Achieving full Nash equilibrium pricing per Liang et al. requires competitor premium data and a fixed-point solver — possible in principle, a significant practical lift. But correcting the frequency model for underreporting bias does not require the full competitive equilibrium machinery. Four steps:

**Step 1: Estimate retention thresholds.** Implement Lemaire's dynamic programming algorithm calibrated to your NCD ladder (your specific discount percentages, your step-back rules), your portfolio's average base premium by segment, and a severity distribution from your own claims data. The algorithm is around 50 lines of NumPy. Output: a retention table $\{b^*_n\}$ for each NCD class $n$.

**Step 2: Compute the censoring correction.** For each NCD class, the probability that a claim exceeds the threshold is:

$$p_n = 1 - F_{\text{Gamma}}\!\left(b^*_n;\, \hat{\alpha}, \hat{\beta}\right)$$

where $\hat{\alpha}, \hat{\beta}$ are your fitted severity parameters. The corrected true frequency is then $\hat{\lambda}_n = \lambda^{\text{obs}}_n / p_n$.

**Step 3: Re-estimate the frequency model.** Either replace observed counts with exposure-weighted corrected frequencies, or fit a truncated Poisson likelihood that treats observed counts as censored from below at $b^*_n$. The latter is more principled — it is a standard survival analysis adjustment applied to count data, well within actuarial toolkits.

**Step 4: Iterate.** The corrected premiums change the retention thresholds (via the Liang et al. barrier formula), which changes the censoring correction, which changes the frequency estimates. Repeat until convergence. In practice, one or two iterations materially close the gap; full convergence to the Nash equilibrium requires competitor pricing data that Steps 1–3 do not need.

Steps 1–3 are achievable with internal data in a few days of implementation. They give an order-of-magnitude estimate of the bias per NCD class that is entirely defensible under standard actuarial methodology. The machinery already exists in any team with a working severity model and a GLM framework.

---

## What FNOL data tells you

There is a direct empirical signal for calibrating $b^*_n$ that most UK insurers are already collecting and largely ignoring for this purpose: FNOL withdrawals.

When a policyholder calls to report a loss and then declines to proceed — "actually, don't worry, I'll sort it myself" — this is a self-settlement: almost certainly a claim below the retention threshold. The distribution of reported loss amounts on FNOL calls that are subsequently withdrawn, filtered by NCD class, is a direct sample from the lower tail of the loss distribution cut off at $b^*_n$.

If FNOL withdrawal rates are 8% for 0-year NCD policyholders and 22% for 7-year-plus NCD policyholders, and the average withdrawn loss amount is larger for high-NCD classes, you have direct evidence of differential suppression — and a dataset with which to calibrate the threshold per class without relying entirely on the theoretical formula.

Most claims systems capture enough data to run this analysis. It requires extracting FNOL-opened-but-not-paid records, matching them to NCD class at incident date, and cross-tabulating withdrawal rates and any available loss estimates against NCD band. The analysis is not exotic. It is just rarely done.

---

## Protected NCD: the complication

Policyholders with protected NCD have a near-zero rational retention threshold for a single claim. If your NCD percentage is preserved regardless, the immediate financial case for suppressing a claim largely disappears.

But not entirely. Three complications remain:

First, the base premium can still increase post-claim even when the discount percentage is protected. A base that reprices from £1,000 to £1,300 after a fault incident turns a protected 70% discount into £390 rather than £300 — the policyholder is paying 30% more despite protection. The decision calculation needs to account for this, and the insurer's own repricing behaviour is not necessarily transparent to the policyholder at point of decision.

Second, a second claim during the protection window — typically 1–2 claims are permitted before protection eligibility is lost — does cross into suppression territory. PNCD holders with one claim already registered face a step-back risk on a second, and their retention threshold for the second claim can be substantial.

Third, the PNCD loading is a sunk cost that distorts framing. Policyholders who have paid for protection and then self-insure anyway have wasted the loading. Behavioural evidence suggests this framing effect is real: PNCD holders report at higher rates partly because not claiming feels like "wasting" the protection they paid for.

The practical consequence for the frequency model: the suppression profile for PNCD-holding policyholders is different from non-protected policyholders at the same NCD level. Lumping them together in a single NCD factor will average across behaviours and miss both. A product type interaction term is the minimum fix.

---

## Why no UK insurer does this

The answer is in Section 7.4 of the research, and it is straightforward to state even if it is depressing: the competitive dynamics punish the first mover.

If you price high-NCD policyholders correctly — higher than competitors who have not corrected for suppression bias — your price for 5-year-plus NCD business will be above market on the aggregator. Those customers will walk. You are left with the low-NCD policyholders at correctly-set rates, but you have lost volume in the most efficient segments of the portfolio.

The insurer who continues using the naive GLM retains the high-NCD business and, because their book looks cleaner (no small claims showing through), achieves apparently good combined ratios for a few years. The bias is not immediately visible in performance data because the suppressed claims never appear.

It is a classic insurance market dynamic: the individually rational choice (do not unilaterally correct a bias that your competitors are exploiting) is collectively irrational (the industry-wide mispricing creates capital misallocation and cross-subsidies that compound over time). Liang et al.'s Nash equilibrium framework makes this precise: the correct equilibrium is well-defined, but no single insurer can reach it unilaterally. Getting there requires either coordinated methodology change or regulatory intervention.

Neither appears imminent. The FCA has published nothing on NCD methodology or the frequency suppression bias. Consumer Duty's fair value requirement could in principle apply — systematically undercharging high-NCD policyholders relative to their true risk is a cross-subsidy that misprices other segments — but no FCA guidance exists on this point.

---

## What a UK pricing team should do now

Not all of this is out of reach. The full Nash equilibrium computation is a research project. Quantifying the bias in your own portfolio is not.

Start with Steps 1–3. Implement Lemaire's algorithm for your NCD ladder — it is roughly 50 lines of NumPy once you have the discount table, step-back rules, and a severity distribution. Compute the implied retention thresholds per NCD class. Compute $p_n$ per class. Compare $\lambda^{\text{obs}}_n / p_n$ against $\lambda^{\text{obs}}_n$. The ratio tells you how much the GLM is underestimating frequency at each NCD level.

Separately, pull your FNOL withdrawal data by NCD class. If the pattern is consistent with the retention threshold estimates — withdrawal rates rising with NCD, average withdrawn loss amounts correlating with the theoretical thresholds — you have empirical confirmation that the suppression effect is real in your portfolio, not just theoretical.

That analysis will almost certainly show material bias at mid-to-high NCD classes. At that point, the pricing team has a decision: quantify the cross-subsidy, present it to senior management, and make an informed choice about whether to act unilaterally or wait for a market-level shift.

We think the magnitude — 15–35% frequency underestimate across the most populated NCD bands — is large enough to warrant that conversation now. The formalisation of the game-theoretic equilibrium in Liang et al. means the phenomenon is now precisely described, not just intuited. That changes the status of inaction from "we're not aware of the problem" to "we've chosen not to correct it."

Whether that choice is defensible under Consumer Duty's fair value obligations is a question for each firm's actuarial function and compliance team to answer. We do not think the answer is obviously yes.

---

## Further reading

- Liang, Z., Zhang, J., Zhou, Z., Zou, B. (2026). "Optimal Underreporting and Competitive Equilibrium." [arXiv:2601.12655](https://arxiv.org/abs/2601.12655)
- Enzi, L., Thonhauser, S. (2026). "To report or not to report: Optimal claim reporting in a bonus-malus system." [arXiv:2601.07655](https://arxiv.org/abs/2601.07655)
- Lemaire, J. (1977). "La Soif du Bonus." *ASTIN Bulletin* 9(1-2):181–190
- Holtan, J. (2001). "Optimal Loss Financing Under Bonus-Malus Contracts." *ASTIN Bulletin* 31(1):161–173
- Abbring, J.H., Chiappori, P-A., Zavadil, T. (2008). "Better Safe than Sorry? Ex Ante and Ex Post Moral Hazard in Dynamic Insurance Data." SSRN 1260168
- Norberg, R. (1976). "A credibility theory for automobile bonus systems." *Scandinavian Actuarial Journal*, 2:92–107
- FCA PS21/5 (2021). General Insurance Pricing Practices
- FCA EP25/2 (2025). Evaluation Paper: GIPP Remedies
