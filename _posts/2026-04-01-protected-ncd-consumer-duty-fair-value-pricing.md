---
layout: post
title: "Protected NCD Is a Fair Value Problem Waiting to Happen"
date: 2026-04-01
categories: [regulation, pricing]
tags: [ncd, protected-ncd, pncd, consumer-duty, fair-value, motor-insurance, uk-motor, fca, pricing-bias, bms, bonus-malus, ps22-9, gipp, lemaire, claim-suppression, underreporting, personal-lines]
description: "Protected NCD is widely misunderstood by consumers, and the product may not deliver the value it charges for. The Consumer Duty fair value test and the hunger-for-bonus literature together suggest UK motor pricing teams should look harder at PNCD than they currently are."
math: true
author: burning-cost
---

Protected NCD is sold on the basis that it preserves your discount if you make a fault claim. That is broadly true. What is less clearly communicated — and what the FCA has already flagged — is that it preserves the *percentage*, not the total premium. After a fault claim, a policyholder with protected NCD keeps their 70% discount and watches their base premium increase at renewal anyway. The protection they paid for absorbed part of the impact, not all of it.

This is a Consumer Understanding problem with a documented FCA concern attached to it. But there is a second issue, less discussed: whether PNCD delivers fair value *at the actuarial level*. Whether the premium loading charged for protection reflects the statistical benefit provided. This is where the pricing literature on strategic claim underreporting becomes directly relevant, and where the Consumer Duty Price and Value outcome starts to bite in a way most UK motor teams have not worked through.

---

## What protected NCD actually does

A fault claim under standard NCD steps a policyholder back, typically two years on a UK ladder. A 9-year NCD holder goes to 7-year. A 5-year holder goes to 3-year. The premium implications can be substantial: a 5-year holder at £350 (65% discount on a £1,000 base) steps back to 3-year at £500 (50% discount), an increase of £150 per year, sustained for two years until they rebuild.

Protected NCD, sold at a typical loading of 5-15% of the total premium, preserves the *percentage* — 65% becomes 65% — but the *base premium* on which that percentage is applied can still increase at renewal. If the base reprices from £1,000 to £1,300 post-claim (which is plausible after a fault incident given that actuarial factors change), the policyholder paying 35% of £1,300 is paying £455 — worse than the unprotected post-claim premium at the old base rate. The protection evaporated.

The FCA's Consumer Understanding outcome under Consumer Duty requires that products are understood by customers at point of sale and renewal. PNCD has a documented misunderstanding problem: research consistently shows that a substantial proportion of policyholders believe protection covers the total premium, not just the discount percentage. The FCA has signalled this is on their radar. It is a live compliance question for every UK motor team that sells PNCD.

---

## The fair value test is harder than it looks

Consumer Duty's Price and Value outcome asks whether the price charged is reasonable relative to the benefit delivered. For PNCD, that means asking: does the loading charged reflect the expected cost saving provided?

The expected benefit of PNCD is the discounted premium saving from preserving the NCD percentage after a fault claim, weighted by the probability of a fault claim occurring during the policy year. At a claim frequency of 0.07 per year (reasonable for a mid-NCD class UK motor policyholder), and an expected annual premium saving of £150 if the discount is preserved, the expected monetary benefit of PNCD in a given year is approximately:

```
E[benefit] ≈ 0.07 × £150 = £10.50
```

At a loading of 5% on a total premium of £350, the cost is £17.50. The loading exceeds the expected benefit by two-thirds. On these numbers, PNCD does not deliver fair value at the loading level charged.

These are illustrative. The actual calculation requires the insurer's fault claim frequency at each NCD class eligible for PNCD (typically 4-year NCD and above), the actual premium stepping rules, and a view on post-claim base premium repricing. But the direction of the concern is structural: PNCD eligibility requires 4-5 years' NCD minimum, which means it is sold to precisely the customers with the lowest fault claim frequencies — the customers least likely to trigger the protection. For them, the loading-to-expected-benefit ratio is most unfavourable.

---

## The hunger-for-bonus complication

The underreporting literature introduces a further wrinkle. Our [earlier post](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/) worked through Lemaire's result that policyholders rationally suppress claims below a threshold — the "hunger for bonus" — and showed that mid-ladder NCD holders (3-5 years) face rational retention thresholds of £278-£370 at a £1,000 base premium.

Protected NCD materially lowers the rational retention threshold. If the discount percentage is preserved after a claim, the NPV of the penalty is near zero for a single claim. The policyholder with PNCD should claim almost everything, rather than self-insuring up to the retention threshold. In theory, PNCD holders ought to report more claims than unprotected policyholders at the same NCD level, because they face no financial penalty from the NCD mechanism for doing so.

If that is true, the fair value calculation for PNCD is not just about the frequency of fault claims. It is about the *change* in claim reporting behaviour between PNCD and unprotected policyholders at the same NCD class. If PNCD holders report 25% more claims than unprotected holders — because their rational retention threshold has fallen from £278 to near zero — the expected cost to the insurer of PNCD is far higher than the frequency-weighted calculation above suggests. The loading must cover not just the premium-step-back saving but the increased claims liability from the shift in reporting behaviour.

We are not aware of any published UK analysis that estimates this reporting-behaviour effect empirically. The FNOL withdrawal rate — calls opened but not converted to paid claims — is the cleanest available signal: PNCD holders should show materially lower FNOL withdrawal rates than unprotected holders at the same NCD level. If that signal exists in your claims data and the difference is large, the fair value calculation needs revisiting.

---

## What a defensible PNCD pricing methodology looks like

A PNCD loading that can survive a Consumer Duty fair value challenge needs to be grounded in three things.

**The premium step-back saving.** At each NCD class where PNCD is eligible, what is the expected premium saving from preserving the discount percentage after a fault claim? This requires modelling the post-claim base premium trajectory — not just the NCD transition — because base premium repricing partly erodes the protection. The modelling needs to account for the fact that policyholders with PNCD may not renew with the same insurer post-claim, which affects the value of the protection to both parties.

**The fault claim frequency for eligible classes.** Expected benefit is zero if the policyholder never makes a fault claim. At 4-year NCD and above, observed fault claim frequencies are low — but they are also suppressed by strategic underreporting. As we noted in our [earlier post](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/), observed frequency at high-NCD classes may understate true frequency by 15-35% at plausible retention thresholds. The fair value calculation should use corrected (true) frequencies, not the observed (suppressed) frequencies from the claims system.

**The reporting behaviour shift.** The expected cost of providing PNCD includes the increase in reported claims from policyholders whose retention threshold has dropped. Estimating this from data requires comparing FNOL withdrawal rates between matched PNCD and non-PNCD policyholders at the same NCD class. A difference-in-differences approach, controlling for base risk factors, gives an approximation of the reporting elasticity. This is achievable with internal data.

If the loading covers items (1) and (2) but not (3), PNCD is systematically underpriced as a product — the insurer is collecting a loading calibrated to the step-back saving while absorbing an unreserved claims increase from the reporting shift. If the loading overcorrects and exceeds all three components, it may not represent fair value to the customer.

---

## Regulatory exposure

The FCA's October 2023 multi-firm review of Consumer Duty implementation found that firms' fair value assessments were often not quantitative at the benefit level. Many firms were asserting fair value without demonstrating that the price charged reflected the expected benefit delivered. PNCD is exactly the product type — ancillary cover with a clear expected benefit structure — where a qualitative fair value assertion is weakest.

The FCA has published nothing specifically on PNCD pricing methodology. The obligation is on firms to apply the fair value framework. But the combination of documented consumer misunderstanding and a structurally unfavourable loading-to-benefit ratio — particularly for high-NCD policyholders who are least likely to claim — makes PNCD a plausible candidate for a targeted FCA review or a skilled person review request under the Product and Service outcome.

The GIPP evaluation paper (EP25/2, 2025) did not examine PNCD fair value. The FCA's planned 2025 motor claims review was focused on claims costs and repair inflation. Neither document closes the question of whether PNCD loading is appropriately calibrated. Firms that have not done the quantitative fair value work are carrying regulatory exposure they may not have fully assessed.

---

## What this means for pricing teams

Protected NCD is, in most UK motor books, a product that has been loaded based on market convention rather than actuarially derived benefit estimates. The industry standard loading of 5-15% reflects what the market charges, not what the protection is worth — a charge calibrated against competitor quotes is circular if every competitor is doing the same thing, and it does not constitute a documented quantitative fair value assessment.

The most practical first step is examining FNOL withdrawal data by NCD class and PNCD status. If PNCD holders show significantly lower FNOL withdrawal rates — they report more of what happens, because their rational retention threshold is near zero — that is the empirical foundation for the reporting behaviour component. If withdrawal rates are similar, the reporting shift effect is small and the simpler premium-step-back calculation is the right framework.

The second step is running the Lemaire retention threshold calculation for your own NCD ladder (we described the 50-line NumPy implementation in our [earlier post](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/)) and using the implied reporting probabilities to correct the fault claim frequencies used in the PNCD fair value calculation. If your 5-year NCD observed frequency is 0.05 but the corrected true frequency is 0.077 (at a retention threshold of £278 and P(Y > £278) ≈ 0.65), the fair value benefit estimate goes up by the same ratio. That might mean the loading is actually reasonable — or it might make the mismatch worse, depending on whether the loading was set against observed or corrected frequencies.

The calculation is not complex. The data is, for most large UK motor insurers, already available. What has been missing is the motivation to do it. Consumer Duty provides the motivation.

---

## Related reading

- [Your Policyholders Are Playing a Game with Your NCD Ladder](/2026/03/31/ncd-game-theory-bms-calibration-underreporting/) — the hunger-for-bonus framework, Lemaire's retention thresholds, and the Liang et al. competitive equilibrium result
- [Profit-Fairness Market Equilibria: Why Consumer Duty Audits the Wrong Unit](/2026/03/31/profit-fairness-market-equilibria-consumer-duty/) — the market-level fair value problem that firm-level Consumer Duty compliance cannot capture
