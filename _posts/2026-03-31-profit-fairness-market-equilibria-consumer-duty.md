---
layout: post
title: "Profit-Fairness Market Equilibria: Why Consumer Duty Audits the Wrong Unit"
date: 2026-03-31
categories: [fairness, regulation]
tags: [Consumer-Duty, FCA, EP25-2, market-equilibrium, collusion-pathology, fairness-metrics, RL-regulation, MarketSim, income-exclusion, Citizens-Advice, motor-insurance, proxy-discrimination, Nash-equilibrium, access-to-insurance]
description: "Thibodeau et al. build a multi-firm market simulator and demonstrate the collusion pathology: a cartel that excludes every income group equally passes standard demand-fairness metrics. Consumer Duty audits one firm. The real problem is market-level."
author: burning-cost
---

Here is a fairness audit scenario worth sitting with. Three insurers collude. They set prices high enough that low-income customers cannot afford cover. Every income group opts out at the same rate. Your fairness metric — the one that checks whether uptake rates differ across income groups — passes them all with flying colours. The exclusion is equal, therefore the market is fair.

This is not a hypothetical constructed to embarrass fairness researchers. It is a formal result from Thibodeau, Nekoei, Taïk, Rajendran and Farnadi (arXiv:2506.00140, May 2025), demonstrated in a calibrated multi-firm market simulation. And it identifies a genuine blind spot in how UK regulators currently think about pricing fairness.

---

## What Consumer Duty actually audits

The FCA's Consumer Duty framework asks individual firms to demonstrate fair value outcomes. For pricing, that means showing your pricing model does not produce systematic disadvantage for identifiable customer groups — income cohorts, demographic groups, customers with protected characteristics. The toolkit is firm-level: proxy detection, demographic A/E analysis, the DoubleFairnessAudit Pareto front. You demonstrate that *your* model is not the problem.

The FCA's EP25/2 paper (July 2025) evaluated whether the GIPP remedies introduced under PS21/5 reduced the loyalty penalty for renewing customers. It found they did. It explicitly did not assess market-wide access gaps by income or demographic group. This is not a criticism of EP25/2 — that was not its remit. But the gap it leaves open is exactly the one this paper addresses.

The paper's contribution: showing that a set of firms can each satisfy individual fairness requirements while the market equilibrium systematically excludes low-income consumers. Not because any single firm is behaving badly. Because the competitive equilibrium itself produces exclusion.

---

## The model

MarketSim, the simulator the authors build, is sparse in the way economics models usually are. Three income groups (High, Middle, Low — proportioned from US Census and Pew Research data). Two firms in the insurance case, five in the credit case. Consumer utility follows a multinomial logit: utility equals a baseline attraction minus a price-sensitivity coefficient times the premium, where price sensitivity increases sharply with lower income. Low-income consumers in the insurance scenario are 3.3 times more price-sensitive than high-income consumers.

Firms maximise profit via Powell's method against a Nash equilibrium concept — they best-respond simultaneously. There is no coordination and no regulatory intervention in the baseline scenario.

The regulator in the paper is a Soft Actor-Critic reinforcement learning agent that learns a tax schedule. It partitions firms into twenty fairness brackets based on their demographic selection gap (how much their uptake rates differ across income groups). Each bracket carries a tax rate between 0 and 1 that reduces effective margin. The RL agent learns the tax schedule to maximise a product of mean firm profit and a global fairness score.

---

## The collusion result

In the unregulated competitive scenario, firms maximise profit by pricing low-income consumers out of the market. The opt-out rate for the low-income group is substantially higher than for the high-income group. This is the access-to-insurance problem — familiar to anyone who has read the Citizens Advice motor insurance research showing a £280/year premium gap attributable to ethnicity proxies (2022 data).

In the collusion scenario, firms coordinate on high prices. All income groups opt out at high rates. The demographic *gap* in opt-out rates is small — all groups are excluded near-equally. The demand-fairness metric, which checks whether uptake rates differ across groups, scores this as approximately fair.

The authors acknowledge this result but do not resolve it in their metric design. We think the failure to resolve it is actually clarifying: it shows that any fairness metric defined over uptake *ratios* is fundamentally inadequate as a welfare measure. Equal exclusion is not fair access. The metric needs to incorporate absolute opt-out levels, not just relative gaps between groups.

---

## The RL planner results

The RL social planner produces genuine improvements in the health insurance simulation:

- Welfare (defined as mean profit times global fairness score): 0.633 under the RL planner versus 0.572 unregulated — an 11% gain
- Fairness score: 0.895 under the RL planner
- Opt-out rate: 0.121 versus 0.137 unregulated

These are the best numbers in the paper and appear in the title's "16% improvement" claim (which comes from the consumer credit scenario, not the insurance one). In consumer credit, fairness improves from 0.660 to 0.767 — a 16% gain. But the opt-out rate *worsens*, from 0.173 to 0.218. More consumers are excluded from credit under the RL-regulated market than under the unregulated one.

The paper notes this contradiction without resolving it. An RL agent that improves relative fairness by driving prices high enough that an extra 4.5 percentage points of consumers exit the market is not achieving what any regulator would actually want. This is partly a metric design problem — the welfare function W = mean profit × fairness score has no principled basis for why a 5% profit gain should exactly offset a 5% fairness loss — and partly a consequence of using a stylised simulator calibrated to aggregate US statistics.

We want to be direct about what these numbers are and are not: they are simulation results from a model with three income groups, 2–5 firms, and US Census calibration. They are not UK evidence. A UK motor insurer looking at this paper should not take the 11% welfare improvement as a prediction for what would happen under a similar regulatory intervention in the UK market. The direction of effect is plausible; the magnitude is not transferable.

---

## Where this connects to UK regulation

Citizens Advice documented in 2022 that households in predominantly ethnic-minority postcodes pay approximately £280 more per year for motor insurance than households in comparable-risk predominantly white postcodes. This is the real-world version of the market-level exclusion the paper models. The mechanism is not individual firm malice — it is that legitimate rating factors (postcode, vehicle type, claims history) encode income and demographic information, and competitive pricing dynamics amplify rather than dampen those differentials.

Consumer Duty asks each firm to demonstrate that its pricing does not produce systematic disadvantage. This is the right question at the firm level. But if every firm individually complies while competitive dynamics continue to price low-income consumers out, the market outcome is still exclusion. The paper formalises why that can happen, and why firm-level compliance metrics cannot detect it.

The tax bracket mechanism the paper proposes — taxing firms on their demographic selection *outcomes*, not on their methodology — is structurally similar to Consumer Duty's outcome framing. The FCA does not tell you which rating factors to use; it asks whether your outcomes are fair. A differential tax on firms whose market presence produces bad access outcomes, regardless of the specific pricing methodology that causes it, is a plausible formalisation of where outcome-based regulation could evolve.

The caveat is significant: the paper assumes a regulator with real-time, accurate fairness scores for every firm in the market, and the authority to impose differential tax rates on that basis. No such regulator exists anywhere. The FCA's toolkit consists of supervisory reviews, skilled persons reviews, and principle-based rules. The gap between the paper's assumed regulatory capability and the FCA's actual capability is not small.

---

## The honest assessment

This paper is theoretical infrastructure for a direction of regulatory thinking, not a description of where regulation currently is or where it will be in two years. The appropriate time horizon for the ideas it contains is a Consumer Duty five-year review — plausible 2028 onwards — if the FCA decides to extend its scope from individual firm compliance to market-level access outcomes.

The specific numerical results are US-calibrated simulations. The collusion pathology is a genuine and underappreciated finding. The RL mechanism design is genuinely novel — no existing open-source tool models competitive equilibrium effects of pricing regulation with demographic heterogeneity, and MarketSim is a useful contribution if the research community takes it up.

For a UK pricing actuary today: the relevant insight is not the tax schedule. It is the framing — that a market can produce systematic exclusion even when every individual firm satisfies individual fairness requirements. This is a board-level argument for why industry coordination on access standards, not just individual compliance, is the correct long-run response to Consumer Duty.

---

## Further reading

- [Sequential Optimal Transport for Multi-Attribute Fairness in Insurance Pricing](/fairness/machine-learning/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — firm-level fairness correction in the insurance-fairness library
- [Measuring Proxy Discrimination in Insurance Pricing: The LRTW Scalar Metric](/fairness/2026/03/31/sensitivity-based-proxy-discrimination-insurance-pricing/) — rigorous single-firm proxy discrimination measure
