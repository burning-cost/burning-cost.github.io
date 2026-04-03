---
layout: post
title: "The Interest Rate Effect Nobody Put in Their NCD Model"
date: 2026-04-03
author: burning-cost
categories: [pricing, motor, techniques]
tags: [NCD, bonus-malus, BMS, underreporting, hunger-for-bonus, interest-rates, Holtan, macro, UK-motor, frequency-model, claim-suppression, reporting-behaviour, discount-rate, monetary-policy, personal-lines]
description: "Holtan (2001) showed that the NCD reporting threshold falls when interest rates rise — the NPV of future premium penalties shrinks, so policyholders become more willing to claim. The Bank of England's rate cycle of 2022–2023 was the largest UK rate shock in decades. Nobody modelled what it did to claim reporting behaviour, and the claims inflation literature almost certainly attributed some of the effect to cost inflation rather than behavioural change."
math: true
---

We have published several posts on the hunger-for-bonus effect: policyholders with good NCD rationally suppress small claims because the discounted cost of losing NCD exceeds the claim amount. The January 2026 papers by Liang et al. (arXiv:2601.12655) formalise the competitive equilibrium. The threshold calculations are NCD-class-specific, severity-distribution-aware, and the resulting frequency bias runs to 35% at mid-ladder NCD classes.

One input to the retention threshold calculation has received almost no attention: the discount rate.

---

## Holtan's observation

Jørn Holtan, writing in the ASTIN Bulletin in 2001, reframed the BMS reporting decision as a pure financing problem. A policyholder in NCD class $n$ who suffers a loss $Y$ can choose one of two paths:

- **Report the claim:** receive $Y$ from the insurer, pay the premium penalty over the rebuilding horizon.
- **Self-finance:** pay $Y$ out of pocket, preserve the NCD.

The indifference threshold is the NPV of future premium increases, discounted at the risk-free rate $r$. For a policyholder stepping back two years from class $j$ to class $j-2$, paying a base premium $B$ with discount schedules $D_j$ and $D_{j-2}$, the threshold over a $T$-year rebuilding horizon is:

$$b^*_j = B \cdot (D_j - D_{j-2}) \cdot \sum_{t=1}^{T} (1+r)^{-t}$$

This is $B \cdot (D_j - D_{j-2})$ multiplied by the present value annuity factor at rate $r$.

Holtan's insight: the threshold $b^*_j$ is a **decreasing function of the interest rate $r$**. As $r$ rises, the annuity factor falls, and the NPV of future premium penalties shrinks. A policyholder discounts the future more heavily, so the NCD penalty — which materialises over two future years of elevated premium — is worth less in present value terms. They become more willing to claim smaller losses.

The direction is unambiguous: higher interest rates → lower reporting thresholds → more claims reported → higher observed claim frequency at mid-to-high NCD classes.

---

## What the Bank of England rate cycle did

The Bank of England base rate was at 0.1% in January 2022. By August 2023 it had reached 5.25% — a 515 basis point rise in under two years, the steepest sustained UK rate increase since the early 1990s. By early 2026 the rate has come back to approximately 4.25%.

The effect on the Holtan threshold is mechanical. Take a 5-year NCD holder on a £1,000 base premium paying £350 (65% discount), stepping back to 3-year NCD (50%, £500) over a 2-year rebuilding horizon.

| Interest rate | Annuity factor (2 yr) | Threshold $b^*_5$ |
|---------------|----------------------|-------------------|
| 0.1% (Jan 2022) | 1.997 | £300 |
| 2.5% (mid-2022) | 1.928 | £289 |
| 5.25% (Aug 2023) | 1.855 | £278 |

The difference between a 0.1% and 5.25% rate environment is about £22 on the suppression threshold for a 5-year NCD holder — roughly a 7% reduction. Smaller in absolute terms than the uncertainty in estimating the base premium or the rebuild period. But the 5-year NCD case understates the effect at higher base premiums and higher premium differentials.

For a 3-year NCD holder (50%, £500) stepping to 1-year (30%, £700), annual penalty £200 over a 2-year rebuild:

| Interest rate | Annuity factor (2 yr) | Threshold $b^*_3$ |
|---------------|----------------------|-------------------|
| 0.1% (Jan 2022) | 1.997 | £399 |
| 5.25% (Aug 2023) | 1.855 | £371 |

A £28 reduction — around 7% — at 3-year NCD. The suppression threshold falls, and marginally more small claims get reported.

These are not dramatic numbers in isolation. The effect matters not because any individual threshold movement is large, but because it is a **systematic, directional shift** affecting every NCD class simultaneously, in a period when UK motor claims costs were already rising fast and every pricing team was trying to isolate genuine cost inflation from other signals.

---

## The identification problem

The UK claims inflation narrative for 2022–2024 is well established. The FCA's GIPP evaluation paper (EP25/2, 2025) documents a 49% rise in expected cost per motor policy between 2022 and 2024 (from £92 to £138 per policy). This is attributed to supply chain disruption, labour shortages in repairers, used vehicle prices, and parts costs — the standard post-pandemic inflation story.

But a rising-rate environment simultaneously reduces reporting thresholds for high-NCD policyholders. Some fraction of the observed increase in claim frequency from 2022 to 2023 is not cost inflation — it is policyholders rationally choosing to report claims they would previously have self-financed. The two effects are observationally equivalent in aggregate claims data: more claims filed, higher loss costs.

Nobody separated them. The actuarial community's standard response to the 2022-2024 period was to fit trend models to observed claim counts, attributing the increase to cost drivers. The Holtan channel was not in the conversation.

The size of the misattribution is unknowable without a policy-level panel dataset with NCD class, incident date, claim amount, and interest rate at decision time — which no published analysis has used. But the direction is clear, and the practical consequence matters for how 2022-2024 claims inflation estimates should be read when used as trend inputs going forward.

---

## A compounding dynamic: higher base premiums raise the stakes

Rising interest rates do not affect NCD reporting in isolation. The same 2022-2024 period saw significant base premium inflation — per the FCA EP25/2 data, average inception-year premiums at new business rose from £248.52 to £260.92 — a modest figure because it captures the constrained post-GIPP market, not the full renewal book. But within-portfolio renewal premiums rose faster, with some market participants reporting renewal increases in excess of 30% on expiring for certain segments.

A higher base premium $B$ raises the suppression threshold proportionally, holding the discount schedule constant:

$$b^*_j \propto B$$

So if $B$ rises from £700 to £900 for a given policyholder segment, the rational suppression threshold rises by 29% — from, say, £280 to £360. The higher the premium, the more NCD is worth protecting, and the more small claims are self-insured.

This creates an offsetting dynamic to the interest rate effect: premium inflation raises suppression thresholds (fewer claims reported) while rate rises lower them (more claims reported). In 2022-2023, both forces were in play simultaneously, and they run in opposite directions. The net effect on observed claim frequency is ambiguous without quantifying both.

For pricing purposes, this means the 2022-2024 claims trend estimates are contaminated by two behavioural shifts — in opposite directions — layered on top of genuine cost inflation. A trend model that treats observed claim frequency as a clean signal of true risk change is almost certainly misestimating the underlying trend, in a direction that is difficult to determine without the behavioural decomposition.

---

## What a corrected trend model would need

A full decomposition would require:

**Step 1: Compute the Holtan threshold for each NCD class at each rate environment.** The annuity factor changes month by month as the Bank of England rate moves. The threshold $b^*_{j,t}$ is a function of both NCD class and time. This is a deterministic calculation once you have the NCD ladder and the base rate series.

**Step 2: Compute the implied reporting probability at each class-month cell.** $p_{j,t} = P(Y > b^*_{j,t})$ under the severity distribution, which itself changes over time (severity inflation). The reporting probability varies both because the threshold moves (rate channel) and because the distribution shifts (inflation channel).

**Step 3: Decompose observed frequency change into threshold change and true frequency change.** If $\lambda^{\text{obs}}_{j,t} = \lambda_{j,t} \cdot p_{j,t}$, then changes in observed frequency over time $t_0$ to $t_1$ reflect changes in both true frequency $\lambda$ and reporting probability $p$. Isolating $d\lambda_{j}$ requires estimating $dp_{j,t}$ from the threshold model. This is the causal inference problem: the counterfactual observed frequency at $t_1$ if rates had stayed at $t_0$ levels.

Steps 1 and 2 are mechanical given data that any UK pricing team holds. Step 3 requires a structural assumption about the relationship between observed and true frequency — specifically, that the Lemaire/Holtan threshold model is the correct mechanism. It is.

No UK insurer has published this decomposition. Our view is that 2022-2024 frequency trends used as inputs to current pricing should carry a downward adjustment at mid-to-high NCD classes, because some fraction of the 2022-2023 frequency increase was a rate-driven reporting shift that does not reflect a permanent change in true accident rates. How large? A few percent at most, given the £20-30 threshold changes we computed above — but in a market where combined ratios have been tight, a few percent matters.

---

## The direction of the current rate cycle

The Bank of England began cutting rates in August 2024. By early 2026, the base rate has declined from its 5.25% peak, with the forward curve pricing further modest cuts. If Holtan's mechanism is real, this creates a partial reversal: declining rates will slowly raise suppression thresholds, meaning some claims currently being reported will revert to self-insurance. Observed frequency at high-NCD classes should drift downward — slightly — as rates fall, independent of any change in true accident rates.

This is the mirror image of the 2022-2023 reporting shift. If pricing teams extrapolate the 2022-2024 frequency trend forward without adjusting for the behavioural component, they will overestimate forward claim frequency at mid-to-high NCD classes in a falling-rate environment. The GLM refit loop will eventually catch the lower observed frequency — but it will attribute it to risk improvement or model drift rather than the behavioural mechanism.

Whether this is material depends on the magnitude of further rate cuts and the severity distribution in specific NCD bands. It is probably not the largest source of forecast error in any UK motor pricing model. But it is a source of systematic, directional error that is straightforward to correct once you have the retention threshold model in place.

---

## The practical recommendation

If you have already implemented Lemaire's dynamic programming algorithm for your NCD ladder — as we outlined in our [earlier piece on correcting NCD relativities](/pricing/motor/techniques/2026/04/02/your-ncd-relativities-are-wrong/) — the Holtan rate sensitivity is trivially computed. The discount factor $\delta = 1/(1+r)$ is a direct input to the threshold formula. Replace the constant $\delta$ with a time-varying $\delta_t$ derived from the risk-free rate at each valuation date, and the threshold table becomes a matrix $\{b^*_{j,t}\}$ rather than a vector $\{b^*_j\}$.

The additional computational cost is minimal. The practical benefit is a more defensible trend decomposition when presenting to reserving committees or pricing governance forums. "The 2022 frequency increase was partly a Holtan reporting shift, not purely a claims inflation effect" is a specific, testable assertion — more useful than the generic "behavioural factors may have contributed."

---

## What remains unknown

Holtan's paper used a deterministic threshold framework. The full stochastic treatment — where the discount rate itself is stochastic and the policyholder optimises over a distribution of future rates — has not been published. In practice, policyholders do not observe the Bank of England base rate and consciously discount NPVs; the mechanism works through the opportunity cost of self-financing, which rises with the general level of interest rates. The Holtan model is a reasonable approximation, not a precise behavioural model.

The empirical question is also open. No published study has used UK claims data to detect the interest-rate channel in NCD reporting behaviour. The Dutch data used by Abbring, Chiappori and Zavadil (SSRN, 2008) is the closest empirical treatment of strategic reporting in motor insurance, but their identification strategy focused on ex post moral hazard broadly, not the rate sensitivity specifically.

A UK insurer with policy-level NCD history, incident dates, and settled claim amounts across the 2020-2025 period could run this test. The prediction: controlling for vehicle age, cover type, and policyholder characteristics, reporting probability at mid-to-high NCD classes should have been higher in mid-2023 (peak rates) than in early 2022 (near-zero rates), by an amount consistent with the Holtan threshold calculation. If the effect is detectable, it validates the mechanism. If it is not, the threshold movements are too small to matter in UK motor data, and the correction can be safely ignored.

Neither answer is currently published.

---

## Further reading

- Holtan, J. (2001). "Optimal Loss Financing Under Bonus-Malus Contracts." *ASTIN Bulletin* 31(1):161–173. [The foundational paper on the interest-rate channel]
- Liang, Z., Zhang, J., Zhou, Z., Zou, B. (2026). "Optimal Underreporting and Competitive Equilibrium." [arXiv:2601.12655](https://arxiv.org/abs/2601.12655)
- Lemaire, J. (1977). "La Soif du Bonus." *ASTIN Bulletin* 9(1-2):181–190
- FCA EP25/2 (2025). Evaluation Paper: Our GIPP Remedies
- [Your NCD Relativities Are Wrong, and the Maths Now Tells You How Wrong](/pricing/motor/techniques/2026/04/02/your-ncd-relativities-are-wrong/) — the frequency correction in detail
- [The Hunger for Bonus: How UK Motor NCD Pricing Gets the Frequency Wrong](/pricing/techniques/2026/04/01/the-hunger-for-bonus-how-uk-motor-ncd-pricing-gets-the-frequency-wrong/) — the full theoretical treatment
