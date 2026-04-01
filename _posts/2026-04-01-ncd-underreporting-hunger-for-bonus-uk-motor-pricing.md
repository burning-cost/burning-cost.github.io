---
layout: post
title: "Your NCD Relativities Are Biased. Here's Why and What to Do About It."
date: 2026-04-01
categories: [motor, pricing]
tags: [NCD, BMS, bonus-malus, underreporting, hunger-for-bonus, game-theory, Nash-equilibrium, GLM, frequency-bias, credibility, Lemaire, Norberg, UK-motor, motor-pricing, claim-suppression, dynamic-programming, severity, Poisson-GLM, uk-insurance, pricing-methodology, competitive-equilibrium, arXiv-2601]
description: "Two January 2026 arXiv papers formalise what the profession has known since Lemaire (1977): policyholders at high NCD levels strategically suppress small claims, and the GLM you fit on observed data is calibrated on a biased sample. At mid-ladder NCD classes, the suppression rate may exceed 35%. UK practice ignores this entirely."
math: true
author: burning-cost
---

Every UK motor pricing actuary knows that high-NCD policyholders are better risks. The 65% discount for 9-year NCD policyholders versus full premium for new drivers reflects decades of experience and a GLM coefficient that consistently comes out below 1.0 for higher NCD classes. The model looks right.

The problem is what is missing from the data. Policyholders with 5 or more years of NCD are rationally self-insuring small claims to protect their discount. They have an accident; they do a quick mental calculation; they pay out of pocket. The insurer never sees the claim, the data never has the event, and the GLM absorbs the resulting frequency suppression into the NCD relativity as if it were pure risk signal.

It is not. The NCD relativity contains two things simultaneously: a real signal about who is a better driver, and a statistical artefact from selective reporting. The GLM cannot distinguish them. Neither can you, without doing something different.

---

## The size of what you are missing

The rational retention threshold — the claim size below which it is cheaper to pay out of pocket than to claim and lose NCD — is not a small number.

For a policyholder on a £1,000 base premium with 5-year NCD at 65% (paying £350), a fault claim steps them back to 3-year NCD at 50% (paying £500). That £150/year increase persists for 2 years while they rebuild. At a 5% discount rate:

$$\text{PV(penalty)} \approx £150 \times (0.952 + 0.907) \approx £279$$

It is rational not to claim for any loss below approximately £280. If UK small-to-medium claims follow something like a Gamma distribution with mean £1,500, the probability of a claim exceeding £280 is roughly 0.65. Observed frequency at 5-year NCD is therefore approximately 65% of true frequency — a 35% underestimate.

The direction at 9-year NCD is reversed in absolute terms: the threshold is lower (~£90 on a similar calculation) because the premium difference between 9-year and 7-year NCD is smaller at the cap. But the mid-ladder classes — 3 to 7 years NCD — face the largest thresholds in absolute terms, and that is where the majority of a renewal book sits.

No published estimate exists for UK motor specifically. The European literature (Belgian, French BMS) finds similar orders of magnitude for comparable ladder structures. Jean Lemaire formalised the optimal retention table in 1977 (ASTIN Bulletin 9(1-2):181–190) — his "La Soif du Bonus" — but neither he nor anyone since has published a UK calibration. It remains an open research question.

---

## What Lemaire actually proved

Lemaire's dynamic programming algorithm works backwards through the NCD ladder. For each class j, it computes the NPV of future premium costs under two scenarios — retain the loss, keep NCD intact; claim, step back to class j−k. The retention threshold $X_j$ is the loss size at which the policyholder is indifferent.

The key result: $X_j$ is **higher at better NCD classes**. A 9-year NCD holder self-insures larger losses than a 2-year NCD holder, because the absolute premium penalty from a fault claim is larger at the top of the ladder. This direction is what makes the bias so significant for pricing: the suppression is concentrated exactly where you have the most policyholders (high-NCD retention book) and the most precise GLM estimates (high exposure, low standard errors). The precision is false.

Norberg (1976, Scandinavian Actuarial Journal, 2:92–107) showed the theoretically optimal BMS relativity minimises the asymptotic MSE between the BMS premium and the policyholder's true expected claims. For a Poisson-Gamma heterogeneity model, this is the Bühlmann credibility premium. The problem is that Norberg's algorithm fit to observed (underreported) data produces a posterior that underestimates the true claim intensity at high-NCD classes. The mathematical framework is correct; it is being applied to a biased input.

---

## The competitive angle: a new paper makes it worse

Liang, Zhang, Zhou and Zou (arXiv:2601.12655, submitted January 2026) are the first to model strategic claim underreporting in an oligopolistic market rather than assuming a single monopolist insurer. This matters.

The paper addresses two questions: given insurer premiums, what is the policyholders' optimal reporting strategy? And given that policyholders respond optimally, what is the Nash equilibrium premium pair for two competing insurers?

The core result (Theorem 3.1) confirms Lemaire's barrier-strategy structure: for any fixed premium pair, there exists a unique optimal threshold $b^*_{n}$ for each NCD class $n$. Crucially, the threshold is insurer-independent — it depends on the rate class and the premium structure, but not on which company the policyholder is with. The barrier has a natural interpretation: it is the NPV of the future premium penalty from a downgrade, probability-weighted by the likelihood of staying with each insurer.

In the two-class BMS, this collapses to a closed form:

$$b^*(\theta_1, \theta_2) = \delta(\kappa - 1) \cdot \left[\theta_1 \cdot \eta(\theta_1 - \theta_2) + \theta_2 \cdot (1 - \eta(\theta_1 - \theta_2))\right]$$

where $\delta$ is the discount factor, $\kappa \in (1,2)$ is the penalty ratio (Class 2 premium = $\kappa$ × Class 1 premium), and $\eta(\cdot)$ is the probability of switching to the competitor given a price differential. The threshold is exactly the expected cost of a downgrade, weighted by the probability of being with each insurer.

Theorem 4.2 proves Nash equilibrium premiums $(\theta^*_1, \theta^*_2)$ exist under regularity conditions on the loss density and the premium cap. The paper notes condition (3) is "rather weak because $\kappa \in (1.2, 1.5)$ in most BMS models" — so the existence result is robust in practice.

**Proposition 4.1** is the practically interesting result: when the preferred insurer has brand preference parameter $k_2 > 0.5$, it charges a higher equilibrium premium — not because its risk book is different, but because market power allows it to extract rent. The equilibrium premium gap (7% in the base case with $k_2 = 0.8$) is a pure competitive distortion. In the UK, this maps directly to the question of whether PCW-dominant carriers (Direct Line, Admiral, Aviva) price above the competitive equilibrium that a fully-informed market would produce.

The paper's base case produces equilibrium premiums of approximately £35.83 (preferred insurer) and £33.45 (competitor) — a 7% gap attributable entirely to brand preference asymmetry.

---

## Why the standard GLM workflow doesn't converge to the right answer

UK pricing practice: fit a Poisson GLM on reported claims with NCD as a categorical factor; set rates based on fitted coefficients; update annually with rolling data.

This procedure has no feedback loop correction. The premiums set this year determine policyholders' optimal reporting thresholds next year (via the Liang et al. closed form). Those thresholds determine the observed claim frequencies used to calibrate next year's GLM. The re-fitting loop converges to a fixed point — but that fixed point is not a Nash equilibrium unless the insurer has explicitly accounted for the reporting suppression.

Per Liang et al., the Nash equilibrium requires solving a three-way fixed point: premiums determine the reporting barrier, the barrier determines the stationary distribution of policyholders across NCD classes, the stationary distribution determines expected profit, and expected profit determines best-response premiums. A GLM re-fit solves none of this; it just fits the observed data and stops.

The competitive distortion makes it worse. If Insurer A corrects for the bias and prices high-NCD policyholders at their true frequency, while Insurer B uses naive GLM premiums, Insurer B undercharges high-NCD policyholders. Those policyholders select to Insurer B (correctly responding to the price differential). Insurer A loses exactly the customers it correctly identified as underpriced. The market dynamics push both insurers toward the naive equilibrium unless all significant competitors move simultaneously. This is a coordination problem.

---

## How to correct for it

Four steps, in increasing order of difficulty.

**Step 1: Estimate the retention thresholds.** Implement Lemaire's dynamic programming algorithm calibrated to your own NCD ladder (discount percentages, step-back rules), the portfolio average base premium by segment, and an estimated severity distribution. Output is a retention table $\{b^*_n\}$ for each NCD class. This is roughly 50 lines of NumPy; the maths is in Lemaire (1977) and fully worked in Chapter 12 of *Loss Data Analytics* (openacttexts.github.io, freely available).

**Step 2: Estimate the censoring fraction.** If severity follows Gamma($\alpha, \beta$) and the retention threshold in class $n$ is $b^*_n$, the probability that a claim is actually reported is:

$$p_n = P(Y > b^*_n) = 1 - F_{\text{Gamma}}(b^*_n;\, \alpha, \beta)$$

True frequency: $\hat{\lambda}_n = \lambda^{\text{obs}}_n / p_n$.

**Step 3: Re-estimate the frequency model.** Replace observed counts with exposure-weighted corrected frequencies, or use a truncated Poisson likelihood treating observed counts as right-censored from below at $b^*_n$. This is standard survival analysis applied to count data. The `scipy.optimize` machinery handles the custom log-likelihood cleanly.

**Step 4 (optional): Iterate toward Nash equilibrium.** Per Liang et al., premiums and reporting thresholds are jointly determined. An iterative solver can approach the equilibrium: start with initial premiums, compute $b^*$, compute $p_n$, compute expected profit at each NCD class given competitor pricing, update premiums by best response, repeat. This requires competitor premium data by NCD class — available from PCW aggregators, not from internal data alone. Steps 1–3 are achievable internally.

---

## A practical first step: quantify the bias before correcting it

For a team that wants an order-of-magnitude estimate before committing to a corrected GLM, this is achievable in a few days:

1. Implement Lemaire's algorithm for your own NCD ladder
2. Use your existing GLM's Gamma/LogNormal severity parameters by cover type
3. Compute $b^*_n$ at the portfolio average base premium per segment
4. Compute $p_n = P(Y > b^*_n)$ per NCD class
5. Compare $\lambda^{\text{obs}}_n / p_n$ to $\lambda^{\text{obs}}_n$ — this is the estimated frequency correction factor per class
6. Check whether the resulting corrected NCD relativities differ materially from your current pricing

If the correction factors are large (>10% at any class), you have a problem worth fixing. If they are small, you have documented due diligence.

One data source that is particularly informative but usually ignored: **FNOL calls that are opened but not converted to a paid claim**. These self-settlements are the direct empirical signal for calibrating $b^*$. If your claims system captures the reported loss amount at FNOL alongside the final outcome, the distribution of self-settled amounts is a direct estimate of what policyholders are retaining. The rate of FNOL withdrawal by NCD class is itself diagnostic.

---

## What not to do

Do not apply a flat percentage underreporting loading uniformly across NCD classes. The suppression is non-uniform — it is largest in absolute terms at mid-ladder classes and smallest at the top and bottom of the ladder. A uniform uplift corrects the direction but not the pattern.

Do not conflate frequency suppression with adverse selection. High-NCD policyholders are both genuinely better drivers *and* strategically suppressing claims. The GLM NCD coefficient captures both effects simultaneously. They cannot be separated without the retention threshold estimate — which is why the standard credibility argument ("the data shows they claim less, so they're better risks") is partially but not entirely correct.

Do not assume Protected NCD eliminates the problem. A PNCD holder has a near-zero rational retention threshold for a single claim — their NCD percentage is preserved. But the base premium can still increase post-claim; a second claim may trigger loss of protection eligibility; and the sunk cost of the PNCD loading affects the decision framing. PNCD holders still suppress some claims, but the threshold is lower than for equivalent unprotected policyholders.

---

## What the literature still hasn't answered

The Liang et al. result proves Nash equilibrium for N = 2 NCD classes only. Real UK NCD has 10 classes (or more). The barrier characterisation (Theorem 3.1) holds for general N, but equilibrium existence for N > 2 is conjectured, not proved. Numerical fixed-point approaches are the practical route forward; the mathematics is open.

The complementary January 2026 paper (arXiv:2601.07655) addresses a single insurer in continuous time using Piecewise Deterministic Markov Processes. Its key result is that the optimal barrier decreases monotonically as contract maturity approaches — near policy expiry, even small claims are worth reporting, because the NCD disadvantage lasts a shorter time. This has an immediate practical implication: claim reporting behaviour is not stationary within the policy year. Mid-year claims and year-end claims come from different points on the barrier curve. No UK pricing model accounts for this.

Three other open questions the literature has not resolved:

**UK empirical calibration.** Abbring, Chiappori and Zavadil (SSRN 2008) used Dutch longitudinal motor data to separate ex ante from ex post moral hazard. They found evidence consistent with strategic non-reporting but limited evidence of ex ante moral hazard. The Dutch BMS is more punitive than UK, so direct transfer is limited. A UK-specific study using FNOL data and NCD histories has not been published.

**Telematics interaction.** A telematics policyholder may be unable to suppress a large claim if the insurer already has FNOL data from the device. How telematics changes the hunger-for-bonus equilibrium — and whether it shifts the equilibrium away from high-NCD suppression — is entirely unmodelled in the literature.

**Protected NCD theory.** The decision tree for a PNCD holder is more complex than the standard BMS problem (second-claim threshold, eligibility expiry window, base premium repricing post-claim) and has not been formally analysed.

---

## The regulatory angle

The FCA has published nothing on this. No guidance on correcting for strategic underreporting in NCD methodology; no mention of frequency bias from hunger for bonus; no reference to the game-theoretic calibration literature. GIPP (PS21/5) constrains renewal pricing but says nothing about NCD ladder design. Consumer Duty's fair value outcome could, in principle, apply if a pricing methodology systematically mislabels a customer segment's risk — but no FCA guidance has drawn that connection.

This is not a comfort. The absence of regulatory obligation does not change the fact that the NCD relativities in your model are calibrated on biased data, and that the policyholders being cross-subsidised (high-NCD classes) are being charged less than their true risk. That is a pricing accuracy problem regardless of what the FCA says about it.

The Consumer Duty question that is live concerns Protected NCD specifically. FCA scrutiny of PNCD fair value is plausible given the 2024 thematic review findings on product governance. The Consumer Understanding outcome requires insurers to communicate clearly that PNCD protects the discount percentage, not the total premium — a distinction that is widely misunderstood. Whether PNCD pricing adequately reflects the statistical benefit provided is a fair value question that no published FCA guidance has addressed.

---

## The papers

**Primary:**

Liang, Z., Zhang, J., Zhou, Z. & Zou, B. (2026). 'Optimal Underreporting and Competitive Equilibrium.' arXiv:2601.12655. Submitted 19 January 2026. [https://arxiv.org/abs/2601.12655](https://arxiv.org/abs/2601.12655)

Authors unnamed in source (2026). 'To report or not to report: Optimal claim reporting in a bonus-malus system.' arXiv:2601.07655. Submitted January 2026. [https://arxiv.org/abs/2601.07655](https://arxiv.org/abs/2601.07655)

**Classical:**

Lemaire, J. (1977). 'La Soif du Bonus.' *ASTIN Bulletin*, 9(1-2):181–190. The foundational hunger-for-bonus analysis; still the right starting point.

Norberg, R. (1976). 'A credibility theory for automobile bonus systems.' *Scandinavian Actuarial Journal*, 2:92–107.

Holtan, J. (2001). 'Optimal Loss Financing Under Bonus-Malus Contracts.' *ASTIN Bulletin*, 31(1):161–173.

Denuit, M., Marechal, X., Pitrebois, S. & Walhin, J-F. (2007). *Actuarial Modelling of Claim Counts*. Wiley.

**Empirical:**

Abbring, J.H., Chiappori, P-A. & Zavadil, T. (2008). 'Better Safe than Sorry? Ex Ante and Ex Post Moral Hazard in Dynamic Insurance Data.' SSRN 1260168.

---

## Related posts

- [Applying Bonus-Malus to Driving Behaviour, Not Just Claims](/telematics/techniques/2026/04/01/weekly-dynamic-telematics-bms-bonus-malus-driving-behaviour/) — the Yanez/Guillen/Nielsen ASTIN 2025 paper applying BMS to telematics signals weekly, which implicitly sidesteps the underreporting problem by observing behaviour directly
