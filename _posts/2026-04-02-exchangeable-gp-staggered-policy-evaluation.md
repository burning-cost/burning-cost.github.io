---
layout: post
title: "When Parallel Trends Break: Gaussian Processes for Staggered Rate Change Evaluation"
date: 2026-04-02
categories: [techniques, causal-inference]
tags: [causal-inference, staggered-adoption, gaussian-processes, diff-in-diff, sdid, synthetic-controls, rate-change-evaluation, panel-data, pricing, personal-lines, insurance-causal-policy, arXiv-2602-21031, Gevorgyan, Kalogeropoulos, Alexopoulos]
description: "Gevorgyan et al. propose exchangeable multi-task Gaussian processes for causal effect estimation in staggered-adoption designs. The method handles nonlinear trends that break SDID's parallel-trends assumption. We explain what this means for UK pricing teams evaluating rate changes across regions that rolled out at different dates."
math: true
author: burning-cost
---

Staggered adoption is the default in UK personal lines pricing. A motor insurer running a rating algorithm change does not flip every region on the same day — it pilots in the North East in Q1, extends to the Midlands in Q2, completes national rollout by Q4. The same is true for telematics scheme launches, broker commission restructures, and any change that requires system migration or FCA notification. You end up with a panel where different units adopt at different times, and you need to estimate the causal effect of the change.

[`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) handles this with Synthetic Difference-in-Differences (SDID). SDID is good. We published benchmarks showing it removes a 3.8pp naive before-after bias on a synthetic UK motor book. But SDID relies on a parallel trends assumption: in the absence of treatment, treated and control units would have evolved on parallel (linear) trajectories. When that assumption fails — when the underlying trend is nonlinear, regime-dependent, or unit-specific — SDID's point estimates are biased and the pre-treatment fit test may not flag the problem.

Gevorgyan, Kalogeropoulos, and Alexopoulos (arXiv:2602.21031, February 2026) propose a different approach: model the joint outcome evolution with an **exchangeable multi-task Gaussian process**, use the posterior predictive distribution over untreated counterfactual trajectories, and compute treatment effects as the difference between observed and counterfactual.

---

## What exchangeability buys you

The GP framework is built around two scenarios. The simpler case — one treated unit, one intervention time — has a clean GP intuition: fit a GP to control unit outcomes, condition on pre-treatment outcomes in the treated unit, and the posterior predictive gives you the counterfactual trajectory for the treated unit post-intervention. The credible interval is honest about extrapolation uncertainty in a way that bootstrap procedures for SDID typically are not.

The staggered case is more interesting and more relevant to insurance. Multiple units adopt at different times; the "never treated" set shrinks as the panel progresses. The exchangeability assumption says that units are exchangeable in distribution — not that they have identical trajectories, but that their outcome functions are drawn from a common GP prior. This is weaker than parallel trends. Parallel trends says the slope is the same; exchangeability says the functions come from the same distribution, which allows different shapes as long as they are draws from the same prior.

The causal estimand is a posterior predictive distribution over the cumulative treatment effect, integrating over the uncertainty in the counterfactual trajectories. The output: a pointwise treatment effect estimate at each post-treatment period, plus credible intervals. The cumulative effect is the sum over the post-treatment window — the right quantity for assessing whether a rate change actually moved aggregate loss ratio.

---

## The parallel trends problem in UK insurance

Parallel trends breaks in predictable ways in UK personal lines data.

**Regulatory step-changes.** The FCA's GIPP pricing remedies (PS21/5) came into force in January 2022. Renewal premium inflation for long-standing customers was mechanically eliminated across all insurers simultaneously. Any panel spanning 2021–2023 has a step-change in the outcome variable that is not a market trend — it is a structural break affecting treated and control units differently depending on their prior book composition. SDID time weights should absorb some of this, but a GP prior that can model step-change behaviour in the latent trend is strictly more expressive.

**Motor claims inflation volatility.** UK motor claims severity ran at approximately 12% in 2022, stabilised to around 4% by late 2024, and remains uncertain heading into 2026 (ABI data). If the claims inflation trajectory is different across regions — because of regional variation in repair network access, parts pricing, or bodily injury claim propensity — then regional outcome trends are not parallel even before treatment. A GP with a flexible kernel (squared exponential, Matérn 5/2) can capture this variation in the counterfactual model.

**Pilot design contamination.** Insurers typically choose pilot regions that are operationally convenient, not statistically matched to control regions. A North East pilot before a national rollout is convenient for a team based in Newcastle; it is not a randomised experiment. The treated and control regions have different baseline trends, different mix, and different broker concentration. SDID's unit weights try to construct a synthetic control from the control pool, but if the pre-treatment fit is poor — which you can see in the placebo tests — you are extrapolating.

---

## What the paper actually delivers

The methodology is sound. The placebo-style pre-treatment validation — fitting the GP to artificial intervention dates in the pre-treatment period and checking that the detected effect is near zero — is the right diagnostic. The kernel function sensitivity analysis (testing multiple kernels for robustness) is appropriate for a practitioner context where the true correlation structure of outcomes is unknown.

The implementation gap is significant. There is no code released with the paper as at April 2026. The model requires a GP library — GPyTorch (Python) or GPflow are the obvious candidates — and a custom multi-task likelihood layer with the exchangeability constraint. This is not a week's implementation work; getting the posterior inference correct for the staggered case, where the set of "not yet treated" units changes at each adoption wave, requires careful handling of the conditioning sets.

The paper uses synthetic datasets for validation. The authors do not demonstrate the method on a real panel with a known intervention effect, which makes it hard to assess performance relative to SDID on realistic data. The benchmark should exist before we integrate this into [`insurance-causal-policy`].

---

## How this relates to SDID

SDID and the GP approach are not in competition for the same use case. The right framing is:

- **Use SDID** when the panel has sufficient pre-treatment periods (seven or more), the number of control units is moderate (20–200), and the pre-treatment parallel trends test passes. This is the common case for UK motor or home rate change evaluation at a segment or regional level.

- **Use the GP approach** when the parallel trends assumption is visibly violated (poor pre-treatment fit in SDID placebo tests), the intervention is embedded in a period of structural change (post-GIPP, post-Ogden rate change, post-COVID return to normal), or when you want a full posterior over the counterfactual trajectory rather than a point estimate with a bootstrapped confidence interval.

The GP approach is also more natural for **heterogeneous effect estimation across time**. SDID produces an average treatment effect on the treated (ATT) over the post-treatment window. The GP gives you a pointwise estimate at each post-treatment period, which matters for assessing whether a rate change effect appeared immediately or built over 6–12 months as the book turned over.

---

## What a UK pricing team needs to implement this

The data requirements are identical to SDID: a balanced (or near-balanced) panel of segments or regions, with outcome metrics (loss ratio, pure premium, claim frequency) at a quarterly or monthly grain, observed pre- and post-adoption for each unit.

The computational requirements are heavier. SDID on a 100-segment, 12-period panel runs in seconds. GP inference on the same panel — with a full covariance matrix over units and time — scales as $O(N^3)$ in the number of observations, with $N = \text{units} \times \text{periods}$. For 100 segments and 20 quarters, $N = 2000$ and full GP inference is manageable. For 1000 segments and 40 quarters, $N = 40000$ and you need sparse GP approximations or structured kernel exploits, neither of which is trivially available in the insurance context.

The authors test multiple kernels. For insurance panel data, we would start with a Matérn 5/2 kernel (smooth but not infinitely differentiable, which better reflects the irregular trajectory of loss ratios than a squared exponential). The multi-task structure requires a kernel over units (capturing the exchangeability assumption) and a kernel over time (capturing temporal autocorrelation). The product of unit and time kernels is the standard multi-task GP construction.

---

## Verdict

The exchangeable GP framework is theoretically stronger than SDID for panels with nonlinear or heterogeneous trends. In UK insurance, where post-GIPP structural breaks are a real feature of 2022–2024 pricing data, this matters.

We are not extending [`insurance-causal-policy`](https://github.com/burning-cost/insurance-causal-policy) to include GP-based counterfactuals until the authors release code and demonstrate performance on a panel with a known causal effect. The implementation complexity is non-trivial, the computational cost is higher than SDID, and the practical improvement over a well-specified SDID on insurance data is unproven. "Theoretically more expressive" is not a sufficient condition for production inclusion.

What we will do: add a parallel trends diagnostic to `insurance-causal-policy` that flags when the pre-treatment SDID fit is poor enough to warrant the GP approach. If the pre-treatment placebo rejection rate exceeds a threshold, the output will recommend reviewing the GP literature, with this paper as the reference.

The paper is open access. For teams with a mature SDID workflow and an adjacent GP infrastructure — which in practice means teams using GPyTorch for other purposes — the methodological extension is worth prototyping on real data from a known rate change event. The validation approach (placebo tests on artificial treatment dates) is directly portable and well explained.

---

## References

Gevorgyan, H., Kalogeropoulos, K., and Alexopoulos, A. (2026). 'Exchangeable Gaussian Processes for Staggered-Adoption Policy Evaluation.' arXiv:2602.21031. Submitted February 2026.

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., and Wager, S. (2021). 'Synthetic Difference-in-Differences.' *American Economic Review* 111(12):4088–4118.

---

## Related posts

- [Your Rate Change Didn't Prove Anything](/2026/03/13/your-rate-change-didnt-prove-anything/) — SDID for rate change evaluation and the FCA Consumer Duty evidence gap
- [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy) — SDID and Doubly Robust Synthetic Controls for insurance panels
