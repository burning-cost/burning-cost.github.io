---
layout: post
title: "How Long Does an Inflationary Shock Last? Lessons from Pandemic Mortality Persistence"
date: 2026-04-02
categories: [insurance-pricing, claims-inflation, techniques]
tags: [claims-inflation, trend-analysis, motor, reserving, gamma-decay, persistence, supply-chain, labour-costs, insurance-trend, arXiv-2603-23707, Liu, Zhou]
description: "A new mortality model from Liu & Zhou (2026) shows that cause-specific shocks decay heterogeneously — some fast, some slow. The analogy to UK claims inflation is exact, and the implications for reserving are uncomfortable."
math: true
author: burning-cost
---

The UK motor market spent 2022–2024 arguing about when claims inflation would normalise. Parts costs spiked with semiconductors, labour rates rose with wage inflation, credit hire spiralled, and everyone had a different view on which components would return to trend and when. Most reserve bases picked one blended inflation rate, applied it forward, and called it done.

That is almost certainly wrong — not because the inflation rate was wrong, but because a single number cannot capture what is actually happening when multiple cost components each have their own persistence dynamics. A new mortality paper from Liu and Zhou (arXiv:2603.23707, March 2026) gives us a principled framework for thinking about exactly this problem, and the analogy is close enough to be practically useful.

---

## The mortality model, briefly

Liu (Nebraska-Lincoln) and Zhou (Waterloo) study how cause-specific mortality shocks decay over time. Their starting point is that existing mortality models treat shocks in one of two ways: either purely transitory (one-period spike, then gone) or fully permanent (the shock shifts the baseline forever). Neither description fits COVID-19 well, and their intuition is that most real shocks are neither — they linger and then fade, at a rate that depends on the underlying cause.

Their formal model embeds a gamma-density decay function into a stochastic mortality framework. The excess log-mortality from a shock at time $t_0$ follows:

$$\delta(s) \propto s^{\alpha-1} e^{-\beta s}, \quad s = t - t_0$$

This is the gamma probability density in disguise, parameterised by shape $\alpha$ and rate $\beta$. The shape parameter $\alpha$ determines what kind of decay you get:

- $\alpha = 1$: exponential decay from the moment of the shock — impact peaks immediately and fades
- $\alpha > 1$: delayed peak — impact builds after the shock, reaches a maximum at $s = (\alpha-1)/\beta$, then declines
- Large $\alpha$, small $\beta$: the effect lingers for years before subsiding

This is a more expressive functional form than the standard alternatives, and critically, it can be fitted separately to each cause of death rather than to the aggregate.

We are not recommending you build this mortality model. The full implementation involves a Gaussian mixture likelihood with 135 parameters fitted by BFGS optimisation on cause-coded US death data. It is a mortality tool, and we are not in life insurance. What we are saying is that the *conceptual structure* maps exactly onto a problem general insurance pricers encounter every time they build a trend assumption.

---

## The empirical results, translated

Liu and Zhou apply their model to US COVID mortality data disaggregated by cause. The findings are striking, and the cause-level patterns map cleanly onto a claims inflation decomposition.

**High initial shock, fast decay.** COVID-direct and "other" causes (their cause group 6) showed the largest initial surge — males aged 65–74 saw mortality roughly 77% above the 2019 baseline in 2020 — followed by a relatively sharp reversal. By 2023 this component was largely normalised. The decay function for this group has high $\beta$: fast rate, short half-life.

**Moderate initial shock, slow persistent decay.** Circulatory disease mortality (cause group 3) peaked later — 2021 to 2022 rather than 2020 — and remained elevated at the 2023 data cutoff. This is a delayed-peak pattern: $\alpha > 1$, with the maximum effect arriving roughly 12–18 months after the initial shock. The cause is plausible: cardiovascular deaths associated with COVID sequelae, deferred treatment, and long-term post-acute effects were still accumulating two years after the acute wave.

**Shock with a rapid reversal.** Infectious disease mortality (non-COVID, cause group 1) went *negative* during COVID — mask mandates and reduced social mixing protected against influenza and other respiratory infections, temporarily cutting death rates. That benefit disappeared within roughly 12 months of restrictions lifting. A textbook $\alpha = 1$ decay in both directions.

**Near-zero impact.** Cancer mortality barely moved. The data shows changes consistently within ±5% across ages. Either cancer mortality is driven by factors orthogonal to the pandemic, or the timescales involved (cancer progression runs over years to decades) mean the pandemic's effect has not yet appeared in the data. Either way: effectively no signal.

The key result is that these patterns are genuinely different, and fitting a single decay rate to aggregate all-cause mortality would produce a weighted average that is wrong in both directions — overstating persistence for the fast-decaying components, understating it for the slow ones.

---

## The claims inflation parallel

UK motor claims inflation from 2021 onwards has the same structure. Multiple cost components each inflated for partially overlapping reasons, and those reasons have different persistence dynamics.

**Direct vehicle parts costs** (analogous to cause group 6 — high shock, fast decay). New car semiconductor shortages from 2021 drove up used car prices and parts costs. The ABI data showed average repair costs up roughly 30–40% at the peak in 2022–2023. As semiconductor supply normalised through 2023–2024, this component began recovering. Not fully — structural factors in parts supply chains have not vanished — but the acute spike is largely past. Shape parameter: probably $\alpha$ near 1 or slightly above, meaning the decay started almost immediately once supply constraints eased.

**Labour costs** (analogous to cause group 3 — delayed peak, slow decay). Vehicle technician shortages emerged more gradually and have not reversed. The 2021–2023 wage inflation cycle hit bodyshop labour particularly hard: the skilled trades shortage predates COVID and the pandemic accelerated it through early retirements. ABI and Thatcham data both showed labour rates still rising through 2024. This is the high-$\alpha$ pattern: the peak impact arrived later than the initial shock and the decay rate is slow.

**Credit hire** (analogous to cause group 1 — spike then reversal). Rental fleet availability collapsed in 2021 as fleets were not replenished, which drove credit hire rates sharply upward. Fleet normalisation through 2022–2023 brought this component back more quickly. There are structural elements (litigation behaviour, third-party capture economics) that complicate a clean reversal, but the fleet-driven spike is largely resolved.

**Social inflation / large loss trends** (loosely analogous to cause group 5 — low initial signal, slow creep). This is the hardest component to track in UK motor because it shows up as a slow drift in large loss frequencies rather than a sharp spike. It is also the hardest to attribute: behavioural change, access to justice reforms, CMC market evolution, and MedCo are all contributing. The pattern here is low $\alpha$, low $\beta$ — small initial signal, but very long tail.

We are offering this as an analogy, not a calibration. We do not have Liu and Zhou's fitted parameters for UK motor claims inflation, because that dataset does not exist in the form their model requires. The parallel is conceptual: the reason to decompose is the same reason they decompose, and the decay shapes they identify in mortality are recognisably present in claims cost data.

---

## What this means for reserving

The reserving implication is uncomfortable.

Standard reserve development assumes a trend rate. You fit one number — perhaps 8% pa aggregate claims inflation — and project it forward. The implicit assumption is that this trend rate is stable over the projection period: the same forces that drove inflation historically will drive it at the same rate going forward.

If the Liu and Zhou framework is right — and the empirical evidence from both mortality and claims data suggests it is — then the aggregate trend rate is a weighted average of components with heterogeneous decay rates. At the time you observe the trend, the weights reflect the current mix of active inflationary forces. As time passes, the fast-decaying components diminish and the slow-decaying ones persist. The trend rate falls, then stabilises at whatever the structural baseline is for the persistent components.

This means:

1. **Aggregate trend fitting biases near-term projections upward.** If parts costs are fast-decaying and labour is slow-decaying, and you fit one trend to their weighted average at the peak, you are projecting the fast-decaying component forward longer than its actual half-life. Your IBNR for accident years 2022 and 2023 is probably too high — you are assuming parts inflation persists at levels that had already partially reversed by the development date.

2. **Aggregate trend fitting biases long-run projections downward.** Conversely, if you back off the aggregate trend to account for observed normalisation, you may be inadvertently assuming the slow-decaying structural components (labour, large loss) have also normalised. They have not. Your ultimate loss for accident years 2024–2025 may be understated.

3. **Reserve uncertainty is asymmetric by component.** Parts costs and labour costs have different uncertainty profiles. Parts cost recovery depends on supply chain geography, semiconductor cycles, and EV transition dynamics — high variance, plausibly mean-reverting. Labour cost persistence depends on training pipeline economics, trade union dynamics, and immigration policy — lower variance, probably not mean-reverting to pre-2021 levels. Treating them identically in your reserve uncertainty analysis is wrong.

None of this is new intuition — good reserving actuaries have been decomposing their trend assumptions for years. What the Liu and Zhou paper adds is the mathematical framing that clarifies *why* aggregate fitting is wrong and what the correct functional form for component persistence should look like.

---

## Connecting to the insurance-trend library

The `insurance-trend` library's `SeverityTrendFitter` and `FrequencyTrendFitter` handle component-level trend fitting, but they fit log-linear trends rather than gamma-decay functions. That is appropriate when you believe the inflationary shock has already passed and you are in the decay phase — log-linear trend is roughly equivalent to the exponential tail of the gamma-decay function when $\alpha$ is close to 1.

For components that are clearly in a decay phase (parts costs, credit hire), `SeverityTrendFitter` with a break point at the supply chain shock date is the right tool. Fit a trend up to the break, fit a separate (probably lower, possibly negative) trend after the break, and use the post-break trend for your projection.

The `MultiIndexDecomposer` is more relevant for components where you have an external index to track the driver — ONS AWE for labour costs, ONS UK Used Car Prices for parts, for instance. Decomposing severity trend against those indices lets you isolate the residual superimposed inflation and, critically, lets you build forward scenarios that are anchored to external forecasts for each index rather than a single blended rate.

Neither class currently fits a gamma-decay function explicitly — that would require a bespoke implementation of the Liu-Zhou persistence model, which is a non-trivial project. But the conceptual architecture of component-level analysis that the library supports is exactly what the persistence heterogeneity argument requires. You do not need the gamma-decay functional form to benefit from doing the decomposition; you just need to stop fitting one number to everything.

---

## What we would actually do

If you are building a 2025 reserve basis for a UK motor book, the immediate practical steps are:

1. **Decompose your paid development by cost component.** Parts, labour, credit hire, and large losses need separate trend analysis. If you cannot do this exactly from your data, use industry indices (ABI, Thatcham) to calibrate relative weights.

2. **Set different forward assumptions by component.** Parts costs: use an external forecast anchored to supply chain and EV parts pricing trends. Labour: assume persistence — assume mid-single-digit continuing increase — trades wage inflation had not normalised by end-2024, and the trained technician pipeline is thin. Credit hire: assume normalised unless you have specific evidence of continued fleet pressure. Large loss: most uncertain; scenario-test against a low, central, and high social inflation assumption.

3. **Do not let the aggregate trend be your single output.** It will be wrong on both sides simultaneously.

The Liu and Zhou paper is primarily a mortality paper. But the principle — that persistence is cause-specific, that decay shapes vary, and that aggregate models necessarily obscure this heterogeneity — is transferable. We think it is the right framing for anyone who has spent the last three years staring at blended claims inflation numbers and wondering why their reserves keep moving.

The paper is at [arXiv:2603.23707](https://arxiv.org/abs/2603.23707).
