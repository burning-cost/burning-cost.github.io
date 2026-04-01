---
layout: post
title: "How Long Does an Inflationary Shock Last? A Gamma-Decay Model for Claims Cost Persistence"
date: 2026-03-31
categories: [pricing, reserving]
tags: [claims-inflation, trend, motor, severity, reserving, shock-persistence, gamma-decay, insurance-trend, structural-break, Liu-Zhou-2026, arXiv-2603-23707]
description: "Fitting one aggregate trend to UK motor claims 2019–2024 embeds a single implicit decay rate across parts shortage, labour shortage, and social inflation — components that normalise at completely different speeds. A pandemic mortality paper gives us the mathematics to do it properly."
author: burning-cost
---

UK motor average claims cost hit approximately £4,900 in 2024. The 2019 figure was around £3,600. That is 36% over five years, but the composition of that 36% matters more than the headline. Parts costs spiked then largely recovered. Bodyshop labour costs rose and stayed up. Credit hire normalised as rental fleets recovered. Injury settlement trends have crept upward but not catastrophically — in the UK at least.

Pricing and reserving actuaries are being asked the same question every quarter: when does this normalise? The answer depends entirely on *which component* you are talking about. A recently published paper on pandemic mortality shocks — Liu and Zhou, arXiv:2603.23707, submitted 24 March 2026 — gives us a defensible mathematical framework for that heterogeneity, and it maps directly onto the claims inflation problem.

---

## The problem with one trend line

Standard severity trend analysis fits a log-linear slope across the available accident years. If your data runs 2019–2024, the slope is estimated from all of it. Apply it forward two years to project the 2026 rating period. Simple, auditable, defensible.

Except: that single slope embeds an implicit assumption about how the 2021–2022 inflation shock decays. When you fit a straight line through data that contains a shock followed by partial recovery, the slope is a weighted average of the shock's entry and exit. Whether that average is the right number to project depends on where you are on the decay curve for each component.

If parts costs normalised by mid-2023 but labour costs are still elevated, the aggregate trend line is simultaneously:

- **Too pessimistic at short horizons** — the fast-decaying parts component is pulling the line upward, but it has already returned to trend
- **Too optimistic at medium horizons** — the slow-decaying labour component persists longer than the aggregate line implies

This is a reserving error in opposite directions at different time horizons, which is the worst kind: it does not cancel out across accident years. The recent accident years, still developing, get wrong IBNR estimates at both ends.

The [insurance-trend](https://github.com/burning-cost/insurance-trend) library addresses half of this with structural break detection — PELT algorithm on the log-transformed series, piecewise trend from the final segment. That is the right approach for handling the COVID frequency distortion. But structural breaks assume the trend shifts once and stabilises. A decaying shock is a different animal: the apparent trend changes continuously as the shock fades, at a rate that depends on which components are in your aggregate.

---

## What pandemic mortality data tells us about decay

Mortality shocks give us unusually clean evidence about how a shock of known origin decays over time, because we can observe multiple post-shock years and because the data is stratified by cause of death. COVID-19 is the natural experiment.

Liu and Zhou fit a stochastic mortality model to US CDC data from 1968 to 2023 — 56 years, 13 age groups, 6 cause-of-death categories. The model separates a common long-run trend, cause-specific deviations from that trend, and a jump component that decays through a gamma-density function. They fit each component separately and compare the decay patterns across causes.

The results are stark. Three distinct regimes emerge:

**Fast decay:** Direct COVID/other mortality (their CoD 6). Large initial shock — for males aged 65–74, the 2019–2020 mortality rate increase was 77.0%. But by 2022–2023, the excess had fallen sharply. The acute phase dominates; the shock peaks at tau=0 and decays quickly.

**Moderate but persistent:** Circulatory disease (CoD 3). Smaller initial shock — +6.4% in 65–74 males — but it *peaks in 2021–2022*, a year after the initial COVID shock, then decays slowly. Still materially elevated in 2023 US data, three years after the pandemic onset. The mechanism is well-understood: deferred GP visits, disrupted cardiac monitoring, long COVID cardiovascular sequelae.

**Very slow decay:** External causes — accidents, drug overdoses, mental health-related mortality (CoD 5). Driven by behavioural secondary effects of the pandemic. These were still elevated in 2022–2023 and showed no clear peak in the data window. Structurally altered behaviour does not normalise on a medical timeline.

This is not a modelling artefact. It is visible in the raw data. The point is that three components of the same original shock — COVID-19 — follow completely different post-shock trajectories. Fitting one aggregate mortality trend to 2019–2023 data would blend them into a single weighted-average decay that is wrong for all three individually.

---

## The gamma-decay framework

Liu and Zhou model the lingering effect of a shock using a gamma density function. For each cause c and years elapsed since the shock tau, the proportion of the original jump effect remaining is:

```
pi_c(tau) = gamma_c * beta_c^alpha_c * tau^(alpha_c - 1) * exp(-beta_c * tau)
```

Two parameters govern the shape:

**alpha_c** controls when the effect peaks. With alpha=1, the impact is at its maximum at the moment of the shock and decays immediately (exponential decay). With alpha=2, the peak is delayed — secondary damage accumulates before recovery begins. With alpha=5, the peak is much later, representing a slow-building structural response.

**beta_c** controls how fast the effect dissipates once it has peaked. Higher beta means faster decay. A shock with alpha=2 and beta=2 peaks at tau=0.5 years and is substantially gone by tau=3. A shock with alpha=2 and beta=0.4 peaks later and persists past tau=8.

The function integrates to gamma_c, bounding the total lingering impact. alpha/beta gives the peak timing.

The estimated parameters for US male mortality (their Table 2) show the three regimes concretely:

| Cause | Estimated alpha | Estimated beta | Peak timing | Character |
|---|---|---|---|---|
| CoD 6 (COVID direct/Other) | 2.21 | 1.21 | tau ~1yr | Moderate persistence, then normalises |
| CoD 3 (Circulatory) | 2.98 | 1.38 | tau ~1.5yr | Delayed peak, slow decay |
| CoD 5 (External) | 0.75 | 33.4 | immediate | Peaks at tau=0, very slow decay |
| CoD 1 (Infectious) | 4.21 | 1.08 | tau ~3yr | Long lag, moderate persistence |
| CoD 4 (Respiratory) | 4.02 | 1.33 | tau ~2.3yr | Delayed, then normalises |
| CoD 2 (Cancer) | 1.45 | 0.52 | tau ~0.9yr | Slow but jump is tiny |

The External Causes result (CoD 5) is the most instructive: alpha < 1 means the function has no interior peak — it starts at its maximum and decays from there. But beta = 33.4 in conjunction with the very small alpha produces an extremely slow tail. The mathematics is capturing exactly what the epidemiology would predict: drug overdose rates during and after a pandemic are determined by social and economic conditions that take years to shift.

You can visualise the three decay regimes with five lines of Python:

```python
import numpy as np
from scipy.stats import gamma

tau = np.linspace(0.01, 8, 300)  # years elapsed since shock

# Fast: parts shortage analogue
fast   = gamma.pdf(tau, a=2.0, scale=1/2.0)

# Moderate: circulatory / labour analogue
medium = gamma.pdf(tau, a=3.0, scale=1/1.4)

# Slow: external / structural analogue
slow   = gamma.pdf(tau, a=1.5, scale=1/0.5)

# Normalise each to peak=1 for visual comparison
fast   /= fast.max()
medium /= medium.max()
slow   /= slow.max()
```

At tau=3 — where UK motor is roughly sitting if you date the shock to Q1 2021 — the fast component is near zero, the medium component is substantially decayed, and the slow component is still above 50% of its peak effect. An aggregate trend fitted across 2021–2024 cannot distinguish between these.

---

## The mapping to UK motor claims inflation

The parallel is direct. UK motor claims costs 2021–2024 were a multi-component shock with identifiable drivers, each with its own post-shock trajectory. The gamma-decay framework gives each a mathematical character.

| UK motor cost component | Mortality analogue | Decay character | Where we are (tau ~3yr) |
|---|---|---|---|
| Semiconductor / parts shortage 2021–22 | CoD 6 (COVID direct) | Fast: alpha ~2, beta ~2. Peaked tau~1yr | Largely resolved. ABI repair cost data shows parts inflation normalising through 2023. |
| Bodyshop labour shortage | CoD 3 (Circulatory, persistent) | Slow: alpha >2, beta small. Structural | Still elevated. IMI (Institute of the Motor Industry) reported 23,000+ unfilled technician positions in 2024. Apprenticeship pipeline 3–5 years. |
| Social / legal inflation | CoD 5 (External, persistent) | Very slow: peaks immediately, fades slowly | Creeping. UK social inflation is materially below US levels — perhaps 2–4% additional severity trend — but it is directionally the same structural shift in litigation behaviour. |
| Credit hire costs | CoD 1 (Infectious, initially negative) | Fast reversal | Normalised by 2022–23 as rental fleet supply recovered. Pulled aggregate inflation down. |
| General CPI component | CoD 2 (Cancer, minimal shock) | Trend-following, no discrete jump | Embedded in the structural trend. Separate from the 2021–22 shock. |

This table is not claimed to be calibrated — we do not have 50 years of component-level motor data. The point is the structure. Each component has a different alpha and beta. When you fit an aggregate severity trend to total cost, you implicitly assign all components the same decay curve: the weighted average. That average is wrong for every single component individually.

Liu and Zhou demonstrate the cost of getting this wrong. When they fit their model without cause-specific jump components — forcing a single aggregate shock — the estimated long-run age sensitivity parameter absorbs the shock's asymmetric age effects as if they were permanent structural changes in mortality improvement rates. The estimates come out flatter. Projections over the following decade are systematically wrong.

The claims inflation parallel is exact: fit a single log-linear trend to aggregate severity 2019–2024 without decomposing the shock, and the fitted slope embeds the shock's aggregate decay as if it were a permanent shift in the underlying inflation regime. You project a trend rate that is somewhere between the fast-decaying components and the slow-decaying ones — and wrong for your actual projection horizon.

---

## What this means for reserving

The PRA's June 2023 Dear Chief Actuary letter on claims inflation explicitly flagged concern about insurers not adequately capturing the persistence of cost pressures in their reserve estimates. This is the mechanism behind that concern, made specific.

IBNR triangles fitted over the 2019–2024 period contain three types of accident-year data:

1. **Pre-shock years (2019–2020):** Stable development pattern at pre-inflation costs
2. **Shock years (2021–2022):** Elevated development with inflationary distortion
3. **Post-shock years (2023–2024):** Still-elevated but partially decaying, with heterogeneous component behaviour

Standard chain ladder assumes each accident year follows the same development pattern. If 2021 closed at elevated costs while 2024 will close at lower costs as the fast-decaying components normalise, the chain ladder applied uniformly overstates IBNR for recent accident years in those fast-decaying components and understates it for the slow-decaying components.

The gamma-decay framework gives you a stress-testing language. Instead of "does the inflation normalise?" the question becomes: what alpha and beta describe each component, and what are the IBNR implications under different parameter combinations?

Concretely:

- **Fast normalisation assumption (alpha=2, beta=2 for all components):** Parts-like decay everywhere. The 2024 accident year settles close to pre-shock severity. IBNR is lower than the chain ladder implies.
- **Slow normalisation assumption (alpha=3, beta=0.6 for labour component):** Labour costs remain elevated through 2027. The 2024 and 2025 accident years continue to develop at above-trend severity. IBNR is higher.
- **Component-specific (differentiated):** Fast parts, slow labour, structural social inflation. This is the most defensible position and gives a range by component.

The scenario analysis in the mortality paper makes a related point about aggregate vs. endemic shocks. In their Scenario II — frequent mild shocks (4x frequency, 0.5x severity) versus the baseline single large shock — the portfolio risk (standard deviation) falls from 0.67 to 0.54. Many small normalising shocks produce less tail risk than one big one that persists. The claims inflation equivalent: sustained mild inflation at 3–4% pa is more predictable and less dangerous for reserving than one spike-and-crash cycle, because the aggregate trend fitter handles a sustained trend better than a decaying transient.

---

## What to do about it

**1. Decompose before trending.**

If you have access to component-level claims data — repair cost versus injury settlement versus credit hire, at minimum — fit separate trends to each. The [insurance-trend](https://github.com/burning-cost/insurance-trend) library's `SeverityTrendFitter` runs on any time series with structural break detection included. Run it on each component separately, then combine the projected components into a total severity projection.

```python
from insurance_trend import SeverityTrendFitter

# Fit repair cost inflation separately from injury settlement inflation
repair_result = SeverityTrendFitter(
    periods=repair_periods,
    severities=repair_severities,
    weights=repair_claim_counts,
).fit(detect_breaks=True)

injury_result = SeverityTrendFitter(
    periods=injury_periods,
    severities=injury_severities,
    weights=injury_claim_counts,
).fit(detect_breaks=True)

print(f"Repair severity trend:  {repair_result.trend_rate:.1%} pa")
print(f"Injury severity trend:  {injury_result.trend_rate:.1%} pa")
```

The structural break detection will identify whether the repair cost series shows a break around Q1 2021 and another potential break as normalisation occurred in 2023. The injury trend will likely show a different break profile. That difference is informative — it is the alpha/beta differential made visible in your own data.

**2. Add a shock residual layer.**

The current insurance-trend library fits log-linear or local linear trends. The gamma-decay framework suggests a two-layer structure: fit the structural trend (the long-run component not affected by the shock), then model the shock residual as a separate term with its own decay assumption. This does not require implementing the full 3WPF-CLJ model. A simple approach:

- Define T_J = Q1 2021 (UK motor shock onset)
- For each component, estimate the "shock excess" above pre-2020 trend
- Apply a gamma-decay multiplier to project the remaining shock excess forward
- Add to the structural trend

The gamma multiplier for a given alpha, beta, and tau is `scipy.stats.gamma.pdf(tau, a=alpha, scale=1/beta)` normalised to 1 at tau=0. The sensitivity of your projection to the alpha and beta assumptions is your parameter risk, and it can be quantified and disclosed.

**3. Scenario-frame your reserve range.**

Rather than presenting a point estimate with a vague "inflation risk" disclosure, frame the IBNR range explicitly as a function of decay assumptions:

- Central: alpha=3, beta=1.4 for labour (moderate persistence, peaks tau~1.5yr from 2021, substantially resolved by 2026)
- Adverse: alpha=2, beta=0.5 for labour (slower decay, still material 2027+)
- Optimistic: alpha=2, beta=2 for labour (rapid normalisation, approaching resolved by end 2025)

This is a defensible and transparent way to present reserving uncertainty on an inflationary shock that has not yet fully worked through the triangles.

**4. Separate structural trend from shock residual when pricing.**

For a 2026 rate filing, you need to project claims cost to the 2026–2027 rating period. Your historical data includes the shock. If you fit a single trend to 2019–2024 and project it forward, you are implicitly assuming the shock persists at the same rate through 2027. If parts costs have already normalised, this overstates your rate need. If labour costs persist, this might understate it.

The correct approach: identify what proportion of current above-baseline severity is attributable to each component, apply component-specific decay to project each to the rating period, and use the sum as your projection basis. The uncertainty is larger than a point estimate implies, but it is structured uncertainty that you can defend.

---

## The honest limitations

The gamma-decay model will not tell you exactly when UK motor costs normalise. The parameters alpha and beta for each component are not directly estimable from three years of post-shock UK data — you would need external priors from macroeconomic data (ONS PPI series for vehicle parts, IMI data on technician vacancy rates), or Bayesian estimation, or both.

The Liu and Zhou model was calibrated on 56 years of US CDC mortality data with multiple historical shock events providing some information about pre-COVID jump characteristics. UK motor has one shock event and at most ten years of quarterly severity data. The standard errors on any estimated alpha and beta would be enormous.

What the framework gives you is not a precise calibration. It is a structured way of thinking about the problem — one that distinguishes "which component" from "the shock" and asks what shape that component's decay curve takes. That distinction produces better questions, better scenario analysis, and more defensible reserve disclosures than fitting one line and calling it done.

The pandemic mortality paper — Liu, Y. & Zhou, K.Q., "The Long Shadow of Pandemic: Understanding the Lingering Effects of Cause-Specific Mortality Shocks" (arXiv:2603.23707, March 2026) — is not a claims inflation paper. But the mathematical framework it establishes is the right one for the problem, and the empirical evidence from three years of post-COVID mortality data is the clearest demonstration we have that heterogeneous decay is not just theoretically plausible but quantitatively real.

The gamma function is not the complicated part. The hard part is collecting your claims cost by component, being honest about where you are on each decay curve, and presenting the resulting range to a pricing committee that would prefer a single number. That is still a judgement problem. The framework just makes the judgement explicit.
