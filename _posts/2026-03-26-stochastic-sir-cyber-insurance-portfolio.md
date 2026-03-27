---
layout: post
title: "Cyber Portfolio Risk Is an Epidemic Problem, Not a Frequency-Severity One"
date: 2026-03-26
categories: [pricing, cyber, research]
tags: [cyber-insurance, contagion, sir-model, systemic-risk, ransomware, lockbit, aggregate-loss, portfolio-risk, catastrophe-modelling, correlated-losses]
description: "A stochastic SIR model calibrated to LockBit ransomware data shows why treating cyber losses as independent events badly underestimates portfolio-level risk."
---

Standard frequency-severity modelling assumes that one policyholder's claim tells you nothing about the next. For motor, household, and most commercial lines, that independence assumption is close enough. For cyber it is wrong in a way that matters.

When LockBit ransomware tore through Île-de-France businesses between May and July 2024, infected firms did not arrive independently into your claim queue. They arrived in clusters, seeded by common attack infrastructure, and then propagated through supplier relationships, shared IT services, and affiliated subsidiaries. That is an epidemic, not a Poisson process.

A new paper from CREST/ENSAE — Hillairet, Lopez, and Sopgoui, arXiv:2603.15369, submitted March 2026 — models it as one.

## What the SIR Model Actually Says

SIR stands for Susceptible-Infected-Recovered. Originally designed for disease spread through populations, the three-compartment structure maps cleanly onto a portfolio of insured businesses:

- **Susceptible**: firms exposed but not yet hit
- **Infected**: firms currently under attack or in active incident
- **Recovered**: firms that have resolved their incident (and, in this version, exit the pool)

The paper's contribution is not simply borrowing the SIR label. They construct a **stochastic multi-group SIR** where transmission rates β and recovery rates γ vary by firm size and evolve over time via Cox-Ingersoll-Ross (CIR) processes. That is: the epidemic's intensity is itself a random variable, reflecting the genuine uncertainty in how aggressively a given ransomware campaign propagates.

The force of infection at time t is:

```
Yₜ = (1/N₀) × Σₖ βₖ,ₜ · k · Iₖ,ₜ
```

where k is firm size (number of subunits) and Iₖ,ₜ is the count of infected firms of that size. Larger firms contribute disproportionately to systemic spread because they have more subsidiaries and subunits that can carry secondary infections internally.

Two transmission channels are modelled separately:

1. **External contagion**: a Cox process with intensity driven by Yₜ — the background campaign threat affecting all portfolio firms
2. **Internal contagion**: a Bernoulli mechanism for affiliated subsidiaries, where an infected parent entity has over a 58% probability of transmitting to a sister subunit

That second channel is the one that produces the heavy tail. A single large conglomerate hit by a ransomware campaign becomes an internal amplifier.

## Calibrating to LockBit

The authors fit the model to 2,929 firms in Île-de-France, using publicly reported LockBit victim counts from May–July 2024 as calibration data. Firm sizes follow Zipf's law with α = 1.76 — a fairly standard heavy-tailed distribution consistent with what you see in most regional SME portfolios.

The key finding: **with 50% probability, the insurer compensates losses equivalent to up to two days of revenue per firm over a 100-day cyber incident**. That median figure is not especially alarming on its own. The distribution around it is.

Because the model produces right-skewed aggregate loss distributions — driven by tail scenarios where a large firm cascades into its subsidiary network — the expected loss significantly exceeds the median. A pricing actuary writing expected-value loadings would underprice relative to the true risk premium needed to remain solvent in the tail.

## What This Means for Cyber Pricing

The practical consequence is that cyber underwriting is a **portfolio-level problem**, not a policy-level one.

You can model individual firm risk as carefully as you like — sector, revenue, IT maturity, prior incidents. But if your portfolio has correlated exposure to a single ransomware family, a single cloud provider, or a single managed service supplier, individual policy-level loadings will not accumulate to adequate aggregate protection. The correlation structure between policies is the risk.

This has three implications:

**Accumulation monitoring over independent pricing.** A cyber book needs the same aggregate exposure management that catastrophe-exposed property lines get. The monitoring infrastructure needed to track these aggregate signals — A/E drift, distribution shift, anomaly detection — is exactly what [insurance model monitoring](/2026/03/21/insurance-model-monitoring-beyond-generic-drift/) provides. Sub-limits, PML monitoring by attack vector, and explicit aggregate stop-loss structures should be on the table. Underwriters who think of cyber as a series of individual firm decisions are missing the portfolio dimension.

**The basic reproduction number R matters.** The paper derives a time-varying R_max — the equivalent of the epidemic reproduction number — as a stability threshold. When R_max drops below 1, the epidemic decays. For pricing, this suggests that the market-level intensity of a given ransomware campaign is a systematic risk driver that pricing models should track, not absorb into an undifferentiated frequency load.

**Large firms are disproportionate risk contributors.** The model shows vulnerability increases with firm size due to internal contagion across subsidiaries. A corporate portfolio with several large conglomerates is not simply a scaled-up SME portfolio — it has fundamentally different tail behaviour. Aggregate exposed revenue is not a sufficient exposure measure.

## Limitations Worth Taking Seriously

The paper is upfront about what it cannot do.

**Parameter uncertainty is large.** Public ransomware data does not report victim firm sizes, so the calibration relies on proxy assumptions to distribute infections across size bands. The CIR parameters driving β and γ are estimated from a single 100-day episode. Transferring this calibration to a UK portfolio — where sector mix, firm size distribution, and the specific ransomware family may all differ — requires assumptions that cannot be validated against UK event data we largely do not have.

**The model covers one episode.** There is no re-infection, no multi-event scenario, and no interaction between concurrent campaigns. A real cyber portfolio experiences overlapping events — LockBit, ALPHV/BlackCat, Cl0p — not a clean sequential series. The model framework supports extension but has not been built out to cover it.

**Severity is static.** Loss severity for each firm is drawn from a Beta distribution and held constant over the attack period. In reality, incident costs depend heavily on how quickly containment and response occur — a firm that detects and isolates within 24 hours faces a materially different loss than one that takes two weeks. Dynamic severity modelling is acknowledged as future work.

**Île-de-France is not the UK market.** French firm size distributions, sector composition, and the penetration of cyber insurance differ from the UK. The NCSC's annual threat assessments and Lloyd's market exposure data would be necessary inputs for a credible UK calibration.

## Where This Sits in the Toolbox

This is not a pricing model you hand to an underwriter and ask them to run quotations through. It is an aggregate risk model — the kind that should live alongside PML benchmarks and aggregate XL pricing inputs when a reinsurer or portfolio manager is setting cyber capacity limits.

The authors have built something useful: a theoretically grounded, stochastically consistent framework for simulating what happens to a portfolio under an epidemic-style cyber event. The LockBit calibration makes it concrete rather than abstract.

We think the right next step — for UK insurers and Lloyd's syndicates with cyber portfolios — is to treat this kind of epidemiological modelling as input to the aggregate exposure management process, not as a replacement for granular risk selection. Use it to size aggregate XL cover, stress-test PML estimates, and think carefully about subsidiary-network concentration. Use your standard models for individual policy pricing.

The paper is at [arXiv:2603.15369](https://arxiv.org/abs/2603.15369).
