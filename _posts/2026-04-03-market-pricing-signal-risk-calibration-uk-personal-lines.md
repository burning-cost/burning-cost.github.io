---
layout: post
title: "Competitor Quotes Are Risk Data. We Just Pretend They Are Not."
date: 2026-04-03
categories: [techniques, pricing, strategy]
tags: [market-ratemaking, entry-pricing, abc-smc, isotonic-regression, approximate-bayesian-computation, pcw, uk-motor, pet-insurance, fca-prod4, fair-value, mga, poisson-lognormal, consumer-intelligence, abi, astin-bulletin, arxiv-2502.04082]
description: "UK personal lines generates hundreds of millions of competitor quotes per year. The industry treats them as competitive positioning data. They are, in fact, risk calibration data — if you know how to read them. A look at what market-based ratemaking actually is, why it matters for UK entry pricing, and which products it works for."
author: burning-cost
---

Every day, UK personal lines generates an enormous quantity of risk data that the industry does not treat as risk data.

When MoneySuperMarket serves a motor quote, that premium represents the output of a GLM applied to a risk profile. The underlying frequency and severity assumptions are embedded in the number. When four carriers quote the same 35-year-old in Leeds on a Ford Focus, and three of them cluster within 5% of each other while one sits 20% higher, that clustering is an observable signal about where the market's collective risk estimates converge. When French Bulldog pet insurance quotes run at 1.8× the rate of Golden Retriever quotes across five providers, that ratio encodes something real about veterinary cost differentials.

The industry reads these numbers as competitive intelligence — "we are third cheapest on this profile." It does not typically ask: "given these premiums, what claim frequency and severity parameters is the market implying?"

These are different questions. The second is harder. But for a UK pricing team setting rates on a new product, it is the only question that matters under PROD 4.

---

## What the industry actually does with competitor quotes

There are two mainstream approaches to competitor data in UK personal lines.

The first is quote scraping plus GLM reverse-engineering. You collect a large grid of competitor quotes — Milliman's 2015 analysis used approximately 20,000 — and fit a machine learning model or multiplicative GLM to each competitor's effective rate surface. The output is a model that predicts what Competitor A will charge for a given risk profile. This answers "am I competitive?" with reasonable accuracy.

Milliman also documented why this cannot answer "is my premium actuarially sound?" They identified eight structural obstacles: unknown enrichment variables (telematics scores, TPI data, fraud flags), unknown variable groupings and interaction structures, proprietary third-party data invisible in the PCW journey, and non-multiplicative premium structures that a multiplicative GLM cannot recover. The conclusion: you learn commercial rates, not loss loading. Those two things can diverge significantly — and in UK motor, they have, twice in the past decade.

The second approach is commercial rate index subscriptions. Consumer Intelligence tracks over 600,000 prices daily across motor, home, travel, and pet. Their data is genuine — they appear on PCW panels as a quoting entity, not as a scraper, and have data agreements with insurers. The product answers: where do you rank on your target risk profiles? What has the market average done in the past 12 months? This is genuinely useful for a live book. For a new entrant calibrating launch rates, it answers the competitive positioning question and leaves the risk calibration question unanswered. The same gap remains.

Both methods share a common flaw: they treat commercial premiums as endpoints, not as encodings of underlying risk assumptions. They ask what the market charges; they do not ask what risk the market is pricing.

---

## The inversion problem

Here is the thing about a commercial insurance premium: it is a function of underlying risk parameters. The relationship is not direct — commercial premium includes expense loading, profit margin, reinsurance cost, and competitive positioning adjustment on top of the pure premium (expected claims cost). But the pure premium is in there. If you knew the market's assumptions about claim frequency and severity, you could derive what the pure premium should be, and you could verify whether the commercial premium is consistent with those assumptions.

Can you run this in reverse? You observe the commercial premium. Can you back out the risk parameters?

This is an identification problem, not a data problem. The data exists — you are looking at it in the PCW quote. The difficulty is that the mapping from risk parameters to commercial premium involves unknowns (the specific loading structure each insurer applies) that vary across competitors and are not directly observable.

Goffard, Piette, and Peters (ASTIN Bulletin, 2025, arXiv:2502.04082) showed that this inversion is tractable under one key assumption: commercial premium should increase monotonically as underlying expected claims cost increases. This is not a strong assumption — it just says that riskier profiles should cost more in the market. If the ordering of competitors' risk assessments is broadly similar, then the relationship between pure premium and commercial premium, while not linear and not identical across carriers, should be monotone.

With that assumption, you can use Approximate Bayesian Computation. Sample candidate risk parameters (claim frequency λ, log-mean severity μ). Simulate what the expected claims cost would be under those parameters across your risk classes. Apply Pool Adjacent Violators (PAVA) isotonic regression to map simulated costs to commercial premiums. Compare against observed market quotes. Accept or reject the candidates. Iterate until you have a posterior.

The output is not a commercial rate model. It is a posterior distribution over the underlying risk parameters that the market's commercial rates are consistent with. The distinction matters: you are learning about the risk, not copying the rate.

---

## Where UK personal lines makes this hard

The paper's dataset — 1,080 quotes from five French pet insurance providers, 12 risk classes — is well-suited to the method. Pet insurance has low dimensionality, relatively homogeneous coverage structures, and a tight mapping between breed, age, and expected veterinary cost.

UK personal lines motor is harder, for three specific reasons that the research makes clear.

**Dimensionality.** A UK motor GLM has 40–60 rating factors: vehicle group, driver age, occupation, NCD level, annual mileage, territory, voluntary excess, convictions, claims history. The effective rating space has millions of cells. For ABC to identify the isotonic relationship, you need enough distinct risk classes with meaningful price variation. A 200-quote collection spanning 40+ effective dimensions is extremely sparse.

**Coverage heterogeneity.** UK motor PCW quotes mix coverage tiers — different voluntary excess levels, courtesy car inclusion, legal expenses add-ons — in a way that makes the coverage function g_i(x) different across competitors for the same risk profile. The ABC method requires that the coverage structure be comparable across the quotes you collect. For pet insurance with standard vet-fee limits, this is approximately true. For motor, you need to restrict to a single coverage tier (comprehensive, £250 voluntary excess, no add-ons) and accept residual noise.

**Market pricing anomalies.** UK motor in 2023–2024 was absorbing the post-whiplash reform repricing. The market's collective implied risk assumptions were in flux. ABC calibrated to a market in transition will produce wider posteriors and more uncertain estimates — which is the honest answer, but not necessarily a useful base rate for launch pricing. A market that has just absorbed a large regulatory shock is a poor reference population for backing out fundamental risk parameters.

For standard UK motor, we think the ABC approach works best as an intercept calibration: use it to validate that your average rate level is consistent with what the market implies about expected frequency and severity, not to derive the full rating relativities. The relativities come from industry statistics, reinsurer benchmarks, or capacity provider data — sources that provide claims-based evidence the ABC method cannot.

For pet insurance, travel, simple home contents, and embedded covers with homogeneous terms, the method is the right day-1 tool. The dimensionality is workable, coverage structures are comparable, and the risk-to-commercial-premium relationship is stable enough for the isotonic link to identify cleanly.

| Product | ABC fit for UK | Main constraint |
|---------|---------------|-----------------|
| Pet insurance | High | None significant |
| Travel insurance | High | Trip type homogeneity required |
| Home contents (simple) | Medium | Some coverage variation; manageable |
| Motor (standard) | Low–medium | Too many dimensions for rate calibration; useful for intercept only |
| Motor (telematics/UBI) | Low | Telematics score invisible in competitor quotes |
| Commercial lines | Low | Rating factors not captured in standard journeys |

---

## The loss ratio corridor: the one prior that matters

ABC-SMC is not assumption-free. It has one prior that dominates the results: the loss ratio corridor.

The corridor [LR_low, LR_high] constrains which candidate risk parameters are accepted. A particle whose simulated pure premiums imply a loss ratio below LR_low or above LR_high is penalised. This is sensible — it rules out degenerate solutions — but it means the posterior is conditional on your corridor choice being correct.

The Goffard paper uses [40%, 70%] for French pet insurance in 2024. That is not the right corridor for UK personal lines. Getting it wrong biases all downstream estimates.

Our calibrated starting points for UK personal lines in 2024–2025:

| Product | LR_low | LR_high | Calibration basis |
|---------|--------|---------|------------------|
| Pet | 50% | 75% | ILAG 2023 data; UK combined ratios 95–105%, expense base ~30–35% |
| Travel | 45% | 70% | High expense ratio (~40% typical) |
| Home contents | 45% | 65% | ABI home statistics 2023; stable years |
| Motor | 55% | 75% | ABI motor 2024; post-reform normalisation |

Stress-test these. Run the ABC with LR_low ± 5 percentage points and check how the MAP estimate for λ shifts. If it moves by more than 15% on a 5pp corridor shift, your data are not strongly identifying the parameters — you need more risk classes or a wider prior.

---

## Collecting the data in the UK context

Manual PCW harvest is legal. Automated scraping is not — all four major UK PCWs (Confused, MoneySuperMarket, Compare the Market, GoCompare) prohibit it in their terms of service. Consumer Intelligence operates under data-sharing agreements; they are a panel member, not a scraper. This means the automated route to a large quote dataset is not available to a new MGA without a Consumer Intelligence contract, which itself requires being an operational insurer.

Manual collection of 200–500 profiles is feasible. For a 4×3 breed-age grid in pet insurance, 12 risk classes is 60–90 minutes of quote journeys across five providers. For a 20-profile motor grid: 2–4 hours.

The snapshot problem is real. Motor rates change weekly, sometimes materially after weekend claims processing. The practical protocol: collect Tuesday to Thursday mid-week, over 2–3 consecutive weeks, and take the median quote by provider and risk class. This gives a representative snapshot rather than a single-day slice, reduces the noise from weekly volatility, and produces a dataset the ABC can work with cleanly.

For motor, restrict to: comprehensive cover, £250 voluntary excess, no add-ons, a synthetic driver profile with no claims or convictions. Any variation in coverage terms across the profiles defeats the coverage normalisation assumption.

---

## What this gives a UK pricing team

A team running this method can document: we collected 400 competitor quotes in February 2026 across a 4×3 breed-age grid for UK pet insurance. We ran 500-particle ABC-SMC for seven generations under a Poisson-LogNormal loss model. The MAP estimate is λ = 0.28 (95% CI: 0.18–0.41), μ = 6.21 (95% CI: 5.84–6.58). Implied loss ratios across all risk classes sit within [56%, 72%], inside the calibrated corridor [50%, 75%]. The MAP estimates became our Bühlmann-Straub collective mean prior; we will update the credibility weight as own-book claims emerge over the next 18 months.

That is not certainty. The posterior is wide — λ's 95% CI spans a factor of 2.3. The method is honest about that. It is not claiming to know the claim frequency precisely; it is saying these are the frequency values consistent with what the market is charging, given that the market's loss ratios are in the range [50%, 75%].

Under PROD 4, that documentation is substantially stronger than a competitor quote table. It provides a documented, reproducible, quantitative link between market data and risk parameters. It gives a capacity provider something to review. It gives an FCA thematic review something to engage with. And it connects forward to the credibility framework the pricing team will need once claims arrive.

The method does not replace internal claims data — nothing does. But for the period before internal data is credible, it is the only published approach that extracts risk calibration information from the market data that UK personal lines is already generating in vast quantities, and simply treating as competitive intelligence rather than the risk signal it actually is.

---

*Goffard, P.-O., Piette, P., and Peters, G. W. (2025). Market-based insurance ratemaking: application to pet insurance. ASTIN Bulletin. arXiv:2502.04082.*

*Milliman (2015). Analysing competitor tariffs with machine learning. uk.milliman.com/en-gb/insight/2015/analysing-competitor-tariffs-with-machine-learning/.*

- [From Competitor Quotes to Risk Parameters: Implementing Market-Based Ratemaking in Python](/2026/03/31/market-based-ratemaking-python-implementation/) — the Python implementation this post describes conceptually
- [Market-Based Ratemaking Without Claims History](/2026/03/25/market-based-ratemaking-no-claims-history/) — the Goffard/Piette/Peters method in detail
- [Five Ways to Get Market Data for Entry Pricing](/2026/04/01/entry-pricing-market-data-methods-uk-mga/) — where this approach sits in the broader landscape
