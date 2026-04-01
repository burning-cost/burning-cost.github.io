---
layout: post
title: "Five Ways to Get Market Data for Entry Pricing — and What Each Actually Delivers"
date: 2026-04-01
categories: [techniques, pricing, strategy]
tags: [entry-pricing, mga, market-data, competitor-analysis, consumer-intelligence, abi, pcw, quote-scraping, glm-reverse-engineering, milliman, capacity-provider, expert-judgment, abc, approximate-bayesian-computation, isotonic-regression, fca, prod4, fair-value, uk-motor, pet-insurance, uk-insurance, personal-lines, new-product]
description: "A UK MGA at launch has five routes to market data: competitor quote reverse-engineering, rate indices from Consumer Intelligence, ABI aggregate statistics, capacity provider data sharing, and expert judgment with commercial buffers. We work through what each delivers, where each breaks, and where the Goffard ABC approach sits in that landscape."
math: false
author: burning-cost
---

A UK MGA at launch faces a specific problem: the FCA's Consumer Duty requires that commercial premiums demonstrably reflect the risks customers face. "We priced to match the market" is not a compliant standalone answer under PROD 4. But you have no claims history.

There are five routes the industry uses to get around this. We think most teams reach for the first two without properly understanding their failure modes, and often do not know the third and fourth exist in useful form. The fifth is the fallback for genuinely novel risks and should be used deliberately, not by default.

This post maps the landscape. The [ABC implementation post](/2026/03/31/market-based-ratemaking-python-implementation/) and the [four-stage MGA architecture post](/2026/04/01/mga-entry-pricing-four-stage-architecture/) cover the Goffard approach in detail; here we focus on what you should know about the alternatives before deciding how to combine them.

---

## Method 1: Competitor quote scraping and GLM reverse-engineering

The most common approach, and the one with the most elaborate documented failure mode.

The method is straightforward in concept. You collect a large grid of competitor quotes from PCW journeys — typically 15,000–30,000 quotes across a synthetic risk grid — and fit a machine learning or multiplicative GLM to model each competitor's effective rate surface. The output is a model that predicts what Competitor A will quote for a given risk profile, which you use as a proxy for market pricing.

Milliman documented this in 2015 in a widely-cited UK market analysis. They collected approximately 20,000 quotes and fitted an ensemble model. Their headline finding was ~4% average absolute relative error on held-out competitors. That sounds reasonable. Their follow-up finding was more useful: they identified eight structural obstacles to recovering the actual rating formula.

The eight obstacles are worth stating explicitly because they each map to a specific type of systematic error, not random noise:

**Unknown risk variables.** PCW quote journeys do not expose all rating factors. A motor insurer using DVLA driving licence checking, TPI third-party identity, or proprietary telematics score enrichment applies those factors after the customer journey. You see the output premium; the enrichment is invisible.

**Unknown variable groupings.** Insurers group continuous variables into bands before applying rating factors. You can observe that a 21-year-old pays more than a 35-year-old, but you cannot recover from quotes alone whether the insurer uses 5-year or 10-year age bands, or where the breakpoints are.

**Unknown interactions.** A GLM with a driver-age-by-vehicle-group interaction produces different premiums than a GLM with main effects only. You can fit an interaction model to the observed quotes, but you are recovering the aggregate effect of the actual model structure, not the model itself.

**Proprietary third-party data.** Experian, Equifax, and specialist data providers give insurers rating factors that are entirely invisible to a quote-monitoring system. Indices of claims propensity by postcode sector, CCJ history flags, CAIS credit data — these are present in the premium, absent from the risk grid.

**Unknown vehicle classification.** ABI vehicle group is public, but insurers maintain proprietary supplements: historical claims cost by vehicle, theft flags, repair cost volatility. Two vehicles in the same ABI group can attract materially different premiums because of non-public vehicle classification.

**Multiple additive and multiplicative elements.** A motor premium contains a base rate, NCD discount, voluntary excess discount, additional driver loading, multi-car discount if applicable, and potentially PCW-specific promotional adjustments. Fitting a single model to the total premium averages across this structure without decomposing it.

**Non-multiplicative structures.** Standard GLM pricing is multiplicative. Some insurers use additive loading structures for specific risk features (young driver loading as a flat amount, not a multiplier) or capped structures (excess discount capped at 25% regardless of voluntary excess level). A multiplicative model fitted to non-multiplicative output will be systematically wrong at the extremes.

**Aggregated premium data.** PCWs often display a "from £X" headline rather than the exact premium for the profile you submitted. The aggregated display masks the actual risk profile the premium was calculated on.

The conclusion from the Milliman analysis is not that competitor quote modelling is useless. It is that it tells you what competitors charge, not why they charge it. You learn market pricing, not risk loading. The two things can diverge substantially, particularly in segments where the market has been systematically overpriced (motor 2018–2020) or underpriced (motor 2021–2022 before the spike).

For a new entrant using this method alone, there is a more specific failure mode: you may be copying competitor pricing errors. If the market has collectively mispriced a segment — and UK motor has done this at scale twice in the past decade — a competitor-quote model trained at that time will reproduce the market's error in your rates. The Goffard ABC method explicitly asks which risk parameters are consistent with market pricing; it does not assume those parameters are correct. Copying quotes skips that step entirely.

**What this method is genuinely useful for:** understanding competitive positioning — where you are cheap relative to the market, where you are expensive. That is useful for conversion modelling and market segmentation even when it is insufficient for pricing soundness. Run it in parallel with something else, not as a standalone.

**The legal constraint:** Automated PCW scraping violates all four major UK PCW terms of service. Manual quote collection (a human completing the journey) is legal. For 200-500 profiles the manual route is feasible. For 20,000 profiles it is not, which limits the GLM reverse-engineering approach to operators with either a special data relationship or the resources to build a large manual collection operation.

---

## Method 2: Commercial rate indices — Consumer Intelligence and Pearson Ham

Consumer Intelligence is the dominant provider of competitive pricing intelligence in UK personal lines. They track over 600,000 prices daily — approximately 10.5 million data points per month — across motor, home, travel, and pet insurance. Their data collection works because they appear on PCW panels as a legitimate quoting entity, not as a scraper. They have data-sharing agreements with the insurers and aggregators, which means they see actual quoted prices, not scraped HTML.

The product offers competitive positioning intelligence at the PCW level: where you rank against specific competitors on specific risk profiles, in real time. The rate index is the aggregate output — how the market's average quoted price has moved over time. Pearson Ham provides a comparable service with slightly different methodology and a stronger focus on claims analytics integration.

For a pricing team at an established insurer, this is genuinely valuable. For an MGA at launch, there is a specific bootstrapping problem.

Access to Consumer Intelligence data requires an insurer or MGA status and a signed data use agreement. You must be operational before you can sign the agreement. You need the data most acutely before you launch — for pre-launch calibration and launch rate setting. But you cannot access it before you are live. Enterprise contracts run approximately £50,000–150,000 per year based on industry estimates; there is no published pricing. For a pre-revenue MGA, this cost comes before any premium income.

The practical consequence: most new MGAs do not have Consumer Intelligence access at launch. They access it after year one, if at all, and by then they have shifted from "pricing a new product" to "monitoring competitive positioning on a book with emerging claims experience." It is a post-launch tool dressed up as a pre-launch solution.

What the rate index provides, even if you have access, is the same thing as Method 1 in aggregate form: commercial rates, not risk parameters. It answers "am I competitive on my premium?" not "is my premium actuarially sound?" Under PROD 4 you need both questions answered, not just the first.

**What this method is genuinely useful for:** ongoing competitive monitoring on a live book, renewal pricing decisions, PCW rank management. It is not a risk calibration tool; it is a market intelligence tool. These are different things, and conflating them is where most new-entrant pricing frameworks go wrong.

---

## Method 3: Industry aggregate statistics — ABI, Thatcham, MIB

The ABI and Thatcham Research publish aggregate loss statistics for UK motor insurance: claim frequency and average cost by vehicle group band, driver age group, and geographic region. The Motor Insurers Bureau provides claims data relevant to uninsured driver exposure. These are public or near-public; ABI data is available to members, and the headline statistics are widely reported.

This is the most credible source type — actual claims data, not market prices — but it has two structural limitations that restrict its utility for entry pricing.

**Resolution is too coarse.** ABI vehicle groups are 50 bands; modern pricing models use 500+ effective vehicle cells combining the public group with proprietary enrichment. Age groupings in the aggregate statistics are typically five-band (under 25, 25–34, 35–50, 51–65, 65+), whereas a full motor GLM uses 15–20 effective age cells. The aggregate data tells you the intercept of your model (market average claim frequency and severity); it does not tell you the rating relativities that make up the slope.

**Lag is 12–24 months.** ABI motor statistics for calendar year 2025 are published in late 2026. For pre-launch calibration in April 2026, the most recent available data reflects 2024 experience. For a market with the claims inflation volatility UK motor has shown since 2021, 12 months of lag is not trivial.

The right use of ABI data for entry pricing is as a corridor calibration — setting bounds on what your model should produce, rather than providing the model itself. If your ABC-derived claim frequency estimate for young male drivers is 0.35 and the ABI data implies the market average is 0.18, that gap requires explanation. It might be justified (you are entering a high-risk segment deliberately), or it might indicate a miscalibrated loss ratio corridor in your ABC run. ABI data lets you catch that discrepancy; it cannot produce the estimate.

For home insurance, building sum insured distributions by postcode sector are available from various sources including Land Registry and ONS. For pet insurance, PDSA and BSAVA publish veterinary cost statistics by species and breed that can calibrate severity priors. The principle is the same: external data as constraint, not as model.

**What this method is genuinely useful for:** setting the loss ratio corridor for the ABC loop (the most critical prior parameter, as we discuss in our [MGA architecture post](/2026/04/01/mga-entry-pricing-four-stage-architecture/)), validating that your ABC output is plausible at the aggregate level, and providing documented reference data for the FCA evidence chain.

---

## Method 4: Capacity provider and reinsurer data sharing

When an MGA secures a capacity provider — a Lloyd's managing agent, a large composite insurer, or a specialist carrier writing on a delegated authority basis — the capacity provider often has relevant claims data from their own book. Some provide it; some do not; the terms vary significantly.

For a capacity provider with an existing pet insurance book, sharing aggregate frequency and severity statistics with a new MGA writing pet on delegated authority is a natural extension of the commercial relationship. The capacity provider has an interest in the MGA pricing soundly — their capital is at risk. In practice, some provide rate tables that the MGA is expected to use as a floor; others provide aggregate statistics with no prescription; others provide nothing and rely on the MGA's own actuarial function.

At Lloyd's, the dynamic is slightly different. The Business Plan submission requires documented pricing methodology, and the Lloyd's actuarial function reviews it. A new syndicate writing personal lines — a small population, primarily commercial and specialty — would be expected to cite reinsurer benchmarks as part of its pricing evidence. Swiss Re, Munich Re, and Hannover Re all provide proprietary rate benchmarks for segments where they are active reinsurers, and those benchmarks are grounded in actual global claims data rather than local market quotes.

The limitation is selection bias. A capacity provider's claims data reflects the risks they have historically written, which is not the same as the risks the MGA will attract. An MGA launching on a UK PCW is selecting for customers who are comparison-active, which skews the book toward younger, more cost-sensitive, more likely to be non-standard risks relative to a direct-sold portfolio. If the capacity provider's historical book was direct-sold or broker-introduced, their frequency statistics may not be the right prior for a PCW-written book.

The Goffard ABC method actually handles this better than direct data transfer, in one specific sense: the isotonic regression link is calibrated on PCW quotes from the market the MGA is actually entering. Capacity provider data reflects the risk population they wrote; competitor PCW quotes reflect the risk population currently quoting on the channel the MGA will use. For a PCW-entering MGA, that difference matters.

**What this method is genuinely useful for:** providing credible aggregate loss statistics where the capacity provider's book is genuinely comparable, validating that your entry pricing is not wildly inconsistent with what the capacity provider's own actuaries believe, and satisfying the Lloyd's actuarial review requirement. It is not a substitute for market-calibrated entry pricing; it is a sanity check and regulatory evidence piece.

---

## Method 5: Expert judgment with commercial risk-transfer structures

For genuinely novel risks — cyber for SMEs in 2016, pandemic event cancellation cover before COVID, parametric climate triggers for new perils — none of the above four methods produces usable input. There is no claims data, no comparable market, no relevant aggregate statistics, and no capacity provider history to borrow. The fallback is actuarial expert elicitation structured formally as a prior distribution, combined with commercial structures that transfer actuarial uncertainty from the MGA to the capacity arrangement.

The commercial structures are not the same as accurate pricing. They are a mechanism for absorbing pricing uncertainty into the commercial terms:

**Quota-share arrangements** with loss ratio corridors. The capacity provider takes a quota-share of all business; if the loss ratio exceeds a corridor ceiling (say, 85%), the MGA's commission reduces proportionally. If the loss ratio beats a floor (say, 55%), the MGA earns an additional profit commission. This structure gives the MGA a stake in pricing accuracy without requiring accurate pricing on day one.

**Profit-commission arrangements.** Standard in Lloyd's delegated authority. The MGA earns additional commission if the combined ratio comes in below the target. This creates pricing incentive alignment without requiring actuarial sign-off on the absolute rate level.

**Stop-loss reinsurance.** For lines with severe parameter uncertainty, the MGA purchases stop-loss protection above the target loss ratio. This caps the downside from systematic mispricing. The reinsurer prices the stop-loss using their own view of the risk distribution; the cost of the stop-loss is the market's implied price of uncertainty.

These structures are not a pricing method. They are a commercial expression of the fact that the pricing is uncertain and all parties know it. Under PROD 4, they do not substitute for a documented risk basis. The FCA does not accept "our capacity provider agreed to absorb losses above 85% loss ratio" as evidence that the premium reflects the customer's risk. The commercial buffer addresses the MGA's financial exposure, not the fair value question.

Lloyd's Capital Guidance (February 2024) requires strong links between pricing assumptions and SCR inputs on Business Plan submission. The 35% FAL uplift on SCR provides a buffer for model uncertainty in new syndicates — but it is a capital response to uncertainty, not a pricing response. A new Lloyd's syndicate still needs documented pricing methodology, just with a larger capital buffer behind it.

**What this method is genuinely useful for:** novel risks with no market comparators, early-stage capital structuring where the pricing uncertainty is genuinely unresolvable, and Lloyd's syndicate capital planning. For any product where a PCW market exists, this method should be a complement, not a substitute.

---

## How the Goffard ABC method fits in this landscape

The ABC approach (Goffard, Piette, and Peters, ASTIN Bulletin 2025, arXiv:2502.04082) is not a sixth method sitting outside this list. It is a way to extract more signal from the data you collect for Method 1.

Where Method 1 asks "what does the market charge?", the ABC method asks "which risk parameters are consistent with what the market charges?" The distinction is between learning market rates and inferring market-implied loss distributions. Those are different outputs. The first answers "am I competitive?"; the second answers "what is my expected claims cost?" Under PROD 4, you need the second.

The practical workflow for a UK MGA combines several of these:

1. **Collect market quotes manually** (Method 1 data, without the GLM reverse-engineering step that hits the eight obstacles).
2. **Run ABC-SMC with Poisson-LogNormal** to extract (λ, μ) posterior from those quotes — turning Method 1 data into risk parameter estimates rather than rate surface approximations.
3. **Calibrate the loss ratio corridor** from ABI aggregate statistics (Method 3) — this is the most important prior for the ABC, and it is the one input where ABI data is the right tool.
4. **Sanity-check the ABC posterior** against capacity provider statistics (Method 4) if available — not to override the market-calibrated estimate, but to catch gross miscalibration.
5. **Document the full chain** for the FCA evidence file: quotes collected, date and method, ABC output, corridor calibration source, MAP estimates, posterior spread.

Methods 2 and 5 sit at the edges: Consumer Intelligence becomes valuable at the competitive monitoring stage after launch, not at the pre-launch calibration stage. Expert judgment plus commercial structures is the fallback when there is no PCW market to harvest from.

---

## What regulators actually want

The FCA's October 2023 multi-firm review of Consumer Duty implementation found that firms' fair value assessments were frequently not quantitative at the benefit-and-cost level. Many firms asserted fair value without demonstrating the link between the price charged and the expected benefit delivered. For entry pricing, the analogous gap is asserting that a price reflects risk without demonstrating how the risk was estimated.

The ABC posterior is a documented, quantitative answer to the question "how did you estimate claim frequency and severity for this new product?" It is not perfect — the paper's authors are explicit about the method's limitations, which we cover in our [earlier post](/2026/03/25/market-based-ratemaking-no-claims-history/) — but it is auditable, reproducible, and grounded in observable market data rather than gut feel. That combination is what PROD 4 compliance requires.

None of the five methods above produces that independently. The Milliman competitor GLM gives you rate surface, not risk parameters. Consumer Intelligence gives you competitive positioning, not risk parameters. ABI data gives you industry aggregate statistics, not risk parameters. Capacity provider data shares someone else's risk parameters, not ones calibrated to your market. Expert judgment gives you ranges, not calibrated distributions.

The ABC method is the only published approach that inverts commercial premium data into risk-parameter estimates without internal claims data. That it was only published in 2025 explains why most UK entry pricing frameworks have not incorporated it yet. We think that changes quickly once teams understand both what it delivers and how the regulatory context makes it necessary.

---

## Related reading

- [From Competitor Quotes to Risk Parameters: Implementing Market-Based Ratemaking in Python](/2026/03/31/market-based-ratemaking-python-implementation/) — the ABC-SMC Python code and UK pet insurance worked example
- [MGA Entry Pricing: A Four-Stage Architecture from Day Zero to Year Three](/2026/04/01/mga-entry-pricing-four-stage-architecture/) — how the ABC posterior feeds into Bühlmann-Straub credibility, DML selection bias correction, and covariate shift monitoring
- [Market-Based Ratemaking Without Claims History](/2026/03/25/market-based-ratemaking-no-claims-history/) — the Goffard/Piette/Peters method and its limitations

---

*Goffard, P.-O., Piette, P. and Peters, G.W. (2025). Market-based insurance ratemaking: application to pet insurance. ASTIN Bulletin. arXiv:2502.04082.*

*Milliman (2015). Analysing competitor tariffs with machine learning. uk.milliman.com.*

*FCA (2022). PROD 4: Product Oversight and Governance. FCA Handbook.*

*FCA (2023). Consumer Duty multi-firm review: fair value assessments. October 2023.*
