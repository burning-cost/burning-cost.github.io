---
layout: post
title: "Market-Based Ratemaking Without Claims History"
date: 2026-03-25
categories: [techniques, getting-started]
tags: [ratemaking, pet-insurance, embedded-insurance, abc, approximate-bayesian-computation, isotonic-regression, market-data, competitive-pricing, no-claims-history, new-product, insurance-credibility, insurance-thin-data, astin-bulletin, arxiv, python, r, uk-insurance, personal-lines]
description: "Goffard, Piette, and Peters (ASTIN Bulletin 2025) show how to calibrate insurance rates using competitor premiums and no internal claims data — using ABC and isotonic regression. Here is what the method actually does and when a UK pricing team should reach for it."
---

Our post on [Pricing a New Product with No Claims History](/2026/03/25/pricing-new-product-no-claims-history/) covers the usual toolbox: Bühlmann-Straub credibility from adjacent books, expert elicitation as Beta priors, transfer learning via `GLMTransfer`. All of those approaches have one thing in common — they need *some* internal claims experience eventually. Even the most credibility-heavy prior has to be calibrated against something your company has actually observed.

What if you have none of that? No proxy portfolio. No adjacent book. You are entering a genuinely new market — UK pet insurance for a motor insurer, embedded cover sold through a retail partner, parametric climate cover for a broker — and all you have is what competitors are charging.

Goffard, Piette, and Peters answered this question in arXiv:2502.04082, published in ASTIN Bulletin in May 2025. Their method, with accompanying R package `IsoPriceR`, derives pure premium estimates using only observed commercial rates. No internal claims data required at any point.

---

## The core idea

The fundamental problem is an identification problem. You observe market premiums — what competitors charge. You do not observe the underlying loss distributions that justify those premiums. The relationship between the two is not direct: a commercial premium includes loading for expenses, profit, reinsurance cost, and competitive positioning. Pure premium (expected claims cost) is embedded inside it, but you cannot simply divide and recover it.

The paper's insight is to treat the loss distribution parameters as latent quantities and use the market premiums as the observable data. They frame this as an Approximate Bayesian Computation (ABC) problem: if you could sample candidate parameter values, simulate what premiums *would* be under those parameters, and check whether the simulated premiums match what you actually observe in the market — you could back out a posterior over the parameters.

The catch is that the mapping from loss parameters to commercial premiums requires an assumption about how the market prices. Goffard et al. use isotonic regression: they require that as pure premium increases, commercial premium increases monotonically. This is enforced via the Pool Adjacent Violators Algorithm (PAVA), which iteratively averages violating sequences until monotonicity holds. The authors note isotonic regression "provides robustness to outliers, superior to that of simple linear regression" — an important property when market quotes include one or two outliers from aggressive market entrants.

---

## The algorithm in detail

The full procedure is a Population Monte Carlo ABC loop. Each iteration:

1. Sample a candidate parameter vector θ from the current intermediate posterior.
2. Simulate R = 2,000 loss outcomes per risk class using Monte Carlo.
3. Convert simulated losses to implied commercial premiums via the isotonic regression mapping.
4. Compute the discrepancy between simulated and observed market premiums.
5. Accept or reject θ based on a tolerance threshold ϵ.
6. Update the intermediate posterior via kernel density estimation over accepted particles.
7. Reduce ϵ and repeat for up to G = 9 generations.

They run J = 1,000 particles per generation. The tolerance decreases across generations, which concentrates the posterior progressively — standard SMC-ABC machinery, applied cleanly to the pricing identification problem.

The paper tests three loss models: Poisson-LogNorm with free (μ, σ), Poisson-LogNorm with σ fixed at 1, and Poisson-Gamma. On their pet insurance dataset — 1,080 quotes from five French insurers, covering 12 risk classes (four breeds × three age bands) — Poisson-LogNorm with σ = 1 performed best in terms of final tolerance. The MAP estimator outperformed the Mode estimator on synthetic data: lower variability, more reliable convergence, approaching true parameter values as the sample size increased from n = 25 to n = 200.

For the Australian Shepherd (4-year-old female), the MAP estimates were λ = 0.31 (claim frequency), μ = 6.14 (log-mean severity). Expected claim amount came out at €239-€245 depending on estimator. The loss ratio corridor used was 40-70% — the authors constrain the ABC to accept only particles where the implied loss ratio is commercially plausible, which is a sensible guard against degenerate solutions.

Across all 12 risk classes, the methodology produced consistent loss ratios of 62-65%. Breed risk ranking (ascending): Australian Shepherd < Golden Retriever < German Shepherd < French Bulldog. Expected claim amounts increased with age in all breeds, which matches the veterinary literature.

---

## What this is not

The method has real limitations that matter for a UK team deciding whether to use it.

**It inherits the market's pricing.** If the market is wrong — if competitors have mispriced a risk class for two years and you are using their quotes as your data — you will replicate that error. ABC calibrated to market premiums converges toward the market's implied view of risk, not the true loss distribution. In a well-functioning competitive market this is a reasonable approximation; in a market with a dominant player who has made systematic errors, you are learning from the dominant player's mistakes.

**Loss ratio corridor is a strong prior.** Setting LR_low = 40% and LR_high = 70% constrains which parameter values are accepted. This is not unreasonable for pet insurance in France (2023 data), but a UK pricing team would need to set their own corridor based on what they know about the market's expense base. Pet insurance in the UK runs at higher combined ratios than France in this period — the corridor matters, and the results are sensitive to it.

**It requires enough distinct risk classes to identify the isotonic regression.** The paper uses 12 risk classes (4 × 3 grid). With fewer risk classes — say, only 4 — the isotonic regression has limited data to work with. The ABC approach is more robust than linear regression here, but you still need enough price variation across risk classes to identify the relationship. A product with uniform pricing across all segments provides no gradient to learn from.

**IsoPriceR is in R.** This is not a fatal objection, but UK pricing teams running Python-first pipelines will need to either port the methodology or call R from Python. The paper's GitHub repository (`market_based_insurance_ratemaking`) provides the R implementation. A Python port is not complex — scipy's isotonic regression (`isotonic_regression` from scipy 1.12+) and any ABC library (pyabc, or a custom loop) covers the same ground.

---

## When a UK pricing team actually uses this

This method fills a gap that our other toolbox approaches do not cover cleanly. Let us be specific about the scenarios.

**Pet insurance from scratch.** A motor insurer entering pet does not have an adjacent book in any meaningful sense. Motor claim distributions share nothing useful with veterinary costs. The `GLMTransfer` approach from `insurance-thin-data` requires a source portfolio with similar structure; it is not going to help here. What you *do* have is competitor premiums from comparison sites — Compare the Market and MoneySuperMarket publish live quotes, and three months of quote monitoring gives you a rich dataset of market rates across breed, age, and excess combinations. That is exactly the data this method is designed for.

**Embedded cover through a new distribution channel.** An insurer distributing cover through a retailer (gadget cover embedded in a broadband contract, travel cover embedded in a current account) often cannot price off internal data from other channels — the risk mix, the claims behaviour, the customer profile can differ substantially. But they can observe what other embedded products in adjacent channels are charging. Market-based ABC is the right starting point here.

**New geographic market entry.** A UK insurer entering Ireland, or an EU insurer post-Brexit setting up a UK entity, may have pricing models for their home market but nothing for the new geography. Market quotes for the UK product are available; internal claims experience in the UK is not. This is the cleanest application of the method — use local market quotes to calibrate local loss parameters, then validate against your home-market priors as a sanity check.

**Early stage of a Bühlmann-Straub sequence.** This is where the connection to our earlier post is most direct. Bühlmann-Straub credibility blending requires a prior. If you have no adjacent book, the market-based ABC posterior *becomes* your prior — your estimate of μ (collective mean loss rate) and your uncertainty around it. As your own claims emerge over the first 12-18 months, the credibility framework takes over and Bühlmann-Straub blends that prior with your emerging experience. The two methods are not alternatives; they are sequential stages in the same process.

---

## Running it against UK market data

The practical workflow for a UK team would look like this:

**Step 1: Quote monitoring.** Scrape or manually collect market quotes across a risk dimension grid. For pet, the natural grid is {breed × age × excess level}. You need price *variation* — quotes that differ meaningfully across cells. 200-500 quotes across 15-20 risk classes is workable; the paper's 1,080-quote dataset across 12 classes gives a sense of what is needed.

**Step 2: Define your loss model family.** Poisson frequency × LogNormal severity is the standard UK personal lines starting point and the model the paper validates most thoroughly. Parameterise with (λ, μ) — claim frequency per policy year, and log-mean severity. If you have any prior information on severity distribution (from reinsurer studies, BSAVA data for pet, NAPHIA for breed-level frequency patterns), encode it as a prior on (λ, μ) before running ABC.

**Step 3: Set the loss ratio corridor.** UK pet insurance expense ratios run higher than France. UK pet market combined ratios in 2023-2024 were in the range 95-105% depending on the provider (ILAG market data). That implies loss ratios of roughly 60-75% for well-run operators. Set LR_low = 50%, LR_high = 80% as a starting point, and stress-test the sensitivity.

**Step 4: Run the ABC loop.** J = 1,000 particles, R = 2,000 simulations per particle per generation, G = 9 generations maximum. On a single laptop this takes minutes rather than hours. The `IsoPriceR` package handles this directly. A Python implementation using `pyabc` and `scipy.optimize.isotonic_regression` replicates the method straightforwardly.

**Step 5: Extract MAP estimates and uncertainty.** The ABC posterior gives you a distribution over (λ, μ) — not point estimates. Report the MAP for your base rates and use the posterior spread to quantify parameter uncertainty for your reserve basis. This is directly usable in a Solvency II internal model context where parameter risk loading is required.

**Step 6: Feed into Bühlmann-Straub.** Take the MAP estimates and their credibility intervals as your prior (μ, v, a) in `BuhlmannStraub` from `insurance-credibility`. Set k conservatively based on the posterior variance — a wide ABC posterior means your prior is uncertain, which should translate into a larger k (more exposure needed before your own data dominates).

---

## Our position

The paper's contribution is genuine and the method is practical. We think it is underused relative to its usefulness — most UK pricing teams entering new markets reach for reinsurer benchmarks and expert elicitation (which we covered in our earlier post) without recognising that competitor quote data is a richer and more systematically exploitable signal.

The isotonic regression link is the paper's key practical innovation. It handles a real problem — the relationship between pure and commercial premium is monotone but non-linear and noisy — in a way that pure parametric assumptions would get wrong. ABC provides the right framework for parameter inference when the likelihood is intractable. Both choices are defensible.

The method does not replace the approaches in our earlier post. It sits upstream of them: use market-based ABC to establish your prior, then use Bühlmann-Straub to update that prior as your own claims emerge. If you have an adjacent book, `GLMTransfer` can borrow rating factor structure on top of that. The sequence is additive, not competitive.

For a UK team entering pet insurance, embedded cover, or any other line where competitor data is available before internal experience is, this is the correct starting point.

---

*Goffard, P.-O., Piette, P., and Peters, G. W. (2025). Market-based insurance ratemaking: application to pet insurance. ASTIN Bulletin. arXiv:2502.04082.*

*The `insurance-credibility` and `insurance-thin-data` libraries are open source and available via `uv add`. IsoPriceR is available on the paper's GitHub repository.*

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/) — the underlying credibility framework that `insurance-thin-data`'s `BorrowStrength` builds on
