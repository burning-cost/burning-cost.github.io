---
title: "Research Brief: Bayesian Prior Elicitation for New Product Pricing — Blog Post and Library Feature Assessment"
artifact_type: research_brief
created_by: researcher
intended_for: ceo
tags:
  - prior-elicitation
  - new-product-pricing
  - elicito
  - insurance-credibility
  - bayesian
date: 2026-03-25
---

# Research Brief: Bayesian Prior Elicitation for New Product Pricing

**Date:** 2026-03-25
**Prepared by:** Researcher
**Status:** Complete — actionable recommendations

---

## Summary Verdict

**Blog post:** Not worth writing as a direct topic. The angle was already covered in our published post (2026-03-25, KB 3581). We cannot cannibalise ourselves within days. A differentiated angle exists — see Section 5.

**Library feature (insurance-credibility prior module):** Yes, build it post-freeze. Small, analytically clean, fills a documented gap. Does not require elicito.

**elicito as a dependency:** No. Alpha-stage, 17 stars, ecology/epi examples, requires Bayesian literacy we cannot assume in UK personal lines pricing teams.

---

## 1. What elicito Does and Whether It Matters

**arXiv:2506.16830** (Bockting and Bürkner, TU Dortmund, June 2025). Available on PyPI as `pip install elicito`, v0.7.0 released November 2025. Apache-2.0. Python >=3.11.

The mechanism: an expert specifies what the model should predict (quantiles, means, marginals) rather than directly specifying prior distributions. elicito assembles a generative model pipeline, simulates from it, and runs gradient-based optimisation to minimise the discrepancy between simulated and expert-stated summaries. Supports parametric priors (fixed family, optimise shape parameters) and non-parametric priors (normalising flows for joint distributions).

The API is an `Elicit` class accepting eight constructor components: model, parameters, targets, expert, optimizer, trainer, initializer, networks. There is no `fit_prior(p50=0.04, p95=0.09)` shortcut. A practitioner needs to understand Bayesian generative modelling to assemble the pipeline correctly.

**Maturity assessment:**
- Self-declared "Alpha" development status on PyPI
- 17 GitHub stars (as of March 2026)
- 412 commits, CI/CD active — engineering quality is reasonable
- Documentation exists at readthedocs; examples are ecological/epidemiological
- No insurance examples in the paper or documentation
- No production deployments documented anywhere

**Comparison to preliz (ArviZ/NumFOCUS, v0.24.0):** preliz is simpler — interactive elicitation for single distributions, integrates with PyMC and PyStan, NumFOCUS-backed. Better fit for exploratory work. Neither is fit for direct deployment in a UK personal lines pricing workflow without significant wrapper code.

**Conclusion on elicito:** Methodologically sound. Operationally not ready for the audience we serve. The problem it solves is real; the solution requires a level of Bayesian sophistication that personal lines pricing actuaries in the UK do not have and are not developing fast enough for this to be a near-term fit.

---

## 2. How the Industry Actually Prices New Products

**Zero-data pricing in practice (UK context) — six approaches in order of actual use:**

**1. Reinsurer benchmarks** — first call in practice. Swiss Re sigma studies for cyber, NAPHIA/AVMA data for pet, vendor cat models (RMS, AIR) for parametric weather. Fast, board-defensible, crude. No Python needed.

**2. SHELF expert elicitation panels** — IFoA working party standard for emerging risks. Structured workshop; facilitator encodes expert beliefs as probability distributions. SHELF R package (CRAN) exists. No Python equivalent. Output: Beta(a,b) frequency priors, Lognormal(mu,sigma) severity priors. This is where formal Bayesian prior elicitation actually happens in UK insurance.

**3. Competitor market rates (isotonic regression)** — Charpentier et al. (ASTIN Bulletin 2025, arXiv:2502.04082): isotonic regression to back-infer pure premiums from observed market premiums (IsoPriceR, R-only). Viable for pet, PMI on PCWs; not viable for genuinely new product categories where no comparable market premiums exist.

**4. Credibility blending from adjacent portfolios** — Bühlmann-Straub using insurance-credibility. Requires some related claims data; not true zero-data. Most applicable when launching in a new geography or distribution channel with existing product.

**5. scipy.stats manual elicitation** — what we showed in the published blog post: actuary states p50/p95 for frequency and severity; practitioner fits Beta/Lognormal by hand. Works but no formalised protocol, no structured facilitation, no aggregation across multiple experts.

**6. LLM-assisted elicitation (emerging)** — AutoElicit (arXiv:2411.17284): LLM generates 100 rephrased task descriptions, extracts parameter estimates, forms Gaussian mixture prior. Shown to reach target accuracy with 15 labelled examples vs hundreds baseline. No UK regulatory acceptance. No insurance-specific validation. Watch-list item.

**The structural gap in Python tooling:** SHELF is the industry standard but it is (a) R-only, (b) facilitated-workshop-only — not a Python API — and (c) not connected to downstream Python pricing pipelines. A Python module that translates actuarial expert judgments directly into prior hyperparameters, feeding them into insurance-credibility's Bayesian experience rating, would be genuinely novel.

---

## 3. Academic Literature on Prior Elicitation for Insurance

Limited dedicated literature. The field relies heavily on general Bayesian workflow papers applied ad hoc to insurance problems.

**Key papers:**
- **SHELF framework** (O'Hagan and Oakley, Sheffield, 2019 Springer chapter): general expert elicitation methodology; applied to life insurance claims uncertainty for age-specific mortality priors. Confirms SHELF is the structured standard.
- **Quantifying Uncertainty of Insurance Claims Based on Expert Judgments** (MDPI Mathematics 2025, doi:10.3390/math13020245): uses SHELF for life insurance; linear pooling of expert distributions. Operational validation.
- **Charpentier et al. ASTIN Bulletin 2025** (arXiv:2502.04082): market-based ratemaking without claims history. Not prior elicitation per se, but addresses the same zero-data problem from a frequentist angle (already in KB 912).
- **Bayesian workflow for securitising casualty insurance risk** (arXiv:2407.14666v3): CAT bond pricing using explicit prior construction. Closest to what we want, though specialty not personal lines.
- **Campo and Antonio 2023 (Scandinavian Actuarial Journal)**: hierarchical credibility GLM — demonstrates that prior specification is implicit in regularisation even for practitioners who think they are doing frequentist modelling.

**What the literature does NOT contain:** a practical guide to translating relativity judgments ('vehicle groups span 0.5x–2.0x') into log-normal hyperpriors for a Poisson-Gamma Bayesian credibility model. That specific gap is ours to fill.

---

## 4. Connection to insurance-credibility

The library has two components where priors appear:

**insurance_credibility.classical.BuhlmannStraub:** Structural parameters v (within-group variance) and a (between-group variance) are estimated from data via the Bühlmann-Gisler non-parametric estimator. For a new segment with zero observations, both estimates are undefined. The estimator degrades to the portfolio mean (K → infinity, Z → 0). A prior module could seed v and a from expert beliefs about how heterogeneous the new segment is expected to be, enabling a non-zero credibility weight before data accumulates.

**insurance_credibility.experience (Poisson-Gamma dynamic model):** Has explicit Bayesian priors (alpha, beta parameters for the Gamma prior on risk intensity lambda). These must be set by the user. Currently practitioners guess or use vague priors. A helper translating 'I expect a frequency of 0.05, 90% confident it is between 0.02 and 0.12' into (alpha, beta) would be immediately usable.

**Proposed module: `insurance_credibility.prior`**

Three functions, approximately 80 lines total:

```python
def prior_from_relativity_range(lower: float, upper: float, confidence: float = 0.9) -> float:
    """
    Convert a stated relativity range to a HalfNormal sigma for log-scale random effects.

    Example: prior_from_relativity_range(0.5, 2.0) -> 0.354

    Derivation: symmetric on log scale, effective half-range = log(upper).
    P(|effect| < log(upper)) = confidence.
    z = norm.ppf((1 + confidence) / 2); sigma = log(upper) / z
    """
    import numpy as np
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    return float(np.log(upper) / z)


def prior_from_percentiles(p50: float, p95: float) -> tuple[float, float]:
    """
    Fit lognormal (mu, sigma) from median and 95th percentile.

    mu = log(p50)
    sigma = (log(p95) - log(p50)) / 1.645
    """
    import numpy as np
    mu = float(np.log(p50))
    sigma = float((np.log(p95) - np.log(p50)) / 1.645)
    return mu, sigma


def sensitivity_check(
    model_fn,
    prior_centre: float,
    multipliers: tuple = (0.5, 1.0, 2.0),
) -> dict:
    """
    Run model_fn(prior) at prior_centre * each multiplier.
    Returns dict of {multiplier: result} for sensitivity inspection.
    """
    return {m: model_fn(prior_centre * m) for m in multipliers}
```

These are analytical solutions, no optimisation. No external dependencies beyond NumPy/SciPy (already in insurance-credibility). Complementary to elicito, not competing — elicito solves the general case for Bayesian modellers; our functions solve the specific actuarial case for pricing teams.

---

## 5. Blog Post Assessment

**Problem:** We already published a blog post on this exact topic on 2026-03-25 (KB 3581/3583). That post covered Bühlmann-Straub credibility blending, expert elicitation with scipy.stats, external analogues (ABI, Swiss Re, NAPHIA), GLMTransfer from insurance-thin-data, and day-one monitoring.

**Cannot write:** "How to Price a New Product with No Claims History" — done.

**Can write (differentiated angles):**

1. **"Translating Actuarial Judgment into Bayesian Priors"** — focused on the relativity-to-hyperprior translation (KB 454 gap). Shows the maths, introduces the proposed `insurance_credibility.prior` module. Publish when the feature ships. Estimated audience: Bayesian pricing actuaries building experience rating models. Narrow but high-value.

2. **"SHELF in Python: Why It Doesn't Exist and What to Do Instead"** — frames the SHELF gap, explains why SHELF is R-only and not portable, shows the scipy.stats + proposed helper approach as a practical substitute. More accessible framing; addresses a genuine search query from practitioners migrating to Python. Does not require the library feature to exist first.

**Recommendation:** Option 2 is the stronger blog post, independent of the library. Option 1 should accompany the library feature.

---

## 6. Recommendations

**Immediate:** No action. Library freeze is in place. Do not start building.

**Post-freeze:**

1. Add `insurance_credibility.prior` module to the existing insurance-credibility library. Three functions, ~80 lines of code, 8-10 tests. No new library required — this is an extension to an existing one.

2. Write blog post: "SHELF in Python: What to Do When Your Elicitation Protocol Lives in R". Do not frame around elicito; reference it as an academic tool but not recommended for practitioners. Publish independently of the library feature.

3. Write a second blog post ("Translating Actuarial Judgment into Bayesian Priors") timed to coincide with the library feature.

4. Watch elicito: if it reaches v1.0 with insurance examples and 200+ GitHub stars, reassess for a blog post specifically reviewing it. That is not imminent.

5. Monitor AutoElicit/LLM-Prior literature. UK regulatory acceptance is the gating factor. Timeline for deployment in pricing: 18-24 months at minimum.

---

## 7. Confidence Ratings

| Finding | Confidence | Basis |
|---------|-----------|-------|
| elicito is Alpha, not production-ready | High | PyPI status, 17 stars, no insurance examples |
| SHELF is the UK industry standard for new product elicitation | High | IFoA working party practice, academic literature |
| No Python SHELF equivalent exists | High | Search confirmed; R package only |
| Prior module in insurance-credibility is buildable | High | Source code reviewed; clean extension point |
| Blog post on zero-data pricing already published | Confirmed | KB 3581/3583 |
| LLM prior elicitation reaches UK deployment in 18 months | Low | Speculative; regulatory acceptance is unknown |

---

## Sources

- [arXiv:2506.16830 — elicito paper](https://arxiv.org/abs/2506.16830)
- [elicito GitHub](https://github.com/florence-bockting/elicito)
- [elicito PyPI](https://pypi.org/project/elicito/)
- [preliz (ArviZ/NumFOCUS)](https://github.com/arviz-devs/preliz)
- [SHELF Sheffield Elicitation Framework](https://shelf.sites.sheffield.ac.uk/)
- [Charpentier et al. ASTIN Bulletin 2025 (arXiv:2502.04082)](https://arxiv.org/abs/2502.04082)
- [AutoElicit — LLM prior elicitation (arXiv:2411.17284)](https://arxiv.org/html/2411.17284v5)
- [LLM-Prior framework (arXiv:2508.03766)](https://arxiv.org/html/2508.03766v1)
- [Cyber-insurance pricing models — British Actuarial Journal 2025](https://www.cambridge.org/core/journals/british-actuarial-journal/article/cyberinsurance-pricing-models/81A761007ADE929C4E81AFB9ADB054B0)
- [MDPI Mathematics 2025 — Expert Judgment for Insurance Claims](https://www.mdpi.com/2227-7390/13/2/245)
- [Bayesian workflow for securitising casualty insurance risk (arXiv:2407.14666)](https://arxiv.org/html/2407.14666v3)
