---
layout: post
title: "Multi-Dimensional Mortality Modelling with Tucker Decomposition"
date: 2026-03-26
categories: [research, mortality, protection-pricing]
tags: [mortality, tucker-decomposition, protection-pricing, life-tables, forecasting, fca, cmi, human-mortality-database, tensor-methods]
description: "Samuel Clark's MDMx organises the Human Mortality Database as a four-way tensor and applies Tucker decomposition to produce structurally coherent mortality models across 50 countries. The sex-gap forecasting improvement over Lee-Carter is 46%. The UK insurance hook is narrow but real: a benchmark and disruption-detection tool at a moment when the FCA is actively scrutinising stale mortality assumptions."
---

The FCA's January 2026 interim report on the pure protection market (MS24/1.4) contained a finding that should make mortality actuaries uncomfortable. Some firms, the FCA said, are carrying legacy mortality assumptions that have not been updated to capture recent improvements — with the problem concentrated in reinsurance pricing. Separately, CMI_2025 reported that all-age England & Wales mortality in 2025 was the lowest on record, roughly 2% below 2024. If your pricing mortality basis pre-dates COVID and has not been revisited since, you are exposed on two fronts: regulatory scrutiny and genuine economic error.

Into this context arrives Samuel Clark's MDMx (arXiv:2603.20518, Ohio State, March 2026): a multi-dimensional mortality model that organises the entire Human Mortality Database as a four-way tensor and extracts structure from it via Tucker decomposition. It will not replace CMI_2025 as your projection model, and it will not replace Whittaker-Henderson smoothing of your own experience data. But it does offer something neither of those tools provides: a principled, internationally benchmarked view of where UK mortality sits relative to peer countries, and an automated mechanism for flagging years in which UK mortality deviated anomalously from expectations.

That is a narrower claim than many mortality papers make for themselves. It is also an honest one.

---

## What Tucker decomposition is

Tucker decomposition is the higher-order generalisation of matrix SVD. Where SVD factors a matrix M into U Σ Vᵀ — left factors, singular values, right factors — Tucker factors a tensor into a dense core G multiplied along each mode by a factor matrix.

For a four-way tensor M ∈ ℝ^(S × A × C × T), Tucker decomposition gives:

```
M ≈ G ×₁ S_mat ×₂ A_mat ×₃ C_mat ×₄ T_mat
```

Here ×ₙ denotes the mode-n product: G is the core tensor of shape (r₁ × r₂ × r₃ × r₄), and each factor matrix maps the latent dimension back to the observed dimension along that mode. S_mat is (2 × r₁), A_mat is (A × r₂), C_mat is (C × r₃), T_mat is (T × r₄).

The rank tuple (r₁, r₂, r₃, r₄) controls compression. Lower ranks mean fewer parameters and a smoother, more constrained reconstruction. Higher ranks allow more flexibility but risk overfitting and lose the interpretability of the factors.

Tucker differs from CP decomposition (CANDECOMP/PARAFAC), which is the special case where the core is superdiagonal — effectively a sum of rank-1 terms with no interaction between modes. Tucker allows mode interactions through the full core, which matters here: the relationship between age patterns and country effects is not assumed to be independent.

---

## The MDMx tensor

Clark organises the Human Mortality Database as:

- **Mode 1 (sex):** 2 categories, male and female
- **Mode 2 (age):** single-year ages from 0 to 110+
- **Mode 3 (country):** ~50 countries with HMD coverage
- **Mode 4 (year):** calendar years from 1751 to present

The response variable is logit(₁qₓ) — the logit of the one-year probability of death at age x — not log death rates. This is a deliberate choice. Logit transforms keep the response on the real line while respecting the [0,1] constraint on probabilities, and they produce better-behaved residuals in the tails where log-rate transforms can become unstable for very young or very old ages.

The total tensor contains roughly 2.9 million observed values. After applying Tucker decomposition with selected ranks r₁=2, r₂=4, r₃=8, r₄=7, the model has approximately 50,000 parameters — a compression ratio of about 58:1. The rank selection criterion is 99.99% variance retention: ranks are increased until adding another factor contributes less than 0.01% of total variance.

What the factor matrices encode:
- **S_mat (2×2):** trivially — sex differences in mortality level and age pattern
- **A_mat (ages×4):** four latent age patterns. The first resembles the standard mortality curve; subsequent factors capture curvature at different life stages
- **C_mat (countries×8):** eight latent country mortality profiles. Countries cluster by development level, healthcare system, and historical mortality transition stage
- **T_mat (years×7):** seven temporal patterns, including long-run improvement trends and abrupt shocks

The core tensor G ties these together. An entry G[s,a,c,t] gives the interaction weight between latent sex component s, age component a, country component c, and temporal component t.

---

## Four tasks from one decomposition

The structural value of Tucker over a suite of separate models is that the decomposition is shared. All four of MDMx's capabilities draw on the same estimated factor matrices.

**1. Model life table clustering.** Project each country's mortality schedule into the factor space defined by C_mat. Apply Gaussian mixture modelling in that low-dimensional space. Countries in the same mixture component share a latent mortality regime. The UK sits in a Western European cluster with Germany, the Netherlands, and the Nordic countries. Within-cluster trajectories can be smoothed to give regime-consistent model life tables.

**2. Life table fitting with disruption detection.** Given a country's observed mortality time series, fit it within the Tucker framework using a three-stage algorithm: grid search initialisation, Gauss-Newton refinement, then a Bayes-factor test on each year. Years where the Laplace-approximated Bayes factor exceeds a threshold are flagged as disrupted — genuine departures from the underlying trend that should not be fitted into the structural model. Clark curates event dictionaries of known shocks (major wars, the 1918 influenza pandemic, COVID-19); the Bayes-factor criterion provides an automated screen for anomalies the dictionaries might miss.

**3. Summary indicator prediction.** Clark reformulates his prior SVD-Comp model in tensor coordinates. Given only two summary mortality indicators — the under-5 mortality rate and the 45q15 adult mortality probability — MDMx recovers the full age schedule of logit(₁qₓ). This is valuable for countries with incomplete vital statistics. For UK insurers, the relevance is indirect: the mechanism confirms that MDMx's age structure is internally coherent.

**4. Forecasting.** The approach proceeds in two stages. First, PCA is applied to the country-year slice of the core tensor — the G_ct matrices — to extract the dominant temporal dynamics. Second, a damped local linear trend Kalman filter projects each principal component forward, with hierarchical drift: 80% from HMD-wide trends and 20% from the individual country's own history. Prediction intervals come from the Kalman uncertainty, which gives the 93.7% empirical coverage at nominal 95% seen in the cross-validation.

---

## Accuracy

Clark reports rolling-origin cross-validation with six origins and a 15-year forecast horizon. We reproduce his headline table:

| Metric | MDMx | Hyndman-Ullah | Lee-Carter |
|--------|------|---------------|------------|
| e₀ MAE (years) | 1.44 | 1.44 | 1.73 |
| Sex-gap MAE (years) | 0.60 | 0.84 | 1.11 |
| PI coverage (95% nominal) | 93.7% | — | — |

The life expectancy MAE result is a tie with Hyndman-Ullah and a modest improvement over Lee-Carter. The sex-gap result is the clearest win. MDMx's 0.60-year MAE on the male-female life expectancy differential represents a 46% improvement over Lee-Carter's 1.11 years.

This is structural, not coincidental. In Lee-Carter, the age effect bₓ is a fixed vector and sex is handled by estimating separate models. The male-female differential evolves only through the interaction of two independent κₜ trajectories, with nothing in the model structure that enforces coherent joint dynamics. In MDMx, sex is mode 1 of the shared tensor, so male and female age patterns and temporal trends are jointly constrained by the same factor matrices. Sex coherence is built into the architecture.

Hyndman-Ullah (2007), the functional data approach using smoothed log mortality curves and product-ratio decomposition, gets close to MDMx on life expectancy but provides no prediction interval coverage comparison in Clark's experiments — the Hyndman-Ullah forecasts are point estimates for this comparison.

---

## What MDMx is not

We want to be unambiguous about two non-replacements.

**MDMx does not replace CMI_2025.** The CMI Mortality Projections Model is calibrated to England & Wales population mortality, governed by the IFoA, updated annually, and designed to produce the mortality improvement factors that UK insurers apply to their base tables. CMI_2025 is authoritative for UK population projection. MDMx is a research tool for understanding international mortality structure and forecasting across the HMD's 50-country panel. Using MDMx instead of CMI for UK projection assumptions would be methodologically indefensible and likely ungovernable under Solvency II internal model standards.

**MDMx does not replace Whittaker-Henderson smoothing or P-splines for experience analysis.** Our [`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker) library handles the core actuary's task: smoothing an insurer's own A/E mortality experience across age and duration cells to produce a credible insured-lives basis. This is a fundamentally different problem from fitting a population model to HMD data. An insured book has hundreds of thousands of exposure years at most; HMD has 2.9 million. The signal-to-noise ratio and the distributional assumptions differ accordingly.

---

## Where MDMx is actually useful for UK insurers

The useful applications are real but specific.

**International benchmarking.** MDMx clusters the UK into a peer group of Western European countries based on its mortality structure. If your pricing team is trying to assess whether UK assured-lives mortality assumptions are moving in the direction consistent with international peer experience, the Tucker-derived country factors give you a principled answer. "The UK has remained in its long-run cluster since 1990 with no structural divergence" is a different governance statement from "we looked at some ONS charts."

**Disruption detection for COVID and governance.** The Bayes-factor disruption detection is the most directly applicable piece. For a model validation exercise under the FCA's current scrutiny, being able to quantify: "UK mortality in 2021 deviated X standard deviations from its Tucker-predicted path, which is consistent with the Bayes factor exceeding the disruption threshold" is more defensible than eyeballing excess deaths charts. The framework makes the anomaly classification auditable. See [`insurance-governance`](https://github.com/burning-cost/insurance-governance) for the model validation infrastructure this would plug into.

**FCA MS24/1 response.** The FCA has given firms until 31 March 2026 to respond to the interim report. Any firm carrying pre-2020 mortality assumptions and hoping to address the FCA's legacy-assumption concern will need to demonstrate both that they are aware of current mortality trajectories and that their assumptions have been benchmarked against them. MDMx provides the international benchmark; CMI_2025 provides the UK projection. Used together, they cover both angles.

---

## Software and implementation

There is no MDMx Python package. Clark's prior model (SVD-Comp) is available on CRAN as `SVDMx` and on GitHub at `sinafala/svd-comp`, but MDMx is not yet publicly released as software.

The Tucker algebra itself is straightforward in Python. TensorLy handles the decomposition cleanly:

```python
import tensorly as tl
from tensorly.decomposition import tucker

# M has shape (2, n_ages, n_countries, n_years), entries are logit(q_x)
core, factors = tucker(M, rank=[2, 4, 8, 7], init='svd', random_state=42)
# factors[0]: sex (2×2), factors[1]: age (n_ages×4),
# factors[2]: country (n_countries×8), factors[3]: year (n_years×7)
```

The harder work is upstream: assembling the HMD data pipeline (the HMD requires registration for data access), computing logit(₁qₓ) from the period life tables, handling missing data for countries with incomplete historical coverage, and then implementing the three-stage fitting algorithm and Bayes-factor disruption detection. A faithful replication of the full MDMx methodology from the paper alone, without a reference implementation, is probably two to three weeks of careful work.

If Clark publishes an R package or Python implementation, the landscape changes significantly. We will cover it then.

---

## The honest verdict

MDMx is a technically accomplished piece of work. The Tucker architecture solves a real problem — structural sex coherence in mortality forecasting — in an elegant way. The unified four-task framework means that model life tables, fitting, prediction from summaries, and forecasting all draw on the same decomposition, which enforces a consistency that separate models cannot provide.

For UK protection pricing actuaries, this is a frontier research awareness piece, not an immediate implementation target. The regulatory hook is genuine: if the FCA asks whether your mortality assumptions are consistent with international experience, MDMx gives you a principled framework for answering. But the primary tools remain CMI_2025 for projection and Whittaker-Henderson (or P-splines) for experience graduation. MDMx sits alongside them as a benchmark and governance aid, not in their place.

The paper is at [arXiv:2603.20518](https://arxiv.org/abs/2603.20518).
