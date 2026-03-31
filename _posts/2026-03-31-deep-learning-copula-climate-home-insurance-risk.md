---
layout: post
title: "Climate Claims Forecasting with Deep Learning and Copulas"
date: 2026-03-31
categories: [climate]
tags: [climate-risk, flood, deep-learning, copula, home-insurance, ukcp18, flood-re, pra, ss5-25, precipitation, ensemble, scenario-analysis, python, uk-personal-lines]
description: "Dey (arXiv:2601.11949) builds a two-step pipeline — MLP for precipitation-to-claims, Gumbel copula for climate model uncertainty — that is methodologically sound, Canadian-only, and directly relevant to the problem UK insurers face between now and Flood Re's 2039 exit."
---

The question that keeps UK pricing actuaries up at night is not "what is the flood risk on this property?" — that one has commercial answers, imperfect but usable. The harder question is: how does the aggregate claims experience for our entire home insurance book shift under different climate trajectories over the next 10-15 years? Building-level scoring tells you the cross-sectional distribution of risk today. It does not tell you how the loss distribution for a given portfolio moves through time as precipitation regimes change.

A paper published in January 2026 by Asim Dey at Texas Tech — [arXiv:2601.11949](https://arxiv.org/abs/2601.11949) — takes a direct run at this problem. The methodology is credible, the results are encouraging, and the data is entirely Canadian. For UK insurers, all three of those facts matter.

---

## The two-step pipeline

The paper's contribution is architectural. It replaces two weak conventional choices — linear regression for claims and simple averaging of climate model outputs — with a neural network and a copula respectively.

**Step 1: DNN claim prediction**

The first step trains a feedforward MLP on observed weekly water-damage insurance claims from two unnamed mid-sized Canadian Prairie cities, using ERA-Interim reanalysis precipitation data for 2002-2011 as input. The response variable is normalised by exposure: weekly average water-damage claims per insured home. Getting the exposure denominator right is the unglamorous detail that separates a usable model from one that confounds claims frequency with portfolio growth.

The architecture is deliberately simple: three hidden layers, 64 units each, ReLU activations throughout, L2 weight regularisation and 0.2 dropout. This is not a seq2seq model or an LSTM — there is no learned recurrent structure. Instead, precipitation at multiple lags is engineered as features:

```
N_t = φ(X_t, X_{t-1}, X_{t-2}, D_t, D_{t-1}) + ε_t
```

where X_t is total precipitation in week t and D_t is maximum daily precipitation in week t. The lag structure captures the fact that riverine flooding often reflects precipitation accumulated over multiple preceding weeks, not just the current week's rainfall. Lag order was selected empirically per city (one to five weeks back). City A's best model used X_t through X_{t-2} plus D_t; City B added D_{t-1}.

Once the DNN is trained on observed 2002-2011 claims against ERA-Interim precipitation, it is applied to six sets of projected precipitation from six CMIP5 regional climate model combinations spanning RCP 4.5 and RCP 8.5 scenarios for 2021-2030. Each application produces one time series of projected weekly claims per insured home. You now have six futures, not one.

**Step 2: Gumbel copula ensemble**

Six separate projections are useful but awkward. Presenting six time series to a board is noise. Averaging them loses the information about inter-model disagreement, which is precisely where the uncertainty lives.

The solution is to treat the six projected claim series as the marginals of a multivariate distribution. Each marginal is fitted as a negative binomial (Type I, selected over Poisson, NB-II, zero-inflated Poisson, and Sichel by AIC/BIC). Then a 6-variate Gumbel copula captures the dependence structure across the six climate model outputs:

```
C(u_1,...,u_6) = exp{ -[ Σᵢ (-ln uᵢ)^θ ]^(1/θ) },  θ ≥ 1
```

The Gumbel family is the natural choice here because of its upper tail dependence — when you are asking whether extreme claims materialise, you want a copula that appropriately captures the probability of all models projecting extremes simultaneously. The fitted θ̂ = 1.327 (SE 0.024) for City A implies an upper tail dependence coefficient of roughly 0.52 — the six models are meaningfully correlated in their extreme projections. City B shows θ̂ = 1.101 (SE 0.012), meaning much weaker inter-model agreement (λ_U ≈ 0.07): the models diverge on what the tail looks like there.

The output is a single risk measure:

```
Φ(z) = P(Y_1 > z, Y_2 > z, ..., Y_6 > z)
```

This is the probability that all six climate models simultaneously project weekly claims exceeding z per insured home. It is a conservative measure — exceedance under the joint tail — and a communicable one. "Under all our climate scenarios, weekly claims exceeding X properties per thousand happens with probability Y" is a board-ready statement.

The model's in-sample RMSE sits at 0.453 (City A) and 0.461 (City B) on the weekly average claims per home scale. Both cities project noticeable increases in extreme claim probability for 2021-2030 compared with the 2002-2011 control period.

---

## Where the methodology is incomplete

This is not a production model, and treating it as one would be a mistake. The gaps are significant enough to say explicitly.

There is no documented train/validation/test split. The reported RMSE appears to be training-set performance. We do not know how the model generalises to held-out weeks or to events outside the training period — and for a model intended to extrapolate under climate change, out-of-sample performance is the only performance that matters.

There is no baseline comparison. SVM regression (from a prior paper in the same research programme) and linear models are referenced in the literature review but not benchmarked against the DNN. Without that, we cannot assess whether the neural network buys anything over a simpler model.

The Gumbel copula is selected by convention, not by AIC/BIC test against Clayton, Frank, Gaussian, or other Archimedean alternatives. For upper tail dependence the Gumbel is a reasonable prior, but the sensitivity of Φ(z) to copula family choice is unexamined.

The DNN has no seasonal component. Winter pipe freezes and summer convective flooding are different physical processes with different precipitation-to-claims relationships. Treating weekly precipitation as a fungible input across all 52 weeks of the year merges two regimes that should probably be modelled separately.

The climate scenarios are CMIP5, running RCP pathways. IPCC AR6 (2021) replaced CMIP5 with CMIP6 and RCPs with SSPs. For regulatory applications in 2026, CMIP6 alignment is increasingly expected — the Network for Greening the Financial System (NGFS) scenarios that the PRA references are CMIP6-based.

These limitations do not undermine the methodological contribution. They are reasons to treat this as an architectural template rather than a deployable model.

---

## UK data: what exists, what does not

The methodology transfers to UK data with no structural changes. The data itself is more complicated.

**Training data — precipitation**

The ERA-Interim control period precipitation in the paper has a direct UK equivalent in HadUK-Grid (v1.3.0): an observation-based gridded dataset covering the UK at 1km resolution, daily, from 1836 to 2023. Variables include daily rainfall, maximum and minimum temperature, wind speed, sunshine hours, and humidity. It is freely available via the CEDA Archive under the NERC Open Government Licence — bulk NetCDF download, no API, registration required but no cost. For flood modelling specifically, a 2023 intercomparison in Hydrology and Earth System Sciences found HadUK-Grid to outperform ERA-5 reanalysis for UK application given its better validation against rain gauge observations.

**Future projections — climate models**

The paper's six CMIP5 downscaled models have a UK equivalent in UKCP18: the Met Office's 12-member perturbed physics ensemble (PPE) at 12km regional model resolution, plus a 2.2km convective-permitting model (CPM) for historic simulations. The PPE is a larger ensemble than the paper's six models, which improves the copula calibration. The 2.2km CPM resolves sub-daily convective rainfall — critical for surface water flood risk, which is increasingly the dominant UK peril as convective storm intensity increases under warming. UKCP18 is free via CEDA; data availability guidance was updated by the Met Office in January 2026.

**EA NaFRA (January 2025)**

The Environment Agency's National Flood Risk Assessment — overhauled and published in January 2025 — now covers rivers, sea, and surface water at ~2m resolution across England. It identifies 6.3 million properties at flood risk, projecting to 8 million by mid-century. NaFRA is useful as a static risk stratifier: it tells you which properties carry flood risk, but not how that aggregate changes week by week. It complements rather than feeds the DNN-copula pipeline.

**What you cannot get for free: claims data**

This is the actual barrier. HadUK-Grid exists. UKCP18 exists. What does not exist in any public form is weekly water-damage claims data, normalised by insured exposure, for any UK geographic unit. The Dey model was trained on exactly this. UK insurers hold it internally. The ABI publishes aggregate annual figures — £714 million expected annual flood losses, £4.6 billion total property payouts in the first three quarters of 2025. But weekly claims aggregated to a city or region, with insured home counts as the denominator, is proprietary.

---

## The Flood Re complication

The UK data gap is bad enough on its own. Flood Re makes it structurally worse.

Flood Re is the household flood reinsurance backstop, launched in 2016, covering roughly 250,000 policies in high-risk properties. For eligible properties — pre-2009 residential, lower council tax bands — insurers can cede flood risk into the scheme, priced on council tax band rather than actual risk. Claims on Flood Re-ceded policies are settled by Flood Re, not by the ceding insurer. Those claims appear in the insurer's accounts as reinsurance recoveries, not as gross claims.

Any insurer attempting to train a claims-versus-precipitation model on its own data will be working with a truncated claims series. The properties most likely to generate large precipitation-driven claims are those with the highest flood risk — and precisely those are most likely to be ceded into Flood Re. The DNN learns on a biased signal. The portion of the tail that most matters for calibrating extreme scenarios is also the portion most likely to have been transferred out of the observable claims data.

Reconstructing the true gross claims series requires the Flood Re ceded claims, which means a data sharing arrangement with Flood Re Ltd. Without that, any UK implementation of this methodology carries a systematic downward bias in the claim response variable.

Flood Re has a designed end date of 2039. The scheme's quinquennial review (2023) reaffirmed the exit, and a staircase of increasing cession premiums is intended to force market pricing progressively before the backstop disappears. When the scheme closes, the Flood Re truncation problem disappears — but so does the reinsurance subsidy, and pricing teams without credible climate-conditioned loss models will face a difficult transition. Building the capability now, working with synthetic or reconstructed claims where necessary, is the sensible path.

---

## PRA SS5/25 and where this fits

The PRA's final supervisory statement SS5/25 (PS25/25, effective 3 December 2025, superseding SS3/19) requires insurers to quantify potential losses on their underwriting portfolio under climate scenarios, integrate physical risk into ORSA stress testing, and for firms with material exposures, apply "more mathematically sophisticated methods." The gap review deadline is 3 June 2026.

The DNN-copula framework maps reasonably well onto three SS5/25 requirements:

**Physical risk quantification.** The Φ(z) output is exactly the kind of scenario-conditioned loss distribution SS5/25 requires. "Under all six climate models we consider, the probability that weekly aggregate claims exceed X per insured home is Y%" is a direct SS5/25-compatible statement.

**Multi-scenario coverage.** Running the copula across 12 UKCP18 PPE members (vs the paper's six) produces a richer characterisation of climate uncertainty than manually defined narrative scenarios. The copula parameterises the inter-model agreement, giving Φ(z) a clean communication for boards: the θ parameter is directly interpretable as "how much do our climate models agree on extreme outcomes?"

**ORSA stress testing.** If implemented at portfolio level with realistic UK claims data (including Flood Re reconstruction), Φ(z) maps onto reserve adequacy questions: given our technical provisions, what is the probability that climate-conditioned claims exhaust them under each scenario?

What the framework does not deliver is the Solvency II 1-in-200-year loss for SCR purposes. Φ(z) operates at weekly resolution on aggregate claims. Translating it into an annual aggregate VaR at the 99.5th percentile for a given year requires additional simulation architecture — summing the weekly series under each joint scenario draw, taking the 99.5th percentile of the resulting annual distribution, and computing the capital requirement. Achievable, but not in the paper.

---

## How this fits alongside existing UK flood modelling

This post sits alongside [our earlier building-level flood risk piece](/2026/03/31/beyond-flood-zone-3-building-level-property-risk-python/), which covered the cross-sectional underwriting problem: given this specific property's characteristics, what is its expected flood loss? That is a pricing question. The DNN-copula framework answers a different question: how does aggregate portfolio loss shift over time under different climate trajectories? That is a reserving and ORSA question.

The two approaches are complementary and, combined, produce something more useful than either alone. Score individual risks with a building-level model (Moriah-type, EA NaFRA features, UKCP18 local weather history). Aggregate to portfolio level. Apply DNN-copula climate stress factors derived from UKCP18 PPE projections. The result is a portfolio-level climate stress output with building-level risk stratification underneath — the kind of architecture that SS5/25 asks for when it references "more mathematically sophisticated methods" for firms with material exposures.

The commercial CAT models — JBA UK Flood at 5m, Fathom UK at 10m, Moody's RMS HD at up to 1m — handle this well for regulatory ICA and Lloyd's RDS requirements, but through a different route: hydraulic routing, event sets, vulnerability functions calibrated to depth-damage curves. Their strength is physical fidelity for individual events. The DNN-copula framework's strength is that it learns from actual claims data, which means it captures insurer-specific portfolio effects and behavioural factors (claims reporting, settlement timing, sub-limit structures) that generic vulnerability curves do not. For a reasonably-sized insurer with quality internal data, building this as a supplementary climate scenario tool for ORSA purposes is plausible. It does not replace JBA or RMS for capital modelling, but it does not need to.

---

## What a UK implementation needs

To move from the paper's template to a working UK model, five things are required:

1. **Weekly claims data, gross of Flood Re.** This means either a data sharing arrangement with Flood Re Ltd (which provides access to ceded claims by insurer), or restricting the training data to the non-Flood-Re segment of the book and accepting the calibration bias explicitly.

2. **Exposure normalisation.** Weekly counts of insured homes, by geographic unit, over the training period. This is internal policy data. It sounds simple and it is not — exposure changes as policies lapse, renew, and are written, and matching exposure to the weekly claims aggregation requires a disciplined policy database.

3. **HadUK-Grid alignment.** Aligning the 1km daily precipitation grids to geographic units of observation (city-equivalent regions in the paper, postcode sectors or Local Authority districts in a UK context). Bulk NetCDF download and spatial aggregation. Not technically hard, but not small data engineering either.

4. **UKCP18 PPE download and preprocessing.** Twelve-member ensemble at 12km for the projection period, matched to the same geographic units. Same NetCDF pipeline as HadUK-Grid, plus bias correction between the HadUK-Grid control climatology and the UKCP18 baseline.

5. **Copula selection, not just Gumbel by convention.** Run AIC/BIC selection across Gumbel, Clayton, Frank, and Gaussian copula families. The Gumbel is a reasonable prior for flood-driven claims (upper tail dependence is appropriate) but the result should be tested, not assumed.

The compute costs are minimal — the MLP trains in seconds, the copula MLE is a one-parameter optimisation. The bottleneck is data engineering and claims reconstruction.

---

## Our read on the paper

Dey (2026) does one thing well and one thing less well. What it does well is the ensemble framing: the insight that climate model uncertainty should be represented as a multivariate distribution, not as a range of scalars, is correct and underused. The Gumbel copula over climate model outputs is a clean solution to the "what do I do with six different futures?" problem. That framing transfers to UK data directly.

What it does less well is validation. A model with no held-out test set, no baseline comparison, and copula selection by convention is a proof of concept, not a production model. The RMSE figures reported are likely optimistic. The absence of seasonal structure is a meaningful simplification that would need addressing for UK perils where winter pipe freeze dominates claims frequency.

The right use for this work in a UK insurance context is as a template for an ORSA climate scenario tool. It is not a replacement for commercial CAT models and should not be presented to regulators as one. Used correctly — paired with HadUK-Grid, UKCP18 PPE, Flood Re-adjusted claims data, and proper train/validation splits — it provides an interpretable, internally-built pathway to the multi-scenario physical risk quantification that SS5/25 asks for. The June 2026 gap review deadline gives teams roughly ten weeks. That is not enough time to build this properly from scratch, but it is enough time to demonstrate a credible plan.

---

*Paper: Dey (2026), "A Deep Learning-Copula Framework for Climate-Related Home Insurance Risk," arXiv:2601.11949.*
*Related: [Building-Level Flood Risk with Python](/2026/03/31/beyond-flood-zone-3-building-level-property-risk-python/) — [Physical Climate Risk in UK Home Insurance Pricing](/2026/03/25/physical-climate-risk-uk-home-insurance-pricing/)*
