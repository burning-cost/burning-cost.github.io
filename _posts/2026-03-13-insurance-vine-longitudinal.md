---
layout: post
title: "D-Vine Copulas for Longitudinal Claim Histories: What NCD Discards"
date: 2026-03-13
categories: [libraries, experience-rating, copula]
tags: [d-vine-copula, ncd, experience-rating, longitudinal, temporal-dependence, python, pyvinecopulib]
description: "NCD collapses 20 years of claim history into a single discount level. D-vine copulas model the full temporal dependence structure — what actually happened, when, and how it predicts what comes next. We've built the first Python implementation of the Yang-Czado two-part longitudinal vine framework."
---

An NCD level is a single integer. For most UK motor policyholders it represents one of seven discount steps — 0%, 20%, 30%, 40%, 50%, 60%, or 65% — and that integer is the sum total of everything your pricing model knows about a policyholder's individual claim history.

Not when they claimed. Not how much. Not whether they had a single severe claim or multiple small ones. Not whether their claim frequency is trending up or whether their last claim was five years ago and atypical. Not the correlation structure between what they did in year one and what they did in year three. One integer.

Yang and Czado (2022, *Scandinavian Journal of Statistics*, 49(4):1534–1561) built a model that uses all of that. It captures the full joint distribution of a policyholder's claim history across years — the temporal dependence structure — and conditions on it to predict next year's claim probability and severity distribution. The model is a two-part D-vine copula. The technique has existed for four years. There was no Python implementation.

We have built [`insurance-vine-longitudinal`](https://github.com/burning-cost/insurance-copula).

---

## What is wrong with NCD

NCD is a Markov chain on a discrete state space. From any given NCD level, a claim drops you by a fixed number of steps and a clean year advances you by one. The transitions are deterministic and the same for everyone. The model treats a single claim by a 20-year claim-free policyholder identically to a single claim by a policyholder who claimed three of the last five years.

Three structural failures follow from this.

**Severity is ignored.** A £150 windscreen claim and a £45,000 third-party personal injury both trigger the same NCD drop. From the NCD model's perspective they are the same event. From a risk perspective they signal very different things about the underlying policyholder.

**The state space is too coarse.** Seven NCD levels compress a continuous risk signal into seven integers. The transition from 0% to 20% NCD on a single claim-free year after a claim is the same regardless of whether the policyholder previously had a clean 10-year record or a spotty 3-year one. History before the last claim is erased.

**The dependence structure is misspecified.** NCD assumes that your NCD level is a sufficient statistic for your claim history. It is not. The specific timing, frequency, and severity pattern of your claims over time carries additional predictive information. The copula model captures this; NCD discards it.

The non-monotone threshold problem is a fourth failure that NCD practitioners know but rarely discuss. At maximum NCD (65% in most UK schemes), a policyholder who claims loses their NCD discount, raising their premium on renewal by more than the claim cost. The optimal policyholder strategy is to self-fund claims below a threshold — actuarially estimated at roughly 20–30% of the maximum NCD discount value. The 65% maximum NCD creates worse selection at the top of the scheme than at the bottom. The scheme design is perverse.

---

## What D-vine copulas actually model

A copula separates the marginal distributions of random variables from their dependence structure. For a bivariate case: instead of fitting the joint distribution of (X, Y) directly, you fit each margin separately and then model the dependence between the probability integral transforms U = F_X(X) and V = F_Y(Y). Sklar's theorem guarantees this is lossless — any joint distribution can be decomposed this way.

A vine copula extends this to T dimensions using a sequence of nested bivariate copulas, one pair at a time. The D-vine is a specific vine architecture designed for temporal data: it arranges the variables in a sequence (year 1, year 2, ..., year T) and builds the copula tree by pairing adjacent years first, then conditioning on intermediate years to model higher-order lags.

The structure for T=5 policy years looks like this:

```
Tree 1:  (1,2)  (2,3)  (3,4)  (4,5)         — adjacent-year pairs
Tree 2:  (1,3|2)  (2,4|3)  (3,5|4)           — lag-2 pairs, conditioned on year between
Tree 3:  (1,4|2,3)  (2,5|3,4)                — lag-3 pairs
Tree 4:  (1,5|2,3,4)                          — lag-4 pair
```

Each node in each tree is a bivariate copula chosen from a parametric family (Gaussian, Frank, Clayton, Gumbel). The tree structure means the model can capture the dependence between year 1 and year 5 separately from the dependence between year 4 and year 5 — they are different pair copulas fitted independently.

For insurance, Yang and Czado impose two simplifications that make the model tractable.

**Stationarity.** The pair copula at lag k (e.g., the Clayton copula capturing year-to-year dependence for adjacent years) is assumed to be the same regardless of which two adjacent years you are looking at. A 2019–2020 pair has the same dependence structure as a 2022–2023 pair. This reduces the parameter count from T(T-1)/2 distinct copulas to T-1 distinct parameters — one per lag level.

**Truncation.** A D-vine truncated at order p treats all pair copulas beyond tree level p as independence copulas. This is equivalent to a p-th order Markov assumption: year T's claim is conditionally independent of year T-3 given years T-1 and T-2, if p=2. BIC selects the truncation order. For most insurance panel datasets, p=1 or p=2 suffices — recent years are what matters.

---

## The two-part structure for insurance claims

Insurance claim data is not continuous. For any given policy year, most policyholders record zero claims. The positive claim amounts follow a heavy-tailed distribution. Modelling these with a single vine copula would require mixing continuous and point-mass components in a way that complicates the pair copula fitting substantially.

Yang and Czado split the problem cleanly into two separate D-vines.

**Occurrence D-vine.** A vine copula fitted on binary indicators Y_t ∈ {0, 1}: did policyholder i claim in year t? The discrete marginal is handled by treating a binary variable as a degenerate continuous case using [F(1), F(0)] two-block pseudo-observations — pyvinecopulib's `var_types=['d']` mechanism. The resulting vine captures temporal persistence in claim occurrence.

**Severity D-vine.** A vine copula fitted on positive claim amounts X_t | X_t > 0: given that a claim occurred, what was its size? Only positive observations enter. The marginal is a gamma or log-normal GLM fitted on policyholder and vehicle covariates. The copula is fitted on the probability integral transform residuals from that GLM — it captures what the marginal GLM misses.

The two vines are fitted independently. Prediction for next year's risk premium combines them:

```
E[cost in year T | history] = P(claim | occurrence history) × E[severity | severity history, claim occurred]
```

Both conditional quantities come from the respective D-vine's h-function recursion.

---

## Prediction via h-function recursion

The computational core of D-vine prediction is the h-function. For a bivariate copula C(u, v), the h-function is the conditional distribution of U given V:

```
h(u | v) = ∂C(u, v) / ∂v
```

Every parametric copula family has an analytical h-function. Gaussian: it is a normal CDF evaluated at a linear function of the conditioning value. Clayton: it is a power function. These are fast to evaluate.

To predict year T given history (u_1, ..., u_{T-1}), you apply h-functions recursively up the vine tree:

1. Start with the PIT-transformed observed history: u_1, ..., u_{T-1}.
2. Apply the lag-1 pair copula h-function to adjacent pairs: this gives pseudo-conditioned quantities at tree level 2.
3. Continue up the tree until you have consumed all observations up to tree level p.
4. The resulting quantity is the conditional CDF evaluated at an unobserved u_T.

For the occurrence vine, this gives P(Y_T = 1 | y_1, ..., y_{T-1}). For the severity vine, it gives F(X_T | x_1, ..., x_{T-1}), which you invert numerically for quantiles or integrate for the conditional mean.

pyvinecopulib provides these h-functions. The tricky part is the partial Rosenblatt transform needed to condition on a subset of dimensions — the library's `inverse_rosenblatt` operates on the full vine. We implement the partial transform directly using pyvinecopulib's individual `hfunc1` and `hfunc2` methods per pair copula, following the algorithm in arXiv:2102.06416, Section 4.2.

---

## Using `insurance-vine-longitudinal`

```bash
pip install insurance-vine-longitudinal
# or
uv add insurance-vine-longitudinal
```

The library expects panel data: a DataFrame where each row is a (policyholder, year) observation. The core class is `TwoPartDVine`.

### Fitting the model

```python
import polars as pl
from insurance_vine_longitudinal import TwoPartDVine, PanelDataset

# Panel data: one row per (policy_id, year)
# Columns: policy_id, policy_year, had_claim (0/1), claim_amount,
#          age_band, region, vehicle_group, ncd_years
df = pl.read_parquet("motor_panel_2015_2024.parquet")

panel = PanelDataset.from_dataframe(
    df,
    id_col="policy_id",
    year_col="policy_year",
    claim_col="had_claim",
    severity_col="claim_amount",
    covariate_cols=["age_band", "region", "vehicle_group"],
)

# Fit: GLM marginals + stationary D-vine + BIC truncation selection
model = TwoPartDVine(max_trunc=4, family_set="parametric")
model.fit(panel)

print(f"Occurrence vine: truncation order p={model.occurrence_vine_.truncation_level}")
print(f"Severity vine:   truncation order q={model.severity_vine_.truncation_level}")
# Occurrence vine: truncation order p=1
# Severity vine:   truncation order q=2
```

A truncation order of p=1 for occurrence means: knowing last year's claim outcome is sufficient. Once you condition on year T-1, year T-2 and earlier add no further information about year T's claim probability. This is the most common result on motor books.

Truncation order q=2 for severity is slightly less common but intuitive: large claims are correlated across adjacent years (the same underlying risk environment produces large severity in consecutive years), and conditioning on the last two years of severity carries information about next year's severity even after conditioning on the GLM covariates.

### Predicting risk premiums

```python
# History data: policyholders we want to price for next year
# Provide their last 1-3 years of claims history (more years = better prediction)
history_df = pl.read_parquet("renewals_2025.parquet")

# Full conditional prediction
predictions = model.predict_premium(history_df, loading=0.0)
# Returns Polars DataFrame with columns:
#   policy_id, claim_prob, expected_severity, risk_premium

print(predictions.head(5))
# shape: (5, 4)
# ┌───────────┬────────────┬──────────────────┬──────────────┐
# │ policy_id ┆ claim_prob ┆ expected_severity ┆ risk_premium │
# │ ---       ┆ ---        ┆ ---               ┆ ---          │
# │ str       ┆ f64        ┆ f64               ┆ f64          │
# ╞═══════════╪════════════╪══════════════════╪══════════════╡
# │ P001      ┆ 0.041      ┆ 2840.0            ┆ 116.44       │
# │ P002      ┆ 0.089      ┆ 3210.0            ┆ 285.69       │
# │ P003      ┆ 0.031      ┆ 1950.0            ┆ 60.45        │
# │ P004      ┆ 0.112      ┆ 4100.0            ┆ 459.20       │
# │ P005      ┆ 0.038      ┆ 2600.0            ┆ 98.80        │
# └───────────┴────────────┴──────────────────┘──────────────┘
```

### Experience relativities: the actuarial output

The prediction above includes copula-estimated conditional claim probability, not just a priori GLM prediction. The *experience relativity* is the ratio of the copula-adjusted premium to the a priori GLM premium — how much this policyholder's specific history is moving their price relative to their demographic baseline.

```python
# Extract relativities: how much does history move the price?
relativities = model.experience_relativity(history_df)

# And generate a relativity table for underwriting guidelines
table = model.extract_relativity_curve(
    claim_counts=[0, 1, 2, 3],
    n_years=[1, 2, 3, 4, 5],
)
print(table)
#    n_years  claim_count  relativity
# 0        1            0        0.94
# 1        1            1        1.18
# 2        2            0        0.88
# 3        2            1        1.09
# 4        2            2        1.41
# 5        3            0        0.83
# 6        3            1        1.04
# 7        3            2        1.28
# 8        3            3        1.67
# ...
```

This table is directly usable by a pricing team. Three years claim-free is a 0.83 relativity on the a priori rate. One claim in three years is 1.04. Three claims in three years is 1.67. These come from the fitted copula parameters — they are data-driven, not assumed. A traditional NCD scheme imposes these factors by design; the vine model estimates what the data actually supports.

### Severity quantiles for excess-of-loss reinsurance

```python
# Conditional severity quantiles for a policyholder with specific history
quantiles = model.predict_severity(
    history_df,
    quantiles=[0.50, 0.75, 0.90, 0.95, 0.99],
)
# Returns conditional quantile of claim severity given history
# Useful for individual excess-of-loss retention decisions
```

The 99th percentile of conditional severity is not accessible from NCD. NCD does not model severity at all. The vine model gives the full conditional distribution — you can extract any quantile needed for reinsurance pricing or retention layer optimisation.

---

## When to use this versus NCD or credibility

This is not a wholesale replacement for NCD. It is a fundamentally different tool with a different purpose.

**NCD** is a pricing input in UK personal motor. Post FCA PS21-5 (January 2022, which banned renewal price-walking), NCD level remains one of the primary legitimate experience signals a motor insurer can use at renewal without FCA concern. It is simple, auditable, and every policyholder understands it. We are not suggesting you bin your NCD scheme.

**Bühlmann-Straub credibility** is appropriate for fleet accounts and SME commercial where you have enough exposure to estimate an individual account's long-run mean loss ratio and blend it with the portfolio mean. It is linear in the observed loss ratio and uses only the first moment of claim history. For personal lines with 1–3 years per policyholder, credibility weights are very low and the model adds little beyond the a priori GLM.

**D-vine copulas** are appropriate when:

- You have 3+ years of panel data per policyholder (the model requires minimum T=2, but T=3–5 produces meaningfully better predictions)
- Temporal dependence in occurrence or severity is statistically detectable (BIC selecting p>=1 tells you this)
- You want to price renewal risk beyond what the NCD level captures
- You need the full conditional severity distribution, not just expected cost
- You are building a reinsurance pricing model that requires tail-conditional quantities

The natural home for this model in UK motor is not the consumer-facing NCD discount (where simplicity and FCA auditability govern) but the internal risk segmentation that sits underneath: the model that determines what the a priori rate should be before NCD is applied, adjusted for claim history signals that NCD cannot capture.

The Consumer Duty angle is also real. FCA PS21-5 created the need to justify renewal pricing as reflecting genuine risk differences, not just price-walking. A documented D-vine model that explicitly decomposes temporal claim persistence into a legitimate risk signal satisfies that requirement in a way a gut-feel NCD override does not. The model's conditional probability estimate is an auditable number. You can show the calculation. You can explain which years of history drove it and why.

---

## Limitations worth knowing before you build

**Panel data requirement.** The model is useless without multiple years per policyholder. If you run a book with high annual turnover — aggregator-driven personal lines where 60%+ of policies lapse after year one — the effective panel depth is too low for the vine to learn temporal dependence. The copula needs pairs of years for the same policyholder. Without pairs, there is no within-policyholder temporal signal.

**Stationarity assumption.** The model assumes year-to-year temporal dependence is stable through calendar time. COVID-era lockdowns in 2020–2021 broke that assumption for UK motor: claim frequency dropped sharply and rebounded. Claims inflation in 2022–2023 broke it for severity. If your training window spans structural breaks, the stationary vine will average over them rather than capturing the true dependence at any given time. Fit the model on a 3–5 year rolling window, not a 10-year panel.

**Minimum T per policyholder.** The vine needs T>=2. Policyholders with only one year of history get the a priori GLM prediction with no copula adjustment. This is correct — you have no temporal information about them — but it means the model's experience rating signal is unavailable for first-year policyholders. That is not a limitation of the vine; it is a genuine information constraint.

**Computational cost of the full severity integral.** Computing E[X_T | history] requires numerical quadrature over the conditional severity CDF. For large renewal portfolios (100,000+ policyholders), this adds seconds per policyholder. We recommend using a 100-point grid integration (the library default) rather than simulation for premium prediction; simulation is available for quantile estimation but is slower. The occurrence probability prediction is fast — it is a single h-function recursion chain, not an integral.

**Interpretation of copula family.** BIC selects the copula family at each lag level from a candidate set (Gaussian, Clayton, Frank, Gumbel). The selected family tells you about the *tail dependence* structure: Clayton has lower tail dependence (bad years cluster together more than good years), Gumbel has upper tail dependence (large claims cluster). In UK motor data, Clayton at lag 1 is the most common selection — which means consecutive-year claim occurrence clusters in the left tail (either both years clean or both years claiming). This is interpretable and matches intuition. Report the selected family in your documentation.

---

## The Wisconsin benchmark

Yang and Czado (2022) validated the model on the Wisconsin Local Government Property Insurance Fund — a panel of 1,000+ US local government entities (counties, cities, school districts) covering building, contents, and inland marine risks over multiple years. It is the standard benchmark for longitudinal insurance copula methods and has since been used by Shi and Zhao (2024, *Journal of Econometrics*, 240(1):105676) to demonstrate a 9% lift in insurer net revenue from dependence-aware experience rating versus an independence-assuming model.

Shi and Zhao extend the Yang-Czado framework to multi-peril bundled risks: a two-tier architecture where within-risk temporal D-vines (the Yang-Czado part) are integrated with a cross-peril contemporaneous copula to model the correlation between, say, building claims and inland marine claims in the same year. For a UK insurer with combined household policies, this is the v2 direction. `insurance-vine-longitudinal` implements the single-peril Yang-Czado core. Multi-peril extension is on the roadmap.

The 9% revenue figure from Shi and Zhao is worth calibrating. It comes from a US commercial property fund, not UK personal motor. The lift reflects dependence-aware renewal pricing against an insurer that was ignoring temporal dependence entirely — not a comparison against an insurer already running a sophisticated credibility model. In a UK motor context where credibility models are already common for fleet accounts, the marginal gain from the vine approach over credibility will be smaller. For personal lines where experience rating is dominated by NCD, the comparison is more favourable.

---

## What is new in this library

No Python implementation of the Yang-Czado two-part longitudinal D-vine existed before this. The R implementation in their 2022 paper uses the `VineCopula` package directly and is research code — not a practitioner workflow, no panel ingestion, no marginal fitting pipeline, no experience relativity output.

pyvinecopulib 0.7.5 provides the engine: the `DVineStructure` class, stationary vine fitting via `Vinecop.from_data()`, truncation level selection via `FitControlsVinecop(truncation_level=p)`, and the pair copula h-functions we use for conditional prediction. What it does not provide — and what this library adds — is the insurance workflow: two-part separation, panel data ingestion and validation, GLM marginal fitting and PIT extraction, BIC truncation selection across both vines, and conditional prediction output in actuarial units (relativities, quantiles, risk premium).

```bash
pip install insurance-vine-longitudinal
```

Source: [github.com/burning-cost/insurance-copula](https://github.com/burning-cost/insurance-copula)

---

*Papers: Yang & Czado (2022) Scandinavian Journal of Statistics doi:10.1111/sjos.12566 — Shi & Zhao (2024) Journal of Econometrics doi:10.1016/j.jeconom.2024.105676*
**Related articles from Burning Cost:**
- [Individual Experience Rating Beyond NCD: From Bühlmann-Straub to Neural Credibility](/2026/03/24/insurance-experience/)
- [Vine Copulas for Correlated Peril Pricing](/2026/03/12/insurance-copula/)
- [Shared Frailty Models for Recurrent Insurance Claims](/2026/03/12/insurance-recurrent/)
