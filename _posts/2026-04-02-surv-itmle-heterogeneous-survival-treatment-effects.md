---
layout: post
title: "The Survival Treatment Effect Your Retention Model Is Not Computing"
date: 2026-04-02
categories: [techniques, causal-inference, libraries]
tags: [survival-analysis, causal-inference, heterogeneous-treatment-effects, targeted-learning, tmle, ipcw, insurance-uplift, insurance-survival, lapse, renewal, telematics, python, econml, lifelines, arXiv-2603-26502, Pryce, Diaz-Ordaz, Keogh, Vansteelandt]
description: "If you have called RetentionUpliftModel with outcome='survival', your model silently ran as binary. There is no pip-installable Python package for survival CATE. We explain the correct IPCW approximation, what surv-iTMLE would add on top, and when each matters for UK pricing."
math: true
author: burning-cost
---

If you have called `RetentionUpliftModel` from `insurance-uplift` with `outcome='survival'`, your model ran as binary. The survival pathway is a documented stub — the parameter is accepted, stored, and then ignored. Every uncensored and censored renewal observation goes through the standard `CausalForestDML` path, with no censoring correction applied. This is not a subtle numerical issue. It produces biased CATE estimates whenever your data has censored observations, which is most renewal data extracts.

This post explains what the correct computation looks like, what to use now, and what a paper published last week — Pryce, Diaz-Ordaz, Keogh, and Vansteelandt (arXiv:2603.26502, March 2026) — would add on top when its code eventually arrives.

---

## Why the stub exists and what it costs you

The `outcome='survival'` parameter was designed to handle a real problem: renewal data is time-to-event data, not binary data. A customer observed still in-force at the data extract date is censored — you do not know when or whether they will lapse. Dropping censored observations (which is what the current binary path effectively does) biases the CATE in a predictable direction: customers who were in-force longer at the extract date are systematically removed, and those tend to be the stickier, less price-sensitive customers. The CATE estimate for price sensitivity is therefore biased upward — you overestimate how much a price increase drives lapse.

In a standard UK motor book with a rolling 12-month data extract, approximately 30–50% of the renewal cohort is right-censored at any point-in-time extract. Running binary CATE on the uncensored subset treats those customers as if they do not exist. The resulting treatment effect estimates are wrong in a way that is hard to detect from model diagnostics alone — the model converges, the coverage tests pass, the CATE distribution looks plausible.

The bug is silent precisely because the parameter is accepted. If `outcome='survival'` raised a `NotImplementedError`, teams would know to handle censoring manually. Because it does not, the incorrect model runs without warning.

---

## The correct approximation: IPCW + CausalForestDML

The right Python-native approach is **inverse probability of censoring weighting (IPCW)** combined with a standard causal forest. The logic: if we know the probability that observation $i$ was not censored — call it $G(t_i \mid X_i)$ — we can upweight uncensored observations by $1/G(t_i \mid X_i)$ to account for the missing censored mass. The causal forest then receives a reweighted dataset that approximately represents the full population.

The steps:

```python
import numpy as np
from lifelines import KaplanMeierFitter
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Fit censoring survival function G(t | X)
# Simple marginal KM; stratified by treatment arm for better adjustment
kmf = KaplanMeierFitter()
kmf.fit(durations=time_obs, event_observed=~censored)   # event = NOT censored

# IPCW weight for each uncensored observation
# G_hat(t_i) = P(C >= t_i) estimated from KM
def ipcw_weight(t, kmf, clip_min=0.01):
    g = kmf.survival_function_at_times(t).values
    return 1.0 / np.clip(g, clip_min, 1.0)

# Apply to uncensored observations only
mask_uncensored = ~censored
w = ipcw_weight(time_obs[mask_uncensored], kmf)

# Binary outcome: did the event (lapse) occur at observed time?
y_binary = event_indicator[mask_uncensored].astype(float)

# IPCW-weighted causal forest
cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=200),
    model_t=GradientBoostingClassifier(n_estimators=200),
    n_estimators=500,
    min_samples_leaf=10,
    random_state=42,
)
cf.fit(
    Y=y_binary,
    T=treatment[mask_uncensored],
    X=features[mask_uncensored],
    sample_weight=w,
)

tau_hat = cf.effect(features_score)
```

This is approximately 200 lines of production code to do correctly — handling the stratification of the censoring model by treatment arm, clipping extreme weights, and propagating weights through the forest's honest splitting. It is implementable in a week. It is what `outcome='survival'` in `insurance-uplift` should do.

Two limitations of this approximation:

1. **Right-censored data only.** If your data has left truncation — policies that entered your extract mid-term because they started after the observation window opened — IPCW does not correct for the truncation bias. More on this below.
2. **No smooth effect curves over time.** The result is a scalar CATE per customer: the estimated effect of treatment on lapse probability at the observed time horizon. It does not give you the lapse trajectory over 0–365 days post-renewal.

For most standard UK personal lines renewal analyses, limitation 1 is manageable: extract data aligned to policy inception dates largely avoids left truncation. Limitation 2 is real but often acceptable — a single CATE at the 12-month mark is sufficient for many pricing decisions.

---

## What surv-iTMLE adds

Pryce, Diaz-Ordaz, Keogh, and Vansteelandt (arXiv:2603.26502) published a method last week that addresses both limitations. The estimand is:

$$\tau(x, t) = \mathbb{E}[S(t \mid X=x, A=1) - S(t \mid X=x, A=0)]$$

for all $t$ in $[0, T_{\max}]$ simultaneously. Not a scalar per customer — a **curve over time**, conditioned on covariates.

On top of handling right censoring and left truncation jointly, the algorithm's 'individual' targeting step — shared fluctuation parameter $\epsilon$ across all time points — produces smooth, monotone effect curves rather than the jagged per-point estimates that IPCW + causal forest generates.

The algorithm is a TMLE in three stages: cross-fitted nuisance estimation (propensity, conditional survival, censoring survival, and an entry time hazard for left truncation), a single logistic-regression TMLE fluctuation step targeting the efficient influence function across all time points jointly, and influence-function variance with simultaneous confidence bands over $t$.

We covered the methodology in detail [yesterday's post](/techniques/causal-inference/2026/04/01/surv-itmle-survival-causal-heterogeneous-treatment-effects-insurance-pricing/). The short version for practitioners: it is theoretically the right answer, it handles data structures that IPCW cannot, and there is currently **no Python package for it**. No R package either. The paper is five days old.

---

## Where left truncation actually bites in UK data

Left truncation is less exotic than it sounds in UK insurance. Any time your data extract is not aligned to policy inception dates, you have it.

The most common scenario: you pull a snapshot of in-force policies on 31 December. A policy that started in March is in your data only because it survived April through December without lapsing. Modelling renewal intent on this extract without adjusting for truncation means your hazard estimates are biased downward — the policies look stickier than the full cohort would, because you are conditioning on nine months of survival.

For a comparison-site motor book where policies start continuously through the year, a fixed-date extract has approximately 30–40% of policies in a left-truncated state. For commercial lines with multi-year contracts and annual data pulls, left truncation is the dominant structure.

The `insurance-survival` library handles left-truncated survival modelling for prediction. `CauseSpecificMortality` and `AJRecalibrator` both accept entry times. The gap is on the causal side: left-truncated CATE has no Python implementation anywhere.

---

## Decision guide

| Data structure | Available in Python now | What to use |
|---------------|------------------------|-------------|
| Right-censored only, scalar CATE sufficient | Yes | IPCW + `CausalForestDML` — fix the `insurance-uplift` stub |
| Right-censored only, effect curve over $t$ needed | Partially | `grf::causal_survival_forest` via `rpy2` (right-censored, no smoothness guarantee) |
| Left-truncated + right-censored | No | Wait for surv-iTMLE code; log the truncation bias as a known limitation |
| Any of the above, R acceptable | Yes | `grf::causal_survival_forest` for scalar; `survtmle` for marginal ATE |

In practice: for most UK personal lines renewal analyses with a renewal-date-aligned extract, IPCW + CausalForestDML is the right move today. For comparison-site books where the extract is not inception-aligned, document the left truncation bias, use the IPCW approximation, and treat the results with appropriate scepticism for the most recently acquired policies.

---

## The broader Python gap

There is no pip-installable package for survival CATE in Python. The state of play as of April 2026:

- `econml` (v0.16.0): causal forest for binary outcomes; IPCW weighting is manual
- `grf` in R: mature, right-censored causal survival forest since 2021; accessible via `rpy2` but fragile in CI
- `SurvITE` (NeurIPS 2021, chl8856/survITE): deep learning, research code, GPU required
- `survtmle` (Benkeser, R/CRAN): marginal ATE only, not heterogeneous
- surv-iTMLE (this paper): no code released
- Orthogonal survival learners (Frauen et al., arXiv:2505.13072, May 2025): no code released

This is a genuine gap that the `insurance-uplift` stub was intended to fill. The IPCW approach we describe above is what the stub should implement as its first pass — it handles right-censored data correctly, does not require R, and gives you heterogeneous treatment effects. Implementing it requires `lifelines`, which is already in the ecosystem.

We are not implementing surv-iTMLE from scratch. The joint temporal targeting step requires getting the TMLE fluctuation numerics right without a reference implementation to validate against, and the risk of shipping incorrect influence function variance is too high. When Pryce and co-authors release code, we will reassess.

---

## What to watch

The paper is from a strong team: Vansteelandt (Ghent/UCL) is one of the foremost causal inference methodologists in Europe; Diaz-Ordaz (UCL) has a track record on causal ML for health policy; Keogh (LSHTM) is a leading survival methodologist. A code release is likely — the question is timeline. When it arrives, surv-iTMLE becomes the correct tool for any book with left truncation, and the case for a full `insurance-uplift` integration is straightforward.

Until then: fix the stub, use IPCW, and document your data structure assumptions.

---

## References

Pryce, M., Diaz-Ordaz, K., Keogh, R. H., and Vansteelandt, S. (2026). 'Targeted learning of heterogeneous treatment effect curves for right censored or left truncated time-to-event data.' arXiv:2603.26502. Submitted 27 March 2026.

Cui, R., Kosorok, M., Sverdrup, E., Wager, S., and Zhu, R. (2023). 'Estimating heterogeneous treatment effects with right-censored data via causal survival forests.' *Journal of the Royal Statistical Society Series B* 85(2):179–211. arXiv:2001.09887.

---

## Related posts

- [Your Retention Model Is Wrong About When Customers Lapse](/techniques/causal-inference/2026/04/01/surv-itmle-survival-causal-heterogeneous-treatment-effects-insurance-pricing/) — full methodology explainer for surv-iTMLE
- [insurance-survival: Survival Modelling for Insurance Pricing](/libraries/pricing/insurance-survival/) — `CauseSpecificMortality`, `AJRecalibrator`, and left-truncation support
- [insurance-uplift: Individual-Level Treatment Effect Estimation](/libraries/causal-inference/insurance-uplift/) — library overview and current scope
