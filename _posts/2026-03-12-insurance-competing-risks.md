---
layout: post
title: "Treating Competing Risks as Censored Is Biasing Your Retention and Home Insurance Pricing"
date: 2026-03-12
categories: [libraries, pricing, survival, retention, home-insurance]
tags: [competing-risks, Fine-Gray, subdistribution-hazard, CIF, Aalen-Johansen, Gray-test, IPCW, retention, lapse, home-insurance, motor, insurance-survival, python]
description: "Fine-Gray subdistribution hazard for UK insurance competing risks. Separates lapse, MTC, and NTU correctly - insurance-survival Python, not naive censoring."
---

There is a mistake buried in most UK insurers' retention models. It is not exotic: it is a standard decision made in lifelines or scikit-survival when you set up a Cox proportional hazards model and treat every exit event the same way. When a policyholder leaves via mid-term cancellation rather than voluntary lapse, you mark them as censored for the lapse analysis and move on. The same pattern appears in home insurance peril modelling - fire claim analysis treats the policy that had an escape of water as censored. Motor claim models treat the windscreen claimant as censored for the own-damage analysis.

This is the competing risks problem, and treating it as censoring biases your estimated event probabilities upward. The scale of the bias depends on how common the competing events are; in retention, where MTC, NTU, and lapse may each account for a quarter or more of exits, the effect is not small.

The fix has existed since 1999 - Fine & Gray published their subdistribution hazard model in JASA that year - but until now there has been no pure-Python, pip-installable implementation. [`insurance-survival`](https://github.com/burning-cost/insurance-survival) is that implementation.

```bash
uv add insurance-survival
```

---

## Why the standard approach is wrong

To see the problem concretely, suppose you are modelling voluntary lapse in a home book. Policies exit in four ways: voluntary lapse, mid-term cancellation (MTC), non-renewal at anniversary (NTU), and renewal. You want to know the probability that a given policy voluntarily lapses within 12 months.

The standard survival analysis approach is to fit a Cox model or Kaplan-Meier on lapse, treating MTC, NTU, and renewal as censored. This seems reasonable - those events are not lapses. But treating them as censored makes a specific assumption: that the censoring is non-informative. That is, that knowing a policy had an MTC tells you nothing about what its lapse probability would have been. This is almost certainly false. MTC-prone policyholders and lapse-prone policyholders differ systematically. When you censor the MTCs, you are progressively removing from the risk set a population that is not representative of the remaining policies.

The result is that `1 - KM(t)` overestimates the true probability of voluntary lapse. The Kaplan-Meier estimator inflates the cause-specific CIF because it treats competing events as uninformative departures from a homogeneous risk set.

The correct non-parametric estimator is the **Aalen-Johansen** estimator:

$$F_k(t) = \sum_{t_i \le t} \frac{d_{k,i}}{n_i} \cdot S(t_{i-})$$

where $d_{k,i}$ is the number of cause-$k$ events at time $t_i$, $n_i$ is the total risk set, and $S(t)$ is the Kaplan-Meier overall survival treating all event types as events. Unlike `1 - KM`, this properly partitions the total event probability across causes. It is available in the library:

```python
from insurance_survival.competing_risks import AalenJohansenFitter

# E: 0=censored (active), 1=voluntary lapse, 2=MTC, 3=NTU
aj_lapse = AalenJohansenFitter()
aj_lapse.fit(df["policy_months"], df["exit_type"], event_of_interest=1, label="Voluntary lapse")

aj_mtc = AalenJohansenFitter()
aj_mtc.fit(df["policy_months"], df["exit_type"], event_of_interest=2, label="MTC")

ax = aj_lapse.plot()
aj_mtc.plot(ax=ax)
```

The stacked CIF view is often more informative than individual cause plots:

```python
from insurance_survival.competing_risks.cif import plot_stacked_cif

plot_stacked_cif(
    df["policy_months"],
    df["exit_type"],
    cause_labels={1: "Lapse", 2: "MTC", 3: "NTU"},
)
```

This gives you a single chart where the total height at any point represents the overall exit probability, partitioned by cause. In practice, it often reveals that NTU is eating into the lapse probability more than the team expected - not because lapse rates are lower than thought, but because NTU was being misattributed.

---

## From descriptive to regression: Fine-Gray

The Aalen-Johansen estimator is descriptive. For pricing, you need to understand how the CIF varies with covariates - vehicle type, property age, NCD history, whatever your rating factors are. That requires regression.

Fine & Gray (1999) defined the **subdistribution hazard** for cause $k$:

$$\lambda_k(t) = -\frac{d}{dt} \log(1 - F_k(t))$$

Unlike the cause-specific hazard from a standard Cox model, the subdistribution hazard has a one-to-one relationship with the CIF. If $\lambda_k(t)$ follows proportional hazards - that is, if $\lambda_k(t|x) = \lambda_{k0}(t) \cdot \exp(\beta_k' x)$ - then the CIF is directly:

$$F_k(t|x) = 1 - \exp\left(-\Lambda_{k0}(t) \cdot \exp(\beta_k' x)\right)$$

This is the quantity pricing actuaries actually want. Not a hazard rate, but a probability: the chance that cause $k$ occurs before time $t$, given the covariates. You can directly use $F_k(t|x)$ as a component of your technical premium.

The estimation wrinkle is that the subdistribution risk set is non-standard. Subjects who have experienced a competing event are kept in the risk set (whereas in standard Cox they would be censored out). This would break the partial likelihood unless they are downweighted using inverse probability of censoring weighting (IPCW). The library implements this correctly:

```python
from insurance_survival.competing_risks import FineGrayFitter

fg = FineGrayFitter(penaliser=0.1)
fg.fit(
    df,
    duration_col="policy_months",
    event_col="exit_type",
    event_of_interest=1,   # voluntary lapse
)
fg.print_summary()
```

The output mirrors the lifelines `CoxPHFitter` format - covariates, coefficients, subdistribution hazard ratios (SHRs), standard errors, z-scores, p-values, confidence intervals.

### What the SHR means (and doesn't mean)

An SHR of 1.8 on a rating factor means the CIF for that cause is elevated. It does not mean the probability of that cause is exactly 1.8 times higher - that would only be true at baseline. Because the CIF is bounded by 1, the proportional relationship between SHR and absolute risk depends on the baseline level. For rare causes (small baseline CIF), SHR and relative risk are approximately equal. For common causes, they diverge.

Practically: always report predicted CIF values directly, not just SHRs. The `predict_cumulative_incidence()` method returns a DataFrame of CIF estimates for each subject at each evaluation time:

```python
times_to_evaluate = [3, 6, 9, 12]  # months

cif_preds = fg.predict_cumulative_incidence(df_new, times=times_to_evaluate)
# Shape: (n_subjects, 4) — lapse probability at 3, 6, 9, 12 months
```

For a covariate effect plot - the standard actuarial partial effect chart - use:

```python
fg.plot_partial_effects_on_outcome(
    covariate="ncd_years",
    values=[0, 1, 2, 3, 4, 5],
)
```

---

## The home insurance use case

Competing risks are, if anything, a bigger problem in home insurance peril modelling than in retention. A property in a policy year can generate a claim from: fire or explosion, escape of water (EoW), flood, storm or weather damage, subsidence, theft, or accidental damage. The first claim that occurs changes everything - the excess is partially exhausted, the insurer may schedule a survey, and the policyholder's perception of the property changes.

The standard approach is to model each peril independently with a Poisson GLM, treating the claim count per policy year as the response. This ignores that a fire claim removes the policy from the EoW risk set for the rest of the year, that subsidence claims correlate with property age in ways that create competing risk structure, and that high-EoW properties in the UK (older houses with original plumbing, according to BIBA property risk data) are systematically different in their fire risk profile.

The Fine-Gray formulation is the right model: fit one `FineGrayFitter` per peril, treating all other perils as competing events, with policy lifetime as the duration variable. The output is:

$$\text{Technical premium} = \sum_{\text{peril } k} \text{peril loading}_k \times F_k(t=12|x)$$

where $F_k(t=12|x)$ is the Fine-Gray predicted probability of peril $k$ being the first claim in a 12-month policy, given covariates $x$. This is actuarially correct in a way that the Poisson GLM approach is not, and the difference is non-trivial for high-frequency perils: Milhaud & Dutang (2018, *European Actuarial Journal*) showed that Fine-Gray outperforms cause-specific Cox for predicting event trajectory in life insurance lapse, and the same logic applies to EoW in a UK home book where EoW frequency is 4-6 times higher than fire.

---

## Gray's test: before you model, test

Before fitting separate Fine-Gray models per subgroup, it is worth asking whether the CIFs actually differ across groups. The log-rank test is wrong here - it tests equality of cause-specific hazards, not CIFs. Gray (1988) derived the correct test, and the library implements it:

```python
from insurance_survival.competing_risks import gray_test

# Does voluntary lapse CIF differ between direct and broker policyholders?
result = gray_test(
    df["policy_months"],
    df["exit_type"],
    groups=df["channel"],
    event_of_interest=1,
)
print(result)
# Gray's 2-Sample CIF Test (cause 1)
#   chi^2 = 14.23  df = 1  p = 0.0002
```

The Gray test weights the log-rank-type process by the CIF itself, giving more power for detecting differences in CIF shape rather than just differences in hazard rate. In practice, it catches differences that the log-rank test misses because they manifest as early/late divergence in cumulative probability rather than a persistent hazard differential.

---

## Model evaluation

A competing-risks model needs competing-risks evaluation metrics. The standard survival C-index and Brier score are both wrong when competing events are present - the IPCW censoring weights need to be adapted.

The library provides both, correctly implemented:

```python
from insurance_survival.competing_risks import competing_risks_brier_score, competing_risks_c_index
from insurance_survival.competing_risks.metrics import integrated_brier_score
import numpy as np

times = np.arange(1, 13)  # months 1-12

cif_test = fg.predict_cumulative_incidence(df_test, times=times)

ibs = integrated_brier_score(
    cif_test,
    T_test=df_test["policy_months"].values,
    E_test=df_test["exit_type"].values,
    T_train=df_train["policy_months"].values,
    E_train=df_train["exit_type"].values,
    times=times,
    event_of_interest=1,
)

c_idx = competing_risks_c_index(
    cif_test,
    T_test=df_test["policy_months"].values,
    E_test=df_test["exit_type"].values,
    T_train=df_train["policy_months"].values,
    E_train=df_train["exit_type"].values,
    eval_time=12.0,
    event_of_interest=1,
)

print(f"IBS: {ibs:.4f}  C-index at 12m: {c_idx:.4f}")
```

The Brier score benchmark: a model that always predicts $F_k(t) = 0.5$ scores 0.25. Anything below that is informative. `integrated_brier_score()` uses the trapezoid rule to average across the time horizon - this is the primary tuning metric, not the raw Brier scores at individual time points.

Calibration is assessed via `calibration_curve()`, which groups subjects by quantile of predicted probability and compares mean predicted CIF to the Aalen-Johansen empirical CIF within each group:

```python
from insurance_survival.competing_risks.metrics import calibration_curve, plot_calibration

cal = calibration_curve(
    cif_test,
    T_test=df_test["policy_months"].values,
    E_test=df_test["exit_type"].values,
    eval_time=12.0,
    event_of_interest=1,
    n_quantiles=10,
)
plot_calibration(cal, title="Lapse CIF calibration at 12 months")
```

---

## The Python gap this fills

Lifelines has had `AalenJohansenFitter` since v0.27 and it is perfectly adequate for non-parametric CIF estimation. It does not have Fine-Gray regression - this has been an open request since at least GitHub issue #539 (2018). scikit-survival added `cumulative_incidence_competing_risks()` in v0.24 (February 2025), but also has no Fine-Gray model. The `cmprsk` Python package wraps R's `cmprsk` via `rpy2` - which means it requires an R runtime installed at path, which is not an option for most production Python environments. `pydts` has Fine-Gray in discrete time, which is the wrong model class for continuous policy lifetime data.

This is a genuine gap: no pure-Python, pip-installable library provides Fine-Gray regression with IPCW, proper Aalen-Johansen CIF with confidence bands, Gray's test, and competing-risks calibration metrics. `insurance-survival` is 2,323 lines across seven modules, with 150 tests verifying numerical accuracy against the Klein & Moeschberger (1997) bone marrow transplant benchmark dataset and against simulated data with known coefficients.

---

## Limitations worth knowing

**Computational cost.** The algorithm is O(n²) in sample size because the extended risk set must be reconstructed at each event time. For a typical UK home book renewal dataset of 50,000 policies, fitting takes 30-90 seconds on a standard laptop. For 500,000 policies, this becomes impractical. The R `fastcmprsk` package addresses this with forward-backward scan algorithms achieving O(n log n), but we have not yet implemented that optimisation. If your book is large, fit on a random subsample (stratified by cause) and use the Breslow estimator on the full data.

**SHR consistency.** Fitting separate Fine-Gray models per cause does not guarantee that the sum of predicted CIFs across causes is at most 1.0. In practice, on realistic covariate ranges, this is not a problem. At extreme covariate combinations, you may see CIF sums exceed 1. If this matters for your use case, add an L2 penaliser (`penaliser=0.5`) to regularise the coefficients.

**Time-varying covariates.** The current implementation handles baseline covariates only. If your retention model includes covariates that change over the policy year (cumulative claims, mid-term updates), you cannot use the current version for those. This is on the roadmap for v0.2.

**Causal interpretation.** A covariate with a large positive SHR for cause 1 will often appear to have a negative effect when you fit a Fine-Gray model for cause 2. This is a mathematical artefact of the subdistribution hazard, not a causal finding - a factor that accelerates lapse reduces the proportion of the risk set that is exposed to MTC, so the MTC Fine-Gray coefficient is attenuated or reversed. Do not use Fine-Gray SHRs to understand mechanisms. For etiology, fit cause-specific Cox models (treating competing events as censored). Use Fine-Gray for pricing output - absolute risk prediction - and cause-specific Cox for understanding what drives each event type.

---

## When to use it

Use `insurance-survival` when:

1. You are modelling retention and have at least two distinct exit types (lapse, MTC, NTU) that you want to price separately
2. You are building a home insurance peril model and want to account for the fact that EoW, fire, and subsidence compete within a policy year
3. You need a calibrated probability - not a score or rank - for customer lifetime value calculations
4. You want to test whether a rating factor has a differential effect on two competing causes before fitting separate models

Do not use it when:

- You have a single event type and administrative censoring (use lifelines `CoxPHFitter`)
- You need non-proportional hazards (the Fine-Gray model assumes proportionality of the subdistribution hazard)
- Your sample size is above 500,000 and you cannot subsample (performance will be limiting)
- The research question is causal (use cause-specific Cox)

The Milhaud & Dutang (2018) paper is the clearest actuarial evidence that this matters for pricing prediction. They fit both cause-specific Cox and Fine-Gray to a French life insurance whole-life portfolio, and found that Fine-Gray produced meaningfully better prediction of the empirical lapse trajectory at the portfolio level. The gap was largest in the middle of the policy lifetime - exactly where the CLV calculation is most sensitive.

---

**[insurance-survival on GitHub](https://github.com/burning-cost/insurance-survival)** - MIT-licensed, PyPI. 2,323 lines, 150 tests, 7 modules.


- **[Survival Models for Insurance Retention](/2026/03/11/survival-models-for-insurance-retention/)** - The baseline: Cox PH and Kaplan-Meier for retention modelling, before competing risks are added
- **[Separating Structural Non-Claimers from Risk: Mixture Cure Models](/2026/03/11/insurance-cure/)** - When some policies will never claim (structural zeros), not just right-censored; the cure fraction approach
- **[Vine Copulas for Multi-Peril Home: The Flood-Subsidence Correlation That Costs 9% in Mispriced Revenue](/2026/03/12/insurance-copula/)** - Copula models for the joint distribution of home perils; pairs well with competing risks for first-claim modelling

---

## Related articles

- [Separating Structural Non-Claimers from Risk: Mixture Cure Models for Insurance Pricing](/2026/03/11/insurance-cure/)
- [Your Lapse Model Ignores Cure: The Customers Who Were Never Going to Leave](/2027/11/15/cure-models-lapse-retention/)
- [Experience Rating: NCD and Bonus-Malus](/2026/02/27/experience-rating-ncd-bonus-malus/)
