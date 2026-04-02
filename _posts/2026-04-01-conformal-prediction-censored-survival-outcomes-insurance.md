---
layout: post
title: "Conformal Prediction for Censored Survival Outcomes: ConformalisedSurvival in insurance-conformal v1.2.0"
date: 2026-04-01
categories: [techniques, pricing]
tags: [conformal-prediction, survival-analysis, censored-data, IPCW, doubly-robust, ConformalisedSurvival, right-censored, mortality, lapse, time-to-claim, Sesia-Svetnik, arXiv-2412-09729, insurance-conformal, lifelines, scikit-survival, python]
description: "Standard conformal prediction fails with right-censored survival data because you never observe the true event time for censored policies. ConformalisedSurvival in insurance-conformal v1.2.0 implements the Sesia & Svetnik (arXiv:2412.09729) doubly robust approach: impute censoring times, reweight with IPCW, and get asymptotically valid lower prediction bounds for mortality, lapse timing, and time-to-claim — even if your survival model or censoring model is misspecified."
math: true
author: burning-cost
---

Insurance is full of survival problems. Time to death in a protection book. Time to lapse in motor or home. Time to claim closure in liability. Time to a large development event in a long-tail commercial line. In every case, you have the same structural problem: the observation ends before the event, and you observe not the true event time $T$ but $\tilde{T} = \min(T, C)$, where $C$ is the censoring time — the point at which the policy expired, the study window closed, or the policyholder left your book.

This is right-censoring, and it is the standard setup of survival analysis. Standard conformal prediction has no answer to it. Split conformal and CQR both require fully observed outcomes at calibration. If you try to apply them directly to survival data — using $\tilde{T}$ as the outcome — your intervals will be biased towards shorter times, systematically undercovering the policyholders whose events you never observed.

`insurance-conformal` v1.2.0 ships `ConformalisedSurvival`, which implements the doubly robust conformal survival method of Sesia & Svetnik (arXiv:2412.09729). This post explains the problem, the algorithm, and how to use it.

---

## Why standard conformal breaks

Split conformal prediction gives you a scalar threshold $\hat{q}$ from calibration nonconformity scores $s_i = \hat{q}_\alpha(x_i) - y_i$, where $\hat{q}_\alpha$ is the survival model's predicted $\alpha$-quantile and $y_i$ is the observed event time. You then form a lower prediction bound $L(x) = \hat{q}_\alpha(x) - \hat{q}$.

The exchange argument underlying this is simple: if the calibration and test data are exchangeable (i.i.d. from the same distribution), the rank of the test nonconformity score among calibration scores is uniform over $\{1, \ldots, n+1\}$, so the $(1-\alpha)$ quantile of calibration scores gives a valid bound.

With right-censored data, this fails immediately. For a censored calibration point with $e_i = 0$, you observe $\tilde{T}_i = C_i$, not $T_i$. If you use $\tilde{T}_i$ in the nonconformity score, you are scoring against the censoring time, not the event time. The score $\hat{q}_\alpha(x_i) - \tilde{T}_i = \hat{q}_\alpha(x_i) - C_i$ overstates the model error for policies with long in-force times — those policyholders may still be alive or lapsed-free, but you counted them as early events. The calibration threshold $\hat{q}$ is therefore deflated, and the LPB undershoots the true event distribution.

The magnitude of this bias depends on the censoring fraction. UK protection books typically have death rates of roughly 0.15–0.3 per 1,000 per year for a 40–60 year old insured population. In a five-year calibration window, you might observe deaths in only 1–2% of policies. The remaining 98% are censored at policy expiry or observation close. Standard conformal applied to this data is effectively calibrating on noise.

---

## The Sesia & Svetnik approach

The paper (arXiv:2412.09729, December 2024) solves this with three ideas: imputation, filtering, and importance weighting.

**Censoring time imputation (Algorithm 1).** For censored calibration points ($e_i = 0$), the censoring time is directly observed: $\hat{C}_i = \tilde{T}_i$. For event-observed points ($e_i = 1$), we know $T_i < C_i$ but $C_i$ is unobserved. The method draws $C_i$ from the truncated conditional censoring distribution $f_{C|X}(c \mid x_i) / P[C > T_i \mid X = x_i]$ for $c > T_i$, using the fitted censoring model. In practice, `n_impute=10` Monte Carlo draws per point are averaged to reduce imputation variance. The output is an imputed censoring time $\hat{C}_i$ for every calibration point.

**Fixed-cutoff filtering (Algorithm 2).** A cutoff $c_0$ is chosen (defaulting to the median observed time). The calibration set is then restricted to $\{i : \hat{C}_i \geq c_0\}$ — points for which we are confident the censoring time exceeds the LPB level we want to certify. This filtering step discards points that cannot contribute reliable information about coverage at time $c_0$. It reduces the effective calibration size, but the remaining points are the ones that matter.

**IPCW reweighting.** Filtering on $\hat{C}_i \geq c_0$ introduces selection bias: policyholders with high-risk covariates may have short censoring times and be systematically excluded. Inverse probability of censoring weighting corrects this. Each filtered calibration point receives weight $1/P[C > c_0 \mid X = x_i]$, estimated from the censoring model. Points that were unlikely to remain in the data (short expected censoring times) get upweighted; long-in-force policyholders who were certain to survive to $c_0$ get weight near 1.

The resulting IPCW-weighted conformal quantile of the nonconformity scores produces a lower prediction bound satisfying:

$$P[T \geq \hat{L}(X)] \geq 1 - \alpha$$

asymptotically, provided the conditional independence $T \perp C \mid X$ holds.

**The doubly robust property.** The coverage guarantee holds as long as at least one of the two models — survival model or censoring model — is correctly specified. If your Random Survival Forest for mortality is well-calibrated but your Cox censoring model is slightly misspecified, coverage holds. If your censoring model is perfect but the survival model is wrong about the distributional shape, coverage still holds. Both need to fail simultaneously for the guarantee to break down, and even then degradation is gradual.

This is a meaningful property for insurance. You are never fully confident in either model. Knowing that misspecification in one does not invalidate the interval is exactly the kind of robustness you want when these bounds are being used in regulatory capital calculations or reserving sign-offs.

---

## Setting it up

```bash
uv add insurance-conformal[survival]
```

The `[survival]` extra installs `lifelines>=0.27` and `scikit-survival>=0.22`. Both are optional — if you only need the `KaplanMeierCensoringModel` (which assumes covariate-independent censoring), the base `uv add insurance-conformal` is sufficient.

---

## A worked example: lapse timing in a motor book

The setup below uses scikit-survival for both the survival model (a Random Survival Forest for lapse time) and the censoring model (a Cox model for the renewal/admin censoring process). The data has the standard survival structure: observed time $\tilde{t}_i = \min(T_i, C_i)$ and event indicator $e_i \in \{0, 1\}$.

```python
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from insurance_conformal import ConformalisedSurvival
from insurance_conformal.survival import SksurvCoxCensoringAdapter

# Assume: X_train, t_train, e_train  (training split)
#         X_cal, t_cal, e_cal        (calibration split)
#         X_test                     (test split, no labels needed for prediction)

# --- Step 1: Fit the survival model for lapse time ---
# sksurv uses structured arrays: (event_bool, time_float)
y_train = np.array(
    [(bool(e), t) for e, t in zip(e_train, t_train)],
    dtype=[("event", bool), ("time", float)],
)
rsf = RandomSurvivalForest(n_estimators=200, random_state=42, n_jobs=-1)
rsf.fit(X_train, y_train)

# ConformalisedSurvival needs predict_quantile(X, alpha) -> np.ndarray.
# sksurv RSF exposes predict_quantile(X, percentiles) returning StepFunctions;
# a thin shim converts the signature:
class RSFSurvivalAdapter:
    def __init__(self, model): self.model = model
    def predict_quantile(self, X, alpha):
        # sksurv predict_quantile takes percentile in [0,1]; returns (n, 1) array
        return self.model.predict_quantile(X, percentiles=alpha).ravel()

survival_model = RSFSurvivalAdapter(rsf)

# --- Step 2: Fit the censoring model ---
# Reverse event indicator: censoring is the 'event' for the censoring model.
y_cens = np.array(
    [(bool(1 - e), t) for e, t in zip(e_train, t_train)],
    dtype=[("event", bool), ("time", float)],
)
cox_cens = CoxPHSurvivalAnalysis(ties="efron")
cox_cens.fit(X_train, y_cens)

censoring_model = SksurvCoxCensoringAdapter(cox_cens)

# --- Step 3: Calibrate ---
cs = ConformalisedSurvival(
    survival_model=survival_model,
    censoring_model=censoring_model,
    alpha=0.10,       # 90% lower prediction bound
    n_impute=10,      # 10 MC draws per event-observed cal point
    random_state=0,
)
cs.calibrate(X_cal, t_cal, e_cal)

print(cs)
# ConformalisedSurvival(method='fixed_cutoff', alpha=0.1, n_impute=10,
#   calibrated on 847 filtered points (of 2000 cal, cutoff=2.3412))

# Calibration summary
print(cs.calibration_summary())
# ┌──────────────────────┬───────────┐
# │ statistic            │ value     │
# │ ---                  │ ---       │
# │ str                  │ f64       │
# ╞══════════════════════╪═══════════╡
# │ n_calibration        │ 2000.0    │
# │ n_filtered           │ 847.0     │
# │ cutoff_c0            │ 2.341     │
# │ alpha                │ 0.1       │
# │ calibration_quantile │ -0.183    │
# │ mean_ipcw_weight     │ 1.847     │
# │ max_ipcw_weight      │ 12.310    │
# │ effective_n          │ 398.2     │
# └──────────────────────┴───────────┘

# --- Step 4: Predict ---
bounds = cs.predict_lower_bound(X_test)
# shape: (n_test, 2) polars DataFrame
# columns: lower_bound (years), q_alpha_model (raw RSF quantile)

print(bounds.head())
# ┌─────────────┬──────────────┐
# │ lower_bound │ q_alpha_model │
# │ f64         │ f64           │
# ╞═════════════╪══════════════╡
# │ 1.84        │ 2.02          │
# │ 3.11        │ 3.29          │
# │ 0.97        │ 1.15          │
# │ 4.22        │ 4.40          │
# │ 2.56        │ 2.74          │
# └─────────────┴──────────────┘
```

The `lower_bound` column is the conformally corrected LPB — the years before which, at 90% confidence, the policyholder will not lapse. The `q_alpha_model` is the raw survival model quantile before conformal correction. The difference is the calibration adjustment (`calibration_quantile_` = −0.183 here, which adds 0.183 years to each bound — the model was slightly over-pessimistic).

---

## The KaplanMeier shortcut

If censoring is genuinely administrative — every policy in your cohort was subject to the same observation window close date with no covariate-driven lapse — you can use `KaplanMeierCensoringModel` instead of a Cox adapter:

```python
from insurance_conformal.survival import KaplanMeierCensoringModel

km_cens = KaplanMeierCensoringModel(random_state=0)
# KM fits on the calibration data itself; no separate training set needed
km_cens.fit(t_cal, e_cal)

cs_km = ConformalisedSurvival(
    survival_model=survival_model,
    censoring_model=km_cens,
    alpha=0.10,
)
cs_km.calibrate(X_cal, t_cal, e_cal)
```

This is faster and has no additional model fitting, but it assumes $C \perp X$. For protection books where all policies have the same term, this may be defensible. For motor, where policyholders switch providers at varying rates depending on NCD and claims history, a covariate-dependent censoring model is better.

---

## Mortality: what a 99.5% LPB means

For protection pricing under Solvency II, you sometimes want to stress-test the reserve by asking: at the 99.5th percentile, how early could a large proportion of policyholders die? This is not quite what the LPB provides (it bounds a single policyholder's time-to-event, not the portfolio distribution), but the individual-level bound is a component of the stress.

Set `alpha=0.005` for a 99.5% LPB:

```python
cs_solvency = ConformalisedSurvival(
    survival_model=survival_model,
    censoring_model=censoring_model,
    alpha=0.005,  # 99.5% LPB
    n_impute=20,  # more imputation draws for stability at extreme quantiles
    random_state=0,
)
cs_solvency.calibrate(X_cal, t_cal, e_cal)
```

The calibration summary will show a smaller `effective_n` because the IPCW-weighted quantile at 0.5% is estimated from fewer effective calibration points. The paper's experiments suggest $n_{\text{cal}} \geq 500$ filtered points for reasonable asymptotic coverage at $\alpha = 0.10$. At $\alpha = 0.005$, you need more. For a UK protection book with 20,000 policies and 0.2% annual mortality, a five-year calibration window gives roughly 200 deaths — well below the paper's recommended threshold. The bound will be noisy. The KB entry for DR-COSARC limitations (ID 5222) has worked numbers on this, and it is worth reading before using `ConformalisedSurvival` for regulatory capital work.

---

## Coverage diagnostics

The `.coverage_diagnostics()` method evaluates empirical coverage on the test set, but only on event-observed points ($e = 1$):

```python
diag = cs.coverage_diagnostics(X_test, t_test, e_test)
# ┌─────────┬──────────────┬───────────────────┬──────────────────┬──────────────┐
# │ n_total │ n_uncensored │ empirical_coverage │ target_coverage  │ coverage_gap │
# │ i32     │ i32          │ f64               │ f64              │ f64          │
# ╞═════════╪══════════════╪═══════════════════╪══════════════════╪══════════════╡
# │ 1500    │ 312          │ 0.917             │ 0.900            │ -0.017       │
# └─────────┴──────────────┴───────────────────┴──────────────────┴──────────────┘
```

A `coverage_gap` of −0.017 means the bound is slightly conservative (covering 91.7% when 90% was targeted). Mild overcoverage is expected — the IPCW-weighted quantile at a finite sample is typically slightly above the true population quantile. Undercoverage (positive gap) means the double robustness assumption may be violated, or the calibration set is too small.

Coverage on censored test points is unknowable: you do not observe $T_i$ for those policyholders. This is the fundamental limitation of any survival conformal method. Diagnostics on the event-observed subset are the best you can do with real data.

---

## Where to use this and where not to

**Good fits:**

- **Lapse timing in personal lines.** Censoring (policy expiry, non-renewal) is covariate-driven and you can fit a Cox censoring model on renewal behaviour. The event of interest (voluntary mid-term lapse) is well-defined.
- **Time to first claim in motor or home.** Claims arrive during the policy term; policies that expire without a claim are censored. You observe enough claims to calibrate the method.
- **Mortality in group risk or protection.** Deaths are the event; policy expiry and surrenders are censoring. With enough lives (20k+), the asymptotic regime is reachable.

**Caution required:**

- **Lapse-then-claim.** If a policyholder lapses because of deteriorating health that also predicts a future claim, the conditional independence $T \perp C \mid X$ is violated. The censoring (lapse) and the event (claim or death) share unobserved drivers. The method degrades but the coverage guarantee disappears entirely in this case.
- **Small books.** UK regional mutual insurers with 2,000 policyholders and 30 deaths over five years are nowhere near the asymptotic regime. The IPCW-weighted quantile at $n_{\text{filtered}} = 30$ is a noisy estimate. Run `cs.calibration_summary()` and check `effective_n` — if it is below 50, treat the bounds as directional only.
- **Informative censoring in claims development.** Late-reported injury claims in motor CTP may be systematically different from early-reported claims. Censoring (reporting lag) and event severity are not independent. Standard IPCW does not handle this without modification.

---

## The `calibration_summary()` diagnostic

Before trusting any output, run `cs.calibration_summary()` and check three things:

**`n_filtered / n_calibration`.** The fraction of calibration points retained after the $c_0$ filter. A ratio below 0.3 means most of your calibration data was discarded. The default cutoff (median observed time) is usually reasonable; if it is too aggressive, lower it with the `cutoff=` parameter.

**`max_ipcw_weight`.** Large maximum weights (above 20–30) indicate regions of covariate space where the censoring model assigns very low $P[C > c_0 \mid X]$. A single observation with weight 80 can dominate the weighted quantile and make the bound erratic. Check your censoring model calibration, and consider raising $c_0$ to reduce the weight dispersion.

**`effective_n` and `coverage_shrinkage`.** The effective sample size after IPCW weighting is $(\sum w_i)^2 / \sum w_i^2$. If `effective_n / n_filtered` (the `coverage_shrinkage` field) is below 0.3, the IPCW weights are highly unequal and you have less information than the raw $n_{\text{filtered}}$ count suggests.

---

## What the method does not do

`ConformalisedSurvival` produces lower prediction bounds only — $P[T \geq \hat{L}(X)] \geq 1 - \alpha$. It does not produce upper bounds on survival time. The paper's Algorithm 3 (adaptive cutoff, which can produce upper bounds) is not yet implemented; the current release covers the fixed-cutoff Algorithm 2.

The coverage guarantee is asymptotic, not finite-sample. Standard split conformal gives you exact finite-sample coverage $\geq 1 - \alpha$ by construction, for any $n$. DR-COSARC does not. For small calibration sets, empirical coverage may fall materially below the nominal $1 - \alpha$. The paper's experiments use $n_{\text{cal}} = 1{,}000$–$2{,}000$; below 500 filtered points, treat the bound with scepticism.

---

## Getting it

```bash
uv add "insurance-conformal[survival]>=1.2.0"
```

```python
from insurance_conformal import ConformalisedSurvival
from insurance_conformal.survival import (
    KaplanMeierCensoringModel,
    LifelinesCoxCensoringAdapter,
    SksurvCoxCensoringAdapter,
    SurvivalModelProtocol,
    CensoringModelProtocol,  # typing.Protocol for type-checked adapters
)
```

Source: `src/insurance_conformal/survival.py`. Tests: `tests/test_survival.py` (35+ tests). Commit: `c1dcdc4`.

Library: [github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

---

## The paper

Sesia, Matteo and Svetnik, Vladimir. "Doubly Robust Conformalized Survival Analysis with Right-Censored Data." arXiv:2412.09729 [stat.ML]. December 2024.

---

## Related posts

- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/techniques/pricing/2026/04/01/shape-adaptive-conformal-prediction/) — MOPI and conditional coverage for Tweedie severity models
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/techniques/pricing/uncertainty/2026/03/13/insurance-conformal-risk/) — conformal risk control and the `insurance-conformal` library overview
- [Conformal Prediction for Insurance Pricing: Intervals, Risk Control, and the Practical Toolkit](/techniques/pricing/2026/03/23/does-conformal-prediction-work-insurance-pricing/) — when conformal prediction works, when it does not, and what calibration data you actually need
