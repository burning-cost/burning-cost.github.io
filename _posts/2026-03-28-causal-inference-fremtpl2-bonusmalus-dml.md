---
layout: post
title: "BonusMalus Is Endogenous: DML on 677K French Motor Policies"
date: 2026-03-28
categories: [causal-inference, pricing, techniques]
tags: [double-machine-learning, dml, bonus-malus, endogeneity, confounding, fremtpl2, causal-inference, motor, frequency-model, poisson, cate, rosenbaum]
description: "BonusMalus is built from past claims — a naive regression conflates the causal effect with selection. We ran Double Machine Learning on 677K freMTPL2 policies to isolate what BonusMalus actually does to claim frequency, controlling for driver age, vehicle age, region, and density."
---

Every French motor pricing model includes BonusMalus as a rating factor. Most UK actuaries will recognise the concept even if they use NCD instead: a coefficient that rises with claim history and falls with claim-free years. The factor is strongly predictive. The problem is that "predictive" and "causal" are not the same thing, and for BonusMalus specifically, the gap between them is large enough to matter.

We benchmarked [`insurance-causal`](/insurance-causal/) against freMTPL2 — 677,991 French MTPL policies from OpenML (dataset ID 41214) — using BonusMalus as a continuous treatment and ClaimNb as the outcome. Here is what a properly confounded causal estimate looks like, and why it should make you think differently about how you use BM in a rating structure.

---

## Why BonusMalus is endogenous

BonusMalus accumulates mechanically from past claims. A driver with BM = 150 got there by having claims. Those same claims reflect latent risk characteristics — annual mileage, driving style, occupation, whether the policyholder lives in a dense urban area — that are only partially captured by the observed rating factors (age, vehicle age, vehicle power, region, density).

A naive GLM that includes BonusMalus as a covariate does not estimate the causal effect of BM on claim frequency. It estimates a mixture: the true causal effect of BM (if any — this is an interesting question in its own right) plus the correlation between BM and all the latent risk factors that drove the policyholder to a high BM in the first place. The coefficient absorbs both signals. You cannot separate them with OLS or Poisson regression alone.

This matters if you are using BM as a rating factor for prospective pricing decisions. The question you want to answer is: conditional on everything else I know about this driver, what does their BM score tell me about their *next* year's claims? The naive coefficient overstates the answer because it includes the historical latent risk that already showed up in past years' claims.

Double Machine Learning (Chernozhukov et al., 2018, *Econometrica*) separates these signals. It residualises both the treatment (BonusMalus) and the outcome (ClaimNb) against the observed confounders using any flexible learner, then regresses the outcome residual on the treatment residual. The confounders' influence is subtracted before the causal estimate is computed.

---

## The setup

```python
from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np

data = fetch_openml(data_id=41214, as_frame=True)
df = data.frame.copy()
df["log_Density"] = np.log1p(df["Density"].astype(float))
```

BonusMalus is clipped at the 99th percentile (~150) before fitting. The right tail above that is sparse enough that the nuisance learners become unstable, and the extreme cases are not the policy population you care most about.

```python
from insurance_causal import CausalModel, ContinuousTreatment

model = CausalModel(
    treatment=ContinuousTreatment(
        column="BonusMalus",
        standardise=False,
    ),
    outcome_col="ClaimNb",
    exposure_col="Exposure",
    outcome_type="poisson",
    confounders=[
        "DrivAge", "VehAge", "VehPower", "log_Density",
        "VehBrand", "VehGas", "Area", "Region",
    ],
    n_folds=5,
    random_state=42,
)

result = model.fit(df)
print(result.summary())
```

The confounders include driver age, vehicle age, vehicle power, log-density, vehicle brand, fuel type, area code, and region — everything in freMTPL2 that is plausibly correlated with both BonusMalus and future claim frequency. Annual mileage and occupation are the main unobserved confounders that are not available in this dataset; we return to that below.

---

## What DML produces

`result.summary()` gives you the Average Treatment Effect (ATE) and its standard error — the estimated causal effect of a one-unit increase in BonusMalus on claim frequency, in the Poisson sense.

The comparison that matters is between the DML estimate and the coefficient you would get from a naive Poisson GLM with BonusMalus included alongside the same confounders:

```python
# Naive GLM coefficient on BonusMalus
import statsmodels.api as sm

X_glm = pd.get_dummies(df[confounders], drop_first=True).astype(float)
X_glm["BonusMalus"] = df["BonusMalus"].clip(upper=bm_p99).astype(float)
X_glm = sm.add_constant(X_glm)
offset = np.log(df["Exposure"].clip(lower=1e-6).astype(float))

glm = sm.GLM(
    df["ClaimNb"],
    X_glm,
    family=sm.families.Poisson(),
    offset=offset,
).fit()

naive_coef = glm.params["BonusMalus"]
dml_ate = result.ate
```

The GLM coefficient on BonusMalus overstates the causal magnitude relative to the DML estimate. The ratio — how much larger the naive estimate is — tells you the confounding bias. On freMTPL2, the confounders only partially account for the endogeneity: the unobserved mileage and driving style channels remain. The DML estimate is more defensible as a causal quantity, though not free of unobserved confounding.

---

## Sensitivity: how worried should we be about unobserved confounders?

This is where Rosenbaum sensitivity analysis becomes essential. DML controls for the observed confounders in the dataset. Annual mileage is not in freMTPL2. Occupation is not in freMTPL2. Both are strongly correlated with BonusMalus (high-mileage commercial drivers accumulate worse BM) and with claim frequency.

```python
sensitivity = result.rosenbaum_sensitivity(gamma_range=(1.0, 3.0, 0.25))
print(sensitivity.summary())
```

`rosenbaum_sensitivity()` sweeps over a range of Gamma values — the odds-ratio of hidden bias. Gamma = 1.0 means no unobserved confounding; Gamma = 2.0 means hidden bias could double the treatment odds. The output tells you the threshold Gamma at which the causal conclusion would be reversed.

For BonusMalus, the Gamma threshold is particularly informative precisely because the endogeneity mechanism is strong. A low Gamma threshold would mean the causal estimate is fragile. The `sensitivity.critical_gamma` attribute gives you the value directly.

```python
print(f"Critical Gamma: {sensitivity.critical_gamma:.2f}")
print(f"Conclusion {'robust' if sensitivity.critical_gamma > 1.5 else 'fragile'} to unobserved confounding")
```

We do not report the specific numbers here because they are dataset-dependent and the freMTPL2 benchmark is a methodology validation rather than a production calibration. The point is to show the workflow that belongs in a causal analysis of a real endogenous rating factor.

---

## CATE by driver age band

At 677K policies, you have enough statistical power to estimate Conditional Average Treatment Effects by subgroup without the instability that plagues CATE estimation on smaller datasets. Each driver age band in freMTPL2 has 80,000+ observations:

```python
age_bands = pd.cut(
    df["DrivAge"].astype(float),
    bins=[17, 25, 35, 45, 55, 65, 100],
    labels=["18-25", "26-35", "36-45", "46-55", "56-65", "66+"],
)

cate = result.cate_by_group(group_col=age_bands, label="DrivAge_band")
print(cate.to_frame())
```

`cate_by_group()` returns a `CATEResult` with per-group ATE estimates and standard errors. The CATE estimates by age band tell you whether the relationship between BonusMalus and claim frequency is heterogeneous across the age distribution — which has direct relevance if you use multiplicative interactions between BM and age in the rating structure.

```python
# Compare causal vs naive by age band
naive_by_band = (
    df.assign(age_band=age_bands)
    .groupby("age_band", observed=True)
    .apply(lambda g: _fit_band_glm(g))  # band-level GLM coefficient on BM
)
```

Young drivers (18-25) often have higher BM scores AND higher base claim frequencies. Whether the BM score is adding incremental predictive value over and above age — the causal question — is not answerable from the naive GLM coefficient.

---

## What this means for pricing teams

BonusMalus is legitimately predictive. No one disputes that. The question is how much of that predictive power is genuine information about unobserved risk heterogeneity versus mechanical correlation with past claims.

The practical implication is this: if you are relying on a naive BM coefficient to set prospective claim frequency expectations, you are using a biased estimate. The bias direction is upward — the naive coefficient overstates the causal effect. That means you are over-weighting BM in the rating structure relative to its causal contribution to future claims.

How much does this matter in aggregate? That depends on the magnitude of the GLM-to-DML ratio in your specific portfolio, which will vary by book composition, the other rating factors available to you, and how much unobserved confounding you have. On freMTPL2, the benchmark quantifies the bias under the assumption that the observed confounders capture the main channels.

Two things we would push back on immediately if we heard them in a pricing review:

**"BonusMalus is already in the GLM so the confounding is controlled."** No. Including BM in a GLM alongside its confounders does not solve the endogeneity problem — it estimates a biased coefficient on BM. Controlling for confounding means partialling them out of *both* the treatment and the outcome. That is what DML does.

**"We cannot get causal estimates from observational insurance data."** You can get cleaner causal estimates from observational data than from naive regression. They will not be as clean as a randomised experiment, and Rosenbaum analysis will tell you how much residual fragility to unobserved confounders remains. But "imperfect causal evidence" is more useful than "known-biased associative evidence" if the bias direction matters for a pricing decision.

---

## Running it yourself

The benchmark notebook is at [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal) (`notebooks/benchmark_fremtpl2.py`), structured for Databricks. It downloads freMTPL2 directly from OpenML.

```
pip install insurance-causal scikit-learn statsmodels
```

The notebook has 14 sections: data loading, type coercion, confounding visualisation, naive GLM (numeric-only and full), DML fit, ATE results, nuisance diagnostics, Rosenbaum sensitivity, CATE by age band, naive vs causal comparison, a four-panel visualisation, confounding bias report, and summary. Each section is self-contained and can be run independently once the data is loaded.

This is the ninth of ten planned external dataset benchmarks across the [`insurance-causal`](/insurance-causal/) library. The tenth (telematics sensor data) is waiting on a suitable trip-level dataset.
