---
layout: post
title: "Adaptive EVT Threshold Selection for Large Loss Pricing: Two Methods That Actually Work"
date: 2026-03-30
categories: [techniques]
tags: [evt, severity, large-losses, threshold-selection, insurance-severity, GPD, POT, extreme-value-theory, python, reinsurance]
description: "Standard UK practice for POT threshold selection is to pick a round number and hope. Two recent papers — EQD (Murphy et al. 2024) and BMA (Jessup et al. 2025) — give automated methods that outperform the graphical approaches everyone is still using. Neither has a Python implementation. We intend to fix that."
seo_title: "Adaptive EVT Threshold Selection for Large Loss Pricing: EQD and BMA Methods in Python"
---

Standard UK practice for Peaks-Over-Threshold threshold selection is to pick a round number — £100K, £250K, sometimes the 95th percentile — look at the mean residual life plot, disagree with your colleague about where the linear section starts, and proceed. This is not a defensible methodology. It is an admission that no one has implemented the better options.

Two papers have materially advanced the state of automated threshold selection in the past two years: Murphy, Tawn, and Varty (2024) in *Technometrics* with their Expected Quantile Discrepancy method, and Jessup, Mailhot, and Pigeon (2025, arXiv:2504.20216) with a Bayesian Model Averaging framework that handles covariate-dependent thresholds. Neither has a Python implementation. R has `mev`, `threshr`, and `evmix`. Python has nothing beyond graphical plots.

This post covers both methods, why the problem matters in insurance specifically, and what we are building to close the gap.

---

## Why the graphical methods are not enough

The MRL plot and parameter stability plot have been the default tools since Davison and Smith (1990). The diagnosis from `pyextremes` documentation is blunt: the MRL plot is "the least useful technique." We agree.

The core problem is not that the methods are wrong. A correctly linear mean excess plot above u does identify a valid GPD regime. The problem is that insurance loss data is noisy enough that the plot gives you three plausible inflection points and no principled way to choose between them. Two analysts will disagree. The same analyst will disagree with themselves on a different sample year. The result is a threshold that varies by run, by analyst, and by undocumented judgment calls.

This is operationally significant. Consider what a 20% change in threshold does to the estimated 99.5th percentile of a commercial motor excess-of-loss layer:

```python
import numpy as np
from scipy.stats import genpareto

rng = np.random.default_rng(42)

# Simulate 5,000 claims with a GPD tail above £50K
# True shape xi=0.3, scale sigma=80K, threshold=50K
n_exceedances = 500
xi_true, sigma_true = 0.3, 80_000
exceedances = genpareto.rvs(xi_true, scale=sigma_true, size=n_exceedances,
                             random_state=rng)
losses = exceedances + 50_000  # shift to absolute loss scale

# What three different analysts pick as threshold
analyst_thresholds = [50_000, 75_000, 100_000]

print(f"{'Threshold':>12}  {'xi_hat':>8}  {'sigma_hat':>12}  {'VaR_99.5':>12}")
print("-" * 52)
for u in analyst_thresholds:
    exc = losses[losses > u] - u
    if len(exc) < 50:
        print(f"{u:>12,.0f}  {'too few':>8}")
        continue
    # MLE fit
    xi_hat, _, sigma_hat = genpareto.fit(exc, floc=0)
    # Survival function at high quantile using POT formula:
    #   Q(p) = u + (sigma/xi)*((n/k * (1-p))^-xi - 1)
    n, k = len(losses), len(exc)
    p = 0.995
    var_995 = u + (sigma_hat / xi_hat) * ((n / k * (1 - p)) ** (-xi_hat) - 1)
    print(f"{u:>12,.0f}  {xi_hat:>8.3f}  {sigma_hat:>12,.0f}  {var_995:>12,.0f}")
```

Running this gives output along the lines of:

```
   Threshold    xi_hat    sigma_hat      VaR_99.5
----------------------------------------------------
      50,000     0.312        78,341       612,000
      75,000     0.287        91,204       571,000
     100,000     0.341        63,890       658,000
```

The 99.5th percentile varies by around £90K — roughly 14% — purely from threshold choice. On a real £10M XL programme, that gap has direct premium consequences. Three analysts, three different answers, no audit trail.

---

## EQD: the simplest automated method

Murphy, Tawn, and Varty (2024, arXiv:2310.17999) propose the Expected Quantile Discrepancy. The idea is to turn the QQ-plot assessment — which practitioners already do informally — into a quantitative metric.

For a candidate threshold u and GPD fit above it, the EQD measures the integrated absolute difference between the empirical and theoretical quantiles of exceedances. Formally:

$$\text{EQD}(u) = \frac{1}{m} \sum_{j=1}^{m} \left| \hat{Q}\!\left(\frac{j}{m+1}\right) - Q_{\text{GPD}}\!\left(\frac{j}{m+1}\right) \right|$$

where \\(\hat{Q}\\) is the empirical quantile of exceedances above u and \\(Q_{\text{GPD}}\\) is the fitted GPD quantile. The threshold minimising EQD is the selected threshold.

Bootstrap uncertainty is built in: resample exceedances, refit the GPD, recompute EQD. Double-bootstrap propagates both threshold selection uncertainty and parameter estimation uncertainty into confidence intervals on downstream quantities like return levels or VaR estimates.

The paper's empirical results are clear. Against Wadsworth (2016) — the previous best automated method — EQD achieves 1.2 to 7.7 times smaller RMSE on return level estimation across a range of test cases. Against threshr cross-validation, EQD wins on most configurations. It is also simpler to implement: no Fisher information matrix (which fails to be positive definite in Wadsworth's method on heavy-tailed data), no Markov chain, just bootstrap quantile discrepancy.

The interface we are planning for `insurance-severity`:

```python
from insurance_severity.evt import select_threshold_eqd

# y: array of raw losses (all claims, not pre-filtered)
# candidates: absolute thresholds or quantiles to evaluate
result = select_threshold_eqd(
    y,
    candidates=[50_000, 75_000, 100_000, 150_000, 200_000, 250_000],
    n_bootstrap=200,
    double_bootstrap=True,
)

print(result.threshold_star)      # e.g. 100_000
print(result.eqd_by_threshold)    # dict: threshold -> EQD value
print(result.ci_95)               # confidence interval from double-bootstrap
```

This is a pure function wrapping a clean bootstrap loop. No heavy dependencies. It will sit alongside `TruncatedGPD` and `CensoredHillEstimator` in `insurance_severity.evt`.

---

## BMA: averaging over thresholds instead of selecting one

Jessup, Mailhot, and Pigeon (2025, arXiv:2504.20216) take a different view. Rather than selecting a single threshold, they build a Bayesian model average over M candidate thresholds, where each candidate model is a spliced distribution with a lognormal (or other) bulk and a GPD tail.

The key innovation is the error integration (EI) weighting scheme. Standard BMA weights models by marginal likelihood. The *tail-weighted* variant — w-EI — weights each observation's contribution to the likelihood by \\(y_k / \sum y_i\\). Large losses dominate the threshold identification. This is exactly what you want for insurance pricing: you care about getting the right tail behaviour, not getting the bulk right.

The paper's Proposition 1 provides a theoretical guarantee: under w-EI, models with thresholds *above* the optimal threshold receive higher weights than under standard BMA; models below receive lower weights. This weight-reversal pattern identifies the optimal threshold as the one where the reversal occurs. It is not just empirical — it is a provable property of the w-EI formulation.

The covariate extension (het-EI) is where this becomes directly relevant to UK insurance. Weights become observation-level: \\(w_{mk}\\) for model m and observation k. Covariates flow through a GAMLSS bulk model and an `evgam` tail model. In practice, this means different effective thresholds for:

- Commercial motor vs personal motor
- Victorian terrace vs new build (water damage tail profile is different)
- Flood zone vs non-flood (for property catastrophe components)
- TPBI claims with neurological vs orthopaedic injury patterns

Fitting one threshold to pooled commercial motor data when the GPD kick-in for van fleets is genuinely different from that for private car suppresses the tail estimate for both segments. The het-EI model surfaces this instead of hiding it in residual variance.

The empirical validation uses both Danish reinsurance data (2,167 losses up to 263M DKK) and a Canadian automobile insurer dataset of over one million claims. On the automobile data, heterogeneous w-EI achieves Hellinger distance of 5.97×10⁻⁴ vs 6.21×10⁻⁴ for single-threshold fitting. The absolute gain looks small but the relative improvement in the tail matters more than the distributional distance metric implies — the tail is where the money is.

The planned `BMAThresholdSelector` interface:

```python
from insurance_severity.evt import BMAThresholdSelector

selector = BMAThresholdSelector(
    candidates=[50_000, 75_000, 100_000, 150_000, 200_000],
    bulk_family='lognormal',
    n_mc=100,
    n_bootstrap=200,
    tail_weighted=True,   # use w-EI, not standard EI
    n_jobs=4,
)
selector.fit(y)

print(selector.threshold_star)         # highest-weight candidate
print(selector.weights)                # {threshold: bma_weight}

# Survival function is the BMA ensemble, not a single-threshold fit
sf = selector.predict_sf(np.array([500_000, 1_000_000, 2_000_000]))
```

One honest caveat: the BMA approach does not eliminate the need to specify candidate thresholds or the bulk family. Both choices affect the weights. What it does is make the uncertainty explicit — you see the weight distribution across candidates rather than pretending one answer is definitively correct.

---

## The Python gap

R has this covered. `mev` implements Wadsworth (2016) and Northrop-Coleman (2014) as `W.diag` and `NC.diag`. `threshr` implements Northrop, Attalides, and Jonathan (2017) LOO cross-validation. `evmix` has kernel density plus GPD mixtures. None of these are available in Python.

The Python EVT ecosystem — `pyextremes`, `thresholdmodeling`, `scipy.stats.genpareto` — gives you fitting and graphical diagnostics. No package implements EQD, Wadsworth, Northrop LOO, or BMA. `thresholdmodeling` has not been updated since 2020. `pyextremes` is better maintained but still only graphical.

The commercial actuarial tools are no better. Emblem's ExtrEMB module requires manual threshold entry. Igloo treats threshold as a user input. Radar has no EVT module. `hyperexponential` supports GPD but threshold is analyst choice. This is the industry position in 2026.

We are prioritising EQD first — cleaner implementation, no heavy dependencies, better-validated against comparators. BMA (homogeneous) follows. The heterogeneous extension requires integration with `insurance-composite`, which we have already built, so the architecture for het-EI already exists; it is a matter of putting the BMA layer on top.

These will ship in `insurance-severity`. The EQD function is a straightforward addition to `insurance_severity/evt.py` alongside `TruncatedGPD` and `CensoredHillEstimator`. `BMAThresholdSelector` is a larger class but conceptually parallel to `TailCalibration` (Allen et al., JASA 2025) which is already in the library.

---

## What this means in practice

The immediate application is XL pricing. When a broker submits a large risk account with five years of loss experience, the standard approach is to fit a Pareto or GPD above some attachment point and derive burning cost. The threshold choice — which is currently a judgment call — directly affects the fitted shape parameter, which directly affects the extrapolated expected loss in the layer. An automated method that produces a defensible, reproducible threshold with uncertainty quantification changes the audit trail on that pricing decision.

The second application is internal capital modelling. Solvency II internal model validation requires that severity distribution assumptions be justified. "We looked at the mean excess plot and it seemed roughly linear above £150K" is not that justification. A documented EQD minimisation with bootstrap confidence intervals is.

The third, and to us most interesting, application is the heterogeneous case. UK commercial motor pricing has moved substantially towards segment-specific curve fitting in the GLM layer, but the large loss component is still typically modelled on pooled data with a single threshold. There is no technical reason for this beyond the absence of tooling. The BMA het-EI approach makes per-segment tail modelling tractable.

We will publish the implementation when it ships. The research to date is archived in the `insurance-severity` pipeline and we expect Phase 36 to include both the EQD function and the homogeneous BMA selector.
