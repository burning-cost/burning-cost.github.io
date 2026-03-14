---
layout: post
title: "Your Postcode Premium Isn't Discriminatory. Probably. Can You Prove It?"
date: 2026-03-11
categories: [libraries, pricing, fairness, causal-inference]
tags: [mediation-analysis, CDE, NDE, NIE, causal-fairness, FCA, CP22-9, proxy-discrimination, IMD, postcode, Imai, VanderWeele, E-value, sensitivity-analysis, Poisson, Gamma, Tweedie, GLM, insurance-mediation, python]
description: "Postcode correlates with dozens of protected characteristics. If you charge more for E1 than SW1, how much of that differential runs through legitimate risk (flood risk, crime rates, IMD deprivation) and how much is a residual effect you cannot explain? insurance-mediation decomposes the answer using causal mediation analysis: CDE, NDE, NIE with GLM outcome models, Imai rho sensitivity, E-values, and FCA-ready audit reports."
---

Postcode is legal to use in UK personal lines pricing. It correlates with flood risk, crime rates, subsidence susceptibility, repair costs — all legitimate actuarial variables. It also correlates with ethnicity, religion, and socioeconomic background — protected characteristics under the Equality Act 2010. When you charge more for E1 than SW1, you are doing both things simultaneously, and you probably cannot tell which is which.

The Citizens Advice 2020 ethnicity penalty report documented systematic variation in motor premiums that could not be explained by individual risk factors. The variation ran through postcode. The question their report could not answer — because the methodology was not designed to — is how much of that variation reflects genuinely higher costs of claims in those postcodes and how much is a residual effect that survives controlling for every legitimate risk factor you have.

FCA CP22/9 is explicit about what you need to demonstrate: that a differential attributable to a proxy for a protected characteristic is proportionate to a legitimate aim. "Proportionate" has a specific legal meaning. It requires you to show that the protected characteristic route — the path from postcode through ethnicity to price — is not the operative mechanism. Saying "our GLM is fair because we didn't include ethnicity" is not a response to that requirement.

[`insurance-mediation`](https://github.com/burning-cost/insurance-mediation) is the tool for that demonstration. It decomposes the total causal effect of postcode on price into the portion that runs through legitimate mediators (IMD deprivation score, flood risk, crime statistics) and the portion that does not. The residual is the number the FCA cares about.

```bash
uv add insurance-mediation
```

---

## What causal mediation analysis actually does

A standard GLM regression answers: holding all covariates constant, what is the association between postcode group and claim frequency? That is not the same question as: what is the causal effect of postcode on price that operates through ethnicity rather than through deprivation?

The difference matters because postcode, IMD deprivation, and ethnicity are all correlated. A regression that controls for IMD while asking about postcode has already partly adjusted away the ethnic composition pathway — but without being explicit about what causal pathway is being blocked or preserved.

Causal mediation analysis is explicit. You specify a causal DAG:

```
Postcode → IMD Deprivation → Claim Frequency
Postcode →                 → Claim Frequency (direct)
```

Then you estimate three quantities:

**Total Effect (TE):** E1 vs SW1 — what is the overall difference in expected claim frequency, accounting for all pathways?

**Natural Indirect Effect (NIE):** how much of that difference operates *through* IMD deprivation? If postcode affects claims only because E1 has higher deprivation and higher-deprivation areas have more claims, the NIE captures this.

**Natural Direct Effect (NDE):** how much remains if you hold IMD deprivation fixed at the level it would take under the control condition? This is the effect that cannot be attributed to the deprivation pathway.

The decomposition: TE = NDE + NIE (subject to Monte Carlo integration error).

For FCA purposes, a large NIE relative to TE is the defensible case: most of the postcode differential operates through a legitimate, measurable risk factor. A large NDE is the uncomfortable case: most of the differential survives the deprivation adjustment, and you need a better explanation for it.

---

## CDE: the one that actually matters for compliance

The Natural Effects require a strong causal assumption called sequential ignorability: conditional on covariates, there are no unmeasured confounders of the mediator-outcome relationship. This is an untestable assumption. In an insurance context — where IMD, crime rates, flood risk, and ethnicity are all correlated with each other and with the unobservable characteristics of individual policyholders — this assumption is heroic.

The **Controlled Direct Effect (CDE)** requires a weaker assumption. It answers: if we intervened to set everyone's IMD deprivation to the same value m, what postcode price differential would remain? Unlike the NDE, it does not require that you can imagine a world where the mediator takes one value but the treatment takes another (the cross-world counterfactual). It requires only that you can imagine a uniform intervention on the mediator — which corresponds to the actual policy intervention of removing IMD from the rating factors.

This is why CDE is the default for FCA compliance work. If you ran a model with IMD set to the population median for every policy, and you still saw a postcode differential, that differential is your CDE estimate. `insurance-mediation` computes this directly from the fitted outcome model, with bootstrap confidence intervals.

---

## The basic analysis

The API is a single fit call:

```python
import pandas as pd
from insurance_mediation import MediationAnalysis

# df has columns: postcode_group, imd_decile, claim_count, exposure,
#                 vehicle_age, driver_age, cover_type

ma = MediationAnalysis(
    outcome_model="poisson",    # Poisson GLM with log link
    mediator_model="linear",    # OLS for continuous IMD decile
    exposure_col="exposure",    # log(exposure) offset in Poisson model
    n_mc_samples=1000,          # Monte Carlo samples for NDE/NIE
    n_bootstrap=200,            # bootstrap replicates for CIs
)

results = ma.fit(
    data=df,
    treatment="postcode_group",
    mediator="imd_decile",
    outcome="claim_count",
    covariates=["vehicle_age", "driver_age", "cover_type"],
    treatment_value="E1",
    control_value="SW1",
)

print(results.summary())
```

Output:

```
Mediation Analysis Summary
Treatment: postcode_group  (E1 vs SW1)
Mediator:  imd_decile
Outcome:   claim_count  (poisson)
N obs:     47,823

Total Effect:              +0.2341  [+0.1893, +0.2789]
Natural Direct Effect:     +0.0587  [+0.0211, +0.0963]
Natural Indirect Effect:   +0.1754  [+0.1312, +0.2196]
CDE (m=5.00):              +0.0612  [+0.0234, +0.0990]
CDE (m=3.00):              +0.0498  [+0.0143, +0.0853]
CDE (m=7.00):              +0.0731  [+0.0312, +0.1150]
```

These are log-rate ratios. To get relativities:

```python
te = results.total_effect()
print(f"Total effect:   {te.ratio:.3f}x  [{te.ratio_ci[0]:.3f}, {te.ratio_ci[1]:.3f}]")
# Total effect:   1.264x  [1.208, 1.322]

nde = results.nde()
print(f"Direct effect:  {nde.ratio:.3f}x  [{nde.ratio_ci[0]:.3f}, {nde.ratio_ci[1]:.3f}]")
# Direct effect:  1.060x  [1.021, 1.101]

nie = results.nie()
print(f"Mediated:       {nie.ratio:.3f}x  [{nie.ratio_ci[0]:.3f}, {nie.ratio_ci[1]:.3f}]")
# Mediated:       1.192x  [1.140, 1.246]
```

The E1/SW1 total relativity is 1.264. Of that, 1.192 is attributable to the IMD pathway — E1 has higher deprivation, higher deprivation drives more claims, so the postcode differential is largely explained. The residual NDE is 1.060 — a 6% effect that survives holding deprivation fixed. Whether that 6% is defensible depends on whether you have other legitimate explanatory variables (crime statistics, repair costs, weather) that you have not yet included as mediators or covariates.

The CDE at IMD decile 5 (population median) is 1.063. Consistent with the NDE estimate, which is reassuring: the two estimands are answering the same question via different routes and arriving at similar answers.

---

## Using your production GLM

If you already have a fitted pricing GLM, you do not need to re-fit:

```python
import statsmodels.api as sm
from insurance_mediation import MediationAnalysis

# Your production Poisson GLM
fitted_glm = sm.GLM(y, X, family=sm.families.Poisson()).fit()

ma = MediationAnalysis(
    outcome_model=fitted_glm,   # pass the fitted result directly
    mediator_model="linear",    # still fit a mediator model
    exposure_col="earned_years",
)

results = ma.fit(
    data=df,
    treatment="postcode_group",
    mediator="imd_decile",
    outcome="claim_count",
    covariates=["vehicle_age", "driver_age"],
    treatment_value="E1",
    control_value="SW1",
)
```

The library detects the GLM family from the statsmodels result object and uses the appropriate link function throughout. For Gamma and Tweedie models:

```python
ma = MediationAnalysis(
    outcome_model="gamma",       # for severity models
    mediator_model="linear",
    exposure_col=None,           # severity — no exposure offset
)

# Or Tweedie (compound Poisson-Gamma, burning cost):
ma = MediationAnalysis(
    outcome_model="tweedie",
    tweedie_var_power=1.5,       # typical for UK personal lines
    mediator_model="linear",
    exposure_col="exposure",
)
```

---

## Sensitivity analysis: what if there is unmeasured confounding?

The NIE estimate assumes sequential ignorability — no unmeasured mediator-outcome confounders. In practice, IMD decile and claim frequency share common causes that are not in your dataset: individual income, housing quality, household size, local council maintenance standards. The assumption is violated.

`insurance-mediation` implements Imai et al. (2010, *Psychological Methods* 15:309–334) sensitivity analysis via the `rho` parameter. Rho is the correlation between the residuals of the outcome model and the mediator model. Under sequential ignorability, rho = 0. Varying rho shows how much unmeasured confounding is required to reduce the NIE to zero.

```python
results = ma.fit(
    data=df,
    treatment="postcode_group",
    mediator="imd_decile",
    outcome="claim_count",
    covariates=["vehicle_age", "driver_age", "cover_type"],
    treatment_value="E1",
    control_value="SW1",
    compute_sensitivity=True,
    rho_range=(-0.5, 0.5),
)

sens = results.sensitivity()
print(sens)
# SensitivityResult(E-value=2.847, E-value(CI)=2.143, rho_at_zero=0.312)
```

The `rho_at_zero=0.312` means the NIE only vanishes if the residual correlation between the IMD model and the claim model is 0.312. Given that IMD decile already captures the main observable channel, a residual correlation of 0.31 is a substantial unmeasured confounder. The evidence for a genuine indirect effect is not fragile.

The **E-value** (VanderWeele & Ding 2017, *Annals of Internal Medicine* 167:268–274) gives a complementary answer on a more interpretable scale: 2.847 means any unmeasured confounder would need risk ratios of at least 2.85 with both the mediator and the outcome to explain away the NIE. A confounder of that strength, given the covariates already in the model, is implausible. The E-value for the confidence interval bound (2.143) is the relevant number for a committee presentation — "even being generous about uncertainty, the confounder would need to be twice as strong as anything we have controlled for".

These numbers belong in the FCA report. `rho_at_zero` and `e_value` appear automatically in the HTML output.

---

## The FCA report

```python
results.report(
    output="mediation_audit_E1_vs_SW1.html",
    title="E1 vs SW1 Mediation Analysis — Motor Frequency",
    protected_attribute="ethnicity",
)
```

The report generates:

- The assumed causal DAG (rendered as an inline SVG — no external dependencies)
- Total effect decomposition table with CDE at five mediator percentiles
- NDE and NIE with 95% confidence intervals
- Sensitivity analysis bounds across the rho range
- E-value and E-value(CI) in a highlighted box
- The causal identification assumptions for each estimand, listed explicitly
- A template Section 19 proportionality statement, filled in with your numbers, ready for review

The proportionality statement reads something like: "The E1/SW1 differential (1.264x) is attributable to the IMD deprivation pathway (NIE: 1.192x, 95% CI [1.140, 1.246]) and a residual direct effect (NDE: 1.060x, 95% CI [1.021, 1.101]). The direct effect is robust to mediator-outcome confounding with rho up to 0.312 (E-value: 2.847). This analysis provides evidence that the pricing differential is proportionate to legitimate actuarial variables. The residual direct effect warrants further investigation via additional mediators (crime statistics, weather risk, repair cost indices)."

That is a statement you can put in front of your Actuarial Function Holder and your compliance team. It is specific, it has numbers, and it is honest about what it does and does not prove.

---

## What the library does not do

It does not verify that your causal DAG is correct. The mediation decomposition is conditional on the causal structure you specify. If IMD is not actually a mediator between postcode and claim frequency — if, say, postcode affects claims through crime rates and IMD is a proxy for crime rates rather than a genuine intermediary — the decomposition is wrong. Specifying the DAG is your responsibility. The library fits whatever DAG you provide.

It is scoped to a single mediator per analysis run in v0.1.0. Multiple-mediator analysis (postcode → [IMD, crime rate, flood risk] → claims) requires running separate analyses and interpreting them jointly. The interventional indirect effects framework (which handles correlated mediators cleanly) is in the roadmap for v0.2.0.

It does not replace the need to find additional mediators. A residual NDE of 1.06 is small enough to be defensible with a good argument. A residual NDE of 1.25 requires either more mediators or a harder conversation about why the postcode differential survives all your adjustments.

---

## The academic lineage

This is applied epidemiology. Imai, Keele, and Yamamoto (2010) published the rho sensitivity framework in *Psychological Methods* as a general tool for behavioural research. Tyler VanderWeele's 2015 textbook *Explanation in Causal Inference* (Oxford University Press) consolidated the NDE/NIE framework and showed how to apply it with GLM outcome models. The E-value appeared in *Annals of Internal Medicine* in 2017.

DoWhy (Microsoft Research) has basic mediation functionality. Its mediation analysis is linear-only: it can decompose effects for a Gaussian outcome model, but cannot handle Poisson claim counts with a log-link offset, Gamma severity, or Tweedie burning cost. None of that machinery exists in DoWhy. The FCA reporting and the Imai sensitivity framework for GLMs are not there either.

`insurance-mediation` does the GLM case properly. It handles the log-link throughout: effects are on the log-mean scale, ratios are exponentiated correctly, and the Monte Carlo integration for NDE/NIE marginalises over the mediator distribution using the fitted mediator model rather than a Gaussian approximation.

---

## Where this fits in the causal fairness toolkit

`insurance-mediation` completes a set of five libraries for causal fairness analysis in UK personal lines:

- **[insurance-fairness](https://github.com/burning-cost/insurance-fairness)** — baseline proxy discrimination audit, quantifies indirect discrimination risk
- **[insurance-fairness-diag](https://github.com/burning-cost/insurance-fairness-diag)** — D_proxy scalar, which rating factors are responsible, per-policyholder vulnerability
- **[insurance-fairness-ot](https://github.com/burning-cost/insurance-fairness-ot)** — optimal transport correction, discrimination-free premium computation
- **[insurance-counterfactual-sets](https://github.com/burning-cost/insurance-counterfactual-sets)** — policyholder-level counterfactual sets for individual fairness
- **[insurance-mediation](https://github.com/burning-cost/insurance-mediation)** — causal pathway decomposition, separates legitimate from illegitimate effects

The diagnostic workflow is: `insurance-fairness-diag` first (does your model discriminate, and through which factors?). If postcode is identified as a discrimination channel, `insurance-mediation` to decompose how much of the postcode effect is defensible. If correction is needed, `insurance-fairness-ot` for the premium adjustment. If FCA audit requires individual-level evidence, `insurance-counterfactual-sets`.

These are not alternatives. They answer different questions at different levels of causal specificity.

---

**[insurance-mediation on GitHub](https://github.com/burning-cost/insurance-mediation)** — MIT-licensed, PyPI, 121 tests, 8 modules. Decomposes the postcode penalty you already knew was there.
---

**Related articles from Burning Cost:**
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Causal Inference for Insurance Pricing](/2026/02/25/causal-inference-for-insurance-pricing/)
- [Regression Discontinuity Design for Insurance Pricing](/2026/03/11/insurance-rdd/)
