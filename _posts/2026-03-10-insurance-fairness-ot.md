---
layout: post
title: "Discrimination-Free Pricing in Python: Causal Paths, Optimal Transport, and the FCA"
date: 2026-03-10
author: Burning Cost
categories: [techniques, compliance, libraries]
tags: [fairness, discrimination-free-pricing, optimal-transport, causal-inference, FCA, Consumer-Duty, Equality-Act, Lindholm, Wasserstein, LRTW, motor, python, insurance-fairness]
description: "Discrimination-free UK insurance pricing via Wasserstein barycenter and causal path decomposition. insurance-fairness-ot for FCA EP25/2 compliance in Python."
---

Detecting proxy discrimination in your pricing model is the first problem. Fixing it is harder.

Our earlier library, [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness), handles the audit side: given a trained model, it tells you whether postcode is proxying for ethnicity, how much of the motor premium disparity between high- and low-BME postcodes is attributable to each rating factor, and what the Equality Act exposure looks like. That is useful. But the output of an audit is a finding, not a price. The question after the audit is: what do we charge instead?

That is what [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) answers. It computes discrimination-free premiums by (a) decomposing the effect of a protected attribute on the price into causal paths, (b) removing only the paths that constitute discrimination, and (c) handling the multi-attribute case -- where you are adjusting for gender and disability status simultaneously -- via Wasserstein barycenter correction. The result is a corrected premium that can go straight into a GLM tariff table as a set of multiplicative relativities.

```bash
uv add insurance-fairness
# or
uv add insurance-fairness
```

---

## The regulatory context

The FCA's Consumer Duty (PS22/9, effective July 2023) requires firms to demonstrate fair value across groups defined by protected characteristics. TR24/2 (August 2024) found that most fair value assessments were too high level to satisfy this -- the FCA's phrasing was that firms "lacked the granularity to adequately evidence good outcomes across customer groups." EP25/2 (July 2025), which evaluated the GIPP pricing remedies, confirmed the FCA is looking beyond price-walking and at how firms monitor distributional outcomes within their models.

The Equality Act 2010 Section 19 captures indirect discrimination: a pricing practice that puts customers with a protected characteristic at a particular disadvantage, even where the characteristic is not explicitly used, is unlawful unless the firm can show proportionate justification through legitimate actuarial purpose.

None of this is new. What is new is that the FCA has moved from supervisory letters to enforcement referrals, and that TR24/2 made clear the standard of evidence required. Asserting that your model is actuarially justified is not enough. You need to demonstrate which effects are justified, which are proxy discrimination, and what you have done about the latter.

---

## Why auditing is not enough

The insurance-fairness library identifies the problem. But knowing that annual mileage proxies for gender is not the same as knowing what premium to charge instead.

The naive approach -- drop the offending variable and retrain -- is usually wrong. Annual mileage is a genuine predictor of claim frequency. Young male drivers tend to drive more miles than young female drivers; the mileage effect on claims is real and actuarially defensible. Dropping mileage removes legitimate predictive signal along with the discriminatory proxy effect, which produces a worse model and does not eliminate the discrimination either -- the remaining variables will partially pick it up.

What you want is to preserve the mileage-to-claims signal while removing the gender-to-mileage-to-claims path that operates as indirect discrimination. The Lindholm et al. (2022) framework provides exactly this distinction, and the Côté-Genest-Abdallah (2025) causal graph formalism makes it operationally precise.

---

## The Lindholm (2022) framework

Lindholm, Richman, Tsanakas and Wüthrich published "Discrimination-Free Insurance Pricing" in ASTIN Bulletin 52(1) in 2022. The paper defines a discrimination-free price as one where the expected premium for a customer does not depend on the protected attribute -- not because the attribute is excluded from the model, but because the pricing function averages over the marginal distribution of that attribute.

Formally, given a model mu_hat(X, D) where X is rating factors and D is the protected attribute, the discrimination-free price is:

```
h*(X) = sum_d mu_hat(X, d) * P(D = d)
```

You evaluate the model at every possible value of D and take the exposure-weighted average. This is marginalisation over D. The result, h*(X), does not depend on D -- by construction. A male policyholder and a female policyholder with identical risk profiles (same age, same vehicle, same postcode, same claims history, same mileage) receive the same premium.

Critically, the model is still trained with D as a feature. That is the point: you need mu_hat(X, D) to be well-defined for each value of D so the marginalisation is meaningful. If you train without D, you lose the ability to disentangle the D-specific effects at all.

There is a bias correction step. The marginalised premium h*(X) has a slightly different mean from the original mu_hat(X, D), because the portfolio is not balanced across D groups. The LindholmCorrector fits a proportional correction factor on calibration data. Lindholm et al. Example 8 (smoker pricing) gives concrete values: the marginalised fair price h* for smokers is £0.200, for non-smokers £0.184, with a portfolio-level bias factor of 1.011. We reproduce these to four significant figures in the test suite.

The LindholmCorrector in insurance-fairness implements this:

```python
from insurance_fairness.optimal_transport import LindholmCorrector
import polars as pl

# X_calib: non-protected features. D_calib: protected attribute column(s).
# model_fn takes a combined DataFrame (X + D columns) and returns predictions.

corrector = LindholmCorrector(
    protected_attrs=["gender"],
    bias_correction="proportional",
    log_space=True,   # operates in log(mu) space for GLM models
)

corrector.fit(
    model_fn=my_glm_predict,
    X_calib=X_train,
    D_calib=D_train,
    exposure=exposure_train,
)

# Fair premiums: same for M and F at identical risk profiles
fair_premiums = corrector.transform(my_glm_predict, X_test, D_test)

print(f"Bias correction factor: {corrector.bias_correction_factor_:.4f}")
# Bias correction factor: 1.0113
```

The `log_space=True` flag is important for GLM models. The Lindholm formula averages over D, which is additive. GLMs predict on the log scale (eta = log(mu)), so the averaging should happen there -- geometric mean, not arithmetic mean -- then exponentiate. Mixing the two produces incorrect results for log-linked models.

---

## Causal path decomposition (Côté-Genest-Abdallah, 2025)

Côté, Genest and Abdallah published "A fair price to pay: Exploiting causal graphs for fairness in insurance" in the Journal of Risk and Insurance 92(1) in 2025 (DOI: 10.1111/jori.12503). This paper is the key upgrade from Lindholm.

Lindholm's framework gives you a mathematically clean discrimination-free price, but it treats the feature set X as a monolith: any variable in X is treated identically in the marginalisation. In practice, this is wrong. Some variables in X are legitimately caused by the protected attribute and carry genuine actuarial content. Others are correlated with the protected attribute but carry no independent risk signal -- they are proxies. Applying the Lindholm correction without distinguishing these groups will either over-correct (removing justified predictive signal) or under-correct (leaving proxy effects in place), depending on how the variables are specified.

The Côté-Genest-Abdallah DAG formalises this with four node types:

- **S (protected)**: gender, disability status
- **R (justified mediator)**: variables caused by S but with genuine independent causal effect on claims -- claims history is an example; male drivers in their 20s accumulate more claims, and those claims predict future frequency regardless of current gender
- **V (proxy)**: variables that are correlated with S but carry no independent causal effect on claims once you condition on S and R -- annual mileage in a usage-pattern-proxy sense, where the mileage signal is almost entirely gender correlation rather than genuine exposure information
- **Y (outcome)**: claims frequency or severity

Causal paths from S to Y are then classified:
1. **Direct path** S -> Y: direct effect of the protected attribute on claims. Must be removed.
2. **Proxy path** S -> V -> Y: indirect discrimination via a proxy variable. Must be removed.
3. **Justified path** S -> R -> Y: effect operates through a legitimate risk mediator. Permissible to retain.

The Lindholm marginalisation applied to the full DAG removes paths 1 and 2 while preserving path 3 -- but only if you specify the DAG correctly. That specification is the modelling and governance decision. It cannot be read off the data; it requires domain knowledge and sign-off from your actuarial governance structure.

In insurance-fairness, the DAG specification uses the CausalGraph builder:

```python
from insurance_fairness.optimal_transport import CausalGraph, PathDecomposer, DiscriminationFreePrice

# UK motor example
graph = (CausalGraph()
    .add_protected("gender")
    .add_justified_mediator("claims_history", parents=["gender"])
    .add_proxy("annual_mileage", parents=["gender"])
    .add_covariate("vehicle_group")
    .add_covariate("age_band")
    .add_outcome("claim_freq")
    .add_edge("claims_history", "claim_freq")
    .add_edge("annual_mileage", "claim_freq")
    .add_edge("vehicle_group", "claim_freq")
    .add_edge("age_band", "claim_freq"))

graph.validate()   # checks DAG is acyclic, outcome exists, protected node reaches outcome
```

Here `claims_history` is declared a justified mediator (R): the assertion is that past claims independently predict future claims, and that the gender-claims_history correlation does not make claims history a discriminatory proxy. `annual_mileage` is declared a proxy (V): the assertion is that the mileage signal in this model is predominantly gender correlation, not genuine exposure information that independently predicts claims.

The PathDecomposer then attributes the premium impact of gender to each causal path:

```python
decomposer = PathDecomposer(graph=graph, model_fn=my_glm_predict)

decomp = decomposer.decompose(
    X=X_test,
    D_values={"gender": ["M", "F"]},
)

# decomp.direct_effect: premium variance from direct S -> Y path
# decomp.proxy_effect: premium variance via annual_mileage
# decomp.justified_effect: premium variance via claims_history (permissible)

df = decomp.as_polars()
print(df.describe())
```

This decomposition is the audit evidence. The proxy_effect column is what you need to report to the FCA: this is the premium impact attributable to indirect discrimination via annual mileage, which we are removing. The justified_effect is what you retain -- it represents the legitimate actuarial content of the path S -> claims_history -> claims.

The critical limitation: the DAG is user-provided. Wrong DAG means wrong decomposition. CausalGraph.validate() catches structural errors (cycles, disconnected nodes, missing outcome), but it cannot check actuarial validity. Misclassifying a proxy as a justified mediator is a governance error that no algorithm can detect. Document the DAG specification as a model governance decision with the appropriate sign-off.

---

## Wasserstein barycenter for multiple protected attributes

The Lindholm formula handles one protected attribute at a time. With two attributes -- say, gender and disability status -- sequential application introduces order dependence and distortions. Remove gender first, then disability, and you get a different answer from doing it in the opposite order.

Wasserstein barycenter correction provides a geometrically principled simultaneous correction. The 1D Wasserstein barycenter of distributions {P_A, P_B} with weights {omega_A, omega_B} has quantile function:

```
Q_bar(u) = omega_A * F_A^{-1}(u) + omega_B * F_B^{-1}(u)
```

The corrected score for group d is: m*(x_i) = Q_bar(F_d(mu_hat(x_i))). In plain terms: map the observed prediction to a probability via the group's empirical CDF, then map that probability back through the barycenter quantile function. Every group ends up with the same prediction distribution.

In 1D with two groups this is algebraically identical to the Lindholm formula. The value of the OT framing appears with more than two groups and in the multi-attribute case: the barycenter is well-defined and order-independent.

For GLM models, the correction operates in log(mu) space -- the eta scale -- then exponentiates. This preserves the multiplicative structure of GLM-based pricing. The WassersteinCorrector also computes the pre-correction Wasserstein distance between group prediction distributions, which is a useful diagnostic: a W2 distance of 0.02 on the log scale is barely worth reporting; a distance of 0.15 is a material disparity that needs addressing.

```python
from insurance_fairness.optimal_transport import WassersteinCorrector
import numpy as np

corrector_ot = WassersteinCorrector(
    protected_attrs=["gender", "disability"],
    epsilon=0.0,             # 0 = fully fair, 1 = no correction
    log_space=True,
    exposure_weighted=True,
    method="sequential",
)

corrector_ot.fit(
    predictions=my_glm_predict(XD_train),
    D_calib=D_train,
    exposure=exposure_train,
)

fair_premiums = corrector_ot.transform(
    predictions=my_glm_predict(XD_test),
    D_test=D_test,
)

# Check pre-correction Wasserstein distances
for attr, w2 in corrector_ot.wasserstein_distances_.items():
    print(f"{attr}: W2 = {w2:.4f}")
# gender: W2 = 0.0831
# disability: W2 = 0.1247
```

The epsilon parameter allows a partial correction -- a blend between fully fair and uncorrected. This is not a recommendation; it is there for sensitivity analysis and for cases where a regulator or governance body wants to understand what a 50% correction looks like relative to a full one.

One important clarification on fairness criteria. The OT barycenter achieves demographic parity: the distribution of predictions is identical across groups unconditionally. This is the wrong standard for UK insurance under both the Equality Act and FCA Consumer Duty, which require conditional fairness -- equal price for equal risk. Young drivers and older drivers have genuinely different risk profiles; equalising their premium distributions unconditionally would over-correct.

The correct approach -- which is what insurance-fairness implements by default -- is Lindholm marginalisation as the primary correction. OT is used only as a secondary tool, for the simultaneous multi-attribute case, applied to predictions that have already been Lindholm-corrected. Do not use WassersteinCorrector in isolation and call the result FCA-compliant. It is not.

---

## The main interface: DiscriminationFreePrice

The DiscriminationFreePrice class orchestrates graph specification, Lindholm correction, optional OT correction, and bias adjustment:

```python
from insurance_fairness.optimal_transport import (
    CausalGraph,
    DiscriminationFreePrice,
    FairnessReport,
    FCAReport,
)

graph = (CausalGraph()
    .add_protected("gender")
    .add_justified_mediator("claims_history", parents=["gender"])
    .add_proxy("annual_mileage", parents=["gender"])
    .add_outcome("claim_freq"))

dfp = DiscriminationFreePrice(
    graph=graph,
    correction="lindholm",           # or "wasserstein", or "lindholm+wasserstein"
    combined_model_fn=my_glm_predict,
    bias_correction="proportional",
    log_space=True,
)

result = dfp.fit_transform(
    X_train,
    D_train,
    exposure=exposure_train,
    y_combined=y_train,
)

# result.fair_premium: discrimination-free premiums
# result.best_estimate: original model predictions (for comparison)
# result.bias_correction_factor: should be close to 1.0
# result.decomposition: PathDecomposition with direct/proxy/justified splits

print(f"Average premium change: {(result.fair_premium / result.best_estimate - 1).mean():.3%}")
```

For separate frequency and severity models -- which is the right architecture for most UK motor and home pricing -- pass them separately:

```python
dfp = DiscriminationFreePrice(
    graph=graph,
    correction="lindholm",
    frequency_model_fn=freq_glm_predict,
    severity_model_fn=sev_glm_predict,
)
```

The fairness correction is applied independently to frequency and severity. This is correct. Applying it to the combined pure premium (frequency x severity) and calling the result fair does not guarantee fairness in the individual components -- if female drivers have systematically higher severity predictions due to a proxy effect, correcting the combined premium may mask the severity problem while over-correcting frequency.

---

## FCA reporting

FairnessReport produces the discrimination metrics table and path attribution summary:

```python
report = FairnessReport(result=result, graph=graph)

# Metrics: demographic_parity_ratio, conditional_parity_ratio, wasserstein_distance
metrics = report.discrimination_metrics(D=D_test, exposure=exposure_test)

# Attribution: variable | direct_effect | proxy_effect | justified_effect
attribution = report.path_attribution()

# Write FCA-format markdown with PS21/11 and Consumer Duty framing
report.to_fca_report("fairness_audit_q1_2026.md")
```

The FCAReport class produces a structured document with the premium comparison table by protected group (before and after correction), the path attribution showing what was removed and why, the bias correction factor, and a template for the Equality Act proportionality analysis. This is the document your model risk committee and compliance team need, not a notebook screenshot.

---

## How this differs from insurance-fairness

The original [insurance-fairness library](https://github.com/burning-cost/insurance-fairness) and this one solve different parts of the same problem.

insurance-fairness is an audit tool. It measures proxy correlation, runs disparate impact tests, and tells you whether your model has a problem. It does not produce corrected premiums. Use it to answer "do we have a discrimination issue?"

insurance-fairness is a pricing correction tool. It assumes you already know you have (or might have) an issue -- either from an insurance-fairness audit or from a governance decision about protected attributes in your market -- and produces the discrimination-free price. The causal graph specification forces you to be explicit about which effects you consider justified and which you are removing. That explicitness is itself regulatory value: when the FCA asks why you are charging what you charge, "we applied the Lindholm marginalisation after a governance-approved causal graph sign-off" is a more defensible answer than "we removed postcode and hoped for the best."

The other comparable Python tool is EquiPy (arXiv:2503.09866, March 2025, from UQAM and SCOR). EquiPy implements the Wasserstein barycenter approach and is theoretically sound. It lacks exposure weighting, causal graph integration, GLM-compatible output, and UK regulatory framing. It also has a sklearn 1.3.0 version pin that conflicts with modern environments. We built this optimal transport module because EquiPy is not fit for purpose in a UK personal lines pricing context.

---

## What the DAG decision really means

The most important thing in this workflow is not the mathematics. It is the causal graph.

When you call `.add_justified_mediator("claims_history", parents=["gender"])`, you are asserting that past claims independently predict future claims, and that the gender-claims history correlation does not make claims history a discriminatory proxy. This is a governance decision with legal significance. If the assertion is wrong -- if in fact claims history is proxying for gender effects rather than reflecting genuine independent risk -- the correction will be incomplete.

The library cannot validate this for you. What it can do is make the assertion explicit and auditable. The DAG is in source control. The path attribution report shows what was removed and what was retained. A peer reviewer or model validation team can challenge the classification of each variable. That is the correct governance process.

Our view: the DAG should be agreed between the Chief Actuary, the Compliance function, and Legal before any pricing correction is implemented. The sign-off should be documented. The assumptions should be revisited annually or when the model is materially changed.

---

`insurance-fairness` is open source under the MIT licence at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). Install with `uv add insurance-fairness`. Python 3.11+, NumPy, SciPy, networkx, POT (Python Optimal Transport), and Polars.

---

**Related articles from Burning Cost:**
- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) - the audit side: how to detect discrimination using insurance-fairness before reaching for the correction tool
- [Causal Inference for Insurance Pricing](/2026/02/25/causal-inference-for-insurance-pricing/) - the do-calculus foundations that underpin the causal path decomposition here
- [Building a Modern Insurance Pricing Pipeline in Python](/2026/03/12/modern-pricing-pipeline/) - where discrimination-free pricing sits in the full end-to-end workflow
