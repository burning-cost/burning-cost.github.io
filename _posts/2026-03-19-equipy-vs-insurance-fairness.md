---
layout: post
title: "EquiPy vs insurance-fairness: Two Different Questions About Fairness in Insurance Pricing"
date: 2026-03-19
author: Burning Cost
categories: [fairness, model-risk, libraries, regulation]
description: "EquiPy is a technically excellent fairness correction tool built on optimal transport theory, from Arthur Charpentier's group at UQAM. insurance-fairness is an FCA-focused proxy discrimination auditor. They are not competing — they address different stages of the same compliance problem."
tags: [equipy, insurance-fairness, FCA, proxy-discrimination, EP25/2, Consumer-Duty, pricing, fairness, python, uk-insurance, optimal-transport, Wasserstein, demographic-parity, Charpentier]
---

A Python fairness library published on arXiv in March 2025: [EquiPy](https://arxiv.org/abs/2503.09866), from Agathe Fernandes Machado, Suzie Grondin, Philipp Ratz, Arthur Charpentier, and François Hu. Charpentier is at UQAM and is one of the most credible actuarial academics working on fairness — his group has published several papers on Wasserstein-based discrimination correction in insurance contexts, and the theoretical foundations here are strong.

EquiPy and `insurance-fairness` both touch on pricing fairness. The comparison is worth making precisely, because the tools address fundamentally different stages of the compliance problem. Running EquiPy in place of `insurance-fairness` — or vice versa — will leave a UK insurer with the wrong deliverable for their situation.

```bash
pip install equipy
uv add insurance-fairness
```

---

## What EquiPy does

EquiPy is a post-processing fairness correction tool. You train your model without any fairness constraint. You then run EquiPy on the model's predictions to adjust them towards demographic parity. The correction is optimal in the Wasserstein-2 sense — it moves predicted values towards the Wasserstein barycenter of the group-conditional distributions, minimising the total "transportation cost" of the correction.

The single-attribute correction (`FairWasserstein`) implements the Chzhen et al. (2020) result. The multi-attribute extension (`MultiWasserstein`) handles sequential correction across several sensitive features simultaneously — the key theoretical contribution is that the sequential application order does not affect the result.

```python
from equipy.fairness import MultiWasserstein

calibrator = MultiWasserstein(sigma=0.0001)
calibrator.fit(predictions_calib, sensitive_features_calib)

# sensitive_features is a DataFrame with columns for each sensitive attribute
fair_predictions = calibrator.transform(predictions, sensitive_features)
```

The `epsilon` parameter allows approximate demographic parity — you can ask for corrections that bring unfairness below a threshold rather than eliminating it entirely:

```python
# Tolerate residual unfairness of 0.1 on gender, 0.2 on ethnicity proxy
fair_predictions = calibrator.transform(
    predictions, sensitive_features, epsilon=[0.1, 0.2]
)
```

The library also provides an `unfairness()` metric and visualisation utilities for inspecting the pre- and post-correction distributions.

It is model-agnostic. You pass in predictions as a numpy array or Series. It does not care what generated them.

---

## What insurance-fairness does

`[insurance-fairness](/2026/03/03/your-pricing-model-might-be-discriminating/)` is a pre-deployment audit tool. It does not modify model predictions. It asks whether a model discriminates, how that discrimination operates, and what evidence you can present to a pricing committee and the FCA.

The central output is a proxy vulnerability assessment: which of your rating factors are statistically correlated with protected characteristics, and how much of that correlation transmits into pricing variation?

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_rate",
    outcome_col="claim_amount",
    exposure_col="exposure",
    factor_cols=["postcode_district", "vehicle_age", "ncd_years", "vehicle_group"],
    model_name="Motor Model Q4 2024",
    run_proxy_detection=True,
)

report = audit.run()
report.to_markdown("audit_q4_2024.md")   # FCA-ready Markdown with regulatory mapping
```

This produces a structured report with RAG statuses, proxy R-squared per factor, mutual information scores, counterfactual premium impacts, and an explicit mapping to FCA EP25/2 and Consumer Duty Outcome 4. It is designed to sit in a pricing committee pack and survive FCA file review.

---

## The detection/correction distinction

This is the core reason the tools are not substitutes.

`insurance-fairness` answers: **does my model discriminate, and through which mechanism?** It is an investigative tool. If it finds that your postcode factor has a proxy R-squared of 0.22 against ethnicity and is generating a 9% premium differential attributable to proxy effects, you now have a documented finding that requires a documented response.

EquiPy answers: **given a model with unfair outputs, how do I produce fairer outputs?** It is a correction tool. It tells you nothing about why the unfairness is happening or which inputs are responsible.

A UK insurer working under FCA Consumer Duty should run detection first, for two reasons. First, the FCA's EP25/2 framework explicitly asks firms to *measure and understand* proxy effects — not just remove them. A firm that applies EquiPy corrections without a detection audit cannot explain to the FCA what they found and why they intervened. Second, detection may reveal that no material proxy effect exists, in which case applying post-processing corrections is unnecessary and will likely worsen calibration without regulatory benefit.

The correct workflow is: audit with `insurance-fairness`, characterise the proxy effects, decide whether intervention is warranted, then consider correction. EquiPy is a reasonable tool for the correction step.

---

## The demographic parity question

This is the harder technical tension, and it is worth being direct about it.

EquiPy enforces demographic parity: the corrected predictions have the same distribution across protected groups. This is the standard academic fairness criterion and it is, as the authors note, the only criterion applicable to both classifiers and regression models.

Demographic parity is probably too aggressive for insurance pricing, and the FCA does not require it.

The FCA's concern, as articulated in EP25/2 and Consumer Duty PS22/9, is proxy discrimination: protected characteristics transmitting into prices through non-protected rating factors. An insurer that eliminates all premium variation correlated with gender would satisfy demographic parity — but it would also eliminate legitimate risk-based differentiation that correlates with gender for actuarially justifiable reasons. The Gender Directive (2012) banned the *direct* use of gender as a rating factor, not the existence of gender-correlated pricing differences arising from risk-differentiated rating.

Wasserstein barycenter corrections remove *all* correlation between predictions and protected attributes, including the portion driven by genuine risk differences. This is more than Equality Act 2010 Section 19 requires. Section 19 prohibits *unjustified* indirect discrimination — a provision that explicitly permits employers and service providers to apply conditions that produce group disparities if those conditions are proportionate to a legitimate aim. Risk-based pricing is a legitimate aim. The actuarial exemption (Schedule 3 Part 5, EA2010) provides additional cover for insurers using actuarially justified factors.

EquiPy's authors are aware of this tension — their previous work on Wasserstein barycenters in insurance contexts discusses the tradeoff between fairness and actuarial soundness. But the library as shipped does not offer a mechanism to separate the discriminatory component of a group differential from the actuarially justified component. `insurance-fairness`'s `ProxyVulnerabilityScore` does: it estimates the gap between the unaware premium (no protected attribute used) and the aware premium (marginalised over the protected characteristic distribution), isolating the component attributable to proxy transmission specifically.

That said: for use cases where full demographic parity correction is appropriate — reinsurance pricing on non-actuarially-differentiated pools, underwriting scores used in automated decisions that must satisfy algorithmic fairness standards — EquiPy's theoretical guarantees are stronger than anything else currently available in Python.

---

## Insurance-specific data structure

One practical gap: EquiPy takes predictions as a flat array or Series. It does not have an exposure parameter.

Insurance portfolios are not flat rows. A fleet vehicle with 1.0 earned exposure is not the same as a private car with 0.3 earned exposure. Every fairness metric in `insurance-fairness` is exposure-weighted throughout. Running EquiPy on a portfolio with highly variable exposures will produce corrections that are technically optimal in unweighted prediction space but actuarially incorrect when translated into premium income.

This is not a criticism of EquiPy — it was not designed for insurance-specific data structures. It is a reason why exposure weighting matters and why a general-purpose tool requires adaptation before use in a pricing context.

`insurance-fairness` is also built specifically around Poisson and Gamma model outputs — the frequency/severity pair that underlies most UK motor, home, and commercial pricing models. Its calibration metrics (`calibration_by_group`) operate on rates with exposure offsets, and demographic parity is computed in log-space because the relevant comparison for a multiplicative model is a ratio of rates, not a difference in levels.

---

## Where EquiPy is genuinely better

On the correction problem itself, EquiPy's technical foundations are more rigorous than anything in `insurance-fairness`.

The Wasserstein barycenter approach provides optimality guarantees: the corrected prediction is the closest distribution (in Wasserstein-2 distance) to the original that achieves demographic parity. The sequential correction for multiple attributes (`MultiWasserstein`) is theoretically elegant — the order-invariance result means you are not making an arbitrary choice about which sensitive attribute to correct first. The `epsilon` parameter enables soft constraints, which is practically useful when full demographic parity is too aggressive.

The `insurance-fairness` optimal transport module (`insurance_fairness.optimal_transport`) implements Wasserstein corrections for the corrective premium component of `ProxyVulnerabilityScore`, but it does not expose the full multi-attribute sequential correction that EquiPy provides.

Charpentier's group has also produced a companion paper specifically on Wasserstein barycenters in insurance contexts (ECML PKDD 2023), which provides theoretical grounding for the application of these methods to actuarial data. The academic pedigree here is serious.

---

## Side-by-side

| | EquiPy | insurance-fairness |
|---|---|---|
| Purpose | Post-processing fairness correction | Pre-deployment discrimination audit |
| Core method | Wasserstein barycenter, optimal transport | DML proxy detection, CatBoost R-squared, mutual information |
| Output | Corrected predictions | FCA-ready audit report with RAG statuses |
| Fairness criterion | Demographic parity (enforced) | Proxy discrimination detection (measured, not enforced) |
| Multiple sensitive attributes | Yes — sequential, order-invariant | Yes |
| Exposure weighting | No | Throughout |
| Insurance model structure (Poisson/Gamma) | No | Yes |
| FCA regulatory mapping | No | EP25/2, Consumer Duty PRIN 2A, Equality Act s.19 |
| Assumes sensitive attribute available at correction time | Yes | Tests whether non-protected factors proxy for protected ones |
| Financial impact quantification | No | ProxyVulnerabilityScore, parity cost per policyholder |
| Theoretical guarantees | Wasserstein-2 optimality | Monte Carlo significance testing, DML asymptotics |
| Double fairness (action vs outcome Pareto) | No | DoubleFairnessAudit |
| PyPI | `equipy` | `insurance-fairness` |

---

## Our view

EquiPy is a technically strong library from credible authors. Arthur Charpentier has done more than almost anyone in actuarial science to apply rigorous mathematical fairness methods to insurance problems, and the optimal transport foundation gives EquiPy properties that ad-hoc correction approaches do not have.

We would not use it as a replacement for `insurance-fairness` in a UK regulatory context, for two reasons.

The first is the detection gap. UK insurers need to *explain* their fairness position to the FCA, not just satisfy a mathematical criterion. That requires knowing which factors are causing which effects and through what mechanism. EquiPy tells you what the corrected predictions look like. It does not tell you why the uncorrected predictions were unfair or which rating factors were responsible.

The second is the demographic parity standard. The FCA does not require premium equality across protected groups. It requires that non-protected rating factors not act as conduits for prohibited characteristics in ways the insurer cannot justify. Correcting to full demographic parity goes beyond what EP25/2 asks for, and it will degrade actuarial calibration by removing risk-differentiated pricing that correlates with protected characteristics for legitimate reasons.

For a UK insurer, the right order is: audit first with `insurance-fairness`, characterise the proxy effects, take those findings to the pricing committee, document the decision, and consider whether any correction is warranted. If correction is warranted, EquiPy is a better-grounded tool for the post-processing step than writing your own quantile-matching correction.

The two libraries are complementary. They are not competing for the same job.

---

`insurance-fairness` is at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). EquiPy is at [github.com/equilibration/equipy](https://github.com/equilibration/equipy).

---

**Related posts:**
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/) — the Lindholm-Richman-Tsanakas-Wüthrich framework, the Citizens Advice data in full, and what a defensible audit trail looks like
- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) — why generic ML fairness tools miss what the FCA cares about
- [FCA is Investigating Home and Travel Insurers](/2026/03/19/the-fca-is-investigating-home-and-travel-insurers/) — what the live enforcement risk looks like

---

**More library comparisons:** How our insurance-specific libraries compare to popular open-source alternatives.

- [Fairlearn vs insurance-fairness](/2026/03/20/fairlearn-vs-insurance-fairness-fca-proxy-discrimination/) — proxy discrimination auditing
- [MAPIE vs insurance-conformal](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/) — conformal prediction intervals
- [EconML vs insurance-causal](/2026/03/19/econml-vs-insurance-causal-inference-pricing/) — causal inference for pricing
- [DoWhy vs insurance-causal](/2026/03/18/dowhy-vs-insurance-causal-inference-insurance-pricing/) — causal graphs and refutation
- [Evidently vs insurance-monitoring](/2026/03/22/insurance-model-monitoring-evidently-alternative/) — model monitoring
- [NannyML vs insurance-monitoring](/2026/03/21/nannyml-vs-insurance-monitoring-drift-detection-insurance/) — drift detection
- [Alibi Detect vs insurance-monitoring](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) — statistical drift tests
- [sklearn TweedieRegressor vs insurance-distributional](/2026/03/22/sklearn-tweedie-vs-insurance-distributional-regression/) — distributional regression
