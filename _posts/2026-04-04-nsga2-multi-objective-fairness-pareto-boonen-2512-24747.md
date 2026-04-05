---
layout: post
title: "The Fairness Impossibility You Cannot Optimise Away — and What Boonen et al. Get Right About It"
date: 2026-04-04
categories: [fairness, insurance-fairness, regulatory]
tags: [nsga2, pareto, multi-objective, fairness, consumer-duty, fca, mills-review, arXiv-2512.24747, counterfactual-fairness, group-fairness, individual-fairness, boonen, impossibility-theorem, pricing]
description: "Chouldechova (2017) proved that when group base rates differ, no classifier can simultaneously achieve calibration within groups, equal false positive rates, and equal false negative rates — a constraint every insurance pricing model must navigate. Boonen, Fan & Quan (arXiv:2512.24747) have now put a price tag on the trade-off in insurance. That price tag is what the FCA's summer 2026 AI review is going to ask for."
author: Burning Cost
math: true
---

Chouldechova (2017) proved that when group base rates differ, calibration within groups, equal false positive rates, and equal false negative rates cannot simultaneously hold. Kleinberg, Mullainathan & Raghavan (2017) proved a related result: calibration and equalised odds are mutually incompatible under unequal base rates. Together they established what the fairness literature calls the impossibility theorem: if two groups have different underlying claim rates, no pricing model can simultaneously satisfy all mainstream fairness criteria.

This is not a technical problem awaiting a better algorithm. It is a mathematical constraint. A pricing model that appears to satisfy all fairness criteria simultaneously either operates in a degenerate case where group base rates are equal, or it is measuring the criteria badly.

Boonen, Fan & Quan (arXiv:2512.24747, December 2025) have produced the most complete investigation we have seen of what this impossibility means in practice for insurance pricing. They apply NSGA-II multi-objective Pareto optimisation to map the trade-off surface explicitly across four fairness criteria, with concrete numbers on what each trade-off costs. The paper has real limitations. But the core move — quantify the trade-off rather than pretend it does not exist — is the right one, and it is the move UK pricing teams need to make before the FCA's Mills Review reports this summer.

---

## What Boonen et al. actually show

The paper compares four model types — GLM, XGBoost, an orthogonal debiasing model, and a synthetic control model — and then proposes an NSGA-II Pareto approach that blends them. It examines four fairness criteria simultaneously:

**Predictive accuracy (normalised Gini).** The ranking accuracy of the model — how well it sorts policyholders by actual loss cost. Higher Gini means better discrimination and fewer cross-subsidies.

**Group fairness (demographic parity).** The ratio of group-specific mean predicted premiums to the overall mean. A ratio of 1.0 means no group is being systematically over- or under-charged relative to the portfolio average.

**Individual fairness (local Lipschitz constant).** Similar policyholders should receive similar prices. The Lipschitz constant measures how rapidly premiums change across feature space — a lower constant means pricing is smoother and less erratic for policyholders with similar risk profiles.

**Counterfactual fairness.** The premium for an individual would not change materially if their protected characteristic were counterfactually different, holding everything causally independent of that characteristic constant.

The key finding on single-model comparison: XGBoost achieves the best predictive accuracy but amplifies fairness disparities most severely across all three fairness criteria. The orthogonal model leads on group fairness. The synthetic control model leads on individual and counterfactual fairness. No single model wins across all four dimensions — which is exactly the point.

The numbers, from the paper's empirical analysis: pushing the model ensemble toward maximum accuracy costs roughly 30–40% in group fairness. Recovering group fairness to near-parity costs roughly 10–15% in Gini. These are not trivial trade-offs. They are the numbers a pricing committee needs to see before deciding where to operate.

The NSGA-II approach generates a Pareto front across the objective space. Rather than selecting a single operating point in advance, it produces the full set of non-dominated solutions — every combination of objectives where you cannot improve on one criterion without worsening at least one other. TOPSIS then selects a specific point based on stated preference weights. The paper uses weights of [0.3, 0.3, 0.3, 0.1] across the four objectives, treating accuracy, group fairness, and individual fairness as equal priorities and downweighting counterfactual fairness as less actionable in practice.

The claimed result — that the NSGA-II ensemble "consistently achieves a balanced compromise, outperforming single-model approaches" — is true in the sense that any Pareto-optimal solution dominates or equals any single-model result. That is definitional. The more useful claim is that the Pareto approach makes the trade-off legible and auditable in a way that no single-model approach can.

---

## The insight the paper makes concrete

The impossibility theorem is abstract. Pricing committees can acknowledge it in principle and then proceed to optimise a single fairness criterion anyway, because the impossibility seems theoretical rather than operational. Boonen et al. make it operational.

When the paper shows that moving from maximum-accuracy XGBoost to the fairness-constrained alternatives costs 10–15 Gini points to recover group parity, it is producing evidence that the impossibility theorem makes necessary. You cannot satisfy all criteria — so you must choose, and the choice must be documented.

This is what FCA Consumer Duty PRIN 2A requires: that firms can demonstrate their pricing approach produces fair outcomes for all groups of customers sharing protected characteristics. "We optimised accuracy" is not an answer to this requirement. "We examined the full trade-off surface, selected an operating point with weights of 0.50 / 0.25 / 0.25 across accuracy, group fairness, and counterfactual fairness, and here is the Pareto front from which we selected" is.

---

## The FCA Mills Review angle

The FCA's Mills Review, launched January 2026 with a response deadline of February 2026 and reporting expected summer 2026, explicitly examines how AI and ML affect consumer outcomes in retail financial services. The themes from the Call for Input include:

- Whether AI-enabled hyper-personalisation creates risks of discrimination or exclusion
- Whether existing regulatory tools are adequate for AI-driven pricing
- What governance and auditability obligations should apply to AI pricing decisions
- Whether market concentration in AI-enhanced pricing warrants structural intervention

The Boonen et al. framework speaks directly to the first and third themes. The review is not asking whether insurance pricing is fair in some absolute sense — it is asking whether firms can *demonstrate* they have examined the fairness implications of their AI pricing decisions and made considered, documented choices about trade-offs.

A firm that presents the Mills Review team with a single demographic parity ratio and a statement that it sits within acceptable bounds has answered a different question than the one being asked. A firm that can present a Pareto front, state the preference weights used to select an operating point, and explain why those weights reflect their Consumer Duty obligations is engaging with the actual question.

We do not know what the Mills Review will recommend. We are confident it will not accept "we measured demographic parity and it was fine" as an adequate governance record for an AI pricing model. The Pareto approach generates something more substantive.

---

## What the paper does not give you

**The dataset is unnamed.** Boonen et al. do not name the insurance dataset used in their empirical analysis — possibly a proprietary French motor dataset, possibly freMTPL, not confirmed. This makes the specific numbers (the 30–40% fairness cost at maximum accuracy) impossible to reproduce externally. They are useful as direction-of-travel estimates; do not use them as benchmarks for your own portfolio. Run the Pareto search on your own book.

**The causal structure for counterfactual fairness is assumed, not derived.** The "flip the protected attribute" approach implicitly assumes that no features in the model are downstream consequences of the protected characteristic. For UK motor insurance, this is questionable. If annual mileage is correlated with gender partly through causal mechanisms — differing commuting patterns, occupational patterns — then flipping gender while holding mileage constant produces a counterfactual that is causally inconsistent. The paper does not discuss this. Neither does most of the fairness literature that uses the same flip approach. It is a genuine limitation.

**NSGA-II does not find the exact Pareto front.** It finds a good approximation. For governance purposes, the key question is whether the approximation is stable: does a different random seed produce meaningfully different TOPSIS-selected operating points? The paper does not report seed stability. Our implementation in `insurance-fairness` addresses this explicitly: run at least three seeds and treat materially different TOPSIS selections as a signal to increase `pop_size` and `n_gen` until the front stabilises.

**Individual Lipschitz fairness is included as a Pareto objective but the distance function is unspecified.** The Lipschitz constant depends critically on the distance metric chosen to measure "similarity" between policyholders. The paper does not specify what distance function was used. This is not a minor methodological gap — the Lipschitz constant under a Euclidean distance on raw feature values is essentially uninterpretable for heterogeneous insurance features such as vehicle age (integer), annual mileage (continuous), and no-claims discount years (ordinal). Any results citing individual Lipschitz fairness from this paper should be treated as illustrative only until the distance function is specified and defended.

---

## A note on UK data availability

Most UK personal lines insurers do not hold self-declared protected characteristic data for their policyholders. Gender, ethnicity, disability status — these are not collected at point of sale, and in most cases cannot be collected without creating adverse selection and legal risks under the Equality Act 2010.

The `protected_col` parameter in the library example below assumes this is resolved: it takes a column name that holds the protected attribute value for each policyholder. In practice, UK teams face two situations.

**If only a proxy is available** (e.g., a name-based gender inference or a postcode-level deprivation score as a proxy for socioeconomic status): the group fairness and demographic parity calculations still run, but the results measure proxy discrimination, not direct discrimination against the protected characteristic. This is both useful (proxy discrimination is actionable under FCA EP25/2 and Consumer Duty) and limited (a model that scores well on proxy fairness may still discriminate against the underlying protected group if the proxy is imperfect).

**The counterfactual fairness flip is not well-defined for a proxy attribute.** Counterfactual fairness asks: would the premium change if the protected characteristic were different? For a genuine protected attribute, this is interpretable (with caveats about causal structure). For a proxy, it is not — flipping the proxy value does not correspond to a coherent counterfactual about the underlying characteristic. The `cf_tolerance` objective should be disabled or used only as a diagnostic when `protected_col` is a proxy.

Until insurers routinely collect verified protected characteristic data — which the regulatory direction of travel under Consumer Duty and EP25/2 may eventually require — the most defensible approach is to run the group fairness and accuracy objectives, report the proxy results transparently, and document the limitation explicitly in the governance record.

---

## How this relates to the insurance-fairness library

`insurance-fairness` v0.6.9 ships `NSGA2FairnessOptimiser`, implementing the Boonen et al. framework adapted for UK personal lines use. The key differences between the paper and the library:

**Exposure weighting throughout.** Gini, demographic parity, and counterfactual fairness are all computed on earned-exposure-weighted bases. Boonen et al. do not discuss exposure weighting. For insurance data where policies have varying exposure periods, unweighted metrics produce incorrect results — an annual policy with 0.5 years exposure should receive half the weight of a full-year policy in any fairness metric.

**Three Pareto objectives, not four.** We implement group fairness and counterfactual fairness as NSGA-II objectives. Individual Lipschitz fairness is computed post-selection via `LipschitzMetric` as a diagnostic. The reason is not that Lipschitz fairness is unimportant — it is that the result is only meaningful with a carefully specified distance function, and adding a poorly-specified fourth Pareto objective does more harm than good. A governance committee should not approve a model on the basis of a Lipschitz constant computed under Euclidean distance on raw features.

**Counterfactual fairness for multi-category attributes.** For binary protected attributes, we flip and recompute. For multi-category attributes, the objective is auto-disabled with an explicit warning: the flip operation is not well-defined without a specified causal structure. Boonen et al. do not address this case.

**TOPSIS with explicit weight documentation.** The JSON audit trail from `ParetoResult.to_json()` records the weights, seed, population size, and selected solution. This is designed to satisfy the FCA evidence pack requirement that governance decisions about AI be documented and reproducible.

```python
from insurance_fairness.pareto import NSGA2FairnessOptimiser

optimiser = NSGA2FairnessOptimiser(
    models={"base": model_accuracy, "fair": model_fairness},
    X=df_test,
    y=y_test,
    exposure=exposure_test,
    protected_col="gender_proxy",
    cf_tolerance=0.05,
)

result = optimiser.run(pop_size=100, n_gen=200, seed=42)

# The preference weights are the governance decision, not a technical parameter
idx = result.selected_point(weights=[0.50, 0.25, 0.25])

# Serialise for the evidence pack
result.to_json("pareto_audit_Q2_2026.json")
```

The preference weights `[0.50, 0.25, 0.25]` are not arbitrary — they are the documented governance decision. Accuracy receives 50% weight because poor Gini drives cross-subsidies that harm consumers. Group and counterfactual fairness receive equal weight at 25% each. If the pricing committee decides group fairness warrants more weight following a regulatory finding, rerunning with `[0.40, 0.40, 0.20]` produces a different operating point from the same Pareto front, and both runs are in the audit record.

---

## The incompatibility theorem as a governance communication tool

We think the Chouldechova-Kleinberg impossibility result is underused in pricing governance. Pricing teams sometimes approach fairness as a compliance box: demonstrate demographic parity, file the evidence, move on. The impossibility theorem shows why this approach is structurally inadequate.

If group base rates differ — and in insurance, they do, across virtually every protected characteristic — then a model that appears to satisfy all fairness criteria simultaneously is doing one of three things: it is operating in a degenerate case where the criteria happen to align numerically, it is satisfying them approximately rather than exactly (and the approximations are mutually inconsistent), or it is measuring the criteria incorrectly. There is no fourth option.

This means a pricing governance pack that claims the model "passes fairness checks" without specifying *which* fairness criterion was checked and *what was traded off against it* is making an implicitly false claim. The model satisfies some criteria and violates others. The question is whether the criteria it violates are the ones that matter for regulatory compliance, and whether the violation level is documented and justified.

The Pareto front approach makes this explicit. It does not claim to satisfy all criteria — it shows the trade-off surface and documents which point was chosen and why. That is, paradoxically, a more defensible regulatory position than a governance pack that claims to have "solved" fairness.

A pricing committee that approves a Pareto-selected model with documented weights has made a considered, auditable decision. A pricing committee that approves a single-objective fairness-constrained model has made the same decision implicitly, without examining what it cost or whether the criterion being constrained was the right one. The first is defensible under SM&CR accountability obligations. The second is not.

---

## Our assessment

Boonen et al. (arXiv:2512.24747) is the best academic treatment we have seen of multi-objective fairness trade-offs in insurance pricing. The use of NSGA-II over weighted-sum scalarisation is correct — weighted sums can miss non-convex regions of the Pareto front. The inclusion of counterfactual fairness is right in principle. The Lipschitz individual fairness criterion is included but insufficiently operationalised.

The empirical result that XGBoost amplifies fairness disparities despite leading on accuracy is useful and credible, consistent with what we see on UK personal lines data. GBMs learn to exploit proxy correlations more effectively than GLMs, which is precisely why FCA EP25/2 (the GIPP remedies evaluation, July 2025) and the Consumer Duty Outcome 4 monitoring are so focused on ML-driven pricing outcomes rather than GLM-based pricing.

The paper's main contribution is not the algorithm — NSGA-II is two decades old — but the framing. Treating fairness as a multi-dimensional trade-off to be mapped rather than a constraint to be satisfied is the correct framing for regulatory compliance, and the paper demonstrates it with numbers from an insurance context.

What the paper cannot give you is a drop-in solution. The dataset is unnamed, the causal structure for counterfactual fairness is assumed, and the distance function for Lipschitz fairness is unspecified. The `insurance-fairness` library fills these gaps for UK personal lines use.

For the FCA's summer 2026 AI review: run the Pareto search on your own book. Document the trade-offs. State the preference weights explicitly. That is not a research methodology — it is a governance requirement the FCA is increasingly likely to enforce.

---

## Reference

Boonen, T.J., Fan, X. and Quan, Z. (2025). Fairness-Aware Insurance Pricing: A Multi-Objective Optimization Approach. arXiv:2512.24747. Submitted December 31, 2025. University of Hong Kong.

Related on Burning Cost:

- [Multi-Objective Insurance Pricing: Finding the Pareto Front of Accuracy, Fairness, and Profit](/2026/03/25/multi-objective-pareto-insurance-pricing-nsga2/)
- [Detection vs Mitigation: insurance-fairness and EquiPy Are Not Competing](/2026/04/04/insurance-fairness-equipy-comparison/) — the distinction between fairness auditing (what we need for the FCA) and fairness correction (what NSGA-II is doing) — full practitioner workflow with worked UK motor example
- [Fairness-Accuracy Tradeoffs in Insurance Pricing — Pareto Frontiers with NSGA-II](/2026/03/28/fairness-accuracy-pareto-nsga2-insurance-pricing/) — `NSGA2FairnessOptimiser` API and three-objective implementation detail
- [The Mills Review: What the FCA's Long-Term AI Inquiry Means for Pricing Teams](/2026/03/26/fca-long-term-ai-review/) — summer 2026 reporting context
