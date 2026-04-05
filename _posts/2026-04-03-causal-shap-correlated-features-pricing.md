---
layout: post
title: "Causal SHAP: Fixing Correlated Feature Attribution — and Why It Is Harder Than It Looks for Pricing"
date: 2026-04-03
categories: [research, techniques, explainability]
tags: [shap, causal-inference, shap-relativities, correlated-features, feature-attribution, causal-discovery, uk-motor, pricing, arXiv-2509.00846, PC-algorithm, DAG]
description: "IJCNN 2025 paper arXiv:2509.00846 introduces Causal SHAP: it uses causal discovery to estimate a DAG, then computes SHAP values that respect causal structure. The correlated-feature problem it solves is real for pricing teams. The assumptions it requires are not satisfied by UK motor data."
seo_title: "Causal SHAP for Insurance Pricing: Correlated Features and the DAG Problem"
seo_description: "Causal SHAP (arXiv:2509.00846) fixes unstable SHAP attribution for correlated features using causal discovery. We explain the method, why the problem is genuine for pricing actuaries, and why the hidden-confounder assumption kills direct application to UK motor GBMs."
author: burning-cost
---

Every pricing actuary who has sat in front of a SHAP bar chart from a UK motor model has encountered some version of this problem. NCD years and vehicle age are both in the model. Both are correlated with each other — drivers with low NCD tend to have newer vehicles because they are newer drivers. You run your SHAP attribution. The NCD bar is 40% bigger than vehicle age on one model refit; on the next, the bars switch. The combined attribution is stable. The individual attributions are not.

This is not a bug in your code. It is a structural property of how standard SHAP handles correlated features, and it has been documented since at least 2020. A paper published at IJCNN 2025 — Ng, Wang, Liu, and Fan, arXiv:2509.00846 — proposes a fix. The fix is genuine. The implementation is available on PyPI (`pip install fast-causal-shap`, v0.3.0, January 2026). Whether it is applicable to insurance pricing GBMs is a different question, and the answer is mostly no — for reasons that matter.

---

## The correlated features problem

Standard interventional SHAP computes a value function by drawing out-of-coalition features independently from their marginal distributions. For a coalition S that includes vehicle age but not NCD, it samples NCD from its marginal — effectively treating NCD as if it were independent of the vehicle age value you have just fixed.

For UK motor data, vehicle age 2 years and NCD 0 is a plausible combination. Vehicle age 2 years and NCD 5 is almost impossible — a driver with a 2-year-old car who has 5 years of no-claims has an inconsistency that almost never appears in training data. When SHAP constructs such combinations as part of its coalition evaluation, it is querying the model in a region it has never seen. The prediction it gets back is extrapolation, and the attribution computed from it is not grounded in anything real.

In practice this means that when two features share a common cause — driver age drives both NCD accumulation and vehicle tenure, which affects vehicle age — standard SHAP distributes attribution between them in a way that depends on the model's internal tree structure rather than on anything economically meaningful. The split is arbitrary and unstable.

The Pinnacle Actuarial article on SHAP in personal auto insurance documented exactly this: insurance score and prior claims history both correlate with each other and with driver quality; SHAP attribution between them can flip across random seeds. Our own `shap-relativities` docstring flags it directly: "SHAP attribution for correlated features is not uniquely defined. Correlated features share attribution in a way that depends on tree split order."

---

## What Causal SHAP does

The paper's insight is that the problem is one of the value function, not the Shapley weights. In standard interventional SHAP, out-of-coalition features are sampled independently from their marginals:

```
v(S) = E[f(X) | do(X_S = x_S)]
```

where X_{S-bar} ~ marginal. This produces the impossible-data-point problem.

Causal SHAP replaces this with a causally-consistent sampling distribution. Out-of-coalition features are drawn from the post-intervention distribution that respects the causal graph: if NCD is downstream of a shared cause of vehicle age, and vehicle age is in the coalition, NCD is sampled from its conditional on the intervened vehicle age value — not from its marginal.

The causal structure is estimated using the PC algorithm (Peter-Clark), which runs conditional independence tests with increasingly large conditioning sets to recover a skeleton, then orients edges using v-collider rules. From the resulting CPDAG (a partially directed acyclic graph), the IDA algorithm extracts multiple candidate DAGs and estimates edge weights by regression. Feature i gets a causal strength weight:

```
gamma_i = |W_i| / sum_j |W_j|
```

where W_i sums the products of edge weights along all causal paths from i to the target. The final causal SHAP value multiplies the standard Shapley marginal contribution by this weight. The headline theorem: any feature with no causal path to the target receives exactly zero attribution.

This directly addresses the coffee-smoking-lung-cancer example the paper uses. Drinking coffee is correlated with lung cancer only because it correlates with smoking. Standard SHAP assigns coffee non-zero importance. Causal SHAP assigns it zero, correctly.

The paper tests this on synthetic datasets with known causal ground truth and gets an RMSE improvement of roughly 129x on the lung cancer example. On real datasets (IBS and colorectal cancer, 21-31 features), gains are marginal: AUROC delta of 0.0005 to 0.0008. The paper is honest about this — the main claim is correctness on ground-truth benchmarks, not performance improvement.

---

## Why the key assumption breaks for insurance data

The PC algorithm requires causal sufficiency: every common cause of any two observed features must itself be observed. If there is a hidden confounder — an unobserved variable that causes both — the PC algorithm will draw a spurious direct edge between the two features, and the causal graph will be wrong. Wrong causal graphs produce worse attribution than standard SHAP, not better.

UK motor pricing features violate causal sufficiency systematically.

Driver age causes both NCD accumulation and vehicle age (through tenure effects). Driver age is typically in the model, so this is recoverable. But policyholder wealth — which is not in any rating model — simultaneously affects vehicle value, voluntary excess selection, parking conditions, annual mileage, and the quality of the garage where repairs are done. None of these features can be causally separated from each other without conditioning on wealth, which you do not observe.

Postcode-derived features (crime index, urban density, flood risk) all share latent neighbourhood socioeconomic structure that no rating file captures. Telematics features (hard braking frequency, motorway proportion, night driving share) are correlated with each other through driving personality traits and trip types that are not directly observed.

A UK motor model typically has 20 to 50 rating features. In a dataset of that dimension, with the kind of feature correlations typical of UK personal lines, there are likely dozens of unobserved common causes. The PC algorithm will produce a graph littered with false edges, and the causal SHAP values will be attributed according to a causal structure that does not exist.

---

## A second problem: factor tables and model consistency

Even if the causal graph were perfectly specified, Causal SHAP has a second problem for pricing applications.

`shap-relativities` extracts multiplicative relativities that reconstruct the model's own predictions: for any policy, `sum of SHAP log-contributions = log(predicted frequency)`. This is the efficiency axiom of Shapley values. The factor table is an exact decomposition of what the model actually does. That is what makes it auditable and what makes it suitable as a governance artefact.

Causal SHAP preserves additivity via renormalisation (Equation 11 in the paper: each causal SHAP value is rescaled so that they still sum to the model output). But the individual feature attributions will deviate from the unconditional Shapley decomposition, because the causal weighting changes how marginal contributions are distributed across the coalition.

The practical consequence: a factor table derived from Causal SHAP will not match the model's internal pricing behaviour for any given segment. NCD years 0 will have a different relativity than what the model actually implies when you vary NCD holding everything else constant. This undermines the central purpose of extracting factor tables from GBMs — which is to understand and communicate what the model actually prices.

---

## Where it could legitimately apply

We are not dismissing the paper. The correlated-feature attribution problem is real and the causal value function is the right theoretical framework for addressing it. Three narrower applications are worth considering.

**Pre-deployment audit against a specified DAG.** If your model validation team has a domain-knowledge causal structure — not one estimated from training data, but one specified independently by subject-matter experts and reviewed by actuaries — you could run Causal SHAP to check whether any feature receives material attribution only through correlation with a variable that has no plausible direct causal path to claims. This is a sanity check, not a factor table.

**Telematics feature selection.** Telematics data has genuine causal structure that practitioners understand reasonably well: trip distance causes exposure to claims; hard braking indicates driving style; night driving is partly a proxy for occupation type. With a defensible domain-knowledge DAG over 8-12 telematics features, the hidden-confounder problem is smaller than for a full rating dataset. Causal SHAP could help decide which telematics features are measuring genuinely independent risk dimensions and which are redundant.

**FCA proxy discrimination evidence.** Under the Consumer Duty (PRIN 2A, PS22/9) and informed by the FCA's December 2025 Research Note on motor insurance pricing and local area ethnicity, firms should be able to demonstrate they have investigated proxy discrimination and taken corrective action where found. If you want to argue that postcode has no direct discriminatory effect after conditioning on socioeconomic features, you need a causal argument. Causal SHAP provides the framework — but the argument only holds if you have an independently validated causal DAG, not one estimated from the very data you are explaining.

---

## The implementation gap

The `fast-causal-shap` package (v0.3.0, January 2026, MIT licence) is a research codebase. The repository is 98.5% Jupyter Notebooks. The README usage section says "// To be added". The package requires a pre-specified causal graph as a JSON dictionary — the PC/IDA discovery described in the paper is not run internally. Model support is scikit-learn estimators only; CatBoost, LightGBM, and XGBoost are not supported natively.

The method name `compute_modified_shap_proba()` suggests the implementation targets classification tasks only. For insurance frequency-severity modelling — Poisson or Tweedie GBMs with log-link — there is no supported path.

We are not planning to integrate Causal SHAP into `shap-relativities` in its current form. The methodological objections above are the main reason. The implementation gap is secondary.

---

## What to watch

The paper's stated future work includes FCI (Fast Causal Inference) as a replacement for PC when hidden confounders are present. FCI relaxes the causal sufficiency assumption by allowing for latent common causes and returning a PAG (partial ancestral graph) rather than a CPDAG. If a follow-up paper delivers FCI-based causal SHAP with demonstrated performance on datasets with realistic hidden confounder structure, the applicability to insurance data improves significantly.

We are also watching for any insurance-specific application that demonstrates factor table improvement on a public benchmark — freMTPL2 or similar — with features that have the kind of known correlational structure typical of UK motor data.

Until then: if you are running `shap-relativities` and you are worried about correlated-feature attribution stability, the immediate mitigations are to use `feature_perturbation="interventional"` with a representative background dataset (which corrects for impossible-data-point sampling without requiring a causal graph), and to document in your governance materials that paired features like NCD and vehicle age have shared attribution that cannot be uniquely decomposed. That is an honest answer to a hard question. It is more honest than a Causal SHAP factor table derived from a causal graph you estimated from your own training data.

---

*Paper: Ng, Wang, Liu, Fan (2025). "Causal SHAP: Feature Attribution with Dependency Awareness through Causal Discovery." IJCNN 2025. arXiv:2509.00846.*
*Implementation: [fast-causal-shap v0.3.0](https://pypi.org/project/fast-causal-shap/) (PyPI, MIT).*
*Related: [Extracting SHAP Relativities from Insurance GBMs](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/), [shap-relativities](https://github.com/burning-cost/shap-relativities).*
