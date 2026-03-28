---
layout: post
title: "Fairness-Accuracy Tradeoffs in Insurance Pricing — Pareto Frontiers with NSGA-II"
date: 2026-04-20
categories: [tutorials, fairness]
tags: [fairness, NSGA-II, pareto, multi-objective-optimisation, insurance-fairness, consumer-duty, FCA, TOPSIS, pymoo, pricing, python, tutorial]
description: "Single-objective fairness constraints force a binary choice. NSGA-II finds the full tradeoff surface, so governance committees can make an explicit, documented decision about where to operate — not discover the choice post-deployment."
seo_title: "Fairness-Accuracy Pareto Frontiers in Insurance Pricing with NSGA-II and TOPSIS"
---

Most fairness work in insurance pricing treats the accuracy-fairness tradeoff as a knob to turn: add a demographic parity constraint to the loss function, choose a penalty weight, retrain. This obscures what is actually happening. The question is not "how fair do we want to be?" but "what does it cost, in Gini, to achieve a given level of demographic parity — and is that cost visible to the people who should be making the governance decision?"

NSGA-II gives you the answer to the first question. TOPSIS turns that answer into a documented governance choice. Together they turn fairness from a post-hoc audit finding into an explicit design decision.

The implementation we are discussing is `NSGA2FairnessOptimiser` in `insurance-fairness` v0.6.9, following the framework published in Boonen, Fan & Quan (arXiv:2512.24747, December 2025). That paper is, as far as we can determine, the first to apply NSGA-II specifically to insurance pricing fairness; the practitioner implementation in `insurance-fairness` is the only public code that ships it.

---

## Why single-objective constraints are not enough

The standard approach adds fairness as a constraint: `maximise Gini subject to demographic_parity_ratio >= 0.9`. This has two problems.

First, it encodes a specific tradeoff point in code, invisibly, before anyone has seen the tradeoff surface. The pricing team might discover that achieving a 90% parity ratio costs 3 Gini points — or it might cost 0.3. They do not know until after the constraint is hard-coded.

Second, there are multiple fairness criteria, and they pull in different directions. Chouldechova (2017) and Kleinberg et al. (2017) proved that demographic parity, equalised odds, and predictive rate parity are pairwise incompatible unless base rates are equal across groups. In insurance, groups defined by protected characteristics have different loss rates — that is precisely why proxy discrimination is a risk. You cannot achieve all fairness criteria simultaneously. Single-objective constraints pretend otherwise.

The Pareto front approach does not. It surfaces the complete set of non-dominated solutions across all objectives simultaneously, then hands the choice to the governance committee with explicit, auditable preference weights.

---

## NSGA-II in two minutes

Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002) is the standard method for multi-objective evolutionary optimisation. The practitioner-level summary:

NSGA-II maintains a population of candidate solutions. At each generation, it produces offspring via crossover and mutation, then selects survivors using two criteria. The first is **non-domination rank**: a solution ranks higher if no other solution is simultaneously better on every objective. The second is **crowding distance**: among solutions at the same rank, prefer those in less-crowded regions of objective space. This second criterion pushes the population toward a well-spread approximation of the full Pareto front, rather than clustering around a few good points.

The result after several hundred generations is a set of Pareto-non-dominated solutions: no solution in the set can be improved on any objective without worsening at least one other. That set is the Pareto front.

For insurance fairness, the decision variables are not model weights or hyperparameters — retraining 20,000 times per optimisation run would take days. Instead, we pre-train K models representing different points on the accuracy/fairness spectrum, then let NSGA-II search over **ensemble mixing weights** across those models. Each candidate solution is a vector of weights `w_1, ..., w_K` (constrained to sum to 1). Evaluation is a weighted sum of pre-computed predictions: approximately 1ms per candidate, making 20,000 evaluations feasible on a standard CPU in under a minute.

---

## Three objectives (not four)

Boonen et al. define four objectives: predictive accuracy, group fairness, individual Lipschitz fairness, and counterfactual fairness. We implement three of them as NSGA-II objectives. Here is why, and what each one means.

**Negative Gini coefficient** — we minimise the negative of the exposure-weighted Gini coefficient of model predictions. Minimising negative Gini maximises predictive discrimination. This is the accuracy objective. We use Gini rather than RMSE because Gini measures ranking accuracy — the ability of the model to sort policyholders by risk — which is what pricing lift depends on.

**Group unfairness** — `1 - demographic_parity_ratio`, where demographic parity ratio is the exposure-weighted ratio of mean predicted premium for the least-favoured group to the most-favoured group. Zero means perfect parity; one means maximum disparity. This directly targets s19 Equality Act 2010 indirect discrimination.

**Counterfactual unfairness** — `1 - counterfactual_fairness_score`, where the score is the proportion of policies for which flipping the protected characteristic (e.g., gender) changes the predicted premium by less than 5% in log-space. A policy is counterfactually fair if gender is genuinely irrelevant to its premium. This is the most legally precise of the three objectives.

**Why we do not include individual Lipschitz fairness as an NSGA-II objective:** the Lipschitz constant measures how rapidly premiums change across the feature space. A lower constant indicates a smoother, more stable model. The problem is that computing a meaningful Lipschitz constant requires a user-defined distance function `d(x, x')` specifying what it means for two policyholders to be "similar." The paper does not specify what distance function was used. Using a poorly-specified distance function as an NSGA-II objective would silently bias the optimiser toward regions of feature space where pairs happen to be close under that metric — not toward genuinely fairer models. We ship `LipschitzMetric` as a diagnostic tool, not an optimisation objective.

---

## Working example

Install with:

```bash
pip install insurance-fairness[pareto]
```

This pulls in `pymoo>=0.6.1` as the NSGA-II engine. The example below uses synthetic UK motor data with 5,000 policies and a binary gender column.

### Step 1 — train models representing the tradeoff

The quality of the Pareto front depends on the diversity of the ensemble. You need at least two models representing genuinely different points on the accuracy/fairness spectrum:

```python
import numpy as np
import polars as pl
from catboost import CatBoostRegressor

# model_base: trained on all features including gender — maximises accuracy
model_base = CatBoostRegressor(iterations=200, depth=4, loss_function="RMSE",
                               random_seed=42, verbose=False)
model_base.fit(X_train_with_gender, y_train, sample_weight=exposure_train)

# model_fair: trained without gender — sacrifices some accuracy for group fairness
model_fair = CatBoostRegressor(iterations=200, depth=4, loss_function="RMSE",
                               random_seed=42, verbose=False)
model_fair.fit(X_train_without_gender, y_train, sample_weight=exposure_train)
```

For richer fronts, add a third model trained with a demographic parity constraint, or a counterfactual-fair model trained with gender held constant. With three models you get a two-dimensional simplex as the weight space; NSGA-II explores it thoroughly.

### Step 2 — run NSGA-II

```python
from insurance_fairness.pareto import NSGA2FairnessOptimiser

optimiser = NSGA2FairnessOptimiser(
    models={"base": model_base, "fair": model_fair},
    X=df_test.select(["gender", "age", "vehicle_age", "ncd_years", "annual_mileage"]),
    y=y_test,
    exposure=exposure_test,
    protected_col="gender",
    cf_tolerance=0.05,   # 5% premium change = materially different
)

result = optimiser.run(pop_size=100, n_gen=200, seed=42)
print(result.summary())
```

`result.F` is an `(n_pareto, 3)` array of objective values. `result.weights` is the corresponding `(n_pareto, 2)` array of ensemble weights. Typical output for a two-model ensemble:

```
NSGA-II Pareto Front Summary
=============================
Solutions on front: 47
Objectives (all minimised — lower is better):

  neg_gini (accuracy):
    best  = -0.3821   worst = -0.2944
    range =  0.0877

  group_unfairness:
    best  =  0.0031   worst =  0.1842
    range =  0.1811

  cf_unfairness:
    best  =  0.0114   worst =  0.1773
    range =  0.1659

TOPSIS-selected solution (equal weights):
  neg_gini        = -0.3412
  group_unfairness =  0.0612
  cf_unfairness    =  0.0803
  Ensemble weights: base=0.483, fair=0.517
```

The range lines tell you the actual cost of tradeoffs. Here: moving from maximum accuracy to maximum group fairness costs 8.8 Gini points. Whether that is acceptable is a governance question, not a technical one — but now the committee has a number to approve rather than an assumption to discover.

### Step 3 — plot the front

```python
idx = result.selected_point(weights=[0.4, 0.3, 0.3])  # prefer accuracy slightly
fig = result.plot_front(highlight=idx)
fig.savefig("pareto_front.png", dpi=150, bbox_inches="tight")
```

`plot_front()` produces three pairwise scatter plots: accuracy vs group fairness, accuracy vs counterfactual fairness, and group vs counterfactual fairness. Each dot is a Pareto-non-dominated solution; the highlighted point is the TOPSIS selection. This is the chart that goes in the model governance pack.

### Step 4 — select an operating point and serialise

```python
# Equal weights — balanced compromise
idx_equal = result.selected_point()

# Accuracy-weighted — for lines where ranking quality matters most
idx_accuracy = result.selected_point(weights=[0.6, 0.2, 0.2])

# Fairness-weighted — for FCA evidence pack or consumer-facing products
idx_fair = result.selected_point(weights=[0.2, 0.5, 0.3])

# Get the ensemble weights for deployment
selected_weights = result.weights[idx_equal]
print(dict(zip(result.model_names, selected_weights)))
# {'base': 0.483, 'fair': 0.517}

# Serialise for audit trail
result.to_json("pareto_result_2026Q1.json")
```

The JSON output captures the full front, the selected point, NSGA-II parameters, and the TOPSIS weights used. This is your audit evidence: a dated, reproducible record that the tradeoff was examined, the preference weights were stated explicitly, and the selected operating point was a deliberate governance choice.

---

## TOPSIS: choosing one point from the front

TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) selects the Pareto point that is simultaneously closest to the ideal solution (best value on each objective independently) and furthest from the anti-ideal solution (worst value on each objective).

The key parameter is the `weights` vector. Weights `[0.4, 0.3, 0.3]` say: "accuracy accounts for 40% of our preference, group fairness 30%, counterfactual fairness 30%." These weights are normalised internally, applied to column-normalised objectives, and appear verbatim in the audit record.

TOPSIS is preferable to lexicographic ordering (rank objectives in strict priority) for governance purposes because preference weight changes produce smooth changes in the selected point. With lexicographic ordering, a small change in priority ranking can flip the selection discontinuously — a poor property when you are doing sensitivity analysis to show a regulator that the choice is robust.

The weights are not magic numbers. Varying them systematically is part of the governance process: does the selected operating point change materially if you shift weight between accuracy and fairness? If it does, the front is steep in that region and the committee should understand why.

---

## The FCA Consumer Duty angle

FCA Consumer Duty (PRIN 2A, FG22/5) requires firms to demonstrate that pricing algorithms do not produce systematically worse outcomes for groups of customers unless differences can be objectively justified. The 2026 insurance supervisory priorities — published February 2026 — named AI outcome testing as a specific supervisory focus and explicitly flagged that "AI-enabled hyper-personalisation could render some customers uninsurable, or enable discrimination."

The Pareto front approach answers the Consumer Duty requirement directly: it produces a documented, dated record that the accuracy-fairness tradeoff was examined, the preference ordering was stated explicitly, and the selected model was approved at the appropriate governance level (SM&CR accountability falls to the senior manager responsible for the pricing algorithm). A regulator reviewing an s166 report can see the front, the TOPSIS weights, and the rationale — not just the outcome.

What the Pareto approach does not replace: you still need `calibration_by_group()` to verify the selected model is not mispricing risk for any group, and `DoubleFairnessAudit` to check Consumer Duty Outcome 4 (equal value, not just equal premiums). The Pareto front is the governance record; the audit metrics are the ongoing monitoring.

---

## Integration with FairnessAudit

If you are already using `FairnessAudit` for routine monitoring, the Pareto run is available as a flag:

```python
from insurance_fairness import FairnessAudit

audit = FairnessAudit(
    model=primary_model,
    data=df,
    protected_cols=["gender"],
    prediction_col="predicted_premium",
    outcome_col="claims",
    exposure_col="exposure",
    run_pareto=True,
    pareto_models={"base": model_base, "fair": model_fair},
    pareto_pop_size=100,
    pareto_n_gen=200,
    pareto_seed=42,
)
report = audit.run()
# report.pareto_result is a ParetoResult instance
```

`report.to_dict()` and `report.to_json()` include the Pareto result, so it flows into whatever evidence pack or model monitoring store you are maintaining.

---

## Honest limitations

**Computational cost scales with models and generations.** With two models and `pop_size=100, n_gen=200`, a run takes roughly 30–60 seconds on a standard laptop. With five models the weight space is four-dimensional; convergence typically requires more generations. Profile before adding models to the ensemble.

**NSGA-II is stochastic.** Different random seeds produce different Pareto fronts. For production use, run at least three seeds and compare. If the TOPSIS-selected operating point changes materially between seeds, increase `pop_size` and `n_gen` until the front stabilises. The `seed` parameter makes any single run reproducible; the multi-seed stability check is not yet automated in the library.

**The quality of the front depends on ensemble diversity.** Two similar models produce a thin, uninformative front that spans a narrow range of the tradeoff space. The ensemble should genuinely cover different approaches: an accuracy-maximising model, a group-fairness-constrained model, and — if counterfactual fairness is a priority — a model trained without the protected characteristic entirely.

**Counterfactual fairness requires a defined causal structure.** The counterfactual objective works by flipping the protected characteristic and recomputing predictions. For binary attributes (gender 0/1) this is straightforward. For multi-class attributes (e.g., ethnicity with multiple categories), the flip is ambiguous — the implementation auto-disables the counterfactual objective in this case. More fundamentally, the "flip" operation assumes no mediated effects: it is methodologically valid only if the protected characteristic does not cause other features in your dataset. If your feature matrix includes post-treatment variables correlated with the protected characteristic, the counterfactual computation is not a true counterfactual in the Pearl sense.

**Individual Lipschitz fairness is available but experimental.** `LipschitzMetric` estimates the maximum premium change per unit of feature-space distance. The result is only interpretable if the distance function encodes meaningful similarity between policyholders — Euclidean distance on raw feature values almost never qualifies. Do not include Lipschitz figures in any regulatory report without a written justification of the distance function.

---

## What gets shipped

`insurance-fairness` v0.6.9 ships:

- `FairnessProblem` — the three-objective pymoo problem definition, separately instantiable for unit testing
- `NSGA2FairnessOptimiser` — the pymoo NSGA-II wrapper with pre-computed predictions
- `topsis_select()` — standalone TOPSIS, usable independently of the optimiser
- `ParetoResult` — dataclass with `plot_front()`, `selected_point()`, `to_json()`, `from_dict()`, `summary()`
- `LipschitzMetric` (experimental) — Lipschitz constant estimation from sampled policy pairs
- `FairnessAudit` integration via `run_pareto=True`

The paper by Boonen, Fan & Quan is the academic grounding; the library is the practitioner implementation. There is no code in the paper — this is, as far as we know, the only public implementation.

```bash
pip install "insurance-fairness[pareto]>=0.6.9"
```

Full API documentation at [insurance-fairness.readthedocs.io](https://insurance-fairness.readthedocs.io). Source and 145 tests at [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness).
