---
layout: post
title: "The Fairness Pareto Front: Why There Is No Single Dial"
date: 2026-04-02
categories: [fairness, regulation, machine-learning]
tags: [fairness, pareto-front, nsga2, topsis, consumer-duty, fca, equality-act, insurance-fairness, demographic-parity, counterfactual-fairness, lipschitz, multi-objective-optimisation, arXiv-2512.24747]
description: "Pricing teams treat fairness as a single slider between accuracy and parity. NSGA-II reveals it is a landscape with multiple competing criteria. Here is what the Pareto front looks like, and how TOPSIS gives you an auditable selection for FCA Consumer Duty."
author: burning-cost
---

The standard framing of the fairness problem in insurance pricing goes something like this: you have a model that is accurate, and you want to make it fairer. You adjust something — perhaps you drop a proxy variable, or apply an orthogonalisation step — and you accept the accuracy loss. There is a dial, and you choose where to set it.

This framing is wrong, and the error is consequential.

There is no single dial because there is no single notion of fairness. Demographic parity (group fairness), Lipschitz continuity (individual fairness), and counterfactual fairness are mathematically distinct properties, and they trade off against each other as well as against accuracy. A model optimised for demographic parity can be highly counterfactually unfair. A model with low group-level disparity can violate individual fairness locally. No single model dominates on all criteria simultaneously — this is a mathematical result, not an empirical observation about your particular dataset.

Once you accept that, the useful question changes. It is no longer "how much accuracy do I give up for fairness?" It is "which point on the multi-dimensional trade-off surface do I want to occupy, and can I document why?"

---

## The Pareto front as a management tool

Suppose you have a set of candidate pricing models. Perhaps a standard XGBoost trained on all features, a GLM trained on a reduced feature set, an orthogonalised variant that projects out the protected characteristic's influence, and a synthetic control model. These sit at different points in the space of accuracy versus fairness — but that space has at least three dimensions, not one.

NSGA-II (Non-dominated Sorting Genetic Algorithm, generation 2) is a multi-objective evolutionary algorithm that finds the Pareto front of non-dominated solutions across all objectives simultaneously. A solution is Pareto-optimal if there is no other solution that is at least as good on every objective and strictly better on at least one. The full front gives you the set of defensible operating points.

The `insurance-fairness` library implements this in `pareto.py`. The entry point is `NSGA2FairnessOptimiser`:

```python
import polars as pl
from insurance_fairness.pareto import NSGA2FairnessOptimiser

# models: dict of {name: fitted_model}, at least two
# Each model must implement predict(X) — sklearn, CatBoost, or XGBoost all work
optimiser = NSGA2FairnessOptimiser(
    models={
        'base': xgb_model,
        'orthogonal': ortho_model,
        'glm': glm_model,
    },
    X=X_test,          # Polars DataFrame
    y=y_test,          # actual outcomes (claim frequency, severity, etc.)
    exposure=exposure, # policy-year exposure
    protected_col='region',  # or 'vehicle_age_band', etc.
    cf_tolerance=0.05,       # 5% log-space tolerance for counterfactual fairness
)

result = optimiser.run(pop_size=100, n_gen=200, seed=42)
print(result.summary())
```

The decision variables are not the model hyperparameters — they are the ensemble mixing weights over the pre-trained models. NSGA-II searches for the weight combinations that trace the full Pareto front across three objectives:

1. **Negative Gini coefficient** — accuracy; minimising this maximises predictive lift (Gini as a rank-ordering measure, not group discrimination).
2. **Group unfairness** — `1 − demographic_parity_ratio`; zero means perfect demographic parity between protected groups.
3. **Counterfactual unfairness** — the exposure-weighted proportion of policies where flipping the protected characteristic changes the predicted premium by more than 5% in log-space.

The pre-computation step matters here. `FairnessProblem` calculates all model predictions and counterfactual predictions at construction time. NSGA-II calls the objective function thousands of times; if that required re-running each model on every call, the runtime would be prohibitive. With pre-computation, evaluation is cheap array arithmetic over the cached predictions.

A typical run with `pop_size=100, n_gen=200` takes a few minutes on a modern laptop for a portfolio of 50,000 policies and three models. The result is a `ParetoResult` containing the objective values and ensemble weights for every non-dominated solution.

---

## What the front looks like

The Pareto front is a surface, not a curve. Plot any two objectives against each other and you see the achievable boundary — below and to the left of each point, no solution exists that improves both simultaneously.

```python
# After running optimiser.run()
idx = result.selected_point(weights=[0.5, 0.3, 0.2])
result.plot_front(highlight=idx)
```

`plot_front()` generates three 2D scatter plots: accuracy vs group fairness, accuracy vs counterfactual fairness, and group fairness vs counterfactual fairness. The full three-dimensional structure is not visible in any single 2D projection, which is part of the point: the space is genuinely multi-dimensional and cannot be collapsed to a single axis without information loss.

What you typically see is that accuracy and group fairness trade off roughly monotonically — pushing toward demographic parity costs Gini. But the counterfactual fairness dimension is less predictable. Orthogonalised models, which are designed to be group-fair, can still be highly counterfactually unfair because the orthogonalisation step operates at the group-mean level, not at the individual-flip level. This is not obvious before you plot the front.

It is also common to find that the pure-accuracy model (e.g. XGBoost with all features) sits well off the Pareto front in fairness dimensions — meaning there exist ensemble combinations that are strictly fairer on both group and counterfactual metrics at only a marginal accuracy cost. Those intermediate points are the ones worth examining. The Pareto front makes them visible.

---

## TOPSIS: an auditable decision

Once you have the Pareto front, you need to pick a point. This is where most pricing teams either hand-wave ("we chose a reasonable trade-off") or freeze ("there are 80 non-dominated solutions, which one do we submit to the model governance committee?").

TOPSIS — Technique for Order of Preference by Similarity to Ideal Solution (Hwang & Yoon, 1981) — gives a principled, auditable answer. The method identifies the solution that is simultaneously closest to the best achievable value on every objective and farthest from the worst. Critically, it requires the analyst to state explicit weights over the objectives before the algorithm runs.

```python
from insurance_fairness.pareto import topsis_select
import numpy as np

# Explicit stakeholder weights: we prioritise accuracy,
# then group fairness, then counterfactual fairness
stakeholder_weights = np.array([0.5, 0.3, 0.2])

selected_idx = topsis_select(result.F, weights=stakeholder_weights)

# The selected ensemble weights
chosen_model_weights = result.weights[selected_idx]
print(dict(zip(result.model_names, chosen_model_weights)))

# The achieved objective values at the selected point
print(dict(zip(result.objective_names, result.F[selected_idx])))
```

The `selected_point()` convenience method on `ParetoResult` calls `topsis_select` internally:

```python
# Equivalent to the above
selected_idx = result.selected_point(weights=[0.5, 0.3, 0.2])
```

The TOPSIS procedure normalises the objective matrix column-wise to remove scale differences (Gini is typically in [0, 0.4] for insurance; demographic parity ratio deviations are in [0, 1]; counterfactual unfairness is in [0, 1]), then applies the weights to the normalised matrix. The selected point maximises the relative closeness score — the ratio of distance from the anti-ideal to total distance.

The result is an index into the Pareto front, with the ensemble mixing weights that achieve the chosen operating point. You have a single model to deploy, plus a complete audit trail: the Pareto front data, the stakeholder weights, the TOPSIS procedure, and the resulting index. All of this can be serialised to JSON:

```python
# Serialise the full result for model governance documentation
result.to_json("pareto_result_v1.json")
```

---

## The regulatory argument

The FCA's Consumer Duty (PS22/9, effective July 2023) requires firms to demonstrate fair outcomes for retail customers. The Equality Act 2010 prohibits indirect discrimination — using neutral factors that disproportionately disadvantage protected groups without objective justification. What neither piece of legislation does is specify which mathematical definition of fairness you must satisfy.

This is deliberate, and it creates both a problem and an opportunity.

The problem: "we checked for fairness" is not a credible statement to a supervisor conducting a pricing review. They will ask how you checked, what you found, and how your model selection reflected what you found.

The opportunity: because the FCA does not prescribe a specific fairness metric, you have latitude to choose an approach that fits your product line and risk population — provided you can explain it.

TOPSIS with an explicit Pareto front provides exactly the documentation structure regulators need to audit. The argument is: we identified multiple candidate models. We computed the Pareto front across three fairness-accuracy objectives relevant to our protected characteristics (region as a proxy risk, etc.). We stated explicit stakeholder weights reflecting our interpretation of Consumer Duty obligations. We applied TOPSIS to select the model at the corresponding point on the front. Here is the output.

That is auditable. "We ran some fairness checks and the numbers looked OK" is not.

The Lipschitz metric in the library provides a fourth dimension for individual fairness — the maximum premium change per unit of feature-space distance — which can be logged separately as evidence that the model's predictions are locally stable near protected characteristic boundaries. This is not currently integrated into the NSGA-II optimisation (the `FairnessProblem` is three objectives), but `LipschitzMetric` is available as a standalone post-hoc check.

---

## Where this fits alongside existing tools

The Pareto front workflow sits after model development but before model governance sign-off. The typical sequence:

1. Fit candidate models (insurance-gam, CatBoost, GLM, orthogonal variants).
2. Run bias metrics across the candidate set using `insurance_fairness.bias_metrics` — demographic parity ratio, disparate impact, equalised odds, calibration by group.
3. If you want a defensible, auditable single model selection, run `NSGA2FairnessOptimiser` across the candidates and apply `topsis_select` with documented weights.
4. Log the `ParetoResult` JSON alongside the model documentation. Include the stakeholder weights and the TOPSIS rationale in the model governance submission.
5. At monitoring cadence, rerun the bias metrics on the deployed ensemble. If drift occurs, you have the original Pareto front as a reference for re-selection without refitting.

Step 3 adds roughly 10-20 minutes to a model review cycle for a typical personal lines portfolio. The governance benefit is substantially larger than the runtime cost.

The library is available at `pip install insurance-fairness[pareto]`. The `[pareto]` optional dependency installs `pymoo>=0.6.1`. Everything else — `polars`, `numpy` — is a standard dependency.

---

## The honest limitation

The Pareto front we show is over ensemble mixing weights, not over the full model training pipeline. NSGA-II finds the best weighted combination of the models you provide. It does not search over hyperparameter spaces, feature sets, or alternative model families. If your three candidate models all share a fundamental flaw — say, all use the same proxy variable — the Pareto front will not escape that constraint.

This is by design, not limitation. Retraining inside a fitness evaluation would make the approach computationally intractable. The right workflow is to bring meaningfully different candidate models: at minimum one accuracy-optimised model, one that has undergone explicit fairness-constrained training or orthogonalisation, and ideally one that uses a reduced feature set as a naive baseline. With three or four genuinely distinct candidates, the Pareto front reveals real structure in the trade-off space.

The other honest point: Pareto front analysis does not make the fairness problem easier. It makes the trade-off visible and the selection process auditable. The hard decisions — which protected characteristics matter for your product, what weight to give group versus individual versus counterfactual fairness, what level of accuracy loss is commercially acceptable — remain exactly as hard as before. TOPSIS does not answer those questions. It operationalises the answers you have already given.

That is as it should be. The regulator is not asking you to have solved fairness. They are asking you to have thought about it rigorously and documented what you chose.
