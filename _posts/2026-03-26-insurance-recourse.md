---
layout: post
title: "SHAP Tells You Why Your Premium Is High. It Doesn't Tell You What to Do About It."
date: 2026-03-26
categories: [libraries, pricing, explainability]
tags: [algorithmic-recourse, counterfactual, DiCE, alibi, FOCUS, FCA, Consumer-Duty, PRIN-2A, SHAP, GLM, GBM, CatBoost, XGBoost, insurance-recourse, python, motor, home, Wachter, Ustun, Karimi, Mothilal]
description: "SHAP attribution tells a policyholder which features drove their premium. That is not the same as telling them what they can change. insurance-recourse wraps DiCE and alibi with insurance-native constraints, causal propagation, and FCA Consumer Duty reporting to generate actionable premium-reduction recommendations."
---

Your motor pricing model predicts a £1,450 premium for a 22-year-old driving a 2020 hatchback in a high-crime postcode with no security upgrades and 12,000 miles per year. SHAP tells you, with precision, which features contributed to that number: mileage is pushing it up by £180, the postcode risk band by £95, the lack of a dashcam by £40. It is an honest and technically correct attribution.

The policyholder calls your contact centre. They want to know what they can do to lower their premium. You read them the SHAP output. They hang up, still paying £1,450.

SHAP is a backwards-looking explanation. It tells the customer which features caused their price. What it does not tell them is which of those features they can actually change, what it would cost to change them, how long it would take, and what the resulting premium would be. That is algorithmic recourse — and it is the gap that [`insurance-recourse`](https://github.com/burning-cost/insurance-recourse) fills.

```bash
uv add insurance-recourse
```

---

## The regulatory context

Consumer Duty (PRIN 2A, effective July 2023) places a positive obligation on firms to support consumer understanding of their products. Outcome 3 (PRIN 2A.5) is the relevant section: customers must be able to understand their pricing, not merely receive a disclosure document that technically satisfies the letter of the rule.

The FCA's thematic review TR24/2, published August 2024, found most fair value assessments lacked granularity at sub-group level. The individual explanation gap is real and the regulator has noticed it. FCA confirmed in December 2025 it will not issue AI-specific rules — Consumer Duty is the operative framework for algorithmic pricing decisions.

The 'why did I get this price?' question is answered by SHAP, LIME, or coefficient tables. The 'what can I do to get a better price?' question is a different question entirely. It requires recourse, and Consumer Duty in its prescriptive interpretation requires that the answer be actionable, not merely attributional.

---

## What algorithmic recourse is

The academic lineage starts with Wachter, Mittelstadt & Russell (2018, Harvard JOLT), who framed explainability as a counterfactual problem: the smallest change to the world that would produce a different model output. Their objective is:

    argmin_{x'} loss(f(x'), y') + λ · d(x, x')

where x is the current feature vector, x' is the counterfactual, y' is the desired outcome (in our case, a lower premium band), and d is a proximity metric. The λ term trades off closeness to the original against validity of the counterfactual.

Ustun, Spangher & Liu (2019, FAT*, arXiv:1809.06514) tightened this into an actionability problem. Their action set A(x) formally encodes three types of features: immutable (age, claims history — cannot be changed), mutable with direction constraints (mileage can decrease but not increase to improve the outcome; garaging can improve but not degrade), and conditionally mutable (postcode can change, but carries causal downstream effects on garaging risk, crime scores, and flood exposure).

Mothilal, Sharma & Tan (2020, FAccT, arXiv:1905.07697) added diversity via DiCE: instead of generating one closest counterfactual, generate K diverse counterfactuals via Determinantal Point Processes so the policyholder gets a menu of options rather than a single prescribed path.

The key theoretical limit comes from Karimi et al. (2020, NeurIPS, arXiv:2006.06831): without the true structural equations of the data-generating process, you cannot guarantee interventional recourse. Changing postcode in isolation — as a Wachter-style counterfactual would — is not the same as moving house, which also changes garaging location, local crime rates, and flood zone. The causal structure matters, and no existing library encodes it for insurance.

---

## The insurance gaps that existing tools do not fill

DiCE v0.12 (released July 2025, actively maintained) and alibi v0.9.6 (November 2025) are excellent libraries. CARLA, once the canonical recourse framework, has been dead since May 2022. DiCE is our default for GLMs and sklearn pipelines. alibi's CFRL (counterfactual RL) backend handles CatBoost and XGBoost as genuine black-boxes.

What neither provides:

**Causal constraint graphs for insurance covariates.** DiCE's constraint handling assumes feature independence in satisfaction. It will happily suggest 'change postcode from E1 to SW1' without propagating the downstream causal consequences to garaging risk, crime score, and flood exposure. For a UK motor pricing model, those propagations are significant.

**Direction constraints that reflect reality.** Claims history is immutable — you cannot undo a claim. Mileage can decrease but its decrease direction in the feature space happens to be the premium-reducing direction, which is not always true. A Pass Plus qualification can be acquired but not removed. None of this is in DiCE's out-of-the-box constraint encoding.

**Premium-denominated output.** The standard recourse output is a counterfactual feature vector. A policyholder does not want a feature vector. They want: 'Install a Thatcham Category 1 alarm (£195, takes 2–3 days) and reduce your declared mileage by 3,000 miles per year, and your premium drops by £187.'

**FCA Consumer Duty reporting.** There is no existing library that generates the tamper-evident audit JSON and customer-facing HTML that a Consumer Duty examination would want to see.

---

## Encoding constraints

`insurance-recourse` starts with an `ActionabilityGraph` — a DAG encoding the mutability structure of your covariates.

```python
from insurance_recourse.constraints import ActionabilityGraph, FeatureConstraint, Mutability

# Pre-built UK motor template
graph = ActionabilityGraph.from_template("motor_uk")

# The template encodes these decisions out of the box:
# - annual_mileage: MUTABLE, direction='decrease'
# - ncd_years: IMMUTABLE (cannot retroactively earn no-claims)
# - age: IMMUTABLE
# - vehicle_security: MUTABLE, direction='increase', allowed_values=[0,1,2,3]
# - pass_plus: MUTABLE, direction='increase', allowed_values=[0,1]
# - postcode_risk: CONDITIONALLY_MUTABLE with causal children [crime_rate, garaging]

# Custom constraint for a non-standard feature
graph.add_constraint(
    FeatureConstraint(
        name="telematics",
        mutability=Mutability.MUTABLE,
        direction="increase",
        allowed_values=[0, 1],
        effort_weight=2.0,       # higher effort — requires hardware installation
        feasibility_rate=0.65,   # 65% of policyholders can realistically do this
    )
)
```

Causal propagation uses topological sort. When a conditionally mutable feature changes — say, `postcode_risk` — the graph walks downstream dependents in dependency order and updates them via registered propagation functions before re-evaluating the model.

---

## Generating recourse

```python
import pandas as pd
from insurance_recourse.constraints import ActionabilityGraph
from insurance_recourse.cost import InsuranceCostFunction
from insurance_recourse.generator import RecourseGenerator

# Pre-built UK motor constraint graph and monetary cost defaults
graph = ActionabilityGraph.from_template("motor_uk")
cost_fn = InsuranceCostFunction.motor_defaults()

gen = RecourseGenerator(
    model=your_catboost_model,     # any sklearn .predict() or callable
    actionability_graph=graph,
    cost_function=cost_fn,
    backend="focus",               # sigmoid approx for tree ensembles
    n_counterfactuals=5,
)

actions = gen.generate(
    factual=policyholder_features,   # pd.Series
    target_premium=900.0,            # desired premium in GBP
    current_premium=1450.0,
    max_monetary_cost=500.0,         # filter: discard actions costing >£500
    max_days=30,                     # filter: discard actions taking >30 days
)

for action in actions:
    print(f"{action.description}")
    print(f"  Premium reduction: £{action.premium_reduction:.0f} "
          f"({action.premium_reduction_pct:.1f}%)")
    print(f"  Cost: £{action.effort.monetary_cost:.0f}, "
          f"{action.effort.time_days:.0f} days")
    print(f"  Feasibility: {action.effort.feasibility_probability:.0%} of "
          f"similar policyholders can do this")
```

Output for a simulated run (CatBoost trained on synthetic UK motor data):

```
Reduce annual mileage from 12,000 to 9,000 miles
  Premium reduction: £142 (9.8%)
  Cost: £0, 0 days
  Feasibility: 72% of similar policyholders can do this

Upgrade vehicle security from level 0 to level 2
  Premium reduction: £118 (8.1%)
  Cost: £195, 3 days
  Feasibility: 81% of similar policyholders can do this

Complete a Pass Plus advanced driving course
  Premium reduction: £93 (6.4%)
  Cost: £150, 14 days
  Feasibility: 89% of similar policyholders can do this
```

The policyholder now has a menu of concrete, costed, ordered actions. That is what Consumer Duty in its practical sense requires.

---

## The FOCUS backend for tree ensembles

DiCE's gradient-free backends (random sampling, KD-Tree, genetic algorithm) work for any model, but they search by perturbation — they find counterfactuals by trying changes, not by reasoning about the model's structure. For a CatBoost model with hundreds of trees, this is slow and the counterfactuals are often non-minimal.

The FOCUS algorithm (AAAI 2022) solves tree ensemble differentiability by replacing tree split thresholds with sigmoid approximations: each hard `x_{f_j} < θ_j` becomes σ(α(θ_j - x_{f_j})) where α controls the approximation sharpness. The tree ensemble becomes differentiable and gradient descent can find the minimal counterfactual directly.

We implemented FOCUS internally — there is no pip package for it. The `backend="focus"` flag in `RecourseGenerator` activates our implementation. It converges faster than DiCE genetic on CatBoost/XGBoost and produces counterfactuals that are measurably closer to the factual in L1 norm.

The `backend="alibi_cfrl"` option uses alibi's reinforcement-learning-based counterfactual search — genuinely model-agnostic, slower, but useful for complex heterogeneous models where gradient approximation is unreliable.

---

## FCA Consumer Duty reporting

Every recourse generation call should produce an audit record. `RecourseReport` wraps the actions with model metadata and a SHA-256 audit hash:

```python
from insurance_recourse.report import RecourseReport

report = RecourseReport(
    factual=policyholder_features,
    actions=actions,
    model_metadata={
        "model_version": "2025-Q4-motor-v3",
        "product": "motor",
        "effective_date": "2026-01-01",
    },
    policyholder_id="POL-123456",
    current_premium=1450.0,
)

# JSON audit record for regulatory purposes
audit = report.to_dict()
# audit["audit_hash"] is SHA-256 of model metadata + factual features + all actions
# tamper-evident: any post-hoc modification invalidates the hash

# Customer-facing HTML explanation page
html = report.to_html()
```

The audit hash covers model metadata, the factual feature vector, and all recourse action details in canonical JSON form. Store the hash at report generation time and you can verify later that neither the model version attribution nor the recourse recommendations were altered. That is the kind of tamper-evidence a Consumer Duty audit examination should be able to request and receive.

---

## What we deliberately did not build

We considered adding a SHAP-to-recourse pipeline that automatically selects which features to target based on their SHAP values. We decided against it.

SHAP attribution and recourse actionability are orthogonal. The feature with the largest SHAP contribution to a premium may be immutable (age) or conditionally mutable with prohibitive causal effects (postcode). Automating the link between attribution and recourse conflates two distinct questions and produces misleading results. The pricing actuary should make the decision about which features to include in the actionability graph. The library encodes that decision explicitly and auditably.

We also did not add premium floor constraints — recourse that pushes a policyholder below their technical rate to win them over. That is a fair value problem, not a recourse problem.

---

## Library: 74th. Score: 15/20.

Four modules, approximately 950 lines, 177 tests. The BUILD score of 15/20 reflects one genuine weakness: the causal propagation functions in the `motor_uk` and `home_uk` templates use heuristic placeholders (for instance, `postcode_risk -> crime_rate` via linear interpolation) rather than real UK geographic data. Firms using this in production will need to replace those propagation functions with their own geospatial lookups. The library provides the structure, not the data — but it means the templates are illustrative, not plug-and-play.

The other gap is no native handling of Tweedie GLMs: the `model` argument accepts any sklearn-compatible `.predict()` callable, but there is no built-in understanding of the multiplicative log-linear structure. If you have a Poisson/Gamma GLM, you can pass a wrapper that calls `predict()` and returns a premium, but the analytical gradient opportunity — the premium-ratio interpretation of counterfactual changes in log-linear space — is left on the table.

Both limitations are documented. Neither makes the library unsuitable for its primary purpose.

**[insurance-recourse on GitHub](https://github.com/burning-cost/insurance-recourse)** — MIT-licensed, PyPI, 177 tests. For the question your policyholders are actually asking.

---

**Related articles from Burning Cost:**
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/)
- [Regression Discontinuity Design for Insurance Pricing](/2026/03/11/insurance-rdd/)
- [Insurance Bunching: Detecting Threshold Gaming in Declared Variables](/2026/03/11/insurance-bunching/)
