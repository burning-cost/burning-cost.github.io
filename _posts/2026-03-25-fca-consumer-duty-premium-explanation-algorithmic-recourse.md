---
layout: post
title: "What Can I Change to Lower My Premium? The Consumer Duty Obligation Most Pricing Teams Are Ignoring"
date: 2026-03-25
author: burning-cost
categories: [compliance, pricing, libraries, tutorials]
description: "FCA Consumer Duty PRIN 2A requires insurers to tell policyholders what they can change to get a better outcome. Most pricing teams have not built this. insurance-recourse does it in Python with FCA-audit output."
---

There is an obligation in the FCA's Consumer Duty rules that is separate from fairness monitoring and is, in our view, considerably harder to discharge in practice. It is not about whether your pricing model discriminates. It is about whether a policyholder who phones up and says "why is my premium £1,400 and what can I do about it?" gets a meaningful, accurate, documented answer.

The Consumer Duty product and service outcome (PRIN 2A.4) requires firms to support customers in making effective decisions — which the FCA has interpreted in FG22/5 to mean communicating in a way that allows customers to understand "what they can reasonably do to obtain a better outcome." For insurance pricing, that means knowing — at the individual policy level — what the customer can actually change, how much it would save them, and how much it would cost them to do it. A trained call handler reading from a generic leaflet does not satisfy this. It requires your pricing model to be able to run a constrained counterfactual search.

Most teams have not built this. The ones who have, have built it in Excel. We are going to show you how to do it properly.

---

## What the obligation actually requires

The FCA has been less specific about the recourse obligation than it has been about proxy discrimination. FG22/5 and the FCA's Consumer Duty supervisory work both focus primarily on price fairness and value monitoring. But the supporting principle in PRIN 2A.4 is clear: firms must support customers in making effective decisions, which means giving them actionable information about their options.

The key word is "actionable." It is not enough to tell a customer that younger drivers pay more because they are higher risk. That is correct and useless. The FCA is asking whether you can tell them: add Pass Plus and your premium drops to £1,150. Fit a Thatcham Cat 1 immobiliser and it drops to £960. The savings need to be denominated in pounds, not percentages of a risk loading they cannot interpret.

This creates a specific technical requirement. Your counterfactual search needs to respect mutability constraints — age cannot be changed; mileage can only decrease; a black-box telematics device takes seven days to fit. It needs to model causal propagation — a change in garaging status changes crime exposure, flood exposure, and the territory rating factor simultaneously. And the output needs to be in a format that goes into the audit file the FCA can request.

[`insurance-recourse`](/insurance-governance/) is built around exactly those constraints.

```bash
pip install insurance-recourse
```

- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — the fairness audit step that should precede the recourse generation: if a feature is a proxy, its counterfactual should not suggest customers change their demographics
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the MRM governance layer that links the recourse audit trail to the model's validation record

---

## What a naive counterfactual search gets wrong

The research-grade algorithmic recourse libraries — DiCE, alibi, CARLA — generate counterfactuals in abstract feature space. They will happily suggest that a 26-year-old can reduce their premium by becoming 19 again, or that a customer can reduce their risk group by changing their postcode to SW1A. These are genuine minimisers of the premium function. They are completely useless for Consumer Duty.

The insurance mutability problem has three layers that generic libraries ignore:

**Immutable features.** Age, gender, claims history, years of experience — these are not adjustable. A counterfactual that changes them is not a recourse option; it is a fantasy. Worse, if you surface it to a customer you face a conduct risk: you are implicitly suggesting they misrepresent their details.

**Directionally constrained features.** Annual mileage can decrease but not increase (a lower-mileage version of the customer is a genuine option; a higher-mileage version is not). NCD bonus can only improve over time, not be granted artificially. Presenting "earn two additional years of NCD immediately" as an option is not something a customer can act on today.

**Causal propagation.** Changing a customer's garaging from street to private garage is a genuine, actionable change. But it does not only affect the garaging rating factor. In most UK motor tariffs, garaging interacts with territory (a car parked in a private garage in a high-crime area has different theft exposure than one left on the street), and with vehicle security (a garaged car with a tracker and an immobiliser is different from a garaged car with nothing). A counterfactual search that treats features as independent will produce savings estimates that do not match what the customer would actually be quoted if they rang up and made the change.

---

## The setup

The library requires three inputs: an actionability constraint graph, a cost function, and your existing pricing model. Nothing is bespoke to the library's architecture — it wraps whatever model you already have.

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from insurance_recourse.constraints import ActionabilityGraph
from insurance_recourse.cost import InsuranceCostFunction
from insurance_recourse.generator import RecourseGenerator
from insurance_recourse.report import RecourseReport
```

The constraint graph for motor comes as a template:

```python
graph = ActionabilityGraph.from_template("motor")
```

The template marks age, gender, claims history, and licence tenure as immutable. It marks mileage as mutable with direction="decrease" only — you can find counterfactuals with lower mileage but not higher. Garaging, telematics, vehicle security, and Pass Plus are mutable in either direction, with causal propagation functions pre-defined.

If your model uses features that are not in the standard motor template, you add constraints explicitly:

```python
from insurance_recourse.constraints import FeatureConstraint, Mutability

graph.add_constraint(FeatureConstraint(
    name="advanced_driver_course",
    mutability=Mutability.MUTABLE,
    direction="increase",        # can acquire but not un-acquire
    effort_weight=2.0,           # weighted cost relative to other actions
    feasibility_rate=0.25,       # 25% of customers will actually do this
    allowed_values=[0, 1],
))
```

The cost function maps actions to monetary costs, timelines, and feasibility rates. Defaults come from the template but override them with your market data — if your insurer subsidises telematics to £0 upfront cost, the default £50 is wrong and will produce misleading savings calculations.

```python
cost_fn = InsuranceCostFunction.motor_defaults()
# Or override specific items:
cost_fn = InsuranceCostFunction(
    monetary_costs={
        "vehicle_security": 350.0,   # Thatcham Cat 1 installation, 2026 market rate
        "pass_plus": 175.0,
        "telematics": 0.0,           # your insurer subsidises it
    },
    time_costs_days={
        "vehicle_security": 14.0,
        "pass_plus": 90.0,           # training course, not just a product purchase
        "telematics": 3.0,
    },
    feasibility_rates={
        "vehicle_security": 0.80,
        "pass_plus": 0.35,
        "telematics": 0.70,
    },
)
```

Then wrap your pricing model:

```python
model = GradientBoostingRegressor(...)  # already fitted to your training data
gen = RecourseGenerator(model, graph, cost_fn, backend="dice", n_counterfactuals=5)
```

---

## Generating the recourse options

For a specific policyholder:

```python
factual = pd.Series({
    "age": 28,
    "annual_mileage": 14000,
    "vehicle_security": 1,       # alarm only, no immobiliser
    "pass_plus": 0,
    "garaging": 0,               # street parking
    "telematics": 0,
    "ncd_years": 3,
    "driver_age_band": 2,
    "vehicle_group": 15,
    # ... other factors
})

current_premium = float(model.predict(factual.to_frame().T)[0])
# £1,200

actions = gen.generate(
    factual,
    target_premium=current_premium * 0.85,  # find options giving 15%+ saving
    current_premium=current_premium,
    max_monetary_cost=500.0,                 # filter out anything costing more than £500
    max_days=30,                             # must be actionable within a month
)
```

The generator searches for counterfactuals that satisfy the mutability constraints, evaluates the monetary and time cost via the cost function, and ranks by saving. The output is a list of `RecourseAction` objects:

```
Rank  Action                              New premium  Saving       Cost    Days  Feasibility
1     Add Thatcham Cat 1 immobiliser      £960.00      £240 (20%)   £350    14    80%
2     Reduce annual mileage to 8k         £1,050.00    £150 (12.5%) none    1     60%
3     Install telematics black box        £1,100.00    £100 (8.3%)  £0      3     70%
```

The causal propagation matters here. When the generator considers "add immobiliser," the constraint graph propagates that to the vehicle security factor and also adjusts the garaging-crime interaction term in the features passed to the model. The £240 saving reflects what your model would actually quote for that risk profile — not just the main effect of the vehicle security factor in isolation.

---

## The FCA audit record

The output that matters for compliance is not the customer-facing HTML. It is the JSON audit record.

```python
report = RecourseReport(
    factual=factual,
    actions=actions,
    model_metadata={
        "model_version": "2026-Q1-motor-v2",
        "product": "private-motor",
        "effective_date": "2026-01-01",
    },
    policyholder_id="POL-123456",
    current_premium=current_premium,
)

html = report.to_html()     # customer-facing explanation letter
audit = report.to_dict()    # for your records system
```

The `audit` dict contains the full feature vector at the time of explanation, the ranked actions with all cost parameters, a timestamp, the model version, and a SHA-256 hash over all fields. Store the hash alongside the policyholder record in your policy admin system. If the FCA asks you to demonstrate that you gave a specific customer specific advice on a specific date, you recompute the hash from stored inputs and show that the record has not been altered.

```python
print(audit["audit_hash"])
# "a3f8c2d1e7b9f4a2c6d8e1f3a5b7c9d2..."
```

This is the piece most teams miss. Generating the counterfactuals is the technical challenge. Storing a tamper-evident record of what you told customers, and when, is the compliance challenge.

---

## Backend choices

Three counterfactual search backends are available. Which one to use depends on your pricing model.

**`focus`** is the fastest option and requires no extra dependencies. It implements the FOCUS sigmoid approximation from Lucic et al. (AAAI 2022) directly: tree split thresholds are replaced with smooth sigmoid functions, which makes the forest differentiable with respect to inputs, then gradient descent finds the minimum under constraint. It works with any sklearn `GradientBoostingRegressor`, `RandomForestRegressor`, or `DecisionTreeRegressor`. For a single policyholder on a 500-estimator GBM, this runs in under a second.

**`dice`** (default) wraps DiCE's genetic algorithm and works with any sklearn-compatible model, including CatBoost and XGBoost if you expose a sklearn-compatible `predict` method. It is slower than FOCUS but more general. Requires `pip install insurance-recourse[dice]`.

**`alibi_cfrl`** uses alibi's Counterfactual RL approach — an RL agent is trained to find counterfactuals without needing model differentiability. Useful for production CatBoost models where the sklearn interface is not clean. Requires TensorFlow or PyTorch and takes several minutes for RL training. For most teams, FOCUS or DiCE will be sufficient.

---

## The implementation path

Integrating this into a UK motor renewal process has three steps that are not all technical.

First, your constraint graph needs to reflect your actual tariff structure, not the template defaults. The propagation functions for garaging and territory interactions are placeholders — you need to implement the specific garaging × territory × vehicle_security interaction as it appears in your model. If you do not, the counterfactual premiums the generator produces will not match what your pricing system actually quotes.

Second, the cost and feasibility rates in the default cost function are our estimates of the 2026 UK market. The Thatcham Cat 1 installation cost of £350 reflects current auto-electrician rates in the South East; it will be different in your geography and different again in 18 months. These are inputs you own, not library defaults you can deploy and forget.

Third, the audit record needs to be wired into your existing document retention infrastructure. The library produces the hash; your team needs to decide where it is stored, for how long, and under what retrieval key. The FCA has not set a specific retention period for Consumer Duty recourse records, but the general COBS 9.5 requirement of five years is a reasonable floor.

None of those three steps are hard. The technical effort is small relative to, say, building a new pricing model. The reason most teams have not done this is that the obligation is not as visible as the proxy discrimination obligation, and the tooling to automate it has not existed until recently.

---

## When this matters most

The obligation is not equally salient across all your customers. It matters most in three cases.

**High-premium renewing customers.** A customer at £1,400+ who is considering shopping around has a clear interest in knowing whether there is anything they can do to justify staying. If the honest answer is "nothing plausible — your risk profile is what it is," that is also a valid output. But producing no answer at all, or a generic leaflet, is not.

**Newly declined or loaded customers.** A customer who receives a decline or a significant loading (say, >£200 above the previous year's premium excluding inflation) has a particularly strong claim to an explanation. The FCA's recent supervisory actions have focused partly on firms that load customers without giving them any recourse pathway.

**Customers in deprivation deciles 1–3.** The intersection of Consumer Duty's outcome focus and the Section 19 indirect discrimination concern lands hardest here. A customer in a high-crime, high-deprivation postcode is already paying more, and is less likely to be able to afford the actions (Thatcham Cat 1 immobiliser, private garage) that would reduce their premium. Surfacing those options honestly and letting customers make informed decisions is the right outcome from both a duty and a fairness perspective.

---

## What this is not

This is not a substitute for actuarial pricing. The counterfactuals reflect what your model would price — they inherit all of the biases, instabilities, and calibration gaps in your underlying GBM or GLM. If your vehicle security factor is mis-estimated because the training data has sparse coverage of Thatcham Cat 1 fitters, the £240 saving estimate will be wrong. Recourse explanations are only as accurate as the model they are drawn from.

It is also not a mechanism for customers to game your pricing model. The constraint graph prevents that by marking immutable features as immutable. A customer cannot present a counterfactual that requires them to pretend to be younger, or to claim NCD they do not have. The mutability constraints are not advisory; the generator will not produce an action that violates them.

What it is: a systematic, auditable way to discharge a specific FCA obligation that currently has no good tooling in the Python ecosystem. [`insurance-recourse`](https://github.com/burning-cost/insurance-recourse) is on PyPI. The FCA is watching renewal pricing closely. This seems like a good time to sort this out.

```bash
pip install insurance-recourse
```

- [Proxy Discrimination in UK Motor Pricing: Detection and Correction](/2026/03/03/your-pricing-model-might-be-discriminating/) — the fairness audit step that should precede the recourse generation: if a feature is a proxy, its counterfactual should not suggest customers change their demographics
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — the MRM governance layer that links the recourse audit trail to the model's validation record
