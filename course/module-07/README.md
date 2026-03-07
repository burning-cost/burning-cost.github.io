# Module 7: Constrained Rate Optimisation

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## The module most courses do not have

Every other module in this course covers something you can find in a textbook or a conference paper. GLM theory, SHAP, credibility - these are established. The literature is rich.

Rate optimisation - the formal problem of finding which factors to move, by how much, to hit a target loss ratio without breaching volume constraints or regulatory rules - is not in any course we have seen. It is done in Excel by most UK pricing teams. The efficient frontier is never traced. Shadow prices on constraints are unknown. Whether a different mix of factor changes could have achieved the same loss ratio with less customer disruption: nobody computes it.

This module is about computing it.

---

## What you will build

- A constrained rate optimisation pipeline using the `rate-optimiser` library on synthetic UK motor data
- The efficient frontier of achievable (loss ratio, volume) outcomes for a rating review cycle
- Shadow price analysis: the marginal cost of tightening the LR target, quantified
- A stochastic formulation that holds the LR constraint with 95% confidence rather than just in expectation
- An FCA PS21/5 / Consumer Duty compliance check built into the optimisation constraints
- A results pack structured for presentation to a pricing committee and a commercial director

---

## Prerequisites

- Comfortable with GLM frequency-severity pricing. You should know what a rating factor relativity is and how a multiplicative tariff works.
- Have read Module 4 (SHAP Relativities) or understand how policy-level technical premiums are produced from a model
- Basic Python. You will be reading and modifying code, not writing from scratch.
- Access to a Databricks workspace. Databricks Free Edition is sufficient.

You do not need to know linear programming or optimisation theory. We explain the LP formulation from first principles and why it requires SLSQP rather than a linear solver.

---

## Estimated time

4-5 hours for the tutorial plus exercises. The optimisation itself runs in seconds for a book of 200,000 policies. The frontier trace takes 1-2 minutes. The stochastic variant is marginally slower.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - full end-to-end workflow |
| `exercises.md` | Four exercises with full worked solutions |

---

## Library

```bash
uv add rate-optimiser

# With stochastic chance-constrained formulation (requires cvxpy):
uv add "rate-optimiser[stochastic]"
```

Source: [github.com/burningcost/rate-optimiser](https://github.com/burningcost/rate-optimiser)

The library runs on standard Python. On Databricks, install in your notebook's first cell:

```python
%pip install rate-optimiser --quiet
```

---

## What you will be able to do after this module

- Formalise a rate review as a constrained optimisation problem with a defined objective function
- Explain the Markowitz analogy: why tracing the efficient frontier is the right way to think about rate strategy trade-offs
- Implement the SLSQP optimisation using `rate-optimiser` and interpret the shadow prices on each constraint
- Build the efficient frontier of (LR, volume) outcomes and identify the knee - the point where further LR improvement costs disproportionate volume
- Apply the stochastic (chance-constrained) formulation: P(LR <= target) >= 0.95
- Encode the FCA PS21/5 ENBP constraint and compute the dislocation cost of regulatory compliance
- Present optimisation results to a pricing committee: what numbers matter, what the frontier means, and why "we moved this factor because the optimiser said so" is not a complete answer
- Write results to Unity Catalog with a full audit trail

---

## What this covers that nothing else does

Commercial tools (Radar Optimiser, Earnix, Akur8) have optimisation modules. They have opaque solvers, inflexible constraint specifications, and no Python API. They do not expose shadow prices. They do not give you the full efficient frontier. They cannot be extended without vendor involvement.

The `rate-optimiser` library is an auditable, extensible alternative built on scipy. You can read every line of the solver. You can add constraints the commercial tools do not support. You can trace the frontier and compute shadow prices without asking anyone for a licence upgrade.

---

## Pricing context

Module 7 sits at the end of the modelling workflow. You have your technical premiums (from Module 2 or Module 4). You have your demand model. The question is: given what the models tell you about expected claims and price elasticity, what rate action should you take? That is an optimisation problem, not a modelling problem. This module is where the two come together.

---

## Part of the MVP bundle

This module is included in the £295 MVP bundle alongside Module 1 (Databricks for Pricing Teams), Module 2 (GLMs in Python), and Module 4 (SHAP Relativities). Individual module: £79.
