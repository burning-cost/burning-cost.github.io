# Module 7: Constrained Rate Optimisation

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

Many UK personal lines pricing teams run a version of the same Excel exercise at each renewal cycle: a spreadsheet with proposed factor adjustments, a loss ratio calculation in one tab, a volume impact estimate in another, and a sensitivity table stitched together manually. The tool is the pricing actuary's judgment, applied iteratively. The problem is not that this produces wrong answers — an experienced actuary working through the spreadsheet usually arrives at a reasonable rate action. The problem is that it is not a formally stated problem. The constraints are implicit. The optimality criterion is undefined. The trade-off between LR improvement and volume retention is explored by hand, not systematically.

This module replaces the Excel exercise with a formally stated constrained optimisation problem. We use the open-source `rate-optimiser` library (github.com/burning-cost/rate-optimiser) to solve for the factor adjustment multipliers that minimise dislocation subject to four constraints: a loss ratio target, a volume floor, per-factor movement caps, and the FCA's equivalent new business pricing (ENBP) requirement introduced under PS 21/5.

The ENBP constraint is not optional. PS 21/5 (effective January 2022) prohibits charging renewal customers more than they would be quoted as new business on the same risk. In a multiplicative tariff, this means the product of your renewal-only factors (tenure discounts, NCB-at-renewal adjustments) cannot increase the renewal premium above the new business equivalent. Getting this wrong is a regulatory breach, not a pricing judgement call. The library encodes ENBP correctly so you cannot accidentally violate it.

Beyond the single-solve result, we cover the efficient frontier: the full trade-off between LR improvement and volume retention across a range of LR targets. The frontier tells you what the pricing committee actually needs to know — not "what is the optimal rate action at 72%?" but "what does each incremental percentage point of LR improvement cost in volume terms, and where does that cost start rising faster than the improvement justifies?" The shadow price on the LR constraint answers this quantitatively.

---

## Prerequisites

- Comfortable with how a multiplicative tariff works: factor tables, base rate, the product of relativities producing a premium
- Have used the GLM or GBM outputs from Modules 2–4 — you should understand what a technical premium represents
- Basic Python and Polars; able to follow a data pipeline
- Module 1 completed: you have a Databricks workspace and know how to run a notebook
- No prior knowledge of constrained optimisation required — we explain SLSQP at the level needed to use it correctly

---

## Estimated time

3–4 hours for the tutorial plus exercises. The notebook runs in 10–15 minutes on any Databricks cluster — the optimiser converges in under 60 seconds.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial — read this first |
| `notebook.py` | Databricks notebook — full rate optimisation workflow on synthetic UK motor data |
| `exercises.md` | Four hands-on exercises with full solutions |

---

## Library

The `rate-optimiser` library is available on GitHub.

```bash
pip install rate-optimiser polars --quiet
```

Source: [github.com/burning-cost/rate-optimiser](https://github.com/burning-cost/rate-optimiser)

The library depends on SciPy (for SLSQP), NumPy, and Polars. No additional solvers required — SLSQP is bundled with SciPy.

---

## What you will be able to do after this module

- State a renewal rate optimisation problem formally: objective function, constraints, decision variables
- Build a `RateChangeOptimiser` with the four standard constraints (LR, volume, ENBP, factor bounds) and check feasibility before solving
- Interpret the ENBP constraint correctly — understand which factors are renewal-only and why misclassifying them produces regulatory exposure
- Solve for the optimal factor adjustments using SLSQP and read the convergence diagnostics
- Trace the efficient frontier across a range of LR targets and identify the knee — the point where marginal dislocation cost accelerates
- Interpret shadow prices: what does the shadow price on the LR constraint actually mean, and what does it tell the pricing committee?
- Translate factor adjustment multipliers into updated Radar-compatible factor tables with a change log suitable for pricing committee sign-off
- Analyse the distribution of individual premium changes across the portfolio and identify any segments with disproportionate impact
- Write the optimisation results and efficient frontier to Unity Catalog Delta tables

---

## Why this is worth your time

The Excel scenario analysis is not a bad tool. It is a bad process. When you manually iterate through LR targets and volume assumptions, you are doing the optimiser's job by hand — without convergence guarantees, without a systematic exploration of the feasible space, and without a documented criterion for when you stop.

The regulatory environment makes this worse. Consumer Duty (PS 22/9, effective July 2023) requires that pricing decisions can be explained and that the methodology for determining rate actions is documented. "The pricing team worked through various scenarios and agreed a compromise" is not documentation. "The optimal factor adjustments minimising sum of squared deviations from current rates, subject to LR ≤ 72%, volume retention ≥ 97%, ENBP compliance on all channels, and factor movements within ±15%" is.

The efficient frontier is the specific output that changes the pricing committee conversation. Instead of presenting a single rate action and defending it against "why not more rate?" or "why not less volume impact?", you present the frontier and the shadow prices. The committee can see the full trade-off and make an informed decision about where on that frontier to operate. That is a different quality of governance.

The ENBP constraint integration is the most practically valuable feature. PS 21/5 compliance cannot be bolted onto an Excel model after the fact. By building it into the optimiser, every solve is guaranteed to be FCA-compliant at the point of calculation.
