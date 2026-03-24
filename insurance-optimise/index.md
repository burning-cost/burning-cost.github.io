---
layout: page
title: "insurance-optimise"
description: "Constrained portfolio rate optimisation with demand modelling. SLSQP, FCA ENBP constraints, Pareto frontier across profit, retention, and fairness."
permalink: /insurance-optimise/
---

[GitHub](https://github.com/burning-cost/insurance-optimise) &middot; `pip install insurance-optimise` &middot; [Full documentation](https://github.com/burning-cost/insurance-optimise#readme)

Demand-curve-aware pricing for UK personal lines. DML elasticity estimation, SLSQP optimisation with analytical Jacobians, FCA ENBP constraints. v0.4.1 adds a Pareto frontier for multi-objective optimisation across profit, retention, and fairness — making the Consumer Duty trade-off visible to pricing committees.

---

## Expected Performance

Benchmarked on synthetic UK motor PCW data (50,000 quotes, true elasticity −2.0) and a 1,000-policy renewal book. Run on Databricks serverless.

- **Demand-curve pricing +143.8% profit lift** over flat loading, even using a biased elasticity estimate — the shape of the demand curve constrains the price in the right direction
- **DML elasticity estimation:** honest result — on PCW data with narrow price variation (std log_price_ratio = 0.045), naive full-controls logistic was closer to truth in point estimate; DML provides confidence intervals and sensitivity analysis (RV = 2.1%) that the logistic cannot
- **DML fit time: 13s** on 50,000 quotes (5 folds, CatBoost nuisance models)
- **Pareto surface:** a 9% reduction in profit (£31,650 → £28,940) buys a fairness disparity reduction from 1.168 to 1.043 with improved retention — the conversation Consumer Duty requires to happen
- **Single-objective SLSQP** is blind to fairness: achieves 16.8% premium disparity across deprivation quintiles without knowing it

| Approach | Profit (£) | Retention | Fairness disparity |
|----------|-----------|-----------|-------------------|
| Flat loading | negative | low | 1.168 |
| DML-optimised | +143.8% vs flat | improved | reduced |
| Pareto balanced point | 28,940 | 0.912 | 1.043 |
| Pareto min-disparity | 22,180 | 0.951 | 1.011 |

Full benchmark methodology: `notebooks/benchmark_demand.py` and `benchmarks/benchmark_pareto.py` in the repo.

---

## Related Libraries

- [insurance-causal](https://github.com/burning-cost/insurance-causal) — DML elasticity estimation feeding into the demand model
- [insurance-fairness](https://github.com/burning-cost/insurance-fairness) — proxy discrimination auditing for the rating variables used in optimisation
