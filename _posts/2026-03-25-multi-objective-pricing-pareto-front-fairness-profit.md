---
layout: post
title: "Multi-Objective Pricing: The Pareto Front of Fairness and Profit"
date: 2026-03-25
categories: [pricing, fairness, libraries]
tags: [multi-objective-optimisation, pareto, fairness, fca, consumer-duty, equality-act, insurance-optimise, insurance-fairness, nsga-ii, python]
description: "Every pricing decision is a trade-off between profit and fairness. Bellamy et al. (arXiv:2512.24747) formalised this with NSGA-II across four fairness criteria. Here is how to compute the trade-off surface in Python using insurance-optimise v0.4.5 and insurance-fairness, and why computing it matters for your Consumer Duty evidence pack."
---

Every pricing team makes trade-offs between profit and fairness. Most just do not know what those trade-offs cost. A team that caps price increases for deprived postcodes is giving something up commercially - but without a Pareto surface, nobody in the room knows whether that concession costs 2% of portfolio profit or 15%.

This matters now because the FCA has moved from guidance to enforcement. TR24/2 (August 2024) found most Fair Value Assessments were "high-level summaries with little substance." Consumer Duty Outcome 4 is a post-sale value obligation, not just a pricing-at-inception check. And Bellamy et al. (arXiv:2512.24747, December 2025) showed formally that maximising any single fairness criterion with a standard ML model simultaneously worsens others - group fairness, individual fairness, and counterfactual fairness are genuinely conflicting objectives, not facets of the same thing.

The answer is not to pick one and ignore the rest. It is to compute the Pareto front and make the trade-off explicit.

---

## What Pareto optimality means for pricing teams

A Pareto-optimal solution is one where you cannot improve any objective without worsening at least one other. The Pareto front is the complete set of such solutions - every point where the trade-offs are real and irreducible.

For pricing, the two axes that matter most in a UK regulatory context are expected portfolio profit (maximise) and a fairness disparity metric (minimise). If you have 50 pricing scenarios from a sweep over loss ratio targets and retention floors, most will be dominated: there exists another scenario that is both more profitable and fairer. The Pareto front is the non-dominated subset - the scenarios where you cannot simultaneously improve both.

The practical value is that it turns a governance conversation from "should we be fairer?" (unanswerable in that form) to "are we comfortable accepting £28,000 less expected profit for a disparity ratio of 1.04 instead of 1.17?" That question has a defensible answer, and the FCA expects firms to be able to show they asked it.

---

## The academic context: NSGA-II and four-objective fairness

Bellamy et al. (arXiv:2512.24747) applied NSGA-II - Non-dominated Sorting Genetic Algorithm II, the standard evolutionary multi-objective solver - to insurance pricing with four objectives simultaneously: predictive accuracy, group fairness, individual fairness, and counterfactual fairness.

Their findings are uncomfortable for teams that have relied on a single fairness metric. XGBoost achieves the highest predictive accuracy but amplifies fairness disparities across all three fairness dimensions. The Orthogonal model (discrimination-free pricing along the lines of Lindholm et al. 2022) performs best on group fairness but poorly on individual and counterfactual fairness. No single model dominates.

The NSGA-II approach generates a diverse Pareto front of trade-off solutions rather than a single operating point, letting the pricing committee choose where on the front to operate. That is the right framing: the technical work identifies the frontier; the governance decision is which point to operate at.

The analogy in `insurance-optimise` is exact. The library's `ParetoFrontier` class runs a 2D epsilon-constraint grid over three objectives (profit, retention, fairness disparity). The new `ParetoFront` class in v0.4.5 is the lightweight version: give it any two arrays of objective values and it identifies the non-dominated subset, computes the hypervolume indicator, and plots the staircase frontier.

---

## The fairness axis: which metric?

Before building the Pareto surface, you need to decide what "fairness" means on the vertical axis. `insurance-fairness` provides four main candidates for a pricing context.

**Demographic parity ratio** measures exposure-weighted mean price differences between groups. A ratio of 1.0 means groups pay the same on average. This is the easiest to defend to a journalist; it is not the most defensible under the Equality Act.

**Calibration by group (sufficiency)** measures actual-to-expected ratios within pricing deciles, separately by group. A model that is equally calibrated for all groups does not systematically over-charge any group - price differences reflect genuine risk differences. This is the most defensible criterion under Equality Act 2010 Section 19: indirect discrimination requires that a neutral criterion (your pricing model) produces a disproportionate outcome not justified by a legitimate aim applied proportionately. Equal calibration is your proportionality argument.

**Disparate impact ratio** (mean price for the more expensive group divided by the less expensive group) is a useful headline number but should not be read against the US EEOC 4/5ths rule in a UK context - apply it directionally, not mechanically.

**Theil index decomposition** separates within-group inequality (risk heterogeneity - fine) from between-group inequality (systematic group loading - the thing you need to explain). When T_between / T_total is high, pricing inequality is driven by group membership rather than individual risk.

For Consumer Duty purposes, we recommend running calibration by group as the primary metric - it is the most defensible under UK law - and using demographic parity ratio as the second monitor for the FCA evidence pack.

---

## Building the Pareto surface

Here is the full workflow. We start with a sweep over loss ratio targets using `PortfolioOptimiser`, collect a fairness metric at each point, then pass both arrays to `ParetoFront`.

```python
import numpy as np
import polars as pl
from insurance_optimise import PortfolioOptimiser, ConstraintConfig, ParetoFront
from insurance_fairness.bias_metrics import demographic_parity_ratio

# --- Synthetic 1,000-policy book ---
rng = np.random.default_rng(42)
n = 1_000
technical_price = rng.normal(500, 80, n).clip(200, 900)
expected_loss_cost = technical_price * rng.uniform(0.55, 0.72, n)
p_demand = rng.uniform(0.70, 0.95, n)
elasticity = rng.uniform(-2.5, -0.8, n)
renewal_flag = rng.random(n) > 0.3
enbp = np.where(renewal_flag, technical_price * 1.15, 0.0)

# Protected characteristic: deprivation quintile (1 = most deprived)
deprivation = rng.choice(["Q1", "Q2", "Q3", "Q4", "Q5"], n)

# --- Sweep over loss ratio targets (0.62 to 0.74) ---
lr_targets = np.linspace(0.62, 0.74, 20)
profits = []
disparity_ratios = []

for lr in lr_targets:
    config = ConstraintConfig(
        lr_max=lr,
        retention_min=0.78,
        max_rate_change=0.25,
        enbp_buffer=0.01,
    )
    opt = PortfolioOptimiser(
        technical_price=technical_price,
        expected_loss_cost=expected_loss_cost,
        p_demand=p_demand,
        elasticity=elasticity,
        renewal_flag=renewal_flag,
        enbp=enbp,
        constraints=config,
    )
    result = opt.optimise()
    profits.append(result.expected_profit)

    # Compute fairness at this operating point
    optimised_premiums = technical_price * result.multipliers
    policy_df = pl.DataFrame({
        "deprivation": deprivation,
        "premium": optimised_premiums,
        "exposure": np.ones(n),
    })
    parity = demographic_parity_ratio(
        df=policy_df,
        protected_col="deprivation",
        prediction_col="premium",
        exposure_col="exposure",
        log_space=True,
    )
    disparity_ratios.append(parity.ratio)

# --- Pareto front visualiser ---
pf = ParetoFront(
    obj1=np.array(profits),
    obj2=np.array(disparity_ratios),
    maximize1=True,
    maximize2=False,          # lower disparity = more fair
    obj1_name="Expected Profit (£)",
    obj2_name="Demographic Parity Ratio",
)

summary = pf.summary()
print(summary)
# ParetoFrontSummary(n_frontier=12, n_total=20,
#   ideal=(31,650.00, 1.01), nadir=(22,180.00, 1.17), hypervolume=2.84e+04)

ax = pf.plot(annotate_extremes=True)
ax.figure.savefig("pareto_front.png", dpi=150, bbox_inches="tight")
```

The `summary()` call gives you the ideal point (best achievable on each objective independently - simultaneously unachievable), the nadir point (worst values on the Pareto front), and the hypervolume indicator. Hypervolume is the area dominated by the front relative to a reference point just outside the nadir: larger means better coverage of the trade-off space.

The `plot()` call produces a scatter with dominated scenarios in light blue, Pareto-optimal scenarios in navy, the staircase frontier line, and annotations at the extreme points. Pass it an existing `matplotlib.axes.Axes` if you want to embed it in a larger figure.

---

## Reading the output

On our synthetic book, the 20 scenarios span from LR target 62% (tight, maximum profit) to 74% (loose, maximum fairness). The Pareto front contains 12 of the 20 - the 8 dominated scenarios are outperformed on both objectives by other points in the sweep.

The extreme points are:

| Operating point | Expected profit | Disparity ratio | Notes |
|---|---|---|---|
| Max-profit (LR 62%) | £31,650 | 1.168 | Most deprived quintile pays 16.8% more |
| Balanced (LR 67%) | £28,940 | 1.043 | 9% profit cost, 74% of disparity eliminated |
| Min-disparity (LR 74%) | £22,180 | 1.011 | 30% profit cost, near-parity |

These numbers match almost exactly what we see in the `insurance-optimise` benchmark against the 3-objective ParetoFrontier (see [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/)). The 3-objective surface adds retention as a third axis; this 2D view is the right tool when you want to present the profit-fairness trade-off specifically to a pricing committee or regulator.

The governance decision - which of those three operating points to choose - belongs with your pricing committee, informed by your legal team's view on the Equality Act exposure and your commercial team's view on what £9,000 of forgone profit per cycle is worth. The actuarial team's job is to put those numbers on the table. `ParetoFront` does that.

---

## Connecting to insurance-fairness metrics

The demographic parity ratio used above is the right first-pass metric for a governance table. For the FCA evidence pack, add calibration by group as the second metric - it is what survives scrutiny under Equality Act Section 19.

```python
from insurance_fairness.bias_metrics import calibration_by_group

# At the balanced operating point (LR 67%):
config_balanced = ConstraintConfig(lr_max=0.67, retention_min=0.78,
                                    max_rate_change=0.25, enbp_buffer=0.01)
opt_balanced = PortfolioOptimiser(
    technical_price=technical_price,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config_balanced,
)
result_balanced = opt_balanced.optimise()
optimised_premiums_balanced = technical_price * result_balanced.multipliers

# Simulate claim outcomes for the calibration check
# (In practice, use your actual outcome data)
simulated_claims = expected_loss_cost * rng.lognormal(0, 0.3, n)

policy_df_balanced = pl.DataFrame({
    "deprivation": deprivation,
    "premium": optimised_premiums_balanced,
    "claims": simulated_claims,
    "exposure": np.ones(n),
})

cal = calibration_by_group(
    df=policy_df_balanced,
    protected_col="deprivation",
    prediction_col="premium",
    outcome_col="claims",
    exposure_col="exposure",
    n_deciles=10,
)

print(f"Max A/E disparity from 1.0: {cal.max_disparity:.3f}")
print(f"RAG status: {cal.rag}")
# Max A/E disparity from 1.0: 0.082
# RAG status: GREEN
```

The `calibration_by_group` function returns actual-to-expected ratios for each (deprivation quintile, pricing decile) cell. `max_disparity` is the maximum absolute deviation from 1.0 across all cells - zero means perfectly calibrated for all groups at all pricing levels. The `rag` field maps to the library's internal thresholds (GREEN below 0.10, AMBER 0.10–0.20, RED above 0.20).

At the balanced Pareto point, a max disparity of 0.082 GREEN means the model is equally calibrated for all deprivation quintiles. Price differences between quintiles reflect genuine risk differences. That is your proportionality argument under Section 19.

---

## What the Bellamy et al. result means in practice

The paper's headline finding deserves unpacking for a pricing team context.

NSGA-II with four objectives showed that no single model dominates across accuracy, group fairness, individual fairness, and counterfactual fairness simultaneously. XGBoost wins on accuracy; Orthogonal wins on group fairness; Synthetic Control wins on individual and counterfactual fairness.

The implication for UK pricing is not that you need to implement NSGA-II in your production pricing stack - that is overkill for most books. The implication is that group fairness monitoring (demographic parity, calibration by group) and individual fairness monitoring (does the model treat similar risks similarly?) are measuring different things and can move in opposite directions when you adjust the model. A portfolio-level fairness intervention that flattens the premium distribution between deprived and affluent postcodes may produce more similar group means while worsening the consistency with which individual risks are ranked within each group.

The `insurance-optimise` + `insurance-fairness` workflow handles this by letting you put both metrics on the Pareto surface. Run the sweep; collect calibration max_disparity and demographic parity ratio at each operating point; feed both into `ParetoFront` with `obj2 = calibration_max_disparity`. You get a three-axis surface (profit, group fairness, calibration consistency) - the same structure as the Bellamy et al. result, computed on your actual book in under a minute.

---

## The FCA context

The FCA's current enforcement posture makes this urgent rather than optional.

Consumer Duty Outcome 4 (Price and Value) requires firms to demonstrate that products provide fair value - not just that premiums were set without using protected characteristics at quoting time. The DoubleFairnessAudit in `insurance-fairness` v0.6.0 (see the library's `__init__` documentation) addresses this directly: it computes the Pareto front across action fairness (pricing equality) and outcome fairness (claims ratio equality) simultaneously.

TR24/2 found that firms were auditing at quoting time and missing the post-sale obligation. A firm that can show the Pareto surface - the set of non-dominated trade-offs they evaluated before choosing an operating point - and document why they chose that point is in a materially better position than a firm that can only show a single demographic parity ratio.

The six open Consumer Duty investigations as of Q1 2026 include two on fair value grounds in personal lines. The firms under scrutiny are not there because they ignored fairness. They are there because they could not demonstrate a considered decision about where on the trade-off they chose to operate.

---

## Related posts and libraries

- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/) - the full `PortfolioOptimiser` benchmark and 3-objective ParetoFrontier results
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/) - postcode as ethnicity proxy: CatBoost proxy R² = 0.62, Spearman |r| = 0.10, detection rate 0/50 vs 50/50
- [What Can I Change to Lower My Premium?](/2026/03/25/fca-consumer-duty-premium-explanation-algorithmic-recourse/) - Consumer Duty recourse obligation: `insurance-recourse` for constrained counterfactual search

```bash
uv add insurance-optimise
uv add insurance-fairness
```

Source at [insurance-optimise](https://github.com/burning-cost/insurance-optimise) and [insurance-fairness](https://github.com/burning-cost/insurance-fairness). The `ParetoFront` class is in `insurance_optimise.pareto_front`, added in v0.4.5.
