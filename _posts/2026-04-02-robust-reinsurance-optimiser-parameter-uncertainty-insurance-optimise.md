---
layout: post
title: "RobustReinsuranceOptimiser: Optimal Cession When You Don't Trust Your Parameters"
date: 2026-04-02
categories: [libraries]
tags: [reinsurance, optimisation, robust-control, ambiguity-aversion, solvency-ii, insurance-optimise, lloyds, orsa, arXiv-2603.25350]
description: "insurance-optimise v0.5.0 adds RobustReinsuranceOptimiser — closed-form and numerical optimal proportional reinsurance cession under model uncertainty, based on Boonen, Dela Vega & Garces (2026)."
author: burning-cost
---

The standard reinsurance optimisation question — what cession rate maximises shareholder value — has a clean answer when you know your loss distribution parameters. The practical problem is that you do not know them. You have estimates, with uncertainty. When those estimates are wrong, a cession schedule optimised against point estimates can be substantially suboptimal or actively harmful.

[insurance-optimise v0.5.0](https://pypi.org/project/insurance-optimise/) adds `RobustReinsuranceOptimiser`, implementing the optimal dividend-reinsurance framework from Boonen, Dela Vega, and Garces (arXiv:2603.25350, March 2026). The framework treats parameter uncertainty as a first-class input via an ambiguity parameter, produces closed-form cession schedules for symmetric portfolios, and falls back to a numerical PDE solver for the general asymmetric case.

---

## The theoretical setup

The insurer controls a surplus process X(t) across multiple lines of business. Each line i pays a net drift mu_i, has volatility sigma_i, and can cede a fraction pi_i(x) of its variance to a reinsurer at loading c_i (c_i > 1 means the reinsurer charges above actuarial fair value). When surplus reaches a barrier b*, dividends are paid out; at zero, ruin occurs.

Model uncertainty enters as an ambiguity parameter theta_i >= 0 for each line. Under the worst-case measure (Gilboa-Schmeidler robust utility), the effective drift of line i is reduced to:

```
mu_eff_i = mu_i - theta_i * sigma_i^2 * v'(x)
```

where v(x) is the value function and v'(x) is its derivative. The ambiguity parameter thus enters multiplicatively through the Arrow-Pratt risk aversion of the value function. Higher theta means the optimiser hedges harder against the possibility that the estimated drift is too high.

The paper proves that the optimal robust cession at surplus level x is:

```
pi*(x) = 1 - mu_eff / (c * sigma^2 * (-v''(x)/v'(x)))
```

clamped to [0, 1]. As surplus grows, the value function becomes less concave (curvature decreases), so pi*(x) falls — you need less reinsurance the safer you are. Near ruin (x -> 0), pi* -> 1: full cession is optimal. This is qualitatively sensible and the paper provides the theoretical proof.

---

## The API

```python
from insurance_optimise import ReinsuranceLine, RobustReinsuranceOptimiser

# Define two correlated motor lines: personal and commercial
lines = [
    ReinsuranceLine(
        mu=0.15,           # expected net profit per unit exposure
        sigma=0.30,        # volatility
        reins_loading=1.15, # reinsurer charges 15% above technical price
        ambiguity=0.05,    # theta: ambiguity parameter
        correlation=0.0,   # rho with other lines (symmetric: set to 0)
    ),
    ReinsuranceLine(
        mu=0.12,
        sigma=0.25,
        reins_loading=1.20,
        ambiguity=0.05,
        correlation=0.0,
    ),
]

opt = RobustReinsuranceOptimiser(
    lines=lines,
    delta=0.05,        # discount rate
    surplus_max=5.0,   # grid upper bound in surplus units
    n_grid=200,        # PDE grid points
    tol=1e-6,
)

result = opt.optimise()
```

The result object carries the key outputs:

```python
print(result.dividend_barrier)    # b* — surplus level triggering dividend payout
print(result.cession_schedule)    # Polars DataFrame: x1, x2, pi_1, pi_2 columns
result.plot_cession_schedule()    # heatmap of pi*(x1, x2) for each line
print(result.to_json())           # audit trail for ORSA documentation
```

For symmetric portfolios (identical mu, sigma, loading, and ambiguity across lines with rho=0), the library uses the closed-form ODE solver — scipy `solve_ivp` with brentq shooting for b*. This is exact to numerical tolerance and typically runs in milliseconds. When lines differ, the code switches to finite-difference value iteration on the 2D surplus grid.

---

## The two solvers

**Symmetric closed-form**: the 2D HJB equation reduces to a 1D ODE for the aggregate surplus v(x). `solve_ivp` integrates this ODE from x=0 outward; `brentq` finds the barrier b* where v'(b*) = 1 and v''(b*) = 0. The ambiguity adjustment is exact. The solver uses the `force_numerical=False` default and will raise an informative error if called on asymmetric inputs.

**Asymmetric PDE**: value iteration on [0, surplus_max]^2. At each iteration, the Bellman operator T[V^n](x) is applied: at each grid point, we solve the pointwise optimisation over pi to find the optimal cession, then advance V. Convergence is declared when max|V^{n+1} - V^n| < tol (default 1e-6). This typically converges in fewer than 500 iterations for standard parameterisations.

You can force the numerical solver on a symmetric input via `force_numerical=True` — useful for validating that the two paths agree:

```python
result_closed = opt.optimise(solver='symmetric_closed_form')
result_numerical = opt.optimise(solver='asymmetric_pde', force_numerical=True)

# These should agree to ~1% for standard parameterisations
```

---

## Sensitivity analysis

`RobustReinsuranceOptimiser.sensitivity()` sweeps one parameter and returns a Polars DataFrame of cession fraction vs that parameter:

```python
df = opt.sensitivity(param='ambiguity',
                     values=[0.0, 0.02, 0.05, 0.10, 0.20],
                     surplus=1.0)
# Columns: ambiguity, cession_fraction, dividend_barrier
```

The result for ambiguity is monotone increasing — more ambiguity aversion implies higher cession at any fixed surplus level. For reins_loading, the relationship is decreasing: more expensive reinsurance means you retain more risk and compensate by holding more surplus before paying dividends (b* rises).

These monotonicity properties are tested in the library's test suite and can be read as sanity checks on any calibration. If your fitted ambiguity parameter implies cession fractions your reinsurance team considers implausible, the ambiguity is probably miscalibrated — not an arbitrary number to tune until the output looks reasonable.

---

## A Lloyd's syndicate use case

A Lloyd's syndicate writes two correlated marine lines — cargo and hull. Both have mu roughly 0.12, sigma roughly 0.28, and quota share treaties with loadings near 1.18. The correlation between lines is estimated at 0.35 from the last five years of combined loss experience, but with wide uncertainty.

Prior to this release, the team used the classical de Finetti proportional reinsurance result (theta=0), which gives a fixed optimal cession independent of surplus. The new framework reveals that at low surplus (x < 0.5, say), the optimal cession is substantially higher than the de Finetti recommendation — the near-ruin premium. At high surplus (x > 3.0), cession drops below the de Finetti value.

Concretely: at ambiguity=0.05, the surplus-weighted average cession comes out 4–7 percentage points above the theta=0 baseline, depending on the correlation assumption. For a syndicate placing £200m of risk, that difference is material to treaty economics.

The `to_json()` output provides the full cession schedule, parameter inputs, and solver details in a format suitable for ORSA documentation. PRA SS3/19 requires that internal models document how parameter uncertainty is handled in capital calculations; the ambiguity framework provides a principled, theoretically-grounded answer.

---

## What this does not do

`RobustReinsuranceOptimiser` solves the continuous-time stochastic control problem under the Boonen et al. framework. It does not:

- Handle non-proportional (excess-of-loss) treaties — the proportional cession structure is baked into the HJB formulation.
- Model correlations via copulas — the current asymmetric solver uses linear correlation in the diffusion. Higher-order tail dependence is not captured.
- Optimise across reinsurer counterparty credit risk. The loading parameter captures price but not default risk.

These are not gaps we expect to fill quickly. Extending to XL structures would require a different control problem; tail copulas would substantially complicate the PDE. For proportional quota share treaty sizing with parameter uncertainty, the current implementation covers the main use case.

The library is on [PyPI](https://pypi.org/project/insurance-optimise/) and the source is on [GitHub](https://github.com/burning-cost/insurance-optimise). The paper is at [arXiv:2603.25350](https://arxiv.org/abs/2603.25350).
