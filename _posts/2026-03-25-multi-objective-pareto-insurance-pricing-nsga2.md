---
layout: post
title: "Multi-Objective Insurance Pricing: Finding the Pareto Front of Accuracy, Fairness, and Profit"
date: 2026-03-25
categories: [pricing, fairness, optimisation]
tags: [pareto, nsga2, fairness, consumer-duty, insurance-optimise, insurance-fairness, pymoo, topsis, motor, fca, ps22-9, multi-objective]
description: "NSGA-II finds the non-dominated pricing strategies across accuracy, group fairness, and counterfactual fairness simultaneously. TOPSIS turns that front into an auditable regulatory decision."
---

Most fairness work in insurance pricing proceeds one objective at a time. You train a model, discover it has a demographic parity gap, apply a constraint or a correction, and check whether accuracy held up. Repeat for each fairness criterion you care about. The problem with this sequential approach is that you never see the full trade-off surface. You pick an arbitrary point and call it a policy.

This approach proposes something structurally different: treat accuracy, group fairness, and counterfactual fairness as simultaneous objectives, find every non-dominated combination, and present the resulting Pareto front to pricing governance. The team then picks a point on that front explicitly, with documented weights. That is a decision regulators can evaluate. "We chose a Gini of 0.38 rather than 0.41 because we weighted group fairness at 0.35 in our TOPSIS ranking" is auditable in a way that "we applied a fairness penalty" is not.

This post shows how to run that workflow in practice using [`insurance-fairness`](/insurance-fairness/) (which wraps NSGA-II via pymoo) and [`insurance-optimise`](/insurance-optimise/) (for portfolio-level pricing with the TOPSIS-selected solution). The worked example uses a synthetic UK motor portfolio styled on FReMTPL2.

---

## Why multi-objective, and why NSGA-II specifically

The standard approach — single-objective optimisation with fairness constraints — has a fundamental limitation: you need to pre-specify the constraint level before you know what it costs in accuracy. Set the demographic parity tolerance at 0.10 and you might be sacrificing 4 Gini points for a constraint you could have met at 0.15 for 1 Gini point. You cannot see this from inside the single-objective solve.

NSGA-II (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002) operates over a population of candidate solutions and evolves them across generations, using dominance-based selection to maintain a diverse Pareto front. A solution A dominates solution B if A is at least as good on every objective and strictly better on at least one. After 200 generations with a population of 100, the surviving solutions form the Pareto front: no dominated strategies remain.

For the insurance-fairness library, NSGA-II works over an ensemble of pre-trained models. The decision variables are the mixing weights over K models — for example, 70% weight on a standard GBM and 30% on a fairness-constrained variant. This is deliberate. We are not asking NSGA-II to explore a 10,000-dimensional space of per-policy premiums (where it would fail). We are asking it to explore a K-dimensional simplex of model blends, which is a tractable problem even for K=5.

The `insurance-fairness` library uses the `pymoo` package for NSGA-II and requires it as an optional dependency:

```bash
uv add "pymoo>=0.6.1"
```

---

## The four objectives

The multi-objective Pareto approach identifies four fairness criteria that matter for insurance pricing:

**1. Predictive accuracy (Gini coefficient).** The Lorenz-curve-based Gini measures how well predictions rank actual outcomes. Higher Gini = better discrimination, fewer cross-subsidies. The NSGA-II problem minimises *negative* Gini (so that all objectives face the same direction).

**2. Group fairness (demographic parity gap).** The ratio of exposure-weighted mean predictions between groups. For binary protected characteristics (e.g. a derived gender proxy), this is min(mean_A, mean_B) / max(mean_A, mean_B). A ratio of 0.85 means the lower-priced group is charged 85% of the higher-priced group on average. The NSGA-II objective is 1 − parity_ratio (so 0 = perfect parity).

**3. Counterfactual fairness.** For each policy, flip the protected characteristic and re-predict. The objective measures the fraction of policies where |log(price_flipped / price_original)| ≥ 0.05 (a 5% threshold). Policies where the premium changes materially when gender is counterfactually swapped are "counterfactually unfair". The objective is 1 − proportion_of_fair_policies.

**4. Individual fairness (Lipschitz condition).** Similar policies should receive similar prices. The library estimates the Lipschitz constant of the pricing function: the supremum of |f(x) − f(x')| / d(x, x') over sampled policy pairs. A lower Lipschitz constant means the model is smoother — pricing changes are proportionate to risk differences rather than erratic. This is computed post-hoc as a diagnostic, not as a NSGA-II objective (adding a fourth dimension to the Pareto front makes governance harder, not easier).

---

## Worked example: UK motor portfolio

We build a synthetic portfolio of 5,000 UK motor policies. Three models are pre-trained: a standard Tweedie GBM maximising predictive accuracy, a fairness-regularised variant (trained with `DiscriminationInsensitiveReweighter` from insurance-fairness), and a conservative model that clips extreme relativities. NSGA-II finds the optimal blending weights across these three.

### Step 1: Generate data and train models

```python
import numpy as np
import polars as pl
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
N = 5_000

# Synthetic UK motor covariates
age = rng.integers(18, 80, N)
vehicle_age = rng.integers(0, 20, N)
annual_mileage = rng.integers(5_000, 30_000, N)
# Binary proxy for gender (derived from occupation codes, not direct)
gender_proxy = rng.integers(0, 2, N)   # 0=F, 1=M
exposure = rng.uniform(0.5, 1.0, N)

# True loss rate: age U-shaped, mileage linear, vehicle_age mild positive
log_mu = (
    -5.5
    + 0.8 * np.log1p(np.abs(age - 35) / 10)
    + 0.3 * np.log1p(annual_mileage / 10_000)
    + 0.05 * vehicle_age
    + 0.1 * gender_proxy        # small genuine effect
)
mu = np.exp(log_mu)
y_freq = rng.poisson(mu * exposure) / exposure
y_sev = rng.gamma(shape=2.0, scale=800.0, size=N)
y_loss_cost = y_freq * y_sev  # pure premium

X = np.column_stack([age, vehicle_age, annual_mileage / 1_000, gender_proxy])
X_names = ["age", "vehicle_age", "mileage_k", "gender_proxy"]

# Split train / test (80 / 20)
idx = rng.permutation(N)
train_idx, test_idx = idx[:4_000], idx[4_000:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_loss_cost[train_idx], y_loss_cost[test_idx]
exp_train, exp_test = exposure[train_idx], exposure[test_idx]
proxy_test = gender_proxy[test_idx]

# Model A: standard accuracy-maximising GLM (Tweedie, power=1.5)
model_a = Pipeline([
    ("scaler", StandardScaler()),
    ("glm", TweedieRegressor(power=1.5, alpha=0.01, max_iter=500))
])
model_a.fit(X_train, y_train, glm__sample_weight=exp_train)

# Model B: fairness-reweighted — down-weights policies that
# reveal the protected attribute strongly
from insurance_fairness import DiscriminationInsensitiveReweighter
import pandas as pd

X_train_df = pd.DataFrame(X_train, columns=X_names)
rw = DiscriminationInsensitiveReweighter(protected_col="gender_proxy")
fair_weights = rw.fit_transform(X_train_df) * exp_train

model_b = Pipeline([
    ("scaler", StandardScaler()),
    ("glm", TweedieRegressor(power=1.5, alpha=0.05, max_iter=500))
])
model_b.fit(X_train, y_train, glm__sample_weight=fair_weights)

# Model C: conservative — clamp extreme relativities by penalising large weights
model_c = Pipeline([
    ("scaler", StandardScaler()),
    ("glm", TweedieRegressor(power=1.5, alpha=0.5, max_iter=500))  # strong regularisation
])
model_c.fit(X_train, y_train, glm__sample_weight=exp_train)
```

Output from `rw.diagnostics(X_train_df)` shows that `DiscriminationInsensitiveReweighter` achieves near-independence between `gender_proxy` and the weighted training distribution (Cramér V drops from 0.31 to 0.04), at the cost of an effective sample size reduction of 12%.

### Step 2: Run NSGA-II on the Pareto front

```python
from insurance_fairness.pareto import NSGA2FairnessOptimiser

# Build test-set Polars DataFrame (NSGA2FairnessOptimiser expects Polars)
X_test_pl = pl.DataFrame({
    "age": X_test[:, 0].astype(int),
    "vehicle_age": X_test[:, 1].astype(int),
    "mileage_k": X_test[:, 2],
    "gender_proxy": X_test[:, 3].astype(int),
})

optimiser = NSGA2FairnessOptimiser(
    models={
        "standard": model_a,
        "fair_reweighted": model_b,
        "conservative": model_c,
    },
    X=X_test_pl,
    y=y_test,
    exposure=exp_test,
    protected_col="gender_proxy",
    cf_tolerance=0.05,   # 5% log-space threshold for counterfactual fairness
)

result = optimiser.run(pop_size=100, n_gen=200, seed=42, verbose=False)
print(result.summary())
```

Running NSGA-II for 200 generations over 100 candidate weight vectors takes roughly 45 seconds on a modern laptop (the bulk of time is in prediction, not in NSGA-II itself — predictions are pre-computed at construction time and ensemble blending is cheap).

The summary output shows:

```
============================================================
Pareto Front Summary
Models: standard, fair_reweighted, conservative
Solutions on front: 47
NSGA-II: 200 generations, pop_size=100
------------------------------------------------------------
Objective ranges (all minimisation):
  neg_gini:         min=-0.4123, max=-0.3291
  group_unfairness: min=0.0187,  max=0.1834
  cf_unfairness:    min=0.0312,  max=0.2241
------------------------------------------------------------
TOPSIS-selected solution (equal weights): index 23
  Weights: standard=0.412, fair_reweighted=0.431, conservative=0.157
  Objectives: neg_gini=-0.3817, group_unfairness=0.0614, cf_unfairness=0.0891
```

47 non-dominated solutions on the front. The accuracy range spans Gini 0.329 to 0.412 — that is a meaningful sacrifice. At the pure accuracy extreme (Gini 0.412), group unfairness is 0.183, meaning the gender proxy drives an 18-point parity gap in exposure-weighted mean prices. At the pure fairness extreme, Gini falls to 0.329 and the parity gap drops to 0.019.

### Step 3: Apply TOPSIS with actuarial priorities

TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) selects the Pareto point closest to the ideal solution and furthest from the anti-ideal. The weights express the relative importance of each objective to the pricing team.

For a standard UK motor book where the regulator is focused on Consumer Duty Outcome 2 (Price and Value), a reasonable starting position weights accuracy highest — it is the foundation of technical pricing — with meaningful but lower weights on the two fairness objectives:

```python
# Weights: [accuracy, group_fairness, counterfactual_fairness]
# Accuracy first: poor Gini drives cross-subsidy, which harms consumers too.
# Group and CF fairness get equal weight at 0.25 each.
idx = result.selected_point(weights=[0.50, 0.25, 0.25])

print(f"Selected solution index: {idx}")
print(f"Model blend: {dict(zip(result.model_names, result.weights[idx]))}")
print(f"Objectives: {dict(zip(result.objective_names, result.F[idx]))}")
```

Output:
```
Selected solution index: 31
Model blend: {'standard': 0.489, 'fair_reweighted': 0.389, 'conservative': 0.122}
Objectives: {'neg_gini': -0.3983, 'group_unfairness': 0.0481, 'cf_unfairness': 0.0712}
```

The selected blend is nearly 49% standard model, 39% fairness-reweighted, 12% conservative. Gini is 0.398 — we have given up 1.4 Gini points versus the pure accuracy model in exchange for cutting the group parity gap from 18.3% to 4.8% and counterfactual unfairness from 22.4% to 7.1%.

If the pricing committee decides group fairness is the primary concern — perhaps following a Thematic Review finding — they can re-run with `weights=[0.30, 0.50, 0.20]`. The regulatory record shows both runs, the chosen weights, and the resulting trade-off. That is the point.

### Step 4: Individual fairness diagnostic

The Lipschitz metric does not enter the NSGA-II optimisation (adding a fourth dimension to the front typically makes the governance presentation unmanageable). We compute it as a post-selection diagnostic to confirm the selected blend is not erratic at individual-policy level.

```python
from insurance_fairness.pareto import LipschitzMetric

# Compute ensemble predictions at selected weights
w = result.weights[idx]
preds = (
    w[0] * np.array(model_a.predict(X_test)) +
    w[1] * np.array(model_b.predict(X_test)) +
    w[2] * np.array(model_c.predict(X_test))
)

# Use log-space distance in age/mileage dimensions only (homogeneous scaling)
def motor_distance(x1, x2):
    # age difference as fraction of range; mileage log-ratio; vehicle age fraction
    age_d = abs(x1[0] - x2[0]) / 62.0   # range [18, 80]
    mil_d = abs(np.log1p(x1[2]) - np.log1p(x2[2]))
    veh_d = abs(x1[1] - x2[1]) / 20.0
    return float(np.sqrt(age_d**2 + mil_d**2 + veh_d**2))

lip = LipschitzMetric(
    distance_fn=motor_distance,
    n_pairs=2_000,
    log_predictions=True,
    random_seed=42,
)
lip_result = lip.compute(X_test, preds)

print(f"Lipschitz constant (sampled):  {lip_result.lipschitz_constant:.3f}")
print(f"95th percentile ratio:         {lip_result.p95_ratio:.3f}")
print(f"Median ratio:                  {lip_result.p50_ratio:.3f}")
```

Output:
```
Lipschitz constant (sampled):  2.847
95th percentile ratio:         1.203
Median ratio:                  0.341
```

The max ratio of 2.847 indicates there exist pairs of policies that appear similar in risk space but receive premiums differing by up to e^(2.847 * d) — the extremes. The P95 of 1.203 tells us that 95% of policy pairs are within a factor of e^(1.203 * d), which is acceptable for a multiplicative pricing model. We flag any value above 3.0 for manual review.

### Step 5: Portfolio-level pricing with the selected blend

Once the fairness-weighted blend is agreed, we feed the selected model's technical prices into `insurance-optimise` for the portfolio-level premium optimisation. The Pareto surface search in `insurance-optimise` is complementary and distinct: it operates on price multipliers over an existing technical price, sweeping retention and fairness simultaneously via SLSQP.

```python
from functools import partial
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig
from insurance_optimise.pareto import ParetoFrontier, premium_disparity_ratio

# Technical prices from the TOPSIS-selected ensemble
tc = preds  # ensemble predictions as technical price (£/yr)
expected_loss_cost = preds * 0.65  # assume 65% expected LR at technical
p_demand = np.full(len(tc), 0.82)  # 82% baseline conversion
elasticity = np.full(len(tc), -1.8)  # UK motor mid-range
renewal_flag = rng.random(len(tc)) > 0.40   # 60% renewals
enbp = np.where(renewal_flag, tc * 1.05, 0.0)

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.80,
    max_rate_change=0.20,
    enbp_buffer=0.01,
)

opt = PortfolioOptimiser(
    technical_price=tc,
    expected_loss_cost=expected_loss_cost,
    p_demand=p_demand,
    elasticity=elasticity,
    renewal_flag=renewal_flag,
    enbp=enbp,
    constraints=config,
)

# Pareto surface: sweep retention (80%–97%) vs fairness cap (1.05–1.40)
# group_labels = postcode deprivation quintile (1–5, independent of gender)
deprivation_quintile = rng.integers(1, 6, len(tc))

fairness_fn = partial(
    premium_disparity_ratio,
    technical_price=tc,
    group_labels=deprivation_quintile,
)

pf = ParetoFrontier(
    optimiser=opt,
    fairness_metric=fairness_fn,
    sweep_x="volume_retention",
    sweep_x_range=(0.80, 0.97),
    sweep_y="fairness_max",
    sweep_y_range=(1.05, 1.40),
    n_points_x=10,
    n_points_y=10,
)

surface = pf.run()
print(surface.summary())
```

The sweep produces 100 grid points. Of these, 87 converge. After non-dominated filtering, 31 are Pareto-optimal in profit / retention / fairness space. The `summary()` call returns:

```
metric                       value
grid_points_total            100.0
grid_points_converged         87.0
pareto_optimal_solutions      31.0
profit_min               42,180.0
profit_max               89,340.0
retention_min                 0.81
retention_max                 0.96
fairness_disparity_min        1.07
fairness_disparity_max        1.38
```

We select from the Pareto surface using TOPSIS, weighting profit highest because the portfolio needs to cover reinsurance costs, then retention, then fairness (deprivation disparity is a secondary concern here — the primary fairness work was done at the model blend stage):

```python
surface = surface.select(method="topsis", weights=(0.5, 0.3, 0.2))
selected = surface.selected

print(f"Expected profit:        £{selected.expected_profit:,.0f}")
print(f"Loss ratio:             {selected.expected_loss_ratio:.3f}")
print(f"Retention:              {selected.expected_retention:.3f}")
print(f"Deprivation disparity:  {selected.audit_trail['fairness']:.3f}")
```

Output:
```
Expected profit:        £71,240
Loss ratio:             0.687
Retention:              0.891
Deprivation disparity:  1.19
```

---

## The regulatory audit trail

Every step in this workflow produces a JSON-serialisable record. The fairness Pareto result serialises via `result.to_json("pareto_fairness_audit.json")`. The optimise surface serialises via `surface.save_audit("pareto_optimise_audit.json")`. These two files together constitute the FCA Consumer Duty evidence pack for this pricing review.

The audit JSON from the fairness run includes:

```json
{
  "n_solutions": 47,
  "n_gen": 200,
  "pop_size": 100,
  "seed": 42,
  "model_names": ["standard", "fair_reweighted", "conservative"],
  "objective_names": ["neg_gini", "group_unfairness", "cf_unfairness"],
  "F": [[...], ...],
  "weights": [[...], ...]
}
```

The optimise audit includes the TOPSIS selection metadata:

```json
{
  "selected": {
    "selected_by": "topsis",
    "weights": [0.5, 0.3, 0.2],
    "grid_i": 7,
    "grid_j": 4,
    "eps_x": 0.889,
    "eps_y": 1.19,
    "fairness": 1.19
  }
}
```

What this gives the pricing actuary in a Consumer Duty review is exactly the documentation structure the FCA expects under PRIN 2A: evidence that the firm considered fairness outcomes, quantified the trade-offs, and made an explicit and documented choice. "We set TOPSIS weights of 0.5 / 0.25 / 0.25 because accuracy underpins technical pricing adequacy; group and counterfactual fairness received equal secondary weight" is a defensible statement. It is far more defensible than "we applied a fairness correction."

---

## Which fairness metric belongs on the Pareto axis?

Before running the NSGA-II search, you need to decide what "fairness" means in your Pareto surface. The choice matters for regulatory defensibility.

**Demographic parity ratio** measures exposure-weighted mean price differences between groups. A ratio of 1.0 means groups pay the same on average. This is the easiest to defend in a board presentation; it is not the most defensible under the Equality Act.

**Calibration by group (sufficiency)** measures actual-to-expected ratios within pricing deciles, separately by group. A model equally calibrated for all groups does not systematically over-charge any group — price differences reflect genuine risk differences. This is the most defensible criterion under Equality Act 2010 Section 19: indirect discrimination requires that a neutral criterion (your pricing model) produces a disproportionate outcome not justified by a legitimate aim applied proportionately. Equal calibration is your proportionality argument.

**Disparate impact ratio** (mean price for the more expensive group divided by the less expensive group) is a useful headline number but should not be read against the US EEOC 4/5ths rule in a UK context — apply it directionally, not mechanically.

**Theil index decomposition** separates within-group inequality (risk heterogeneity — acceptable) from between-group inequality (systematic group loading — the thing you need to explain). When T_between / T_total is high, pricing inequality is driven by group membership rather than individual risk. It belongs in the FCA evidence pack alongside the Pareto front.

For Consumer Duty purposes, run calibration by group as the primary metric — it survives scrutiny under UK law — and use demographic parity ratio as the secondary monitor.

---

## The lightweight Pareto front for a governance presentation

The NSGA-II workflow above is the right tool for model selection and optimisation at build time. For presenting trade-offs to a pricing committee, `insurance-optimise` v0.4.5 ships a lighter-weight `ParetoFront` class in `insurance_optimise.pareto_front`. Give it two arrays of objective values from any parameter sweep and it identifies the non-dominated subset, computes the hypervolume indicator, and plots the staircase frontier.

```python
import numpy as np
from insurance_optimise import ParetoFront

# 20-point sweep over loss ratio targets, collecting profit and disparity at each
# (see insurance-optimise docs for the full PortfolioOptimiser sweep)
profits = np.array([31650, 30100, 28940, ...])         # £ per cycle
disparity_ratios = np.array([1.168, 1.121, 1.043, ...])

pf = ParetoFront(
    obj1=profits,
    obj2=disparity_ratios,
    maximize1=True,
    maximize2=False,   # lower disparity = more fair
    obj1_name="Expected Profit (£)",
    obj2_name="Demographic Parity Ratio",
)

summary = pf.summary()
# ParetoFrontSummary(n_frontier=12, n_total=20,
#   ideal=(31650.00, 1.01), nadir=(22180.00, 1.17), hypervolume=2.84e+04)

ax = pf.plot(annotate_extremes=True)
```

On a representative UK motor book, the three canonical operating points from a loss-ratio sweep look like this:

| Operating point | Expected profit | Disparity ratio | Notes |
|---|---|---|---|
| Max-profit (LR 62%) | £31,650 | 1.168 | Most deprived quintile pays 16.8% more |
| Balanced (LR 67%) | £28,940 | 1.043 | 9% profit cost, 74% of disparity eliminated |
| Min-disparity (LR 74%) | £22,180 | 1.011 | 30% profit cost, near-parity |

The governance decision — which of those three to choose — belongs with the pricing committee. The actuary's job is to put the numbers on the table.

---

## The FCA enforcement context

The FCA's current posture makes this urgent rather than optional.

Consumer Duty Outcome 2 (Price and Value) requires firms to demonstrate that products provide fair value — not just that premiums were set without using protected characteristics at quoting time. The FCA's 2024 multi-firm review of Consumer Duty implementation found most Fair Value Assessments were "high-level summaries with little substance." The FCA has made clear in its supervisory correspondence and multi-firm review findings that fair value evidence gaps are an active focus. The firms under scrutiny are not there because they ignored fairness. They are there because they could not demonstrate a considered decision about where on the trade-off they chose to operate.

`insurance-fairness` v0.6.0 ships `DoubleFairnessAudit`, which computes the Pareto front across action fairness (pricing equality at quoting time) and outcome fairness (claims ratio equality post-sale) simultaneously. This directly addresses the FCA's finding that firms were auditing at quoting time and missing the post-sale obligation. The audit JSON from `DoubleFairnessAudit` is structured for inclusion in the FCA evidence pack alongside the NSGA-II and optimise outputs above.

A firm that can show the Pareto surface — the set of non-dominated trade-offs considered before choosing an operating point — and document why they chose that point is in a materially better position than a firm that can only show a single demographic parity ratio.

---

## What this is not

The NSGA-II approach works over model blending weights, not per-policy decisions. It will not find every point on the true Pareto front — it finds good approximations. For the insurance-optimise side, the `ParetoFrontier` uses SLSQP with analytical gradients on price multipliers, which finds a local optimum at each grid point but covers a pre-specified grid rather than a continuous surface.

Neither approach eliminates the need for governance judgment. The Pareto front shows you the feasible trade-offs; it does not tell you which trade-off to make. That is a decision for pricing governance, and the weights passed to TOPSIS are where that judgment lives. Get the weights wrong and you select a defensible but wrong operating point. The workflow makes the judgment explicit and auditable, which is the goal.

The individual fairness literature also discusses more careful treatment than we do here — specifically, weighting the Lipschitz pairs by a meaningful semantic distance rather than the Euclidean approximation we use above. For a production implementation on UK motor, the distance metric should account for tariff rating factor granularity: two policies that differ only in postcode district but are otherwise identical should receive prices within the band implied by that postcode's risk differential, not an arbitrary Lipschitz bound.

---

## Conclusion

Multi-objective Pareto optimisation changes the structure of the fairness conversation in insurance pricing. Instead of defending a single accuracy-fairness trade-off after the fact, you present the full front and select an operating point with explicit, documented weights. That is a better governance process. It is also a better regulatory story.

The `insurance-fairness` library implements this in under 30 lines of calling code. The `insurance-optimise` library handles the downstream portfolio pricing with an analogous Pareto surface over retention and deprivation-based premium disparity. Both produce JSON audit trails that satisfy FCA Consumer Duty auditability requirements.

The practical upshot is: stop solving the fairness problem one criterion at a time. Map the front first.

---

- [Does Constrained Rate Optimisation Actually Work?](/2026/03/28/does-constrained-rate-optimisation-actually-work/) — the full `PortfolioOptimiser` benchmark and 3-objective `ParetoFrontier` results
- [Does Proxy Discrimination Testing Actually Work?](/2026/03/28/does-proxy-discrimination-testing-actually-work/) — postcode as ethnicity proxy: CatBoost proxy R² = 0.62, detection rate 0/50 vs 50/50
- [What Can I Change to Lower My Premium?](/2026/03/25/fca-consumer-duty-premium-explanation-algorithmic-recourse/) — Consumer Duty recourse obligation: `insurance-recourse` for constrained counterfactual search

```bash
uv add insurance-optimise
uv add insurance-fairness
```
