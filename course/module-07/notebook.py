# Databricks notebook source
# MAGIC %md
# MAGIC # Module 7: Constrained Rate Optimisation
# MAGIC ## Modern Insurance Pricing with Python and Databricks
# MAGIC
# MAGIC This notebook covers the full Module 7 workflow:
# MAGIC
# MAGIC 1. Install packages and generate a synthetic UK motor renewal portfolio
# MAGIC 2. Wrap data in the rate-optimiser data layer
# MAGIC 3. Declare the factor structure and demand model
# MAGIC 4. Check feasibility before solving
# MAGIC 5. Solve for the optimal rate action
# MAGIC 6. Translate adjustments into updated factor tables
# MAGIC 7. Trace the efficient frontier
# MAGIC 8. Cross-subsidy and Consumer Duty analysis
# MAGIC 9. Shadow price interpretation
# MAGIC 10. Write outputs to Unity Catalog
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks Runtime 14.3 LTS or later
# MAGIC - Unity Catalog enabled (Free Edition includes this)
# MAGIC
# MAGIC **Free Edition note:** All cells run on Databricks Free Edition. The optimiser
# MAGIC uses SLSQP and converges in under 60 seconds for a 5,000-policy portfolio.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Install packages

# COMMAND ----------

# MAGIC %sh uv pip install rate-optimiser polars scipy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

CATALOG = "pricing"
SCHEMA  = "motor"

# Loss ratio target for the rate review
LR_TARGET    = 0.72
VOLUME_FLOOR = 0.97   # Maximum acceptable volume loss: 3%

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Imports

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.special import expit

print(f"NumPy version:  {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"rate-optimiser version: {__import__('rate_optimiser').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Synthetic renewal portfolio
# MAGIC
# MAGIC We generate 5,000 motor renewal policies with realistic UK structure:
# MAGIC five rating factors, a price-sensitive demand model, and a current book
# MAGIC running at approximately 75% loss ratio (we need to take rate to 72%).
# MAGIC
# MAGIC In production, replace this cell with your GLM scoring output. The required
# MAGIC columns are documented in the rate-optimiser data schema.

# COMMAND ----------

rng = np.random.default_rng(2026)
N   = 5_000

# Factor relativities - what the current tariff produces per policy
age_relativity     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N, p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_relativity     = rng.choice([0.70, 0.80, 0.90, 1.00],       N, p=[0.30, 0.30, 0.25, 0.15])
vehicle_relativity = rng.choice([0.90, 1.00, 1.10, 1.30],       N, p=[0.25, 0.35, 0.25, 0.15])
region_relativity  = rng.choice([0.85, 1.00, 1.10, 1.20],       N, p=[0.20, 0.40, 0.25, 0.15])
tenure             = rng.integers(0, 10, N).astype(float)
tenure_discount    = np.ones(N)   # Renewal-only factor; currently neutral

base_rate         = 350.0
technical_premium = (
    base_rate
    * age_relativity
    * ncb_relativity
    * vehicle_relativity
    * region_relativity
    * rng.uniform(0.97, 1.03, N)
)

# Current premium: book running at approximately 75% LR
current_premium = technical_premium / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: competitive market slightly below our current rate
market_premium  = technical_premium / 0.73 * rng.uniform(0.90, 1.10, N)

renewal_flag = rng.random(N) < 0.60
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.70, 0.30]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# Demand model: logistic with log price semi-elasticity = -2.0
price_ratio = current_premium / market_premium
logit_p     = 1.0 + (-2.0) * np.log(price_ratio) + 0.05 * tenure
renewal_prob = expit(logit_p)

df = pd.DataFrame({
    "policy_id":          [f"MTR{i:06d}" for i in range(N)],
    "channel":            channel,
    "renewal_flag":       renewal_flag,
    "technical_premium":  technical_premium,
    "current_premium":    current_premium,
    "market_premium":     market_premium,
    "renewal_prob":       renewal_prob,
    "tenure":             tenure,
    "f_age":              age_relativity,
    "f_ncb":              ncb_relativity,
    "f_vehicle":          vehicle_relativity,
    "f_region":           region_relativity,
    "f_tenure_discount":  tenure_discount,
})

print(f"Portfolio:    {len(df):,} policies")
print(f"Renewals:     {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean():.1%})")
print(f"Current LR:   {df['technical_premium'].sum() / df['current_premium'].sum():.3f}  (target: {LR_TARGET})")
print(f"Channel mix:  {df.groupby('channel')['policy_id'].count().to_dict()}")
print(f"\nMean renewal probability at current rates: {renewal_prob.mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wrap the data in the PolicyData layer
# MAGIC
# MAGIC `PolicyData` validates the input columns and computes baseline statistics.
# MAGIC The `current_loss_ratio()` method gives you the unadjusted ratio at face
# MAGIC premiums - the baseline you are improving.

# COMMAND ----------

from rate_optimiser import (
    PolicyData, FactorStructure, DemandModel,
    RateChangeOptimiser, EfficientFrontier,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

data = PolicyData(df)

print(f"n_policies:  {data.n_policies:,}")
print(f"n_renewals:  {data.n_renewals:,}")
print(f"channels:    {data.channels}")
print(f"current LR:  {data.current_loss_ratio():.4f}  (target: {LR_TARGET})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Declare the factor structure
# MAGIC
# MAGIC The `factor_names` list tells the optimiser which columns contain factor
# MAGIC relativities. The decision variable is one multiplicative scalar per factor.
# MAGIC
# MAGIC `renewal_factor_names` is critical for the ENBP constraint: these are factors
# MAGIC that renewals receive but new business does not. When computing the NB-equivalent
# MAGIC premium for ENBP compliance, the library sets these factors' adjustments to 1.0.
# MAGIC
# MAGIC Include every renewal-only factor here. A false positive (treating an NB factor
# MAGIC as renewal-only) is conservative. A false negative means the ENBP constraint
# MAGIC is computing the wrong NB equivalent.

# COMMAND ----------

factor_names = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=factor_names,
    factor_values=df[factor_names],
    renewal_factor_names=["f_tenure_discount"],   # excluded from NB equivalent
)

print(f"Factor structure: {fs.n_factors} factors")
print(f"All factors:          {fs.factor_names}")
print(f"Renewal-only factors: {fs.renewal_factor_names}")
print(f"NB-equivalent factors: {[f for f in fs.factor_names if f not in fs.renewal_factor_names]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Set up the demand model
# MAGIC
# MAGIC The demand model translates a price ratio (adjusted premium / market premium)
# MAGIC into a renewal probability. A logistic model with log price semi-elasticity of
# MAGIC -2.0 is typical for a PCW-heavy UK motor portfolio. Direct-only portfolios
# MAGIC tend to be less elastic (-1.2 to -1.6).
# MAGIC
# MAGIC Validate the elasticity against observed lapse rates before using it in
# MAGIC production. A badly calibrated elasticity produces an optimised rate strategy
# MAGIC that looks good in the model and performs badly in market.

# COMMAND ----------

params = LogisticDemandParams(
    intercept=1.0,
    price_coef=-2.0,    # log-price semi-elasticity
    tenure_coef=0.05,   # tenure loyalty effect
)
demand = make_logistic_demand(params)

# Verify elasticities
sample_ratios = np.ones(10)
elasticities  = demand.elasticity_at(sample_ratios)
print(f"Price elasticity at market price: {elasticities.mean():.3f}")
print(f"(Typical UK motor PCW: -1.5 to -2.5)")

# Renewal probability curve at current portfolio mean
test_ratios = np.linspace(0.85, 1.20, 30)
test_probs  = demand.predict(test_ratios, tenure=np.zeros(30))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(test_ratios, test_probs, color="steelblue", linewidth=2)
ax.axvline(1.0, color="grey", linestyle="--", alpha=0.7, label="Market price parity")
ax.set_xlabel("Price ratio (our premium / market premium)")
ax.set_ylabel("Renewal probability")
ax.set_title("Demand model: renewal probability vs. price ratio")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Build the optimiser and check feasibility
# MAGIC
# MAGIC Before asking the solver to find an optimal solution, check whether the
# MAGIC constraints are feasible at current rates (m = all ones, no change).
# MAGIC
# MAGIC The loss ratio constraint should be violated at current rates (we are at 0.75,
# MAGIC target is 0.72) - we need to take rate. Volume and ENBP are trivially satisfied
# MAGIC at no-change. If the volume constraint is violated at no-change, something is
# MAGIC wrong with the data or demand model.

# COMMAND ----------

opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

# Add all constraints
opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(lower=0.90, upper=1.15, n_factors=fs.n_factors))

print("Feasibility at current rates (no change):")
print(opt.feasibility_report())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Solve
# MAGIC
# MAGIC SLSQP (Sequential Least Squares Programming) handles the nonlinear LR and
# MAGIC volume constraints. For 5,000 policies and 5 factors, it converges in under
# MAGIC 60 iterations.
# MAGIC
# MAGIC The objective is minimum dislocation: `sum_k (m_k - 1)^2`. A value of 1.0
# MAGIC means no change; departures are penalised symmetrically. The optimiser finds
# MAGIC the smallest rate action that satisfies all constraints.

# COMMAND ----------

result = opt.solve()
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8a. Reading the result
# MAGIC
# MAGIC - **Factor adjustments**: each number is the multiplicative scalar applied to
# MAGIC   that factor's relativities. `+3.8%` means every level of that factor gets
# MAGIC   scaled up 3.8% uniformly.
# MAGIC - **Expected LR**: should land just at or below the target.
# MAGIC - **Expected volume ratio**: should be above the volume floor.
# MAGIC - **Shadow prices**: the most important output. A shadow price of 0.15 on
# MAGIC   `loss_ratio_ub` means: relaxing the LR constraint by 1pp reduces total
# MAGIC   dislocation by 0.15. Zero shadow price means the constraint is not binding.
# MAGIC - **Objective**: sum of squared deviations from 1.0. Lower is better.

# COMMAND ----------

print("Factor adjustments:")
for factor_name, adj in result.factor_adjustments.items():
    direction = "up" if adj > 1.0 else ("down" if adj < 1.0 else "flat")
    print(f"  {factor_name:25s}: {adj:.4f}  ({(adj - 1) * 100:+.1f}%)  [{direction}]")

print(f"\nExpected LR:          {result.expected_lr:.4f}  (target: {LR_TARGET})")
print(f"Expected volume ratio: {result.expected_volume_ratio:.4f}  (floor: {VOLUME_FLOOR})")
print(f"Objective value:       {result.objective:.6f}")

print("\nShadow prices:")
for constraint_name, price in result.shadow_prices.items():
    binding = "BINDING" if price > 0.001 else "slack"
    print(f"  {constraint_name:20s}: {price:.4f}  [{binding}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Translate adjustments into factor table updates
# MAGIC
# MAGIC The `factor_adjustments` dict gives multipliers on the existing factor structure.
# MAGIC Apply them to the current factor tables to produce the new tariff relativities.
# MAGIC These are the numbers that go into your rating engine (Radar, Emblem, Akur8, etc.).

# COMMAND ----------

# Current factor tables (substitute your actual tariff tables in production)
current_tables = {
    "f_age": pd.DataFrame({
        "band":       ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "relativity": [2.00,    1.50,    1.20,    1.00,    0.92,    0.95,    1.10],
    }),
    "f_ncb": pd.DataFrame({
        "ncd_years":  [0, 1, 2, 3, 4, 5],
        "relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
    "f_vehicle": pd.DataFrame({
        "group":      ["A", "B", "C", "D"],
        "relativity": [0.90, 1.00, 1.10, 1.30],
    }),
    "f_region": pd.DataFrame({
        "region":     ["North", "Midlands", "SouthEast", "London"],
        "relativity": [0.85, 1.00, 1.10, 1.20],
    }),
}

factor_adj = result.factor_adjustments

updated_tables = {}
for factor_name, tbl in current_tables.items():
    if factor_name in factor_adj:
        m = factor_adj[factor_name]
        updated_tables[factor_name] = tbl.copy()
        updated_tables[factor_name]["relativity_new"] = tbl["relativity"] * m
        updated_tables[factor_name]["pct_change"]     = (m - 1) * 100
        print(f"\n{factor_name}:  adjustment = {m:.4f}  ({(m - 1) * 100:+.1f}%)")
        print(updated_tables[factor_name].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Trace the efficient frontier
# MAGIC
# MAGIC A single solve gives one point. The frontier requires solving across a range
# MAGIC of LR targets. This takes 1-2 minutes for 20 points on a 5,000-policy book.
# MAGIC
# MAGIC The output is the core deliverable for a pricing committee: for every
# MAGIC achievable LR target, here is the volume outcome and the marginal cost
# MAGIC of pushing one more percentage point.

# COMMAND ----------

frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=20)

print("Efficient frontier (feasible points):")
print(
    frontier_df[frontier_df["feasible"]][[
        "lr_target", "expected_lr", "expected_volume",
        "shadow_lr", "shadow_volume", "feasible",
    ]].to_string(index=False)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10a. Plot the efficient frontier

# COMMAND ----------

feasible = frontier_df[frontier_df["feasible"]].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: frontier curve (LR vs volume)
ax1 = axes[0]
ax1.plot(
    feasible["expected_lr"] * 100,
    feasible["expected_volume"] * 100,
    marker="o", color="steelblue", linewidth=2, markersize=6,
)
ax1.axvline(LR_TARGET * 100, color="red", linestyle="--", alpha=0.6,
            label=f"Target LR {LR_TARGET:.0%}")
ax1.axhline(VOLUME_FLOOR * 100, color="orange", linestyle=":", alpha=0.6,
            label=f"Volume floor {VOLUME_FLOOR:.0%}")

# Mark current position and target solve
ax1.scatter([data.current_loss_ratio() * 100], [100], color="green",
            zorder=5, s=80, label="Current (no change)")
ax1.scatter([result.expected_lr * 100], [result.expected_volume_ratio * 100],
            color="red", zorder=5, s=80, label="Optimal solve")

ax1.set_xlabel("Expected loss ratio (%)")
ax1.set_ylabel("Expected volume retention (%)")
ax1.set_title("Efficient frontier: loss ratio vs. volume retention")
ax1.invert_xaxis()   # Lower LR (better) on the right
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)
ax1.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))

# Right: shadow price curve
ax2 = axes[1]
ax2.plot(
    feasible["lr_target"] * 100,
    feasible["shadow_lr"],
    marker="o", color="darkorange", linewidth=2, markersize=6,
)

# Annotate knee (where shadow price rises sharply)
if len(feasible) > 0:
    knee_idx = feasible["shadow_lr"].diff().fillna(0).abs().argmax()
    knee     = feasible.iloc[knee_idx]
    ax2.annotate(
        f"Knee: {knee['lr_target']:.1%} LR target",
        xy=(knee["lr_target"] * 100, knee["shadow_lr"]),
        xytext=(knee["lr_target"] * 100 + 1.0, knee["shadow_lr"] + 0.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=8, color="red",
    )

ax2.axhline(0.30, color="red", linestyle="--", alpha=0.5, label="Threshold: shadow = 0.30")
ax2.set_xlabel("LR target (%)")
ax2.set_ylabel("Shadow price (marginal dislocation cost per 1pp LR)")
ax2.set_title("Shadow price on loss ratio constraint")
ax2.invert_xaxis()
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8)
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

plt.suptitle("Rate optimisation: efficient frontier and shadow prices", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 10b. Reading the frontier
# MAGIC
# MAGIC The left panel shows the set of achievable (LR, volume) outcomes. Every
# MAGIC point is Pareto-optimal: you cannot improve LR without losing volume.
# MAGIC
# MAGIC The right panel shows the shadow price curve. The shadow price on the LR
# MAGIC constraint is the marginal cost of tightening the LR target by 1pp:
# MAGIC how much additional dislocation the optimiser must accept. When the shadow
# MAGIC price exceeds your internal threshold (say 0.30), you are past the knee -
# MAGIC further LR improvement is disproportionately expensive in volume terms.
# MAGIC
# MAGIC A non-zero shadow price on volume means the volume floor is binding at that
# MAGIC point. Relaxing the volume floor would let the optimiser accept more LR
# MAGIC improvement with less factor movement.

# COMMAND ----------

# Shadow price summary table
print("Shadow price interpretation:")
print(frontier_df[frontier_df["feasible"]][[
    "lr_target", "shadow_lr", "shadow_volume"
]].assign(
    lr_target=lambda d: d["lr_target"].map("{:.1%}".format),
    shadow_lr=lambda d: d["shadow_lr"].map("{:.4f}".format),
    shadow_volume=lambda d: d["shadow_volume"].map("{:.4f}".format),
).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Cross-subsidy and Consumer Duty analysis
# MAGIC
# MAGIC The optimiser minimises total dislocation. It does not guarantee that all
# MAGIC customer segments experience equal rate changes. Before presenting to a
# MAGIC pricing committee, analyse the distribution of premium changes across the
# MAGIC portfolio.
# MAGIC
# MAGIC Under Consumer Duty (July 2023) and PS21/5, you must be able to demonstrate
# MAGIC that rate changes do not disproportionately impact protected characteristic
# MAGIC groups without actuarial justification.

# COMMAND ----------

# Compute per-policy premium change at the optimal adjustments
# The effective adjustment for each policy is the product of all factor adjustments
# applied to the relativities that policy happens to have
df_analysis = df.copy()

# Build the combined adjustment factor for each policy
combined_adj = np.ones(len(df_analysis))
for factor_name, adj in result.factor_adjustments.items():
    combined_adj *= adj

df_analysis["new_premium"] = df_analysis["current_premium"] * combined_adj
df_analysis["pct_change"]  = (
    (df_analysis["new_premium"] - df_analysis["current_premium"])
    / df_analysis["current_premium"]
)

print("Portfolio premium change distribution:")
print(df_analysis["pct_change"].describe().round(4))

# COMMAND ----------

# By age band (using f_age relativity as proxy for driver age segment)
df_analysis["age_segment"] = pd.cut(
    df_analysis["f_age"],
    bins=[0, 0.85, 0.95, 1.05, 1.40, 1.75, 3.0],
    labels=["very_low", "low", "medium", "elevated", "high", "very_high"],
    right=False,
)

by_age = df_analysis.groupby("age_segment", observed=True).agg(
    n_policies=("policy_id", "count"),
    mean_pct_change=("pct_change", "mean"),
    median_pct_change=("pct_change", "median"),
    mean_current_premium=("current_premium", "mean"),
).round(4)

print("\nPremium change by age-risk segment:")
print(by_age.to_string())
print(
    "\nNote: if mean_pct_change differs materially across segments, document "
    "the actuarial justification before submitting to compliance."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 11a. ENBP shadow price - regulatory cost
# MAGIC
# MAGIC The ENBP shadow price quantifies the regulatory cost of PS21/5 compliance:
# MAGIC how much additional dislocation the optimiser accepts because the ENBP
# MAGIC constraint is binding. A non-zero shadow price means the regulation is
# MAGIC materially constraining the rate strategy.

# COMMAND ----------

enbp_shadow = result.shadow_prices.get("enbp", 0.0)
print(f"ENBP constraint shadow price: {enbp_shadow:.4f}")

if enbp_shadow > 0.01:
    print(
        f"\nENBP constraint is binding. The optimiser is constrained by PS21/5 rules.\n"
        f"Relaxing the ENBP constraint would reduce dislocation by approximately "
        f"{enbp_shadow:.4f} per unit of LR improvement.\n"
        f"Document this in your PS21/5 impact analysis."
    )
else:
    print(
        "\nENBP constraint is not binding. The optimal rate strategy is PS21/5 "
        "compliant without material constraint."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Tighter bounds on a specific factor
# MAGIC
# MAGIC Solve again with an 8% cap on the age factor (a common commercial constraint
# MAGIC when the committee is uncomfortable with a large move on young drivers).
# MAGIC The shadow price on the age bounds constraint tells you exactly what you are
# MAGIC paying for the override.

# COMMAND ----------

# Per-factor bounds: age capped at 8% increase, others at default 15%
lower_bounds = np.array([0.90, 0.90, 0.90, 0.90, 0.90])
upper_bounds = np.array([1.08, 1.15, 1.15, 1.15, 1.15])   # age: 8% cap

opt2 = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)
opt2.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt2.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt2.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt2.add_constraint(FactorBoundsConstraint(
    lower=lower_bounds,
    upper=upper_bounds,
    n_factors=fs.n_factors,
))

result2 = opt2.solve()

print("Constrained solve (age capped at +8%):")
print(result2.summary())

print(f"\nObjective comparison:")
print(f"  Unconstrained solve: {result.objective:.6f}")
print(f"  Age-capped solve:    {result2.objective:.6f}")
print(f"  Cost of age cap:     {result2.objective - result.objective:.6f}")
print(
    f"\nThe age cap costs {result2.objective - result.objective:.4f} additional dislocation "
    f"to hit the same {LR_TARGET:.0%} LR target."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Write outputs to Unity Catalog

# COMMAND ----------

from datetime import date

RUN_DATE = str(date.today())

# Frontier DataFrame
frontier_out = frontier_df.copy()
frontier_out["run_date"]  = RUN_DATE
frontier_out["lr_target_configured"] = LR_TARGET

# Factor adjustments as a summary DataFrame
adj_records = [
    {
        "factor":       factor_name,
        "adjustment":   float(adj),
        "pct_change":   float((adj - 1) * 100),
        "run_date":     RUN_DATE,
        "lr_target":    LR_TARGET,
        "achieved_lr":  float(result.expected_lr),
        "volume_ratio": float(result.expected_volume_ratio),
    }
    for factor_name, adj in result.factor_adjustments.items()
]
adj_df = pd.DataFrame(adj_records)

try:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    spark.createDataFrame(frontier_out).write \
        .format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_frontier")
    print(f"Frontier written to {CATALOG}.{SCHEMA}.rate_frontier")

    spark.createDataFrame(adj_df).write \
        .format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_adjustments")
    print(f"Adjustments written to {CATALOG}.{SCHEMA}.rate_adjustments")

    # Premium change distribution
    spark.createDataFrame(df_analysis[["policy_id", "current_premium", "new_premium", "pct_change"]]).write \
        .format("delta").mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_change_by_policy")
    print(f"Policy-level changes written to {CATALOG}.{SCHEMA}.rate_change_by_policy")

except Exception as e:
    print(f"Delta write skipped (check catalog access): {e}")
    print("Results available as local DataFrames: adj_df, frontier_df, df_analysis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. Summary
# MAGIC
# MAGIC What we built in this notebook:
# MAGIC
# MAGIC - **Synthetic portfolio**: 5,000 UK motor renewal policies, 5 rating factors,
# MAGIC   logistic demand model with price semi-elasticity -2.0 (PCW-heavy)
# MAGIC - **Formal optimisation problem**: minimum dislocation objective, LR constraint,
# MAGIC   volume floor, ENBP (PS21/5) constraint, per-factor movement caps
# MAGIC - **SLSQP solver**: converges in under 60 iterations for this portfolio size
# MAGIC - **Factor table updates**: multipliers applied to existing tariff relativities -
# MAGIC   the exact format rating engines expect
# MAGIC - **Efficient frontier**: 20 points across LR targets 68-78%, with shadow prices
# MAGIC   at each point
# MAGIC - **Shadow price interpretation**: the marginal dislocation cost of each
# MAGIC   percentage point of LR improvement; identifies the knee of the frontier
# MAGIC - **Consumer Duty analysis**: premium change distribution by age-risk segment
# MAGIC - **Commercial override**: age cap at 8% re-solved, with quantified cost of
# MAGIC   the constraint
# MAGIC
# MAGIC The key output for a pricing committee is not the factor adjustments themselves
# MAGIC but the frontier and shadow prices: "here is every achievable (LR, volume)
# MAGIC outcome, here is the marginal cost of going further, and here is what
# MAGIC tightening the age cap costs in dislocation terms."
# MAGIC
# MAGIC **Next: Module 8 - End-to-End Pipeline**
# MAGIC Everything from Modules 1-7 in a single auditable notebook.

# COMMAND ----------

print("=" * 60)
print("Module 7 complete")
print("=" * 60)
print(f"Portfolio:            {N:,} policies")
print(f"Current LR:           {data.current_loss_ratio():.4f}")
print(f"Target LR:            {LR_TARGET:.4f}")
print(f"")
print(f"Optimal solve:")
print(f"  Achieved LR:        {result.expected_lr:.4f}")
print(f"  Volume ratio:       {result.expected_volume_ratio:.4f}  (floor: {VOLUME_FLOOR})")
print(f"  Objective:          {result.objective:.6f}")
print(f"")
print(f"Factor adjustments:")
for factor_name, adj in result.factor_adjustments.items():
    print(f"  {factor_name:25s}: {(adj - 1) * 100:+.1f}%")
print(f"")
print(f"Shadow prices:")
for cname, price in result.shadow_prices.items():
    print(f"  {cname:20s}: {price:.4f}")
print(f"")
print(f"Frontier points computed: {len(frontier_df[frontier_df['feasible']]):d}")
print(f"")
print("Next: Module 8 - End-to-End Pipeline")
