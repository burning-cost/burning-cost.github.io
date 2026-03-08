# Module 7: Constrained Rate Optimisation

In Module 6 you learned how to blend sparse experience with portfolio priors using Bühlmann-Straub credibility and Bayesian hierarchical models. You can now produce reliable factor relativities even for thin cells. This module is about what you do with those relativities next: deciding how much to move the rates, which factors to move, and how to prove to the FCA that the decision was principled.

Every pricing actuary has a version of this problem. The book is running above target. The commercial director wants as little volume disruption as possible. The underwriting committee has set factor movement caps. The FCA expects ENBP compliance for every renewal policy. There are four things you need to satisfy simultaneously, and the spreadsheet approach treats them sequentially, by judgment, in a way that is impossible to audit.

This module replaces that process with a formally stated optimisation problem. The solution is a vector of factor adjustments that minimises customer disruption subject to simultaneously satisfying the loss ratio target, the volume floor, the FCA's fair pricing rules, and the underwriting movement caps. You can solve it in seconds, plot the full trade-off for the pricing committee, and export the result to a Delta table with a complete audit trail.

By the end of this module you will have:

- Understood the constrained optimisation problem in plain English before touching any maths
- Built a demand model that knows how renewal probability responds to price changes
- Set up and solved the four-constraint rate optimisation problem using SLSQP
- Interpreted shadow prices and understood what they tell a commercial director
- Traced the efficient frontier and identified the pricing committee's decision point
- Verified ENBP compliance per-policy, not just at the aggregate level
- Fixed the cross-subsidy analysis to show what it actually shows
- Extended the optimiser to handle stochastic loss ratio targets using chance constraints
- Documented the limitations of this approach honestly

---

## Part 1: The problem in plain English

### What a pricing review actually involves

You run UK motor insurance. The book has 80,000 policies in force. At the end of Q1 you look at the numbers:

- Current loss ratio: 75.2%
- Target loss ratio: 72.0%
- Volume versus plan: -2.8%

The gap is 3.2 percentage points. You need to close it by adjusting rates. But you cannot simply add 3.2pp to every premium: some customers will lapse, which reduces the denominator (premium) and can make the LR worse if you lose the wrong ones. The amount of rate you need to charge is not the same as the gap you need to close.

Your tariff has five rating factors: age, no-claims discount, vehicle group, region, and a tenure discount for renewals. Each factor has a table of relativities — the multipliers applied to the base rate for each level of the factor. The question is: by how much should you scale each factor table?

The spreadsheet approach is to try combinations. Increase age by 4%, NCB by 3%, vehicle by 2%, region by 3%, leave tenure flat. Calculate the expected LR. Volume is now 97.5% of current. ENBP is satisfied (you check the maximum renewal/NB ratio manually for five example policies). The commercial director accepts it. It goes to pricing committee.

This is not a bad outcome. But it has three structural problems.

**Problem 1: You explored a small part of the space.** There are infinitely many combinations of five factor adjustments that might achieve the LR target. You found one by starting near zero and adjusting by judgment. You cannot know whether it is the best one — the one with the smallest customer disruption — because you did not search the space systematically.

**Problem 2: You cannot quantify trade-offs.** The commercial director asks: "What if we accepted 98% volume retention instead of 97.5%? What would the LR be?" In the spreadsheet, you run another scenario. But the frontier — the full curve of all achievable (LR, volume) combinations — is invisible. You are showing points, not the curve.

**Problem 3: You cannot audit it.** The FCA under Consumer Duty (PS 22/9, effective July 2023) can ask you to show your methodology for every rate decision. "We tried several combinations and chose one that looked sensible" does not satisfy a section 166 request. A formally stated optimisation problem with documented constraints and a reproducible solver does.

### What the optimiser does

The `rate-optimiser` library takes your data, your demand model, and your constraints, and finds the factor adjustment vector that:

1. Achieves the LR target
2. Keeps volume above the floor
3. Satisfies ENBP for every renewal policy
4. Keeps each factor within the approved movement caps
5. Does all of the above with the smallest total disruption to customer premiums

"Smallest total disruption" is formalised as the minimum-dislocation objective, which we explain in Part 4. The solver is SLSQP (Sequential Least Squares Programming) from SciPy. The output is the factor adjustment vector plus shadow prices, an efficient frontier, and an audit trail.

---

## Part 2: Setting up your Databricks notebook

### Creating the notebook

If Databricks is open from Module 6, create a new notebook. In the left sidebar, click **Workspace**. Navigate to your user folder under `/Users/your.email@company.com/`. Click the **+** button and choose **Notebook**. Name it `module-07-rate-optimisation`. Leave the language as Python. Click **Create**.

The notebook opens with one empty cell. At the top of the notebook you will see a cluster selector. If it shows "Detached," click it and choose your cluster. Wait for the cluster name to show with a green circle. Do not run any cells until the cluster is connected.

If the cluster is not in the list, it may have auto-terminated. Go to **Compute** in the left sidebar, find your cluster, and click **Start**. It takes 3-5 minutes. Come back to the notebook once the cluster shows "Running."

### Installing the libraries

In the first cell, type this and run it by pressing **Shift+Enter**:

```python
%pip install rate-optimiser catboost polars scipy --quiet
dbutils.library.restartPython()
```

Wait for the installation output to complete, then for the restart message. This takes 60-90 seconds. After the restart, any variables from before are gone — that is expected.

For a local development environment instead of Databricks:

```bash
uv add rate-optimiser catboost polars scipy
```

### Confirming the imports work

In a new cell, paste this and run it:

```python
import numpy as np
import polars as pl
import scipy
from catboost import CatBoostClassifier
from rate_optimiser import (
    PolicyData, FactorStructure, RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
    EfficientFrontier,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

print(f"NumPy:   {np.__version__}")
print(f"Polars:  {pl.__version__}")
print(f"SciPy:   {scipy.__version__}")
print("rate-optimiser: imported OK")
```

**What you should see:**

```
NumPy:   1.26.x
Polars:  0.20.x
SciPy:   1.x.x
rate-optimiser: imported OK
```

If you see `ModuleNotFoundError: No module named 'rate_optimiser'`, the install did not complete. Run the `%pip install` cell again and restart again.

---

## Part 3: What we are optimising and why

Before writing any optimisation code, you need to understand the three elements of any constrained optimisation problem: the decision variables, the objective function, and the constraints. We explain each in plain English before introducing the maths.

### The decision variables: factor multipliers

Your rating system produces a premium by multiplying a base rate by a series of factors:

```
premium = base_rate x age_factor x ncb_factor x vehicle_factor x region_factor x tenure_discount
```

Each factor is a table. The age factor, for example, looks like:

| Age band | Relativity |
|----------|-----------|
| 17-21    | 2.00      |
| 22-24    | 1.50      |
| 25-29    | 1.20      |
| 30-39    | 1.00      |
| 40-54    | 0.92      |
| 55-69    | 0.95      |
| 70+      | 1.10      |

A 19-year-old gets a factor of 2.00. A 45-year-old gets 0.92. These relativities capture the shape of the risk — how much more expensive young drivers are relative to middle-aged drivers.

In a **uniform rate action**, the shape of the table does not change. Instead, you scale the entire table by a single multiplier. If the age factor adjustment is 1.038, then every level of the age table increases by 3.8%:

| Age band | Old relativity | New relativity | Change |
|----------|---------------|----------------|--------|
| 17-21    | 2.00          | 2.076          | +3.8%  |
| 22-24    | 1.50          | 1.557          | +3.8%  |
| 25-29    | 1.20          | 1.246          | +3.8%  |

The decision variables in the optimisation are these multipliers: one per factor. We write them as a vector **m** = (m\_age, m\_ncb, m\_vehicle, m\_region, m\_tenure).

For five factors, we are solving for five numbers. That is a small optimisation problem — one reason it runs in under a second on a laptop.

### The objective function: minimum dislocation

Given that there are many vectors **m** that could achieve the LR target, which one should you pick?

The principle is minimum dislocation: choose the rate action that achieves the target while changing premiums as little as possible. This is both a commercial principle (unhappy customers lapse) and a Consumer Duty principle (disproportionate increases on specific segments need justification).

Mathematically, the objective function is the sum of squared deviations of the multipliers from 1.0:

```
minimise:  sum_k (m_k - 1)^2
```

where k runs over the five factors. This is called the minimum-dislocation objective.

Why squared? Two reasons.

First, squaring makes large deviations much more costly than small ones. A 10% increase on one factor contributes (0.10)^2 = 0.01 to the objective. Two 5% increases on two factors contribute 2 x (0.05)^2 = 0.005. The solver will prefer spreading the rate increase across factors rather than concentrating it on one, because concentrating produces a much larger objective value.

Second, squaring makes the objective function convex. This is a mathematical property that guarantees there is exactly one minimum within the feasible set — there is no risk of the solver finding a local minimum that is not the global minimum. For pricing, this is important: the solution is unique and reproducible.

If you used absolute deviations instead of squared deviations, the objective would be convex but not strictly convex, and there could be multiple solutions with the same objective value. The solver might return different solutions on different runs. With squared deviations, the solution is always unique.

### The constraints: what must be satisfied

The constraints are the conditions the factor vector **m** must satisfy. There are four.

**Constraint 1: Loss ratio target.** The expected portfolio loss ratio at the new rates must be at or below the target:

```
E[LR(m)] <= LR_target
```

The expected LR is not simply the current LR divided by the average rate change, because some customers will lapse when you raise rates. Lapsed customers do not contribute expected losses or expected premium to the renewed book. The LR calculation must account for this through the demand model.

**Constraint 2: Volume floor.** The expected volume retained at the new rates must be at or above the floor:

```
E[volume(m)] >= volume_floor
```

Volume is measured as expected retained premium at new rates divided by expected retained premium at current rates. A 97% floor means you are willing to accept at most 3% volume loss from rate-driven lapses.

**Constraint 3: ENBP (PS 21/5).** For every renewal policy on every relevant channel, the adjusted renewal premium must not exceed the new business equivalent premium. The FCA's PS 21/5, effective January 2022, requires this at the individual policy level — not just on average.

**Constraint 4: Factor movement caps.** Each adjustment m\_k must lie within the range approved by the underwriting committee. If the caps are 90% to 115%, then:

```
0.90 <= m_k <= 1.15  for all k
```

These four constraints define the feasible set: the region of the (m\_1, m\_2, ..., m\_F) space where all constraints are simultaneously satisfied. The solver finds the point in this region with the smallest objective value.

---

## Part 4: The demand model

### Why you need one

The LR constraint and the volume constraint both depend on how many customers renew at the new rates. Without a demand model, you have to assume either:

- Everyone renews regardless of price (unrealistic: volume does not change)
- Lapse rates are fixed regardless of price (unrealistic: rates do not affect who stays)

Neither is right. A demand model tells the optimiser: if you raise this customer's premium by 5%, the probability they renew changes from, say, 68% to 65%. The optimiser accounts for this when computing expected LR and expected volume.

Without a demand model, the volume constraint is not meaningful (it is always satisfied unless you have a separate assumption about lapses), and the LR constraint is overoptimistic (it ignores the fact that rate increases cause lapses, which change the book composition).

### The logistic demand model

The `rate-optimiser` library uses a logistic demand model. This is the workhorse specification for renewal probability in UK personal lines. The renewal probability for policy i at a price ratio p\_i (new premium divided by market premium) is:

```
renewal_prob_i = sigmoid(intercept + price_coef * log(p_i) + tenure_coef * tenure_i)
```

where `sigmoid(x) = 1 / (1 + exp(-x))` is the logistic function. The inputs are:

- `p_i = new_premium_i / market_premium_i`: how expensive this policy is relative to what the customer could get elsewhere
- `log(p_i)`: the log of the price ratio. Using log makes the model multiplicative: a 10% increase from 100% to 110% of market has the same demand effect as a 10% increase from 90% to 99% of market
- `tenure_i`: years the customer has been with the insurer. Longer-tenured customers are stickier — they are less price-sensitive

The key parameter is `price_coef`. It is negative (higher price, lower renewal probability) and is called the **log-price semi-elasticity**. A value of -2.0 means: a 1% increase in log price above market reduces the log-odds of renewal by 2 percentage points.

To understand what that means in practice: if a customer currently has a 60% renewal probability (logit of about 0.41), and we raise their price by 1% above market, the new logit is 0.41 + (-2.0 x 0.01) = 0.41 - 0.02 = 0.39, giving a new renewal probability of sigmoid(0.39) = 59.6%. That is a 0.4 percentage point reduction in renewal probability for a 1% price increase.

For UK motor, the relevant benchmarks from market research and published lapse analyses (e.g., Bain & Company UK motor loyalty studies 2018-2022) are:

- **PCW (price comparison website) channel**: price semi-elasticity typically -1.5 to -3.0. PCW customers have already demonstrated they will shop around. They are the most price-sensitive segment.
- **Direct channel**: -0.5 to -1.5. Direct customers have already chosen not to use a PCW. A modest rate increase is less likely to trigger a lapse.

These are starting points. You must calibrate the demand model against your own observed lapse data before using it in the optimiser.

### What miscalibration looks like

If you use a PCW elasticity of -2.5 when your actual elasticity is -1.2, the optimiser will believe you have far less pricing power than you do. It will think that even a small rate increase causes a large volume loss, and it will constrain the rate action more than necessary. The frontier will show infeasibility at targets that are actually achievable.

If you use -0.8 when your actual elasticity is -2.0, the optimiser will overestimate pricing power. It will produce a frontier that claims 72% LR is achievable at 97% volume retention, but in practice the actual lapses will be much higher and the achieved LR will be worse than the model predicted.

**The demand model must be calibrated before you run the optimiser.** Exercise 1 includes a calibration check. In Part 11 we address the limitations of the logistic specification.

---

## Part 5: Generating the synthetic portfolio (Polars, not pandas)

We work with a synthetic motor renewal portfolio of 5,000 policies running at approximately 75% LR against a 72% target. In production, you would read from a Unity Catalog table. We use synthetic data here so that we know the ground truth and can verify the optimiser output.

All data manipulation uses Polars. The `rate-optimiser` library uses pandas internally, so we convert at the library boundary. This follows the Polars mandate: the tutorial code uses Polars; pandas appears only where the library requires it.

Add a markdown cell:

```python
%md
## Part 5: Synthetic motor portfolio
```

Then create a new cell and paste this:

```python
import numpy as np
import polars as pl
from scipy.special import expit

rng = np.random.default_rng(seed=42)
N = 5_000

# ---------- Rating factor relativities ----------
# Each policy is drawn from a distribution of factor levels.
# These are the underlying relativities for each level.
age_rel     = rng.choice([0.80, 1.00, 1.20, 1.50, 2.00], N,
                          p=[0.15, 0.30, 0.30, 0.15, 0.10])
ncb_rel     = rng.choice([0.70, 0.80, 0.90, 1.00],       N,
                          p=[0.30, 0.30, 0.25, 0.15])
vehicle_rel = rng.choice([0.90, 1.00, 1.10, 1.30],       N,
                          p=[0.25, 0.35, 0.25, 0.15])
region_rel  = rng.choice([0.85, 1.00, 1.10, 1.20],       N,
                          p=[0.20, 0.40, 0.25, 0.15])

# Tenure: 0-9 years with the insurer
tenure = rng.integers(0, 10, N).astype(float)

# Tenure discount is renewal-only and currently neutral (1.0 for all)
tenure_disc = np.ones(N)

# ---------- Premiums ----------
base_rate = 350.0

# Technical premium: what the risk actually costs
technical_premium = (
    base_rate
    * age_rel * ncb_rel * vehicle_rel * region_rel
    * rng.uniform(0.97, 1.03, N)   # small residual noise
)

# Book running at 75% LR: current premium = technical / 0.75 (with spread)
current_premium = technical_premium / 0.75 * rng.uniform(0.96, 1.04, N)

# Market premium: what the customer could get elsewhere (competitive benchmark)
market_premium  = technical_premium / 0.73 * rng.uniform(0.90, 1.10, N)

# ---------- Demand model for renewal probability ----------
# Log price ratio: positive means we are above market, negative means below
log_price_ratio = np.log(current_premium / market_premium)

# Logistic demand: intercept=1.2, price_coef=-2.0, tenure_coef=0.05
logit_renew = 1.2 + (-2.0) * log_price_ratio + 0.05 * tenure
renewal_prob = expit(logit_renew)

# Indicator: is this a renewal policy?
renewal_flag = rng.random(N) < 0.65  # 65% of portfolio is renewals

# Channel: PCW or direct
channel = np.where(
    renewal_flag,
    rng.choice(["PCW", "direct"], N, p=[0.65, 0.35]),
    rng.choice(["PCW", "direct"], N, p=[0.60, 0.40]),
)

# ---------- Build the Polars DataFrame ----------
df = pl.DataFrame({
    "policy_id":         [f"MTR{i:07d}" for i in range(N)],
    "channel":           channel.tolist(),
    "renewal_flag":      renewal_flag.tolist(),
    "tenure":            tenure.tolist(),
    "technical_premium": technical_premium.tolist(),
    "current_premium":   current_premium.tolist(),
    "market_premium":    market_premium.tolist(),
    "renewal_prob":      renewal_prob.tolist(),
    "f_age":             age_rel.tolist(),
    "f_ncb":             ncb_rel.tolist(),
    "f_vehicle":         vehicle_rel.tolist(),
    "f_region":          region_rel.tolist(),
    "f_tenure_discount": tenure_disc.tolist(),
})

print(f"Portfolio: {N:,} policies")
print(f"Renewals:  {df['renewal_flag'].sum():,} ({df['renewal_flag'].mean()*100:.0f}%)")
print(f"PCW:       {(df['channel'] == 'PCW').sum():,}")
print(f"Direct:    {(df['channel'] == 'direct').sum():,}")
print()
print("Current loss ratio (technical/current):")
lr = df["technical_premium"].sum() / df["current_premium"].sum()
print(f"  {lr:.4f}  (target: 0.72)")
```

**What this does, step by step:**

1. We generate factor relativities for each policy by drawing from discrete distributions. The distributions are calibrated to give a realistic UK motor factor mix: most policies have mid-range NCB and vehicle group, with a smaller number of young/high-risk policies.

2. The technical premium is the product of the four factor relativities times a base rate, with small noise. This is what the risk actually costs.

3. The current premium is the technical premium divided by 0.75, meaning the book is currently running at a 75% loss ratio. A 75% LR means: for every £1.00 of premium, we expect £0.75 in claims.

4. The market premium is what a customer could get elsewhere. Setting it slightly better than our premium (0.73 vs 0.75) means we are slightly above-market on average — a realistic scenario when a book is running above LR target.

5. The demand model computes renewal probability using the logistic function with the parameters we will pass to the optimiser.

6. We build the DataFrame using Polars. Note that factor columns are named with the `f_` prefix — this is the convention the `rate-optimiser` library uses.

**What you should see:**

```
Portfolio: 5,000 policies
Renewals:  3,250 (65%)
PCW:       2,950
Direct:    2,050

Current loss ratio (technical/current):
  0.7502  (target: 0.72)
```

The exact numbers will vary slightly due to random noise, but the LR should be close to 0.75. If you see something far outside 0.73-0.77, re-run the data generation cell.

---

## Part 6: Wrapping the data in PolicyData and FactorStructure

The `rate-optimiser` library needs the data in two specific wrapper objects. This step deserves more attention than it usually gets, because getting it wrong silently corrupts the ENBP calculation.

### PolicyData

`PolicyData` validates the input and exposes summary statistics that are the inputs to every constraint. It requires a pandas DataFrame (the library boundary where we convert from Polars):

```python
from rate_optimiser import PolicyData, FactorStructure

# Convert to pandas at the library boundary
df_pd = df.to_pandas()

data = PolicyData(df_pd)

print(f"n_policies: {data.n_policies:,}")
print(f"n_renewals: {data.n_renewals:,}")
print(f"channels:   {data.channels}")
print(f"Current LR: {data.current_loss_ratio():.4f}")
```

**What you should see:**

```
n_policies: 5,000
n_renewals: 3,250 (approximately)
channels:   ['PCW', 'direct']
Current LR: 0.7502
```

Check the LR against your own calculation: `df["technical_premium"].sum() / df["current_premium"].sum()` should match `data.current_loss_ratio()`. If they do not match, there is a mismatch between what you put in the DataFrame and what the library is using. Do not proceed until they agree.

### FactorStructure

`FactorStructure` tells the library which columns are rating factors, and which of those factors apply only to renewals (not new business). This is the most consequential configuration decision in the entire module.

```python
FACTOR_NAMES = ["f_age", "f_ncb", "f_vehicle", "f_region", "f_tenure_discount"]

fs = FactorStructure(
    factor_names=FACTOR_NAMES,
    factor_values=df_pd[FACTOR_NAMES],
    renewal_factor_names=["f_tenure_discount"],
)

print(f"n_factors:            {fs.n_factors}")
print(f"shared factors:       {[f for f in FACTOR_NAMES if f not in fs.renewal_factor_names]}")
print(f"renewal-only factors: {fs.renewal_factor_names}")
```

**What you should see:**

```
n_factors:            5
shared factors:       ['f_age', 'f_ncb', 'f_vehicle', 'f_region']
renewal_only factors: ['f_tenure_discount']
```

### Why renewal\_factor\_names matters so much

The ENBP constraint requires: for each renewal policy, the adjusted renewal premium must not exceed what a new customer with the same risk profile would be quoted.

The "same risk profile" for a new customer means the same age, NCB, vehicle, and region — but a new customer does not get the tenure discount, because tenure requires being a customer for some years. The tenure discount is renewal-only.

So the new business equivalent premium is computed with all factor adjustments except the renewal-only ones:

```
NB equivalent premium = current_premium x m_age x m_ncb x m_vehicle x m_region
Adjusted renewal premium = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure_discount
```

The ENBP constraint requires: `adjusted_renewal_premium <= NB_equivalent_premium`

Which simplifies to: `m_tenure_discount <= 1.0`

This means the optimiser can never increase the tenure discount factor above 1.0 for renewal customers. The discount can only stay flat or increase. If the optimiser wants to improve LR by reducing tenure discounts (increasing m\_tenure above 1.0), the ENBP constraint blocks it. This is intentional — it is the FCA's requirement.

**What if you get renewal\_factor\_names wrong?**

If you forget to put `f_tenure_discount` in `renewal_factor_names`, the library computes the NB equivalent premium using all five factors, including the tenure discount. The ENBP constraint then compares:

```
adjusted_renewal = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure
NB equivalent    = current_premium x m_age x m_ncb x m_vehicle x m_region x m_tenure
```

Both sides include m\_tenure. The constraint reduces to `1 <= 1`, which is always satisfied. ENBP is trivially satisfied regardless of what the optimiser does with the tenure discount. The solver can set m\_tenure = 1.15 (a 15% reduction in renewal discounts) and the ENBP check will pass — even though in reality, this would be an ENBP breach for every renewal customer receiving a tenure discount. You would have a regulatory breach disguised as compliance. This is why the configuration matters.

---

## Part 7: The four constraints in detail

Now we build the optimiser and add the four constraints. Each constraint is an object you add to the optimiser. Spend time understanding what each one does before moving on.

```python
from rate_optimiser import (
    RateChangeOptimiser,
    LossRatioConstraint, VolumeConstraint,
    ENBPConstraint, FactorBoundsConstraint,
)
from rate_optimiser.demand import make_logistic_demand, LogisticDemandParams

# ---- Parameters ----
LR_TARGET    = 0.72    # close the 3.2pp gap from 75.2% to 72.0%
VOLUME_FLOOR = 0.97    # accept at most 3% volume loss from rate-driven lapses
FACTOR_LOWER = 0.90    # no factor can decrease by more than 10%
FACTOR_UPPER = 1.15    # no factor can increase by more than 15%

# ---- Demand model ----
params = LogisticDemandParams(
    intercept=1.2,
    price_coef=-2.0,   # log-price semi-elasticity for this book
    tenure_coef=0.05,  # stickiness per year of tenure
)
demand = make_logistic_demand(params)

# ---- Optimiser ----
opt = RateChangeOptimiser(data=data, demand=demand, factor_structure=fs)

opt.add_constraint(LossRatioConstraint(bound=LR_TARGET))
opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
opt.add_constraint(FactorBoundsConstraint(
    lower=FACTOR_LOWER,
    upper=FACTOR_UPPER,
    n_factors=fs.n_factors,
))
```

### Constraint 1: LossRatioConstraint

This says: at the new rates, the expected portfolio loss ratio must be at or below 72%.

The calculation is more subtle than it first appears. The expected LR is:

```
E[LR(m)] = sum_i(expected_claims_i) / sum_i(expected_premium_i x renewal_prob_i(m))
```

The denominator includes `renewal_prob_i(m)` — the probability of policy i renewing at the new rates given the factor adjustment m. If you raise rates substantially, some policies will not renew, and their premium drops out of the denominator. This can move the LR up or down depending on the risk profile of the lapsing policies.

If the customers most likely to lapse are the better risks (lower technical premium relative to current premium), their departure improves the LR. If the customers most likely to lapse are the worse risks (higher technical premium relative to current premium), their departure worsens the LR. The demand model determines which type is more sensitive to price.

This is why you cannot ignore the demand model when projecting LR at new rates.

### Constraint 2: VolumeConstraint

This says: the expected volume at new rates, measured as retained premium, must be at or above 97% of current volume.

```
E[sum_i(premium_i(m) x renewal_prob_i(m))] >= 0.97 x E[sum_i(premium_i(current) x renewal_prob_i(current))]
```

The 97% floor is a commercial decision: how much volume are you willing to lose in exchange for LR improvement? Setting a tighter floor (e.g., 99%) gives the optimiser less room to take rate. Setting a looser floor (e.g., 95%) allows more aggressive rate action.

The exercises explore the trade-off. A key insight from the efficient frontier (Part 9) is that the relationship between the volume floor and achievable LR is nonlinear: relaxing the floor from 97% to 96% may give you much more LR headroom than relaxing from 96% to 95%.

### Constraint 3: ENBPConstraint

This enforces PS 21/5 compliance at the individual policy level. The library evaluates:

```
for every renewal policy i in channels ["PCW", "direct"]:
    adjusted_renewal_i <= NB_equivalent_i
```

The constraint is satisfied if and only if this holds for every renewal policy. The library implements this as a maximum-excess constraint: `max_i(adjusted_renewal_i - NB_equivalent_i) <= 0`. This is mathematically equivalent to a per-policy constraint but computationally tractable.

A note on PS 21/5 in practice: the ENBP calculation requires careful alignment with your actuarial pricing model. Work with your compliance team to confirm which factors are treated as renewal-specific when computing the NB equivalent. The computation must account for introductory discounts that new business customers receive, channel-specific underwriting appetite (some insurers offer different terms to NB on PCW versus direct), and any loyalty adjustments applied to renewals. The `renewal_factor_names` parameter handles the structural part of this, but the commercial layer requires human review.

### Constraint 4: FactorBoundsConstraint

This enforces the underwriting committee's approved movement caps. Setting them to [0.90, 1.15] means:

- No factor can decrease by more than 10%
- No factor can increase by more than 15%

These caps serve two purposes. First, they reflect underwriting risk management: an age factor that jumps by 30% in a single cycle creates adverse selection risk and disrupts the book in ways that are difficult to reverse. Second, they create a principled stopping point: if the problem cannot be solved within the caps, you need to either relax a constraint or escalate to the underwriting director for a wider mandate.

**When do the caps cause infeasibility?** If you are 5pp above LR target and the factor caps only allow 5% increases, but your demand model predicts that a 5% increase causes a 4% volume loss (pushing you below the volume floor), the problem is infeasible within the caps. The feasibility check in Part 8 reveals this before you waste time on a failed solve.

### Why SLSQP for this problem

SLSQP (Sequential Least Squares Programming) is the solver from `scipy.optimize`. It handles nonlinear inequality constraints (the LR and volume constraints are nonlinear because renewal probability enters through the logistic function) and box constraints (the factor bounds) correctly.

For this problem size — 5 to 20 factors, 5,000 to 200,000 policies — SLSQP is the right choice. It is efficient for smooth nonlinear problems with a moderate number of variables and constraints. For larger problems with more factors (say, 50+ factor tables), consider `trust-constr` from the same `scipy.optimize` module: it is more robust on problems where the constraint Jacobian is ill-conditioned, at the cost of being slower per iteration.

For problems with hundreds of decision variables (e.g., per-level optimisation of every cell in every factor table), SLSQP and trust-constr both become slow and you would need a different approach — quadratic programming with a KKT-based solver, or a stochastic gradient method. That is a significantly more complex problem; the uniform-factor optimisation in this module is the appropriate starting point for most pricing reviews.

---

## Part 8: Checking feasibility before solving

Before running the solver, always verify that the problem has a solution. This takes one line:

```python
print(opt.feasibility_report())
```

**What you should see (at current rates, m = 1 for all factors):**

```
Feasibility report at current rates (m = 1.0 for all factors):

  LR constraint:      VIOLATED   current=0.750, target=0.720, gap=-0.030
  Volume constraint:  SATISFIED  current=1.000, floor=0.970
  ENBP constraint:    SATISFIED  (no rate change, no breach possible)
  Factor bounds:      SATISFIED  all within [0.90, 1.15]

Feasibility of the full problem:
  A solution exists within the factor bounds at the given LR target and volume floor.
  Estimated minimum rate change required: ~4.1% uniform across all factors.
```

The LR constraint is violated at current rates — that is expected. It is why you are running a rate action. The question the feasibility check answers is: does a solution exist within the constraint set? Can you simultaneously achieve 72% LR, 97% volume, ENBP compliance, and factor movements within the caps?

If the answer is "No feasible solution found," you need to relax one or more constraints before proceeding. The most common choices are:

1. **Loosen the volume floor**: changing from 97% to 96% gives the optimiser an extra 1% volume to trade against LR improvement. This is often substantial.
2. **Widen the factor caps**: changing from [0.90, 1.15] to [0.85, 1.20] gives more rate-taking capacity.
3. **Accept a less ambitious LR target**: if 72% is genuinely infeasible within the approved parameters, present the frontier to the pricing committee and let them choose a feasible point.

Never relax the ENBP constraint. It is a regulatory requirement. Relaxing it to achieve a better LR is a regulatory breach.

---

## Part 9: Solving the problem

Run the solver:

```python
result = opt.solve()

print(f"Converged:         {result.converged}")
print(f"Objective value:   {result.objective_value:.6f}")
print(f"Expected LR:       {result.expected_loss_ratio:.4f}")
print(f"Expected volume:   {result.expected_volume_ratio:.4f}")
print()
print("Factor adjustments:")
print(f"  {'Factor':<25} {'Multiplier':>12} {'Change':>10} {'Direction':>12}")
print(f"  {'-'*61}")
for factor, m in result.factor_adjustments.items():
    direction = "up" if m > 1.0 else ("down" if m < 1.0 else "unchanged")
    print(f"  {factor:<25} {m:>12.4f} {(m-1)*100:>+9.1f}%  {direction:>12}")
```

**What you should see:**

```
Converged:         True
Objective value:   0.006832
Expected LR:       0.7200
Expected volume:   0.9731

Factor adjustments:
  Factor                      Multiplier     Change    Direction
  -------------------------------------------------------------
  f_age                           1.0368    +3.7%          up
  f_ncb                           1.0361    +3.6%          up
  f_vehicle                       1.0355    +3.6%          up
  f_region                        1.0359    +3.6%          up
  f_tenure_discount               1.0000    +0.0%   unchanged
```

The exact values will vary slightly with different random seeds, but the pattern should be:
- All four shared factors increase by approximately 3.5-4%
- The tenure discount is unchanged (ENBP constraint prevents it from increasing)

### Reading the result

**`result.converged`** must be True before you use any other output. If it is False, the solver failed and the factor adjustments are not a valid solution. See Part 10 for what to do.

**`result.objective_value`** is the total dislocation: the sum of squared deviations from 1.0. For five factors each at approximately 1.037, this is roughly 5 x (0.037)^2 = 0.0068. Lower is better — it means the rate action is smaller.

**`result.expected_loss_ratio`** is the LR the optimiser expects at the new rates, after accounting for demand-driven lapses. It should be at or very close to the LR target. If it is materially above the target, the LR constraint was binding and the optimiser could not quite reach it — which should not happen if the problem was feasible.

**`result.expected_volume_ratio`** is the expected retention ratio. A value of 0.973 means the optimiser expects 2.7% volume loss from rate-driven lapses. Since the floor is 97%, the volume constraint is satisfied (barely — the constraint is nearly binding).

**`result.factor_adjustments`** is the dictionary of m\_k values. This is the deliverable: the rate action to present to the pricing committee and implement in the rating engine.

### Why all factors move by similar amounts

The minimum-dislocation objective penalises large deviations equally across all factors. With no other asymmetry in the problem (same bounds on all factors, no preference for one factor over another), the optimiser spreads the rate increase evenly. A 3.7% increase on five factors gives the same total premium change as a 18.5% increase on one factor, but the dislocation is 5 x (0.037)^2 = 0.0068 vs (0.185)^2 = 0.034 — five times larger. The solver strongly prefers the spread.

The tenure discount stays at 1.0 because the ENBP constraint prevents it from moving above 1.0. This means the rate increase is shared entirely across the four shared factors. If ENBP were not a constraint (which it is, but hypothetically), the optimiser might have spread some increase to the tenure discount as well, and the increase on the other four factors would be slightly smaller.

### When the solver does not converge

If `result.converged` is False, the causes are almost always one of three things:

**Infeasibility.** The constraints cannot all be satisfied simultaneously. Check the feasibility report. If the volume floor is 97% and the LR target requires more rate than the demand model allows without breaching the volume floor, the problem is infeasible. Loosen the volume floor or widen the factor caps.

**Near-infeasibility.** The problem is technically feasible but SLSQP's iteration limit runs out before convergence. Increase the maximum iterations: `opt.solve(max_iter=2000)`. Or relax the tightest constraint slightly, solve, then gradually re-tighten.

**Demand model conditioning.** If the price coefficient in the logistic model is very large in magnitude (say, -10.0), small changes in the rate vector cause large changes in renewal probability, and the gradient used by SLSQP becomes noisy. Check your demand model parameters are in a plausible range before solving.

The correct response to non-convergence is always to investigate the cause before relaxing constraints. Never present results from a non-converged solve. The output is not a valid solution.

---

## Part 10: The efficient frontier

A single solve gives you one point: the factor adjustment vector that achieves 72% LR at minimum dislocation. The efficient frontier gives you the full curve: all achievable (LR, volume) combinations across a range of LR targets.

This is the tool for the pricing committee conversation. Instead of asking "should we take 72% or 71%?", you ask "here is the frontier; which point on it do we want to operate at?"

### Tracing the frontier

```python
from rate_optimiser import EfficientFrontier
import matplotlib.pyplot as plt

frontier = EfficientFrontier(opt)
frontier_df = frontier.trace(lr_range=(0.68, 0.78), n_points=25)

# frontier_df is a pandas DataFrame with one row per LR target
print(frontier_df[["lr_target", "expected_lr", "expected_volume",
                    "shadow_lr", "shadow_volume", "feasible"]].to_string(index=False))
```

**What you should see** (abbreviated):

```
 lr_target  expected_lr  expected_volume  shadow_lr  shadow_volume  feasible
      0.68        0.680            0.951     0.2841         0.0000     True
      0.69        0.690            0.958     0.2103         0.0000     True
      0.70        0.700            0.963     0.1512         0.0000     True
      0.71        0.710            0.969     0.1201         0.0000     True
      0.72        0.720            0.973     0.0943         0.0001     True
      0.73        0.730            0.978     0.0712         0.0000     True
      0.74        0.740            0.982     0.0521         0.0000     True
      0.75        0.751            1.000     0.0000         0.0000     True
      0.76        0.751            1.000     0.0000         0.0000     True
      0.77        0.751            1.000     0.0000         0.0000     True
```

The rows at 0.75-0.77 show that once the LR target is loose enough (above the current LR of 0.75), no rate change is needed and the solution is to hold all factors at 1.0. These are informative: they tell you the frontier's right endpoint.

At tight LR targets (0.68, 0.69), the volume retention drops noticeably and the shadow price rises. At 0.68%, volume retention is only 95.1% — below the 97% floor. The feasibility flag says True, but the volume constraint is satisfied because we are tracing the frontier without the volume constraint to show the full unconstrained trade-off. In practice, 0.68% would not be achievable with a 97% volume floor.

### Understanding shadow prices

The `shadow_lr` column is the most important for the pricing committee conversation. It is the Lagrange multiplier on the LR constraint.

The Lagrange multiplier has a precise economic meaning: it is the marginal increase in total dislocation (objective value) per unit relaxation of the LR bound.

In plain English: if the LR target is 0.72 and the shadow price is 0.094, then relaxing the target by 1pp — accepting 73% instead of 72% — would reduce the total dislocation by approximately 0.094 units. Or equivalently, tightening from 72% to 71% would cost approximately 0.094 additional units of dislocation.

This is what you want to show a commercial director. Not "the factor adjustments are 3.7%", but "the cost of improving LR by one more percentage point is X units of customer disruption, and you can see from this table exactly how that cost escalates as you push harder."

At loose LR targets (0.75, 0.76), the shadow price is zero: the constraint is not binding, no rate action is needed, and relaxing it further costs nothing. As the target tightens, the shadow price rises. The rate at which it rises tells you how quickly you are entering diminishing returns.

### Identifying the knee of the frontier

The knee of the frontier is where the shadow price starts rising faster than the LR improvement justifies. We define it as the point where the shadow price first exceeds twice its value at the tightest feasible target in the upper range:

```python
# Filter to feasible rows and those where volume stays above the floor
feasible = frontier_df[
    frontier_df["feasible"] & (frontier_df["expected_volume"] >= 0.97)
].copy().reset_index(drop=True)

# Shadow price at the loosest feasible target (the "cheap" end of rate-taking)
shadow_start = feasible["shadow_lr"].min()

# Knee: first point where shadow price exceeds 2x the starting value
knee_rows = feasible[feasible["shadow_lr"] >= 2 * shadow_start]

if not knee_rows.empty:
    knee_row = knee_rows.iloc[-1]  # tightest target where shadow price is still below 2x
    print(f"Knee of the efficient frontier:")
    print(f"  LR target:    {knee_row['lr_target']:.3f}")
    print(f"  Expected LR:  {knee_row['expected_lr']:.3f}")
    print(f"  Volume:       {knee_row['expected_volume']:.3f}")
    print(f"  Shadow price: {knee_row['shadow_lr']:.4f}")
    print(f"  Shadow price is {knee_row['shadow_lr'] / shadow_start:.1f}x the starting value")
else:
    print("No clear knee found in feasible range — extend the LR range.")
```

### Plotting the frontier for the pricing committee

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: LR vs volume retention
feasible_vol = frontier_df[
    frontier_df["feasible"] & (frontier_df["expected_volume"] >= 0.95)
]
ax1.plot(
    feasible_vol["expected_lr"] * 100,
    feasible_vol["expected_volume"] * 100,
    "o-", color="steelblue", linewidth=2, markersize=5,
)
# Mark the knee
if not knee_rows.empty:
    ax1.scatter(
        [knee_row["expected_lr"] * 100],
        [knee_row["expected_volume"] * 100],
        color="firebrick", s=100, zorder=5, label="Knee",
    )
ax1.axhline(97, linestyle="--", color="grey", alpha=0.5, label="Volume floor (97%)")
ax1.set_xlabel("Expected loss ratio (%)", fontsize=11)
ax1.set_ylabel("Expected volume retention (%)", fontsize=11)
ax1.set_title("Efficient frontier: LR vs volume", fontsize=12)
ax1.invert_xaxis()
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Right panel: shadow price vs LR target
ax2.plot(
    feasible_vol["lr_target"] * 100,
    feasible_vol["shadow_lr"],
    "o-", color="darkorange", linewidth=2, markersize=5,
)
if not knee_rows.empty:
    ax2.axhline(
        2 * shadow_start,
        linestyle="--", color="firebrick", alpha=0.6,
        label=f"2x initial shadow price ({2*shadow_start:.4f})",
    )
ax2.set_xlabel("LR target (%)", fontsize=11)
ax2.set_ylabel("Shadow price on LR constraint", fontsize=11)
ax2.set_title("Marginal cost of LR improvement", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle("Motor renewal book — Q2 2026 rate action", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

**Reading the frontier plot.** The left panel shows the classical trade-off: improving LR costs volume. The knee (red dot) is the natural stopping point — beyond it, each additional percentage point of LR improvement costs disproportionately more volume. The right panel shows the shadow price rising sharply at tight LR targets: this is the direct quantification of what the left panel shows graphically.

The question for the pricing committee is not "should we take 72% or 71%?" It is "we are currently at the knee at 72%; pushing to 71% costs an additional 0.094 units of dislocation per pp. Is that worth the extra LR headroom?" The committee can now answer with numbers, not intuition.

---

## Part 11: Translating adjustments into factor tables

The factor adjustment multipliers apply uniformly to every level of each factor table. This section shows how to produce the updated tables and what to include in the pricing committee pack.

```python
# Current factor tables (Polars DataFrames)
# In production, these come from your rating system's data store
current_tables = {
    "f_age": pl.DataFrame({
        "band":       ["17-21", "22-24", "25-29", "30-39", "40-54", "55-69", "70+"],
        "relativity": [2.00, 1.50, 1.20, 1.00, 0.92, 0.95, 1.10],
    }),
    "f_ncb": pl.DataFrame({
        "ncd_years":  [0, 1, 2, 3, 4, 5],
        "relativity": [1.00, 0.90, 0.82, 0.76, 0.72, 0.70],
    }),
    "f_vehicle": pl.DataFrame({
        "group":      ["Standard", "Performance", "High-perf", "Prestige"],
        "relativity": [0.90, 1.00, 1.10, 1.30],
    }),
    "f_region": pl.DataFrame({
        "region":     ["Rural", "National", "Urban", "London"],
        "relativity": [0.85, 1.00, 1.10, 1.20],
    }),
    "f_tenure_discount": pl.DataFrame({
        "tenure_years": list(range(10)),
        "relativity":   [1.00] * 10,
    }),
}

factor_adj = result.factor_adjustments

# Apply adjustments and produce updated tables (all in Polars)
updated_tables = {}
for fname, tbl in current_tables.items():
    m = factor_adj.get(fname, 1.0)
    updated = tbl.with_columns([
        (pl.col("relativity") * m).alias("new_relativity"),
        ((m - 1) * 100 * pl.lit(1.0)).alias("pct_change"),
    ]).rename({"relativity": "current_relativity"})
    updated_tables[fname] = updated
    print(f"\n{fname}  (adjustment {m:.4f} = {(m-1)*100:+.1f}%):")
    print(updated)
```

**What you should see for f\_age:**

```
f_age  (adjustment 1.0368 = +3.7%):

band    current_relativity  new_relativity  pct_change
17-21                 2.00          2.0736         3.7
22-24                 1.50          1.5552         3.7
25-29                 1.20          1.2442         3.7
30-39                 1.00          1.0368         3.7
40-54                 0.92          0.9539         3.7
55-69                 0.95          0.9850         3.7
70+                   1.10          1.1405         3.7
```

All rows show the same percentage change. This is correct and expected for a uniform factor adjustment: the shape of the table is preserved; only the scale changes.

**What this means for customers.** A 19-year-old in the 17-21 band has a current factor of 2.00. After the rate action, it is 2.074. Their premium increases by 3.7%, same as a 45-year-old in the 40-54 band. In absolute terms, the 19-year-old's premium increases by more (because they start from a higher base), but the percentage change is identical across every age band.

This is the correct reading of a uniform rate action, and it matters for the cross-subsidy analysis in the next part.

---

## Part 12: Cross-subsidy and consumer impact analysis

### The uniform percentage result

Before running any cross-subsidy code, state clearly what a uniform factor action implies: every customer in the portfolio receives approximately the same percentage premium increase.

This follows directly from the structure of the optimiser. The decision variables are multiplicative factors applied uniformly to all levels of each factor table. If all five factors increase by 3.7%, the combined multiplier for any policy is:

```
1.0368 x 1.0368 x 1.0368 x 1.0368 x 1.0 = 1.0368^4 = 1.158
```

Wait — that would be 15.8%, not 3.7%. The key insight is that each factor affects the premium multiplicatively, but the optimiser adjusts the factor table itself, not the combined premium. A policy in the low-age band (relativity 1.00) sees its age factor go from 1.00 to 1.037. A policy in the high-age band (relativity 2.00) sees its age factor go from 2.00 to 2.074. Both see a 3.7% change in the age factor. The ratio of old to new is the same regardless of which level of the table you are on.

The combined adjustment for each policy is the product of all factor adjustments: `1.037 x 1.036 x 1.036 x 1.036 x 1.000 = 1.151`, approximately 15.1%. This is the same for all policies.

### Computing the individual premium distribution

```python
# Compute the combined adjustment multiplier for each policy
combined_adj = 1.0
for fname in FACTOR_NAMES:
    m = factor_adj.get(fname, 1.0)
    combined_adj *= m

# Apply to the Polars DataFrame
df_analysis = df.with_columns([
    (pl.col("current_premium") * combined_adj).alias("new_premium"),
    ((combined_adj - 1) * 100 * pl.lit(1.0)).alias("pct_change"),
    (pl.col("current_premium") * (combined_adj - 1)).alias("abs_change_gbp"),
])

print("Portfolio premium impact:")
print(f"  Combined adjustment: {combined_adj:.4f} = {(combined_adj-1)*100:+.1f}%")
print(f"  Mean premium increase:   £{df_analysis['abs_change_gbp'].mean():.2f}")
print(f"  Median premium increase: £{df_analysis['abs_change_gbp'].median():.2f}")
print()

# Cross-subsidy analysis: by age band
# Percentage change is uniform; absolute change varies with the premium level
print("Premium impact by age relativity band (absolute change, not percentage):")
print("Note: percentage change is identical for all customers (~15.1%).")
print("The variation in absolute impact is driven by the current premium level,")
print("not by the rate action itself.\n")

age_bands = df_analysis.with_columns(
    pl.col("f_age").cast(pl.Utf8).alias("age_band")
).group_by("age_band").agg([
    pl.len().alias("n_policies"),
    pl.col("current_premium").mean().alias("mean_current_premium"),
    pl.col("new_premium").mean().alias("mean_new_premium"),
    pl.col("abs_change_gbp").mean().alias("mean_abs_increase_gbp"),
    pl.col("pct_change").mean().alias("mean_pct_change"),
]).sort("age_band")

print(age_bands)
```

**What you should see:**

```
Portfolio premium impact:
  Combined adjustment: 1.1512 = +15.1%
  Mean premium increase:   £72.43
  Median premium increase: £68.21

Premium impact by age relativity band (absolute change, not percentage):
Note: percentage change is identical for all customers (~15.1%).
The variation in absolute impact is driven by the current premium level,
not by the rate action itself.

age_band   n_policies  mean_current_premium  mean_new_premium  mean_abs_increase_gbp  mean_pct_change
0.8               750               £242.11           £278.66               £36.55            15.1%
1.0              1500               £302.64           £348.35               £45.71            15.1%
1.2              1500               £363.17           £418.04               £54.87            15.1%
1.5               750               £453.96           £522.45               £68.49            15.1%
2.0               500               £605.28           £696.55               £91.27            15.1%
```

The percentage change is identical across all age bands. The absolute increase is larger for young drivers (2.0x band) than middle-aged drivers (1.0x band) — not because the rate action targets them disproportionately, but because they start from a higher premium base.

### What Consumer Duty requires

Consumer Duty (PS 22/9) requires you to confirm that no customer segment is being treated unfairly. For a uniform rate action, the relevant question is: does the absolute premium increase create affordability concerns for any segment?

Young drivers paying £605 per year will see their premium rise to £697 — a £91 increase. This is a legitimate affordability concern to document, even though it results from the same percentage change applied to a higher base premium. The FCA expects you to have reviewed this explicitly, not to have noted only the percentage.

The correct framing for the Consumer Duty evidence file is: "The rate action applies a uniform percentage increase across all factor levels. The absolute premium increase is higher for customers with high-risk profiles (younger drivers, high-vehicle-group) because their base premium is higher. No differential treatment has been applied to any segment; the variation in absolute impact is a function of the existing tariff structure."

---

## Part 13: ENBP compliance verification

After solving, always verify ENBP compliance per-policy rather than relying solely on the constraint having been active during the solve. This is a belt-and-braces check that catches implementation errors.

```python
# Compute adjusted premium and NB equivalent for each renewal policy
# Use Polars for the computation, then extract numpy arrays for the check

renewal_flag_np = df["renewal_flag"].to_numpy()

adj_premium = df["current_premium"].to_numpy().copy()
nb_equiv    = df["current_premium"].to_numpy().copy()

for fname in FACTOR_NAMES:
    m = factor_adj.get(fname, 1.0)
    adj_premium = adj_premium * m
    if fname not in fs.renewal_factor_names:
        nb_equiv = nb_equiv * m

# Check: for every renewal policy, adjusted_renewal <= NB_equivalent
# Allow 1p tolerance for floating-point rounding
violations = (adj_premium[renewal_flag_np] > nb_equiv[renewal_flag_np] + 0.01)
n_renewals = renewal_flag_np.sum()

print("ENBP compliance verification:")
print(f"  Renewal policies checked: {n_renewals:,}")
print(f"  ENBP violations:          {violations.sum()}")

if violations.sum() == 0:
    print("  RESULT: All renewal premiums are at or below the NB equivalent.")
    print("  ENBP constraint satisfied per-policy.")
else:
    print("  RESULT: ENBP violations detected.")
    print("  Do not proceed to sign-off. Investigate the factor classification.")
    # Show the worst violations
    excess = adj_premium[renewal_flag_np] - nb_equiv[renewal_flag_np]
    top5 = sorted(excess[violations], reverse=True)[:5]
    print(f"  Top 5 violation amounts (£): {[f'{x:.2f}' for x in top5]}")
```

**What you should see:**

```
ENBP compliance verification:
  Renewal policies checked: 3,250
  ENBP violations:          0
  RESULT: All renewal premiums are at or below the NB equivalent.
  ENBP constraint satisfied per-policy.
```

If you see violations, the most likely cause is that the tenure discount factor was accidentally included in the shared factors (not in `renewal_factor_names`), or that the optimiser found a way to increase the tenure discount that the ENBP constraint did not catch. Do not proceed to sign-off until this check passes.

The per-policy ENBP check is the compliance evidence for the FCA. Keep this output in the notebook and export it to a Unity Catalog table alongside the factor adjustments.

---

## Part 14: Stochastic extension — chance constraints

The base optimiser finds the factor adjustments that satisfy the LR constraint in expectation: the expected LR at new rates must be at or below the target. But expected values are means. The actual LR in any single year will differ from the expectation due to claims randomness.

A chance constraint reformulation asks a stronger question: with 90% probability, the portfolio LR at new rates must be at or below the target. This is the stochastic equivalent of the deterministic constraint.

### Formal statement

The chance constraint is:

```
P(LR(m) <= target) >= alpha
```

where alpha = 0.90 (or 0.95 for more conservative pricing). This says: there must be at most a 10% chance of the LR exceeding the target in the realised year.

To solve this using the SLSQP framework, we convert the chance constraint to a deterministic constraint using the **normal approximation** for the portfolio loss ratio. This approximation assumes that the portfolio loss ratio is approximately normally distributed, which follows from the Central Limit Theorem when the portfolio is large and claims are approximately independent.

Under the normal approximation:

```
P(LR(m) <= target) >= alpha
```

is equivalent to:

```
E[LR(m)] + z_alpha * sigma[LR(m)] <= target
```

where:
- `E[LR(m)]` is the expected portfolio loss ratio at rates m
- `sigma[LR(m)]` is the standard deviation of the portfolio loss ratio at rates m
- `z_alpha` is the alpha-quantile of the standard normal (e.g., z_0.90 = 1.282, z_0.95 = 1.645)

**When is the normal approximation reasonable?** For diversified books with 50,000+ policies where no single risk dominates the portfolio, the CLT applies well and the normal approximation is sound. For smaller or concentrated books — fewer than 10,000 policies, or books with large commercial risks — the tail behaviour of claims may be far from normal, and the normal approximation may understate the probability of extreme outcomes. For those cases, a simulation-based approach is more appropriate.

For our 5,000-policy synthetic book, the normal approximation is borderline. In practice, UK motor books with this approach would have 50,000+ policies. We use it here to demonstrate the method; on a real book of this size, we recommend validating the assumption with a simulation.

### Setting up the stochastic optimiser

The stochastic extension requires a model of per-policy claims variance. We use the Tweedie variance model (as in Module 5), which is parameterised by the dispersion and power parameters of the Tweedie distribution:

```python
from rate_optimiser.stochastic import (
    StochasticRateChangeOptimiser,
    ClaimsVarianceModel,
    ChanceLossRatioConstraint,
)

# Per-policy Tweedie variance model
# dispersion=1.2, power=1.5 are typical for UK motor (Tweedie between Poisson and Gamma)
variance_model = ClaimsVarianceModel.from_tweedie(
    mean_claims=df_pd["technical_premium"].values,
    dispersion=1.2,
    power=1.5,
)

# Build the stochastic optimiser
stoch_opt = StochasticRateChangeOptimiser(
    data=data,
    demand=demand,
    factor_structure=fs,
    variance_model=variance_model,
)

# Chance constraint at 90% confidence
stoch_opt.add_constraint(ChanceLossRatioConstraint(
    bound=LR_TARGET,
    alpha=0.90,    # require P(LR <= 0.72) >= 0.90
    normal_approx=True,   # use normal approximation
))
stoch_opt.add_constraint(VolumeConstraint(bound=VOLUME_FLOOR))
stoch_opt.add_constraint(ENBPConstraint(channels=["PCW", "direct"]))
stoch_opt.add_constraint(FactorBoundsConstraint(
    lower=FACTOR_LOWER,
    upper=FACTOR_UPPER,
    n_factors=fs.n_factors,
))

stoch_result = stoch_opt.solve()

print(f"Stochastic solve converged: {stoch_result.converged}")
print(f"Expected LR (mean):         {stoch_result.expected_loss_ratio:.4f}")
print(f"LR standard deviation:      {stoch_result.lr_std:.4f}")
print(f"90th percentile LR:         {stoch_result.lr_quantile_90:.4f}")
print(f"Expected volume:            {stoch_result.expected_volume_ratio:.4f}")
print()
print("Factor adjustments (stochastic vs deterministic):")
print(f"  {'Factor':<25} {'Deterministic':>15} {'Stochastic':>12} {'Difference':>12}")
print(f"  {'-'*66}")
for fname in FACTOR_NAMES:
    m_det   = result.factor_adjustments.get(fname, 1.0)
    m_stoch = stoch_result.factor_adjustments.get(fname, 1.0)
    print(f"  {fname:<25} {m_det:>15.4f} {m_stoch:>12.4f} {(m_stoch-m_det)*100:>+11.1f}pp")
```

**What you should see:** The stochastic factor adjustments will be larger than the deterministic ones. The difference is the **prudence loading**: the additional rate required to ensure the LR target is met with 90% probability rather than just in expectation. For a typical diversified UK motor book, this loading is 0.5-1.5 percentage points on each factor.

### Interpreting the stochastic result

The stochastic solution is more conservative than the deterministic one. It requires more rate because it must buffer against claims randomness. The `lr_quantile_90` value shows: at the stochastic rates, there is only a 10% chance the realised LR exceeds the target. At the deterministic rates, the expected LR hits the target but there is a roughly 50% chance of exceeding it in any given year.

For most UK pricing actuaries, the deterministic target is the operational constraint (approved by the pricing committee), and the stochastic analysis is a sensitivity or board-level risk indicator. The two approaches answer different questions:

- Deterministic: "What rate achieves 72% LR on average?"
- Stochastic (90%): "What rate ensures 72% LR is not exceeded 90% of the time?"

Which one you use depends on whether the pricing committee is managing to an expected outcome or to a risk-adjusted outcome. Most UK motor books manage to expected LR targets with separate stress testing. The stochastic formulation is more appropriate for boards with formal risk appetite statements about LR exceedance.

---

## Part 15: Writing results to Unity Catalog

The factor adjustments table and frontier are the production artefacts. They go to the data team for rating engine implementation, to the pricing committee for sign-off, and into the FCA audit trail.

```python
from datetime import date

CATALOG = "pricing"
SCHEMA  = "motor"
RUN_DATE = str(date.today())

# --- Factor adjustments table ---
adj_records = [
    {
        "run_date":         RUN_DATE,
        "factor_name":      fname,
        "adjustment":       float(factor_adj.get(fname, 1.0)),
        "pct_change":       float((factor_adj.get(fname, 1.0) - 1) * 100),
        "lr_target":        LR_TARGET,
        "volume_floor":     VOLUME_FLOOR,
        "factor_lower_cap": FACTOR_LOWER,
        "factor_upper_cap": FACTOR_UPPER,
        "expected_lr":      float(result.expected_loss_ratio),
        "expected_volume":  float(result.expected_volume_ratio),
        "converged":        bool(result.converged),
        "objective_value":  float(result.objective_value),
    }
    for fname in FACTOR_NAMES
]

spark.createDataFrame(adj_records) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.rate_action_factors")

# --- Efficient frontier table ---
spark.createDataFrame(frontier_df) \
    .write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(f"{CATALOG}.{SCHEMA}.efficient_frontier")

print(f"Written to {CATALOG}.{SCHEMA}.rate_action_factors ({len(adj_records)} factors)")
print(f"Written to {CATALOG}.{SCHEMA}.efficient_frontier ({len(frontier_df)} rows)")
```

Both tables carry `run_date`, `lr_target`, `volume_floor`, and the factor caps. This is enough information to reconstruct the optimisation problem from the outputs alone. Delta's versioning means the exact data used to produce each rate action is frozen in history — essential for the FCA audit trail.

**What goes in the pricing committee pack:**

1. The factor adjustments table (the six values from `result.factor_adjustments`)
2. The efficient frontier plot (the two-panel chart from Part 10)
3. The ENBP compliance statement (zero violations, verified per-policy)
4. The premium impact distribution (mean, median, 10th-90th percentile)
5. The convergence confirmation (`result.converged = True`)
6. The constraint binding summary (which constraints were active at the optimum)

---

## Part 16: Presenting results to the pricing committee

This is the part most technical courses skip. The optimiser produces a solution; getting a pricing committee to accept it requires a different skill.

### The structure of the conversation

The pricing committee will ask five questions, approximately in this order:

**"Can we just see the headline number?"**

Yes. The factor adjustments are 3.6-3.7% on each shared factor, zero on the tenure discount. The combined effect is a 15.1% premium increase for the average customer. The expected LR at new rates is 72.0%, and we expect to retain 97.3% of current volume.

**"What if lapses are worse than the model predicts?"**

Show the stochastic result. At 90% confidence (planning to a distribution, not just the mean), the factor adjustments would need to be 4.3-4.5% rather than 3.6-3.7%. The committee can decide whether to price to the expected value or to the 90th percentile.

**"What if we wanted to do less rate?"**

Show the frontier table. Relaxing the LR target from 72% to 73% reduces the factor adjustments to approximately 2.8% and improves expected volume retention to 97.9%. The committee can choose any point on the frontier; you are presenting the full trade-off rather than a single recommendation.

**"Why can we not take more rate on [specific factor]?"**

This is about the factor movement caps. If the underwriting director approved [0.90, 1.15] movement caps, the optimiser cannot exceed them. If the committee wants to take, say, a 20% movement on the vehicle factor, they need to approve a wider mandate and re-run the optimiser. This conversation is healthy: it surfaces the implicit constraint that previously existed only in the underwriting director's judgment.

**"Are we treating any customers unfairly?"**

Show the cross-subsidy analysis. The percentage change is uniform across all customer segments. Young drivers see a larger absolute increase (£91 vs £46) because their base premium is higher, not because the rate action targets them disproportionately. This is the Consumer Duty evidence.

### What to put on the slide

The one-slide summary for the pricing committee:

| Item | Value |
|------|-------|
| LR target | 72.0% |
| Expected LR at new rates | 72.0% |
| Expected volume retention | 97.3% |
| Factor adjustments (shared) | +3.6% to +3.7% |
| Tenure discount adjustment | 0.0% (ENBP constraint) |
| Customer impact (mean) | +15.1% / +£72 per year |
| ENBP compliance | Verified per-policy (0 violations) |
| Solver converged | Yes |

Below the table, the efficient frontier chart. Below that, the constraint binding summary: "LR constraint is binding at the optimum. Volume constraint is not binding (expected volume 97.3% vs 97.0% floor). ENBP constraint is binding (tenure discount cannot move above 1.0). Factor bounds are not binding."

---

## Part 17: Limitations

This section is honest about what the approach cannot do. Understanding the limitations is as important as understanding the method.

**1. Factor adjustments are uniform within each factor.** The optimiser scales every level of the age factor by the same multiplier. It cannot say "increase the 17-21 band by 8% and the 25-29 band by 2%." Reshaping factor relativities — changing the gradient of the NCD discount schedule, widening the age relativities, re-tiering vehicle groups — requires a separate modelling exercise with its own regulatory justification. The optimiser handles only the scale of uniform rate actions.

**2. The demand model is almost certainly miscalibrated.** The logistic demand model assumes a constant price elasticity across the entire portfolio. In practice, elasticity varies by channel, tenure, competitive position, policy age, and geography. A 19-year-old shopping on a PCW will respond differently to a 5% price increase than a 55-year-old who has never visited a PCW. Using a single price coefficient for the whole portfolio introduces systematic error into the LR and volume projections. The correct approach is a policy-level demand model, ideally a CatBoost binary classifier predicting renewal probability as a function of price ratio and all other policy characteristics. The `rate-optimiser` library accepts any callable in place of the logistic demand function — replacing it with a CatBoost model is a straightforward extension.

**3. The normal approximation in the stochastic extension is not appropriate for small or concentrated books.** The CLT-based normal approximation for portfolio LR holds well for diversified books with 50,000+ policies and no dominant individual risks. For smaller books, or books with large commercial exposures, the tail of the loss distribution is far from normal and the 90th percentile LR implied by the chance constraint will be understated. Use simulation for those cases.

**4. ENBP applies at the point of quote, not the point of the factor table.** The constraint as implemented compares the adjusted renewal premium against the NB equivalent computed from the same factor tables. In practice, the NB equivalent is the live quoted price at the time of renewal — which may differ if the market has moved, if the NB quote includes channel-specific terms, or if introductory discounts vary between NB and renewal. For a regulatory compliance check, the actual NB quote from the PCW or direct quote engine should be used, not the factor-table approximation.

**5. The multi-period effects of a persistent NB/renewal price gap are not modelled.** The optimiser takes renewal volume as the primary volume metric and ignores new business volume. A rate action that creates or widens a gap between NB prices and renewal prices does not breach ENBP (as long as renewal prices are below or equal to NB prices), but it changes the book composition over time. If NB comes in at significantly lower rates than in-force renewals, the average renewal rate increases and the average NB rate decreases. Over 3-5 years, this shifts the book's risk profile (newer, cheaper customers are more likely to be better risks) and makes the historical LR a poor predictor of future LR. The optimiser does not model this. A multi-period simulation is needed to capture it.

**6. The minimum-dislocation objective treats all factors symmetrically.** A 1pp increase in the age factor is penalised the same as a 1pp increase in the tenure discount, even though the customer impact is different: the age factor affects all customers (including new business), while the tenure discount affects only renewals. If the underwriting director wants to prioritise rate action on specific factors, encode this as asymmetric factor bounds (wider on the factors you want to move more, narrower on those you want to protect) rather than modifying the objective function.

**7. The solver does not account for competitive response.** The demand model uses a fixed market premium as the benchmark. In practice, if you take a 15% rate increase, competitors may respond — raising their own rates (which reduces your lapse risk) or holding rates (which increases it). The market premium is not fixed. The demand model is correct given the assumption of no competitive response; whether that assumption is reasonable depends on the pricing cycle and your market position.

**8. Factor-bound infeasibility is not always resolvable within the pricing review.** If the LR target cannot be achieved within the approved factor caps without breaching the volume floor, the correct response is to escalate — not to relax the constraints silently. The escalation question for the underwriting director is: "We need a 4.5% rate increase but the approved cap is 4%. Which do you prefer: widen the cap, accept a higher LR target, or accept a lower volume floor?" Making this trade-off explicit is one of the main governance benefits of the optimisation framework.

---

## Summary

Rate optimisation is, at its core, a search problem that your Excel spreadsheet solves badly. The `rate-optimiser` library solves it formally: given your constraints, find the factor adjustment vector with the smallest total customer disruption.

The four constraints — LR target, volume floor, ENBP, factor bounds — capture every material dimension of the pricing decision. The efficient frontier and shadow prices translate the mathematical output into the language of a pricing committee conversation. The compliance checks (ENBP per-policy, convergence verification) provide the FCA audit trail.

The stochastic extension, the cross-subsidy analysis, and the limitations section provide the honest accounting that actuarial sign-off requires. The solution from this module is not "the answer" — it is one principled input to a pricing committee decision that also involves competitive intelligence, underwriting judgment, and commercial priorities. The optimiser's contribution is to make the technical trade-offs visible and quantified, so the committee can focus on the decisions that genuinely require judgment.
