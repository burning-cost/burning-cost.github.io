---
layout: post
title: "UBI Adverse Selection: When Telematics Discounts Drive Away Your Best Risks"
date: 2026-03-25
categories: [pricing, techniques]
tags: [telematics, ubi, adverse-selection, uk-motor, motor-pricing, insurance-telematics, insurance-optimise, insurance-monitoring, self-selection, portfolio-management, python]
description: "The adverse selection trap in opt-in UBI: why telematics discounts attract the risks you least want to retain, and what to do about it."
---

The logic of usage-based insurance sounds impeccable. Fit a black box or ask customers to install an app. Score their driving. Discount the good ones. Your book improves, your competitors stand still, and you write more profit per policy year than anyone using traditional GLM pricing.

The problem is that logic is backwards. In an opt-in UBI programme, the discount does not create a better book. It selects one from the pool of drivers who were already going to be better than your traditional model said. And the mechanism that produces that selection is exactly the mechanism that poisons what remains.

This is the UBI adverse selection trap, and it is operating in the UK motor market right now.

---

## How opt-in selection works against you

Start with a vanilla motor portfolio. You have a traditional GLM pricing model — call it the technical price. Your book is a mixture: some drivers the model overprices slightly, some it underprices slightly, most it gets approximately right. The cross-subsidies are small and they roughly cancel.

Now launch a UBI opt-in product. Who responds?

Low-mileage drivers. People who already know their driving is safe. Young drivers whose parents told them they are careful. Anyone whose actual risk behaviour is meaningfully better than what the GLM predicts from their observable rating factors — age, NCB, vehicle group, postcode.

These are precisely your most profitable risks: the ones where the traditional price, before any UBI discount, was generating the highest margin. They opt in, they get their discount, and the effective premium they pay drops. The discount is genuine — they really are lower risk — but the profit per policy falls.

Meanwhile, the drivers who do not opt in are also self-selecting. The reckless 24-year-old knows, consciously or not, that a black box will not help him. The commuter doing 40,000 miles a year in the outside lane of the M25 knows his score will be poor. The driver with three speeding convictions that have not yet reached his renewal does not opt in. They stay on the traditional book. The GLM prices them; the GLM does not know what a telematics device would reveal.

The result is a portfolio that has split itself along risk lines, with the telematics opt-ins clustered at the good end and the non-participants clustered at the bad end. Your total expected loss has not changed — the same drivers are still insured — but the pricing structure now systematically overcharges the bad risks (who have low demand elasticity, so they stay) and undercharges the good risks (who got a discount they earned).

That is not an adverse selection problem in the classical sense of hidden types. It is self-selection on known types: the drivers who can pass a telematics test know they can, and they opt in. The ones who cannot, stay off the scheme.

---

## The portfolio composition shift

The dangerous version of this is not what happens to the telematics book. It is what happens to the non-telematics book over time.

Suppose your traditional motor book has a loss ratio of 72%. After 18 months of UBI opt-ins, the telematics cohort is running at 61%. Looks like the programme is working. Your chief actuary shows this at the board and the UBI budget gets doubled.

But look at the non-telematics book. Its loss ratio has moved from 72% to 76%. The best risks have migrated out, leaving behind a residual pool that is incrementally worse than the overall book was before. The model has not changed. It is still pricing this pool as if it represented the full market distribution it was trained on. It does not know the distribution has shifted.

This is a covariate-like shift but driven by endogenous selection rather than an exogenous change in the insured population. The model's feature distribution has not changed much — the non-opt-ins still look like your historical book in terms of age, NCB, vehicle group. But the conditional distribution of actual driving behaviour, given those traditional features, has worsened. The drivers who remain are systematically worse drivers than the people who share their rating factors in your training data.

Quantifying this shift is non-trivial. Traditional PSI monitoring on your rating factors will not flag it, because the distribution of observable factors has barely moved. The deterioration is in the unobservable dimension — the one the telematics device was supposed to measure — but you only measure that dimension for the opt-ins. For the non-participants, you are flying blind.

---

## Detecting and quantifying the problem

We will use three tools from the library suite:

- [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) for HMM-based driver risk scoring
- [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) for modelling the pricing pressure from UBI discounts on portfolio composition
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) for tracking the non-telematics book deterioration

```bash
uv add insurance-telematics insurance-optimise insurance-monitoring
```

### Step 1: Score the opt-in cohort

The first thing to establish is whether your telematics opt-ins are actually better drivers than the traditional model predicted — or whether the discount is simply following the traditional risk signal into a new data format.

```python
from insurance_telematics import TelematicsScoringPipeline, TripSimulator

# Simulate two cohorts: opt-ins and non-opt-ins
# In practice, trips_df comes from your telematics provider
sim = TripSimulator(seed=2026)
trips_df, claims_df = sim.simulate(n_drivers=500, trips_per_driver=40)

pipe = TelematicsScoringPipeline(n_hmm_states=3, credibility_threshold=30)
pipe.fit(trips_df, claims_df)
scored = pipe.predict(trips_df)

print(scored.head())
# driver_id  predicted_claim_frequency
# D0001      0.0812
# D0002      0.1341
# ...
```

The pipeline runs GPS cleaning, trip feature extraction, a 3-state Hidden Markov Model (cautious / normal / aggressive), Bühlmann-Straub credibility weighting, and a Poisson GLM, all in a single `fit()` call. The output is predicted annual claim frequency per driver — a number you can directly compare to the GLM prediction from traditional factors.

What you are looking for: if `predicted_claim_frequency` from telematics is systematically lower than the traditional GLM price implied, your opt-ins are selecting on telematics risk, not just on traditional risk. The more negative this gap, the larger the adverse selection premium embedded in your telematics discount.

You want this gap to be zero or small. A large negative gap means the discount is doing real work — good, it is accurate pricing — but it is doing that work by pulling your best traditional-model risks into the telematics book. Your non-telematics book is being drained.

### Step 2: Model the pricing pressure

Once you have the opt-in claim frequency scores, you can model what the appropriate pricing structure looks like under different opt-in penetration rates. The key question is: at what discount level does the telematics book become economically self-sustaining without being so generous that it pulls in everyone with a good traditional-model profile?

```python
import numpy as np
from insurance_optimise import PortfolioOptimiser, ConstraintConfig
from insurance_optimise.demand import RetentionModel

# Construct a stylised portfolio: 1,000 policies
# Half have telematics scores materially better than GLM prediction
rng = np.random.default_rng(2026)
n = 1_000

glm_technical_price = rng.uniform(350, 900, n)
glm_loss_cost = glm_technical_price * rng.uniform(0.55, 0.80, n)

# Telematics score: opt-ins are 15% better on average (adverse selection at work)
# Non-opt-ins: unknown — assume GLM is correct
tele_uplift = np.where(
    rng.uniform(0, 1, n) < 0.30,   # 30% opt-in rate
    rng.uniform(0.75, 0.92, n),     # opt-ins: true cost 8-25% below GLM
    1.0,                            # non-opt-ins: GLM assumed correct
)
true_loss_cost = glm_loss_cost * tele_uplift

# Price elasticities: UBI-eligible customers are more price-sensitive
# (they opted in because the discount matters to them)
elasticity = rng.uniform(-1.8, -1.2, n)

# Renewal probabilities at current price
p_renew = rng.uniform(0.82, 0.95, n)

config = ConstraintConfig(
    lr_max=0.72,
    retention_min=0.85,
    max_rate_change=0.15,
)

opt = PortfolioOptimiser(
    technical_price=glm_technical_price,
    expected_loss_cost=true_loss_cost,
    p_demand=p_renew,
    elasticity=elasticity,
    renewal_flag=np.ones(n, dtype=bool),
    constraints=config,
)

result = opt.optimise()
print(f"Optimised loss ratio:  {result.expected_loss_ratio:.3f}")
print(f"Optimised retention:   {result.expected_retention:.3f}")
print(f"Expected profit:       £{result.expected_profit:,.0f}")
```

The constraint `lr_max=0.72` is important here. Without it, the optimiser will price down the telematics opt-ins aggressively to maximise retention — exactly the behaviour that accelerates adverse selection on the non-telematics book. The loss ratio constraint forces a floor on the discount, ensuring the telematics cohort is still generating underwriting profit even after the telematics discount is applied.

The shadow price on the `lr_max` constraint tells you how much profit you are leaving on the table to maintain that floor. If the shadow price is high, you have a viable business case for a mandatory-telematics product where the constraint is relaxed because non-opt-in adverse selection no longer applies.

### Step 3: Monitor the non-telematics book

The non-telematics book is where the adverse selection accrues, and it is where the deterioration will first show up if the programme is pulling too aggressively on the good risks.

```python
import polars as pl
from insurance_monitoring import psi, ae_ratio, ae_ratio_ci, MonitoringReport
from insurance_monitoring.calibration import CalibrationChecker

# reference_claims: non-telematics book at UBI launch (Month 0)
# current_claims:   non-telematics book 12 months later

# Track the A/E ratio on the non-telematics book quarterly
# A rising A/E indicates the residual book is worse than the model expects
reference_ae = ae_ratio(
    actual=reference_claims["n_claims"].to_numpy(),
    predicted=reference_claims["glm_predicted"].to_numpy(),
    exposure=reference_claims["exposure_years"].to_numpy(),
)

current_ae = ae_ratio(
    actual=current_claims["n_claims"].to_numpy(),
    predicted=current_claims["glm_predicted"].to_numpy(),
    exposure=current_claims["exposure_years"].to_numpy(),
)

print(f"Non-telematics A/E at launch:   {reference_ae:.3f}")
print(f"Non-telematics A/E at 12 months: {current_ae:.3f}")
# Example output:
# Non-telematics A/E at launch:     1.002
# Non-telematics A/E at 12 months:  1.089

# PSI on the non-telematics book rating factors
# This will typically NOT flag the problem — the factor distribution
# is stable even as the underlying risk worsens
psi_age = psi(
    reference=reference_claims["driver_age"].to_numpy(),
    current=current_claims["driver_age"].to_numpy(),
    n_bins=10,
)
print(f"PSI on driver_age: {psi_age:.4f}")  # Likely < 0.10 — no flag raised
```

The A/E drift is the signal you are looking for. A non-telematics A/E that rises from 1.00 to 1.09 over 12 months while the telematics A/E holds at 0.85 is not evidence that the telematics programme is working. It is evidence that the programme is redistributing risk within the portfolio, with the non-telematics residual bearing the cost.

Notice that PSI on the traditional rating factors will very likely not flag this. The feature distribution of the non-telematics book is stable — the ages, NCB bands, and vehicle groups of the non-opt-ins look similar to the historical book. The deterioration is in the unobservable behaviour dimension. The only direct signal is the A/E ratio climbing on the non-telematics cohort.

The full `MonitoringReport` ties these checks together:

```python
report = MonitoringReport(
    reference_actual=reference_claims["n_claims"].to_numpy(),
    reference_predicted=reference_claims["glm_predicted"].to_numpy(),
    current_actual=current_claims["n_claims"].to_numpy(),
    current_predicted=current_claims["glm_predicted"].to_numpy(),
    exposure=current_claims["exposure_years"].to_numpy(),
)
# Results computed on construction — no run() call needed
print(report.results_["ae_ratio"])   # A/E with CI and traffic-light band
print(report.recommendation)         # PASS / RECALIBRATE / REFIT
```

Run this monthly on the non-telematics cohort and set an A/E alert at 1.05. If you hit that threshold within 6 months of launching a generous UBI discount, your opt-in rate has a selection problem.

---

## What to do about it

### Option 1: Mandatory telematics

The cleanest fix is to eliminate opt-in selection by making telematics the default. If everyone is scored, the adverse selection mechanism disappears: the good risks no longer self-select in, because there is no alternative product to self-select away from. You price everyone on their actual behaviour, and the non-telematics residual pool stops growing.

The UK market is moving this way for young driver products, where some insurers (notably those backed by Direct Line Group and Ageas) have made black box telematics mandatory for drivers under 25. The adverse selection argument is one of the cleaner actuarial justifications for mandatory data collection — you are not gathering the data because you want to surveil customers, but because opt-in data collection actively harms the customers who choose not to participate by concentrating risk in their cohort and eventually pushing their prices up.

### Option 2: Discount caps and credibility floors

If mandatory telematics is not viable for your customer segment, discount caps directly limit the adverse selection magnitude. A maximum UBI discount of 15% above the traditional technical price caps the incentive for the very best risks to churn to a competitor's opt-in scheme.

The `ConstraintConfig` in `insurance-optimise` handles this directly via `max_rate_change`:

```python
config = ConstraintConfig(
    lr_max=0.72,
    max_rate_change=0.15,   # no more than 15% reduction from technical price
)
```

The 15% ceiling is not a number to pick arbitrarily — it should come from the A/E gap analysis: the maximum discount where the expected A/E on the opt-in cohort plus the implied deterioration on the non-telematics book still leaves the combined book below your loss ratio target.

The credibility floor matters too. `TelematicsScoringPipeline` uses a default of 30 trips for full Bühlmann-Straub credibility. A driver with 5 trips in their first month should not get a full 20% discount — the HMM state probabilities are too noisy to support it. Use the `credibility_threshold` parameter:

```python
pipe = TelematicsScoringPipeline(
    n_hmm_states=3,
    credibility_threshold=50,   # stricter credibility floor — more trips required
)
```

With `credibility_threshold=50`, a driver with 10 trips gets a Bühlmann weight of 10/60 = 0.17 — their score is shrunk 83% towards the portfolio mean. The discount is minimal until there is genuine driving evidence.

### Option 3: Portfolio-split monitoring with A/E alerts

Regardless of which structure you use, run a persistent split-book monitor: telematics cohort, non-telematics cohort, and combined. Alert on A/E divergence above 1.05 in the non-telematics book. Alert on the spread between the two A/E ratios widening beyond 20 percentage points. Both of those thresholds are leading indicators that the selection is accelerating.

The academic reference here is the EJOR 2025 paper by Shi et al., "A usage-based insurance (UBI) pricing model considering customer retention" (Insurance: Mathematics and Economics, 2025, doi:10.1016/j.insmatheco.2025.01.008). It formulates the joint optimisation problem directly: maximise underwriting profit subject to retention constraints on both the telematics and non-telematics cohorts simultaneously. Their empirical work on Chinese motor data finds that ignoring the retention feedback in the non-telematics cohort leads to a 3-5% understatement of the true adverse selection cost. In the UK market, with a higher proportion of price-sensitive renewal customers (driven by the PS21/11 ENBP constraint), we suspect the understatement is larger.

---

## The number that matters

There is one diagnostic that cuts through all the complexity: the A/E ratio on your non-telematics book, tracked monthly against the pre-UBI baseline.

If it is flat or declining, your telematics programme is selecting good risks and your traditional pricing is holding. You may have a margin issue on the opt-in cohort, but the book is not degrading.

If it is rising by more than 2-3 percentage points per quarter, the selection effect is materialising. The non-telematics residual is becoming a worse pool faster than the model knows. At that rate, you have roughly 18-24 months before the A/E divergence forces a significant rate correction on the non-telematics cohort — a correction that will itself provoke further adverse selection as better traditional-model risks hunt for telematics alternatives.

The UK market had this conversation in 2013-2015, when the first generation of young driver black box products launched, and then largely forgot about it when telematics adoption plateaued at under 15% of motor policies. With smartphone-based UBI now at lower friction — no installation required, immediate score feedback, discounts visible in the first month — adoption rates are rising fast. The adverse selection mechanism is the same. The speed of the feedback loop is not.

Run the monitoring. Set the alerts. Do not wait for the loss ratio to tell you what the A/E has been saying for two years.

---

- [`insurance-telematics`](https://github.com/burning-cost/insurance-telematics) — HMM scoring pipeline for driver risk from raw trip data
- [`insurance-optimise`](https://github.com/burning-cost/insurance-optimise) — constrained portfolio optimisation with FCA PS21/11 compliance
- [`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) — A/E monitoring, Gini drift, and calibration checks for production pricing models
- [Does HMM Telematics Risk Scoring Actually Work?](/2026/03/31/does-hmm-telematics-risk-scoring-actually-work/) — the benchmark post for the telematics scoring pipeline
- [Does Constrained Rate Optimisation Actually Work?](/2026/03/29/does-constrained-rate-optimisation-actually-work/) — the benchmark post for the optimisation library
- [Does Automated Model Monitoring Actually Work?](/2026/03/27/does-automated-model-monitoring-actually-work/) — the benchmark post for the monitoring library
