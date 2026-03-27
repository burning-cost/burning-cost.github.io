---
layout: post
title: "How to Model BI Claims Trajectory Under Whiplash Reform Uncertainty"
date: 2026-03-25
categories: [reserving, severity, regulation]
tags: [insurance-severity, insurance-trend, insurance-whittaker, insurance-conformal, whiplash-reform, bodily-injury]
description: "Scenario modelling for UK motor bodily injury claims under whiplash reform uncertainty — Taylor separation, severity distributions, and conformal prediction intervals."
---

The Civil Liability Act 2018 came into effect on 31 May 2021. Within two years it had done something that most regulatory interventions only claim to do: it measurably changed claims behaviour. Bodily injury as a share of UK motor spend fell from roughly 16% in 2021 to 9% by 2025 (ABI data). BI claim frequency dropped approximately 30% for the relevant low-value injuries. Average cost per sub-£5,000 claim fell around 25% as the fixed tariff replaced general damages negotiations.

Five years on, the MoJ is reviewing the tariff regime. The Spring 2026 review has three plausible outcomes, none of which you can rule out at 95% confidence. That uncertainty is not a reason to wait. It is a modelling problem.

This post shows how to structure the problem for a pricing team: what the reform actually did, what the scenarios are, and how to build scenario-weighted severity curves and a reserve margin that reflects genuine distributional uncertainty rather than a single-point assumption.

---

## What the reform did

The Civil Liability Act 2018 introduced three interlocking changes for road traffic accident personal injury claims below £5,000:

1. **Fixed tariffs for whiplash injuries.** General damages for soft-tissue injuries lasting up to two years are now set by statutory table, not negotiated between solicitors. A one-month whiplash injury pays £240. A two-year injury pays £3,910. The tariff replaces the Judicial College Guidelines for this class.

2. **The Official Injury Claim portal.** RTA personal injury claims must go through the OIC portal before any litigation. This removes the pre-litigation negotiation stage that previously inflated settlement values — and the claimant legal costs that came with it.

3. **No pre-medical offers.** Insurers cannot settle a whiplash claim without a medical report. This was the direct mechanism that suppressed fraudulent and exaggerated claims: no quick offer, no incentive to claim.

The combined effect was structural, not cyclical. MoJ data for the first year post-reform (31 May 2021–30 May 2022) showed 386,000 personal injury claims, against pre-pandemic levels above 650,000. By 2023/24 the figure was approximately 360,000. Settlement times increased — the OIC portal created a new bottleneck, with average settlement taking 251 days in Q2 2023 against 227 days pre-reform — but the volume compression was real and sustained.

For reserving, the premium impact matters: ABI estimated the reform saved £4 per policy in 2020–21, rising to £15 per policy by 2022–23 versus the counterfactual.

What the reform did *not* change: injury severity above £5,000, the treatment of mixed injuries (whiplash plus other soft tissue), or the OIC portal's ability to handle genuinely contested liability. The Supreme Court's mixed-injury ruling in *ABI v Aviva* (2024) added further complexity to the tariff boundary — claims with both a whiplash and a non-whiplash component require careful apportionment.

---

## The uncertainty: three scenarios for the Spring 2026 review

The MoJ committed to reviewing the tariff regime, and that review is running now. Three credible outcomes:

**Scenario A — Tariffs uprated with inflation (status quo+).** The tariffs have not been uprated since 2021. CPI over the period has been material. Under this scenario, the MoJ applies an inflationary uplift — call it 15–20% on the existing bands. Claim counts do not change significantly. Severity increases modestly within the tariffed band. BI as a share of spend ticks back up but remains well below pre-reform levels. This is the lowest-disruption outcome.

**Scenario B — Tariff scope expanded to higher values.** The £5,000 threshold moves up — say, to £7,000 or £8,000. More claims fall within the fixed tariff regime, including some that currently settle via conventional negotiation. Frequency in the £5,000–£8,000 band falls. Average severity falls for affected claims. The reform's reach extends further into the BI book. This is the further-suppression scenario.

**Scenario C — Tariffs loosened or portal simplified.** Political pressure from claimant solicitors and access-to-justice advocates leads to tariff band increases above inflation, reduced medical reporting requirements, or portal simplification that makes it easier for claims to exit into litigation. The reform's friction effects partially unwind. Some pre-litigation negotiation re-enters. BI frequency and severity partially revert toward pre-2021 trajectory. This is the partial reversal scenario and the one that worries pricing teams most.

We assign illustrative probabilities of 45%, 25%, 30% respectively. Your team's view will differ. The point is that Scenario C has non-trivial weight and a severity impact that is asymmetric: the downside (claims re-entering) is larger in absolute terms than the upside from Scenario B.

---

## Modelling the trajectory

### Step 1: Fit scenario-conditional severity distributions

For each scenario, we need a severity distribution over the BI claims that will actually emerge — not the settled portfolio average, but the per-claim distribution that drives the reserve.

[insurance-severity](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) gives us three tools here. For the body of the distribution (attritional BI below £5,000) we use a lognormal body. For the tail — the claims above £5,000 that sit outside the tariff regardless of reform scenario — we use a GPD tail. The spliced model handles the structural break at the tariff threshold.

```python
from insurance_severity import LognormalBurrComposite

# Fit to post-reform settled BI claims
model_base = LognormalBurrComposite(threshold_method="mode_matching")
model_base.fit(bi_claims_post_2021)

# Scenario A: uprate the body distribution by 17%
# (approximate CPI adjustment, May 2021 - May 2026)
# Achieved by scaling the lognormal mu parameter
model_a = LognormalBurrComposite(threshold_method="mode_matching")
model_a.fit(bi_claims_post_2021 * 1.17)

# Scenario C: partial reversion — use 2018-2021 pre-reform data
# blended 70/30 with post-reform
import numpy as np
blended = np.concatenate([
    bi_claims_post_2021,
    bi_claims_pre_reform * 0.3 / 0.7,  # weight pre-reform data
])
model_c = LognormalBurrComposite(threshold_method="mode_matching")
model_c.fit(blended)
```

The `LognormalBurrComposite` handles the mode-matching threshold automatically — you do not need to specify a splice point. For the EVT tail (large BI claims in commercial lines or high-value personal injury), use `TruncatedGPD` to account for any policy limit truncation in the data.

To build the scenario-weighted severity curve:

```python
import numpy as np

weights = np.array([0.45, 0.25, 0.30])  # A, B, C
quantile_grid = np.linspace(0.01, 0.99, 200)

# Weighted quantile function (approximate via quantile blending)
quantiles_a = np.array([model_a.ppf(q) for q in quantile_grid])
quantiles_b = np.array([model_b.ppf(q) for q in quantile_grid])
quantiles_c = np.array([model_c.ppf(q) for q in quantile_grid])

weighted_quantiles = (
    weights[0] * quantiles_a
    + weights[1] * quantiles_b
    + weights[2] * quantiles_c
)
```

This is not a mixture distribution in the formal sense — it is a probability-weighted average of scenario outcomes at each quantile. For reserving purposes, this is the right framing: each scenario is a possible state of the world, not a component of a latent mixture.

### Step 2: Decompose development patterns with Taylor separation

Raw BI development triangles conflate two distinct effects: calendar-year inflation (claims costs rising over time regardless of when the accident happened) and development-year settlement patterns (how a cohort of claims matures through to final settlement). Under reform uncertainty, these effects behave differently:

- Calendar-year inflation still runs at whatever claims inflation rate the market faces (currently elevated by soft tissue treatment costs and legal costs for claims that exit the portal).
- Development-year patterns have been distorted by the OIC portal backlog — settlement extending to 251+ days on average compared to 227 days pre-reform.

Use `insurance-trend` to decompose the severity trend into these components.

```python
from insurance_trend import SeverityTrendFitter

fitter = SeverityTrendFitter(
    periods=['2021Q3', '2021Q4', '2022Q1', '2022Q2',
             '2022Q3', '2022Q4', '2023Q1', '2023Q2',
             '2023Q3', '2023Q4', '2024Q1', '2024Q2'],
    total_paid=bi_paid_by_quarter,
    claim_counts=bi_counts_by_quarter,
)
result = fitter.fit()
print(result.trend_rate)      # annualised log-linear severity trend
print(fitter.superimposed_inflation())  # above ONS index
```

The `superimposed_inflation()` method strips out the ONS motor repair index (series HPTH) and returns the residual — the part of severity growth not explained by general price inflation. For post-reform BI, this residual is the key number: it reflects changing claim mix, increasing legal representation among portal users, and any partial reversion already visible in the data.

### Step 3: Smooth the development factors

The development factors from a BI triangle are noisy at long tails — credibility thins out in the 36–72 month development range. [insurance-whittaker](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) smooths these without imposing a parametric form.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# dev_months: array of development periods (e.g., 12, 24, ..., 96)
# raw_factors: link ratios from each development period
# claim_counts: number of claims at each development age (for weighting)

dev_months = np.array([12, 24, 36, 48, 60, 72, 84, 96])
raw_factors = np.array([2.41, 1.38, 1.18, 1.09, 1.04, 1.02, 1.01, 1.00])
credibility_weights = np.array([8200, 6100, 4300, 2900, 1800, 950, 420, 180])

wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(dev_months, raw_factors, weights=credibility_weights)

# result.fitted: smoothed development factors
# result.ci_lower, result.ci_upper: 95% posterior credible interval
```

The `order=2` penalty smoothes second differences — equivalent to fitting a local quadratic through the factor pattern. REML lambda selection balances the noisy long-tail data against the well-credible early development. The resulting `result.ci_upper` on each development factor feeds directly into the next step.

---

## Quantifying reserve uncertainty with conformal prediction intervals

The smoothed development factors from Whittaker-Henderson carry Bayesian credible intervals — but those intervals assume Gaussian errors and the correctness of the smoothing model. Under reform scenario uncertainty, neither assumption is comfortable.

[insurance-conformal](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) gives us a distribution-free alternative. We treat the development factor as a prediction target, the calibration set is our historical factor observations by development age, and the conformal interval carries a finite-sample coverage guarantee without any distributional assumption.

```python
from insurance_conformal import InsuranceConformalPredictor

# factor_model: any fitted model for development factors
# (e.g., a simple log-linear model of factors against development age)
cp = InsuranceConformalPredictor(
    model=factor_model,
    nonconformity="pearson_weighted",
    distribution="gamma",   # factors are positive, right-skewed
)
cp.calibrate(dev_ages_cal, factors_cal)
intervals = cp.predict_interval(dev_ages_new, alpha=0.10)

# intervals["lower"], intervals["upper"]: 90% distribution-free interval
# for each development factor
```

The gamma distribution choice for `distribution` reflects the fact that development factors are strictly positive and the residuals are heteroscedastic — noisier at long development ages where credibility is low. The `pearson_weighted` score corrects for this heteroscedasticity before computing conformal quantiles.

The resulting interval width on, say, the 60-month factor tells you something concrete: if the smoothed factor is 1.04 and the 90% interval runs from 1.01 to 1.09, your reserve is sensitive to a 5 percentage point swing in that factor. Under Scenario C, where settlement patterns revert toward pre-reform norms, the long-tail factors are the ones most likely to move.

---

## What to hold as reserve margin

Pulling these together into a practical reserve margin:

1. **Scenario-weighted ultimate severity.** Take the probability-weighted average of the three scenario severity distributions. Your central estimate is not Scenario A; it is the scenario-weighted mean, which sits above Scenario A because Scenario C has non-trivial weight and a higher mean.

2. **Development factor uncertainty.** Apply the conformal upper bounds (say, 75th percentile of the interval, not the 95th — you are holding a margin, not an extreme solvency buffer) to the smoothed development factors. This is your uncertainty margin on timing, not magnitude.

3. **Stress the book.** Run the full reserve calculation under Scenario C with conformal upper development factors. The difference between this and your central estimate is the reform uncertainty reserve margin. For a typical UK motor book with significant BI exposure, we would expect this to be in the range of 2–4 percentage points of the overall BI reserve, though it scales with your portal mix and average claim size.

The point of the margin is not to hold capital against every bad scenario — it is to make the uncertainty visible as a number that can be discussed and either held or released as the Spring 2026 review resolves.

---

## What to do now

Three concrete steps before the MoJ announces anything:

**Fit your scenarios now, not after the announcement.** The temptation is to wait for clarity. The problem is that if the announcement is Scenario C, you will be repricing into a hardening market with a cold model. Build the three scenario models on current data while you have a stable baseline.

**Monitor the leading indicators.** OIC portal claim volumes and average settlement times are published quarterly by MoJ. A meaningful uptick in portal exits to litigation — claims stepping out of the portal into court — would be an early signal of Scenario C dynamics before any formal tariff change. Wire this into your monitoring process now.

**Separate the tariff effect from the cost-of-living effect.** The severity trend in the post-reform period reflects both the tariff suppression and general claims inflation. When the tariff effect partially unwinds (Scenario C), the cost-of-living component does not go away — you are not reverting to 2019 severity, you are reverting to 2019 tariff structure on top of 2026 cost levels. The `SeverityTrendFitter` superimposed inflation decomposition is the right tool for keeping these apart.

The code pattern for scenario-weighted pricing adjustment:

```python
import numpy as np

# Three scenario ultimate loss ratios on BI book
lr_a = 0.68  # status quo + inflation
lr_b = 0.64  # expanded tariff scope
lr_c = 0.79  # partial reversion

weights = np.array([0.45, 0.25, 0.30])
scenario_lrs = np.array([lr_a, lr_b, lr_c])

expected_lr = np.dot(weights, scenario_lrs)
# expected_lr = 0.714

# Stress: 90th percentile across scenarios (weighted)
# Use the spread, not just the weighted mean
spread = np.dot(weights, (scenario_lrs - expected_lr) ** 2) ** 0.5
margin_90 = expected_lr + 1.28 * spread
# margin_90 ~ 0.748 under these illustrative numbers
```

The margin of 3.4 percentage points over the central estimate is not a buffer for catastrophe — it is the cost of the MoJ review being unresolved. Once the review concludes, one of the scenarios collapses to probability 1 and the spread disappears. The margin releases.

That is the honest way to hold it: not as a permanent loading, but as a time-limited uncertainty reserve that has a defined trigger for release.
