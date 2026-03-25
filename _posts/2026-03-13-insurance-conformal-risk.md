---
layout: post
title: "Coverage Is the Wrong Guarantee for Pricing Actuaries"
date: 2026-03-13
categories: [libraries, pricing, uncertainty]
tags: [conformal-prediction, conformal-risk-control, premium-sufficiency, loading-factor, underwriting, selective-risk, CRC, Angelopoulos, ICLR-2024, interval-width, python, motor, polars, insurance-conformal]
description: "Conformal risk control for UK insurance: coverage calibrated to financial shortfall, not miscoverage rate. insurance-conformal - beyond standard intervals."
---

Your conformal prediction interval covers 90% of outcomes. Standard conformal prediction guarantees that. The guarantee is finite-sample valid, distribution-free, and entirely appropriate for many applications.

For insurance pricing, it is the wrong guarantee.

A 10% miscoverage on a portfolio of 5,000 motor policies could be 500 policies each generating a £200 shortfall, or it could be 3 policies with a £50,000 shortfall apiece. The coverage guarantee says nothing about which. The expected monetary loss — the thing that shows up in your loss ratio — depends entirely on which case you are in. But both satisfy the same 90% coverage constraint.

Conformal risk control, from Angelopoulos, Bates, Fisch, Lei and Schuster (ICLR 2024, arXiv:2208.02814), changes what you control. Instead of bounding the probability of being wrong, it bounds the expected magnitude of being wrong. The formal guarantee:

```
E[L(Y, d(X))] ≤ α
```

where `L` is your loss function and `α` is the risk level. Not the probability that `L` exceeds a threshold — the expected value of `L` itself.

[`insurance-conformal-risk`](https://github.com/burning-cost/insurance-conformal-risk) implements this for UK pricing workflows. Three controllers, each targeting a different risk the pricing team cares about.

```bash
pip install insurance-conformal-risk
```

---

## The confusion worth clearing up

`insurance-conformal` and `insurance-conformal-risk` address different questions. They are related but distinct.

`insurance-conformal` controls **coverage probability**: the claim falls within the prediction interval on at least `1 - α` of policies. It finds the smallest interval such that `P(Y ≤ upper) ≥ 1 - α`. The output is an interval. The guarantee is about frequency of inclusion.

`insurance-conformal-risk` controls **expected monetary loss**: the expected shortfall from underpriced policies, as a fraction of premium income, is at most `α`. It finds the smallest loading multiplier λ such that `E[max(claim - λ·premium, 0) / premium] ≤ α`. The output is a loading factor. The guarantee is about financial exposure.

Coverage control suits regulators who think in terms of miscoverage rates. Risk control suits actuaries who think in terms of loss ratios.

You can use both in sequence: `insurance-conformal` for the interval, then `PremiumSufficiencyController` calibrated on the resulting intervals. This is documented in the library's API and produces a loading factor that controls the expected shortfall from claims that breach the conformal upper bound.

---

## PremiumSufficiencyController: the loading factor with a proof

The primary use case: your GBM produces predicted pure premiums. After expenses and profit margin, you charge some multiple of those predictions. The question is what multiple you need to be confident about the shortfall.

The shortfall loss for a single policy is:

```
L(y, λ, p) = max(y - λ·p, 0) / p
```

where `y` is the actual claim, `p` is the base premium, and `λ` is the loading factor. At `λ = 1`, this is the normalised excess of claim over premium. At `λ = 1.3`, the bound is 30% above the model premium. The denominator `p` makes this dimensionless: it is the fraction of premium income lost to underpriced policies.

`PremiumSufficiencyController` finds the smallest `λ*` such that the expected shortfall does not exceed `α`:

```python
from insurance_conformal_risk import PremiumSufficiencyController
import numpy as np

# y_cal: claims on calibration set
# premium_cal: model-predicted pure premiums on calibration set
psc = PremiumSufficiencyController(
    alpha=0.05,   # expected shortfall ≤ 5% of premium income
    B=3.0,        # upper bound on normalised loss; set to max(y_cal / premium_cal)
)
psc.calibrate(y_cal, premium_cal)

print(psc.lambda_hat_)   # e.g., 1.27
print(psc.risk_summary())
# {
#   'lambda_hat': 1.274,
#   'alpha': 0.05,
#   'n_calibration': 2847,
#   'corrected_risk_at_lambda': 0.0498,
#   ...
# }
```

After calibration, apply to new business:

```python
result = psc.predict(premium_new)
# Polars DataFrame:
# base_premium | upper_bound     | safety_loading | lambda_hat
# 423.00       | 538.90          | 0.274          | 1.274
# 891.50       | 1135.77         | 0.274          | 1.274
```

The loading is the same for all policies — `λ*` is a scalar multiplier. The guarantee is marginal over the calibration distribution, not per-segment. We discuss the implications of that below.

### The correction that makes it valid

The calibration algorithm (Algorithm 1 from the paper) applies a finite-sample correction before comparing the empirical risk to `α`:

```
corrected_risk(λ) = (n/(n+1)) · R̂_n(λ) + B/(n+1)
```

`R̂_n(λ)` is the empirical mean shortfall at loading `λ` over the `n` calibration observations. The correction inflates this by the `B/(n+1)` term to account for the unseen test point. This is not cosmetic. Dropping it produces invalid guarantees for calibration sets of the sizes common in insurance pricing (n = 500 to 2,000). At n = 1,000 and B = 3.0, the correction adds 0.003 to every empirical risk estimate — small, but it is the difference between a valid guarantee and a broken one.

Setting `B` correctly matters as much as the correction. `B` is the maximum possible value of the normalised shortfall loss. If a policy's claim can be at most 5 times its premium, then B = 5. Pass `B = np.percentile(y_cal / premium_cal, 99.9)` as a conservative finite empirical bound.

### Where the marginal guarantee can mislead you

The marginal guarantee holds over the calibration distribution. It does not guarantee that any particular segment is controlled.

If your calibration set is 70% standard risks and 30% high-value properties, and all of the shortfall sits in the high-value segment, the average looks fine. `shortfall_report()` surfaces this:

```python
report = psc.shortfall_report(y_cal, premium_cal, n_deciles=10)
# decile | mean_premium | n_obs | mean_shortfall | max_shortfall | pct_underpriced
# 1      | 312          | 285   | 0.012          | 0.31          | 4.2%
# ...
# 10     | 2,847        | 284   | 0.089          | 1.43          | 18.7%
```

If decile 10 has mean shortfall of 8.9% against an `α` of 5%, the marginal guarantee is masking a structural problem. The response is either to fit a segment-specific controller on the high-value book separately, or to extend the calibration set to better represent that segment's behaviour.

---

## IntervalWidthController: paying for efficiency

The second controller addresses a different optimisation problem. You have conformal prediction intervals with valid coverage. They are wider than you would like — conservative models, or heterogeneous risk populations. You want the tightest quantile level such that expected interval width stays below a budget.

This is relevant in any workflow where interval width is the cost. Underwriting might quote on the basis of the upper bound: wider intervals mean higher quoted premiums, which means less competitive pricing on well-priced risks. The width budget is a business constraint.

```python
from insurance_conformal_risk import IntervalWidthController

# upper_cal, lower_cal: conformal intervals on calibration set
iwc = IntervalWidthController(
    width_target=0.15,  # expected normalised width ≤ 0.15 of the scale budget
    scale=5000.0,       # normalisation: max plausible interval width in £
)
# calibrate_from_widths takes an array of (upper - lower) interval widths
widths_cal = upper_cal - lower_cal
iwc.calibrate_from_widths(widths_cal)
print(f"Optimal quantile: {iwc.lambda_hat_:.3f}")  # e.g., 0.923
```

The controller finds the quantile level `λ*` such that the expected normalised width `E[(upper - lower) / scale] ≤ width_target`. Higher quantile = wider intervals = smaller `λ*` minimises width subject to the risk bound.

---

## SelectiveRiskController: underwriting with a guaranteed expected loss

The third controller solves a structurally different problem: selective acceptance. You have a risk score model for underwriting. You want to find the score threshold that guarantees the expected loss on the accepted book is bounded, while keeping at least `xi_min` of the book.

This is SCRC-I from arXiv:2512.12844 (Selective Conformal Risk Control) with a DKW-based correction on the selection rate.

```python
from insurance_conformal_risk import SelectiveRiskController

# Loss function: fraction of accepted policies with large claims
def large_claim_loss(y, scores):
    # Loss = 1 if claim exceeds £10,000; 0 otherwise
    return (y > 10_000).astype(float)

src = SelectiveRiskController(
    alpha=0.08,              # expected large-claim rate ≤ 8% on accepted book
    loss_fn=large_claim_loss,
    xi_min=0.60,             # must accept at least 60% of risks
    delta=0.05,              # DKW confidence level for selection rate bound
)
src.calibrate(y_cal, scores_cal)

decisions = src.predict(scores_new)
# Polars DataFrame:
# score     | accept  | threshold
# 0.812     | True    | 0.654
# 0.493     | False   | 0.654
# 0.721     | True    | 0.654

summary = src.portfolio_summary(y_cal, scores_cal)
# {
#   'threshold': 0.654,
#   'selection_rate': 0.71,
#   'n_accepted': 2,023,
#   'mean_loss_accepted': 0.073,
#   'alpha': 0.08,
#   ...
# }
```

The DKW correction handles the bias in selection rate estimation. When you choose the threshold based on the same calibration set you use to estimate the conditional loss, naive thresholding is biased: you are selecting the risks you already know were fine. The DKW bound sets a conservative lower bound on the true selection rate, preventing the controller from accepting a threshold that only appears to meet `xi_min` due to sampling noise.

A note on the loss function: it must return values in `[0, B]` and must be fixed once calibrated — it is not re-estimated at deployment. The guarantee is that the expected value of `loss_fn(Y, score)` on the accepted population is at most `α`, where the expectation is over the calibration distribution. The practical interpretation depends entirely on what you put in `loss_fn`.

---

## Composing with insurance-conformal

The library is designed to stack with `insurance-conformal` intervals. A typical workflow:

1. Fit a GBM and calibrate conformal intervals using `InsuranceConformalPredictor` from `insurance-conformal`.
2. Collect the upper bounds on the calibration set.
3. Calibrate `PremiumSufficiencyController` using those upper bounds as the "premium" input.
4. The resulting `λ*` is a loading on top of the conformal upper bound that controls the expected shortfall above it.

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal_risk import PremiumSufficiencyController

# Step 1: conformal intervals
cp = InsuranceConformalPredictor(model=fitted_gbm)
cp.calibrate(X_cal, y_cal)
upper_cal = cp.predict_interval(X_cal)["upper"].to_numpy()

# Step 2: risk control on top of intervals
psc = PremiumSufficiencyController(alpha=0.03, B=2.0)
psc.calibrate(y_cal, upper_cal)
print(f"Combined loading: {psc.lambda_hat_:.3f}")
```

This gives you a two-layer guarantee. The conformal interval controls coverage. The risk controller controls the expected shortfall above that interval.

---

## Calibration set size and what happens when it is small

The Hoeffding bound underlying the finite-sample correction requires a calibration set that is representative of the future population. For motor pricing, 1,000 to 2,000 policies is a workable minimum for typical `α` values (0.03 to 0.10). Below 500, the `B/(n+1)` correction becomes material — at n = 200 and B = 3.0, it adds 0.015 to every empirical risk estimate, which forces `λ*` to be noticeably higher than the empirical minimum.

This is not a deficiency of the library. It is a statement about the information content of small samples. If you have 300 calibration observations and want to guarantee expected shortfall ≤ 3%, the honest answer is that you need a larger loading factor than you would with 3,000 observations. The correction makes that cost explicit rather than hiding it.

If calibration fails entirely — `RuntimeError: No lambda controls expected risk at alpha` — the options are: increase `alpha`, extend `lambda_grid` to larger values, or increase `n_cal`. The error message reports the minimum achievable corrected risk so you know which constraint is binding.

---

## Technical summary

- Implements Algorithm 1 from Angelopoulos et al. (ICLR 2024, arXiv:2208.02814)
- Finite-sample correction `(n/(n+1)) · R̂_n(λ) + B/(n+1)` applied by default, not optional
- `SelectiveRiskController` follows SCRC-I from arXiv:2512.12844 with DKW correction on selection rate
- Loss functions: `scaled_shortfall_loss`, `shortfall_loss`, `interval_width_loss`, `coverage_loss`, `xl_recovery_loss`, `exposure_weighted_mean`
- Polars DataFrames for all output
- 121 tests, MIT-licensed, Python 3.10+

```bash
pip install insurance-conformal-risk
```

Source at [github.com/burning-cost/insurance-conformal-risk](https://github.com/burning-cost/insurance-conformal-risk).

---

## See also

- [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal) — coverage probability control, conformal prediction intervals, SCR upper bounds. The companion library that controls frequency of error rather than magnitude. [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)
- [`insurance-evt`](https://github.com/burning-cost/insurance-evt) — when the shortfall you care about is in the extreme tail, EVT gives you return levels and XL layer pricing from the GPD. [Extreme Value Theory for UK Motor Large Loss Pricing](/2026/03/13/insurance-evt/)
- [`insurance-dro`](https://github.com/burning-cost/insurance-dro) — distributional robustness at the optimisation stage. DRO protects the rate recommendation against demand model misspecification; CRC protects the premium against claims model error. Different problems, complementary tools. [Robust Rate Optimisation: Pricing Against Demand Model Misspecification](/2026/03/13/insurance-dro/)
