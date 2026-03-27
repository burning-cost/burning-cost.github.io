---
layout: post
title: "Stop Peeking at Your A/B Test — Or Use a Test That Doesn't Care"
date: 2026-03-24
author: Burning Cost
categories: [pricing, experimentation, libraries, champion-challenger]
description: "Every insurance team checks their champion/challenger results monthly. Every month you look, you inflate the false positive rate. Here is how to do it correctly using sequential testing."
tags: [sequential-testing, msprt, e-values, champion-challenger, a-b-testing, claim-frequency, poisson, type-i-error, insurance-monitoring, motor, python, polars, gipp, fca]
---

Your champion/challenger experiment has been running for three months. You pull the monthly MI report. The challenger is showing a 4% lower claim frequency. Not quite significant yet, but trending nicely. You make a note and check again next month.

You have just done something that silently broke your experiment.

---

## The peeking problem

Fixed-horizon tests — the standard two-sample t-test or Poisson rate comparison you learned at university — come with one precondition that nobody reads: you commit to the sample size before looking at the data. The 5% false positive rate holds only if you look exactly once, at the pre-specified end point.

When you check monthly and stop early if the result looks good, the actual false positive rate is not 5%. Simulations by Johari et al. (2015) showed that peeking at fixed-horizon results just five times at evenly spaced intervals inflates the type I error to around 14%. Check it every single period and you will reach roughly 20–25% false positives across a typical 18-month experiment horizon. On a portfolio where you are running 10 simultaneous champion/challenger tests — which is common on a mid-size UK motor book — you can expect two or three false winners by pure chance.

Insurance makes this worse than most industries. You cannot not look. Claims MI lands monthly. Underwriting committees ask quarterly. Reinsurance treaties need exposure data twice a year. The fix-and-forget experimental discipline that might work in a tech company running a one-week website test is not available to you.

---

## What the mSPRT does differently

The mixture Sequential Probability Ratio Test (mSPRT), formalised by Johari et al. (2022, _Operations Research_), solves this by replacing the p-value with an e-process.

The e-process Lambda_n is a statistic with a remarkable property: under the null hypothesis (no difference between arms), its expectation is at most 1 at every point in time — not just at the end. By Markov's inequality, P(Lambda_n ever exceeds 1/alpha) is at most alpha, for any stopping rule, however opportunistic. You can look daily, stop early when you see a signal, continue past a planned end date because the reinsurer asked a question, and the type I error guarantee still holds exactly.

The "mixture" part is where tau enters. The mSPRT integrates the likelihood ratio over a prior distribution on the effect size (a Gaussian with standard deviation tau on the log-rate-ratio). This makes the test sensitive to effects in a neighbourhood around tau. It is not free sensitivity — if the true effect is much larger or smaller than tau, the test is slower to reject than an oracle test tuned to the exact effect. But it gives you a single parameter to calibrate upfront rather than a power calculation that requires knowing the answer in advance.

You also get an anytime-valid confidence sequence for free. Unlike a standard CI which covers the true value 95% of the time at a fixed n, the CS from Howard et al. (2021) covers the true rate ratio simultaneously for every look you take. When it excludes 1.0, your test has rejected.

---

## Code example

The `insurance-monitoring` library (v0.8.2) implements this in `sequential.py`. The main class is `SequentialTest`. You pass incremental data — new claims and new exposure per reporting period — and it maintains the cumulative state and history internally.

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",       # Poisson rate ratio
    alternative="less",       # challenger has lower frequency than champion
    alpha=0.05,               # threshold Lambda = 20
    tau=0.03,                 # prior std dev on log-rate-ratio: ~3% effect
    max_duration_years=2.0,
    min_exposure_per_arm=500.0,  # car-years before any stopping decision
    futility_threshold=0.1,   # stop if evidence falls below 0.1
)

# Monthly reporting loop
monthly_data = [
    # (champion_claims, champion_exposure, challenger_claims, challenger_exposure)
    (42, 420.0, 38, 418.0),
    (45, 425.0, 36, 421.0),
    (43, 418.0, 34, 415.0),
    # ... more months
]

for month_idx, (cc, ce, lc, le) in enumerate(monthly_data):
    date = datetime.date(2025, 4 + month_idx, 1)
    result = test.update(
        champion_claims=cc,
        champion_exposure=ce,
        challenger_claims=lc,
        challenger_exposure=le,
        calendar_date=date,
    )
    print(result.summary)
    if result.should_stop:
        print(f"Stopping: {result.decision}")
        break
```

The `result.summary` string at each step looks like:

```
Challenger freq 5.8% lower (95% CS: 0.881–0.991). Evidence: 4.2 (threshold 20.0). Inconclusive.
```

Once the evidence value reaches 20 (= 1/0.05), the decision flips to `reject_H0`. If it falls below `futility_threshold` after `min_exposure_per_arm` is reached, it declares `futility` — the challenger is not going to win within the experiment horizon.

To replay a historical experiment from a Polars DataFrame, use the convenience function:

```python
import polars as pl
from insurance_monitoring.sequential import sequential_test_from_df

result = sequential_test_from_df(
    df=monthly_df,                        # one row per reporting month
    champion_claims_col="champ_claims",
    champion_exposure_col="champ_caryears",
    challenger_claims_col="chal_claims",
    challenger_exposure_col="chal_caryears",
    date_col="period_end",
    metric="frequency",
    tau=0.03,
    alpha=0.05,
)
print(result.summary)
```

To inspect the full history and plot Lambda_n over time:

```python
history = test.history()   # Polars DataFrame, one row per update()
# Columns include: lambda_value, log_lambda_value, rate_ratio, ci_lower, ci_upper, decision
```

`log_lambda_value` is the one to plot — it is monotone with evidence, stays bounded, and the threshold becomes a horizontal line at `log(1/alpha)`.

---

## Tau calibration

Tau controls the sensitivity of the test. The mSPRT is most powerful against effects near tau on the log scale. Set tau too small and you will be slow to detect the effects you actually care about; set it too large and the test is poorly calibrated.

Our rule of thumb for UK motor:

| Scenario | Recommended tau |
|---|---|
| Pricing rule change, expected 2–5% frequency lift | 0.03 |
| New telematics scoring model, expected 5–10% lift | 0.05 |
| Major rating factor restructure, expected >10% lift | 0.10 |
| Fraud detection model (frequency proxy) | 0.05 |
| Severity-focused change (repair network, excess) | 0.05 on severity metric |

These are starting points. If you have a strong prior from modelling — e.g., your frequency model predicts a 6.3% lift on the test segment — use that directly: tau = 0.063. The test is robust to moderate misspecification of tau; a factor of 2x wrong does not collapse the type I error guarantee, it just costs power.

---

## Insurance-specific notes

**Exposure units.** The Poisson mSPRT uses car-years as the exposure denominator. Be consistent: do not mix policy-years and car-years within a test, and do not use written premium as a proxy for exposure. For fleets or commercial lines where one policy covers multiple vehicles, use vehicle-years.

**IBNR on long-tail lines.** For motor liability, reported claims at 3 months are materially incomplete. Running a frequency test on 3-month-old accident periods treats IBNR as if it does not exist. We recommend either: (a) restricting to accidental damage and windscreen claims for early-stage tests where you need a fast signal; or (b) using the `calendar_date` parameter and setting `min_exposure_per_arm` generously — 12 months of exposure minimum for liability-heavy tests, to give claims adequate time to emerge before the test can trigger a decision.

**Loss ratio metric.** For `metric="loss_ratio"`, the library multiplies the frequency and severity e-values. This is valid because e-values compose multiplicatively under independence. Passing pre-calculated `champion_severity_sum` and `champion_severity_ss` (sums of log(claim cost) and log(claim cost)^2 respectively) triggers the log-normal severity component. We find this most useful when testing repair network or excess changes where frequency is stable but severity is the signal.

**GIPP compliance.** FCA GIPP (General Insurance Pricing Practices) requires that pricing changes affecting renewal premiums are documented with supporting evidence of their effect on risk. A sequential test log — Lambda_n over time, the anytime-valid CS, and the declared stopping reason — is cleaner audit evidence than a batch t-test run at an arbitrary point after the experiment started. Store `test.history()` to your results warehouse at the end of each reporting period.

---

## When not to use this

The CLT approximation inside `_poisson_msprt` requires at least 5 claims in each arm before it returns a non-trivial value. With fewer than 5 cumulative claims per arm, Lambda_n is fixed at 1.0 — evidence level 1, indistinguishable from the null. If you are testing on a niche product with 20 new policies a month, you will not accumulate 5 claims per arm for a very long time, and the mSPRT will give you nothing useful.

Rough minimums before a frequency test has any realistic chance of rejecting within 2 years:

- At least 50–100 claims per arm total at experiment end (depends on effect size)
- At least 500 car-years per arm before any stopping rule fires (set via `min_exposure_per_arm`)
- Realistic expected effect size above 3% — if your model predicts a 0.5% frequency improvement, you need a much longer experiment regardless of method

If your book is too thin for sequential testing, you need either a longer horizon, a wider segment definition, or a different primary outcome metric (something you can measure faster than attritional claim frequency).

The mSPRT also assumes the units within each arm are independent. Shared policy-level random effects — a fleet, a broker block — can deflate the estimated variance and produce overconfident Lambda_n values. If your challenger arm is dominated by a single large fleet account, the test statistics are not reliable.

---

## The operational case

The hard thing about peeking is not the statistics — it is that the people asking "can we look?" are not wrong to want an answer. You are running an experiment on live policies. Decisions compound. The alternative to peeking is not the principled fixed-horizon test; it is pressure to declare a winner anyway, on whatever basis is available at the time.

Sequential testing does not require you to change how often you look at data. It requires you to change the test statistic you use when you do. Monthly MI, quarterly reviews, ad-hoc checks when a line goes adverse — all of that is still permitted. The Lambda_n e-process is valid at every one of those checkpoints simultaneously.

[`insurance-monitoring`](/insurance-monitoring/) is on [GitHub](https://github.com/burning-cost/insurance-monitoring). The `sequential` module has no external dependencies beyond polars, scipy and numpy.

- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) — the `insurance-monitoring` library's broader framework: PSI, segmented A/E, and Gini drift detection alongside the sequential testing module
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/) — how to register champion/challenger test results in the MRM inventory with a machine-readable audit trail
