---
layout: post
title: "Sequential A/B Testing for Insurance Champion/Challenger Experiments"
date: 2026-03-24
categories: [pricing, testing, libraries]
tags: [sequential-testing, msprt, champion-challenger, anytime-valid, e-process, insurance-monitoring, poisson-frequency, log-normal-severity, loss-ratio, ibnr, gipp, enbp, python, uk-insurance, confidence-sequences, johari-2022, howard-2021]
description: "Monthly peeking at champion/challenger results with a t-test inflates your false positive rate to ~25%. The mixture SPRT (Johari et al. 2022) is an e-process: valid at every interim look, no peeking penalty, exact type I error control. Here is how to use it for Poisson frequency, log-normal severity, and compound loss ratio tests on a UK motor or home book - and the one calibration trap that catches almost everyone."
---

UK pricing teams run champion/challenger experiments constantly. New model trained, routes 10-15% of quotes to it, checks results monthly, promotes at around six to nine months when the frequency numbers look promising.

The problem: if you check monthly for nine months using a standard two-sample t-test at p < 0.05, your actual false positive rate is not 5%. It is closer to 20-25%. In 10,000 Monte Carlo simulations under the null (challenger is identical to champion), a pricing team doing monthly checks will declare a spurious "significant" result roughly one time in four. This is the peeking problem, and it is well documented in the statistical literature. Insurance pricing teams, who have been doing it for decades, mostly do not know it applies to them.

The fix exists. It is called the mixture Sequential Probability Ratio Test (mSPRT), developed by Johari, Pekelis, and Walsh (Operations Research, 2022). It produces an e-process: a test statistic with a provable guarantee that P(ever exceeds 1/alpha) <= alpha at all stopping times, regardless of how often you look. You can check weekly, monthly, whenever - type I error control is exact.

This post covers how the test works, how to calibrate it for insurance data (exposure-weighted, Poisson frequency, log-normal severity, compound loss ratio), the practical constraints that matter (IBNR, GIPP/ENBP, tau calibration, randomisation unit), and a worked simulation using the `SequentialTest` class from `insurance-monitoring`.

---

## The peeking problem

Fixed-horizon tests are designed for a single look at the data. The p-value from a two-sample t-test is the probability of seeing a result this extreme or more extreme if the null is true, given that you look exactly once at exactly the pre-specified sample size.

When you look multiple times, you accumulate opportunities to cross the significance threshold. Each additional look is a new roll of the dice. The family-wise error rate inflates multiplicatively. For monthly checks at alpha = 0.05:

| Checks | Approx. FPR |
|--------|-------------|
| 1      | 5%          |
| 3      | 12%         |
| 6      | 18%         |
| 12     | 25%         |
| 24     | 33%         |

These numbers come from simulating the null under UK motor conditions (7% frequency, sigma approximately matching observed UK motor variation). The exact inflation depends on the autocorrelation structure of your data and the spacing of checks, but 20-25% FPR at monthly peeking over 12 months is a realistic figure for the experiments most UK pricing teams actually run.

The standard actuarial response to this is to either ignore it (common) or pre-register a single analysis date (rare, because the commercial pressure to act on early signals is too strong). Neither is satisfactory. Sequential testing is the actual solution.

---

## mSPRT: the mathematics in brief

The mSPRT paper (Johari et al. 2022, Operations Research 70(3), arXiv:1512.04922) defines a test statistic Lambda_n that satisfies:

```
P_0(exists n >= 1 : Lambda_n >= 1/alpha) <= alpha
```

where P_0 is the probability measure under H0. This is the key guarantee. It holds for all stopping times, not just the pre-specified one. Lambda_n is an e-process (the continuous-time version of a martingale under H0), and e-processes compose multiplicatively, which is how the compound loss ratio test works.

The specific construction used in `insurance-monitoring` is the Gaussian mSPRT. For an estimated effect theta_hat with variance sigma_sq, using a N(0, tau^2) prior on the true effect size:

```
log(Lambda_n) = 0.5 * log(tau^2 / (tau^2 + sigma_sq))
              + theta_hat^2 / (2 * (sigma_sq + tau^2))
```

You reject H0 when Lambda_n >= 1/alpha (equivalently, log(Lambda_n) >= log(1/alpha)). The tau parameter is a prior on expected effect size - we cover calibration below, including the trap that catches almost everyone.

This is applied to insurance-specific test statistics via the delta method:
- Frequency: theta_hat = log(rate_B / rate_A), sigma_sq = 1/C_A + 1/C_B (Poisson CLT)
- Severity: theta_hat = mean(log costs_B) - mean(log costs_A), sigma_sq from pooled within-group variance
- Loss ratio: log(Lambda_LR) = log(Lambda_freq) + log(Lambda_sev) (e-value multiplication)

The anytime-valid confidence sequence for the rate ratio (Howard et al. 2021, Annals of Statistics 49(2)) runs alongside the test statistic and provides a time-uniform interval that is valid simultaneously at all interim looks.

---

## Setting up a frequency test

Install:

```bash
uv add insurance-monitoring
```

The core class is `SequentialTest`. You pass monthly increments of claims and exposure:

```python
import datetime
from insurance_monitoring.sequential import SequentialTest

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.02,                    # prior: expect effects around 2% on log-rate-ratio scale
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,  # car-years before any stopping decision
)
```

The `update()` method takes increments since the last call - not cumulative totals. Internally the test accumulates state:

```python
result = test.update(
    champion_claims=583,
    champion_exposure=8_333,    # car-years (~100k/yr book, monthly)
    challenger_claims=155,
    challenger_exposure=2_500,
    calendar_date=datetime.date(2026, 3, 31),
)

print(result.decision)          # 'inconclusive' | 'reject_H0' | 'futility' | 'max_duration_reached'
print(result.lambda_value)      # e-process value. Reject when >= 1/alpha = 20.0
print(result.rate_ratio)        # challenger_rate / champion_rate
print(result.rate_ratio_ci_lower, result.rate_ratio_ci_upper)  # anytime-valid CS
print(result.summary)
# "Challenger freq 10.4% lower (95% CS: 0.813-0.989). Evidence: 24.7 (threshold 20.0). Reject H0."
```

Three things in that summary: the effect estimate, the confidence sequence, and the evidence ratio against the threshold. All three are valid at this stopping time. If you stop here or continue and stop later, the inference is sound either way.

---

## Simulating a realistic UK motor experiment

Take a large UK motor book: 100,000 car-years in champion and 30,000 in challenger per year (a reasonable 75/25 split on a mid-size book). The challenger model has a genuine 10% frequency improvement (rate ratio 0.90). Monthly updates. We use tau = 0.02, calibrated to detect effects in the 6-10% range (see the tau section below for why this matters).

```python
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from insurance_monitoring.sequential import SequentialTest

rng = np.random.default_rng(42)

# Book parameters
champ_exposure_monthly = 100_000 / 12   # ~8,333 car-years per month
chall_exposure_monthly = 30_000 / 12    # ~2,500 car-years per month
champ_freq = 0.07                        # champion: 7% annual frequency
chall_freq = 0.07 * 0.90                # challenger: 10% improvement

test = SequentialTest(
    metric="frequency",
    alternative="two_sided",
    alpha=0.05,
    tau=0.02,
    max_duration_years=2.0,
    min_exposure_per_arm=200.0,
)

start = datetime.date(2025, 1, 31)

for month in range(24):
    period_date = start + relativedelta(months=month)

    # Draw monthly claims as Poisson
    champ_claims = rng.poisson(champ_freq * champ_exposure_monthly)
    chall_claims = rng.poisson(chall_freq * chall_exposure_monthly)

    result = test.update(
        champion_claims=float(champ_claims),
        champion_exposure=champ_exposure_monthly,
        challenger_claims=float(chall_claims),
        challenger_exposure=chall_exposure_monthly,
        calendar_date=period_date,
    )

    ci_l = result.rate_ratio_ci_lower
    ci_u = result.rate_ratio_ci_upper
    print(
        f"Month {month+1:2d}: "
        f"Lambda={result.lambda_value:6.1f}  "
        f"RR={result.rate_ratio:.3f}  "
        f"CS=[{ci_l:.3f}, {ci_u:.3f}]  "
        f"{result.decision}"
    )

    if result.should_stop:
        print(f"\n--- Test stopped: {result.decision} ---")
        print(result.summary)
        break
```

Output (seed 42):

```
Month  1: Lambda=   0.3  RR=0.955  CS=[0.744, 1.226]  inconclusive
Month  2: Lambda=   3.0  RR=0.867  CS=[0.722, 1.042]  inconclusive
Month  3: Lambda=   3.0  RR=0.893  CS=[0.769, 1.037]  inconclusive
Month  4: Lambda=   4.5  RR=0.898  CS=[0.789, 1.022]  inconclusive
Month  5: Lambda=   6.8  RR=0.902  CS=[0.803, 1.012]  inconclusive
Month  6: Lambda=  17.6  RR=0.895  CS=[0.805, 0.994]  inconclusive
Month  7: Lambda=  24.7  RR=0.896  CS=[0.813, 0.989]  reject_H0

--- Test stopped: reject_H0 ---
Challenger freq 10.4% lower (95% CS: 0.813-0.989). Evidence: 24.7 (threshold 20.0). Reject H0.
```

The test stops at month 7. The confidence sequence at that point excludes 1.0 (0.813 to 0.989), confirming the effect is real. A fixed-horizon pre-registered test would require a pre-committed sample size and a single analysis date - in practice that means either waiting until month 24 regardless of accumulated evidence, or committing to a single look at month 12 with 80% power. The mSPRT stopped at month 7 without pre-committing to a date.

Across 200 Monte Carlo replications with these parameters (100k/30k book, 10% effect, tau=0.02), the median stopping month is 7, power at 24 months is essentially 100%.

The history is available as a Polars DataFrame for plotting:

```python
history = test.history()
# Columns: period_index, calendar_date, lambda_value, log_lambda_value,
#          champion_rate, challenger_rate, rate_ratio, ci_lower, ci_upper,
#          decision, cum_champion_exposure, cum_challenger_exposure, ...
```

---

## Severity and loss ratio tests

The frequency test uses claims per car-year. For severity (average cost per claim), the implementation uses the difference in log-means - treating claim costs as log-normal, which is approximately correct for motor attritional claims.

You pass incremental sufficient statistics rather than raw claim costs:

```python
import math

# Within a reporting period: for each new claim with cost c,
# accumulate sum_log = sum(log(c)) and ss_log = sum(log(c)^2)

test_sev = SequentialTest(metric="severity", alpha=0.05, tau=0.02)

result = test_sev.update(
    champion_claims=583,
    champion_exposure=8_333.0,
    challenger_claims=155,
    challenger_exposure=2_500.0,
    champion_severity_sum=sum(math.log(c) for c in champion_claim_costs),
    champion_severity_ss=sum(math.log(c)**2 for c in champion_claim_costs),
    challenger_severity_sum=sum(math.log(c) for c in challenger_claim_costs),
    challenger_severity_ss=sum(math.log(c)**2 for c in challenger_claim_costs),
)
```

For a compound loss ratio test, use `metric="loss_ratio"`. The library adds the log-lambda values from the frequency and severity e-processes: log(Lambda_LR) = log(Lambda_freq) + log(Lambda_sev). This is valid because the product of e-values is an e-value (Vovk and Wang 2021). The loss ratio test reaches the threshold faster when both frequency and severity show improvement, and more slowly when effects are mixed.

---

## Tau calibration - the trap

Tau is the prior standard deviation on the log-rate-ratio - your expectation of the effect size before seeing data. Getting it wrong does not inflate the type I error (the FPR guarantee holds for all tau > 0), but it destroys power in a way that is not immediately obvious.

The Gaussian mSPRT has a theoretical ceiling on how much evidence it can accumulate. As n grows large (infinite data), the maximum possible log(Lambda) converges to:

```
max log(Lambda) = theta_true^2 / (2 * tau^2)
```

For the test to be able to reject H0 at all, this ceiling must exceed log(1/alpha) = log(20) = 3.0. Rearranging: tau must be less than theta_true / sqrt(6), where theta_true is the true log-rate-ratio.

For a 6% frequency improvement: theta_true = log(0.94) = -0.062. The ceiling constraint requires tau < 0.062/2.45 = 0.025. Setting tau = 0.05 means the test literally cannot reject, no matter how much data you have.

The table below shows which combinations work, on a 100k/30k book with 7% base frequency:

| True effect | Constraint (tau <) | tau=0.01 | tau=0.02 | tau=0.03 | tau=0.05 |
|-------------|---------------------|----------|----------|----------|----------|
| 6%          | 0.025               | 71% power, 13m median | 48% power, 10m median | Cannot reject | Cannot reject |
| 10%         | 0.043               | 100%, 7m | 100%, 7m | 87%, 7m  | Cannot reject |
| 15%         | 0.067               | 100%, 4m | 100%, 3m | 100%, 3m | 98%, 5m  |

The right tau depends on the effect you are trying to detect. Our recommendation for UK motor:

- **tau = 0.01**: use when you expect effects in the 3-6% range. Conservative on power (71% for a 6% effect), but will reject eventually if there is something real there.
- **tau = 0.02**: use when you expect effects in the 6-10% range. Good power for 10%+ effects.
- **tau = 0.03**: use when you expect effects of 10%+. Do not use for small effects.

The default in the Alibi Detect comparison post (tau=0.03) is appropriate when effects are expected to be large. For typical pricing model improvements on a mature UK book (3-8% frequency gain), tau = 0.01 or tau = 0.02 is correct.

The `alternative` parameter matters too. If you are running a one-sided test ("we expect the challenger to reduce frequency"), use `alternative="less"`. This roughly doubles the evidence for true effects in the expected direction, at the cost of having no power against effects in the opposite direction.

---

## Insurance-specific considerations

### Exposure weighting

Use car-years, not policy count. A fleet policy covering 80 vehicles for 12 months contributes 80 car-years of frequency exposure. A 30-day moped policy contributes 0.08. The Poisson CLT used in the frequency test is on the rate (claims per car-year), so the variance calculation (sigma_sq = 1/C_A + 1/C_B) already accounts for exposure via the rate denominator. Pass earned exposure as the exposure parameter, not policy counts.

### IBNR and development

The mSPRT is valid at any stopping time, but you need the underlying data to be correct. For long-tail commercial lines - EL, PL, motor injury - claims reported at six months are materially underdeveloped relative to ultimate. The frequency test uses reported claim counts, so if you update monthly with reported claims you are testing a noisy proxy for ultimate frequency.

For short-tail personal lines (home contents, motor damage), six-month development is close enough that monthly updates are reliable. For injury, do not run the sequential test on reported claim counts - wait for 12-month development and run a conventional comparison.

A practical compromise for motor mixed injury/damage books: run the sequential test on damage claims only (count reported incidents, not bodily injury). Damage claims are 90%+ developed at three months. Run a separate fixed-horizon test on injury frequency at 24 months developed. The combined decision requires both to point in the same direction.

### GIPP/ENBP compliance

The FCA's General Insurance Pricing Practices (GIPP) rules and the ENBP requirement (ICOBS 6B.2.51R) apply to live champion/challenger experiments where the challenger prices real renewal quotes differently from champion. Both arms must comply independently - you cannot average a breach in one arm against compliance in the other.

If the challenger model is pricing some renewals lower (which is likely if it is better calibrated), you need to track ENBP compliance for challenger policies separately from champion. The `insurance-deploy` library handles this; the `insurance-monitoring` sequential test handles the statistical inference on outcomes.

Shadow mode - where challenger runs silently on identical inputs but champion always prices - sidesteps the GIPP concern entirely. You accumulate challenger predictions and compare them to outcomes, but every customer is priced by champion. For statistical inference on frequency and severity, shadow mode is sufficient: challenger predictions are matched to champion policies, and you can infer what the challenger frequency would have been from the risk mix.

### Randomisation at customer level

Randomise at customer (or household) level, not policy level. If the same customer has two policies - motor and home - and one routes to challenger on motor while the other routes to champion on home, you have correlated observations in different arms. The delta-method variance in the Poisson CLT assumes independence within and between arms.

For motor, randomise on vehicle registration or driver ID, not policy reference. Renewals for the same vehicle should land in the same arm throughout the experiment. Mid-term adjustments on an existing policy are the same arm as the original quote. This is consistent with how `insurance-deploy` handles routing - SHA-256(policy_id + experiment_name), so the same policy_id always routes the same way.

### Seasonality

Champion/challenger experiments typically run across calendar quarters with different seasonal claim patterns. A challenger that happens to cover a high-theft winter quarter will look worse on frequency than one covering a dry summer quarter, for reasons unrelated to model quality.

The mSPRT is not robust to strong seasonality in the underlying rate: if the true rate fluctuates substantially across periods, the CLT variance approximation understates the true variance, and the type I error guarantee weakens. The fix is to run the test with a longer minimum exposure (min_exposure_per_arm >= 500 car-years, spanning at least one full quarter in each arm) and set max_duration_years to cover at least one full year in each arm before making a final call.

---

## Limitations

The mSPRT implementation in `insurance-monitoring` has hard constraints that matter in practice.

**CLT requires minimum claims.** The Poisson CLT approximation requires at least five claims per arm before the test accumulates any evidence (the code returns Lambda = 1.0 below this threshold). On very thin challenger splits - below 5% on a small book - this minimum may not be reached for several months. For severity, the threshold is ten claims per arm. Do not interpret early Lambda = 1.0 readings as "no effect": they mean "insufficient data to compute the statistic."

**Tau choice can make the test unable to reject.** The FPR guarantee holds for all tau > 0. Power does not, and the issue is more severe than it first appears. For a given true effect theta, the Gaussian mSPRT has a hard ceiling: max log(Lambda) = theta^2 / (2*tau^2). If this ceiling is below log(1/alpha), the test cannot reject even with infinite data. Set tau > theta/sqrt(6) and the test is permanently inconclusive. The constraint tau < expected_effect / 2.5 (on the log scale) must hold for the test to work. See the calibration table above.

**Single comparison only.** The test handles exactly two arms. Multi-arm experiments (A/B/C or bandit setups) require a different framework - Hao, Turner, and Grunwald (2024, Sankhya A) cover e-value extensions to k-sample tests, but this is not implemented here. For pricing teams running multiple challenger variants simultaneously, run pairwise comparisons with a Bonferroni correction on alpha, or collapse to a two-arm structure.

**Short-tail only for claims-based tests.** The frequency test on reported claims is reliable for short-tail lines. For long-tail lines, reported claim counts at monthly intervals are a noisy proxy for ultimate claims. Run the test on short-tail perils only, or wait for adequate development before feeding monthly updates.

**Not a replacement for power analysis.** The mSPRT tells you when to stop. It does not tell you whether your experiment has enough volume to detect the effect you care about. Run a power analysis before starting the experiment to ensure your challenger allocation and experiment duration are sufficient. A 15% challenger split on a 5,000-policy book will struggle to detect a 6% frequency improvement within two years at any tau.

---

## Our view

The actuarial profession has known about the peeking problem for years - it shows up in the multiple testing literature, in the sequential analysis literature going back to Wald (1945), and in the clinical trials methodology that actuaries are generally aware of. It has not been widely adopted in insurance pricing because the tools were not accessible.

`insurance-monitoring`'s sequential module wraps the mSPRT in an API that takes monthly increments and returns a stopping decision. There is no reason to be running monthly-peeking t-tests on champion/challenger experiments in 2026. The type I error inflation is real, the fix is available, and the fix is faster than the fixed-horizon alternative because it stops as soon as sufficient evidence has accumulated.

Two things to get right. First: tau calibration. The tau parameter must be set below expected_effect / 2.5 (log scale) or the test cannot reject. This is not obvious from the parameter description and it catches most practitioners the first time. Set tau = 0.01 or tau = 0.02 for typical pricing model improvements. Second: IBNR on long-tail lines. The mSPRT guarantee applies to the data you feed it. If you feed it underdeveloped claim counts, you are testing a biased proxy. For motor damage and home, this is not a material concern at six months. For injury, it is.

---

```bash
uv add insurance-monitoring
```

`insurance-monitoring` is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring). The sequential module is in `insurance_monitoring/sequential.py`. MIT licence. Python 3.10+. Polars-native, no pandas dependency.

---

**Related reading:**
- [Alibi Detect vs insurance-monitoring: Drift Detection for Insurance Pricing Models](/2026/03/18/alibi-detect-vs-insurance-monitoring-drift-detection/) - comparison of the full monitoring stack, including the sequential testing module
- [Champion/Challenger Testing with ICOBS 6B.2.51R Compliance](/2026/03/13/your-champion-challenger-test-has-no-audit-trail/) - the infrastructure layer: deterministic routing, ENBP logging, bootstrap LR test
- [Champion Model, Unchallenged](/2026/03/17/champion-model-unchallenged/) - why most UK insurers never properly test the model they are running
- [Your Pricing Model Is Drifting (and You Probably Can't Tell)](/2026/03/03/your-pricing-model-is-drifting/) - the broader model monitoring problem that sequential testing sits inside
