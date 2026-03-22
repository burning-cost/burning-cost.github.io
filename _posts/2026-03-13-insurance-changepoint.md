---
layout: post
title: "When Did Your Loss Ratio Actually Change?"
date: 2026-03-13
categories: [libraries, pricing, monitoring]
tags: [changepoint, BOCPD, PELT, regime-detection, Bayesian, Poisson-Gamma, exposure-weighted, Consumer-Duty, FCA, Whiplash-Reform, Ogden, GIPP, motor, insurance-dynamics, python]
description: "Bayesian Online Changepoint Detection for UK insurance loss ratios. Poisson-Gamma conjugates, regulatory event priors, Consumer Duty evidence pack - Python."
---

There is a conversation that happens in almost every UK personal lines pricing team, usually in a quarterly experience review. Someone puts up the monthly frequency chart. There is a clear kink, somewhere around Q2 of last year. One actuary says it looks like something changed around April. Another says maybe June. A third points out that the storms in February could have influenced it. The team agrees that the trend has probably shifted, applies a judgement-based adjustment to the rate filing, and documents it as "technical pricing review, Q2 inflection noted."

This is the entire monitoring methodology for detecting regime changes at most UK insurers. An eyeball test, a committee, and a sentence in a Word document.

The problem is not that actuaries are bad at spotting patterns — they are not. The problem is that eyeballing gives you no posterior probability, no uncertainty on the break location, no formal detection threshold, and no audit trail that satisfies an FCA reviewer. Under Consumer Duty (PRIN 2A.9, effective July 2023), firms must evidence that pricing models are being monitored for continued fair value delivery. "We looked at the chart and it seemed fine" is not evidence in any meaningful sense.

[`insurance-dynamics`](https://github.com/burning-cost/insurance-dynamics) implements formal Bayesian change-point detection for insurance pricing time series. MIT-licensed, on PyPI.

```bash
uv add insurance-dynamics
```

---

## What BOCPD actually gives you

The core algorithm is Bayesian Online Changepoint Detection (BOCPD) from Adams & MacKay (2007), arXiv:0710.3742. The original paper considers a stream of observations and maintains, at each time step, a posterior distribution over the current run length — how many periods have elapsed since the last regime change. A short run length means we think a change happened recently. A long run length means we think we are still in the same regime we started in.

At each new period t, the algorithm updates this posterior and outputs P(changepoint at t | all data up to t). Not a flag. Not a threshold breach. A probability. If the frequency data for Q3 2023 generates P = 0.91, the algorithm is telling you: conditional on everything we have seen, there is a 91% posterior probability that the data-generating process changed this quarter.

This matters for documentation. There is a qualitative difference between "the actuary reviewed the chart and identified a possible inflection" and "the BOCPD algorithm assigned P = 0.91 to a regime change in Q3 2023, using an exposure-weighted Poisson-Gamma conjugate model with Ogden and Whiplash Reform event priors." The second statement is auditable, reproducible, and defensible.

The online component means this runs in real time. You do not wait for enough data to perform a retrospective analysis. You feed each quarter's claims and exposure as it arrives, and the posterior updates. If a regime change is happening, you see it as it happens, not six months later in the experience review.

---

## The insurance gap in existing tools

The standard Python ecosystem for change-point detection is `ruptures`, with some additional coverage from R's `changepoint` package and the `changepoint-online` Python library. All of these are general-purpose. None of them understand insurance exposure.

This matters for a specific reason. Insurance frequency data is not a sequence of uniformly-weighted observations. Q1 might have 500 earned vehicle-years; Q4 of the same year might have 50,000. A naive Poisson model applied to raw claim counts treats these quarters as equally informative. They are not. A quarter with 50,000 vehicle-years is 100 times more informative than a quarter with 500. If you do not weight by exposure, you will over-react to regime signals from thin periods and under-react to signals from heavy periods.

The natural conjugate model for insurance frequency data is Poisson-Gamma: claim counts are Poisson with rate λ × exposure, where λ is drawn from a Gamma prior. Under this model, the conjugate update at each time step is analytic — you update the Gamma shape and rate parameters from the observed claims and exposure without MCMC. The BOCPD algorithm extends to this conjugate structure straightforwardly, and the exposure weight enters the Poisson likelihood directly.

`insurance-dynamics` implements the exposure-weighted Poisson-Gamma BOCPD. To our knowledge, no existing Python package handles this. `ruptures` is minimum-description-length based and does not have a Poisson-Gamma cost model. `changepoint-online` implements the exponential family BOCPD but without insurance-aware exposure weighting or UK-specific priors. The gap is real, and it is not a packaging gap — it is a modelling gap.

---

## Detecting frequency regime changes

The `FrequencyChangeDetector` is the main class for claims frequency monitoring. You supply claim counts, exposure (vehicle-years or policy-years), and optionally period labels.

```python
from insurance_dynamics.changepoint import FrequencyChangeDetector
import numpy as np

detector = FrequencyChangeDetector(
    hazard=0.1,       # Prior probability of a changepoint at any period
    alpha0=1.0,       # Gamma prior shape for claim rate
    beta0=10.0,       # Gamma prior rate for claim rate
    threshold=0.5,    # Posterior probability threshold to flag a break
)

# Quarterly data: 20 quarters of BI frequency
claims = np.array([41, 38, 44, 39, 42, 37, 40, 43, 38, 41,
                   28, 25, 31, 27, 29, 26, 30, 28, 27, 25])
exposure = np.array([920, 910, 935, 905, 915, 900, 925, 930, 910, 920,
                     890, 905, 920, 895, 910, 885, 915, 900, 895, 880])
periods = [f"Q{q}" for q in range(1, 21)]

result = detector.fit(claims=claims, exposure=exposure, periods=periods)

print(f"Detected breaks: {result.n_breaks}")
for brk in result.detected_breaks:
    print(f"  {brk}")
# DetectedBreak(period=Q11, prob=0.847)
```

The `ChangeResult` object carries the full posterior: `changepoint_probs` is a length-T array of P(changepoint at t) for every period, `run_length_probs` is the T×T matrix of run-length posteriors, and `detected_breaks` is the list of periods where the posterior exceeded the threshold.

The Gamma prior parameters `alpha0` and `beta0` encode your prior belief about the claim rate. `alpha0=1.0, beta0=10.0` means a prior mean rate of α/β = 0.10 claims per vehicle-year — roughly in the right ballpark for UK motor BI frequency. These are weakly informative priors; after even a few quarters of data the likelihood dominates. If you have a historical estimate of your frequency you can set them to reflect it: `alpha0=2.0, beta0=25.0` implies a prior mean of 0.08 with tighter concentration.

For severity monitoring, `SeverityChangeDetector` uses a Normal-Normal conjugate (or Normal-InverseGamma for unknown variance) applied to log-transformed claim amounts. The API is identical.

---

## UK regulatory event priors

Bayesian change-point detection is more powerful when you have prior knowledge about when breaks might occur. UK insurance has a well-documented history of regulatory and catastrophic events that create structural breaks in loss experience. Encoding these as informative priors is not cheating — it is correct statistical practice, and it makes the model faster to detect known-type breaks and more cautious about spurious signals in non-event periods.

`UKEventPrior` encodes ten known break dates as a hazard function multiplier. In the base BOCPD model, the hazard is a constant: P(changepoint at t) = h for all t. With event priors, the hazard spikes at known break dates. The algorithm becomes more sensitive at those dates and retains its standard sensitivity elsewhere.

```python
from insurance_dynamics.changepoint import FrequencyChangeDetector, UKEventPrior

prior = UKEventPrior(
    events=["whiplash_reform", "ogden_2017", "ogden_2021",
            "gipp", "covid_q1_2020", "storm_ciara"],
    spike_multiplier=3.0,   # hazard × 3.0 at event periods
    spike_width=1,          # apply spike to the event quarter ± 1 period
)

detector = FrequencyChangeDetector(
    hazard=0.1,
    uk_event_prior=prior,
    threshold=0.5,
)
```

The ten encoded events are: Ogden rate change (March 2017), Ogden partial correction (August 2019), Whiplash Reform Act (Royal Assent November 2018, OIC portal effective May 2021), GIPP pricing rules (PS21/11, effective January 2022), COVID lockdowns (Q1 and Q2 2020), Storm Ciara (February 2020), Storm Dennis (February 2020), Storm Eunice (February 2022), and FCA Motor Finance review (2024). You can add custom events for portfolio-specific events — a major panel change, a broker acquisition, a claims handling outsource.

The spike multiplier of 3.0 means the prior probability of a changepoint is three times higher at event periods than non-event periods. This is not aggressive. The posterior is still driven by the data; if the data shows no inflection at the whiplash reform date, the algorithm will not flag a break there. What the prior does is lower the evidence threshold: the data only needs to be moderately consistent with a break at a known-event period, whereas at an arbitrary period it needs to be more compelling.

The practical benefit is faster detection. A retrospective analysis of motor BI frequency from 2021 onwards benefits substantially from encoding the May 2021 OIC portal launch as an event prior. The actual effect on claim frequency was real — the portal and fixed whiplash tariff reduced minor BI claims frequency — and a model that knew to look for it would have flagged it within two quarters rather than six.

---

## Streaming updates

One of the key properties of BOCPD is that it is genuinely online. You do not refit the model from scratch each quarter. You call `.update()` with the new period's data and the posteriors update incrementally.

```python
# Initial fit on historical data
result = detector.fit(
    claims=historical_claims,
    exposure=historical_exposure,
    periods=historical_periods,
)

# Quarter ends; new data arrives
new_result = detector.update(
    claims=np.array([22]),
    exposure=np.array([870]),
    period="Q21",
)

print(f"P(changepoint at Q21): {new_result.changepoint_probs[-1]:.3f}")
# P(changepoint at Q21): 0.127
# No alert — consistent with ongoing low-frequency regime
```

This is the operational pattern for a monitoring dashboard. Each quarter, feed in the new data, read the posterior probability, compare to threshold, and document the output. No refit, no analyst involvement unless the probability exceeds your alert threshold. The methodology is running continuously in the background.

For a Consumer Duty evidence pack, this gives you a time-stamped table of posterior probabilities, with explicit detection thresholds, covering every period since deployment. If the FCA asks how you monitored motor BI frequency between January 2023 and March 2025, you hand them a table.

---

## Retrospective analysis with PELT

Online BOCPD tells you when something changed as it happens. `RetrospectiveBreakFinder` tells you, given the full history, where the breaks most probably were and how uncertain we are about their timing.

The algorithm is PELT (Penalised Exact Linear Time) from Killick, Fearnhead & Eckley (2012), JASA 107(500). PELT finds the globally optimal segmentation of a time series into piecewise homogeneous segments in O(n) time — faster than the O(n²) naive search and exact rather than approximate. The penalty parameter controls the trade-off between model fit and number of segments.

The `insurance-dynamics` enhancement is bootstrap confidence intervals on break locations. PELT gives you a point estimate: "the break was at period 14." The bootstrap CI tells you the uncertainty: "95% CI is [12, 16]." This is important when break timing matters for decisions — for example, determining which data falls into the new regime for a model refit.

```python
from insurance_dynamics.changepoint import RetrospectiveBreakFinder

finder = RetrospectiveBreakFinder(
    penalty=3.0,        # BIC-style penalty; higher -> fewer breaks
    model="poisson",    # Poisson cost function for count data
    n_bootstraps=500,   # Bootstrap resamples for CI estimation
    ci_level=0.95,
)

break_result = finder.fit(
    claims=claims,
    exposure=exposure,
    periods=periods,
)

print(f"Breaks found: {break_result.n_breaks}")
for ci in break_result.break_cis:
    print(f"  {ci}")
# BreakInterval(break=Q11, CI=[Q10, Q12])
```

A break CI of [Q10, Q12] means the most probable break location is Q11, but the data is consistent with the break having occurred anywhere between Q10 and Q12. For a model refit, this tells you to include at least data from Q13 onwards in the new regime — Q12 is still ambiguous.

The retrospective finder is the right tool for historical analyses and annual experience reviews. The online BOCPD is the right tool for routine quarterly monitoring. They complement each other: run BOCPD continuously to catch breaks in real time; run PELT at each annual review to get precise, uncertainty-quantified break locations for the governance record.

---

## Combining frequency and severity

A loss ratio regime change is usually driven by either frequency, severity, or both shifting simultaneously. Monitoring them separately and combining the signals is cleaner than monitoring the loss ratio directly, because the loss ratio conflates two independent processes.

`LossRatioMonitor` runs `FrequencyChangeDetector` and `SeverityChangeDetector` in parallel and combines the posterior probabilities into a single monitoring signal.

```python
from insurance_dynamics.changepoint import LossRatioMonitor

monitor = LossRatioMonitor(
    frequency_hazard=0.1,
    severity_hazard=0.1,
    uk_event_prior=prior,
    threshold=0.5,
)

monitor_result = monitor.fit(
    claims=claims,
    exposure=exposure,
    avg_claim_amounts=severities,
    periods=periods,
)

print(monitor_result.recommendation)
# 'retrain' or 'monitor'

print(f"Combined max signal: {monitor_result.combined_probs.max():.3f}")
```

The `combined_probs` array is the element-wise maximum of the frequency and severity posterior probabilities. A combined probability above 0.5 at any period means at least one component is signalling a probable regime change. The `recommendation` field is either `'retrain'` (a break was detected; consider refitting the pricing model on post-break data) or `'monitor'` (no break detected; continue routine monitoring).

The recommendation is deliberately conservative. It does not tell you to retrain; it tells you that the evidence supports considering a retrain. The decision is still yours. But the evidence base for that decision is now documented, reproducible, and posterior-probability-grounded rather than eyeball-based.

---

## Consumer Duty reporting

PRIN 2A.9 requires firms to monitor customer outcomes on an ongoing basis and retain evidence that they have done so. For pricing models, this means evidencing that the model is being monitored for deterioration and that material changes trigger appropriate responses. The regulation does not specify methodology, but it requires documentation that would withstand scrutiny in a supervisory review.

`ConsumerDutyReport` generates a structured evidence pack from a `ChangeResult` or `MonitorResult`. The report includes: the full posterior probability time series; detected breaks with probabilities; the event priors applied and their rationale; the detection threshold and its justification; and a recommendation with the evidential basis.

```python
from insurance_dynamics.changepoint import ConsumerDutyReport

report = ConsumerDutyReport(
    result=monitor_result,
    model_name="Motor BI Frequency Model v3.2",
    reporting_period="Q4 2025",
    analyst="J. Smith",
)

report.to_html("motor_bi_monitoring_Q4_2025.html")
report.to_csv("motor_bi_monitoring_Q4_2025.csv")
```

The HTML report is formatted for inclusion in a model governance pack. The CSV is for data retention. Both are generated from the same `MonitorResult` object, so there is no discrepancy between what was computed and what was reported.

We want to be clear about what this does and does not do. It does not make a Consumer Duty compliance decision for you. Consumer Duty compliance requires a firm-level governance framework that goes well beyond a single monitoring report. What `ConsumerDutyReport` does is operationalise the monitoring evidence generation — the part that is currently being done informally, inconsistently, and incompletely at most UK insurers.

---

## Comparison to the informal approach

The standard alternative to this library is a combination of: periodic experience reviews, manual inspection of frequency and severity plots, and actuary judgment. This approach is not worthless — experienced actuaries do catch regime changes — but it has three structural weaknesses.

The first is latency. Visual inspection of a chart tends to identify breaks only after two to three quarters of post-break data have accumulated. The break is already confirmed before it is flagged. BOCPD, with a threshold of 0.5, will typically flag a genuine break within one to two quarters of it occurring, depending on the magnitude and the exposure volume.

The second is calibration. An actuary who says "it looks like something changed around Q3" is not giving you a probability. They are giving you a judgement, which may be correct but cannot be formally compared against other evidence. When a pricing committee needs to decide whether to trigger a model refit — a two-to-eight-week piece of work — "P = 0.87 posterior from the BOCPD model" is a more defensible basis than "the chart looks kinked."

The third is audit. Informal monitoring leaves no reproducible record. If you are asked, in a supervisory review two years from now, to demonstrate that your motor BI frequency model was monitored appropriately in 2024, you want a table of posterior probabilities, thresholds, and documented responses — not a set of monthly experience packs and someone's recollection.

---

## The library

[`insurance-dynamics`](https://github.com/burning-cost/insurance-dynamics) implements the full detection stack described above.

The main classes are `FrequencyChangeDetector` (online BOCPD with exposure-weighted Poisson-Gamma conjugate), `SeverityChangeDetector` (online BOCPD with Normal-InverseGamma conjugate for log-transformed severities), `RetrospectiveBreakFinder` (PELT with bootstrap CIs), `LossRatioMonitor` (combined frequency and severity monitoring with recommendation), `UKEventPrior` (ten encoded UK insurance events as hazard-function priors), and `ConsumerDutyReport` (PRIN 2A.9 evidence pack generation).

Install with `uv add insurance-dynamics`.

The only genuine limitation to document: BOCPD is a univariate method. It operates on a single time series — claim frequency, severity, or loss ratio — not on the multivariate joint distribution of all your rating factors simultaneously. For detecting whether a specific rating factor's effect has changed (concept drift in the P(Y|X) relationship), the correct tool is the Gini drift test in [`insurance-monitoring`](/2026/03/03/your-pricing-model-is-drifting/). Change-point detection and model monitoring solve adjacent but distinct problems: the former asks "has the data-generating process changed?", the latter asks "is my model still accurate?". In a well-run pricing function, both are running in parallel.

---

## References

- Adams, R.P. & MacKay, D.J.C. (2007). Bayesian Online Changepoint Detection. *arXiv:0710.3742*.
- Killick, R., Fearnhead, P. & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.
- FCA (2022). PS22/9: A new Consumer Duty. Final rules and non-Handbook guidance.
- Civil Liability Act 2018. Whiplash reforms, Official Injury Claim portal effective 31 May 2021.
- FCA (2021). PS21/11: Pricing practices in home and motor insurance. Effective 1 January 2022.

- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/) — the PSI/A-E/Gini framework for detecting whether your model is still accurate
- [Covariate-Conditioned IBNR Completion: Why Aggregate LDFs Mismatch Your Recent Book](/2026/03/13/insurance-nowcast/) — covariate-conditioned completion factors for immature accident periods
- [Bayesian Trend Models for Insurance Frequency and Severity](/2026/03/13/insurance-trend/) — fitting piecewise trend models after a regime change is detected
- [PRA SS1/23-Compliant Model Validation in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)
