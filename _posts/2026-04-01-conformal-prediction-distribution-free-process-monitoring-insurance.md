---
layout: post
title: "Distribution-Free Process Monitoring: Replace Your Arbitrary Thresholds with Calibrated Ones"
date: 2026-04-01
categories: [techniques, monitoring, pricing]
tags: [conformal-prediction, process-monitoring, SPC, PSI, AE-ratio, distribution-free, calibration, PRA-SS317, insurance-monitoring, Burger, arXiv-2512-23602, tweedie, poisson, gamma, CUSUM, exchangeability, UK-regulatory, python]
description: "Burger (arXiv:2512.23602, Dec 2025) applies conformal prediction to insurance model monitoring, replacing PSI > 0.2 and A/E > 1.15 with thresholds that are calibrated from data and carry a provable false alarm rate guarantee. We explain how it works, where it fits, and where it does not replace existing approaches."
math: true
author: burning-cost
---

Insurance model monitoring runs on conventions dressed up as standards. PSI > 0.2 means "investigate". A/E > 1.15 means "something has changed". Residuals more than 30% from expectation get flagged. These numbers appear in monitoring frameworks, governance papers, and regulatory submissions across the UK market. Nobody can tell you where they came from. Nobody can tell you what false alarm rate they produce on your data. They were reasonable guesses that calcified into doctrine.

The deeper problem is not just that the thresholds are arbitrary. It is that the implicit statistical model behind them is wrong. A/E control limits derived from normal approximations are wrong for Poisson frequency data. PSI has no distributional assumptions baked in at all — the 0.1 and 0.2 thresholds were proposed by a vendor in the 1990s on heuristic grounds and have never been formally justified. When your regulator asks "why this threshold?", the honest answer is "convention".

Burger (arXiv:2512.23602, "Distribution-Free Process Monitoring Using Conformal Prediction", December 2025) offers a different approach: set control limits from the data itself, with a guaranteed false alarm rate that holds regardless of the underlying distribution. The method is split conformal prediction applied to the monitoring context. It is simple, it is implementable, and it produces the audit trail PRA SS3/17 actually wants.

---

## What conformal process monitoring does

The setup mirrors classical statistical process control, but without the normality assumption.

You designate a calibration window — call it the "in-control" period, typically the first 12 months of a portfolio after model deployment, or the most recent period before a known structural change. You compute nonconformity scores for each observation in that window. The simplest choice is the absolute residual:

$$s_i = |y_i - \hat{y}_i|$$

For a frequency model, $y_i$ is realised claim count and $\hat{y}_i$ is modelled expected frequency times exposure. For severity, $y_i$ is average cost per claim and $\hat{y}_i$ is the modelled severity. For a combined Tweedie model, $y_i$ is pure premium.

You then set the control limit at the appropriate order statistic of the calibration scores. Formally, for a target false alarm rate of $\alpha$:

$$\hat{q} = s_{(\lceil (1-\alpha)(n+1) \rceil)}$$

where $n$ is the number of in-control calibration points and $s_{(k)}$ denotes the $k$-th order statistic of the calibration scores. Under exchangeability, the probability that a new in-control observation exceeds $\hat{q}$ is at most $\alpha$, regardless of the distribution of $y_i$.

A monitoring cycle — say, a quarterly actuals run — produces a new score $s_{\text{new}}$. If $s_{\text{new}} > \hat{q}$, the cycle signals. If it does not, it passes. The false alarm rate is controlled at $\alpha$ by construction.

That is the whole method. There is no distributional assumption, no asymptotic argument, no moment condition. It works for Poisson counts, Gamma severities, Tweedie pure premiums, log-normal large losses, whatever distribution your data actually follows.

---

## The studentised variant

The basic absolute residual does not adjust for heteroscedasticity. A high-frequency commercial vehicle risk will naturally produce larger absolute residuals than a low-frequency household risk, even when both are perfectly well-predicted. Flagging the commercial risk more often is a monitoring artefact, not a signal.

The fix is a studentised nonconformity score:

$$s_i = \frac{|y_i - \hat{y}_i|}{\hat{\sigma}(x_i)}$$

where $\hat{\sigma}(x_i)$ is an estimate of the conditional standard deviation — either from a separate dispersion model or from a locally weighted approximation using similar risks in the calibration set. This is the locally weighted conformal (LWC) approach described in Fan and Sesia (arXiv:2512.15383), which the `insurance-conformal` library implements for prediction intervals. The same idea applies directly to the monitoring context.

With studentisation, the control limit is on a standardised scale that is comparable across the book. High-frequency commercial risks are held to the same relative standard as household risks.

---

## The multivariate case: portfolio health as a single number

Individual-observation monitoring catches local failures. Portfolio monitoring needs something broader: a single health metric that flags when the overall distribution of a monitoring period has drifted from the in-control baseline, not just when individual observations are outliers.

Burger proposes using a one-class support vector machine (OC-SVM) trained on the feature vectors of the in-control calibration window. At each monitoring cycle, you compute the OC-SVM's decision function on the new period's feature vectors and convert it to a conformal p-value using the calibration set as the reference distribution. Small p-values indicate that the new period's feature space looks unlike the in-control baseline.

The resulting health metric — a single number between 0 and 1 — combines evidence across all features simultaneously. It is not a replacement for per-variable PSI. It is a summary test for "does this quarter look anything like the training period?" before you have run the claim actuals.

This is the genuinely novel contribution. Most monitoring frameworks apply univariate checks to individual features one at a time, then aggregate by hand. The OC-SVM approach gives you a joint test whose false alarm rate is still controlled via the conformal calibration, not via Bonferroni.

---

## The uncertainty spike: early warning before the mean shifts

There is a subtler diagnostic that the conformal framing enables.

In adaptive conformal prediction — where the calibration window rolls forward as new data arrives — the width of the control interval at each period reflects the local uncertainty of the model. When a model is well-calibrated and the portfolio is stable, interval widths should be roughly constant over time. When something changes — a new distribution channel, a rate change that attracted a different risk profile, a claims inflation spike — interval widths tend to widen before the point estimates shift.

Burger calls these "uncertainty spikes". They are detectable several monitoring cycles before the A/E ratio moves. That early warning matters: if you are waiting for A/E > 1.15 to act, you are already pricing mispriced risks. A rising interval width trend is an earlier signal that the model's fit is degrading.

The diagnostic is simple to implement: track the rolling mean width of the conformal interval across your monitoring period and flag when it exceeds the in-control baseline by more than a multiple of its in-control standard deviation. The threshold for "spike" is itself calibrated from the in-control period, so the false alarm rate on this secondary diagnostic is also controlled.

---

## What this means for PRA SS3/17

PRA Supervisory Statement SS3/17 requires insurers to document their model risk management frameworks, including monitoring approaches with pre-specified thresholds and defined remediation triggers. The PRA has been increasingly specific in recent years about what "pre-specified" means: it wants to see thresholds set before the monitoring run, with a documented basis for the chosen values.

"PSI > 0.2 per industry convention" does not fully satisfy this. It satisfies the letter of the requirement but not its spirit. The PRA wants to see that you understand what your thresholds mean statistically.

The conformal framing gives you this directly. The submission reads: "Control limits were calibrated on the in-control period of Q1 2024 to Q4 2024 ($n = 1{,}200$ quarterly risk groups) to achieve a false alarm rate of $\alpha = 0.05$. The resulting per-cycle threshold is the 1,141st order statistic of the 1,200 calibration nonconformity scores, computed as $\lceil 0.95 \times 1201 \rceil = 1141$. Under exchangeability, the probability of a false alarm in any in-control quarter is at most 5%."

That is a complete statistical statement. It is defensible, it is reproducible, and it is honest about the assumption it rests on (exchangeability — more on that below).

---

## How it compares with what you are already running

**PSI and CSI.** PSI measures the shift in the marginal distribution of a single feature. The 0.1 and 0.2 thresholds carry no calibrated false alarm rate. You can have PSI = 0.25 that is sampling variation on a thin segment and PSI = 0.12 that is a genuine trend in a high-volume segment. Conformal monitoring on the multivariate OC-SVM score gives a joint test with controlled FAR. PSI remains useful as a diagnostic — when the conformal health metric signals, PSI tells you which features drove it.

**A/E with normal confidence intervals.** Normal approximations for A/E control limits are wrong for Poisson counts at the typical insurance monitoring level. They are computed as $\hat{\mu} \pm z \cdot \sqrt{\hat{\mu}/n}$, which assumes large-sample Gaussian behaviour of a ratio of Poisson sums. For motor frequency monitoring on quarterly segments with thin exposure, the normal approximation fails. Exact Poisson confidence intervals are better but are rarely implemented in practice. Conformal control limits are distribution-free and correct by construction.

**CUSUM.** This is the most important comparison. CUSUM accumulates evidence of slow, persistent drift over many monitoring cycles. It is the right tool for detecting gradual claims inflation over 18 months that never crosses any single-period threshold. Conformal monitoring gives a per-cycle signal: each quarterly run is tested independently against the calibration baseline. The two approaches are complementary. We use both: conformal for per-cycle flagging, CUSUM for drift detection over time. The `insurance-monitoring` library implements both.

---

## The exchangeability assumption

Nothing in this post should suggest the method is free of assumptions. The conformal guarantee rests on exchangeability: the calibration observations and new observations must be exchangeable — loosely, the in-control data and the new period must be draws from the same distribution.

Insurance data violates this in predictable ways:

- **Seasonality.** Motor frequency has strong quarterly patterns. A Q4 calibration window will produce systematically different scores than a Q2 monitoring period, even on a perfectly stable book. The calibration window must cover a full annual cycle to be representative.
- **Claims development.** Severity estimates change as claims develop. A monitoring cycle that uses undeveloped severities will look systematically different from a calibration window of fully developed data. Use consistent development assumptions across both windows.
- **Portfolio growth.** If the in-control period covers 10,000 risks and the monitoring period covers 25,000, the score distributions may differ due to sampling variation alone. This is not exchangeability violation per se, but the power of the test changes.

The paper's recommendation — and ours — is to choose the calibration window carefully, verify it covers a representative seasonal cycle, and document the choice explicitly. If the calibration window is poorly chosen, the false alarm rate guarantee is still mathematically valid for the exchangeability-satisfying part of the data, but the useful signal deteriorates.

---

## The `insurance-monitoring` library

The conformal process monitoring approach described in Burger (2025) is implemented in our `insurance-monitoring` library:

```bash
uv add insurance-monitoring
```

The library covers:

- Per-observation monitoring with absolute and studentised NCS
- Rolling adaptive conformal intervals with uncertainty spike detection
- Multivariate OC-SVM health metric with conformal p-values
- CUSUM for slow drift detection, calibrated to align with the conformal FAR
- Export to the standard monitoring report format for governance packs

The API for a basic conformal monitoring setup:

```python
from insurance_monitoring import ConformalMonitor

monitor = ConformalMonitor(alpha=0.05)
monitor.fit(y_cal, y_hat_cal)          # calibrate on in-control period
result = monitor.check(y_new, y_hat_new)  # test new monitoring cycle
print(result.signal, result.threshold, result.score)
```

The `fit` step computes $\hat{q}$ from the calibration scores. The `check` step computes $s_{\text{new}}$ and compares it to $\hat{q}$. If `result.signal` is `True`, the cycle has exceeded the control limit at the calibrated FAR.

---

## What to do now

If you are running PSI and A/E monitoring on a deployed pricing model, the practical steps are:

1. Identify your in-control period. This should be a window you are confident represents normal model behaviour — ideally after any initial deployment settling and covering a full annual cycle.
2. Compute NCS for each observation in that window. Start with absolute residuals. Add studentisation if you have material heteroscedasticity across risk segments.
3. Set $\hat{q}$ using the formula above at $\alpha = 0.05$ (or whatever FAR your governance framework specifies).
4. Run conformal checks alongside your existing PSI and A/E. Do not immediately replace the existing checks — PSI in particular is useful for diagnosing which features drove a conformal signal.
5. Document the calibration basis. This is the part the PRA actually wants.

You do not need to throw away your existing monitoring infrastructure to benefit from this. Add conformal thresholds as a layer on top of existing checks. Over time, as you accumulate evidence that the conformal thresholds are better calibrated than the heuristic ones, the case for replacing them strengthens.

---

## The paper

Burger, Andreas. "Distribution-Free Process Monitoring Using Conformal Prediction." arXiv:2512.23602 [stat.AP]. December 2025.

---

## Related

- [Covariate Shift in Motor Pricing: Detection, Correction, and Conformal Intervals](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) — shift detection and importance weighting before conformal correction
- [Coverage Is the Wrong Guarantee for Pricing Actuaries](/2026/03/13/insurance-conformal-risk/) — conformal risk control and the full `insurance-conformal` library
- [Shape-Adaptive Conformal Prediction: Why Your Intervals Are Wrong for Skewed Claims](/2026/04/01/shape-adaptive-conformal-prediction/) — MOPI and group-conditional calibration for Tweedie models
