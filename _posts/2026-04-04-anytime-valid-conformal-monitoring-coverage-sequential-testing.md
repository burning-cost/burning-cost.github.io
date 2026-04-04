---
layout: post
title: "When Does Your Conformal Model Break? Monitoring Coverage Without False Alarms"
date: 2026-04-04
categories: [model-monitoring, conformal-prediction]
tags: [conformal-prediction, anytime-valid, e-values, betting-martingale, sequential-testing, coverage-monitoring, model-monitoring, insurance-conformal, insurance-monitoring, multiple-testing, PSI, arXiv-2602.04364, hultberg-bates-candes, python, actuarial]
description: "You deploy a conformal pricing model and check its coverage every month. After twelve months, you see a coverage failure. Is the model broken, or did you just run twelve tests? Anytime-valid conformal monitoring (Hultberg, Bates & Candès, arXiv:2602.04364) solves the multiple-testing problem that conventional coverage monitoring ignores."
math: true
author: burning-cost
---

You deploy a conformal prediction model in January. The calibration set gave you 90% coverage on held-out data — the intervals contain the true loss at least 90% of the time, finite-sample guarantee, no distributional assumptions. You review coverage every month. In October, the empirical coverage for that month dips to 84%. You raise an action, convene a governance meeting, and start a refitting exercise.

Was the model actually broken? Or did you just run ten tests and get one that looked bad?

This is not a hypothetical. Monthly coverage reviews are standard practice in UK pricing governance, and coverage failure is a plausible reason to trigger a model refitting cycle. Refitting a CatBoost pricing model costs a few weeks of data scientist time, plus validation, plus committee sign-off. The expected cost of a false alarm is material.

---

## The multiple testing problem hiding in your monitoring process

Standard split conformal prediction gives you a one-shot guarantee:

$$P(Y_{\text{new}} \in C(X_{\text{new}})) \geq 1 - \alpha$$

The phrase "one-shot" is load-bearing. The guarantee is valid at a single, fixed evaluation time. It does not say anything about what happens when you evaluate coverage repeatedly over a sequence of batches.

Consider what monthly coverage monitoring actually is. You collect the batch of policies from month 1, compute the empirical coverage rate, and check whether it is above some threshold. Then you do the same for month 2. Then month 3. After twelve months you have run twelve coverage tests. If each test uses a 10% per-test false alarm rate — a common default — the family-wise error rate across the twelve checks is not 10%. It is substantially higher.

For independent monthly tests, the compounded FWER is $1 - (1 - \alpha_{\text{per-test}})^k$. At $\alpha_{\text{per-test}} = 0.10$ and $k = 12$: roughly 72%. You would expect a false alarm about three quarters of the time over a model's first year of deployment, even when coverage is genuinely valid throughout. At $k = 36$ (three years): 97%.

The batches are not actually independent — adjacent months share calibration set history — but the inflation is real regardless. Coverage monitoring without sequential error control is less a monitoring system and more a mechanism for generating work.

---

## What existing tools do and do not address

Our [`insurance-monitoring`](/insurance-monitoring/) library already implements several sequential monitoring procedures. `PSICalculator` and `CSICalculator` track feature distribution drift. `MonitoringReport` tracks A/E ratios, Gini drift, and segmented calibration. `PITMonitor` tests whether the probability integral transforms of model predictions remain uniform — a necessary condition for correct calibration — using a sequential e-process with anytime-valid false alarm control.

`PITMonitor` is the closest thing we already have to what we need here. It detects *distributional* changes in calibration via PIT uniformity. But there is a gap: PIT uniformity tests whether the whole predictive distribution is correctly calibrated. A conformal prediction interval has a binary coverage indicator — either the true value fell inside the interval or it did not. Monitoring conformal coverage sequentially requires testing whether the empirical coverage rate from a sequence of Bernoulli trials is consistent with the nominal $1 - \alpha$.

That sounds simpler, but it is a different problem. And it has the same multiple testing structure.

---

## The anytime-valid solution: betting martingales

Hultberg, Bates and Candès (arXiv:2602.04364, February 2026) — the same paper we discussed in the context of [sequential CRC recalibration](/2026/04/02/anytime-valid-conformal-premium-sufficiency-guarantee/) — establishes the formal framework for this. The monitoring problem is: at each time step $t$, a new coverage indicator arrives,

$$Z_t = \mathbf{1}[Y_t \in C(X_t)] \in \{0, 1\}$$

Under a correctly calibrated model, these are i.i.d. Bernoulli($1 - \alpha$) — or at least exchangeable with that marginal. The null hypothesis is $H_0: E[Z_t] \geq 1 - \alpha$. You want to reject $H_0$ — declare coverage failure — with controlled FWER across all possible stopping times $\tau$.

The key tool is an **e-value**: a non-negative random variable $E$ such that $E[E] \leq 1$ under the null hypothesis. E-values have a useful property: if you sequentially multiply e-values, the product is still a valid test statistic — you can check the product at any time, and the probability of it ever exceeding $1/\alpha$ under the null is bounded by $\alpha$. This is the "anytime-valid" property.

For the coverage monitoring problem, the natural construction is a **betting martingale**. At each step:

$$M_t = \prod_{s=1}^{t} \left(1 + \lambda_s (Z_s - (1 - \alpha))\right)$$

where $\lambda_s \in [0, 1/(1-\alpha))$ is a "bet size" chosen before seeing $Z_s$. Under $H_0$, $M_t$ is a non-negative martingale with $E[M_t] \leq 1$. By Ville's inequality:

$$P\left(\exists t : M_t \geq \frac{1}{\delta}\right) \leq \delta$$

So: declare coverage failure when $M_t$ first exceeds $1/\delta$, and the family-wise error rate across all possible stopping times is at most $\delta$. You can check $M_t$ every month, every week, or every day, and the probability of a false alarm over the entire monitoring horizon is bounded at $\delta$.

This is what "anytime-valid" means in practice. Not that the test is valid for a single pre-specified sample size, but that it is valid for every sample size simultaneously.

---

## Choosing the bet size

The bet size $\lambda_s$ controls the power of the test. Set $\lambda_s = 0$ and the martingale is identically 1 — never detects anything. Set it too large and the martingale is volatile.

The optimal adaptive strategy is to set $\lambda_s$ at each step to maximise the expected log-growth of $M_t$ under the alternative you consider most relevant — a specific value of $E[Z_t] = \beta < 1 - \alpha$. This gives the "Kelly bet":

$$\lambda_s^* = \frac{(1 - \alpha) - \hat{\beta}_s}{\alpha \cdot (1 - \hat{\beta}_s)}$$

where $\hat{\beta}_s$ is your running estimate of the actual coverage rate from observations so far. In practice, you initialise $\hat{\beta}_0 = 1 - \alpha$ (assume the model is correct) and update as observations accumulate. This is the **predictably mixed martingale** approach from the e-values literature.

The practical implementation is about twenty lines:

```python
import numpy as np

class CoverageMonitor:
    """
    Anytime-valid sequential monitoring of conformal coverage.

    Declare failure when martingale exceeds 1/delta.
    False alarm rate bounded at delta across all stopping times.
    """

    def __init__(self, alpha: float = 0.10, delta: float = 0.05):
        self.alpha = alpha          # target miscoverage rate
        self.delta = delta          # FWER threshold
        self.nominal_coverage = 1 - alpha

        self.M = 1.0                # current martingale value
        self.t = 0                  # number of observations
        self.n_covered = 0          # running count of covered observations

    def update(self, covered: bool) -> dict:
        """
        Update with a new coverage indicator.

        covered: True if the true value fell inside the conformal interval.
        Returns dict with current state: martingale value, flagged, coverage estimate.
        """
        self.t += 1
        z = float(covered)
        self.n_covered += int(covered)

        # Running coverage estimate (initialised at nominal)
        beta_hat = (self.n_covered + self.nominal_coverage) / (self.t + 1)

        # Kelly bet: maximise expected log-growth under estimated alternative
        denom = self.alpha * (1 - beta_hat)
        if denom < 1e-9:
            lam = 0.0
        else:
            lam = min(
                (self.nominal_coverage - beta_hat) / denom,
                1.0 / self.nominal_coverage - 1e-6,  # stay in valid range
            )

        # Martingale update
        increment = 1 + lam * (z - self.nominal_coverage)
        self.M *= max(increment, 0.0)  # non-negativity

        return {
            "t": self.t,
            "M": self.M,
            "flagged": self.M >= 1.0 / self.delta,
            "empirical_coverage": self.n_covered / self.t,
            "nominal_coverage": self.nominal_coverage,
        }

    def reset(self):
        """Reset after a confirmed model intervention."""
        self.M = 1.0
        self.t = 0
        self.n_covered = 0
```

Usage is straightforward. After each policy reaches its maturity date and the coverage indicator can be computed:

```python
monitor = CoverageMonitor(alpha=0.10, delta=0.05)

for batch in monthly_batches:
    for _, row in batch.iterrows():
        result = monitor.update(covered=row["inside_interval"])

    print(f"Month {batch.name}: M={result['M']:.2f}, "
          f"coverage={result['empirical_coverage']:.3f}, "
          f"flagged={result['flagged']}")
```

When `flagged` becomes `True`, that is a genuine signal: coverage has degraded with FWER at most 5% across the entire monitoring period. Not "this month's coverage is below 90%", which is a statement with no multiple-testing correction. A formal signal with bounded false alarm rate.

---

## How this fits the existing monitoring workflow

The practical workflow in `insurance-monitoring` already has two stages that precede a model refit decision: feature drift (PSI/CSI) and model performance (A/E, Gini, segmented calibration). The anytime-valid coverage monitor adds a third, conformal-specific layer.

A reasonable ordering:

**Stage 1 — Feature drift.** Run PSI on the model score and CSI on key rating factors. Amber on two or more factors, or red on any factor, triggers closer inspection. This detects covariate shift — the population has changed — which may not affect coverage immediately but is a leading indicator.

**Stage 2 — Performance drift.** Check A/E by segment, Gini stability, and Murphy decomposition. These detect concept drift and miscalibration at the point-prediction level. A degrading A/E that is not accompanied by PSI movement points to concept drift.

**Stage 3 — Coverage monitoring.** `CoverageMonitor.update()` runs every time a policy matures and its conformal interval can be evaluated. This is independent of the batch rhythm for stages 1 and 2. For a fast-maturing product (motor, home), you can run it daily. For long-tail lines, weekly or monthly is more practical.

The signal hierarchy: a coverage failure in stage 3 with no drift in stages 1 or 2 suggests the conformal calibration set is no longer representative — recalibrate the conformal predictor on more recent data without refitting the underlying model. A coverage failure accompanied by stage 2 drift suggests the underlying model has degraded — the pricing model itself needs refitting. A stage 1 signal without stages 2 or 3 points to covariate shift that may just need an intercept correction.

This mirrors the "recalibrate vs refit" framework from [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/), but now with a statistically calibrated trigger for the conformal layer specifically.

---

## A worked example

Suppose you have a motor portfolio of 3,000 active policies at any given time. Each month, roughly 250 policies reach their annual renewal date and you can check whether their realised loss fell within the conformal interval. You target 90% coverage, $\alpha = 0.10$, and you want a 5% FWER over a three-year monitoring horizon.

Under a correctly calibrated model, the expected monthly coverage rate is 90% — about 225 covered out of 250. The standard deviation of a monthly coverage estimate is $\sqrt{0.9 \times 0.1 / 250} \approx 0.019$. A month at 84% coverage is a $3.2\sigma$ event under the null — it happens about 0.2% of the time. But over 36 monthly checks with naive thresholding at some fixed threshold, the probability of at least one false flag grows substantially with the number of checks — even using a stricter threshold than 10% per-test does not help once the monitoring horizon is long enough.

The betting martingale accumulates evidence without the multiple-testing inflation. After 36 months of perfectly valid 90% coverage, the martingale fluctuates around 1 (by construction: $E[M_t] \leq 1$ under the null). A single month at 84% coverage from a valid model gives a small upward tick; a sustained coverage deficit at 80% across five consecutive months drives $M_t$ above $1/0.05 = 20$ with high probability.

To put numbers on "how many bad months does it take to trigger a flag": if true coverage has dropped to 80% and you see 250 observations per month, the expected time to detection (ETD) with an adaptive Kelly bet is approximately 3–4 months. At 85% true coverage, ETD is roughly 8–10 months. At 89% coverage (marginal degradation), detection takes over a year — which is appropriate; a 1pp shortfall is probably not operationally significant.

---

## The exchangeability caveat

The betting martingale construction requires that coverage indicators $Z_t$ are exchangeable under the null — i.e., drawn i.i.d. from Bernoulli($1 - \alpha$). For a conformal model calibrated on a truly held-out set, this follows from the standard conformal guarantee.

Two things can break it in production.

**Temporal autocorrelation.** If your policies have correlated exposures — a block of fleet policies, a household scheme where multiple policies in the same area are affected by the same storm — coverage indicators for that month will be correlated. The martingale construction still provides valid inference, but the effective sample size is smaller than the nominal policy count. You will need more months to accumulate detection power.

**Distribution shift.** If the underlying risk environment is changing — claims inflation, regulatory reform, a new underwriting appetite — the null hypothesis "$E[Z_t] \geq 0.90$" may never have been true for recent cohorts. The martingale is then a valid test of whether coverage has degraded since the last calibration, not since time immemorial. This is the right question to ask: has anything changed since we last calibrated? If the answer is yes, recalibrate.

The `insurance-conformal-ts` library's temporal calibration split already accounts for this partially — calibrating on recent data rather than a random holdout means the reference distribution is the most recent one. The monitoring martingale should be reset whenever you recalibrate: a new calibration set means a fresh null hypothesis.

---

## What is not yet in the libraries

The `CoverageMonitor` implementation above is a standalone component. We intend to integrate it into `insurance-monitoring` as a first-class citizen alongside `PITMonitor`, with `MonitoringReport` automatically computing both PIT-based and coverage-based sequential statistics where conformal intervals are available.

For teams using `insurance-conformal` directly, the gap between producing intervals and monitoring their coverage over time is currently a manual step — you have to join the interval outputs back to realised claims and compute coverage indicators yourself. A `ConformalCoverageTracker` that manages this join and feeds `CoverageMonitor` is on the roadmap for `insurance-conformal` v1.4.0.

The open question is multivariate coverage: if you are monitoring conformal intervals separately for frequency and severity models, should you combine the coverage martingales into a single signal? The e-values framework supports this via product martingales (the product of valid e-processes is a valid e-process), but the right way to present the combined signal to a pricing committee without obscuring which component is driving it is a design decision we have not finalised.

---

## Summary

Checking conformal coverage every month without sequential correction is approximately equivalent to conducting a new significance test each month with no Bonferroni adjustment — it inflates false alarm rates to the point where your monitoring process is generating spurious governance actions on valid models.

The betting martingale construction from Hultberg, Bates and Candès (arXiv:2602.04364) solves this directly: the martingale $M_t$ can be inspected at any time, and the probability that it ever exceeds $1/\delta$ on a well-calibrated model is bounded at $\delta$. A flag from this monitor means something — unlike "coverage was 87% last month".

For practical UK insurance pricing governance, this means three things. First, the quarterly coverage review should report the martingale value $M_t$, not just the empirical coverage rate for that quarter — the latter has no stopping-time guarantee. Second, the threshold for action should be $M_t \geq 1/0.05 = 20$, not "coverage below 90%". Third, when $M_t$ does cross the threshold, the signal tells you coverage has genuinely degraded; the PSI/A/E signals in `insurance-monitoring` then tell you whether the fix is recalibration of the conformal set or a full model refit.

The `CoverageMonitor` implementation above is the whole thing — about forty lines including comments. It will move into `insurance-monitoring` in a future release; until then it is self-contained.

The paper: Hultberg, E., Bates, S. & Candès, E.J. — "Anytime-Valid Conformal Risk Control" (arXiv:2602.04364, February 2026).

---

Related:
- [Your Premium Sufficiency Guarantee Has an Expiry Date](/2026/04/02/anytime-valid-conformal-premium-sufficiency-guarantee/) — the same paper, from the angle of sequential recalibration of conformal risk control rather than coverage monitoring
- [Your Monitoring Thresholds Are Made Up](/2026/04/03/conformal-control-charts-insurance-model-monitoring/) — conformal SPC as a replacement for arbitrary PSI/A/E thresholds; the distribution-free control chart complement to the sequential martingale approach here
- [Three-Layer Drift Detection: What PSI and A&E Ratios Miss](/2026/03/03/your-pricing-model-is-drifting/) — the broader monitoring framework into which coverage monitoring fits as a third layer
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the baseline conformal predictor whose coverage this post is monitoring
- [Conformal Prediction with Change Points](/2026/03/31/conformal-prediction-change-points-regime-switching-insurance-monitoring/) — what to do when coverage monitoring signals a structural break rather than gradual drift
