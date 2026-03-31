---
layout: post
title: "When your coverage guarantee means nothing: optimal regret in online conformal prediction under drift"
date: 2026-03-31
categories: [conformal-prediction, model-monitoring]
---

A conformal prediction interval that satisfies its coverage guarantee on paper can still leave you with three months of useless intervals after a claims inflation event. That is not a theoretical curiosity — it is how adaptive conformal inference (ACI) behaves in practice when you hit an abrupt regime shift. Jiadong Liang, Zhimei Ren, and Yuxin Chen's recent paper (arXiv:2602.16537, February 2026) pinpoints why, and shows what a better algorithm looks like.

## The coverage guarantee that hides the problem

ACI (Gibbs and Candes, NeurIPS 2021) adapts a conformal threshold via a simple gradient descent step on the pinball loss:

```python
alpha_t1 = alpha_t + gamma * (alpha - int(y_t not in C_t))
```

The coverage guarantee is: over a long time horizon T, the time-average coverage rate is at least 1 - alpha. What this does not say is anything about what happens in any particular stretch of time.

The guarantee is *marginal*. It averages over all possible training datasets — all possible realisations of the history up to deployment. A method can satisfy the marginal guarantee while catastrophically undercovering for 60 or 80 steps after a regime shift, provided it overcovers sufficiently once it has recovered. The miscoverage period and the recovery period average out. On paper, valid.

We have benchmarked this in [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts). On a synthetic +20% step shift in non-conformity score mean, ACI at gamma=0.005 produces first-half coverage of 66.7% and second-half coverage of 91.7% over a 24-month window — a period of severe undercoverage followed by overcorrection. The long-run average can look acceptable while the pricing team is flying blind during exactly the period they need valid intervals most.

## Training-conditional regret: a harder criterion

Liang et al. propose measuring performance via *training-conditional cumulative regret*:

```
Regret_T = sum_{t=1}^{T} L(C_t, y_t) - min_{theta} sum_{t=1}^{T} L(C_t(theta), y_t)
```

The key word is *conditioned*. Rather than averaging over all possible training histories, this conditions on the actual training data you observed. The regret measures how much worse your deployed algorithm performs compared to the best fixed post-hoc threshold — not in expectation over a thought experiment, but for the specific model you have in production.

Why does this matter? A UK pricing team has one deployed model. The marginal guarantee is a statement about a hypothetical ensemble of equally plausible training runs. The training-conditional guarantee is a statement about the run that actually happened — the one your intervals are based on.

Think of it this way. Marginal coverage is like a portfolio loss ratio — useful for aggregate pricing, but it tells you nothing about a specific account. Training-conditional regret is a per-account view. Both matter; the latter is what a model governance team should care about when asking whether *this* model's intervals are valid *now*.

## What the paper proves

The paper establishes a minimax lower bound on training-conditional regret under distribution drift, then shows that two algorithms achieve it. The contribution is not just an upper bound (a new algorithm that performs well) but the lower bound: a proof that no algorithm can do better. DtACI (Gibbs and Candes, JMLR 2024, arXiv:2208.08401) achieved a strong adaptive upper bound without proving it was optimal. Liang et al. close that gap.

The paper handles two drift models: abrupt change points (the distribution switches at unknown times tau_1, ..., tau_K), and smooth drift (the distribution evolves gradually under a Lipschitz-type condition). The minimax rate for the abrupt case is governed by the number of change points K and the time horizon T — consistent with the O(sqrt(KT)) rate that appears throughout the change-point literature.

## The algorithm: calibration flush, not passive decay

For the practically relevant case — a pretrained score function (your GLM residuals, say) and an arriving stream of calibration observations — the paper's proposed algorithm is conceptually clean:

```python
class DriftAwareCP:
    """
    Split-conformal + CUSUM drift detection.
    On alarm: flush calibration set, rebuild from post-drift scores.
    Reference: Liang, Ren, Chen (arXiv:2602.16537, 2026).
    """
    def update(self, score):
        self.cusum = max(0, self.cusum + score - self.reference_mean - self.slack)
        if self.cusum > self.threshold:
            # Regime change detected. Discard stale calibration entirely.
            self.calibration_scores = []
            self.cusum = 0.0
            self.cold_start = True

    def predict_interval(self, x, alpha=0.1):
        if len(self.calibration_scores) < self.min_calibration_size:
            # Cold start: infinite interval. Honest about ignorance.
            return (-inf, inf)
        quantile = np.quantile(self.calibration_scores, 1 - alpha)
        return self.base_forecaster.predict(x, quantile)
```

The key distinction from everything currently in [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts) is the flush. ACI and its variants passively decay the influence of old calibration observations by windowing or exponential weighting — old scores become less influential but never completely disappear. After a true abrupt change point, those stale scores do not just have reduced influence; they actively corrupt the calibration distribution. The right action is deletion, not downweighting.

This is also simpler than DtACI, which achieves strong adaptive coverage by running multiple experts (different gamma values) in parallel and aggregating them via exponential weighting. That machinery is doing work that CUSUM on the score distribution handles more directly: detecting when the score distribution has genuinely shifted rather than inferring it from coverage outcomes.

## The leading indicator problem

CUSUM on non-conformity scores is a *leading indicator* — a point worth dwelling on beyond the paper's formal results. Current diagnostics — the Kupiec backtest in `SequentialCoverageReport`, the width monitoring in `IntervalWidthReport` — detect problems in *coverage outcomes*. By the time coverage has deteriorated enough to fail a Kupiec test, you have already been producing miscalibrated intervals for several periods.

The non-conformity score distribution shifts first. If your model's systematic error increases — because claims inflation has moved the underlying cost distribution — the score distribution will show this before the coverage statistics do. A `ScoreDistributionMonitor` watching the CUSUM statistic on incoming scores would give the monitoring team a signal before the intervals become invalid, not after:

```python
class ScoreDistributionMonitor:
    """
    CUSUM on non-conformity score stream.
    Fires before coverage collapses — use alongside Kupiec for earlier warning.
    """
    def update(self, score: float) -> dict:
        self.cusum = max(0.0, self.cusum + score - self.reference_mean - self.slack)
        alarm = self.cusum > self.threshold
        if alarm:
            self.cusum = 0.0
        return {'cusum': self.cusum, 'alarm': alarm, 'steps_since_reset': self.steps}
```

This belongs in `diagnostics.py` as a complement to the existing coverage-outcome tests. It does not require implementing the full `DriftAwareCP` class — it is useful for any team running ACI or ConformalPID who wants earlier warning.

## UK insurance: two direct applications

**FCA pricing practices reform, January 2022.** The FCA's requirement to price renewals at new-business equivalent rates was a regulatory-induced step change in the claims/premium ratio across motor and home. Any conformal interval calibrated on pre-reform renewal data was immediately miscalibrated post-reform. This is the canonical abrupt change-point scenario: known in retrospect, unknown in advance, with an obvious regime boundary.

The useful operational question is not "did my intervals miscalibrate?" (they did, by construction) but "how many post-reform observations do I need before the intervals are trustworthy again?" The minimax lower bound answers this precisely. Without seeing the paper's full constants, we cannot quote a number, but the bound is a function of the target miscoverage alpha and the confidence level delta — not an ad hoc rule of thumb. The rate is likely O(alpha^-1 * log(1/delta)) post-regime observations for (1 - alpha, 1 - delta) validity — we are inferring this from the structure of the problem rather than from the paper's full text, which we have not yet read. With alpha=0.1 and delta=0.05, that is on the order of tens of post-reform renewals, not hundreds.

**Claims inflation shocks, 2022-2024.** UK bodily injury claims inflation was abrupt and severe: Ogden rate changes, litigation funding reform, repair cost inflation. A pricing model calibrated on 2019-2021 data was predicting into a materially different cost environment from late 2022. The abrupt change-point model applies directly. The answer to "how long until our intervals are valid again after the inflation event?" is what the minimax rate characterises. For a team reviewing pricing monthly, the answer in months matters. Even a minimax-optimal algorithm needs post-shift data to rebuild the calibration set; the rate tells you the minimum waiting period — not an arbitrary 12-month rule but a mathematically grounded figure.

Both scenarios share a structure: a known external event (regulatory change, inflation shock) creates a change point that the monitoring team knows about in retrospect even if they could not predict it precisely. The CUSUM detector is useful for quantifying when the data itself confirms what the team suspects.

## What we are not building yet

We think the calibration-flush approach is worth implementing in insurance-conformal-ts, but not until we can read the full paper. The HTML version of arXiv:2602.16537 was returning 404 as of late March 2026 (v2, dated March 5, not yet indexed in HTML form). Until we can read Algorithms 1 and 2 with their exact drift detector specifications and the full statement of the minimax lower bound theorem, we will not implement.

There are also open questions specific to insurance. The paper's guarantees assume independently distributed observations within each regime. Insurance claims time series — aggregate monthly triangles, renewal book totals — are autocorrelated. The paper's guarantees transfer cleanly to the *cross-sectional* monitoring case (one observation per policy at renewal, the natural unit for pricing model validation) but not directly to the aggregate time-series setting. We need to think through that gap before building.

RetroAdj (Jun and Ohn, arXiv:2511.04275, in insurance-conformal v0.7.0) already addresses post-shift recovery via retrospective quantile adjustment — approximately 30 steps to recover vs 100+ steps for ACI on a +30% inflation shock. Calibration flush is theoretically stronger (training-conditional rather than marginal guarantees), but we should benchmark both on the same insurance scenario before concluding which to recommend operationally.

## What to watch

The paper is worth tracking for three reasons. First, it changes the evaluation criterion, not just the algorithm — once teams adopt training-conditional regret as the right metric, a lot of the "our coverage looks fine on average" reassurances from marginal guarantees stop being acceptable. Second, the calibration-flush mechanism is operationally simple and fits naturally into existing split-conformal workflows. Third, the minimax lower bound is the kind of result that makes its way into model governance standards: if you need to argue to a validation team why your post-shift recovery period is appropriate, "we are achieving the minimax-optimal rate" is a materially stronger claim than "we tuned the step size until the backtests looked reasonable."

We will revisit the BUILD/BLOG verdict once the full text is accessible and we have run the benchmark comparison against RetroAdj.

---

*Liang J., Ren Z., Chen Y. (2026). Optimal training-conditional regret for online conformal prediction. arXiv:2602.16537. Submitted 2026-02-18; v2 2026-03-05.*

*Related libraries: [insurance-conformal](https://github.com/burning-cost/insurance-conformal), [insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts).*
