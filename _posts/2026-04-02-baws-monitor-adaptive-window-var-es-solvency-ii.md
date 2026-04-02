---
layout: post
title: "BAWSMonitor: Data-Driven Window Selection for Rolling VaR and ES"
date: 2026-04-02
categories: [libraries]
tags: [model-monitoring, solvency-ii, internal-models, var, expected-shortfall, insurance-monitoring, bootstrap, arXiv-2603.01157, pra-ss319]
description: "insurance-monitoring v1.1.0 adds BAWSMonitor — bootstrap adaptive window selection for rolling VaR and ES estimates, replacing arbitrary lookback choices with a principled, audit-ready method."
author: burning-cost
---

Every internal model team picks a window length at some point and then defends it forever. Three years because that is what the previous actuary used. Five years because it spans a full economic cycle. One year because the portfolio changed. None of these are wrong exactly — they are just not grounded in anything you could put in front of a PRA validator and call a methodology.

[insurance-monitoring v1.1.0](https://pypi.org/project/insurance-monitoring/) adds `BAWSMonitor` — Bootstrap Adaptive Window Selection — which selects the rolling window length from data using the strictly consistent scoring framework of Li, Lyu, and Wang (arXiv:2603.01157, March 2026). The window that produces the most accurate VaR and ES estimates wins. That is the whole idea, and it is more defensible than any fixed-length heuristic.

---

## What BAWS does

At each monitoring step, `BAWSMonitor` has a set of candidate windows — say, 52, 104, 156, 208 weeks. For each candidate, it takes the most recent `w` observations, generates block-bootstrap replicates (preserving serial dependence), computes the mean Fissler-Ziegel score across those replicates, and selects the window with the lowest score. It then computes VaR and ES from the winning window.

The Fissler-Ziegel score is the important piece. It is the only strictly consistent scoring function for the pair (VaR, ES) jointly — meaning that the true data-generating distribution minimises expected score, and no other distribution can do better. Using it for window selection means you are optimising for the actual target, not a proxy. Squared error on historical exceedance rates would give you something plausible but not provably aligned with VaR/ES accuracy.

Block bootstrap rather than iid bootstrap is essential. Insurance loss data — even annual — has serial correlation from reserve development, macro cycles, and claims inflation trends. An iid bootstrap destroys this structure and overstates confidence in short windows. The default block length follows the standard T^{1/3} rule; you can override it.

---

## The API

```python
from insurance_monitoring.baws import BAWSMonitor, BAWSResult

monitor = BAWSMonitor(
    alpha=0.05,                   # VaR/ES level
    candidate_windows=[52, 104, 156, 208],  # weeks
    n_bootstrap=500,
    block_length=None,            # None => T^(1/3) automatic
    score_fn='fissler_ziegel',    # or 'asymm_abs_loss' for VaR only
    random_state=42,
)

# Feed in historical weekly loss ratio observations
monitor.fit(historical_returns)

# At each monitoring cycle, call update() with the new observation
result: BAWSResult = monitor.update(this_week_return)

print(result.selected_window)  # int — winning window in observations
print(result.var_estimate)     # float — VaR_{0.05} from winning window
print(result.es_estimate)      # float — ES_{0.05} from winning window
print(result.scores)           # dict mapping window -> mean bootstrap score
```

The full window selection history is available as a Polars DataFrame:

```python
df = monitor.history()
# Columns: time_step, selected_window, var_estimate, es_estimate, n_obs
```

And `monitor.plot()` produces the standard BAWS governance chart: selected window over time as a step function, with VaR and ES on a second axis.

For batch processing — backtesting, annual reviews — `update_batch()` feeds multiple observations in one call and produces identical results to sequential `update()` calls:

```python
results = monitor.update_batch(new_weekly_returns)
# list[BAWSResult], one per observation, in order
```

---

## The two score functions

`score_fn='fissler_ziegel'` uses the negentropy Fissler-Ziegel score:

```
S(x, v, y; alpha) = (1/alpha) * [1{y > x} * (y - x) / (-v)] + log(-v) + (1 - 1/alpha) * x/v
```

where x is the VaR estimate, v is the ES estimate (negative convention), and y is the realised return. This is the default and the right choice when you are reporting both VaR and ES — it selects the window that is simultaneously most accurate for both measures.

`score_fn='asymm_abs_loss'` uses the tick loss:

```
S(x, y; alpha) = (alpha - 1{y <= x}) * (y - x)
```

This is strictly consistent for VaR alone. Use it if your governance framework only cares about VaR coverage and you want to maximise accuracy of the VaR estimate without ES pulling window selection in a different direction.

You can inspect both scores for any set of inputs directly:

```python
from insurance_monitoring.baws import fissler_ziegel_score, asymm_abs_loss

fz = fissler_ziegel_score(var_estimate=-0.15, es_estimate=-0.22,
                          realised=-0.18, alpha=0.05)
al = asymm_abs_loss(var_estimate=-0.15, realised=-0.18, alpha=0.05)
```

---

## A realistic internal model use case

A Lloyd's syndicate runs weekly loss ratio monitoring on its catastrophe book. Historical data goes back 10 years (520 weeks). Every quarter, the capital team needs to update the rolling VaR estimate used in the SCR calculation and file supporting evidence with Lloyd's.

Previously, the team used a fixed 260-week (five-year) window — not wrong, but the choice pre-dated the post-2017 weather event period and was never revisited.

With `BAWSMonitor`, the team fits on the first 200 weeks of data and then updates weekly. The history DataFrame shows that during 2022–23, when the book was relatively stable, the model consistently preferred 208-week windows (longer history is more informative). After the large 2024 event season, the selected window shortened sharply to 52 weeks as the long history began to distort the recent distribution.

This is exactly the adaptive behaviour the paper proves is optimal. The governance report can now show: *window selection follows an auditable algorithm, responds to structural breaks in the data, and is evaluated against a strictly consistent scoring function.* That is a stronger submission than "five years because cycle length."

---

## Structural breaks and the key invariant

One explicit design choice: when all candidate windows fail the bootstrap test (an edge case where the bootstrap variance is so high that no window is reliable), `BAWSMonitor` retains the previous selected window rather than crashing or emitting a degenerate estimate. Stability is more important than correctness in this edge case — you still have a reported VaR and ES to take to committee, and the anomaly is recorded in the history.

The invariant that ES <= VaR (in absolute terms) is enforced in the output. On constant or near-constant series, empirical quantiles can produce ES slightly above VaR due to floating point; this is clipped.

---

## Where it fits in the monitoring stack

`BAWSMonitor` sits alongside the existing sequential and CUSUM components, but it addresses a different question. `CalibrationCUSUM` tells you when a deployed pricing model has gone out of calibration. `PITMonitor` tells you when the probability integral transform distribution has shifted. `BAWSMonitor` tells you how much history to use when computing the current-period VaR and ES — a question that arises in capital allocation and internal model validation rather than in pricing model governance.

The library is on [PyPI](https://pypi.org/project/insurance-monitoring/) and the source is on [GitHub](https://github.com/burning-cost/insurance-monitoring). The BAWS paper is at [arXiv:2603.01157](https://arxiv.org/abs/2603.01157).
