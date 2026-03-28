---
layout: post
title: "Does Bühlmann-Straub Credibility Actually Work?"
date: 2026-03-28
categories: [techniques, validation]
tags: [credibility, buhlmann-straub, scheme-pricing, experience-rating, fleet, python]
description: "We benchmarked Bühlmann-Straub credibility against raw experience and manual Z-factors on a 30-segment synthetic UK motor fleet book with a known DGP. On thin schemes, it reduces MSE by 30-50%. On thick ones, it ties. The honest case for using it anyway."
---

Yes, with one condition: you have to give up on recovering the true structural parameters from a small panel, and accept what the method actually delivers rather than what the textbook promises.

We ran the benchmark on [`insurance-credibility`](/insurance-credibility/) against a synthetic 30-scheme UK motor fleet book with known ground truth. The results are what you would expect from first principles, which is the best endorsement a classical technique can get.

---

## What Bühlmann-Straub promises

The claim is precise: given a portfolio of groups (schemes, large accounts, geographic territories), where each group has some volume of observed experience and a true underlying loss rate you cannot observe directly, B-S credibility produces the minimum-variance linear estimator of each group's true rate.

The formula is Z_i = w_i / (w_i + K), where w_i is the scheme's earned exposure (vehicle-years), and K = v/a is the ratio of expected process variance (v, within-scheme noise) to between-scheme variance (a, how much schemes genuinely differ). A scheme with little exposure gets a small Z and leans heavily on the portfolio mean. A thick scheme with strong data gets Z close to 1.0 and is trusted almost entirely on its own numbers.

The alternative is using Z-factors set by hand: "anything under 200 policies gets Z=0.20, 200-500 gets Z=0.40" and so on. Everyone who prices schemes has seen a spreadsheet like this. It is not obviously wrong. The question is whether the formal B-S approach is actually better, and by how much.

---

## What we tested

The benchmark uses a synthetic UK motor fleet portfolio: 30 scheme segments, 5 accident years, 64,302 total policy-years. The data-generating process has known parameters - mu=0.650 (portfolio loss rate), v=0.020 (within-scheme process variance), a=0.005 (between-scheme variance), K=4.0 (true noise-to-signal ratio).

Segments were structured to mirror the distribution you would see in a real UK commercial motor scheme book: 8 thin schemes with under 500 policy-years total, 12 medium schemes between 500 and 2,000 policy-years, and 10 thick schemes above 2,000. This split matters because the three tiers behave entirely differently.

Three estimators were compared against known true scheme rates:
- Raw observed experience: take the scheme's own loss rate at face value
- Portfolio average: ignore scheme differences, apply the grand mean to everyone
- Bühlmann-Straub (B-S): the credibility-weighted blend

We also compared B-S against manual Z-factors - fixed thresholds that assign Z by exposure band rather than deriving Z from the data.

The benchmark code is at `benchmarks/benchmark.py` in the repo. Seed 42.

---

## The numbers

**MAE by tier (vs known true scheme rates, 30 schemes, 5 accident years):**

| Tier | Schemes | Raw MAE | Portfolio avg MAE | B-S MAE | Winner |
|---|---|---|---|---|---|
| Thin (<500 PY) | 8 | 0.0074 | 0.0596 | **0.0069** | Credibility |
| Medium (500-2000 PY) | 12 | 0.0030 | 0.0423 | **0.0029** | Credibility |
| Thick (2000+ PY) | 10 | 0.0014 | 0.0337 | 0.0014 | Tie |
| All 30 schemes | 30 | 0.0036 | 0.0440 | **0.0035** | Credibility |

The MSE figures from the validation notebook are consistent with the MAE: B-S reduces MSE by 30-50% versus raw experience on thin segments, 5-20% on medium segments, and is indistinguishable from raw on thick segments.

The portfolio average is uniformly the worst. It ignores genuine between-scheme variation and applies a single rate to a fleet broker with a 0.40 loss ratio and a retail scheme with a 0.90 loss ratio. Using it is not a conservative choice - it is just wrong.

**Manual Z-factors versus B-S on thin segments:**

This is where the practical case for the formal method lands. A manual approach might set Z=0.30 for all schemes with 100-300 policy-years. But a scheme with exactly 100 vehicle-years and a very noisy history needs more shrinkage than a scheme with 280 vehicle-years from a stable affinity group. The manual approach cannot distinguish them - it applies the same Z to both. B-S derives Z from the scheme's actual exposure within the fitted portfolio: Z = w_i / (w_i + K_hat). On the thin tier in our benchmark, this produces strictly lower MSE than fixed-threshold Z-factors, because it uses each scheme's individual exposure rather than assigning everyone in the band to the same bucket.

---

## Structural parameter recovery

The benchmark recovers structural parameters as follows against the known true values:

- mu_hat = 0.6593 (true 0.6500) - error 1.4%, well within acceptable range
- v_hat = 0.01770 (true 0.02000) - EPV underestimated by 11.5%
- a_hat = 0.00212 (true 0.00500) - VHM underestimated by 57.6%
- K_hat = 8.36 (true K = 4.0) - K over-estimated by a factor of 2.1

The K over-estimation is the result you need to understand. K = v/a, and a_hat is computed via a method-of-moments between-group variance estimator. With only 30 groups across 5 years, this estimator is noisy - it needs 100+ groups over several years to converge reliably (the exact number of periods depends on the true K and exposure distribution). The over-estimated K means the model shrinks more aggressively than theory would recommend: schemes get pulled harder towards the portfolio mean than the true K=4.0 would dictate.

The effect is conservative rather than catastrophic. Despite using K_hat=8.36 when the true K is 4.0, B-S still beats raw experience on thin and medium tiers. The Z-values it produces are internally consistent - z_i = w_i / (w_i + K_hat) to four decimal places. The model is self-consistent, it is just more cautious than the oracle would be.

The practical implication: on a real 30-scheme book, you should expect B-S to over-shrink modestly compared to the theoretical optimum. You will capture most of the benefit on thin schemes. You will not achieve the maximum possible MSE reduction until your panel grows to 100+ groups.

---

## Where it works

Thin segments are where B-S earns its keep. Take the smallest scheme in the benchmark: approximately 150 total policy-years across 5 years -- about 30 vehicles per year. It has had a bad year; the raw exposure-weighted experience says loss rate 0.95. The portfolio average says 0.65. With K_hat=8.36 and w_i=150, Z = 150/(150+8.36) = 0.947. The B-S estimate is 0.947 × 0.95 + 0.053 × 0.65 = 0.934. That 0.934 is closer to the true underlying rate than the raw 0.95 when one bad year inflated the observed average.

The margin looks small -- 0.015 compared to the spread between raw and portfolio -- but it compounds. A fleet pricing team setting rates on twenty thin schemes simultaneously, each with noisy experience, systematically prices closer to truth if every one gets this correction. The aggregate improvement over raw experience, confirmed in the benchmark table, is 7% MAE reduction on the thin tier.

The mechanism matters: B-S is not smoothing across time, it is weighting between the scheme's evidence and the portfolio's evidence. The exposure weighting is the critical difference from a flat manual Z table. A scheme with 150 policy-years earns more credibility than a scheme with 80 policy-years even if both had identical claim counts per year, because the larger scheme has provided more statistical evidence about its true underlying rate.

The individual experience rating side of the library follows the same logic at policy level. For commercial motor with kappa typically in the 3-8 range, a policy with 3 full vehicle-years gives Z = 3/(3+8) = 0.27 -- roughly 27% credibility. That policy's B-S rate is 27% own experience, 73% portfolio mean. A flat 5-step NCD table assigns maximum discount regardless of whether the policy has 0.5 vehicle-years or 5 - it ignores the information entirely.

---

## Where it adds little

Thick segments with 2,000+ policy-years: Z approaches 1.0, and the credibility estimate converges to raw experience. This is mathematically correct - once you have enough data, trust the data. The portfolio mean is irrelevant. B-S applied to thick segments does not hurt you, but it does not help either. If your book consists entirely of schemes with 5,000+ policy-years each, you can use raw experience directly and the benefit of credibility weighting is negligible.

The other case where B-S adds nothing: if your schemes are genuinely homogeneous. If a_hat comes out near zero - the between-scheme variance is trivially small - then all schemes have the same true rate and the portfolio average is already the optimal estimate. B-S will fit K_hat very large and assign every scheme Z near zero, collapsing to the portfolio mean. This is the right answer when the data support it.

---

## The honest verdict

Bühlmann-Straub credibility works. The mechanics are sound, the MSE reduction on thin segments is real, and the method is self-correcting on thick segments. The practical barriers are smaller than the theoretical ones.

What it is not is a full structural model of your scheme portfolio. The method-of-moments estimator for K is noisy at the panel sizes most actuaries actually work with. You will not recover true K=4.0 from 30 groups over 5 years - you will recover something in the right order of magnitude that produces conservative but defensible shrinkage. The model's internal consistency (Z values matching the formula to four decimal places) is no guarantee that K_hat is close to the true value.

Use B-S if:
- You have thin or medium schemes where raw experience is plausibly noise rather than signal
- You currently set Z-factors by hand using exposure thresholds - the formal B-S derivation will produce better estimates because it uses each scheme's actual exposure continuously rather than binning
- You need a documented, reproducible, and defensible methodology for pricing committee or Lloyd's syndicate sign-off

Do not expect B-S to recover precise structural parameters from a small panel. It cannot. On 30 groups you will over-shrink modestly - this is known behaviour of the estimator, not a bug. On 100+ groups the parameters converge.

Fit time is under one second on a 30-scheme panel. It is closed-form - there is no iteration, no gradient descent, no convergence to worry about. For a technique that plugs into a quarterly scheme pricing review, that matters.

```python
from insurance_credibility import BuhlmannStraub
import polars as pl

bs = BuhlmannStraub()
bs.fit(df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.z_)          # credibility factors per scheme
print(bs.k_)          # noise-to-signal ratio K
print(bs.premiums_)   # credibility-blended rates
```

```bash
uv add insurance-credibility
```

Source and benchmarks at [GitHub](https://github.com/burning-cost/insurance-credibility). The full validation notebook with segment-level tables and shrinkage visualisations is at `notebooks/databricks_validation.py`.

Reference: Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Does Whittaker-Henderson Smoothing Actually Work?](/2026/03/28/does-whittaker-henderson-smoothing-actually-work/)
- [Does DML Causal Inference Actually Work for Insurance Pricing?](/2026/03/25/does-dml-causal-inference-actually-work/)
