---
layout: post
title: "Testing Conditional Coverage in Conformal Prediction — The ERT Diagnostic"
seo_title: "Conditional Coverage Diagnostics for Conformal Prediction: ERT Test for Insurance Pricing"
date: 2026-03-31
categories: [conformal-prediction, model-validation]
tags: [conformal-prediction, coverage, diagnostics, ERT, insurance-conformal, covmetrics, conditional-coverage, Consumer-Duty, model-validation, LightGBM, arXiv-2512-11779, Braun, prediction-intervals, UK-insurance]
description: "Conformal prediction gives valid marginal coverage but says nothing about conditional coverage — your intervals can fail for young drivers or flood-zone properties while the portfolio-level metric looks fine. ERT (Braun et al., arXiv:2512.11779) is a classifier-based test that detects these failures with 500 samples instead of the 5,000+ competitors require."
author: burning-cost
---

The guarantee that makes conformal prediction attractive is also the one that misleads the most practitioners: split conformal intervals have valid marginal coverage at level 1−α. On average, 90% of future claims will fall inside a 90% prediction interval. That sentence is true and largely useless.

What matters for insurance pricing is whether 90% of claims fall inside the interval for young drivers in particular, for flood-zone properties in particular, for long-tail liability claims in particular. Marginal coverage says nothing about this. A conformal predictor can achieve exactly 90.0% marginal coverage while covering only 60% of claims from the highest-risk decile — and the miscoverage is structurally masked by over-coverage in the low-risk majority.

Braun, Holzmüller, Jordan, and Bach formalised this problem and proposed a practical solution in December 2025 (arXiv:2512.11779). The core contribution is ERT — Excess Risk of Target Coverage — a classifier-based diagnostic that detects conditional coverage failures in small samples. We think it is the right tool for conformal model validation in insurance, and this post explains why, how to use it, and where it connects to FCA Consumer Duty.

---

## Why marginal coverage is not enough

Conformal prediction's finite-sample guarantee is conditional on exchangeability: the calibration and test data must be drawn from the same distribution. In most insurance applications, this assumption is violated in exactly the places where coverage failure is most consequential.

Young drivers have structurally different frequency and severity distributions than older drivers. Their claims are right-skewed in a way that Pearson residuals — the standard basis for conformal scores — systematically compress. If the conformity score is `|y - ŷ| / ŷ`, large residuals get divided by large fitted values, and the resulting scores are better behaved for high-risk policyholders than for average ones. The conformal quantile is set at the portfolio level. High-risk policyholders are under-covered.

Flood-zone properties have spatially correlated claims. A flood event affects adjacent postcodes simultaneously, violating the exchangeability assumption at the cluster level. Standard split conformal treats each property as an independent observation. The spatial correlation means the effective calibration sample size is smaller than the number of policies suggests, and the marginal coverage guarantee formally applies to a distribution it was not estimated from.

High-value claims — roughly the top 2% by severity — suffer from the same Pearson score compression. The denominator of the conformity score is proportional to the fitted value. Large claims get shrunk scores. The quantile that controls marginal coverage is determined by the mass of small-to-medium claims. The result is intervals that are too narrow for the very claims that drive reserving risk.

Long-tenure renewal customers post-GIPP are a different population than new business customers in the same pricing cell. The January 2022 intervention changed the mix. A conformal predictor calibrated on a multi-vintage calibration set inherits the pre-intervention distribution. Conditional miscoverage on long-tenure renewals is a plausible consequence.

In all four cases, marginal coverage will look fine. The problem is invisible to standard diagnostics.

---

## ERT: the diagnostic

The key insight from Braun et al. is a reframing. If coverage is conditionally valid — if the probability of coverage is genuinely 1−α given any feature vector X — then the binary indicator

```
Z_i = 1{y_i ∈ C(X_i)}
```

is independent of X. Knowing X tells you nothing about whether an interval will cover the observation; coverage probability is constant at 1−α everywhere. Any classifier that can predict Z from X better than the constant predictor h*(X) = 1−α has found a coverage violation.

ERT converts this into a scalar metric:

```
ℓ-ERT = R_ℓ(1−α) − R_ℓ(p̂)
```

where R_ℓ(h) is the expected loss under proper scoring rule ℓ, p̂(X) is the classifier's predicted coverage probability, and 1−α is the constant-predictor baseline. The difference measures how much a classifier can improve on the naive prediction by using features. Under valid conditional coverage, no classifier can improve — ERT = 0.

Three concrete variants (Table 1 of the paper):

- **L₁-ERT**: uses the sign-based proper score; estimates the expected absolute gap E[|p(X) − (1−α)|]. If p(X) = 85% for young drivers in a 90%-coverage model, L₁-ERT detects a 5 percentage point gap.
- **L₂-ERT**: Brier score analogue; estimates E[(p(X) − (1−α))²]. Penalises large gaps more than small ones.
- **KL-ERT**: log-loss analogue; estimates E[D_KL(p(X) ‖ 1−α)]. Most sensitive to extreme under-coverage.

The **directional variants** are what make ERT practically useful for insurance. L₋-ERT counts only under-coverage (p(X) < 1−α); L₊-ERT counts only over-coverage. Under-coverage means inadequate reserves or capital for a segment. Over-coverage means intervals that are wider than necessary — an efficiency loss, but not a safety issue. We want to know about under-coverage.

One property worth emphasising: the classifier gives a lower bound on true ERT (Proposition 2.3 of the paper). If the LightGBM classifier finds a coverage violation, it is real — a non-zero ERT value is a confirmed violation, not a false alarm. The classifier cannot manufacture violations that do not exist in the data. It may fail to detect violations that a better classifier would find, but what it reports is conservative.

---

## Sample efficiency — why this matters

The practical obstacle to conditional coverage testing has always been sample size. Alternative approaches need several thousand test observations before they stabilise:

- **CovGap** (Angelopoulos et al., 2021): needs roughly 5,000+ test samples
- **WSC** (worst-slab coverage, Cauchois et al., 2021): needs 5,000+
- **ERT**: stabilises at approximately 500 test samples

For a UK motor book with 20,000 policies, a calibration set of 5,000 is plausible. For subgroup analysis — young male drivers in London, for instance — the effective sample may be 400–800. ERT is the only method with reliable power at that scale.

The sample efficiency comes from the classifier framing. Rather than searching for slabs in feature space (WSC) or computing coverage over fine bins (CovGap), ERT uses the full predictive power of a gradient-boosted classifier. The classifier automatically finds the subspace where coverage varies most, rather than requiring the analyst to pre-specify the slicing dimensions.

---

## Using covmetrics

The reference implementation is `covmetrics`, an open-source Python package by one of the paper authors:

```bash
uv add covmetrics
```

Given a calibrated conformal predictor, compute coverage indicators and run ERT:

```python
import numpy as np
from covmetrics import ERT

# cp is your fitted conformal predictor (e.g. from insurance-conformal)
# X_test, y_test: held-out test features and observed claims

intervals = cp.predict_interval(X_test, alpha=0.10)
lower = intervals["lower"].to_numpy()
upper = intervals["upper"].to_numpy()

# Binary coverage indicator: did the interval capture this claim?
cover = np.asarray((y_test >= lower) & (y_test <= upper), dtype=float)

# Run ERT with 5-fold cross-validation
# alpha here is the miscoverage rate (1 - target coverage)
ert_value = ERT().evaluate(X_test, cover, alpha=0.10, n_splits=5)
print(f"L2-ERT: {ert_value:.6f}")
```

For the directional variant that catches under-coverage:

```python
from covmetrics import ERT
from covmetrics.loss_functions import brier_score_under

ert_under = ERT(loss=brier_score_under).evaluate(X_test, cover, alpha=0.10, n_splits=5)
print(f"L2-ERT (under-coverage only): {ert_under:.6f}")
# Non-zero value = confirmed segments with coverage below 90%
```

One practical note: `covmetrics` carries a PyTorch dependency. For production environments where adding PyTorch to a pricing toolkit is impractical, the computation is achievable with LightGBM directly. We are building a native implementation in `insurance-conformal` (see below).

---

## The insurance diagnostic workflow

Here is the full diagnostic workflow using `insurance-conformal` for the conformal predictor and `covmetrics` for ERT:

```python
import numpy as np
import polars as pl
from insurance_conformal import ConformalPredictor
from insurance_conformal.diagnostics import CoverageDiagnostics
from insurance_conformal.diagnostics_ext import ert_coverage_gap, subgroup_coverage
from covmetrics import ERT
from covmetrics.loss_functions import brier_score_under, brier_score

# -----------------------------------------------------------------
# Step 1: marginal coverage — the baseline check
# -----------------------------------------------------------------
diag = CoverageDiagnostics(cp, X_test, y_test, alpha=0.10)
print(diag.marginal_coverage())          # should be close to 0.90
print(diag.coverage_by_decile())         # by predicted-value decile

# -----------------------------------------------------------------
# Step 2: ERT — the conditional coverage check
# Using covmetrics for the full classifier-based test
# -----------------------------------------------------------------
intervals = cp.predict_interval(X_test, alpha=0.10)
cover = np.asarray(
    (y_test >= intervals["lower"].to_numpy()) &
    (y_test <= intervals["upper"].to_numpy()),
    dtype=float
)

ert_overall = ERT(loss=brier_score).evaluate(
    X_test, cover, alpha=0.10, n_splits=5
)
ert_under = ERT(loss=brier_score_under).evaluate(
    X_test, cover, alpha=0.10, n_splits=5
)

print(f"L2-ERT (overall):         {ert_overall:.6f}")
print(f"L2-ERT (under-coverage):  {ert_under:.6f}")

if ert_under > 1e-4:
    print("WARNING: confirmed conditional under-coverage")
    print("Some feature subspaces have coverage below target")

# -----------------------------------------------------------------
# Step 3: subgroup diagnostics — identify where
# The ERT test tells you *that* there is a violation.
# Subgroup coverage tells you *where* to look.
# -----------------------------------------------------------------
segment_labels = df["segment"].to_numpy()   # young_driver, standard, etc.

seg_coverage = subgroup_coverage(
    cp, X_test, y_test, alpha=0.10,
    groups=segment_labels,
    group_name="segment"
)
print(seg_coverage.sort("coverage_gap", descending=True))
```

The distinction between these two steps matters. ERT is a global test: it confirms a violation exists somewhere in feature space. The subgroup breakdown is the follow-up investigation. If ERT is zero, the subgroup analysis is reassurance. If ERT is non-zero, subgroup analysis is diagnosis.

The existing `ert_coverage_gap` function in `insurance-conformal/diagnostics_ext.py` is a pragmatic approximation — it bins by predicted value rather than training a classifier on the full feature vector. It will catch simple coverage failures (monotone relationship between predicted value and coverage rate) but miss complex conditional patterns. The full ERT test from `covmetrics` is strictly more powerful.

---

## Four UK insurance cases

**Young drivers.** Expected L₋-ERT > 0. The combination of structural right-skew in young-driver claims and Pearson-score compression is not hypothetical — it is a predictable consequence of how standard conformity scores are computed. If you are running CQR (conformalised quantile regression) with a separate high-risk quantile model, ERT on the young-driver subset is the validation step that confirms whether the adaptive scoring has corrected the gap.

**Flood-zone properties.** Expected L₋-ERT > 0 in post-event years. Spatial correlation violates exchangeability at the cluster level, and a standard calibration set that happened to include no major flood events will systematically under-estimate the quantile needed for conditional coverage in high-flood-risk postcodes. Running ERT separately on high-flood-risk and low-flood-risk subsets will show divergence if the calibration set is not event-representative.

**High-value claims — top 2% by severity.** Pearson scores divide residuals by fitted values. For a £200k liability claim with a fitted value of £15k, the conformity score is 12.3. For a £2k home claim with a fitted value of £1.5k, the conformity score is also 0.33. The 90th-percentile quantile across the portfolio is dominated by the second type of observation. Intervals for high-severity claims are almost certainly too narrow. L₋-ERT on severity decile ≥ 9 will confirm this.

**Long-tenure renewals post-GIPP.** Consumer Duty and PS21/11 both require pricing model validation by customer segment. If a conformal predictor calibrated on 2019–2022 data is being applied to 2025 renewal portfolios, the post-GIPP customer mix shift is a material covariate shift. Running ERT segmented by tenure band (new business, 1–3 years, 4+ years) is the validation evidence you need for a Consumer Duty fair value assessment.

---

## The regulatory connection

ERT is not directly mandated by any FCA or PRA rule. We will not claim otherwise. But it is a precise answer to regulatory questions that are currently being asked imprecisely.

**FCA Consumer Duty (PS22/9, PRIN 2A)** requires firms to deliver good outcomes for all retail customer segments and to demonstrate this with evidence. FCA TR24/2 (August 2024) found firms specifically non-compliant with product governance at the segment level — adequate portfolio-level metrics were not sufficient. If your pricing uses conformal prediction intervals for risk quantification, a non-zero L₋-ERT on a customer segment is evidence of inadequate coverage for that segment. A positive L₋-ERT for elderly policyholders, in a context where you are using prediction intervals to inform capital allocation, is a Consumer Duty issue.

**PS21/11 (General Insurance Pricing Practices)** requires firms to validate that model outputs are appropriate by renewal vintage. Conditional miscoverage on long-tenure customers is precisely the kind of pricing model failure this rule was designed to catch.

**Model validation practice, PRA SS1/23-inspired.** SS1/23 formally scopes to internal models for regulatory capital, not insurance pricing models. But the Principle 4 requirement — independent validation including subpopulation testing — has been widely adopted as best practice. A model validation report that includes ERT values by risk segment is materially more defensible than one that reports only marginal metrics.

The practical argument is simpler: if you are using conformal prediction intervals to communicate uncertainty in a pricing or reserving context, you owe your governance committee a test that checks whether those intervals actually work for the subgroups that matter. Marginal coverage is a necessary condition, not a sufficient one. ERT provides the sufficient test.

---

## What we are building

`insurance-conformal` v0.6.4 has `ert_coverage_gap` in `diagnostics_ext.py` — a binned approximation that is useful for simple diagnostics but lacks the statistical power of the full classifier-based test.

The next step is a native `ConditionalCoverageERT` class that implements the full Braun et al. methodology using LightGBM as the classifier, avoiding the PyTorch dependency that makes `covmetrics` difficult to deploy in production pricing environments. The planned API:

```python
from insurance_conformal.diagnostics_ert import ConditionalCoverageERT

ert = ConditionalCoverageERT(n_splits=5, random_state=42)
ert.fit(X_cal, cover_cal)
result = ert.evaluate(X_test, cover_test, alpha=0.10)

print(result.ert_l2)           # overall L2-ERT
print(result.ert_l2_under)     # under-coverage component
print(result.per_fold)         # fold-by-fold breakdown for stability assessment
print(result.is_violation)     # bool: ert_l2_under > threshold
```

LightGBM is already a dependency of `insurance-conformal`. The implementation adds no new mandatory packages and runs on the same environments as the rest of the library. Estimated at around 200 lines. This is targeted for v0.7.x.

Until then, `covmetrics` is the right tool for the full ERT test, with the understanding that it requires PyTorch. For quick segment-level screening without the additional dependency, `subgroup_coverage` in `diagnostics_ext.py` gives coverage by business-relevant groupings and is sufficient when you have pre-specified the subgroups you want to check.

---

## What ERT does not do

It detects that conditional coverage is invalid. It does not fix it.

The correction for conditional miscoverage is a separate problem. Locally weighted conformal prediction — which scales the conformity score by an estimated difficulty function — is the standard approach, and it is implemented in `insurance-conformal` as `LocallyWeightedPredictor`. Conformalised quantile regression with separate quantile models for high-risk subgroups is an alternative. Neither is guaranteed to produce conditional validity; ERT is the test that confirms whether they have succeeded.

ERT also does not distinguish between violations caused by covariate shift, violations caused by poorly specified conformity scores, and violations caused by genuine model failures. The positive test result tells you coverage is conditionally invalid for some subgroup. Investigation — which subgroup, what mechanism — requires the subgroup breakdown and domain knowledge.

---

## The paper

Braun, S., Holzmüller, D., Jordan, M. I. & Bach, F. (2025). *Conditional Coverage Diagnostics for Conformal Prediction*. arXiv:2512.11779. Submitted 12 December 2025.

The `covmetrics` package (open source, MIT licence): `uv add covmetrics`. GitHub: [ElSacho/covmetrics](https://github.com/ElSacho/covmetrics).

Our implementation of locally weighted conformal prediction and CQR, with the existing approximate coverage diagnostics: [`insurance-conformal`](/insurance-conformal/).
