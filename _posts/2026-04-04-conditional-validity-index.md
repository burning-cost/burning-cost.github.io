---
layout: post
title: "Conformal Prediction Works on Average. Does It Work for Your Riskiest Customers?"
date: 2026-04-04
categories: [techniques, regulation]
tags: [conformal-prediction, conditional-coverage, CVI, consumer-duty, FCA-PS22-9, prediction-intervals, insurance-conformal, subgroup-monitoring, Zhou-2026, arXiv-2603-27189, fair-value, undercoverage, reliability-estimator, CC-Select, actuarial, UK-insurance, model-governance]
description: "Marginal conformal coverage is an average — and averages hide subgroup failures. Zhou et al. (arXiv:2603.27189, March 2026) reframe conditional coverage assessment as a supervised learning problem, producing the Conditional Validity Index. A conformal predictor covering 90% overall but 82% for young drivers is a Consumer Duty exposure. CVI makes it measurable."
math: true
author: burning-cost
---

A conformal prediction interval with 90% coverage has a finite-sample theorem behind it. The proof is real. The guarantee holds. What the proof does not say — what it was never designed to say — is that coverage is 90% for every identifiable group of policyholders in your portfolio.

This distinction matters more now than it did a few years ago, for two reasons. First, actuaries are using conformal intervals further downstream than they used to: feeding them into reserve ranges, into capacity models, into automated acceptability decisions. When the interval is wrong for an identifiable segment, the downstream decision is systematically wrong for that segment. Second, the FCA's Consumer Duty (PS22/9, July 2022) introduced a positive obligation to evidence good outcomes across customer groups — not just in aggregate.

A conformal predictor that covers young drivers at 82% while covering the rest of the portfolio at 93% is not a calibration curiosity. It is a regulatory exposure.

Zhou, Zhang, Tao, and Yang (arXiv:2603.27189, March 2026) have given us the right tool for measuring this. Their paper introduces the Conditional Validity Index (CVI) — a scalar assessment of conditional coverage quality that decomposes into two components with distinct commercial and regulatory interpretations. The underlying insight is simple and powerful: treat conditional coverage assessment as a classification problem, not a statistical hypothesis test.

---

## What marginal coverage actually guarantees

Split conformal prediction constructs intervals $C_\alpha(x)$ such that:

$$P\!\bigl(Y_{n+1} \in C_\alpha(X_{n+1})\bigr) \;\geq\; 1 - \alpha$$

The randomness here is over the joint draw of the calibration set and the new test point. For exchangeable data, this holds exactly under mild conditions. For insurance claims, "exchangeable" is a modelling assumption, and it is usually reasonable enough to proceed.

What this does not imply is:

$$P\!\bigl(Y \in C_\alpha(X) \;\big|\; X \in \mathcal{S}\bigr) \;\geq\; 1 - \alpha \quad \text{for any subgroup } \mathcal{S}$$

The conditional statement is strictly stronger. The marginal guarantee is consistent with arbitrarily bad conditional coverage in any subgroup small enough not to dominate the aggregate.

To see how this plays out: suppose you are pricing household insurance with a calibration set of 20,000 policies. Of those, 1,800 are in flood-zone postcodes with significantly more variable claims. Your conformal calibration — the 90th percentile of nonconformity scores — is determined by the 18,200 non-flood-zone policies. At that threshold, your flood-zone policyholders may have systematically higher nonconformity scores. The interval that captures 90% of standard claims may capture 74% of flood-zone claims. Your marginal coverage is 89.7% — indistinguishable from target. Your conditional coverage for flood-zone policyholders is 74%. Nobody has checked.

---

## The reliability estimator

The paper's core contribution is to stop treating conditional coverage assessment as a hypothesis testing problem and start treating it as a prediction problem.

Given prediction intervals and a labelled evaluation set, define the coverage indicator for each observation:

$$Z_i = \mathbf{1}\!\bigl\{Y_i \in C_\alpha(X_i)\bigr\}$$

$Z_i = 1$ if the interval contains the true outcome, $Z_i = 0$ if it misses. The marginal coverage is just the mean of $Z_i$. But that hides everything interesting.

Now train a binary classifier — the paper uses LightGBM with isotonic calibration, which is the right choice for monotone probability estimation on tabular data — on $(X_i, Z_i)$: can you predict, from an observation's features alone, whether the conformal interval will cover it?

If coverage is truly uniform, this classifier gains nothing over a constant prediction. It will be no better than predicting $1 - \alpha$ everywhere. If coverage varies systematically with features, the classifier will find the structure.

The classifier's calibrated output $\hat{\eta}(x)$ is the estimated local coverage probability at $x$ — the probability that the conformal interval covers a new observation with features $x$. The paper calls this the reliability estimator. It is doing something that no hypothesis test can do: it produces a coverage estimate for any specific risk profile, not just a global assessment.

For a 25-year-old male driver in North London, $\hat{\eta}(x)$ might be 0.71. For a 45-year-old female driver in a rural postcode, $\hat{\eta}(x)$ might be 0.94. The marginal coverage of 0.90 tells you nothing about either of these people individually.

---

## The CVI decomposition

The Conditional Validity Index aggregates the reliability estimator into a scalar:

$$\text{CVI} = \frac{1}{n} \sum_{i=1}^n \bigl|\hat{\eta}(X_i) - (1-\alpha)\bigr|$$

This is the mean absolute deviation of estimated local coverage from the target. A CVI of zero means the reliability estimator is flat at the target — truly uniform conditional coverage. Any positive CVI means coverage is not uniform.

The useful structure is in the decomposition. Define a tolerance band around the target:

$$\mathcal{A}^- = \bigl\{i : \hat{\eta}(X_i) < (1-\gamma)(1-\alpha)\bigr\}, \quad \mathcal{A}^+ = \bigl\{i : \hat{\eta}(X_i) > (1+\gamma)(1-\alpha)\bigr\}$$

for a tolerance parameter $\gamma > 0$ (default 0.1). Observations within the band — within 10% of the target — are considered adequately calibrated and contribute to neither component.

**Safety (CVI$_U$):** undercoverage risk.

$$\text{CVI}_U = \underbrace{\frac{|\mathcal{A}^-|}{n}}_{\pi^-} \times \underbrace{\frac{1}{|\mathcal{A}^-|}\sum_{i \in \mathcal{A}^-}\bigl[(1-\alpha) - \hat{\eta}(X_i)\bigr]}_{\text{CMU}}$$

$\pi^-$ is the proportion of observations below the tolerance band. CMU is how far below target they sit on average. $\text{CVI}_U$ is their product.

**Efficiency (CVI$_O$):** overcoverage cost.

$$\text{CVI}_O = \underbrace{\frac{|\mathcal{A}^+|}{n}}_{\pi^+} \times \underbrace{\frac{1}{|\mathcal{A}^+|}\sum_{i \in \mathcal{A}^+}\bigl[\hat{\eta}(X_i) - (1-\alpha)\bigr]}_{\text{CMO}}$$

The Safety component is the one that matters for regulation. Overcoverage is a capital efficiency problem. Undercoverage for an identifiable group is a Consumer Duty problem.

The decomposition gives you three useful numbers: how many policyholders are being undercovered ($\pi^-$), by how much on average (CMU), and the product as a single scalar to track over time or to rank competitor models.

---

## Why CVI is not the same as stratified coverage

An obvious question: why not just compute coverage by subgroup? Calculate actual coverage for young drivers, for urban policyholders, for any segment of interest, and report the results.

The answer is twofold.

First, you need to know which subgroups to check. A stratified analysis requires you to pre-specify the segments. The reliability estimator makes no such assumption — it uses the full feature matrix and finds the structure the data contains. It will surface undercoverage in a combination of features you had not thought to stratify on. A 23-year-old male in a specific occupation class in a high-crime postcode may sit in an undercovered region for reasons that no single-dimension stratification would reveal.

Second, the reliability estimator produces a continuous score at the level of the individual, not a point estimate for each subgroup. The CVP curve (empirical CDF of $\hat{\eta}$ values) shows you the full distribution of local coverage quality across the portfolio, not just summaries for predefined buckets.

That said, stratified coverage by key dimensions — age band, region, prior claims — remains a useful complementary diagnostic. The reliability estimator identifies the problem; stratified analysis helps explain it.

---

## CC-Select: choosing between conformal predictors

Suppose you have three candidate conformal predictors for your motor book. All three achieve 90% marginal coverage — they must, by construction, if calibrated correctly. Standard model selection metrics are useless for distinguishing between them on coverage quality.

CC-Select uses CVI to solve this selection problem. For each of $B$ random subsamples of the calibration data:

1. Split the subsample into calibration and evaluation halves.
2. Recalibrate each candidate predictor on the calibration half.
3. Compute CVI for each predictor on the evaluation half.

The predictor with the lowest mean CVI across subsamples is selected. The 50/50 split is theoretically motivated in the paper — it minimises the combined estimation and approximation error in the reliability estimator (Theorem 2). The selection rule is provably consistent: if predictor A has strictly lower true conditional CVI than predictor B, the probability that CC-Select identifies A correctly converges to one as sample size grows (Theorem 4).

This is model selection with teeth. You are not comparing AIC scores or cross-validated RMSE. You are asking: which model's intervals fail the least for identifiable subgroups of policyholders?

For a pricing team that has been using the same non-conformity score for two years because nobody had a principled way to compare alternatives, CC-Select is the missing piece.

---

## The Consumer Duty framing

Consumer Duty (PS22/9) places a positive obligation on firms to understand and evidence outcomes across different customer groups. The FCA has been explicit that this is not satisfied by aggregate metrics that happen to pass threshold. The question is whether outcomes are materially worse for any identifiable cohort.

Conformal prediction intervals are not traditionally thought of as something Consumer Duty applies to. But if those intervals are feeding downstream decisions — reserve ranges, automated underwriting thresholds, capacity flags — then their failure modes for specific customer segments are exactly the kind of outcome differentiation that Duty is designed to catch.

The parallel with FCA EP25/2 on proxy discrimination is direct. EP25/2 is concerned with whether protected characteristics influence pricing outcomes through proxy variables. The same logic applies to interval reliability: if your intervals are systematically less reliable for younger drivers, and age is a characteristic the regulator watches, you have a potential conduct risk even if the interval construction never explicitly references age.

CVI provides the measurement tool for a Consumer Duty subgroup monitoring process:

1. Run your conformal predictor across a representative evaluation cohort.
2. Fit the reliability estimator on $(X_i, Z_i)$ pairs.
3. Report $\pi^-$, CMU, and $\text{CVI}_U$ to the model risk governance committee alongside the standard marginal coverage figure.
4. Cross-tabulate the bottom quintile of $\hat{\eta}$ against protected characteristics and rating factors.
5. If the concentration is material, investigate whether the calibration data is representative of the undercovered segment.

There is no regulatory bright line on what constitutes a material CVI$_U$. We think $\pi^- > 0.10$ at $\gamma = 0.10$ warrants formal investigation — that is more than one in ten policyholders with estimated local coverage more than 9 percentage points below target on a 90% interval. That is not a calibration tolerance. That is a systematic failure for an identifiable group.

---

## A concrete illustration in Python

The following shows the core mechanics without the library machinery — useful for understanding what is happening before you reach for `ConditionalCoverageAssessor`.

```python
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

def fit_reliability_estimator(X_eval, y_lower, y_upper, y_true, n_splits=5):
    """
    Train a reliability estimator: predict whether the conformal interval
    covers the true outcome, from the observation's features.

    Returns eta_hat: estimated local coverage probability for each observation.
    """
    n = len(y_true)
    z = ((y_true >= y_lower) & (y_true <= y_upper)).astype(int)

    eta_hat_raw = np.zeros(n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_eval):
        clf = LGBMClassifier(n_estimators=200, learning_rate=0.05,
                             num_leaves=31, verbose=-1)
        clf.fit(X_eval[train_idx], z[train_idx])
        eta_hat_raw[val_idx] = clf.predict_proba(X_eval[val_idx])[:, 1]

    # Isotonic calibration on out-of-fold estimates
    iso = IsotonicRegression(out_of_bounds="clip")
    eta_hat = iso.fit_transform(eta_hat_raw, z)

    return eta_hat, z


def compute_cvi(eta_hat, alpha=0.10, gamma=0.10):
    """
    Decompose CVI into Safety (undercoverage) and Efficiency (overcoverage).

    Returns dict with cvi, cvi_u, cvi_o, pi_minus, pi_plus, cmu, cmo.
    """
    target = 1.0 - alpha
    lo_band = (1 - gamma) * target
    hi_band = (1 + gamma) * target

    under_mask = eta_hat < lo_band
    over_mask  = eta_hat > hi_band

    pi_minus = under_mask.mean()
    pi_plus  = over_mask.mean()

    cmu = (target - eta_hat[under_mask]).mean() if under_mask.any() else 0.0
    cmo = (eta_hat[over_mask] - target).mean()  if over_mask.any()  else 0.0

    cvi_u = pi_minus * cmu
    cvi_o = pi_plus  * cmo
    cvi   = np.abs(eta_hat - target).mean()

    return {
        "cvi": cvi, "cvi_u": cvi_u, "cvi_o": cvi_o,
        "pi_minus": pi_minus, "pi_plus": pi_plus,
        "cmu": cmu, "cmo": cmo,
    }
```

Running this on your evaluation set and printing the results:

```python
eta_hat, z = fit_reliability_estimator(
    X_eval, intervals["lower"], intervals["upper"], y_eval
)
result = compute_cvi(eta_hat, alpha=0.10, gamma=0.10)

print(f"Marginal coverage:   {z.mean():.3f}   (target: 0.900)")
print(f"CVI:                 {result['cvi']:.4f}")
print(f"  Safety  (CVI_U):   {result['cvi_u']:.4f}  "
      f"({result['pi_minus']:.1%} of obs, mean shortfall {result['cmu']:.3f})")
print(f"  Efficiency (CVI_O):{result['cvi_o']:.4f}  "
      f"({result['pi_plus']:.1%} of obs, mean excess {result['cmo']:.3f})")
```

Suppose you see:

```
Marginal coverage:   0.901   (target: 0.900)
CVI:                 0.0384
  Safety  (CVI_U):   0.0261  (13.4% of obs, mean shortfall 0.095)
  Efficiency (CVI_O):0.0123  (6.1% of obs, mean excess 0.040)
```

The marginal coverage passes. The Safety number tells you that 13.4% of policyholders have estimated local coverage 9.5 percentage points below target on average. That is a 90% interval that is effectively covering them at 80.5%. Those policyholders are identifiable from features — the reliability estimator found them.

To find who they are:

```python
import polars as pl

df = pl.DataFrame({
    **{col: X_eval[:, i] for i, col in enumerate(feature_names)},
    "eta_hat": eta_hat,
    "covered": z,
})

# Bottom quintile of reliability
low_coverage = df.filter(pl.col("eta_hat") < df["eta_hat"].quantile(0.2))

# Cross-tab against age band and region
print(low_coverage.group_by("age_band").agg(
    pl.col("eta_hat").mean().alias("mean_eta_hat"),
    pl.len().alias("n"),
).sort("mean_eta_hat"))
```

If the bottom quintile is disproportionately concentrated in a particular age band or region, you have found the segment. The next question is whether the calibration set was representative of those policyholders — in most cases, it was not.

---

## Theoretical guarantees

The paper proves two key results.

First, the CVI estimator converges. Under conditions on the reliability estimator's approximation quality, the estimated CVI converges to the true CVI at a rate that depends on the classifier's performance. The paper quantifies the convergence rate in Theorem 1 and the finite-sample deviation bounds in Theorem 2. For a pricing actuary, the practical implication is that CVI estimates stabilise as evaluation set size grows, and the 800-observation lower bound for meaningful results is a genuine constraint, not a conservative heuristic.

Second, CC-Select is consistent. If predictor A has strictly lower true CVI than predictor B, CC-Select selects A with probability approaching 1 as sample size grows (Theorem 4). This is a proper model selection guarantee, not a frequentist comparison of point estimates.

What the guarantees do not cover: the reliability estimator is only as good as LightGBM's ability to model the coverage structure from the features provided. A feature matrix that omits the relevant risk drivers will produce a flat reliability estimator and a near-zero CVI regardless of actual conditional coverage failures. This is the main practical failure mode.

---

## How this fits the existing toolkit

The CVI framework sits alongside, not instead of, the conditional coverage hypothesis tests. The ERT (Excess Risk Test) in `insurance-conformal` v0.7.1 answers the binary question: is there evidence of conditional coverage failure at all? CVI answers the continuous question: how bad is it, and between two predictors, which has less of it?

Run ERT first. If ERT is near zero, conditional coverage failures are not detectable from this data — either the predictor is well-calibrated or the evaluation set is too small to find structure. If ERT is positive, proceed to CVI for quantification and CC-Select for model selection.

The `ConditionalCoverageAssessor` in `insurance-conformal` v1.2.0 implements this full workflow, including the stateful fit/query interface that avoids refitting the LightGBM classifier for each candidate predictor comparison. See [that post](/2026/04/02/conditional-coverage-assessor-cvi-consumer-duty-insurance-conformal-v120/) for the library API. The code above shows the underlying mechanics for those who prefer to understand before they import.

---

## What to put in the governance pack

Consumer Duty governance documentation for a conformal prediction model should include, at minimum:

- Marginal coverage on the evaluation set with a 95% binomial confidence interval.
- CVI, CVI$_U$, and CVI$_O$ at $\gamma = 0.10$, with $\pi^-$ and CMU reported separately.
- The CVP curve (empirical CDF of $\hat{\eta}$), annotated with the target and tolerance band.
- A cross-tabulation of the bottom quintile of $\hat{\eta}$ against age band, region, and any protected characteristics that are indirectly represented in the rating structure.
- If multiple predictor variants were evaluated, the CC-Select output showing CVI rankings.

The CVP curve is worth including in any presentation to a non-technical audience. A flat line at the target is what uniform conditional coverage looks like. A line that dips well below the target for the first 15% of observations is what a systematic undercoverage problem looks like. The shape communicates the problem in a way that a single scalar cannot.

---

The paper is Zhou, Zhang, Tao, Yang — "Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation and Selection" (arXiv:2603.27189, March 2026). Implementation in [insurance-conformal](https://github.com/burning-cost/insurance-conformal) from v0.8.0, with the stateful `ConditionalCoverageAssessor` API from v1.2.0.

---

Related:
- [Conditional Coverage and Conformal Prediction Model Selection: CVI and CC-Select](/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — the v0.8.0 library release with `ConditionalValidityIndex` and `CCSelect`
- [Your Conformal Intervals Lie About Some Policyholders](/2026/04/02/conditional-coverage-assessor-cvi-consumer-duty-insurance-conformal-v120/) — the v1.2.0 `ConditionalCoverageAssessor` API and Consumer Duty framing
- [Your Prediction Intervals Are Unfair (And You Haven't Checked)](/2026/04/01/conformal-prediction-intervals-fairness-coverage-parity/) — Vadlamani et al. on conformal fairness criteria and the four-fifths rule
- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the baseline split conformal implementation
