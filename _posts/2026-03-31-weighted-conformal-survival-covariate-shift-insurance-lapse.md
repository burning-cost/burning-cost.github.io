---
layout: post
title: "Conformal Prediction for Lapse Timing When Your Book Has Shifted"
date: 2026-03-31
categories: [techniques]
tags: [conformal-prediction, survival-analysis, covariate-shift, lapse, time-to-event, IPCW, censoring, insurance-conformal, insurance-survival, CLV, Consumer-Duty, PS21/11, fair-value, uncertainty-quantification, python, arXiv-2512-03738]
description: "Shin, Lee and Kang (arXiv:2512.03738, Dec 2025) provide the first finite-sample coverage guarantee for time-to-event prediction under covariate shift. Here is what it means for lapse timing models deployed on portfolios that no longer match their training data."
author: burning-cost
---

Conformal prediction for survival analysis has been technically possible since Candès, Lei and Ren published their censoring-adjusted framework in the JRSSB in 2023. The problem is that their coverage guarantee rests on a quietly uncomfortable assumption: the distribution of covariates in your calibration set matches the distribution at test time. In insurance, this assumption is wrong as a matter of routine.

A lapse model calibrated on 2022–23 renewals and deployed on the 2025 book is not operating on IID data. Market hardening between 2022 and 2025 changed who stayed and who shopped. Aggregator growth shifted the NCD distribution. The post-GIPP loyalty penalty ban altered the propensity-to-shop for previously captive customers. You are asking a conformal predictor trained on one population to produce coverage guarantees for a materially different one. The existing conformal machinery — including our own `insurance-conformal` library — has no answer for this. The intervals are computed, but the coverage guarantee has silently evaporated.

Shin, Lee and Kang (arXiv:2512.03738, Yonsei University, submitted December 2025) fix this. Their paper is the first to combine right-censored survival outcomes, covariate shift, and a finite-sample coverage theorem in a single framework. This post explains what they did, what it actually buys you in a UK pricing context, and why we are not building it into `insurance-conformal` yet.

---

## Why conformal prediction for survival is harder than for regression

The standard split conformal recipe is straightforward: fit a model on a training set, compute conformity scores on a held-out calibration set, take a quantile of those scores, and use it to construct prediction intervals for test observations. Coverage follows from exchangeability — if calibration and test are drawn from the same distribution, the test point's score ranks uniformly among the calibration scores.

Survival data breaks this in two distinct ways.

The first is censoring. You observe not the true event time $T_i$ but $\tilde{T}_i = \min(T_i, C_i)$, where $C_i$ is the censoring time — in insurance terms, the policy cancellation date, the data cutoff, or mid-term termination. For a policyholder whose policy lapsed at month 8, you know the event time. For one still active at month 8 with a 12-month data window, you know only that they survived to month 8. You cannot compute a conformity score in the usual sense for the censored observations.

Candès, Lei and Ren (2023, JRSSB) addressed censoring by subsetting the calibration set to observations where the censoring time exceeded a threshold $c_0$: if $C_i \geq c_0$, the observation is either a confirmed lapse before $c_0$ or a confirmed survivor beyond $c_0$, so you know the ordering relative to the threshold. This is elegant, but the subsetting step introduces its own covariate shift: the distribution of $X$ among calibration points satisfying $C_i \geq c_0$ differs from the full population distribution.

The second problem is the separate covariate shift between your calibration cohort and the test population — what we described above for the 2022 vs 2025 book.

The Shin et al. paper makes both corrections rigorous. Their theorem states that, under correctly specified density ratio weights, the prediction set achieves test-marginal coverage $P(T_{n+1} \geq \hat{q}^\omega) \geq 1 - \alpha$ in finite samples. This is not asymptotic. It holds for any sample size given correct weights.

---

## The methodology

The method has four components.

**Conformity scores via the survival function.** For a fitted survival model outputting $\hat{S}(t \mid x)$, the conformity score for a calibration observation $(x_i, \tilde{T}_i, \delta_i)$ — where $\delta_i = 1$ if the event was observed — compares the model's predicted survival probability at the event time to the empirical outcome. The score is 1 minus the predicted survival probability at the observed event time, so a well-calibrated model assigns low scores to its calibration points.

**IPCW weights for censored observations.** For censored observations, the score cannot be computed directly. Inverse probability of censoring weights (IPCW), derived from a Kaplan-Meier censoring estimator, reweight the observed-event calibration points to represent the full population. This is the standard IPCW approach from Uno et al. (2011, Statistics in Medicine), applied to the conformal calibration step.

**Density ratio weights for covariate shift.** Covariate shift between calibration and test is corrected via importance weights $w(x) = p_\text{test}(x) / p_\text{train}(x)$. These are estimated by training a logistic classifier to distinguish calibration from test observations, then reading off the probability ratio — the same approach used in Tibshirani et al. (NeurIPS 2019). Each calibration point receives a weight proportional to its density under the test distribution.

**Weighted quantile.** The two weight systems are combined. The weighted calibration quantile $\hat{q}^\omega_\alpha$ replaces the standard unweighted quantile. The lower prediction bound is the time $t$ at which the model predicts $\hat{S}(t \mid x_\text{test}) = 1 - \hat{q}^\omega_\alpha$.

The result is a prediction lower bound: a time before which the test observation's event will occur with probability at most $\alpha$. In lapse terms: a date by which a specific policyholder will have lapsed with probability at most 10%.

---

## What this looks like in practice

The method accepts any survival model that outputs $\hat{S}(t \mid x)$. In our stack, that means anything from `lifelines` — Cox PH, Weibull AFT, the LogNormal — or the `WeibullMixtureCureFitter` in `insurance-survival`. The library code does not exist yet (this is explicitly a blog-first situation), but the algorithm is transparent enough to implement directly.

Here is a sketch of the calibration and prediction steps, using a Cox model and the standard logistic density ratio estimator:

```python
import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── 1. Fit survival model on training data ──────────────────────────────────
cox = CoxPHFitter()
cox.fit(train_df, duration_col="months_to_lapse", event_col="lapsed")

# ── 2. Estimate censoring weights via Kaplan-Meier on calibration set ───────
# Censoring indicator: 1 if censored, 0 if event observed
kmf_cens = KaplanMeierFitter()
kmf_cens.fit(
    cal_df["months_to_lapse"],
    event_observed=(1 - cal_df["lapsed"])
)

def ipcw(t, delta, km_cens):
    """IPCW weight for observation with event time t, event indicator delta."""
    if delta == 0:
        return 0.0  # censored — excluded from calibration scores
    g = km_cens.survival_function_at_times(t).iloc[0]
    return 1.0 / max(g, 1e-6)

ipcw_weights = np.array([
    ipcw(row["months_to_lapse"], row["lapsed"], kmf_cens)
    for _, row in cal_df.iterrows()
])

# ── 3. Estimate covariate shift weights via logistic classifier ─────────────
X_cal = cal_df[feature_cols].values
X_test = test_df[feature_cols].values
X_combined = np.vstack([X_cal, X_test])
y_combined = np.array([0] * len(X_cal) + [1] * len(X_test))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

clf = LogisticRegression(max_iter=500, C=0.1)
clf.fit(X_scaled, y_combined)
probs = clf.predict_proba(X_scaled)

p_test_given_x = probs[:, 1]
p_train_given_x = probs[:, 0]
# Avoid divide-by-zero
ratio = p_test_given_x / np.clip(p_train_given_x, 1e-6, None)

shift_weights_cal = ratio[:len(X_cal)]

# ── 4. Compute conformity scores on calibration set ─────────────────────────
def conformity_score(row, model):
    """1 - S(T_i | X_i): lower is better-calibrated."""
    t = row["months_to_lapse"]
    sf = model.predict_survival_function(row[feature_cols].to_frame().T)
    s_at_t = float(sf.loc[t] if t in sf.index else np.interp(
        t, sf.index, sf.values.ravel()
    ))
    return 1.0 - s_at_t

scores_cal = np.array([
    conformity_score(row, cox)
    for _, row in cal_df[cal_df["lapsed"] == 1].iterrows()
])
observed_cal = cal_df[cal_df["lapsed"] == 1]

ipcw_w = ipcw_weights[cal_df["lapsed"].values == 1]
shift_w = shift_weights_cal[cal_df["lapsed"].values == 1]
combined_weights = ipcw_w * shift_w
combined_weights /= combined_weights.sum()

# ── 5. Weighted quantile ─────────────────────────────────────────────────────
alpha = 0.10  # target miscoverage
# Weighted quantile: find score q such that sum of weights for scores <= q >= 1-alpha
sorted_idx = np.argsort(scores_cal)
sorted_scores = scores_cal[sorted_idx]
sorted_weights = combined_weights[sorted_idx]
cumulative = np.cumsum(sorted_weights)
q_hat = sorted_scores[np.searchsorted(cumulative, 1 - alpha)]

# ── 6. Prediction lower bound for new test observations ─────────────────────
def predict_lapse_lower_bound(x_test_row, model, q_hat):
    """Return month by which lapse occurs with probability <= alpha."""
    sf = model.predict_survival_function(x_test_row[feature_cols].to_frame().T)
    # Find t such that 1 - S(t|x) = q_hat, i.e. S(t|x) = 1 - q_hat
    target_survival = 1.0 - q_hat
    times = sf.index.values
    survival = sf.values.ravel()
    # First time survival drops to or below target
    idx = np.searchsorted(-survival, -target_survival)
    return times[min(idx, len(times) - 1)]

# Example: predict lower bound for each test observation
test_df["lapse_lower_bound_months"] = [
    predict_lapse_lower_bound(row, cox, q_hat)
    for _, row in test_df.iterrows()
]
```

The calibration overhead is modest. Computing IPCW weights takes one Kaplan-Meier fit. The density ratio classifier is a standard logistic regression on the combined calibration-plus-test feature matrix. At prediction time, the only cost is looking up the survival function at the quantile threshold.

---

## The insurance application: lapse timing under portfolio composition shift

The natural use case is a lapse model deployed across a renewal cycle where the portfolio composition has changed materially from the calibration cohort.

Consider a UK home insurer who calibrated their lapse model in 2022 on a book that was 70% direct channel, 30% aggregator. By 2025, the mix had inverted following Confused.com and Compare the Market's pricing algorithm changes: 65% aggregator, 35% direct. Aggregator customers have structurally different lapse dynamics — higher price sensitivity, shorter tenures, different NCD distributions. A conformal lapse timing model calibrated on the 2022 cohort will be overconfident about when aggregator customers will lapse. The prediction intervals will be too narrow for the aggregator segment and too wide for the direct segment. The shift weights correct for this.

The correction is measurable. The Shin et al. simulations, run on Weibull survival times with 20–40% censoring and a deliberate covariate shift (mean shift of one standard deviation on a key predictor), show that unweighted conformal miscoverage runs at 5–10 percentage points below nominal at 90% coverage targets. The weighted method restores nominal coverage. Intervals are also 15–30% narrower than naive unweighted intervals under shift, because the weights prevent the calibration quantile from inflating to compensate for distribution mismatch.

---

## The regulatory hook: CLV under Consumer Duty

The FCA's Consumer Duty requires demonstrable fair value across customer segments. `insurance-survival`'s `SurvivalCLV` computes expected customer lifetime value from a survival function $\hat{S}(t \mid x)$. If that survival function is uncertain — because the model is deployed under covariate shift — the CLV calculation inherits that uncertainty.

The survival conformal lower bound gives you a principled way to put error bars on the CLV estimate. If the 90% conformal lower bound on lapse time for a policyholder is 18 months, and the point estimate of expected lapse time is 28 months, you can compute both a point-estimate CLV and a conservative CLV based on the lower bound. This range is directly relevant to fair value assessments: if the conservative CLV is below the premium paid, that is a signal worth investigating under the Consumer Duty outcomes framework.

FCA supervisors have begun asking insurers how confident they are that fair value assessments hold across customer segments with different risk and behavioural profiles. Survival conformal intervals, properly adjusted for the segment-level distribution shift, are a direct answer to that question.

---

## What the existing stack does and does not provide

`insurance-conformal` at v0.7.1 has four relevant components:

- `InsuranceConformalPredictor`: split conformal for Tweedie and Poisson regression — pure claims severity and frequency, no survival outcomes
- `LocallyWeightedConformal`: adaptive-width intervals for heteroscedastic claims
- `ConditionalCoverageERT`: coverage diagnostics (the [conditional coverage post]({{ site.baseurl }}{% post_url 2026-03-31-conditional-coverage-diagnostics-conformal-prediction-ert %}) covers this in detail)
- `RetroAdj`: online conformal with retrospective adjustment for temporal drift

None of these handle censored outcomes. The covariate shift correction in `ShiftRobustConformal('weighted')` from `insurance-covariate-shift` exists for regression, but uses mean calibration weights rather than per-test-point density ratios. For a thorough treatment of covariate shift in conformal prediction for regression, the [KMM-CP post from earlier today]({{ site.baseurl }}{% post_url 2026-03-31-kmm-conformal-prediction-covariate-shift-insurance-pricing %}) is the right starting point.

`insurance-survival` has `WeibullMixtureCureFitter`, `LapseTable`, and `SurvivalCLV` — the right infrastructure for lapse modelling — but nothing relating to conformal prediction or uncertainty quantification on survival predictions.

The gap is real: there is no pip-installable Python library, as of March 2026, that produces finite-sample-valid prediction intervals for survival outcomes under covariate shift. The code above is implementable from scratch in a few hundred lines using `lifelines` and `scikit-learn`, both of which are already dependencies of the Burning Cost stack.

---

## Why we are not building this yet

The case for building `insurance_conformal.survival` as a new subpackage is genuine — the method fills a real gap, the home is clear, the dependency footprint is minimal. [Survival models for lapse prediction]({{ site.baseurl }}{% post_url 2026-03-28-survival-models-lapse-prediction %}) and the [conformal prediction for non-exchangeable claims]({{ site.baseurl }}{% post_url 2026-03-15-conformal-prediction-for-non-exchangeable-claims-time-series %}) posts have both attracted the kind of interest that suggests there is an audience for this. But a few things give us pause.

The paper comes from a clinical research group at Yonsei University. The examples and framing are oncology-first. Translating that to insurance lapse involves non-trivial domain adaptation, not just interface wrapping. The question of what constitutes the right conformity score for a lapse model with a cure component — where a proportion of policyholders will never lapse within a product lifetime — is not addressed in the paper. `WeibullMixtureCureFitter` outputs a two-component survival function; applying the Shin et al. scores naively to a cure model prediction ignores the structural zero in the cure fraction.

We also do not yet have a clear signal that UK pricing teams are querying per-policy lapse timing intervals rather than expected survival curves. If the question is "when will this customer lapse?", survival conformal intervals are the right tool. If the question is "what is the probability this customer lapses before renewal?", standard binary conformal predictors on the 12-month lapse indicator work fine and are simpler.

If this post generates demand for the former use case, the build decision is straightforward. The interface would be:

```python
from insurance_conformal.survival import SurvivalConformalPredictor

scp = SurvivalConformalPredictor(
    model=lifelines_fitter,
    shift_estimator='logistic',  # or 'kernel' for KMM-style weights
    censoring_estimator='kaplan-meier'
)
scp.calibrate(X_cal, T_cal, event_cal, X_test=X_test)
bounds = scp.predict_lower_bound(X_test, alpha=0.10)
```

The `X_test` parameter at calibration time is required because the shift weights depend on the test distribution — the weights are estimated jointly over calibration and test.

---

## What to read next

The theoretical foundation sits on two earlier papers: Candès, Lei and Ren (JRSSB 2023, arXiv:2103.09763) for censored conformal prediction without shift, and Tibshirani, Barber, Candès and Ramdas (NeurIPS 2019, arXiv:1904.06019) for weighted conformal prediction under shift for regression. Both are worth reading in that order before the Shin et al. paper.

For the insurance-survival infrastructure this would build on, the [survival models for insurance retention]({{ site.baseurl }}{% post_url 2026-03-11-survival-models-for-insurance-retention %}) post covers the `insurance-survival` library in full. For covariate shift correction in the regression setting, start with the [KMM-CP post]({{ site.baseurl }}{% post_url 2026-03-31-kmm-conformal-prediction-covariate-shift-insurance-pricing %}) before reading arXiv:2512.03738.

The full paper is at [arXiv:2512.03738](https://arxiv.org/abs/2512.03738).
