---
layout: post
title: "The ENBP Counterfactual Problem Has a Statistical Answer"
date: 2026-03-11
categories: [libraries, causal-inference, compliance]
tags: [conformal-prediction, causal-inference, fca, consumer-duty, icobs-6b, lei-candes, weighted-conformal, propensity-score, sensitivity-analysis, python, insurance-counterfactual-sets]
description: "FCA EP25/2 admitted that individual-level ENBP counterfactual analysis 'is unable to be conducted'. Lei & Candès 2021 weighted conformal inference says otherwise. insurance-counterfactual-sets brings finite-sample valid harm assessments to the policy level for the first time."
---
> **Note:** The `insurance-counterfactual-sets` library referenced in this post has been archived. The code remains available at its [GitHub repository](https://github.com/burning-cost/insurance-counterfactual-sets) for reference, but is no longer actively maintained.


In July 2025, the FCA published Evaluation Paper 25/2 — its assessment of the GIPP remedies three years on. Buried in the methodology section is a sentence that should unsettle every pricing actuary doing ICOBS 6B compliance:

> "Policy-level counterfactual analysis is unable to be conducted because the ENBP counterfactual scenario is not observable at the policy level."

Read that again. The regulator that mandated equal pricing for renewals and equivalent new business customers is simultaneously acknowledging that we cannot actually verify, at the level of any individual policyholder, whether that mandate has been met. Current compliance rests on aggregate MI reports — average renewal versus new business price ratios, segment-level statistics, portfolio-wide summaries. Nobody is producing individual-level statistical evidence of harm or its absence.

This is the ENBP counterfactual problem. And it has, as of now, a statistical answer.

---

## What the problem actually is

The ENBP counterfactual is the price a renewing customer *would have been offered* had they been treated as a new business customer. It is unobservable. A customer either renews or they do not; they either receive the renewal price or the new business price. Nobody receives both.

This is the fundamental identification problem of causal inference: we observe Y(1) or Y(0), never both. In the ICOBS 6B context, Y(1) is what actually happened (the renewal price) and Y(0) is the counterfactual (what would have happened under new business pricing). The FCA wants firms to demonstrate that Y(1) ≤ Y(0), i.e., that renewal customers are not paying more than they would as new customers. But because Y(0) is never observed for renewers, any analysis of Y(0) is necessarily an estimate under assumptions.

The actuarial response to this has been to work on the average. Match renewal policyholders to comparable new business policyholders, compute the aggregate premium ratio, and report it to the board. This satisfies the letter of the current supervisory approach. It does not answer the individual-level question: for *this specific policyholder*, do we have statistical evidence that they are not harmed?

EP25/2's conclusion — that such analysis "is unable to be conducted" — reflects the state of actuarial practice, not the state of statistics. The statistics has caught up.

---

## The method: weighted conformal inference for counterfactuals

In 2021, Lei and Candès published a paper in the *Journal of the Royal Statistical Society Series B* — "Conformal Inference of Counterfactuals and Individual Treatment Effects" — that directly addresses this problem. The core insight is that conformal prediction, a distribution-free method for constructing prediction sets with finite-sample coverage guarantees, can be extended to the counterfactual setting by reweighting the conformal scores by the propensity score.

The standard split conformal construction works as follows. Train a model on half the calibration data, compute residuals on the held-out half, and find the empirical (1−α) quantile of those residuals. A prediction interval for a new observation is then: model prediction ± that quantile. This interval achieves exactly (1−α) marginal coverage — 95% of future observations will fall inside it, with no distributional assumptions.

Lei and Candès show that if you weight the residuals by inverse propensity scores — the probability that each calibration unit received the treatment it actually received — you get valid counterfactual prediction sets. For a renewing policyholder, you can construct a set [L, U] such that, under strong ignorability, at least 95% of renewal policyholders have their Y(0) contained in that interval. If the renewal premium sits above U, you have statistical evidence — finite-sample valid, not asymptotic — that this policyholder's renewal price exceeds what they would have paid as a new customer.

In observational settings (as opposed to randomised experiments), the guarantee relaxes slightly to asymptotic doubly-robust coverage: valid if either the outcome model or the propensity model is correctly specified, but not necessarily both. This is the standard DML robustness condition. For insurance applications, where we have good covariate data but no random assignment, this is the realistic case.

The Conformalized Quantile Regression variant — CQR, from Romano et al. 2019 — replaces a single residual score with quantile residuals. This produces tighter intervals when the response is heteroskedastic, which insurance claims data always is. A policyholder with a high expected loss has wider uncertainty than one with a low expected loss; CQR respects this rather than imposing uniform interval width.

---

## Using the library

`insurance-counterfactual-sets` is a pip-installable Python implementation of these methods, built specifically for the insurance pricing context. Install it:

```bash
pip install insurance-counterfactual-sets
```

The core workflow is three steps: fit, calibrate, predict.

```python
from insurance_counterfactual_sets import WeightedConformalITE, SensitivityAnalyzer, FCAHarmReport

# Fit outcome model and propensity model on training data
ite = WeightedConformalITE(
    outcome_model=catboost_model,
    propensity_model=lr,
    alpha=0.05  # 95% prediction sets
)
ite.fit(X_train, Y_train, T_train)

# Calibrate conformal scores on held-out data
ite.calibrate(X_cal, Y_cal, T_cal)

# Produce counterfactual prediction sets for renewals
# treatment_arm=0 means: what would this person's claims be under new-business pricing?
counterfactuals = ite.predict_counterfactual(X_test, treatment_arm=0)
```

`counterfactuals` is a DataFrame with columns `lower`, `upper`, and `point_estimate` for each policyholder. If the renewal premium sits above `upper`, the policyholder falls into the flagged set.

The `calibrate` call is where the propensity reweighting happens. It weights the calibration residuals by the inverse probability of treatment, so that the resulting quantile reflects the counterfactual distribution rather than the observed distribution. You need a reasonably large calibration set — we recommend at least 500 policyholders per treatment arm — and propensity scores that are neither too close to 0 nor 1. The library raises a warning if the effective sample size falls below 200 due to extreme weights.

---

## Sensitivity analysis: the Gamma report

Strong ignorability — the assumption that all confounders are observed — is never literally true. There will always be characteristics the pricing model does not capture that predict both treatment assignment (whether a customer received the renewal price or the new business price) and claims outcome. The question is not whether unmeasured confounding exists, but how much of it would be needed to overturn the finding.

Jin, Ren and Candès (2023, PNAS) extended the Lei-Candès framework to incorporate a marginal sensitivity model, following Tan (2006) and Rosenbaum (2002). The Gamma parameter bounds the degree of unmeasured confounding: Gamma=1 means no confounding beyond what is observed; Gamma=2 means the odds of treatment assignment could differ by a factor of 2 due to unmeasured variables. The library computes a Gamma value for each policyholder — the threshold at which their harm finding would be overturned.

```python
sa = SensitivityAnalyzer(ite)
gamma_values = sa.gamma_report(X_test)
```

The Cornfield condition — the standard epidemiological threshold — is Gamma=9: an unmeasured confounder would need to be associated with a ninefold increase in both the odds of treatment and the outcome to explain away the effect. In insurance, unmeasured confounders of that magnitude are implausible given the data richness of a modern pricing model. If your flagged policyholders have Gamma > 2, you have a reasonably robust finding.

The sensitivity report is the document you want in an FCA attestation pack alongside the interval estimates.

---

## Producing the attestation pack

The `FCAHarmReport` class wraps the full workflow into a structured output:

```python
report = FCAHarmReport(ite, sensitivity=sa)
report.fca_attestation_pack(
    X_renewals,
    Y_actual,
    renewal_premium,
    output_dir="./fca_evidence"
)
```

This writes to `output_dir`:

- `harm_flags.csv`: one row per policyholder, with predicted counterfactual interval, renewal premium, flag status, and Gamma value
- `coverage_audit.json`: empirical coverage on the calibration set with bootstrap confidence intervals
- `sensitivity_summary.pdf`: distribution of Gamma values across flagged policyholders
- `methodology_note.md`: auto-generated methodological description citing Lei & Candès 2021 and Jin et al. 2023

The methodology note is not legal advice and should be reviewed by your compliance team before submission. But it gives you the factual skeleton: which methods were used, what their coverage guarantees are, and what the sensitivity analysis reveals.

---

## Limitations: read these before using

We wrote these down because they matter. The library's test suite checks for correct calibration set size and propensity score overlap, but it cannot check whether your data satisfies the underlying assumptions.

**Marginal not conditional coverage.** The 95% guarantee is across the population, not for any specific policyholder. "95% of renewal policyholders have their counterfactual within the interval" is what the theory delivers. "This specific policyholder's counterfactual is within the interval with 95% probability" is a stronger claim that conformal inference does not make. The distinction matters for individual attestation. We think the population-level guarantee is still meaningful and defensible — it is more than any current compliance approach provides — but be precise about what you are claiming.

**Strong ignorability is an assumption.** The sensitivity analysis quantifies robustness to violations; it does not eliminate the requirement. If your pricing model is missing a major predictor of claims that also predicts treatment assignment — say, telematics data that you have for some customers but not others — the coverage guarantee is conditional on that assumption holding. The Gamma report tells you how bad the violation would need to be.

**Intervals can be wide for zero-inflated claims.** Motor claims frequency is typically 5-15% per annum. Most calibration observations have Y=0. The conformal intervals will often span zero to some moderate positive value, with a lower bound of zero. This is technically correct — the counterfactual is genuinely uncertain — but it limits the power to flag individual harm for low-claim-frequency products. Contents insurance is worse than motor in this respect. The CQR variant helps at the margin but does not resolve the fundamental signal-to-noise problem.

**FCA has not mandated ITE-level evidence.** EP25/2's methodology acknowledges the limitation but does not require individual-level analysis. Aggregate MI reporting remains sufficient for current compliance. We think the regulatory direction of travel is clear — you cannot indefinitely satisfy a rule about individual treatment by reporting only on averages — but we will not tell you this is currently required, because it is not.

---

## When to use it

Use this library if you are doing any of the following:

**Proactive harm scanning at renewal.** Before issuing renewal invitations, run the model against your renewing portfolio. Flag policyholders whose renewal price sits above the upper bound of the counterfactual interval. Investigate the flagged set before the invitations go out. This is prevention, not post-hoc attestation.

**FCA s.166 preparation.** If you are facing a skilled persons review of your GIPP compliance, "here is individual-level statistical evidence with finite-sample coverage guarantees" is a materially stronger position than "here is our quarterly aggregate MI report."

**Consumer Duty fair value assessment.** The Consumer Duty's fair value requirement is substantively about whether customers are paying a price that reflects value. A pricing model that produces counterfactual prediction sets is producing exactly the evidence that a fair value assessment should be based on.

**Research and model development.** The library is also useful as a model evaluation tool during pricing model development, for assessing heterogeneity in treatment effects across segments, and for academic work on insurance pricing methodology.

---

## Related reading

For the causal inference foundations — DML, TMLE, and propensity score methods in insurance — see our post on [insurance-causal](https://burning-cost.github.io/2026/02/25/causal-inference-for-insurance-pricing/). The conformal prediction methods that underpin the counterfactual sets are introduced in our post on [insurance-conformal](https://burning-cost.github.io/2026/03/11/conformal-scr-solvency/). Questions about protected characteristics and the interaction between counterfactual analysis and indirect discrimination law are addressed in the [insurance-fairness](https://burning-cost.github.io/2026/03/03/your-pricing-model-might-be-discriminating/) post. Deploying models of this kind in a production environment is covered in [insurance-deploy](https://burning-cost.github.io/2026/03/13/your-champion-challenger-test-has-no-audit-trail/).

---

The library is at [github.com/burning-cost/insurance-counterfactual-sets](https://github.com/burning-cost/insurance-counterfactual-sets).

```bash
pip install insurance-counterfactual-sets
```

The FCA said individual counterfactual analysis cannot be conducted at the policy level. That is a description of current practice. It does not have to stay that way.
