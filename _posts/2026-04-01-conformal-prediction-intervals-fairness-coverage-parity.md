---
layout: post
title: "Your Prediction Intervals Are Unfair (And You Haven't Checked)"
date: 2026-04-01
categories: [fairness, techniques]
tags: [conformal-prediction, fairness, coverage-parity, Consumer-Duty, Equality-Act, GCF, demographic-parity, equal-opportunity, insurance-conformal, claims-frequency, UK-insurance, arXiv-2505.16115, Vadlamani, ICLR-2025, EqA-s19, FCA-TR24-2, python]
description: "Vadlamani et al. (ICLR 2025, arXiv:2505.16115) formalise fairness at the prediction-set level. A model can be statistically valid at 90% coverage while covering elderly policyholders at 70% and younger ones at 95%. Here is what that means for UK insurance, and what to do about it."
math: true
author: burning-cost
---

Pricing teams have spent years on point-estimate fairness. Mean premium parity. Demographic parity ratios on GLM outputs. Proxy discrimination audits checking whether age or postcode correlates with the score. This work matters, and the FCA's multi-firm review TR24/2 (August 2024) was unambiguous that most firms were not doing enough of it.

But there is a fairness dimension that has gone almost completely unexamined, even in the most diligent audit workflows: whether the uncertainty quantification itself is fair.

Conformal prediction has entered insurance modelling as the distribution-free alternative to parametric uncertainty bands. A 90% prediction interval that contains the true outcome 90% of the time, with no assumptions about the error distribution. That guarantee is real. What it does not tell you is 90% for whom.

---

## The problem with marginal coverage

Split conformal prediction offers a marginal coverage guarantee: across the full calibration population, $P(Y \in C(X)) \geq 1 - \alpha$. For $\alpha = 0.10$, your intervals contain the true value at least 90% of the time.

This is an aggregate statement. It is entirely consistent with the following scenario.

Your calibration set is 10,000 policyholders. 1,000 are aged 70 or over; 9,000 are under 70. You fit a standard conformal predictor using a single nonconformity threshold — the 90th percentile of calibration scores. That threshold is dominated by the score distribution of the 9,000-policyholder majority. The elderly minority get whatever coverage their score distribution happens to produce at the threshold that the majority determined.

If the model's errors are larger and more variable for elderly policyholders — which is common, because they are underrepresented in training data and have more heterogeneous risk profiles — their nonconformity scores will be higher on average. A threshold calibrated on the majority will under-cover them. You can hit 90% marginal coverage while covering under-70s at 93% and over-70s at 71%. The intervals are statistically valid. They are demographically asymmetric.

Vadlamani, Srinivasan, Maneriker, Payani and Parthasarathy (ICLR 2025, [arXiv:2505.16115](https://arxiv.org/abs/2505.16115)) tested how bad this can get on real benchmark datasets. They found that baseline conformal prediction can produce coverage ratios across demographic groups as low as 0.25 — one group's intervals are four times less reliable than another's. Not a marginal imbalance. A four-to-one disparity, across a calibration procedure that everyone involved would describe as "well-calibrated."

---

## What conformal fairness means

The GCF paper formalises three fairness criteria at the prediction-set level, directly mapping the classical fairness literature into the conformal setting.

Let $G$ be the set of demographic groups, $C_\lambda(x)$ the prediction set under nonconformity threshold $\lambda$, and $\tilde{y}$ the advantaged label (e.g., "no claim" or "claim included in prediction set"). The tolerance $c$ is a user-specified maximum acceptable gap.

**Conformal Demographic Parity** asks whether the probability of $\tilde{y}$ appearing in the prediction set is equal across groups, conditioning only on group membership:

$$|P(\tilde{y} \in C_\lambda(X) \mid X \in g_a) - P(\tilde{y} \in C_\lambda(X) \mid X \in g_b)| < c$$

**Conformal Equal Opportunity** conditions on the true label as well — among policyholders who actually belong to the advantaged class, does the model include that class equally for all demographic groups?

$$|P(\tilde{y} \in C_\lambda(X) \mid Y = \tilde{y}, X \in g_a) - P(\tilde{y} \in C_\lambda(X) \mid Y = \tilde{y}, X \in g_b)| < c$$

**Conformal Disparate Impact** applies the four-fifths rule — the EEOC standard that a protected group should receive at least 80% of the rate accorded to the reference group:

$$\frac{1 - \alpha_{\max}}{1 - \alpha_{\min}} \geq 0.8$$

where $\alpha_{\max}$ and $\alpha_{\min}$ are the per-group miscoverage rates. The paper shows baseline conformal can produce ratios as low as 0.25. The GCF algorithm recovers ratios of 0.79–0.82 under a tight tolerance $c$.

None of these are exotic criteria. They are the same concepts — demographic parity, equal opportunity, disparate impact — that actuaries already apply to point predictions. The paper's contribution is applying them to prediction sets rather than point estimates, and providing a constructive algorithm for satisfying them.

---

## The GCF algorithm

The existing alternative, from Romano, Barber, Sabatti and Candès (2020, HDSR), handles this by splitting the calibration set by demographic group and computing a separate quantile threshold per group. The problem with this approach is efficiency: when group sizes differ, the per-group calibration sets have very different sizes, and the minority group's threshold is based on far fewer observations. Romano's approach forces coverage equality; GCF's insight is that you do not need equality, only a bounded gap, and a single threshold can achieve that gap while being considerably more efficient.

The algorithm works as follows. For each candidate threshold $\lambda$ in a search space built from the nonconformity scores, and for each group-label combination, the algorithm computes the empirical coverage that threshold would produce for that group:

$$\text{coverage}_{g,\tilde{y}} = Q^{-1}(\lambda, S_\text{calib}^{g,\tilde{y}})$$

This is simply the proportion of the group's calibration scores that fall below $\lambda$ — the fraction of that group who would be covered. The fairness check is then:

$$\max_g \text{coverage}_{g,\tilde{y}} - \min_g \text{coverage}_{g,\tilde{y}} \leq c$$

Run this for every $\lambda$ in the search space. Collect all the $\lambda$ values that satisfy the fairness constraint. The optimal threshold is the smallest one that does — the minimum threshold that satisfies fairness, which maximises efficiency by minimising prediction set size.

The practical efficiency cost of this relative to unconstrained conformal prediction is typically 5–15% larger prediction sets. Romano's equalized coverage approach inflates sets by 30–50%. The difference matters at scale.

---

## Where this applies in UK insurance

The GCF framework is built for classification. Prediction sets over discrete outcome categories. This does not include continuous premium regression — that is worth stating plainly, because it is the dominant use case in UK personal lines pricing. If you are fitting a Tweedie GLM or a gradient boosted pure premium model, GCF does not apply to your output directly.

What it does apply to, right now, within a standard UK insurance modelling stack:

**Claims frequency models.** Binary: claim / no-claim. The prediction set is one of $\{\text{no-claim}\}$, $\{\text{claim}\}$, or $\{\text{no-claim, claim}\}$. Conformal Equal Opportunity asks: among policyholders who will actually make a claim, does the model include "claim" in the prediction set equally for 25-year-olds and 75-year-olds?

A model that systematically excludes elderly true claimants from the "claim" prediction set is applying a provision, criterion or practice — the calibration procedure — that puts older policyholders at a particular disadvantage. That is the definition of indirect discrimination under Equality Act 2010 s.19.

**Decline / refer / accept models.** Underwriting triage with three-class output. The four-fifths disparate impact check directly tests whether the acceptance prediction set appears less frequently for protected groups.

**Tariff band assignment.** If continuous pricing outputs are discretised into $N$ rating bands for quote presentation or group policyholder products, GCF applies to the band as the classification label.

**Reserve adequacy models.** A model that produces prediction intervals for ultimate claims. GCF-style coverage parity would check that reserve intervals are equally valid for policies from different geographic areas — postcode as a proxy for ethnicity via ONS Census 2021 MSOA-level ethnicity data.

---

## A claims frequency example

Consider a motor claims frequency model trained on a portfolio of 200,000 policies annually. You have fitted a LightGBM binary classifier, calibrated it using split conformal prediction at $\alpha = 0.10$, and hit 90.3% marginal coverage on a held-out test set. Everything looks correct.

Now segment the test set by age band: under 30, 30–69, 70 and over. Compute empirical coverage per group. The over-70 segment might look like this:

| Age group | n (calibration) | Coverage | Gap from 90% |
|-----------|----------------|----------|--------------|
| Under 30  | 8,200          | 91.4%    | +1.4%        |
| 30–69     | 38,600         | 90.5%    | +0.5%        |
| 70+       | 3,200          | 79.8%    | −10.2%       |

A 10-point coverage gap for elderly policyholders. The intervals are narrower for this group than the model's confidence warrants. Predictions about elderly claimants are presented with more apparent certainty than is justified.

The Equal Opportunity framing makes this concrete: among actual claimants aged 70+, only 79.8% have their true outcome covered by the prediction set, against 91.4% for under-30 claimants. The model is less reliable precisely for the group whose claims it asserts to know about.

Under Consumer Duty PRIN 2A, this is an outcome. Outcomes monitoring requires evidence that model outputs are fair across customer segments including those sharing protected characteristics. "We hit 90% marginal coverage" does not discharge that obligation. "We hit 90% marginal coverage and our coverage gap across age bands is below 3 percentage points" does.

---

## What to do today

You do not need to implement GCF from scratch to start checking coverage parity. The `ConditionalCoverageERT` in `insurance-conformal` includes a `subgroup_coverage` method that does exactly the diagnostic step.

```python
from insurance_conformal.conditional_coverage import ConditionalCoverageERT
import numpy as np

ert = ConditionalCoverageERT(loss="l1", direction="under", n_splits=5)

# y_lower, y_upper: interval bounds from your conformal predictor
# y_true: observed outcomes
# X_groups: array with a column encoding group membership
# (age band as integer, 0=<30, 1=30-69, 2=70+)

subgroup_report = ert.subgroup_coverage(
    X=X_groups,
    y_lower=y_lower,
    y_upper=y_upper,
    y_true=y_true,
    alpha=0.10,
    feature_names=["age_band"],
    n_bins=3,
)
print(subgroup_report)
```

This returns a Polars DataFrame with empirical coverage, target coverage, and coverage gap per bin. A coverage_gap above 0.03 for any age or sex group is worth investigating further and documenting for your model governance record.

For continuous outcome models (severity, pure premium), this is the right diagnostic. You are not checking conformal fairness in the GCF sense — your outputs are regression intervals, not classification prediction sets — but you are checking whether the intervals are equally valid across protected groups, which is the operationally important question.

For classification outputs (frequency, declines), the GCF threshold search itself is a one-day implementation on top of the calibration scores your conformal predictor already produces. The algorithm requires the calibration nonconformity scores, group labels for the calibration set, a choice of fairness metric (Equal Opportunity is the most defensible under EqA s.19), and a tolerance $c$ — we would suggest $c = 0.05$ as a starting point. Implement `Satisfy_lambda(lambda)` as a loop over the threshold grid, check the coverage gap, return the minimum satisfying threshold.

---

## The finite-sample constraint

GCF's coverage guarantee for subgroup $g$ carries a slack term of $1/(n_g + 1)$ from Lemma 3.1. This is not a theoretical nuisance — it is a binding practical constraint.

| Group calibration size | Slack    |
|------------------------|----------|
| 1,000                  | 0.1%     |
| 200                    | 0.5%     |
| 100                    | 1.0%     |
| 25                     | 3.8%     |

For a 500,000-policy personal lines portfolio, age groups over 70 will typically have 20,000–40,000 calibration observations — no problem. But intersectional groups collapse quickly. Elderly women with specific occupation codes in a specific region might have 150 calibration observations. That is not enough for GCF to make reliable guarantees.

Our recommendation: run GCF at the single-characteristic level (age band, sex). Use `ConditionalCoverageERT` for finer-grained conditional coverage diagnostics where group sizes are small. The ERT test does not require large per-group samples — it tests overall conditional coverage quality, not per-group statistics.

---

## The regulatory case

Consumer Duty (PS22/9, PRIN 2A) requires firms to evidence fair outcomes across customer segments. The FCA's TR24/2 review found that most firms relied on high-level aggregate metrics without granular evidence of differential outcomes by segment. A coverage-parity report — showing that prediction set coverage is within $c = 0.05$ of target across age and sex groups — is exactly the kind of quantitative evidence TR24/2 identified as missing.

Equality Act 2010 s.19 is the sharper instrument. Indirect discrimination occurs when a provision, criterion or practice puts persons sharing a protected characteristic at a particular disadvantage. A calibration procedure that produces systematically narrower intervals for elderly policyholders — resulting in model outputs that are less reliable for that group — is a PCP with potentially discriminatory effect.

The paper's authors are not insurance regulators, and the connection between GCF and EqA s.19 requires actuarial and legal judgement to apply. But the logic is not strained: if your model's uncertainty quantification is demonstrably less valid for protected groups, you are not providing equal service, and you will struggle to argue objective justification.

---

## What this does not solve

We are being explicit about the limits.

GCF is a classification framework. If your output is a continuous premium estimate or a severity prediction, GCF does not apply. Regression coverage fairness — ensuring that prediction intervals for continuous outcomes are equally valid across demographic groups — is an open problem. The right approach for regression outputs today is the `subgroup_coverage` diagnostic above, which tells you the empirical gap but does not provide a constructive correction.

The method requires group membership at calibration time, which is fine for an audit. Using group-specific thresholds at prediction time — adjusting the threshold because you know a policyholder is elderly — raises different legal questions under the Equality Act and requires separate analysis.

GCF also assumes exchangeability between calibration and deployment. If your portfolio's demographic composition shifts between calibration and deployment (plausible if calibration data is from 2022–2023 and the model is deployed in 2026), the coverage-parity guarantees may be stale. Re-certify after each model refresh.

---

## Comparison to Romano 2020

Romano, Barber, Sabatti and Candès ("With Malice Toward None," HDSR 2020) proposed equalized coverage: compute a separate quantile threshold per demographic group. The threshold for group $g$ is calibrated only on group $g$'s calibration data.

That is simpler to implement and explain to regulators. When group sizes are roughly equal and you genuinely want identical coverage, it is adequate. GCF's advantages are efficiency (5–15% larger sets vs. 30–50%), compatibility with any nonconformity score function, and tolerance-based control rather than forced equality. They are not incompatible approaches — Romano's method can be derived as a special case of GCF with $c = 0$ and per-group threshold search.

---

## The broader point

Insurance modelling has a habit of treating fairness as a point-estimate problem. The audit checks whether mean premiums are similar across groups, whether demographic parity ratios are within acceptable bounds on GLM outputs, whether proxy variables are correlated with protected characteristics. These checks are necessary and the FCA expects to see them.

They are not sufficient. Every model that outputs uncertainty estimates — prediction intervals, quantile bands, probability sets — is making a claim about how confident it is. If that claim is systematically more reliable for some groups than others, the uncertainty itself is discriminatory in effect. A policyholder whose claim outcome falls outside the model's 90% interval is not getting the service the model promised them. If that happens more often for elderly policyholders, that is not a coincidence to note and move on from.

The GCF framework gives this a precise formulation and a practical remedy. It is classification-only, which limits the primary application to frequency models and underwriting decisions rather than pure premium regression. But within that scope, it is the first method that provides a formal fairness guarantee at the prediction-set level with a constructive, efficient algorithm for achieving it.

Check your coverage by group. It takes an afternoon and three lines of code. You probably have not done it.

---

## The paper

Vadlamani, A.T., Srinivasan, A., Maneriker, P., Payani, A. & Parthasarathy, S. (2025). 'A Generic Framework for Conformal Fairness.' ICLR 2025. [arXiv:2505.16115](https://arxiv.org/abs/2505.16115). Authors affiliated with The Ohio State University and Cisco Research.

---

## Related posts

- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/03/31/conformal-prediction-intervals-insurance-pricing-python/) — conformal interval foundations; locally weighted calibration (LWC) coverage forthcoming
- [Proxy Discrimination in Insurance Pricing: What the Fairness Audit Actually Tests](/insurance-fairness/) — the point-estimate audit that GCF complements, not replaces
- [Selective Conformal Prediction for Automated Underwriting](/2026/04/01/selective-conformal-prediction-automated-underwriting-conditional-vs-marginal-risk/) — conditional vs. marginal coverage in underwriting triage; the coverage distinction that motivates the GCF framework
