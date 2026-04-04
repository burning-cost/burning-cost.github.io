---
layout: post
title: "Conformal Prediction Works on Average — But Does It Work for Your Riskiest Customers?"
date: 2026-04-04
categories: [conformal-prediction, insurance-pricing]
tags: [conformal-prediction, conditional-coverage, cvi, cc-select, model-selection, insurance-conformal, zhou-2026, arXiv-2603.27189, consumer-duty, motor-insurance, fairness, actuarial, uncertainty-quantification, subpopulation-coverage]
description: "Marginal coverage guarantees hold on average — but average is not good enough for pricing. Zhou, Zhang, Tao and Yang (arXiv:2603.27189, March 2026) reframe conditional coverage assessment as supervised learning: train a reliability estimator, compute the Conditional Validity Index, and select among conformal predictors by CVI. We connect this to Consumer Duty and the insurance-conformal assessment module."
math: true
author: burning-cost
---

A conformal predictor targeting 90% coverage achieves 90% coverage. That is the guarantee, and it holds: finite-sample, distribution-free, no ifs.

Here is what it does not guarantee. It does not guarantee that your young urban drivers are covered at 90%. Or your motor trade accounts. Or your high-value vehicle segment. The 90% is a portfolio average. A model that covers 97% of your mature standard risks and 78% of your high-risk tail passes the guarantee with room to spare. The average looks fine. The tail does not.

This matters more for pricing than for most prediction applications. The riskiest customers are where your model's uncertainty is greatest, where premium inadequacy is most likely, and where — under FCA Consumer Duty — the obligation to evidence fair outcomes is hardest to satisfy. A conformal predictor that systematically underpredicts its own uncertainty for those customers is a liability even if its portfolio statistics are clean.

Zhou, Zhang, Tao and Yang (arXiv:2603.27189, March 2026) give us a way to measure this precisely and use the measurement to choose better models.

---

## The conditional coverage problem

The distinction between marginal and conditional coverage is worth stating clearly.

**Marginal coverage**: $P(Y \in C(X)) \geq 1 - \alpha$. The probability is over the joint distribution of $(X, Y)$. Any correctly implemented split-conformal predictor achieves this by construction — it is a theorem, not an assumption.

**Conditional coverage**: $P(Y \in C(X) \mid X = x) \geq 1 - \alpha$ for all $x$. Point-wise validity across the feature space. This is what you actually need for per-risk applications.

Foygel Barber et al. (2019) showed that exact finite-sample conditional coverage is impossible in a distribution-free setting — you cannot have both marginal guarantees and exact conditional validity without restricting the model class. So conditional coverage is a spectrum, not a binary. The question is not "does this predictor achieve conditional coverage?" but "how badly does it fail conditional coverage, and where?"

That is the question CVI answers.

---

## The CVI construction

The insight is to reframe conditional coverage assessment as a supervised learning problem.

Take your calibrated conformal predictor and an evaluation set $(X_i, Y_i)$ that it has not seen. For each observation, generate a binary coverage indicator:

$$I_i = \mathbf{1}\{Y_i \in C(X_i)\}$$

This is 1 if the prediction interval contained the true outcome, 0 if it missed. Now train a probabilistic classifier to predict $I_i$ from $X_i$. After isotonic probability calibration, this classifier produces $\hat{\eta}(x)$: an estimate of the local coverage probability at each point in feature space.

If coverage is uniform, $\hat{\eta}(x) \approx 1 - \alpha$ everywhere and the classifier learns nothing useful. If coverage varies across the feature space — if it is lower for young high-risk drivers and higher for mature standard risks — the classifier finds that pattern.

The Conditional Validity Index is then the expected deviation of $\hat{\eta}(x)$ from target:

$$\text{CVI} = \mathbb{E}[\phi(\hat{\eta}(X))]$$

where the penalty function $\phi$ decomposes into two components with different economic content:

$$\phi(c) = \begin{cases} (1 - \alpha) - c & \text{if } c < 1 - \alpha \quad \text{(Safety: undercoverage)} \\ c - (1 - \alpha) & \text{if } c > 1 - \alpha \quad \text{(Efficiency: overcoverage)} \end{cases}$$

In practice, CVI decomposes as:

$$\text{CVI} = \text{CVI}_U + \text{CVI}_O$$

$$\text{CVI}_U = \hat{\pi}_- \cdot \text{CMU}, \qquad \text{CVI}_O = \hat{\pi}_+ \cdot \text{CMO}$$

where $\hat{\pi}_-$ is the proportion of observations with $\hat{\eta}(x) < 1 - \alpha$, CMU (Conditional Mean Undercoverage) is the average shortfall among those points, and the overcoverage terms are defined symmetrically.

**CVI$_U$ is the dangerous component.** It tells you what fraction of your portfolio has systematically undercovered intervals, and by how much. High CVI$_U$ means you are issuing confident-looking uncertainty estimates to precisely the customers where your model is most uncertain. For a 90% predictor, a CVI$_U$ of 0.04 driven by $\hat{\pi}_- = 0.15$ and CMU = 0.27 means 15% of your policyholders are experiencing roughly 63% coverage — not 90%.

**CVI$_O$ is the costly component.** Overcoverage wastes margin. If you are holding capital against interval uncertainty that is wider than necessary for the well-understood segments of your book, you are paying a competitive cost for a non-existent safety benefit.

The ratio CVI$_U$ / CVI$_O$ is a directional read on whether miscoverage is biased toward the dangerous side. For insurance applications, if this ratio is substantially above 1, the undercoverage problem deserves investigation before the overcoverage problem.

---

## Why stratification does not work at scale

The standard alternative — compute coverage separately by rating factor cell — sounds sensible until you try it on a real motor book.

A UK personal lines motor rating structure with age band (8 levels) × vehicle group (20 levels) × area (6 levels) × NCB (6 levels) × occupation (15 levels) produces over 86,000 cells on five factors alone. A calibration set of 20,000 policies gives an average of 0.23 policies per cell. Most cells have zero observations. The ones that have a few have coverage estimates with confidence intervals of ±40 percentage points.

CVI sidesteps this by treating conditional coverage assessment as a generalisation problem. The LightGBM classifier learns that coverage is low for young drivers in urban areas — not just for the exact cell "age band 3, vehicle group 15, area 2, NCB 1, occupation 7", but for the combination of features that characterises that risk type. It transfers from cells with observations to cells without them, in the same way any other supervised learner would.

The output is more useful too. Instead of a sparse table of unreliable cell-level estimates, you get: a single scalar CVI (comparator-safe), the U/O decomposition (tells you which problem to address), and per-observation $\hat{\eta}(x)$ scores that tell you how much to trust the interval for any specific policy.

---

## CC-Select: choosing between conformal predictors

If you have multiple candidate conformal predictors — say, `pearson_weighted`, `deviance`, and `anscombe` non-conformity scores, or different calibration window lengths, or different base models — CVI gives you a principled basis for choosing between them.

The CC-Select algorithm from the paper runs CVI estimation over $K$ repeated random splits of the evaluation data (default $K = 5$, to reduce Monte Carlo noise), and selects the predictor with the lowest mean CVI. The paper establishes consistency: as the sample size grows, CC-Select converges to the oracle choice — the predictor that genuinely has better conditional coverage.

The key point is that CC-Select selects on conditional coverage, not marginal coverage. Two predictors can have identical marginal coverage (both 90%) while one has CVI$_U$ = 0.01 and the other has CVI$_U$ = 0.06. The marginal metric cannot distinguish them. CVI can.

---

## Consumer Duty: the number you need to report

FCA Consumer Duty (PS22/9) requires firms to evidence good outcomes for retail customers. The FCA's EP25/2 discussion paper on proxy discrimination is specific: pricing tools that produce materially worse outcomes for identifiable customer groups carry regulatory exposure regardless of aggregate performance.

Conditional coverage of prediction intervals is exactly the kind of differential outcome EP25/2 is concerned with. If your uncertainty estimates are systematically less reliable for customers with certain characteristics — young drivers, certain occupations, certain geographic areas — that is a measurable fairness failure, not just a statistical curiosity.

A conformal predictor covering 90% overall but 82% for young urban drivers is a Consumer Duty exposure. CVI makes the 82% visible. Before CVI, the 90% aggregate masked it completely.

The practical governance case: a model risk sign-off that includes "CVI$_U$ = 0.008, $\hat{\pi}_- = 0.04$, no significant concentration in protected characteristics by $\hat{\eta}$ quintile" is substantively different from a sign-off that reports only marginal coverage. The first can be tested, challenged, and reproduced. The second is an average with no diagnostic value.

---

## Implementation overhead is low

The attractive feature of this approach is that it does not require any changes to the conformal predictor itself. You train the conformal predictor normally. You then train a binary classifier on the coverage outcomes from an evaluation set. Any classifier will do — LightGBM is the natural choice because it handles mixed feature types, categorical variables, and interaction effects without feature engineering.

The only input the classifier needs is the feature matrix and the binary coverage vector. It outputs $\hat{\eta}(x)$ for any new observation. Total implementation cost: one classifier fit, one evaluation.

In `insurance-conformal`, this is `ConditionalCoverageAssessor` from `insurance_conformal.assessment`. The stateful design matters for production: fit the assessor once against a held-out evaluation set, then query it repeatedly at different gamma tolerances, against different candidate predictors, or to score new policies as they arrive.

```bash
uv add "insurance-conformal[lightgbm]"
```

```python
from insurance_conformal import InsuranceConformalPredictor
from insurance_conformal.assessment import ConditionalCoverageAssessor

# Your predictor, calibrated as usual
predictor = InsuranceConformalPredictor(
    model=fitted_gbm,
    nonconformity="pearson_weighted",
    tweedie_power=1.5,
)
predictor.calibrate(X_cal, y_cal)

# Generate intervals on a held-out evaluation set
intervals_df = predictor.predict_interval(X_eval, alpha=0.10)
intervals_np = intervals_df[["lower", "upper"]].to_numpy()

# Fit the assessor — trains LightGBM classifier on coverage indicators
assessor = ConditionalCoverageAssessor(alpha=0.10, n_splits=5, random_state=42)
assessor.fit(X_eval, y_eval, intervals_np)

# CVI decomposition
result = assessor.cvi(gamma=0.1)
print(result)
# CVIResult(cvi=0.0421, cvi_u=0.0312, cvi_o=0.0109,
#           pi_minus=0.147, pi_plus=0.063, alpha=0.100, gamma=0.100, n_eval=1842)
```

`pi_minus = 0.147` means 14.7% of evaluation observations have estimated local coverage below the 90% target (outside the tolerance band at gamma=0.1). That is the population that needs attention.

To identify which policies are undercovered, cross-reference `assessor.eta_hat_` against your feature columns. If low-`eta_hat` observations concentrate in a particular age band or area code, you have a conditional coverage problem in a specific segment, not random noise across the portfolio.

---

## Limitations

**Sample size.** The classifier is trained on binary coverage indicators — a noisy signal. Below around 800 evaluation observations, the $\hat{\eta}$ estimates are unreliable. The library warns at instantiation. On a 50,000-policy book this is irrelevant. On a specialist commercial lines book with 900 calibration policies, CVI is indicative rather than actionable.

**Classifier capacity.** If LightGBM cannot find the conditional coverage pattern — because the signal is driven by high-order interactions in a small dataset — CVI will understate the true problem. This is the biased-conservative failure mode. Check the classifier's cross-validated AUC: if it is near 0.5, the classifier has no signal and the CVI estimate is unreliable.

**What CVI cannot do.** It tells you which of your candidate predictors has better conditional coverage on your data. It does not tell you whether the best candidate is good enough. CVI = 0.02 is better than CVI = 0.06. Whether CVI = 0.02 is acceptable for your portfolio is a judgment about the CVI$_U$ decomposition and where the undercoverage concentrates — not a number the paper gives you.

**No insurance benchmark.** The paper evaluates on nine datasets — Bike, Computer, Kin8nm, Meps21 and five others — none of which are insurance. The coverage heterogeneity patterns in insurance pricing (Poisson frequency with structural zeros, gamma severity with heavy tails, strong age-vehicle interactions) differ from the UCI regression benchmark collection. We are running this on FrenchMTPL via `insurance-datasets` and will publish results.

---

## The paper

Zheng Zhou, Xiangfei Zhang, Chongguang Tao and Yuhong Yang, "Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation and Selection", arXiv:2603.27189, March 2026. The CVI definition and phi-decomposition are in Section 2; the CC-Select algorithm and consistency results are in Sections 3–4.

---

## Related

- [Conformal Prediction Intervals for Insurance Pricing](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/) — the base `InsuranceConformalPredictor` with `pearson_weighted` scoring
- [Your Conformal Intervals Lie About Some Policyholders](/2026/04/02/conditional-coverage-assessor-cvi-consumer-duty-insurance-conformal-v120/) — stateful `ConditionalCoverageAssessor` in v1.2.0 with CVP curve and production interface
- [Conformal Risk Control Assumes Your Loss Decreases With Interval Width](/2026/04/04/non-monotone-conformal-risk-control-insurance-losses/) — non-monotone CRC and the Winkler score
- [Your Joint Prediction Sets Are 20–40% Too Wide](/conformal-prediction/insurance-pricing/2026/04/03/multivariate-conformal-prediction-mahalanobis-insurance/) — Mahalanobis nonconformity scores for correlated outputs
