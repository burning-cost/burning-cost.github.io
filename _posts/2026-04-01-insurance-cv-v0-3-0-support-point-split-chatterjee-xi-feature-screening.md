---
layout: post
title: "Two Things Random Splits and Pearson Correlation Get Wrong in Insurance Data"
date: 2026-04-01
categories: [machine-learning, model-validation]
tags: [insurance-cv, train-test-split, feature-selection, SupportPointSplit, ChatterjeeSelector, Chatterjee-Xi, energy-distance, support-points, SPlit, Mak-Joseph-2018, zero-inflation, heavy-tails, InsurAutoML, arXiv-2603-18190]
description: "insurance-cv v0.3.0 adds SupportPointSplit (distributional train-test splitting via energy distance minimisation) and ChatterjeeSelector (nonlinear feature screening using Chatterjee's Xi correlation). Both address structural failures of the standard approaches on insurance data."
author: burning-cost
---

Every insurance pricing workflow does two things before model training: splits data into train and test, and screens features. Both are treated as solved problems. They are not.

Random train-test splits produce unstable hold-out sets. On a book with 5% large-loss frequency, you will see 90 large losses in your test set on one run and 110 on the next. That is not noise you can shrug at — it is the difference between a model that looks well-calibrated and one that looks systematically wrong in the tail, depending on random seed. Pearson and Spearman correlation miss the nonlinear relationships — U-shaped age curves, threshold effects near deductibles, log-linear sum-insured relationships — that are the actual signal in insurance loss data. If you screen features using either, you are discarding variables that matter.

`insurance-cv` v0.3.0 adds two new tools to fix these. `SupportPointSplit` produces distributional train-test splits that represent the full data distribution in a principled, reproducible way. `ChatterjeeSelector` screens features using Chatterjee's Xi correlation, which is consistent under any dependence structure including the nonlinear patterns that Pearson and Spearman miss.

---

## The problem with random splits on insurance data

The failure mode of random splitting is not random in the bad sense — it is systematic. Insurance loss distributions are heavy-tailed and zero-inflated. The bulk of observations are small or zero losses. The claims that drive most of your profit variability, your reinsurance layers, and your reserving uncertainty are rare events: large property losses, serious bodily injury cases, catastrophe-affected policies.

A random 80/20 split on a book of 100,000 motor policies might give you between 83 and 97 large losses (say, losses above £50,000) in the test set across repeated splits. Mean: 90 observations. Standard deviation: 7.7 (Guo et al., arXiv:2603.18190, simulated insurance data with 5% rare events, 2026). That 7.7 observation standard deviation translates to a coefficient estimation error of 0.0059 ± 0.0044 (2.6% relative error). For a feature with a true coefficient of 0.23, your estimate ranges from 0.22 to 0.24 depending on which random seed you use. Validation results are not stable.

Stratified splitting reduces this, but only if you know what to stratify on. Stratify on the target? You need to discretise it, which loses information in continuous severity models. Stratify on a large-loss indicator? Better, but now you are making a modelling decision about what constitutes a large loss, and you are stratifying only on the marginal target distribution, not on the joint (X, y) distribution. Policies at the 95th percentile of sum insured are not uniformly distributed across the target strata.

The deeper problem: you want your test set to represent the full population you will encounter in production. That means matching the joint distribution of all features and the target simultaneously. Stratified splitting handles one dimension at a time.

---

## What support points actually are

A support point set is a finite sample that best represents a distribution in a well-defined sense. "Best" here means minimising energy distance — a measure of discrepancy between two distributions that generalises the Cramér-von Mises statistic to multivariate data.

Given your full dataset of N observations in d dimensions, the support points $\{o^*_j\}_{j=1}^{n_{\text{test}}}$ are the $n_{\text{test}}$ points that minimise:

$$\text{ED}(n_{\text{test}}, N) = \frac{2}{n_{\text{test}} \cdot N} \sum_j \sum_i \|o_j - x_i\|_2 - \frac{1}{n_{\text{test}}^2} \sum_j \sum_{j'} \|o_j - o_{j'}\|_2 - \frac{1}{N^2} \sum_i \sum_{i'} \|x_i - x_{i'}\|_2$$

The first term pulls the support points towards the data. The second term pushes the support points apart. The third term is a constant (the energy of the empirical distribution itself) that normalises the objective. The result is a set of points spread across the data distribution in a way that trades off coverage and spread — not random, not on a regular grid, but adapting to the actual density of your data.

The convergence guarantee is what makes this useful in practice. Random sampling achieves O($n_{\text{test}}^{-1/2}$) error in approximating distributional moments. Support points achieve O($n_{\text{test}}^{-1}$) — the quasi-Monte Carlo rate, substantially faster convergence. In the insurance simulation, this reduces the large-loss standard deviation in the test set from 7.72 (random) to 1.28 (SPlit) — a six-fold variance reduction. Coefficient estimation error falls from 0.0059 to 0.0015, a four-fold improvement.

The SPlit algorithm (Mak & Joseph, JASA 2018; Guo et al. 2026) finds the support points of the full dataset and uses them as the test set. The training set is everything else. The test set is not drawn randomly — it is computed deterministically (given a random initialisation) to represent the full distribution as well as $n_{\text{test}}$ points can.

For insurance this has a concrete meaning: the test set will contain a representative number of large losses, a representative distribution of sum insured, a representative mix of young drivers and elderly drivers, a representative number of policies near the deductible threshold. Not by construction — by optimisation.

---

## Using SupportPointSplit

```python
from insurance_cv.distributional import SupportPointSplit
import numpy as np

# X: feature matrix, shape (N, d)
# y: target vector, shape (N,)
# Append y to X so the split respects the joint distribution
X_with_y = np.column_stack([X, y])

splitter = SupportPointSplit(
    test_size=0.2,
    n_iter=100,
    random_state=42,
    standardize=True,   # recommended for multivariate data
)

train_idx, test_idx = splitter.split(X_with_y)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
```

The `standardize=True` flag normalises each column to zero mean and unit variance before computing distances. This is important when your features have very different scales — sum insured in pounds versus binary indicators versus vehicle age in years. Without standardisation, the distance computation is dominated by whichever feature has the largest numeric range.

`n_iter` controls the number of gradient descent steps in the energy distance optimisation. For production use, 100 iterations is adequate for N up to 50,000; push to 300 for larger datasets to ensure convergence. The computational cost scales as O(N × n_test × d) per iteration — for N=100,000, d=20, n_test=20,000, this is millions of floating-point operations per step, which is fast on modern hardware but not instantaneous. Expect a few seconds per split at that scale.

The sklearn-compatible interface means `SupportPointSplit` integrates naturally with pipelines and cross-validation wrappers:

```python
from sklearn.model_selection import cross_val_score
from insurance_cv.distributional import SupportPointSplit

# get_n_splits() returns 1 — it is a single train/test split, not k-fold
for train_idx, test_idx in splitter.split(X_with_y):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### When stratified splitting is still appropriate

`SupportPointSplit` is not a universal replacement for stratified splitting. For temporal data — accident year, policy inception cohort — you should use the temporal splitters in `insurance_cv.splits` (`WalkForwardSplit`, `AccidentYearSplit`). Distributional splitting is correct for cross-sectional model validation; temporal splitters are correct when you need to respect time ordering and avoid look-ahead.

For class-imbalanced classification (fraud detection, cancellation prediction), `SupportPointSplit` will represent the minority class in the test set in proportion to its true frequency. If you want to over-represent the minority class for evaluation purposes, a stratified split with explicit class weighting is the right tool. If you want the test set to reflect the real deployment distribution — which is usually correct — `SupportPointSplit` does this naturally.

---

## The problem with Pearson and Spearman on insurance loss data

Insurance loss data has a specific structure that breaks both standard correlation measures.

**Pearson** captures linear dependence only. It will tell you that sum insured has a positive correlation with loss severity — true, and it captures the linear part. It will not tell you whether that relationship is linear, log-linear, or follows a threshold (policies below the reinsurance attachment behave differently from policies above it). A U-shaped relationship — claim frequency by driver age — produces a Pearson correlation near zero even though age is one of the strongest predictors in the model. Pearson will tell you to drop it.

**Spearman** is better. It captures monotone relationships without requiring linearity, so it handles log-curves correctly. It will not catch U-shapes. A 19-year-old and a 75-year-old have similar claim frequencies; a 40-year-old has much lower frequency. The rank correlation between age and frequency is near zero. Spearman will also suggest dropping age.

The correct measure for feature screening is one that is consistent under any dependence structure — one that converges to zero if and only if the feature is truly independent of the target. Chatterjee's Xi is that measure.

---

## Chatterjee's Xi correlation

Xi was introduced by Sourav Chatterjee (JASA, 2021) and is defined as follows. Sort your dataset by the feature $X$ in ascending order. Let $r_i$ be the rank of the corresponding $Y$ value in the sorted order. Then:

$$\xi_N(X, Y) = 1 - \frac{3 \sum_{i=1}^{N-1} |r_{i+1} - r_i|}{N^2 - 1}$$

(This is the version without ties in $X$; the ties case uses 20 random tie-breaking repetitions, averaged, which matters for binary features and capped variables in insurance data.)

Xi takes values in [0, 1]. Xi = 0 if and only if $Y$ is independent of $X$. Xi = 1 if and only if $Y$ is a measurable function of $X$ — a perfect predictor. Under the null hypothesis of independence, $\sqrt{N} \cdot \xi_N \to \mathcal{N}(0, 2/5)$, so hypothesis testing is straightforward.

The computation is O(N log N): sort once, compute ranks, sum adjacent differences. For a 500,000-row dataset with 50 features, the full Xi screening takes seconds.

The properties that matter for insurance screening:

- **Consistent for any dependence structure.** U-shaped relationships, threshold effects, log-linear relationships, zero-inflation — Xi converges to the correct value in all of these. Pearson and Spearman do not.
- **Heavy-tail robustness.** Xi is a rank-based statistic. It is not influenced by extreme values in the way that sample covariance is. A small number of very large losses does not dominate the Xi estimate the way it would dominate Pearson correlation.
- **Zero-inflation handling.** Insurance severity distributions frequently have large point masses at zero. Xi handles ties in $Y$ (zeros) correctly — the formula above handles ties in the ranks, and a large mass at zero just produces tied ranks in $Y$, which is fine.

### Pearson, Spearman, and Xi on a concrete example

Consider driver age against annual claim frequency on a UK motor book. The true relationship is U-shaped: young drivers (under 25) and elderly drivers (over 70) have materially higher frequency than middle-aged drivers.

| Measure | Value | Interpretation |
|---|---|---|
| Pearson r | -0.03 | Near-zero: suggests independence |
| Spearman rho | -0.04 | Near-zero: suggests independence |
| Chatterjee Xi | 0.21 | Strong dependence — both tails are informative |

Pearson and Spearman would lead you to drop age from the feature set. Xi correctly identifies it as a strong predictor. The sign of the Xi measure is not informative — it only tells you whether dependence exists, not its direction. For screening purposes, this is the right question.

Similarly for sum insured and claim severity on a property book:

| Measure | Value | Interpretation |
|---|---|---|
| Pearson r | 0.18 | Weak linear correlation (dominated by zeros) |
| Spearman rho | 0.31 | Moderate monotone correlation |
| Chatterjee Xi | 0.47 | Strong predictive relationship |

The zero-inflated severity distribution (many policies with zero losses) suppresses Pearson and Spearman. Xi, being rank-based and treating zeros as tied observations, correctly identifies that sum insured contains substantial predictive information about severity.

---

## Using ChatterjeeSelector

```python
from insurance_cv.feature_selection import ChatterjeeSelector
import pandas as pd

# Select top-10 features by Xi against the target
selector = ChatterjeeSelector(
    k=10,             # keep top-k features by Xi score
    ties='random',    # tie-breaking method for X: 'random' (default) or 'average'
    n_reps=20,        # repetitions for random tie-breaking (matches paper)
)

selector.fit(X_df, y)

# Inspect Xi scores for all features (descending)
print(selector.scores_)
# {'vehicle_age': 0.51, 'driver_age': 0.47, 'sum_insured': 0.44, ...}

# Transform: returns DataFrame with only the selected features
X_selected = selector.transform(X_df)

# Or combined:
X_selected = selector.fit_transform(X_df, y)
```

`selector.scores_` is an ordered dictionary of feature name to Xi value, sorted descending. You can use this directly to make selection decisions rather than relying on the `k` threshold.

If you prefer a threshold-based selection:

```python
selector = ChatterjeeSelector(
    threshold=0.1,    # keep features with Xi > 0.1
    ties='random',
    n_reps=20,
)
selector.fit_transform(X_df, y)
```

Neither `k` nor `threshold` is universally correct. The Xi values depend on the data — their absolute level is less meaningful than their relative ordering. We recommend plotting the sorted Xi scores and looking for an elbow: a point where scores drop sharply. Features above the elbow have meaningful predictive relationships; features below it are weaker candidates.

`ChatterjeeSelector` implements sklearn's `BaseEstimator` and `TransformerMixin`, so it works inside `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import TweedieRegressor

pipe = Pipeline([
    ('screen', ChatterjeeSelector(k=15)),
    ('scale', StandardScaler()),
    ('model', TweedieRegressor(power=1.5, link='log')),
])

pipe.fit(X_train, y_train)
```

### The multivariate extension

For datasets with more than one target or when you want to screen features against a composite outcome, `ChatterjeeSelector` supports the ACCC (Azadkia-Chatterjee Conditional Correlation Coefficient), the multivariate extension from Azadkia & Chatterjee (2021):

$$\text{ACCC}_N(X, Y) = \frac{\sum_i \min(r_i, r_{\text{NN}(i)}) - \frac{(N+1)(2N+1)}{6}}{\frac{N^2 - 1}{6}}$$

where $r_{\text{NN}(i)}$ is the rank of the target at the nearest neighbour of observation $i$ in feature space. This is useful when you have a vector-valued target (frequency and severity jointly) and want a single screening statistic against both.

---

## Limitations to be honest about

**SupportPointSplit and dimensionality.** Energy distance optimisation in high-dimensional feature spaces is poorly behaved. The practical limit for `SupportPointSplit` is d ≤ 20 features in the split input. For richer feature sets, pre-select the features most important for distributional balance — typically the target, sum insured, and your key risk factors — and pass only those to the splitter. The O(N²) distance matrix computation also makes very large datasets (N > 200,000) slow unless you subsample for the optimisation and snap to the full dataset afterwards. The library documents these constraints.

**Xi is not symmetric.** $\xi(X, Y) \neq \xi(Y, X)$ in general. Xi measures whether $Y$ is predictable from $X$, not whether $X$ is predictable from $Y$. For feature screening, the correct direction is always Xi(feature, target) — how well does the target depend on the feature. If you compute it the wrong way, you get a different (and less useful) quantity. The `ChatterjeeSelector` API enforces the correct direction: `fit(X, y)` always computes Xi(X_j, y) for each feature column.

**Xi detects dependence but does not quantify it in familiar units.** A feature with Xi = 0.4 is not twice as important as a feature with Xi = 0.2, in the sense that doubling the coefficient would give you twice the predictive power. Xi is a screening tool — it tells you whether a relationship exists and how strong it is relative to other features. It does not replace model-based importance measures (SHAP, permutation importance) for understanding the shape and magnitude of effects.

**Ties in binary features.** If your feature is binary — a flag for whether the policyholder has a telematics device, say — then X has only two values, and the tie-breaking in Xi is doing a lot of work. With `n_reps=20` (the paper's recommendation), the Xi estimate averages over 20 random orderings of the tied observations, which is correct but means the Xi value for a binary feature has higher variance than for a continuous feature. For binary features, the Xi screening result is reliable when N > 2,000 per value; below that, treat it with caution.

---

## Installation

```bash
pip install "insurance-cv>=0.3.0"
```

`SupportPointSplit` has no new dependencies beyond what insurance-cv already requires — the energy distance computation uses `scipy.spatial.distance.cdist`, which comes in via scikit-learn. `ChatterjeeSelector` implements Xi natively in ~15 lines using `scipy.stats.rankdata`, so it also adds no new PyPI dependencies.

```python
from insurance_cv.distributional import SupportPointSplit
from insurance_cv.feature_selection import ChatterjeeSelector
```

Source: [github.com/burning-cost/insurance-cv](https://github.com/burning-cost/insurance-cv). The underlying theory is in Mak & Joseph (JASA 2018) for support points, Chatterjee (JASA 2021) for Xi, and Guo et al. (arXiv:2603.18190, 2026) for the insurance application of both.

---

## Related posts

- [Conformal Prediction Intervals for Insurance Pricing Models](https://burning-cost.github.io/machine-learning/2026/03/31/conformal-prediction-intervals-insurance-pricing-python/) — distribution-free uncertainty quantification for Tweedie models; uses `insurance-conformal`, which pairs naturally with `SupportPointSplit` for the train/calibrate/test split
- [Conditional Coverage and Conformal Prediction in Model Selection](https://burning-cost.github.io/machine-learning/model-validation/2026/03/31/conditional-coverage-conformal-prediction-model-selection-cvi/) — using conformal prediction diagnostics in model validation workflows
