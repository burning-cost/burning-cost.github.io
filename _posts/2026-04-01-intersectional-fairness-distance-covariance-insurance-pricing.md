---
layout: post
title: "Intersectional Fairness in Insurance Pricing: Why Auditing Age and Gender Separately Is Not Enough"
date: 2026-04-01
categories: [fairness, machine-learning]
tags: [intersectional-fairness, distance-covariance, CCdCov, insurance-fairness, FCA-Consumer-Duty, demographic-parity, fairness-gerrymandering, regularisation, protected-characteristics, motor-insurance, gender-ban, Lee-Antonio-Avanzi-2025, arXiv-2509-08163, dcor]
description: "A model can pass its age fairness audit and its gender fairness audit and still systematically overprice young women. This is fairness gerrymandering. We explain the CCdCov measure from Lee et al. (2025), how insurance-fairness v1.0.0 implements it, and why the FCA's 2026 AI review makes this now."
author: burning-cost
---

Here is a pricing model scenario. Gender fairness audit: passed. Age fairness audit: passed. You close the file and sign off. Six months later an actuary notices that young women in your portfolio are paying materially more than both young men and elderly women with identical risk profiles. Every marginal test was clean. The intersectional subgroup was invisible.

This is not a contrived edge case. It is a known failure mode with a name — fairness gerrymandering — and it is the specific pattern the FCA's 2026 multi-firm AI review flags under "demographic combination bias." The paper that gives us the right tool to detect and correct it is Lee, Antonio, Avanzi, Marchi & Zhou (arXiv:2509.08163, September 2025). We have implemented their method in `insurance-fairness` v1.0.0 as `IntersectionalFairnessAudit` and `DistanceCovFairnessRegulariser`.

---

## Why marginal audits fail

Suppose your model produces predictions $\hat{y}$ and you have two protected attributes: gender $s_1$ and age $s_2$. The standard approach runs two audits:

$$\tilde{d}\text{Cov}^2(\hat{y},\, s_1) \approx 0 \quad \text{and} \quad \tilde{d}\text{Cov}^2(\hat{y},\, s_2) \approx 0$$

Both are near-zero: the model's predictions are approximately independent of gender and approximately independent of age. You declare it fair.

The problem is that statistical independence of $\hat{y}$ and $s_1$, plus independence of $\hat{y}$ and $s_2$, does not imply independence of $\hat{y}$ and the joint vector $(s_1, s_2)$. A model can have zero marginal correlation with each attribute while exploiting intersectional structure: pricing the *combination* of young and female differently, even though neither young nor female alone carries a signal in the marginals.

This is the insurance analogue of Simpson's paradox. A classic example: a hospital appears unbiased when you split outcomes by gender and again by age. But young female patients have worse outcomes than any other group. The aggregation conceals what the intersection reveals.

For UK motor insurance this is not academic. The post-2012 gender ban (following the Test-Achats ruling, applied to UK by GIC) removed gender as a direct rating factor. Insurers have had fourteen years to adjust. What the data shows is that intersectional combinations — young female, specifically — remain an area where pricing models can absorb the signal through correlated features like vehicle type, telematics behaviour, and NCD profile. Each of those features has a clean individual audit. Their intersection does not.

---

## What CCdCov measures

The Lee et al. paper introduces Concatenated Distance Covariance (CCdCov) to close this gap. It builds on distance covariance — a measure of statistical dependence developed by Székely, Rizzo & Bakirov (Annals of Statistics, 2007) — but applies it to the joint protected attribute vector rather than each attribute independently.

Distance covariance has a key property that correlation does not: $\text{dCov}(\hat{y}, s) = 0$ if and only if $\hat{y}$ and $s$ are statistically independent. Pearson correlation can be zero for non-linearly dependent variables. Distance covariance catches everything. This matters for insurance where premium structures are nonlinear and indirect discrimination can live in interaction terms.

CCdCov treats the concatenated attribute matrix $S = (s_1, \ldots, s_d)$ as a single multivariate entity:

$$\text{CCdCov}^2(\hat{y},\, S) = \sum_{k=1}^{d} \tilde{d}\text{Cov}^2(\hat{y},\, s_k) \;+\; \eta(\hat{y},\, S)$$

The first term is the sum of individual marginal penalties — what the naive approach already computes. The second term, $\eta$, is the intersectional residual: the additional dependence between predictions and the *joint* attribute distribution that the marginals miss.

$\text{CCdCov}^2 = 0$ if and only if $\hat{y}$ is jointly independent of all protected attributes. This is the correct target for intersectional demographic parity.

One practical note: $\eta$ can be negative in finite samples, specifically when the unbiased CCdCov estimator is smaller than the sum of marginal estimators. This is a finite-sample artefact of the bias-corrected estimator, more pronounced when attributes are strongly collinear or $n$ is small. The library reports $\eta$ with its sign, and the Jensen-Shannon calibration protocol (described below) is robust to this.

### CCdCov versus JdCov

The paper also defines Joint Distance Covariance (JdCov), which adds pairwise inter-attribute terms $\tilde{d}\text{Cov}^2(s_k, s_l)$ to the marginal sum. The instability of JdCov arises when the protected attributes are themselves correlated — age and gender are not independent across insurance portfolios (younger cohorts have different gender compositions, NCD profiles differ). JdCov penalises the model for associations between the attributes themselves, not just between the model's predictions and those attributes. CCdCov avoids this by treating $S$ as a unified vector: it penalises $\hat{y}$'s dependence on $S$, full stop, without contamination from inter-attribute structure.

In our implementation, CCdCov is the default and the recommended method for production use. JdCov is available for comparison.

---

## Auditing an existing model

The `IntersectionalFairnessAudit` class computes CCdCov, marginal dCov per attribute, $\eta$, JdCov, and Jensen-Shannon divergence across all intersectional subgroups. It takes model predictions and a DataFrame of protected attribute values — it does not need the model itself.

```python
from insurance_fairness.intersectional import IntersectionalFairnessAudit
import pandas as pd
import numpy as np

# y_hat: your model's predicted claim frequencies, shape (n,)
# df_protected: DataFrame with the protected attribute columns
# e.g., df_protected = df[["gender", "age_band"]]

audit = IntersectionalFairnessAudit(
    protected_attrs=["gender", "age_band"],
    continuous_attrs=["age_band"],   # normalise to [0,1]; leave out for categoricals
)
report = audit.audit(y_hat, df_protected)
print(report.summary())
```

The output breaks down each component:

```
IntersectionalFairnessAudit
========================================
  n = 45,231  |  6 intersectional subgroups
  Protected attributes: gender, age_band

Distance covariance metrics (lower = fairer, 0 = independent):
  CCdCov²(ŷ, S)       = 0.003847
  Σ marginal dCov²    = 0.001203
  η (intersect. resid)= 0.002644
  JdCov²(ŷ, S)        = 0.004112

Marginal d̃Cov² per attribute:
    gender:   0.000841
    age_band: 0.000362

Jensen-Shannon divergence (lower = fairer):
  D_JS overall = 0.0412
  D_JS by attribute pair:
    (gender, age_band): 0.0317
```

In this example, the marginal dCov values for gender (0.000841) and age\_band (0.000362) are both small — the marginal audits would pass. But $\eta = 0.002644$ is larger than either marginal term individually. The model's intersectional dependence exceeds its marginal dependence. The audit would have been signed off incorrectly.

The `IntersectionalAuditReport` also contains a `subgroup_statistics` DataFrame with mean predictions by subgroup. This is where you see the young female / elderly female divergence directly.

### Encoding

Protected attributes are encoded automatically before the distance covariance computation. Categorical attributes (gender, vehicle type) receive ordinal integer coding — sorted unique values mapped to 0, 1, 2, ... . The library uses ordinal integers rather than one-hot encoding because Euclidean distance on integers is the standard approach in the distance covariance literature. One-hot encoding inflates the dimensionality and distorts the distance structure. Continuous or ordinal attributes listed in `continuous_attrs` are min-max normalised to $[0,1]$ to prevent scale dominance.

---

## Training-time regularisation

Auditing tells you whether the problem exists. The `DistanceCovFairnessRegulariser` addresses it during model training by adding a $\lambda \cdot \text{CCdCov}^2$ penalty to the loss function.

The training objective becomes:

$$\min_\Theta \;\left\{ \frac{1}{n} \sum_{i=1}^n L(\hat{y}_{\Theta,i},\, y_i) \;+\; \lambda \cdot \text{CCdCov}^2(\hat{y}_\Theta,\, S) \right\}$$

where $L$ is your predictive loss (Poisson deviance for claim counts, MSE for severity) and $\lambda$ controls the fairness-accuracy trade-off. Higher $\lambda$ = more fairness pressure, more accuracy cost.

```python
from insurance_fairness.intersectional import DistanceCovFairnessRegulariser

reg = DistanceCovFairnessRegulariser(
    protected_attrs=["gender", "age_band"],
    continuous_attrs=["age_band"],
    method="ccDcov",   # "ccDcov" (default), "jdCov", or "sum_dcov"
    lambda_val=0.5,
)

# Inside your training loop:
y_hat = model.predict(X_batch)
fairness_penalty = reg.penalty(y_hat, df_protected_batch)
total_loss = poisson_deviance(y_hat, y_batch) + fairness_penalty
```

The `penalty()` call computes $\lambda \cdot \text{CCdCov}^2(\hat{y}, S)$ on the current predictions and returns a scalar you add to your loss. For neural network training with autograd, you will want to compute the gradient of the penalty with respect to $\hat{y}$ using `penalty_grad()`, which returns $\partial(\lambda \cdot \text{CCdCov}^2) / \partial \hat{y}_i$ for each observation.

---

## Calibrating lambda

Lambda calibration is not optional. Too small and the penalty is cosmetic; too large and you destroy the risk signal. The paper recommends the Jensen-Shannon divergence approach rather than hypothesis tests — correctly, because hypothesis tests on large portfolios become oversensitive and will flag as statistically significant any fairness gap that is practically negligible.

The `calibrate_lambda()` method runs the JS-divergence protocol automatically:

```python
result = reg.calibrate_lambda(
    model_fn=model.predict,
    X_val=X_validation,
    D_val=df_protected_validation,
    y_val=y_validation,
    lambda_grid=[0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    loss="poisson",
)
print(result.summary())
result.plot()  # Pareto front: validation loss vs D_JS across lambda values
```

The protocol:

1. Fix model hyperparameters at $\lambda = 0$ (tune accuracy first, separate from fairness)
2. Train over a grid of $\lambda$ values on a 70/30 subtrain/validation split of the training set
3. For each $\lambda$, record validation Poisson deviance and $D_{JS}$ across intersectional subgroups
4. Plot the Pareto frontier of (accuracy, fairness)
5. Select $\lambda$ based on your explicit fairness requirement — regulatory threshold, business judgment, or a specific $D_{JS}$ target

Step 5 is intentionally exogenous. The library does not select $\lambda$ for you. This is the right design for regulatory defensibility: the fairness-accuracy trade-off is a governance decision, not a technical optimisation. A model validation report should show the Pareto plot and the explicit basis for the $\lambda$ selection.

The `LambdaCalibrationResult` contains `lambda_values`, `js_divergence`, `validation_loss`, `pareto_front`, and `selected_lambda`. The Pareto plot is the primary output — you want to see the $D_{JS}$ falling as $\lambda$ increases, and identify where the accuracy loss becomes unacceptable.

---

## Scalability: O(n²) and what to do about it

Distance covariance on a concatenated multivariate matrix $S$ of shape $(n, d)$ is $O(n^2)$. The `dcor` library uses an $O(n \log n)$ AVL-based method for one-dimensional inputs, but multivariate distance covariance falls back to the $O(n^2)$ naive computation. A 100,000-row dataset is feasible on modern hardware (roughly 30 seconds). A 500,000-row dataset is not — the distance matrix alone requires 2TB of intermediate storage.

The library issues a warning above 50,000 observations and suggests subsampling:

```python
# For large portfolios: subsample before auditing
rng = np.random.default_rng(42)
idx = rng.choice(len(df), size=50_000, replace=False)

report = audit.audit(y_hat[idx], df_protected.iloc[idx])
```

For training-time regularisation on large datasets, the natural approach is to compute the CCdCov penalty on the current mini-batch rather than the full training set. Mini-batch CCdCov is a noisy estimator of the full-data penalty, but the gradient direction is correct on average, which is sufficient for stochastic optimisation. Use batch sizes of at least 2,000 to keep the estimator variance manageable.

The $O(n^2)$ constraint does not affect the `js_divergence_overall` metric — that runs on the full dataset and is $O(n \cdot B)$ where $B$ is the number of histogram bins. You can compute $D_{JS}$ on a full 500,000-policy book without subsampling.

---

## The regulatory hook

The FCA's 2026 AI multi-firm review explicitly flags "algorithmic bias against specific demographic combinations" as a concern distinct from single-attribute bias. This is the first time FCA guidance has used language that maps directly to the intersectional fairness problem rather than treating age, gender, and ethnicity as separate issues to be audited independently.

Consumer Duty outcome monitoring under PS22/9 requires firms to demonstrate that their pricing outcomes are not disproportionately adverse for groups sharing protected characteristics. If your monitoring is structured as separate tests per characteristic, you are not demonstrating intersectional fairness. A firm that can show CCdCov-based audit results, a Pareto plot of $\lambda$ calibration, and a documented governance decision on the fairness-accuracy trade-off is in materially better shape than one that can only show per-attribute distributional tests.

The Equality Act 2010 angle is subtler. Section 19 (indirect discrimination) requires showing that a provision, criterion, or practice puts a group with a protected characteristic at a particular disadvantage. The key phrase is "a group" — which the FCA interprets as potentially intersectional. A 17-year-old woman is not the same protected group as a 17-year-old man or a 65-year-old woman. The Act does not prevent them from having different premiums; it requires that the difference is justified by legitimate risk factors, not by the protected characteristic. CCdCov tests for the latter.

One important distinction: CCdCov targets demographic parity ($\hat{y} \perp S$), meaning the distribution of predictions is the same across intersectional subgroups. The Equality Act standard is closer to conditional fairness — equal prices for equal risks, regardless of protected attribute. Demographic parity and conditional fairness are different objectives. For UK regulatory compliance, [Lindholm marginalisation](https://burning-cost.github.io/2026/03/31/discrimination-free-pricing-privatised-attributes-insurance/) targets conditional fairness and should be your primary correction tool. CCdCov then serves as a complementary training-time regulariser and intersectional audit mechanism.

---

## What sits in the existing library

`insurance-fairness` had seven fairness components before v1.0.0. None of them targeted the joint distribution of multiple protected attributes at training time:

- `LindholmCorrector`: post-hoc marginalisation over individual attributes — conditional fairness, not demographic parity
- `DiscriminationInsensitiveReweighter`: KL-optimal reweighting for a single protected attribute
- `MulticalibrationAudit`: can audit intersectional cells if you pass composite group labels, but requires discretisation of continuous attributes
- `WassersteinCorrector` / `SequentialOTCorrector`: distributional correction for marginal or sequential parity
- `IndirectDiscriminationAudit`: proxy vulnerability and the five benchmark premiums (Côté et al.)

`IntersectionalFairnessAudit` and `DistanceCovFairnessRegulariser` fill the gap that was genuinely absent. The intended workflow is CCdCov regularisation during training, followed by Lindholm marginalisation as post-hoc correction, followed by `IntersectionalFairnessAudit` as the sign-off diagnostic.

---

## Installation

```bash
uv add "insurance-fairness[intersectional]"
```

The `[intersectional]` extra pulls in `dcor>=0.6`. Without it, importing from `insurance_fairness.intersectional` will raise an `ImportError` with installation instructions. The base `insurance-fairness` package does not require `dcor` for the other modules.

```python
from insurance_fairness.intersectional import (
    IntersectionalFairnessAudit,
    DistanceCovFairnessRegulariser,
    IntersectionalAuditReport,
    LambdaCalibrationResult,
)
```

Source: [github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness). The underlying paper is arXiv:2509.08163; the `dcor` library is by Ramos-Carreño & García-Fernández and documented at [dcor.readthedocs.io](https://dcor.readthedocs.io/).

---

## Related posts

- [Discrimination-Free Insurance Pricing: The Lindholm Approach](https://burning-cost.github.io/2026/03/31/discrimination-free-pricing-privatised-attributes-insurance/) — conditional fairness via marginalisation; the correct primary correction for UK regulatory compliance
- [Sequential Optimal Transport for Multi-Attribute Fairness](https://burning-cost.github.io/2026/03/31/sequential-ot-fairness-multi-attribute-insurance-pricing/) — distributional corrections for joint demographic parity across multiple protected attributes
