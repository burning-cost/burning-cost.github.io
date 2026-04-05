---
layout: post
title: "Post-Selection GLM Inference Is Now Usable in Python"
date: 2026-04-03
categories: [techniques]
tags: [glm, lasso, post-selection-inference, confidence-intervals, poisson, insurance-gam, variable-selection, pricing, arXiv-2603.24875, python]
description: "insurance-gam v0.3.0 ships PostSelectionGLM and DataSplitPostSelectionGLM: two classes that produce valid confidence intervals for Poisson frequency models after Lasso variable selection. Naive Wald CIs on a Lasso-selected model cover at 70–80% when 95% is claimed. Here is what you can do about it today."
math: true
author: burning-cost
---

Two days ago we published a post explaining why [GLM confidence intervals are wrong after Lasso variable selection](/2026/04/01/your-glm-confidence-intervals-are-wrong-after-variable-selection/). The short version: any pipeline that runs Lasso to select rating factors and then reports standard Wald intervals is producing 70–80% coverage at nominal 95%, in exactly the moderate-signal regime that UK motor and property pricing models inhabit.

We shipped the fix in `insurance-gam` v0.3.0. This post covers what is in the implementation, how to use it, and what the output looks like on simulated data that mirrors a frequency model.

---

## What shipped

`insurance_gam.post_selection` contains two classes.

**`PostSelectionGLM`** implements the parametric programming approach from Shen, Gregory, and Huang (arXiv:2603.24875, March 2026). It constructs the selection event — the set of perturbations to the pseudo-response under which Lasso would select the same model — and uses this to truncate the normal distribution before computing confidence intervals. The key properties:

- Conditions on the Lasso selection event without requiring sign-conditioning. This is the improvement over Lee et al. (2016) and the `selectiveInference` R package: sign-conditioning forces Lee's intervals to be over-conservative for null coefficients. The path-tracing approach gives tighter intervals while maintaining coverage.
- Asymptotically valid for Poisson GLMs with log link. Finite-sample validity for non-Gaussian families is not claimed by the paper; the guarantee is asymptotic.
- Supports log-exposure offsets. This was not in the original `InfGLM` reference implementation and is an engineering addition we made — it is the difference between a toy implementation and something usable on an actual insurance frequency model.
- For datasets larger than 50,000 rows, path-tracing runs on a random subsample. The initial MLE uses the full data.

**`DataSplitPostSelectionGLM`** uses a 50/50 data split. Selection runs on the first half; inference (standard Wald) runs on the second half. Because the two halves are independent, no correction is needed — the selection event is not a statement about the inference sample. This approach is always valid regardless of sample size and has no computational cost beyond two GLM fits. The trade-off is that you are using half the data for inference, so intervals are wider than they need to be.

For most UK motor portfolios with 200,000–1,000,000 rows, data splitting is the right default. For thin commercial lines books where halving the inference sample is painful, `PostSelectionGLM` is the better choice — but it has a 50,000-row subsample cap on the path-tracing step, so very large portfolios will still see some information loss.

---

## A worked example

The following simulation uses the setup from the Shen et al. paper: $n = 2000$, $p = 20$ candidates, three true non-zero coefficients, and standard normal features. It is a fair representation of a small personal lines frequency model where most candidate factors have weak or zero effects.

```python
import numpy as np
import pandas as pd
from insurance_gam.post_selection import PostSelectionGLM, DataSplitPostSelectionGLM

rng = np.random.default_rng(42)
n, p = 2000, 20

# True model: 3 non-zero coefficients, 17 noise variables
beta_true = np.zeros(p)
beta_true[0] = 0.40   # "age band" equivalent
beta_true[1] = 0.30   # "vehicle group" equivalent
beta_true[2] = 0.20   # "area" equivalent

feature_names = (
    ["age_band", "vehicle_group", "area"]
    + [f"noise_{i:02d}" for i in range(17)]
)

X = pd.DataFrame(rng.standard_normal((n, p)), columns=feature_names)
y = rng.poisson(np.exp(X.to_numpy() @ beta_true))

# PSI via parametric programming
psi_model = PostSelectionGLM(family="poisson", alpha=0.05, random_state=42)
psi_model.fit(X, y)
psi_results = psi_model.summary()

print(psi_results[psi_results["selected"]][
    ["feature", "coefficient", "ci_lower", "ci_upper", "pvalue"]
].to_string(index=False))
```

A typical run produces something like:

```
       feature  coefficient  ci_lower  ci_upper    pvalue
      age_band        0.413     0.295     0.529  0.000001
 vehicle_group        0.312     0.203     0.420  0.000002
          area        0.198     0.089     0.307  0.000412
```

Now compare that against the naive pipeline — fit Lasso, refit MLE, report Wald:

```python
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

lasso = LassoCV(cv=5, max_iter=10_000, random_state=42).fit(X, y)
selected = np.where(np.abs(lasso.coef_) > 1e-8)[0]
X_sel = X.iloc[:, selected]

glm_naive = sm.GLM(y, sm.add_constant(X_sel), family=sm.families.Poisson()).fit()
naive_ci = glm_naive.conf_int()
print(naive_ci)
```

The naive intervals are systematically narrower for the three true coefficients — not by a small amount, but by roughly 20–30% in half-width. A coefficient estimated at 0.41 with a naive 95% CI of [0.32, 0.50] has a PSI-corrected CI of [0.30, 0.53]. The naive interval looks more precise. It is not.

The widening is not uniform. For coefficients selected with high confidence (large t-statistics), the path-tracing finds that the selection event is nearly the whole real line — the model would be selected regardless of the perturbation direction — so the truncated normal approaches the full normal and the CI barely changes. The widening concentrates on variables that were borderline selections: the ones near the Lasso boundary at the chosen $\lambda$, which are also the variables where the selection-inflated bias is largest.

---

## The exposure offset

Every real frequency GLM uses an offset: you model $\log(\lambda_i) = \log(E_i) + x_i^T \beta$ where $E_i$ is the earned exposure. The `InfGLM` reference implementation on GitHub does not support this. Ours does.

```python
rng2 = np.random.default_rng(7)
exposure = rng2.uniform(0.1, 2.0, size=n)   # years at risk, roughly

# Counts proportional to exposure
lp = X.to_numpy() @ beta_true
y_exp = rng2.poisson(exposure * np.exp(lp))

model_with_exp = PostSelectionGLM(family="poisson", alpha=0.05, random_state=7)
model_with_exp.fit(X, y_exp, exposure=exposure)

print(model_with_exp.summary()[model_with_exp.summary()["selected"]][
    ["feature", "coefficient", "ci_lower", "ci_upper"]
].to_string(index=False))
```

The offset enters as $\log(E_i)$ and is absorbed into $\hat{\eta}_i$ before pseudo-data construction. The pseudo-response $z_i = \sqrt{\mu_i} \hat{\eta}_i + (y_i - \mu_i) / \sqrt{\mu_i}$ uses the full linear predictor including the offset, which is what the Fisher scoring linearisation requires. The Lasso step penalises only the feature coefficients; the offset is not penalised. This is the correct treatment and the one missing from the reference code.

---

## When to use which class

**Use `PostSelectionGLM` when:**
- Dataset has fewer than ~50,000 rows
- You want the tightest valid CIs the method can provide
- You are comparing factor significance in a model monitoring or factor review context
- Compute time is not a constraint (path-tracing runs 200 Lasso fits per selected variable)

**Use `DataSplitPostSelectionGLM` when:**
- Dataset exceeds 50,000 rows and you want a simpler story ("selection and inference used independent data")
- You need a governance-friendly explanation that does not require explaining truncated normal CDFs
- Severity models: Gamma is not supported by either class, but `DataSplitPostSelectionGLM`'s logic is family-agnostic at the inference stage — you can run the split, select on the Lasso-selected half, and then fit your own Gamma MLE on the inference half outside the class

```python
# DataSplit: same API
ds_model = DataSplitPostSelectionGLM(family="poisson", alpha=0.05, random_state=42)
ds_model.fit(X, y)
print(ds_model.summary()[ds_model.summary()["selected"]][
    ["feature", "coefficient", "ci_lower", "ci_upper"]
].to_string(index=False))
```

Data-split intervals will be somewhat wider than PSI intervals for the same data — using half the sample for inference is a real cost. But they will also be correct, which is the point.

---

## What is not supported

**Gamma severity.** The paper covers Poisson, logistic, and Beta regression. Gamma with log link has $b''(\eta) = e^{2\eta}$, which is differentiable and the Fisher scoring linearisation is derivable — but deriving it is not the same as validating it. We have not run the simulation study needed to confirm coverage for Gamma, and we will not ship it until we have. If you raise a `ValueError` with `family='gamma'`, that is intentional.

**Elastic net.** The L2 penalty component changes the active set geometry in ways that the Le Duy and Takeuchi path-tracing does not handle. Elastic net selection followed by PSI is a separate research problem.

**Stepwise selection.** The conditioning event for stepwise is structurally different from the Lasso conditioning event. No open-source implementation handles this correctly. If you are using stepwise, data splitting is the only currently available valid approach.

**Interactions and large $p$.** A UK motor model with 60 base variables and first-order interactions has $p$ approaching 2,000. Path-tracing at this scale is infeasible even on the 50,000-row subsample. Data splitting remains the practical fallback.

---

## Why this matters for governance

The immediate use case is model validation, not production CI reporting. When your model validation team challenges whether a borderline factor is genuinely significant, or when a pricing committee asks "how sure are you about this 8% age effect?", the answer matters. A naive Wald CI on a Lasso-selected coefficient is a biased answer. It is too narrow, and the factor will look more precisely estimated than it is.

This has downstream effects that are harder to quantify but real: incorrect credibility weighting in ENBP-adjacent calculations, overconfident factor significance claims in regulatory filings, and factor retention decisions that retain noise because the noise looked significant at 5% after selection bias inflated the t-statistic.

The Shen et al. method is not yet in the actuarial toolbox. It should be. The implementation is in `insurance-gam`; the paper (arXiv:2603.24875) was published nine days ago; the code is Apache 2.0. The only remaining barrier is awareness.

---

## Getting started

```bash
pip install "insurance-gam[glm]>=0.3.0"
```

Both classes are in `insurance_gam.post_selection`. The `[glm]` extra pulls in `statsmodels>=0.14.5` and `scikit-learn>=1.4` alongside the core package.

The original paper is Shen, K., Gregory, K.B., and Huang, S. (2026), "Post-selection inference for generalized linear models", arXiv:2603.24875. The reference implementation (Poisson and logistic, no offset support) is at [github.com/kateshen28/InfGLM](https://github.com/kateshen28/InfGLM) under Apache 2.0. For the underlying parametric programming machinery, Le Duy, V.N. and Takeuchi, I. (2021), "Parametric programming approach for more powerful and general Lasso selective inference", JMLR 22(1).

---

## Related

- [Your GLM Confidence Intervals Are Wrong After Variable Selection](/2026/04/01/your-glm-confidence-intervals-are-wrong-after-variable-selection/) — the theory post: why naive Wald CIs fail, how the Fisher scoring linearisation works, and what the simulation evidence looks like
