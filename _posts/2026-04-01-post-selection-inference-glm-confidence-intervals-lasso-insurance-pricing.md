---
layout: post
title: "Your GLM Confidence Intervals Are Wrong After Variable Selection"
date: 2026-04-01
categories: [techniques]
tags: [glm, variable-selection, confidence-intervals, lasso, poisson, inference, pricing]
description: "Every UK pricing GLM pipeline that uses Lasso variable selection then reports Wald confidence intervals is producing coverage rates of 70–80% at a nominal 95%. A March 2026 paper provides the fix for Poisson frequency models."
math: true
author: burning-cost
---

Pick up a typical UK personal lines pricing pipeline. It will do something like this: fit a Lasso or elastic net on your Poisson frequency or Gamma severity model to knock out the weak variables, then refit OLS or MLE on the surviving features and report confidence intervals on the selected coefficients. The CIs tell you which factors are statistically significant, which ones are noise, and how precisely you've estimated each effect.

Those confidence intervals are wrong. Not wrong in a "technically suboptimal" way. Wrong in a "covering at 70–80% when you think you're at 95%" way, according to simulation results in a March 2026 paper from Shen, Gregory, and Huang at the University of South Carolina (arXiv:2603.24875).

This post explains why, what the fix looks like, and what you can actually do about it today.

---

## What goes wrong

The issue has a name — post-selection inference — and has been understood in the statistics literature since at least Lee et al. (2016). The problem is simple once you see it: Lasso selects a model $\hat{M}$ by looking at your data. You then treat $\hat{M}$ as if it were chosen in advance, independently of the data, and report inference as if you'd specified the model before seeing anything.

You haven't. The selection event $\{\hat{M} = M\}$ — the fact that exactly these variables survived and these didn't — is a statement about your data. Every Wald interval you compute afterwards conditions implicitly on having passed through that filter. Standard asymptotic theory for MLEs doesn't account for the filter, so the intervals are anti-conservative: they're too narrow, p-values are too small, and you reject null effects more often than you should.

The severity of the problem depends on the signal-to-noise regime. Shen et al. simulate this directly for Poisson regression with Lasso selection across different sample sizes and effect sizes. In the moderate-signal regime — moderate $n$, moderate true coefficients — they observe actual coverage rates of 70–80% at nominal 95%. That is the regime that most insurance frequency models live in. We are not dealing with tiny effects that Lasso correctly drives to zero, nor with overwhelming signals that Lasso correctly retains regardless of tuning. We are in the middle, where the selection is doing real work and the inference machinery behind it is quietly broken.

To make this concrete, here is a minimal simulation showing the problem:

```python
import numpy as np
from sklearn.linear_model import LassoCV
from statsmodels.api import GLM, families
from scipy import stats

rng = np.random.default_rng(42)

n, p = 2000, 20
# True model: 5 non-zero coefficients, 15 pure noise
beta_true = np.array([0.3, -0.25, 0.2, -0.15, 0.1] + [0.0] * 15)

X = rng.normal(size=(n, p))
lp = X @ beta_true  # log-linear predictor, no intercept for simplicity
y = rng.poisson(np.exp(lp))

coverage_naive = []
coverage_oracle = []

for _ in range(500):
    X_sim = rng.normal(size=(n, p))
    lp_sim = X_sim @ beta_true
    y_sim = rng.poisson(np.exp(lp_sim))

    # Step 1: Lasso selection
    lasso = LassoCV(cv=5, max_iter=5000).fit(X_sim, y_sim)
    selected = np.where(lasso.coef_ != 0)[0]

    if len(selected) == 0:
        continue

    # Step 2: Refit MLE on selected variables, report naive Wald CIs
    X_sel = X_sim[:, selected]
    glm = GLM(y_sim, X_sel, family=families.Poisson()).fit()

    # Check coverage for first true non-zero coefficient if selected
    if 0 in selected:
        idx = list(selected).index(0)
        ci = glm.conf_int().iloc[idx]
        coverage_naive.append(ci[0] <= beta_true[0] <= ci[1])

print(f"Naive CI coverage (nominal 95%): {np.mean(coverage_naive):.1%}")
# Typical output: 72-79% coverage
```

Run this and you will see something in the 72–79% range. The confidence interval that says "95%" is, in practice, a 75% confidence interval. If you're using these intervals to decide which rating factors to retain, which interactions to include, or whether a competitor-matching adjustment is statistically supported, you are making decisions on misleading evidence.

---

## Why the Wald interval fails here

The Wald interval for coefficient $\hat{\beta}_j$ on selected model $M$ is:

$$\hat{\beta}_j \pm z_{\alpha/2} \cdot \widehat{\text{se}}(\hat{\beta}_j)$$

This treats $\hat{\beta}_j$ as if it were the MLE from a pre-specified model. But $\hat{\beta}_j$ here is the MLE conditional on $\hat{M} = M$ — it has been computed only because $j$ survived selection. For variables near the noise boundary, the event "variable $j$ was selected" is correlated with "variable $j$'s estimated coefficient happened to be large enough." This inflates the point estimate relative to what you'd expect from the unconditional distribution, and the naive standard error doesn't correct for it.

The formal statement is that $\hat{\beta}_j \mid \{\hat{M} = M\}$ does not have the distribution that Wald interval theory assumes. The true conditional distribution has a truncation structure imposed by the selection event.

---

## The fix: parametric programming for post-selection inference

The Shen, Gregory, Huang (2026) paper, building on Le Duy and Takeuchi (2021, JMLR), provides a tractable correction for GLMs. The method has three components.

**Step 1: Linearise the GLM.** At the Lasso solution, construct pseudo-data via the Fisher scoring expansion. The pseudo-response is:

$$z_i = \sqrt{b''(\hat{\eta}_i)} \cdot \hat{\eta}_i + \frac{Y_i - b'(\hat{\eta}_i)}{\sqrt{b''(\hat{\eta}_i)}}$$

and the pseudo-covariates are $\tilde{x}_i = \sqrt{b''(\hat{\eta}_i)} \cdot x_i$, where $b(\cdot)$ is the cumulant function for the GLM family. For Poisson, $b''(\eta) = e^\eta$, so this is the standard IRLS working response and weights. This reduction converts the GLM Lasso to a weighted least-squares Lasso, on which the parametric programming machinery applies.

**Step 2: Trace the selection event.** Parameterise the pseudo-response along a scalar path $\tau$ and trace how the Lasso selection set $\hat{M}(\lambda, \tau)$ changes as $\tau$ varies. The key result from Le Duy and Takeuchi is that this tracing identifies the selection event $\{\hat{M} = M\}$ as a union of intervals on $\tau$ — a 1D object, regardless of how many variables are in the model. This avoids the combinatorial explosion of conditioning on a high-dimensional event directly.

**Step 3: Truncated normal CIs.** Construct confidence intervals using the truncated normal distribution conditional on the identified intervals. For each selected variable $j$ and test direction $e_j$, the pivot:

$$T_{M,j} = \frac{e_j^T \hat{\beta}_M}{\sigma \| P_M e_j \|}$$

has a truncated standard normal distribution conditional on the selection event. Inverting this pivot gives asymptotically valid CIs — not finite-sample exact, but asymptotically correct under standard regularity conditions (Theorem 1 of Shen et al.).

The practical difference from Lee et al. (2016) — implemented in the `selectiveInference` R package — is that this method does not require conditioning on the sign of the Lasso solution. Sign-conditioning makes Lee's CIs over-conservative for null coefficients, producing unnecessarily wide intervals. The path-tracing approach avoids sign-conditioning, giving narrower but still valid intervals. In insurance terms: fewer false "this factor is significant" readings, but also fewer false "this factor is insignificant" readings for genuine effects.

---

## What this means for a pricing pipeline

### The selection stage contaminates the inference stage

The most important shift in thinking is this: the moment you use Lasso or elastic net to choose your model, you cannot treat the model as fixed for inference purposes. The selection and the inference are entangled. Running `statsmodels.GLM().fit()` on the Lasso-selected variables and reading off confidence intervals is producing numbers that look right but aren't.

This applies to every standard UK motor and home pricing GLM pipeline we're aware of that does Lasso-based variable selection. The variables chosen, the coefficient intervals reported, the "significant at 5%" statements made — all of them are based on the assumption that the model was fixed in advance.

### Frequency models are the priority

The Shen et al. paper covers Logistic, Poisson, and Beta regression. It does not yet cover Gamma regression. This matters for insurance: Poisson frequency models are covered, but Gamma severity models are not. The Fisher scoring linearisation approach should extend to Gamma in principle — the machinery only requires differentiability of the cumulant function — but the theoretical guarantees and code do not currently exist for severity.

If you are running a frequency-severity GLM and using Lasso for variable selection on both components, the frequency CIs can be corrected with the current method. The severity CIs cannot yet; you need either a different approach or the fallback described below.

### Offset terms are not supported

Another practical constraint: the current `InfGLM` code on GitHub does not handle offset terms. Offsets are standard in Poisson frequency models — you typically offset by log(exposure) to model claim rate rather than claim count. Without offset support, the code does not apply directly to a standard insurance frequency GLM. This is a known limitation of the current implementation, not a theoretical barrier, but it means some engineering work before production use.

---

## Practical workflow for UK pricing teams

Given the above constraints, here is what we'd recommend today.

### Option 1: Data splitting (viable now, no new code)

Split your modelling dataset: use half for variable selection (fit Lasso, identify $\hat{M}$), use the other half for inference (fit MLE on $\hat{M}$ using only the held-out data, report Wald CIs). Because the held-out data was not used for selection, the selection event is independent of the inference sample. Wald CIs on the held-out MLE are valid.

The cost is statistical efficiency — you're using half your data for inference rather than all of it. For a book with 500,000 policies, this is not a material problem. For a thin commercial portfolio with 8,000 policies, halving the inference sample is painful. It's also worth noting that the data-splitting CIs are wider than they need to be: you are discarding information. But they are correct, which is the priority.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from statsmodels.api import GLM, families
import numpy as np

# Assume X, y, exposure are your full dataset
X_sel, X_inf, y_sel, y_inf, exp_sel, exp_inf = train_test_split(
    X, y, exposure, test_size=0.5, random_state=42
)

# Variable selection on first half
lasso = LassoCV(cv=5, max_iter=5000).fit(X_sel, y_sel)
selected_vars = np.where(lasso.coef_ != 0)[0]

# Valid inference on second half — Wald CIs are now legitimate
X_inf_sel = X_inf[:, selected_vars]
glm = GLM(y_inf, X_inf_sel,
          family=families.Poisson(),
          offset=np.log(exp_inf)).fit()

print(glm.summary())
# These confidence intervals are valid
```

### Option 2: InfGLM (for non-offset Poisson, frequency only, current state)

The `InfGLM` library (GitHub: [kateshen28/InfGLM](https://github.com/kateshen28/InfGLM), Apache 2.0) implements the Shen et al. method. As noted, it does not currently support offset terms. If you can fold your exposure into the response (working with rates rather than counts, or restricting to unit-exposure records), it applies.

The library has notebooks for Poisson regression showing the full pipeline: fit Lasso, run path-tracing, compute truncated normal CIs. The corrected CIs will be wider than naive Wald CIs for variables that were weakly selected (those near the Lasso boundary) and approximately equal for variables selected with high confidence. That widening is not a flaw — it's the method correctly acknowledging that selection uncertainty exists.

### Option 3: Wait for Gamma support

For severity models, there is no off-the-shelf corrected CI method available today. The Le Duy-Takeuchi parametric programming framework should extend to Gamma — the cumulant function $b(\eta) = -\log(-\eta)$ is twice differentiable — but the theoretical development and code for this have not appeared. We expect it to follow from the Shen et al. work relatively quickly given how cleanly the Fisher scoring reduction works across families.

In the interim, data splitting is the correct approach for severity CIs post-Lasso.

---

## How to read your existing pricing model outputs

If your team has already built a GLM with Lasso selection and Wald CIs, here is a rough guide to how worried to be:

**Variables you retained with very large coefficients and t-statistics well above 3–4:** These were selected regardless of the Lasso path. The selection event carries little information about the MLE. Coverage loss is small; inference is roughly fine.

**Variables retained with moderate t-statistics (1.5–3):** These are exactly where the problem bites hardest. The coefficient is large enough to survive Lasso, but the selection event is correlated with the point estimate being above average. Your 95% CI is probably a 75–80% CI. Be sceptical about significance claims here.

**Variables that narrowly survived (Lasso coefficient very close to zero at the chosen $\lambda$):** These are the most contaminated. The selection event is essentially a statement that the data happened to show a large enough effect to cross the threshold. Do not interpret narrow Wald CIs on these variables as evidence of precision.

The implication for pricing: where you have used "statistically significant at 5%" as a criterion for retaining a rating factor, you should be more conservative. Some of those factors passed the threshold partly because the selection filter and the inference stage were not independent.

---

## Connecting to the GLM library

Our [`insurance-gam`](/insurance-gam/) library, which handles penalised GLM and GAM fits for pricing, does not currently implement post-selection corrected inference — that's fair given the method is new and the code is still maturing. The interaction between GAM spline selection and coefficient inference is a harder version of the same problem: if you use any data-driven method to choose which spline terms to include or how many knots to place, the resulting coefficient intervals have the same contamination issue.

We will be watching the InfGLM development and will assess integration once offset support is available. For teams currently using `insurance-gam` with penalised variable selection, data splitting is the pragmatic near-term fix.

---

## The bottom line

Post-selection inference is not a theoretical curiosity. It's a systematic bias in how UK pricing actuaries read their GLM outputs. Any pipeline that uses Lasso or elastic net for variable selection and then reports standard confidence intervals is operating with miscalibrated uncertainty estimates — typically 15–20 percentage points below the nominal coverage rate in the moderate-signal regime where most insurance models sit.

The Shen, Gregory, Huang paper provides a working solution for Poisson frequency models, available in Python under Apache 2.0. Gamma severity is not yet covered. Until it is, data splitting is the correct approach: use half your data to select, the other half to infer. The intervals are wider; they are also true.

The fix for frequency models exists. The only question is whether teams building GLM pricing pipelines will use it.

---

## The paper

Shen, K., Gregory, K.B., and Huang, S. (2026). *Post-selection Inference for Generalized Linear Models.* arXiv:2603.24875. Submitted 25 March 2026.

Code: [github.com/kateshen28/InfGLM](https://github.com/kateshen28/InfGLM) — Apache 2.0, Python.

Prior work: Lee, J.D., Sun, D.L., Sun, Y., Taylor, J.E. (2016). 'Exact post-selection inference, with application to the lasso.' *Annals of Statistics*, 44(3), 907–927. Le Duy, V.N. and Takeuchi, I. (2021). 'Parametric programming approach for more powerful and general lasso selective inference.' *JMLR*, 22(1), 1–37.

---

## Related posts

- [GAM Pricing with insurance-gam](/insurance-gam/) — the GLM/GAM library where post-selection inference will eventually integrate
- [Sparse Deep Two-Part Models for Frequency-Severity](/2026/03/31/sparse-deep-two-part-frequency-severity/) — joint variable selection across frequency and severity, same inference problem applies
