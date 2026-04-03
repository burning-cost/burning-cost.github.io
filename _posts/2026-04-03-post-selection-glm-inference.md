---
layout: post
title: "Post-Selection Inference Fixes Your Frequency Model. Your Severity Model Is Still Broken."
date: 2026-04-03
categories: [techniques]
tags: [glm, lasso, post-selection-inference, confidence-intervals, poisson, gamma, severity, frequency-severity, insurance-gam, variable-selection, pricing, arXiv-2603.24875, two-part-model]
description: "Shen et al. (arXiv:2603.24875) gives valid CIs for Lasso-selected Poisson GLMs. That fixes your frequency model. UK motor pricing is Poisson × Gamma — and the Gamma severity model has the same post-selection bias with no fix in sight. We explain what this means for the two-part model in practice."
math: true
author: burning-cost
---

We [published last week](/2026/04/01/post-selection-inference-glm-confidence-intervals-lasso-insurance-pricing/) on post-selection inference: the paper by Shen, Gregory, and Huang (arXiv:2603.24875, March 2026) that provides valid confidence intervals for Poisson GLMs after Lasso variable selection. If your pricing pipeline runs Lasso to select rating factors and then reports Wald CIs, those CIs are covering at 70–80% when you think you are at 95%. The paper fixes this for Poisson.

The fix matters. But it is easy to read the paper and conclude that the post-selection inference problem is now solved for UK motor pricing. It is not.

UK personal lines motor pricing is a two-part model: claim frequency (Poisson) multiplied by claim severity (Gamma with log link). Shen et al. cover Poisson. They do not cover Gamma. The frequency side of the problem has a working solution. The severity side has the same bias and no equivalent solution in any published tool.

---

## The structure of the two-part model

The standard UK motor pricing model for pure premium is:

$$\mathbb{E}[\text{pure premium}_i] = \lambda_i \cdot \mu_i$$

where $\lambda_i = \exp(\mathbf{x}_i^T \boldsymbol{\beta}_\lambda)$ is the Poisson frequency component and $\mu_i = \exp(\mathbf{x}_i^T \boldsymbol{\beta}_\mu)$ is the Gamma severity component. Both models are typically estimated on their own feature sets, with Lasso or elastic net selecting rating factors for each independently.

The frequency model gets a Lasso pass over candidate variables — age band, vehicle group, area, NCD, occupation, annual mileage and so on. The severity model gets its own Lasso pass, possibly over the same candidates, possibly over a different set that includes claim type and channel alongside the rating factors.

Both models then have their selected variables refit with standard MLE and report Wald CIs. Both suffer from post-selection bias. Both are understating their uncertainty by roughly the same degree.

Shen et al. fix the Poisson frequency side. The Gamma severity side is left as it was: naive CIs on a Lasso-selected model, with the same 70–80% coverage at nominal 95%.

---

## Why Gamma is harder

The Fisher scoring linearisation that makes Shen et al. work for Poisson is not Poisson-specific — it is a general exponential family technique. For a GLM with cumulant function $b(\eta)$, the pseudo-response at the MLE is:

$$z_i = \sqrt{b''(\hat{\eta}_i)} \cdot \hat{\eta}_i + \frac{y_i - b'(\hat{\eta}_i)}{\sqrt{b''(\hat{\eta}_i)}}$$

For Poisson with log link, $b'(\eta) = b''(\eta) = e^\eta = \mu_i$. Simple. For Gamma with log link, $b'(\eta) = \mu_i = e^\eta$ and $b''(\eta) = \mu_i^2 = e^{2\eta}$. The pseudo-data formula is derivable — it is not mathematically obstructed.

The obstruction is empirical, not algebraic. Shen et al. validate their method by simulation: 1,000 Monte Carlo replicates, $n = 500$, three true non-zero coefficients, comparing achieved coverage against nominal 95%. That simulation evidence is what makes the Poisson result usable. For Gamma, no equivalent simulation study exists. The `InfGLM` reference implementation on GitHub does not include Gamma. The research paper does not include Gamma. Until someone runs the simulation study and it passes, shipping a Gamma post-selection CI would be marketing, not statistics.

There is also a genuine additional complication. In the Poisson case, the dispersion is fixed at 1 by construction — one less thing to worry about. For a Gamma GLM, the dispersion parameter $\phi$ (equivalently, the shape parameter) must be estimated from data. Estimation error in $\phi$ propagates into the CI width via the variance term $a(\phi) \cdot \|c_{M,j}\|^2$. How badly this affects coverage for realistic insurance severity distributions — skewed, right-tailed, often with large individual claims — is unknown. The paper uses known dispersion in its simulations. Real pricing does not have that luxury.

---

## What this means for your validation workflow

In the two-part model, the pure premium relativity for factor $j$ is the product of the frequency relativity and the severity relativity. If you are checking whether a rating factor belongs in the model, you need both CIs to be valid.

Suppose you run the corrected Poisson inference on your frequency model using `PostSelectionGLM` from `insurance-gam`. You get a valid 95% CI for the frequency effect of vehicle group band 5: say, [+0.18, +0.31] on the log scale. That is honest. You then report the severity CI for the same variable from your Gamma model: say, [+0.04, +0.12]. That CI is still naive. It is probably covering at 75% rather than 95%.

The combined pure premium relativity looks more precisely estimated than it is, because the severity CI is too tight. If the borderline call on whether vehicle group band 5 belongs in the model is closer to the severity estimate than the frequency estimate — which it often is for severity, where effects are smaller and harder to detect — you are making that call with false precision.

There is no clean solution for Gamma today. The choices are:

**Data splitting.** Select variables for the severity model on one half of the data and run inference on the other. The inference half has not seen the selection, so standard Wald CIs are valid. This is the `DataSplitPostSelectionGLM` approach applied to severity, running outside the class (which currently only handles Poisson). Split the data, fit Lasso on half, refit Gamma MLE on the inference half using the selected features, report the Wald CIs from the inference half. This is always valid, uses no special machinery, and works for any GLM family. The cost is halving the inference sample, which for a UK motor portfolio with 300,000+ claims per year is typically acceptable.

**Explicit acknowledgement in governance.** If data splitting is not feasible for your severity model — thin commercial book, immature development, unusual feature set — then at minimum the model validation documentation should state clearly that severity CIs are naive post-selection estimates with unknown actual coverage. "We acknowledge that severity confidence intervals do not account for variable selection; we are exploring corrections" is a materially better statement than implying the intervals are valid. Under PRA CP6/24 insurance model risk guidance, governance documentation should be honest about known limitations. Pretending naive Wald CIs are valid when their coverage is known to be inflated is not an acceptable position once the problem is in the literature.

**Wait for the Gamma extension.** We plan to ship Gamma post-selection inference in `insurance-gam` once we have a simulation study confirming coverage. We will not ship it before then. The simulation requires care: realistic dispersion values for UK motor severity (coefficient of variation around 1.5–2.5), realistic sample sizes, and testing across the range of Lasso regularisation strengths that UK pricing teams actually use. When it is ready, we will publish the simulation results alongside the code.

---

## The elastic net complication

The problem is somewhat worse than it looks. UK pricing teams increasingly use elastic net (combined L1 + L2 penalty) rather than pure Lasso, because it handles correlated features more gracefully. Age and occupation are correlated. Area and vehicle type interact. Elastic net handles the multicollinearity more stably than Lasso, which can arbitrarily select one variable from a correlated group.

Shen et al. cover Lasso. Elastic net post-selection inference is a separate unsolved problem — the L2 penalty component changes the active set geometry in ways the Le Duy-Takeuchi path-tracing algorithm does not handle. There is no published elastic net PSI tool for GLMs, Poisson or Gamma. If your frequency or severity model uses elastic net, even the Poisson result from Shen et al. does not apply.

If you are using Lasso, the Poisson frequency side is now fixable. If you are using elastic net, you are in data-splitting territory for both components of the two-part model. This is not a catastrophic finding — data splitting is a perfectly valid approach with a straightforward justification — but it is worth knowing.

---

## The practical checklist

For a UK personal lines motor pricing team, right now:

**Frequency model, Lasso-selected, $n < 50{,}000$:** Use `PostSelectionGLM(family='poisson')` from `insurance-gam>=0.3.0`. Post-selection CIs are valid (asymptotically). Exposure offset is supported.

**Frequency model, Lasso-selected, $n > 50{,}000$:** Use `DataSplitPostSelectionGLM`. Or use `PostSelectionGLM` with the default 50,000-row subsample cap on path-tracing — the initial MLE uses all data; only the path-tracing step subsamples. For most UK motor portfolios, data splitting on $n = 300{,}000+$ is the cleaner story.

**Frequency model, elastic net-selected:** Data split. No alternative.

**Severity model, Lasso or elastic net-selected:** Data split. No alternative until the Gamma simulation study is complete.

**Stepwise-selected models (either component):** Data split. Post-selection inference for stepwise has a structurally different conditioning event; no open-source tool handles it. Stepwise should be on its way out of your pipeline regardless.

The one unambiguous improvement available today is on the Poisson frequency side with Lasso. That is real and worth taking. The severity side requires a workaround that is somewhat blunter but still valid.

---

## A note on regulatory framing

The post-selection inference problem is relevant to how you defend model significance in governance. If a validation team challenges a borderline rating factor — "what is the evidence that vehicle group band 5 belongs in the model?" — the answer involves a confidence interval. Under PRA CP6/24 principles for insurance model risk, the CI you quote should be one you believe is correctly calibrated.

A naive Wald CI on a Lasso-selected coefficient is not correctly calibrated. It is too narrow. Quoting it in a governance document is presenting a number with a known downward bias in uncertainty. The paper that identifies this problem is now nine days old and publicly available. The implication is that pricing teams who are aware of arXiv:2603.24875 — which, as of today, includes every team that reads this — cannot plausibly claim the problem is unknown to them.

This is not a regulatory crisis. No regulator has cited post-selection inference in an insurance context. It is a question of intellectual honesty in model documentation: if you know your CIs are probably miscalibrated, say so.

---

## Getting started

```bash
pip install "insurance-gam[glm]>=0.3.0"
```

For the Poisson frequency fix:

```python
from insurance_gam.post_selection import PostSelectionGLM, DataSplitPostSelectionGLM

# Frequency model (Poisson, Lasso, with exposure offset)
freq_model = PostSelectionGLM(family="poisson", alpha=0.05)
freq_model.fit(X_freq, y_counts, exposure=earned_exposure)
print(freq_model.coefficients())

# Severity model: implement the data split manually for Gamma
import numpy as np
from sklearn.linear_model import LassoCV
import statsmodels.api as sm

rng = np.random.default_rng(42)
n = len(X_sev)
select_idx = rng.choice(n, n // 2, replace=False)
infer_idx = np.setdiff1d(np.arange(n), select_idx)

# Step 1: select variables on the selection half
lasso = LassoCV(cv=5, max_iter=10_000, random_state=42)
lasso.fit(X_sev[select_idx], y_sev[select_idx])
selected_cols = np.where(np.abs(lasso.coef_) > 1e-8)[0]

# Step 2: infer on the inference half — Wald CIs are valid here
X_infer = X_sev[infer_idx][:, selected_cols]
glm_sev = sm.GLM(
    y_sev[infer_idx], sm.add_constant(X_infer),
    family=sm.families.Gamma(sm.families.links.Log())
).fit()
print(glm_sev.conf_int())  # these CIs are valid: no selection bias on the inference half
```

The paper is Shen, K., Gregory, K.B., and Huang, S. (2026), "Post-selection inference for generalized linear models", arXiv:2603.24875. Reference implementation at [github.com/kateshen28/InfGLM](https://github.com/kateshen28/InfGLM) (Poisson and logistic only, no offset, no Gamma).

---

## Related

- [Your GLM Confidence Intervals Are Wrong After Variable Selection](/2026/04/01/post-selection-inference-glm-confidence-intervals-lasso-insurance-pricing/) — the theory post: what post-selection bias is, why it matters, and how the Fisher scoring linearisation works
- [Post-Selection GLM Inference Is Now Usable in Python](/techniques/2026/04/03/post-selection-glm-confidence-intervals-insurance-pricing/) — the implementation post: `PostSelectionGLM` and `DataSplitPostSelectionGLM` in detail, with worked examples
