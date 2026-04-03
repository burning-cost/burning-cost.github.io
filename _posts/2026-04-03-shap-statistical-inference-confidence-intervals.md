---
layout: post
title: "Your SHAP Feature Importance Bar Chart Has No Error Bars — Here Is How to Fix That"
date: 2026-04-03
categories: [research, machine-learning, governance]
tags: [SHAP, feature-importance, statistical-inference, confidence-intervals, model-governance, Consumer-Duty, shap-relativities, U-statistics, Neyman-orthogonal, Whitehouse-Sawarni-Syrgkanis-2026, UK-motor, FCA, debiasing]
description: "A February 2026 paper provides the first statistically valid confidence intervals for global SHAP feature importance. We explain what changes for UK insurance pricing teams, when the theory holds, and how we plan to implement it in shap-relativities."
author: burning-cost
---

See also our earlier coverage of [SHAP feature importance inference and governance implications](/2026/04/02/shap-inference-error-bars-feature-importance-governance/).

Every pricing model sign-off we have seen includes a SHAP importance bar chart. Feature A is the tallest bar. Feature B is slightly shorter. The committee nods and moves on.

Nobody asks the obvious question: are those bars actually different from each other, or are we reading noise?

Until February 2026 there was no statistically valid answer. The standard approach was either a naive bootstrap (which carries the bias of the SHAP estimator itself) or nothing at all. A paper from Whitehouse, Sawarni, and Syrgkanis — arXiv:2602.10532 — fixes this. It provides asymptotically normal estimators for global SHAP importance with confidence intervals that have correct frequentist coverage. We think this is one of the more practically useful pieces of ML theory we have seen in a while, and we intend to implement it in [shap-relativities](https://github.com/burning-cost/shap-relativities).

---

## What global SHAP importance actually means — and why inference is hard

The standard global importance metric for feature *a* is the mean absolute SHAP: take the absolute value of each policy's SHAP value for that feature and average across the book. Call this theta_1. The p=2 variant uses mean squared SHAP (theta_2).

The difficulty with putting a confidence interval on theta_1 is that it is not a simple sample mean of observed quantities. SHAP values are themselves estimated from the data — they depend on a conditional mean function mu(X) that you have fitted. A naive CI treats the SHAP values as if they were fixed known numbers. They are not. The estimation uncertainty in mu propagates into the SHAP values and then into your importance estimate. The result is a biased interval with incorrect coverage.

The paper uses the same debiasing machinery as double/debiased machine learning (Chernozhukov et al., 2018). It constructs a Neyman-orthogonal score — a modified estimating equation whose first-order sensitivity to errors in nuisance estimation is zero. Plug in machine-learning estimates of the nuisance functions, and those estimation errors vanish from the leading term. What remains is a standard asymptotically normal U-statistic. The confidence interval is:

```
theta_hat ± z_{α/2} × (σ_hat / √n)
```

This is a valid asymptotic interval for the true population theta_p, provided the nuisance estimates converge at the n^{-1/4} rate — achievable by gradient boosting on any UK motor book.

Mean absolute SHAP (p=1) requires extra work because |x| is not twice differentiable at zero. The paper handles this with a smoothed approximation using a tanh-based regulariser, parametrised by a temperature beta_n that grows with sample size. The asymptotics recover at the end. It is more involved but the intuition is the same.

---

## When the theory holds for UK insurance

The theory rests on four assumptions. Three are straightforward for UK personal lines. One is not.

**Bounded outcomes.** For claim count models (Poisson, Tweedie) this is satisfied — counts are small integers. For severity models, UK motor bodily injury has a heavy tail. You need to winsorise at the 99.9th percentile before fitting. This is standard practice and introduces only modest bias in the SHAP estimates.

**Interventional SHAP, not path-dependent TreeSHAP.** This is the binding constraint. The paper's estimand is the population mean of |phi_a(X)|^p where phi_a is the true interventional SHAP — the one that marginalises over X_{-S} using the data distribution. The default TreeSHAP in the shap library uses the in-tree distribution, which is fast but does not match this definition when features are correlated. For UK motor — where age, NCD, and vehicle group are all correlated — the default will give you CIs that do not have the stated coverage.

The fix: use `feature_perturbation='interventional'` with a representative background sample. This is already what we recommend in shap-relativities for governance use. If you are already doing that, you are set.

**Sample size.** The nuisance convergence requirement is roughly n >= 10,000 for gradient boosting on smooth problems. UK motor books run 100k–500k policies. No issue. Specialty lines (professional indemnity, cyber) at n=5,000–20,000 will get wider CIs and approximate rather than exact coverage — but approximate is still better than nothing.

**Mass at zero SHAP.** If a feature has more than around 20% of policies with exactly zero SHAP (e.g. a feature that only applies to one policy type), the smoothing for p=1 needs a larger beta parameter. Document it and move on.

---

## What this changes for a pricing team

**Model sign-off.** The SHAP bar chart becomes a chart with error bars. Features whose CIs overlap cannot be treated as having a confirmed ranking. "Vehicle group is our third most important feature" either survives this scrutiny or it does not. We think most pricing committees will find some surprises.

**Model version comparison.** When you refresh a model, the current standard is a side-by-side bar chart and a qualitative judgment about whether the importance order changed. With valid inference you can run a two-sample test:

```
T = (theta_hat_a,v2 − theta_hat_a,v1) / √(Var_v2/n + Var_v1/n)
```

This is a standard z-test using the asymptotic variances from the two models. You get a p-value. "Postcode's importance dropped significantly in the refresh" becomes a statement you can defend to the pricing committee, not a visual impression.

**FCA Consumer Duty.** FCA PS22/9 (in effect July 2023, evidence requirements from July 2024) requires firms to demonstrate fair value. When the FCA asks "is postcode a material pricing driver?", a CI that excludes zero at the 95% level is defensible evidence that it is. When they ask "has the contribution of this proxy changed post-model refresh?", a two-sample test answers that directly. The paper turns qualitative SHAP charts into quantitative governance evidence. We think this is the most practically significant application for UK pricing teams.

---

## How this fits into shap-relativities

The library currently provides per-level CIs: for a given level of a categorical feature (say, postcode area = "SW1"), you get a CI on the mean SHAP within that level. That answers "how uncertain is this level's relativity?" — a within-group question.

The new `SHAPInference` class (planned for v0.3.0) answers a different question: "how uncertain is this feature's global importance rank?" These are complementary tools, not alternatives.

The planned API looks like this:

```python
from shap_relativities import SHAPRelativities, SHAPInference

sr = SHAPRelativities(model, feature_perturbation="interventional", background=bg_data)
shap_vals = sr.shap_values(X_val)

inf = SHAPInference(
    shap_values=shap_vals,
    y=y_val,
    feature_names=feature_names,
    p=2,          # mean squared SHAP; use p=1 for mean absolute
    n_folds=5,
    ci_level=0.95,
    random_state=42,
)
inf.fit()

table = inf.importance_table()
# Returns: feature, theta_hat, theta_lower, theta_upper, se, rank, p_value_nonzero
```

The `ranking_ci` method tests whether feature A is genuinely more important than feature B, using the joint asymptotic distribution of the two importance estimates. The covariance comes from the cross-product of their influence functions — estimable from the same data.

One honest caveat: the alpha nuisance function (the response correction term) requires, in principle, density ratio estimation over all feature subsets. For a 20-feature model that is 2^19 subsets — computationally infeasible in the general case. For interventional SHAP under approximate feature independence, the density ratios simplify and the correction reduces to a regression of weighted residuals on X. That is what we will implement, with the assumption clearly documented. For UK motor with the kind of feature sets typically used (age, NCD, vehicle group, area, telematics), the independence approximation is reasonable. For models where features are heavily engineered to be orthogonal, it is exact.

---

## Build status

The paper is from February 2026. There is no public reference implementation. We have assessed the methodology and our shap-relativities build plan is:

- **v0.3.0 (p >= 2):** Core `SHAPInference` class with fit, importance_table, and ranking_ci. The p=2 case has clean theory and no smoothing complications. This is the version most useful for governance.
- **v0.3.1:** plot_importance with CI error bars, multiple testing correction (Bonferroni/BH).
- **v0.4.0 (p=1):** Mean absolute SHAP with the tanh smoothing. This is the headline number most people use.
- **v0.5.0:** Convenience method `sr.importance_inference()` on `SHAPRelativities` for one-line access.

If you are working on pricing model governance and want to pilot this before it is in the library, get in touch. We are particularly interested in cases where the ranking uncertainty changes a committee's decision.

---

## The bigger picture

The actuarial profession has spent twenty years debating whether GBMs are explainable enough for model governance. SHAP answered that question well enough for most purposes. The remaining gap has been that the explanations are point estimates — they look precise but carry no uncertainty.

This paper closes that gap for global feature importance. The methodology is sound, the assumptions are achievable for most UK personal lines books, and the implementation is tractable. We do not think there is a good reason to present a SHAP importance chart without confidence intervals once this is available. The bar chart without error bars will start to look like what it is: a point estimate of something uncertain.

---

*Paper: Whitehouse, Sawarni, Syrgkanis (2026). "Statistical Inference and Learning for SHAP." arXiv:2602.10532. Regulatory references: FCA PS22/9 (Consumer Duty fair value), FCA EP25/2 (proxy discrimination).*
