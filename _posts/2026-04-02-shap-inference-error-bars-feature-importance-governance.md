---
layout: post
title: "Your SHAP Bar Chart Has No Error Bars"
date: 2026-04-02
categories: [libraries]
tags: [shap, governance, fca, shap-relativities, consumer-duty, feature-importance, inference, arXiv-2602.10532, Whitehouse, Sawarni, Syrgkanis]
description: "SHAPInference (in development for shap-relativities): asymptotically valid confidence intervals on global SHAP feature importance using de-biased U-statistics. Every SHAP importance ranking you have ever put in a governance document is a point estimate. This is the fix."
author: burning-cost
---

Every SHAP importance bar chart you have ever put in a pricing committee pack, a Consumer Duty evidence file, or a model validation report has the same property: it is a point estimate with no uncertainty quantification. The bars have no error bars. Feature A is ranked above feature B because E[|phi_A(X)|] > E[|phi_B(X)|] in your sample. You do not know whether that ordering reflects the true population importance or just sampling noise.

This is not a minor presentation issue. It is a governance problem. When you write "vehicle group is the third most important pricing factor", you are making a claim about your model's structure that your evidence does not actually support. You have a single realisation from the distribution of importance estimates. The ranking is a random variable, and you have no interval on it.

`SHAPInference` is being added to [shap-relativities](https://pypi.org/project/shap-relativities/), implementing the de-biased U-statistic estimator from Whitehouse, Sawarni, and Syrgkanis (arXiv:2602.10532, February 2026). It produces asymptotically valid confidence intervals on theta_p = E[|phi_a(X)|^p] for each feature — the standard global SHAP importance measure — and a direct test for whether feature A is truly more important than feature B.

---

## Why the naive bootstrap fails

The natural response is: just bootstrap. Resample your policy data with replacement 1000 times, recompute SHAP importance each time, take the 2.5th and 97.5th percentiles. This gives an interval with a problem.

SHAP values are not observed quantities. They are functionals of the conditional mean function mu(X) = E[Y | X], which is itself estimated from the same data. When you bootstrap, you simultaneously resample the outcome Y and re-estimate the model that computes SHAP from Y. The SHAP importance estimate inherits bias from the nuisance estimation of mu. The bootstrap interval, centred on a biased point estimate, has systematically incorrect coverage — it does not converge to a valid 95% interval as n grows.

This is the same problem that makes naive bootstrap invalid for treatment effect estimation when the propensity score is estimated. The fix, developed across two decades of semi-parametric inference, is de-biasing.

---

## What the paper does

Whitehouse et al. work out the asymptotic theory for estimating theta_p = E[|phi_a(X)|^p]. The two most relevant cases are p=2 (mean squared SHAP, the theoretically cleaner case) and p=1 (mean absolute SHAP, the number on every standard SHAP importance plot).

The core idea is Neyman-orthogonal de-biasing. They construct a score function whose first-order sensitivity to errors in the nuisance estimates (errors in mu, errors in the sensitivity function gamma_p) is exactly zero. This means that even if mu is estimated at the modest rate of n^{-1/4} — achievable by gradient boosting on smooth problems — the resulting estimator for theta_p is sqrt(n)-consistent and asymptotically normal. You get a proper central limit theorem.

The estimator takes the form of a U-statistic with a symmetrised two-observation score. In practice, the naive O(n^2) U-statistic computation is unnecessary: the Hoeffding projection identity lets you reduce it to an O(n) calculation after estimating the nuisance functions with cross-fitting. This matters for insurance datasets where n might be 200,000 policies.

For p=1, there is an additional complication. The function |x|^p is not twice differentiable at x=0, so the standard de-biasing argument breaks down if SHAP values have mass near zero. The paper's solution is smooth approximation: replace |phi|^1 with phi_{1,beta}(phi) = |phi| * tanh(beta * |phi|), where the temperature parameter beta grows with n. As beta tends to infinity, the approximation converges to |phi|. The rate requirement is beta_n = O(n^{1/4}) under mild density regularity. The resulting CI is asymptotically valid for the unsmoothed theta_1.

---

## The API

```python
from shap_relativities import SHAPRelativities, SHAPInference

# Fit your model and get interventional SHAP values
sr = SHAPRelativities(model, background_data, feature_perturbation="interventional")
shap_vals = sr.shap_values(X_test)  # shape (n_obs, n_features)

# Compute CIs on global importance
inf = SHAPInference(
    shap_values=shap_vals,
    y=y_test,
    feature_names=feature_names,
    p=2,          # mean squared SHAP; cleaner theory than p=1
    n_folds=5,
    ci_level=0.95,
    random_state=42,
)
inf.fit()

# Importance table with CIs
table = inf.importance_table()
# columns: feature, theta_hat, theta_lower, theta_upper, se, rank, rank_lower, rank_upper, p_value_nonzero

# Direct ranking test: is vehicle_group truly more important than postcode?
result = inf.ranking_ci("vehicle_group", "postcode")
# returns: diff, se_diff, z_stat, p_value, ci_lower, ci_upper

# Bar chart with error bars
inf.plot_importance(top_n=15)
```

The `importance_table()` output gives you two ranks per feature: the point-estimate rank (by theta_hat) and the conservative and optimistic ranks computed from the CI bounds. When the confidence intervals of two features overlap substantially, their conservative and optimistic ranks will differ — that is the signal that the ordering is uncertain.

---

## What this means for FCA Consumer Duty evidence

PS22/9 requires firms to demonstrate fair value. In practice, this means documenting which features drive pricing and whether protected characteristic proxies are contributing more than legitimate risk factors. Most firms currently do this by presenting SHAP importance bar charts with no uncertainty quantification.

`SHAPInference` makes three specific improvements to that evidence:

**Materiality with a threshold.** "Is postcode a material pricing driver?" becomes answerable: if the 95% CI for theta_2 of postcode excludes zero, postcode is a material driver at the 5% significance level. This is a defensible, reproducible claim — not a judgment call about whether the bar on the chart looks big.

**Ranking claims.** "Vehicle group is more important than postcode" can be tested directly with `ranking_ci()`. A p-value below 0.05 means the difference is statistically confirmed at your sample size. A p-value of 0.3 means the ordering could easily have been reversed in a different sample. Governance packs should state which rankings are and are not confirmed.

**Version comparison.** "Has the relative importance of occupation changed since the last model refresh?" is currently answered by inspection. With two fitted `SHAPInference` objects (one per model version), you run a standard two-sample z-test: T = (theta_hat_a_v2 - theta_hat_a_v1) / sqrt(Var_v2/n_v2 + Var_v1/n_v1). Asymptotic normality holds by independence of the two models. You now have a significance level on the change.

---

## Honest limitations

We are not going to oversell this.

**Requires interventional SHAP.** The theoretical guarantees require SHAP values computed by marginalising over the background distribution — interventional SHAP (`feature_perturbation="interventional"` in the shap library). The default path-dependent TreeSHAP uses the in-tree distribution, not the true marginal, and the resulting values are not the quantity the theory is about. Using `SHAPInference` with path-dependent SHAP values will produce confidence intervals with unknown coverage properties. We emit a warning; in a future version we may make this an error. For most pricing models with correlated features (and age and NCD are always correlated), this is the binding constraint on adoption.

**p=1 is approximate.** The mean absolute SHAP case uses the tanh smoothing described above. The approximation is valid asymptotically and conservative in practice — the default beta schedule (beta_n = n^{1/4} under delta=1 density regularity) is designed to err on the side of wider intervals. But p=2 has cleaner theory, and for governance purposes mean squared SHAP is equally defensible. If you have a strong preference for p=1, ship it with the caveat; if you are choosing fresh, use p=2.

**Large n required.** The nuisance estimators need to achieve n^{-1/4} convergence rates. For UK motor books (100k-500k policies) this is easy. For specialty lines — professional indemnity, cyber, small commercial with 5,000-20,000 observations — the convergence rate is marginal and the CIs should be treated as approximate rather than exact at finite sample. We suggest checking empirically that CI width is decreasing roughly as 1/sqrt(n) at your actual data size.

**Multiple testing.** `importance_table()` returns p-values for all features. With 20 features in a typical motor model, five-percent significance against a single feature is not the same as five-percent significance against the most extreme feature. Apply Bonferroni or Benjamini-Hochberg correction before drawing conclusions across the full feature set. A `correction` parameter is on the roadmap for v0.5.1.

---

## What this does not replace

The existing shap-relativities per-level CIs remain. Those answer a different question: "how uncertain is the mean SHAP for area=London?". `SHAPInference` answers "how uncertain is area's global importance rank?". They are complementary. Governance packs benefit from both: per-level CIs for relativity validation, global importance CIs for feature ranking claims.

---

## Getting it

**Note: `SHAPInference` is not yet released.** The current library version is v0.2.6; v0.5.0 is in development. The API shown above reflects the planned implementation. Watch the [GitHub repository](https://github.com/burning-cost/shap-relativities) for the release.

The underlying paper is Whitehouse, Sawarni, Syrgkanis (2026), arXiv:2602.10532.

If you are writing a Consumer Duty model governance pack and want to put defensible confidence intervals on your feature importance rankings, this is the tool we are building. Questions and use-case input welcome via the issue tracker.
