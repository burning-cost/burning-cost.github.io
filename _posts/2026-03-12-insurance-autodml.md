---
layout: post
title: "Continuous Treatment Causal Inference for Insurance Pricing: insurance-causal"
date: 2026-03-12
categories: [techniques, libraries]
tags: [causal-inference, double-machine-learning, price-elasticity, riesz-representer, continuous-treatment, debiased-ml, dml, fca, motor, home, retention, python]
description: "Automatic Debiased ML via Riesz Representers for continuous price elasticity. insurance-causal - no GPS density blow-up at tails. UK personal lines Python."
---

Your pricing team has a demand model. It says retention drops as premium rises. What it almost certainly cannot tell you is: by exactly how much, with what uncertainty, and whether that effect is the same for young drivers in postcodes with thin data as it is for your core 40-year-old motor book.

The standard approaches break down here in a predictable way. Binary treatment DML — split the portfolio into "high price" and "low price", apply CausalForestDML — works acceptably when the treatment is genuinely discrete. Premium is not discrete. Discretising it means choosing a cutpoint, and the cutpoint is arbitrary. Two otherwise identical analyses with cutpoints at the median versus the 60th percentile will give you different answers. You have introduced a researcher degree of freedom into what should be an estimation procedure.

The generalised propensity score (GPS) approach handles continuous treatments correctly in theory. In practice, GPS requires estimating the density of the premium distribution conditional on covariates — and then dividing by it. At the tails of the premium distribution, where renewal rates are 20–30% and your high-risk book is concentrated, those density values are small. You are dividing by small numbers. The resulting weights blow up, the variance explodes, and the confidence intervals become too wide to be actionable.

We built [`insurance-causal`](https://github.com/burning-cost/insurance-causal) — library 77 in the Burning Cost portfolio — to handle this correctly. It implements Automatic Debiased Machine Learning via Riesz Representers for continuous treatment causal inference. No density estimation. No arbitrary discretisation. Sqrt(n) inference with double robustness.

---

## Why standard DML breaks on continuous treatments

DoubleML and EconML both implement the Neyman-orthogonal score for partially linear models:

```
Y = theta * D + g(X) + epsilon
```

where theta is the treatment effect. For binary D (price up vs. price down), this is clean. For continuous D (the actual premium in pounds), you need an additional nuisance object: something that captures how the treatment is distributed conditional on covariates, so you can appropriately weight the residuals.

The standard answer is the GPS: estimate p(D | X) and use it to construct IPW weights. The problem is density estimation in high dimensions is hard, and the weights 1/p(D | X) are unstable when p is small. For a motor renewal portfolio where a 10% premium increase cuts renewal probability from 80% to 35%, the density of high premiums is genuinely thin. The GPS at the high end of the distribution is small, so the IPW weights are large, so the variance of the estimator inflates.

Hirshberg & Wager (2021, *Annals of Statistics*) showed there is a better path. Instead of estimating the GPS, estimate the *Riesz representer* of the target functional directly. For the Average Marginal Effect (AME), the Riesz representer is the function alpha(X) satisfying:

```
E[h(X) * alpha(X)] = E[m(W, h)]  for all test functions h
```

where m(W, h) is the Riesz functional evaluated at h. This can be estimated by minimising:

```
E[alpha(X)^2 - 2 * m(W, alpha)]
```

Any ML model can learn this — it is a regression problem. No density estimation. No division by small values. The minimax identity means you are directly learning the reweighting function you need for debiased inference, not computing it indirectly through a density.

---

## Three estimands

```bash
pip install insurance-causal
```

The library provides three estimands for continuous treatment pricing questions.

### PremiumElasticity: Average Marginal Effect

The AME answers: how does a £1 premium increase affect claims or retention, on average across the portfolio?

```python
from insurance_autodml import PremiumElasticity

model = PremiumElasticity(
    outcome_family="poisson",   # "tweedie" for pure premium, "gaussian" for log-transformed
    n_folds=5,
    riesz_type="forest",        # ForestRiesz: recommended for most insurance data
    inference="eif",            # Efficient influence function; faster than bootstrap
)
model.fit(X, D, Y, exposure=exposure)
result = model.estimate()
print(result.summary())
# AME = -0.0031 (SE = 0.0004, 95% CI: [-0.0039, -0.0023])
# n = 142,817 | folds = 5
```

The AME is a linear functional of the data distribution, which means it achieves sqrt(n) convergence rates even when the nuisance models (E[Y|D,X] and the Riesz representer) converge at slower rates. You can use forests, gradient boosting, or any other ML method for the nuisances without contaminating the rate of convergence of the main estimate.

The `outcome_family` parameter matters. For claim counts with an exposure offset, use `"poisson"` — the nuisance model fits Y/exposure and the AME is on the rate scale. For pure premium (frequency × severity combined), use `"tweedie"`. For retention (lapse indicator), use `"gaussian"` or leave it as default.

Segment-level effects come from the same fit — no refitting required. The efficient influence function decomposes additively over subgroups:

```python
segment_results = model.effect_by_segment(df["vehicle_group"])
for seg in segment_results:
    print(f"{seg.segment_name}: {seg.result.estimate:.4f} "
          f"(95% CI: {seg.result.ci_low:.4f} to {seg.result.ci_high:.4f})")
```

This is important: you do not pay a multiple comparisons penalty if you have not specified the segments in advance. Each segment_result is using the same pooled nuisance estimates, only splitting the influence function scores for inference.

### DoseResponseCurve: E[Y(d)] across premium levels

The dose-response curve answers a different question: what would the average claim rate (or retention) be if every policyholder in the portfolio paid premium d? Not "what is the marginal effect", but "what is the level at a specific price point?"

This uses the Colangelo-Lee kernel-DML approach (JBES 2025, arXiv:2004.03036), which wraps kernel-weighted doubly-robust scores around the cross-fitted nuisance:

```python
from insurance_autodml import DoseResponseCurve
import numpy as np

drc = DoseResponseCurve(
    outcome_family="poisson",
    bandwidth="silverman",      # or "cv" for data-driven bandwidth, or a float
    kernel="gaussian",
)
drc.fit(X, D, Y, exposure=exposure)

d_grid = np.linspace(300, 800, 50)     # £300 to £800 premium grid
result = drc.predict(d_grid)

# result.ate: array of shape (50,) — estimated E[Y(d)] at each grid point
# result.ci_low, result.ci_high: pointwise 95% confidence bands
```

The key difference from a naive regression of Y on D is confounding adjustment. The dose-response curve estimates a *causal* counterfactual — what would happen to your portfolio if you could set everyone's premium to d — not a conditional expectation that conflates risk selection with price response.

For visualisation:

```python
ax = drc.plot(d_grid, xlabel="Premium (£)", ylabel="Claim rate")
```

The bandwidth matters for the dose-response curve in a way it does not for the AME. Too narrow and you get high-variance point estimates at sparse premium values; too wide and you smooth over real non-linearity. Silverman's rule is a reasonable default for unimodal premium distributions. If your portfolio has a multi-modal premium distribution (e.g. a commercial book mixed with personal lines), try `bandwidth="cv"` to let the data choose.

### PolicyShiftEffect: portfolio-level counterfactuals

The policy shift estimand answers the question pricing teams actually ask before an annual renewal cycle: if we increase all premiums by 5%, what happens to aggregate claims or retention?

```python
from insurance_autodml import PolicyShiftEffect

pse = PolicyShiftEffect(outcome_family="gaussian")
pse.fit(X, D, Y)

# 5% increase
result_5pct = pse.estimate(delta=0.05)
print(f"Impact of +5%: {result_5pct.estimate:.4f} "
      f"(95% CI: {result_5pct.ci_low:.4f} to {result_5pct.ci_high:.4f})")

# Sweep across a range of changes
deltas = np.linspace(-0.10, 0.10, 21)
curve = pse.estimate_curve(deltas)
```

The `estimate_curve` method is efficient: it fits the nuisance models once in `fit()` and reuses them across all delta values. The underlying Riesz representer for the shift functional is approximated using the same ForestRiesz estimates from the AME, which is valid for small delta (|delta| < 0.15 in practice).

---

## ForestRiesz: why it works at the tails

The GPS instability problem is worst at the tails of the premium distribution. At £750 renewal premium for a young driver in a high-crime postcode, the number of observations is small and the renewal rate is low. GPS estimation here produces small density values that, when inverted, give enormous IPW weights.

ForestRiesz avoids this entirely. The Riesz regression target is the numerical derivative of the nuisance model:

```
z_i = [g(D_i + eps, X_i) - g(D_i - eps, X_i)] / (2 * eps)
```

This is the partial derivative of E[Y|D,X] with respect to D, evaluated at the observed data point. A random forest learns alpha(X) by regressing onto z. There is no density in this calculation. There is no inversion. The 99th percentile of |z| is clipped before fitting to handle any residual numerical instability at extreme premium values.

The default hyperparameters — 200 trees, max_depth=6, min_samples_leaf=10 — work well on motor and home renewal books in the 50k–500k policy range. For smaller books:

```python
from insurance_autodml import PremiumElasticity

model = PremiumElasticity(
    outcome_family="poisson",
    n_folds=3,                              # 3 folds for n < 2000
    riesz_type="forest",
    riesz_kwargs={"n_estimators": 100, "min_samples_leaf": 20},
)
```

You can assess Riesz representer quality using the held-out minimax loss:

```python
loss = model.riesz_loss()
# Lower is better. Values above 0.1 (on standardised scale) suggest poor
# Riesz estimation — consider more flexible nuisance models.
```

For fast experimentation or when you want a linear baseline before committing to the forest:

```python
model = PremiumElasticity(riesz_type="linear")
```

LinearRiesz solves the same minimax objective using ridge regression. It is ten times faster and gives you a sanity check on the forest estimate. If the two agree within uncertainty, you have some confidence the nonparametric estimate is not overfit.

---

## Selection correction for renewal portfolios

UK motor and home renewal portfolios have a selection problem that goes beyond confounding: you only observe claims for policies that renewed. Non-renewers lapse, and their outcomes are missing. Higher premiums cause more lapses, so the observed claim experience is systematically drawn from the lower-risk portion of the portfolio that stayed.

If you estimate the AME naively on renewals, you understate the true causal effect of premium increases on claims — because the retained sample at high premiums is risk-selected, not representative.

`SelectionCorrectedElasticity` handles this. The identification follows the recent extension of the Riesz framework to missing outcomes (arXiv:2601.08643). A selection model P(S=1 | X, D) is estimated jointly with the outcome nuisance, and the EIF score is IPW-corrected:

```python
from insurance_autodml import SelectionCorrectedElasticity

# S: renewal indicator (1 = renewed, 0 = lapsed)
# Y: claims — for S=0 observations, Y is ignored (can be 0 or NaN)
model = SelectionCorrectedElasticity(
    outcome_family="poisson",
    n_folds=5,
)
model.fit(X, D, Y, S=S, exposure=exposure)
result = model.estimate()
```

The corrected EIF score is:

```
psi_i = S_i / pi_hat_i * alpha_hat_i * (Y_i - g_hat_i) + alpha_hat_i
```

where pi_hat_i = P(S=1 | X_i, D_i) estimated by cross-fitted gradient boosting. Selection probabilities are clipped to [0.05, 0.95] to prevent extreme IPW weights at the tails.

The identification assumption is that there are no *unobserved* confounders of selection — that conditional on X and D, which observable features you have, the decision to renew is as good as random. This is a strong assumption. The library tests robustness to violations:

```python
# Gamma sensitivity bounds: how much unobserved confounding can the estimate survive?
# Gamma=1: no unobserved confounding (point identified)
# Gamma=2: unobserved confounders could double or halve the selection odds
bounds = model.sensitivity_bounds(gamma_grid=[1.0, 1.5, 2.0, 3.0])

for gamma, b in bounds.items():
    print(f"Gamma={gamma}: AME in [{b['lower']:.4f}, {b['upper']:.4f}]")
```

If the estimate is stable across Gamma=1 through Gamma=2, you have evidence that moderate unobserved selection confounding does not overturn the conclusion. If the bounds cross zero at Gamma=1.5, the result is fragile and you should say so in any FCA submission.

---

## FCA evidence output

The `ElasticityReport` class generates HTML reports suitable for inclusion in a pricing review submission:

```python
from insurance_autodml import ElasticityReport

report = ElasticityReport(
    estimator=model,
    segment_results=model.effect_by_segment(df["vehicle_group"]),
    sensitivity_bounds=bounds,          # optional: from SelectionCorrectedElasticity
)
report.to_html("elasticity_report.html")
```

The HTML output includes the overall AME with confidence interval and p-value, the segment table, sensitivity analysis plots, and a plain-English methodology section written for a regulatory audience. The methodology notes explain the Riesz approach without requiring the reader to know what a Riesz representer is — which is the right level for an FCA evidence pack.

Under FCA pricing review, insurers need to demonstrate that pricing differentials reflect genuine risk differentiation, not differential treatment of loyal customers. An AME estimate with appropriate confidence intervals and sensitivity analysis is the right form of evidence. A naive regression is not.

---

## When to use which estimand

**PremiumElasticity (AME)** is the right starting point for almost every analysis. It answers "what is the marginal effect of premium on outcomes, on average?" It is a single number with a confidence interval. It is what you present to the underwriting committee before a rate walk. The segment decomposition lets you check whether the effect is uniform or concentrated in a particular book segment.

**DoseResponseCurve** is for when you need the full price-outcome relationship, not just the slope at the current premium level. Use it when you are evaluating a proposed rate change that moves premiums significantly from current levels — for example, repricing a motor sub-book from a mean of £450 to a mean of £600. The dose-response curve tells you what to expect at £600 given causal identification, not just a regression extrapolation.

**PolicyShiftEffect** is for portfolio-level projections. It is faster than computing the full dose-response curve and directly answers the commercial question: if we apply a 7% rate increase, what is the projected impact on claims frequency? The delta sweep gives you a curve that feeds directly into the optimisation layer.

**SelectionCorrectedElasticity** should replace PremiumElasticity when you have renewal portfolios with observable selection (i.e. you observe whether policies lapse). For new business conversion, where the selection process is the same as the treatment process, the standard AME is appropriate.

---

## Library details

`insurance-causal` has 199 tests across all estimators, nuisance backends, and inference modes. Dependencies: numpy, pandas, scikit-learn. Optional: catboost for `nuisance_backend="catboost"`, matplotlib for `drc.plot()`.

```bash
pip install insurance-causal

# With CatBoost nuisance backend:
pip install "insurance-causal[catboost]"

# With plotting:
pip install "insurance-causal[plots]"
```

Source and tests: [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal).

The academic groundwork is Hirshberg & Wager (2021), *Double Robustness of Local Average Treatment Effects in the Frequentist and Bayesian Settings*, *Annals of Statistics*; the ForestRiesz construction and the missing-outcome extension are from arXiv:2601.08643; the dose-response curve implementation follows Colangelo & Lee, *Journal of Business & Economic Statistics* (2025), arXiv:2004.03036.

- [Your Demand Model Is Confounded](/2026/03/01/your-demand-model-is-confounded/)
- [Your Rating Factor Might Be Confounded](/2026/03/05/your-rating-factor-might-be-confounded/)
- [Your Rate Change Didn't Prove Anything](/2026/03/13/your-rate-change-didnt-prove-anything/)
