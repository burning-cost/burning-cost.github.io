---
layout: post
title: "Double/Debiased Machine Learning for Insurance Pricing: A Practitioner Guide"
date: 2026-04-01
categories: [techniques, causal-inference]
tags: [DML, double-machine-learning, causal-inference, price-elasticity, telematics, treatment-effects, insurance-causal, omitted-variable-bias, nuisance-functions, cross-fitting, riesz-representer, python, arXiv-2504-08324, Chernozhukov, Ahrens, Hansen]
description: "DML removes the omitted variable bias that makes naive GLM price elasticity estimates wrong by 20–80%. We explain why it works, show the two core insurance applications — price elasticity and telematics treatment effects — and provide working code using insurance-causal."
math: true
author: burning-cost
---

A new tutorial from the method's original authors — Achim Ahrens, Victor Chernozhukov, Christian Hansen, Damian Kozbur, Mark Schaffer, and Thomas Wiemann (arXiv:2504.08324, revised February 2026) — is the most complete accessible treatment of Double/Debiased Machine Learning (DML) yet published. The paper covers the two biases that arise when you use the same data to estimate nuisance functions and treatment effects, how cross-fitting resolves them, and practical guidance on ML method choice. The paper includes code and simulations throughout.

We implemented DML in `insurance-causal` before this tutorial appeared. This post bridges the gap between the econometrics exposition and what a pricing actuary actually needs to do with observational insurance data.

---

## The problem DML solves

You have a portfolio of UK motor renewals. You want to know: if we increase a customer's premium by 1%, how does their claim frequency change? You fit a Poisson GLM with log-premium as a covariate alongside the usual rating factors — age, NCB, vehicle group, region. The coefficient on log-premium is your elasticity estimate.

That estimate is biased. Not slightly — in our benchmarks on synthetic UK motor data with realistic confounding structure, OLS/GLM bias relative to the true elasticity is 20–80%.

The reason is structural. Your pricing model generates the premium as a function of rating factors: higher-risk vehicles get higher premiums, lower-NCB drivers pay more. When you then regress claim frequency on premium, you are partially measuring the premium effect and partially measuring the rating factor effects that drove the premium in the first place. The correlation between premium and claims has two sources — causal price effect, and shared risk factor variation — and the GLM coefficient confounds them.

The formal name for this is **omitted variable bias**. Even with a long list of rating factors in the model, any factor that jointly influences premium assignment and claims and that is not perfectly observed produces bias. Historically, the remedy was a randomised price experiment. DML gives you a valid estimate from the observational data you already have.

---

## What DML does: the residualisation idea

The core of DML is simple enough to state in three lines:

1. Fit $\hat{m}(X) = \mathbb{E}[Y \mid X]$ (outcome nuisance) on all data except the current fold. Compute residuals $\tilde{Y} = Y - \hat{m}(X)$.
2. Fit $\hat{\ell}(X) = \mathbb{E}[D \mid X]$ (treatment nuisance) on all data except the current fold. Compute residuals $\tilde{D} = D - \hat{\ell}(X)$.
3. Regress $\tilde{Y}$ on $\tilde{D}$. The coefficient $\hat{\theta}$ is the causal treatment effect.

This is the Frisch-Waugh-Lovell theorem extended to nonparametric nuisance functions. By partialling out $X$ from both $Y$ and $D$ before the causal regression, you remove the confounding variation. What remains in $\tilde{D}$ is the part of the premium that is *not* explained by the rating factors — genuinely exogenous variation. The regression of $\tilde{Y}$ on $\tilde{D}$ then recovers the causal effect.

The Neyman-orthogonal score for the partially linear model is:

$$\psi(W; \theta, \eta) = (\tilde{Y} - \theta \tilde{D}) \cdot \tilde{D}$$

where $W = (Y, D, X)$ and $\eta = (m, \ell)$ are the nuisance functions. The key property: the score's derivative with respect to $\eta$ is zero at the true nuisance values. This means errors in the nuisance estimation contribute only second-order bias to $\hat{\theta}$, not first-order. You can use flexible ML methods (gradient boosting, random forests, Lasso) for the nuisance functions without contaminating the final causal estimate — provided you use **cross-fitting** to avoid overfitting.

---

## Why cross-fitting matters

Cross-fitting is non-negotiable and is where practitioners most often make mistakes.

If you fit the nuisance models on the same data used to compute residuals, the residuals are contaminated by in-sample overfitting. A gradient-boosted tree that achieves near-perfect in-sample fit for $\hat{m}(X)$ will produce near-zero residuals $\tilde{Y}$, and the causal estimate $\hat{\theta}$ will be undefined or degenerate. More subtly, moderate overfitting produces finite-sample bias that does not vanish as $n \to \infty$ — the Neyman orthogonality property only holds for nuisance estimates that are asymptotically independent of the score data.

The fix is $K$-fold cross-fitting:
- Split the data into $K$ folds (5 is the standard; 3 for small samples).
- For each fold $k$, fit the nuisance models on all folds except $k$.
- Predict residuals on fold $k$ using the out-of-fold models.
- Stack the residuals across all $K$ folds to cover the full sample.
- The final causal regression uses these out-of-fold residuals.

The Ahrens et al. tutorial (Section 3) formalises this and shows that with $K \geq 2$ folds and nuisance models estimating at the $n^{-1/4}$ rate, the resulting $\hat{\theta}$ achieves $\sqrt{n}$-consistency and asymptotic normality. Most reasonable ML methods exceed the $n^{-1/4}$ rate requirement in practice.

---

## Two biases, one fix

The Ahrens et al. tutorial title 'two biases' refers to a distinction worth understanding.

**Regularisation bias.** ML nuisance models trade variance for bias — Lasso shrinks coefficients, forests average over noisy splits. If you use a Lasso to estimate $\hat{\ell}(X)$ and the Lasso omits a relevant confounder (sets its coefficient to zero), the residual $\tilde{D}$ still contains that confounder's variation. The final regression absorbs it into $\hat{\theta}$, biasing the causal estimate. Cross-fitting does not fix this; it requires the nuisance models to be *consistent* for their targets.

**Overfitting bias.** When nuisance estimation and score computation share the same data, the in-sample fit quality of the nuisance model inflates the signal in the score. Cross-fitting removes this by ensuring the nuisance models never see the data they score.

The practical implication: choose nuisance models that are flexible enough to capture the true confounding structure (avoid badly mis-specified parametric models) but do not over-regularise. In insurance data — where confounders are often multiplicative interactions between rating factors — gradient-boosted trees with modest depth (3–5) and L2 regularisation work well as nuisance models.

---

## Insurance application 1: price elasticity of demand

The canonical application is estimating how a premium change causally affects renewal probability or claim frequency. The treatment $D$ is log-premium or percentage price change. The outcome $Y$ is the renewal indicator or claim count. The confounders $X$ are the rating factors.

The partially linear model is:

$$Y_i = \theta D_i + g_0(X_i) + \varepsilon_i$$

where $\theta$ is the causal price elasticity and $g_0$ absorbs all confounder effects on the outcome. We do not need to specify $g_0$ parametrically.

Using `insurance-causal`:

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import PriceChangeTreatment

model = CausalPricingModel(
    outcome="renewal",
    outcome_type="binary",
    treatment=PriceChangeTreatment(column="pct_price_change"),
    confounders=["age", "vehicle_age", "postcode_band", "ncb", "channel"],
)
model.fit(df)
ate = model.average_treatment_effect()
print(ate)
```

Output:

```
Average Treatment Effect
  Treatment: pct_price_change
  Outcome:   renewal
  Estimate:  -0.0231
  Std Error: 0.0018
  95% CI:    (-0.0266, -0.0196)
  p-value:   0.0000
  N:         120,000
```

The GLM alternative — a logistic regression or Poisson with log-premium as a covariate alongside the rating factors — typically yields something like $-0.041$ on the same data. The difference is the confounding. The true elasticity in this data-generating process is $-0.023$. DML is right; the GLM is wrong by a factor of nearly two.

### Why is the GLM so wrong?

In a formula-rated UK motor book, the premium is approximately $D \approx \hat{\mu}(X) + \varepsilon$ where $\hat{\mu}(X)$ is the technical price and $\varepsilon$ is an exogenous noise component (market adjustments, commercial loadings, pricing errors). The exogenous component $\varepsilon = \tilde{D}$ is what DML recovers. In a tightly managed book with low pricing error, $\text{Var}(\varepsilon) / \text{Var}(D)$ can be as low as 5–15%. The GLM sees the full correlation between $D$ and $Y$; DML isolates the causal fraction.

The `ElasticityDiagnostics` class in `insurance-causal` measures this before you fit:

```python
from insurance_causal.elasticity.diagnostics import ElasticityDiagnostics

diag = ElasticityDiagnostics()
report = diag.treatment_variation_report(
    df,
    treatment="log_price_change",
    confounders=["age", "ncd_years", "vehicle_group", "region", "channel"],
)
# treatment_r2: 0.91 — 91% of price variation is explained by rating factors.
# exogenous_fraction: 0.09 — only 9% is genuinely exogenous.
```

An exogenous fraction below 10% is a warning sign. The confidence intervals will be wide and the point estimate sensitive to nuisance model choice. You need more pricing noise (e.g. from a historical experiment window) or an instrumental variable.

---

## Insurance application 2: telematics treatment effect on claim frequency

Telematics enrolment is a binary treatment: customer consents to black-box or app-based monitoring, $D_i \in \{0, 1\}$. The outcome $Y_i$ is claim count in the policy year. The confounders $X_i$ are the rating factors plus any pre-enrolment driving indicators.

The naive approach — compare mean claim frequency between telematics and non-telematics customers — is severely confounded. Telematics customers self-select: younger drivers with fewer miles driven and lower pre-registration risk scores disproportionately enrol. A simple comparison attributes the selection effect to the telematics programme.

DML handles binary treatments via the interactive regression model (IRM), which does not impose the partially linear structure:

$$Y_i = g_0(D_i, X_i) + \varepsilon_i, \quad D_i = \mathbb{1}[m_0(X_i) \geq U_i]$$

The nuisance models are now $g_0(D, X) = \mathbb{E}[Y \mid D, X]$ and a propensity score $m_0(X) = \mathbb{P}(D = 1 \mid X)$. The ATE is identified as:

$$\theta_0 = \mathbb{E}\left[\frac{D_i(Y_i - g_0(0, X_i))}{m_0(X_i)} - \frac{(1-D_i)(Y_i - g_0(1, X_i))}{1 - m_0(X_i)}\right]$$

This is the augmented IPW (AIPW) estimator, doubly robust: consistent if either $g_0$ or $m_0$ is correctly specified.

```python
from insurance_causal import CausalPricingModel
from insurance_causal.treatments import BinaryTreatment

model = CausalPricingModel(
    outcome="claim_count",
    outcome_type="poisson",
    treatment=BinaryTreatment(column="telematics_enrolled"),
    confounders=[
        "age", "vehicle_age", "annual_mileage_band",
        "ncb_years", "postcode_risk", "channel",
    ],
)
model.fit(df, exposure_col="policy_years")
ate = model.average_treatment_effect()
print(ate)
```

For heterogeneous effects — does telematics help young urban drivers more than older rural ones? — use the causal forest:

```python
from insurance_causal.causal_forest import HeterogeneousElasticityEstimator

est = HeterogeneousElasticityEstimator(n_estimators=300)
est.fit(
    df,
    confounders=["age", "ncd_years", "vehicle_age", "annual_mileage_band"],
)
cates = est.cate(df)
# cates is a Series of per-customer estimated treatment effects.
# Negative CATE: telematics reduces claim frequency for this customer.
```

The `HeterogeneousInference` class provides BLP and GATES tests (Chernozhukov et al., 2020) for whether the heterogeneity is statistically real, not just model noise.

---

## Continuous treatments: the Riesz representer approach

For continuous treatments — the actual premium charged, not a percentage change — the standard DML partially linear model assumes a constant treatment effect $\theta$. That is almost certainly wrong: a £50 premium increase on a £300 policy has a different effect than the same £50 on a £800 policy. And the IRM requires a binary treatment.

The `autodml` subpackage implements the average marginal effect (AME) estimator via Riesz representers (Chernozhukov et al., 2022, Econometrica). This avoids modelling the generalised propensity score $p(D \mid X)$ — which is ill-posed for continuous treatments with near-deterministic pricing formulas — by directly learning the reweighting functional from data.

The estimand is:

$$\theta_0 = \mathbb{E}\left[\frac{\partial \mathbb{E}[Y \mid D, X]}{\partial D}\right]$$

the population-average of the local treatment effect, integrated over the data distribution. For a renewal model, this is the average change in lapse probability per £1 increase in premium.

```python
from insurance_causal.autodml import PremiumElasticity

model = PremiumElasticity(
    outcome_family="poisson",
    n_folds=5,
    nuisance_backend="catboost",
    riesz_type="forest",
    inference="eif",
    ci_level=0.95,
    random_state=42,
)
model.fit(X, D, Y, exposure=policy_years)
result = model.estimate()
print(result)
# AME estimate: -0.0023 (95% CI: -0.0028, -0.0018)
# Interpretation: +£1 in premium reduces claim rate by 0.0023 per policy year.
```

For segment-level effects — does the AME differ between high-value and standard policies?

```python
segment_results = model.effect_by_segment(segments=df["customer_segment"])
for seg in segment_results:
    print(f"{seg.segment_name}: {seg.result.estimate:.4f} "
          f"(95% CI: {seg.result.ci_low:.4f}, {seg.result.ci_high:.4f})")
```

This is valid because the efficient influence function scores decompose additively over subgroups — no refitting is required, and the inference is still asymptotically correct.

### Renewal selection bias

A complication specific to UK personal lines: you observe claim outcomes only for customers who renew. Non-renewers lapse and their claim experience is unobserved. Since premium increases cause lapse, the observed renewers at high premium levels are a *selected* sample — lower-risk customers who stayed despite the increase. Naively fitting on renewers understates the causal effect because the selected sample is systematically less risky.

`SelectionCorrectedElasticity` handles this by jointly modelling the renewal propensity $\pi(X, D) = \mathbb{P}(S = 1 \mid X, D)$ and incorporating IPW correction into the EIF score:

```python
from insurance_causal.autodml import SelectionCorrectedElasticity

model = SelectionCorrectedElasticity(
    outcome_family="poisson",
    n_folds=5,
    inference="eif",
    random_state=42,
)
model.fit(X, D, Y, S=renewal_indicator, exposure=policy_years)
result = model.estimate()

# Sensitivity analysis: how robust is the estimate to unobserved confounding?
bounds = model.sensitivity_bounds(gamma_grid=[1.0, 1.5, 2.0, 3.0])
# gamma=1: point-identified result
# gamma=2: worst-case bounds if unobserved confounders could double the selection odds
```

The sensitivity bounds follow the Manski–Rosenbaum approach. If your estimate remains negative and economically meaningful across $\Gamma = 1$ to $\Gamma = 2$, you have a robust finding. If the bounds widen to include zero at $\Gamma = 1.5$, you have a marginal result that requires transparency in any pricing review or Solvency II internal model validation.

---

## What goes wrong without DML

It is worth being concrete about the failure modes of naive approaches.

**Naive GLM.** Fitting a Poisson or logistic GLM with treatment as a covariate produces a biased coefficient if any unobserved confounder jointly affects treatment assignment and outcome. In insurance, this is almost always true: pricing formulas are never complete representations of risk. The bias is first-order and does not decrease as sample size grows. Adding more data to a confounded model gives you a more precise wrong answer.

**Propensity score matching.** Matching on the estimated propensity score $\hat{m}(X)$ removes observed confounding but relies on the unconfoundedness assumption: all relevant confounders are in $X$. In insurance, the scoring model is never the full risk picture. Matching also throws away data (the unmatched majority) and is sensitive to the matching algorithm. DML uses all observations and provides valid inference without the parametric propensity score model.

**Two-stage least squares (IV).** A valid instrument — a variable that affects treatment but not outcome except through treatment — is the gold standard. In renewal pricing, mid-market competitor prices are sometimes used as instruments: competitor price changes shift your relative price without directly affecting claims. But good instruments are scarce and the exclusion restriction is hard to verify. DML does not require an instrument; it requires unconfoundedness, which is weaker.

**Fixed effects panel regression.** If you have panel data (multiple policy years per customer), fixed effects remove time-invariant confounders. But individual fixed effects cannot remove confounders that vary over time with the treatment — a customer's improving driving record that drives both lower premiums and fewer claims is a time-varying confounder. DML with time-varying confounders in $X$ handles this directly.

---

## Nuisance model choice in practice

The Ahrens et al. tutorial (Section 4) is thorough on this and we agree with their main recommendations.

For $n > 50{,}000$ (a standard UK motor or home book): gradient-boosted trees (CatBoost with depth 4–6, L2 regularisation 3–10) work well for both nuisance functions. Trees naturally handle the multiplicative interaction structure of insurance rating factors.

For $n < 10{,}000$ (commercial or specialty lines): regularised linear models (elastic net, LASSO) are more reliable because they impose structure that prevents overfitting at small samples. The DML partially linear model is also more interpretable in thin-data settings where a fully nonparametric nuisance model cannot learn reliable representations.

For the propensity model in binary treatment DML: logistic regression with all pairwise interactions as features is surprisingly competitive with gradient boosting when the propensity is a reasonably smooth function of $X$. The advantage of simple models here is that they are less likely to produce extreme propensity scores that make the AIPW estimator unstable.

The `insurance-causal` defaults — CatBoost nuisance with adaptive regularisation, 5-fold cross-fitting — are calibrated for UK personal lines portfolio sizes. The adaptive regularisation logic adjusts CatBoost depth and L2 penalty based on sample size so that nuisance quality is stable across portfolios ranging from 5,000 to 500,000 policies.

---

## The reference

Ahrens, A., Chernozhukov, V., Hansen, C., Kozbur, D., Schaffer, M. E., & Wiemann, T. (2025). 'An Introduction to Double/Debiased Machine Learning.' arXiv:2504.08324.

The original DML paper: Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). 'Double/Debiased Machine Learning for Treatment and Structural Parameters.' *The Econometrics Journal*, 21(1): C1–C68.

The AutoDML Riesz representer approach used in `PremiumElasticity`: Chernozhukov, V., Newey, W. K., & Singh, R. (2022). 'Automatic Debiased Machine Learning of Causal and Structural Effects.' *Econometrica*, 90(3): 967–1027.

---

## Installation and source

```bash
pip install "insurance-causal[all]"
```

```python
from insurance_causal import CausalPricingModel
from insurance_causal.autodml import PremiumElasticity, SelectionCorrectedElasticity
from insurance_causal.causal_forest import HeterogeneousElasticityEstimator
```

Source: [github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal) — MIT licence.

---

## Related posts

- [OLS Elasticity in a Formula-Rated Book Measures the Wrong Thing](/libraries/pricing/causal-inference/2026/03/14/causal-price-elasticity-for-uk-renewal-pricing/) — the foundational post on why GLM elasticity is biased in UK motor renewal
- [Heterogeneous Price Elasticity via Causal Forests](/techniques/causal-inference/2026/03/25/heterogeneous-price-elasticity-causal-forests-insurance-pricing/) — per-customer CATE estimation and the ENBP optimiser
- [Does DML Causal Inference Actually Work on Insurance Data?](/techniques/causal-inference/2026/03/25/does-dml-causal-inference-actually-work/) — our benchmark results on synthetic and FrEMTPL2 data
