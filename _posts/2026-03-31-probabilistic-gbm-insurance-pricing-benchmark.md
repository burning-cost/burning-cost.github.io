---
layout: post
title: "Probabilistic GBM for Insurance Pricing: What the Chevalier & Côté Benchmark Actually Tells Us"
date: 2026-03-31
categories: [pricing, machine-learning]
tags: [distributional-GBM, XGBoostLSS, LightGBMLSS, NGBoost, PGBM, CatBoost, insurance-distributional, Chevalier-Cote-2025, EAJ-2025, GAMLSS, ZI-Tweedie, Tweedie, frequency-severity, calibration, CRPS, Solvency-II, reinsurance, per-risk-volatility, UK-personal-lines, FreMTPL, BelgianMTPL]
description: "Chevalier & Côté (EAJ 2025) benchmarked 11 probabilistic GBM methods on 5 insurance datasets. The headline: XGBoostLSS wins on confidence interval coverage; LightGBM is fastest; NGBoost and cyc-GBM lose on 3 of 4 metrics. We explain what the results mean for UK pricing teams, where each method falls short on insurance-specific distributions, and what insurance-distributional fills in."
math: true
author: burning-cost
---

Most pricing models produce a point estimate. Expected claims frequency. Expected severity. Expected pure premium. The model trains to minimise a deviance or RMSE, and the single number that comes out is what gets loaded and filed. The distribution of outcomes around that point — how fat the tail is, what the 95th percentile looks like, what volatility the insurer is accepting per risk — is handled separately, if at all, via capital models that typically use aggregate assumptions rather than risk-level distributions.

Probabilistic gradient boosting changes this. Instead of training a tree ensemble to predict $\mathbb{E}[Y \mid X]$, you train it to estimate the full conditional distribution $P(Y \mid X)$. The output per risk is not a single number but a fitted distribution — a Gamma, a negative binomial, a Tweedie — parameterised by covariates. From that distribution you can read off the mean, the variance, the 99th percentile, the coefficient of variation, or the expected cost of a per-risk excess-of-loss layer. One model, many actuarial outputs.

The field has been noisy: multiple competing libraries, inconsistent benchmarks, and methods that look similar but are technically quite different. Chevalier & Côté's benchmark, published in the European Actuarial Journal in August 2025 (arXiv:2412.14916, doi:10.1007/s13385-025-00428-5), is the most rigorous actuarial comparison to date. It runs 11 algorithms across 5 insurance datasets, covering computational efficiency, predictive accuracy, calibration, and portfolio balance. The results cut through most of the noise.

Below we explain what that benchmark found, where each method fits in the taxonomy, what the distribution support gaps are for UK pricing work, and when it is actually worth reaching for a distributional model instead of a plain LightGBM.

---

## Three approaches that are not the same thing

Before the library names, the taxonomy. There are three fundamentally different ways to get a conditional distribution out of a gradient boosted tree, and mixing them up leads to wrong assumptions about what you can and cannot do with each method.

**Approach A: Multi-parameter distributional boosting (GBMLSS)**

Parameterise the target distribution explicitly — for a Gamma model, you estimate both $\mu$ (the mean parameter) and $\phi$ (the dispersion parameter) as functions of $X$. Grow separate tree ensembles for each parameter. At each boosting step, compute gradients and Hessians of the negative log-likelihood via autodiff (in practice, via PyTorch autograd), and fit one tree per parameter per round. XGBoostLSS and LightGBMLSS both use this approach.

The key property: you are directly maximising

$$\mathcal{L}(\theta_1(X), \ldots, \theta_K(X)) = \sum_i \log p(y_i \mid \theta_1(x_i), \ldots, \theta_K(x_i))$$

where $K$ is the number of distributional parameters. For a Gamma this is 2. For a ZI-Poisson this is also 2 (the rate $\lambda$ and the zero-inflation probability $\pi$). Computational cost scales with $K$ but is manageable.

**Approach B: Natural gradient boosting (NGBoost)**

Use the Fisher information matrix $\mathcal{F}$ to transform ordinary gradients into natural gradients: $\tilde{g} = \mathcal{F}^{-1} g$. Natural gradients are parameterisation-invariant — the update direction does not depend on whether you model the Gamma dispersion as $\phi$ or $\log \phi$. This is theoretically cleaner than GBMLSS. In practice, the base learners are sklearn decision trees, not a native GBDT engine, which is the main source of NGBoost's computational disadvantage.

**Approach C: Variance approximation (PGBM)**

Train a single model. Estimate mean and variance from variability of leaf prediction values across trees. Post-hoc fit a distribution to those two moments. Fast, but indirect: you are fitting a distribution to estimated moments, not maximising the target distribution's likelihood. Fine for generic uncertainty quantification; inadequate for distributions where moment parameterisation is insufficient — you cannot separately estimate the zero-inflation probability $\pi$ from the aggregate mean and variance.

The distinction between A and C matters for zero-inflated models. GBMLSS can fit a ZI-Poisson by growing one tree for $\lambda$ and one tree for $\pi$. PGBM cannot — it only has (mean, variance).

---

## What Chevalier & Côté found

The benchmark covers: XGBoost, LightGBM, CatBoost (point predictors), XGBoostLSS, NGBoost, PGBM, cyc-GBM (distributional methods), GLM and GAMLSS (traditional baselines), plus EGBM (InterpretML's Explainable Boosting Machine). Five datasets including BelgianMTPL and FreMTPL variants with claim frequency and severity targets.

**On predictive accuracy:** All algorithms achieve similar Poisson deviance on homogeneous datasets. The differences are small and mostly inside noise. This is the honest headline: for the basic frequency prediction task, you are not going to beat a well-tuned LightGBM on deviance by using a distributional method. The distributional approach earns its keep elsewhere.

**On computational efficiency:** LightGBM is the fastest algorithm overall, with essentially no accuracy penalty for using it. Among probabilistic methods, XGBoostLSS is fastest. NGBoost and PGBM are materially slower — NGBoost because of its sklearn tree base learners, PGBM because of its stochastic tree update equations. cyc-GBM ranges from slow to very slow. For a UK personal lines book of 300,000+ policies with quarterly repricing cycles, training time matters. LightGBMLSS (the LightGBM-backed LSS variant) is the right default if speed is a constraint.

**On calibration and confidence interval coverage:** This is where probabilistic models justify their existence. Point models produce coverage intervals that are "too narrow" for severity data — the intervals are miscalibrated, covering fewer observations than their stated confidence level. XGBoostLSS, with its shrinkage and DART settings, gives "adequate fit in terms of coverage of confidence intervals" on both 75% and 95% intervals. NGBoost also gives "good improvement" on lognormal severity coverage alongside XGBoostLSS.

**On portfolio balance:** cyc-GBM is explicitly flagged for portfolio balance issues — its lift curves show miscalibration even when the aggregate deviance looks acceptable. This is a failure mode that deviance-based model selection misses entirely. It matters for pricing because a model can minimise training deviance while systematically overpricing low-risk and underpricing high-risk, which is exactly the Consumer Duty and adverse selection problem you do not want.

**The cyc-GBM and NGBoost verdict:** Both "lose in 3 of 4 tables" for average rank. cyc-GBM is additionally abandoned — last GitHub commit July 2023, no PyPI package, Gaussian distribution only. There is no production argument for either.

**On high-cardinality categoricals:** CatBoost gives "a small improvement when the dataset contains many high-cardinality categorical variables." For UK personal lines, this is nearly every pricing dataset: vehicle make/model (~2,000 values), occupation (100–400 insurer-specific codes), postcode sector (~9,000 sectors). CatBoost's ordered target statistics handle these without requiring manual target encoding with holdout. The improvement is small on clean benchmark data; it may be larger on real UK books where encoding artefacts accumulate across multiple high-cardinality fields.

**On EGBM:** InterpretML's Explainable Boosting Machine — a generalised additive model trained via boosted tree stumps — achieves "fully interpretable, competitive performance, no trade-off." We think this result deserves more attention from UK pricing teams than it currently gets, but that is a separate post.

---

## The distribution support problem

The benchmark finding ("calibration improves, accuracy roughly equal") is only useful if the libraries actually support the distributions you need. They largely do not.

Here is the current state as of March 2026:

| Method | Poisson | NegBin | Gamma | Tweedie | ZI-Poisson | ZA-Gamma | Exposure offset |
|--------|---------|--------|-------|---------|------------|----------|----------------|
| XGBoostLSS v0.6.1 | Y | Y | Y | **No** | Y | Y | Unclear |
| LightGBMLSS v0.6.x | Y | Y | Y | **No** | Y | Y | Unclear |
| NGBoost v0.5.10 | Y | **No** | **No** | **No** | **No** | **No** | **No** |
| PGBM v2.3.0 | indirect | indirect | indirect | indirect | **No** | **No** | Via LGBM |
| cyc-GBM | **No** | **No** | **No** | **No** | **No** | **No** | **No** |
| insurance-distributional | **No** | Y | Y | Y | Y | **No** | **Yes** (`exposure=`) |

The most important gap: neither XGBoostLSS nor LightGBMLSS support Tweedie. The compound Poisson-Gamma family — the standard combined frequency-severity distribution for pure premium modelling — does not fit cleanly into the GBMLSS framework, because Tweedie is characterised by a power parameter $p \in (1,2)$ that controls the frequency-severity mix, not a set of separately estimated moment parameters.

NGBoost's coverage is thin throughout. No Gamma. No negative binomial. No zero-inflated variants. NGBoost is well-maintained (v0.5.10 released March 2026) but its distribution library has not kept pace with what insurance pricing requires.

The exposure offset issue is underappreciated. A frequency model without proper exposure handling is not a frequency model — it is a count model that happens to be fitted on insurance data. Standard actuarial practice is `log(exposure)` as an offset: a six-month policy has half the expected claims of a twelve-month policy at the same risk. GLMs handle this via the offset parameter. How GBM frameworks handle it varies:

- Chevalier & Côté describe exposure being handled "as part of the initial score/baseline prediction" — `log(exposure)` set as `base_score` in XGBoost/LightGBM. This is a workaround, not a first-class offset.
- `insurance-distributional` takes `exposure=` as an explicit argument to `fit()`. This is the right API.
- NGBoost does not support offset at all.

---

## Where insurance-distributional fits

[insurance-distributional](https://github.com/burning-cost/insurance-distributional) is our CatBoost-native distributional GBM library. It covers the frequency-severity distributions that UK pricing teams actually use: Tweedie, Gamma, zero-inflated Poisson, negative binomial. The `exposure=` argument to `fit()` handles offset correctly.

```python
from insurance_distributional import TweedieGBM

model = TweedieGBM(iterations=500, learning_rate=0.05)
model.fit(X_train, y_train, exposure=exposure_train)

# Returns conditional distribution per risk
dist = model.predict_distribution(X_val)

# Point predictions
print(dist.mean())

# Per-risk coefficient of variation — useful for reinsurance loading
print(dist.std() / dist.mean())

# 99th percentile per risk — relevant for SII capital allocation
print(dist.ppf(0.99))
```

What it does not do: it does not implement ZI-Tweedie. The So & Valdez paper (arXiv:2406.16206, ASTIN Best Paper 2024) introduced zero-inflated Tweedie with CatBoost as the primary reference for insurance-distributional, and the library's cited motivation is directly that work. But the implementation only covers standard Tweedie, Gamma, ZIP, and NegBinomial — not ZI-Tweedie. This is the largest gap in the current open-source distributional GBM ecosystem. No pip-installable library implements So & Valdez's ZI-Tweedie model. ZI-Tweedie is the right distribution for a personal lines motor own damage book where most policies are claim-free and the non-zero claims follow a compound Poisson-Gamma structure.

---

## When a distributional model actually earns its keep

For the vanilla expected pure premium use case — price the mean — a plain LightGBM with Tweedie loss is faster, more widely understood, and broadly as accurate. We are not going to pretend otherwise.

Distributional GBM earns its keep in three specific situations.

**Per-risk volatility for reinsurance pricing.** If you are quoting per-risk XL covers, the layer expected value depends on the full conditional severity distribution, not just the mean. A risk with expected annual loss of £2,000 and high severity volatility loads the £5,000 xs £2,000 layer differently from a risk with the same expected loss but low volatility. The distributional model gives you per-risk CoV; the standard GBM does not. The calculation is:

$$\mathbb{E}[\min(S, L+D) - D]_+ = \int_D^{L+D} S(x)\, dx$$

where $S(x) = P(Y > x)$ is the survival function from the fitted conditional distribution. This is a closed-form integral for Gamma and log-normal; you can compute it per risk in a vectorised loop.

**Reserve ranges at IFRS 17 and Solvency II.** IFRS 17 requires a risk adjustment above the best-estimate liability. The approved methods include confidence intervals and CoV-based loadings. A distributional GBM fitted to your claims dataset gives you, per policy cohort, the conditional distributional parameters from which you can derive statistically grounded CoV estimates. This is more defensible than a global percentage applied to IBNR. The Solvency II SCR calculation is a 99.5th percentile VaR — per-risk distributional parameters can feed a portfolio simulation to produce a bottom-up capital estimate.

**Detecting systematic severity miscalibration.** A point model can fit the mean while leaving the dispersion entirely unconstrained — and in practice the standard Tweedie GBM does exactly this. The dispersion parameter is estimated from residuals after training, not modelled as a function of covariates. If higher-risk segments genuinely have higher severity variance (not just higher severity mean), a model that treats dispersion as constant will produce miscalibrated confidence intervals — "too narrow" intervals for the segments where it matters most, which is precisely the Chevalier & Côté finding. A distributional model fits $\phi(X)$ explicitly and catches this.

For routine frequency or pure premium modelling with no reinsurance or capital modelling application, keep LightGBM with Tweedie or Poisson loss. It is faster, the practitioner pool is larger, and the regulatory precedent is cleaner. Deploy distributional GBM where the full distribution is the output you actually need.

---

## Practical guidance for UK pricing teams

**If you need a distributional model today and your primary loss distribution is Gamma or log-normal severity:** XGBoostLSS (`pip install xgboostlss`, v0.6.1, Python >=3.11) is the best option. Best calibrated CI coverage in the Chevalier & Côté benchmark. Slowest installation (PyTorch, Pyro, Optuna as dependencies) but the training speed is fine for most insurance datasets. SHAP works on each distributional parameter's tree ensemble separately — you can explain why a risk has high predicted dispersion, not just a high predicted mean.

**If training speed matters:** LightGBMLSS (`pip install lightgbmlss`, v0.6.x). Same distribution family as XGBoostLSS, LightGBM backend, materially faster. The Chevalier & Côté benchmark used LightGBM as a point predictor (fastest overall); LightGBMLSS inherits that speed advantage for the distributional case.

**If your data has many high-cardinality categoricals and your target distribution is Tweedie, Gamma, ZIP, or NegBinomial:** insurance-distributional with CatBoost. The exposure offset API is correct; the CatBoost backend handles vehicle make/model, occupation, and postcode properly; Tweedie is supported (unlike in XGBoostLSS/LightGBMLSS).

**If you need ZI-Tweedie:** There is no production library for this yet. The So & Valdez (2024) paper and the So & Deng NAAJ extension (doi:10.1080/10920277.2025.2454460) describe the method in full; Scenario 2 (single tree ensemble where $q = 1/(1+\mu^\gamma)$, with $\gamma$ a scalar) is the simpler implementation and the better starting point. This is the next planned addition to insurance-distributional.

**If someone on your team is pushing NGBoost:** Ask them which insurance distributions they intend to use. If the answer is anything other than Poisson or Normal, there is no support for it. The natural gradient property is theoretically appealing but has not translated into benchmark wins for insurance data. XGBoostLSS with standard gradients beats it on both speed and CI coverage.

---

## Calibration is not optional

We will close on the point the Chevalier & Côté benchmark makes most clearly, even if it is easy to bury in 25 tables.

Calibration — whether your 90% prediction interval covers 90% of actuals — is a different axis from deviance. A model can minimise Poisson deviance on a test set while producing intervals that are systematically too narrow. For severity prediction on insurance data, point models do this routinely. The correct validation workflow for a distributional model is not deviance alone; it is:

- **PIT histogram**: probability integral transform. If the model's distributional predictions are correct, $F(y_i \mid x_i)$ should be uniform on $[0,1]$. Deviations from uniform indicate where the distribution is miscalibrated — usually the tails.
- **Coverage curves**: for each confidence level $\alpha \in \{0.50, 0.75, 0.90, 0.95\}$, what fraction of observations fall inside the $\alpha$-prediction interval? Plot empirical coverage against nominal coverage. A well-calibrated model sits on the 45-degree line.
- **CRPS**: the Continuous Ranked Probability Score is a proper scoring rule for full distributional forecasts. Proper scoring rules cannot be gamed — a model that reports false distributions to improve its CRPS score will score worse than the true model. Use CRPS for model selection when choosing between distributional families (does Gamma or log-normal fit this severity data better?).

None of these diagnostics are exotic. The `insurance-distributional` library includes PIT and coverage utilities. Running them takes less time than the model training. If you fit a distributional model and only report deviance, you have missed the point of fitting a distribution.

---

## References

- Chevalier, D. & Côté, M.-P. (2025). From Point to Probabilistic Gradient Boosting for Claim Frequency and Severity Prediction. *European Actuarial Journal*. arXiv:2412.14916, doi:10.1007/s13385-025-00428-5.
- März, A. (2019–2024). XGBoostLSS — An extension of XGBoost to probabilistic forecasting. arXiv:1907.03178.
- Duan, T. et al. (2020). NGBoost: Natural Gradient Boosting for Probabilistic Prediction. *ICML 2020*. arXiv:1910.03225.
- Sprangers, O., Schelter, S. & de Rijke, M. (2021). Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression. *KDD 2021*. arXiv:2106.01682.
- So, B. & Valdez, E.A. (2024/2025). Zero-inflated Tweedie gradient boosting for extremely imbalanced zero-inflated insurance data. *Applied Soft Computing*. arXiv:2406.16206.
- So, B. & Deng, M. (2025). *North American Actuarial Journal*, Vol. 29, No. 4. doi:10.1080/10920277.2025.2454460.
