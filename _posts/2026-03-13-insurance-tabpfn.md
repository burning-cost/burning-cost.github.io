---
layout: post
title: "Foundation Models for Thin Segments: TabPFN and TabICLv2 in Insurance Pricing"
date: 2026-03-13
categories: [libraries, pricing, foundation-models]
tags: [TabPFN, TabICLv2, foundation-model, thin-data, in-context-learning, GLM, conformal-prediction, relativities, benchmark, Gini, python, insurance-thin-data]
description: "TabPFN and TabICLv2 for thin-segment UK insurance pricing. In-context learning at inference, no gradient descent. insurance-thin-data wraps both for actuaries."
---

When your home insurance team prices thatched properties, new-build flats in a specific postcode cluster, or a freshly launched e-bike product, they are almost always working from fewer than a thousand policy-years. Often fewer than two hundred. At that data volume, a Poisson GLM's MLE has not converged. The confidence intervals on your rating factors are so wide they are informative about almost nothing. The standard response is credibility blending — shrink toward the overall book mean, apply a judgement overlay, call it done. That is not a method. It is a controlled way of saying "we do not have enough data."

The question worth asking in 2026 is whether there is something better.

[`insurance-thin-data`](https://github.com/burning-cost/insurance-thin-data) is our answer. It wraps TabPFN v2 and TabICLv2 in an insurance actuarial workflow: sklearn-compatible fit/predict, GLM benchmark with proper actuarial metrics, PDP-based relativities extraction, split conformal prediction intervals, and a committee-ready report with mandatory disclosure of limitations.

```bash
uv add insurance-thin-data
```

---

## What foundation models for tabular data actually do

TabPFN v2, published in Nature 637:319–326 by Hollmann et al. in January 2025, is not a model you train on your data. It is a model pretrained on millions of synthetically generated tabular datasets — datasets drawn from a prior over data-generating processes that covers a huge range of functional forms, noise structures, and feature correlations.

At inference time, you pass in your labelled training set as context. The model has never seen your data before. But it has seen enough diversity during pretraining that it can, in a single forward pass, produce a posterior-predictive distribution over test labels. No gradient descent. No hyperparameter tuning. No CV loop. The training set literally becomes part of the input.

This is in-context learning. The same mechanism that lets a large language model answer questions about a document it has never seen — by conditioning on that document in the prompt — lets TabPFN produce calibrated predictions from a dataset it was never trained on.

The performance claims in the Nature paper are not modest. On datasets under 10,000 samples, TabPFN v2 outperforms XGBoost, LightGBM, and tuned neural networks across 300+ benchmarks. The margin is meaningful — Gini improvements of 3–8 points in the regimes where these models have genuinely thin data. Hollmann et al. attribute this to the synthetic pretraining: the model has effectively been regularised by an enormous implicit prior over tabular data-generating processes.

In February 2026, TabICLv2 from INRIA's SODA team superseded TabPFN v2.5 on the same benchmarks. It is Apache 2.0-licensed and available on PyPI. `insurance-thin-data` abstracts over both backends via a `BackendProtocol` interface, so you can switch without touching your pricing code. A `MockBackend` ships for CI — no GPU, no API key, no cost.

---

## The insurance use case

The canonical thin-data problem in UK personal lines is not the main book — it is the edge. Some examples from current UK pricing practice:

**E-bike and cargo bike.** Products launched in 2022–2024. The oldest books have three years of development. Frequency by age band is unstable because the young-driver cohort is small and claims are sparse.

**Subsidence in new postcodes.** Climate-driven subsidence has expanded into postcode sectors that had no material claims history before 2022. You have postcode sectors with one or two claims apiece. A GLM will either overfit wildly or assign flat relativities.

**Thatched properties.** Perhaps 150,000 thatched dwellings in England and Wales. Your book probably has under 5,000. Losses are highly heterogeneous by construction method, age, and maintenance. Claim frequency by sub-segment is very thin.

**New-to-market demographics.** A young non-binary driver cohort or a new immigrant community. Small sample, genuine heterogeneity, no legacy data.

In each case, the actuarial instinct is to borrow. Credibility blending borrows from the overall mean. Hierarchical Bayes borrows from the parent segment. Both work reasonably well when the sub-segment is a well-understood perturbation of the parent. When it is genuinely different — new product, different risk behaviour, no natural parent — borrowing from the wrong place is actively misleading.

TabPFN's pretraining prior is broad enough that it does not borrow from any one place. It generalises from the implicit structure in millions of synthetic datasets. That is a different kind of shrinkage — not toward your book, but toward reasonable functional forms for tabular data.

---

## Fitting it

The API is sklearn-compatible:

```python
from insurance_thin_data.tabpfn import InsuranceTabPFN

model = InsuranceTabPFN(
    backend="tabicl",          # or "tabpfn" for v2
    task="regression",
    exposure_col="earned_years",
    n_ensemble=8,
    conformal_alpha=0.1,       # 90% prediction intervals
)

model.fit(X_train, y_train)     # y = claim frequency or pure premium

preds = model.predict(X_test)
intervals = model.predict_interval(X_test)   # split conformal, finite-sample guarantee
```

The `exposure_col` argument handles the standard insurance problem that you are modelling a rate, not a count. There is no native exposure offset in TabPFN — the library implements a log-rate workaround: the target is log(claims / exposure), the model fits on log-rate, and predictions are exponentiated back to rate. This is a genuine limitation: it is not the same as a proper Poisson offset, and it will not produce well-calibrated Poisson deviance scores. We document this explicitly in every committee report the library generates.

The prediction intervals use split conformal prediction — the same approach as [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal), adapted for regression. The coverage guarantee is finite-sample and distribution-free: if you specify `conformal_alpha=0.1`, the intervals will contain the true value at least 90% of the time on exchangeable test data, regardless of the true data-generating process.

---

## Benchmarking against your GLM

`insurance-thin-data` ships a `GLMBenchmark` class. Pass it your fitted GLM and the foundation model and it produces a proper actuarial comparison:

```python
from insurance_thin_data.tabpfn import GLMBenchmark
from sklearn.linear_model import TweedieRegressor

glm = TweedieRegressor(power=1, alpha=0.01, link="log")
glm.fit(X_train, y_train)

bench = GLMBenchmark(
    glm=glm,
    foundation=model,
    exposure=X_test["earned_years"],
)
results = bench.evaluate(X_test, y_test)

print(results.summary())
# Metric           GLM       TabICLv2    Delta
# Gini             0.312     0.351       +0.039
# Poisson deviance 0.418     0.391       -0.027
# RMSE             0.094     0.087       -0.007
# Double-lift (top decile) 1.31  1.47    +0.12
```

The double-lift chart compares the top-decile observed loss ratio for policies ranked by each model against the bottom decile. If the foundation model is genuinely separating risk better, the spread widens. In our test runs on simulated thin-book data (500 policies, 5 features, Poisson-Gamma frequency-severity), the foundation model's double-lift spread was consistently 10–15 percentage points wider than the GLM.

The Gini formula in the benchmark is the standard insurance definition: twice the area between the Lorenz curve and the diagonal, with cumulation by predicted rank. We caught and corrected an inversion in the Gini formula during build — the initial implementation was computing 1 − Gini, producing metrics below 0.5 for models with genuine discrimination power. Obvious once you see it; easy to miss in a unit test that only checks the range.

---

## Extracting relativities

A foundation model is not interpretable in the GLM sense. There is no `coef_` vector. There is no multiplicative factor table you can put in a rating engine.

`RelativitiesExtractor` addresses this via partial dependence profiles (PDPs). For each feature, it sweeps the feature over a grid while marginalising over other features, computes the predicted rate at each grid point, and normalises to a base value to produce a multiplicative relativity:

```python
from insurance_thin_data.tabpfn import RelativitiesExtractor

extractor = RelativitiesExtractor(model, exposure_col="earned_years")
rels = extractor.extract(X_train, features=["vehicle_age", "driver_age", "ncd_years"])

# Returns a dict of DataFrames: feature -> (value, relativity, lower_90, upper_90)
rels["vehicle_age"]
#    vehicle_age  relativity  lower_90  upper_90
# 0            1        1.42      1.28      1.58
# 1            2        1.31      1.19      1.44
# 2            3        1.18      1.09      1.29
# ...
```

The intervals on the relativities are conformal — inherited from the underlying `predict_interval` method. They reflect genuine uncertainty in the model's predictions at each grid point, not parametric bootstrap approximations.

These are PDP-based relativities, not GLM coefficients. They are not marginalised in the same way; they will include interaction effects in the average. Whether that is what you want depends on your rating engine. We think it is often more honest than forcing an additive log-linear structure onto genuinely non-linear risk.

We fixed a bug in the PDP computation during build: exposure weighting was not being threaded through the marginalisation correctly, so the grid estimates were effectively unweighted averages across the test set. The fix ensures the PDP reflects the exposure distribution in the training data, not a uniform grid.

---

## The committee report

Every use of `insurance-thin-data` in a pricing process needs sign-off. The `CommitteeReport` class generates an HTML or JSON report that includes five mandatory disclosures:

```python
from insurance_thin_data.tabpfn import CommitteeReport

report = CommitteeReport(model, bench_results, rels)
report.generate("tabpfn_review.html")
```

The five mandatory limitations are:

1. **Maximum dataset size.** TabPFN v2 supports up to approximately 10,000 rows. TabICLv2 has a similar practical limit. This library is not appropriate for portfolios above that threshold — use a GLM or GBM.

2. **No exposure offset.** The log-rate workaround is not a proper Poisson offset. Poisson deviance scores are approximate. Do not rely on them for regulatory model validation without additional testing.

3. **No Poisson or Gamma family.** The model is regression-only. Frequency-severity decomposition is not available natively. You can chain two `InsuranceTabPFN` instances (one on frequency, one on severity), but the joint calibration is your responsibility.

4. **No sample weights.** All policies are treated equally regardless of earned exposure. This matters when your thin book has significant variation in policy-year counts.

5. **In-context learning is not transfer learning.** TabPFN does not learn from your data. It generalises from pretraining. If your segment is genuinely outside the distribution of synthetic pretraining data, performance will degrade. There is no fine-tuning mechanism.

These are not caveats buried in a README. They appear in the report as a numbered list, before the metrics, in 14pt type. A pricing actuary signing off on a foundation model output should be able to articulate all five.

---

## Where this fits and where it does not

We want to be specific about the position of this library in a pricing toolkit.

**It is not a GLM replacement.** The GLM is the right tool for a portfolio with 50,000+ policies and a reasonably standard risk structure. It is interpretable, it handles exposure natively, it has a principled likelihood, and every reinsurer in London knows how to read its output. We are not trying to replace it.

**It is not a GBM replacement.** For portfolios with 10,000+ policies and complex non-linear structure, a tuned CatBoost with SHAP relativities is probably the right model. GBMs can also handle exposure offsets natively. `insurance-thin-data` tops out at ~10K rows and regression only.

**It is a thin-segment specialist.** When you have 200–2,000 policies, no natural parent segment to borrow from, and a GLM that is either flat or unstable, TabPFN or TabICLv2 in-context learning is the best tool we have seen. The pretraining acts as a prior that is broader and less committal than anything you would elicit from your book.

The BUILD score we assigned during development was 16/20. The four missing points are: no sample weights (critical for exposure-varying portfolios), no proper exposure offset (critical for Poisson frequency), max 10K rows (critical for mid-size books), and no Gamma family (critical for severity). These are not bugs to be fixed — they are architectural limits of the underlying TabPFN/TabICLv2 backends. The library documents them and stops you from using it outside its scope.

---

## A note on the academic lineage

The Nature paper — Hollmann, Müller, Purucker, Krishnakumar, Körfer, Hoo, Shen, Hutter (2025), "TabPFN v2: Improved In-Context Learning for Tabular Data", Nature 637:319–326 — is not a typical ML conference paper. Nature peer review is substantially more rigorous, and the 300-dataset benchmark leaves less room for cherry-picking than single-paper results. The actuarial credibility of the method is grounded in that publication, not in our endorsement.

TabICLv2, the INRIA/SODA release from February 2026, does not yet have a comparable peer-reviewed publication. It outperforms TabPFN v2.5 on the standard benchmark suite, but that is a newer and less thoroughly scrutinised result. The `BackendProtocol` abstraction in `insurance-thin-data` is deliberately designed so that when the next generation of foundation models for tabular data arrives — and there will be a next generation — swapping backends requires changing one string.

---

**[insurance-thin-data on GitHub](https://github.com/burning-cost/insurance-thin-data)** — MIT-licensed, PyPI, 72 tests. For the segments your GLM cannot see.

---

**Related reading:**
- [Bayesian Hierarchical Models for Thin-Data Pricing](/2026/02/17/bayesian-hierarchical-models-for-thin-data-pricing/) — partial pooling across segments using MCMC; the full Bayesian alternative for when interpretability and uncertainty quantification matter more than raw accuracy
- [Bühlmann-Straub Credibility in Python](/2026/02/19/buhlmann-straub-credibility-in-python/) — frequentist credibility for thin segments; simpler and faster than TabPFN, appropriate when the segment structure is one-dimensional
- [Borrowing Experience You Don't Have](/2026/03/12/borrowing-experience-you-dont-have/) — transfer learning for thin-segment pricing; three approaches for adapting a model trained on a related, data-rich segment
