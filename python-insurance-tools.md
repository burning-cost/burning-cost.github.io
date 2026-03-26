---
layout: page
title: "Python Packages for Insurance Pricing — A Practitioner's Guide by Task"
description: "What Python package solves this pricing problem? A task-framed reference covering GLM/GBM modelling, conformal prediction, causal inference, fairness auditing, rate optimisation, telematics, and more. Burning Cost libraries and honest alternatives."
permalink: /python-insurance-tools/
---

This page exists because most Python package lists are organised by library, not by problem. If you know what tool you have, that is useful. If you have a pricing problem to solve, it is not.

What follows is organised around tasks — the things a pricing actuary or data scientist actually searches for. For each task we list the packages that address it, including third-party alternatives where they are relevant or better. We are honest about where gaps exist, including cases where R is still the practical answer.

A note on scope: we focus on the technical pricing workflow — model building, validation, uncertainty quantification, causal inference, fairness, and rate optimisation. We do not cover reserving (see [chainladder-python](https://github.com/casact/chainladder-python)) or capital modelling.

---

## Contents

- [Data Preparation & Feature Engineering](#data-preparation--feature-engineering)
- [Frequency-Severity Modelling](#frequency-severity-modelling)
- [Interpretability & Factor Extraction](#interpretability--factor-extraction)
- [Uncertainty Quantification](#uncertainty-quantification)
- [Causal Inference](#causal-inference)
- [Fairness & Discrimination](#fairness--discrimination)
- [Model Monitoring & Governance](#model-monitoring--governance)
- [Rate Optimisation](#rate-optimisation)
- [Smoothing & Spatial](#smoothing--spatial)
- [Telematics](#telematics)
- [Reserving & Severity](#reserving--severity)

---

## Data Preparation & Feature Engineering

### One-way analysis / exposure-weighted profiling

Raw one-way tables — observed frequency or loss ratio by factor level, weighted by exposure — are the first diagnostic every pricing actuary runs. There is no dedicated package for this because **pandas** handles it well: `groupby` + `agg` with exposure as a weight. The missing piece is visual output and significance testing.

**packages:** pandas (standard), [insurance-datasets](https://github.com/burning-cost/insurance-datasets)

We use `insurance-datasets` to provide a synthetic UK motor portfolio with known DGP parameters. It is primarily useful for testing whether a method recovers the true relativities before applying it to real data — a use case that has no equivalent in general-purpose packages. The `load_motor()` and `load_home()` functions return pandas DataFrames with exposure, claim counts, and claim amounts; Polars output is available via `polars=True`.

```python
from insurance_datasets import load_motor
df = load_motor(n_policies=50_000, seed=42)
one_way = (df.groupby("driver_age_band")
             .agg(exposure=("exposure","sum"),
                  claim_count=("claim_count","sum"))
             .assign(frequency=lambda x: x.claim_count / x.exposure))
```

---

### Temporal train/test splitting

Standard k-fold cross-validation applied to insurance data produces optimistic Poisson deviance estimates because IBNR claims from the most recent months appear in both training and validation sets. The fix is walk-forward CV with an explicit IBNR buffer.

**packages:** [insurance-cv](https://github.com/burning-cost/insurance-cv), scikit-learn `TimeSeriesSplit` (partial fix only)

scikit-learn's `TimeSeriesSplit` handles the ordering correctly but does not understand exposure windows or IBNR buffers — it will still include partially-developed claims in the calibration set unless you manually trim dates. Our `insurance-cv` builds the buffer into the splitter. In our benchmarks, walk-forward CV catches 10.5% Poisson deviance optimism that k-fold hides on synthetic UK motor data.

```python
from insurance_cv import InsuranceWalkForwardCV
cv = InsuranceWalkForwardCV(n_splits=5, ibnr_buffer_months=6)
for train_idx, val_idx in cv.split(df, date_col="inception_date"):
    X_train, X_val = X[train_idx], X[val_idx]
```

---

### Synthetic data generation

When your model development environment cannot touch production data, or when you want to benchmark a method against a known ground truth, you need synthetic data that preserves the multivariate dependence structure of the real book.

**packages:** [insurance-synthetic](https://github.com/burning-cost/insurance-synthetic), [SDV](https://github.com/sdv-dev/SDV), [CTGAN](https://github.com/sdv-dev/CTGAN)

SDV and CTGAN are general-purpose tabular data synthesis tools with good community support. They do not know about insurance concepts: exposure-weighted Poisson marginals, claim count distributions, or TSTR (Train on Synthetic, Test on Real) fidelity metrics specific to Gini-based pricing evaluation.

Our `insurance-synthetic` uses vine copulas to preserve multivariate dependence, including the correlation structure between vehicle age, driver age, and claim frequency that naive synthesis methods break. It ships with a TSTR Gini fidelity report. The trade-off: SDV/CTGAN have wider documentation and larger communities; use them if your synthesis requirement is general-purpose. Use `insurance-synthetic` if you specifically need a portfolio that behaves like insurance data under Poisson/Gamma modelling.

---

## Frequency-Severity Modelling

### GLM for frequency/severity

Poisson GLM for claim frequency and Gamma GLM for average severity is the foundation of UK personal lines pricing. The standard Python tools are mature.

**packages:** [statsmodels](https://www.statsmodels.org/), scikit-learn `TweedieRegressor`, [insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools)

statsmodels' `GLM` with `families.Poisson()` and `families.Gamma()` is the standard. It gives you proper MLE, coefficient standard errors, and likelihood ratio tests — everything a GLM in Emblem would give you, minus the GUI. scikit-learn's `TweedieRegressor` is faster for large datasets but does not expose standard errors or deviance diagnostics.

Our `insurance-glm-tools` adds the actuarial layer: R2VF factor level clustering (collapsing 500 vehicle makes to pricing bands), fused lasso for ordered factor levels, and SKATER-based territory banding. None of this is in statsmodels.

```python
import statsmodels.api as sm

freq_model = sm.GLM(
    y_freq, X,
    family=sm.families.Poisson(),
    exposure=df["exposure"]
).fit()
print(freq_model.summary())
```

---

### GBM for insurance

GBMs outperform GLMs on predictive accuracy for most non-linear insurance datasets. The standard Python ecosystem is well-served here.

**packages:** [CatBoost](https://catboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/)

All three support Poisson and Tweedie objectives natively, handle categorical variables, and work with sample weights for exposure. CatBoost handles categoricals without pre-encoding, which matters for insurance data with many nominal variables. LightGBM is fastest for large datasets. XGBoost has the largest community and the most third-party integrations.

We build our libraries on CatBoost for internal work because of its categorical handling and the availability of SHAP interaction values. If your organisation has standardised on LightGBM or XGBoost, SHAP values are available for both.

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(
    loss_function="Poisson",
    cat_features=cat_cols,
    iterations=1000,
    learning_rate=0.05
)
model.fit(X_train, y_train, sample_weight=exposure_train,
          eval_set=(X_val, y_val), verbose=200)
```

---

### Joint frequency-severity modelling with dependency

Most pricing teams fit frequency and severity models independently and multiply them together. This is wrong if frequency and severity are positively correlated — high-frequency risks also tend to have higher average severity, meaning the premium is understated for the worst risks.

**packages:** [insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity)

There is no well-maintained general-purpose Python package for Sarmanov copula joint frequency-severity modelling. Our `insurance-frequency-severity` is the only Python implementation we know of. It uses the Sarmanov family to model frequency-severity dependence with Poisson-Gamma GLM marginals, IFM estimation, and an analytical premium correction for the dependence term.

The honest caveat: whether your portfolio has material frequency-severity dependence depends on the line of business and data. The library includes a dependence test — run that first before committing to the joint model.

```python
from insurance_frequency_severity import SarmanovModel

model = SarmanovModel(freq_family="poisson", sev_family="gamma")
model.fit(X, n_claims, claim_amounts, exposure=exposure)
print(f"Dependence parameter: {model.omega_:.4f}")
corrected_premium = model.predict_premium(X_new, exposure_new)
```

---

### Zero-inflated and hurdle models

Some insurance data has structural excess zeros — a large fraction of policies make no claims not because their expected frequency is low but because of genuine zero-inflation from unobserved heterogeneity or policy structure (e.g., high excess policies). Standard Poisson GLMs underfit this pattern.

**packages:** statsmodels `ZeroInflatedPoisson` (limited), [insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm)

statsmodels has `ZeroInflatedPoisson` and `ZeroInflatedNegativeBinomialP`, but they lack the covariate-dependent inflation component and the actuarial output format. For a full GAMLSS-style zero-inflated model where both the mean and inflation components are functions of rating variables, use our `insurance-distributional-glm`.

**Honest gap:** for zero-inflated Tweedie in particular — compound Poisson-Gamma with structural zero mass — there is no mature Python package. R's `gamlss` package with `ZAGA` or `ZAIG` families is the practical answer for this problem until something better exists in Python. If you are working in Python and cannot use R, the closest approximation is to fit a Tweedie GBM with the `tweedie` power parameter set near 1, which implicitly handles mass at zero, but this is not the same thing and the interpretation is different.

---

## Interpretability & Factor Extraction

### SHAP relativities / factor tables

A GBM outperforms your production GLM on every holdout metric. The problem: actuarial sign-off, filing requirements, and your rating engine all need multiplicative factor tables. There is no `exp(beta)` in CatBoost.

**packages:** [shap-relativities](https://github.com/burning-cost/shap-relativities), [shap](https://shap.readthedocs.io/)

The base `shap` library gives you SHAP values but not factor tables. Converting SHAP values to multiplicative relativities requires exposure weighting, a reference level for each factor, and a reconstruction validation to check the factors reproduce the model's predictions to sufficient accuracy. Our `shap-relativities` automates this pipeline and produces output in the format actuarial committees expect: a relativity table per rating variable with confidence intervals and an R² reconstruction score.

In our benchmarks against a direct GLM fit, `shap-relativities` produces a +2.85pp Gini lift while reducing NCD relativity extraction error from 9.44% (GLM) to 4.47%.

```python
from shap_relativities import SHAPRelativities

extractor = SHAPRelativities(model, cat_features=cat_cols)
extractor.fit(X_train, exposure=exposure_train)
tables = extractor.relativities()          # dict of DataFrames
reconstruction_r2 = extractor.validate(X_val)
```

---

### EBM / GAM tariffs

If you want a model that is inherently interpretable — shape functions per rating factor, not SHAP post-hoc approximations — Explainable Boosting Machines (EBMs) and Neural Additive Models (NAMs) are the current state of the art.

**packages:** [insurance-gam](https://github.com/burning-cost/insurance-gam), [interpret](https://interpret.ml/) (Microsoft)

Microsoft's `interpret` library provides EBMs and is the reference implementation. Our `insurance-gam` wraps EBM with actuarial-specific output (factor tables with uncertainty bands, pairwise interaction network visualisation, exact Shapley values) and adds Neural Additive Models with insurance loss objectives. If you want a pure EBM with maximum community support, use `interpret` directly. If you need actuarial output format and NAM variants, use `insurance-gam`.

---

### GBM-to-GLM distillation

Sometimes you need the predictive power of a GBM but your rating engine accepts only a GLM — Emblem and Radar work with multiplicative factor tables, not gradient boosted trees. One approach is to treat the GBM as a teacher model and fit a surrogate GLM to its predictions.

**packages:** [insurance-distill](https://github.com/burning-cost/insurance-distill)

There is no general-purpose Python package for this. Our `insurance-distill` fits a surrogate Poisson or Gamma GLM to CatBoost predictions and exports multiplicative factor tables in formats suitable for Radar and Emblem rating engines. In benchmarks, the distilled GLM achieves 90–97% R² match against GBM predictions — good enough for most production uses, but the 3–10% residual is genuine information loss that the actuary needs to understand and document.

---

## Uncertainty Quantification

### Conformal prediction intervals

Point estimates per risk are insufficient for Solvency II internal models, capital allocation, and underwriting decisions on individual large risks. Conformal prediction provides distribution-free prediction intervals with a finite-sample coverage guarantee — unlike parametric bootstrap intervals, the guarantee holds without distributional assumptions.

**packages:** [insurance-conformal](https://github.com/burning-cost/insurance-conformal), [MAPIE](https://mapie.readthedocs.io/)

MAPIE is a well-maintained general-purpose conformal prediction library with sklearn integration. It is the right choice if you want broad method coverage, active community support, and are not working specifically with Tweedie or Poisson claims data.

Our `insurance-conformal` is optimised for insurance-specific non-conformity scores: the default `pearson_weighted` score, which uses exposure-weighted Pearson residuals, produces intervals 13.4% narrower than MAPIE's default at identical 90% coverage on 50k synthetic UK motor policies. We also implement frequency-severity conformal intervals (Graziadei et al.) and online retrospective adjustment. If you are doing general ML, use MAPIE. If you are doing insurance pricing with Tweedie losses and exposure weights, the difference matters.

```python
from insurance_conformal import InsuranceConformalPredictor

cp = InsuranceConformalPredictor(model, nonconformity="pearson_weighted")
cp.calibrate(X_cal, y_cal, exposure=exposure_cal)
lower, upper = cp.predict(X_test, exposure=exposure_test, alpha=0.10)
```

---

### Distributional regression

A point-estimate GBM tells you the expected loss cost. It does not tell you whether that expected value comes from a narrow or a wide predictive distribution — and for capital allocation and individual risk assessment, the variance matters as much as the mean.

**packages:** [insurance-distributional](https://github.com/burning-cost/insurance-distributional), [NGBoost](https://github.com/stanfordmlgroup/ngboost), [PGBM](https://github.com/elephaint/pgbm)

NGBoost (Stanford) and PGBM both provide distributional GBMs with natural gradient and probabilistic gradient boosting respectively. They are well-documented general-purpose tools.

Our `insurance-distributional` uses CatBoost with custom Tweedie, Gamma, ZIP, and NegBin objectives and produces per-risk volatility scores in insurance-interpretable format. The GammaGBM variant produces +1.5% log-likelihood improvement over a standard CatBoost Gamma objective in our benchmarks. The practical difference from NGBoost is insurance-specific: correct exposure handling, per-risk volatility scoring rather than full distribution output, and output in the format pricing models expect.

---

### Bayesian credibility

Credibility theory provides a formal mechanism for blending a segment's observed experience with a prior estimate, weighted by the credibility of the observations.

**packages:** [insurance-credibility](https://github.com/burning-cost/insurance-credibility), [bayesian-pricing](https://github.com/burning-cost/bayesian-pricing), [PyMC](https://www.pymc.io/)

PyMC (version 5) provides general hierarchical Bayesian modelling and is the right foundation if you want full flexibility in model specification. Our `insurance-credibility` provides Bühlmann-Straub credibility specifically, with mixed-model equivalence checks, static and dynamic experience rating, and a deep attention credibility variant. Our `bayesian-pricing` wraps PyMC 5 for hierarchical Bayesian pricing with actuarial factor output.

Use PyMC directly if your problem requires custom priors or model structure. Use `insurance-credibility` if you want a drop-in Bühlmann-Straub implementation that outputs factors in actuarial format and includes the experience rating workflow.

---

## Causal Inference

### Treatment effect estimation (DML)

Standard GLM coefficients are biased whenever rating variables correlate with distribution channel or policyholder selection. A vehicle value factor that looks significant in the GLM may be picking up channel effects — direct customers buy cheaper cars. Double Machine Learning (DML) removes that confounding without requiring a structural model.

**packages:** [insurance-causal](https://github.com/burning-cost/insurance-causal), [EconML](https://econml.azurewebsites.net/), [DoubleML](https://docs.doubleml.org/)

EconML (Microsoft) and DoubleML are mature, well-documented DML implementations with active development. EconML in particular has extensive coverage of causal methods and good documentation.

Our `insurance-causal` provides DML with CatBoost nuisance models, a confounding bias report (showing how much each variable's coefficient changes under DML versus naive OLS), and price elasticity estimation via `insurance_causal.elasticity`. The difference from EconML is focus: `insurance-causal` is set up for the specific question a pricing actuary asks — "how much of this factor's coefficient is real risk signal versus channel or selection bias?" — and outputs results in that framing.

In benchmarks, DML correctly removes non-linear confounding bias at scale (n≥50k). Be honest about the limitation: at small n, DML over-partials — the nuisance model absorbs signal as well as confounding. It is not magic; it requires sufficient data for the nuisance models to converge.

```python
from insurance_causal import DeconfoundedPricer

pricer = DeconfoundedPricer(treatment="vehicle_value", controls=controls)
pricer.fit(df, outcome="claim_freq", exposure="exposure")
print(pricer.bias_report())         # coefficient shift: naive vs deconfounded
```

---

### Heterogeneous treatment effects (causal forests)

A portfolio-level price elasticity from DML tells you the average. Causal forests tell you which segments respond most to a rate change — a critical input for differential pricing strategy.

**packages:** [insurance-causal](https://github.com/burning-cost/insurance-causal) (`causal_forest` module), [EconML](https://econml.azurewebsites.net/)

EconML's `CausalForestDML` is the reference implementation. Our `insurance-causal` wraps it with insurance-specific inference: GATES (grouped average treatment effects) for segment-level effects, CLAN for profiling the most/least responsive segments, and RATE/AUTOC/QINI targeting evaluation scores for assessing whether the heterogeneity is actionable.

One important caveat we document explicitly: individual-level CATEs (conditional average treatment effects) are too noisy to act on directly — arXiv:2509.11381 demonstrates this. Work with GATE aggregates at n≥2,000 per group. Using causal forest output to rank individuals and price them accordingly is a mistake.

---

### Rate change evaluation (DiD/ITS)

You put through a rate increase in Q3. Conversion dropped. How much of that drop was your rate change versus market conditions? Without a control group, the pre/post comparison is confounded by seasonal effects and competitor movements.

**packages:** [insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy), [EconML](https://econml.azurewebsites.net/), [pyfixest](https://github.com/py-econometrics/pyfixest)

`pyfixest` provides standard difference-in-differences with fixed effects and is well-maintained for general econometric purposes.

Our `insurance-causal-policy` implements Synthetic Difference-in-Differences (SDID) — which constructs a synthetic control group from comparable segments when no clean control exists — and the Doubly Robust SC estimator (DRSC, arXiv:2503.11375), which is consistent if either parallel trends or the synthetic control weights are correct. In benchmarks, naive before-after comparison is biased +3.8pp by market inflation; SDID achieves 98% CI coverage. DRSC achieves 24% lower RMSE than SDID when you have few donor segments (N_co=6). The output includes an HonestDiD sensitivity analysis and a FCA evidence pack for Consumer Duty documentation.

---

## Fairness & Discrimination

### Proxy discrimination auditing

Under the Equality Act 2010 and FCA Consumer Duty, using a rating variable that is a proxy for a protected characteristic requires proportionality justification. Postcode correlated with ethnicity is the most common example in UK motor. The naive Spearman correlation test routinely misses proxy relationships that a classification-based audit catches.

**packages:** [insurance-fairness](https://github.com/burning-cost/insurance-fairness), [Fairlearn](https://fairlearn.org/), [AIF360](https://aif360.res.ibm.com/)

Fairlearn (Microsoft) and AIF360 (IBM) are the most widely used general-purpose fairness libraries. Both compute standard fairness metrics (demographic parity, equalised odds, etc.) and provide mitigation algorithms.

The gap with general-purpose tools is the UK regulatory framing. FCA Consumer Duty does not map directly onto the fairness notions in US-origin libraries, which are primarily focused on employment and credit decisions under US law. Our `insurance-fairness` is built around the specific legal obligations in the Equality Act 2010 and the Consumer Duty outcome requirements, with a proxy audit methodology (classification-based R² proxy detection) that catches proxy relationships the correlation-based methods miss: in benchmarks, proxy R²=0.78 catches a postcode/ethnicity proxy that Spearman r=0.06 misses entirely.

```python
from insurance_fairness import ProxyAudit

audit = ProxyAudit(protected="ethnicity_proxy", rating_vars=rating_vars)
audit.fit(df)
report = audit.consumer_duty_report()    # structured for FCA documentation
print(audit.proxy_r2_table())
```

---

### FCA Consumer Duty compliance

The Consumer Duty (July 2023) requires insurers to demonstrate their pricing does not lead to poor outcomes for protected groups. This requires documentation, not just an absence of discrimination.

**packages:** [insurance-fairness](https://github.com/burning-cost/insurance-fairness)

No third-party library addresses FCA Consumer Duty specifically. The `insurance-fairness` library produces a structured evidence pack with the proxy audit results, disparate impact metrics, and proportionality documentation in the format an FCA review would expect. It does not tell you whether you are compliant — that is a legal question — but it gives you the documented evidence base to make the proportionality argument.

---

## Model Monitoring & Governance

### Drift detection (PSI/CSI/A/E)

Once a model is live, you need to know when it has stopped working — before the loss ratio tells you. The standard indicators are Population Stability Index (PSI) for input distribution drift, Characteristic Stability Index (CSI) for individual variable drift, and Actual-vs-Expected (A/E) ratios with IBNR adjustment for predictive accuracy.

**packages:** [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring), [Evidently](https://www.evidentlyai.com/), [NannyML](https://nannyml.com/)

Evidently and NannyML are strong general-purpose model monitoring tools with good dashboards and broad method coverage. Evidently in particular has good PSI and data drift implementations.

The insurance-specific gap is exposure weighting. General-purpose monitoring tools treat all observations equally; insurance drift detection should weight by exposure because a PSI spike driven by a change in your book's exposure distribution means something different from the same PSI spike with flat exposure. Our `insurance-monitoring` computes exposure-weighted PSI/CSI, A/E ratios with IBNR adjustment, and a Gini drift z-test with a formal decision rule for whether to recalibrate the intercept or refit with new data.

It also includes `PITMonitor` for calibration drift detection via PIT e-process martingale, which achieves ~3% false positive rate versus 46% for repeated Hosmer-Lemeshow tests — a straightforward argument for using the Bayesian approach when you are running monthly checks. The `InterpretableDriftDetector` attributes feature-interaction drift with BH FDR control.

---

### Sequential A/B testing

Champion/challenger experiments are the right way to validate a new pricing model before full rollout. The naive approach — run a t-test at a fixed interval — has a 25% false positive rate if you peek at results before the experiment ends.

**packages:** [insurance-monitoring](https://github.com/burning-cost/insurance-monitoring) (`SequentialTest`), [scipy](https://scipy.org/) (fixed-horizon only)

scipy provides standard t-tests and z-tests but no sequential testing infrastructure. The correct tool for anytime-valid inference — where you can look at results at any point without inflating the false positive rate — is a mixture Sequential Probability Ratio Test (mSPRT). Our `insurance-monitoring`'s `SequentialTest` implements mSPRT with exposure-weighted Poisson observations. In benchmarks, a standard t-test run at interim checkpoints has 25% FPR (five times nominal); mSPRT holds at 1%.

---

### Model validation reports

PRA Supervisory Statement SS1/23 requires insurers to maintain model validation documentation covering statistical tests, expert review, model limitations, and ongoing performance monitoring.

**packages:** [insurance-governance](https://github.com/burning-cost/insurance-governance)

No general-purpose Python library addresses PRA SS1/23 specifically. Commercial platforms like Emblem and Radar produce model reports, but they do not expose the underlying methodology or produce machine-readable output. Our `insurance-governance` generates HTML and JSON validation reports structured for model risk committees and PRA review: bootstrap Gini CI, Poisson A/E CI, double-lift charts, renewal cohort tests, ModelCard, ModelInventory, and GovernanceReport objects. The automated suite catches miscalibration at the age-band level that manual checklists miss.

---

## Rate Optimisation

### Constrained rate changes

Given a technical price per segment, a loss ratio target, and maximum movement caps, what is the optimal rate change per segment? This is typically done in a spreadsheet where the constraints interact and the solution is not verifiably optimal.

**packages:** [insurance-optimise](https://github.com/burning-cost/insurance-optimise), [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html)

`scipy.optimize` provides the underlying SLSQP solver and is appropriate if you want to define the problem yourself. Our `insurance-optimise` wraps it with insurance-specific structure: FCA ENBP (equivalent net premium) constraints as required by GIPP (PS21/5), analytical Jacobians for faster convergence, an efficient frontier between loss ratio improvement and movement constraints, and a JSON audit trail for regulatory documentation.

```python
from insurance_optimise import RateOptimiser

opt = RateOptimiser(
    technical_price=df["technical_price"],
    current_rate=df["current_rate"],
    exposure=df["exposure"]
)
result = opt.optimise(
    target_lr=0.68,
    max_movement=0.20,
    enbp_constraint=True
)
```

---

### Portfolio optimisation (multi-objective)

Single-objective rate optimisation ignores the trade-off between profit, retention, and fairness. Tightening rates in high-risk segments improves the loss ratio but increases the premium disparity ratio. The efficient frontier makes this trade-off explicit so you can defend the operating point.

**packages:** [insurance-optimise](https://github.com/burning-cost/insurance-optimise) (`ParetoFrontier`)

There is no off-the-shelf Python tool for multi-objective insurance rate optimisation. `insurance-optimise`'s `ParetoFrontier` uses an epsilon-constraint sweep across profit, retention, and fairness objectives with TOPSIS selection for the recommended operating point. The built-in `premium_disparity_ratio` and `loss_ratio_disparity` metrics quantify the fairness dimension. In our benchmark, single-objective SLSQP produces a premium disparity ratio of 1.168; the Pareto surface makes visible that operating points exist with equivalent loss ratio improvement at disparity ratios below 1.05.

---

## Smoothing & Spatial

### Whittaker-Henderson smoothing

Raw one-way relativities from a GLM are spiky when data is thin in some cells. The standard actuarial fix is Whittaker-Henderson smoothing, which applies a penalty on second differences to produce a smooth curve while respecting the underlying data.

**packages:** [insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)

There is no well-maintained Python implementation of Whittaker-Henderson smoothing with REML lambda selection prior to our library. R's `JOPS` package is the reference implementation. Our `insurance-whittaker` provides 1D, 2D, and Poisson-weighted variants with REML lambda selection and Bayesian credible intervals. In benchmarks, it achieves a 57.2% MSE reduction versus raw rates on synthetic one-way curves. The 2D variant is useful for age/duration surfaces in NCD ratemaking.

```python
from insurance_whittaker import WhittakerSmoother

smoother = WhittakerSmoother(order=2)
smoother.fit(rates, weights=exposures)
smooth_rates = smoother.smooth()
lambda_reml = smoother.lambda_
```

---

### Spatial territory rating

Raw postcode-level relativities are noisy when data is thin — adjacent postcodes with very different relativities often reflect data noise rather than genuine risk differences. Spatial smoothing that borrows strength from neighbouring areas produces more stable territory factors.

**packages:** [insurance-spatial](https://github.com/burning-cost/insurance-spatial), [PyMC](https://www.pymc.io/), [PySAL](https://pysal.org/)

PySAL provides spatial statistics and spatial weights matrices. PyMC provides the Bayesian modelling infrastructure. Our `insurance-spatial` combines them to implement the BYM2 model — a structured additive regression with an ICAR (intrinsic conditional autoregressive) spatial random effect and an unstructured component — with adjacency matrix construction from postcode data, Moran's I diagnostics, and factor output per postcode. BYM2 is not available as a drop-in in any general-purpose Python package; if you want to implement it yourself using PyMC, the adjacency matrix construction and ICAR prior specification are non-trivial.

---

## Telematics

### HMM driving state classification

Raw 1Hz GPS and accelerometer data from UBI products contains signal about driving behaviour, but aggregating trip-level events into a driver-level risk score that is stable across different journey mixes and road types is not straightforward. Hidden Markov Models (HMMs) provide a principled approach.

**packages:** [insurance-telematics](https://github.com/burning-cost/insurance-telematics), [hmmlearn](https://hmmlearn.readthedocs.io/)

`hmmlearn` provides general-purpose HMM fitting. Our `insurance-telematics` goes further: it classifies driving states (cautious/normal/aggressive), applies Bühlmann-Straub credibility aggregation to driver level to stabilise scores across drivers with different trip histories, and outputs a Poisson GLM-compatible risk variable. In benchmarks, HMM state features produce a 3–8pp Gini improvement over raw trip averages.

The limitation to state clearly: telematics data pipelines are expensive to build and maintain. If your organisation does not have reliable raw trip data at 1Hz resolution, start there before investing in the HMM model. The model is the last 5% of the problem.

---

## Reserving & Severity

### Tail modelling / Extreme Value Theory

Standard Gamma GLM severity models underfit the upper tail of the loss distribution. For large loss loading and increased limits factors, you need a model that captures the Pareto tail correctly.

**packages:** [insurance-severity](https://github.com/burning-cost/insurance-severity), [thresholdmodeling](https://github.com/iagolemos1/thresholdmodeling), [pyextremes](https://georgebv.github.io/pyextremes/)

`pyextremes` provides EVT methods (GEV, GPD) for extreme value analysis. Neither it nor `thresholdmodeling` is designed for insurance severity with covariate-dependent thresholds.

Our `insurance-severity` implements spliced severity models with covariate-dependent threshold selection (separating attritional and large loss components), composite Lognormal-GPD and Pareto-Gamma models, Deep Regression Networks (DRN) for neural severity modelling, and EQRN extreme quantile neural networks. In benchmarks, the composite model reduces tail error 5.6% versus single lognormal; the improvement is 15–20% on heavy-tail DGPs (Pareto α=1.5).

```python
from insurance_severity import SplicedSeverityModel

model = SplicedSeverityModel(
    body_family="gamma",
    tail_family="pareto",
    threshold_method="covariate_dependent"
)
model.fit(X, claim_amounts, exposure=exposure)
ilf_table = model.increased_limits_factors([1e5, 2.5e5, 5e5, 1e6])
```

---

### Large loss handling

Policies with very large claims distort GLM fits. The common ad-hoc fixes — capping losses at a percentile, excluding the top N claims — are defensible if documented but throw away information. Spliced models that fit body and tail separately are the principled approach.

**packages:** [insurance-severity](https://github.com/burning-cost/insurance-severity)

The spliced model approach is the same as the EVT case above — see that section. For the simpler task of large loss identification and basic Pareto tail fitting, `scipy.stats` provides `pareto`, `genpareto`, and `lognorm` distributions with MLE fitting. This is adequate for ILF tables if your data is clean and you do not need covariate-dependent thresholds.

---

## What this page does not cover

**Commercial pricing platforms.** If you need a comparison of open-source versus Emblem, Radar, Akur8, or DataRobot, see our [/compare/](/compare/) page.

**Reserving.** For chain-ladder and stochastic reserving, [chainladder-python](https://github.com/casact/chainladder-python) is the standard Python library. We do not have overlap with it.

**Claims fraud.** Fraud scoring is a separate discipline that shares methods (anomaly detection, network analysis) but has different ground truth and regulatory constraints. We do not cover it.

**Catastrophe modelling.** Cat modelling is its own specialised field. We do not touch it.

**R packages.** We mention R where no adequate Python alternative exists. For a comprehensive survey of actuarial R packages, the [CRAN Task View for Insurance](https://cran.r-project.org/view=Insurance) maintained by Christophe Dutang is the reference.

---

All Burning Cost libraries are MIT-licensed and install from PyPI. The [full library index](/tools/) has installation commands and links to GitHub. For the decision guide mapping problems to libraries, see [/guide/](/guide/).

Questions or corrections: [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com).
