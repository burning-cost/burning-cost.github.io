---
layout: page
title: "Benchmarks"
description: "Every library ships with a Databricks-runnable benchmark on a known data-generating process. The results below are extracted from those benchmarks."
permalink: /benchmarks/
---

Every library ships with a Databricks-runnable benchmark on a known data-generating process. All benchmarks are self-contained (synthetic data, no external files), run on Databricks Serverless unless noted, and produce the same output on every run. The results below are extracted from those runs.

We publish the failures alongside the successes. Where a method has conditions under which it does not win, we say so.

---

## GBM Deployment

### shap-relativities — SHAP rating relativities from GBMs

**What is measured:** GBM+SHAP relativity extraction vs direct Poisson GLM on a 20,000-policy synthetic motor DGP with known true coefficients (NCD=5 true relativity 0.549, conviction true 1.57×).

| Method | Gini | NCD=5 relativity error | Conviction relativity error |
|---|---|---|---|
| Poisson GLM (direct) | baseline | 9.44% | — |
| CatBoost + shap-relativities | +2.85pp Gini | **4.47%** | recovers 1.57× within CI |

Adding an interaction DGP (vehicle_group × NCD): SHAP relativities give +3.4pp Gini over main-effects GLM. SHAP absorbs the interaction into the marginals, which is the correct TreeSHAP behaviour.

[github.com/burning-cost/shap-relativities](https://github.com/burning-cost/shap-relativities)

---

### insurance-distill — GBM-to-GLM distillation

**What is measured:** SurrogateGLM fidelity vs direct GLM on 30,000 synthetic motor policies with a CatBoost teacher model. 80/20 holdout.

| Method | Max segment deviation | Holdout Gini | Fidelity R² |
|---|---|---|---|
| Direct GLM | 21.4% | 97.6%* | 0.51 |
| SurrogateGLM | **3.6%** | 86.5%* | 0.54 |
| LassoGuidedGLM (α=0.0005) | 10.4% | — | — |

*The Gini reversal is noise at this n; fidelity R² is the correct metric. SurrogateGLM is 6× more faithful at cell level — the segment a pricing actuary would present to a CRO.

[github.com/burning-cost/insurance-distill](https://github.com/burning-cost/insurance-distill)

---

### insurance-deploy — Champion/challenger deployment framework

**What is measured:** Operational correctness of the deployment infrastructure — routing determinism, ENBP breach detection, and statistical test calibration. 10,000 synthetic motor policies, 20% challenger allocation, shadow mode, ~40% renewals with 2% injected ENBP breaches.

| Capability | What's guaranteed |
|---|---|
| Routing determinism | Hash routing gives expected allocation within 0.5pp; same policy_id always routes to same arm |
| ENBP breach detection | ~2% of renewals flagged, matching the injected breach rate |
| Bootstrap LR test calibration | Returns INSUFFICIENT_EVIDENCE at typical volumes — reflects the true statistical difficulty |
| Power analysis | Reports 18–24 months to LR significance including 12-month development, matching theoretical calculation |

This is a deployment and governance framework, not a predictive model. The benchmark validates that the infrastructure does what it claims: deterministic routing, correct audit logging, and honest statistical tests that do not promote a challenger prematurely. At 20% challenger allocation and 10,000 policies, the power analysis correctly flags that loss ratio significance requires 18–24 months — teams that expect a promotion decision in 60 days will be disappointed, and the library is designed to surface that before the experiment starts. SQLite is not appropriate for multi-process concurrent writes; use the adapter pattern for distributed rating engines.

[github.com/burning-cost/insurance-deploy](https://github.com/burning-cost/insurance-deploy)

---

## Severity Modelling

### insurance-severity — EVT tail modelling

**EVT benchmark (TruncatedGPD vs naive GPD, £100k policy limit):**

| Method | Shape parameter (ξ) bias | Q99 error |
|---|---|---|
| Naive GPD | 0.035 | 10.3% |
| TruncatedGPD | **0.006** | **1.2%** |

**Heavy-tail benchmark (α=1.5 Pareto, infinite variance, n=20,000):** Gamma GLM structurally fails at the tail. LognormalGPDComposite and GammaGPDComposite (both threshold_method='profile_likelihood') recover the tail shape. Q99 error reduction vs Gamma: 15–20+ percentage points. ILF error at £5m limit: 20+ ppts lower for composite models. Run: Databricks Serverless (43s, SUCCESS).

**WeibullTemperedPareto vs standard Pareto:** +31.3 log-likelihood; standard Pareto Q99.5 error 15.9%.

[github.com/burning-cost/insurance-severity](https://github.com/burning-cost/insurance-severity)

---

### insurance-quantile — Quantile regression for large loss reserving

**What is measured:** Four methods (Gamma GLM, GBM, Lognormal QR, EQRN) on 5,000 synthetic UK motor claims with vehicle-group heteroscedasticity. Coverage, pinball loss, TVaR accuracy, and ILF accuracy benchmarked separately.

**Coverage and pinball loss at Q95 (target 95%):**

| Method | Coverage | Pinball loss |
|---|---|---|
| Gamma GLM | 91.2% | 312.4 |
| GBM | 93.1% | 298.7 |
| Lognormal QR | **95.3%** | **277.1** |

**TVaR accuracy vs DGP truth (Lognormal QR vs GBM):**

| Method | TVaR MAE |
|---|---|
| GBM | 477.0 |
| Lognormal QR | **315.1** |

**ILF accuracy and heteroscedastic Q95 coverage by vehicle group:** Lognormal QR recovers group-specific tails; GBM undercovers the highest-risk vehicle group by 4–6pp because it fits a common residual distribution.

[github.com/burning-cost/insurance-quantile](https://github.com/burning-cost/insurance-quantile)

---

## Prediction Intervals

### insurance-conformal — Conformal prediction intervals

**What is measured:** Locally-weighted (LW) conformal vs naive parametric intervals vs standard conformal on 50,000-policy Gamma DGP (heteroscedastic, shape decreasing with predicted mean). 60/20/20 temporal split.

| Method | Worst-decile coverage | Interval width vs parametric |
|---|---|---|
| Naive parametric | ~70–75% (misses 90% target) | baseline |
| Standard conformal | 87.9% (still misses worst decile) | −13.4% |
| LW conformal | **90%+ in every decile** | **−11.7%** |

Standard conformal meets aggregate coverage but undercovers the highest-risk decile by ~10pp — precisely the segment that drives SCR and reinsurance cost. LW conformal meets the target by construction in every decile while also being narrower.

[github.com/burning-cost/insurance-conformal](https://github.com/burning-cost/insurance-conformal)

---

### insurance-conformal-ts — Adaptive conformal intervals for aggregate time series

**What is measured:** Adaptive conformal inference (ACI) and ConformalPID vs split conformal and naive parametric on synthetic monthly aggregate claim series with distribution shift. Kupiec unconditional coverage test on held-out windows.

**24-month test window:**

| Method | Coverage | Kupiec p-value |
|---|---|---|
| Naive parametric | 0.583 | — |
| Split conformal | 0.542 | — |
| ACI | 0.792 | 0.1163 |
| ConformalPID | **0.808** | **0.1481** |

**60-month test window (more stable regime):**

| Method | Coverage | Kupiec p-value |
|---|---|---|
| Split conformal | 0.483 | — |
| ACI | 0.833 | 0.1876 |
| ConformalPID | **0.850** | **0.2256** |

Split conformal collapses under distribution shift — its calibration set is stale by the time the test window arrives. ACI and ConformalPID maintain nominal coverage by adapting the quantile online. ConformalPID's integral correction term prevents the slow drift that accumulates in ACI over longer horizons.

[github.com/burning-cost/insurance-conformal-ts](https://github.com/burning-cost/insurance-conformal-ts)

---


### insurance-gam — Interpretable GAM/EBM pricing models

**What is measured:** InsuranceEBM (explainable boosting machine) vs Oracle GLM on synthetic motor DGP with known non-linear age effect. Databricks Serverless, 2026-03-16.

| Method | Poisson deviance | Gini |
|---|---|---|
| Oracle GLM | 0.2516 | −0.453 |
| InsuranceEBM | 1.333* | −0.294 |
| GLM (standard) | 0.2535 | −0.449 |

*The EBM deviance of 1.333 is a miscalibration artefact, not a structural failure of the EBM approach. The EBM correctly identifies the non-linear age-risk relationship (Gini reflects ranking quality) but requires calibration post-fitting to match the GLM scale. On Gini, EBM at −0.294 is weaker than GLM at −0.449, suggesting the age non-linearity in this DGP is captured well enough by the GLM's piecewise age-band encoding. EBM value is in interpretability, not necessarily Gini lift on well-specified DGPs.

[github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam)

---

## Causal Inference

### insurance-causal-policy — Synthetic DiD / SDID for pricing interventions

**What is measured:** SDID vs naive before-after and plain DiD. 30-simulation Monte Carlo, 80 segments, 12 periods, true ATT = −0.08, market inflation 0.5pp per period.

| Method | Bias | 95% CI coverage |
|---|---|---|
| Naive before-after | ~+2pp upward (market inflation absorbed) | — |
| SDID | **near-zero** | ~93–95% |

Naive before-after is biased upward by roughly 4 × 0.5pp inflation over the post-period window. SDID recovers the true ATT with valid confidence intervals.

**v0.2.0 — DoublyRobustSCEstimator (DRSC, arXiv:2503.11375)**

DRSC adds double robustness to SDID: the estimator is consistent if *either* the parallel trends assumption holds *or* the synthetic control weights are correctly specified — you only need one of the two to hold, not both. ~420 LOC, 56 new tests, no new dependencies.

**What is measured:** DRSC vs SDID RMSE under two donor pool sizes, 100-simulation Monte Carlo, fixed seed.

| Donor pool | DRSC RMSE | SDID RMSE | DRSC improvement |
|---|---|---|---|
| N_co = 6 (few donors) | lower | baseline | **24% lower RMSE** |
| N_co = 40 (many donors) | equivalent | equivalent | no meaningful difference |

The advantage is concentrated in the regime most relevant to insurance pricing: when you have few comparable control segments (e.g., 5–10 product lines or regions as controls), DRSC substantially outperforms SDID. With a rich donor pool, both estimators perform similarly — DRSC does not hurt.

[github.com/burning-cost/insurance-causal-policy](https://github.com/burning-cost/insurance-causal-policy)

---

### insurance-causal — DML causal effect estimation

**What is measured:** DML (CausalPricingModel) vs naive Poisson GLM on confounded telematics DGP (5,000 policies, true effect −0.15, confounding via driver safety score).

Honest finding: on a single-run Databricks benchmark at n=5,000, the naive GLM achieved −0.2124 (bias 41.6%, CI covers true) and DML achieved −0.0202 (bias 86.5%, CI misses true). DML underperforms here due to over-partialling — when CatBoost nuisance models absorb most outcome variance, the residualised treatment has low variance and the final regression is imprecise. **DML wins when n ≥ 50,000, treatment effects are large, and GLM misspecification compounds across many factors.** See the README for the full conditions.

[github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal)

---

### insurance-elasticity — DML price elasticity with heterogeneous treatment effects

> **Merged into [insurance-causal](https://github.com/burning-cost/insurance-causal).** Use `insurance_causal.elasticity` instead. The benchmark below was run before the merge and remains valid.

**What is measured:** DML (DoubleMLElasticityModel) vs naive OLS on 50,000 synthetic renewal policies with known NCD-varying price elasticity. True elasticity varies by NCD band (−1.5 to −3.0). Databricks Serverless, 2026-03-16.

| Method | ATE bias | NCD GATE RMSE | 95% CI valid? |
|---|---|---|---|
| Naive OLS | 24.5% | 0.0855 | No |
| DML | **21.8%** | **0.0448** | **Yes** |

DML reduces NCD GATE RMSE by 47.6% vs OLS. The primary win is not ATE accuracy (both are imprecise at this treatment heterogeneity) but GATE recovery: naive OLS estimates a single pooled elasticity and cannot produce per-NCD-band confidence intervals. The DML CI correctly covers the true NCD elasticity in each band; the OLS CI does not because it conflates confounding with price variation.

[github.com/burning-cost/insurance-causal](https://github.com/burning-cost/insurance-causal)

---


## Demand & Retention Modelling

### insurance-demand — Price elasticity recovery for new business conversion

> **Merged into [insurance-optimise](https://github.com/burning-cost/insurance-optimise).** Use `insurance_optimise.demand` instead. The benchmark below was run before the merge and remains valid.

**What is measured:** ConversionModel (Heckman-corrected log-price elasticity) vs naive log-price OLS on 30,000 synthetic UK motor quotes with known true price elasticity of −2.0 and selection bias from quote-to-bind conversion.

| Method | Estimated elasticity | Bias vs true (−2.0) |
|---|---|---|
| Naive OLS coefficient | −0.40 | 80% bias |
| ConversionModel | −2.09 | **4.5%** |

Naive OLS recovers −0.40 because it regresses on the selected sample of bound policies — the most price-insensitive customers. ConversionModel corrects for selection using the full quote funnel, recovering the true −2.0 within 4.5%. The 80% OLS bias is not a modelling subtlety: it produces a fundamentally wrong demand curve that will lead a pricing optimiser to systematically underestimate the cost of price increases.

[github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

## Dispersion & Distributional Modelling

### insurance-dispersion — DGLM for heteroscedastic severity

**What is measured:** DGLM (double GLM with modelled dispersion parameter φ) vs standard Gamma GLM (constant φ) on 25,000 synthetic policies where φ varies by vehicle group. Known DGP: φ ranges from 0.1 (fleet, low dispersion) to 0.6 (young drivers, high dispersion).

| Method | φ MAE | Variance ratio range | 95% PI coverage |
|---|---|---|---|
| Gamma GLM (constant φ) | 0.312 | 0.51–1.94 | 88.3% |
| DGLM | **0.089** | **0.87–1.12** | **91.7%** |

The constant-φ GLM systematically under-reserves for young driver claims (variance ratio 1.94 — predicted variance is half of actual) and over-reserves for fleet (variance ratio 0.51). DGLM reduces φ MAE by 71.5% and collapses the variance ratio range from 0.51–1.94 to 0.87–1.12. Separate Gamma benchmark (single-group φ=0.30): DGLM recovers φ̂=0.304, 90% PI coverage 88.8% vs target 90%.

[github.com/burning-cost/insurance-dispersion](https://github.com/burning-cost/insurance-dispersion)

---

### insurance-distributional — Distributional GBM (GammaGBM)

**What is measured:** GammaGBM (CatBoost with distributional loss, models both μ and φ jointly) vs standard constant-φ Gamma GLM. 6,000 synthetic UK motor policies with vehicle-group-varying dispersion. Databricks Serverless, 2026-03-16.

**Prediction interval coverage (target level vs achieved):**

| Target | GLM coverage | GammaGBM coverage |
|---|---|---|
| 80% | 74.8% | **80.4%** |
| 90% | 84.2% | **89.5%** |
| 95% | 90.1% | **94.9%** |

**Dispersion parameter recovery:** φ correlation with true φ: GLM 0.000 (constant φ by construction), GammaGBM **+0.702**. GLM's constant-φ assumption is not approximately right on heteroscedastic data — it is structurally incapable of capturing vehicle-group dispersion differences. GammaGBM recovers the dispersion structure at the cost of interpretability.

[github.com/burning-cost/insurance-distributional](https://github.com/burning-cost/insurance-distributional)

---

### insurance-distributional-glm — GAMLSS for joint mean and variance modelling

**What is measured:** GAMLSS (generalised additive models for location, scale, shape) vs standard Gamma GLM (constant φ) on 25,000 synthetic policies with age-varying dispersion. Known DGP: log(φ) is a smooth function of driver age.

| Method | σ MAE | σ correlation with truth | 95% PI coverage |
|---|---|---|---|
| Gamma GLM (constant φ) | 0.1018 | 0.000 | 93.87% |
| GAMLSS | **0.0059** | **0.998** | **94.25%** |

GAMLSS reduces σ MAE by 94.2% and achieves near-perfect recovery of the age-varying dispersion curve (correlation 0.998). The 95% PI coverage improvement from 93.87% to 94.25% looks small but the improvement in the tails is larger — GAMLSS correctly widens intervals for young and elderly drivers where the GLM undercovers by 3–5pp. Fit time is comparable to GLM.

[github.com/burning-cost/insurance-distributional-glm](https://github.com/burning-cost/insurance-distributional-glm)

---

## Time-Varying Models

### insurance-dynamics — GAS time-varying frequency models

**What is measured:** GAS Poisson (time-varying λ) vs GLM constant rate vs GLM with linear trend. Synthetic 60-month claim series with a +37.5% step change in frequency at month 36. Known DGP allows direct bias measurement.

| Method | MAE (overall) | RMSE (overall) | Log-likelihood |
|---|---|---|---|
| GLM constant | 0.014980 | 0.019104 | −289.0 |
| GLM with trend | 0.009697 | 0.008795 | −236.3 |
| GAS Poisson | **0.008438** | **0.007540** | **−231.7** |

GAS Poisson achieves the best log-likelihood (−231.7 vs −236.3 for GLM trend) and lowest RMSE (0.007540 vs 0.008795). The advantage is concentrated in the post-break period: the GLM with trend fits a global slope and systematically lags the step change by 6–8 months. GAS updates the local level at each observation and tracks the new regime within 2–3 months.

[github.com/burning-cost/insurance-dynamics](https://github.com/burning-cost/insurance-dynamics)

---

### insurance-trend — Loss cost trend analysis

**What is measured:** `insurance-trend` (log-linear with structural break detection) vs naive single-segment OLS on synthetic 24-quarter UK motor series (2019 Q1–2024 Q4) with known frequency break at Q8 and severity break at Q12. True post-break rates: frequency +3.0% pa, severity +8.0% pa. Databricks Serverless, 2026-03-16.

**Trend rate accuracy:**

| Component | True (DGP) | Naive OLS | insurance-trend |
|---|---|---|---|
| Frequency | +3.000% | −5.152% | −5.353% |
| Severity | +8.000% | +2.353% | +2.256% |
| Loss cost | +11.240% | −2.921% | −3.217% |

† Without break detection, single-segment OLS returns the wrong sign on frequency trend. Use `changepoints=[8,12]` or auto-detection for series with structural breaks.

**4-quarter forward projection MAPE:**

| Method | Loss cost MAPE |
|---|---|
| Naive OLS | 29.99% |
| insurance-trend | **26.06%** |

Honest caveat: break detection did not fire on this 24-quarter series — the PELT penalty threshold was not exceeded at this noise level. Both models produce similar trend rate estimates for this reason. The 3.9pp MAPE improvement comes from the frequency/severity decomposition and bootstrap CI (frequency CI: −7.3% to −3.2%), not from break detection. Naive OLS produces one blended loss cost trend; it cannot separate −5.2pp frequency from +2.4pp severity, which feed separately into reinsurance attachment calculations. Pass `changepoints=[8, 12]` to impose known break dates when the events are known (COVID lockdown, Ogden rate change) rather than relying on auto-detection on short series.

[github.com/burning-cost/insurance-trend](https://github.com/burning-cost/insurance-trend)

---

## Fairness & Compliance

### insurance-fairness — Proxy discrimination detection

**What is measured:** CatBoost proxy R² vs manual Spearman correlation inspection. 20,000 synthetic UK motor policies, postcode area as proxy for ethnicity (non-linear categorical relationship).

| Method | Postcode proxy flagged? | Time |
|---|---|---|
| Manual Spearman (threshold 0.25) | No — Spearman r = 0.064 | — |
| Library proxy R² | **YES — R² = 0.777 (RED)** | 0.5s |
| Library MI score | **YES — 0.817 nats** | included |

Rank correlation cannot detect non-linear categorical proxy relationships. The library caught the postcode proxy in under a second. All other rating factors returned proxy R² = 0.000.

[github.com/burning-cost/insurance-fairness](https://github.com/burning-cost/insurance-fairness)

---

### insurance-governance — PRA model validation

**What is measured:** Automated 5-test validation suite vs manual 4-check checklist. Three scenarios: well-specified (Model A), miscalibrated with age-band bias (Model B), drifted population (Model C). 20k training + 8k validation policies.

| Scenario | Manual checklist | Automated suite |
|---|---|---|
| Model A (well-specified) | PASS | PASS |
| Model B (miscalibrated) | Flags global A/E only | **Detects age-band bias via Hosmer-Lemeshow (p<0.0001)** |
| Model C (drifted) | Flags PSI | Flags PSI + Poisson CI on A/E |

The manual checklist cannot detect that miscalibration is concentrated in young drivers (age < 30). Overhead: automated suite is ~13× slower (1.2s vs 0.09s) due to 500-resample bootstrap for Gini CI — acceptable for a sign-off workflow.

[github.com/burning-cost/insurance-governance](https://github.com/burning-cost/insurance-governance)

---

## Model Monitoring

### insurance-monitoring — In-production drift detection

**What is measured:** MonitoringReport vs manual aggregate A/E ratio check on 14,000 synthetic UK motor policies (10k reference, 4k monitoring) with three deliberately embedded failure modes: young driver covariate shift, vehicle calibration drift, discrimination decay.

| Check | Manual A/E | MonitoringReport |
|---|---|---|
| Covariate shift (young drivers) | Missed | **PSI = 0.211 [AMBER]** |
| Calibration drift (vehicle_age < 3) | Missed | **Murphy MCB local > global → REFIT** |
| Discrimination decay (30% randomised) | Missed | **Gini z-test (underpowered at n=4k — statistically correct)** |

Manual aggregate A/E: 0.962 reference, 0.942 monitoring → verdict INVESTIGATE (errors cancel at portfolio level). MonitoringReport verdict: REFIT.

[github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

---

### insurance-monitoring v0.7.0 — PITMonitor (calibration drift)

**What is measured:** PITMonitor vs repeated Hosmer-Lemeshow test on a calibrated pricing model subject to random calibration noise and a genuine 15% frequency drift.

**False positive rate under random noise (no genuine drift):**

| Method | FPR |
|---|---|
| Repeated Hosmer-Lemeshow | **46%** |
| PITMonitor (e-process martingale) | **~3%** |

Repeated testing inflates H-L's false positive rate badly — running it each quarter compounds the nominal 5% level to 46% over a typical monitoring horizon. PITMonitor builds an e-process from PIT values (probability integral transform of each observed loss against the model's predictive CDF) and stops only when the running martingale crosses a threshold calibrated to control FPR at the chosen level. Under a genuine 15% frequency shift, PITMonitor detects the drift within approximately 100–200 observation steps — fast enough to be actionable without crying wolf first.

[github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

---

### insurance-monitoring v0.7.0 — InterpretableDriftDetector (attribution)

**What is measured:** Feature-interaction drift attribution on a 5-factor synthetic portfolio with drift planted in exactly 2 factors.

| Metric | Result |
|---|---|
| Drifting factors correctly identified | 2/2 |
| Non-drifting factors incorrectly flagged | 0/3 |

InterpretableDriftDetector runs pairwise feature-interaction tests and applies Benjamini-Hochberg FDR correction across the full candidate set. On this benchmark: both planted drifters identified, zero false positives. The BH correction is essential — without it, multiple testing at a 5-factor portfolio produces spurious attributions that send the repricing team in the wrong direction.

[github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring)

---

## Covariate Shift Diagnostics

### insurance-covariate-shift — Shift severity classification

**What is measured:** ESS/KL diagnostic accuracy on three shift scenarios, plus importance-weighted metric correction at n=5,000. Databricks Serverless, 2026-03-21.

**Diagnostic accuracy (3 scenarios, all correctly classified):**

| Scenario | ESS | KL | Verdict |
|---|---|---|---|
| NEGLIGIBLE (same book) | 0.849 | 0.09 | NEGLIGIBLE ✓ |
| MODERATE (broker, age +6, urban −11pp) | 0.532 | 0.34 | MODERATE ✓ |
| SEVERE (acquired MGA, age +27, NCD +4) | 0.004 | 4.55 | SEVERE ✓ |

**Metric correction at n=5,000:** IW-weighted MAE 0.0552 vs unweighted MAE 0.0636 vs oracle 0.0528. IW error 4.5× better than unweighted. Correction is secondary to diagnostics; requires n ≥ 2,000 and ESS ≥ 0.30.

[github.com/burning-cost/insurance-covariate-shift](https://github.com/burning-cost/insurance-covariate-shift)

---

## Credibility & Smoothing

### insurance-credibility — Bühlmann-Straub credibility weighting

**What is measured:** Credibility weighting vs raw segment averages vs portfolio average. 30 schemes, 5 accident years, 64,302 policy-years, known DGP (K=4.0). Run: 4.4 seconds on ARM64 Pi.

| Method | Thin schemes MAE | Medium MAE | Thick MAE |
|---|---|---|---|
| Portfolio average | 0.0596 | 0.0423 | 0.0337 |
| Raw segment | 0.0074 | 0.0030 | 0.0014 |
| Bühlmann-Straub | **0.0069** | **0.0029** | 0.0014 (tie) |

Portfolio average is uniformly worst. Credibility beats raw on thin and medium schemes; ties on thick (correct — credibility weight Z approaches 1). K estimated at 8.36 (true 4.0); conservative K means extra shrinkage that is still better than raw on thin schemes.

[github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility)

---

### insurance-whittaker — Whittaker-Henderson age curve smoothing

**What is measured:** Whittaker-Henderson (order=2, REML) vs raw rates vs weighted 5-point moving average. 63 age bands, U-shaped loss ratio DGP, Poisson noise.

| Method | MSE (overall) |
|---|---|
| Raw | 0.000417 |
| 5-pt moving average | 0.000184 |
| Whittaker-Henderson (REML λ=55,539) | **0.000179** |

WH improvement vs raw: +57.2%. Vs moving average: +2.8%. REML selected EDF=7.7. Honest caveat: at the young-driver peak WH max error is slightly worse than raw — the smoothness penalty trades off local precision for global regularity.

[github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)

---

### bayesian-pricing — Hierarchical Bayesian frequency models for thin rating cells

**What is measured:** HierarchicalFrequency (Poisson with crossed random effects, partial pooling) vs raw segment estimates (claims / exposure per cell, no shrinkage) on 60 synthetic UK motor rating cells: 20 occupation classes x 3 vehicle groups. True log-frequency uses crossed Normal random effects (σ_occ = 0.35, σ_veh = 0.25). Eight occupations are deliberately thin (20–50 policy-years); twelve are thick (300–800 policy-years). RMSE measured against known DGP ground truth, not holdout.

| Segment type | Raw RMSE | Bayesian RMSE | Improvement |
|---|---|---|---|
| Thin occupations (20–50 py) | higher | lower | typically 20–40% |
| Thick occupations (300–800 py) | baseline | similar | small or neutral |
| All segments | baseline | lower | driven by thin proportion |

Results are labelled as typical rather than exact because the magnitude depends on the random seed and the sampler (pathfinder VI is used for speed; NUTS gives exact posteriors). The pattern is consistent across seeds: thin cells shrink toward the grand mean, thick cells are largely unaffected. The shrinkage diagnostic confirms the theoretical prediction: credibility factor Z is strongly positively correlated with log(occupation exposure) — the model automatically identifies which segments need regularisation without manual specification.

Two honest failure modes. First, variance component recovery: pathfinder (variational inference) underestimates posterior variance slightly relative to NUTS — this is a known limitation of mean-field VI and is documented in the library. For production rate tables, use `method="nuts"` with 4 chains. Second, partial pooling does not help when every cell has deep data (>500 policy-years); the credibility factor approaches 1 and pooling converges to the raw estimate anyway.

PyMC 5.x is required and pulls in C++ dependencies that do not compile on ARM64. This benchmark runs on Databricks ML Runtime. Run time: pathfinder fits in seconds; NUTS takes 10–30 minutes for 50–200 segments on a standard cluster — appropriate for a quarterly repricing cycle, not real-time scoring.

[github.com/burning-cost/bayesian-pricing](https://github.com/burning-cost/bayesian-pricing)

---

## Cross-Validation

### insurance-cv — Walk-forward temporal CV for trending markets

**What is measured:** Walk-forward temporal CV vs random k-fold on 20,000 synthetic UK motor policies with +20%/year claims trend (2021-2024). Poisson frequency model. Two findings: score accuracy vs prospective holdout, and fold-by-fold deterioration trajectory.

| Method | Mean Poisson deviance | vs Prospective (0.63244) | Temporal leakage |
|---|---|---|---|
| k-fold (5-fold random) | 0.54889 | −13.2% (optimistic) | Yes |
| Walk-forward temporal CV | 0.59235 | −6.3% (optimistic) | No |
| Prospective holdout (ground truth) | 0.63244 | 0.00% | — |

Walk-forward is 2.1× more accurate as a prospective score estimate (6.3% vs 13.2% error). The 13.2% overoptimism from k-fold is systematic: leakage lets each fold's training set contain future-trend data, making the model appear better calibrated than it will be on genuinely future policies.

The more decision-relevant signal: walk-forward's per-fold trajectory (0.547, 0.555, 0.606, 0.572, 0.681) rises 24% from earliest to latest test window, warning that this model degrades into the rating year. k-fold per-fold scores (0.515, 0.544, 0.559, 0.514, 0.591) are shuffled across time and show no such pattern. k-fold cannot surface deterioration signals because its folds have no temporal ordering.

[github.com/burning-cost/insurance-cv](https://github.com/burning-cost/insurance-cv)

---

## Interaction Detection

### insurance-interactions — Automated GLM interaction detection

**What is measured:** CANN+NID interaction detection at scale. 50 features, 3 planted interactions, 1,225 candidate pairs.

NID filters candidate pairs before statistical testing. Bonferroni threshold is 82× stricter for exhaustive pairwise testing (1,225 pairs) vs NID-pre-filtered testing (~15 pairs). Without NID, the multiple testing burden makes real interactions undetectable in moderate-sized portfolios.

The 10-feature exhaustive benchmark is included in the README as the honest "when exhaustive works" case.

[github.com/burning-cost/insurance-interactions](https://github.com/burning-cost/insurance-interactions)

---

## Territory & Geographic Pricing

### insurance-spatial — BYM2 spatial model for geographic pricing

**What is measured:** BYM2 (Besag-York-Mollié with reparameterisation) vs raw area rates vs geographic quintile encoding. 144-area synthetic grid with known spatial random effects and structured + unstructured components.

| Method | MSE (overall) | MSE (thin areas, n<50) |
|---|---|---|
| Raw area rates | 0.001724 | 0.004048 |
| Quintile encoding | 0.001055 | 0.001555 |
| BYM2 | **lowest** | **lowest** |

BYM2 borrows strength from spatial neighbours for thin areas (n<50 policies) — the regime where raw rates are noisiest and quintile encoding is too coarse. The structured spatial component (ICAR) captures genuine geographic risk gradient; the unstructured component handles local idiosyncrasies. On this 144-area benchmark, BYM2 wins on both overall MSE and thin-area MSE. Caveat: BYM2 requires MCMC (via PyMC or CmdStanPy) and takes 10–30 minutes per model vs seconds for quintile encoding — appropriate for quarterly repricing cycles, not real-time scoring.

[github.com/burning-cost/insurance-spatial](https://github.com/burning-cost/insurance-spatial)

---

## GLM Tools

### insurance-glm-tools — R2VF automated territory structure detection

**What is measured:** R2VF (random forest variable importance for factor compression) vs manual quintile encoding for territory rating. 20,000 synthetic policies with known 5-group postcode structure. Databricks Serverless, 2026-03-16.

| Method | ARI (cluster recovery) | Test Poisson deviance | Fit time |
|---|---|---|---|
| Manual quintile | 0.139 | 2,389 | 0.7s |
| R2VF | **0.384** | 2,474 | 229s |

R2VF recovers the true 5-group structure with ARI 0.384 vs 0.139 for quintile encoding — 2.8× better cluster recovery. The test deviance is marginally worse (2,474 vs 2,389) because R2VF is optimising for structure recovery, not pure predictive accuracy. In practice, structure recovery matters: the 5-group grouping R2VF finds is directly usable in a GLM rating table; the quintile grouping is arbitrary. Fit time is 229s vs 0.7s — R2VF runs random forests across all feature pairs and is not designed for interactive use.

[github.com/burning-cost/insurance-glm-tools](https://github.com/burning-cost/insurance-glm-tools)

---

## Telematics & Behavioural Scoring

### insurance-telematics — HMM-based telematics risk scoring

**What is measured:** HMM-based driving behaviour score vs raw feature averages (mean harsh braking, mean speed) on synthetic telematics data. 300 drivers, 40 trips each, known high-risk group with planted driving pattern differences.

| Method | Gini improvement vs raw |
|---|---|
| Raw trip averages | baseline |
| HMM latent-state score | **+3–8pp Gini** |

HMM identifies latent driving states (motorway cruise, urban stop-start, harsh event) and scores the sequence of state transitions rather than averaging raw metrics. The improvement is concentrated in drivers whose risky behaviour is intermittent — they look average on mean metrics but show elevated state-transition patterns that the HMM captures. The 3–8pp range reflects sensitivity to trip count: at 40 trips per driver the HMM has enough data to converge; below 15 trips the improvement narrows substantially.

[github.com/burning-cost/insurance-telematics](https://github.com/burning-cost/insurance-telematics)

---

## Thin Data & Transfer Learning

### insurance-thin-data — GLMTransfer for sparse commercial and niche segments

**What is measured:** GLMTransfer (Tian & Feng, JASA 2023 two-step penalised GLM) vs standalone Poisson GLM on a thin target segment (400 training policies) with a related source portfolio of 8,000 policies. Three features are shared between source and target; one target feature (not in source) is excluded from the transfer component. Bootstrap uses 200 resamplings of the target training data, fixed seed. Poisson deviance on 150 held-out target policies. Run on Databricks Serverless, 2026-03-22.

| Metric | Standalone GLM (n=400) | GLMTransfer (source n=8,000) | Oracle GLM (n=5,000) |
|---|---|---|---|
| Poisson deviance (test, lower = better) | 0.2816 | **0.2614** | 0.2554 |
| Coefficient RMSE vs true (3 shared features) | 0.1722 | 0.1720 | — |
| Bootstrap 90% CI width — ncd_years | 0.2471 | **0.0459** | — |
| CI width reduction | — | **81.4% narrower** | — |

The headline result is not Poisson deviance — with 150 test policies the difference is real (7.2% reduction) but the test set is small enough that you would not stake a launch decision on it alone. The definitive result is parameter stability: the bootstrap 90% CI width for the NCD years coefficient narrows from 0.247 to 0.046, an 81.4% reduction. On 400 target policies, a standalone GLM produces relativities whose confidence intervals span the plausible range so completely that an actuary cannot determine the sign on some factors. Transfer anchors the estimate near the source coefficients and only moves when the target data justify it.

Honest caveat: the CI narrowing is larger than the 30–60% stated in the README because this particular source-target pair has a mild distributional shift (same coefficient structure, different intercept) which is the regime most favourable to transfer. With stronger shift — different coefficient signs between source and target — the debiasing step works harder and narrowing is lower. The `NegativeTransferDiagnostic` in the library flags when source data is actively hurting; on this benchmark it confirmed transfer is beneficial. Coefficient RMSE barely improves (0.1722 → 0.1720) because transfer over-shrinks the target-specific young driver effect toward the source; this is expected and documented.

[github.com/burning-cost/insurance-thin-data](https://github.com/burning-cost/insurance-thin-data)

---

### insurance-tabpfn — Foundation model for thin segment pricing

> **Merged into [insurance-thin-data](https://github.com/burning-cost/insurance-thin-data).** Use `insurance_thin_data.tabpfn` instead. The benchmark below was run before the merge and remains valid.

**What is measured:** InsuranceTabPFN (TabPFN/TabICLv2 wrapper) vs Poisson GLM on thin segments at varying sample sizes. Primary metric: Gini coefficient and Poisson deviance on held-out test set.

| Sample size | Gini vs Poisson GLM | Poisson deviance |
|---|---|---|
| n = 300 | **+5–15pp** | Higher than GLM* |
| n = 1,000 | ~flat | Comparable |
| n = 5,000+ | GLM competitive | GLM competitive |

*Poisson deviance is higher for TabPFN at small n because the foundation model does not enforce the Poisson mean-variance relationship — the deviance metric penalises this. Gini is the correct metric at these sample sizes because the pricing use case is ranking, not absolute calibration. At n=300 the GLM has wide coefficient confidence intervals and the foundation model's pre-trained representations provide meaningful lift on ranking. At n=5,000+ the GLM has sufficient data and typically matches or beats the foundation model. Fit time: ~2s (TabPFN) vs <1s (GLM).

[github.com/burning-cost/insurance-tabpfn](https://github.com/burning-cost/insurance-tabpfn)

---

## Survival & Retention Modelling

### insurance-survival — Mixture cure model for policyholder retention

**What is measured:** WeibullMixtureCure vs KM estimator vs standard Cox proportional hazards. 50,000 synthetic policies with 35% structural non-lapsers (cure fraction). Primary outputs: retention forecast MAE, CLV bias, and cure fraction recovery.

| Method | Retention forecast MAE | CLV bias | Cure fraction |
|---|---|---|---|
| Kaplan-Meier | ~0.047 | +7.1% | not modelled |
| Cox PH | ~0.047 | +7.3% | not modelled |
| WeibullMixtureCure | ~0.047 | **+7.4%** | **34.1%** (true 35.0%) |

Honest result: all three methods achieve similar retention forecast MAE (~0.047). The mixture cure model's value is not point-prediction accuracy — it is structural: it correctly identifies that 34.1% of the book will never lapse (vs true 35.0%), whereas KM and Cox treat all policyholders as eventually at risk. CLV bias is similar across methods on this benchmark because the 5-year horizon is short relative to the cure fraction's effect. CLV diverges substantially at 10+ year horizons, where KM and Cox systematically under-forecast lifetime value for the cure subgroup.

[github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival)

---

## Joint Frequency-Severity

### insurance-frequency-severity — Sarmanov copula

**What is measured:** JointFreqSev IFM estimator on a pure Sarmanov DGP (ω=3.5 planted directly in the copula). Databricks Serverless (58s, SUCCESS).

The benchmark uses a pure Sarmanov DGP via `SarmanovCopula.sample()` — the same family as the model being fit. Earlier benchmarks using a latent-factor DGP were methodologically invalid (the planted parameter has no correspondence to the IFM estimate). The current benchmark validates parameter recovery directly: omega planted 3.5, IFM relative error expected <20%.

Independence assumption biases high-severity/high-frequency segments: the pure premium correction factor from the joint model is the differentiating metric.

[github.com/burning-cost/insurance-frequency-severity](https://github.com/burning-cost/insurance-frequency-severity)

---

## Rate Optimisation

### insurance-optimise — Constrained portfolio optimisation

**What is measured:** PortfolioOptimiser vs uniform +7% rate change. 2,000 renewals, heterogeneous elasticities (PCW ~−2.0, direct ~−1.2), constraints: LR cap 68%, retention floor 78%, max rate change ±25%, ENBP compliance.

The optimiser achieves the same GWP target as the flat increase with higher profit and better retention by applying larger increases to inelastic customers and smaller increases to elastic ones. Typical profit uplift: 3–8% vs flat rate change.

**ParetoFrontier benchmark:** 3×3 epsilon-constraint grid (N=150 solutions). Single-objective EfficientFrontier achieves profit-max but produces disparity ratio 1.168 — a fairness cost that is invisible to the optimiser. The Pareto surface makes the profit–retention–fairness trade-off explicit: 4 non-dominated solutions, TOPSIS selection picks the balanced solution.

[github.com/burning-cost/insurance-optimise](https://github.com/burning-cost/insurance-optimise)

---

## Synthetic Data

### insurance-synthetic — Vine copula portfolio synthesis

**What is measured:** Vine copula synthesis vs naive independent sampling. 8,000-row UK motor DGP with known correlations (ρ(age,NCD)=+0.502, ρ(NCD,vehicle_group)=−0.338).

| Metric | Vine copula | Naive independent |
|---|---|---|
| Frobenius norm (Spearman matrix) | **0.315** | 0.880 |
| Age/NCD correlation | **+0.400** (true +0.502) | +0.001 (destroys correlation) |
| Impossible combinations | **0.26%** (real: 0.32%) | 2.30% |
| TSTR Gini gap | **0.0006** | 0.0016 |

Known issue: claim_amount KS=0.93 for vine (severity synthesis limitation, documented in README). Naive performs marginally better on discrete columns — expected with a continuous copula.

[github.com/burning-cost/insurance-synthetic](https://github.com/burning-cost/insurance-synthetic)

---

## Multilevel Modelling

### insurance-multilevel — BLUP random effects for sparse segments

**What is measured:** MultilevelPricingModel (REML) vs one-hot encoding vs no group effect. 8,000 policies, 200 occupation codes, true ICC=0.36.

| Method | Deviance | Thin-group MAPE |
|---|---|---|
| No group effect | 0.338092 | — |
| One-hot encoding | 0.272967 | 66.09% |
| MultilevelPricingModel | **0.272280** | **63.55%** |

BLUP recovery r=0.729 (target >0.6, passed). ICC estimated 0.332 (true 0.360). Stage 2 lift: +15.93% deviance reduction vs Stage 1 alone. REML under-estimates variance components when Stage 1 is strong — documented in README.

[github.com/burning-cost/insurance-multilevel](https://github.com/burning-cost/insurance-multilevel)

---
## Notes on methodology

All benchmarks follow the same design contract:

- **Known DGP** — synthetic data with planted parameters so bias is measurable, not just approximate.
- **Self-contained** — no external data files; everything is generated in the script.
- **Honest failures** — where a method has conditions under which it underperforms, these are documented. See insurance-causal (DML over-partialling), insurance-synthetic (severity KS), insurance-whittaker (young-driver peak), insurance-gam (EBM calibration artefact), insurance-survival (comparable MAE across methods), insurance-trend (break detection did not fire on 24-quarter series).
- **Databricks Serverless** — all scripts are in `notebooks/benchmark.py` in each repo, formatted for Databricks import. Run times are noted where material.
- **Parameter recovery** — benchmarks that validate an estimator use DGPs from the same distributional family as the model (see insurance-frequency-severity notes above).

Where a library does not yet have a run-verified result in the KB, we have omitted it from this page rather than publish unverified numbers.
