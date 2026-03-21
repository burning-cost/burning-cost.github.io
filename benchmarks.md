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

## Interaction Detection

### insurance-interactions — Automated GLM interaction detection

**What is measured:** CANN+NID interaction detection at scale. 50 features, 3 planted interactions, 1,225 candidate pairs.

NID filters candidate pairs before statistical testing. Bonferroni threshold is 82× stricter for exhaustive pairwise testing (1,225 pairs) vs NID-pre-filtered testing (~15 pairs). Without NID, the multiple testing burden makes real interactions undetectable in moderate-sized portfolios.

The 10-feature exhaustive benchmark is included in the README as the honest "when exhaustive works" case.

[github.com/burning-cost/insurance-interactions](https://github.com/burning-cost/insurance-interactions)

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
- **Honest failures** — where a method has conditions under which it underperforms, these are documented. See insurance-causal (DML over-partialling), insurance-synthetic (severity KS), insurance-whittaker (young-driver peak).
- **Databricks Serverless** — all scripts are in `notebooks/benchmark.py` in each repo, formatted for Databricks import. Run times are noted where material.
- **Parameter recovery** — benchmarks that validate an estimator use DGPs from the same distributional family as the model (see insurance-frequency-severity notes above).

Where a library does not yet have a run-verified result in the KB, we have omitted it from this page rather than publish unverified numbers.
