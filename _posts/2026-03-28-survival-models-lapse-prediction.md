---
layout: post
title: "Survival Models for Insurance Lapse Prediction: What Actually Works"
date: 2026-03-28
categories: [techniques, research]
tags: [survival-analysis, lapse, cure-models, DeepSurv, DeepHit, lifelines, scikit-survival, pycox, torchsurv, GIPP, consumer-duty, CLV, retention, python, tutorial]
description: "Deep learning survival models underperform Cox regression on tabular insurance data. Cure models are the real story post-GIPP. Here is what the research says and what UK pricing teams should actually do."
seo_title: "Survival Models for Insurance Lapse Prediction: Cure Models, DeepSurv, and What Works in Python"
---

Most UK personal lines pricing teams model lapse as a binary event. Will this policyholder renew? Yes or no. They fit logistic regression on a 12-month outcome window, generate propensity scores, and use those to drive retention campaigns. This is not wrong, but it is incomplete — and post-GIPP it leaves commercial value on the table.

The right framing is time-to-event. Not will they lapse, but when. A policyholder with a 40% 1-year lapse probability is very different from one who survives 4 years and then lapses: the CLV calculation that drives your discount decision is entirely different. Survival analysis was designed for this problem. The question is which survival methods are worth using in a production UK pricing context.

We have spent time going through the recent literature on deep learning survival models, the empirical benchmarks, and the post-GIPP data from FCA EP25/2. The conclusion is clear enough to state plainly: deep learning survival models (DeepSurv, DeepHit, NeuralSurv, SurvUnc) underperform classical methods on standard insurance tabular data. Cure models are the genuine methodological story, and they have become more important, not less, since the loyalty penalty ban came into force in January 2022.

---

## Why binary classification is the wrong model

When you fit logistic regression on a 12-month lapse indicator, you make three choices implicitly, without necessarily noticing them.

First, you decide that the timing of lapse within the year carries no information. A policyholder who cancels mid-term in month 2 gets the same label as one who shops around at renewal but ultimately stays. The survival function distinguishes these; logistic regression does not.

Second, you handle censored policies badly. Policies that are mid-term at your observation cutoff have not lapsed, but they have not been observed through to renewal either. The standard approaches — exclude them, or treat them as non-lapsers — both introduce bias. Survival likelihood handles censoring correctly by construction.

Third, you commit to a single horizon. Your model answers one question: "will this policyholder lapse in the next 12 months?" To answer "what is the CLV of this policyholder over 5 years?", you either need five separate models or a survival model. Post-PS21/11, where your only renewal pricing lever is how much to discount below ENBP, multi-period CLV is the decision variable. The CLV formula is:

```
CLV(x) = Σ_{t=1}^{T} S(t|x) * (P_t - C_t) * (1+r)^{-t}
```

You cannot compute this without `S(t|x)` — a survival function that gives you retention probability at any horizon. A 12-month propensity score produces a CLV approximation, not a CLV calculation.

Survival analysis is not superior to logistic regression on discrimination: C-index values for Cox PH on typical insurance datasets run around 0.60–0.65, which is similar to what a well-specified logistic regression achieves. The advantage is calibration across multiple horizons and correct handling of censoring and the cure subgroup.

---

## When logistic regression is adequate

Being honest here: for pure 1-year renewal scoring — ranking policyholders by lapse propensity for retention campaign prioritisation — logistic regression with tenure, NCD level, premium change, and channel interactions is competitive. You do not need Cox regression to generate a decile of high-lapse-risk customers. The AUC difference will be small.

Survival analysis becomes materially better in these situations:

- You need multi-period CLV for discount optimisation under GIPP
- You have a genuine cure subgroup (structural non-lapsers) whose survival curve plateaus rather than going to zero
- You have informative censoring — mid-term cancellations that are not random with respect to lapse propensity
- You need calibrated retention curves beyond the 1-year horizon
- You are dealing with competing risks: voluntary non-renewal, mid-term cancellation, and total loss are different events that warrant separate treatment

For a UK motor book with 5+ years of policy history, all five of these apply. You have cure-fraction loyalists (high NCD, direct debit, long tenure). Mid-term cancellations correlate with price sensitivity. And CLV is the correct optimisation target under PS21/11.

---

## Classical survival methods: the honest benchmark

The reference benchmark we are working from is PMC11531126, which evaluated Cox PH, Random Survival Forest, and DeepHit across 11 high-dimensional datasets:

| Method | C-index (avg) | IBS (avg) |
|---|---|---|
| Cox PH | 0.799 | 0.165 |
| Random Survival Forest | 0.817 | — |
| DeepHit (default) | 0.624 | 0.197 |

Cox PH wins on discrimination against DeepHit out of the box. RSF wins on C-index. DeepHit is the worst-performing method on both discrimination and calibration. These are not cherry-picked results — they are averages across 11 datasets with default or lightly tuned hyperparameters.

For insurance tabular data (30–50 features, annual structure, no sequential measurement), our recommendation is:

1. **Cox PH** (via lifelines or scikit-survival) as baseline. Fits quickly, interpretable coefficients, handles tied event times, good diagnostics.
2. **GradientBoostingSurvivalAnalysis** from scikit-survival if you have reason to believe non-linearities are important. C-index of 0.70–0.76 on non-linear datasets — better than Cox PH at the cost of interpretability.
3. **WeibullMixtureCure** (insurance-survival) when the cure subgroup is present. More on this below.

Do not start with DeepSurv or DeepHit on a standard tabular lapse dataset. They require careful tuning, are harder to deploy and explain, and will not outperform Cox PH on typical personal lines feature sets.

---

## The deep learning reality check

### DeepSurv

DeepSurv (Katzman et al., 2018, arXiv:1606.00931) replaces the linear predictor in Cox PH with a feed-forward neural network. The loss function is identical to Cox — negative partial log-likelihood. The claim is that it learns non-linear covariate-hazard relationships without manual feature engineering.

The claim is true in theory and narrow in practice. When the true data-generating process has non-linear risk functions, DeepSurv outperforms linear Cox. When risk is approximately linear — which is the realistic case for standard personal lines tabular data — DeepSurv performs similarly to Cox PH at 10–100x the deployment complexity.

There is also a maintenance problem. The original repository (jaredleekatzman/DeepSurv on GitHub) is archived and has had no updates since 2020. If you want a working DeepSurv implementation, use pycox or torchsurv — but the case for doing so on a 40-feature motor dataset is weak.

Where DeepSurv might genuinely add value: telematics time-series features, high-cardinality behavioural data, raw app-usage sequences. When the feature space is high-dimensional and structured in ways that benefit from representation learning, the neural architecture earns its complexity. Standard price/NCD/tenure tabular data does not.

### DeepHit

DeepHit (Lee et al., AAAI 2018) is the dominant discrete-time deep survival model. Unlike DeepSurv, it directly learns the joint distribution `P(T=t, K=k)` over discrete time steps and competing event types — no proportional hazards assumption, and competing risks modelled natively.

The competing risks capability is genuinely useful. Voluntary non-renewal, mid-term cancellation, and total loss are distinct events; modelling them jointly without the Fine-Gray proportionality assumption is methodologically sound.

The benchmark performance is not. C-index of 0.624 against Cox PH's 0.799 in PMC11531126. The explanation is that DeepHit requires careful tuning of the time discretisation (bin count) and the ranking loss weight `alpha`. Default hyperparameters perform poorly. A well-tuned DeepHit can match RSF on specific datasets, but this requires significant effort that is hard to justify when Fine-Gray regression (available in insurance-survival) provides competing risk modelling with better calibration and direct interpretability for actuaries and regulators.

The canonical Python implementation is pycox (havakv/pycox). It is functional but slow to maintain — active issues unresolved since 2022. If you need dynamic covariate handling (relevant for telematics), Dynamic-DeepHit is the extension, also in pycox.

### NeuralSurv and SurvUnc: the UQ papers

Two May 2025 preprints address uncertainty quantification for survival models.

NeuralSurv (arXiv:2505.11054) embeds Bayesian UQ into deep survival via variational inference. Two augmentation tricks — Polya-Gamma augmentation for the partial likelihood sigmoid and marked Poisson process augmentation for the baseline hazard — make otherwise intractable Bayesian updates tractable. The result is closed-form CAVI updates that scale linearly in network size. Theoretically, this is superior to MC Dropout (computationally cheap but not a true posterior) and deep ensembles (reliable but 10+ training runs). In benchmark results it shows superior calibration over DeepSurv and DeepHit.

SurvUnc (arXiv:2505.14803, KDD 2025) takes a different approach: post-hoc uncertainty scores without modifying the underlying model. A meta-model is trained on the same covariates; its labels capture how often the base model incorrectly ranks a subject relative to anchor samples. Higher label = higher uncertainty = less reliable prediction. In SEER-BC experiments, discarding the 50% most uncertain predictions improved C-index from 0.842 to 0.961 — a substantial gain from selective prediction.

Both papers have meaningful limitations for insurance. Neither has been validated on insurance or churn data — all experiments are on medical datasets. Both rely on a missing-completely-at-random censoring assumption that is routinely violated in UK insurance: mid-term cancellations correlate with price sensitivity, which correlates with lapse propensity. Neither has a pip-installable implementation. Production use requires substantial adaptation work.

The SurvUnc selective prediction concept is directly applicable in principle: use uncertainty scores to decide which policies get automated CLV-based retention offers versus manual review. But proving that the approach is correctly calibrated on your specific data is a research task before it is a production task.

---

## Cure models: the real innovation post-GIPP

This is the part that matters most for UK personal lines pricing teams right now.

In any mature motor or home book, a significant fraction of policyholders will not voluntarily lapse. They have maximum NCD, pay by direct debit, have been with the insurer for 7+ years, and their switching probability is near zero even at meaningful price differentials. In our benchmarks with synthetic motor data (50k policies, 5-year window, known data-generating process), the structural cure fraction is around 30%. Empirical UK motor data suggests 20–30% of customers with 5+ years tenure have near-zero switching probability.

Standard survival models — Cox PH, Weibull AFT, Kaplan-Meier — force `S(∞) → 0`. They assume that, given enough time, every policyholder will eventually lapse. This is wrong for the loyalist subgroup, and the misspecification distorts hazard estimates for everyone else.

The mixture cure model corrects this:

```
S(t|x) = π(x) + (1 - π(x)) * S_u(t|x)
```

where `π(x)` is the probability of being in the "cured" (never-lapse) subgroup — modelled by logistic regression on covariates — and `S_u(t|x)` is the survival function for the susceptible subgroup only, which follows a proper parametric distribution (Weibull, log-logistic, log-normal). The full survival curve plateaus at `π(x)` rather than going to zero.

### What the benchmarks show

From our Databricks benchmark on 15,000 synthetic motor policies (5-year window, 30% true cure fraction, Weibull(1.2, 36-month) latency for susceptibles):

| Method | C-index | 5-yr CLV bias | Notes |
|---|---|---|---|
| Kaplan-Meier | ~0.50 | Underestimates | Non-parametric, no covariates |
| Cox PH | ~0.62 | ~£40–80/policy | Drives toward zero at long horizons |
| WeibullMixtureCure | ~0.62 | <£5/policy | Recovers cure fraction to within 1–2pp |

The concordance difference between Cox PH and the cure model is negligible. If ranking policyholders by lapse risk is your only goal, Cox PH is fine. The cure model's advantage is **calibration** — getting the survival curve at the right level — and that advantage compounds at longer horizons.

At a 10,000-policy scale, the CLV mis-valuation from ignoring the cure fraction runs to £400k–£800k. That is the order of magnitude of error in book value assessments and retention budget allocation under Consumer Duty fair value obligations.

### The GIPP connection

FCA EP25/2 (2025) evaluated the GIPP remedies three years after implementation. Key findings relevant to survival modelling:

- Policies held for 4–6 years rose from 15% to 18% of the portfolio — structural loyalists stayed longer post-GIPP
- Auto-renewal at inception jumped from 54% to 70% — more policies set up as auto-renewing from day one
- The renewal hazard function has shifted: pre-2022, hazard peaked at years 3–5 (when price walking became material). Post-2022, the hazard is flatter for long-tenure customers

The implication is that the cure fraction has increased post-GIPP. Customers who previously left due to loyalty penalty are now staying. A survival model trained on pre-2022 data will overestimate hazard for long-tenure policyholders. If your model was last fitted on data spanning 2018–2023, the pre-2022 cohorts are contaminating your hazard estimates.

The practical response: train on post-January 2022 data only, or use a changepoint model that allows for a structural break in the baseline hazard at the GIPP implementation date.

### The Python gap and what fills it

lifelines' `MixtureCureFitter` is univariate only — it fits a single cure fraction scalar with no covariate support. To model *which* customers are in the cured subgroup (NCD, tenure, channel), you need a covariate-adjusted cure model. R has `flexsurvcure` and `smcure` for this. Python did not, until insurance-survival.

`WeibullMixtureCure` in insurance-survival fits the mixture cure model via EM algorithm with L-BFGS-B for the M-step. The API follows lifelines conventions:

```python
from insurance_survival.cure import WeibullMixtureCure

model = WeibullMixtureCure(
    incidence_formula="ncd + tenure + channel",   # logistic for π(x)
    latency_formula="premium_change + ncd",        # Weibull AFT for S_u
)
model.fit(df, duration_col="months_to_event", event_col="lapsed")

# Per-policy cure probability
df["cure_prob"] = model.predict_cure_fraction(df)

# Survival curve at any horizon
df["s36"] = model.predict_survival_function(df, times=[36]).loc[36]
```

The `cure_prob` column is directly actionable: policies with `cure_prob > 0.7` are structural loyalists. They do not need retention discounts. Under GIPP, spending retention budget on this segment is waste.

---

## Python tooling: the honest landscape

The four main libraries for survival analysis in Python:

**lifelines (v0.29+)** — the default starting point. Best API, best documentation, parametric and semiparametric models, built-in plotting. Gaps: no deep learning, no Fine-Gray competing risks, no covariate-adjusted cure models (univariate only). For standard Cox PH and Weibull AFT on insurance tabular data, lifelines is the right tool.

**scikit-survival (v0.27+)** — adds gradient boosting and random survival forest with sklearn-compatible API. `GradientBoostingSurvivalAnalysis` achieves C-index 0.70–0.76 on non-linear datasets, meaningfully above Cox PH. `RandomSurvivalForest` runs at ~0.82 C-index on average. If you suspect non-linear risk structure and can accept less interpretability, add scikit-survival alongside lifelines. Gaps: no Fine-Gray, no cure models, no CLV integration.

**pycox (v0.3+)** — DeepSurv, DeepHit, LogisticHazard, CoxTime, MTLR. Built on PyTorch. Functional but slow to maintain; last major update around 2022. Use it if you specifically need one of these architectures.

**torchsurv (Novartis, v0.1.5, 2024)** — pure PyTorch toolkit. Unlike pycox, which ships fixed architectures, torchsurv gives you Cox and Weibull loss functions and metrics (time-dependent C-index, AUC, Brier score) to plug into your own PyTorch model. Listed in the FDA CDRH regulatory science tool catalog as of October 2025, which is the strongest production credibility signal available for a Python survival library. Use this if you are building custom deep survival architectures for telematics or high-dimensional feature spaces.

The recommended stack for UK insurance lapse modelling:

1. **insurance-survival** — primary. WeibullMixtureCure, FineGrayFitter, SurvivalCLV, MLflow deployment wrappers. Fills all the gaps that lifelines and scikit-survival leave open for insurance-specific use cases.
2. **scikit-survival** — add if non-linearity is present and GBM-level interpretability is acceptable.
3. **torchsurv** — only if telematics or other high-dimensional features justify the neural architecture overhead.

---

## Uncertainty quantification: what is achievable now

There is a hierarchy here, and it is worth being precise about which level is achievable with current tools.

**Level 1 — achievable now.** Calibration metrics on fitted models: time-dependent Brier score (IBS), C-index by tenure decile, actual-versus-expected retention curves. insurance-survival and scikit-survival both provide these. This is the minimum for Consumer Duty model governance documentation — the FCA's September 2024 guidance on fair value data analytics expects firms to "understand the uncertainty in those analytics." A model with good IBS and well-characterised A/E analysis by segment meets this bar.

**Level 2 — near-term with implementation effort.** Conformal prediction intervals on survival outputs. Three recent papers (arXiv:2412.09729 DR-COSARC; arXiv:2512.03738 weighted conformal under covariate shift; arXiv:2410.24136 two-sided conformal survival) establish the theory. The common challenge is the exchangeability assumption — survival data collected across time violates it when there are seasonal patterns, market hardening cycles, or structural breaks like GIPP. insurance-conformal-ts addresses this for claims time series; the extension to survival outputs requires manual implementation but is feasible.

**Level 3 — research.** SurvUnc meta-model or NeuralSurv Bayesian UQ. No pip-installable implementation. Medical-domain validation only. MCAR censoring assumption violated in insurance. Interesting methodology, not a production route in 2026.

The commercial case for Level 2 is genuine: if SurvUnc's result (50% uncertain-sample discard improving C-index from 0.842 to 0.961) transfers at all to insurance tabular data, the retention campaign implications are material. Spend budget on the predictions you trust; flag the uncertain ones for human review. This maps directly to how retention analysts already work — the uncertainty framing gives them a principled basis for prioritisation rather than a gut-feel override.

---

## Practical recommendations

For a UK personal lines pricing team building or upgrading their lapse modelling stack:

**If you have a working logistic regression and need quick wins:** Add tenure non-linearity (cubic spline or restricted cubic spline on log-tenure), NCD × premium change interaction, and proper censoring treatment (exclude mid-term cancellations from the denominator, or treat them as competing risk). You will improve calibration without changing model class.

**If you are ready to move to survival analysis:** Start with Cox PH in lifelines. Verify proportional hazards assumption using Schoenfeld residuals — if violated for tenure or NCD, stratify. Add scikit-survival's `GradientBoostingSurvivalAnalysis` as a challenger. Evaluate on time-dependent Brier score and C-index by tenure decile, not overall AUC.

**If you have 5+ years of post-2022 data:** Fit a mixture cure model. Estimate the cure fraction for your book. If it is above 20%, the cure model materially affects CLV estimates and you should be using it for discount optimisation. Use `WeibullMixtureCure` from insurance-survival.

**Do not train survival models on pre-2022 data without a GIPP structural break adjustment.** The hazard function shifted materially in January 2022. A single model spanning 2018–2025 is fitting a mixture of two different lapse regimes. Either train on post-2022 data only, or model the changepoint explicitly.

**On deep learning:** DeepSurv and DeepHit are not the answer for standard personal lines tabular data. The benchmark evidence is clear. If you have telematics sequences or other high-dimensional structured data, torchsurv with a custom architecture is the right route. For 40-feature motor tabular data, Cox PH + cure model + competing risks is more defensible, more accurate on calibration, and vastly easier to explain to Lloyd's, the FCA, or your model governance committee.

The post-GIPP world has made CLV calculation mandatory for rational discount optimisation. Survival analysis is the right mathematical framework for CLV. Cure models are the correct specification for a UK personal lines book where structural loyalists make up 20–30% of the portfolio. The tools are available. The benchmark evidence is clear enough to act on.

---

*insurance-survival is available at [github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival). The Databricks benchmark notebook demonstrating cure model vs Cox PH on synthetic motor data is in [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples).*
