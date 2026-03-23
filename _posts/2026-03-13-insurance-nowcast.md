---
layout: post
title: "Covariate-Conditioned IBNR Completion: Segment-Level Development Factors with insurance-nowcast"
date: 2026-03-13
categories: [libraries, pricing, reserving]
tags: [nowcasting, IBNR, EM-algorithm, XGBoost, reporting-delay, completion-factors, frequency-GLM, insurance-nowcast, python, poisson, multinomial]
description: "Covariate-conditioned IBNR completion by risk segment using ML-EM algorithm with XGBoost M-step. Exposure-weighted Poisson occurrence plus multinomial delay model - insurance-nowcast Python."
---

Aggregate LDFs from the quarterly triangle assume recent business has the same risk profile as the historical average. When the book has shifted — more fleet in the last 18 months, more high-NCD motor at the younger end of the age distribution — the aggregate factor will be wrong for every segment it is applied to. Fleet motor PD has a median reporting delay of around 2 months. Young driver motor BI runs closer to 4–5 months with a long right tail past 18 months. Averaging those two development patterns produces a factor that understates IBNR for the slower-reporting segment and overstates it for the faster one.

The subtler version of this failure is invisible in diagnostics. A Q4 frequency model built with aggregate LDF correction looks plausible — it doesn't flag as a model failure. The bias is towards the historical risk mix rather than the current one, and it compounds at the rate at which the book shifts.

[`insurance-nowcast`](https://github.com/burning-cost/insurance-nowcast) gives you covariate-conditioned completion factors: the development factor for a young driver motor BI claim reflects the reporting delay distribution for that segment, not the whole book.

[`insurance-nowcast`](https://github.com/burning-cost/insurance-nowcast) gives you covariate-conditioned completion factors. The development factor for a young driver motor BI claim reflects the reporting delay distribution for young driver motor BI claims, not for your whole book. 3,023 lines of source, 1,722 lines of tests, 171 tests passing. MIT-licensed, on PyPI.

```bash
uv add insurance-nowcast
```

---

## The model

The library implements the ML-EM framework from Wilsens, Antonio & Claeskens (KU Leuven), arXiv:2512.07335, December 2024. The authors' R implementation is on GitHub with three commits, no PyPI package, and no documentation beyond the paper itself. There is no existing Python implementation.

The model treats claim occurrence and reporting delay as a joint Poisson-Multinomial process. Each risk unit with features x generates claims at a rate λ(x) per unit of exposure — the occurrence process. Each claim then takes a random number of periods to be reported, with the delay distribution p(x) depending on those same risk features — the delay process. The key insight is that the delay distribution is covariate-conditioned: different kinds of risks have different reporting speed.

**The problem**: at any evaluation date, claims from recent occurrence periods are partially missing. You can observe claims that have already been reported, but you cannot observe claims that occurred and have not yet been reported (IBNR). The reported data is censored.

**The solution**: treat the unreported claim counts as latent variables and estimate them by expectation-maximisation.

---

## EM for actuaries who haven't seen it

EM is an algorithm for maximum likelihood estimation when some of the data is missing. In our case, the "missing data" is the unreported claims.

You want to find the parameters (λ, p) that maximise the likelihood of your observed data. But the likelihood of the observed data — claims that actually showed up — is complicated, because it involves integrating over all the ways the IBNR claims could have been distributed.

EM sidesteps this by iterating between two steps:

**E-step (Expectation):** Given your current parameter estimates, fill in the missing data. For each occurrence period i and delay j that falls beyond the observation window, impute the expected claim count:

    N̂(i,j) = λ̂(xᵢ) × p̂ⱼ(xᵢ)

This gives you a completed dataset — observed claims where you have them, imputed expected counts where you don't.

**M-step (Maximisation):** Fit your models on the completed dataset as if the imputed counts were real. Fit the occurrence model (Poisson regression on claim frequency) and the delay model (multinomial regression on delay probabilities) separately. Because the complete-data log-likelihood factors into an occurrence term plus a delay term, you can maximise each independently.

Repeat. The EM guarantee is that each iteration cannot decrease the likelihood of the observed data — you are monotonically improving. In practice, convergence is fast: typically 15–30 iterations for well-specified data.

The Wilsens contribution is plugging XGBoost into the M-step instead of a GLM. Standard EM with a GLM M-step has a clean mathematical guarantee. XGBoost is more complicated — you cannot refit XGBoost from scratch at each iteration without breaking the convergence structure. The solution: add new trees to the existing model at each EM iteration rather than refitting. Each iteration adds boosting rounds to the model from the previous iteration. This additive construction preserves monotone likelihood improvement without requiring the clean convexity properties that classical EM exploits.

---

## The insurance adaptation

The original paper has no exposure offset. This matters enormously for pricing use.

If occurrence period i has 10,000 policy-years of exposure and occurrence period i+1 has 2,000 policy-years, you would expect five times as many claims from period i even if the underlying frequency rate is identical. Without an exposure offset, the occurrence model would learn a spurious pattern — it would treat the difference in raw claim counts as a frequency signal rather than a volume effect.

`insurance-nowcast` adds log(exposure) as an offset in the Poisson occurrence model. XGBoost supports this via the `base_margin` parameter. Without this, the model is not usable for pricing.

---

## Usage

```python
from insurance_nowcast import ReportingDelayModel, NowcastSimulator

# Generate synthetic data — useful for validating your setup
sim = NowcastSimulator(
    n_occurrence_periods=24,
    max_delay_periods=12,
    base_frequency=0.08,
    delay_shape="geometric",
)
df = sim.generate(n_policies=2000, eval_period=23)

# Fit the model
model = ReportingDelayModel(
    occurrence_model="xgboost",
    delay_model="xgboost",
    max_delay_periods=12,
    verbose=True,
)
model.fit(
    df,
    occurrence_col="occurrence_period",
    report_col="report_period",
    exposure_col="exposure",
    feature_cols=["age_group", "risk_score", "channel"],
    eval_date=23,
)

# Completion factors by occurrence period
cf = model.predict_completion_factors()
print(cf[["occurrence_period", "completion_factor", "ibnr_count"]])

# Segment-level factors — what you actually want for GLM adjustment
cf_by_channel = model.predict_completion_factors(df=df, by=["channel"])
```

The `by=` parameter is the pricing-relevant output. You get a separate completion factor for each segment combination — age group × channel, or whatever covariates matter for reporting delay in your book. You then apply these per-segment factors when constructing your GLM training dataset rather than the flat aggregate LDF from the triangle.

---

## GLM vs XGBoost

Use `occurrence_model="glm", delay_model="glm"` for books below roughly 5,000 claims in the development window, or when you need interpretable delay distributions you can defend to the reserving team. GLM produces a development pattern you can read and explain: "young drivers take 2.3× longer to report BI claims than experienced drivers."

Use XGBoost when you have scale and expect non-linear interactions between covariates and reporting delay. Wilsens et al. demonstrate on simulated data with non-linear covariate effects that XGBoost materially outperforms GLM-EM — the GLM M-step is misspecified relative to the data generating process and convergence is slower, producing worse estimates at iteration termination.

---

## max_delay_periods

This is the most consequential parameter. Set it too small and your completion factors will systematically understate IBNR. The model will warn you if more than 10% of observed delays are at or beyond the boundary — but you should not rely on the warning as your only check.

Indicative values for UK lines:

| Line | max_delay_periods |
|------|-------------------|
| Motor PD | 6 months |
| Motor BI | 18–24 months |
| Employers' liability | 36–48 months |
| Public liability | 24–36 months |
| Professional indemnity | 36–60 months |

Motor PD and motor BI should not share a model. Fit them separately. The delay structures are different enough that pooling them gives you a model that fits neither well.

---

## Confidence intervals

The model produces bootstrap confidence intervals on completion factors. The default is 100 bootstrap replications at 90% confidence:

```python
model = ReportingDelayModel(
    n_bootstrap=100,
    bootstrap_confidence=0.90,
)
```

Set `n_bootstrap=0` to skip bootstrapping — completion factor estimation only, no CIs. For production use, CIs are worth having: a completion factor of 1.42 with a 90% CI of [1.28, 1.58] tells you something different about your confidence in the Q4 data than one with a CI of [1.39, 1.45].

---

## Diagnostics

```python
from insurance_nowcast import ReportingDelayDiagnostic

diag = ReportingDelayDiagnostic()
diag.plot_convergence(model)             # EM log-likelihood by iteration
diag.plot_development_pattern(model)    # Cumulative delay curves by period
diag.plot_ibnr_by_period(model)         # Observed vs IBNR bar chart
diag.plot_delay_distribution(model, X)  # Delay PMF by risk profile
```

The convergence plot is the first thing to check. You want to see monotone increase that flattens. If you see oscillation, either the data is very sparse or the model is misspecified — consider switching from XGBoost to GLM or reducing `max_delay_periods`.

The delay distribution plot by risk profile is the sanity check. Run it for two contrasting risk profiles — your fastest-reporting segment and your slowest. If the model has learned the covariate signal, these curves should look materially different. If they look identical, the model hasn't found the delay heterogeneity, and covariate-conditioning is not helping you.

---

## What it is not

This is a pricing tool. The outputs — completion factors and IBNR counts by segment — are designed for adjusting a frequency GLM training dataset. They are not statutory IBNR reserves.

The model handles IBNR (unreported claims) only, not RBNS (reported but not settled). For pricing frequency models this is the right choice: ultimate claim counts matter, not ultimate paid amounts. A claim is a claim at the point of notification.

If your completion factors diverge materially from the reserving team's LDFs, investigate before assuming either side is wrong. The divergence might be telling you something about a shift in risk mix, a change in claims handling process, or a methodological difference that is worth understanding.

---

**[insurance-nowcast on GitHub](https://github.com/burning-cost/insurance-nowcast)** — 171 tests, MIT-licensed, PyPI. Library #40.

- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)
- [Temporal Leakage and IBNR Contamination in Insurance Model Validation](/2026/02/23/why-your-cross-validation-is-lying-to-you/)
- [Calibration Testing for Insurance Pricing Models](/2026/03/09/insurance-calibration/)
