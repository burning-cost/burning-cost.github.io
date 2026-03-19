---
layout: post
title: "Heterogeneous Lapse Effects with Bayesian Causal Forests: Beyond the Average Elasticity"
date: 2026-03-12
categories: [libraries, pricing, causal-inference]
tags: [BCF, bayesian-causal-forest, BART, CATE, heterogeneous-treatment-effects, stochtree, price-elasticity, FCA, Consumer-Duty, RIC, MCMC, propensity, observational-data, UK-motor, insurance-bcf, python]
description: "Bayesian Causal Forests for heterogeneous lapse effects in UK insurance pricing. Segment-level elasticity with posteriors - insurance-bcf wrapping stochtree."
---

Your motor book took an 8% rate increase last October. Aggregate lapse rose 1.8 percentage points. The GLM returned an elasticity of −0.22. The pricing team noted it in the experience review and moved on.

That number is probably wrong for most of your book — and wrong in ways that are costing you margin. If young price comparison website customers are lapsing at three times the rate of mature direct customers under the same increase, the average elasticity tells you nothing useful about where to push rates and where to hold back. The segments with high lapse sensitivity need a softer increase. The segments that barely noticed need more. Pricing to the average means you are simultaneously leaving margin on the table and over-lapsin the customers you most want to keep.

The GLM cannot tell you this. It is built to estimate population-averaged effects. Segment interactions help, but they require you to specify in advance which interactions matter — and in a UK personal lines book with dozens of rating factors, you will miss the ones that matter most.

[`insurance-bcf`](https://github.com/burning-cost/insurance-bcf) wraps Bayesian Causal Forests for insurance pricing teams. It estimates the treatment effect for every policy in your portfolio — not an average — with a posterior distribution and credible intervals suitable for Consumer Duty audit documentation.

```bash
pip install insurance-bcf
```

---

## Why the average elasticity is the wrong number

When you run a rate increase, you are conducting a natural experiment. Some policies got the increase; some did not (not renewed, mid-term adjustments, band effects). The question is not "what was the average lapse effect?" but "what was the lapse effect *for this type of customer*, and how certain are we?"

The answer to that question is a function of the covariates — age, channel, NCD level, vehicle type, postcode deprivation index, years as a customer. The function is unknown. It is almost certainly non-linear and has interactions. And it is the function that determines your optimal rating structure.

Conditional Average Treatment Effect (CATE) estimation is the right framing. For policy *i*, we want:

```
tau(x_i) = E[Y_i(1) - Y_i(0) | X_i = x_i]
```

where Y_i(1) is the renewal outcome if the rate increase was applied and Y_i(0) if it was not. Since we only observe one of these, CATE estimation is a causal inference problem on observational data.

The standard causal forest (Wager & Athey 2018, implemented in `econml`) gives you CATE estimates. It is frequentist, uses honest splitting, and produces asymptotically valid confidence intervals. It is a good tool.

BCF does something different, and for insurance observational data the difference is material.

---

## The Regularization-Induced Confounding problem

In a standard BART model fitted to insurance observational data, there is a systematic bias towards over-estimating the treatment effect. Hahn, Murray and Carvalho (2020) named it Regularization-Induced Confounding.

The mechanism: your risk model drives both the premium and the renewal probability. High-risk policies get higher premiums; they also have different renewal behaviour for reasons unrelated to the premium (younger drivers move more often, change policies at life events). In the outcome model, E[Y|X] is almost entirely explained by the propensity score pi(X) — the probability that a given policy received the rate increase. BART regularization over-shrinks the prognostic function mu, leaving unexplained variance in the outcome. That residual gets attributed to the treatment, because the treatment is correlated with the propensity. The result: tau absorbs confounding that mu failed to capture. The estimated treatment effect is biased towards the covariate-driven renewal effect, not the rate-driven one.

BCF corrects this by running *two separate* Bayesian tree ensembles with deliberately asymmetric priors, and by including the propensity score explicitly in the prognostic function:

```
Y_i = mu(x_i, pi_hat(x_i)) + tau(x_i) * z_i + epsilon_i
```

`mu` — 250 trees, `alpha=0.95`, `beta=2` — is expressive and allowed to soak up the renewal surface under control. `tau` — 50 trees, `alpha=0.25`, `beta=3` — has a shrink-to-homogeneity prior: it is regularised towards a *constant* treatment effect, and only heterogeneity with genuine data support will survive. Including `pi_hat` in `mu` removes the collinearity between Z and the unexplained residual. The confounding cannot bleed into `tau`.

This is not optional when you are fitting to insurance observational data where the risk model drives treatment assignment. With the default `propensity_covariate='prognostic'` setting in `insurance-bcf`, RIC correction is always on.

---

## Fitting the model

`insurance-bcf` wraps [stochtree](https://github.com/StochasticTree/stochtree) 0.4.0 — the reference Python BCF implementation, released March 2026, authored by Herren, Hahn, Murray, and Carvalho, the original BCF paper authors. It is the only production-quality Python BCF engine. The C++ MCMC engine (XBART GFR warm-start plus full MCMC) is fast enough for book-level analysis, and `num_threads` defaults to all available cores.

```python
from insurance_bcf import BayesianCausalForest, ElasticityEstimator
from insurance_bcf.simulate import simulate_renewal, SimulationParams

# Simulate 10k motor renewals with known heterogeneous treatment effects
data = simulate_renewal(SimulationParams(n_policies=10_000, random_seed=42))

model = BayesianCausalForest(
    outcome='binary',    # renewal flag: probit link on latent scale
    num_mcmc=500,        # retained posterior samples
    num_gfr=10,          # GFR warm-start (eliminates burn-in)
    random_seed=42,
)
model.fit(
    X=data.X,               # pd.DataFrame of rating factors
    treatment=data.treatment,  # 1 = rate increase applied, 0 = not
    outcome=data.outcome,   # 1 = renewed, 0 = lapsed
)

# Policy-level CATE: posterior mean and 95% credible interval
cate_df = model.cate(data.X)
print(cate_df.head())
#    cate_mean  cate_lower  cate_upper  cate_std
# 0     -0.061      -0.074      -0.048     0.007
# 1     -0.042      -0.051      -0.033     0.005
# 2     -0.018      -0.024      -0.012     0.003
```

Each row is a posterior mean treatment effect for a policy, with a credible interval derived from 500 MCMC samples. Policy 0 is estimated to have a 6.1 percentage point lapse effect from the rate increase; policy 2 has a 1.8 point effect — one-third as sensitive. The same rate increase, very different consequences.

For pre-computed propensity scores — preferred when you have domain knowledge about what drives treatment assignment — pass them directly:

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(data.X, data.treatment)
pi_hat = lr.predict_proba(data.X)[:, 1]

model.fit(data.X, data.treatment, data.outcome, propensity=pi_hat)
```

---

## Segment effects

The policy-level CATE is useful for individual pricing decisions. For rate strategy, you want the segment picture:

```python
est = ElasticityEstimator(model)
seg = est.segment_effects(data.X, segment_cols=['age_band', 'channel'])
print(seg)
#   age_band  channel  effect_mean  effect_lower  effect_upper  n_policies
# 0        0        1       -0.082        -0.094        -0.071        1241
# 1        0        0       -0.041        -0.049        -0.033         420
# 2        1        1       -0.035        -0.041        -0.029        3410
# 3        5        0       -0.011        -0.018        -0.004        1892
```

Young PCW customers (age_band=0, channel=1): lapse effect of 8.2 percentage points. Mature direct customers (age_band=5, channel=0): 1.1 percentage points. That is a 7.5x difference in sensitivity to the same rate movement. If you price to the portfolio average, you are leaving 6+ points of margin on mature direct and burning retention on young PCW. Neither is intentional.

The segment credible intervals matter here. The young PCW CI is `[-0.094, -0.071]` — tight and firmly negative. The mature direct CI is `[-0.018, -0.004]` — the effect is small and the data supports it confidently. You can act on both segments without worrying that you are acting on noise.

How does sensitivity vary along a single feature, averaging over the book distribution?

```python
pd_df = est.partial_dependence(data.X, feature='ncb_steps', grid_points=6)
print(pd_df)
# feature_value  pdp_mean  pdp_lower  pdp_upper
#             0    -0.071     -0.082     -0.060
#             1    -0.065     -0.074     -0.055
#             5    -0.031     -0.039     -0.023
```

Higher NCD customers are less lapse-sensitive. They have more to lose by switching insurer — their NCD would not transfer at full value, or there is risk of losing it. This is directionally what you would expect, and BCF quantifies it.

---

## Rate adjustment recommendations

Given segment-level CATEs, the library will estimate implied rate adjustments to hit a target margin, bounded by a maximum adjustment:

```python
import pandas as pd
import numpy as np

current_premium = pd.Series(np.random.uniform(400, 1200, len(data.X)))

adj = est.optimal_rate_adjustment(
    data.X,
    target_margin=0.05,
    current_premium=current_premium,
    max_adjustment=0.20,
)
print(adj[['suggested_adjustment', 'adjustment_confidence']].head())
```

This is a tool, not a decision. The output gives you a direction and a confidence measure derived from the posterior width. Where the CATE posterior is tight, the adjustment confidence is high. Where the posterior is wide — thin segments with uncertain effects — the confidence is low, and you should apply more caution.

---

## Consumer Duty compliance

Consumer Duty (PS22/9, PRIN 2A) requires firms to monitor fair value across groups defined by protected characteristics. The credible interval structure of BCF — you get a posterior over the treatment effect for every protected characteristic group — is exactly what audit documentation needs.

`BCFAuditReport` generates a structured HTML report. The `protected_characteristic_check` method tests whether the treatment effect for each protected group falls within the credible interval of the portfolio average:

```python
from insurance_bcf import BCFAuditReport

report = BCFAuditReport(model, est)

pc_df = report.protected_characteristic_check(
    data.X,
    protected_cols=['age_band'],
)
print(pc_df[['characteristic', 'group', 'effect_mean', 'flag']])
#   characteristic  group  effect_mean   flag
# 0       age_band      0       -0.082   True   ← outside portfolio CI
# 1       age_band      5       -0.011  False

report.render(
    output_path='bcf_audit_2025Q4.html',
    X=data.X,
    Z=data.treatment,
    protected_cols=['age_band'],
    segment_cols=[['age_band'], ['channel'], ['age_band', 'channel']],
)
```

A `flag=True` for the youngest age band does not mean the pricing is discriminatory — it means the pricing effect differs materially by age group, which should be documented and explained. The report includes model configuration, MCMC convergence diagnostics, segment effects, and a methodology appendix. It is designed for internal model governance, not for FCA submission.

One point on scope: BCF is a Bayesian black box. It is harder to audit under Solvency II model governance than a GLM with explicit parameters. The library includes MCMC convergence diagnostics (R-hat via arviz for multi-chain runs) and serialisation (`to_json`/`from_json`) to support model governance requirements, but you should expect pushback from model risk teams if you have not done the groundwork on explainability.

---

## When to use this vs. insurance-causal

These are not competing tools. They answer different questions.

Use DML ([`insurance-causal`](https://github.com/burning-cost/insurance-causal)) when the treatment is the actual premium level — a continuous variable — and you have some exogenous price variation from an A/B test or the GIPP structural break as a natural experiment. DML gives you a scalar elasticity for the rate optimiser.

Use BCF (this library) when:

- The treatment is binary or categorical: rate increase applied yes/no, NCD tier change, telematics option
- You want posterior uncertainty over segment effects — not confidence intervals, but a full posterior that propagates through the audit documentation
- Confounding is heavy: the risk model drives both premium assignment and renewal probability, making RIC correction critical
- You want counterfactual analysis under the full Bayesian framework

If you have both tools fitted to overlapping data, divergence in segment rankings is a useful diagnostic. It suggests model misspecification in at least one of them, and forces you to think carefully about which causal assumptions each approach is making.

A practical note on runtime: 500 MCMC samples on a 100k-policy book with 30 features will take 10–30 minutes, depending on your hardware and `num_threads`. For exploratory analysis, 100 samples with `num_gfr=5` is a reasonable starting point. The GFR warm-start removes the need for burn-in iterations, so the effective chain length is close to what you specify.

---

## GIPP date handling

If your dataset spans January 2022, the library will warn you:

```python
X_with_dates = data.X.copy()
X_with_dates['renewal_date'] = pd.date_range('2021-06-01', periods=len(data.X), freq='D')

model.fit(X_with_dates, data.treatment, data.outcome, gipp_date_col='renewal_date')
# GIPPBreakWarning: Column 'renewal_date' spans the GIPP implementation date (January 2022).
```

Data spanning the GIPP break should generally be split. Pre-GIPP renewal behaviour is not a valid basis for estimating post-GIPP treatment effects — the structural change in retention pricing means the treatment assignment mechanism changed. Fitting a single BCF model across the break conflates two regimes.

---

## The library

[`insurance-bcf`](https://github.com/burning-cost/insurance-bcf) is MIT-licensed and on PyPI. 149 tests across four modules: `BayesianCausalForest`, `ElasticityEstimator`, `BCFAuditReport`, and `simulate`. It wraps stochtree 0.4.0 and requires a C++ build; wheels are available for Linux x86_64, macOS (Intel + Apple Silicon), and Windows x86_64.

```bash
pip install stochtree>=0.4.0
pip install insurance-bcf
pip install insurance-bcf[diagnostics]   # includes arviz for multi-chain R-hat
```

It is the 73rd library in the Burning Cost open-source portfolio.

---

**[insurance-bcf on GitHub](https://github.com/burning-cost/insurance-bcf)** — MIT-licensed, PyPI. Posterior treatment effects, not averaged ones.

---

## References

- Hahn, P.R., Murray, J.S., Carvalho, C.M. (2020). Bayesian Regression Tree Models for Causal Inference. *Bayesian Analysis* 15(3): 965–1056.
- Herren, A., Hahn, P.R., Murray, J.S., Carvalho, C.M. (2026). StochTree. arXiv:2512.12051v2.
- Chipman, H.A., George, E.I., McCulloch, R.E. (2010). BART. *Annals of Applied Statistics* 4(1): 266–298.
- Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA* 113(523): 1228–1242.
- FCA Evaluation Paper EP25/2 (2025). Evaluation of GIPP Remedies.

---

## See Also

- **[insurance-causal](https://github.com/burning-cost/insurance-causal)** — DML price elasticity for continuous treatment (actual premium level). Use when you have exogenous price variation.
- **[insurance-fairness](https://github.com/burning-cost/insurance-fairness)** — Proxy discrimination diagnostics for fitted pricing models. Complements BCFAuditReport for Consumer Duty evidence packs.
- **[insurance-dynamics](https://github.com/burning-cost/insurance-dynamics)** — Detect when your loss experience regime changed. Segment the dataset before fitting BCF if a structural break is present.

- [Causal Inference for Insurance Pricing](/2026/02/25/causal-inference-for-insurance-pricing/)
- [DML for Insurance: Benchmarks and When It Beats Naive Regression](/2026/03/09/dml-insurance-benchmarks/)
- [Your Demand Model Is Confounded](/2026/03/01/your-demand-model-is-confounded/)
