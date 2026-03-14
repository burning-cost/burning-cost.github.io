---
layout: post
title: "Doubly Robust Causal Inference for Insurance: TMLE With Poisson Outcomes"
date: 2026-03-12
categories: [libraries, causal-inference, pricing]
tags: [TMLE, targeted-learning, doubly-robust, SuperLearner, EIF, Poisson, causal-inference, DML, price-elasticity, insurance-tmle, python, motor, telematics]
description: "Doubly robust TMLE for insurance pricing with Poisson outcomes and exposure offsets. insurance-tmle - first Python library with the implementation AIPW lacks."
---

You ran a causal inference analysis on your telematics trial. You used Double Machine Learning, as you should. The DML pipeline gave you an ATE estimate of −0.031 with a tight 95% CI: [−0.048, −0.014]. Telematics devices reduce claim frequency by 3.1 claims per hundred policy-years. The result went into the business case.

Now someone asks: how good is your propensity model?

That is the question you do not want. The propensity model — P(telematics device fitted | risk characteristics) — depends on which customers were offered the device, at what price, through which channel, and whether they accepted. You modelled it on the observable covariates. But the acceptance decision depends on things you do not observe: whether the customer was price-sensitive that day, whether a competitor was running a no-device promotion, whether the aggregator ranked you first.

Your propensity model is misspecified. This is not a modelling failure. It is structural.

DML handles propensity misspecification through cross-fitting and Neyman orthogonality. When the propensity model is wrong but the outcome model is right, DML is still consistent — the orthogonalisation absorbs the error. But DML uses a non-substitution estimator: in finite samples, especially when propensity scores are near 0 or 1, the estimator can produce predicted rates outside the parameter space. Negative claim rates. Conversion probabilities above 1. These get silently clipped or averaged away, but the underlying instability is real.

TMLE — Targeted Maximum Likelihood Estimation — solves the same problem differently. It is also doubly robust. But its targeting step is a substitution estimator: the final ATE is obtained by plugging a corrected outcome model into the parameter mapping, so the estimate always respects parameter space constraints. And critically for insurance: TMLE tolerates propensity misspecification at least as well as DML, and outperforms it in finite samples when the propensity model is the weaker of the two.

[`insurance-tmle`](https://github.com/burning-cost/insurance-tmle) is our implementation. It is the first Python library with Poisson TMLE and exposure offsets for insurance count outcomes. The R ecosystem has had mature TMLE tooling since 2015 (`tmle`, `tmle3`, `lmtp`). Python has had nothing that handles the Poisson frequency / Gamma severity structure that insurance actuaries actually need.

```bash
pip install insurance-tmle
```

---

## What TMLE does, precisely

The algorithm has five steps (van der Laan & Rubin 2006, *International Journal of Biostatistics* 2(1)):

**Step 1 — Initial outcome model.** Fit E[Y | A, W] using any flexible estimator. Call this Q_n(A, W). For claim frequency: a Poisson GLM with log link and exposure offset. For conversion: logistic regression or a gradient boosting classifier. The initial model can be as flexible as you like — random forest, CatBoost, a SuperLearner ensemble.

**Step 2 — Propensity model.** Fit P(A=1 | W) = g_n(W). Again, any estimator. This models the probability that a policy received treatment (telematics fitted, price increase applied, intervention offered) given observable characteristics.

**Step 3 — Clever covariate.** Construct H = A/g_n(W) − (1−A)/(1−g_n(W)). This is the efficient influence function's treatment component. For each observation, it encodes how much weight to give based on how surprising its treatment assignment was. Policies that received treatment despite low predicted propensity get high positive weight; policies that did not receive treatment despite high propensity get high negative weight.

**Step 4 — Targeting step.** This is what separates TMLE from every other doubly-robust estimator. Run a one-parameter regression: for binary outcomes, fit logit(Q\*) = logit(Q_n) + ε·H using Q_n as a fixed offset, and solve for ε. For Poisson outcomes, fit log(Q\*) = log(Q_n) + ε·H, which is equivalent to Q\* = Q_n · exp(ε·H). The update is multiplicative on the rate scale — the final predicted rates stay positive by construction.

The targeting step solves the efficient influence function estimating equation exactly: (1/n) Σ φ_i = 0. This is the mathematical condition for semiparametric efficiency. The initial estimate Q_n is perturbed by the minimum amount necessary to satisfy it.

**Step 5 — Plug-in ATE and standard error.** ATE = (1/n) Σ [Q\*(1, W_i) − Q\*(0, W_i)]. Standard error from the efficient influence function: SE = sqrt(Var(φ)/n), where φ_i = Q\*(1, W_i) − Q\*(0, W_i) + H_i · (Y_i − Q\*(A_i, W_i)) − ATE.

No bootstrap. The EIF-based standard error is analytically exact and computationally trivial.

---

## Why Poisson TMLE matters

The standard TMLE literature is almost entirely binary outcome (treatment effects in epidemiology: disease incidence, mortality, hospitalisation). The logistic targeting step is well-implemented in R and partially in Python (IBM's `causallib` covers binary outcomes for ATE estimation, as of April 2025).

Insurance frequency modelling is Poisson, not binary. A claim count over a policy year, scaled by exposure, follows a Poisson distribution with a log-link outcome model. The targeting step for Poisson is mathematically straightforward but technically different from the binary case:

```
log(Q*) = log(Q_n) + ε · H    with offset = log(Q_n · exposure)
```

This is a Poisson GLM with H as the single covariate and log(Q_n · exposure) as the fixed offset. The update gives:

```
Q* = Q_n · exp(ε · H)
```

The multiplicative structure on the rate scale is exactly what insurance pricing uses. Relativities compose multiplicatively in a log-linear GLM. The TMLE targeting step speaks the same language.

The exposure offset in the counterfactual evaluation also deserves care. When we evaluate Q\*(1, W_i) — the predicted claim rate if policy i had received treatment — we standardise to unit exposure. The counterfactual is a rate, not a count. `insurance-tmle` handles this automatically.

No Python library implemented this before `insurance-tmle`. The `zepid` library (abandoned October 2022) had binary TMLE only. `causallib` (IBM, active) has binary and continuous bounded outcomes. `EconML` (Microsoft) has an open GitHub issue requesting TMLE; as of March 2026 it has not shipped one. `PyTMLE` handles survival outcomes, not cross-sectional count data.

---

## Using it

### Claim frequency: telematics causal effect

```python
from insurance_tmle import TMLE

# treatment: 1 = telematics device fitted, 0 = not
# exposure: policy duration in years
tmle = TMLE(outcome_family="poisson")
tmle.fit(
    Y=claim_counts,
    A=has_telematics,
    W=rating_factors,   # age, vehicle group, NCB, region, ...
    exposure=years,
)

print(tmle.summary())
#    method   ate       se   ci_lower  ci_upper  p_value
# 0    TMLE  -0.031  0.0082   -0.047    -0.015   0.0002

print(f"Telematics reduces claim rate by {-tmle.ate:.3f} per year")
print(f"Relative reduction: {tmle.ate / baseline_rate:.1%}")
```

The `ate_` attribute is in rate units (claims per year). The EIF is accessible via `tmle.eif_` if you want to inspect the observation-level influence.

### Conversion: price sensitivity

```python
from insurance_tmle import TMLE

# treatment: 1 = price increase applied, 0 = control
tmle = TMLE(outcome_family="binary")
tmle.fit(Y=converted, A=price_increase, W=rating_factors)

lo, hi = tmle.ate_ci()
print(f"Price increase reduced conversion by {-tmle.ate:.1%} "
      f"(95% CI: [{-hi:.1%}, {-lo:.1%}])")
```

### Checking the targeting step converged

The epsilon parameter should be close to zero if the initial outcome model was already well-specified. A large epsilon means the targeting step had substantial corrective work to do — which typically indicates propensity model issues or initial outcome model misspecification.

```python
from insurance_tmle import ConvergenceCheck

cc = ConvergenceCheck(tmle.Q_init_, tmle.Q_star_, tmle.epsilon_)
print(cc.is_converged())   # True if |epsilon| < 0.01
print(f"Epsilon: {tmle.epsilon_:.5f}")
```

---

## Running TMLE and DML side by side

We recommend running both estimators. Agreement is evidence of robustness. Divergence is a model sensitivity signal that needs investigating before anything goes to committee.

```python
from insurance_tmle import CausalComparison

comp = CausalComparison(outcome_family="binary", include_cvtmle=True)
comp.fit(Y, A, W)
print(comp.results)
#                method     ate      se  ci_lower  ci_upper  p_value
# 0  Outcome Regression  -0.028  0.0078   -0.044    -0.013   0.0003
# 1                 IPW  -0.035  0.0095   -0.053    -0.016   0.0002
# 2                AIPW  -0.032  0.0082   -0.048    -0.016   0.0001
# 3                TMLE  -0.032  0.0082   -0.048    -0.016   0.0001
# 4             CV-TMLE  -0.032  0.0083   -0.048    -0.016   0.0001
```

When TMLE and outcome regression agree but IPW disagrees sharply, the propensity model is probably misspecified and the outcome model is carrying the weight. When TMLE and IPW agree but outcome regression is far off, the outcome model is misspecified and the propensity model is saving you. When all five agree, you can present the result with confidence. When TMLE and DML disagree materially, you have work to do.

The `CausalComparison` output is built for a confounding bias table in a committee paper: the "naive" outcome regression estimate against the doubly-robust estimates shows the magnitude of the bias that causal adjustment is correcting.

---

## SuperLearner: not just better, theoretically optimal

TMLE's consistency guarantee requires that at least one nuisance model is correctly specified. In practice, "correctly specified" means "converging at a rate faster than n^(−1/4)" — which gives TMLE its n^(−1/2) efficiency. You cannot guarantee this with a single GLM.

The SuperLearner (van der Laan, Polley & Hubbard 2007) maximises the probability of achieving the required convergence rate by stacking multiple learners with cross-validated NNLS weights. The ensemble is provably no worse than the best single learner in the library — it achieves the oracle convergence rate asymptotically.

`insurance-tmle` ships a `SuperLearner` class with insurance-appropriate default libraries:

- **Poisson frequency**: `PoissonRegressor` (unpenalised and L2-penalised), `HistGradientBoostingRegressor(loss="poisson")` at two learning rates, CatBoost with Poisson objective if installed.
- **Propensity (binary treatment)**: `LogisticRegression(elastic net)`, `HistGradientBoostingClassifier`, CatBoost classifier if installed.

```python
from insurance_tmle import SuperLearner, TMLE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

# Custom propensity library when you have a prior about the treatment mechanism
prop_sl = SuperLearner(
    library=[
        LogisticRegression(penalty="elasticnet", solver="saga",
                           l1_ratio=0.5, max_iter=2000),
        HistGradientBoostingClassifier(max_iter=300),
    ],
    task="binary",
    cv=5,
)

tmle = TMLE(outcome_family="poisson", propensity_estimator=prop_sl)
tmle.fit(Y, A, W, exposure=exposure)
```

The SuperLearner's weight report tells you how much each base learner contributed. A weight of 0.0 on the GLM and 1.0 on the gradient booster means the data strongly favour the non-parametric model. A weight of 0.7 on the GLM means the Poisson log-linear structure fits well and the boosted model is not adding much. This is useful diagnostic information independent of the TMLE result.

---

## CV-TMLE for smaller portfolios

Standard TMLE relies on empirical process conditions — specifically, the Donsker class condition — that guarantee the plug-in estimator behaves well. These conditions are hard to verify in finite samples and are violated when you use very flexible ML nuisance models with small datasets.

CV-TMLE addresses this by fitting nuisance models on complementary folds (cross-fitting), then applying the targeting step to the held-out fold predictions. This eliminates the Donsker class condition entirely. The cost is computation time.

We recommend CV-TMLE for n < 5,000 or when you are using complex nuisance models (gradient boosting, neural networks) on moderate-sized datasets:

```python
from insurance_tmle import CVTMLE

cvtmle = CVTMLE(outcome_family="binary", cv=5)
cvtmle.fit(Y, A, W)
print(cvtmle.summary())
```

Levy & van der Laan (2024, arXiv:2409.11265) provide the most recent simulation evidence on CV-TMLE performance. The finite-sample coverage improvements over standard TMLE are largest at n = 500–2,000 and taper off above n = 10,000.

---

## Propensity overlap: the positivity assumption

TMLE requires the positivity assumption: every unit must have non-zero probability of receiving either treatment, P(A=1 | W=w) ∈ (0, 1) for all w in the support of W. Violations mean there are subgroups where you cannot estimate a causal effect, because you never observe them under counterfactual treatment.

In insurance, positivity violations are common. Certain risk segments may have been systematically excluded from a telematics trial. Certain postcodes may have always received a specific pricing tier. The propensity score distribution tells you where your causal estimate is and is not supported by the data.

```python
from insurance_tmle import PropensityDiagnostics

diag = PropensityDiagnostics(tmle.g_, A=A)
print(diag.positivity_warning())
# "WARNING: 3.2% of observations have propensity < 0.05 or > 0.95.
#  Causal estimates for these observations are unreliable."
diag.plot()  # Overlapping histograms of g(W) for treated vs control
```

The library trims propensity scores to [0.01, 0.99] by default. You can adjust this, but the default is conservative enough to prevent the clever covariate H from blowing up while retaining the vast majority of observations in well-overlapped data.

---

## Heterogeneous effects: CATE by segment

Average treatment effects hide heterogeneity. The average telematics effect might be −3.1 claims per hundred policy-years overall, with −5.2 in under-25 drivers and −1.8 in the 40–55 band. A pricing decision that does not distinguish these will underprice telematics discount for young drivers.

`StratumTMLE` runs a separate TMLE within each stratum, pooling the nuisance models but computing stratum-specific ATEs:

```python
from insurance_tmle import StratumTMLE

st = StratumTMLE(outcome_family="binary")
st.fit(Y, A, W, strata=age_band)
print(st.results)
#   stratum    n     ate      se   ci_lower  ci_upper  p_value
# 0    17-25  843  -0.052  0.0144   -0.080    -0.024   0.0003
# 1    26-39  2104 -0.031  0.0091   -0.049    -0.013   0.0007
# 2    40-55  3218 -0.018  0.0075   -0.033    -0.003   0.0168
# 3      55+  1891 -0.009  0.0081   -0.025     0.007   0.2674
```

The stratum CIs are EIF-based within each stratum and are honest about the smaller sample sizes in each band.

---

## The HTML report

The library ships `TMLEReport`, which generates an HTML committee document combining the method comparison, propensity diagnostics, and CATE breakdown:

```python
from insurance_tmle import TMLEReport

report = TMLEReport(
    tmle,
    comparison=comp,
    stratum_tmle=st,
    title="Telematics Causal Impact — Motor Book 2025 Q4",
)
with open("tmle_report.html", "w") as f:
    f.write(report.to_html())
```

The report includes five sections: ATE summary table (all estimators side by side), confounding bias table (naive vs TMLE), propensity overlap plot and warning flags, CATE breakdown by stratum, and the convergence diagnostics (epsilon, EIF distribution, n_trimmed). It is designed to be attached to a pricing committee paper without further editing.

---

## The academic lineage and why it took this long

TMLE was introduced by van der Laan & Rubin in 2006. The core theory — doubly-robust semiparametric efficiency, the targeting step as minimum-perturbation projection onto the tangent space — was well-established by 2011, when van der Laan & Rose published *Targeted Learning* (Springer). The R `tmle` package shipped the same year.

The Python gap reflects where the methodology comes from. TMLE was developed and deployed in epidemiology and health economics, not in insurance. Of 300+ TMLE applications reviewed in a 2023 systematic review (Lodi et al., *Annals of Epidemiology*), the breakdown is: epidemiology, public health, clinical trials, health technology assessment. Insurance: zero. The survey of causal inference papers in banking, finance, and insurance (arXiv:2307.16427, 2023) finds 45 papers over 30 years — and TMLE is not mentioned once.

The closest prior art in insurance is Guelman & Guillén (2014, *Expert Systems with Applications*), which uses propensity score matching for automobile insurance price elasticity. Matching, not TMLE. The PSM approach has neither double robustness nor semiparametric efficiency. It is a 2014 paper using 2005 methodology.

`insurance-tmle` is, as far as we can determine, the first Python implementation of Poisson TMLE with exposure offsets anywhere. It is certainly the first implementation designed for insurance causal inference workflows. The claim about academic novelty — that applying TMLE to P&C insurance pricing is unpublished territory — is grounded in a systematic literature search conducted in March 2026. If there is a paper we missed, we would be genuinely interested to see it.

---

## Where this sits relative to insurance-causal (DML)

We built [`insurance-causal`](https://github.com/burning-cost/insurance-causal) for DML-based price elasticity estimation. It wraps EconML, handles the insurance-specific workflow (exposure offset, confounding report, segment heterogeneity), and is the right tool for large portfolios where you have a strong prior that your outcome model (the claim GLM) is well-specified.

`insurance-tmle` is the natural next step for teams where that prior is weaker — novel product lines, thin segments, situations where you genuinely are not sure whether your outcome model or your propensity model is better specified.

We think the right workflow is: run both. Use `CausalComparison` to see what each method gives you. Agreement strengthens the result. Divergence tells you something important about model sensitivity. The EIF standard errors are honest about the precision of the TMLE estimate; you can compare them directly to the DML standard errors.

TMLE is not strictly better than DML in all cases. For large portfolios with well-specified Poisson GLMs, DML and TMLE converge to the same estimate. The TMLE advantage concentrates in finite samples, misspecified propensity models, and non-standard outcome families — which is to say, the situations that UK pricing actuaries encounter routinely.

---

**[insurance-tmle on GitHub](https://github.com/burning-cost/insurance-tmle)** — MIT-licensed, PyPI, 120 tests. For the causal analyses where your propensity model is structurally wrong and you know it.

---

**Related reading:**
- [How Much of Your GLM Coefficient Is Actually Causal?](/2026/02/25/causal-inference-for-insurance-pricing/) — DML-based causal elasticity estimation; the right tool when your outcome model is well-specified and the propensity model is less critical
- [When exp(beta) Lies: Confounding in GLM Rating Factors](/2026/03/05/your-rating-factor-might-be-confounded/) — the confounding problem that motivates both DML and TMLE, explained through GLM rating factor examples
- [Heterogeneous Lapse Effects with Bayesian Causal Forests: Beyond the Average Elasticity](/2026/03/12/insurance-bcf/) — Bayesian Causal Forests for heterogeneous treatment effects when the causal question involves policy-level variation rather than a single average effect
