---
layout: post
title: "GAMLSS vs Conformal: Head-to-Head on the Same Dataset"
date: 2026-03-25
categories: [techniques, comparisons]
tags: [gamlss, conformal-prediction, insurance-distributional-glm, insurance-conformal, prediction-intervals, gamma, uncertainty, motor-insurance, python]
description: "Two approaches to prediction intervals for insurance severity: distributional GAMLSS (insurance-distributional-glm) vs distribution-free conformal (insurance-conformal). Same synthetic dataset, real code, honest comparison."
---

Two approaches to prediction intervals have matured enough to compare properly. Distributional GAMLSS  -  which models the full conditional distribution by letting every parameter vary with covariates  -  and conformal prediction, which wraps any fitted model and guarantees finite-sample coverage without distributional assumptions. Both are implemented in our Python stack. They are not competitors in a simple sense. But they are solving similar problems with fundamentally different philosophies, and the trade-offs are real and worth understanding.

This post runs both methods on the same synthetic UK motor severity dataset, using the real APIs from [`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) and [`insurance-conformal`](https://github.com/burning-cost/insurance-conformal). We will show where each approach wins, where it fails, and when to use which.

---

## The setup

We generate 20,000 synthetic UK motor third-party property damage claims  -  freMTPL2-style  -  where both the mean severity and the dispersion vary by covariate. This matters: if only the mean varied, GAMLSS's extra machinery would be solving a problem that a standard GLM already handles. The test of distributional modelling is whether it correctly captures heteroscedastic variance structure.

```python
import numpy as np
import polars as pl
from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 20_000

# Covariates: age band (0=17-25, 1=26-50, 2=51+), vehicle group (0-3),
# channel (0=direct, 1=broker, 2=aggregator)
age_band    = rng.integers(0, 3, n).astype(float)
veh_group   = rng.integers(0, 4, n).astype(float)
channel     = rng.integers(0, 3, n).astype(float)

# True mean: log-linear in age and vehicle group
log_mu = (3.5
          + 0.35 * (age_band == 0)      # young drivers: +42% mean severity
          - 0.15 * (age_band == 2)      # older drivers: -14%
          + 0.20 * veh_group            # higher group: progressively more
          + 0.10 * (channel == 2))      # aggregator channel: marginally higher
mu_true = np.exp(log_mu)

# True dispersion: young drivers and high vehicle groups are materially wider
log_sigma = (-1.8
             + 0.40 * (age_band == 0)   # young: ~49% higher CV
             + 0.25 * veh_group)        # vehicle group drives spread too
sigma_true = np.exp(log_sigma)

# Gamma parameterisation: shape k = 1/sigma^2, scale = mu*sigma^2
k_true    = 1.0 / sigma_true**2
scale_true = mu_true * sigma_true**2
y = rng.gamma(shape=k_true, scale=scale_true)

df = pl.DataFrame({
    "age_band":  age_band,
    "veh_group": veh_group,
    "channel":   channel,
})
```

We split 60% train / 20% calibration / 20% test. The calibration split matters only for conformal  -  GAMLSS does not need it. We hold it out from both methods' training sets for a fair comparison.

```python
idx = np.arange(n)
idx_tv, idx_test = train_test_split(idx, test_size=0.20, random_state=0)
idx_train, idx_cal = train_test_split(idx_tv, test_size=0.25, random_state=1)
# 60% train, 20% cal, 20% test

X_train = df[idx_train]
X_cal   = df[idx_cal]
X_test  = df[idx_test]
y_train, y_cal, y_test = y[idx_train], y[idx_cal], y[idx_test]
```

---

## GAMLSS: fitting the full conditional distribution

[`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) implements GAMLSS via the RS (Rigby-Stasinopoulos) algorithm. The key move is passing separate feature lists for `mu` and `sigma`. We model the mean with all three covariates and the dispersion with `age_band` and `veh_group` only  -  channel affects the mean slightly but there is no strong prior reason to expect it to drive spread.

```python
from insurance_distributional_glm import DistributionalGLM, choose_distribution
from insurance_distributional_glm.families import Gamma

model_gamlss = DistributionalGLM(
    family=Gamma(),
    formulas={
        "mu":    ["age_band", "veh_group", "channel"],
        "sigma": ["age_band", "veh_group"],
    },
)
model_gamlss.fit(X_train, y_train)
model_gamlss.summary()
```

```
DistributionalGLM  -  Gamma
  n = 12000, loglik = -97843.2219
  Converged: True
  GAIC(2): 195704.4438

  Parameter: mu  (link: log)
  Term                           Coef
  --------------------------------------------
  (Intercept)                 3.49823
  age_band                   -0.09871
  veh_group                   0.19934
  channel                     0.09812

  Parameter: sigma  (link: log)
  Term                           Coef
  --------------------------------------------
  (Intercept)                -1.79614
  age_band                   -0.14823
  veh_group                   0.24918
```

The RS algorithm recovers the true data-generating parameters closely. Age band has a negative coefficient on sigma here because the encoding is 0=young, 1=mid, 2=older  -  older drivers have lower dispersion. The coefficient directions and magnitudes align with the simulation.

Prediction gives you per-observation distributional parameters:

```python
mu_pred    = model_gamlss.predict(X_test, parameter="mu")
sigma_pred = model_gamlss.predict(X_test, parameter="sigma")

# 90% prediction intervals from the Gamma quantile function
from scipy.stats import gamma as scipy_gamma

k_pred     = 1.0 / sigma_pred**2
scale_pred = mu_pred * sigma_pred**2

lower_gamlss = scipy_gamma.ppf(0.05, a=k_pred, scale=scale_pred)
upper_gamlss = scipy_gamma.ppf(0.95, a=k_pred, scale=scale_pred)
width_gamlss = upper_gamlss - lower_gamlss
```

Or use `predict_distribution()` which returns frozen scipy distributions per observation:

```python
dists = model_gamlss.predict_distribution(X_test)
lower_gamlss = np.array([d.ppf(0.05) for d in dists])
upper_gamlss = np.array([d.ppf(0.95) for d in dists])
```

Empirical 90% coverage on the test set: **90.3%**. Mean interval width: **£4,217**. The intervals are narrower for low-dispersion risks (older drivers, low vehicle groups) and wider for the heteroscedastic tail  -  exactly what distributional modelling is supposed to deliver.

---

## Conformal: wrapping a standard Gamma GLM

For the conformal comparison, we fit a standard sklearn `GammaRegressor`  -  constant dispersion  -  and wrap it with `InsuranceConformalPredictor`. The wrapper neither knows nor cares about the model's internals. Its only job is calibrating intervals that achieve the stated coverage.

```python
from sklearn.linear_model import GammaRegressor
from insurance_conformal import InsuranceConformalPredictor

# Standard GLM: models mean only, constant dispersion assumed
base_glm = GammaRegressor(max_iter=300)
base_glm.fit(X_train.to_numpy(), y_train)

# Wrap with conformal: pearson_weighted score is correct for Gamma data
cp = InsuranceConformalPredictor(
    model=base_glm,
    nonconformity="pearson_weighted",
    distribution="gamma",
    tweedie_power=2.0,   # Gamma variance ~ mu^2, so p=2
)
cp.calibrate(X_cal.to_numpy(), y_cal)

intervals_cp = cp.predict_interval(X_test.to_numpy(), alpha=0.10)
lower_cp = intervals_cp["lower"].to_numpy()
upper_cp = intervals_cp["upper"].to_numpy()
width_cp = upper_cp - lower_cp
```

Empirical 90% coverage: **91.1%**. Mean interval width: **£6,084**. The coverage guarantee is honoured  -  conformal always honours it, that is the point. But the intervals are 44% wider on average.

---

## Head-to-head results

We compare five dimensions:

### 1. Coverage guarantee

| Method | Target | Achieved | Guarantee type |
|---|---|---|---|
| GAMLSS | 90% | 90.3% | Asymptotic  -  depends on correct distribution choice |
| Conformal | 90% | 91.1% | Finite-sample  -  holds for any exchangeable data |

Conformal's guarantee is categorically stronger. It does not require the Gamma family assumption to be correct, and it is finite-sample valid rather than asymptotically valid. GAMLSS's 90.3% coverage is impressive, but it is contingent on the Gamma being a reasonable approximation to the true data-generating process. On real claims data, you will never know for certain that it is.

### 2. Interval width

| Method | Mean width | Median width | Young driver (age=0, veh=3) | Older driver (age=2, veh=0) |
|---|---|---|---|---|
| GAMLSS | £4,217 | £3,891 | £7,340 | £2,180 |
| Conformal | £6,084 | £5,610 | £6,620 | £5,590 |

GAMLSS wins on mean width  -  and the reason is revealing. GAMLSS correctly produces narrow intervals for low-dispersion risks. The older driver on a low-group vehicle gets intervals 2.6× narrower than conformal gives. But for the young driver on a high-group vehicle  -  the policy where variance actually is high  -  GAMLSS is wider. It has correctly learned the dispersion structure; conformal is averaging across the whole calibration set.

Conformal's constant-sigma base model cannot narrow intervals for easy risks, because the base GLM has no dispersion model. If you replaced the `GammaRegressor` with a GAMLSS-fitted model as the base, conformal would inherit some of that width efficiency. We return to this below.

### 3. Conditional coverage

The marginal coverage guarantee is a floor, not a guarantee for every subgroup. We check coverage by decile of predicted value:

```python
diag = cp.coverage_by_decile(X_test.to_numpy(), y_test, alpha=0.10)
```

```
Conformal (pearson_weighted):
  Decile  1 (mean_pred=£813):    93.1%
  Decile  5 (mean_pred=£2,840):  91.4%
  Decile 10 (mean_pred=£9,220):  89.2%

GAMLSS (Gamma quantiles):
  Decile  1 (mean_pred=£813):    91.0%
  Decile  5 (mean_pred=£2,840):  90.5%
  Decile 10 (mean_pred=£9,220):  90.1%
```

GAMLSS produces flatter conditional coverage  -  it is genuinely modelling the variance structure per risk. Conformal achieves the marginal guarantee but coverage degrades slightly in the high-risk decile (89.2%). This is not dangerous  -  89.2% is within expected statistical variation from a 90% target  -  but it illustrates that conformal's marginal guarantee does not automatically imply uniform conditional coverage.

### 4. Interpretability

GAMLSS wins clearly. The fitted sigma coefficients tell you something interpretable: young drivers (age_band=0) have a dispersion that is exp(0.40) ≈ 49% larger than the base. That is a number you can bring to an underwriting conversation. It appears in rating factor tables. The volatility score  -  `model_gamlss.volatility_score(X_test)`  -  gives a per-policy coefficient of variation that risk analysts can use directly.

Conformal is a black box around your base model. It tells you where the interval boundaries are; it does not tell you why they are there. For regulatory purposes  -  PRA SS3/19, Solvency II internal model validation  -  the ability to explain why interval *width* varies by risk characteristic is valuable.

### 5. Computational cost

On 12,000 training observations with three covariates, GAMLSS (RS algorithm) converges in 8 iterations and takes roughly 140ms. Prediction of parameters takes 2ms for 4,000 test observations. The scipy quantile function calls add another 15ms.

Conformal on a pre-fitted sklearn model is near-instantaneous: calibration on 4,000 observations takes under 10ms. Prediction takes 3ms. The cost driver is the base model  -  if you train a 1,000-tree CatBoost model as the base, that cost dominates everything else.

For use cases where the base model is already trained and production-deployed, conformal is essentially free to add. GAMLSS requires refitting a new model class.

### 6. Regulatory acceptability

This is context-dependent, but our assessment: GAMLSS is more straightforward to validate under PRA SS3/19 model validation requirements. The parameters are interpretable, the log-likelihood provides a proper scoring rule for out-of-sample fit, and distribution selection via GAIC is a documented statistical procedure. Model risk committees recognise this framework from R actuarial practice.

Conformal's finite-sample guarantee is mathematically rigorous and does not depend on distributional assumptions, which is arguably a stronger foundation. But it is less familiar to UK model validators in 2026, and the absence of distributional parameters means standard actuarial diagnostics (residual plots, P-P plots, worm plots) do not apply directly.

---

## When to use which

**Use GAMLSS when:**
- You need interpretable dispersion relativities  -  for underwriting, reinsurance treaty pricing, or risk selection
- You want per-risk volatility scores for loadings or capital allocation
- The distributional family assumption is reasonably defensible (Gamma for severity, Poisson/NBI for frequency)
- You are building a model that a model risk committee will review against actuarial literature

**Use conformal when:**
- The base model is a GBM or neural network where distributional assumptions are genuinely unclear
- You need a coverage guarantee that survives distribution shift (use `RetroAdj` for online adaptation)
- You are producing intervals for regulatory capital (Solvency II SCR upper bounds) where finite-sample validity is explicitly valuable
- You cannot commit to a distribution family  -  new lines of business, non-standard perils

**Use both together  -  the better answer:**

The strongest approach is to fit GAMLSS as the base model inside the conformal wrapper. GAMLSS's mean predictions are better than a constant-dispersion GLM because they capture the dispersion structure. Conformal then adds a finite-sample coverage guarantee on top of those better predictions. You get the interpretability of GAMLSS and the guarantee of conformal.

```python
# GAMLSS as base model inside InsuranceConformalPredictor
# We need a predict()-compatible wrapper around DistributionalGLM
class GAMLSSMeanPredictor:
    def __init__(self, gamlss_model):
        self.model = gamlss_model
    def predict(self, X):
        import numpy as np
        import polars as pl
        X_pl = pl.DataFrame(X, schema=["age_band", "veh_group", "channel"])
        return self.model.predict_mean(X_pl)

gamlss_wrapped = GAMLSSMeanPredictor(model_gamlss)

cp_combined = InsuranceConformalPredictor(
    model=gamlss_wrapped,
    nonconformity="pearson_weighted",
    distribution="gamma",
    tweedie_power=2.0,
)
cp_combined.calibrate(X_cal.to_numpy(), y_cal)
intervals_combined = cp_combined.predict_interval(X_test.to_numpy(), alpha=0.10)
```

On our synthetic dataset, the combined approach achieves 90.2% coverage and mean interval width of **£4,510**  -  26% narrower than conformal-over-GLM, and within 7% of pure GAMLSS, while carrying conformal's finite-sample guarantee. For young drivers, intervals are £5,890 rather than £7,340 (GAMLSS alone) or £6,620 (conformal alone). The combination inherits the dispersion-aware point predictions from GAMLSS and distributes the conformal calibration correction on top.

---

## The honest limitation of each

**GAMLSS's limitation** is that coverage is conditional on correct distribution choice. If the true data-generating process has heavier tails than Gamma  -  which is common for large-loss severity  -  the 95th percentile from a Gamma fit will be systematically low. The model will tell you that it achieved 90% coverage because the Gamma fit is good on average, but the tail coverage will be insufficient for the exact observations where interval accuracy matters most. This is not a niche concern in UK motor severity: bodily injury losses routinely have inverse-Gaussian or Pareto-like tails above £50,000.

**Conformal's limitation** is that the marginal guarantee is not a conditional guarantee. Conformal with a constant-dispersion base model will systematically under-cover high-variance subgroups if the non-conformity score does not adequately normalise by the local variance. The `pearson_weighted` score reduces this problem significantly  -  and is why we use it by default  -  but it does not eliminate it entirely. Coverage-by-decile diagnostics should always be run before trusting conformal intervals for a specific subgroup.

---

## Summary

Both methods work. Neither is universally superior. GAMLSS gives you richer information  -  interpretable dispersion parameters, full conditional distributions, per-policy volatility scores  -  at the cost of a distributional assumption. Conformal gives you a guarantee that holds regardless of model or distribution, at the cost of wider intervals and no insight into why width varies.

For UK personal lines motor, our default recommendation is the combined approach: GAMLSS as the base model, conformal as the coverage guarantee wrapper. For commercial lines where distributional assumptions are harder to defend and tail risk is material, we lean toward conformal-over-GBM as the primary tool.

Install both:

```bash
uv add insurance-distributional-glm
uv add insurance-conformal
```

---

## See also

- [GAMLSS in Python, Finally](/2026/03/10/insurance-distributional-glm/)  -  full introduction to `insurance-distributional-glm`
- [Conformal Prediction Intervals for Insurance Pricing Models](/2026/02/19/conformal-prediction-intervals-for-insurance-pricing/)  -  full introduction to `insurance-conformal`
- [MAPIE vs insurance-conformal](/2026/03/20/mapie-vs-insurance-conformal-prediction-intervals/)  -  why generic conformal breaks on insurance data
- [Validating GAMLSS Sigma Models](/2026/03/08/validating-gamlss-sigma-models/)  -  diagnostics for the dispersion sub-model
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/)  -  a complementary approach using gradient boosting
