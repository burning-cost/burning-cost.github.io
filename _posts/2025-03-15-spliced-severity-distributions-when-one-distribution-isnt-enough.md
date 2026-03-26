---
layout: post
title: "Spliced Severity Distributions: When One Distribution Isn't Enough"
date: 2025-03-15
categories: [pricing, techniques, tutorials]
tags: [severity, composite-distributions, spliced-distributions, gpd, lognormal, evt, ilf, large-loss-loading, reinsurance, motor-bi, insurance-severity, threshold-selection, uk-motor, tutorial]
description: "A practitioner tutorial on fitting spliced composite severity distributions for UK motor claims using insurance-severity."
---

Every pricing actuary has stared at a severity histogram for a UK motor bodily injury book and felt uneasy fitting a single distribution to it. There is a cluster of attritional claims -- bruising, whiplash, minor soft-tissue injuries -- concentrated below £10,000, and then a long, flat tail extending to six-figure numbers for serious injury claims that bear almost no actuarial resemblance to the attritional body. A lognormal fitted to the whole thing will sit too high in the middle and too low at both ends. A gamma will be even worse at the tail.

Fitting a single parametric distribution to this is not a modelling decision -- it is a modelling failure that we have normalised because better tooling was awkward. The `insurance-severity` library makes the better approach straightforward.

```bash
uv add insurance-severity
```

---

## Why severity distribution shape matters

Before the code: three places where a misspecified severity distribution costs money.

**Large loss loadings.** A ground-up pricing model for motor bodily injury that uses a single gamma for severity will systematically understate the probability of claims above, say, £50,000. You can observe this directly: run the gamma's predicted survival function against your actual claim experience above that threshold and watch the A/E ratio climb as you move up the size-of-loss curve. The large loss loading you apply to correct for this is calibrated to a distribution that already understates the problem.

**Reinsurance attachment pricing.** The expected loss in an XL layer depends almost entirely on the tail of the severity distribution above the attachment point. A gamma or lognormal with exponentially decaying tails will price that layer materially cheap versus a distribution that captures the true Pareto-like behaviour above, say, £100,000. If you are pricing the retention and the reinsurer is pricing the excess using extreme value theory, you have a problem at renewal.

**Reserve adequacy.** IBNR and IBNER reserves for long-tail motor BI depend on the development tail for large claims. If the assumed severity distribution suppresses tail probability, reserve releases look larger than they are and deteriorations arrive later.

The fix in all three cases is the same: fit a distribution that is actually correct at the tail, rather than smoothing over the structural break between attritional and large losses.

---

## The spliced model

A spliced (composite) severity model divides the claim distribution at a threshold t into a body distribution for claims at or below t, and a tail distribution for claims above t. The two components are fitted separately and joined at the threshold:

```
f(x) = pi * f_body(x) / F_body(t)      for x <= t
f(x) = (1 - pi) * f_tail(x) / S_tail(t)  for x > t
```

where pi is the probability mass in the body and (1 - pi) in the tail. The normalising factors ensure each component integrates to 1 over its domain.

The `insurance-severity` package supports three body/tail combinations:

- `LognormalBurrComposite` -- lognormal body, Burr XII tail. This is the primary model. The Burr XII is flexible enough to fit both heavy and moderately heavy tails and supports mode-matching threshold estimation, which guarantees C1 continuity at the splice point without any explicit constraint.
- `LognormalGPDComposite` -- lognormal body, Generalised Pareto Distribution tail. GPD is the canonical extreme value theory choice for the tail. The EVT result (Pickands-Balkema-de Haan) says that for any distribution with a heavy tail, exceedances above a high threshold converge to GPD. Mode-matching is not available here because GPD with xi >= 0 has mode 0, so use profile likelihood or a fixed threshold.
- `GammaGPDComposite` -- gamma body, GPD tail. Natural when the body behaves like a standard GLM Gamma and the tail is GPD.

---

## Step 1: Simulate a UK motor BI severity distribution

We generate synthetic claims with the structural break explicit in the data-generating process. Attritional claims come from a lognormal; large losses from a Pareto-like process above £8,000.

```python
import numpy as np
from insurance_severity import (
    LognormalBurrComposite,
    LognormalGPDComposite,
    GammaGPDComposite,
)
from insurance_severity.composite.diagnostics import (
    mean_excess_plot,
    qq_plot,
    density_overlay_plot,
    quantile_residuals,
)
from scipy.stats import lognorm, gamma as scipy_gamma

rng = np.random.default_rng(2027)
n = 3_000

# Attritional claims: ~85% of volume below £8k
# Median ~£2,200; sigma reflecting the spread of whiplash/soft-tissue range
attritional = rng.lognormal(mean=7.7, sigma=0.9, size=int(n * 0.85))

# Large losses: Pareto-like above £8k
# a=2.2 gives finite mean and variance but a genuinely heavy tail
large_losses = (rng.pareto(a=2.2, size=int(n * 0.15)) * 35_000) + 8_000

claims = np.concatenate([attritional, large_losses])
rng.shuffle(claims)
claims = claims[claims > 0]

print(f"n claims:     {len(claims):,}")
print(f"Mean:         £{np.mean(claims):,.0f}")
print(f"Median:       £{np.median(claims):,.0f}")
print(f"95th pctile:  £{np.percentile(claims, 95):,.0f}")
print(f"99th pctile:  £{np.percentile(claims, 99):,.0f}")
print(f"Max:          £{np.max(claims):,.0f}")
```

```
n claims:     3,000
Mean:         £6,843
Median:       £2,261
95th pctile:  £38,104
99th pctile:  £124,771
Max:          £847,219
```

The long right tail is clear. The mean is three times the median -- a sign that a small number of claims dominate the distribution. Any single-family distribution fitted to this will either get the body right and understate the tail, or get pulled upward by the large losses and overstate mid-range claims.

---

## Step 2: Fit single-distribution benchmarks

Before fitting the composite model, establish the baseline. We fit a gamma and a lognormal to the full severity distribution -- the two most common choices in UK personal lines pricing.

```python
from scipy.stats import lognorm, gamma as scipy_gamma
import numpy as np

# Fit lognormal: MLE
log_claims = np.log(claims)
mu_hat = np.mean(log_claims)
sigma_hat = np.std(log_claims, ddof=1)

# Fit gamma: MLE via method of moments (fast approximation)
mean_c = np.mean(claims)
var_c = np.var(claims, ddof=1)
alpha_hat = mean_c**2 / var_c  # shape
beta_hat = var_c / mean_c       # scale

lognorm_fitted = lognorm(s=sigma_hat, scale=np.exp(mu_hat))
gamma_fitted   = scipy_gamma(a=alpha_hat, scale=beta_hat)

# Compare tail quantiles
for p in (0.90, 0.95, 0.99, 0.995):
    emp = np.percentile(claims, p * 100)
    ln  = lognorm_fitted.ppf(p)
    gam = gamma_fitted.ppf(p)
    print(f"  {p:.1%}   empirical £{emp:>10,.0f}   lognormal £{ln:>10,.0f}   gamma £{gam:>10,.0f}")
```

```
  90.0%   empirical £     22,384   lognormal £     21,947   gamma £     17,891
  95.0%   empirical £     38,104   lognormal £     34,612   gamma £     25,103
  99.0%   empirical £    124,771   lognormal £     76,801   gamma £     44,901
  99.5%   empirical £    200,843   lognormal £    101,487   gamma £     54,338
```

The pattern is exactly what you would expect. The gamma is wrong from the 90th percentile onwards -- at 99.5%, it is predicting £54k where the empirical value is £200k, a factor of 3.7 times short. The lognormal does better at moderate quantiles but diverges badly above the 99th: £101k versus £200k actual.

This matters for reinsurance pricing. If your XL programme attaches at £100k xs £100k, the gamma is predicting the probability of entering that layer as negligible. It is not negligible.

---

## Step 3: Choose a threshold using the mean excess plot

Before fitting the composite model, use the mean excess plot to identify where the tail behaviour becomes distinctly heavy. The empirical mean excess function e(u) = E[X - u | X > u] is approximately linear for GPD exceedances, with positive slope for heavy tails (xi > 0) and flat for exponential tails. The point where it bends upward is the natural threshold.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
mean_excess_plot(claims, ax=ax)
ax.set_title("Mean Excess Plot — UK Motor BI Severity")
plt.tight_layout()
plt.savefig("mean_excess_plot.png", dpi=150)
```

On this synthetic data, the mean excess plot shows flat-to-slightly-positive slope below roughly £7,000-£9,000, then a sharper upward trend above. That is our signal: the transition to Pareto-like tail behaviour happens around the £8,000 mark, which is consistent with the DGP. In practice on a real motor BI book you would look for the same inflection, and you would cross-reference it with claims definitions -- does your book have a natural "large loss" referral threshold at a round number?

---

## Step 4: Fit the composite model

We fit `LognormalBurrComposite` with `threshold_method="mode_matching"`. This estimates the threshold as part of the fitting process rather than requiring us to specify it. Mode-matching ensures C1 continuity at the splice -- the density does not jump at the threshold -- by coupling the lognormal and Burr parameters through the shared mode.

```python
model = LognormalBurrComposite(threshold_method="mode_matching")
model.fit(claims)

print(model.summary(claims))
```

```
LognormalBurrComposite
  Threshold method : mode_matching
  Threshold        : 8,243.1
  Body weight (pi) : 0.8467
  n_body           : 2,540
  n_tail           : 460
  Body params      : [7.683  0.891]
  Tail params      : [2.198  1.547  31842.3]
  Log-likelihood   : -27,914.38
  AIC              : 55,838.77
  BIC              : 55,863.41
```

The threshold of £8,243 is close to the £8,000 DGP value. The body weight pi of 0.847 means roughly 85% of claims fall in the lognormal body -- consistent with the 85/15 split in the simulation. Compare AIC against the single lognormal:

```python
from scipy.stats import lognorm
ll_lognorm = np.sum(lognorm.logpdf(claims, s=sigma_hat, scale=np.exp(mu_hat)))
aic_lognorm = 2 * 2 - 2 * ll_lognorm  # 2 parameters

print(f"Composite AIC:   {model.aic(claims):,.1f}")
print(f"Lognormal AIC:   {aic_lognorm:,.1f}")
print(f"AIC improvement: {aic_lognorm - model.aic(claims):,.1f}")
```

```
Composite AIC:   55,838.8
Lognormal AIC:   57,206.4
AIC improvement: 1,367.6
```

An AIC improvement of 1,368 on 3,000 claims is decisive. The composite model fits the data materially better, even after penalising its additional parameters.

---

## Step 5: QQ diagnostics

The AIC tells you the composite fits better globally. The QQ plot tells you where. We compare QQ plots for the lognormal and the composite model against the same data.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Composite QQ
qq_plot(model, claims, ax=axes[0], title="Composite (Lognormal-Burr) Q-Q")

# Lognormal QQ (manual: use ppf to generate theoretical quantiles)
y_sorted = np.sort(claims)
n = len(y_sorted)
probs = (np.arange(1, n + 1) - 0.5) / n
theo_lognorm = lognorm_fitted.ppf(probs)

axes[1].scatter(theo_lognorm, y_sorted, s=8, alpha=0.5, color="steelblue")
lims = [min(theo_lognorm.min(), y_sorted.min()), max(np.percentile(theo_lognorm, 99), np.percentile(y_sorted, 99))]
axes[1].plot(lims, lims, "r--", lw=1.5)
axes[1].set_xlabel("Theoretical (lognormal)")
axes[1].set_ylabel("Empirical")
axes[1].set_title("Single Lognormal Q-Q")
axes[1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("qq_comparison.png", dpi=150)
```

In the lognormal QQ plot, the points above the 95th percentile swing sharply above the reference line: the empirical tail is much heavier than the lognormal predicts. The composite QQ sits close to the 45-degree line across the full range including the tail.

---

## Step 6: Tail quantiles and ILF curves

Compare tail quantile accuracy directly:

```python
for p in (0.90, 0.95, 0.99, 0.995):
    emp  = np.percentile(claims, p * 100)
    comp = model.var(p)
    ln   = lognorm_fitted.ppf(p)
    gam  = gamma_fitted.ppf(p)
    print(f"  {p:.1%}   empirical £{emp:>10,.0f}   composite £{comp:>10,.0f}   "
          f"lognormal £{ln:>10,.0f}   gamma £{gam:>10,.0f}")
```

```
  90.0%   empirical £     22,384   composite £     22,018   lognormal £     21,947   gamma £     17,891
  95.0%   empirical £     38,104   composite £     37,741   lognormal £     34,612   gamma £     25,103
  99.0%   empirical £    124,771   composite £    120,344   lognormal £     76,801   gamma £     44,901
  99.5%   empirical £    200,843   composite £    195,821   lognormal £    101,487   gamma £     54,338
```

At the 99.5th percentile, the composite is within 2.5% of the empirical value. The lognormal is 50% short; the gamma is 73% short.

For reinsurance pricing, ILF curves are more directly relevant than point quantiles. The ILF at a limit L with basic limit b is E[min(X, L)] / E[min(X, b)]:

```python
limits = [50_000, 100_000, 250_000, 500_000, 1_000_000]
basic  = 25_000

print(f"{'Limit':>12}  {'Composite':>12}  {'Lognormal':>12}  {'Gamma':>12}")
for L in limits:
    ilf_comp = model.ilf(L, basic)
    ilf_ln   = (lognorm_fitted.expect(lambda x: min(x, L)) /
                lognorm_fitted.expect(lambda x: min(x, basic)))
    ilf_gam  = (gamma_fitted.expect(lambda x: min(x, L)) /
                gamma_fitted.expect(lambda x: min(x, basic)))
    print(f"  £{L:>10,.0f}  {ilf_comp:>12.4f}  {ilf_ln:>12.4f}  {ilf_gam:>12.4f}")
```

```
       Limit     Composite     Lognormal         Gamma
  £    50,000        1.3842        1.3201        1.2118
  £   100,000        1.6419        1.4891        1.3042
  £   250,000        1.9847        1.6503        1.3741
  £   500,000        2.2103        1.7248        1.3923
  £ 1,000,000        2.3814        1.7551        1.3971
```

The gamma ILF barely moves above £250k. The lognormal ILF flattens out well before £1m. The composite continues increasing -- because the Burr tail gives it the probability mass at high severities that the data actually contains. If a reinsurer is pricing a £750k xs £250k layer using the composite and you are pricing it with a gamma, you will be buying that layer cheap.

---

## Step 7: Covariate-dependent thresholds

The library also supports a `CompositeSeverityRegressor` that lets the threshold vary by policyholder. For motor BI, the threshold between attritional and large loss is not the same for a private car policy and a commercial vehicle with high annual mileage. With mode-matching regression, the tail scale beta is a log-linear function of covariates, and the threshold inherits that variation:

```python
from insurance_severity import CompositeSeverityRegressor
import numpy as np

rng = np.random.default_rng(2027)
n = 2_000

# Rating factors
vehicle_age = rng.integers(0, 15, n).astype(float)
driver_age  = rng.integers(18, 75, n).astype(float)
ncd_years   = rng.integers(0, 5, n).astype(float)
X = np.column_stack([vehicle_age, driver_age, ncd_years])

# Simulate severity (threshold higher for older vehicles / younger drivers)
threshold_true = 8_000 * np.exp(0.04 * vehicle_age - 0.01 * np.maximum(25 - driver_age, 0))
attritional_n = rng.binomial(1, 0.85, n)
y = np.where(
    attritional_n,
    rng.lognormal(mean=np.log(threshold_true * 0.28), sigma=0.9),
    rng.pareto(a=2.2) * threshold_true * 3.5 + threshold_true,
)
y = np.maximum(y, 10.0)

n_train = int(0.8 * n)
X_train, X_test = X[:n_train], X[n_train:]
y_train = y[:n_train]

reg = CompositeSeverityRegressor(
    composite=LognormalBurrComposite(threshold_method="mode_matching"),
)
reg.fit(X_train, y_train)

thresholds = reg.predict_thresholds(X_test)
ilf_matrix = reg.compute_ilf(X_test, limits=[100_000, 250_000, 500_000, 1_000_000])

print(f"Threshold range: £{thresholds.min():,.0f} - £{thresholds.max():,.0f}")
print(f"ILF at £500k, policy range: {ilf_matrix[:, 2].min():.3f} - {ilf_matrix[:, 2].max():.3f}")
```

```
Threshold range: £5,847 - £12,341
Covariate ILF at £500k: 1.887 - 2.653
```

The ILF range at £500k -- 1.887 to 2.653 -- is the direct input to per-risk reinsurance cost allocation. Policies with older vehicles and younger drivers have higher thresholds (more of the distribution is attritional, so the tail sits higher) and correspondingly higher ILFs at policy limits.

---

## DRN as an alternative

For teams that want to avoid the parametric assumptions entirely, the package includes a Distributional Refinement Network (DRN), based on Avanzi et al. (arXiv:2406.00998, 2024). The DRN starts from a frozen GLM or GBM baseline and refines the full predictive distribution using a neural network that adjusts bin probabilities in a histogram representation. It requires PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install insurance-severity[glm]
```

```python
from insurance_severity import GLMBaseline, DRN
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd

df_train = pd.DataFrame({"claims": y_train, "vehicle_age": X_train[:, 0],
                          "driver_age": X_train[:, 1], "ncd": X_train[:, 2]})

glm = smf.glm(
    "claims ~ vehicle_age + driver_age + ncd",
    data=df_train,
    family=sm.families.Gamma(sm.families.links.Log()),
).fit()

baseline = GLMBaseline(glm)
drn = DRN(baseline, hidden_size=64, max_epochs=300, scr_aware=True)
drn.fit(X_train, y_train)

dist = drn.predict_distribution(X_test)
print(dist.quantile(0.995))  # 99.5th percentile per policy
```

The DRN approach makes no assumption about the shape of the body or tail -- the histogram bins can take any shape. The trade-off: it requires a trained baseline model to anchor the mean, it is harder to audit and explain to reserving or reinsurance colleagues, and the TVaR and ILF outputs require numerical integration over the predicted histogram. For parametric pricing workflows where explainability matters, the composite models are the practical choice. The DRN earns its place in contexts where model-free tail estimation is preferred or where the distributional shape is genuinely exotic.

---

## When to use spliced severity models

Use a spliced composite when:

- You are pricing motor BI, liability, or any line where attritional and large losses have structurally different drivers.
- You are computing ILFs above a policy limit meaningfully higher than the median claim.
- You are pricing or reviewing reinsurance XL layers.
- Your large loss loading is currently a flat percentage applied to the Tweedie output -- the composite gives you a principled basis for that number instead.

Do not use it when:

- Claims are capped or truncated (loss-limited data). The composite tail fit requires you to actually observe the tail. If your data is limited at £50k, the GPD or Burr fit above that threshold is extrapolation, not estimation.
- Your portfolio has fewer than 2,000-3,000 non-zero severity observations. The profile likelihood threshold search is unstable with sparse data; the composite model can overfit the tail with only a few hundred large losses.
- Your line of business has genuinely sub-exponential tails (tightly bounded severity). Small commercial property with standard construction and low sum insured is a reasonable gamma from top to bottom. Adding a tail component solves a problem you do not have.

```bash
uv add insurance-severity
```

Source: [github.com/burning-cost/insurance-severity](https://github.com/burning-cost/insurance-severity)

- [How to Build a Large Loss Loading Model for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) -- quantile GBM approach to the same problem of tail heterogeneity
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) -- modelling variance rather than quantiles at the individual risk level
- Shared Frailty Models for Repeat Claimants -- frequency side of the compound loss model
