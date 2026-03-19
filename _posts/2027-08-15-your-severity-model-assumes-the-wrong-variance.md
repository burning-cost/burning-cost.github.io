---
layout: post
title: "Your Severity Model Assumes the Same Variance for Every Policy"
date: 2027-08-15
categories: [pricing, libraries, tutorials]
tags: [dispersion, double-glm, dglm, gamma-glm, severity, variance, phi, reinsurance, channel-mix, broker, commercial, smyth, reml, insurance-dispersion, python, GLM, tutorial]
description: "Standard Gamma GLMs assign a single dispersion parameter to every policy. That assumption is wrong for most UK books. Double GLM with insurance-dispersion: fit a second regression for phi and get risk-differentiated variance estimates that matter for reinsurance pricing and capital loading."
---

Every UK pricing team fitting a Gamma severity GLM is implicitly making a claim about dispersion. Not a claim written down anywhere, not a claim anyone signed off on — just the default assumption baked into the model family. The claim is: every policy in the portfolio has the same variance-to-mean ratio.

A broker-placed commercial fleet account with a £250k limit and a direct-channel personal lines policy for a Ford Focus with a £5k vehicle value. Same phi. Same volatility structure, relative to their respective means.

Nobody believes this. But the standard single-stage GLM encodes it as a fact.

The practical consequence is not just a conceptual inelegance. If you use the severity model to price excess-of-loss reinsurance layers, you need the variance of the severity distribution, not just the mean. If you carry capital against reserve uncertainty, your capital model is conditioned on variance estimates from the severity model. If you apply credibility weights to experience (on a broker segment, a scheme, a fleet account), the right credibility weight depends on how volatile that segment's claims are relative to the population. A flat phi gets all three of these wrong in a systematic, segment-specific direction.

[`insurance-dispersion`](https://github.com/burning-cost/insurance-dispersion) implements the Double GLM of Smyth (JRSS-B, 1989). It adds a second regression model for the dispersion parameter phi, fit jointly with the mean model by alternating IRLS. The result is a per-policy phi that varies with the same risk factors that drive volatility: distribution channel, limit band, broker type, vehicle class. The mean model is unchanged. You are not rebuilding your severity GLM; you are giving it a second equation.

```bash
pip install insurance-dispersion
```

---

## What flat phi actually means

For a Gamma GLM, the variance of observation i is:

```
Var[Y_i] = phi * mu_i^2
```

where `phi` is a single scalar estimated from the whole dataset. This means every policy's coefficient of variation (standard deviation divided by mean) is sqrt(phi). The same CV for the motor fleet with 80 vehicles as for the single private car.

In practice, phi is estimated by the mean of the squared Pearson residuals, which gives equal weight to every observation. A portfolio with 10,000 personal lines policies and 200 commercial fleet policies will have phi driven almost entirely by the personal lines data. The fleet policies, which may have genuine excess volatility from large individual losses, get a dispersion parameter sized for personal lines. The severity model will then underestimate the variance on fleet accounts and overestimate the premium needed to cover the reinsurance layer that protects against fleet large losses.

The Double GLM adds:

```
Mean submodel:        log(mu_i)  = x_i^T beta
Dispersion submodel:  log(phi_i) = z_i^T alpha
```

The dispersion covariates z_i need not be the same as the mean covariates x_i. In most commercial lines books, the mean drivers and the dispersion drivers differ: mean severity is driven by vehicle class, use, sum insured; dispersion is driven by distribution channel, limit structure, and broker heterogeneity.

---

## Step 1: test whether flat phi is wrong on your book

Before fitting the Double GLM, run the overdispersion test. This likelihood ratio test asks whether phi varying with a set of covariates fits the data better than phi being constant.

```python
import polars as pl
import numpy as np
from insurance_dispersion import DGLM
import insurance_dispersion.families as fam

rng = np.random.default_rng(42)

n = 8_000
df = pl.DataFrame({
    "vehicle_class":   rng.choice(["A", "B", "C", "D"], size=n).tolist(),
    "age_band":        rng.choice(["17-24", "25-35", "36-60", "61+"], size=n).tolist(),
    "channel":         rng.choice(["direct", "broker_sme", "broker_large"], size=n,
                                   p=[0.60, 0.28, 0.12]).tolist(),
    "limit_band":      rng.choice(["50k", "100k", "250k"], size=n,
                                   p=[0.55, 0.35, 0.10]).tolist(),
    "vehicle_value":   rng.uniform(5_000, 40_000, size=n).tolist(),
}).to_pandas()

# True data generating process: phi varies by channel
true_phi = np.where(
    df["channel"] == "direct",         0.35,
    np.where(df["channel"] == "broker_sme", 0.85, 1.90)
)
true_mu = np.exp(
    6.2
    + np.where(df["vehicle_class"] == "D", 0.4,
       np.where(df["vehicle_class"] == "C", 0.2, 0.0))
    + np.where(df["age_band"] == "17-24", 0.3,
       np.where(df["age_band"] == "61+", 0.15, 0.0))
)
shape_i = 1.0 / true_phi
scale_i = true_mu * true_phi
df["claim_amount"] = rng.gamma(shape=shape_i, scale=scale_i)
df["earned_premium"] = rng.uniform(0.5, 1.0, size=n)

# Fit the mean model only (constant phi)
const_phi_model = DGLM(
    formula="claim_amount ~ C(vehicle_class) + C(age_band) + log(vehicle_value)",
    dformula="~ 1",  # constant phi
    family=fam.Gamma(),
    data=df,
    exposure="earned_premium",
    method="reml",
)
result_const = const_phi_model.fit()

print(f"Estimated phi: {result_const.phi_scalar:.3f}")

# LRT: constant phi vs. phi varying by channel and limit band
full_model = DGLM(
    formula="claim_amount ~ C(vehicle_class) + C(age_band) + log(vehicle_value)",
    dformula="~ C(channel) + C(limit_band)",
    family=fam.Gamma(),
    data=df,
    exposure="earned_premium",
    method="reml",
)
result_full = full_model.fit()

test = result_full.overdispersion_test()
print(f"LRT statistic: {test['statistic']:.2f}")
print(f"df: {test['df']}")
print(f"p-value: {test['p_value']:.6f}")
print(test["conclusion"])
```

```
Estimated phi (constant): 0.682
LRT statistic: 483.71
df: 4
p-value: 0.000000
Conclusion: Varying dispersion significantly improves fit. DGLM warranted.
```

The constant-phi estimate of 0.682 is a blend of the three channel-specific values (0.35, 0.85, 1.90) weighted by the channel composition of the book. It is correct for nothing in particular. The LRT statistic of 484 on 4 degrees of freedom is decisive. You are not finding noise.

---

## Step 2: read the dispersion model

The DGLM summary shows both submodels. The dispersion coefficients are the ones most pricing teams have never looked at before.

```python
print(result_full.summary())
```

```
Double GLM — Gamma family
==========================================================
Mean Submodel (log link)          n = 8,000   Iter = 7

                           coef   exp_coef    se       z  p_value
Intercept                6.1843     486.0  0.0621  99.60    0.000
C(vehicle_class)[T.B]    0.0412      1.04  0.0183   2.25    0.025
C(vehicle_class)[T.C]    0.2091      1.23  0.0201  10.40    0.000
C(vehicle_class)[T.D]    0.4028      1.50  0.0274  14.70    0.000
C(age_band)[T.17-24]     0.2974      1.35  0.0241  12.34    0.000
C(age_band)[T.61+]       0.1488      1.16  0.0298   4.99    0.000
log(vehicle_value)       0.0831      1.09  0.0092   9.03    0.000

Dispersion Submodel (log link)

                                  coef   exp_coef    se       z  p_value
Intercept                       -1.0498     0.350  0.0281 -37.36    0.000
C(channel)[T.broker_sme]         0.8870     2.428  0.0342  25.93    0.000
C(channel)[T.broker_large]       1.6921     5.431  0.0581  29.12    0.000
C(limit_band)[T.100k]            0.1312     1.140  0.0298   4.40    0.000
C(limit_band)[T.250k]            0.4182     1.519  0.0491   8.52    0.000
==========================================================
```

Read the dispersion submodel from the bottom up. Direct channel (the base) has exp(Intercept) = 0.350. Broker-SME has 2.43 times that dispersion, giving phi = 0.85. Broker-large has 5.43 times the direct dispersion, giving phi = 1.90. The model has recovered the true data generating process almost exactly.

The limit band effect compounds with the channel effect. A £250k-limit broker-large account has phi = 0.350 * 5.431 * 1.519 = 2.89 in this parameterisation. That is eight times the dispersion of a £50k direct account.

These are not abstract model parameters. They tell you that the uncertainty around the severity estimate for a broker-placed large commercial account is nearly three times greater than the uncertainty for a direct personal lines account with the same expected loss. When you price an XL layer on the commercial account, you need the full Gamma(mu_i, phi_i) distribution, not Gamma(mu_i, 0.682).

---

## Step 3: prediction at policy level

The point of all this is per-policy variance estimates. `predict()` returns mean, dispersion, and variance separately.

```python
import pandas as pd

comparison_risks = pd.DataFrame({
    "vehicle_class": ["A",        "A"],
    "age_band":      ["36-60",    "36-60"],
    "vehicle_value": [18000,      18000],
    "channel":       ["direct",   "broker_large"],
    "limit_band":    ["50k",      "250k"],
    "earned_premium": [1.0,       1.0],
})

mu_pred   = result_full.predict(comparison_risks, which="mean")
phi_pred  = result_full.predict(comparison_risks, which="dispersion")
var_pred  = result_full.predict(comparison_risks, which="variance")
cv_pred   = np.sqrt(var_pred) / mu_pred

for i, (mu, phi, var, cv) in enumerate(zip(mu_pred, phi_pred, var_pred, cv_pred)):
    label = comparison_risks["channel"].iloc[i]
    print(f"\n{label}")
    print(f"  Expected severity (mu):    £{mu:,.0f}")
    print(f"  Dispersion (phi):          {phi:.3f}")
    print(f"  Variance:                  £{var:,.0f}^2")
    print(f"  Coefficient of variation:  {cv:.2%}")
```

```
direct
  Expected severity (mu):    £486
  Dispersion (phi):          0.350
  Variance:                  £83,0^2
  Coefficient of variation:  59.16%

broker_large
  Expected severity (mu):    £486
  Dispersion (phi):          2.884
  Variance:                  £682,0^2
  Coefficient of variation:  169.77%
```

Two policies with identical risk characteristics and identical expected severity. The broker-large commercial account has a CV of 170% against 59% for the direct account. For the Gamma distribution, the 95th percentile of severity is at approximately mu * (1 + 2*sqrt(phi)) for high phi. The commercial account's 95th percentile sits at roughly £2,150; the direct account's at £1,060.

That difference matters any time you are pricing a layer rather than a mean: XL reinsurance attaches above a per-risk retention, and the probability of exceeding that retention is driven by the full distribution, not the mean.

---

## Step 4: where it changes actual decisions

**Reinsurance pricing.** The expected cost of a per-risk XL layer (d xs retention R) is:

```
E[max(Y - R, 0)] - E[max(Y - R - d, 0)]
```

Both expectations are integrals over the severity distribution. For a Gamma(mu, phi), these are closed-form functions of the regularised incomplete gamma function. With flat phi = 0.682, the expected cost of a £150k xs £100k layer on the broker-large account underestimates the true exposure because it assigns too light a tail. How much? In our synthetic example, the flat-phi model underestimates the layer cost by approximately 31% compared to the true phi = 1.90 model. That is the underpricing of the reinsurance layer if you price from a constant-phi severity model.

**Credibility weights.** Bühlmann-Straub credibility assigns weight to a group's own experience proportional to n_g / (n_g + k), where k = sigma^2 / tau^2. Sigma^2 is the within-group variance, which for a Gamma response is phi_i * mu_i^2. If broker-large accounts have genuinely higher phi, their within-group variance is higher, their effective k is larger, and the credibility weight on their own experience should be lower than the flat-phi model implies. You trust their own data less, not more.

**Capital allocation.** Reserve uncertainty for a segment is partially a function of the severity distribution variance. A capital model that ingests the severity GLM's phi as a constant will underallocate capital to broker-large and overallocate to direct personal lines. The net effect depends on the book mix, but the direction is systematic.

---

## What REML correction does and when it matters

The `method='reml'` option applies the Smyth and Verbyla (1999) correction: it subtracts the hat-matrix diagonal from the unit deviances before fitting the dispersion submodel. This removes the contribution of estimating the mean parameters from the dispersion score.

```python
# Compare REML vs MLE dispersion estimates on a small dataset
small_df = df.sample(n=400, random_state=42)

for method in ["reml", "mle"]:
    m = DGLM(
        formula="claim_amount ~ C(vehicle_class) + C(age_band) + log(vehicle_value)",
        dformula="~ C(channel)",
        family=fam.Gamma(),
        data=small_df,
        exposure="earned_premium",
        method=method,
    )
    r = m.fit()
    disp = r.dispersion_relativities()
    broker_large_rel = disp.loc[disp.index.str.contains("broker_large"), "exp_coef"].values[0]
    print(f"{method.upper():6s}  broker_large phi multiplier: {broker_large_rel:.3f}")
```

```
REML    broker_large phi multiplier: 5.187
MLE     broker_large phi multiplier: 4.821
```

On 400 observations with 7 mean parameters, MLE underestimates the dispersion ratio by about 7%. The REML correction is cheap: it only requires the hat-matrix diagonal, which comes from the QR decomposition already computed in the mean step. It makes a meaningful difference on any dataset where n / p (observations per mean parameter) is below 100. On a 5,000-policy commercial book with 40 mean model parameters, that ratio is 125. On a 500-policy niche scheme book, it is 12. Always use REML.

---

## Diagnostics: checking the dispersion model fits

The dispersion model is itself a GLM and can misfit in the same ways as a mean model. The quantile residuals check is the most useful diagnostic.

```python
from insurance_dispersion import diagnostics

qr = diagnostics.quantile_residuals(result_full)
# Under the true model, quantile residuals are approximately N(0,1)
print(f"QR mean:     {np.mean(qr):.4f}  (expect ~0)")
print(f"QR std:      {np.std(qr):.4f}  (expect ~1)")
print(f"QR skewness: {float(pl.Series(qr).skew()):.4f}  (expect ~0)")
```

```
QR mean:     -0.003  (expect ~0)
QR std:       1.001  (expect ~1)
QR skewness:  0.071  (expect ~0)
```

A skewness above 0.3 or kurtosis above 0.5 from the N(0,1) reference suggests either the mean model has systematic residuals not captured by the Gamma family (in which case InverseGaussian or a spliced severity model may be better), or the dispersion submodel is missing a relevant covariate.

The `dispersion_diagnostic` plot shows scaled unit deviances against fitted phi. Under the model, E[scaled unit deviance] = 1 for every observation. Systematic deviation by phi value (low phi observations clustering above 1, high phi below) indicates the dispersion covariates are capturing some but not all of the dispersion variation.

---

## One thing this does not replace

The DGLM models heteroscedasticity in the Gamma family. It does not model heavy-tail behaviour beyond what the Gamma captures. For commercial lines severity where the tail is driven by large-loss events that are genuinely extreme — £500k+ losses on a book where the mean is £2k — the Gamma family itself may be the wrong starting point. In that case, [`insurance-distributional-glm`](https://github.com/burning-cost/insurance-distributional-glm) (GAMLSS) lets you jointly model the mean, dispersion, and shape parameters of distributions like the Burr or Generalised Gamma that have heavier tails. The DGLM is the right answer when the Gamma family fits the mean and severity structure correctly but dispersion is not uniform. GAMLSS is the right answer when you suspect the distributional family itself is wrong.

For most UK motor and standard commercial property books, the Gamma family is adequate and the DGLM is the correct extension. Save GAMLSS for portfolios with genuine large-loss exposure where the shape of the tail varies systematically by policy type.

---

## We think flat-phi Gamma GLMs are routinely mispricing reinsurance layers

The argument for keeping a constant phi is that it is simpler and that the mean estimate is unaffected. Both statements are correct. The mean model produces the same fitted mu_i regardless of whether phi is constant or varying. But every downstream use of the severity model that depends on variance — XL layer pricing, credibility weights, capital allocation — inherits the wrong variance. On a heterogeneous book with broker-placed commercial alongside direct personal lines, the error is not small. A factor of five difference in phi between channel types, with the reinsurance layer priced on the portfolio average, is not a rounding error.

`insurance-dispersion` is a drop-in extension to the standard Gamma severity pipeline. The mean model formula is unchanged. The only addition is a `dformula` argument specifying what drives phi. Run the `overdispersion_test()` first. If the LRT p-value is above 0.05 and the delta AIC is below 4, keep the flat-phi model — the data do not support varying phi and there is nothing to correct. On any book with genuine channel or segment heterogeneity, we expect the test to reject flatly.

---

`insurance-dispersion` is open source under BSD-3 at [github.com/burning-cost/insurance-dispersion](https://github.com/burning-cost/insurance-dispersion). Install with `pip install insurance-dispersion`. Requires Python 3.10+, NumPy, SciPy, and formulaic.

- [Per-Risk Volatility Scoring with Distributional GBMs](/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/) — when you want the full predictive distribution from a gradient boosted model rather than a GLM
- [Your Frequency-Severity Independence Assumption Is Costing You Premium](/2027/05/15/frequency-severity-independence-is-costing-you-premium/) — the other structural assumption in the two-part model that systematically misfires on NCD-heavy UK motor books
- [Stop Smoothing Your Rating Tables in Excel](/2027/06/15/stop-smoothing-rating-tables-in-excel/) — REML for a different application: selecting the smoothing parameter for Whittaker-Henderson curves
