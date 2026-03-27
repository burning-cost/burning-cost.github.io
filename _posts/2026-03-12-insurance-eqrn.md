---
layout: post
title: "EQRN: Covariate-Dependent Extreme Value Modelling for XL Pricing and Tail Reserving"
date: 2026-03-12
categories: [libraries, pricing, severity, reinsurance]
tags: [EVT, extreme-value-theory, GPD, neural-networks, extreme-quantile, EQRN, covariate-dependent, tail-modelling, XL-pricing, reinsurance, Solvency-II, TPBI, motor, insurance-quantile, python, pytorch]
description: "EQRN covariate-dependent GPD tail modelling for XL pricing. Per-risk shape and scale via neural networks - insurance-quantile Python, Pasche and Engelke (2024)."
---

<div class="notice--warning" markdown="1">
**Package update:** `insurance-eqrn` has been consolidated into [`insurance-quantile`](/insurance-distributional/). Install with `pip install insurance-quantile` — all functionality described here is available as a submodule. [View on GitHub →](https://github.com/burning-cost/insurance-quantile)
</div>


Standard GPD fitting gives you one shape parameter and one scale parameter for the whole book. That is fine for portfolio capital reporting. It is not fine for per-risk XL pricing, TPBI reserving by driver profile, or any use case where tail heaviness varies across your portfolio. EQRN - Extreme Quantile Regression Neural Networks, from Pasche & Engelke's 2024 paper in the *Annals of Applied Statistics* - solves this. [`insurance-quantile`](https://github.com/burning-cost/insurance-quantile) is the first Python implementation.

```bash
uv add insurance-quantile
```

---

## What the pooled model gets wrong

Fit a GPD to your motor bodily injury claims above £500k and you get one xi and one sigma. The 99.5th percentile severity is the same for every policyholder. That is clearly wrong: catastrophic TPBI claims involving young injured parties with decades of annuity ahead of them have a heavier tail than claims involving older parties with shorter projected futures. The pooled shape parameter averages these together. Your XL pricing for the younger-driver segment is too cheap; for the older-driver segment it is too expensive.

The same problem appears in property: the tail for timber-frame commercial buildings differs from masonry. In liability: claims involving solicitor representation have a different tail from direct settlements. Anywhere the covariate structure of the claim changes the shape of the extreme tail, a single GPD is wrong.

What you need is xi(x) and sigma(x) - GPD parameters as functions of risk characteristics rather than pooled scalars. This is what Pasche & Engelke (2024) call the conditional tail approximation: for a covariate vector x and intermediate threshold u(x) = Q_x(tau_0),

```
P(Y > y | X=x) ≈ (1 - tau_0) * (1 + xi(x) * (y - u(x)) / sigma(x))^(-1/xi(x))
```

Inverting gives the extreme conditional quantile:

```
Q_x(tau) = Q_x(tau_0) + sigma(x)/xi(x) * [((1-tau_0)/(1-tau))^xi(x) - 1]
```

Valid for tau well above tau_0. The neural network learns xi(x) and sigma(x) from data.

---

## The two-step method

EQRN fits in two stages. Neither step can be skipped without breaking the other.

**Step 1: Intermediate quantile with CatBoost (out-of-fold)**

Fit a quantile regression at a moderate level - tau_0 = 0.8 or 0.85 - using K-fold cross-validation. The critical requirement is that the intermediate quantile predictions used in Step 2 must be out-of-fold. In-sample predictions give artificially accurate thresholds, and the network in Step 2 then learns the wrong exceedance set.

```python
from insurance_quantile.eqrn import EQRNModel

model = EQRNModel(
    tau_0=0.85,
    hidden_sizes=(32, 16, 8),
    n_epochs=300,
    shape_fixed=False,   # covariate-dependent xi
    n_folds=5,           # K-fold OOF for Step 1
    seed=42,
)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```

**Step 2: GPD neural network on exceedances**

Observations above their predicted threshold (~15–20% of training data at tau_0=0.85) form the exceedance set. The network maps (X, Q_hat(tau_0)) - covariates plus the intermediate quantile as an extra feature - to (nu(x), xi(x)).

The orthogonal reparameterisation is the key numerical trick. Rather than fitting sigma(x) and xi(x) directly, the network fits nu(x) = sigma(x) * (xi(x) + 1) and xi(x). This makes the Fisher information matrix diagonal, which stabilises Adam training substantially. In the (sigma, xi) parameterisation, gradient descent oscillates between the two parameters. In (nu, xi), it does not.

The xi output is constrained to (-0.5, 0.7) via a scaled tanh: `0.6 * tanh(z) + 0.1`. This covers all practical insurance cases - UK motor TPBI typically sits in xi ∈ [0.30, 0.55], property large loss around [0.20, 0.40] - while preventing numerical blowup during early training.

---

## Prediction at extreme quantile levels

```python
# Per-segment 99.5th percentile severity
var_995 = model.predict_quantile(X_test, q=0.995)

# TVaR for layer pricing
tvar_99 = model.predict_tvar(X_test, q=0.99)

# Per-risk XL: £500k xs £500k
xl_loss = model.predict_xl_layer(X_test, attachment=500_000, limit=500_000)

# GPD parameters per observation
params = model.predict_params(X_test)
# DataFrame: xi, sigma, nu, threshold
```

The XL layer pricing uses the closed-form expected loss in a layer conditional on the fitted GPD - exact under the GPD assumption, not a simulation. For an XL tower, price each layer by calling `predict_xl_layer` with the appropriate attachment and limit per policy.

For exceedance probability - useful for binder underwriting and risk screening:

```python
# P(claim > £1m | risk profile)
exceed_prob = model.predict_exceedance_prob(X_test, threshold=1_000_000)
```

---

## Diagnostics

The three plots you need before trusting any EQRN output:

```python
from insurance_quantile.eqrn import EQRNDiagnostics

diag = EQRNDiagnostics(model)

# Threshold stability: fit shape_fixed models at each tau_0 level
# Look for a plateau in xi — that is your valid tau_0 range
diag.threshold_stability_plot(X_train, y_train)

# GPD QQ plot on exceedances — should track the diagonal
# Systematic deviation above the line means the tail is heavier than estimated
diag.qq_plot(X_test, y_test)

# Coverage calibration: predicted vs empirical exceedance rates
# At q=0.99, roughly 1% of test observations should exceed predict_quantile
diag.calibration_plot(X_test, y_test, levels=[0.9, 0.95, 0.99, 0.995])
```

The calibration plot is the most practically important. Systematic undercoverage at extreme levels means the model is underestimating tail risk - dangerous for capital work. Systematic overcoverage is conservative - acceptable for pricing, costly for competitiveness.

---

## When to start with shape_fixed=True

The full model - covariate-dependent xi(x) and sigma(x) - requires enough data in the exceedance set to estimate a surface rather than a scalar. Below roughly 500 exceedances, the shape surface estimates become unstable. Start with `shape_fixed=True`:

```python
model_fixed = EQRNModel(
    tau_0=0.85,
    shape_fixed=True,   # scalar xi, covariate-dependent sigma only
    seed=42,
)
model_fixed.fit(X_train, y_train)
```

This fits a global xi with covariate-dependent scale - a sensible intermediate between pooled GPD and the full EQRN. Standard EVT wisdom applies: xi is harder to estimate than sigma; fixing it when data is limited often improves out-of-sample performance. Move to `shape_fixed=False` once you have at least 500 exceedances and the stability plot shows consistent estimates across tau_0 levels.

---

## Insurance applications

**Motor TPBI per driver profile**

Injured party age, vehicle type, claim type, and solicitor involvement all change the tail shape. EQRN gives you P(claim > £500k | risk profile) per policy - the right input for per-risk XL attachment analysis and Solvency II segment-level VaR.

**Commercial property large loss**

Construction class, sum insured, sprinkler status, flood zone: the tail for a non-sprinklered timber-frame unit is not the tail for a sprinklered masonry office. EQRN provides a 1-in-200 loss estimate conditional on risk characteristics, which is the correct input to CAT reinsurance programme design.

**Per-risk XL pricing from first principles**

```python
# Price layer: £1m xs £500k, conditional on policyholder risk profile
xl = model.predict_xl_layer(X_cedant, attachment=500_000, limit=1_000_000)
# xl is an array: one expected layer loss per policy
print(f"Layer expected loss: £{xl.sum():,.0f}")
```

Current market practice for per-risk XL pricing typically uses a marginal GPD fit to the cedant's aggregate losses - no covariate conditioning. EQRN provides per-class burn cost, which is a more defensible basis for bespoke programmes where the cedant's book has a systematic risk skew.

**Solvency II internal models**

The Pasche & Engelke (2024) publication in *Annals of Applied Statistics* - a peer-reviewed journal, not an industry white paper - provides the academic credential that regulators expect for covariate-dependent tail estimation methodology. Segment-level conditional VaR at 99.5% from EQRN is more conservative for high-risk segments and more accurate for low-risk segments than a pooled EVT estimate applied uniformly.

---

## Where EQRN sits in the toolkit

Three other libraries address tail risk but none addresses covariate-dependent EVT:

- **[insurance-evt](/2026/03/13/insurance-evt/)**: marginal GPD/GEV, no covariate conditioning, full censored MLE and profile likelihood CI. Use it for portfolio-level threshold selection and return levels. EQRN extends this to segment level.
- **[insurance-quantile](/2026/03/07/insurance-quantile/)**: CatBoost quantile regression, covariate-dependent, but no EVT extrapolation. Reliable to the 95th percentile; for the 99.5th percentile there is no theoretical basis for direct quantile regression beyond the training data range. EQRN adds the EVT extrapolation.
- **[insurance-nflow](/2026/03/12/insurance-nflow/)**: full conditional density, flexible, but no asymptotic tail guarantees. For extreme quantile work, EQRN is lighter, tail-specific, and theoretically grounded.

The gap EQRN fills: covariate-dependent extrapolation to extreme quantile levels using EVT theory. No other Python library does this.

---

## The Python gap this fills

The CRAN EQRN package (Pasche, March 2025) is the only prior implementation. Four conditional EVT methods exist in the literature: EQRN (neural networks), GBEX (gradient boosted trees, Velthoen et al. 2023, *Extremes*), ERF (extremal random forests, Gnecco et al. 2024, *JASA*), and evgam (GAMs for extremes, Youngman 2019). All four are R-only. `insurance-quantile` is the first Python implementation of any of them. The simulation study in Pasche & Engelke (2024) shows EQRN outperforms the other three methods in high-dimensional settings (p > 10 covariates), which is representative of most insurance rating factor sets.

---

## Practical notes

**Sample size.** You need roughly 1,000 claims above the basic reporting threshold for the exceedance set to be large enough for the GPD network to train reliably. Below that, use `shape_fixed=True` and treat the result as indicative. Below ~100 exceedances, fall back to `insurance-evt` directly.

**tau_0 selection.** Run `threshold_stability_plot` and look for the lowest tau_0 where the xi estimates plateau. Do not pick tau_0 by minimising a single metric - the stability diagnostic is the right approach. Typical range: tau_0 ∈ [0.75, 0.90].

**shape_penalty.** The `shape_penalty` parameter adds an L2 penalty on variance of xi(x) across the batch, smoothing the shape surface and reducing overfitting when the shape signal in the data is weak. Try values in [0.01, 0.1] if the calibration plot shows poor coverage at extreme levels.

---

**[insurance-quantile on GitHub](https://github.com/burning-cost/insurance-quantile)** - MIT-licensed, PyPI. PyTorch GPDNet, 119 tests.

---

## See Also

- **[insurance-evt](/2026/03/13/insurance-evt/)** - Marginal GPD/GEV for portfolio-level EVT, with censored MLE for open TPBI claims and ExcessGPD layer pricing
- **[insurance-quantile](/2026/03/07/insurance-quantile/)** - Quantile GBMs for covariate-dependent tail modelling, without EVT extrapolation
- **[insurance-nflow](/2026/03/12/insurance-nflow/)** - Normalizing flows for the full conditional severity distribution

---

## Related articles

- [Extreme Value Theory for UK Motor Large Loss Pricing](/2026/03/13/insurance-evt/)
- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/)
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/)
