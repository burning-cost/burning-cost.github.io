---
layout: post
title: "Modelling Social Inflation in UK Motor Severity: A Python Approach"
date: 2026-03-24
categories: [pricing, libraries, tutorials]
tags: [social-inflation, superimposed-inflation, bodily-injury, severity, taylor-separation, whittaker-henderson, insurance-whittaker, motor, uk-pricing, ogden, medco, python, tutorial]
description: "UK motor bodily injury severity has outrun CPI since 2022. This post implements a multiplicative severity separation model and Whittaker-Henderson smoothing in Python to separate economic inflation from superimposed inflation."
excerpt: "Every UK motor pricing team wrestling with BI severity knows the number is going up faster than CPI. The question is: how much of that is real, how much is superimposed, and what trend do you project forward? A multiplicative separation model gives us the decomposition. Python makes it tractable."
published: true
---

Every UK motor pricing team wrestling with bodily injury severity knows the same uncomfortable fact: the number is going up faster than CPI, has been since at least 2022, and the standard approach of "take last three years, fit a line, call it a day" is increasingly indefensible.

The ABI reported motor claims hitting a record £11.7bn in 2024 [ABI, *UK Motor Insurance: Premium, Claims and Profit*, 2023/2024]. Average claim severity rose 13% year-on-year to £4,900, with Q4 2024 reaching £5,300 — an all-time high. BI claim volume is down 41% since the 2021 whiplash reforms, but total BI cost is up 7% over the same period. The frequency trend is doing what we hoped. The severity trend is doing something else entirely.

That "something else" has a name: superimposed inflation, or social inflation in American parlance. It is claims severity growth above general economic inflation. Measuring it correctly — separating the component that tracks CPI from the component that does not — is the single hardest and most commercially important calibration problem in UK motor pricing right now.

This post implements a multiplicative severity separation model in Python, then uses [insurance-whittaker](/insurance-whittaker/) to smooth the resulting development patterns. Note that Taylor's original 1977 formulation in the ASTIN Bulletin is a three-factor model: C[i,j] = R[i] × x[j] × y[i+j], where y[i+j] is a calendar-year diagonal parameter that isolates claims inflation directly. The two-factor model used here — C[i,j] = r[i] × l[j] — omits that diagonal component, making it simpler to implement on a severity triangle but unable to extract a direct inflation index from the diagonals. For the full three-factor Taylor decomposition, see [our companion post on claims inflation decomposition](/2026/03/24/claims-inflation-decomposition-taylor-two-factor-separation-python/). Both methods are older than most pricing actuaries' careers. The gap in UK practice is not the theory; it is production-quality Python tooling that makes them routine.

---

## What is driving UK motor BI severity

To calibrate a trend correctly you need to understand what you are calibrating. UK motor BI severity has four identifiable inflationary components since 2022, each with different forward-looking characteristics.

**Legal cost inflation.** Solicitor hourly rates have risen substantially above CPI. The fixed recoverable costs regime, extended in October 2023 to intermediate track claims up to £100,000, was intended to cap legal cost inflation. The evidence so far is mixed — fixed cost caps have held in low-value claims, but cases migrating above the £100k threshold continue to see uncapped legal costs.

**MedCo and medical report fees.** The Ministry of Justice's July 2023 consultation on fixed-cost medical reports resulted in a 25% increase in MedCo fixed-cost medical report fees from April 2025 and a 15% increase to the whiplash tariff from May 2025. For a pricing team building in a 2025 accident year, these step changes need explicit allowance — they are structural shifts, not part of the secular trend.

**Ogden rate.** The personal injury discount rate (PIDR) moved from -0.25% to +0.5% effective 11 January 2025. The direction was favourable for insurers: Kennedys estimated the change would save the industry around £150m per annum, with lump sum payments for the most severe injuries reducing by up to 25%. This is a one-off negative SI component that needs isolating from the secular positive trend — if you naively smooth through the Ogden change you will understate forward severity.

**Care cost inflation.** For serious BI claims, future care costs dominate the settlement. Total pay growth ran at 10.2% for the year to September 2023, well above CPI. Care costs do not track the general CPI basket; they track healthcare wage inflation, which has structurally exceeded CPI.

None of these is captured in CPI. Deflating your severity triangle by CPI and treating the residual as "real superimposed inflation" is therefore wrong by construction. It is still more informative than not deflating at all, but a team should be explicit about what the deflator represents.

---

## The multiplicative separation model

The idea is straightforward. In a severity triangle, the average cost of claims settled in development period j from accident year i is modelled as:

```
C[i, j] = r[i] * l[j]
```

where `r[i]` is an accident year severity index (the real underlying cost level at the time of the accident) and `l[j]` is a payment delay factor (how settlement mix shifts as claims develop — large claims settle later, lifting average cost in later development periods). The inflation embedded in `r[i]` is what you are measuring. Once you have estimated `r[i]` for each accident year, you fit a trend to it and separate the CPI-tracking component from the superimposed component.

This differs from Taylor's (1977) three-factor model, which adds a calendar-year parameter y[i+j] to the right-hand side. That third factor directly captures the inflation diagonal — it is the right formulation when you want to extract an inflation index from the triangle itself. The two-factor form here is appropriate when you have an external inflation index (CPI or HPTH) you plan to use for deflation, and you want the accident-year factors r[i] to carry the combined inflation signal for subsequent trend fitting.

The fitting is a least-squares problem on the log scale. With a triangle of average severity by accident year and development year:

```python
import numpy as np

from scipy.optimize import minimize

def taylor_separation(avg_cost: np.ndarray, weights: np.ndarray) -> dict:
    """
    Multiplicative two-factor separation of a severity triangle.

    Note: this is not the full Taylor (1977) three-factor model. Taylor's
    original formulation includes a calendar-year diagonal parameter y[i+j]
    that isolates the inflation series directly. This implementation fits only
    accident-year factors r[i] and development-year factors l[j].

    Parameters
    ----------
    avg_cost : ndarray, shape (n_acc, n_dev)
        Average cost per settled claim, NaN where unobserved.
    weights : ndarray, shape (n_acc, n_dev)
        Settled claim counts (or exposure). Used as regression weights.
        Zero where avg_cost is NaN.

    Returns
    -------
    dict with keys:
        'r' : accident year indices, length n_acc (normalised so r[0] = 1.0)
        'l' : development year factors, length n_dev (normalised so l[0] = 1.0)
        'fitted' : fitted triangle, same shape as avg_cost
        'residuals' : log residuals
    """
    n_acc, n_dev = avg_cost.shape
    mask = ~np.isnan(avg_cost) & (weights > 0)

    # Work in log space: log(C[i,j]) = log(r[i]) + log(l[j])
    log_c = np.where(mask, np.log(np.where(mask, avg_cost, 1.0)), np.nan)

    # Parameters: [log_r[0..n_acc-1], log_l[0..n_dev-1]]
    # Identification: fix log_r[0] = 0 and log_l[0] = 0
    n_params = (n_acc - 1) + (n_dev - 1)

    def objective(params):
        log_r = np.concatenate([[0.0], params[:n_acc - 1]])
        log_l = np.concatenate([[0.0], params[n_acc - 1:]])
        fitted_log = log_r[:, None] + log_l[None, :]
        resid = np.where(mask, log_c - fitted_log, 0.0)
        w = np.where(mask, weights, 0.0)
        return np.sum(w * resid ** 2)

    x0 = np.zeros(n_params)
    result = minimize(objective, x0, method="L-BFGS-B")

    log_r = np.concatenate([[0.0], result.x[:n_acc - 1]])
    log_l = np.concatenate([[0.0], result.x[n_acc - 1:]])

    r = np.exp(log_r)
    l = np.exp(log_l)
    fitted = r[:, None] * l[None, :]
    residuals = np.where(mask, log_c - np.log(fitted), np.nan)

    return {"r": r, "l": l, "fitted": fitted, "residuals": residuals}
```

Let us construct a realistic synthetic BI severity triangle to demonstrate. We use ten accident years (2015–2024), seven development periods, and build in a step-change at 2022 to represent the onset of superimposed inflation:

```python
rng = np.random.default_rng(42)

n_acc, n_dev = 10, 7
acc_years = np.arange(2015, 2025)

# True accident year severity index
# 3% pa underlying inflation 2015-2021, then 9% pa from 2022 (SI kicks in)
r_true = np.ones(n_acc)
for i in range(1, n_acc):
    rate = 0.09 if acc_years[i] >= 2022 else 0.03
    r_true[i] = r_true[i - 1] * (1 + rate)

# True development year factor: large claims settle later, lifting later periods
l_true = np.array([1.00, 1.08, 1.14, 1.18, 1.21, 1.23, 1.24])

# Observed average costs — upper-left triangle only (latest diagonal = 2024)
avg_cost = np.full((n_acc, n_dev), np.nan)
weights = np.zeros((n_acc, n_dev))

for i in range(n_acc):
    for j in range(n_dev):
        if i + j < n_acc:  # upper triangle observable
            true_val = r_true[i] * l_true[j]
            # Multiplicative noise, larger at thin cells
            n_claims = max(10, rng.poisson(200 - 20 * j - 5 * i))
            noise = rng.lognormal(0, 0.15 / np.sqrt(n_claims))
            avg_cost[i, j] = true_val * noise * 5_000  # scale to £
            weights[i, j] = n_claims

result = taylor_separation(avg_cost, weights)
r_hat = result["r"]
```

Once you have `r_hat`, normalise it and fit the trend:

```python
from scipy.stats import linregress

# Normalise: r[0] = 1.0 (already normalised by construction above)
# Fit log-linear trend to r_hat
log_r = np.log(r_hat)
slope, intercept, r_value, p_value, se = linregress(np.arange(n_acc), log_r)

annual_inflation = np.exp(slope) - 1
print(f"Fitted annual severity inflation: {annual_inflation:.1%}")
# Expected: ~5-6% blended across the full period
```

---

## Where Whittaker-Henderson comes in

The Taylor method gives you a point estimate for each development year factor. With a seven-period development table that is seven numbers, and fitting a monotone development pattern directly from noisy data is the same smoothing problem as fitting a driver age curve — small cells at the extremes (period 1 is thick; period 7 has one or two diagonal entries) need to be handled without over-fitting the noise.

[`insurance-whittaker`](/insurance-whittaker/) solves this. The development year factors `l[j]` estimated from Taylor separation are smoothed using Whittaker-Henderson with automatic REML lambda selection. The result is a smooth, credible development pattern where the uncertainty widens correctly in sparsely-observed development periods.

```bash
uv add insurance-whittaker
```

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# Development year factors from Taylor separation
l_hat = result["l"]
n_dev = len(l_hat)

# Approximate exposure for each development period
# (sum of claim counts observed in each column)
dev_exposures = np.nansum(weights, axis=0)
dev_exposures = np.maximum(dev_exposures, 1.0)  # avoid zero weights

# Smooth the log development factors
# Order-2 penalty penalises departure from linearity in the log-factor curve
log_l_hat = np.log(l_hat)

wh = WhittakerHenderson1D(order=2)
smooth_result = wh.fit(log_l_hat, weights=dev_exposures)

l_smoothed = np.exp(smooth_result.fitted)
l_ci_lower = np.exp(smooth_result.ci_lower)
l_ci_upper = np.exp(smooth_result.ci_upper)

print("Development year factors (raw vs smoothed):")
for j, (raw, smoothed, lo, hi) in enumerate(
    zip(l_hat, l_smoothed, l_ci_lower, l_ci_upper)
):
    print(f"  Period {j+1}: raw={raw:.3f}  smoothed={smoothed:.3f}  "
          f"90% CI=[{lo:.3f}, {hi:.3f}]")
```

The credible intervals on the later development periods are the commercially important output. Period 6 and 7 in a BI triangle might each have two or three diagonal observations. The raw factors from Taylor separation in those cells have wide uncertainty that is invisible until you quantify it explicitly. A pricing team loading a development-year pattern into a reserving or pricing model without those intervals is making implicit assumptions about the reliability of thin-cell estimates that they would not accept if asked to state them out loud.

---

## Separating economic from superimposed inflation

Once you have the accident year indices `r[i]`, the decomposition into economic and superimposed components is:

```python
# CPI index, 2015 = 1.0 (illustrative — replace with ONS CPIH series)
# ONS series D7BT or CPIH annual averages
cpi = np.array([1.000, 1.009, 1.027, 1.049, 1.065, 1.073, 1.082,
                1.155, 1.237, 1.283])  # 2015-2024 approximate

# Deflate r_hat by CPI to get real severity index
r_real = r_hat / cpi

# Fit trend to real index — this is the superimposed inflation estimate
log_r_real = np.log(r_real)
slope_si, intercept_si, *_ = linregress(np.arange(n_acc), log_r_real)
si_annual = np.exp(slope_si) - 1

# Economic component: fitted from CPI series directly
log_cpi = np.log(cpi)
slope_econ, *_ = linregress(np.arange(n_acc), log_cpi)
econ_annual = np.exp(slope_econ) - 1

print(f"Economic inflation (CPI trend):     {econ_annual:.1%} pa")
print(f"Superimposed inflation (residual):  {si_annual:.1%} pa")
print(f"Total severity inflation:           {(1+econ_annual)*(1+si_annual)-1:.1%} pa")
```

The choice of deflator is not innocent. CPI is the most common choice because it is readily available, but for BI claims it is structurally wrong. The CPI basket does not include solicitor hourly rates as a component; it does not weight towards care worker wages; it does not include MedCo report fees. The ONS SPPI series for legal services is more appropriate for the legal cost component. For care costs, the NHS pay settlement index is closer to the right benchmark.

We are not aware of a team in the UK market that uses component-level deflators consistently. Most use CPI as a practical approximation and acknowledge the limitation in model documentation. That is defensible — the error from using CPI rather than a bespoke deflator is likely smaller than the uncertainty in the trend estimate itself. But it should be a documented assumption, not a default that nobody examined.

---

## The Ogden adjustment

The PIDR change to +0.5% effective January 2025 is a discrete negative superimposed inflation event. A naive trend fit across the 2015–2024 accident years will include the pre-2025 Ogden settlement levels and project them forward. From 2025 accident year onwards, the Ogden basis is materially different.

The right approach is to apply the Ogden adjustment explicitly as a loadings step, then fit your superimposed inflation trend to the Ogden-neutral series. This is equivalent to treating the Ogden change as a known structural break and removing it before trend fitting. The forward BI severity projection then has two components: the Ogden-neutral SI trend plus the step adjustment for the new PIDR basis.

The size of the Ogden effect on average BI severity depends heavily on the mix of claim types in your book. For a standard private car portfolio, the impact is concentrated in the serious injury tail — claims above £500k. If your BI mix is predominantly soft-tissue injuries settling under the MoJ portal with average settlements under £5,000, the Ogden adjustment is immaterial. For a commercial fleet book with genuine catastrophic injury exposure, failing to adjust is a material error.

---

## Limitations

The Taylor separation method assumes multiplicative separability: C[i,j] = r[i] * l[j]. This is a strong assumption. In practice, the development pattern itself shifts over time — litigation timelines changed during COVID, the extension of fixed recoverable costs in 2023 altered settlement behaviour in the intermediate track, and the whiplash reforms changed the mix of claim types reaching each development period. A triangle that mixes pre- and post-reform accident years violates the separability assumption. The residual analysis (`result["residuals"]`) is the diagnostic: large systematic residuals in the post-reform diagonals indicate the model is misspecified.

The method also requires average severity per settled claim, not average severity per reported claim. The distinction matters where claims management strategy changes the rate at which claims are settled across development years. If your claims team accelerated settlements in 2021 to clear a backlog, you will see a spurious dip in the development factor for period 1 in the 2021 accident year. This shows up as a residual, but it is indistinguishable from a genuine change in the severity of early-settling claims without external claims management data.

Neither of these limitations makes the method wrong. They make it a model that needs monitoring, not a calculation that produces a definitive number.

---

The IFoA Claims Inflation Working Party (British Actuarial Journal, 2024, runner-up Brian Hey Prize) confirmed there is no standard Python tooling for this class of problem in the UK market — all work is done in bespoke reserving and pricing models. The separation method and `insurance-whittaker` together give you a reusable, tested, version-controlled foundation. The next time someone asks what the superimposed inflation assumption is and whether it is defensible, you want to be able to show them code and output, not a spreadsheet last edited by someone who left in 2022.

```bash
uv add insurance-whittaker
```

Full documentation and notebooks at [`insurance-whittaker`](/insurance-whittaker/).
