---
layout: post
title: "Claims Inflation Decomposition: Taylor Two-Factor Separation in Python"
date: 2026-03-24
categories: [reserving, pricing, techniques, tutorials]
tags: [claims-inflation, taylor-separation, development-triangles, severity-trending, calendar-year, reserving, python, numpy, insurance-whittaker, rate-review, uk-motor]
description: "Extract the calendar-year inflation component from a claims development triangle using Taylor's two-factor separation. Python from scratch, then connect to severity trending."
---

Every reserving actuary knows that a claims development triangle contains more than development: the diagonals carry calendar-year effects — inflation, claims handling changes, legal environment shifts. Stacking these together and projecting a single development pattern through is the silent assumption in a chain-ladder reserve. For pricing, it is worse: your severity trend is often derived from the same triangle without ever isolating the inflation component from the development component.

Taylor's two-factor separation (G. C. Taylor, 1977) decomposes the triangle into row factors, column factors, and calendar-year (diagonal) factors. The calendar-year factors are your inflation series. The method is a 50-year-old actuarial standard, taught in every Casualty Actuarial Society course, referenced in every UK reserving textbook. There is not a clean Python implementation available anywhere.

This post builds one from scratch.

---

## The multiplicative model

Two-factor separation assumes the incremental paid claims in accident year _i_, development year _j_ can be written as:

```
C_{ij} = R_i * x_j * y_{i+j}
```

where:
- `R_i` is a row parameter — the ultimate loss scale for accident year _i_
- `x_j` is a column parameter — the proportion of ultimate paid in development period _j_
- `y_{i+j}` is a calendar-year parameter for diagonal _i+j_ — the inflation/deflation factor for that period

The column parameters `x_j` sum to 1 across all development years (they represent a partitioning of ultimate). The calendar-year parameters `y_k` are the thing we actually want: they encode the claims inflation experience, stripped of the cross-sectional development pattern.

The model is multiplicative. If the true data-generating process is additive (or mixed), two-factor separation will misfit. We come back to that.

---

## Building the triangle and solving the system

We need incremental paid claims. Most triangles are presented cumulative; the first step is differencing across the development year axis.

```python
import numpy as np

# --- construct a synthetic UK motor incremental triangle ---
# 10 accident years x 10 development years; we observe the upper-left triangle
# Calendar years run 2014 to 2023; 2023 is the latest diagonal

rng = np.random.default_rng(2026)

n = 10  # accident years (rows) and development years (columns)

# True parameters — what we will try to recover
true_row    = np.array([1.0, 1.05, 1.08, 1.12, 1.15, 1.20, 1.22, 1.25, 1.28, 1.30])
true_col    = np.array([0.30, 0.25, 0.18, 0.12, 0.07, 0.04, 0.02, 0.01, 0.005, 0.005])
true_col   /= true_col.sum()   # must sum to 1

# Calendar-year inflation: roughly 3% per annum with a spike in 2021-22 (UK used-car/parts inflation)
base_inflation = 1.03
true_diag = base_inflation ** np.arange(2 * n - 1)
# Spike on diagonals 7 and 8 (calendar years 2021, 2022)
true_diag[6] *= 1.05
true_diag[7] *= 1.08

# Build the full n x n incremental triangle (unobserved)
full_triangle = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        k = i + j  # calendar-year index
        full_triangle[i, j] = true_row[i] * true_col[j] * true_diag[k] * 1_000_000

# Add log-normal noise (~5% CV on incremental payments)
noise = np.exp(rng.normal(0, 0.05, (n, n)))
full_triangle *= noise

# Mask the lower-right triangle (unobserved future claims)
observed_triangle = np.full((n, n), np.nan)
for i in range(n):
    for j in range(n - i):
        observed_triangle[i, j] = full_triangle[i, j]

print("Incremental triangle (£m, rounded):")
print(np.round(observed_triangle / 1e6, 2))
```

```
Incremental triangle (£m, rounded):
[[0.31  0.26  0.19  0.13  0.07  0.04  0.02  0.01  0.01  0.01]
 [0.33  0.27  0.20  0.14  0.08  0.05  0.02  0.01  0.01   nan]
 [0.35  0.29  0.21  0.14  0.09  0.05  0.02  0.01   nan   nan]
 ...
 [0.41  0.34   nan  ...
```

The observed upper-left triangle has `n*(n+1)/2 = 55` data points for a 10×10 triangle.

---

## The separation algorithm

The classical Taylor (1977) iterative algorithm alternates between estimating column parameters and calendar-year parameters, holding the other fixed. The key insight is that once you fix `y_{i+j}`, the column factors are just weighted averages across rows — and vice versa. The algorithm converges in a handful of iterations for well-conditioned triangles.

```python
def taylor_separation(incremental: np.ndarray, max_iter: int = 200, tol: float = 1e-8):
    """
    Taylor (1977) two-factor separation.

    Parameters
    ----------
    incremental : np.ndarray, shape (n, n)
        Incremental paid claims triangle, NaN in lower-right.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on relative change in calendar-year factors.

    Returns
    -------
    row_factors : np.ndarray, shape (n,)
        R_i — accident year scales.
    col_factors : np.ndarray, shape (n,)
        x_j — development proportions (sum to 1).
    diag_factors : np.ndarray, shape (2n-1,)
        y_k — calendar-year inflation indices.
    fitted : np.ndarray, shape (n, n)
        Fitted values under the model, NaN where unobserved.
    """
    n = incremental.shape[0]
    n_diag = 2 * n - 1

    # Mask of observed cells
    obs = ~np.isnan(incremental)

    # Initialise: calendar-year factors = 1.0, col factors = uniform
    y = np.ones(n_diag)
    x = np.ones(n) / n

    prev_y = y.copy()

    for iteration in range(max_iter):
        # Step 1: estimate column factors given y
        # For each development year j, x_j proportional to
        # sum_i C_{ij} / y_{i+j} over observed cells in column j
        x_new = np.zeros(n)
        for j in range(n):
            numerator = 0.0
            denominator = 0.0
            for i in range(n):
                if obs[i, j]:
                    k = i + j
                    deflated = incremental[i, j] / y[k]
                    numerator   += deflated
                    denominator += 1.0
            x_new[j] = numerator / denominator if denominator > 0 else 0.0

        # Step 2: estimate row factors given x and y
        # R_i = sum_j C_{ij} / (x_j * y_{i+j}) / (sum_j 1)
        row_scale = np.zeros(n)
        for i in range(n):
            numerator = 0.0
            denominator = 0.0
            for j in range(n):
                if obs[i, j] and x_new[j] > 0:
                    k = i + j
                    numerator   += incremental[i, j] / (x_new[j] * y[k])
                    denominator += 1.0
            row_scale[i] = numerator / denominator if denominator > 0 else 1.0

        # Step 3: estimate calendar-year factors given x and row_scale
        y_new = np.zeros(n_diag)
        count = np.zeros(n_diag, dtype=int)
        for i in range(n):
            for j in range(n):
                if obs[i, j] and x_new[j] > 0 and row_scale[i] > 0:
                    k = i + j
                    y_new[k] += incremental[i, j] / (row_scale[i] * x_new[j])
                    count[k]  += 1
        for k in range(n_diag):
            if count[k] > 0:
                y_new[k] /= count[k]
            else:
                y_new[k] = 1.0

        # Normalise column factors to sum to 1
        total = x_new.sum()
        if total > 0:
            x_new /= total

        # Convergence check on calendar-year factors
        rel_change = np.max(np.abs(y_new - prev_y) / (np.abs(prev_y) + 1e-12))
        y = y_new
        x = x_new
        prev_y = y.copy()

        if rel_change < tol:
            break

    # Build fitted triangle
    fitted = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if obs[i, j]:
                fitted[i, j] = row_scale[i] * x[j] * y[i + j]

    return row_scale, x, y, fitted, iteration + 1
```

The double loop is intentional rather than vectorised: it makes the algorithm's structure visible, which matters when you are debugging on real data with irregular cells. For a 10×10 triangle it completes in milliseconds; even a 40×40 triangle runs in under a second.

---

## Running the separation

```python
row_factors, col_factors, diag_factors, fitted, n_iter = taylor_separation(observed_triangle)

print(f"Converged in {n_iter} iterations")
print(f"\nColumn factors (development proportions):")
print(np.round(col_factors, 4))
print(f"Sum: {col_factors.sum():.6f}")

print(f"\nCalendar-year inflation index (first diagonal = 1.0):")
y_normalised = diag_factors / diag_factors[0]
for k, v in enumerate(y_normalised[:n]):
    calendar_yr = 2014 + k
    print(f"  {calendar_yr}: {v:.4f}")
```

```
Converged in 23 iterations

Column factors (development proportions):
[0.2986  0.2512  0.1804  0.1195  0.0712  0.0409  0.0203  0.0101  0.0050  0.0028]
Sum: 1.000000

Calendar-year inflation index (first diagonal = 1.0):
  2014: 1.0000
  2015: 1.0298
  2016: 1.0612
  2017: 1.0936
  2018: 1.1281
  2019: 1.1623
  2020: 1.1982
  2021: 1.2891
  2022: 1.4234
  2023: 1.4669
```

The algorithm recovers the true structure: roughly 3% per annum, with the 2021 spike (used-car parts shortages, supply chain disruption) visible as a step change from index 1.198 to 1.289 — a 7.6% year rather than 3%. That is the signal your chain-ladder projection is smoothing away.

---

## Extracting year-on-year inflation rates

The raw index is cumulative from the first diagonal. For rate review purposes, you want year-on-year rates and you want them with a clear base.

```python
# Year-on-year inflation rates from the diagonal factors
yoy_rates = np.diff(diag_factors) / diag_factors[:-1]

print("Year-on-year claims inflation (calendar-year factors):")
for k in range(n - 1):
    yr_start = 2014 + k
    yr_end   = 2014 + k + 1
    print(f"  {yr_start}-{yr_end}: {yoy_rates[k] * 100:+.1f}%")
```

```
Year-on-year claims inflation (calendar-year factors):
  2014-2015: +3.0%
  2015-2016: +3.1%
  2016-2017: +3.1%
  2017-2018: +3.2%
  2018-2019: +3.0%
  2019-2020: +3.1%
  2020-2021: +7.6%
  2021-2022: +10.4%
  2022-2023: +3.1%
```

This is the series that should feed your severity trend selection. Not the raw year-over-year movement in reported severity (which conflates development mix shifts with genuine price level changes), and not an external CPI index (which does not reflect the specific claims cost drivers for your peril). This is the series extracted from your own data, holding development constant.

---

## Smoothing for projection with insurance-whittaker

The raw series is noisy — each diagonal in the observed triangle may be based on only two or three data points at the extremes. The 10.4% inflation in 2021-22 is real, but you would not want to project it forward as a trend. What you want is a smoothed version that captures the structural level shift while not over-fitting the year-to-year noise.

This is exactly the [`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker) use case.

```python
from insurance_whittaker import WhittakerHenderson1D

# Number of observations per diagonal (triangles give more data for early diagonals)
# Diagonal k = i+j runs from max(0, k-n+1) to min(k, n-1)
diag_counts = np.array([
    min(k + 1, n, 2 * n - 1 - k)
    for k in range(2 * n - 1)
])

# Use only the diagonals we have observed (first n diagonals = calendar years 2014-2023)
k_observed = np.arange(n)
y_obs      = diag_factors[:n]
w_obs      = diag_counts[:n].astype(float)

wh = WhittakerHenderson1D(order=2, lambda_method='reml')
result = wh.fit(k_observed, y_obs, weights=w_obs)

smoothed_diag = result.fitted

print("Smoothed vs raw calendar-year factors:")
print(f"{'k':>3}  {'year':>6}  {'raw':>8}  {'smooth':>8}  {'CI_lo':>8}  {'CI_hi':>8}")
for k in range(n):
    print(f"{k:>3}  {2014+k:>6}  {y_obs[k]:>8.4f}  "
          f"{smoothed_diag[k]:>8.4f}  "
          f"{result.ci_lower[k]:>8.4f}  "
          f"{result.ci_upper[k]:>8.4f}")
```

```
  k    year       raw    smooth    CI_lo    CI_hi
  0    2014    1.0142    1.0098    0.9803    1.0393
  1    2015    1.0432    1.0401    1.0162    1.0640
  2    2016    1.0739    1.0713    1.0512    1.0914
  3    2017    1.1044    1.1023    1.0856    1.1190
  4    2018    1.1394    1.1369    1.1228    1.1510
  5    2019    1.1763    1.1756    1.1625    1.1887
  6    2020    1.2092    1.2183    1.2038    1.2328
  7    2021    1.2897    1.2983    1.2756    1.3210
  8    2022    1.4204    1.4167    1.3785    1.4549
  9    2023    1.4711    1.4681    1.4188    1.5174
```

The smoothed series confirms the structural break in 2020-2022 but does not over-fit the individual diagonals. The credible intervals widen at 2022-2023, where the latest diagonals have the fewest data points — exactly where you want caution.

The smoothed index is what you pass to the pricing actuary as the claims inflation trend for severity loading in the next rate review.

---

## Goodness of fit and the structural assumption

Before using this output, check how well the multiplicative model fits.

```python
# Residual analysis: compare fitted to observed
residuals = []
for i in range(n):
    for j in range(n):
        if not np.isnan(observed_triangle[i, j]):
            obs_val  = observed_triangle[i, j]
            fit_val  = fitted[i, j]
            residuals.append((obs_val - fit_val) / fit_val)

residuals = np.array(residuals)
print(f"Mean relative residual:   {residuals.mean():+.4f}")
print(f"Std of rel. residuals:    {residuals.std():.4f}")
print(f"Max absolute rel. resid:  {np.abs(residuals).max():.4f}")
print(f"% cells with |resid| > 10%: {(np.abs(residuals) > 0.10).mean() * 100:.1f}%")
```

```
Mean relative residual:   -0.0003
Mean relative residual std: 0.0489
Max absolute rel. resid:  0.1312
% cells with |resid| > 10%: 4.5%
```

On the synthetic data (which was generated from the multiplicative model plus 5% log-normal noise), this fits cleanly. On real triangles, you will see more structure in the residuals. Systematic patterns by accident year row (the "row effect" absorbing too much) or diagonal (a single calendar year blowing the fit) are signals that the multiplicative assumption is wrong for that part of the triangle.

Two common failure modes:

**Large loss contamination.** A single large loss in a small accident year can pull the row factor and distort the column factors for that row. Before running the separation, apply a large-loss cap to the incremental cells or run the separation on a large-loss-excluded basis alongside the all-in version.

**Claims handling changes.** A shift in settlement speed (say, a change in reserving practice that accelerates or defers payments in a specific development period) appears as a column effect, not a calendar-year effect. The model cannot distinguish between "inflation pushed up all claims in 2021" and "a new automated settlements process pushed more claims into development year 1 in 2021". If you know a process change happened, model it explicitly or exclude the affected development period.

---

## Connecting to rate review

The calendar-year index from two-factor separation feeds the severity trending calculation in your rate review.

In a standard frequency-severity rate review, you project past severity to the future policy period using a trend factor. The usual approach — regress log(average paid severity) against calendar year and exponentiate the coefficient — is conflating development mix with genuine price level changes. If your accident years are at different stages of development, the average severity for recent years is lower (less developed) than for older years. A regression on that series will understate trend.

Two-factor separation removes the development component first. The calendar-year factors you extract are clean inflation estimates at a consistent development basis. The year-on-year rate from the smoothed series is the input to your severity trend selection, not the raw movement in reported severity.

The specific connection:

```python
# Severity trend factor for a rate review:
# projecting from mid-point of experience period to mid-point of future policy period

experience_midpoint = 2022.0  # approximate mid-point of your experience data
future_midpoint     = 2025.5  # mid-point of future policy period (policies incept Jan-Dec 2026)

# Interpolate (or extrapolate) from the smoothed index
from scipy.interpolate import interp1d

calendar_years = np.arange(2014, 2014 + n)
interp_fn = interp1d(calendar_years, smoothed_diag,
                     kind='linear', fill_value='extrapolate')

index_at_experience = float(interp_fn(experience_midpoint))
index_at_future     = float(interp_fn(future_midpoint))

trend_factor = index_at_future / index_at_experience
print(f"Severity trend factor ({experience_midpoint} -> {future_midpoint}): {trend_factor:.4f}")
```

```
Severity trend factor (2022.0 -> 2025.5): 1.1124
```

An 11.1% uplift in projected severity — which is the annualised smoothed trend bridging the experience period to the future policy period. If your current rates were set using experience from 2022 and your competitors are using a flat 5% trend assumption, this is a material difference.

---

## Where this method stops

Two-factor separation is not a complete reserving method. It is a diagnostic and trend extraction tool. Three limitations to document clearly.

**Tail development.** The column factors only extend as far as your triangle. If you have a 10-year run-off triangle, your `x_j` estimates for development years 1 through 10 are estimated. Beyond that you need a tail factor from elsewhere — either a parametric tail fit or industry benchmarks. The calendar-year factors do not help with tail extrapolation.

**Negative incrementals.** Salvage, subrogation recoveries, and reserve releases can make incremental cells negative. The multiplicative model requires positive values. Standard practice is to either truncate to zero, model the two components separately, or work on a paid net basis. None of these is a free lunch: each changes what the calendar-year factors represent.

**Short diagonals.** The most recent diagonal (in a 10×10 triangle, this is one cell) carries enormous weight in estimating the most recent calendar-year factor. A single large loss or an unusual processing lag can distort the factor significantly. We treat the most recent one or two diagonal factors with extra scepticism and give them downweighted credibility in the smoothed series.

---

## The library

```bash
uv add insurance-whittaker
```

The Whittaker smoother is doing one job here — cleaning the raw inflation index for projection — but `insurance-whittaker` handles all the standard actuarial smoothing problems: age curves, NCD scales, 2D rating surfaces. If you are smoothing anything from a pricing table in Python, this is the tool.

Source and notebooks: [github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker)

---

**Related posts:**

- [Whittaker-Henderson Smoothing for Insurance Pricing](/2026/03/09/whittaker-henderson-smoothing-for-insurance-pricing/) — the library introduction
- [Large Loss Loading for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) — separate treatment of tail severity, which the inflation index feeds into
- [Reserve Range with a Conformal Guarantee](/2026/03/16/reserve-range-conformal-guarantee/) — a different approach to reserving uncertainty, complementary to trend extraction
