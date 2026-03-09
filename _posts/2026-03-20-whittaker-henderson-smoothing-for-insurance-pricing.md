---
layout: post
title: "Whittaker-Henderson Smoothing for Insurance Pricing: A Python Library"
date: 2026-03-20
categories: [techniques, libraries, pricing]
tags: [smoothing, whittaker-henderson, graduation, driver-age, ncd, vehicle-group, penalised-least-squares, REML, bayesian, credible-intervals, python, insurance-whittaker]
description: "Every pricing actuary smooths experience rating tables. Most do it with ad hoc moving averages in Excel. insurance-whittaker brings Whittaker-Henderson penalised least squares — with REML lambda selection, Bayesian credible intervals, and 2D cross-table support — to Python."
---

Every rating factor curve needs smoothing before it goes into the rating engine. The raw experience data is noisy. Age 23 has a loss ratio of 140%. Age 24 has 118%. Age 25 has 131%. The underlying risk profile does not look like that: the 24-year-old cohort did not genuinely have a 15-point better loss ratio than the cohorts on either side. You are looking at sampling variation.

The standard responses are: apply a moving average, eyeball it and draw something that looks plausible, or use a parametric curve like a Makeham function fitted to the points. None of these is principled. The moving average has no data-driven bandwidth selection. The freehand adjustment is irreproducible. The parametric curve imposes a functional form that may not match your portfolio. And none of them gives you uncertainty bounds on the smoothed estimate, which means you cannot quantify how much credibility to place in the curve at the extremes where data is thin.

[`insurance-whittaker`](https://github.com/burning-cost/insurance-whittaker) is our solution: Whittaker-Henderson penalised least squares smoothing for insurance pricing. Three classes, 73 tests, pure NumPy/SciPy. The method has been used in actuarial science since the 1920s — E.T. Whittaker proposed it for mortality graduation in 1923. Biessy (2026, ASTIN Bulletin) modernised it with REML lambda selection and Bayesian credible intervals. We have built that into a Python library designed around UK personal lines pricing problems.

```bash
uv add insurance-whittaker
```

---

## The method

Whittaker-Henderson is a penalised least squares problem. Given observed values $y_i$ with weights $w_i$ (typically exposures), find a smooth vector $z$ that minimises:

$$\sum_i w_i (y_i - z_i)^2 + \lambda \sum_i (\Delta^q z_i)^2$$

The first term is the fit term — stay close to the observations. The second is the roughness penalty — penalise the $q$-th differences of the smooth. $\lambda$ controls the trade-off. Large $\lambda$ produces a near-linear smooth; small $\lambda$ interpolates the data.

The order $q$ determines what kind of smoothness is enforced. $q=1$ penalises first differences — the smooth should not change rapidly from one cell to the next. $q=2$ penalises second differences — the smooth should not have rapidly changing slope. For driver age curves, $q=2$ is the natural choice: we expect the loss ratio to follow a broadly smooth arc, not just be locally flat.

This is the same mathematical structure as ridge regression and, with an appropriate prior interpretation, as Bayesian smoothing. The connection to Bayesian inference is what makes the credible interval calculation tractable: the smooth estimate has a known posterior distribution given the roughness prior.

The critical practical question is: what value of $\lambda$? This is where the library earns its keep.

---

## Lambda selection

Setting $\lambda$ by hand is what practitioners do in SAS or Excel. You pick a value, look at the resulting curve, pick another, look again. This is not reproducible and does not account for the actual information content of the data.

`insurance-whittaker` implements four automatic selection methods:

**REML** (restricted maximum likelihood): treats the roughness penalty as a Gaussian prior on the differences and maximises the restricted likelihood. This is the method from Biessy (2026) and is the default. It tends to select smoothness appropriate to the signal-to-noise ratio in the data — thin, volatile data gets smoothed more; dense, stable data is allowed to show structure.

**GCV** (generalised cross-validation): approximately minimises leave-one-out prediction error. A reasonable alternative to REML that does not require the Gaussian prior interpretation.

**AIC/BIC**: information criteria using the effective degrees of freedom of the smoother. BIC penalises complexity more heavily than AIC and will favour smoother curves.

In our experience on UK motor driver age data, REML and GCV typically agree to within a factor of two on $\lambda$ and produce visually indistinguishable curves. The differences emerge at the extremes of the age range, where data is thinnest, and REML's prior interpretation gives it a slight edge in producing actuarially credible estimates.

---

## Smoothing a driver age curve

This is the core use case. We have 50 age bands — ages 17 to 66 — with earned car years and observed loss ratios. The extremes are noisy: 17–20 year olds are a small proportion of most UK books, and the 60+ bands reflect a mix of low-frequency, higher-severity risk that samples badly on smaller portfolios.

```python
import numpy as np
from insurance_whittaker import WhittakerHenderson1D

# Age bands 17-66, exposure-weighted
ages = np.arange(17, 67)
loss_ratios = np.array([...])   # 50 observed loss ratios
exposures = np.array([...])     # 50 exposure weights (earned car years)

wh = WhittakerHenderson1D(order=2, method='reml')
result = wh.fit(loss_ratios, weights=exposures)

print(result.smoothed)          # smoothed loss ratio per age band
print(result.lambda_)           # selected lambda
print(result.effective_df)      # effective degrees of freedom
```

The `effective_df` is important. It tells you how many free parameters the smoother is using. A curve fitted with `effective_df=4.2` is approximately a fourth-order polynomial; a curve with `effective_df=12.5` has more local structure. This is the number to put in your assumptions documentation — "age curve smoothed with Whittaker-Henderson q=2, REML-selected lambda=47.3, effective df=6.1" — because it gives the peer reviewer a meaningful handle on the degree of smoothing applied.

---

## Credible intervals

The posterior distribution of the smooth under the Gaussian roughness prior gives us a credible interval at each point. For pricing purposes this matters most at the extremes of the curve.

```python
ci = result.credible_interval(level=0.95)

print(ci.lower)    # lower bound of 95% credible interval, by age
print(ci.upper)    # upper bound of 95% credible interval, by age
```

At age 17–18, on a book of typical UK size, the 95% credible interval around the smoothed driver age relativities is wide — easily ±20–30 percentage points. This is the correct quantification of the uncertainty, and it should flow into pricing decisions. If a direct-to-consumer quote engine applies a driver age relativity of 1.45 at age 17 with a 95% CI of [1.20, 1.75], the pricing team should be loading for that uncertainty rather than treating 1.45 as a point estimate.

This is the piece that no moving average gives you. Moving averages smooth the point estimate. They do not tell you how smooth it should be or how uncertain the smoothed estimate is. A Whittaker-Henderson fit with REML and credible intervals gives you both.

---

## 2D cross-table smoothing

Many UK motor rating structures include interaction effects: vehicle group by driver age, vehicle group by NCD level, or claim type by vehicle age. These produce cross-tables of relativities, and the cells in the corners of the table — young driver, high vehicle group; older driver, very new vehicle — are almost always sparse.

`WhittakerHenderson2D` applies the same penalised least squares logic along both axes simultaneously, with independent lambda values for each dimension:

```python
from insurance_whittaker import WhittakerHenderson2D

# observed_table: (n_ages, n_veh_groups) array of loss ratios
# weight_table: (n_ages, n_veh_groups) array of exposures

wh2d = WhittakerHenderson2D(order_row=2, order_col=1, method='reml')
result = wh2d.fit(observed_table, weights=weight_table)

print(result.smoothed)          # (n_ages, n_veh_groups) smoothed table
print(result.lambda_row)        # lambda selected for the age dimension
print(result.lambda_col)        # lambda selected for the vehicle group dimension
```

The two lambdas are selected independently. This matters because the signal is typically richer along the age axis (continuous risk gradient, well-understood shape) than along the vehicle group axis (ABI groups are ordered but not uniformly spaced by risk). The smoother learns this from the data and applies more regularisation where the data supports it.

---

## Poisson smoothing for count data

Loss ratios are ratios: they can be smoothed directly. Claim frequencies are counts: a cell with 3 claims from 150 car years has a frequency of 0.020, but the uncertainty around that 0.020 is Poisson, not Gaussian. The WLS approximation embedded in the standard Whittaker-Henderson fit is not quite right.

`WhittakerHendersonPoisson` uses penalised iteratively reweighted least squares (PIRLS) to fit directly to the count data under a Poisson likelihood:

```python
from insurance_whittaker import WhittakerHendersonPoisson

# claims: integer count per age band
# exposures: earned car years per age band

wh_pois = WhittakerHendersonPoisson(order=2, method='reml')
result = wh_pois.fit(claims, exposures=exposures)

print(result.smoothed_rate)     # smoothed claim frequency per age band
print(result.credible_interval(level=0.95))
```

For frequency smoothing, this is the correct model. On a typical UK motor portfolio, the difference between Poisson PIRLS and Gaussian WLS is small in the middle of the age range where claims are plentiful, but meaningful at the extremes: the Poisson model gives wider intervals at age 17–19 where claim counts are low, correctly reflecting the additional uncertainty from small counts rather than just the sampling variation in the loss ratio.

---

## What this replaces

The standard practice on UK pricing teams is one of the following: five-point moving average applied to the raw relativities, cubic spline smoothed by eye, or a parametric Makeham-type curve. All three are done in Excel. None is reproducible in the sense that a different actuary starting from the same data will arrive at the same curve, because the bandwidth selection or the manual adjustment or the parametric form choice involves undocumented judgment.

The case for Whittaker-Henderson over these approaches:

**The data selects the smoothness.** REML-selected $\lambda$ is a function of the signal-to-noise ratio in the data. A book with high exposures and stable experience gets a lower $\lambda$ (less smoothing, more structure preserved). A thin book gets a higher $\lambda$ (more smoothing, more regularisation toward the linear trend). This is correct behaviour. A five-point moving average applies the same bandwidth regardless.

**Credible intervals are computed, not estimated.** The posterior distribution of the smooth gives genuine uncertainty bounds. You can say "we are 95% confident the true age-17 loss ratio is between 135% and 185% of base, after smoothing." No moving average gives you this.

**The output is reproducible.** Given the same data, the same library version, and the same method parameter, the output is identical every time. The assumptions are in the code, not in a modeller's head.

**It is auditable.** A model documentation appendix can record the REML lambda, the effective degrees of freedom, the order of differencing, and the method. The reviewer can run the same code. This satisfies the documentation requirement under SS1/23 in a way that "smoothed by eye using a five-point moving average" does not.

---

## Installation and requirements

```bash
uv add insurance-whittaker
```

Python 3.10+. NumPy and SciPy only — no ML framework dependencies. 73 tests, all passing, on Python 3.10–3.12. MIT licence at [github.com/burning-cost/insurance-whittaker](https://github.com/burning-cost/insurance-whittaker).

The library follows the same conventions as the rest of our pricing stack: `fit()` returns a result object, lambdas are selected automatically by default with manual override available, and the API is consistent across 1D, 2D, and Poisson variants.

---

**Related articles from Burning Cost:**
- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)
- [Spatial Territory Ratemaking with BYM2](/2026/03/09/spatial-territory-ratemaking-bym2/)
- [Trend Selection Is Not Actuarial Judgment: A Python Approach](/2026/03/13/insurance-trend/)
