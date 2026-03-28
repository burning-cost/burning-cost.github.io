---
layout: post
title: "Stochastic Reserving in Python: Mack and Bootstrap ODP with chainladder"
date: 2026-03-28
categories: [tutorials]
tags: [reserving, IBNR, stochastic-reserving, bootstrap-ODP, Mack, chainladder, python, Solvency-II, IFRS-17, tutorial]
description: "How to produce a full IBNR distribution in Python using the Mack method and Bootstrap ODP sampling. Covers analytical standard errors, 5,000-simulation bootstrap, percentile tables for Solvency II SCR and IFRS 17 risk adjustment, and a Mack vs Bootstrap comparison on the RAA dataset."
---

[Part 1 of this series](/2026/04/13/chain-ladder-python-reserving-tutorial/) showed how to build and fit a chain ladder model on the RAA triangle using the `chainladder` library, and how to extract the IBNR estimate of 52,135. That figure is a point estimate. It tells you the expected shortfall between current reserves and ultimate losses. It does not tell you how wrong it could be.

This post addresses that gap. We cover two methods for quantifying reserve uncertainty:

1. **Mack** — an analytical approach that derives standard errors directly from the development pattern, without simulation
2. **Bootstrap ODP** — a simulation method that generates a full empirical distribution of possible reserve outcomes

By the end you will have a percentile table covering the 50th through 99.5th percentiles of total IBNR, the numbers you need for a Solvency II SCR calculation or an IFRS 17 risk adjustment.

---

## Why the point estimate is not enough

The chain ladder gives you an *expected* reserve. That is the right number for setting a best estimate liability. But capital, pricing, and accounting all require more.

**Solvency II SCR.** Under Article 101 of the Directive, the Solvency Capital Requirement is calibrated to a 99.5th percentile loss over a one-year horizon. For reserve risk, that means you need to know where the 99.5th percentile of the reserve distribution sits relative to your best estimate. The gap between the two is roughly your reserve SCR before diversification.

**IFRS 17 risk adjustment.** Under IFRS 17, the risk adjustment for non-financial risk represents the compensation an insurer requires for bearing uncertainty in claims cash flows. It must be disclosed with reference to a confidence level, which requires a quantified reserve distribution.

**Internal management.** Capital allocation, re-insurance purchasing, and business planning all depend on understanding how bad a reserve adverse development scenario could be — not just where the central estimate sits.

The Mack method and Bootstrap ODP are the two workhorses for answering these questions at the aggregate triangle level. Both are implemented in `chainladder` and both take about a dozen lines of Python.

---

## Setup and the pandas compatibility fix

`chainladder` 0.9.1 (the current release) has a timestamp precision mismatch with pandas 3.x: internally it stores valuation dates at microsecond precision but pandas 3.0 uses nanoseconds by default, which causes the diagonal selector to return an empty slice and raise `ValueError: Slice returns empty Triangle`. The fix is a one-line patch applied before any triangle fitting.

```python
import chainladder as cl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------------------------------------------------
# Patch: fix microsecond/nanosecond mismatch in chainladder 0.9.1
# with pandas 3.x. Floor valuation timestamps to microsecond precision
# so the diagonal selector comparison works correctly.
# -----------------------------------------------------------------------
_orig_valuation = cl.Triangle.valuation.fget

def _patched_valuation(self):
    return _orig_valuation(self).floor("us")

cl.Triangle.valuation = property(_patched_valuation)
```

Apply this once at the top of your script. Every subsequent `fit()` call works normally.

```python
# Load the RAA triangle — the same dataset used in Part 1
raa = cl.load_sample("raa")
print(raa)
```

```
         12       24       36       48       60       72       84       96       108      120
1981  5012.0   8269.0  10907.0  11805.0  13539.0  16181.0  18009.0  18608.0  18662.0  18834.0
1982   106.0   4285.0   5396.0  10666.0  13782.0  15599.0  15496.0  16169.0  16704.0      NaN
1983  3410.0   8992.0  13873.0  16141.0  18735.0  22214.0  22863.0  23466.0      NaN      NaN
...
1990  2063.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
```

Ten accident years (1981–1990), ten development periods (12–120 months). The 1981 accident year is fully developed. The 1990 accident year has only the 12-month diagonal.

---

## Method 1: Mack chain ladder

The Mack method (Thomas Mack, 1993) is an analytical extension of the chain ladder. It derives standard errors for each origin year's IBNR estimate without making distributional assumptions — treating the triangle as a distribution-free model and computing variance from the weighted residuals around each development factor.

The key decomposition is:

- **Parameter variance** — uncertainty in the development factors themselves, arising because we estimate them from a finite number of years
- **Process variance** — inherent randomness in future claims even if the development factors were known exactly

The sum of these gives the Mack standard error per origin year. Squared and summed across years (with a covariance correction), this gives the total reserve standard error.

```python
mack = cl.MackChainladder().fit(raa)
print(mack.summary_)
```

```
       Latest          IBNR      Ultimate  Mack Std Err
1981  18834.0           NaN  18834.000000           NaN
1982  16704.0    153.953917  16857.953917    142.931716
1983  23466.0    617.370924  24083.370924    592.148304
1984  27067.0   1636.142163  28703.142163    712.853921
1985  26180.0   2746.736343  28926.736343   1452.090330
1986  15852.0   3649.103184  19501.103184   1994.987807
1987  12314.0   5435.302590  17749.302590   2203.838469
1988  13112.0  10907.192510  24019.192510   5354.340512
1989   5395.0  10649.984101  16044.984101   6331.543044
1990   2063.0  16339.442529  18402.442529  24565.775709
```

The IBNR figures match the chain ladder point estimates from Part 1 exactly — Mack gives identical central estimates, just adds standard errors alongside them. Notice that 1990 has the largest standard error (24,566) despite only having the second-largest IBNR (16,339). That is because it has only one diagonal of data; almost all its reserve is uncertain.

```python
total_ibnr = float(np.nansum(mack.ibnr_.values))
total_se   = float(mack.total_mack_std_err_.values.flatten()[0])

print(f"Total IBNR:           {total_ibnr:>10,.0f}")
print(f"Total Mack Std Err:   {total_se:>10,.0f}")
print(f"Coefficient of Var:   {total_se / total_ibnr:>10.1%}")
```

```
Total IBNR:               52,135
Total Mack Std Err:       26,881
Coefficient of Var:         51.6%
```

A CV of 51.6% is high. On a ten-year triangle with thin early diagonals, this is realistic — the 1990 accident year alone has a standard error larger than its IBNR. On a mature, stable portfolio with 15+ accident years, you would typically see CVs of 10–25%.

### Converting Mack standard errors to percentiles

Mack does not specify a distribution. The conventional actuarial practice is to fit a lognormal distribution matched to the mean and standard error. This is a common exam standard (CT4/ST7 syllabus), accepted in PRA submissions, and reasonable for unimodal, right-skewed reserve distributions.

```python
# Lognormal parameterisation: match mean and variance to Mack outputs
cv2   = (total_se / total_ibnr) ** 2
sigma = np.sqrt(np.log(1 + cv2))          # lognormal shape
mu    = np.log(total_ibnr) - 0.5 * sigma**2  # lognormal location

mack_percentiles = {}
for p in [50, 75, 90, 95, 99.5]:
    mack_percentiles[p] = stats.lognorm.ppf(p / 100, s=sigma, scale=np.exp(mu))
    print(f"Mack {p:>5.1f}th: {mack_percentiles[p]:>10,.0f}")
```

```
Mack  50.0th:     46,338
Mack  75.0th:     64,293
Mack  90.0th:     86,332
Mack  95.0th:    102,986
Mack  99.5th:    161,840
```

The lognormal median (46,338) sits below the mean (52,135) — this is always true for a lognormal; the distribution is right-skewed. The 99.5th percentile of 161,840 is three times the central estimate, illustrating the capital requirement implied by reserve uncertainty on a small, immature triangle.

---

## Method 2: Bootstrap ODP

The Over-Dispersed Poisson bootstrap (England and Verrall, 2002) is a simulation method. Rather than deriving standard errors analytically, it:

1. Fits the chain ladder to the original triangle and computes Pearson residuals
2. Samples from those residuals with replacement to construct 5,000 pseudo-triangles
3. Fits the chain ladder to each pseudo-triangle
4. Projects each pseudo-triangle to ultimate

The resulting 5,000 total IBNR values form an empirical distribution. No distributional assumption is required — the shape falls out of the data.

In `chainladder`, `BootstrapODPSample` is a development transformer (it produces simulated triangles), not a reserving method. The workflow is: bootstrap to get simulated triangles, then fit `Chainladder` to those triangles.

```python
# Step 1: generate 5,000 bootstrapped triangles from the RAA data
boot = cl.BootstrapODPSample(n_sims=5000, random_state=42)
boot.fit(raa)
sims = boot.transform(raa)

print(f"Simulated triangles shape: {sims.shape}")
# (5000, 1, 10, 10) — 5,000 simulations, 1 column, 10 origins, 10 dev periods
```

```
Simulated triangles shape: (5000, 1, 10, 10)
```

```python
# Step 2: fit chain ladder to all 5,000 simulated triangles simultaneously
cl_sims = cl.Chainladder().fit(sims)

# Step 3: extract total IBNR per simulation
# ibnr_ has shape (5000, 1, 10, 1) — sum across origin axis to get total per sim
total_ibnr_sims = cl_sims.ibnr_.sum(axis=2).values.flatten()
total_ibnr_sims = total_ibnr_sims[~np.isnan(total_ibnr_sims)]

print(f"Simulations retained: {len(total_ibnr_sims):,}")
print(f"Mean IBNR:   {np.mean(total_ibnr_sims):>10,.0f}")
print(f"Std dev:     {np.std(total_ibnr_sims):>10,.0f}")
```

```
Simulations retained: 5,000
Mean IBNR:     53,826
Std dev:       19,059
```

The mean of 53,826 sits close to the chain ladder point estimate (52,135). The slight upward bias is expected from the bootstrap resampling — the ODP bootstrap consistently overestimates slightly because the pseudo-triangles include process noise that the point estimate does not.

### The percentile table

```python
boot_percentiles = {}
for p in [50, 75, 90, 95, 99.5]:
    boot_percentiles[p] = np.percentile(total_ibnr_sims, p)
    print(f"Bootstrap {p:>5.1f}th: {boot_percentiles[p]:>10,.0f}")
```

```
Bootstrap  50.0th:     51,587
Bootstrap  75.0th:     65,111
Bootstrap  90.0th:     79,636
Bootstrap  95.0th:     87,589
Bootstrap  99.5th:    111,466
```

### Histogram of the distribution

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(total_ibnr_sims, bins=80, color="#3b82f6", alpha=0.7, edgecolor="white",
        linewidth=0.4)

# Mark key percentiles
colours = {50: "#64748b", 75: "#f59e0b", 95: "#ef4444", 99.5: "#7c3aed"}
labels  = {50: "50th", 75: "75th", 95: "95th", 99.5: "99.5th"}
for p, col in colours.items():
    val = boot_percentiles[p]
    ax.axvline(val, color=col, linewidth=1.8, linestyle="--",
               label=f"{labels[p]} pctl: {val:,.0f}")

ax.axvline(total_ibnr, color="black", linewidth=2,
           label=f"CL point estimate: {total_ibnr:,.0f}")

ax.set_xlabel("Total IBNR", fontsize=12)
ax.set_ylabel("Frequency (5,000 simulations)", fontsize=12)
ax.set_title("RAA Triangle — Bootstrap ODP IBNR Distribution", fontsize=13)
ax.legend(fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.tight_layout()
plt.savefig("raa_bootstrap_ibnr.png", dpi=150)
plt.show()
```

The histogram is right-skewed — a thin but heavy right tail extends well past 100,000. Most of the probability mass sits between 30,000 and 80,000 but the 99.5th percentile is more than twice the median.

---

## Mack vs Bootstrap: a comparison

```python
comparison = pd.DataFrame({
    "Percentile": [50, 75, 90, 95, 99.5],
    "Mack (lognormal)": [mack_percentiles[p] for p in [50, 75, 90, 95, 99.5]],
    "Bootstrap ODP":    [boot_percentiles[p]  for p in [50, 75, 90, 95, 99.5]],
})
comparison["Ratio (B/M)"] = (
    comparison["Bootstrap ODP"] / comparison["Mack (lognormal)"]
).map("{:.2f}".format)
comparison["Mack (lognormal)"] = comparison["Mack (lognormal)"].map("{:,.0f}".format)
comparison["Bootstrap ODP"]    = comparison["Bootstrap ODP"].map("{:,.0f}".format)
print(comparison.to_string(index=False))
```

```
 Percentile Mack (lognormal) Bootstrap ODP Ratio (B/M)
         50           46,338        51,587        1.11
         75           64,293        65,111        1.01
         90           86,332        79,636        0.92
         95          102,986        87,589        0.85
         99.5         161,840       111,466        0.69
```

The methods agree reasonably well in the middle of the distribution. The divergence at the tail is the interesting result: Mack (with a lognormal overlay) produces a heavier tail than the bootstrap. At the 99.5th percentile, Mack gives 161,840 versus 111,466 from the bootstrap — a 45% difference.

This is a well-known pattern. The lognormal overlay on Mack is somewhat arbitrary — a log-t or inverse Gaussian can be more appropriate depending on the triangle — and the lognormal tail is inherently heavier than the empirical bootstrap distribution. England and Verrall (2002) found similar divergence: the bootstrap's empirical tail is bounded by the range of observed residuals, while a parametric Mack distribution can extend further.

Which to use?

**Use Mack when** you need a quick, closed-form answer that does not depend on simulation stability. Mack is also the standard in many PRA internal model submissions and has an explicit analytical interpretation. The lognormal overlay is accepted practice at Lloyd's and in most UK reserving opinions.

**Use Bootstrap ODP when** you want an empirical distribution and are comfortable with the interpretation: these are the outcomes that are consistent with your observed residuals, not a parametric extrapolation. It is also better behaved when the triangle has outlier diagonals that distort the lognormal fit, and it naturally extends to including process variance (the `BootstrapODPSample` draws residuals from the ODP model's process variance as well as parameter variance).

**Do not use either** as a substitute for scenario analysis. Both methods assume the future will look like the past. If you have a known emerging liability — a new legal precedent, a change in settlement rates, a line in run-off — the stochastic framework will not capture it. Supplement with deterministic stress scenarios.

---

## A note on the `BootstrapODPSample` warning

When you run the bootstrap on the RAA triangle you will see a `RuntimeWarning: invalid value encountered in sqrt`. This comes from the hat matrix adjustment in the standardised residual calculation. A handful of cells in the RAA triangle produce leverage factors close to 1, which causes the denominator in the adjustment to approach zero. The warning is expected for small triangles and does not affect the simulation results materially. You can suppress it with:

```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="chainladder")
```

Or leave it — it is informative rather than alarming.

---

## What's next

[Part 3](/2026/04/17/bornhuetter-ferguson-python-reserving/) will cover the **Bornhuetter-Ferguson method** — the industry's preferred approach when the chain ladder is unreliable on thin or volatile triangles. BF blends a prior expected loss ratio with the emerging development pattern, borrowing strength from your pricing assumptions to stabilise immature accident years. We will fit it to the RAA triangle, compare it with the chain ladder, and show how to set and test the prior assumption.
