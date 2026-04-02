---
layout: post
title: "Chain Ladder Reserving in Python: A Practical Tutorial with chainladder"
description: "Build loss development triangles, calculate IBNR reserves, and plot development patterns using Python and the chainladder library."
date: 2026-03-28
categories: [reserving, tutorials]
tags: [chainladder, IBNR, loss-triangles, reserving, Python]
---

Reserving and pricing feel like separate disciplines but they feed each other directly. If reserves are inadequate, the loss ratios your pricing team are loading into the rate review are understated. If they are over-adequate, you are chasing phantom profitability. Either way, the pricing actuary who cannot read a reserve analysis is flying partially blind.

This post works through the chain ladder method in Python using the [chainladder](https://github.com/casact/chainladder-python) library (500+ GitHub stars, actively maintained by CAS volunteers). We cover: what a loss development triangle is, how to build and manipulate one, volume-weighted versus simple average development factors, IBNR calculation, and how to visualise development patterns. In part 2 we will add stochastic reserving via the bootstrap ODP method.

---

## What is a loss development triangle?

When a claim is reported, the insurer does not pay it immediately. Losses develop over time: initial reserves get revised upward or downward, partial payments are made, coverage disputes are resolved. At any point in time, your reported losses for a given accident year are less than the ultimate losses you will eventually pay.

A loss development triangle captures this process. Rows are accident years (or underwriting years, or policy years). Columns are development ages, typically in months: 12, 24, 36, and so on. Each cell contains cumulative losses reported as of that age. The right edge of the triangle, the latest diagonal, is where you are right now. Everything to the right of the diagonal is unknown: that is the IBNR (incurred but not reported) you need to estimate.

```
         12      24      36      48      60
1981   5,012   8,269  10,907  11,805  13,539  ...
1982     106   4,285   5,396  10,666  13,782  ...
1983   3,410   8,992  13,873  16,141  18,735  ...
...
1990   2,063     NaN     NaN     NaN     NaN
```

The 1990 row has only one data point: we are at development age 12. The 1981 row is complete at age 120. Chain ladder uses the completed rows to estimate development factors, then projects the incomplete rows to ultimate.

---

## Setup

```bash
uv add chainladder
```

> **Version note:** Tested with `chainladder==0.8.19` and `pandas<3.0`. The `chainladder` 0.9.x series has a known compatibility issue with pandas 3.0+ that causes `Chainladder().fit()` to fail.

We will use the RAA dataset throughout: a classic 10x10 cumulative triangle of general liability losses, originally from the 1989 CAS study on loss development. It is widely used as a textbook example precisely because its development factors are well-behaved, which makes it a good teaching dataset before you get to real-world misbehaviour.

```python
import chainladder as cl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the RAA triangle
triangle = cl.load_sample('raa')
print(triangle)
```

Output:

```
         12       24       36       48       60       72       84       96      108      120
1981  5012.0   8269.0  10907.0  11805.0  13539.0  16181.0  18009.0  18608.0  18662.0  18834.0
1982   106.0   4285.0   5396.0  10666.0  13782.0  15599.0  15496.0  16169.0  16704.0      NaN
1983  3410.0   8992.0  13873.0  16141.0  18735.0  22214.0  22863.0  23466.0      NaN      NaN
1984  5655.0  11555.0  15766.0  21266.0  23425.0  26083.0  27067.0      NaN      NaN      NaN
1985  1092.0   9565.0  15836.0  22169.0  25955.0  26180.0      NaN      NaN      NaN      NaN
1986  1513.0   6445.0  11702.0  12935.0  15852.0      NaN      NaN      NaN      NaN      NaN
1987   557.0   4020.0  10946.0  12314.0      NaN      NaN      NaN      NaN      NaN      NaN
1988  1351.0   6947.0  13112.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
1989  3133.0   5395.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
1990  2063.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
```

The `chainladder` library stores triangles as its own `Triangle` object. It is backed by NumPy arrays but displays as a labelled grid. It handles the NaN-filled lower-right of the triangle automatically.

---

## Development factors: volume-weighted vs simple average

The chain ladder method works by computing age-to-age development factors: the ratio of losses at age N+1 to losses at age N, across all years that have data at both ages. There are two standard approaches.

### Volume-weighted factors

Volume-weighted factors weight each year's ratio by the losses at the earlier age. A year with 20,000 of losses at age 12 carries 10 times the weight of a year with 2,000. This is the default in chainladder and the approach most practitioners use for large, volatile loss triangles where size matters.

```python
dev_vw = cl.Development()  # volume-weighted by default
dev_vw.fit(triangle)

print(dev_vw.ldf_)
```

Output:

```
          12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
(All)  2.999359  1.623523  1.270888  1.171675  1.113385  1.041935  1.033264  1.016936  1.009217
```

The 12-24 factor of 3.00 means that, on volume-weighted average across all ten years, cumulative losses at age 24 are three times losses at age 12. The factors taper toward 1.0 at later ages as development matures.

### Simple average factors

Simple average gives equal weight to each year regardless of size. For homogeneous books where size variation is noise rather than signal, it can be more stable.

```python
dev_simple = cl.Development(average='simple')
dev_simple.fit(triangle)

print(dev_simple.ldf_)
```

Output:

```
          12-24     24-36    36-48     48-60     60-72     72-84     84-96    96-108   108-120
(All)  8.206099  1.695894  1.31451  1.182926  1.126962  1.043328  1.034355  1.017995  1.009217
```

The 12-24 factor jumps to 8.21 under simple average. The 1982 accident year had only 106 of losses at age 12 and 4,285 at age 24, giving a raw link ratio of 40.4. Under volume weighting that outlier is suppressed; under simple averaging it drags the mean up substantially. This is exactly the kind of distortion that makes volume weighting the default for most reserving work.

### CDF to ultimate

The cumulative development factor (CDF) from age N to ultimate is the product of all the individual age-to-age factors from age N onwards. It tells you how much remaining development to expect from where you are now.

```python
print(dev_vw.cdf_)
```

Output:

```
         12-Ult    24-Ult    36-Ult    48-Ult    60-Ult    72-Ult    84-Ult    96-Ult   108-Ult
(All)  8.920234  2.974047  1.831848  1.441392  1.230198  1.104917  1.060448  1.026309  1.009217
```

A 1990 accident year at age 12 needs a factor of 8.92 applied to its current losses to project to ultimate. 1989 at age 24 needs 2.97. By age 96, less than 3% of development remains.

---

## Running the chain ladder projection

With development factors estimated, the chain ladder projection is straightforward: multiply each year's latest diagonal by the CDF for that development age.

The chainladder library follows a scikit-learn-style API. You fit the `Development` estimator first, use `transform` to apply the fitted factors to the triangle, then fit `Chainladder` to produce the ultimate estimates.

```python
import pandas as pd

# Step 1: fit development factors
dev = cl.Development()
dev.fit(triangle)

# Step 2: apply to the triangle
triangle_developed = dev.transform(triangle)

# Step 3: project to ultimate
cl_model = cl.Chainladder()
cl_model.fit(triangle_developed)

# IBNR by accident year
print(cl_model.ibnr_.to_frame())

# Total
print(f"\nTotal IBNR: {float(cl_model.ibnr_.sum()):,.0f}")
```

Output:

```
                    2261
1981-01-01           NaN
1982-01-01    153.953917
1983-01-01    617.370924
1984-01-01  1636.142163
1985-01-01  2746.736343
1986-01-01  3649.103184
1987-01-01  5435.302590
1988-01-01  10907.192510
1989-01-01  10649.984101
1990-01-01  16339.442529

Total IBNR: 52,135
```

The 1981 row shows NaN because it is fully developed: there is no remaining IBNR. The 1990 row, with only one diagonal of data, carries the largest IBNR at 16,339. The numbers here are in thousands of dollars (the RAA dataset is not currency-labelled but is conventionally quoted in thousands).

You can also compare reported losses against projected ultimates in a single summary table:

```python
summary = pd.concat(
    [cl_model.latest_diagonal.to_frame(),
     cl_model.ultimate_.to_frame(),
     cl_model.ibnr_.to_frame()],
    axis=1
)
summary.columns = ['Reported', 'Ultimate', 'IBNR']
summary['% Developed'] = (summary['Reported'] / summary['Ultimate'] * 100).round(1)
print(summary.dropna())
```

Output:

```
            Reported      Ultimate         IBNR  % Developed
1982-01-01   16704.0  16857.953917    153.953917         99.1
1983-01-01   23466.0  24083.370924    617.370924         97.4
1984-01-01   27067.0  28703.142163   1636.142163         94.3
1985-01-01   26180.0  28926.736343   2746.736343         90.5
1986-01-01   15852.0  19501.103184   3649.103184         81.3
1987-01-01   12314.0  17749.302590   5435.302590         69.4
1988-01-01   13112.0  24019.192510  10907.192510         54.6
1989-01-01    5395.0  16044.984101  10649.984101         33.6
1990-01-01    2063.0  18402.442529  16339.442529         11.2
```

The 1982 accident year is 99% developed: the chain ladder adds only 154 of IBNR. The 1990 accident year is 11% developed, meaning 89% of its ultimate losses are still to emerge.

---

## Visualising the development pattern

Seeing the development factors visually makes it much easier to spot anomalies: a factor that sits above or below the trend, a year that behaved differently from the cohort.

```python
import matplotlib.pyplot as plt

# Individual link ratios as a DataFrame
link_ratios = triangle.link_ratio.to_frame()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: scatter of all age-to-age ratios with volume-weighted mean
ax = axes[0]
for year in link_ratios.index:
    ax.plot(link_ratios.columns, link_ratios.loc[year], 'o', alpha=0.5, color='steelblue')

ldf_series = dev_vw.ldf_.to_frame().iloc[0]
ax.plot(ldf_series.index, ldf_series.values, 'r-o', linewidth=2, label='Volume-weighted LDF')
ax.set_title('Age-to-age link ratios: RAA triangle')
ax.set_xlabel('Development period')
ax.set_ylabel('Link ratio')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Right panel: CDF to ultimate
ax2 = axes[1]
cdf_series = dev_vw.cdf_.to_frame().iloc[0]
ax2.bar(range(len(cdf_series)), cdf_series.values, color='steelblue', alpha=0.7)
ax2.set_xticks(range(len(cdf_series)))
ax2.set_xticklabels(cdf_series.index, rotation=45)
ax2.set_title('CDF to ultimate by development age')
ax2.set_xlabel('Development period')
ax2.set_ylabel('CDF to ultimate')

plt.tight_layout()
plt.savefig('raa_development_pattern.png', dpi=150)
plt.show()
```

The left panel shows all individual link ratios as points, with the volume-weighted LDF overlaid in red. For the RAA data, the 12-24 period is visibly noisy: the 1982 accident year outlier (link ratio 40.4) appears as a clear high point. This is exactly the kind of plot you want before accepting your factor selections.

The right panel shows the CDF to ultimate by development age. It illustrates the exposure the older accident years have already worked off versus the residual uncertainty in the newer ones.

---

## Where chain ladder breaks down

Chain ladder is a workhorse, not a silver bullet. It assumes that future development patterns will look like past ones. That assumption fails in predictable ways.

**Thin data.** With fewer than five or six accident years contributing to an age-to-age factor, the volume-weighted average is dominated by one or two observations. A single large claim in the early years of a new line can produce a 12-24 factor that is two or three times higher than reality. In practice, many actuaries cap the number of years used in the factor calculation, or exclude the most recent year when it behaves as a clear outlier.

**Changing mix.** If the mix of business changes materially between cohorts, historic development patterns may not apply to newer ones. A book that added a new class with faster settlement in 2022 will see development patterns shift; chain ladder applied to the combined triangle will blend the old and new patterns in a way that can be meaningless for projecting the newest years.

**Operational time.** Chain ladder works in calendar-year development ages. Some lines develop better on a claims-handled basis (operational time). If claims processing speed has changed, perhaps due to a staffing change or a new TPA, the development triangle in calendar time will look different from prior years even if the underlying liability profile has not changed. The 72-84 column might now represent "claims processed to 80% completion" rather than "12 months of incremental development". You will not see this in the triangle until it is too late.

**Superimposed inflation.** If economic or social inflation is running at a different rate than is embedded in the historical development triangle, your ultimates will be understated. Chain ladder has no mechanism to capture this; you need to either trend the triangle explicitly before applying development factors, or use an alternative method (Bornhuetter-Ferguson with explicit a priori loss ratios, or a generalised linear model for the full triangle).

The bottom line: treat chain ladder estimates as a starting point for analysis, not an endpoint. Run it alongside Bornhuetter-Ferguson. Look at the diagnostic plots. Ask whether anything has changed since your oldest accident years that would invalidate the assumption that history repeats.

---

## Part 2: stochastic reserving

In part 2 we cover stochastic reserving: the bootstrap ODP (over-dispersed Poisson) method, also available in chainladder via `cl.BootstrapODPSample`. Where this post gives you the point estimate (52,135 of IBNR), part 2 gives you the distribution: what 75th-percentile reserves look like, how to get a range of outcomes, and why the reserving risk capital charge at Lloyd's is based on this kind of model.

```python
# Preview: stochastic chain ladder
from chainladder import BootstrapODPSample

sampler = BootstrapODPSample(n_sims=5000, random_state=42)
sampler.fit(triangle)
cl_stochastic = cl.Chainladder().fit(sampler.transform(triangle))

# Distribution of total IBNR across 5000 simulations
ibnr_dist = cl_stochastic.ibnr_.sum('origin')
print(ibnr_dist.quantile(0.75))  # 75th percentile reserve
```

If you are working through a reserve review and want to understand the variability around your point estimates before then, the [chainladder documentation](https://chainladder-python.readthedocs.io/) covers the full range of stochastic methods.

---

The chainladder library handles triangle data correctly, exposes a clean sklearn-style API, and makes it straightforward to run both deterministic and stochastic methods without writing the development factor arithmetic yourself. For pricing actuaries who want to follow a reserve analysis intelligently, or for reserving actuaries moving their workflow into Python, it is the right starting point.

The full working code for this post is in the [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples) repository.
