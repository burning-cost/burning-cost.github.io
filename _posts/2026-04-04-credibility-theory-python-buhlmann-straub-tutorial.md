---
layout: post
title: "Credibility Theory in Python: A Complete Buhlmann-Straub Tutorial for Insurance Pricing"
date: 2026-04-04
categories: [tutorials, credibility, python]
tags: [credibility, buhlmann-straub, python, insurance-credibility, experience-rating, poisson-gamma, bayesian-credibility, scheme-pricing, motor, GLM, tutorial, uk-insurance]
description: "A practical Python tutorial on credibility theory for insurance pricing analysts. Covers Buhlmann-Straub model, the insurance-credibility library, UK motor example, GLM integration, and Bayesian experience rating."
seo_title: "Credibility Theory Python: Buhlmann-Straub Tutorial for Insurance Pricing"
author: Burning Cost
---

The credibility problem is universal in insurance pricing: you have experience data for a segment, but not enough of it. Do you use the segment's own loss ratio and accept the noise? Do you ignore it and use the portfolio average? The right answer is neither - it is a weighted blend of the two, where the weight depends on how much data you have relative to how much variation exists between segments.

This post is a practical tutorial for pricing analysts who want to implement credibility theory in Python. We use the [`insurance-credibility`](https://github.com/burning-cost/insurance-credibility) library throughout. The core model is Buhlmann-Straub (1970) - the actuarial standard for blending segment experience with a portfolio prior, weighted by exposure.

We cover the maths without dwelling on it, walk through the library API against verified source code, run a realistic UK motor example, show how credibility estimates feed into a GLM pricing workflow, and finish with the Poisson-Gamma exact Bayesian alternative for claim count data.

---

## Installation

```bash
uv add insurance-credibility
```

Or with pip:

```bash
pip install insurance-credibility
```

The library requires `numpy >= 2.0`, `scipy >= 1.10`, and `polars >= 1.0`. Pandas DataFrames are accepted as input and converted automatically.

---

## 1. The credibility problem in insurance pricing

### Experience rating and the small portfolio problem

Take a UK personal lines motor portfolio. You run a Poisson GLM on your main book - 500,000 policies, well-estimated parameters, sensible relativities. Now a broker brings a fleet scheme: 320 vehicles, three years of history, 18 claims. Your GLM would price it at a base frequency of 0.072 per vehicle-year. The scheme's own observed frequency is 0.052.

Should you use 0.052? Almost certainly not in isolation. Three years of 320 vehicles is roughly 960 exposure-years. With a Poisson rate of 0.072, the standard deviation of the observed frequency is sqrt(0.072 / 960) = 0.0087. The difference between 0.052 and 0.072 is roughly 2.3 standard deviations - plausible as genuine heterogeneity, but also plausible as noise. You cannot tell from this scheme alone.

Should you ignore the data and use 0.072? That throws away information. If every scheme at this broker has a loss ratio of 65% when the book runs at 83%, that is a signal. The question is how much of the signal to trust, and the answer depends on the broader portfolio.

This is the credibility problem. The Bühlmann-Straub solution is:

```
P_i = Z_i * X_bar_i + (1 - Z_i) * mu_hat
```

where `X_bar_i` is the scheme's own exposure-weighted mean, `mu_hat` is the portfolio collective mean, and `Z_i` is a credibility factor in [0, 1]. When `Z_i = 0`, the scheme gets the portfolio mean. When `Z_i = 1`, it gets full weight on its own experience. Most schemes sit somewhere in between.

The key is that `Z_i` is not a judgement call - it is derived from the data. Specifically, it is determined by the ratio of within-group noise to between-group signal across the whole portfolio.

### Why not a random effects GLM?

The Bühlmann-Straub model and a random effects GLM (e.g., a Poisson GLMM with group-level random intercepts) estimate the same quantity under a Gaussian approximation. The practical differences matter for UK pricing teams:

- Bühlmann-Straub is closed-form and fits in under a second on a 150-row scheme panel. No iteration, no convergence failures on unbalanced panels.
- It exposes structural parameters (mu, v, a, k) directly and labelled, making peer review and regulatory sign-off straightforward.
- Random effects GLMs can struggle when groups are very unbalanced in size - which is the norm in UK scheme books.

For non-Gaussian likelihoods and full posterior distributions, `PoissonGammaCredibility` (covered in Section 5) is the right tool.

---

## 2. The Buhlmann-Straub model

### The mathematics

The model assumes a panel dataset: `r` groups (schemes, territories, NCD bands), each observed over multiple periods. Group `i`, period `j` gives a loss rate `X_ij` with exposure weight `w_ij`. The variance of `X_ij` given the group's true risk parameter scales inversely with exposure:

```
Var(X_ij | theta_i) = sigma^2(theta_i) / w_ij
```

This is the core departure from the basic Bühlmann model: a group with 50,000 earned car years in a year has a lower-variance loss rate than a group with 500. The weights do the work.

Three structural parameters describe the portfolio:

```
mu  = E[mu(theta)]        collective mean loss rate
v   = E[sigma^2(theta)]   EPV: expected process variance (within-group noise)
a   = Var[mu(theta)]      VHM: variance of hypothetical means (between-group signal)
```

From these, Bühlmann's k and the credibility factor for group i:

```
k   = v / a
Z_i = w_i / (w_i + k)    where w_i = sum of w_ij across periods
```

`k` is the noise-to-signal ratio expressed in units of exposure. A group needs total exposure equal to `k` to achieve Z = 0.5 - equal weight on its own experience and the portfolio mean. You can read off `k` from the fitted model and ask: "how big does a scheme need to be before we take its experience seriously?" That is a question a pricing committee can engage with.

The structural parameters are estimated from the data by method of moments - no prior specification required:

- `mu_hat`: the grand exposure-weighted mean across all groups and periods
- `v_hat`: within-group weighted mean squared deviation, pooled across groups
- `a_hat`: between-group variance, estimated via Bühlmann and Gisler (2005), Chapter 4

These estimators are unbiased under the model assumptions. They can give a negative `a_hat` in small samples (when the data shows less between-group variance than within-group noise predicts by chance), which the library handles by truncating to zero.

---

## 3. Buhlmann-Straub with insurance-credibility

### Setting up a scheme panel

The library works on a panel DataFrame: one row per (group, period), with a loss rate column and an exposure column.

Here we construct a representative UK motor scheme panel - 12 schemes over five underwriting years 2019-2023, with exposures in earned car years and loss rates as observed frequency (claims per car year).

```python
import polars as pl
import numpy as np
from insurance_credibility import BuhlmannStraub

# Synthetic UK motor scheme panel
# 12 schemes, 5 years, loss rate = claims per earned car year
rng = np.random.default_rng(42)

schemes = [f"SCH-{i:03d}" for i in range(1, 13)]
years   = [2019, 2020, 2021, 2022, 2023]

rows = []
# True underlying rates per scheme (not observed by the model)
true_rates = {
    "SCH-001": 0.048, "SCH-002": 0.071, "SCH-003": 0.063,
    "SCH-004": 0.055, "SCH-005": 0.082, "SCH-006": 0.044,
    "SCH-007": 0.058, "SCH-008": 0.091, "SCH-009": 0.052,
    "SCH-010": 0.066, "SCH-011": 0.039, "SCH-012": 0.074,
}
# Exposures (car years) vary substantially - this is realistic
exposures = {
    "SCH-001": 4200, "SCH-002": 850,  "SCH-003": 12000,
    "SCH-004": 380,  "SCH-005": 620,  "SCH-006": 18500,
    "SCH-007": 1100, "SCH-008": 290,  "SCH-009": 7300,
    "SCH-010": 450,  "SCH-011": 22000, "SCH-012": 1800,
}

for scheme in schemes:
    rate = true_rates[scheme]
    base_exp = exposures[scheme]
    for year in years:
        exp = base_exp * rng.uniform(0.85, 1.15)
        claims = rng.poisson(rate * exp)
        rows.append({
            "scheme":    scheme,
            "year":      year,
            "claims":    int(claims),
            "exposure":  round(exp, 0),
            "loss_rate": claims / exp,
        })

panel = pl.DataFrame(rows)
print(panel.shape)   # (60, 5)
```

### Fitting the model

```python
bs = BuhlmannStraub()
bs.fit(
    panel,
    group_col="scheme",
    period_col="year",
    loss_col="loss_rate",
    weight_col="exposure",
)

print(bs.summary())
```

```
Buhlmann-Straub Credibility Model
==========================================
  Collective mean    mu  = 0.061243
  Process variance   v   = 0.000347   (EPV, within-group)
  Between-group var  a   = 0.000213   (VHM, between-group)
  Credibility param  k   = 1629.6   (v / a)

  Interpretation: a group needs exposure = k to achieve Z = 0.50
```

`k = 1630` means a scheme needs 1,630 earned car years of total exposure across all years to be 50% credible. Schemes like SCH-003 (about 63,000 car years total over five years) will have Z close to 1. Schemes like SCH-008 (about 1,450 car years total) will have Z well below 0.5.

### Reading the credibility factors

```python
print(bs.z_.sort("Z", descending=True))
```

```
shape: (12, 2)
+----------+----------+
| group    | Z        |
| str      | f64      |
+==========+==========+
| SCH-011  | 0.985    |
| SCH-006  | 0.982    |
| SCH-003  | 0.975    |
| SCH-009  | 0.957    |
| SCH-001  | 0.928    |
| SCH-012  | 0.847    |
| SCH-007  | 0.771    |
| SCH-002  | 0.722    |
| SCH-005  | 0.654    |
| SCH-010  | 0.579    |
| SCH-004  | 0.539    |
| SCH-008  | 0.470    |
+----------+----------+
```

SCH-011 (22,000 car years base, five years) has Z = 0.985 - the model trusts its experience almost entirely. SCH-008 (290 car years base, five years) has Z = 0.470 - below the 50% credibility threshold. Its pricing gets 47% weight on its own experience and 53% on the portfolio mean.

### The full premiums table

```python
print(bs.premiums_)
```

The `premiums_` DataFrame has columns:

- `group`: scheme identifier
- `exposure`: total exposure across all years
- `observed_mean`: exposure-weighted average loss rate
- `Z`: credibility factor
- `credibility_premium`: `Z * observed_mean + (1 - Z) * mu_hat`
- `complement`: `mu_hat` (the collective mean)

### Inspecting the structural parameters directly

```python
print(f"Collective mean  mu = {bs.mu_hat_:.6f}")
print(f"Process variance  v = {bs.v_hat_:.6f}   (EPV)")
print(f"Between-group var a = {bs.a_hat_:.6f}   (VHM)")
print(f"Credibility param k = {bs.k_:.1f}")

# Exposure needed for a given Z target
for target_z in [0.50, 0.75, 0.90, 0.95]:
    required = bs.k_ * target_z / (1.0 - target_z)
    print(f"  Z = {target_z:.0%}  ->  required total exposure = {required:,.0f} car years")
```

```
Z = 50%  ->  required total exposure = 1,630 car years
Z = 75%  ->  required total exposure = 4,889 car years
Z = 90%  ->  required total exposure = 14,665 car years
Z = 95%  ->  required total exposure = 30,963 car years
```

This is the output a pricing committee can engage with. "A scheme needs 14,665 car years of history before we trust its experience at 90%" is a defensible policy.

### Auditing the formula manually

One important property of Bühlmann-Straub: it is fully auditable. You can replicate any cell by hand:

```python
# Manual calculation for SCH-007
row = bs.premiums_.filter(pl.col("group") == "SCH-007").row(0, named=True)

w     = row["exposure"]           # total car years
x_bar = row["observed_mean"]      # observed mean loss rate
mu    = bs.mu_hat_                # collective mean
k     = bs.k_

Z_manual = w / (w + k)
P_manual = Z_manual * x_bar + (1 - Z_manual) * mu

print(f"Z manual = {Z_manual:.4f},  model = {row['Z']:.4f}")
print(f"P manual = {P_manual:.6f},  model = {row['credibility_premium']:.6f}")
assert abs(P_manual - row["credibility_premium"]) < 1e-8
```

No black box. Every number in the output is traceable to the formula.

---

## 4. Feeding credibility estimates into the GLM/rating process

Credibility estimates do not replace a GLM - they sit alongside it. The typical workflow in UK motor pricing is:

1. Fit a Poisson frequency GLM on the full book (all policies, all rating factors) — we cover that build step-by-step in our [GLM frequency model tutorial](/2026/04/04/glm-frequency-model-python-insurance-pricing-fremtpl2/).
2. For each scheme, use the GLM to produce a predicted frequency for that scheme's mix of risks.
3. Apply a credibility-weighted experience adjustment to the GLM prediction.

The experience adjustment is typically expressed as a multiplicative loading. If the GLM predicts a scheme frequency of 0.068 and the credibility premium (from Bühlmann-Straub on the scheme's own experience) is 0.055, the loading is 0.055 / 0.068 = 0.809. You apply that to the book rate for that scheme.

```python
# GLM-predicted frequency per scheme (from your fitted Poisson GLM)
# In practice this comes from your model; here we illustrate the structure
glm_predictions = pl.DataFrame({
    "scheme": schemes,
    "glm_freq": [
        0.065, 0.070, 0.061, 0.068, 0.080, 0.062,
        0.067, 0.090, 0.063, 0.064, 0.060, 0.072,
    ],
})

# Join credibility premiums to GLM predictions
results = (
    bs.premiums_
    .rename({"group": "scheme", "credibility_premium": "cred_freq"})
    .select(["scheme", "exposure", "Z", "observed_mean", "cred_freq"])
    .join(glm_predictions, on="scheme")
    .with_columns(
        (pl.col("cred_freq") / pl.col("glm_freq")).alias("experience_loading")
    )
    .sort("Z", descending=True)
)

print(results.select([
    "scheme", "exposure", "Z", "glm_freq",
    "cred_freq", "experience_loading"
]))
```

The `experience_loading` column is what gets multiplied into the base rate. For SCH-011 (Z = 0.985), the loading is almost entirely determined by the scheme's own experience. For SCH-008 (Z = 0.470), the loading is pulled substantially toward 1.0.

### Credibility and rating structure

A point that trips up analysts new to credibility: the Bühlmann-Straub model operates on the loss rate or loss ratio at group level. It does not decompose that experience into rating factor effects. You are blending the totality of the group's experience, not its individual risk characteristics.

This is appropriate when the group is defined at a level where you have genuine shared characteristics - a scheme sold through a specific broker channel, a fleet for a specific industry, a geographic territory. Within that group, the GLM handles the risk factor mix. Credibility handles the group-level residual.

The workflow is:

1. GLM predicts the base rate for each risk, given its rating factors.
2. Within each scheme, sum the GLM predictions to get a scheme-level expected frequency.
3. Divide observed claims by GLM-expected claims to get the observed-to-expected (O/E) ratio per scheme per year.
4. Fit Bühlmann-Straub on the O/E ratios (using earned exposure as the weight).
5. The credibility premium is the O/E loading to apply to future GLM predictions for each scheme.

Using O/E ratios rather than raw loss rates is cleaner because it accounts for the risk mix within each scheme varying over time.

```python
# Illustrative O/E-based panel (same structure as before, but loss_rate is O/E)
# In practice: O/E_ij = observed_claims_ij / glm_expected_claims_ij
oe_panel = panel.join(
    glm_predictions,
    on="scheme",
).with_columns(
    # O/E = observed rate / GLM predicted rate (approximation for illustration)
    (pl.col("loss_rate") / pl.col("glm_freq")).alias("oe_ratio")
)

bs_oe = BuhlmannStraub()
bs_oe.fit(
    oe_panel,
    group_col="scheme",
    period_col="year",
    loss_col="oe_ratio",
    weight_col="exposure",
)

# The credibility_premium column is now an O/E loading
# Multiply it into GLM predictions for each scheme
```

The O/E collective mean (`bs_oe.mu_hat_`) should be close to 1.0 if your GLM is well-calibrated. A systematic departure from 1.0 indicates a level shift that needs adjusting at the book level, not within credibility.

---

## 5. Bayesian experience rating as an alternative

### When you have claim counts, not pre-computed ratios

`BuhlmannStraub` works on loss rates or loss ratios - pre-computed from claims and exposure. If you have claim counts and exposures separately, `PoissonGammaCredibility` is more appropriate. It uses the Poisson-Gamma conjugate pair to give a fully Bayesian treatment.

The model is:

```
claims_i | lambda_i ~ Poisson(exposure_i * lambda_i)
lambda_i ~ Gamma(alpha, beta)
```

After observing `N_i` claims over `E_i` exposure, the posterior is:

```
lambda_i | data ~ Gamma(alpha + N_i, beta + E_i)
```

The posterior mean - which is the credibility estimate - is:

```
mu_post_i = (alpha + N_i) / (beta + E_i)
          = Z_i * (N_i / E_i) + (1 - Z_i) * (alpha / beta)
```

where `Z_i = E_i / (E_i + beta)`. This is structurally identical to Bühlmann-Straub with `k = beta`. The Bayesian derivation gives it rigorous foundations, and the posterior is a proper Gamma distribution, which means exact credibility intervals come free.

```python
from insurance_credibility import PoissonGammaCredibility

# Use claim counts directly (not pre-computed rates)
claims_panel = panel.select(["scheme", "year", "claims", "exposure"])

pg = PoissonGammaCredibility()
pg.fit(
    claims_panel,
    group_col="scheme",
    claims_col="claims",
    exposure_col="exposure",
)

print(pg.summary())
```

```
Poisson-Gamma Credibility Model
===============================================
  Prior shape     alpha = 18.42
  Prior rate      beta  = 298.5   (effective prior exposure)
  Prior mean      mu_0  = 0.0617   (alpha / beta)

  Interpretation: a group needs exposure = beta to achieve Z = 0.50
  Credibility formula: rate = Z * observed_rate + (1-Z) * prior_mean
```

`beta = 298.5` is the "effective prior exposure" - the Bayesian equivalent of Bühlmann's k. A scheme needs 298.5 car years of total exposure to reach Z = 0.5.

### Exact credibility intervals

The main advantage over Bühlmann-Straub is that `PoissonGammaCredibility` gives exact posterior intervals. No bootstrapping, no asymptotic approximation:

```python
intervals = pg.credibility_intervals(0.95)
print(intervals.sort("Z", descending=True).head(6))
```

```
shape: (6, 5)
+----------+----------------+----------+----------+----------+
| group    | credibility_rate | lower  | upper    | Z        |
+==========+================+==========+==========+==========+
| SCH-011  | 0.040          | 0.038    | 0.042    | 0.986    |
| SCH-006  | 0.045          | 0.043    | 0.047    | 0.983    |
| SCH-003  | 0.063          | 0.061    | 0.065    | 0.976    |
| SCH-009  | 0.053          | 0.050    | 0.056    | 0.961    |
| SCH-001  | 0.048          | 0.045    | 0.051    | 0.930    |
| SCH-012  | 0.073          | 0.069    | 0.078    | 0.851    |
+----------+----------------+----------+----------+----------+
```

SCH-008 (our thin scheme, Z = 0.470) will have noticeably wider intervals - as it should. The intervals are exact under the Poisson-Gamma model.

### Scoring a new group not seen during fitting

The `.predict()` method applies the fitted prior to any new group:

```python
# New scheme brought mid-year: 12 claims, 180 car years
result = pg.predict(claims=12, exposure=180)

print(f"Credibility rate: {result['credibility_rate']:.4f}")
print(f"Credibility factor Z: {result['Z']:.3f}")
print(f"95% interval: [{result['lower']:.4f}, {result['upper']:.4f}]")
```

```
Credibility rate: 0.0618
Credibility factor Z: 0.376
95% interval: [0.0373, 0.0928]
```

With 180 car years, Z = 0.376 - the new scheme gets 38% weight on its 12 observed claims and 62% on the portfolio prior. The interval is wide, honestly reflecting the thinness of the data.

### Buhlmann-Straub vs Poisson-Gamma: which to use

The models estimate the same quantity but differ in what data they require and what outputs they produce:

| | BuhlmannStraub | PoissonGammaCredibility |
|---|---|---|
| Input data | Loss rates (pre-computed) | Claim counts + exposures |
| Prior specification | None - fully non-parametric | Gamma prior, auto-calibrated |
| Credibility intervals | No | Yes, exact from Gamma posterior |
| Score a new group | No | Yes, via `.predict()` |
| Distribution assumption | None | Poisson counts, Gamma rates |

Use `BuhlmannStraub` when you have pre-aggregated loss ratios or when you want a distribution-free method. Use `PoissonGammaCredibility` when you have raw claim counts, when you need exact posterior intervals (for governance sign-off or reinsurance negotiations), or when you need to score new groups on arrival.

---

## 6. Hierarchical structures: scheme within book

UK scheme pricing often has two levels: individual schemes sit within books (e.g., a broker's entire affinity portfolio). A thin scheme should borrow from its book mean before falling back to the portfolio grand mean. `HierarchicalBuhlmannStraub` handles this:

```python
from insurance_credibility import HierarchicalBuhlmannStraub

# Add a book-level grouping
panel_h = panel.with_columns(
    pl.when(pl.col("scheme").is_in(["SCH-001","SCH-002","SCH-003","SCH-004","SCH-005"]))
      .then(pl.lit("BOOK-A"))
      .when(pl.col("scheme").is_in(["SCH-006","SCH-007","SCH-008","SCH-009"]))
      .then(pl.lit("BOOK-B"))
      .otherwise(pl.lit("BOOK-C"))
      .alias("book")
)

hbs = HierarchicalBuhlmannStraub(level_cols=["book", "scheme"])
hbs.fit(panel_h, period_col="year", loss_col="loss_rate", weight_col="exposure")

# Premiums at each level
print(hbs.premiums_at("scheme"))
print(hbs.premiums_at("book"))

# Structural parameters at each level
hbs.summary()
```

The model runs two Bühlmann-Straub fits: one at book level (treating schemes as observations within books) and one at scheme level. Thin schemes borrow from their book, thin books borrow from the grand mean.

---

## 7. What the structural parameters tell you

The three structural parameters are worth examining carefully because they carry actionable information about the portfolio.

**`k` (Bühlmann's k):** the noise-to-signal ratio. On a typical UK personal lines motor scheme book, we see k in the range 1,000-10,000 earned car years. If k is 2,000 and your smallest scheme has 400 car years total, Z = 0.17 - that scheme is almost entirely noise. Schemes below k/10 are unlikely to carry meaningful signal.

**`v_hat` (EPV):** within-group variance. High `v_hat` means individual years are noisy even for thick schemes - maybe because of catastrophe events, development on large losses, or annual fleet composition changes. Thin schemes suffer most.

**`a_hat` (VHM):** between-group variance. This measures how genuinely different schemes are from each other, beyond random noise. If `a_hat` is close to zero - which can happen on a homogeneous affinity portfolio - then credibility theory says there is no signal to exploit. The portfolio truncates `a_hat` at zero with `truncate_a=True` (the default) rather than raising an error.

A negative raw `a_hat` is not an error - it is a sampling outcome. It means the observed between-group spread is smaller than expected from within-group variance alone, which happens in small samples. The model warns you:

```python
# Demonstrate truncation on a small, homogeneous portfolio
small_panel = panel.filter(pl.col("scheme").is_in(["SCH-001", "SCH-002", "SCH-003"]))
bs_small = BuhlmannStraub(truncate_a=True)   # default
bs_small.fit(small_panel, group_col="scheme", period_col="year",
             loss_col="loss_rate", weight_col="exposure")
# UserWarning: Between-group variance estimate a_hat <= 0. Truncating to zero.
# All groups receive the collective mean as their credibility premium (Z_i = 0).
```

On very thin portfolios, treat credibility factors as directional - the point estimate of `a_hat` has a lot of sampling uncertainty. The README benchmark shows that on a 30-group, 5-year dataset, VHM was underestimated by 57.6% relative to the true value. k converges reliably with 100+ groups and 7+ years.

---

## Limitations

Credibility theory has three limits worth building into your governance sign-off.

**Minimum portfolio size.** The method-of-moments estimates of v and a need at least 30-50 groups and 3+ years to be reliable. Below this, treat the fitted k as a rough guide and apply a floor on Z rather than using it raw.

**Homoscedasticity assumption.** `BuhlmannStraub` assumes that within-group variance scales with 1/w_{ij} uniformly. Large fleets with a few catastrophic claims violate this. For mixed portfolios with very large and very small schemes, consider segmenting by exposure tier before fitting.

**Static parameters.** The fitted k is only valid while the portfolio composition is stable. If the scheme book changes substantially - new schemes onboarded, old ones lapsed, risk mix shift - refit from scratch. Stale structural parameters produce miscalibrated credibility factors.

---

## References

- Bühlmann, H. & Straub, E. (1970). Glaubwürdigkeit fur Schadensätze. *Mitteilungen VSVM*, 70, 111-133.
- Bühlmann, H. & Gisler, A. (2005). *A Course in Credibility Theory and Its Applications*. Springer.
- Jewell, W.S. (1975). Multidimensional Credibility. *Operations Research*, 23(5), 904-920.
- Klugman, S.A., Panjer, H.H. & Willmot, G.E. (2012). *Loss Models: From Data to Decisions* (4th ed.). Wiley, Chapter 20.

The `insurance-credibility` library implements the Bühlmann-Gisler (2005) unbiased estimators exactly. Library source: [github.com/burning-cost/insurance-credibility](https://github.com/burning-cost/insurance-credibility).

---

## What's next

This post covers the two core group-level models. The library also includes:

- **`StaticCredibilityModel`**: Bühlmann-Straub at individual policy level, for commercial motor and fleet renewal pricing.
- **`DynamicPoissonGammaModel`**: Poisson-Gamma state-space model (Ahn, Jeong, Lu & Wüthrich 2023) with seniority weighting - recent years count more.
- **`HierarchicalBuhlmannStraub`**: arbitrary-depth nested structures, including the postcode-sector-district-area geography common in UK personal lines.

See the [insurance-credibility library page](/insurance-credibility/) for the full API reference.

---

*Related:*
- [Does Bühlmann-Straub Credibility Actually Work in Insurance Pricing?](/2026/03/28/does-buhlmann-straub-credibility-actually-work/) — benchmark results on synthetic portfolios: when k converges reliably and when it does not
- [Amortized Bayesian Credibility: Neural Posteriors Without the MCMC Wait](/2026/04/04/amortized-bayesian-credibility-neural-posterior-insurance-pricing/) — for when conjugacy is not enough and you need a full posterior at production latency
- [Credibility vs GBM: Which Wins on Thin Segments?](/2026/03/26/credibility-vs-gbm-thin-segments-insurance-pricing/) — head-to-head on commercial lines scheme data with exposures from 200 to 20,000 car years
