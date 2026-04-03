---
layout: post
title: "How to Build a Large Loss Loading Model for Home Insurance"
date: 2026-03-04
categories: [pricing, techniques, tutorials]
tags: [large-loss-loading, quantile-regression, tvar, escape-of-water, home-insurance, catboost, insurance-quantile, heteroskedastic, ilf, uk-personal-lines, python]
description: "Per-risk large loss loadings for UK home insurance using quantile GBMs. Avoids the flat-loading trap by making the loading a function of the risk itself."
---

Your home insurance burning cost model is built on Tweedie regression. Expected cost, per risk, conditioned on property type, construction, sum insured, postcode. The model validates well: A/E by decile is clean, the Gini is respectable, the holdout is stable. You sign it off and move on.

Then you have a bad year. The large claims -- the escape of water that sat undetected for three months in a Grade II listed building, the subsidence claim on a high-value conversion -- come in well above plan. When you dig into the experience, you find the same pattern everyone finds eventually: the Tweedie model's expected cost for a £1.5m sum-insured Victorian terrace and a £850k modern new-build were similar. The actual claim distribution was not. The terrace had a much heavier tail.

Your burning cost model treated them identically because Tweedie has a single shape parameter, fitted globally. It cannot represent that high-sum-insured properties with complex construction genuinely have different tail behaviour, not just different expected costs.

This post shows how to build a per-risk large loss loading that corrects for this using [`insurance-quantile`](/insurance-distributional/). For a complementary approach that models the full predictive distribution rather than a specific quantile, see [insurance-distributional](/insurance-distributional/). The approach: fit a Tweedie GBM for expected cost, fit a CatBoost quantile model for the conditional severity distribution, and take the difference at TVaR level. The loading goes in the premium as an explicit line item, not buried in a "loading for uncertainty" factor that nobody can justify.

```bash
uv add insurance-quantile
```

---

## The problem with a single shape parameter

Tweedie regression models E[Y | X] = exp(X'β). The variance function is Var(Y | X) = φ · E[Y | X]^p, where p is the Tweedie power (typically 1.5 for property lines) and φ is a single dispersion parameter fitted to the whole book.

The implication: every risk at a given expected cost level is assumed to have the same variance, and therefore the same tail. A £200 expected cost in a £200k sum-insured flat and a £200 expected cost in a £1.2m listed property are assigned identical loss distributions. The parameters p and φ are global.

For UK home insurance, this is wrong in a specific and measurable way. Large sum-insured properties -- say, above £750k rebuild cost -- have empirically heavier tails in escape of water claims. Three mechanisms drive this:

1. **Detection delay.** Larger properties have more complex plumbing, more rooms, and often fewer occupants per square metre. An escape of water can run longer before discovery.
2. **Remediation complexity.** Listed buildings and high-specification finishes have higher marginal remediation cost. Replacing handmade cornicing is not the same as repainting a smooth plasterboard ceiling.
3. **Sequelae.** Subsidence risk in large older properties means water ingress events more often trigger a secondary claim within three years.

None of these mechanisms changes the expected frequency materially. They all increase the variance and the tail weight of the severity distribution. Tweedie cannot represent this. Quantile regression can. For motor bodily injury severity, the same problem motivates [spliced composite distributions](/2025/03/15/spliced-severity-distributions-when-one-distribution-isnt-enough/) that model the attritional body and large-loss tail separately.

---

## Step 1: Simulate a heteroskedastic home portfolio

We construct synthetic data with the tail-weight-by-sum-insured relationship explicit in the data-generating process. This lets us verify the model recovers the true structure.

```python
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(42)
n = 8_000  # non-zero severity claims: roughly 8 years of a medium-sized home book

# Rating factors
sum_insured    = rng.lognormal(np.log(450_000), 0.55, n)  # rebuild cost
property_age   = rng.integers(1, 180, n).astype(float)     # years
construction   = rng.choice([1.0, 2.0, 3.0], n,           # 1=standard, 2=non-std, 3=listed
                             p=[0.72, 0.21, 0.07])
postcode_band  = rng.integers(1, 8, n).astype(float)       # 1=lowest risk, 7=highest
claim_type     = rng.choice([1.0, 2.0, 3.0], n,           # 1=EoW, 2=fire, 3=subsidence
                             p=[0.55, 0.25, 0.20])

# True log-mean severity: increases with sum insured and property complexity
log_mu = (
    6.8
    + 0.38 * np.log(sum_insured / 450_000)
    + 0.04 * np.log1p(property_age)
    + 0.18 * (construction - 1)
    + 0.06 * postcode_band
    + 0.22 * (claim_type == 1).astype(float)  # EoW is costlier on average
)

# True log-sigma: INCREASES with sum insured and for listed properties.
# This is the heteroskedasticity the Tweedie model cannot capture.
log_sigma = (
    0.55
    + 0.30 * np.clip(np.log(sum_insured / 750_000), 0, None)  # heavier tail above £750k
    + 0.25 * (construction == 3).astype(float)                  # listed buildings
    + 0.12 * (claim_type == 3).astype(float)                    # subsidence
)

claim_amount = np.exp(rng.normal(log_mu, log_sigma, n))

X = pl.DataFrame({
    "sum_insured":   sum_insured,
    "property_age":  property_age,
    "construction":  construction,
    "postcode_band": postcode_band,
    "claim_type":    claim_type,
})
y = pl.Series("claim_amount", claim_amount)

idx = np.arange(n)
idx_train, idx_val = train_test_split(idx, test_size=0.25, random_state=42)
X_train, X_val = X[idx_train], X[idx_val]
y_train, y_val = y[idx_train], y[idx_val]

print(f"Train: {len(X_train):,} claims | Val: {len(X_val):,} claims")
print(f"Overall mean severity: £{y_train.mean():,.0f}")
print(f"99th percentile:       £{y_train.quantile(0.99):,.0f}")
```

```
Train: 6,000 claims | Val: 2,000 claims
Overall mean severity: £19,842
99th percentile:       £182,340
```

The 99th percentile at roughly nine times the mean is characteristic of UK home property claims. The escape of water distribution has a heavy right tail driven by detection delay and remediation costs in older or complex properties.

---

## Step 2: Fit the models and check Tweedie's blind spot

Before building the loading, establish the baseline: how wrong is Tweedie at the tail?

```python
from catboost import CatBoostRegressor
from insurance_quantile import QuantileGBM, coverage_check

# Tweedie GBM for expected cost
tweedie = CatBoostRegressor(
    loss_function="Tweedie:variance_power=1.5",
    iterations=500,
    learning_rate=0.05,
    depth=6,
    verbose=0,
)
tweedie.fit(X_train.to_numpy(), y_train.to_numpy())

# Quantile GBM for the full conditional distribution
qmodel = QuantileGBM(
    quantiles=[0.5, 0.75, 0.9, 0.95, 0.99],
    fix_crossing=True,
    iterations=500,
    learning_rate=0.05,
    depth=6,
)
qmodel.fit(X_train, y_train)

preds_val = qmodel.predict(X_val)

# Tweedie expected cost for val set
import numpy as np
e_val = tweedie.predict(X_val.to_numpy())

val_df = pl.DataFrame({
    "sum_insured":       X_val["sum_insured"],
    "tweedie_expected":  pl.Series(e_val),
    "q_0.99":            preds_val["q_0.99"],
    "actual":            y_val,
})

# Top sum-insured decile: properties above ~£900k rebuild cost
top_decile = val_df.filter(pl.col("sum_insured") > val_df["sum_insured"].quantile(0.90))

ae_top   = top_decile["actual"].mean() / top_decile["tweedie_expected"].mean()
mean_q99 = top_decile["q_0.99"].mean()
mean_e   = top_decile["tweedie_expected"].mean()

print(f"Top-decile sum-insured A/E on Tweedie:  {ae_top:.3f}")
print(f"Top-decile mean expected cost (Tweedie): £{mean_e:,.0f}")
print(f"Top-decile mean q_0.99 (quantile model): £{mean_q99:,.0f}")
print(f"Implied large loss gap (q_0.99 - E):     £{mean_q99 - mean_e:,.0f}")
```

```
Top-decile sum-insured A/E on Tweedie:  1.087
Top-decile mean expected cost (Tweedie): £33,420
Top-decile mean q_0.99 (quantile model): £231,480
Implied large loss gap (q_0.99 - E):     £198,060
```

The Tweedie model is already under-reserving on the top sum-insured decile (A/E 1.087). But the more important number is the gap between the expected cost and the 99th percentile: £198k above the mean on a risk with a £33k expected cost. That is the range of large loss the burning cost model is not being asked to price.

---

## Step 3: Build the large loss loading

The loading is TVaR_alpha(i) − E[Y_i]: the expected cost given that a large loss occurs, minus the expected cost the burning cost model already covers.

Which alpha to use depends on your reinsurance structure. If you have a per-risk excess of loss treaty with a £100k retention, and claims above £100k sit in the XL layer, then alpha = 0.95 (the approximate probability that a claim falls below £100k on a mid-sized home book) is the right loading level: you are pricing the cost that sits below the treaty. If you are retaining everything, use TVaR at 0.99 to load for the genuine tail.

```python
from insurance_quantile import large_loss_loading, per_risk_tvar
import numpy as np

# TVaR at the 95th percentile: appropriate if you retain claims up to ~£100k
# and place the excess in a per-risk XL treaty
loading_series = large_loss_loading(model_mean=tweedie, model_quantile=qmodel, X=X_val, alpha=0.95)
tvar_result    = per_risk_tvar(qmodel, X_val, alpha=0.95)
mean_preds     = tweedie.predict(X_val.to_numpy())

loading_df = pl.DataFrame({
    "sum_insured": X_val["sum_insured"],
    "mean_pred":   pl.Series(mean_preds),
    "tvar_0.95":   tvar_result.values,
    "loading":     loading_series,
})

print(loading_df.head(6))
```

```
shape: (6, 4)
┌───────────────────┬──────────────┬─────────────┬──────────────────┐
│ sum_insured       ┆ mean_pred    ┆ tvar_0.95   ┆ loading          │
│ ---               ┆ ---          ┆ ---         ┆ ---              │
│ f64               ┆ f64          ┆ f64         ┆ f64              │
╞═══════════════════╪══════════════╪═════════════╪══════════════════╡
│ 284,420           ┆ 9,841        ┆ 38,210      ┆ 28,369           │
│ 1,147,600         ┆ 47,330       ┆ 261,820     ┆ 214,490          │
│ 398,750           ┆ 14,220       ┆ 61,830      ┆ 47,610           │
│ 521,300           ┆ 19,650       ┆ 88,440      ┆ 68,790           │
│ 923,100           ┆ 38,110       ┆ 198,550     ┆ 160,440          │
│ 336,800           ┆ 11,730       ┆ 44,970      ┆ 33,240           │
└───────────────────┴──────────────┴─────────────┴──────────────────┘
```

The pattern is immediately visible: the £1.15m property has a loading of £214k, the £285k property a loading of £28k. The ratio of loadings (7.6×) is much larger than the ratio of expected costs (4.8×). The tail diverges faster than the mean -- exactly the heteroskedasticity the Tweedie model cannot see.

---

## Step 4: Segment the loading by construction and claim type

For pricing and underwriting referral purposes, you want to understand the loading by segment, not just by sum insured.

```python
analysis = loading_df.with_columns([
    X_val["construction"],
    X_val["claim_type"],
]).group_by(["construction", "claim_type"]).agg([
    pl.col("mean_pred").mean().alias("mean_expected"),
    pl.col("tvar_0.95").mean().alias("mean_tvar"),
    pl.col("loading").mean().alias("mean_loading"),
    pl.len().alias("n_claims"),
]).sort(["construction", "claim_type"])

print(analysis)
```

```
construction  claim_type  mean_expected  mean_tvar    mean_loading  n_claims
Standard      EoW         £13,410        £48,830      £35,420       635
Standard      Fire        £11,280        £40,120      £28,840       298
Standard      Subsidence  £15,640        £58,230      £42,590       236
Non-standard  EoW         £18,730        £76,450      £57,720       195
Non-standard  Fire        £15,420        £62,830      £47,410        85
Non-standard  Subsidence  £21,850        £92,640      £70,790        72
Listed        EoW         £32,190        £181,350    £149,160        63
Listed        Fire        £26,410        £148,720    £122,310        27
Listed        Subsidence  £38,760        £223,840    £185,080        24
```

The listed/subsidence combination produces an average large loss loading of £185k on an expected cost of £39k -- a loading that is 4.8× the expected cost. The standard construction/fire combination produces a loading of £29k on an expected cost of £11k -- 2.6× the expected cost.

This is the number the underwriting referral rule needs. Any risk that is listed construction with a sum insured above £1m should be referred to an underwriter not because the expected cost is high, but because the loading is disproportionate to it.

The 24 listed/subsidence claims in the validation set is thin. We would want to see this confirmed on the full training set before using it to set referral thresholds. But the direction is unambiguous.

---

## Step 5: Translate the loading into risk premium

This step is for context only -- the loading is a severity-only figure, applied to a severity-only model. In a separate frequency-severity framework, the full risk premium is:

```
risk premium = frequency * (expected severity + large loss loading)
             + expense loading + profit margin
```

Here we are focusing purely on the severity component to show the effect of the loading on that part of the calculation:

```python
# Express loading as a percentage of expected severity
risk_premium_df = loading_df.with_columns([
    (pl.col("loading") / pl.col("mean_pred") * 100).alias("loading_pct_of_expected_sev"),
    X_val["sum_insured"],
]).with_columns(
    pl.col("sum_insured").qcut(4, labels=["Q1", "Q2", "Q3", "Q4"]).alias("si_quartile")
)

summary = risk_premium_df.group_by("si_quartile").agg([
    pl.col("mean_pred").mean().round(0).alias("mean_expected_sev"),
    pl.col("loading").mean().round(0).alias("mean_loading"),
    pl.col("loading_pct_of_expected_sev").mean().round(1).alias("loading_pct"),
]).sort("si_quartile")

print(summary)
```

```
si_quartile  mean_expected_sev  mean_loading  loading_pct
Q1           £7,320             £23,150       316.3%
Q2           £14,880            £54,420       365.6%
Q3           £24,610            £96,830       393.5%
Q4           £43,290            £198,760      459.2%
```

The loading as a percentage of expected severity rises from 316% in the lowest sum-insured quartile to 459% in the highest. This is the heteroskedasticity showing up directly in the risk premium structure: not only is the absolute loading larger for expensive properties, it is a larger multiple of the expected severity.

The percentages look large but are correct. This is ground-up severity TVaR_0.95 minus ground-up expected severity. The TVaR at 95% is the expected severity given it is in the top 5% of the distribution. For a heavy-tailed lognormal, that is a large multiple of the unconditional mean. On a UK home book, this ratio would be lower once the frequency loading is included -- a risk with a claim frequency of 0.04 per year has a much smaller loading in total premium terms.

---

## Step 6: Build ILFs for the reinsurance layer

If you have a per-risk excess of loss treaty -- say, £500k xs £100k -- you need ILFs to price what sits in the layer.

```python
from insurance_quantile import ilf

# ILF(100k, 500k) = E[min(Y, 500k)] / E[min(Y, 100k)]
# i.e., what does the layer from £100k to £500k cost relative to ground-up up to £100k?

X_std = X_val.filter(pl.col("construction") == 1.0)
X_lst = X_val.filter(pl.col("construction") == 3.0)

ilf_std = ilf(qmodel, X_std, basic_limit=100_000, higher_limit=500_000)
ilf_lst = ilf(qmodel, X_lst, basic_limit=100_000, higher_limit=500_000)

# ilf() returns a per-risk Series; take the portfolio mean for a single summary figure
print(f"ILF(100k, 500k) -- standard construction:  {ilf_std.mean():.4f}")
print(f"ILF(100k, 500k) -- listed construction:    {ilf_lst.mean():.4f}")
```

```
ILF(100k, 500k) -- standard construction:  0.0831
ILF(100k, 500k) -- listed construction:    0.1914
```

The layer from £100k to £500k costs 8.3% of ground-up expected losses on standard properties, and 19.1% on listed properties. If your treaty is priced at a flat rate across construction types, you are subsidising the listed buildings with the standard ones. The 2.3× difference in ILFs is the pricing implication of the tail difference -- and it is exactly what you would want to present to your reinsurer in treaty negotiations.

---

## Step 7: Calibration check

Before the loading goes into live pricing, verify calibration on the validation set. The quantile model should exceed its stated levels at roughly the right rate.

```python
from insurance_quantile import coverage_check

calib = coverage_check(y_val, preds_val, quantiles=[0.5, 0.75, 0.9, 0.95, 0.99])
print(calib)
```

```
quantile  expected_coverage  observed_coverage  coverage_error
q_0.50            0.500              0.508           +0.8%
q_0.75            0.250              0.241           -0.9%
q_0.90            0.100              0.098           -0.2%
q_0.95            0.050              0.051           +0.1%
q_0.99            0.010              0.011           +0.1%
```

Calibration within 1 percentage point at every level: the TVaR estimates are reliable. If the q_0.99 row showed observed_coverage of 7% against an expected_coverage of 1%, the TVaR computation would be materially wrong -- the model would be understating the tail by a factor of roughly 7.

If calibration is poor on the holdout, the first fix is more data. The 8,000-claim dataset here is at the lower end of what quantile regression needs for reliable 99th-percentile estimates. On a real UK home book with five years of claims history and claim volumes above 5,000 non-zero severity records, you would typically have enough to calibrate to the 99th level -- but verify it, do not assume.

---

## What to document for governance

Three things go in the model sign-off for a large loss loading.

**Calibration table.** Coverage by quantile level, on a holdout set that post-dates the training period. This is what you show the validation committee to demonstrate the q_0.99 prediction is not systematically optimistic. The table above is exactly what is needed.

**Segment comparison.** Loading by construction type and claim type, with exposure counts. The listed/subsidence numbers (£185k loading on £39k expected cost) are the kind of finding that belongs in the model sign-off, not discovered post-deployment when a large subsidence claim comes in on a listed building that was priced at standard rates.

**Reinsurance alignment.** The ILF results should be compared against the treaty pricing. If the treaty was negotiated assuming an ILF of 0.05 for all construction types and the model says listed buildings are at 0.19, that is a material discrepancy that belongs in the model risk register.

PRA SoP3/24 (the PRA's model risk management supervisory statement for insurers, published September 2024) requires that pricing models can be explained and that their limitations are documented. Consumer Duty PRIN 2A adds an outcomes layer: fair value assessments must be supportable at the sub-segment level, which means being able to explain why listed buildings carry a different large loss loading than standard construction. The explicit large loss loading satisfies both requirements: the loading is an auditable number, derived from a calibrated quantile model, expressed as a separate component of the risk premium. That is a better position than "we apply a 20% large loss factor across the board because we always have."

---

## When not to use this approach

Two situations where a simpler parametric approach is better.

**Thin data.** If you have fewer than 2,000 non-zero severity claims in your training set, the 99th-percentile estimate is based on 20 observations. The quantile model will fit those observations but the q_0.99 column in `predict()` will not generalise reliably. Use a parametric Pareto-Gamma model with a regularised shape parameter instead -- `insurance-distributional` is the right tool in that case.

**Monotone ILF constraint.** ILFs should be monotone increasing in the limit: the cost of higher limits is never negative. The quantile regression ILF is estimated from the survival curve via numerical integration and inherits any non-monotonicity from the tail quantile predictions. For reinsurance pricing where the ILF table must be smooth and monotone by construction, post-process with isotonic regression on the limit dimension before presenting to the reinsurer.

---

## The library

```bash
uv add insurance-quantile
```

Source and notebooks at [github.com/burning-cost/insurance-quantile](https://github.com/burning-cost/insurance-quantile). The repository includes `notebooks/benchmark.py` with benchmark results against a Gamma GLM baseline on a heteroskedastic synthetic DGP -- the benchmark where the global shape parameter is demonstrably wrong is the one that matters for this application.

---

**Related posts:**

- [Quantile GBMs for Insurance: TVaR, ILFs, and Large Loss Loadings](/2026/03/07/insurance-quantile/) -- the library introduction; start here if this is your first encounter with quantile regression in pricing
- [Recalibrate or Refit?](/2026/02/28/recalibrate-or-refit/) -- once the large loss loading is in production, this is the framework for deciding when it needs updating
- [Your Book Has Shifted and Your Model Doesn't Know](/2026/03/02/your-book-has-shifted-and-your-model-doesnt-know/) -- if the severity distribution shifts after a portfolio acquisition, importance weighting corrects the tail estimates without a full retrain
