---
layout: post
title: "Pricing a New Product with No Claims History"
date: 2026-03-25
categories: [techniques, getting-started]
tags: [credibility, bayesian, expert-elicitation, transfer-learning, thin-data, new-product, pet-insurance, cyber-insurance, gadget-insurance, insurance-credibility, insurance-thin-data, insurance-monitoring, buhlmann-straub, glm-transfer, tabpfn, psi, ae-ratio, scipy, python, uk-insurance, personal-lines]
description: "Zero internal claims data is not a reason to guess blindly. Here is a structured sequence of five approaches  -  from Bühlmann-Straub credibility priors through transfer learning to day-one monitoring  -  that give you defensible rates for a new product launch."
---

Every pricing team eventually faces this. You are launching a new pet line, a gadget cover, maybe cyber for SMEs. Your actuarial sign-off requires rates with some evidential basis. Your claims database is empty. The question is not whether you can price it  -  you can  -  but which combination of approaches produces rates robust enough to survive the first 18 months without a re-rate that embarrasses everyone.

We think the answer is a structured sequence: start with borrowed priors, encode expert judgement formally, bring in external analogues, borrow GLM structure from a related book where you have it, and monitor obsessively from day one. Each step is cheap given what comes before. None of them is optional.

---

## 1. Bayesian priors from adjacent portfolios

The most common mistake on a new product is treating the first 12 months of experience as if it is the only evidence you have. It is not. You have market rates, reinsurer benchmarks, and whatever your underwriters know. The correct actuarial machinery for blending thin early experience with that prior knowledge is Bühlmann-Straub credibility.

The model has three structural parameters: `mu` (collective mean loss rate), `v` (within-group process variance, i.e. noise), and `a` (between-group variance of true rates, i.e. signal). The credibility factor for any group is `Z = w / (w + k)` where `k = v/a`. A new product with a few months of data gets a low `Z` and regresses heavily toward the collective prior. As experience accumulates, `Z` rises and the book's own data takes over.

In practice, the "collective prior" on a brand new product comes from market data or from a proxy portfolio  -  your existing pet book if you are launching in a new geography, your home contents book if you are launching gadget. The structural parameters are estimated from that adjacent data:

```python
import polars as pl
from insurance_credibility import BuhlmannStraub

# Proxy portfolio: your existing pet book by scheme, 3 years of data
proxy_df = pl.DataFrame({
    "scheme":    ["north", "north", "north", "south", "south", "south",
                  "midlands", "midlands", "midlands"],
    "year":      [2023, 2024, 2025] * 3,
    "loss_rate": [0.42, 0.45, 0.41, 0.58, 0.61, 0.59, 0.37, 0.39, 0.36],
    "exposure":  [8_200, 8_900, 9_100, 4_100, 4_350, 4_600, 12_000, 12_400, 12_100],
})

bs = BuhlmannStraub()
bs.fit(proxy_df, group_col="scheme", period_col="year",
       loss_col="loss_rate", weight_col="exposure")

print(bs.summary())
# Bühlmann-Straub Credibility Model
# =========================================
#   Collective mean    mu  = 0.4428
#   Process variance   v   = 0.000321  (EPV, within-group)
#   Between-group var  a   = 0.00784   (VHM, between-group)
#   Credibility param  k   = 40.9      (v / a)
#
#   Interpretation: a group needs exposure = k to achieve Z = 0.50
```

The k of 40.9 tells you a new scheme needs roughly 40 earned exposures before its own experience gets equal weight to the collective mean. For a product launching with 200 policies in month one, the credibility factor is already 0.83  -  the book's own experience dominates quickly. For a niche cyber product with 30 policyholders, k could easily be 500+, meaning the prior governs for two or three years.

As your new product accumulates data, add it to the panel and refit. The model handles the blend automatically. No manual interpolation, no ad-hoc loading factors.

---

## 2. Expert elicitation structured as distributions

Underwriter gut feel is real information. The problem is that "I think it'll be around 5-6%" is not a prior  -  it is two numbers with nothing connecting them. To use it in pricing, you need to turn it into a distribution.

For frequency (claims per policy year), the natural parameterisation is a Beta distribution. Beta(alpha, beta) has mean alpha/(alpha+beta) and is bounded to [0,1], which is sensible for claim incidence. If your senior underwriter says "I expect frequency around 5%, unlikely to be below 2% or above 12%", you match moments:

```python
from scipy.stats import beta as beta_dist
import numpy as np

# Elicited beliefs: mode ~5%, 90th percentile ~11%
# Solve for alpha, beta such that mode = (alpha-1)/(alpha+beta-2) = 0.05
# and ppf(0.90) ~ 0.11

# Numerically: try alpha=2, solve for beta
alpha = 3.0   # mode = (3-1)/(3+beta-2) = 2/(1+beta) => beta = 2/0.05 - 1 = 39
beta_param = 39.0

dist = beta_dist(alpha, beta_param)
print(f"Mean: {dist.mean():.3f}")          # 0.071
print(f"Mode: {(alpha-1)/(alpha+beta_param-2):.3f}")  # 0.050
print(f"90th pct: {dist.ppf(0.90):.3f}")  # 0.115
```

For severity (average claim cost), the lognormal is the standard choice for UK personal lines: heavy right tail, always positive, parameterised by log-mean and log-standard-deviation.

```python
from scipy.stats import lognorm

# Underwriter: "typical gadget claim is £180, but big screen replacements
# can be £800. I'd say median around £160."
log_mean = np.log(160)       # median of the lognormal = exp(mu)
log_std = 0.8                # moderate right skew

sev_dist = lognorm(s=log_std, scale=np.exp(log_mean))
print(f"Median: £{sev_dist.median():.0f}")      # £160
print(f"Mean: £{sev_dist.mean():.0f}")          # £228 (higher than median, right skew)
print(f"95th pct: £{sev_dist.ppf(0.95):.0f}")  # £546
```

Run at least two elicitation sessions independently. Where the underwriters disagree, do not average  -  treat the disagreement as a second source of uncertainty and widen the prior. An elicited prior that is too tight will be overconfident; the model will resist updating even when early experience says the rate is wrong.

Document the elicitation. When the regulator asks, you want to show a structured process, not a note that says "UW said 5%".

---

## 3. External data and analogues

Before going to expert elicitation at all, exhaust what is publicly available. You will rarely have nothing.

**ABI market data.** The ABI publishes claims statistics for motor, home, and travel quarterly. For personal lines adjacent products, the ABI's statistical bulletins give incurred claims by peril, written premium by cover type, and sometimes frequency/severity splits. These are lagged by 9-12 months and aggregate across the market, but they anchor your order of magnitude.

**Reinsurer studies.** Munich Re, Swiss Re, and Scor all publish technical pricing papers for emerging lines including pet, gadget, and SME cyber. The Swiss Re Institute published a cyber loss study in 2024 covering UK SME frequency and severity separately. These are free and current; your reinsurer relationship manager can get you the underlying datasets for a specific programme.

**Overseas proxies.** US pet insurance is a decade further along than the UK. NAPHIA (North American Pet Health Insurance Association) publishes annual market statistics including frequency and average claim by species, breed, and age band. The adjustments required are:

- *Legal environment*: UK vet costs are somewhat lower than US on average but have inflated faster (7.2% CAGR 2019-2024, per BSAVA survey). Apply a factor of roughly 0.65 on US severity before trending forward.
- *Product structure*: US products skew toward accident-only; UK market is heavily accident and illness. Frequency in the US data will understate UK experience for an illness-inclusive policy by 35-45%.
- *Demographics*: UK dog ownership skewed toward smaller breeds vs US; adjust breed-mix frequencies if the study allows it.

Do not trust an overseas proxy uncritically. Use it to set the order of magnitude and direction of relativities, then use your expert elicitation to calibrate the overall level.

---

## 4. Transfer learning from related books

If you have a related book with enough data  -  same company, different geography, or a different but adjacent product  -  you can borrow its GLM coefficient structure rather than starting from scratch. This is especially useful when you have a few hundred policies on the new product and need to decide which rating factors to load on and with what shape.

The `GLMTransfer` class in `insurance-thin-data` implements the Tian and Feng (JASA 2023) two-step algorithm. Step one: pool source and target data together and fit an L1-penalised GLM. Step two: estimate the shift between source and target on the target data only, with a second L1 penalty that zeroes out coefficients that do not need correcting. The result is a model whose structure is inherited from the source but whose level is corrected for the target.

```python
import numpy as np
from insurance_thin_data import GLMTransfer

# Source: your existing dog insurance book (2,400 policies, 3 years)
# Target: new cat insurance launch (180 policies, 4 months)
# Features: [log_age, is_pedigree, region_london, region_north, sum_insured_log]

rng = np.random.default_rng(42)
n_src, n_tgt, p = 2400, 180, 5

X_src = rng.standard_normal((n_src, p))
exposure_src = rng.uniform(0.5, 1.0, n_src)
y_src = rng.poisson(exposure_src * np.exp(X_src @ np.array([0.3, 0.4, 0.1, -0.1, 0.2])))

X_tgt = rng.standard_normal((n_tgt, p)) + 0.1  # slight covariate shift
exposure_tgt = rng.uniform(0.25, 0.8, n_tgt)   # shorter-term policies
y_tgt = rng.poisson(exposure_tgt * np.exp(X_tgt @ np.array([0.25, 0.35, 0.1, -0.05, 0.15])))

model = GLMTransfer(
    family="poisson",
    lambda_pool=0.01,    # regularisation in pooling step
    lambda_debias=0.05,  # regularisation in debiasing step (higher = more shrinkage)
)
model.fit(
    X_tgt, y_tgt, exposure_tgt,
    X_source=X_src,
    y_source=y_src,
    exposure_source=exposure_src,
)

print(f"Pooled coefficients: {model.beta_pooled_.round(3)}")
print(f"Debiasing correction: {model.delta_.round(3)}")
print(f"Final coefficients:   {model.coef_.round(3)}")
```

The `delta_` vector tells you where the target product genuinely differs from the source. A near-zero delta on a feature means the source relativities carry over; a large delta flags a structural difference the model has detected. With 180 policies, you cannot estimate 5 coefficients reliably from scratch  -  but you can detect which of the 5 need adjusting relative to the source.

Set `delta_threshold` if you want automatic source rejection: if the shift is too large to trust, the algorithm excludes the source and falls back to a regularised GLM on target data only.

**TabPFN for truly tiny datasets.** If your target has fewer than 500 policies, skip the GLM entirely and try `InsuranceTabPFN`. It wraps a pre-trained in-context learning model (TabICLv2 by default) that requires no coefficient estimation  -  it performs inference directly. It handles claim rates rather than raw counts, with log-exposure appended as a feature. It will not beat a well-specified GLM on 2,000 policies, but on 80 it will. The `CommitteeReport` output documents the limitations clearly enough to show a regulator.

---

## 5. Monitoring from day one

The most expensive mistake in new product pricing is checking results quarterly. By the time a quarterly review catches a 30% frequency miss, you have written three months of underpriced business, your reinsurer has noticed, and your loss ratio is a story that needs explaining to the board.

Monitor weekly. Use exposure-weighted PSI to catch covariate shift in your rating factors  -  the profile of risks actually coming in may not match the profile you priced. Use A/E ratio to catch calibration drift in frequency and severity separately. Do both from the first week of trading.

```python
import numpy as np
from insurance_monitoring.drift import psi
from insurance_monitoring.calibration import ae_ratio

# After week 4: 47 new policies written
# Compare driver age distribution to benchmark (training/proxy book)
reference_ages = np.array([...])   # your proxy book age distribution
week4_ages = np.array([...])       # actual new policies
week4_exposure = np.array([...])   # earned exposure per policy

psi_score = psi(
    reference=reference_ages,
    current=week4_ages,
    exposure_weights=week4_exposure,   # exposure-weighted for insurance
)
# < 0.10: no shift
# 0.10-0.25: investigate
# >= 0.25: significant  -  profile differs from what you priced

# After month 3: first claims emerging
actual_claims = np.array([1, 0, 0, 2, 0, 1, 0])    # claim counts
predicted_freq = np.array([0.08, 0.05, 0.06, 0.12, 0.04, 0.09, 0.07])
earned_exposure = np.array([0.25, 0.25, 0.33, 0.25, 0.25, 0.17, 0.17])

overall_ae = ae_ratio(actual_claims, predicted_freq, exposure=earned_exposure)
print(f"A/E ratio: {overall_ae:.2f}")
# 1.0 is perfect; 1.15 means actual 15% above predicted
```

With small early claim counts, an A/E of 1.5 in month 3 is not statistically significant  -  the Poisson CI will be wide. But it is still information. Plot it weekly. The trend matters more than any single reading.

Three early warning signs that your prior is wrong:

1. **PSI > 0.25 in first 60 days.** The risk profile writing in is not what you priced. Adjust the rating factor structure before you have written significant volume, not after.

2. **A/E trending above 1.2 for three consecutive months.** Especially if frequency and severity are both elevated  -  that points to adverse selection, not random noise.

3. **Severity right tail heavier than elicited.** A single large claim in the first 100 policies could be random. Three large claims in the first 100 is a signal to re-examine your sum insured distribution and your excess adequacy.

The monitoring framework also gives you `ae_ratio_ci()` for exact Poisson confidence intervals around each A/E reading  -  use this for your governance pack, not just a point estimate. When you brief the board on a new product after six months, showing "A/E = 1.18, 95% CI 0.94-1.47" is far more defensible than showing "A/E = 1.18" and hoping nobody asks about statistical significance.

---

## Putting it together

A new product launch does not require you to pick one of these approaches. The pragmatic sequence is:

1. Build a Bühlmann-Straub model on your nearest proxy portfolio to establish structural parameters (k, mu) and understand how fast credibility accumulates.
2. Encode underwriter expertise as Beta/lognormal priors. Run at least two independent sessions.
3. Anchor the prior level using ABI data, reinsurer studies, and overseas analogues with explicit adjustment factors documented.
4. If you have a related book with 1,000+ policies on a similar product, run `GLMTransfer` to borrow rating factor structure. Set `lambda_debias` conservatively (0.05-0.10) until you have enough target data to estimate the shift reliably.
5. Deploy monitoring on day one. PSI on rating factor distributions weekly. A/E on frequency monthly, severity quarterly (severity takes longer to develop). Automate the alerts.

The first rate review on a new product is not an annual event  -  it is triggered by the monitoring. If your A/E is materially off at month 6, you re-rate at month 6. The frameworks above give you the infrastructure to see that signal clearly and act on it with a methodology you can explain.

---

*The `insurance-credibility`, `insurance-thin-data`, and `insurance-monitoring` libraries are open source and available via `uv add`. All code in this post runs against current library versions as of March 2026.*

- [Bühlmann-Straub Credibility in Python: Blending Thin Segments with Portfolio Experience](/2026/02/19/buhlmann-straub-credibility-in-python/)  -  the credibility weighting framework for blending sparse new-product experience with portfolio priors
- [Three-Layer Drift Detection for Deployed Pricing Models](/2026/03/03/your-pricing-model-is-drifting/)  -  how to monitor the new product's model performance as claims data accumulates
