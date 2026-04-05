---
layout: post
title: "Why Your CI Cause-of-Death Models Disagree With Each Other (And How to Fix It)"
date: 2026-04-01
categories: [techniques, pricing]
tags: [mortality, cause-specific-mortality, DMP, Dirichlet-Multinomial-Poisson, Lee-Carter, coherence, CI-pricing, LTC, protection, longevity, NumPyro, HMC, Bayesian, insurance-survival, arXiv-2603-00973, Nigri-Shang-Ungolo, improvement-factors, CMI, IFRS17]
description: "Independent Lee-Carter models per cause-of-death produce forecasts that do not sum to total mortality — a coherence failure that flows directly into CI and LTC reserves. Nigri, Shang and Ungolo (arXiv:2603.00973, March 2026) solve this with a Dirichlet-Multinomial-Poisson hierarchy that enforces sum-to-total by construction. We have implemented it in insurance-survival v0.4.0."
math: true
author: burning-cost
---

Take a typical CI pricing refresh. You have ONS cause-of-death data from 1979 to 2022. You fit a Lee-Carter model to cancer deaths, another to cardiovascular deaths, another to respiratory deaths. You project each forward to 2040. You then sum the projected cause-specific rates to get your implied total mortality improvement.

There is a problem. The sum does not equal what a Lee-Carter model fitted directly to all-cause deaths would produce. It might be close, or it might be off by 15%. The discrepancy is not random error — it is structural. Each model was calibrated independently, so there is nothing forcing the parts to add up to the whole. When you use these rates to discount CI benefits or reserve LTC cashflows under IFRS 17, that incoherence propagates.

This is not an obscure edge case. It is how most UK life and protection actuaries currently do cause-specific mortality analysis, because until now there was no off-the-shelf tool that did it coherently.

Nigri, Shang and Ungolo (arXiv:2603.00973, submitted March 2026) fix this with a three-level probabilistic hierarchy they call the Dirichlet-Multinomial-Poisson (DMP) framework. We have shipped a Python implementation in `insurance-survival` v0.4.0.

---

## The incoherence problem, precisely stated

Let $m_{a,t}$ be the all-cause central death rate for age group $a$ in calendar year $t$, and let $m_{a,t,c}$ be the cause-specific rate for cause $c$. Coherence requires:

$$\sum_c m_{a,t,c} = m_{a,t} \quad \text{for all } a, t$$

When you fit six separate Lee-Carter models — one per ICD cause group — each model has its own latent period trend $\kappa_t^{(c)}$ and its own age loadings $\beta_a^{(c)}$. Nothing links them. The six projected trends can collectively accelerate faster or slower than the all-cause trend. Coherence is violated by construction.

The conventional patch is post-hoc rescaling: compute the incoherent sum, find the ratio to your all-cause projection, and apply that ratio back to each cause. This is actuarially pragmatic but statistically wrong. The rescaled cause-specific rates no longer reflect the uncertainty in the underlying model — their credible intervals are shrunken in a way that is hard to justify. And improvement factors derived from rescaled rates are not comparable across years.

---

## The DMP hierarchy

The DMP model decomposes cause-specific mortality into three layers.

**Layer 1: Total mortality via Lee-Carter.**

$$\log m_{a,t} = \nu + \delta_a + \beta_a \kappa_t$$

$\kappa_t$ follows a random walk with drift. The $\beta_a$ parameters measure how sensitive each age group is to the shared trend. Identifiability constraints: $\sum_t \kappa_t = 0$, $\sum_a \beta_a = 1$. Total deaths observed as $Y_{a,t} \sim \text{Poisson}(E_{a,t} \cdot m_{a,t})$.

**Layer 2: Cause allocation via Dirichlet-Multinomial.**

Given the total death count $Y_{a,t}$, the cause-specific counts $\bar{Y}_{a,t}$ are distributed as:

$$\bar{Y}_{a,t} \sim \text{DirichletMultinomial}(\phi \cdot \gamma_{a,t}, \; Y_{a,t})$$

where $\gamma_{a,t} \in \Delta^{C-1}$ is the Dirichlet mean vector (cause probability simplex) and $\phi > 0$ is the overdispersion parameter. The DM distribution generalises the Multinomial by allowing extra-multinomial variation across years — useful because the cause mix in real data is more variable than a Multinomial with fixed $\gamma$ would imply.

**Layer 3: Cause probabilities via softmax.**

$$\gamma_{a,t,c} = \frac{\exp(\eta_{a,t,c})}{\sum_{c'} \exp(\eta_{a,t,c'})}$$

with the linear predictor:

$$\eta_{a,t,c} = \phi_{0,c} + \zeta_{a,c} + \theta_c \kappa_t$$

The key term is $\theta_c \kappa_t$: the *same* latent trend $\kappa_t$ that drives total mortality also shifts the cause composition. A decade with rapidly falling cardiovascular mortality will show up in $\kappa_t$; the $\theta_c$ loadings determine how much each cause shifts in response.

**Why coherence holds.**

The cause-specific rate is $m_{a,t,c} = m_{a,t} \cdot \gamma_{a,t,c}$. Because $\sum_c \gamma_{a,t,c} = 1$ by the softmax, we have $\sum_c m_{a,t,c} = m_{a,t}$ at every posterior draw. Not on average, not approximately — identically, by construction.

---

## What this means for CI pricing

A typical UK CI product prices cancer, heart attack, stroke, and other conditions separately. The pricing actuary needs cause-specific mortality improvement factors that are coherent with the CMI mortality improvement assumptions embedded in their valuation basis. Under IFRS 17, the valuation basis and pricing basis interact through the CSM, so an incoherent set of cause-specific improvement factors can cause the pricing profit signature to diverge from the valuation emergence. This is not a theoretical concern — it is an IFRS 17 implementation risk.

The DMP model produces improvement factors in exactly the format CMI use. The `MortalityForecast.improvement_factors()` method returns posterior mean improvement by age, year, and cause:

$$F_{a,t,c} = 1 - \frac{m_{a,t,c}}{m_{a,t-1,c}}$$

These are coherent with the total-mortality improvement by construction. You can directly compare them against CMI 2023 model outputs by age band.

---

## The implementation in insurance-survival

Install with the optional mortality dependencies:

```bash
uv add "insurance-survival[mortality]"
```

This pulls in `numpyro>=0.13` and `jax`. NumPyro was chosen over PyStan because it avoids the C++ toolchain — NUTS sampling runs on pure Python/JAX with no compilation step.

### Loading data

The library ships an `HMDLoader` for England and Wales ONS-format data. You need to download `Deaths_1x1.txt` and `Exposures_1x1.txt` from mortality.org (free registration) and supply cause-specific counts from ONS separately:

```python
from insurance_survival.mortality import HMDLoader, CauseSpecificMortality

# Pair HMD all-cause exposure with ONS cause-specific death counts
# ons_cause_deaths: np.ndarray of shape (n_age, n_year, 6)
deaths, exposure, ages, years = HMDLoader.load(
    data_dir="/data/hmd/GBRTENW",
    sex="female",
    years=(1979, 2022),
    cause_deaths=ons_cause_deaths,
    cause_names=["neoplasms", "cardiovascular", "respiratory",
                 "infectious", "external", "other"],
)
```

For development and testing, synthetic data with realistic UK mortality patterns is built in:

```python
deaths, exposure, ages, years = HMDLoader.load_synthetic(
    n_ages=18, n_years=40, n_causes=6, seed=42
)
```

The synthetic data-generating process uses a Lee-Carter structure with ~1% annual improvement, Gompertz age gradient, and a cause mix weighted toward CVD and neoplasms at older ages — similar enough to GBRTENW female mortality to be a useful development target.

### Fitting the LC-DM model

```python
model = CauseSpecificMortality(
    model_type="LC",   # Lee-Carter variant (recommended over "AP")
    n_chains=3,
    n_samples=1500,    # per chain; 3x1500 = 4500 total draws
    n_warmup=750,
)

model.fit(
    deaths,
    exposure,
    age_labels=ages,
    year_labels=years,
    cause_names=["neoplasms", "cardiovascular", "respiratory",
                 "infectious", "external", "other"],
)
```

Fitting 18 age groups × 40 years × 6 causes on a single GPU takes approximately 12 minutes with the default settings. On CPU, expect 45–90 minutes. For a production annual assumption update cycle this is perfectly acceptable; for exploratory work, reduce `n_samples` to 500.

Check convergence before trusting any outputs:

```python
diag = model.diagnostics()
max_rhat = max(diag["rhat"].values())
print(f"Max R-hat: {max_rhat:.3f}")  # should be < 1.05
print(f"Divergences: {diag['n_divergences']}")  # should be 0 or close to 0
```

The paper reports R-hat < 1.05 for the LC variant on US and French data. The AP variant was numerically unstable for some sex/country combinations — we replicate this finding and recommend LC as the default.

### Forecasting and coherence

```python
forecast = model.forecast(horizon=20)  # projects 2023–2042

# This asserts sum_c cause_rates = total_rates at every draw.
# If it raises, something is wrong with the forecast construction.
assert forecast.coherence_check()

# Credible interval summary for all ages and causes
summary = forecast.summary(quantiles=(0.025, 0.5, 0.975))
print(summary.head(12))
```

The `summary()` output is a long-format DataFrame with columns `age`, `period`, `cause`, `mean`, `sd`, `q0025`, `q0500`, `q0975`. Each row is one (age, period, cause) combination. The `cause="total"` rows give the all-cause rate.

### Improvement factors for CMI comparison

```python
impr = forecast.improvement_factors()

# Filter to working ages, cardiovascular
cvd = impr[
    (impr["cause"] == "cardiovascular") &
    (impr["age"].isin(["45-49", "50-54", "55-59"]))
]

print(cvd[["age", "period", "improvement_factor",
           "improvement_q025", "improvement_q975"]])
```

The credible intervals will be wide — this is honest. A 20-year forecast on a Bayesian RW1 carries substantial uncertainty, and the model does not hide it. The key property is that the weighted sum of cause-specific improvement factors equals the all-cause improvement factor at every posterior draw. Your CI improvement assumptions and your all-cause projection cannot contradict each other.

### Cause fractions for a specific cell

If you only need the posterior mean cause allocation for a particular age/year cell — for example to split an all-cause incidence assumption across ICD chapters:

```python
# Posterior mean cause fractions for age group "55-59", last observed year
fractions = model.cause_fractions(
    age_idx=model.age_labels_.index("55-59"),
    year_idx=model.n_years_ - 1,  # last observed year
)
# fractions: array of shape (6,), sums to 1.0
for cause, frac in zip(model.cause_names_, fractions):
    print(f"{cause:20s}  {frac:.3f}")
```

---

## Limitations you should know before using this

**No UK data in the paper.** Nigri et al. validate on US and France (1979–2023). We have not yet run a formal validation against GBRTENW mortality. UK all-cause improvement trends broadly resemble the US pattern post-2000, but the COVID-19 mortality signature and UK excess deaths in 2022–2023 may require care around end-point sensitivity. Run your own out-of-sample tests on GBRTENW data before production use.

**Population rates, not insured lives.** The model uses HMD/ONS population counts. Insured lives are a selected subgroup — typically healthier, employed, non-smokers. Selection differentials are typically applied as a multiplicative adjustment to population rates using occupational/lifestyle class tables. The DMP model does not handle selection internally; apply your standard select-and-ultimate adjustments after forecasting.

**Six cause groups may be too coarse for full CI pricing.** The paper uses ICD chapter aggregation: neoplasms, CVD, respiratory, infectious, external, other. A UK CI product typically prices 30–40 individual conditions, many within the same ICD chapter (breast cancer vs lung cancer vs colon cancer all sit within neoplasms). Disaggregating to individual ICD codes substantially increases model dimensionality and slows inference. For individual-condition pricing, we suggest using the DMP model at chapter level to generate coherent chapter-level improvement factors, then applying internal claim experience ratios within chapters.

**HMC is slow.** This is not a real-time pricing tool. It is an annual assumption-setting tool. The right workflow is: run the model once per year during your annual assumption update, store the posterior draws, and use the `forecast.summary()` output as a lookup table during pricing. Do not run HMC at quote time.

---

## How it connects to competing risks

`insurance-survival` already has a `FineGrayFitter` in the `competing_risks` subpackage, which models time-to-event with multiple competing causes. The DMP mortality model provides the baseline cause-specific hazard rates that `FineGrayFitter` can be calibrated against. If you are building a multi-state LTC model where transitions to death need cause-specific rates (for example, to stress-test mortality assumptions by clinical scenario), the two modules are designed to work together: use DMP to generate coherent baseline rates, then overlay condition-specific stressed rates using Fine-Gray sub-distribution hazards.

---

## What we think

Independent Lee-Carter per cause is, frankly, a modelling error dressed up as a simplifying assumption. It has persisted because coherent alternatives required either bespoke Stan implementations or access to commercial platforms. The DMP framework from Nigri et al. is elegant, the coherence guarantee is mathematically clean, and the NumPyro implementation in `insurance-survival` makes it accessible without a C++ toolchain.

The paper's out-of-sample results show DMP-LC achieving the lowest RMSE for total mortality forecasting on both US and French data — beating the best available CoDa alternatives. That is a meaningful result. Our implementation passes 37 tests covering the RW2 reconstruction, coherence check, and forecast shape validation. Sampling-based integration tests require a GPU environment; the model structure itself is validated against synthetic data on CPU.

We think this is the right way to model cause-specific mortality for CI and LTC pricing. We would be cautious about using it for individual-condition pricing below chapter level until we have more experience with the UK data, but at chapter level the framework is ready to use.

Code and tests at [github.com/burning-cost/insurance-survival](https://github.com/burning-cost/insurance-survival). Paper: Nigri, Shang and Ungolo, arXiv:2603.00973.
