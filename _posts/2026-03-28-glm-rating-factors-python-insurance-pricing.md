---
layout: post
title: "GLM Rating Factors in Python: A Practical Workflow for Insurance Pricing"
description: "Fit a Tweedie GLM to insurance data in Python, extract multiplicative rating relativities, and visualise factor curves — with working code throughout."
date: 2026-03-28
categories: [pricing, tutorials]
tags: [glm, rating-factors, python, tutorial, pricing, statsmodels, tweedie, relativities]
---

A GLM gives you coefficients. What you actually need is a rating factor table: the multiplicative relativity for each variable level, expressed relative to a base. This post works through the full workflow in Python — fitting a Tweedie GLM with statsmodels, extracting relativities, and producing the visualisations you would put in front of a pricing committee.

We use synthetic data so you can run the code directly. The structure mirrors a typical personal lines frequency model: policy count data, categorical and continuous rating variables, exposure offset.

---

## The data

We will build a small synthetic motor dataset. Five variables: driver age band, area (urban/suburban/rural), NCD (no claims discount) band, vehicle age band, and exposure in years. The response variable is claim count.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
n = 10_000

df = pd.DataFrame({
    "age_band": rng.choice(
        ["17-25", "26-35", "36-50", "51-65", "66+"],
        size=n,
        p=[0.12, 0.22, 0.28, 0.25, 0.13],
    ),
    "area": rng.choice(
        ["Urban", "Suburban", "Rural"],
        size=n,
        p=[0.40, 0.40, 0.20],
    ),
    "ncd_band": rng.choice(
        ["0yr", "1-2yr", "3-4yr", "5yr+"],
        size=n,
        p=[0.15, 0.20, 0.25, 0.40],
    ),
    "veh_age_band": rng.choice(
        ["0-2yr", "3-5yr", "6-9yr", "10yr+"],
        size=n,
        p=[0.15, 0.25, 0.30, 0.30],
    ),
    "exposure": rng.uniform(0.1, 1.0, size=n),
})

# True relativities (multiplicative, base = 36-50, Rural, 5yr+, 3-5yr)
freq_base = 0.06
age_rel    = {"17-25": 2.8, "26-35": 1.6, "36-50": 1.0, "51-65": 0.9, "66+": 1.3}
area_rel   = {"Urban": 1.4, "Suburban": 1.1, "Rural": 1.0}
ncd_rel    = {"0yr": 1.8, "1-2yr": 1.4, "3-4yr": 1.1, "5yr+": 1.0}
veh_rel    = {"0-2yr": 0.9, "3-5yr": 1.0, "6-9yr": 1.1, "10yr+": 1.2}

mu = (
    freq_base
    * df["age_band"].map(age_rel)
    * df["area"].map(area_rel)
    * df["ncd_band"].map(ncd_rel)
    * df["veh_age_band"].map(veh_rel)
    * df["exposure"]
)

df["claims"] = rng.poisson(mu)
```

The true model is multiplicative with known relativities. A correctly specified Poisson GLM with a log link should recover these (up to sampling noise at this dataset size).

---

## Fitting the GLM

We fit a Poisson GLM with a log link, log-exposure as an offset, and treatment coding with explicit base levels.

```python
# Set base levels via pd.Categorical ordering
for col, base in [
    ("age_band",    "36-50"),
    ("area",        "Rural"),
    ("ncd_band",    "5yr+"),
    ("veh_age_band","3-5yr"),
]:
    cats = [base] + [c for c in df[col].unique() if c != base]
    df[col] = pd.Categorical(df[col], categories=cats)

model = smf.glm(
    formula="claims ~ age_band + area + ncd_band + veh_age_band",
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df["exposure"]),
).fit()

print(model.summary())
```

A few things worth noting on the setup:

**Base level selection matters.** The base level is the reference category that gets a relativity of 1.0 in the output. Choose a level that is well-populated — sparse base categories produce wide confidence intervals on everything else. We use `pd.Categorical` to control the ordering rather than relying on alphabetical sorting, which often gives you the wrong base.

**The offset, not a weight.** Exposure goes in as `offset=np.log(df["exposure"])`, not as `freq_weights`. These are not equivalent. An offset shifts the linear predictor without being a parameter. A frequency weight changes how much each observation contributes to the likelihood. For a Poisson model where the natural scale is claim count per unit exposure, the offset is correct.

**Tweedie for pure premium.** If you are fitting directly to pure premium (claim cost / exposure) rather than claim count, switch to:

```python
family=sm.families.Tweedie(
    var_power=1.5,
    link=sm.families.links.Log()
)
```

The Tweedie power parameter `p` can be estimated from the data (roughly: Poisson at p=1, Gamma at p=2, compound Poisson-Gamma in between). For a standard personal lines pure premium model, p ≈ 1.5 is a reasonable starting point; validate it by running the model at p=1.3, 1.5, 1.7 and comparing AIC or out-of-sample deviance.

---

## Extracting relativities

The model returns coefficients on the log scale. Relativities are `exp(coefficient)`.

```python
def extract_relativities(result):
    """
    Extract multiplicative relativities and 95% CIs from a statsmodels GLM result.
    Returns a DataFrame with columns: variable, level, relativity, ci_lower, ci_upper.
    """
    rows = []
    for name, coef in result.params.items():
        if name == "Intercept":
            continue
        ci_lo, ci_hi = result.conf_int().loc[name]
        # statsmodels names categorical terms as "variable[T.level]"
        if "[T." in name:
            variable, level = name.split("[T.")
            level = level.rstrip("]")
        else:
            variable, level = name, name  # continuous — handle separately
        rows.append({
            "variable": variable,
            "level":    level,
            "log_coef": coef,
            "relativity": np.exp(coef),
            "ci_lower": np.exp(ci_lo),
            "ci_upper": np.exp(ci_hi),
        })
    return pd.DataFrame(rows)

rels = extract_relativities(model)
print(rels.to_string(index=False))
```

Output (representative — actual values will vary by sample):

```
       variable  level  relativity  ci_lower  ci_upper
       age_band  17-25        2.73      2.48      3.00
       age_band  26-35        1.57      1.43      1.72
       age_band  51-65        0.89      0.81      0.99
       age_band    66+        1.28      1.14      1.44
           area  Urban        1.38      1.28      1.48
           area  Suburban     1.09      1.01      1.18
       ncd_band  0yr          1.76      1.57      1.97
       ncd_band  1-2yr        1.38      1.25      1.52
       ncd_band  3-4yr        1.09      0.99      1.20
   veh_age_band  0-2yr        0.89      0.79      1.01
   veh_age_band  6-9yr        1.09      1.00      1.19
   veh_age_band  10yr+        1.19      1.09      1.30
```

The base levels are absent from this table because they have relativity 1.0 by construction. Add them back explicitly before presenting to a committee — a table with missing rows is a table that confuses people.

```python
# Add base level rows explicitly
base_rows = [
    {"variable": "age_band",    "level": "36-50",  "relativity": 1.0, "ci_lower": 1.0, "ci_upper": 1.0},
    {"variable": "area",        "level": "Rural",   "relativity": 1.0, "ci_lower": 1.0, "ci_upper": 1.0},
    {"variable": "ncd_band",    "level": "5yr+",    "relativity": 1.0, "ci_lower": 1.0, "ci_upper": 1.0},
    {"variable": "veh_age_band","level": "3-5yr",   "relativity": 1.0, "ci_lower": 1.0, "ci_upper": 1.0},
]
rels = pd.concat([rels, pd.DataFrame(base_rows)], ignore_index=True)
```

---

## Visualising relativities

The standard output for a pricing review is a bar chart per variable, ordered by factor level, with confidence intervals.

```python
# Ordered display sequences
level_order = {
    "age_band":     ["17-25", "26-35", "36-50", "51-65", "66+"],
    "area":         ["Urban", "Suburban", "Rural"],
    "ncd_band":     ["0yr", "1-2yr", "3-4yr", "5yr+"],
    "veh_age_band": ["0-2yr", "3-5yr", "6-9yr", "10yr+"],
}

variables = ["age_band", "area", "ncd_band", "veh_age_band"]
titles    = ["Driver age band", "Area", "NCD band", "Vehicle age band"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, var, title in zip(axes, variables, titles):
    sub = (
        rels[rels["variable"] == var]
        .set_index("level")
        .reindex(level_order[var])
    )
    x = range(len(sub))
    ax.bar(x, sub["relativity"], color="steelblue", alpha=0.75, zorder=2)
    ax.errorbar(
        x,
        sub["relativity"],
        yerr=[
            sub["relativity"] - sub["ci_lower"],
            sub["ci_upper"] - sub["relativity"],
        ],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.2,
        zorder=3,
    )
    ax.axhline(1.0, color="red", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(level_order[var], rotation=30, ha="right")
    ax.set_ylabel("Relativity")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("GLM rating relativities — Poisson frequency model", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("glm_relativities.png", dpi=150, bbox_inches="tight")
plt.show()
```

Three things to check on the output before putting it in a report:

1. **Direction.** Does the age curve move in the expected direction — young and old drivers carrying higher risk? Does NCD discount correctly decrease from 0-year to 5-year+?

2. **Confidence intervals.** A relativity with a 95% CI that straddles 1.0 is not statistically distinguishable from no effect. For `veh_age_band 0-2yr` (relativity 0.89, CI 0.79–1.01 in the example above), you cannot rule out that new vehicles carry the same risk as 3–5 year vehicles. This is either a real null result or insufficient exposure in that cell — check the counts.

3. **Lift chart.** A table of relativities says nothing about the model's ability to differentiate risk. Compute a lift chart (sorted by predicted frequency, Gini coefficient) before presenting the relativities as evidence the model works.

---

## Checking fitted values against actuals

Before anything else, confirm the model is calibrated — that predicted claim counts match observed claim counts in aggregate and within key segments.

```python
df["fitted_freq"] = model.fittedvalues / df["exposure"]
df["actual_freq"] = df["claims"] / df["exposure"]

# Overall
print(f"Predicted frequency: {df['fitted_freq'].mean():.4f}")
print(f"Actual frequency:    {df['actual_freq'].mean():.4f}")

# By age band
calibration = (
    df.groupby("age_band", observed=True)
    .agg(
        predicted=("fitted_freq", "mean"),
        actual=("actual_freq", "mean"),
        policies=("exposure", "count"),
    )
    .round(4)
)
print(calibration)
```

A GLM with a log link and offset is not guaranteed to reproduce the overall mean exactly unless the model includes an intercept (which `smf.glm` does by default). Check each major segment. A 5–10% miss on a volume segment is a model failure, not a rounding artefact.

---

## Where the GLM runs out

The workflow above is a solid starting point for any personal lines pricing model. It is also where most UK pricing teams sit today. The GLM has three limitations that become material as book complexity increases.

**Non-linearity.** A driver age variable in a GLM is either a linear slope (one coefficient, cannot represent a U-shape) or a set of dummy levels (you have to choose the bins). The U-shaped age curve is well-established; a GLM needs manual engineering to represent it. Get the bins wrong and the relativities are wrong.

**Interactions.** An area × vehicle age interaction — urban areas with old vehicles carrying additional loading — cannot appear in a main-effects GLM unless you add an explicit interaction term. On a book with five variables and five levels each, the full two-way interaction space has 100 terms. Nobody adds them all.

**Factor extraction from GBMs.** If you have already moved to a gradient boosted model for part of the rating structure, getting a multiplicative factor table back out of it is non-trivial. The model does not produce `exp(beta)` coefficients.

Two libraries address these gaps directly. [insurance-gam](https://github.com/burning-cost/insurance-gam) wraps EBM (Explainable Boosting Machine) with a proper actuarial API — `InsuranceEBM` fits an additive model where each feature gets a smooth shape function, the output is a `RelativitiesTable` comparable to what we extracted above, and it handles Poisson/Tweedie/Gamma losses with exposure offsets. On a 50,000-policy synthetic UK motor book with a known non-linear DGP, it achieves +5–15pp Gini over a linear GLM without manual polynomial terms or bin choices.

[shap-relativities](https://github.com/burning-cost/shap-relativities) does the other half: given a fitted CatBoost model, it extracts multiplicative relativities using SHAP values — including continuous age curves, not just binned levels — with confidence intervals and a validation check that the extracted table reconstructs the model's predictions. If you have a GBM sitting in a notebook that your pricing committee cannot sign off because there is no factor table, that library closes the gap.

Neither of these is a replacement for understanding the GLM workflow. The GLM is still where you should start, particularly when the book is small enough that sample variance dominates any model complexity benefit, or when you need to present to a regulator who expects interpretable coefficients. But the two libraries mean that "GLM or black box" is no longer the only choice.

---

The complete working script for this post is in the [burning-cost-examples](https://github.com/burning-cost/burning-cost-examples) repository.
