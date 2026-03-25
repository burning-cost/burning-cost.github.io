---
layout: post
title: "The EBM Is Sitting in Your Notebook Because You Can't Show It to the Committee"
date: 2026-03-16
author: Burning Cost
description: "GLMComparison and MonotonicityEditor in insurance-gam close the governance gap between EBM shape functions and the GLM factor table your pricing committee..."
tags: [ebm, gam, interpretability, GLM, monotonicity, model-governance, pricing, insurance-gam, motor, python, polars]
---

You built the EBM. Poisson deviance is lower than the production GLM. The driver age shape function picked up the U-curve between 25 and 75 that you have been bucketing by hand for years. The double-lift chart is clean. You showed it to the pricing lead and she said: "Looks interesting. Can you put it alongside the factor table?"

Six months later, it is still in a notebook.

This is not a model quality problem. It is a governance translation problem. The pricing committee reads factor tables. NCD band 0 is 1.00. Band 1 is 0.84. Band 5 is 0.52. They have opinions about these numbers, because they have been debating them for years. When you show them an EBM shape function - a continuous curve in log space - they have no reference point. They cannot tell whether the EBM agrees with the GLM at NCD 3, or whether it is doing something unexpected at the 19-year-old boundary that the GLM handles with a manual override.

`insurance-gam` has tooling specifically for this. `GLMComparison` and `MonotonicityEditor` are not modelling tools. They are governance tools. They answer the question: relative to the model you already approved, what is the EBM doing differently, and on which features?

```bash
uv add "insurance-gam[ebm]"
```

---

## The comparison workflow

Start with what you already have: the GLM factor tables. Every UK motor pricing team has them. Pull out the NCD relativity table from your last approval pack - the one with the levels the committee signed off on in whatever year you last refitted.

```python
import polars as pl
from insurance_gam.ebm import InsuranceEBM, RelativitiesTable, GLMComparison

# Your existing GLM relativities, from the last approval pack.
# These are the numbers the committee already accepted.
glm_ncd = pl.DataFrame({
    "level": ["0", "1", "2", "3", "4", "5+"],
    "relativity": [1.00, 0.84, 0.73, 0.64, 0.57, 0.52],
})

glm_driver_age = pl.DataFrame({
    "level": ["17-20", "21-24", "25-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    "relativity": [2.41, 1.72, 1.31, 1.00, 1.00, 1.04, 1.09, 1.23],
})

glm_vehicle_age = pl.DataFrame({
    "level": ["0-2", "3-5", "6-9", "10-12", "13+"],
    "relativity": [0.91, 0.96, 1.00, 1.07, 1.21],
})

# Your fitted EBM
# (assume model is already fitted - see the March 2026 post for the fitting workflow)
cmp = GLMComparison(ebm_model)
```

The `divergence_summary()` method ranks every feature by maximum absolute relativity difference between the EBM and the GLM. Run this first. It tells you where to spend the committee's time.

```python
summary = cmp.divergence_summary(
    glm_relativities_by_feature={
        "ncd_years": glm_ncd,
        "driver_age": glm_driver_age,
        "vehicle_age": glm_vehicle_age,
    }
)
print(summary)
```

```
feature       max_abs_diff  mean_abs_diff  n_levels_compared
driver_age          0.187          0.061               8
ncd_years           0.043          0.028               6
vehicle_age         0.021          0.013               5
```

Driver age diverges most. Max absolute difference is 0.187 - that is 18.7 relativity points, which is a material difference on a factor that starts at 2.41 for 17-20 year olds. NCD and vehicle age are much closer. Start with driver age.

```python
df = cmp.compare_shapes("driver_age", glm_relativities=glm_driver_age)
print(df)
```

```
level    ebm_relativity  glm_relativity  abs_diff  pct_diff
17-20             2.23            2.41      0.187      7.8%
70+               1.41            1.23      0.180     14.6%
25-29             1.19            1.31      0.120      9.2%
21-24             1.79            1.72      0.070      4.1%
60-69             1.04            1.09      0.050      4.6%
50-59             1.01            1.04      0.030      2.9%
30-39             0.98            1.00      0.020      2.0%
40-49             1.00            1.00      0.000      0.0%
```

The EBM has lower relativities for very young drivers (17-20: 2.23 vs 2.41) and higher relativities for 70+ drivers (1.41 vs 1.23). These are opposite movements. The committee will ask about both.

The young driver question is probably a data question. How many 17-20 year old policies are in the training set? If the answer is 3,200 out of 120,000, the EBM is fitting a thin segment and the 0.18 difference may be within the confidence band. The 70+ question is interesting: the GLM's 1.23 may come from a manual override applied after some large losses in that age band, while the EBM is fitting the underlying data without the override. That is worth investigating, not assuming the EBM is wrong.

The point is not that the EBM is right and the GLM is wrong. The point is that you now have a structured table that names the disagreements precisely. The committee can debate factor-by-factor, on a feature they already understand. That is a governance conversation you can have.

---

## Monotonicity: the other committee question

Vehicle age should increase with risk as the car ages. NCD should decrease with risk as years of claims-free driving accumulate. Driver age should be U-shaped - higher at 17, lower through the working years, rising again above 70. These are not statistical constraints. They are underwriting logic that the committee expects to see enforced.

An EBM fitted without constraints will sometimes violate these. Not because the data says otherwise, but because a thin segment at the boundary of a rating factor can produce a small reversal that is noise, not signal. The committee will find it immediately.

```python
import numpy as np
from insurance_gam.ebm import MonotonicityEditor

editor = MonotonicityEditor(ebm_model)

# check() returns True if the shape is already monotone, False otherwise
is_monotone = editor.check("vehicle_age", direction="increase")
print(f"vehicle_age monotone increasing: {is_monotone}")
# vehicle_age monotone increasing: False

# Inspect where the reversals are before enforcing
scores_before = editor.get_scores("vehicle_age")
diffs = np.diff(scores_before[1:])  # skip the missing-value bin at index 0
reversals = np.where(diffs < 0)[0]
print(f"Reversal bins: {reversals}")           # [8 9]  (bins 9-10 and 10-12 by index)
print(f"Max reversal: {abs(diffs[reversals]).max():.3f} log points")  # 0.031
```

Two reversals in the 9-12 year range, maximum magnitude 0.031 in log space. This is noise - the EBM saw slightly fewer claims for 10-12 year vehicles than for 9-10 year vehicles in the training data, probably a mix effect or a thin-data artefact. The underlying risk direction is clear.

Capture the scores before enforcing (for the governance pack chart), then apply isotonic regression:

```python
# Capture before-scores for the documentation chart
scores_before = editor.get_scores("vehicle_age")

# Enforce: applies isotonic regression to the stored term scores in-place
editor.enforce("vehicle_age", direction="increase")

# Verify
print(f"vehicle_age monotone after enforcement: {editor.check('vehicle_age', direction='increase')}")
# vehicle_age monotone after enforcement: True

# Optional: plot before/after for the governance pack
editor.plot_before_after("vehicle_age", scores_before=scores_before, direction="increase")
```

The adjustment is applied to the stored term scores in-place. The underlying boosting trees are not changed - only the shape function read out from them. This is a soft constraint, and it is the correct approach: you are not claiming the model should have been fitted differently, you are applying actuarial judgement to the output in the same way you would review and override a GLM factor table.

Document the adjustment. In the governance pack, note: "Vehicle age shape function had two minor reversals in the 9-12 year range (max 0.031 log points). Isotonic regression applied via `MonotonicityEditor.enforce()` to enforce actuarially expected monotonicity. No change to model discrimination metrics." That is a defensible statement. It is less defensible to leave the reversal in and wait for the committee to find it themselves.

---

## Putting the committee pack together

After running `divergence_summary()` and handling monotonicity violations, the output of the comparison workflow is a table that looks like this:

| Feature | EBM vs GLM divergence | Action |
|---------|----------------------|--------|
| driver_age | Max 18.7pp at 17-20 and 70+ | Investigate young driver data; document 70+ uplift |
| ncd_years | Max 4.3pp | Within rounding of GLM; no material change |
| vehicle_age | Max 2.1pp; 2 monotonicity violations corrected | Isotonic correction applied and documented |

Features where the EBM agrees closely with the GLM go in the pack as supporting evidence: "The EBM independently recovers the NCD curve shape within 4 percentage points across all bands. This provides independent actuarial evidence that the NCD factor table is appropriately calibrated." Features where it disagrees become the conversation, not the block.

The full relativities table for any feature is one call away:

```python
rt = RelativitiesTable(ebm_model)
print(rt.table("ncd_years"))
```

```
bin_label  raw_score  relativity
0.0            0.000        1.00
1.0           -0.174        0.84
2.0           -0.305        0.74
3.0           -0.426        0.65
4.0           -0.497        0.61
5+            -0.566        0.57
```

This is the format a pricing committee reads. The `relativity` column is normalised to the modal bin (NCD 0 here). You can drop it directly into a governance document.

---

## Why this matters now

The [original insurance-gam post](/2026/03/14/insurance-gam-interpretable-nonlinearity/) covered the three-way decision between EBM, ANAM, and PIN. That is still the right framework for choosing which architecture to use. The regulatory pressure to use it has grown since then.

FCA Consumer Duty, operational since July 2023, requires firms to demonstrate that pricing outcomes are fair on an ongoing basis. The obligation applies to the model as used, not just as documented. A GAM that produces interpretable shape functions at training time but is never reviewed at factor level after deployment is not meeting the spirit of that requirement - the black-box critique applies not to GBMs but to GAMs that are interpretable in principle but unreviewed in practice, because teams lack the tooling to do it systematically at each refresh.

`GLMComparison.divergence_summary()` is the answer to that gap. It produces a quantified, auditable comparison of EBM factors against the previously approved GLM at every refresh. When the EBM moves a factor more than 5 percentage points relative to the approved GLM, you have a documented trigger for review. When it stays within tolerance, you have documented evidence that the EBM is not drifting away from approved rates in ways the governance process did not intend.

The model is not just sitting in a notebook because the committee is unreasonable. It is sitting there because the tooling to connect EBM outputs to GLM governance vocabulary did not exist until recently. It exists now.

---

`insurance-gam` is open source under MIT at [github.com/burning-cost/insurance-gam](https://github.com/burning-cost/insurance-gam). Requires Python 3.10+; the `[ebm]` extra requires interpretML 0.6+.

**Related posts:**
- [Your Model Is Either Interpretable or Accurate. insurance-gam Refuses That Trade-Off.](/2026/03/14/insurance-gam-interpretable-nonlinearity/) - the three-way EBM/ANAM/PIN decision framework
- [Model Validation Is a Checklist, Not a Test](/2026/03/11/model-validation-pra-ss123/) - PRA SS1/23 and what a defensible sign-off process looks like
- [Validating GAMLSS Sigma Models](/2026/03/08/validating-gamlss-sigma-models/) - when the dispersion model also needs governance sign-off
