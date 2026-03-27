---
layout: post
title: "How to Export a Factor Table to Excel in Python"
date: 2026-03-24
categories: [pricing, techniques, tutorials]
tags: [catboost, shap, shap-relativities, rating-factors, relativities, excel, openpyxl, radar, emblem, uk-motor, python, tutorials]
description: "Step-by-step: extract CatBoost factor tables with shap-relativities and write a clean Excel file with openpyxl. Formatted output ready to paste into Radar or Emblem."
---

You have extracted your CatBoost relativities with `shap-relativities`. They are in a Polars DataFrame: feature, level, relativity, confidence intervals, exposure weights. Now someone asks for an Excel file to review before the next pricing committee.

`df.write_excel()` will get you a spreadsheet, but it will not get you a *usable* spreadsheet. The output is a flat dump with six decimal places, no conditional formatting, no exposure column, and no indication of which cells represent thin levels where the estimates are unreliable. A pricing actuary opening that file will either immediately ask for a better version or, worse, use it as-is and miss the thin cells.

This post shows how to produce an Excel file that a pricing actuary can actually work with: one sheet per feature, formatted relativities, conditional formatting for CI width, and a summary tab with a pivot of all factors. The whole thing is roughly 60 lines of openpyxl and runs in under a second.

---

## Setup

```bash
uv add "shap-relativities[all]" openpyxl
```

`shap-relativities[all]` brings in CatBoost, SHAP, and pandas. `openpyxl` is the Excel writer we will use directly — we want control over formatting that pandas `to_excel()` does not give us cleanly.

---

## Step 1: Fit and extract

We will use the synthetic UK motor dataset that ships with the library. If you already have a fitted `SHAPRelativities` object and a `rels` DataFrame from a previous step, skip ahead.

```python
import polars as pl
import numpy as np
import catboost
from shap_relativities import SHAPRelativities
from shap_relativities.datasets.motor import load_motor

df = load_motor(n_policies=50_000, seed=42)

df = df.with_columns([
    ((pl.col("conviction_points") > 0).cast(pl.Int32)).alias("has_convictions"),
    pl.col("area")
      .replace({"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "F": "5"})
      .cast(pl.Int32)
      .alias("area_code"),
])

features     = ["area_code", "ncd_years", "driver_age", "has_convictions"]
cat_features = ["area_code", "ncd_years", "has_convictions"]

X = df.select(features)
pool = catboost.Pool(
    X.to_pandas(),
    label=df["claim_count"].to_numpy(),
    weight=df["exposure"].to_numpy(),
)
model = catboost.CatBoostRegressor(
    loss_function="Poisson", iterations=500,
    learning_rate=0.05, depth=6, random_seed=42, verbose=0,
)
model.fit(pool)

sr = SHAPRelativities(
    model=model, X=X, exposure=df["exposure"],
    categorical_features=cat_features,
)
sr.fit()

rels = sr.extract_relativities(
    normalise_to="base_level",
    base_levels={"area_code": 0, "ncd_years": 0, "has_convictions": 0},
    ci_level=0.95,
)
```

`rels` now has columns: `feature`, `level`, `relativity`, `lower_ci`, `upper_ci`, `mean_shap`, `shap_std`, `n_obs`, `exposure_weight`.

---

## Step 2: Write the Excel file

```python
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment,
    Border, Side,
)
from openpyxl.formatting.rule import ColorScaleRule

HEADER_FILL  = PatternFill("solid", fgColor="1F3864")   # dark navy
ALT_FILL     = PatternFill("solid", fgColor="EEF2F7")   # light blue-grey
BASE_FILL    = PatternFill("solid", fgColor="FFF2CC")   # amber — marks base level
THIN_FILL    = PatternFill("solid", fgColor="FFE0E0")   # light red — thin levels
HEADER_FONT  = Font(bold=True, color="FFFFFF", size=10)
BODY_FONT    = Font(size=10)
BORDER_SIDE  = Side(style="thin", color="CCCCCC")
THIN_BORDER  = Border(
    left=BORDER_SIDE, right=BORDER_SIDE,
    top=BORDER_SIDE,  bottom=BORDER_SIDE,
)

# Column widths for each sheet
COL_WIDTHS = {
    "A": 16,   # level
    "B": 12,   # relativity
    "C": 12,   # lower_ci
    "D": 12,   # upper_ci
    "E": 10,   # n_obs
    "F": 14,   # exposure_weight
    "G": 12,   # ci_width (derived)
}

HEADERS = ["Level", "Relativity", "Lower 95% CI", "Upper 95% CI",
           "Obs", "Exposure", "CI Width"]


def _write_factor_sheet(ws, feature_rels: pl.DataFrame, feature_name: str):
    """Write one feature's relativities to a worksheet."""
    ws.title = feature_name[:31]   # Excel tab names max 31 chars

    # Header row
    for col_idx, header in enumerate(HEADERS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        cell.border = THIN_BORDER

    for col_letter, width in COL_WIDTHS.items():
        ws.column_dimensions[col_letter].width = width
    ws.row_dimensions[1].height = 20

    base_level = feature_rels.filter(pl.col("relativity") == 1.0)["level"][0]

    for row_idx, row in enumerate(feature_rels.iter_rows(named=True), start=2):
        is_base = str(row["level"]) == str(base_level)
        is_thin = int(row["n_obs"]) < 100   # flag levels with fewer than 100 obs

        ci_width = row["upper_ci"] - row["lower_ci"]
        values = [
            row["level"],
            row["relativity"],
            row["lower_ci"],
            row["upper_ci"],
            row["n_obs"],
            row["exposure_weight"],
            ci_width,
        ]

        for col_idx, value in enumerate(values, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.border = THIN_BORDER
            cell.font = BODY_FONT

            # Number formats
            if col_idx in (2, 3, 4, 7):   # relativity and CI columns
                cell.number_format = "0.000"
            elif col_idx == 6:
                cell.number_format = "#,##0.0"
            elif col_idx == 5:
                cell.number_format = "#,##0"

            # Row fills
            if is_base:
                cell.fill = BASE_FILL
            elif is_thin:
                cell.fill = THIN_FILL
            elif row_idx % 2 == 0:
                cell.fill = ALT_FILL

    # Colour scale on CI width column (G) — wider CI = more red
    last_row = len(feature_rels) + 1
    ws.conditional_formatting.add(
        f"G2:G{last_row}",
        ColorScaleRule(
            start_type="min", start_color="63BE7B",    # green: tight CI
            end_type="max",   end_color="F8696B",      # red: wide CI
        ),
    )

    # Freeze header row
    ws.freeze_panes = "A2"


def write_factor_table(rels: pl.DataFrame, path: str) -> None:
    wb = Workbook()
    wb.remove(wb.active)   # remove default empty sheet

    features = rels["feature"].unique(maintain_order=True).to_list()

    for feature in features:
        ws = wb.create_sheet()
        feature_rels = rels.filter(pl.col("feature") == feature)
        _write_factor_sheet(ws, feature_rels, feature)

    # Summary sheet: all features in one pivot
    ws_summary = wb.create_sheet(title="Summary", index=0)
    for col_idx, header in enumerate(
        ["Feature", "Level", "Relativity", "Lower CI", "Upper CI", "Obs"],
        start=1,
    ):
        cell = ws_summary.cell(row=1, column=col_idx, value=header)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        cell.border = THIN_BORDER

    ws_summary.column_dimensions["A"].width = 16
    ws_summary.column_dimensions["B"].width = 16
    for c in "CDEF":
        ws_summary.column_dimensions[c].width = 13

    row_idx = 2
    for row in rels.iter_rows(named=True):
        for col_idx, value in enumerate(
            [row["feature"], row["level"], row["relativity"],
             row["lower_ci"], row["upper_ci"], row["n_obs"]],
            start=1,
        ):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
            cell.border = THIN_BORDER
            cell.font = BODY_FONT
            if col_idx in (3, 4, 5):
                cell.number_format = "0.000"
        row_idx += 1

    ws_summary.freeze_panes = "A2"
    wb.save(path)
    print(f"Saved {path}")


write_factor_table(rels, "factor_table.xlsx")
```

The output:
- A **Summary** tab: all features combined, one row per factor level, relativities to three decimal places.
- One **per-feature tab**: header in navy, alternating row shading, base level in amber, thin levels (fewer than 100 observations) in light red, CI width column colour-scaled from green (tight) to red (wide).
- Frozen header rows on every sheet.

---

## What the formatting is telling you

**Amber row** = base level. This is the level with relativity 1.0. Everything else is expressed relative to it. Make sure the pricing committee knows which level is base before the meeting — "area A is base" is a fact that gets forgotten between quarterly reviews and causes confusion when a column of relativities shows what looks like two rows of 1.000.

**Red row** = thin level. Fewer than 100 observations means the CLT confidence interval is unreliable. You should not present relativities for these levels without a credibility weight or a note in the commentary. The red flag is a reminder to flag them explicitly rather than presenting them on equal footing with well-observed levels.

**CI width colour scale** = data quality gradient. A level with a CI of ±0.05 is well-supported. A level with a CI of ±0.40 is not. The colour scale makes this immediately visible across all levels without requiring anyone to calculate `upper - lower` by eye.

---

## The Radar/Emblem workflow

A Radar import template expects: factor name, factor level, relativity (numeric, to at least 3 decimal places), and usually a min/max override column and a load/deduction flag. The `factor_table.xlsx` structure above is close but not identical — Radar's native import format uses a fixed column order and expects the factor levels in exactly the form they appear in the rating algorithm's factor table reference.

The practical workflow most teams use is: export the `factor_table.xlsx` as a reference document for the pricing committee review, then produce a separate `radar_import.xlsx` with Radar's specific column layout derived from the same `rels` DataFrame. These are two different files for two different audiences. The formatted one is for the governance review. The import file is for the systems team. Conflating them causes problems in both directions.

For Emblem (Guidewire), the equivalent is the `.xml` factors file or the CSV rate table import, depending on which Emblem release your team is on. If you are on Emblem 9.x and using the CSV import path, the `rels` DataFrame maps directly: feature name → factor name, level → factor code, relativity → factor value.

---

## What to tell the pricing committee about thin levels

The model has a relativity for every level it has seen in the training data. Some of those levels have five observations. Some have fifteen. The relativity for those levels is not meaningless — the SHAP values are real — but the uncertainty is very large. The 95% confidence interval for a level with 12 observations typically spans ±0.5 to ±1.2 log-points, which corresponds to a multiplicative uncertainty range of roughly 0.6× to 1.8× on the point estimate.

Our recommendation: for levels with fewer than 100 observations, credibility-weight the GBM relativity against the GLM relativity or against the portfolio mean. The library does not do this automatically — it is a judgment call that depends on how different the GBM and GLM are for that level and how comfortable the pricing committee is with the exposure. What the library does give you is the observation count and the CI, which are the two numbers you need to have that conversation.

If your portfolio has many thin levels — small commercial lines, niche vehicle types, high-performance segments — consider whether the SHAP relativity extraction is the right approach at all. The `shap-relativities` library was designed for personal lines frequency models where most factor levels have sufficient volume. For very thin books, a Bayesian hierarchical model or explicit credibility weighting at the modelling stage is more appropriate.

---

## Full library

Source at [github.com/burning-cost/shap-relativities](https://github.com/burning-cost/shap-relativities). For distilling GBM into factor tables, see [`insurance-distill`](/insurance-distill/).

- [How to Extract GLM-Style Rating Factors from a CatBoost Model](/2026/03/02/how-to-extract-rating-factors-from-catboost/) — the upstream step: fitting the model and running `extract_relativities()`
- [Extracting Rating Relativities from GBMs with SHAP](/2026/02/17/extracting-rating-relativities-from-gbms-with-shap/) — the conceptual background
- [Your GBM and GLM Are Not Competitors](/2026/02/28/your-gbm-and-glm-are-not-competitors/) — when the factor tables diverge and what to do about it
