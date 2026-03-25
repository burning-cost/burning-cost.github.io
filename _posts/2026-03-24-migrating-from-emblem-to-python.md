---
layout: post
title: "Migrating from Emblem to Python: What Actually Changes"
date: 2026-03-24
categories: [pricing]
tags: [emblem, python, glm, polars, catboost, statsmodels, shap-relativities, insurance-distill, insurance-monitoring, radar, migration, workflow, pricing-actuary]
excerpt: "If you are considering moving your GLM workflow from Emblem to Python, the modelling is not the hard part - here is what is."
---

Most UK motor pricing teams have used Emblem for a decade or more. The GUI is familiar, the factor table view is sensible, and everyone in the team knows how to navigate it. If you are thinking about moving to Python, you are probably not doing it because Emblem is bad. You are doing it because Python opens up things Emblem cannot do: CatBoost and XGBoost, conformal prediction intervals, causal inference, proper temporal cross-validation, and a model development process that lives in version control.

This post covers what actually changes in the workflow - not the syntax, but the shape of how you work. We are honest about what is harder in Python and what is genuinely better.

---

## What stays the same

The statistical models are identical. A Poisson GLM with a log link and exposure offset fitted in statsmodels produces the same estimates as Emblem's frequency model, to rounding. A Gamma GLM on claims-with-at-least-one-claim, frequency-weighted, gives the same severity structure. The factor tables express the same multiplicative relativities. If you handed a well-specified statsmodels model to an Emblem user and asked them to verify it, they could.

We showed the numerical equivalence in [How to Reproduce an Emblem GLM in Python](/2026/03/23/how-to-reproduce-emblem-glm-in-python/) - Poisson frequency, Gamma severity, deviance residuals, one-way plots, and lift chart. Everything matches.

The pricing logic is also unchanged. Frequency times severity gives pure premium. You still need to think about base levels, reference categories, factor smoothing, and whether area belongs in the severity model. Python does not make those actuarial judgements for you.

---

## Data preparation: Polars replaces Emblem's built-in

Emblem handles data within its own environment. You load a flat file, define exposure and claim count columns, and the software manages the rest. Moving to Python means you own the data pipeline explicitly.

We use Polars rather than pandas. Polars runs faster on large insurance datasets, has cleaner syntax for the aggregations pricing actuaries need, and handles lazy evaluation properly for Databricks. The learning curve coming from pandas is moderate; coming from Emblem's built-in data handling, you are starting mostly from scratch anyway.

A typical motor data preparation step looks like this:

```python
import polars as pl

df = (
    pl.scan_parquet("motor_policies.parquet")
    .filter(pl.col("policy_year").is_between(2021, 2024))
    .with_columns([
        # Exposure: cap at 1.0 for annual policies, actual fraction for short-term
        pl.col("days_on_cover").truediv(365.25).clip(0.0, 1.0).alias("exposure"),
        # Age band matching Emblem factor definition
        pl.col("driver_age")
          .cut([24, 34, 44, 54, 64], labels=["17-24","25-34","35-44","45-54","55-64","65+"])
          .alias("age_band"),
        # NCD as string for GLM categorical treatment
        pl.col("ncd_years").clip(0, 5).cast(pl.Utf8).alias("ncd_f"),
    ])
    .with_columns([
        # Frequency: avoid division for zero exposure records
        pl.when(pl.col("exposure") > 0)
          .then(pl.col("claim_count") / pl.col("exposure"))
          .otherwise(None)
          .alias("freq"),
    ])
    .collect()
)
```

This is more verbose than clicking a column definition in Emblem's data panel. But it is also in a file, in git, reviewed in a pull request. When someone asks six months later why the 17-24 band starts at 17 rather than 18, the answer is in the commit history.

One Emblem habit that transfers badly: Emblem's GUI tends to encourage working on aggregated summary tables (one row per factor combination). Python and Polars work best on policy-level data. Fit your models on the full policy file, not on a pre-aggregated summary. The information loss from aggregating before fitting is real, especially for the severity model where claim-count weighting matters at policy level.

---

## Model fitting: statsmodels replaces the Emblem GUI

The Emblem model development screen gives you a list of factors, a fit button, an ANOVA table, and a factor table panel that updates on each fit. In Python, `statsmodels.formula.api.glm()` is the equivalent for GLMs.

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

df_pd = df.to_pandas()
df_pd["log_exp"] = np.log(df_pd["exposure"])

freq_model = smf.glm(
    formula=(
        "claim_count ~ "
        "C(ncd_f, Treatment(reference='3')) + "
        "C(age_band, Treatment(reference='35-44')) + "
        "C(veh_group, Treatment(reference='5')) + "
        "C(area, Treatment(reference='D'))"
    ),
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exp"],
).fit()
```

The formula interface maps cleanly to Emblem's factor list. `Treatment(reference='3')` is the equivalent of setting a base level. The offset handles exposure correctly: it enters the linear predictor directly, so the model is fitting expected claim count, not expected frequency. Passing exposure as `freq_weights` is a common mistake that gives biased estimates whenever exposure correlates with rating factors - and in UK motor it always does.

For severity, the model is Gamma with claim-count weights:

```python
claims = df_pd[df_pd["claim_count"] > 0].copy()
claims["avg_sev"] = claims["severity"] / claims["claim_count"]

sev_model = smf.glm(
    formula="avg_sev ~ C(veh_group, Treatment(reference='5')) + C(area, Treatment(reference='D'))",
    data=claims,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    freq_weights=claims["claim_count"],
).fit()
```

NCD is absent from severity for the same reason it would be in Emblem: NCD is a selection mechanism, not a driver of repair costs.

---

## Factor extraction

In Emblem, the factor table appears automatically after fitting. In Python, you build it. This is one of the genuine friction points in migration.

The manual version is a loop over `freq_model.params` and `freq_model.conf_int()`, extracting the parameter names that match your factor prefix and exponentiating. We built [`shap-relativities`](https://github.com/burning-cost/shap-relativities) partly to solve the GBM case, but its `factor_table()` method works for any model whose outputs are already on the log scale.

For GLMs specifically, the extraction is:

```python
import polars as pl
import numpy as np

def glm_factor_table(model, factor_name, base_label):
    params = model.params
    conf   = model.conf_int()
    prefix = f"C({factor_name}"
    rows   = []

    for name, coef in params.items():
        if name.startswith(prefix):
            level = name.split("[T.")[-1].rstrip("]")
            rows.append({
                "level":      level,
                "relativity": round(np.exp(coef), 4),
                "ci_lo":      round(np.exp(conf.loc[name, 0]), 4),
                "ci_hi":      round(np.exp(conf.loc[name, 1]), 4),
            })

    rows.append({"level": base_label, "relativity": 1.0, "ci_lo": 1.0, "ci_hi": 1.0})
    return pl.DataFrame(rows).sort("relativity")

ncd_table = glm_factor_table(freq_model, "ncd_f", "3")
```

When you are working with CatBoost rather than a GLM, `shap-relativities` handles the full extraction using SHAP values to decompose the GBM's predictions into per-factor relativities on a multiplicative scale:

```python
from shap_relativities import SHAPRelativities

sr = SHAPRelativities(cb_model, X_train)
sr.fit()
all_rels = sr.extract_relativities()
ncd_gbm_table = all_rels.filter(all_rels["feature"] == "ncd_f")
```

This lets you compare the GBM's implicit factor structure against the GLM's explicit one. Divergence at extreme levels - say, the GBM giving NCD-0 a relativity of 1.65 where the GLM gives 1.39 - flags either a non-linear effect the GLM is missing or a thin-data artefact in the GBM.

---

## Validation pipeline

This is where Emblem genuinely has an advantage: its diagnostic plots are integrated and automatic. You fit, and the one-way plots and deviance residuals appear.

In Python, you build the validation pipeline once and reuse it. The upside is that the pipeline is explicit, parameterised, and testable. The downside is that the first time you build it, it takes a day.

The components you need:

**One-way plots.** Aggregate observed and fitted frequency (or severity) by each factor level, weighted by exposure. Plot observed vs fitted with an exposure bar chart on a secondary axis. The pattern should show observed and fitted lines tracking each other; a systematic bend means the factor needs finer banding or an interaction.

**Deviance residuals.** `model.resid_deviance` gives you these directly. For a well-specified Poisson GLM on a large book, the standard deviation should be close to 1.0. Values substantially above 1.0 mean overdispersion; consider quasi-Poisson or negative binomial.

**Lift chart.** Sort policies by predicted risk, group into deciles, compare observed vs GLM vs GBM within each decile. CatBoost typically outperforms a GLM in the tails; the question is by how much and whether the difference is large enough to justify the deployment complexity.

**Gini coefficient.** Use the exposure-weighted version - see [our post on the exposure-weighted Gini](/2026/03/23/exposure-weighted-gini-coefficient-python/) for the exact implementation. The naive unweighted version is wrong for insurance because short-term policies contribute less exposure but the same policy count.

**Temporal cross-validation.** Emblem does not do this at all. Standard k-fold CV applied to insurance data is wrong because it mixes time periods: a test fold drawn from the same years as the training set will share the same trend, IBNR pattern, and seasonal effects. Proper insurance CV holds out the most recent period(s) as test data. We made the full case for this in [the k-fold post](/2026/03/21/why-k-fold-cv-is-wrong-for-insurance/).

---

## What's harder in Python

No point pretending otherwise. Here is the genuine list:

**No GUI.** There is no equivalent of clicking a factor in Emblem's list and seeing the one-way plot update. Everything is code. For actuaries who are more comfortable with point-and-click tooling, the transition is uncomfortable. The payoff is that "the code is the analysis" - everything is reproducible, diffable, and reviewable.

**Factor smoothing.** Emblem's GUI lets you manually drag points to smooth a factor curve. In Python you do this programmatically: monotone splines, isotonic regression, GAMs. [insurance-gam](https://github.com/burning-cost/insurance-gam) handles non-linear smooth terms properly. The result is actually better than manual dragging, but it requires knowing what you are doing.

**Building the environment.** Emblem installs as a Windows application with a known set of dependencies. Setting up a Python pricing environment on Databricks - package management, cluster configuration, library versions - takes real effort the first time. Once it is set up and codified, it replicates reliably; that setup phase is real work.

**Stakeholder communication.** Factor tables in a Python DataFrame look different from Emblem's formatted output. You will need to build Excel export logic (or use openpyxl / xlsxwriter) to produce the governance pack format that pricing committees expect. This is a one-time build, but it is not free.

---

## Deployment to Radar via insurance-distill

Once you have a fitted GBM you want to get into Radar, [`insurance-distill`](https://github.com/burning-cost/insurance-distill) handles the GLM distillation step: it fits a surrogate Poisson or Gamma GLM on the GBM's predictions, bins continuous variables at the GBM's natural split points, and produces multiplicative factor tables in a format compatible with Radar's CSV import template.

```bash
uv add insurance-distill
```

```python
from catboost import CatBoostRegressor
from insurance_distill import SurrogateGLM

surrogate = SurrogateGLM(
    model=fitted_catboost,
    X_train=X_train,        # Polars DataFrame
    y_train=claim_counts,
    exposure=exposure,
    family="poisson",
)

surrogate.fit(max_bins=10)

# Export factor tables
# Access all factor tables at once via the report
report = surrogate.report()
tables = report.factor_tables          # dict[str, pl.DataFrame], one per feature

# Export all factor tables as CSVs for Radar/Emblem import
surrogate.export_csv("output/factors/", prefix="motor_freq_")
# Writes: motor_freq_driver_age.csv, motor_freq_vehicle_value.csv, ...
```

The approach is: use the GBM's predictions as targets for the GLM, so the GLM learns to approximate the GBM's structure rather than the noisy raw claims. The factor tables that come out are multiplicative, interpretable, and can be loaded into Radar via its standard import template. The loss in discrimination relative to running the GBM directly is typically 1-2 Gini points - a price worth paying for a rating engine that auditors and regulators can review.

---

## Drift monitoring after deployment

Emblem has no post-deployment monitoring. Once your factors are in Radar, you are checking actual vs expected manually in Excel, usually quarterly, usually without a statistical test for whether what you see is noise or signal.

[`insurance-monitoring`](https://github.com/burning-cost/insurance-monitoring) runs automated drift detection against your deployed model's predictions. It tracks the distribution of model inputs over time (covariate drift), the ratio of observed to expected claims (response drift), and the Gini coefficient on a rolling out-of-time window.

```bash
uv add insurance-monitoring
```

```python
from insurance_monitoring import MonitoringReport

# Run on new quarter — MonitoringReport is a dataclass, no .run() call needed
report = MonitoringReport(
    reference_actual=reference_claims,
    reference_predicted=surrogate.model.predict(X_train),
    current_actual=claims_2024q1,
    current_predicted=surrogate.model.predict(X_2024q1),
    exposure=exposure_2024q1,
    feature_df_reference=X_train,
    feature_df_current=X_2024q1,
)

print(report.recommendation)  # NO_ACTION | RECALIBRATE | REFIT | INVESTIGATE
# Flags: covariate drift on veh_group (PSI=0.14), AE ratio 1.08 (warning threshold 1.05)
```

The Population Stability Index (PSI) flags when the distribution of an input variable has shifted relative to the training period. A PSI of 0.14 and an AE warning threshold of 1.05 are reasonable defaults on a book of 50k+ policies; on smaller books these thresholds need calibrating to your actual data volume. An AE ratio of 1.08 on frequency means claims are running 8% above what the model expects - not necessarily a refitting trigger, but worth monitoring. The Gini coefficient on recent months tells you whether the model is still discriminating correctly or has flattened.

---

## The version control argument

This is the argument that closes the sale for most teams, especially those who have had an Emblem model and are not sure exactly what parameters it was fitted on.

Every model development step in Python is a file. The data preparation script, the model specification, the validation outputs, the factor tables - all of it is in git. The model that went live in January is a tagged commit. The January model versus the March model is a `git diff`. When the governance committee asks what changed between the Q4 model review and the live model, you show them the diff.

Emblem stores model specifications in proprietary binary formats. You can export factor tables to Excel. But "what exactly was this model trained on, with which parameters, on which data extract?" is harder to answer definitively than it should be.

---

## Where to start

If you want to try Python pricing on your own book before committing to a full migration:

1. Install the stack: `uv add statsmodels catboost polars shap-relativities insurance-distill insurance-monitoring`
2. Take your Emblem model specification and reproduce it in Python using the GLM workflow above
3. Compare factor tables between the two - they should agree closely if the base levels and data prep match
4. Add a CatBoost challenger and look at the lift chart
5. Once you trust the Python output, use `insurance-distill` to generate the Radar-ready factor tables

The [Python insurance pricing cookbook](/2026/03/22/python-insurance-pricing-cookbook/) covers the full environment setup including Databricks cluster configuration. The [complete toolkit post](/2026/03/22/complete-python-insurance-pricing-toolkit-2026/) lists every library we use and why.

---

The workflow change is real. Python requires more upfront code than Emblem's GUI. What you get in return is a pipeline that is reproducible, extensible to techniques Emblem cannot support, and auditable by anyone with access to the repository - including future you, when you cannot remember what you did six months ago.
