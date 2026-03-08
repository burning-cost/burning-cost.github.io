# Module 2: GLMs in Python - The Bridge from Emblem

In Module 1 you created a Databricks account, started a cluster, ran your first Python cells, and saved data to a Delta table. This module builds on that foundation. By the end of it you will have fitted a working Poisson frequency GLM and a Gamma severity GLM on a realistic motor dataset, extracted factor relativities, run diagnostics, and seen how the Python output compares to what Emblem would produce.

This is not a statistics lecture. We assume you already understand what a GLM is, what IRLS does, and what a relativity means. The goal here is to show you that the same model you have been running in Emblem can be reproduced in Python - with better auditability, better version control, and a cleaner integration with the rest of a modern pricing stack.

---

## Why this is worth doing

Let us be clear about something before we start. Emblem fits GLMs correctly. It uses IRLS, the same algorithm Python's statsmodels uses. It handles exposure offsets. It produces deviance statistics, factor charts, and actual-versus-expected plots. If you are a pricing actuary who has been fitting frequency-severity models in Emblem for ten years, you are not doing it wrong.

The problem is the infrastructure around the model. The Emblem project file is not version-controlled. Nobody commits it to Git. The data extract you fed it lives on a network drive with a name like `motor_extract_final_v2_ACTUAL.csv`. When the FCA asks you to reproduce the relativities from your Q3 2023 renewal cycle, you go hunting for the right combination of software version, project file, and data extract - and you hope nothing has changed.

This matters now more than it did five years ago. PS 21/5 (the general insurance pricing practices rules, effective January 2022) banned price walking and introduced explicit audit trail requirements for pricing decisions. Consumer Duty (PS 22/9, effective July 2023) extended this further to require demonstrable fair value - meaning the FCA wants to walk into your office, ask about any price charged to any customer in the past three years, and have you show them the model, the inputs, and the decision trail in under an hour.

Moving your GLM to Python and Databricks solves this. The model code is version-controlled. The training data is a Delta table with time travel. The fitted model is logged to MLflow with the parameters, metrics, and artefacts. Running the model from six months ago means checking out the relevant Git tag and pointing at the Delta table at that timestamp. That is reproducibility you can demonstrate to a regulator.

The GLM itself is not the hard part. The encoding, the validation, and the export are where the work is. That is what this module covers.

---

## Part 0: Setting up your notebook

Before we write any modelling code, we need a notebook and the right libraries installed.

### Creating the notebook

In your Databricks workspace, go to the left sidebar and click **Workspace**. Navigate to your user folder (or your Git repo if you set one up in Module 1). Click the **+** button and choose **Notebook**.

Name it something like `module-02-glm-frequency-severity`. Keep the default language as Python. Click **Create**.

Check the cluster selector at the top of the notebook. If it says "Detached," click it and select your cluster. Once it shows the cluster name in green, you are connected and ready to run cells.

### Installing the libraries

The three libraries we need for this module are:

- **Polars** - for data manipulation. Think of it as Excel tables in Python. You define calculations on columns, filter rows, and group data - but it handles millions of rows instantly and the syntax is explicit about what is happening. We introduced Polars briefly in Module 1.
- **numpy** - for numerical computing. Arrays of numbers, mathematical functions (exp, log, etc.), and the random number generators we use to build our synthetic dataset. You will see it abbreviated as `np` throughout.
- **statsmodels** - the library that fits the GLMs. It contains Poisson, Gamma, Tweedie, and quasi-Poisson families, uses IRLS exactly as Emblem does, and produces coefficient tables, deviance statistics, and confidence intervals. You will see it abbreviated as `sm` or `smf`.

In a new cell at the top of your notebook, type this and run it (Shift+Enter):

```python
%pip install polars statsmodels scipy matplotlib
```

You will see a stream of output as pip downloads and installs the packages. Wait for it to finish. At the end it says something like:

```
Note: you may need to restart the Python kernel to use updated packages.
```

In the next cell, run:

```python
dbutils.library.restartPython()
```

This restarts the Python session. Any variables you defined before this point are gone - the session resets. This is expected. Always put your `%pip install` cell at the very top of the notebook so you only need to restart once per session.

**Why numpy is not in the install list:** numpy comes pre-installed with Databricks Runtime. You do not need to pip install it.

### Confirm the imports work

In the next cell, run this to confirm everything is installed correctly:

```python
import polars as pl
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

print(f"Polars version: {pl.__version__}")
print(f"Statsmodels version: {sm.__version__}")
print("All imports OK")
```

You should see version numbers printed with no errors. If you see `ModuleNotFoundError`, the install cell did not work - check that you ran it before the restart, and if necessary run `%pip install` again.

---

## Part 1: The workflow we are building

We are building a motor frequency-severity model: a Poisson GLM for claim frequency and a Gamma GLM for average severity, both with log link and exposure offset. The pure premium estimate is the product: frequency times severity.

The data pipeline, in order:

1. Generate a synthetic UK motor dataset with known true parameters
2. Prepare features: encode categorical factors, handle base levels, check for data quality issues
3. Fit the frequency GLM (Poisson with log link and exposure offset)
4. Fit the severity GLM (Gamma with log link, on claimed policies only)
5. Run diagnostics: deviance residuals, actual-versus-expected by factor level
6. Validate against known true parameters (and later, against Emblem output)
7. Export factor tables in the format Radar expects

We use synthetic data throughout because it has known true parameters. This lets us verify that our GLM is recovering the right answers - if the model works correctly on synthetic data, we can trust it on real data.

---

## Part 2: Building the dataset

### What we are generating

We need 100,000 synthetic motor policies with these attributes:

- Area band (A through F, roughly corresponding to postcode area bands)
- ABI vehicle group (1-50)
- NCD years (0-5)
- Driver age (17-85)
- Conviction flag (0 or 1)
- Earned exposure (fraction of a policy year, 0.05 to 1.0)

For each policy we generate a claim count (from the Poisson process) and, for claimed policies, an average severity (from the Gamma process). We know the true parameters because we define them ourselves.

### A new Python concept: the random number generator

`np.random.default_rng(seed=42)` creates a random number generator with a fixed starting point (`seed=42`). Using the same seed every time means the synthetic data is reproducible - you and a colleague running this same notebook will get exactly the same 100,000 policies. If you change the seed, you get a different dataset with the same statistical properties.

### A new Python concept: `np.where`

`np.where(condition, value_if_true, value_if_false)` applies a condition to every element of an array and returns a new array. It is the vectorised equivalent of an IF statement in Excel. When you see `np.where(area == "B", 0.10, 0)`, it returns an array where each element is 0.10 if the corresponding policy is in area B, and 0 otherwise.

### Generating the data

Create a new cell and run this. We will explain what happened immediately afterwards.

```python
import polars as pl
import numpy as np

rng = np.random.default_rng(seed=42)
n = 100_000

# Rating factors - UK motor conventions
areas = ["A", "B", "C", "D", "E", "F"]
area = rng.choice(areas, size=n, p=[0.10, 0.18, 0.25, 0.22, 0.15, 0.10])

vehicle_group = rng.integers(1, 51, size=n)  # ABI group 1-50
ncd_years = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.08, 0.07, 0.09, 0.12, 0.20, 0.44])
driver_age = rng.integers(17, 86, size=n)
conviction_flag = rng.binomial(1, 0.06, size=n)
exposure = np.clip(rng.beta(8, 2, size=n), 0.05, 1.0)

# True log-frequency parameters (GLM intercept + log-linear effects)
INTERCEPT = -3.10
TRUE_PARAMS = {
    "area_B": 0.10, "area_C": 0.20, "area_D": 0.35,
    "area_E": 0.50, "area_F": 0.65,
    "vehicle_group": 0.018,   # per ABI group unit above 1
    "ncd_years": -0.13,       # per year of NCD
    "young_driver": 0.55,     # age < 25
    "old_driver": 0.28,       # age > 70
    "conviction": 0.42,
}

# Build the log expected claim rate for each policy
log_mu = (
    INTERCEPT
    + np.where(area == "B", TRUE_PARAMS["area_B"], 0)
    + np.where(area == "C", TRUE_PARAMS["area_C"], 0)
    + np.where(area == "D", TRUE_PARAMS["area_D"], 0)
    + np.where(area == "E", TRUE_PARAMS["area_E"], 0)
    + np.where(area == "F", TRUE_PARAMS["area_F"], 0)
    + TRUE_PARAMS["vehicle_group"] * (vehicle_group - 1)
    + TRUE_PARAMS["ncd_years"] * ncd_years
    + np.where(driver_age < 25, TRUE_PARAMS["young_driver"], 0)
    + np.where(driver_age > 70, TRUE_PARAMS["old_driver"], 0)
    + TRUE_PARAMS["conviction"] * conviction_flag
    + np.log(exposure)
)

freq_rate = np.exp(log_mu - np.log(exposure))  # annualised frequency
claim_count = rng.poisson(freq_rate * exposure)

# Severity DGP: Gamma with mean around £3,500, vehicle group effect only.
# NCD reflects driver behaviour and correlates with claim frequency,
# not individual claim size. Including NCD in the severity model would
# capture frequency effects through the back door.
sev_log_mu = (
    np.log(3500)
    + 0.012 * (vehicle_group - 1)
)
true_mean_sev = np.exp(sev_log_mu)
shape_param = 4.0  # coefficient of variation = 1/sqrt(4) = 0.5

has_claim = claim_count > 0
avg_severity = np.where(
    has_claim,
    rng.gamma(shape_param, true_mean_sev / shape_param),
    0.0
)

df = pl.DataFrame({
    "policy_id": np.arange(1, n + 1),
    "area": area,
    "vehicle_group": vehicle_group,
    "ncd_years": ncd_years,
    "driver_age": driver_age,
    "conviction_flag": conviction_flag,
    "exposure": exposure,
    "claim_count": claim_count,
    "avg_severity": avg_severity,
    "incurred": avg_severity * claim_count,
})

print(f"Portfolio: {len(df):,} policies")
print(f"Exposure: {df['exposure'].sum():,.0f} earned years")
print(f"Claims: {df['claim_count'].sum():,} ({df['claim_count'].sum() / df['exposure'].sum():.3f}/year)")
print(f"Total incurred: £{df['incurred'].sum() / 1e6:.1f}m")
```

**What you should see:** Four printed lines showing the portfolio summary. Roughly 88,000 earned years, around 10,000-12,000 claims, total incurred around £40-50m. The exact numbers depend on the random draws but these ranges are correct.

**What the code did:** It generated 100,000 arrays of random numbers (one element per policy), calculated a log-expected-frequency for each policy using the true parameter values, drew actual claim counts from a Poisson distribution with those expected frequencies, and then drew severity amounts from a Gamma distribution for the policies that had claims. The whole thing is stored in a Polars DataFrame called `df`.

### A new Python concept: f-strings

The `f"..."` syntax in the print statements is an f-string. The `f` prefix tells Python to interpret `{...}` inside the string as an expression to evaluate. So `f"{df['claim_count'].sum():,}"` prints the claim count sum with thousand separators. The `:,` inside the braces is a format specifier meaning "use commas as thousand separators." The `:.3f` means "three decimal places."

### Sanity-checking the data

Before fitting any model, look at the data. In a new cell:

```python
# First few rows
df.head(5)
```

You should see a table with 10 columns and 5 rows. Look at the exposure values - they should all be between 0.05 and 1.0. The claim_count should be 0 for most rows (this is a motor book, not a high-frequency line). The avg_severity should be 0.0 for rows with no claims.

In the next cell:

```python
# Summary statistics
df.describe()
```

This shows count, mean, standard deviation, min, and max for each numeric column. The minimum claim_count should be 0, the minimum exposure should be around 0.05. If you see negative values in either column, something went wrong in the generation step.

In the next cell, check a one-way by area:

```python
df.group_by("area").agg(
    pl.col("claim_count").sum().alias("claims"),
    pl.col("exposure").sum().alias("exposure"),
    pl.len().alias("policies"),
).with_columns(
    (pl.col("claims") / pl.col("exposure")).alias("freq")
).sort("area")
```

**What you should see:** Area A should have the lowest claim frequency (it is the base area with no uplift). Area F should have the highest, roughly 1.9x area A. This matches `exp(0.65) ≈ 1.92`, the true area F effect we defined in TRUE_PARAMS.

**A new Python concept: method chaining.** Polars lets you chain operations with `.`. Each method returns a new DataFrame, so you can chain `.group_by()`, `.agg()`, `.with_columns()`, `.sort()` one after another. This is the same idea as Excel's CTRL+T tables with multiple computed columns, but written in a chain rather than spread across cells.

---

## Part 3: Factor encoding

### Why encoding matters

This is where most GLM migrations go wrong, and it is worth spending time on before touching statsmodels.

In a GLM, every categorical factor must have a reference category - what Emblem calls the "base level." For a factor with k levels, the model estimates k-1 coefficients. The reference category has no coefficient; its effect is absorbed into the intercept. The relativity for each non-reference level is `exp(beta)` relative to the reference.

Python's statsmodels picks the base level automatically. By default, it uses the first level alphabetically or numerically. So for area, that is A. For NCD years, that is 0. For vehicle group if treated as a factor, that is 1. These happen to match Emblem's typical defaults for this dataset - but if you are not explicit about it, you are relying on a coincidence.

If your Emblem model uses area A as base and your Python model defaults to something different, every single area relativity will be off by a constant multiplier. It will not look like a coding error. The relativities will have the right shape but the wrong scale. This is the number one source of "why don't the numbers match" on Emblem-to-Python migrations.

### Encoding area as an Enum

A Polars `Enum` is a categorical column with an explicit, ordered list of allowed values. By defining the order ourselves, we control which level appears first when the data is passed to statsmodels, and therefore which level statsmodels treats as the base.

```python
# Encode area with explicit ordering - area A is first, so it becomes the base
area_order = ["A", "B", "C", "D", "E", "F"]
df = df.with_columns(
    pl.col("area").cast(pl.Enum(area_order))
)
```

Run this cell. No output is expected - it modifies the DataFrame in place. Check it worked:

```python
print(df.dtypes)
```

You should see `area` listed as `Enum(categories=['A', 'B', 'C', 'D', 'E', 'F'])`.

### Preparing for statsmodels

statsmodels requires pandas DataFrames, not Polars DataFrames. We convert at the point of model fitting and keep everything else in Polars. The reason: Polars is faster and more readable for data manipulation; statsmodels simply does not support Polars yet. The conversion is a one-liner.

We also need to add the log-exposure column here, which is the exposure offset for the frequency GLM. We will explain exactly what that is in Part 4.

```python
import numpy as np

df_pd = df.to_pandas()
df_pd["log_exposure"] = np.log(df_pd["exposure"].clip(lower=1e-6))

print(f"Converted to pandas: {df_pd.shape[0]:,} rows, {df_pd.shape[1]} columns")
```

**What the `.clip(lower=1e-6)` does:** It replaces any exposure values below 0.000001 with 0.000001 before taking the log. `log(0)` is negative infinity, which will corrupt the GLM fitting. This clip prevents that. In our synthetic data, exposures are always at least 0.05, so the clip makes no difference here. On real data it protects you from data errors.

### Checking for missing values

statsmodels drops rows with missing values silently. If your real dataset has missing vehicle groups and Emblem treats them as "Unknown" while Python drops them, you are fitting two different models on different data. Always check before fitting.

```python
print("Missing value counts:")
print(df.null_count())
```

In our synthetic data, there should be no missing values. On real data, any non-zero count here needs a decision: impute, create an "Unknown" level, or drop and document.

---

## Part 4: Fitting the frequency GLM

### The exposure offset - the most important concept in this module

Before fitting, we need to understand what the exposure offset does and why it matters.

A Poisson GLM without an offset fits raw claim counts. A policy with 0.5 earned years should generate half as many expected claims as an identical policy with 1.0 earned years - not because it is lower risk, but because it was only exposed for half the time. Without the offset, the model cannot distinguish between "low risk" and "short exposure." It will learn the exposure duration as a spurious predictor of claim count, producing biased coefficients for everything else.

The offset fixes this by entering the linear predictor as a term with a fixed coefficient of exactly 1:

```
log(E[claims_i]) = log(exposure_i) + intercept + beta_area × area_i + ...
```

Rearranging: `E[claims_i] = exposure_i × exp(intercept + betas)`. The model is fitting the annualised rate, and the exposure scales the expected claims for each policy. A policy with 0.5 years contributes exactly half of a full-year policy's expected claims.

In statsmodels, the offset argument takes the log-exposure vector we computed above.

### The formula string

statsmodels uses a formula syntax (from the `patsy` library) to specify the model. It looks like R:

```
"claim_count ~ C(area) + C(ncd_years, Treatment(0)) + C(conviction_flag, Treatment(0)) + vehicle_group"
```

- `claim_count` is the response variable (left of the tilde)
- `C(area)` tells patsy to treat area as a categorical factor, creating dummy variables
- `C(ncd_years, Treatment(0))` treats NCD years as categorical with NCD=0 as the explicit base level
- `vehicle_group` (without `C()`) treats vehicle group as a continuous variable - one slope, not k-1 dummies

**Why not `C(vehicle_group)`?** Because with 50 ABI groups, using each as a separate dummy variable costs 49 degrees of freedom. Treating it as continuous costs 1 degree of freedom and gives a smooth, monotone effect if the relationship is approximately linear in logs. The synthetic data was generated with a linear vehicle group effect (0.018 per group), so this is the correct specification here. On real data, plot the one-way A/E first and decide.

### Fitting the model

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

freq_formula = (
    "claim_count ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_freq = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

print(glm_freq.summary())
```

Run this cell. It will take a few seconds (IRLS is iterative). When it completes, you will see a large summary table.

**What to check in the summary:**

- `Converged: True` - if this says False, the model did not reach a solution. Treat all results with suspicion.
- `Method: IRLS` - confirms we are using the same algorithm as Emblem
- `Deviance` - the total deviance. For Poisson, a well-specified model has deviance roughly equal to the residual degrees of freedom (here, around 99,988). If deviance is materially above df_resid, the model may be overdispersed.
- The coefficient table - columns are: `coef` (the log-relativity), `std err` (standard error), `z` (z-statistic), `P>|z|` (p-value), and 95% confidence interval

### Reading the coefficient table

The coefficient for area B will appear as `C(area)[T.B]`. The `[T.B]` suffix means "treatment contrast: level B relative to the reference level (A)." The coefficient is approximately 0.099, so the area B frequency relativity is `exp(0.099) ≈ 1.104`. The true parameter was 0.10, so `exp(0.10) = 1.105`. Our model has recovered it to within rounding error.

Check whether the model has converged correctly:

```python
print(f"Converged: {glm_freq.converged}")
print(f"Iterations: {glm_freq.nit}")
print(f"Deviance: {glm_freq.deviance:,.1f}")
print(f"Residual df: {glm_freq.df_resid:,.0f}")
print(f"Deviance/df: {glm_freq.deviance / glm_freq.df_resid:.3f}")

# Check for aliased (dropped) parameters
nan_params = glm_freq.params[glm_freq.params.isna()]
if len(nan_params) > 0:
    print(f"\nWARNING: {len(nan_params)} aliased parameters (NaN coefficients):")
    print(nan_params)
else:
    print("\nNo aliased parameters - design matrix is full rank")
```

You should see `Converged: True`, iterations in single digits, and a deviance/df ratio close to 1.0. A ratio above 1.3 indicates overdispersion - more on that in the diagnostics section.

**A new Python concept: dictionary.** `TRUE_PARAMS` earlier in the module is a dictionary - a collection of key-value pairs. You access a value with `TRUE_PARAMS["area_B"]`. Dictionaries are useful for storing named parameters like this, where the name (e.g. "area_B") makes the code self-documenting.

### Extracting relativities

The raw output from statsmodels is in log-space (the coefficient table) and uses patsy's naming convention. For a pricing actuary used to Emblem's factor tables, we want multiplicative relativities in a clean format. This function does the conversion:

```python
def extract_freq_relativities(glm_result, base_levels: dict) -> pl.DataFrame:
    """
    Extract multiplicative relativities from a fitted statsmodels GLM.

    Returns a Polars DataFrame with columns:
        feature, level, log_relativity, relativity, se, lower_ci, upper_ci

    base_levels: dict mapping each factor name to its base level value.
                 These get relativity=1.0, which is the definition of a base level.
    """
    records = []
    params = glm_result.params
    conf_int = glm_result.conf_int()

    for param_name, coef in params.items():
        if param_name == "Intercept":
            continue

        lo = conf_int.loc[param_name, 0]
        hi = conf_int.loc[param_name, 1]
        se = glm_result.bse[param_name]

        # Parse the patsy parameter name: "C(area)[T.B]" -> feature="area", level="B"
        if "[T." in param_name:
            feature_part = param_name.split("[T.")[0]
            level_part = param_name.split("[T.")[1].rstrip("]")
            if feature_part.startswith("C("):
                feature_part = feature_part[2:].split(",")[0].split(")")[0].strip()
        else:
            # Continuous feature - single coefficient, not per-level
            feature_part = param_name
            level_part = "continuous"

        records.append({
            "feature": feature_part,
            "level": level_part,
            "log_relativity": coef,
            "relativity": np.exp(coef),
            "se": se,
            "lower_ci": np.exp(lo),
            "upper_ci": np.exp(hi),
        })

    rels = pl.DataFrame(records)

    # Add a row for each base level (relativity = 1.0 by definition)
    base_rows = []
    for feat, base_level in base_levels.items():
        base_rows.append({
            "feature": feat,
            "level": str(base_level),
            "log_relativity": 0.0,
            "relativity": 1.0,
            "se": 0.0,
            "lower_ci": 1.0,
            "upper_ci": 1.0,
        })

    return pl.concat([pl.DataFrame(base_rows), rels]).sort(["feature", "level"])


freq_rels = extract_freq_relativities(
    glm_freq,
    base_levels={"area": "A", "ncd_years": "0", "conviction_flag": "0"},
)

# Look at area relativities
print(freq_rels.filter(pl.col("feature") == "area"))
```

**What you should see:**

```
shape: (6, 7)
┌─────────┬───────┬─────────────────┬────────────┬──────────┬────────────┬────────────┐
│ feature ┆ level ┆ log_relativity  ┆ relativity ┆ se       ┆ lower_ci   ┆ upper_ci   │
│ str     ┆ str   ┆ f64             ┆ f64        ┆ f64      ┆ f64        ┆ f64        │
╞═════════╪═══════╪═════════════════╪════════════╪══════════╪════════════╪════════════╡
│ area    ┆ A     ┆ 0.0             ┆ 1.0        ┆ 0.0      ┆ 1.0        ┆ 1.0        │
│ area    ┆ B     ┆ 0.099...        ┆ 1.104...   ┆ 0.024... ┆ 1.057...   ┆ 1.152...   │
│ area    ┆ C     ┆ 0.197...        ┆ 1.218...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ D     ┆ 0.348...        ┆ 1.417...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ E     ┆ 0.499...        ┆ 1.647...   ┆ ...      ┆ ...        ┆ ...        │
│ area    ┆ F     ┆ 0.648...        ┆ 1.912...   ┆ ...      ┆ ...        ┆ ...        │
└─────────┴───────┴─────────────────┴────────────┴──────────┴────────────┴────────────┘
```

Area F relativity: approximately 1.912. True value: `exp(0.65) = 1.916`. We are within 0.2%.

Now verify the NCD relativities against what we know the true values should be:

```python
# True NCD=5 vs NCD=0 relativity: exp(-0.13 × 5) = exp(-0.65) ≈ 0.522
# True conviction uplift: exp(0.42) ≈ 1.52

print("NCD relativities:")
print(freq_rels.filter(pl.col("feature") == "ncd_years"))

print("\nConviction relativity:")
print(freq_rels.filter(pl.col("feature") == "conviction_flag"))

true_ncd5 = np.exp(-0.13 * 5)
true_conviction = np.exp(0.42)
print(f"\nTrue NCD=5 relativity: {true_ncd5:.4f}")
print(f"True conviction relativity: {true_conviction:.4f}")
```

If the model is working correctly, the estimated relativities will be within a few percent of the true values. They will not be exact - a sample of 100,000 policies has sampling variation.

---

## Part 5: Handling sparse factor levels

Before we move to the severity model, a practical note on something that bites real projects.

Real claims extracts will have factor levels with very few policies: area codes that appear in the policy file but not in the training data, occupation codes with 2 claims, NCD levels with 12 policies. Emblem consolidates sparse levels automatically. Python estimates a separate coefficient for every level unless you intervene. The result is extremely wide confidence intervals (or outright NaN coefficients) for sparse levels.

Here is the pattern to use in Polars for grouping sparse levels by exposure:

```python
# Identify area levels with fewer than 50 earned years of exposure
area_exposure = (
    df
    .group_by("area")
    .agg(pl.col("exposure").sum().alias("total_exposure"))
)

sparse_areas = area_exposure.filter(pl.col("total_exposure") < 50)["area"].to_list()
print(f"Sparse area levels: {sparse_areas}")

# In our synthetic data, this should be empty - all areas are well-populated.
# On real data, group sparse levels into "Other":
if sparse_areas:
    df = df.with_columns(
        pl.when(pl.col("area").is_in(sparse_areas))
        .then(pl.lit("Other"))
        .otherwise(pl.col("area").cast(pl.Utf8))
        .alias("area")
    )
    print(f"Merged {len(sparse_areas)} sparse levels into 'Other'")
```

The threshold (50 earned years here) is a business judgement. A level with fewer than about 30-50 years of exposure will produce a relativity with such a wide confidence interval that it is essentially noise. Merge it with a generic "Other" bucket and document the consolidation in your model notes.

---

## Part 6: Fitting the severity GLM

### What is different about severity

Severity GLMs differ from frequency GLMs in three important ways:

1. **We fit on claimed policies only.** A policy with zero claims contributes nothing to the severity distribution. Fitting severity on all policies (including the zeros) would be mathematically wrong.

2. **The response variable is average severity per claim, not total incurred.** Total incurred mixes severity and claim count. We want to model the cost per event, not the cost per policy.

3. **We use claim count as a variance weight, not an exposure offset.** A policy with 3 claims gives us an average of 3 severities, which is more informative than an average of 1. The `var_weights` argument tells statsmodels to weight each observation's contribution by its claim count.

### NCD does not belong in the severity model

NCD years are excluded from the severity formula and this is a deliberate modelling decision, not an oversight.

NCD reflects driving behaviour and correlates with claim frequency - drivers with zero NCD have more accidents. But conditional on a claim occurring, the claim cost is not systematically different between NCD=0 and NCD=5 drivers. Including NCD in the severity model would capture frequency effects through the back door, double-counting the NCD signal. The frequency model picks it up correctly; the severity model should not.

This is the kind of thing Emblem users sometimes handle by trial and error. In Python, the decision is explicit in the formula string.

### Large loss truncation

Before fitting the severity GLM on real motor data, you need to decide what to do about large personal injury claims. Bodily injury claims on UK motor books are typically 10-100x the average accidental damage claim. An untruncated Gamma severity model will be driven by whichever risk characteristics correlate with the handful of catastrophic PI claims in your portfolio.

Standard practice is to cap large losses at £100k-£250k and model the excess separately, or to separate PI claims from property damage. For the synthetic data here we have no large PI exposure - the severity DGP is a simple Gamma with mean £3,500. On real data, always add this step before the severity GLM:

```python
LARGE_LOSS_THRESHOLD = 100_000  # £100k cap - adjust for your book

df_sev_all = df.with_columns(
    pl.col("incurred").clip(upper_bound=LARGE_LOSS_THRESHOLD * pl.col("claim_count"))
    .alias("incurred_capped")
)
n_capped = df_sev_all.filter(
    pl.col("incurred") > LARGE_LOSS_THRESHOLD * pl.col("claim_count")
).shape[0]
print(f"Policies capped at £{LARGE_LOSS_THRESHOLD/1000:.0f}k: {n_capped}")
```

Document the cap in your model log. It is a modelling assumption that materially affects severity relativities, and anyone trying to reproduce your results without knowing the cap will not match your numbers.

### Fitting the model

```python
# Severity data: claimed policies only
df_sev = df.filter(pl.col("claim_count") > 0)
df_sev = df_sev.with_columns(
    (pl.col("incurred") / pl.col("claim_count")).alias("avg_severity")
)

print(f"Severity model: {len(df_sev):,} claimed policies")
print(f"Mean average severity: £{df_sev['avg_severity'].mean():,.0f}")

df_sev_pd = df_sev.to_pandas()

sev_formula = (
    "avg_severity ~ "
    "C(area) + "
    "vehicle_group"
    # NCD deliberately excluded - see above
)

glm_sev = smf.glm(
    formula=sev_formula,
    data=df_sev_pd,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    var_weights=df_sev_pd["claim_count"],
).fit()

print(f"\nSeverity GLM:")
print(f"Converged: {glm_sev.converged}")
print(f"Gamma scale (phi): {glm_sev.scale:.4f}")
print(f"CV of severity: {np.sqrt(glm_sev.scale):.3f}")
```

**What to check:**

- `Converged: True` - as with the frequency model
- Gamma scale (phi): this is the dispersion parameter for the Gamma family. A coefficient of variation (CV = sqrt(phi)) of 0.5-0.8 is typical for motor accidental damage severity. If it is below 0.3, check whether your data has already been capped. If it is above 1.5, you may have a mixture of claim types (small property damage plus large PI) that would be better separated.

Extract severity relativities:

```python
sev_rels = extract_freq_relativities(
    glm_sev,
    base_levels={"area": "A"},
)

print("Severity area relativities:")
print(sev_rels.filter(pl.col("feature") == "area"))

# The true severity DGP has no area effect. The severity area relativities
# should all be close to 1.0 (within sampling noise).
print("\nIf the model is correct, all area relativities should be close to 1.0")
print("because area was not in the severity data-generating process.")
```

This is an important check. The synthetic data was generated with no area effect on severity. So the severity area relativities should be statistically indistinguishable from 1.0. If they are not, something is wrong with the model specification.

---

## Part 7: Diagnostics

Running a GLM without diagnostics is not modelling. These checks tell you whether the model is well-specified and where it fails.

### Deviance residuals for the frequency model

The deviance residual for each observation measures how far the observed claim count is from the model's prediction, on a scale that accounts for the Poisson distribution.

```python
import matplotlib.pyplot as plt
from scipy import stats

resid_deviance = glm_freq.resid_deviance
resid_std = resid_deviance / np.sqrt(glm_freq.scale)
fitted_vals = glm_freq.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs fitted
axes[0].scatter(np.log(fitted_vals), resid_std, alpha=0.1, s=5, color="steelblue")
axes[0].axhline(0, color="black", linestyle="--", lw=1)
axes[0].axhline(2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].axhline(-2, color="red", linestyle="--", lw=1, alpha=0.5)
axes[0].set_xlabel("log(fitted frequency)")
axes[0].set_ylabel("Deviance residual")
axes[0].set_title("Residuals vs Fitted - Frequency GLM")

# Normal QQ plot
stats.probplot(resid_std, dist="norm", plot=axes[1])
axes[1].set_title("Normal QQ - Deviance Residuals")

plt.tight_layout()
plt.show()
```

**What to look for:**

- Residuals should show no strong pattern against fitted values. A funnel shape (residuals increasing with fitted values) suggests overdispersion.
- More than 5% of residuals outside the red ±2 lines suggests either genuine overdispersion or a systematic missing feature.
- The QQ plot for Poisson GLM deviance residuals will not be perfectly normal (the Poisson is discrete), but the upper tail should not be dramatically heavier than normal.

### Actual vs Expected by factor level

This is the diagnostic Emblem shows in its factor charts, and it is the single most useful check for a pricing model. For each level of each rating factor, you compute the ratio of observed claims to predicted claims. A well-specified model should have A/E close to 1.0 for all levels.

```python
def ae_by_factor(
    df: pl.DataFrame,
    fitted_values: np.ndarray,
    feature: str,
) -> pl.DataFrame:
    """
    Compute actual vs expected claim counts by factor level.
    """
    return (
        df
        .with_columns(
            pl.Series("expected_claims", fitted_values)
        )
        .group_by(feature)
        .agg([
            pl.col("claim_count").sum().alias("actual_claims"),
            pl.col("expected_claims").sum().alias("expected_claims"),
            pl.col("exposure").sum().alias("exposure"),
        ])
        .with_columns(
            (pl.col("actual_claims") / pl.col("expected_claims")).alias("ae_ratio")
        )
        .sort(feature)
    )


ae_area = ae_by_factor(df, glm_freq.fittedvalues, "area")
print("A/E by area:")
print(ae_area)
```

A/E ratios close to 1.0 for area are expected - area is in the model, so the model is calibrated to it. The more important check is for **factors not in the model**. If you omit driver age and the A/E for young drivers is consistently above 1.0, the model is materially underpricing that group. Add age to the model.

```python
# Check A/E by driver age bands - driver age IS in the model (as young/old flags),
# so we should see good A/E here. Try removing it and see what happens.
df_diag = df.with_columns(
    pl.when(pl.col("driver_age") < 25).then(pl.lit("17-24"))
    .when(pl.col("driver_age") < 35).then(pl.lit("25-34"))
    .when(pl.col("driver_age") < 50).then(pl.lit("35-49"))
    .when(pl.col("driver_age") < 65).then(pl.lit("50-64"))
    .otherwise(pl.lit("65+"))
    .alias("age_band")
)

ae_age = ae_by_factor(df_diag, glm_freq.fittedvalues, "age_band")
print("\nA/E by age band:")
print(ae_age)
```

**A new Python concept: `pl.when().then().otherwise()`.** This is Polars' equivalent of a nested IF statement. `pl.when(condition).then(value).when(condition2).then(value2).otherwise(fallback)` evaluates each condition in order and returns the corresponding value. It runs across all 100,000 rows simultaneously.

### Overdispersion

For real UK motor data, the Poisson model will almost always be overdispersed - the deviance will be materially above the residual degrees of freedom. A deviance/df ratio above 1.3 is common; ratios above 2.0 are not unusual on books with bodily injury cover.

When this happens, you have two options:

**Quasi-Poisson:** same point estimates as Poisson, but standard errors are inflated to account for overdispersion. The relativities themselves do not change - only the confidence intervals widen. Use this when you want conservative confidence intervals but do not want to change the relativities.

```python
glm_freq_quasi = smf.glm(
    formula=freq_formula,
    data=df_pd,
    family=sm.families.quasi.Quasipoisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

print(f"Quasi-Poisson dispersion estimate: {glm_freq_quasi.scale:.3f}")
print("(If > 1.0, the data is overdispersed relative to Poisson)")
```

**Negative Binomial:** models overdispersion explicitly with an additional dispersion parameter. The coefficients will differ slightly from Poisson. More appropriate when you expect genuine extra-Poisson variation in the data-generating process.

For a first migration from Emblem (which uses Poisson), quasi-Poisson is the lower-risk option. Relativities are identical to Poisson; only the standard errors change.

---

## Part 8: Validating against Emblem output

### The critical check

The following function compares your Python GLM relativities to an Emblem CSV export. The Emblem CSV format is typically:

```
Factor,Level,Relativity,SE,LowerCI,UpperCI
```

```python
def compare_to_emblem(
    python_rels: pl.DataFrame,
    emblem_path: str,
    tolerance: float = 0.001,
) -> pl.DataFrame:
    """
    Compare Python GLM relativities to an Emblem CSV export.
    Returns a comparison DataFrame with a 'match' column.
    """
    emblem_rels = pl.read_csv(
        emblem_path,
        schema_overrides={"Level": pl.Utf8, "Relativity": pl.Float64},
    )

    comparison = (
        python_rels
        .rename({"feature": "Factor", "level": "Level", "relativity": "Python_Rel"})
        .join(
            emblem_rels.rename({"Relativity": "Emblem_Rel"}).select(["Factor", "Level", "Emblem_Rel"]),
            on=["Factor", "Level"],
            how="inner",
        )
        .with_columns([
            ((pl.col("Python_Rel") - pl.col("Emblem_Rel")).abs()).alias("abs_diff"),
            ((pl.col("Python_Rel") / pl.col("Emblem_Rel") - 1).abs()).alias("rel_diff"),
        ])
        .with_columns(
            (pl.col("rel_diff") < tolerance).alias("match")
        )
        .sort(["Factor", "Level"])
    )

    n_matched = comparison["match"].sum()
    n_total = len(comparison)
    print(f"Matched: {n_matched}/{n_total} relativities within {tolerance*100:.1f}% tolerance")

    if n_matched < n_total:
        mismatches = comparison.filter(~pl.col("match"))
        print("\nMismatches:")
        print(mismatches.select(["Factor", "Level", "Python_Rel", "Emblem_Rel", "rel_diff"]))

    return comparison
```

### Tolerances to accept

- **Identical data, identical specification:** relativities should match to 4+ decimal places. If they do not, there is a specification mismatch.
- **Same data, minor specification differences** (e.g. Emblem rounds vehicle group to bands, Python uses continuous): expect differences that reflect the specification difference. Document them.
- **Different data vintage:** some differences are expected from the additional data. Verify the sign and approximate magnitude are consistent, but do not try to match exactly.

### Common reasons for mismatches

**Base level differs.** Emblem defaults to the highest-exposure level as base; Python defaults to alphabetical first. If you have not pinned the base level explicitly, every relativity for that factor will be off by a constant multiplier - all quotients `Python/Emblem` for that factor will be the same constant.

**Continuous vs categorical encoding.** Emblem often auto-detects whether a factor should be treated as continuous or categorical. Python's formula interface requires you to be explicit. If Emblem fit vehicle group as categorical (27 dummies for groups 1-27, 28-50 collapsed) and your Python model treats it as continuous, the relativities will not match.

**Missing level consolidation.** Emblem consolidates sparse levels automatically. Python estimates a separate coefficient for every level unless you merge them manually.

**Emblem manual overrides.** If the validation fails despite identical data and identical specification, the most likely explanation is a manual override in Emblem that was never documented. Someone clicked on a factor level in the Emblem UI and typed in a relativity. It happens constantly on live books.

Check whether any Emblem relativities are suspiciously round numbers - 1.000, 0.850, 1.250. A likelihood-based GLM will almost never produce a relativity of exactly 0.850 to three decimal places. If you see round-number relativities, talk to whoever built the original Emblem model before concluding there is a coding error in your Python model.

---

## Part 9: Exporting to Radar

Willis Towers Watson Radar imports external relativity tables as CSVs. The format is:

```
Factor,Level,Relativity
area,A,1.0000
area,B,1.1041
area,C,1.2185
```

The factor name must match the rating variable name in the Radar model exactly. Radar is case-sensitive. If your Python column is `area` and your Radar variable is `Area`, the import will fail silently - Radar will apply a relativity of 1.0 for all levels. Check the exact names in your Radar project before export.

```python
def to_radar_csv(
    rels: pl.DataFrame,
    output_path: str,
    factor_name_map: dict | None = None,
    decimal_places: int = 4,
) -> None:
    """
    Export relativities in Radar factor table import format.

    factor_name_map: dict mapping Python column names to Radar variable names.
                     e.g. {"area": "PostcodeArea", "ncd_years": "NCDYears"}
    """
    radar_df = rels.select(["feature", "level", "relativity"])

    if factor_name_map:
        radar_df = radar_df.with_columns(
            pl.col("feature").replace(factor_name_map).alias("feature")
        )

    radar_df = (
        radar_df
        .rename({"feature": "Factor", "level": "Level", "relativity": "Relativity"})
        .with_columns(
            pl.col("Relativity").round(decimal_places).alias("Relativity")
        )
    )

    radar_df.write_csv(output_path)
    print(f"Exported {len(radar_df)} factor table rows to {output_path}")


# Export frequency relativities for Radar
# Only export categorical factors - continuous features need banding first
cat_rels = freq_rels.filter(pl.col("level") != "continuous")

# On Databricks, /tmp/ is a local path accessible from the notebook.
# For production, use /dbfs/mnt/pricing/outputs/ or equivalent.
to_radar_csv(
    cat_rels,
    "/tmp/freq_relativities_radar.csv",
    factor_name_map={
        "area": "PostcodeArea",
        "ncd_years": "NCDYears",
        "conviction_flag": "ConvictionFlag",
    },
)
```

One practical issue: Radar requires every level that exists in the policy file to appear in the import table. If your rating variable has NCD=6 (some insurers allow it for advanced drivers) but your Python model merged NCD=5 and NCD=6 into a single level, you need to add a NCD=6 row to the export. Decide the relativity - usually the NCD=5 relativity, or a mild discount beyond it - and add it explicitly. Do not let this happen quietly; document it in your model notes.

---

## Part 10: Running on Databricks - the full production pattern

### Loading data from Delta tables

Rather than CSV exports, the production pattern loads data from Delta tables registered in Unity Catalog. This gives you time travel (query the data as it was at model-fit time) and full lineage tracking.

```python
# On Databricks, spark is already available - no import needed
# spark = SparkSession.builder.getOrCreate()  # Only needed outside Databricks

# Load current policy data
df_spark = spark.table("main.pricing.motor_policies")

# Convert to Polars for manipulation
df = pl.from_pandas(df_spark.toPandas())
```

To use the data as it was on a specific date:

```python
df_spark = spark.sql(
    "SELECT * FROM main.pricing.motor_policies TIMESTAMP AS OF '2024-03-15T00:00:00'"
)
```

Record the table version number when fitting - this is how you prove to a regulator what data you used:

```python
table_version = spark.sql(
    "DESCRIBE HISTORY main.pricing.motor_policies LIMIT 1"
).first()["version"]

print(f"Training data version: {table_version}")
```

### Logging to MLflow

MLflow is Databricks' experiment tracking system. It stores parameters, metrics, and artefacts from each model run, giving you a queryable history of every model you have fitted.

```python
import mlflow
from datetime import date

mlflow.set_experiment("/pricing/motor-glm")

with mlflow.start_run(run_name="freq_glm_v2") as run:
    # Log parameters - everything that defines the model
    mlflow.log_params({
        "model_type": "Poisson_GLM",
        "formula": freq_formula,
        "n_policies": len(df),
        "training_data_version": table_version,
        "training_date": str(date.today()),
        "base_levels": str({"area": "A", "ncd_years": "0", "conviction_flag": "0"}),
    })

    # Log metrics - numbers that describe model performance
    mlflow.log_metrics({
        "deviance": glm_freq.deviance,
        "null_deviance": glm_freq.null_deviance,
        "pseudo_r2": 1 - (glm_freq.deviance / glm_freq.null_deviance),
        "aic": glm_freq.aic,
        "n_params": len(glm_freq.params),
        "converged": int(glm_freq.converged),
        "n_iterations": glm_freq.nit,
    })

    # Log the relativities as a CSV artefact
    rels_path = "/tmp/freq_relativities.csv"
    freq_rels.write_csv(rels_path)
    mlflow.log_artifact(rels_path, artifact_path="factor_tables")

    # Log the model summary as a text artefact
    summary_path = "/tmp/glm_freq_summary.txt"
    with open(summary_path, "w") as f:
        f.write(str(glm_freq.summary()))
    mlflow.log_artifact(summary_path, artifact_path="diagnostics")

    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")
```

### Writing factor tables to Unity Catalog

```python
from datetime import date

rels_with_meta = freq_rels.with_columns([
    pl.lit(str(date.today())).alias("model_run_date"),
    pl.lit("freq_glm_v2").alias("model_name"),
    pl.lit(run_id).alias("mlflow_run_id"),
    pl.lit(table_version).alias("training_data_version"),
    pl.lit(len(df)).alias("n_policies_trained"),
])

spark.createDataFrame(rels_with_meta.to_pandas()).write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("main.pricing.glm_relativities")

print(f"Written {len(rels_with_meta)} rows to main.pricing.glm_relativities")
```

Using `mode("append")` means every model run adds to the history. You can query how any factor's relativity has changed across model cycles:

```python
area_f_history = spark.sql("""
    SELECT model_run_date, model_name, relativity
    FROM main.pricing.glm_relativities
    WHERE feature = 'area' AND level = 'F'
    ORDER BY model_run_date
""")
display(area_f_history)
```

This is the audit trail that both PS 21/5 and Consumer Duty require.

---

## Part 11: The Tweedie model (alternative approach)

The frequency-severity split (Poisson frequency times Gamma severity) is the standard UK personal lines approach. An alternative is the Tweedie pure premium model, which models the total incurred amount directly using a compound Poisson-Gamma distribution.

The Tweedie fits one model, not two. Its power parameter `p` controls the compound distribution: `p=1` is Poisson, `p=2` is Gamma, and `1 < p < 2` is the compound Poisson-Gamma relevant for insurance.

```python
# Compute pure premium in Polars before converting to pandas
df_pp = df.with_columns(
    (pl.col("incurred") / pl.col("exposure")).alias("pure_premium")
)
df_pp_pd = df_pp.to_pandas()
df_pp_pd["log_exposure"] = np.log(df_pp_pd["exposure"].clip(lower=1e-6))

pp_formula = (
    "pure_premium ~ "
    "C(area) + "
    "C(ncd_years, Treatment(0)) + "
    "C(conviction_flag, Treatment(0)) + "
    "vehicle_group"
)

glm_tweedie = smf.glm(
    formula=pp_formula,
    data=df_pp_pd,
    family=sm.families.Tweedie(
        var_power=1.5,
        link=sm.families.links.Log(),
    ),
    offset=df_pp_pd["log_exposure"],
).fit()

print(f"Tweedie GLM deviance: {glm_tweedie.deviance:,.1f}")
print(f"Pseudo R2: {1 - glm_tweedie.deviance/glm_tweedie.null_deviance:.4f}")
```

**When to use Tweedie vs frequency-severity split:**

**Tweedie** is simpler - one model, not two - and handles the zero-inflated pure premium directly. It is appropriate when you only want a pure premium prediction and are not trying to understand whether a risk is high frequency or high severity.

**Frequency-severity split** gives you more diagnostic power. You can see whether your area F uplift is driven by frequency (more accidents) or severity (more expensive accidents). That distinction matters: high-frequency/low-severity areas have a different risk management profile from low-frequency/high-severity areas, and reinsurance structuring is designed around that distinction.

The Tweedie also requires you to commit to a value of `p`. When the regulator asks why you chose `p=1.5`, you need an answer. The frequency-severity split has a cleaner actuarial justification: claims arrive as a Poisson process and each claim has a Gamma-distributed cost.

We recommend the frequency-severity split for production personal lines pricing.

---

## Part 12: Gotchas when moving from Emblem

We have worked with several teams on this migration. These are the problems that actually bite people.

**Emblem's automatic base level selection.** Emblem picks the most credible level as the base - usually the highest-exposure level. statsmodels picks alphabetically. Forgetting to align these is the number one source of "why don't the numbers match."

**Emblem's missing value handling.** Emblem treats missing values as a separate level, "Unknown," and estimates a relativity for it. statsmodels drops rows with missing values unless you handle them explicitly. If 3% of your policies have missing vehicle group and Emblem is pricing them as "Unknown" while Python is dropping them, your models are fit on different data.

```python
# Check for missing values before fitting
missing_report = df.null_count()
print(missing_report)

# Options:
# 1. Impute with the mean or mode for continuous variables
df = df.with_columns(
    pl.col("vehicle_group").fill_null(strategy="mean").cast(pl.Int32)
)

# 2. Create an "Unknown" level for categorical factors
df = df.with_columns(
    pl.col("area").fill_null("Unknown")
)
```

**Credibility-weighted relativities.** Emblem has a credibility option that shrinks sparse level relativities toward 1.0. By default in statsmodels, you get maximum likelihood estimates with no shrinkage. If Emblem's published relativities show NCD=6 at 1.000 while your Python estimate is 0.78 with a wide CI, Emblem may have applied credibility weighting.

**The deviance statistic and likelihood ratio tests.** Emblem reports "change in scaled deviance" when you add a factor. statsmodels reports total deviance. To get the chi-squared test for adding a factor, compute the deviance difference manually:

```python
# Fit base model and extended model
glm_base = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0))",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

glm_extended = smf.glm(
    "claim_count ~ C(area) + C(ncd_years, Treatment(0)) + vehicle_group",
    data=df_pd,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=df_pd["log_exposure"],
).fit()

# Likelihood ratio test for adding vehicle_group
from scipy import stats as scipy_stats

lr_stat = glm_base.deviance - glm_extended.deviance
df_diff = glm_base.df_resid - glm_extended.df_resid
p_value = scipy_stats.chi2.sf(lr_stat, df_diff)
print(f"LR chi-squared: {lr_stat:.2f}, df: {df_diff}, p-value: {p_value:.4f}")
```

---

## Summary

The Python GLM workflow produces output that is numerically consistent with Emblem when given the same data and the same specification. On synthetic data without manual overrides, the relativities match to four decimal places. On real Emblem models, validate any overrides explicitly before declaring a match.

The difference between Emblem and Python is not in the model. It is in the surrounding infrastructure: version control, reproducibility, auditability, and integration with the rest of the modelling stack.

**The critical steps, in order:**

1. Get the exposure right. Use earned exposure, clip at a small positive number, filter out zeros.
2. Match the base levels to Emblem's explicitly. Do not rely on defaults.
3. Handle missing values deliberately - decide between dropping, imputing, or creating an "Unknown" level.
4. Truncate large losses before the severity GLM. Document the cap.
5. Exclude NCD from the severity formula - it is a frequency signal, not a severity driver.
6. Check for aliased parameters and non-convergence before trusting results.
7. Run A/E diagnostics for factors not in the model as well as those that are.
8. Check deviance/df - if materially above 1, consider quasi-Poisson or negative binomial.
9. Validate against Emblem's published relativities on matched data, accounting for any manual overrides.
10. Log everything to MLflow and Unity Catalog before exporting to Radar.

The model is the easy part.

---

## What's next

**Module 3: Gradient Boosted Models with CatBoost** - replaces the GLM frequency model with a CatBoost model. Covers hyperparameter tuning, cross-validation designed for insurance data, and model comparison against the GLM benchmark from this module.

**Module 4: Validation and Monitoring** - builds the monitoring infrastructure to track model performance month by month, detect drift, and generate the FCA evidence pack automatically.
