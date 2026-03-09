# Module 6: Credibility and Bayesian Methods — The Thin-Cell Problem

In Module 5 you built monitoring dashboards and learned how to detect when a production model drifts. You know how to tell when a model is going wrong. This module is about a problem that exists before the model is even fitted: what do you do when you do not have enough data to trust the observed rate?

This is the thin-cell problem. It is not a niche concern. Every UK personal lines pricer working with postcode districts, vehicle groups, or affinity schemes encounters it every week. By the end of this module you will know the two principled tools for handling it — Bühlmann-Straub credibility and Bayesian hierarchical models — and you will have implemented both from scratch in a Databricks notebook.

We also make a promise: this module will not leave you stuck at any step. If you have never used PyMC, never set up a Python environment for sampling, and never heard of MCMC diagnostics, that is fine. We explain everything from first principles.

---

## Part 1: The problem you are solving

### The postcode district scenario

You are pricing a UK motor book. Your data team has given you claims experience by postcode district for the last five accident years. You have 2,300 postcode districts in the data. You are reviewing the rating for KT (Kingston upon Thames).

KT has 847 policy-years of exposure and 11 claims over five years. The observed claim frequency is:

```
11 / 847 = 1.30%
```

The portfolio mean for risks with similar rating factor profiles is 6.8%.

Should KT's rating factor be based on 1.30%? On 6.8%? Something in between?

The answer to "something in between" is not arbitrary. It depends on two things:

1. **How variable is a district's true risk from year to year?** If KT's true risk fluctuates a lot year to year (high within-district variance), then five years of data with 11 claims is not very informative — a run of good luck is plausible. We should put less weight on KT's own experience.

2. **How different are districts from each other genuinely?** If UK postcode districts have genuinely different risk profiles (high between-district variance), then KT probably is different from the portfolio mean, even if we cannot pin down exactly how different. We should put more weight on KT's own experience.

Bühlmann-Straub credibility gives you the mathematically principled formula that answers this question. Bayesian hierarchical models give you the same answer with richer uncertainty quantification. This module covers both.

### Why GLMs and GBMs do not solve it

You might wonder: doesn't the area factor in your GLM handle geographic risk? Partly. But:

A main-effects GLM assigns a single area coefficient to all districts in a postcode area (SW, KT, EC, etc.). Districts within an area share the coefficient — there is no district-level differentiation. If you fit a district-level factor in a GLM, KT gets a coefficient estimated from 847 policies and 11 claims. The standard error on that coefficient is enormous. The GLM will either return an implausible rate or, with regularisation, shrink KT's coefficient to zero — which is also wrong.

Ridge regularisation shrinks every coefficient toward zero, regardless of the cell's exposure. A district with 50,000 policy-years gets the same shrinkage formula as a district with 50. That is not right: the dense district deserves its own experience; the thin district deserves pooling.

LASSO is worse. It forces thin cells to exactly zero, which means the district gets the portfolio base rate. That is not credibility — it is arbitrary censorship of real geographic variation.

GBMs have `min_data_in_leaf`. Set it too low: the model learns noise in KT as signal. Set it too high: KT vanishes into the portfolio average. Neither is principled.

**Credibility theory answers the question directly.** Each cell gets a blend of its own experience and the portfolio mean, in proportion to how much information its experience actually contains. The blend is not a tuning parameter — it is derived from the data's evidence about the heterogeneity of the portfolio.

### Scale of the problem

Great Britain has approximately 2,800 postcode districts. A UK motor book with 1.5 million policies averages around 535 policies per district. That sounds comfortable, but motor books are not uniformly distributed:

- Inner London districts (SW1, EC1, W1) each contain several thousand policies
- Rural Scottish districts (KW, IV, HS) may have under 50 policies
- Thin cells are a structural feature of personal lines rating, not an edge case

Even with 535 policies average, the variance is enormous. Any district with fewer than 200 policy-years of experience — roughly 40% of UK districts in a typical mid-size motor book — needs credibility treatment.

---

## Part 2: Setting up your Databricks notebook

### Creating the notebook

If Databricks is already open from Module 5, you can create a new notebook. If you are starting fresh, go to your Databricks workspace URL and log in.

In the left sidebar, click the **Workspace** icon (it looks like a folder). Navigate to a folder you have write access to — usually your personal folder under `/Users/your.email@company.com/`. Right-click (or click the three dots next to the folder name) and select **Create > Notebook**. Name it `module-06-credibility-bayesian`. Leave the default language as Python.

The notebook opens in the editor. You will see an empty cell with a grey triangle (run button) on the left.

### Attaching a cluster

If your cluster from Module 5 is still running, click **Connect** in the top right and select it. If it has terminated (clusters auto-terminate after inactivity), start a new cluster: click **Connect > Create new cluster** (or go to the Compute section in the left sidebar). Use the default settings — a single-node cluster with the ML runtime (DBR 15.x) is sufficient for this module.

Databricks Free Edition (Free Edition) clusters have one driver node with no workers. That is fine for everything in this module. MCMC runs on the driver node.

### Installing PyMC

PyMC is not installed on the default Databricks ML runtime. Install it in the first cell of your notebook. Click in the first cell and type:

```python
%pip install pymc arviz --quiet
dbutils.library.restartPython()
```

**What this does:** `%pip install` installs packages into the running Python environment on the cluster. `--quiet` suppresses the progress output so the cell is less noisy. `dbutils.library.restartPython()` restarts the Python interpreter after installation — this is required because Python does not automatically make newly installed packages available in the current session.

**Run this cell** by pressing Shift+Enter or clicking the run button (triangle). This takes about 60-90 seconds. The cell will show the pip installation output and then a message saying the Python kernel has restarted.

**What you should see after it finishes:** The notebook will show a message like "Python interpreter restarted." All your cells after this one will work with PyMC available.

**Important:** After `dbutils.library.restartPython()` runs, the Python kernel resets. Any variables you defined in earlier cells are gone. This is why the installation cell must always be the first cell, and you must run the remaining cells after it completes.

### Importing libraries

Create a new cell (click the + icon below the first cell or press B to add a cell below). Paste in:

```python
import numpy as np
import polars as pl
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("NumPy:", np.__version__)
print("Polars:", pl.__version__)
print("PyMC:", pm.__version__)
print("ArviZ:", az.__version__)
```

**What this does:** Imports the four main libraries for this module. NumPy handles numerical arrays. Polars handles DataFrames (faster than pandas for the data manipulation we need). PyMC is the probabilistic programming library for Bayesian models. ArviZ provides MCMC diagnostics and plotting.

**Run this cell.** It takes about 30-60 seconds on first run because PyMC compiles PyTensor computation graphs on import.

**What you should see:**
```
NumPy: 1.26.x
Polars: 0.20.x
PyMC: 5.x.x
ArviZ: 0.18.x
```

The exact version numbers will differ depending on when you installed. What matters is that PyMC shows version 5.x — this module uses PyMC 5 syntax, which differs from PyMC 3 in a few places. If you see version 3.x, run `%pip install "pymc>=5" --quiet` and restart again.

---

## Part 3: Generating synthetic motor data

We use synthetic data so we know the true underlying risk — this lets us verify that credibility estimation is working correctly. Real data validation is covered in Exercise 1.

### The data generating process

Create a new cell with a markdown header:

```python
%md
## Part 3: Synthetic motor portfolio — postcode districts
```

Create the next cell:

```python
rng = np.random.default_rng(seed=42)

# Portfolio parameters
N_DISTRICTS = 120          # 120 UK postcode districts
N_YEARS = 5                # accident years 2019-2023
PORTFOLIO_FREQUENCY = 0.07 # 7% mean claim frequency (motor)

# True between-district heterogeneity on log scale
# This is the ground truth sigma_district - we will try to recover it
TRUE_SIGMA_DISTRICT = 0.35

# True district-level log-rate deviations from the portfolio mean
# These are the "true" risk levels we are trying to estimate
true_log_rates = rng.normal(0, TRUE_SIGMA_DISTRICT, size=N_DISTRICTS)

# Postcode district names: mix of real UK formats
prefixes = ["SW", "SE", "N", "E", "W", "EC", "WC", "KT", "SM", "CR",
            "BR", "DA", "RM", "EN", "HA", "UB", "TW", "IG", "WD", "SL",
            "GU", "RH", "TN", "ME", "CT", "BN", "PO", "SO", "SP", "RG",
            "OX", "MK", "NN", "LE", "PE", "CB", "IP", "NR", "CO", "CM",
            "SS", "RM", "AL", "LU", "SG", "HP", "SN", "BA", "BS", "GL",
            "HR", "WR", "B", "CV", "DE", "NG", "LN", "HU", "DN", "S",
            "HD", "HX", "BD", "LS", "WF", "WA", "SK", "ST", "WV", "DY",
            "TF", "SY", "LL", "SA", "CF", "NP", "LD", "SY", "LA", "FY",
            "PR", "BB", "OL", "M", "BL", "WN", "L", "CH", "CW", "CA",
            "DL", "HG", "YO", "TS", "NE", "DH", "SR", "TD", "EH", "G",
            "PA", "KA", "FK", "KY", "DD", "PH", "AB", "IV", "KW", "HS"]

# Ensure we have exactly N_DISTRICTS names
district_names = []
for i, prefix in enumerate(prefixes[:N_DISTRICTS]):
    district_names.append(f"{prefix}{i+1}")

print(f"Generated {len(district_names)} district names")
print(f"First 10: {district_names[:10]}")
```

**What this does:** Sets up the ground-truth parameters for the simulation. `TRUE_SIGMA_DISTRICT = 0.35` means the true between-district log-rate standard deviation is 0.35. On the claim rate scale, this means districts range from roughly `exp(-0.35) = 0.70x` to `exp(+0.35) = 1.42x` the portfolio mean at ±1 SD. A significant amount of genuine geographic heterogeneity — typical for UK motor.

**Run this cell.**

**What you should see:** A count of 120 district names and the first 10. The district names look like real UK postcode prefixes with a number appended.

Now create the data for each district and year:

```python
# Generate exposures: highly skewed — inner city districts are dense,
# rural districts are thin. Matches the real distribution of UK motor books.
base_exposures = rng.lognormal(mean=5.5, sigma=1.2, size=N_DISTRICTS)
base_exposures = np.clip(base_exposures, 20, 5000)

# Create a DataFrame with one row per (district, accident_year)
rows = []
for i, district in enumerate(district_names):
    for year in range(2019, 2019 + N_YEARS):
        # Exposure varies slightly by year (lapse and new business fluctuation)
        exposure_this_year = base_exposures[i] * rng.uniform(0.85, 1.15)

        # True log-rate: portfolio base + district effect (stable over time)
        # Real insurance: districts are also affected by portfolio-wide trends,
        # but for simplicity we hold the district effect fixed
        true_log_rate = np.log(PORTFOLIO_FREQUENCY) + true_log_rates[i]
        true_rate = np.exp(true_log_rate)

        # Observed claims: Poisson given true rate and exposure
        observed_claims = rng.poisson(true_rate * exposure_this_year)

        rows.append({
            "postcode_district": district,
            "accident_year": year,
            "earned_years": float(exposure_this_year),
            "claim_count": int(observed_claims),
            "true_rate": float(true_rate),
            "true_log_deviation": float(true_log_rates[i]),
        })

df = pl.DataFrame(rows)

# Compute observed claim frequency per row
df = df.with_columns(
    (pl.col("claim_count") / pl.col("earned_years")).alias("claim_frequency")
)

print(f"Dataset dimensions: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))
```

**What this does:** Generates a panel dataset: 120 districts × 5 years = 600 rows. Each row has the number of claims observed and the earned exposure (policy-years) in that district-year combination. The `true_rate` column contains the ground truth — what the credibility estimator should try to recover. We keep this column in the DataFrame for validation; in real data, of course, you would not have it.

**Run this cell.**

**What you should see:** `Dataset dimensions: (600, 7)` and a table showing district names, accident years, earned years (anywhere from ~20 to ~5,000), claim counts, and the true rate. Look at the `earned_years` column — you should see a large spread. Some districts have hundreds of policy-years per year; others have 20-30. This is realistic.

Now compute district-level aggregate statistics, which are what the credibility estimator uses:

```python
# Aggregate to district level (summing across years)
dist_totals = (
    df
    .group_by("postcode_district")
    .agg([
        pl.col("earned_years").sum().alias("total_earned_years"),
        pl.col("claim_count").sum().alias("total_claims"),
        pl.col("true_rate").mean().alias("true_rate"),        # the ground truth
        pl.col("true_log_deviation").mean().alias("true_log_deviation"),
    ])
    .with_columns([
        (pl.col("total_claims") / pl.col("total_earned_years")).alias("observed_frequency"),
    ])
    .sort("postcode_district")
)

print(f"Number of districts: {dist_totals.height}")
print()
print("Exposure distribution across districts:")
print(dist_totals["total_earned_years"].describe())
print()
print("Top 5 thinnest districts (fewest policy-years):")
print(dist_totals.sort("total_earned_years").head(5))
print()
print("Top 5 densest districts (most policy-years):")
print(dist_totals.sort("total_earned_years", descending=True).head(5))
```

**What this does:** Creates a district-level summary with the total exposure and total claims across all five years. This is the starting point for Bühlmann-Straub credibility — you need group-level totals and the within-group year-by-year variation.

**Run this cell.**

**What you should see:** 120 districts. The exposure distribution will be highly right-skewed — a few dense districts with 5,000-25,000 earned years across 5 years, and many thin districts with 100-500 earned years. The thinnest districts will have observed frequencies that look implausible — a district with 20 claims across 5 years of 50 policy-years each will have an observed frequency that is highly volatile.

### Checkpoint 1: Data check

Before proceeding, verify your dataset is sensible:

```python
# Checkpoint 1: Basic sanity checks on the data
total_policies = dist_totals["total_earned_years"].sum()
total_claims = dist_totals["total_claims"].sum()
portfolio_frequency = total_claims / total_policies

print("=== CHECKPOINT 1: DATA SANITY ===")
print(f"Total earned years (portfolio): {total_policies:,.0f}")
print(f"Total claims:                   {total_claims:,}")
print(f"Portfolio claim frequency:       {portfolio_frequency:.4f}  ({portfolio_frequency*100:.2f}%)")
print()
print(f"Expected portfolio frequency:    {PORTFOLIO_FREQUENCY:.4f}  ({PORTFOLIO_FREQUENCY*100:.2f}%)")
print()

# Check for problematic rows
zero_exposure = dist_totals.filter(pl.col("total_earned_years") == 0).height
zero_claims = dist_totals.filter(pl.col("total_claims") == 0).height
print(f"Districts with zero exposure:  {zero_exposure}  (should be 0)")
print(f"Districts with zero claims:    {zero_claims}  (may be non-zero for thin districts)")
print()

if abs(portfolio_frequency - PORTFOLIO_FREQUENCY) / PORTFOLIO_FREQUENCY < 0.05:
    print("Portfolio frequency is within 5% of target. Proceeding.")
else:
    print("WARNING: Portfolio frequency deviates by more than 5%. Check data generation.")
```

**What you should see:** The portfolio frequency close to 7% (within 5%). Zero rows with zero exposure. Possibly a handful of districts with zero claims (these are the very thinnest districts that happened to have no claims in five years — they are real and legitimate).

If you see warnings here, re-read the data generation cell above and run it again. The random seed is fixed at 42, so results should be reproducible.

---

## Part 4: Bühlmann-Straub credibility — the maths and the code

### Why this section matters

Bühlmann-Straub (1970) is the workhorse credibility method in European non-life insurance. It appears in the Swiss Solvency Test standard formula, in IFoA working papers, and in the actuarial standards of the Netherlands, Germany, and Switzerland. It is not exotic — it is the established method.

UK actuaries who trained on Emblem or Radar have seen GLM coefficients with standard errors. Bühlmann-Straub is giving you something conceptually similar: a parameter estimate (the credibility-weighted rate) with an implicit uncertainty measure (the credibility factor Z). The difference is that B-S is designed explicitly for the problem of blending group-specific evidence with portfolio evidence, rather than estimating a main effect.

### The three structural parameters

The B-S model operates on a dataset of groups (districts, schemes, vehicle classes — any set of segments) observed over multiple periods. Three numbers drive everything:

**mu (grand mean):** The portfolio-wide expected loss rate, weighted by exposure. This is the anchor — the value a thin district's estimate collapses toward.

**v (EPV — Expected value of Process Variance):** Within-group variance, averaged over the portfolio. This captures how much a group's observed rate fluctuates year to year, purely due to Poisson sampling noise, even if its true underlying risk is perfectly stable. High v means groups are inherently volatile.

**a (VHM — Variance of Hypothetical Means):** Between-group variance. This captures how much the true underlying risks differ across groups. High a means the portfolio is genuinely heterogeneous — some districts really are riskier than others.

**K = v / a:** The credibility parameter. Interpretable as: "how many units of exposure does a group need before its own experience is as informative as the portfolio mean?" Low K means you trust group experience quickly. High K means you need many years of data before the portfolio mean is overridden.

### The credibility factor Z

For group i with total exposure w_i, the credibility factor is:

```
Z_i = w_i / (w_i + K)
```

The credibility-weighted estimate is:

```
P_i = Z_i × X̄_i + (1 - Z_i) × mu
```

where X̄_i is the exposure-weighted observed mean for group i.

This formula has a clean intuitive reading:
- When w_i is large (many policy-years), Z_i → 1, and the estimate trusts the group's own experience
- When w_i is small, Z_i → 0, and the estimate collapses to the portfolio mean mu
- K controls the speed of this transition

For KT with 847 policy-years and K = 1,200: Z = 847 / (847 + 1200) = 0.41. KT's rate would be 41% of its own observed 1.30% and 59% of the portfolio mean 6.8%. That gives 0.41 × 0.013 + 0.59 × 0.068 = 0.046, i.e. 4.6%. A meaningful adjustment downward from the portfolio mean, but far from KT's own volatile 1.3%.

### Estimating the structural parameters from data

You do not specify v and a in advance — they are estimated from the data. Here are the formulas, followed immediately by the implementation:

**Grand mean:**
```
mu_hat = Σ_i(w_i × X̄_i) / Σ_i(w_i)
```
Weighted average of group means, weighted by exposure.

**EPV (v_hat) — within-group variance:**
```
v_hat = [Σ_i Σ_j w_{ij} × (X_{ij} - X̄_i)²] / Σ_i(T_i - 1)
```
Sum of within-group squared deviations, weighted by exposure, divided by the total number of within-group degrees of freedom. Groups with only one period (T_i = 1) contribute zero to the numerator but would subtract 1 from the denominator — filter these out before computing v_hat.

**VHM (a_hat) — between-group variance:**
```
c    = Σ_i(w_i) - Σ_i(w_i²) / Σ_i(w_i)
s²   = Σ_i w_i × (X̄_i - mu_hat)²
a_hat = (s² - (r - 1) × v_hat) / c
```
The between-group sum of squares, with sampling noise removed. Important: a_hat can be negative. This happens when within-group variance dominates — the groups look similar not because they are truly similar, but because the data are too noisy to distinguish them. By convention, truncate a_hat at zero. When a_hat = 0, K = infinity and all Z_i = 0 — every group gets the portfolio mean. This is not wrong; it means the data cannot justify any group-level adjustment.

### The implementation

Create a new cell with:

```python
%md
## Part 4: Bühlmann-Straub implementation
```

Create the next cell with the full function:

```python
def buhlmann_straub(
    data: pl.DataFrame,
    group_col: str,
    value_col: str,
    weight_col: str,
    log_transform: bool = True,
) -> dict:
    """
    Bühlmann-Straub credibility estimator.

    Parameters
    ----------
    data : Polars DataFrame with one row per (group, period).
        Each row represents one group in one period.
    group_col : str
        Column identifying the group (e.g. "postcode_district").
    value_col : str
        Observed loss rate per unit of exposure (e.g. "claim_frequency").
    weight_col : str
        Exposure weight (e.g. "earned_years").
    log_transform : bool
        If True, apply B-S in log-rate space to avoid the Jensen's inequality
        bias that arises in multiplicative (log-link) frameworks. Set False for
        additive models or when working with severity in additive space.

    Returns
    -------
    dict with keys:
        grand_mean  : float — portfolio mean (on original scale if log_transform=True)
        v_hat       : float — EPV estimate (on working scale)
        a_hat       : float — VHM estimate, truncated at 0
        a_hat_raw   : float — VHM before truncation (negative values are diagnostic)
        k           : float — v_hat / a_hat (inf if a_hat = 0)
        results     : Polars DataFrame with per-group Z and credibility estimates
    """
    # --- Work in log space if multiplicative framework ---
    if log_transform:
        # Clip near-zero rates to avoid log(0). Do not clip to 0 directly:
        # that silently drops valid thin-cell observations.
        data = data.with_columns(
            pl.col(value_col).clip(lower_bound=1e-9).log().alias("_y")
        )
        y_col = "_y"
    else:
        y_col = value_col

    groups = data[group_col].unique().sort().to_list()
    r = len(groups)

    # --- Per-group sufficient statistics ---
    group_data = (
        data
        .group_by(group_col)
        .agg([
            pl.col(weight_col).sum().alias("w_i"),
            (
                (pl.col(y_col) * pl.col(weight_col)).sum()
                / pl.col(weight_col).sum()
            ).alias("x_bar_i"),
            pl.col(weight_col).count().alias("T_i"),   # number of periods
        ])
        .sort(group_col)
    )

    # Filter single-period groups for EPV calculation.
    # A group with T_i = 1 contributes 0 to the EPV numerator (no within-group
    # deviation possible from a single observation) but would subtract 1 from
    # the denominator. This incorrectly reduces the effective sample size.
    group_data_epv = group_data.filter(pl.col("T_i") > 1)

    w_i = group_data["w_i"].to_numpy()
    x_bar_i = group_data["x_bar_i"].to_numpy()
    w = w_i.sum()

    # --- Collective mean (grand mean) ---
    mu_hat = (w_i * x_bar_i).sum() / w

    # --- EPV: within-group variance ---
    def epv_numerator_for_group(grp_name: str) -> float:
        grp = data.filter(pl.col(group_col) == grp_name)
        if grp.height <= 1:
            return 0.0
        x_bar = float(
            (grp[y_col] * grp[weight_col]).sum() / grp[weight_col].sum()
        )
        resid_sq = (grp[y_col].to_numpy() - x_bar) ** 2
        return float((resid_sq * grp[weight_col].to_numpy()).sum())

    epv_groups = group_data_epv[group_col].to_list()
    epv_num = sum(epv_numerator_for_group(g) for g in epv_groups)
    epv_den = float(group_data_epv["T_i"].sum() - len(epv_groups))
    v_hat = epv_num / epv_den if epv_den > 0 else 0.0

    # --- VHM: between-group variance ---
    c = w - (w_i ** 2).sum() / w
    s_sq = (w_i * (x_bar_i - mu_hat) ** 2).sum()
    a_hat_raw = (s_sq - (r - 1) * v_hat) / c
    # Truncate at zero: negative a_hat means data cannot distinguish groups.
    # This is a valid conclusion, not an error. When a_hat = 0, all Z_i = 0.
    a_hat = max(a_hat_raw, 0.0)

    # --- Credibility factors and estimates ---
    k = v_hat / a_hat if a_hat > 0 else np.inf
    z_i = w_i / (w_i + k) if np.isfinite(k) else np.zeros(r)
    cred_est_log = z_i * x_bar_i + (1 - z_i) * mu_hat

    # Convert back from log space if needed
    if log_transform:
        grand_mean = np.exp(mu_hat)
        cred_est = np.exp(cred_est_log)
        obs_mean = np.exp(x_bar_i)
    else:
        grand_mean = mu_hat
        cred_est = cred_est_log
        obs_mean = x_bar_i

    results_df = pl.DataFrame({
        group_col: group_data[group_col].to_list(),
        "exposure":               w_i.tolist(),
        "obs_mean":               obs_mean.tolist(),
        "Z":                      z_i.tolist(),
        "credibility_estimate":   cred_est.tolist(),
    })

    return {
        "grand_mean":  grand_mean,
        "v_hat":       v_hat,
        "a_hat":       a_hat,
        "a_hat_raw":   a_hat_raw,
        "k":           k,
        "results":     results_df,
    }
```

**What this does:** Defines the `buhlmann_straub` function. It takes a Polars DataFrame with one row per (group, period) — in our case, one row per (postcode_district, accident_year) — and returns the structural parameters and per-group credibility estimates. The `log_transform=True` argument tells it to work in log-rate space, which is correct for Poisson frequency models.

**Run this cell.** There is no output — you are just defining the function.

### Fitting Bühlmann-Straub on claim frequency

Now apply it. First, prepare the district-year data in the format the function expects:

```python
# The function needs one row per (group, period) with:
#   - the loss rate (claim frequency for that district-year)
#   - the exposure weight (earned years for that district-year)

# Filter out district-years with very low exposure to avoid near-infinite
# frequencies from near-zero denominators.
# 0.5 years is the minimum threshold — anything below this is a data artefact
# (mid-year new entrant or incomplete year) rather than a real observation.
dist_year = (
    df
    .filter(pl.col("earned_years") > 0.5)
    .select(["postcode_district", "accident_year", "earned_years",
             "claim_count", "claim_frequency"])
)

print(f"Rows in dist_year: {dist_year.height}  (should be close to 600)")
print(f"\nPreview:")
print(dist_year.head(8))
```

**Run this cell.** You should see approximately 600 rows (600 = 120 districts × 5 years; a few near-zero exposure rows may be filtered out).

Now fit:

```python
bs = buhlmann_straub(
    data=dist_year,
    group_col="postcode_district",
    value_col="claim_frequency",
    weight_col="earned_years",
    log_transform=True,    # working in log-rate space: correct for Poisson/log-link
)

print("=== Bühlmann-Straub Results ===")
print()
print(f"Portfolio grand mean:  {bs['grand_mean']:.4f}  ({bs['grand_mean']*100:.2f}% frequency)")
print()
print(f"EPV (v):   {bs['v_hat']:.6f}   within-district year-to-year variance")
print(f"VHM (a):   {bs['a_hat']:.6f}   between-district variance (true underlying differences)")
print(f"K:         {bs['k']:.1f}         earned years for Z = 0.50")
print()

if bs['a_hat_raw'] < 0:
    print(f"DIAGNOSTIC: a_hat before truncation = {bs['a_hat_raw']:.6f}  (negative)")
    print("  The data cannot distinguish district effects from sampling noise.")
    print("  All Z = 0; every district gets the portfolio mean.")
    print("  This is unusual for a 120-district synthetic dataset — check data quality.")
else:
    print(f"  a_hat raw (before truncation): {bs['a_hat_raw']:.6f}  (positive — good)")

print()
print("Per-district credibility estimates (first 15):")
print(bs['results'].head(15))
```

**What this does:** Runs the Bühlmann-Straub estimator on all 120 districts across 5 years. The output tells you: the portfolio mean frequency, the within-district variance (EPV), the between-district variance (VHM), and the K parameter that controls the blend.

**Run this cell.**

**What you should see:**
- Grand mean close to 7% (our true portfolio frequency)
- VHM (a) positive — the simulation has genuine between-district heterogeneity (sigma=0.35)
- K somewhere in the range 100-800 (depending on the simulated data)
- A DataFrame with 120 rows, one per district, showing Z values ranging from near 0 (thin districts) to near 1 (dense districts)

If a_hat is negative, something has gone wrong with the data. Re-run the data generation cell with `rng = np.random.default_rng(seed=42)` and try again.

### What K means in practice

```python
# Exposure thresholds for different Z values, given this K
k = bs['k']
print(f"With K = {k:.0f} earned years:")
print()
print(f"  Z = 0.25  →  need {k/3:.0f} earned years (75% from portfolio mean)")
print(f"  Z = 0.50  →  need {k:.0f} earned years  (half weight on own experience)")
print(f"  Z = 0.67  →  need {2*k:.0f} earned years  (two-thirds on own experience)")
print(f"  Z = 0.80  →  need {4*k:.0f} earned years  (80% on own experience)")
print(f"  Z = 0.90  →  need {9*k:.0f} earned years  (90% on own experience)")
print()
print("In context: a district needs K earned years for Z = 0.50.")
print(f"A thin district with 100 earned years has Z = {100/(100+k):.2f}")
print(f"A dense district with 5,000 earned years has Z = {5000/(5000+k):.2f}")
```

**Run this cell.** The output shows how many policy-years a district needs before its own experience dominates the portfolio mean. This is the number you bring to the underwriting team when explaining why thin district rates are constrained.

### Checkpoint 2: Validate credibility estimates against the true rates

Since we generated the data, we can check whether the credibility estimator is actually recovering the true rates better than the naive observed rates:

```python
# Merge credibility estimates with true rates for validation
bs_results = bs["results"]

# Get true rates from the aggregated district totals
true_rates_df = dist_totals.select(["postcode_district", "true_rate", "total_earned_years"])

validation = (
    bs_results
    .join(true_rates_df, on="postcode_district", how="inner")
    .with_columns([
        # Error of naive observed rate vs true rate
        ((pl.col("obs_mean") - pl.col("true_rate")).abs()).alias("obs_error"),
        # Error of credibility estimate vs true rate
        ((pl.col("credibility_estimate") - pl.col("true_rate")).abs()).alias("cred_error"),
    ])
)

# Mean absolute error comparison
mae_obs = float(validation["obs_error"].mean())
mae_cred = float(validation["cred_error"].mean())

print("=== CHECKPOINT 2: CREDIBILITY VALIDATION ===")
print()
print("Comparing naive observed rates to credibility estimates vs the true rate:")
print(f"  MAE (observed rate):           {mae_obs:.5f}  ({mae_obs*100:.3f}%)")
print(f"  MAE (credibility estimate):    {mae_cred:.5f}  ({mae_cred*100:.3f}%)")
print()
improvement = (mae_obs - mae_cred) / mae_obs * 100
print(f"  Credibility reduction in MAE:  {improvement:.1f}%")
print()

if improvement > 0:
    print(f"Credibility estimates are closer to the truth on average. Good.")
else:
    print("Observed rates are closer on average. This can happen with a homogeneous")
    print("portfolio (small sigma_district) — check your simulation parameters.")

# Show the districts where credibility helps most (the thinnest ones)
thin = validation.sort("total_earned_years").head(20)
print()
print("20 thinnest districts — credibility vs observed error:")
print(thin.select(["postcode_district", "total_earned_years", "true_rate",
                    "obs_mean", "credibility_estimate",
                    "obs_error", "cred_error"]))
```

**What you should see:** Credibility estimates with lower MAE than naive observed rates, particularly for thin districts. The improvement should be 20-50% for a portfolio with 120 districts and our simulation parameters. For the 20 thinnest districts, the credibility estimate should be substantially closer to the true rate — this is the point of the exercise.

---

## Part 5: The shrinkage plot

The shrinkage plot is the chart that earns credibility modelling its budget. It shows, visually, what credibility is doing: pulling thin cells toward the portfolio mean and leaving dense cells close to their own experience.

```python
%md
## Part 5: Shrinkage plot
```

```python
# Build the data for the shrinkage plot
# We need: observed rate, credibility estimate, exposure (for sizing), Z (for colour)
plot_data = bs_results.join(
    dist_totals.select(["postcode_district", "total_earned_years"]),
    on="postcode_district",
    how="inner",
)

obs_rates = plot_data["obs_mean"].to_numpy()
cred_ests = plot_data["credibility_estimate"].to_numpy()
exposures = plot_data["total_earned_years"].to_numpy()
z_vals = plot_data["Z"].to_numpy()

grand_mean = bs["grand_mean"]

# Point sizes: log exposure, scaled to readable marker sizes
log_exp = np.log1p(exposures)
sizes = 15 + 120 * (log_exp - log_exp.min()) / (log_exp.max() - log_exp.min() + 1e-9)

fig, ax = plt.subplots(figsize=(10, 8))

sc = ax.scatter(
    obs_rates,
    cred_ests,
    s=sizes,
    c=z_vals,
    cmap="RdYlGn",
    alpha=0.7,
    edgecolors="grey",
    linewidths=0.3,
    vmin=0, vmax=1,
)

# 45-degree line: perfect concordance between observed and credibility estimate
# Dense cells (high Z) should sit near this line
all_rates = np.concatenate([obs_rates, cred_ests])
rate_min = all_rates.min() * 0.8
rate_max = all_rates.max() * 1.1
ax.plot([rate_min, rate_max], [rate_min, rate_max],
        "k--", alpha=0.3, lw=1.5, label="No shrinkage (observed = estimate)")

# Horizontal line at grand mean: thin cells (low Z) should sit near this line
ax.axhline(grand_mean, color="steelblue", linestyle=":", alpha=0.6, lw=1.5,
           label=f"Grand mean = {grand_mean:.3f}")

plt.colorbar(sc, label="Credibility factor Z  (green = high Z, red = low Z)")

ax.set_xlabel("Observed claim frequency", fontsize=12)
ax.set_ylabel("Credibility-weighted estimate", fontsize=12)
ax.set_title("Bühlmann-Straub shrinkage plot\n(point size ∝ log exposure; colour = Z)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Plots observed rates on the x-axis against credibility estimates on the y-axis. Points near the 45-degree line are districts whose credibility estimates are close to their observed rate — these are dense districts with high Z. Points near the horizontal grand mean line are districts that have been pulled strongly toward the portfolio mean — these are thin districts with low Z.

**Run this cell.**

**What you should see:** A scatter plot where:
- Large green points (dense districts, high Z) cluster near the 45-degree dashed line
- Small red points (thin districts, low Z) cluster near the horizontal blue dotted line at the grand mean
- No point at an extreme observed rate has its credibility estimate also at that extreme, unless it has high exposure justifying it

This is the chart to show a pricing committee. It demonstrates in one image that the model is doing the right thing: trusting dense districts' experience and pulling thin districts back toward safety.

---

## Part 6: The bridge to Bayesian — what Bühlmann-Straub assumes

### B-S is empirical Bayes

Bühlmann-Straub is an empirical Bayes method. This is rarely stated clearly in actuarial textbooks, but it motivates everything in the second half of this module.

The credibility premium P_i is exactly the posterior mean of this Bayesian model:

```
X_{ij} | theta_i  ~  Normal(theta_i,  v / w_{ij})    [observation model]
theta_i            ~  Normal(mu, a)                    [prior on group mean]
```

The posterior mean of theta_i is:

```
E[theta_i | data] = Z_i × X̄_i + (1 - Z_i) × mu
```

where Z_i = w_i / (w_i + K), K = v/a. That is exactly the Bühlmann-Straub formula.

B-S plugs in point estimates of v and a (the structural parameters we estimated above). Full Bayesian treats v and a as uncertain — it places priors on them and integrates over that uncertainty. Three things follow from this:

**When B-S is sufficient:**
- Many groups (20+) so that v and a are estimated reliably from data
- One grouping variable
- Results needed in seconds, not minutes
- Regulatory documentation needs to be simple and traceable

**When full Bayesian is better:**
- Few groups (fewer than 10 affinity schemes) — with few groups, the estimate of the between-group variance is unreliable, and that uncertainty propagates into Z. Full Bayesian propagates it correctly; B-S ignores it.
- Multiple crossed grouping variables simultaneously (area AND vehicle group AND NCD band)
- Proper Poisson or Gamma likelihood — B-S assumes Normal errors in its derivation
- Credible intervals on individual segment rates are required for regulatory evidence or pricing decisions on thin segments

---

## Part 7: Bayesian hierarchical models with PyMC

### The model we are building

We will fit a Poisson hierarchical model to the district-level claim counts. The model in full:

```
claims_i  ~  Poisson(lambda_i × exposure_i)
log(lambda_i)  =  alpha + u_district[i]
u_district[k]  ~  Normal(0, sigma_district)
alpha           ~  Normal(log(mu_portfolio), 0.5)
sigma_district  ~  HalfNormal(0.3)
```

**What each line means:**

`claims_i ~ Poisson(lambda_i × exposure_i)` — Claims for district i are Poisson-distributed with rate lambda_i per unit of exposure. This is the same distributional assumption as a Poisson GLM.

`log(lambda_i) = alpha + u_district[i]` — The log-rate for district i is the sum of a global intercept (alpha) and a district-specific deviation (u_district[i]).

`u_district[k] ~ Normal(0, sigma_district)` — District deviations are drawn from a Normal distribution centred at zero with standard deviation sigma_district. When sigma_district is large, the model allows districts to differ substantially from the global mean. When sigma_district is small, districts are heavily pooled toward the global mean. The data determine sigma_district.

`alpha ~ Normal(log(mu_portfolio), 0.5)` — The global intercept has a weakly informative prior centred on the log of the observed portfolio frequency. SD = 0.5 on the log scale allows the intercept to range from approximately 0.6× to 1.65× the portfolio mean at ±1 SD — not unduly tight.

`sigma_district ~ HalfNormal(0.3)` — The between-district standard deviation has a HalfNormal prior (constrained to be positive). HalfNormal(0.3) places most prior mass below about 0.6 log points, implying most districts sit within roughly 0.5× to 2.0× the portfolio mean — reasonable for UK motor postcodes.

### Why we need MCMC

Unlike GLMs, which have closed-form maximum likelihood solutions, this hierarchical model cannot be solved analytically. The posterior distribution over all parameters — alpha, all 120 u_district[k] values, and sigma_district — is a 122-dimensional distribution that has no closed form.

MCMC (Markov Chain Monte Carlo) is a family of algorithms that draw samples from this posterior distribution. PyMC uses NUTS (No-U-Turn Sampler), a particularly efficient variant. After sampling, we summarise the posterior with means and quantiles.

You do not need to understand NUTS in detail. You need to understand three things:
1. How to run it (the `pm.sample()` call below)
2. How to check that it worked (convergence diagnostics — covered below)
3. How to interpret the output (posterior means and credible intervals)

### Non-centered parameterisation — mandatory for hierarchical models

Before writing the model code, there is one implementation detail that is non-negotiable: non-centered parameterisation.

The obvious (but wrong) way to write the district random effects:

```python
# CENTERED — do not use this
u_district = pm.Normal("u_district", mu=0, sigma=sigma_district, dims="district")
```

The problem: when sigma_district is near zero (all districts are similar), u_district and sigma_district become highly correlated in the posterior. The posterior geometry forms a "funnel". NUTS cannot pick a step size that works in both the narrow neck and the wide mouth of the funnel simultaneously. The sampler under-samples the region near sigma → 0, biasing the variance component estimates downward. Your credibility factors will be systematically too low — more shrinkage than the data warrant — without any obvious warning.

The correct way — non-centered parameterisation:

```python
# NON-CENTERED — always use this for hierarchical models
u_district_raw = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
u_district = pm.Deterministic("u_district", u_district_raw * sigma_district, dims="district")
```

This decouples the raw offset (u_district_raw, which is standard Normal) from the scale (sigma_district). The posterior geometry is now approximately Gaussian, and NUTS samples it efficiently. This is not optional — it is the standard practice for hierarchical models in PyMC.

### Building and fitting the model

Create a new markdown cell:

```python
%md
## Part 7: Bayesian hierarchical model
```

Create the next cell:

```python
# Prepare numpy arrays for PyMC
# PyMC works with numpy arrays, not Polars DataFrames directly.
# We need to convert our Polars data to numpy for the likelihood.

# Use district-level totals (aggregated over all years)
# For the Bayesian model, we pass total claims and total earned years per district
segments = dist_totals.sort("postcode_district")   # sort for reproducibility

# Encode districts as integer indices
# PyMC identifies array positions by integer index, not by string name.
# We need a mapping from district name to integer.
districts_sorted = segments["postcode_district"].to_list()
district_to_idx = {d: i for i, d in enumerate(districts_sorted)}
n_districts_model = len(districts_sorted)

# Convert to numpy
district_idx_arr = np.array([district_to_idx[d] for d in districts_sorted])
claims_arr = segments["total_claims"].to_numpy().astype(int)
exposure_arr = segments["total_earned_years"].to_numpy()

# Portfolio log-rate: prior centre for alpha
log_mu_portfolio = np.log(claims_arr.sum() / exposure_arr.sum())

print(f"Districts in model:       {n_districts_model}")
print(f"Total claims in model:    {claims_arr.sum():,}")
print(f"Total exposure in model:  {exposure_arr.sum():,.0f} earned years")
print(f"Portfolio log-rate:       {log_mu_portfolio:.4f}  (= log({np.exp(log_mu_portfolio):.4f}))")
print()
print("Exposure range:")
print(f"  Min: {exposure_arr.min():,.0f} earned years  (thinnest district)")
print(f"  Max: {exposure_arr.max():,.0f} earned years  (densest district)")
```

**What this does:** Converts the Polars DataFrame to numpy arrays in the format PyMC expects. The `district_idx_arr` is an integer array that maps each observation to its district. Because `segments` is sorted by district name and `districts_sorted` is the same sorted list, `district_idx_arr` is simply `[0, 1, 2, ..., 119]` — but writing it explicitly makes the indexing transparent.

**Run this cell.**

**What you should see:** 120 districts, total claims around 30,000-50,000 (depending on the random seed), exposure spanning from a thin district with ~100 earned years to a dense district with potentially 20,000+ earned years.

Now define and fit the model:

```python
# PyMC uses a "with pm.Model() as model_name:" context manager.
# Everything inside the with block is part of the model definition.
# This is just Python convention - the model object collects all the
# random variables defined inside it.

coords = {"district": districts_sorted}   # named coordinates for the random effects

with pm.Model(coords=coords) as hierarchical_model:

    # --- Priors ---
    # alpha: global log-rate intercept.
    # Prior: Normal centred on the observed log portfolio frequency.
    # SD = 0.5 on log scale: allows prior range of ~0.6x to ~1.65x portfolio mean at ±1 SD.
    alpha = pm.Normal("alpha", mu=log_mu_portfolio, sigma=0.5)

    # sigma_district: between-district log-rate standard deviation.
    # Prior: HalfNormal(0.3).
    # This constrains sigma to be positive and places most mass below ~0.6 log points.
    # On the rate scale: most districts within ~0.5x to ~2.0x the portfolio mean.
    sigma_district = pm.HalfNormal("sigma_district", sigma=0.3)

    # --- Non-centered parameterisation (mandatory) ---
    # u_district_raw: raw district offsets, standard Normal.
    # u_district: actual district log-rate deviations = raw * scale.
    # The "dims='district'" argument labels the array with the district names
    # defined in coords — this makes results easier to extract later.
    u_district_raw = pm.Normal("u_district_raw", mu=0, sigma=1, dims="district")
    u_district = pm.Deterministic(
        "u_district", u_district_raw * sigma_district, dims="district"
    )

    # --- Linear predictor ---
    # log(lambda_i) = alpha + u_district[district_index_i]
    # district_idx_arr maps each observation to its district's log-rate deviation.
    log_lambda = alpha + u_district[district_idx_arr]

    # --- Likelihood ---
    # Poisson distribution: claims_i ~ Poisson(lambda_i * exposure_i)
    # pm.math.exp(log_lambda) converts from log space to rate space.
    # Multiplying by exposure_arr gives the expected claim count E[claims_i].
    claims_obs = pm.Poisson(
        "claims_obs",
        mu=pm.math.exp(log_lambda) * exposure_arr,
        observed=claims_arr,
    )

# Print a model summary so we can check the structure before sampling.
# This shows all random variables and their shapes.
print("Model structure:")
print(hierarchical_model.debug())
```

**What this does:** Defines the hierarchical Poisson model. The `with pm.Model()` block is how PyMC knows which variables belong to the model. Nothing is computed yet — this is just the model definition.

**Run this cell.**

**What you should see:** A text description of the model showing the random variables (`alpha`, `sigma_district`, `u_district_raw`, `u_district`, `claims_obs`) and their shapes and distributions. If you see an error here, it is almost always a shape mismatch — check that `district_idx_arr`, `claims_arr`, and `exposure_arr` all have length 120.

Now sample. This is the slow step — it runs MCMC:

```python
# pm.sample() runs NUTS (No-U-Turn Sampler) to draw samples from the posterior.
#
# Parameters:
#   draws=1000:          Number of posterior samples per chain to keep.
#   tune=1000:           Number of warmup/adaptation steps before keeping samples.
#                        During tuning, NUTS adapts its step size and mass matrix.
#                        Tuning samples are discarded.
#   chains=4:            Number of independent Markov chains to run.
#                        Running 4 chains lets us check convergence with R-hat.
#   target_accept=0.90:  Target acceptance probability. Higher values → smaller
#                        step sizes → slower but more thorough exploration.
#                        0.90 is appropriate for most hierarchical models.
#   return_inferencedata=True: Return an ArviZ InferenceData object (recommended).
#   random_seed=42:      For reproducibility.
#
# This takes approximately 3-6 minutes on a Databricks Free Edition cluster.
# The progress bar shows chains running in parallel if cores > 1.
# Do not stop it early — incomplete chains cannot be used for diagnostics.

print("Fitting hierarchical Bayesian model via NUTS...")
print("Expected time: 3-6 minutes on Databricks Free Edition.")
print()

with hierarchical_model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
    )

print()
print("Sampling complete.")
```

**What this does:** Runs NUTS sampling. PyMC will print a progress bar showing the sampling status for each chain.

**Run this cell and wait.** On Databricks Free Edition (single node), this takes 3-6 minutes for 120 districts. A multi-core worker cluster would be faster — see the Databricks Deployment section later in this module.

**What you should see during sampling:** A progress bar like:
```
Auto-assigning NUTS sampler...
Initializing NUTS using jitter+adapt_diag...
Multiprocess sampling (4 chains in 2 jobs)
NUTS: [alpha, sigma_district, u_district_raw]
 |████████████| 100.00% [8000/8000 00:03<00:00 Sampling 4 chains, 0 divergences]
```

The number of divergences should be 0. If you see divergences, it is not a catastrophic error, but it needs investigation — covered in the next section.

---

## Part 8: Convergence diagnostics — do not skip this

MCMC sampling can fail silently. A model can complete sampling, print no errors, and produce posteriors that are systematically wrong. Convergence diagnostics are not bureaucratic box-ticking — they are how you detect this.

### What convergence means

MCMC generates a Markov chain — a sequence of samples where each sample depends only on the previous one. If the chain has been running long enough and mixing well across the posterior, the distribution of samples approximates the true posterior. "Convergence" means this approximation is good.

We run 4 independent chains starting from different initial points. If the chains have converged, their distributions should look the same. If they look different, something is wrong.

### R-hat: between-chain versus within-chain variance

R-hat (also written R̂) compares the variance between chains to the variance within chains. A value of 1.0 means the chains are identical — perfect convergence. Values above 1.01 indicate the chains have not converged: some region of the posterior is not being explored consistently.

### ESS: effective sample size

Because MCMC samples are autocorrelated (each sample is correlated with the previous one), 4,000 samples do not give you 4,000 independent pieces of information. ESS adjusts for this autocorrelation. Low ESS (below 400) means the posterior estimate for that parameter is unreliable.

For variance components (sigma_district), the ESS requirement is higher — we recommend 1,000+ — because sigma_district drives the credibility factors for all districts. A poorly sampled sigma_district propagates errors into all district estimates.

### Divergences

A divergence occurs when NUTS takes an extremely large step and the trajectory diverges numerically. Divergences indicate regions of the posterior where the sampler cannot explore correctly. Non-zero divergences require investigation, even if convergence diagnostics look otherwise acceptable.

### Running the diagnostics

```python
%md
## Part 8: Convergence diagnostics
```

```python
# R-hat: should be < 1.01 for all parameters
rhat = az.rhat(trace)

# az.rhat() returns an xarray Dataset. Convert to a single maximum value.
max_rhat = float(rhat.max().to_array().max())
print(f"Max R-hat across all parameters: {max_rhat:.4f}")

if max_rhat < 1.01:
    print("  Status: OK — chains have converged.")
elif max_rhat < 1.05:
    print("  Status: WARNING — some parameters have not fully converged.")
    print("  Consider increasing draws to 2000 or tune to 2000.")
else:
    print("  Status: FAILED — chains have not converged. Do not use these results.")
    print("  Check for model misspecification or try non-centered parameterisation.")

print()

# ESS (bulk): effective sample size for interior of distribution
ess_bulk = az.ess(trace, method="bulk")
min_ess_bulk = float(ess_bulk.min().to_array().min())
print(f"Min ESS (bulk) across all parameters: {min_ess_bulk:.0f}")

if min_ess_bulk > 400:
    print("  Status: OK")
else:
    print("  Status: LOW — increase draws or check for slow mixing.")

print()

# Variance component ESS — especially important
sigma_ess = float(ess_bulk["sigma_district"])
print(f"ESS for sigma_district (the key variance component): {sigma_ess:.0f}")
if sigma_ess > 1000:
    print("  Status: OK — variance component is well-sampled.")
elif sigma_ess > 400:
    print("  Status: ACCEPTABLE — consider 2000 draws for final results.")
else:
    print("  Status: LOW — increase draws to 2000+ for variance component reliability.")

print()

# Divergences: should be 0
n_div = int(trace.sample_stats["diverging"].sum())
print(f"Divergences: {n_div}")
if n_div == 0:
    print("  Status: OK — no divergences.")
elif n_div < 10:
    print("  Status: FEW — investigate but may be acceptable.")
    print("  Try increasing target_accept to 0.95.")
else:
    print("  Status: MANY — model has posterior geometry problems.")
    print("  Verify non-centered parameterisation is correctly implemented.")
    print("  If divergences persist, the model may be misspecified.")
```

**What this does:** Runs the three standard MCMC convergence checks. The ESS check for `sigma_district` specifically is important — the between-district variance drives all the credibility factors, and it needs to be well-sampled.

**Run this cell.**

**What you should see with the synthetic data and the model as written:** Max R-hat below 1.01, min ESS above 400, ESS for sigma_district above 1000, zero divergences. If any check fails, the interpretations above tell you what to try next.

### Trace plots: visual convergence check

```python
# Trace plot for the key parameters
# Shows the chain values over iterations (should look like white noise)
# and the marginal distribution (should be smooth bell-shaped)

az.plot_trace(
    trace,
    var_names=["alpha", "sigma_district"],  # plot the global parameters
    figsize=(10, 5),
)
plt.suptitle("Trace plots: alpha and sigma_district", y=1.02)
plt.tight_layout()
display(plt.gcf())
plt.close()
```

**What this does:** Shows the trace of the MCMC chains over iterations. Good mixing looks like a "hairy caterpillar" — the chain moves freely across its range, all four chains overlap, and there are no obvious trends or stuck periods.

**Run this cell.**

**What you should see:** For alpha: four chains (different colours) that overlap heavily, all fluctuating around the same mean. For sigma_district: same pattern, with the four chains' marginal distributions (right panel) all showing the same shape. If any chain looks like it is stuck in one region or trending, convergence has not been achieved.

### Checkpoint 3: Convergence check

```python
# Hard gate: if convergence criteria are not met, do not proceed with results.
# The results downstream are meaningless if the sampler has not converged.
print("=== CHECKPOINT 3: CONVERGENCE ===")
print()
convergence_ok = (max_rhat < 1.01) and (min_ess_bulk > 400) and (n_div == 0)
print(f"R-hat OK (< 1.01):      {'YES' if max_rhat < 1.01 else 'NO'}")
print(f"Min ESS OK (> 400):     {'YES' if min_ess_bulk > 400 else 'NO'}")
print(f"Divergences OK (= 0):   {'YES' if n_div == 0 else 'NO'}")
print()
if convergence_ok:
    print("All convergence criteria met. Proceeding to extract results.")
else:
    print("CONVERGENCE FAILURE: Do not interpret the following results.")
    print("Investigate the failing criteria before proceeding.")
```

If any convergence check fails here, stop. Fix the issue before proceeding. Results from a model that has not converged are not just uncertain — they are wrong in ways that are hard to detect downstream.

---

## Part 9: Extracting posterior estimates

Once convergence is confirmed, extract the posterior means and credible intervals:

```python
%md
## Part 9: Extracting posterior estimates
```

```python
# Extract posterior samples for alpha and u_district
# trace.posterior is an xarray Dataset.
# trace.posterior["alpha"] has shape (chains, draws) = (4, 1000)
# trace.posterior["u_district"] has shape (chains, draws, districts) = (4, 1000, 120)

# Posterior means: average over chains and draws
alpha_post_mean = float(trace.posterior["alpha"].mean())
u_post_mean = trace.posterior["u_district"].mean(dim=("chain", "draw")).values  # shape: (120,)

# Posterior mean log-rate per district
log_rate_post_mean = alpha_post_mean + u_post_mean

# Convert to rate space (anti-log)
posterior_mean_rate = np.exp(log_rate_post_mean)

print(f"alpha posterior mean: {alpha_post_mean:.4f}")
print(f"Portfolio rate from alpha: {np.exp(alpha_post_mean):.4f}  ({np.exp(alpha_post_mean)*100:.2f}%)")
print()
print(f"sigma_district posterior mean: {float(trace.posterior['sigma_district'].mean()):.4f}")
print(f"  (True sigma_district was:   {TRUE_SIGMA_DISTRICT:.4f})")
print()
```

**What this does:** Extracts the posterior means for the global parameters. The `mean(dim=("chain", "draw"))` call averages across all 4 chains × 1000 draws = 4,000 samples to give a single posterior mean per district.

**Run this cell.**

**What you should see:** The posterior mean for sigma_district should be close to 0.35 (the true value we used to simulate the data). If it is far off, investigate the convergence diagnostics. Alpha posterior mean should be close to `log(0.07) ≈ -2.66`.

Now extract per-district posterior intervals:

```python
# Credible intervals: 90% posterior interval for each district's rate
# We need to sample the full posterior for each district, not just the mean.

# Stack all posterior samples into a 2D array: (total_samples, districts)
# Shape: (4 chains × 1000 draws, 120 districts) = (4000, 120)
alpha_samples = trace.posterior["alpha"].values.flatten()          # shape: (4000,)
u_samples = trace.posterior["u_district"].values.reshape(-1, n_districts_model)  # (4000, 120)

# Add alpha to each district's u_district samples
log_rate_samples = alpha_samples[:, np.newaxis] + u_samples         # (4000, 120)
rate_samples = np.exp(log_rate_samples)                              # (4000, 120)

# 5th and 95th percentile → 90% credible interval
lower_90 = np.percentile(rate_samples, 5, axis=0)   # shape: (120,)
upper_90 = np.percentile(rate_samples, 95, axis=0)  # shape: (120,)
posterior_sd = rate_samples.std(axis=0)              # shape: (120,)

observed_rate = claims_arr / exposure_arr

# Build results DataFrame
results = pl.DataFrame({
    "postcode_district": districts_sorted,
    "total_claims":      claims_arr.tolist(),
    "earned_years":      exposure_arr.tolist(),
    "observed_rate":     observed_rate.tolist(),
    "posterior_mean":    posterior_mean_rate.tolist(),
    "posterior_sd":      posterior_sd.tolist(),
    "lower_90":          lower_90.tolist(),
    "upper_90":          upper_90.tolist(),
    "interval_width":    (upper_90 - lower_90).tolist(),
})

print("Posterior estimates — first 15 districts:")
print(results.head(15))
```

**What this does:** Extracts the full posterior distribution for each district's claim rate, computes the 90% credible interval, and assembles a clean results DataFrame.

**Run this cell.**

**What you should see:** A 120-row DataFrame with each district's observed rate, posterior mean, posterior standard deviation, and 90% credible interval. Notice that thin districts have wide intervals (large `interval_width`) and dense districts have narrow intervals. This is honest: the model is correctly representing its own uncertainty.

### Approximate credibility factor from the Bayesian model

```python
# Compute an approximate Z from the Bayesian posterior.
# The Bayesian model does not directly output a Z value — Z is a B-S concept.
# But we can estimate it from how much the posterior mean was pulled toward
# the grand mean relative to the observed rate.

grand_mean_rate = float(np.exp(alpha_post_mean))

z_bayes_approx = np.where(
    np.abs(observed_rate - grand_mean_rate) > 1e-6,
    1.0 - np.abs(posterior_mean_rate - grand_mean_rate)
        / np.abs(observed_rate - grand_mean_rate),
    1.0,
).clip(0, 1)

results = results.with_columns(
    pl.Series("z_bayes_approx", z_bayes_approx)
)

print(f"Grand mean rate (exp(alpha)): {grand_mean_rate:.4f}")
print()
print("Approximate Bayesian Z vs Bühlmann-Straub Z:")
comparison = bs_results.join(
    results.select(["postcode_district", "posterior_mean", "z_bayes_approx"]),
    on="postcode_district",
    how="inner",
)
print(comparison.select([
    "postcode_district", "exposure", "Z", "z_bayes_approx",
    "credibility_estimate", "posterior_mean",
]).sort("exposure").head(20))
```

**Run this cell.**

**What you should see:** The B-S Z and Bayesian approximate Z should be broadly similar. They will agree closely for dense districts. For thin districts, the Bayesian model typically produces slightly more shrinkage (lower Z) because it propagates uncertainty in sigma_district — the B-S estimate plugs in a_hat as a fixed value, ignoring the uncertainty in the between-district variance estimate.

---

## Part 10: The shrinkage plot — Bayesian version

```python
%md
## Part 10: Bayesian shrinkage plot and uncertainty visualisation
```

```python
# Shrinkage plot comparing Bühlmann-Straub and Bayesian estimates
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

obs_rates = results["observed_rate"].to_numpy()
post_means = results["posterior_mean"].to_numpy()
bs_ests = bs_results.sort("postcode_district")["credibility_estimate"].to_numpy()
exposures_plot = results["earned_years"].to_numpy()

log_exp_p = np.log1p(exposures_plot)
sizes_p = 15 + 120 * (log_exp_p - log_exp_p.min()) / (log_exp_p.max() - log_exp_p.min() + 1e-9)

all_rates = np.concatenate([obs_rates, post_means, bs_ests])
rate_min = all_rates.min() * 0.8
rate_max = all_rates.max() * 1.1

# Left: Bühlmann-Straub
sc1 = axes[0].scatter(
    obs_rates, bs_ests,
    s=sizes_p, c=z_vals, cmap="RdYlGn", alpha=0.7,
    edgecolors="grey", linewidths=0.3, vmin=0, vmax=1,
)
axes[0].plot([rate_min, rate_max], [rate_min, rate_max], "k--", alpha=0.3, lw=1.5)
axes[0].axhline(bs["grand_mean"], color="steelblue", linestyle=":", alpha=0.6, lw=1.5)
axes[0].set_xlabel("Observed frequency")
axes[0].set_ylabel("Credibility estimate")
axes[0].set_title("Bühlmann-Straub")

# Right: Bayesian
sc2 = axes[1].scatter(
    obs_rates, post_means,
    s=sizes_p, c=z_bayes_approx, cmap="RdYlGn", alpha=0.7,
    edgecolors="grey", linewidths=0.3, vmin=0, vmax=1,
)
axes[1].plot([rate_min, rate_max], [rate_min, rate_max], "k--", alpha=0.3, lw=1.5)
axes[1].axhline(grand_mean_rate, color="steelblue", linestyle=":", alpha=0.6, lw=1.5)
axes[1].set_xlabel("Observed frequency")
axes[1].set_ylabel("Posterior mean frequency")
axes[1].set_title("Bayesian hierarchical (PyMC)")

plt.colorbar(sc2, ax=axes[1], label="Approximate Z  (green=high, red=low)")
plt.suptitle("Shrinkage comparison: Bühlmann-Straub vs Bayesian\n(size ∝ log exposure)", fontsize=12)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Side-by-side comparison of the B-S and Bayesian shrinkage. The patterns should look similar — both are applying partial pooling to the same data.

**Run this cell.**

**What you should see:** Two scatter plots with the same overall pattern. The Bayesian plot may show slightly more shrinkage for thin districts (lower z_bayes_approx for small points). Dense districts (large green points) should cluster near the 45-degree line in both plots. This visual confirmation tells you the two methods agree where they should.

### Uncertainty bands for individual districts

```python
# Plot credible intervals for the 20 districts with most evidence (densest)
# and 20 with least evidence (thinnest)
dense_20 = results.sort("earned_years", descending=True).head(20)
thin_20 = results.sort("earned_years").head(20)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

for ax, subset, title in [
    (axes[0], dense_20, "20 densest districts"),
    (axes[1], thin_20, "20 thinnest districts"),
]:
    districts_plot = subset["postcode_district"].to_list()
    obs = subset["observed_rate"].to_numpy()
    post = subset["posterior_mean"].to_numpy()
    lo = subset["lower_90"].to_numpy()
    hi = subset["upper_90"].to_numpy()
    x = np.arange(len(districts_plot))

    ax.barh(x, hi - lo, left=lo, height=0.5, alpha=0.4, color="steelblue",
            label="90% credible interval")
    ax.scatter(obs, x, marker="o", color="crimson", s=50, zorder=5, label="Observed rate")
    ax.scatter(post, x, marker="D", color="navy", s=40, zorder=6, label="Posterior mean")
    ax.axvline(grand_mean_rate, color="grey", linestyle=":", alpha=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(districts_plot, fontsize=8)
    ax.set_xlabel("Claim frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)

plt.suptitle("90% posterior credible intervals\n(navy diamond = posterior mean, red dot = observed)", fontsize=11)
plt.tight_layout()
display(fig)
plt.close(fig)
```

**What this does:** Shows credible intervals for the 20 densest and 20 thinnest districts. This is the chart that answers the pricing committee question "how confident are we in this rate?"

**Run this cell.**

**What you should see:**
- Left panel (dense districts): Narrow bars. The posterior mean (navy diamond) sits close to the observed rate (red dot) — high credibility, trusting own experience.
- Right panel (thin districts): Wide bars spanning much of the plausible rate range. The posterior mean sits noticeably closer to the grand mean (grey dotted line) than the observed rate — strong shrinkage toward the portfolio mean.

This is the honest picture. A thin district's rate might genuinely be anywhere in that wide interval. The pricing decision is: set the rate at the posterior mean, acknowledge the uncertainty, and monitor as experience develops.

---

## Part 11: Applying B-S to model residuals (the correct workflow)

### Why raw rates are often the wrong input

The tutorial above applies credibility to raw observed claim frequencies. In practice, this is rarely the right application.

If you have already fitted a GLM or GBM on the full dataset, you have a model prediction for each district that already accounts for the other rating factors (vehicle group, NCD, driver age, etc.). The credibility question becomes: should the district factor from the model be adjusted based on the district's own experience relative to what the model expects?

You apply B-S to the district-level O/E ratios (observed over expected), not to raw observed rates. This gives a clean decomposition:

- The GLM or GBM handles main effects
- Credibility on residuals handles district-level adjustments on top of the main-effect model

Applying B-S to raw rates when a GBM has already partially handled thin cells produces double shrinkage. The GBM's regularisation (min_data_in_leaf, L2 leaf penalty) already shrinks thin-cell predictions toward the base rate. Applying B-S on raw rates then shrinks again. The correct approach:

```python
# After fitting your GLM or GBM, compute district-level O/E ratios
# Replace "model_predicted_claims" with your actual model predictions

# Example structure (your column names will differ):
# df has columns: postcode_district, accident_year, earned_years,
#                 claim_count, model_predicted_claims

dist_residuals = (
    df
    .with_columns([
        # O/E ratio: observed over expected per policy-year row
        (pl.col("claim_count") / pl.col("model_predicted_claims").clip(lower_bound=1e-6))
        .alias("oe_ratio")
    ])
    .group_by(["postcode_district", "accident_year"])
    .agg([
        pl.col("oe_ratio").mean().alias("oe_ratio"),
        pl.col("earned_years").sum().alias("earned_years"),
    ])
    .filter(pl.col("earned_years") > 0.5)
)

# Apply B-S to the O/E ratios in log space
# The credibility estimate is the district-level log-adjustment factor
# to apply multiplicatively on top of the model's predictions.
bs_residuals = buhlmann_straub(
    data=dist_residuals,
    group_col="postcode_district",
    value_col="oe_ratio",
    weight_col="earned_years",
    log_transform=True,   # log because we want a multiplicative adjustment
)

# Grand mean of O/E should be close to 1.0 for a well-calibrated model
print(f"Grand mean O/E ratio: {bs_residuals['grand_mean']:.4f}  (should be near 1.0 for calibrated model)")
print(f"K (for O/E residuals): {bs_residuals['k']:.1f}")
```

**When to use this pattern:** Any time you are refining a GLM or GBM with district-level experience adjustments. The main model handles the structural rating factors; credibility handles the district-level departure from the model's expectation.

---

## Part 12: Applying B-S in log-rate space — why it matters

### The Jensen's inequality problem

For a multiplicative pricing framework (Poisson log-link), the correct approach is to blend in log-rate space. The `log_transform=True` argument in `buhlmann_straub()` does this automatically, but it is worth understanding why.

If you apply B-S directly to rates and then convert the estimate to a log-scale relativity, you introduce a bias. The log of the expected value does not equal the expected value of the log:

```
log(Z × X̄ + (1-Z) × mu)  ≠  Z × log(X̄) + (1-Z) × log(mu)
```

For typical motor insurance frequencies (5-10%), the difference is small. For extreme relativities — a very thin district with an observed rate of 0.5% against a portfolio mean of 7% — the bias can shift the estimate by 5-10% relative to the log-space calculation. Use `log_transform=True` as the default for any Poisson/multiplicative application.

---

## Part 13: Structural parameter stationarity

The EPV (v) and VHM (a) are estimated from historical data. Which historical data matters.

The 2020-2022 period contained the COVID shock: claim frequencies fell sharply in 2020 (reduced driving), rebounded unevenly in 2021-2022, and claim costs inflated due to supply chain disruption on parts and courtesy cars. Estimating v from that period inflates the EPV, because some of the within-group year-to-year variance is a portfolio-wide shock rather than genuine group-level volatility.

Inflated v → inflated K → Z values too low → more shrinkage than warranted.

In practice:
- Re-estimate structural parameters at each model rebuild cycle, not just when you rebuild the main pricing model
- Consider excluding shock years (or downweighting them) when estimating v and a, if the portfolio experienced a clear external distortion
- Monitor K stability over time: a sudden increase in K without a change in portfolio composition is a signal that the EPV estimate has been contaminated by a portfolio-wide event

For the v estimate specifically: if your within-group variance in 2019-2023 is substantially higher than 2015-2019, investigate whether the difference is genuine geographic volatility or a COVID-era artefact before using the inflated K in production.

---

## Part 14: When to use which method — the decision framework

This framework assumes you have already decided that naive observed rates are not appropriate. Some form of partial pooling is needed. The question is which form.

### Use Bühlmann-Straub when:

- **One grouping variable.** Schemes, vehicle classes, or postcode districts in isolation. B-S is a one-dimensional method; it handles one grouping at a time.
- **Many groups.** At least 5, ideally 20+. With fewer groups, the estimate of between-group variance (a_hat) is unreliable. Five groups gives a very noisy a_hat, and a noisy a_hat makes K unreliable.
- **Speed matters.** B-S runs in milliseconds. PyMC takes minutes. For daily monitoring, real-time pricing, or exploratory analysis, B-S wins.
- **Regulatory transparency.** Bühlmann-Straub has a 55-year track record in actuarial methodology documentation. The FCA Consumer Duty pack can cite Bühlmann & Straub (1970) and explain Z without specialist software. A documented, auditable methodology is substantially stronger than an undocumented one, even if the undocumented approach is technically superior.
- **Downstream of a main model.** If you are applying credibility to residuals from a GLM or GBM, B-S is the natural tool.

### Use full Bayesian hierarchical models when:

- **Multiple crossed grouping variables simultaneously.** Area AND vehicle group AND NCD band, all with partial pooling. B-S handles one dimension at a time; PyMC handles arbitrary combinations.
- **Few groups.** With fewer than 10 affinity schemes, the uncertainty in the structural parameters is material. Bayesian propagates it correctly.
- **Credible intervals are required.** B-S gives point estimates of Z and the credibility estimate. Bayesian gives the full posterior distribution — 90% credible intervals, probability that a rate exceeds a threshold, etc.
- **Two-level geographic hierarchy.** Districts nest within areas. The nested model in Exercise 2 handles this correctly.
- **Proper Poisson or Gamma likelihood.** B-S is derived from a Normal observation model. Full Bayesian uses the correct likelihood for the data type.

### The two-stage approach

For most UK personal lines pricing projects, the correct architecture is:

1. **Stage 1:** CatBoost or GLM on the full dataset for main effects — driver age, vehicle group, NCD, area bands
2. **Stage 2:** Bühlmann-Straub on district-level O/E residuals from Stage 1, with `log_transform=True`

This is not a shortcut. It is the principled decomposition: the main model handles the rating factors the GLM/GBM can identify from large samples; credibility handles the district-level departures that require pooling.

Reserve full Bayesian for cases where Stage 2 is insufficient — multiple crossed groupings, very few groups, or explicit uncertainty quantification requirements.

---

## Part 15: Databricks deployment

### PyMC on Databricks — practical setup

PyMC 5.x runs on the standard Databricks ML runtime (DBR 14.x or later). Install it in the first cell of every notebook that uses it:

```python
%pip install pymc arviz --quiet
dbutils.library.restartPython()
```

The first import of PyMC in a session compiles PyTensor computation graphs. This takes 30-60 seconds on first run in a fresh cluster session. Subsequent cells run faster.

### Parallelising chains

On a multi-core cluster, NUTS chains run in parallel. To use all available cores:

```python
import multiprocessing
n_cores = multiprocessing.cpu_count()
print(f"Available cores: {n_cores}")

with hierarchical_model:
    trace = pm.sample(
        draws=1000,
        tune=1000,
        chains=min(4, n_cores),
        cores=min(4, n_cores),   # run chains in parallel
        target_accept=0.90,
        return_inferencedata=True,
        random_seed=42,
    )
```

On a 4-core single-node cluster: 4 chains run in parallel, cutting wall-clock time by roughly 3-4×. On Databricks Free Edition (typically 1-2 cores), the chains run sequentially or with limited parallelism. For production models, use a standard cluster with 4-8 cores.

### MLflow tracking

Every hierarchical model fit should be tracked in MLflow. The convergence diagnostics are the most important artefacts — a model that failed convergence should not be usable downstream.

```python
import mlflow

mlflow.set_experiment("/pricing/credibility-bayesian/module06")

with mlflow.start_run(run_name="hierarchical_frequency_v1"):

    # Log convergence diagnostics as metrics
    mlflow.log_metric("max_rhat", max_rhat)
    mlflow.log_metric("min_ess_bulk", min_ess_bulk)
    mlflow.log_metric("n_divergences", n_div)
    mlflow.log_metric("n_districts", n_districts_model)
    mlflow.log_metric("sigma_district_mean",
                      float(trace.posterior["sigma_district"].mean()))
    mlflow.log_metric("grand_mean_rate",
                      float(np.exp(trace.posterior["alpha"].mean())))

    # Log the full posterior as an ArviZ netCDF artefact.
    # This lets you reload the posterior for any downstream analysis
    # without re-running MCMC.
    trace.to_netcdf("/tmp/posterior_module06.nc")
    mlflow.log_artifact("/tmp/posterior_module06.nc", "posteriors")

    # Log the results table
    results.write_csv("/tmp/credibility_results_module06.csv")
    mlflow.log_artifact("/tmp/credibility_results_module06.csv", "results")

    print(f"MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")
```

**What this does:** Logs the convergence diagnostics, the full posterior (as a netCDF file), and the results table to MLflow. The netCDF file means you can reload the posterior at any time without re-running MCMC — important because MCMC takes minutes but loading from disk takes seconds.

### Unity Catalog for credibility-weighted estimates

Credibility-weighted factor tables belong in Unity Catalog with the same governance as any other rating artefact:

```python
from datetime import date

RUN_DATE = str(date.today())
MODEL_NAME = "hierarchical_freq_module06_v1"

# Hard gate: do not write unconverged posteriors downstream.
# A model that has not converged produces estimates that are wrong in ways
# that are hard to detect. Fail loudly rather than propagate bad estimates.
if max_rhat > 1.01 or n_div > 0:
    raise ValueError(
        f"Convergence failure: max_rhat={max_rhat:.4f}, divergences={n_div}. "
        "Credibility estimates not written to Unity Catalog."
    )

results_out = results.with_columns([
    pl.lit(MODEL_NAME).alias("model_name"),
    pl.lit("hierarchical_poisson").alias("model_type"),
    pl.lit(RUN_DATE).alias("run_date"),
    pl.lit(max_rhat).alias("max_rhat"),
    pl.lit(n_div).alias("n_divergences"),
])

(
    spark.createDataFrame(results_out.to_pandas())
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("main.pricing.module06_credibility_estimates")
)

print(f"Written {results_out.height} rows to main.pricing.module06_credibility_estimates")
```

**What this does:** Writes the credibility-weighted estimates to a Delta table in Unity Catalog. The hard gate (`raise ValueError` if convergence fails) ensures a model with bad convergence cannot write downstream. This is not defensive programming — it is the only way to prevent silent errors from propagating through a pricing pipeline.

---

## Part 16: Limitations — for regulatory documentation

Regulatory and internal governance presentations require honest documentation of methodology limitations. Here are the limitations for this module's methods, in the precision expected for an FCA Consumer Duty or IFoA actuarial standards filing.

**1. Bühlmann-Straub assumes Normal errors in the working scale.** The BLUP property holds without distributional assumptions, but the quality of the EPV and VHM estimates depends on the data not being severely non-Normal. Very skewed loss rates — common when large bodily injury claims drive thin-cell volatility — inflate the EPV estimate and produce Z values that are too low (over-shrinkage). Log-transforming before applying B-S mitigates this but does not eliminate it for extreme distributions.

**2. MCMC is slow and requires diagnostic skill.** A pricing team accustomed to GLMs that complete in under a minute will find Bayesian MCMC unfamiliar. R-hat diagnostics, divergence checks, and ESS requirements are new concepts. Budget time for this learning curve and compute for exploratory runs before committing to Bayesian methods in a production pipeline.

**3. The Poisson likelihood assumes observed variance equals expected variance.** If claim count data are overdispersed (variance greater than mean) — which is common in insurance due to unobserved heterogeneity (driving behaviour, actual vehicle condition, claims propensity) — the Poisson model understates uncertainty for individual districts. The Negative Binomial likelihood is more appropriate when overdispersion is detected. In PyMC: replace `pm.Poisson` with `pm.NegativeBinomial` and add a dispersion parameter `alpha_disp = pm.HalfNormal("alpha_disp", sigma=1.0)`.

**4. Hierarchical models with few groups are poorly identified at the top level.** If you have 6 affinity schemes, the estimate of sigma — the between-group standard deviation — is itself highly uncertain. The Bayesian posterior for sigma will be wide, propagating uncertainty into all scheme-level credibility factors. This is correct behaviour, but it can be uncomfortable for stakeholders expecting precise answers. Report the posterior for sigma explicitly, not just the point estimate.

**5. Bühlmann-Straub groups must be approximately exchangeable.** The method assumes group hypothetical means are independently drawn from the same prior distribution. For UK postcodes, this assumption is violated: districts in the same urban area are correlated. KT1, KT2, and KT3 share road networks, parking conditions, crime rates, and flood risk. They are not independent draws from a common prior. Flat B-S underestimates the between-group variance in this case and over-shrinks correlated groups. The two-level hierarchical model (Exercise 2) is the correct fix for structured geographic data.

**6. Structural parameters are estimated from the same data used for credibility weighting.** This is empirical Bayes: v and a are estimated from the data and then used in the credibility formula. This introduces a subtle upward bias in Z values — the data that inform the structural parameters are the same data used to evaluate credibility, violating the independence assumption in the theoretical derivation. For portfolios with 50+ groups across 5+ years, this bias is negligible. For fewer groups or shorter histories, it may be material.

**7. The log-Normal random effect distribution may not match the true distribution of district risks.** The Bayesian hierarchical model places a Normal distribution on the log-rates, corresponding to a log-Normal distribution on the rates themselves. If the true distribution has heavier tails — for example, a handful of genuinely extreme-risk micro-areas — the log-Normal will over-shrink the extreme districts. The mitigation is to check the shrinkage plot for evidence of excessive shrinkage on extreme districts and to run a prior sensitivity check with a heavier-tailed Student-t prior: replace `pm.Normal("u_district_raw", ...)` with `pm.StudentT("u_district_raw", nu=4, ...)`.

**8. Credibility estimates are an input to the pricing decision, not the decision itself.** In production, credibility-weighted district relativities are typically capped and floored before entering the rating structure — for example, no district moves more than 50% above or below the portfolio mean in a single review cycle. This prevents any single district driving a loss-making or uncompetitive premium. If you implement capping and flooring, document it explicitly: FCA Consumer Duty requires that the methodology choices — including constraints on outputs — are recorded and explainable.

**9. Posterior predictive validation is mandatory, not optional.** A hierarchical model that passes MCMC convergence diagnostics (R-hat, ESS, no divergences) can still be misspecified. Run posterior predictive checks — simulate datasets from the posterior and compare to observed data — before presenting results. Exercise 3 covers this in detail. A model that fails posterior predictive checks is misspecified regardless of convergence, and misspecified models produce systematically biased credibility factors.

---

## Summary: what just happened

This module covered the two principled methods for handling thin cells in UK insurance pricing.

**Bühlmann-Straub credibility:**
- Derives the optimal blend of group experience and portfolio mean from first principles
- Three parameters: grand mean (mu), within-group variance (EPV, v), between-group variance (VHM, a)
- Credibility factor Z = w / (w + K), K = v/a
- Works in log-rate space for Poisson/multiplicative frameworks (`log_transform=True`)
- Fast, auditable, regulatory-friendly
- Correct for one grouping variable with many groups

**Bayesian hierarchical models:**
- Probabilistic programming with PyMC 5
- Non-centered parameterisation eliminates funnel geometry
- Three mandatory convergence checks: R-hat < 1.01, ESS > 400, zero divergences
- Outputs posterior distributions, not just point estimates — 90% credible intervals on every district rate
- Correct for multiple groupings, few groups, or when uncertainty quantification is required
- Slower, requires diagnostic skill, but more honest about uncertainty

**The two-stage workflow most UK teams need:**
1. CatBoost or GLM for main rating factors
2. Bühlmann-Straub on district-level O/E residuals

The four exercises extend this: Exercise 1 applies B-S to severity data, Exercise 2 builds a two-level geographic hierarchy in PyMC, Exercise 3 validates the Bayesian model with posterior predictive checks, and Exercise 4 formats results for a pricing committee presentation.

---

*This module can also be worked via the burning-cost `credibility` library (github.com/burning-cost/credibility) and `bayesian-pricing` library (github.com/burning-cost/bayesian-pricing) for a higher-level API. The implementations in this tutorial are the reference — they are what those libraries do internally, made explicit so you can understand and adapt them.*
