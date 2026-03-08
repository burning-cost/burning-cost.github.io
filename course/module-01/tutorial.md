# Module 1: Databricks from Scratch

This module is for pricing analysts who have never used Databricks. If you have opened it before, skimmed the documentation and come away confused, or been told "we're moving to Databricks" with no further instruction, this is the right starting point.

We assume you know Excel and probably some R or Python. We assume you use Emblem or Radar. We do not assume you know anything about cloud computing, distributed systems, or data engineering.

By the end of this module you will have a working Databricks environment, a first notebook, and a real dataset loaded and ready to query. Everything else in the course builds on this foundation.

---

## Part 1: What is Databricks and why should you care

### The problem it solves

Most pricing teams work like this: the model lives in a collection of R or Python scripts on someone's laptop. When it needs to run, that person runs it. The training data is a CSV somewhere on SharePoint - possibly the right version, possibly not. The last model run was six months ago and nobody is quite sure what version of the data it used. When someone asks "what would happen if we changed the NCD relativity?" the answer involves finding the right person, hoping their environment still works, and waiting.

This is not a criticism. It is a natural consequence of tools designed for individual use being stretched to support team workflows.

Databricks is a cloud platform designed to fix exactly this. It provides:

- **A shared compute environment** - everyone runs code on the same infrastructure, with the same library versions, without needing to set up anything on their own machine
- **Notebooks that live in the cloud** - open them from any browser, share them with a link, no local installation required
- **Versioned data storage** - the data you trained the model on is preserved, with a version number, and you can query it exactly as it was on any date
- **Scheduled pipelines** - the model can run itself on the 15th of each month, email you if something goes wrong, and not require anyone to be at their desk

The reason pricing teams specifically are moving to Databricks is regulatory pressure. Consumer Duty requires insurers to demonstrate that pricing is consistent with the intended risk segmentation and does not produce unfair outcomes. That demonstration requires being able to show, for any given model: what data it was trained on, when it ran, what it produced, and who approved it. A collection of scripts on individual laptops cannot show any of that. Databricks can.

### What it replaces in your workflow

| What you do now | What Databricks replaces it with |
|---|---|
| Scripts on a laptop | Notebooks in a shared workspace |
| CSV files on SharePoint | Delta tables with version history |
| "Email me the output" | Shared notebooks and scheduled exports |
| Manual model runs | Workflows (automated pipelines) |
| "Which data did we use?" | Delta time travel - query any past version |
| Library conflicts between team members | Shared cluster with fixed library versions |

It does not replace Radar or Emblem. The output of a Databricks pipeline - factor tables, GLM relativities, pure premiums - still feeds into Radar for deployment. Databricks is where you build and run the analysis. Radar is where the output lives.

### Free Edition vs paid

Databricks offers a Free Edition at `databricks.com/try-databricks`. It is real Databricks - the same notebooks, the same Python environment, the same Delta Lake storage format. The limitations are:

- Single small cluster (no choice of size)
- No Unity Catalog (the governance layer - covered in Part 7)
- No scheduled Workflows
- No multi-user workspace (it is a single-person environment)

Free Edition is sufficient for everything in Parts 2 through 6 of this module, and for most of Module 2. When we reach features that require a paid workspace, we will say so explicitly.

For a team production environment, your insurer will have a paid workspace on Azure or AWS - set up by the platform or data engineering team. This module uses Free Edition so you can follow along without needing to ask for access to a shared environment.

---

## Part 2: Setting up your Free Edition account

### Creating an account

Go to `https://www.databricks.com/try-databricks` in your browser and select **Free Edition** (not the 14-day trial).

You will see a sign-up form. Fill in your name, email, and a password. Use your work email if you intend to eventually connect this to your organisation's Databricks workspace - it makes things simpler later. If you just want to experiment, a personal email is fine.

After submitting the form, you will receive a verification email. Click the link in it. If it does not arrive within five minutes, check your spam folder.

Once verified, you will be asked to choose a cloud provider: AWS, Azure, or Google Cloud. For Free Edition this choice does not affect what you can do - pick whichever you like. We use AWS in the screenshots throughout this course, but the interface is identical on Azure and Google Cloud.

### What you see after first login

After logging in you land on the Databricks home screen. It looks like this:

- **Left sidebar** - the main navigation. The icons from top to bottom are: Home, Workspace, Repos, Data, Compute, Workflows, and Settings. We will use most of these.
- **Central panel** - this changes depending on what you have selected in the sidebar. On first login it shows a "Get started" panel with shortcuts to create a notebook, import data, and so on.
- **Top bar** - your username, a help icon, and a search box.

The most important things to understand at this stage:

**Workspace** is where your notebooks live. Think of it like a folder structure in File Explorer, but in the cloud. When you create a notebook, it goes into your Workspace.

**Compute** is where you manage clusters - the actual computers that run your code. Without a running cluster, notebooks cannot execute. We will cover this properly in Part 6.

**Data** is where you browse and manage datasets. For Free Edition users this shows you DBFS (Databricks File System) - the storage layer for your workspace.

### Starting a cluster before anything else

Before a notebook can run, a cluster must be running. On Free Edition, you get one cluster.

Click **Compute** in the left sidebar. If you have never created a cluster, it shows an empty list and a button that says **Create Compute** (or **Create cluster** - the label has changed between versions). Click it.

On the cluster creation screen you will see various configuration options. On Free Edition, most of these are fixed - you cannot choose the instance type or size. What you can choose is the Databricks Runtime version. Select the latest **LTS ML** version (something like "14.3 LTS ML" or similar - LTS means Long Term Support, ML means it includes machine learning libraries). If you are not sure which to pick, choose the one with "LTS ML" in the name.

Click **Create Compute**. The cluster takes 3-5 minutes to start. The status shows "Pending" and then "Running". While it is starting, move on to the next section.

---

## Part 3: Your first notebook

### Creating a notebook

In the left sidebar, click **Workspace**. You will see a folder structure. There may be a folder called "Users" with your email address as a subfolder - click into it.

Click the **Add** or **+** button (it may be in the top right, or appear when you hover over a folder). Select **Notebook**.

A dialog box appears. Give the notebook a name - something like "module-01-getting-started" is fine. The default language is Python, which is what we want. Click **Create**.

The notebook opens. You will see:

- A toolbar at the top with buttons for running cells, connecting to a cluster, and so on
- A cluster selector, probably showing "Detached" if the cluster is not yet running
- One empty cell in the main area, ready for code

If the cluster selector shows "Detached", click it and select your cluster from the dropdown. Once connected, the selector shows the cluster name in green.

### What a cell is

A notebook is made of **cells**. Each cell contains code (or markdown text). You run cells one at a time, or all at once.

Click on the empty cell in your notebook. Type:

```python
print("Hello from Databricks")
```

To run the cell, press **Shift+Enter**. The cell runs and the output appears immediately below it:

```
Hello from Databricks
```

A new empty cell appears below, ready for the next thing you want to try.

This is the core of how notebooks work. You write code in a cell, run it, see the output, write more code in the next cell. The Python session is continuous - variables defined in one cell are available in all subsequent cells, as long as you run them in order.

If you close the notebook and reopen it later, you will need to run the cells again from the top - the Python session does not persist between sessions.

### Basic Python reminder

If you use Python regularly, skip this section. If you mostly use R or Excel, here is the minimum you need for the rest of this module.

**Variables and arithmetic:**

```python
# Variables
claim_count = 42
exposure_years = 1.0
claim_frequency = claim_count / exposure_years

print(f"Claim frequency: {claim_frequency}")
```

Run this (Shift+Enter). The output is `Claim frequency: 42.0`.

The `f"..."` is an f-string - it lets you put variable values inside a string by wrapping them in `{}`.

**Lists and loops:**

```python
accident_years = [2020, 2021, 2022, 2023, 2024]

for year in accident_years:
    print(f"Processing accident year {year}")
```

**Importing libraries:**

```python
import polars as pl
import matplotlib.pyplot as plt
```

The `import` statement loads a library. Libraries need to be installed before you can import them - more on that below.

### Installing a library

Databricks notebooks use a special command for installing libraries: `%pip`. The `%` prefix tells Databricks that this is a notebook magic command, not Python code.

In a new cell, type:

```python
%pip install polars
```

Run it. You will see a stream of output as pip downloads and installs Polars. At the end it says:

```
Note: you may need to restart the Python kernel to use updated packages.
```

After a `%pip install`, you need to restart the Python kernel before you can use the newly installed library. Run this in the next cell:

```python
dbutils.library.restartPython()
```

This restarts the Python interpreter. Any variables you defined before this point are gone - the session resets. This is normal and expected after installing a library. From now on, put all `%pip install` commands at the very top of your notebook, before any other code, so you only need to restart once.

To install multiple libraries at once:

```python
%pip install polars catboost matplotlib
```

Note: in normal Python development outside Databricks you would use `uv add` or `pip install` from a terminal. Inside Databricks notebooks, `%pip` is the right tool. `uv` is not available inside notebook cells.

### Your first DataFrame with Polars

Create a new cell and run this:

```python
import polars as pl

# A simple DataFrame - like a spreadsheet but in Python
df = pl.DataFrame({
    "accident_year": [2020, 2021, 2022, 2023, 2024],
    "claim_count":   [412,  389,  441,  398,  427],
    "exposure":      [8500, 8750, 9100, 9200, 9400],
})

df
```

The last line - just `df` with no print statement - displays the DataFrame as a formatted table in the notebook output. You will see five rows, three columns, with the data you defined.

Try some basic operations in the next cell:

```python
# Add a calculated column: claim frequency per policy-year
df = df.with_columns(
    (pl.col("claim_count") / pl.col("exposure")).alias("claim_freq")
)

df
```

The `.with_columns()` method adds new columns. `pl.col("claim_count")` refers to the existing column, and `.alias("claim_freq")` gives the new column a name.

### A simple plot

Databricks notebooks display matplotlib plots inline, just like Jupyter.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    [str(y) for y in df["accident_year"].to_list()],
    df["claim_freq"].to_list(),
    color="steelblue",
)
ax.set_xlabel("Accident year")
ax.set_ylabel("Claim frequency")
ax.set_title("Motor claim frequency by accident year")
ax.set_ylim(0, 0.07)

plt.tight_layout()
plt.show()
```

Run this. A bar chart appears in the cell output. The `plt.show()` call is what triggers the display.

---

## Part 4: Working with data in Databricks

### Uploading a CSV file

In real pricing work, your data will come from a database or a file upload. Here is how to upload a CSV to your Databricks workspace.

First, create a small CSV on your local machine. Open a text editor (or use Excel and save as CSV) and create a file called `sample_policies.csv` with this content:

```
policy_ref,inception_year,exposure_years,claim_count,area_band,ncd_years,vehicle_group,driver_age
POL00001,2023,1.0,0,B,5,15,42
POL00002,2023,0.75,1,A,0,22,27
POL00003,2023,1.0,0,C,3,18,55
POL00004,2023,0.5,0,D,2,31,34
POL00005,2023,1.0,1,B,1,19,29
POL00006,2023,1.0,0,A,5,12,61
POL00007,2023,0.25,0,E,4,25,38
POL00008,2023,1.0,0,C,3,17,45
POL00009,2023,1.0,2,B,0,28,22
POL00010,2023,1.0,0,A,5,11,58
```

Save the file.

Now, in Databricks, click **Data** in the left sidebar. You will see options for browsing tables and files.

Look for a button or option that says **Add data** or **Upload file** - the exact label varies slightly by Databricks version. Click it.

You will be prompted to upload a file. Drag and drop your `sample_policies.csv`, or use the file browser to find it.

After uploading, Databricks will show you a preview of the file and suggest a table name. For now, note the path where the file was uploaded - it will be something like `/FileStore/tables/sample_policies.csv` or similar. We will read it from that path.

Alternatively, you can create the data directly in the notebook without uploading a file - which is more reliable for this exercise:

```python
import polars as pl

# Create sample policy data directly in the notebook
policies = pl.DataFrame({
    "policy_ref":     ["POL00001", "POL00002", "POL00003", "POL00004", "POL00005",
                       "POL00006", "POL00007", "POL00008", "POL00009", "POL00010"],
    "inception_year": [2023] * 10,
    "exposure_years": [1.0, 0.75, 1.0, 0.5, 1.0, 1.0, 0.25, 1.0, 1.0, 1.0],
    "claim_count":    [0, 1, 0, 0, 1, 0, 0, 0, 2, 0],
    "area_band":      ["B", "A", "C", "D", "B", "A", "E", "C", "B", "A"],
    "ncd_years":      [5, 0, 3, 2, 1, 5, 4, 3, 0, 5],
    "vehicle_group":  [15, 22, 18, 31, 19, 12, 25, 17, 28, 11],
    "driver_age":     [42, 27, 55, 34, 29, 61, 38, 45, 22, 58],
})

policies
```

### Basic data exploration

These are the first things you should do with any new dataset:

```python
# How many rows and columns?
print(f"Shape: {policies.shape}")  # (rows, columns)

# What are the column names and types?
print(policies.dtypes)
print(policies.columns)
```

```python
# First few rows
policies.head(5)
```

```python
# Summary statistics for numeric columns
policies.describe()
```

The `.describe()` output shows count, mean, standard deviation, min, and max for each numeric column. For a real claims dataset, the minimum claim count should never be negative and the exposure should always be positive - these are the first sanity checks to run.

```python
# Count claims by area band
policies.group_by("area_band").agg(
    pl.col("claim_count").sum().alias("total_claims"),
    pl.col("exposure_years").sum().alias("total_exposure"),
    pl.len().alias("policy_count"),
).with_columns(
    (pl.col("total_claims") / pl.col("total_exposure")).alias("claim_freq")
).sort("area_band")
```

This is a basic one-way analysis: claim frequency by area band. It is the same calculation you would do in a spreadsheet pivot table, but it scales to millions of rows without slowing down.

### Saving as a Delta table

Delta Lake is the data storage format that makes Databricks different from just a Python notebook environment. A Delta table is like a database table, but stored in your cloud workspace, with the ability to query historical versions.

To save a Polars DataFrame as a Delta table, we go via Spark (the distributed computing engine underneath Databricks):

```python
# Convert Polars to a Spark DataFrame, then write as Delta
spark.createDataFrame(policies.to_pandas()).write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("policies_sample")
```

There are three things happening here:

1. `policies.to_pandas()` - converts the Polars DataFrame to a pandas DataFrame (an intermediate step)
2. `spark.createDataFrame(...)` - converts the pandas DataFrame to a Spark DataFrame
3. `.write.format("delta").mode("overwrite").saveAsTable("policies_sample")` - writes it to a Delta table

The `spark` variable is already available in any Databricks notebook - you do not need to import or create it.

You should see a confirmation message. The table is now saved and will persist even if you close the notebook or the cluster shuts down.

**What is Delta Lake?** It is a storage format built on top of Parquet (a compressed, columnar file format). What makes it special is that every write creates a new version of the table, and you can query any version at any point in time. If someone accidentally deletes data, or if you need to prove what the data looked like when you ran a model three months ago, Delta time travel gives you that. We will use this extensively in Module 2.

### Reading the table back

```python
# Read the Delta table back into a Polars DataFrame
policies_from_delta = spark.table("policies_sample").toPandas()
df_back = pl.from_pandas(policies_from_delta)

df_back.head()
```

Or, using SQL directly in a notebook cell with the `%sql` magic:

```sql
%sql
SELECT area_band, COUNT(*) as policy_count, SUM(claim_count) as total_claims
FROM policies_sample
GROUP BY area_band
ORDER BY area_band
```

The `%sql` magic lets you run SQL directly in a notebook cell. The result displays as a table. This is useful for quick data queries without writing Python.

---

## Part 5: Organising your work

### Notebooks, Repos, and Files

Databricks gives you three places to put code. Understanding the difference saves headaches later.

**Workspace notebooks** are the default. When you created a notebook in Part 3, it went into your Workspace. These notebooks are stored in Databricks itself - they are not files on your computer. They are fine for exploration and learning. The limitation is that they are not version-controlled: if you delete a cell by accident, there is no undo history beyond the session. If you overwrite work, it is gone.

**Repos** are Git-backed. A Repo in Databricks is a connection to a Git repository (on GitHub, Azure DevOps, or Bitbucket). When you edit a notebook in a Repo, you can commit and push your changes just like you would from VS Code or a terminal. This is the right place for any code that matters - model training notebooks, data preparation scripts, production pipelines. We cover setting up a Repo below.

**Files** are arbitrary files in the workspace filesystem - config files, YAML, JSON, CSV. Not notebooks. Less common, but useful for storing configuration alongside your repo code.

The rule of thumb: **exploratory work in Workspace, anything production-ready in a Repo**.

### Connecting a Git repository

You will need a GitHub account (free). If you do not have one, go to `https://github.com` and sign up.

Create a new repository on GitHub:
1. Go to `https://github.com/new`
2. Give it a name, e.g. `databricks-pricing-practice`
3. Set it to Private
4. Check "Add a README file"
5. Click **Create repository**

Copy the repository URL - it looks like `https://github.com/yourusername/databricks-pricing-practice.git`.

Back in Databricks:
1. Click **Repos** in the left sidebar
2. Click **Add Repo** (or **+** > **Repo**)
3. Paste the GitHub URL
4. Databricks will ask for your GitHub credentials - use a Personal Access Token, not your password. To create one: GitHub > Settings > Developer settings > Personal access tokens > Tokens (classic) > Generate new token. Give it `repo` scope. Copy the token and paste it into Databricks.
5. Click **Create Repo**

Databricks clones the repository. You will see its contents appear in the Repos panel. You can now create notebooks inside this repo, edit them, and commit the changes back to GitHub.

For the rest of this course, we recommend creating new notebooks inside your Repo rather than in the Workspace. It gives you a safety net and is the professional habit to build.

### Folder structure for a pricing project

Here is a sensible structure for a pricing project repo. You do not need to create all of this now - this is the shape to grow into.

```
motor-pricing/
    notebooks/
        01_data_preparation.py
        02_frequency_model.py
        03_severity_model.py
        04_factor_extraction.py
    src/
        motor/
            features.py
            validation.py
    config/
        model_params.yaml
        base_levels.yaml
    README.md
```

- `notebooks/` - numbered in execution order. Each notebook does one thing.
- `src/` - reusable Python code (functions used across multiple notebooks).
- `config/` - configuration files. Putting model parameters in YAML files rather than hardcoding them in notebooks means you can change parameters without touching the code.

For exploratory work - trying things, one-off analyses, checking something quickly - use your Workspace rather than the repo. The repo should contain code that you would be comfortable showing to a colleague or a regulator.

---

## Part 6: Compute basics

### What a cluster actually is

When you run code in a Databricks notebook, it does not run on your laptop. It runs on a **cluster** - a virtual computer (or collection of computers) in the cloud.

The cluster has:
- A specific amount of RAM and CPU cores
- A specific version of Databricks Runtime (which determines the Python version, pre-installed libraries, and so on)
- A start-up time (3-5 minutes, because a virtual machine is being provisioned)
- A cost (on paid tiers - charged by the hour)

When nothing is running and the cluster sits idle, it wastes money. For this reason, clusters are configured to **auto-terminate** after a period of inactivity - typically 30-60 minutes. When a cluster auto-terminates, your notebooks still exist and your Delta tables still exist, but you will need to restart the cluster (or attach to a different running one) before you can run code again.

### Free Edition compute

On Free Edition you have one cluster with fixed specifications. You cannot change the size. It auto-terminates after 2 hours of inactivity. When it terminates, you can restart it from the Compute panel - click the cluster name, then **Start**.

The Free Edition cluster is more than sufficient for learning: loading data, running models on small datasets, exploring the Databricks interface. It is not fast enough for training CatBoost on 500,000 policies - that kind of work requires a paid workspace.

### Starting and stopping a cluster

**To start:** Click **Compute** in the left sidebar. If the cluster shows status "Terminated", click the cluster name and then **Start**. Wait 3-5 minutes for it to reach "Running" status.

**To stop:** Click **Compute**, click the cluster name, click **Terminate** (or the stop button). On Free Edition this is rarely necessary since it auto-terminates anyway, but on paid tiers you should stop clusters when you are not using them.

**To check if a cluster is running from a notebook:** Look at the cluster selector at the top of the notebook. If it shows the cluster name in green, it is running and attached. If it shows "Detached" or the name in grey, either the cluster is stopped or the notebook is not connected to it.

### What happens when a cluster auto-terminates

If a cluster terminates while you have a notebook open:
- The notebook code and outputs that have already run are preserved
- Variables and in-memory data are lost (you will need to rerun the cells)
- Delta tables on disk are preserved
- The next time you run a cell, Databricks will prompt you to restart the cluster (this takes 3-5 minutes)

This is one reason to save important results to Delta tables rather than keeping them as in-memory DataFrames. A Delta table persists across cluster restarts. A Python variable does not.

### Installing libraries: per-notebook vs cluster-level

There are two ways to install libraries on Databricks:

**Per-notebook install using `%pip`** - this is what we used in Part 3. The library is installed for the current session. When the cluster restarts, you will need to run the `%pip install` cell again. This is fine for learning and for notebooks that are not part of automated pipelines.

```python
%pip install polars catboost matplotlib
```

**Cluster-level install** - the library is installed on the cluster itself, available to all notebooks, and persists across cluster restarts. On Free Edition, go to **Compute**, click the cluster, go to the **Libraries** tab, click **Install new**, select PyPI, and enter the package name.

For a team environment, cluster-level installs are better - everyone gets the same library version without each notebook needing its own install step. The Databricks Runtime ML versions (which we recommended in Part 2) pre-install many common libraries including scikit-learn, MLflow, and SHAP.

For this course, `%pip install` at the top of each notebook is fine.

---

## Part 7: Unity Catalog (preview)

You will not use Unity Catalog in Free Edition - it is a paid feature. But understanding what it is now means Part 2 of the course makes sense when you get there.

### What Unity Catalog is

Every Delta table you created in Parts 4 and 6 has a simple name: `policies_sample`. In a team environment with multiple projects, product lines, and data domains, simple names cause problems. Which `claims` table? Is it raw or processed? Which team owns it?

Unity Catalog is Databricks' governance layer. It organises all tables in a three-level hierarchy:

```
catalog . schema . table
```

For example: `pricing.motor.claims_exposure`

- **Catalog** - the top level. Usually corresponds to an environment (production, development) or a business unit.
- **Schema** - the middle level. Usually corresponds to a product line or analytical domain (motor, home, governance).
- **Table** - the specific dataset.

When you reference a table as `pricing.motor.claims_exposure`, you know exactly what it is, who owns it, and where it sits in the organisation's data hierarchy.

### Why it matters for pricing teams

Beyond organisation, Unity Catalog provides:

- **Lineage tracking** - it automatically records which tables fed into which other tables and which notebooks created them. After a model run you can look up a table in the Catalog UI and see the full chain: which source tables it came from, which notebook processed it, which other tables depend on it. This is the audit trail that Consumer Duty requires.

- **Access control** - you can grant read access to `pricing.motor.claims_exposure` to the data science team and write access only to the pricing actuaries, without any manual file permission management.

- **Column-level security** - sensitive columns (policyholder names, dates of birth) can be masked so analysts who do not need PII see redacted values, while those with appropriate access see the real data.

On Free Edition, you work with simple table names (no catalog or schema prefix). When you move to a paid workspace, your platform team will have set up a catalog and schema for your team - they will tell you the prefix to use.

### The catalog hierarchy in practice

In a paid workspace, the first thing to understand is what catalog and schema your team uses. Common patterns:

```
pricing.motor.claims_exposure          <- Motor claims data
pricing.motor.model_relativities       <- Model outputs
pricing.home.claims_exposure           <- Home claims data
pricing.governance.model_run_log       <- Audit log
```

You do not need to create these - in most organisations the platform team sets up the catalog structure and grants your team access. What you do need is the prefix, which goes at the start of every table reference in your notebooks.

This is covered in detail in Module 2, once you are working in a team environment.

---

## What's next

You now have a working Databricks environment. You know how to create notebooks, run Python code, load data, and save it as a Delta table. You understand what a cluster is and how to manage it.

**Module 2: GLMs for Frequency and Severity** - trains Poisson frequency and Gamma severity models on real claims data loaded into Delta tables. Introduces PySpark for reading large datasets, Polars for analysis, and MLflow for experiment tracking. By the end of Module 2 you have a working GLM pipeline that produces factor tables.

**Module 3: Gradient Boosted Models with CatBoost** - replaces the GLM frequency model with a CatBoost model. Covers hyperparameter tuning, cross-validation designed for insurance data, and model comparison against the GLM benchmark.

The infrastructure you set up here - the workspace, the cluster, the Delta tables - is used in every subsequent module. When Module 2 reads from `pricing.motor.claims_exposure`, it is reading from a table structured exactly as described in Part 4 and Part 7.
