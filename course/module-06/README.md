# Module 6: Credibility & Bayesian Pricing - The Thin-Cell Problem

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

You have 200 policies in postcode area SW14 and 3 claims this year. Is 1.5% the true frequency, or is it noise from a cell that happens to have had a bad year?

Every pricing team has this problem. GLMs handle it partially through parameter constraints. GBMs, left unchecked, overfit it completely - min_data_in_leaf is a blunt instrument, not a principled solution.

This module covers two approaches to the thin-cell problem: classical Bühlmann-Straub credibility weighting and full Bayesian hierarchical models. We show where they agree, where they diverge, and how to choose between them. The Bühlmann-Straub calculation is implemented directly in NumPy (about 20 lines). The Bayesian model is written explicitly in PyMC rather than hidden behind a library wrapper. All code runs on Databricks.

---

## Prerequisites

- Comfortable with GLMs for frequency and severity modelling - you know what random effects are and why you might want them
- Have read Module 4 (SHAP Relativities) or understand the concept of segment-level rating factor estimates
- Basic familiarity with Python and Polars
- Databricks access (Databricks Free Edition is sufficient for the exercises)
- Willing to engage with some mathematics - we do not handwave the Bühlmann-Straub estimators

You do not need prior exposure to Bayesian statistics or PyMC. We explain what MCMC is doing and why non-centered parameterization matters.

---

## Estimated time

4–5 hours for the tutorial plus exercises. The notebook's classical credibility section runs in under a minute. The Bayesian section takes 5–15 minutes depending on cluster size - MCMC is not fast.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial - read this first |
| `notebook.py` | Databricks notebook - full workflow on synthetic data |
| `exercises.md` | Four hands-on exercises with full solutions |

---

## Libraries

The notebook requires only standard open-source packages available on PyPI:

```bash
uv pip install pymc arviz polars numpy matplotlib mlflow
```

No private or suspended GitHub repositories. PyMC 5.x and ArviZ are the core Bayesian dependencies. On Databricks, install in your notebook's first cell - the ML runtime (DBR 14.x or later) includes most of these already.

---

## What you will be able to do after this module

- Apply Bühlmann-Straub credibility to frequency and severity data, with correct exposure weighting - using a self-contained NumPy implementation you understand line by line
- Interpret the structural parameters v (EPV), a (VHM), and K, and explain what they tell you about your portfolio's heterogeneity
- Understand why Bühlmann-Straub is an empirical Bayes method and how it connects to hierarchical Bayesian models
- Fit a hierarchical Poisson model directly in PyMC, with non-centered parameterization for convergence
- Produce a shrinkage plot showing observed rates versus credibility-weighted estimates - the most persuasive diagnostic for explaining credibility to a pricing committee
- Extract credibility factors from Bayesian posteriors and compare them to classical Bühlmann-Straub Z values
- Decide when classical credibility is sufficient and when full Bayesian is warranted
- Store credibility-weighted factor tables in Unity Catalog with MLflow tracking of the posteriors

---

## Why this is worth your time

The thin-cell problem is not a niche concern. It affects every factor table you produce: vehicle make/model (thousands of levels, most sparsely populated), occupation (hundreds of levels), postcode (9,500 sectors across Great Britain). The question is never whether to use credibility - you are always implicitly making a choice between pooling, no pooling, or partial pooling. The question is whether you make that choice deliberately and with the right mathematics.

Bühlmann-Straub is the most defensible tool for a large fraction of UK pricing problems and is currently done in Excel universally. The implementation in this module is self-contained - no dependency on any private library. You can read it, audit it, and explain every line to a Chief Actuary.

For cases where Excel credibility is not enough - hierarchical geographic structures, crossed effects, or genuine uncertainty quantification - full Bayesian gives you something classical credibility cannot: a posterior distribution over each segment's true expected loss rate. That distribution is what you need to make defensible, documented decisions on thin cells - and it is substantially stronger evidence than a point estimate when regulators ask how you priced a cell with 8 claims.
