# Module 1: Databricks for Pricing Teams

**Part of: Modern Insurance Pricing with Python and Databricks**

---

## What this module covers

Most large UK insurers are running Databricks. Most pricing teams are not getting close to what it can do for them.

This module is not a generic Databricks tutorial. It is specifically about what pricing actuaries need: how to structure your data so models are reproducible, how to track what data generated a given set of relativities, how to replace the "run the notebook manually on Thursday" pattern with a scheduled job that runs reliably and fails loudly, and what the FCA's Consumer Duty requirements mean in concrete infrastructure terms.

By the end, you have a working Databricks environment configured for a pricing team, a Unity Catalog schema with a versioned motor claims dataset, and a scheduled Workflow that runs automatically and writes results to Delta.

---

## Prerequisites

- Comfortable working with tabular insurance data (policy extracts, claims bordereaux — you know what earned exposure means)
- Basic Python (you do not need to be a software engineer)
- Databricks access — Free Edition works for all exercises except Workflows, which requires a paid workspace
- No prior Databricks experience required

---

## Estimated time

3–4 hours for the tutorial plus exercises. The notebook runs end-to-end in under 15 minutes on a small single-node cluster.

---

## Files

| File | Purpose |
|------|---------|
| `tutorial.md` | Main written tutorial — read this first |
| `notebook.py` | Databricks notebook — Unity Catalog setup, Delta tables, time travel, basic profiling |
| `exercises.md` | Four hands-on exercises with full solutions |

---

## What you will be able to do after this module

- Explain to a data engineer exactly how you want your pricing data structured in Unity Catalog, with the correct table design and partitioning
- Set up a Unity Catalog schema for your team's modelling work with column-level security for PII
- Write and query Delta tables with time travel — essential for reproducing any model run against the exact data snapshot it used
- Load data from the sources pricing teams actually use: CSV exports from Emblem/Radar, SAS datasets, legacy SQL databases (requires network access from Databricks to the source — check with your platform team if the source is on-premises)
- Build a data quality check notebook that validates claim counts, flags negative exposures, and catches the common corruption patterns in bordereaux data
- Set up a Databricks Workflow that runs your pipeline on a schedule and writes results to Unity Catalog — replacing manual notebook execution
- Describe what audit trail the FCA's Consumer Duty requires and which Databricks features satisfy each requirement

---

## Why this is worth your time

The pricing teams that use Databricks well are not the ones with the best data engineers. They are the ones where the actuaries and analysts understand the platform well enough to ask for what they need and to build the parts that only they can build.

The infrastructure you set up in this module is the foundation that every subsequent module depends on. Module 2 trains GLMs on data loaded from Delta. Module 3 trains CatBoost models. Module 4 extracts SHAP relativities and writes them back to Unity Catalog. Module 5 builds out the full scheduled pipeline from raw data to Radar export. None of that works without the setup covered here.

The FCA compliance angle is real, not box-ticking. Consumer Duty's requirements on pricing fairness and explainability have specific, concrete implications for how you store data, version models, and document decisions. Databricks' Unity Catalog lineage and MLflow Model Registry are the right tools for this. We show you how to use them for that purpose, not in the abstract.
