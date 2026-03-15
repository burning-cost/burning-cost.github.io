---
layout: page
title: "Work with Us"
description: "We built 36 open-source pricing libraries used by UK insurance teams. If you need help getting them into production, we can help."
permalink: /work-with-us/
---

We built 36 open-source Python libraries for UK personal lines pricing — everything from causal elasticity estimation and fairness auditing to Databricks-native deployment pipelines. Together they get 8,500+ monthly PyPI downloads and have been benchmarked on Databricks serverless compute.

Most of the teams that find us via GitHub can run the code. The harder problem is usually one of three things: knowing which library to use for your actual problem, adapting the examples to your data schema and modelling conventions, and getting sign-off from a pricing committee or model risk function that doesn't speak Python.

That's where we come in.

---

## What we help with

**Implementation support**

You have the library. You have the data. But the library was written against synthetic motor data and your home book has a different structure — tied agents, mid-term adjustment complexity, or claims that are grouped at household level. We help you wire it in correctly, including the edge cases the README doesn't cover.

**Model validation**

We can run an independent technical review of a pricing model built on our libraries — or any GBM-based pricing model in Python. This covers: temporal cross-validation correctness, calibration by segment, IBNR handling, and whether the feature engineering decisions hold up under actuarial scrutiny. Output is a structured validation report suitable for a model risk committee or PRA SS1/23 submission.

**Databricks workflow setup**

We build all of our tooling to run on Databricks. If your team is migrating from a local Python environment or a legacy SAS/R workflow to Databricks, we can help you set up the pipelines: model training notebooks, MLflow experiment tracking, champion/challenger deployment with the audit trail that ICOBS 6B.2 requires, and scheduled monitoring jobs.

**Team training**

Half-day or full-day sessions for pricing teams. Not generic Python training — pricing-specific: how to use CatBoost correctly for insurance frequency and severity models, how to extract multiplicative factor tables using SHAP that an actuary can review, how to run temporally correct cross-validation, and how to build the governance documentation a model risk function will actually accept.

---

## Typical engagement

Most engagements run two to ten days. We work remotely; we are UK-based and generally available within normal UK working hours.

We don't have a rate card on this page — the right scope varies too much by problem. Send us an email with a brief description of what you're trying to do and we'll tell you honestly whether we can help and roughly what's involved.

---

## Who this is for

UK personal lines pricing teams — motor and home primarily, though the libraries cover the relevant methods for most non-life lines. We are practitioners, not consultants with a deck. If you want someone who can sit in a technical pricing session, review your cross-validation setup, and give you a direct opinion on whether your elasticity estimate is credible, that's what we do.

We are not the right choice if you want a large engagement with a multi-team delivery structure. We work in small, focused bursts on clearly defined technical problems.

---

## Contact

**Email:** [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com)

Tell us what you're working on, what the blocker is, and what a successful outcome looks like. We'll reply within one working day.
