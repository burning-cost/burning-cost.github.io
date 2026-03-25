---
layout: page
title: "Work with Us"
description: "We build open-source pricing tools used by UK insurance teams. We also help teams implement them."
permalink: /work-with-us/
---

We build open-source Python tools for UK personal lines insurance pricing. The toolkit covers 34 libraries: causal inference, fairness auditing, conformal prediction, model governance, severity modelling, constrained rate optimisation, and more. If you've found the libraries useful and need help getting them into production, that's what this page is about.

---

## What we offer

**Pricing model review**

We review frequency and severity model stacks against FCA Consumer Duty and PRA SS1/23 requirements. The output is a written assessment with ranked issues — covering temporal cross-validation correctness, calibration by segment, IBNR handling, feature engineering decisions, and governance documentation. Delivered within two weeks.

*What this looks like in practice*

A two-week sprint. We work through your existing model stack, cross-validation setup, and governance documentation. Deliverables: a written technical assessment with ranked findings, a corrected cross-validation specification, and calibration diagnostics by key segment. We work directly with your pricing team — you can ask questions throughout, not just at the end. One working session mid-sprint to align on priorities, one to present findings.

---

**Databricks deployment**

We deploy and configure the open-source toolkit on your Databricks workspace, connected to your data. This includes MLflow experiment tracking, champion/challenger deployment with the audit trail ICOBS 6B.2 requires, and scheduled monitoring jobs. Typically three to five days.

*What this looks like in practice*

A three-to-five day embedded engagement. Deliverables: the full toolkit installed and configured on your Unity Catalog, champion/challenger pipelines wired to your existing models, monitoring jobs running on a schedule, and a handover document your team can maintain without us. We work alongside your pricing and data engineering team — the goal is that they own it when we leave, not that they depend on us.

---

**Team training**

Practical Python training for pricing teams, using your data and your models — not a generic course. Based on our eight-module curriculum: CatBoost for frequency and severity, SHAP relativities extraction, temporally correct cross-validation, constrained rate optimisation, and governance documentation. Typically two days on-site.

*What this looks like in practice*

Two days on-site. Deliverables: worked examples using your own data, a reusable notebook set your team keeps, and a follow-up Q&A session two weeks later. We adapt the curriculum based on a pre-training conversation — if your team already has a strong CatBoost foundation but struggles with SHAP interpretation or temporal validation, we weight it accordingly. Maximum eight participants; smaller groups work better.

---

## Why us

- **We built the tools.** The libraries you'd be implementing — [insurance-fairness](/insurance-fairness/), [insurance-monitoring](/insurance-monitoring/), [insurance-conformal](/insurance-conformal/), [insurance-governance](/insurance-governance/), [insurance-optimise](/insurance-optimise/) and the rest — are ours. We know where they work well and where they need care.
- **Open-source means no vendor lock-in.** Everything we deploy is on PyPI under a permissive licence. You are not dependent on us to run it, update it, or understand it.
- **UK personal lines focus.** We are not a generic data science consultancy that has pivoted to insurance. Motor and home pricing is what we do. The regulatory context, the data quirks, the actuarial constraints — we do not need these explained to us.
- **Regulatory fluency.** FCA Consumer Duty, PRA SS1/23, ICOBS 6B.2 — we understand what these require technically, not just in principle. If your board is asking for explainability documentation or fairness monitoring, we know what adequate looks like.

---

## How it works

**1. Discovery call**

A 30-minute call to understand your situation — what you have, what is blocking you, and whether we are the right fit. No obligation, no sales process.

**2. Scoping document**

We send a one-page scoping document within two working days: what we will do, what we will deliver, how long it will take, and what we need from your side. Fixed scope, fixed price.

**3. Engagement**

We get on with it. You have direct access to us throughout — not a project manager relaying messages. We deliver, hand over, and make sure your team can run it independently.

---

## Who this is for

UK personal lines pricing teams — motor and home primarily, though the libraries cover the relevant methods for most non-life lines. If you want someone who can sit in a technical pricing session, review your cross-validation setup, and give you a direct opinion on whether your elasticity estimate is credible, that is what we do.

We are not the right choice if you want a large engagement with a multi-team delivery structure. We work in small, focused bursts on clearly defined technical problems.

---

## Contact

Email [pricing.frontier@gmail.com](mailto:pricing.frontier@gmail.com?subject=Consulting%20enquiry) with "Consulting enquiry" in the subject line. Tell us what you are working on, what the blocker is, and what a successful outcome looks like. We reply within one working day.
