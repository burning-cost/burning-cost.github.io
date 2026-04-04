---
layout: post
title: "Pricing SME Cyber with Python - What Works and What Doesn't"
date: 2026-03-31
categories: [techniques, libraries, cyber]
tags: [cyber-insurance, SME, frequency-severity, zero-inflated, LightGBM, GPD, lognormal, FCA, PRC, VCDB, insurance-distributional, insurance-severity, insurance-frequency-severity, insurance-fairness, python, open-source]
description: "UK SME cyber is growing at 13% CAGR with 2.8% standalone penetration. The data is terrible, the commercial tools are six figures, and the best published frequency model explains 13% of variance. Here is an honest account of what open-source Python can and cannot do."
author: burning-cost
---

UK SME cyber insurance has 2.8% standalone penetration. The average SME cyber premium is around £3,715 per year. The average ransomware demand against a UK SME is around £147,044 - roughly 40 times the premium. The market is growing at 13% CAGR and the FCA named cyber insurance as a 2026 review priority in March of this year.

The tools that pricing teams can actually use are either £50k+/year black boxes or a security-posture scanner that produces a letter grade, not a loss distribution.

This post is about what can be built with open data, what the published research actually shows, and where the genuine limits are. We are not going to pretend UK calibration exists when it doesn't.

---

## The best published research, graded honestly

The most interesting recent paper is Guo, Quan, and Zhang (arXiv:2507.08193, July 2025): "Entity-Specific Cyber Risk Assessment Using InsurTech-Empowered Risk Factors." It is genuinely novel in one respect and oversold in another.

**What they do:** They assemble 39,636 firm-year observations from seven open sources (Privacy Rights Clearinghouse, VCDB, CISSM, Have I Been Pwned, SecurityWeek NLP extractions, and others), then enrich each firm with 527 entity-level features from the Carpe Data commercial InsurTech platform. Features span firmographics, customer review metrics, operational proximity scores, and licensing categories. They run LightGBM with multi-output regression to predict annual incident frequency across five types: Privacy Violation, Data Breach, Fraud/Extortion, IT Error, and Others.

**The headline number:** Average correlation coefficient of 0.36 between predicted and actual incident frequency. That is r2 of roughly 0.13. The model explains about 13% of variance in incident counts.

For rare event prediction that is not surprising. But you should hold it in your head while evaluating any vendor claiming "AI-powered" cyber scoring. A correlation of 0.36 is honest and typical. The paper's "InsurTech-empowered" framing makes it sound like a bigger advance than those numbers warrant.

**What the paper does not do:** There is no severity modelling. The authors acknowledge this explicitly and defer it to future work. There is no financial loss data in the assembled dataset at all - just incident counts. There is no zero-inflated model, despite the authors acknowledging that the data is "heavily skewed and dominated by zeros." And the training data is entirely US-sourced. The Privacy Rights Clearinghouse exists because of HIPAA breach notification requirements; there is no UK equivalent until the Cyber Security and Resilience Bill passes.

**What is genuinely novel:** The 527-feature entity-level enrichment methodology is the first paper to systematically do this at scale. More practically useful: their regressor chain analysis shows that incident types are conditionally independent given firm features. Privacy Violation frequency tells you almost nothing about Ransomware frequency once you condition on what the firm looks like. This simplifies the modelling architecture considerably - you can fit five independent models rather than a joint one.

---

## The data problem

This is the binding constraint. Not model architecture. Data.

The open datasets available for cyber modelling are PRC (US, ~9,000 records in the research archive, HIPAA-biased toward healthcare), VCDB (open on GitHub at vz-risk/VCDB, ~10,000 incidents, US-heavy but with partial international coverage), and CISSM from the University of Maryland (~14,000 incidents). Combined and deduplicated, you get roughly 25,000 usable firm-year observations for frequency modelling.

None of them contain financial loss amounts. The only dataset with actual insurance dollar losses is Advisen's Cyber Loss Data, which requires an NDA and has roughly 20 staff maintaining it daily. It is inaccessible for open-source work.

PRC records a count of individuals affected - "records exposed" - which is commonly used as a severity proxy. Using records-exposed as a proxy for financial severity is defensible for a demo but has real problems: a GDPR enforcement penalty scales differently from a US class action; healthcare record exposure is vastly over-represented due to HIPAA; and the UK regulatory penalty structure (ICO fines) is simply different from US state AG settlements.

For UK practitioners: there is currently no UK-specific open dataset for entity-level cyber frequency or severity modelling. ICO publishes aggregate GDPR Article 33 breach statistics, not individual incident records. NCSC Annual Review gives sector-level statistics. If the Cyber Security and Resilience Bill passes and mandates breach reporting, this changes - but that is a future data strategy item, not a today problem.

---

## The commercial tool gap

The market splits into two buckets with a void between them.

**Lloyd's/reinsurance tools:** CyberCube (San Francisco, founded 2018) and Kovrr (Tel Aviv/New York) are the actuarially sophisticated end. CyberCube is widely used for Lloyd's Realistic Disaster Scenario accumulation reporting. Kovrr runs Monte Carlo simulation on FAIR-framework models and produces financial loss distributions per entity. Both are six-figure annual subscriptions. Both target syndicates and global reinsurers, not SME pricing teams.

**Security posture tools:** BitSight and SecurityScorecard scan external IP footprint, patch cadence, and vulnerability exposure. They produce scores (A-F or 0-900, depending on vendor). Hiscox, Beazley, and Aviva use them for SME cyber questionnaire augmentation. They cost in the £15-50k/year range for insurer use. They are not actuarial models and have not been calibrated to insurance losses at published precision. They are a credit score for cyber hygiene, not a loss distribution.

**The void:** A reproducible, actuarially-grounded, entity-level frequency-severity model for SME cyber that a pricing actuary can inspect, validate, and adapt. Something that clearly separates the frequency mechanism from the severity mechanism, uses a proper zero-inflated count model, and runs on open data. This is what we are building with [insurance-distributional](/insurance-distributional/), [insurance-severity](/tools/), and [insurance-frequency-severity](/insurance-frequency-severity/).

---

## What Python can actually do: frequency

The correct model structure for entity-level cyber frequency is zero-inflated. Most firms have zero incidents in any given year. Standard regression conflates the structural zeros (the firm is simply not a realistic target) with the count-generating process for exposed firms. Guo et al. acknowledge this and then ignore it, using MSE regression. That is a methodological gap.

The right two-stage structure is:

```
Stage 1: P(incident > 0 | firm features) - binary classifier
Stage 2: E[count | count > 0, firm features]  - Poisson or NegBin regression
```

[insurance-distributional](/insurance-distributional/) already has `zip.py` (zero-inflated Poisson) and `ZeroInflatedTweedieGBM` from So and Valdez (arXiv:2406.16206, 2024). These map directly to cyber incident frequency. The conditional independence finding from Guo et al. means you fit one ZI-Poisson per incident type independently - five models, not one joint model.

What you cannot do with open data is UK calibration. The frequency parameters you estimate from PRC+VCDB are US-data estimates. They can serve as priors if you have any UK portfolio data, but they are not directly applicable to UK SME pricing without adjustment.

---

## What Python can actually do: severity

The empirical picture on cyber severity is clearer than on frequency, though the data access problem is if anything worse.

Eling, Ibragimov, and Ning (Insurance: Mathematics and Economics, 2025, DOI: 10.1016/j.insmatheco.2025.103196) analyse three cyber loss databases and find that malicious incident losses have increased in severity since 2018 while negligent incident losses have declined. They use Frechet-based change-point detection and inverse probability weighting for selection bias. The tail remains heavy throughout - consistent with a GPD upper tail.

The consensus on distribution shape: log-normal fits small-to-medium losses well; GPD fits the tail; a composite lognormal-GPD splice outperforms either alone. Bee (arXiv:2505.22507, 2025) develops a lognormal-GPD bridge via exponential transition with four free parameters identifiable by iterative MLE.

[insurance-severity](/tools/) already has `TruncatedGPD` (GPD with policy-limit truncation correction, directly relevant to cyber), a composite distributions module with lognormal-GPD splice, and `TailVariableImportance` for identifying which firm characteristics drive extreme losses.

The severity tooling is essentially ready. The binding constraint is data: with no financial loss amounts in PRC or VCDB, we use records-exposed as a proxy. That gives you a severity model that is directionally correct and clearly caveated. If you have Advisen access, substitute it - the API is designed for that.

One additional constraint worth naming: entity-level severity models cannot price the tail risk from systemic events. NotPetya produced over $10 billion in aggregate losses across hundreds of entities simultaneously. MOVEit hit hundreds of organisations through a single vendor vulnerability. These events are not predictable from individual firm features. Entity-level scoring addresses the attritional layer only. For accumulation and peak-zone analysis, you need a different approach - we covered SIR contagion modelling for cyber accumulation in a separate March 2026 post.

---

## The joint model and fairness audit

The frequency-severity join is handled in [insurance-frequency-severity](/insurance-frequency-severity/) via `joint.py` and `copula.py`. For cyber, the frequency and severity of a given incident type are plausibly correlated (larger, more-targeted firms have both more incidents and higher losses per incident), which warrants explicit dependence modelling rather than naive multiplication of marginals.

The fairness question matters specifically for SME cyber because industry sector is the most predictive firmographic variable after firm size. Sector as a predictor is likely to produce adverse outcomes for SMEs in hospitality, retail, and social care relative to financial services - even controlling for actual security posture. Under FCA Consumer Duty, algorithmic underwriting outcomes need to be tested for proxy discrimination. [insurance-fairness](/insurance-fairness/) provides `proxy_detection.py` (which will flag sector as a proxy for variables correlated with protected characteristics) and `audit.py` for the full outcome audit workflow.

This is not hypothetical. If you are underwriting SME cyber and using a GBM model trained on US data with sector as a feature, you should run the fairness audit before you deploy.

---

## What we claim and don't claim

This is an open-source, actuarially-grounded entity-level cyber frequency-severity model calibrated to PRC and VCDB. That is a smaller claim than vendor marketing but a larger claim than currently exists in the open-source space.

We do not claim:
- Better discrimination than CyberCube or Kovrr (we have no access to their holdout data)
- Applicability to UK pricing without UK recalibration
- That entity-level scoring addresses systemic cyber risk
- That a correlation of 0.36 on US data translates to useful pricing lift on UK SME books

We do claim:
- First open-source ZI-Poisson cyber frequency model with entity-level firmographic features
- Auditable severity model with explicit composite distribution and data-source caveats
- Fairness audit workflow addressing FCA Consumer Duty requirements
- A reference implementation that UK pricing actuaries can inspect, adapt, and recalibrate when UK loss data becomes available

The FCA review is exploratory, not enforcement. The Cyber Security and Resilience Bill is pending. The UK cyber data infrastructure may improve significantly over the next two to three years. The modelling framework we are publishing now will be worth more when it can be recalibrated against UK-specific incident data. Build the infrastructure before you need it.

---

## References

- Guo, Quan, Zhang (2025). "Entity-Specific Cyber Risk Assessment Using InsurTech-Empowered Risk Factors." arXiv:2507.08193
- Eling, Ibragimov, Ning (2025). "The Changing Landscape of Cyber Risk: An Empirical Analysis of Loss Severity and Tail Dynamics." *Insurance: Mathematics and Economics*. DOI: 10.1016/j.insmatheco.2025.103196
- Bee (2025). Lognormal-GPD bridge model. arXiv:2505.22507
- So and Valdez (2024). "Boosted Trees for Zero-Inflated Counts." arXiv:2406.16206
- arXiv:2502.04421 (Feb 2025). Ransomware risk prioritisation via LLM CTI extraction
- UK DSIT. "Insuring Resilience: The State of SME Cyber Insurance." August 2025
- FCA. "Insurance Priorities 2026." March 2026
- VCDB: github.com/vz-risk/VCDB
- PRC: privacyrights.org/data-breaches
