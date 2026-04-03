---
layout: post
title: "Entity-Level Cyber Risk Scoring: What the Open Data Actually Tells Us"
date: 2026-04-03
categories: [pricing, cyber]
tags: [cyber, cyber-risk-scoring, entity-scoring, cyber-frequency, zero-inflated-poisson, lightgbm, VCDB, PRC, privacyrightsclearinghouse, CyberCube, Kovrr, BitSight, SecurityScorecard, SME, FCA-2026, cyber-insurance, open-data, InsurTech, Guo-Quan-Zhang, arXiv-2507-08193, Eling-2025, lognormal-GPD, severity, conditional-independence, multilabel, UK-insurance, feature-engineering]
description: "Everyone is buying CyberCube or Kovrr. The open data suggests entity-level cyber frequency scoring is feasible — but the honest numbers (r = 0.36, US-only, no severity) tell a different story from the marketing. Here is what you can actually build today and where the gaps remain."
author: burning-cost
math: true
---

The FCA named cyber insurance as a 2026 review item in its March priorities publication. The stated aim is diagnostic — improving the regulator's own understanding of risks, opportunities, and barriers — but it will push cyber up every pricing actuary's agenda regardless. A UK SME cyber market growing at 13% CAGR, with standalone penetration sitting at 2.8% across all businesses and ransomware demands averaging roughly 40 times the annual premium, is a market with a severe underwriting data problem dressed up as a market opportunity.

The two default responses to that problem are: buy CyberCube or Kovrr (six-figure subscriptions, inaccessible below roughly £50m GWP), or use BitSight or SecurityScorecard as an underwriting input (security posture scores, not calibrated to loss data, not actuarial outputs). We think there is a third option that most pricing teams have not explored: the open incident datasets are larger and richer than commonly assumed, and a paper published in July 2025 by Guo, Quan, and Zhang (arXiv:2507.08193) demonstrates a feature engineering methodology that is genuinely replicable without a commercial data contract.

But the honest numbers matter. The correlation between predicted and actual incident frequency in Guo et al. is 0.36. The data is US-only. The model does not touch severity. Before describing what is achievable, we should be precise about what is not.

---

## What the commercial tools actually do

The commercial landscape splits into two categories that are frequently conflated.

**CyberCube and Kovrr** are actuarial modelling platforms. CyberCube's Account Cypher produces single-risk scoring for Lloyd's syndicates; its Portfolio Manager handles accumulation and PML estimation at the portfolio level. Kovrr uses Monte Carlo simulation with a FAIR framework adaptation to generate financial loss distributions per entity. These are the tools Lloyd's syndicates use for Realistic Disaster Scenario reporting. They model both attritional and catastrophic layers. They cost six figures and their methods are not published.

**BitSight and SecurityScorecard** are security rating products. They scan external IP footprint, patch cadence, vulnerability exposure, and similar signals to produce scores — BitSight on a 0–900 scale, SecurityScorecard A through F. Hiscox, Beazley, and Aviva all use these for SME questionnaire augmentation. Subscriptions are cheaper (£15k–£50k/year for insurer use). The critical limitation: neither company has published a calibration of their scores against insurance loss data. They predict security posture. Whether security posture predicts insurance losses, and with what accuracy, is not demonstrated in open literature.

The marketing from both categories implies more than the evidence supports. "AI-powered cyber scoring" is doing marketing work, not statistical work.

The gap between them — a reproducible, actuarially-grounded, entity-level frequency model that a pricing actuary can inspect, validate, and adapt — is what the open data approach addresses. Partially.

---

## What Guo, Quan, and Zhang actually built

The paper assembles 39,636 firm-year observations by combining seven open incident databases:

| Source | Records | Coverage |
|--------|---------|----------|
| Privacy Rights Clearinghouse | 9,000+ | US, 2005–2019, HIPAA-driven |
| VERIS Community Database (VCDB) | 10,000+ | Open JSON; 2,500 attributes per incident |
| CISSM Cyber Events Database | 14,000+ | University of Maryland |
| University of Queensland | 1,000+ | 2004–2019 |
| Have I Been Pwned | 800+ | Credential exposures |
| Critical Infrastructure Ransomware | 2,000+ | US critical infrastructure |
| SecurityWeek (NLP-extracted) | 11,000+ | Press reports, structured by NLP |

After deduplication and entity alignment across sources: 39,636 observations at the firm-year level. The scale is larger than any previous open-data cyber study we are aware of.

The methodologically interesting contribution is not the dataset assembly — it is the feature enrichment. The authors join each entity to 527 features from the Carpe Data commercial InsurTech platform, spanning eight categories: business information, customer review metrics, risk characteristics, firmographics, licensing categories, classification segment tags, proximity scores, and composite index scores.

Standard cyber risk papers use industry sector dummies and firm size. Carpe Data adds signals like online review sentiment, operational activity proxies, and geographic proximity to high-risk facilities. Most of these signals are available, at some approximation, from Companies House filings, Google Places, and similar open sources. Not at 527 features — but directionally replicable.

The modelling task is multilabel: five incident types (Privacy Violation, Data Breach, Fraud/Extortion, IT Error, Others), each treated as a separate frequency regression. The best-performing model is LightGBM wrapped in a MultiOutputRegressor. The comparison also includes Random Forest variants and regressor chains.

---

## The numbers you need to read carefully

| Metric | Value |
|--------|-------|
| Classification weighted-F1 | 0.6439 |
| Hamming loss | 0.1600 |
| Frequency average correlation | 0.3623 |
| Frequency aRMSE | 0.3477 |

The correlation of 0.36 is the number that matters. It implies $r^2 \approx 0.13$: the model explains roughly 13% of variance in incident frequency. For rare-event prediction this is not unusual — the data-generating process is hostile to supervised learning because most firm-years have zero incidents, the events are heterogeneous, and the signal-to-noise ratio is low by construction.

The authors are honest about this. The framing of the paper as "InsurTech-empowered risk assessment" implies stronger results than 0.36 delivers. What the paper demonstrates is that entity-level feature engineering meaningfully outperforms naive baselines — not that the model is accurate in any absolute sense. Those are different claims.

Three further limitations are not prominently flagged in the abstract but are critical for any UK application:

**No severity modelling.** The paper explicitly defers severity to future work, citing data availability constraints. A frequency model alone is not a pricing model. Pure premium is frequency times severity; 0.36 correlation on the frequency component tells you nothing about what the average loss will be conditional on an incident occurring.

**No zero-inflated structure.** The dataset is heavily zero-dominated — most firm-years have no incidents. Standard MSE regression conflates the zero-generating mechanism (firm is simply not a target) with the count-generating mechanism (firm is a target but the realised count this year is zero). The actuarially appropriate structure is a two-stage model: first predict $P(\text{incident} > 0)$ via binary classification, then predict $E[\text{count} \mid \text{incident} > 0]$ via Poisson or negative binomial regression. Guo et al. acknowledge the zero inflation but do not implement the split. This is a methodological gap that would materially affect the frequency estimates for low-risk firms.

**US data only.** The Privacy Rights Clearinghouse relies on HIPAA breach notification requirements. There is no UK equivalent at the incident level. ICO GDPR Article 33 breach notifications exist but are published in aggregate, not per-entity. The NCSC Annual Review gives sector-level statistics, not entity-level data. A model trained on US healthcare-skewed data and applied to UK SMEs is doing out-of-population inference in a domain where regulatory, sectoral, and organisational differences between the two markets are material.

---

## The one finding worth taking to an architecture discussion

The paper's regressor chain analysis — comparing models that assume incident types are independent with models that explicitly model inter-type correlations — finds no significant correlation among incident type frequencies after conditioning on firm features. Privacy Violation and Ransomware are conditionally independent given the firmographic covariates.

This is a practically useful empirical finding. It means you can fit five separate, independent models — one per incident type — rather than a joint model. For a pricing team that wants to build modular, interpretable underwriting models for different cyber perils, this is good news: the architecture is simpler than it might have been.

The caveat is that this finding is on US data and may not transfer. UK cyber incidents have a different sectoral mix. But conditional independence is the kind of property that tends to be driven by the firm-features used as conditioning variables, not by country-specific incident dynamics. We would expect it to hold qualitatively in a UK context.

---

## The severity data problem is real

No open dataset contains financial loss amounts for cyber incidents. This is not an observation about the quality of Guo et al. — it is a structural fact about the data landscape.

The PRC records the number of individuals affected (records exposed). This is the standard severity proxy in open-data cyber research. Its limitations for UK insurance are threefold:

1. Records exposed does not translate to financial loss. US class action exposure scales with records exposed in ways that UK GDPR regulatory penalties (ICO enforcement) do not replicate.
2. Healthcare records are over-represented due to HIPAA mandates. A UK model trained on this data will systematically over-weight healthcare sector signals.
3. UK regulatory penalties scale differently from US state attorney general settlements. The ICO's maximum fine under UK GDPR (£17.5m or 4% of global turnover) bears no direct relationship to US class action damages.

Advisen's Cyber Loss Data is the only published source with actual insurance loss amounts. It requires a non-disclosure agreement and is not available for open-source calibration. For a UK insurer with its own claims data, the path is different: your own loss experience is the calibration dataset, and the open-data model provides the frequency structure.

Eling, Ibragimov, and Ning (Insurance: Mathematics and Economics, 2025) offer a useful empirical anchor for severity parameterisation without Advisen. Their analysis — using Fréchet change-point detection — finds that malicious cyber losses (ransomware, hacking) have heavier and growing tails since 2018, while negligent incidents (accidental disclosure, lost hardware) have lighter and more stable tails. The divergence began around 2018 and has continued. For severity modelling, this finding argues for fitting separate lognormal-GPD composite distributions by incident type, with the GPD tail parameters for malicious incidents materially heavier than for negligent ones.

The average UK SME ransomware demand of £147,044 (DSIT "Insuring Resilience", August 2025) against an average premium of £3,715 illustrates why severity is the commercially important component. A frequency model that explains 13% of variance is a useful underwriting signal. A pricing model that ignores the severity distribution is not a pricing model.

---

## What you can realistically build with open data

A UK insurer with access to Companies House, the VCDB (available on GitHub at github.com/vz-risk/VCDB), the CISSM database, and standard firmographic sources could build the following:

**A multilabel incident classifier** trained on VCDB + CISSM, with firmographic features from Companies House filings (revenue, employee count, SIC code, registered address, filing status). The Carpe Data signals from Guo et al. (online review sentiment, proximity scores, composite indices) are not freely available, but SIC code granularity, company age, filing regularity, and revenue trajectory from Companies House provide a meaningful subset. Expected discrimination: lower than Guo et al.'s 0.36 correlation — perhaps 0.25–0.30 — because the feature set is shallower.

**A zero-inflated Poisson frequency model** on top of the classifier. The two-stage structure — $P(\text{incident} > 0 \mid x)$ then $E[\text{count} \mid \text{incident} > 0, x]$ — is the methodologically correct approach the Guo paper omits. For an entity with £2m revenue, 15 employees, and an IT services SIC code, the model would output a probability of at least one incident per year and a conditional expected count. The conditional independence finding justifies building five separate models, one per incident type.

**A severity framework using records exposed as a proxy**, with an explicit assumption document noting that this is a records-exposed proxy trained on US data, that UK severity will differ for reasons specific to UK regulatory penalties and UK claims practice, and that the model should be recalibrated against the insurer's own loss experience before deployment. This is not a complete severity model — it is a structured placeholder with honest uncertainty bounds.

**A Consumer Duty–compatible audit trail.** The FCA's cyber review has a fairness dimension: algorithmic underwriting is subject to Consumer Duty (PRIN 2A) whether the input data comes from BitSight or from a Companies House-enriched classifier. Industry sector is a proxy variable with potential indirect discrimination implications under Equality Act 2010. Any cyber scoring model needs to demonstrate that sector-based differentiation reflects genuine risk rather than correlated demographic characteristics. The same proxy detection workflow that applies to motor postcode rating applies to cyber sector rating.

---

## What the commercial tools address that open data cannot

We should be direct about where the commercial tools earn their price.

**Catastrophic accumulation.** Entity-level scoring — ours or anyone else's — cannot price the tail risk from systemic events. NotPetya cost an estimated $10bn+ in aggregate losses. The MOVEit vulnerability affected hundreds of entities simultaneously. For a Lloyd's syndicate or reinsurer writing cyber aggregate, the entity-level frequency model is the wrong layer. Accumulation modelling requires epidemic/contagion structures — network propagation models calibrated to shared software dependency data — not entity-level ML. CyberCube's Portfolio Manager addresses this layer. An open-data entity-level model does not.

**Loss data calibration.** CyberCube and Kovrr have access to insurance loss data that no open-source project can replicate. Their entity-level scores, whatever the underlying model, are calibrated against actual claims outcomes. Open-data models trained on incident counts (not insurance losses) are predicting a different quantity.

**UK coverage.** Commercial tools have UK incident data that the open datasets lack. For UK SME pricing, the US-data limitation of open-source models is a fundamental problem that cannot be engineered around without better UK data. The UK Cyber Security and Resilience Bill — pending legislation that would mandate breach reporting — would materially improve this position if passed. Until then, UK incident data at the entity level does not exist in usable form.

---

## The data watch item

The UK Cyber Security and Resilience Bill would require UK organisations to report cyber incidents in a structured, comparable format. If passed in anything like its current form, it creates the UK equivalent of what HIPAA notifications do for the US data landscape: a national incident database with entity-level records and metadata.

The ITIF argued in December 2025 that mandatory breach reporting could unlock UK cyber market growth by reducing insurer information asymmetry. We think this assessment is right directionally. The specific mechanism is that a UK incident database would allow calibration of frequency models against UK firms, UK regulatory penalties, and UK claims practice — eliminating the primary methodological limitation of the Guo et al. approach for UK application.

Until the Bill passes, UK insurers should build the modelling infrastructure against US data, document the extrapolation assumptions explicitly, and recalibrate against their own loss experience. Waiting for better data is reasonable; building nothing while waiting is not.

---

## The practical recommendation

For a UK insurer or MGA writing SME cyber today:

If your GWP is above roughly £50m: consider whether CyberCube or Kovrr is cost-effective given your Lloyd's obligations and need for PML estimation. Entity-level scoring is part of their proposition but not the most important part — the accumulation modelling is the differentiating value at that scale.

If your GWP is below £50m and you are using BitSight or SecurityScorecard as an underwriting signal: you are using security posture scores, not actuarial models. That is legitimate as a supplementary underwriting input. It is not a substitute for a frequency model. The two address different questions.

If you want to build an internal entity-level frequency model: the Guo et al. methodology is the right starting point. The VCDB and CISSM are freely available. Companies House filings provide a workable firmographic feature set. The zero-inflated two-stage structure is the correct statistical architecture. A correlation of 0.25–0.30 on held-out UK data is a realistic expectation — informative, not transformative.

The honest version of the model's value proposition is not "replaces commercial tools" but "provides a quantitative prior that is better than the underwriter's gut and worse than Advisen-calibrated scores." For a pricing team that currently applies flat sector loadings to cyber, that is a meaningful improvement. For a Lloyd's syndicate with a six-figure CyberCube contract, it is not obviously better.

We think the former is the larger and less-served market.

---

## References

Guo, Y., Quan, Z. and Zhang, X. (2025). Entity-Specific Cyber Risk Assessment Using InsurTech-Empowered Risk Factors. arXiv:2507.08193.

Eling, M., Ibragimov, R. and Ning, N. (2025). The Changing Landscape of Cyber Risk: An Empirical Analysis of Loss Severity and Tail Dynamics. *Insurance: Mathematics and Economics*. DOI: 10.1016/j.insmatheco.2025.103196.

UK DSIT (2025). Insuring Resilience: The State of SME Cyber Insurance. August 2025.

FCA (2026). Insurance Priorities 2026. March 2026.

VERIS Community Database: github.com/vz-risk/VCDB

Privacy Rights Clearinghouse: privacyrights.org/data-breaches

---

**Related posts:**
- [Stochastic SIR Models for Cyber Portfolio Accumulation](/2026/03/26/stochastic-sir-cyber-insurance-portfolio/) — portfolio-level epidemic modelling for systemic cyber events; entity-level scoring and accumulation modelling address different layers
- [Truncation-Corrected GPD: Unbiased Tail Index Under Policy Limits](/2026/03/20/your-gpd-is-lying-because-your-claims-are-truncated/) — EVT for cyber severity if you have your own loss data
- [Composite Severity Regression: Getting the Tail Right Without Throwing Away the Body](/2026/03/13/insurance-composite/) — lognormal-GPD composite for malicious vs negligent incident severity
