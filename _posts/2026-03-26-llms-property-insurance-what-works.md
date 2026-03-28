---
layout: post
title: "LLMs for Property Insurance: What They Can Actually Do"
date: 2026-03-26
categories: [property-insurance, machine-learning]
tags: [llm, gpt-4, nlp, property-risk, underwriting, claims, survey-processing, text-extraction, uk-insurance, pricing]
description: "LLMs can extract structured data from surveys, flag non-standard construction in loss adjuster notes, and rate categorical variables at scale. They cannot reliably assess claim liability. This is what the evidence actually shows."
---

Large language models are not primarily a pricing tool for property insurance. They are an operational tool. The pricing discussion — can we use LLMs to score property risk from images — belongs in a different category to what LLMs are genuinely good at in the insurance workflow: reading documents that humans currently read, extracting structured fields from unstructured text, and processing categorical data at a scale that human experts cannot match.

We think the operational applications are more immediately deployable than the imagery applications and are underinvested by UK insurers. This post is about those applications, where the evidence is solid, where it is weak, and why the claims use case demands more caution than the underwriting use case.

This is the third post in our series on computer vision and LLMs for property insurance. The [first post](/2026/03/26/computer-vision-property-insurance-what-actually-works/) covered the vendor landscape and UK data infrastructure. The [second post](/2026/03/26/open-source-cv-tools-property-risk/) went through the open-source CV stack.

---

## What LLMs demonstrably do: categorical feature enrichment

The most rigorous published evidence for LLM use in insurance pricing comes from the Actuaries Institute of Australia, in work published in 2024–2025 on LLM feature engineering for insurance pricing.

The core result: LLMs can assign risk-relevant ratings to large categorical variable lists at a fraction of the cost and time of human expert review. A motor insurer might have 3,000 distinct vehicle model descriptions in their portfolio — far too many for an actuary to assign relativities to individually, so most models collapse rare levels or use reference tables that go stale. An LLM can process all 3,000 levels in seconds, producing structured ratings along specified dimensions: vehicle year, performance characteristics, size category, repair cost tier. The authors distinguish "factual" ratings (engine size is a verifiable fact) from "subjective" ratings (is this model likely to attract high-risk drivers) and flag that the subjective ratings encode the LLM's training data biases.

The direct property insurance extension: **occupation and property use classification**. UK home insurance applications contain free-text fields — "describe your occupation", "is the property used for any business purpose", "any unusual features". These are currently processed by rules engines or junior underwriters. An LLM can classify these consistently and at scale: flagging business use that increases liability exposure, identifying holiday let or AirBnB declarations, catching descriptions that suggest non-standard construction not elsewhere declared. This is not a research project. It is a configuration task on a deployed LLM API.

---

## Text extraction from surveys and loss adjuster reports

Property surveys and loss adjuster reports are rich sources of risk information that are almost entirely unstructured in most UK insurers' data systems. A typical commercial property survey report might be 15–25 pages of narrative text produced by a chartered surveyor: roof type and condition observations, electrical installation age, plumbing type, occupancy details, security provisions. That information currently sits in a PDF, searched only when a human reads it.

LLMs can extract structured fields from these documents reliably for a defined schema. A prompt asking GPT-4o to extract {roof_material, roof_condition_rating, electrical_installation_age, heating_type, structural_concerns_flagged} from a survey report will produce useful output. The quality degrades for:

- Fields where the source document is ambiguous or contradictory
- Numerical estimates buried in narrative ("approximately 40 years old" vs a precise date)
- Technical assessments that require domain knowledge to interpret correctly ("early signs of sulphate attack in the mortar" is a structural concern; an LLM may or may not recognise its insurance significance)

For loss adjuster reports post-claim, the extraction task is similar but the stakes are higher. We cover that separately below.

**EPC ROOF_DESCRIPTION extraction** is a specific application we would prioritise for UK home insurers. The 30 million EPC records available from MHCLG contain a ROOF_DESCRIPTION free-text field that describes insulation type and condition. These range from "flat, no insulation (assumed)" to "pitched, 300mm loft insulation" to "thatched". A simple classifier — LLM-based or even a regex — can convert these into structured risk flags: thatched construction (specialist underwriting required), flat roof present (higher maintenance cost exposure), no insulation (property age indicator). This is immediately actionable with a free public dataset.

---

## Structuring historical claims narratives

UK home insurance portfolios typically hold years of claims handler notes, loss adjuster summaries, and legal correspondence — all unstructured text. This data is rarely used in pricing models because extracting features from it at scale has been prohibitively expensive.

LLMs change the cost equation. For a modest portfolio of 500,000 historical claims, extracting a defined feature set — cause of loss reclassification, non-standard construction flags, fraudulent indicator keywords, third-party liability involvement — at $0.01–0.02 per document would cost £4,000–£8,000 at current API rates. That is within the exploratory budget for a pricing project.

The validated use case: **cause of loss standardisation**. Claims handlers record cause using free text that varies across handlers, offices, and time. "Leak from upstairs flat", "water ingress via ceiling", "escape of water from neighbour's property" are the same peril but may be coded inconsistently. An LLM applying a defined taxonomy produces consistent codes. This is a data quality improvement that directly benefits frequency model credibility by reducing noise in the target variable.

---

## The hallucination problem in claims contexts

We need to be direct here because the risk is asymmetric.

In underwriting feature extraction, a hallucinated field value (the LLM invents a CONSTRUCTION_AGE_BAND that is not in the source document) causes a pricing error. That is a problem with a feedback loop: pricing errors show up in loss ratios over 12–36 months and can be corrected.

In claims contexts, a hallucinated output can cause an immediate and irreversible wrong decision about a customer.

LLMs hallucinate. This is not a property of poor models or bad prompts — it is an intrinsic characteristic of autoregressive language models that produce plausible-sounding completions even when ground truth is unavailable. The rate varies by model and task: GPT-4o on well-structured extraction from clear source documents produces very low hallucination rates. GPT-4o asked to summarise a long, complex loss adjuster report and produce a liability assessment will occasionally invent details not in the source.

The specific failure mode in property claims: a loss adjuster report might contain ambiguous language about policy coverage, exclusion applicability, or causation. Asked to summarise the coverage position, an LLM might generate a clean, authoritative-sounding summary that resolves the ambiguity in a direction not supported by the underlying documents. If a human reads that summary and acts on it without reviewing the source, a customer gets a wrong decision.

Our position: LLMs should not be used for coverage decisions without human review of the underlying source documents. Full extraction and summarisation assistance — yes. Autonomous coverage determination — no. This is not a conservative position driven by general AI caution; it follows from the specific failure characteristics of these models on ambiguous text.

**The governance parallel**: our [insurance-monitoring](/insurance-monitoring/) and [insurance-governance](/insurance-governance/) libraries both handle model monitoring and output logging. Any LLM deployed in a claims workflow needs the same monitoring infrastructure as a pricing model: output logging, anomaly detection, regular review of samples, and a documented escalation path when the LLM flags uncertainty. LLM uncertainty is not always explicit — a hallucinated output typically does not come with a low-confidence flag — which makes sampling-based human review more important, not less.

---

## The non-standard construction identification use case

This is the application we think most clearly justifies a near-term pilot.

UK home insurance underwriting has a material data quality problem with property type and construction declarations. Customers declare "brick walls, tile roof" because that is the standard answer, often without knowing whether their extension is flat-roofed felt, whether their property has an unexposed timber frame, or whether the roof is partially thatched. Survey data suggests a material proportion of declared standard construction is non-standard in practice.

An LLM combining: (a) Street View or aerial image description, (b) EPC ROOF_DESCRIPTION and WALLS_DESCRIPTION, (c) OS MasterMap building age approximation, and (d) free-text fields from the application or renewal can flag mismatches: "declared standard construction; EPC indicates flat roof; Street View shows visible flat-roof rear extension; OS approximate age pre-1900 suggests potential non-standard materials." That flag then routes to a specialist underwriter, not to automatic declination.

This is a human-in-the-loop workflow, which is appropriate. The LLM's role is triage: identifying the cases that warrant human attention, not making underwriting decisions. It is the same role a rules engine plays today, but with access to richer unstructured data.

---

## What has not been demonstrated

The literature review for this series required stating honestly what has not been done, because the gap between what LLMs can conceivably do and what has been validated is large.

**No published peer-reviewed paper demonstrates an LLM or VLM directly improving UK property claims prediction on a representative dataset.** The Johansson et al. (2026) Swedish building heritage classification result is the closest analogue and it is not an insurance study. The Actuaries Institute Australia work is a practitioner article, not peer-reviewed research. The gap between "plausibly works" and "validated against claims outcomes" is wide in this domain.

**No published study shows LLM-extracted property features outperforming actuary-assessed features on insurance loss ratios.** This is the study that would matter most for a UK pricing team justifying an LLM investment. It does not exist in the public literature. The US vendor claims — AIG and others have cited improvements in underwriting data accuracy from generative AI platforms, with figures in the range of 10–20% quoted in industry presentations rather than published research — conflate data quality improvement (an operational benefit) with pricing model improvement (a different claim requiring actuarial validation). These figures have not been independently verified.

**Hallucination rates on insurance documents have not been systematically benchmarked.** General-purpose hallucination benchmarks exist, but the rate of fabricated details in property survey extraction, claims narrative summarisation, and coverage interpretation tasks has not been published. Any team deploying LLMs in these workflows is operating without knowing their error rate. That needs to be measured, and measuring it requires a labelled evaluation set.

---

## The architecture that actually makes sense

The realistic deployment pattern for a UK insurer in 2026:

**Stage 1 — Data enrichment at underwriting.** LLM processes free-text application fields, EPC descriptions, and Street View visual descriptions (if available) to produce structured risk flags. Output feeds into pricing model as additional covariates. Human review is not required at this stage — errors affect pricing, are observable over time, and the model is explicit about the features it uses.

**Stage 2 — Claims triage and note structuring.** LLM extracts structured fields from incoming claims notifications and loss adjuster reports: cause of loss, property features relevant to the claim, declared vs observed discrepancies. Output supports handler prioritisation. Human review of source documents required before coverage decisions.

**Stage 3 — Portfolio retrospective.** LLM processes historical claims narratives to enrich historical data for pricing model development. This is pure data science; no customer-facing decisions involved. Lowest governance overhead.

This is the same layered approach we use in our actuarial model governance work. The key principle from [insurance-governance](https://github.com/burning-cost/insurance-governance): the riskiness of an automated output scales with how reversible the downstream decision is. Pricing a renewal slightly wrong is more reversible than declining a claim. Calibrate the human review requirement accordingly.

---

## On cost and build time

The cost argument for LLM-based operational tooling is strong relative to the alternatives.

A bespoke OCR and rules-based extraction system for property surveys — the traditional approach — requires significant NLP engineering for each document type, breaks on formatting variation, and needs maintenance when document templates change. An LLM prompt requires no engineering for each new document type, handles formatting variation gracefully, and updates when the base model improves.

The build cost for a survey extraction pipeline: one prompt template, an evaluation dataset of 50–100 manually labelled surveys, and an API integration. Measured in days of engineering time, not months. The ongoing cost: API fees (pennies per document for document-length inputs to GPT-4o) and the human review process for monitoring output quality.

The build cost for a claims narrative structuring system: similar. The additional cost is the human review infrastructure — the sampling, labelling, and escalation processes that convert an LLM pipeline into a governed system.

The operational case is strong. The reason more UK insurers have not deployed this is not technical difficulty. It is the combination of data governance questions (where does the document go when it is sent to an external API), IT integration requirements (getting document pipelines to route through an LLM service), and the audit trail requirements for FCA Consumer Duty purposes. Those are solvable problems that do not require better models.

---

## What to do now

If you are a UK actuarial or data science team with an interest in this area, the following are concrete starting points:

**Build an EPC extraction pipeline.** The 30 million EPC records are free, the fields are partially structured, and the ROOF_DESCRIPTION / WALLS_DESCRIPTION fields contain risk information not systematically used. A day of work with a simple LLM classification prompt produces structured construction flags for a meaningful proportion of your portfolio.

**Label 100 property surveys.** If your portfolio holds commercial or high-net-worth property surveys, pull 100 and manually extract the fields you care about: construction type, roof condition observation, electrical installation age, any concerns flagged by the surveyor. That 100-document labelled set becomes the evaluation dataset for any LLM extraction system you build. Without it, you cannot measure accuracy.

**Do not start with claims.** The hallucination risk and governance overhead in claims contexts make it the wrong place to prototype. Start where errors are observable and reversible.

The tools are available, the cost is accessible, and the data is — in the case of EPC — free. The gap between where most UK insurers are and where they could be on this is primarily organisational, not technical.

---

*This concludes our three-part series on computer vision and LLMs for property insurance. The series covers: [the vendor landscape and UK infrastructure constraints](/2026/03/26/computer-vision-property-insurance-what-actually-works/), [the open-source CV tooling and its honest limitations](/2026/03/26/open-source-cv-tools-property-risk/), and this post on the LLM operational layer. For proxy discrimination governance — the third constraint in deploying any imagery or LLM-derived pricing feature — we cover the technical framework in our [insurance-governance](https://github.com/burning-cost/insurance-governance) library.*
