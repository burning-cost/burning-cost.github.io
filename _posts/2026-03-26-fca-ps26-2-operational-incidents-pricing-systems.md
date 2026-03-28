---
layout: post
title: "Is Your Pricing Engine a Material System Under FCA PS26/2?"
date: 2026-03-26
categories: [regulation, governance]
tags: [FCA, PS26/2, operational-resilience, pricing-governance, model-risk, Consumer-Duty, incident-reporting, third-party, insurance-monitoring, rating-api, renewal-engine, CUE, telematics, material-third-party]
description: "FCA PS26/2 (March 2026) creates mandatory incident reporting and material third party registers for all authorised insurers. Every pricing actuary who owns a rating API, renewal engine, or CUE data feed needs to understand what it requires  -  and what counts as a reportable incident."
---

On 18 March 2026 the FCA, PRA, and Bank of England published PS26/2: the policy statement on operational incident and third-party reporting. Rules come into force on 18 March 2027. There is no pricing-specific guidance in the document. That gap is exactly why pricing actuaries need to read it themselves, because the systems they own  -  rating APIs, renewal engines, PCW integrations, CUE feeds, telematics data connectors  -  sit squarely within the framework.

This post is about what the regime actually requires for typical pricing infrastructure, where the obligations are clear, and where they are genuinely ambiguous.

---

## Two separate obligations, different scopes

PS26/2 creates two new requirements that are easy to conflate:

**1. Operational incident reporting** (SUP 15.18 from March 2027): applies to all Part 4A authorised firms  -  which means every UK-authorised insurer. When an operational incident meets specified thresholds, firms must notify the FCA within 24 hours of becoming aware of it.

**2. Material third-party register** (SUP 15.19 and SUP 16.33 from March 2027): applies to a narrower list  -  Solvency II insurers, banks, designated investment firms, building societies, CASS large firms, and a few others. Not all insurers. If your firm is a small non-Solvency II captive or a Lloyd's managing agent without direct Part 4A authorisation, your exposure on this second obligation needs checking.

The distinction matters because many smaller pricing teams will not be directly in scope for the third-party register  -  but all authorised insurers are in scope for incident reporting.

---

## What counts as a reportable incident

The definition of 'operational incident' in PS26/2 captures "incidents affecting the services provided to clients, or incidents that could affect the integrity of internal processes." An incident is reportable when it meets at least one of three thresholds:

- **Consumer harm**: could cause intolerable harm to the firm's clients
- **Safety and soundness**: poses a risk to the firm's soundness, stability, or resilience
- **Market integrity**: could risk the integrity, stability, or resilience of the UK financial system

For a pricing team, the consumer harm threshold is the one to focus on. The FCA's guidance (FG26/3) is explicit that market integrity exposure can arise where PCW data feeds are affected at scale, because prolonged failures affect price comparison across the market simultaneously. A rating API outage at a large direct insurer with a major PCW share is not just an IT problem.

The 24-hour reporting clock starts when the firm "becomes aware" of the incident  -  not when it is fully diagnosed. This is a hard governance requirement. You need to know who in your organisation becomes "aware," and what the escalation path looks like from a pricing system failure to whoever makes the regulatory notification.

---

## Pricing systems: a classification analysis

The FCA's guidance contains no pricing-specific examples. What follows is our read of how the framework applies to typical pricing infrastructure. These are working judgements, not legal advice.

**Rating APIs**  -  If your rating engine is hosted externally (Guidewire, Earnix, INSTANDA, or equivalent SaaS), a sustained outage preventing quote generation is almost certainly a reportable incident for consumer harm. You cannot issue new business or mid-term adjustments. For insurers with their own on-premises rating infrastructure but a shared-services function providing the API, intragroup rules apply and the third-party register obligation is less clear  -  but the incident reporting obligation remains if clients cannot transact.

**Renewal pricing engines**  -  An outage that prevents the firm from issuing renewal invitations at correct prices, or that applies wrong rates to renewing policies, is a plausible consumer harm trigger. The scenario does not need to be large. If a subset of your book is auto-renewed at incorrect rates and policyholders do not receive correct documentation, you have a potential consumer harm within the PS26/2 definition.

**CUE and NCB data feeds**  -  CUE (Claims and Underwriting Exchange, maintained by MIB) is an external third party. If your underwriting process requires a CUE check before binding  -  as most motor writers do for fraud detection  -  a prolonged CUE outage could mean you either cannot bind policies or are binding them without required checks. Both are consumer harm or regulatory compliance scenarios. For Solvency II insurers, the CUE feed is a strong candidate for inclusion in the material third-party register. There are no quantitative materiality thresholds  -  the FCA explicitly rejected proposals to define thresholds numerically. Firms must assess case-by-case.

**Telematics data connectors**  -  Usage-based insurance introduces a pricing dependency on telemetry data. If a connected insurer cannot receive telematics signals and is using them to price or rate-adjust, a connector outage affects accurate pricing in a way that could cause consumer harm  -  particularly for policyholders mid-policy whose behaviour scores should have reduced their premium. Whether this rises to reportable depends on materiality and duration. It is worth mapping your telematics dependency explicitly.

**PCW integrations**  -  The insurer's integration with GoCompare, Confused.com, or MoneySuperMarket sends prices out; the PCW is a distribution channel, not a supplier. But any data normalisation platform or third-party aggregator API that the insurer relies on to receive and process PCW traffic is a candidate third-party arrangement. The FCA's guidance (FG26/4 3.7) explicitly names "aggregators such as pricing comparison platforms" as examples of non-outsourcing third-party arrangements, with materiality assessed case-by-case.

---

## The third-party register in practice

For Solvency II insurers, the annual register requirement means maintaining and submitting  -  via FCA RegData, within a 90-day window post go-live  -  a standardised record of all material third-party arrangements. The template includes: contract reference, legal name, LEI identifier, whether it is outsourcing or non-outsourcing, service type (a dropdown taxonomy that includes "Artificial intelligence & machine learning"), cloud deployment model, supply chain ranking (direct or subcontractor), and contract dates including notice periods.

The notice period field is not incidental. If your rating engine vendor has a 12-month exit notice period and the FCA asks you to evidence operational resilience planning, that notice period determines how quickly you could exit in a disruption scenario.

The FCA rejected requests for a safe harbour around arrangements that do not affect an Important Business Service (as defined under PS21/3 operational resilience). The materiality test for PS26/2 third-party reporting is wider than the IBS test. An arrangement not linked to an IBS can still be material under PS26/2 if its disruption could cause consumer harm or cast doubt on the firm's ability to satisfy threshold conditions.

---

## The gap between PS21/3 and PS26/2

PS21/3 (Building Operational Resilience, March 2021) required firms to map Important Business Services, set impact tolerances (ITOLs), and test those tolerances by 31 March 2025. Most insurers have done this work. The new PS26/2 obligation sits on top of it but does not replace it  -  and the FCA explicitly rejected proposals to align PS26/2 incident reporting with ITOL breaches.

The stated reasoning is that a cyber breach of a customer data portal might not breach an ITOL  -  the data service continues operating  -  but is still a PS26/2-reportable incident. The regulator wants to know about it regardless of whether it affects a formally designated IBS.

The practical implication: firms that have done their PS21/3 mapping cannot assume that captures their full PS26/2 scope. You need a separate assessment.

---

## What pricing teams should do before March 2027

The implementation period runs until 18 March 2027. Here is what a pricing actuary responsible for model and system governance should do in the next six months:

**Audit your system dependencies.** Map which external services your pricing infrastructure depends on for normal operation. Rating engine vendor, CUE/NCB feed, telematics connector, postcode data supplier (Landmark, Experian QAS), credit reference agencies (if used in pricing), PCW data normalisation platforms. Document whether each is in-house, intragroup, or external.

**Apply the materiality test to each.** For each external dependency: if it failed for 24 hours, what is the worst-case consumer outcome? Could policyholders be unable to renew? Could wrong premiums be charged at scale? Could binding checks (fraud, NCB verification) be bypassed? If yes to any, it is a material third-party candidate for Solvency II firms, and the relevant systems are incident-reporting candidates for all firms.

**Establish the incident escalation path.** PS26/2 requires a 24-hour notification from when the firm "becomes aware." That means someone in the pricing team's operational chain  -  ideally the Chief Actuary or equivalent  -  needs to be in the escalation flow for pricing system failures, with a defined triage against the three reporting thresholds. This is not a natural part of most pricing teams' incident management today.

**Connect to existing monitoring.** The insurance-monitoring library's `CalibrationChecker`, `PSIMonitor`, and `MonitoringReport` classes detect when a pricing model's output has shifted significantly from expectation. That detection capability is not PS26/2 compliance per se  -  model degradation is not an operational incident under the regulation's definition unless it disrupts service delivery or compromises data integrity. But it gives you the baseline from which consumer harm is assessed. A firm that can demonstrate its pricing model was performing normally until a specific third-party data failure occurred is better placed to respond to regulatory enquiry than a firm that cannot.

---

## What this is not

PS26/2 is an operational infrastructure regulation, not a pricing accuracy regulation. It does not require firms to demonstrate that their pricing models are good. It does not add to Consumer Duty pricing fairness obligations. It does not change PS21/5 renewal pricing rules.

What it does do is put formal reporting obligations on firms when infrastructure failures cause consumer harm  -  and it extends those obligations to third-party dependencies that no previous regulation specifically covered. For pricing teams, the relevant question is not whether the rules are burdensome (the FCA's own cost-benefit analysis estimated £16–25m in industry-wide costs over ten years, or roughly £2m per year). The question is whether the systems your team owns and relies on are mapped, assessed, and documented before March 2027.

The FCA has form on enforcement where firms were aware of a regulatory obligation and had twelve months to prepare. "We did not think this applied to our pricing systems" is not a defence that has aged well in previous Consumer Duty and operational resilience enforcement actions.

---

PS26/2 is available at [fca.org.uk/publication/policy/ps26-2.pdf](https://www.fca.org.uk/publication/policy/ps26-2.pdf). The accompanying guidance is FG26/3 (incident reporting) and FG26/4 (material third parties), published the same day.

The insurance-monitoring library for model drift and calibration monitoring is at [github.com/burning-cost/insurance-monitoring](https://github.com/burning-cost/insurance-monitoring).

---

**Related:**
- [FCA SMCR Rollback: What It Actually Means for Pricing Governance](/2026/03/26/fca-smcr-rollback-consumer-duty-disapply-pricing-governance/)  -  the deregulatory signal from the same FCA 2026 priorities document
- [Your Pricing Model Is Drifting](/2026/03/03/your-pricing-model-is-drifting/)  -  model monitoring as the evidence base for incident triage
- [Insurance Governance: PRA SS1/23 Model Risk in Python](/2026/03/14/insurance-governance-unified-pra-ss123-validation/)  -  the model risk governance layer that PS26/2 sits alongside
