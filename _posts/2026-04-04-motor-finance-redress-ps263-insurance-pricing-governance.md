---
layout: post
title: "Motor Finance Redress: What PS26/3 Tells Insurance Pricing Teams About Their Own Exposure"
date: 2026-04-04
categories: [regulation, governance]
tags: [FCA, consumer-duty, motor-finance, PS26/3, FSMA-s404, insurance-governance, fairness, retrospective-enforcement, pricing-governance, insurance-fairness, UK-insurance, fair-value, proxy-discrimination, FCA-EP25-2, PS22-9]
description: "The FCA confirmed its motor finance redress scheme on 30 March 2026 — £7.5bn, 12.1m agreements, going back to 2007. The enforcement pattern is not motor-finance-specific. Insurance pricing teams with undocumented fairness testing from 2021–2024 are carrying the same structural risk."
author: Burning Cost
---

On 30 March 2026 the FCA published Policy Statement PS26/3, confirming an industry-wide consumer redress scheme for motor finance customers. The scheme covers approximately 12.1 million regulated credit agreements entered between 6 April 2007 and 1 November 2024. Total estimated lender costs: £9.1 billion, of which £7.5 billion is direct consumer redress at an average of £620 per agreement.

The motor finance story has been running since the Court of Appeal ruling in October 2024. The Supreme Court's August 2025 judgment narrowed the scope — it rejected the broader "secret commission" framing and upheld only the Johnson case, where commission was 55% of total credit charges and documentation was actively misleading. The FCA nonetheless proceeded with a mandatory industry scheme, using its powers under section 404 of the Financial Services and Markets Act 2000 and the unfair relationship provisions in section 140A of the Consumer Credit Act 1974.

We are not here to relitigate motor finance. The question for insurance pricing teams is different: what does this enforcement pattern tell us about our own exposure?

---

## What the FCA actually did

The mechanics matter. The FCA:

1. **Identified a structural unfairness** — discretionary commission arrangements (DCAs), banned from 28 January 2021, which allowed dealers to vary the interest rate on customer finance and thereby increase their own commission. Customers were not told this was happening.

2. **Let the court test the legal basis** — rather than enforcing through its own supervisory powers immediately, the FCA waited for litigation to establish the unfair relationship finding under Consumer Credit Act s.140A.

3. **Then used FSMA s404 to mandate industry-wide redress**, covering firms that were never individually litigated against. The scheme applies to any lender whose agreements met the commission disclosure criteria — regardless of whether a complaint was made.

4. **Extended the scheme back to 2007** — seventeen years before the redress scheme was announced. Scheme 1 (2007–2014) went ahead despite legal challenge from lenders arguing the FCA lacked s404 powers for pre-2014 agreements. The FCA acknowledged "elevated legal risk" and proceeded anyway, splitting the scheme in two precisely to allow Scheme 2 (2014–2024) to proceed if Scheme 1 is challenged successfully in court.

The critical point: the DCA was banned in 2021 because it produced unfair consumer outcomes. The FCA then required redress going back fourteen years before the ban — to 2007. The historical unfairness, and the absence of documentation showing firms had identified and addressed it, is what made the retrospective scheme viable.

---

## Why this is an insurance pricing problem

Motor finance and insurance pricing are different products under different regulatory frameworks. The FCA is not suggesting an insurance pricing redress scheme is imminent.

What the pattern shows is how the FCA constructs retrospective enforcement:

- Identify a structural feature of pricing or remuneration that produces poor outcomes for consumers
- Let the legal framework establish the unfairness finding
- Then use s404 (or equivalent) to require industry-wide remediation, including historical cases

For insurance pricing, the structural feature that maps most directly onto DCAs is **the absence of documented fairness testing for pricing models between 2021 and 2024** — the period during which Consumer Duty was being designed and implemented.

Consumer Duty came into force for open products on 31 July 2023. The FCA's own multi-firm review, published as TR24/2 in August 2024, found that most fair value assessments "lacked granularity to identify whether specific groups were receiving poor outcomes." The review covered 28 product manufacturers and 39 distributors. The finding was not that firms were producing discriminatory outcomes. The finding was that firms could not demonstrate they were not.

That is the motor finance pattern. DCAs were not unlawful because every individual lender chose to harm consumers. They were unlawful because the structure created incentive misalignment, lenders could not document consumer-level fairness, and the courts found the relationships unfair.

---

## The specific exposure period

PS26/3 applies Consumer Duty communication standards to current correspondence about a historical scheme — pricing teams should note the direction of travel. The FCA explicitly notes that "firms should ensure that communications with consumers are aligned with relevant Consumer Duty requirements," and this framing applies to a scheme covering conduct going back to 2007, predating Consumer Duty by sixteen years.

For insurance, the highest-risk period is **2021–2024**:

- 2021: FCA published PS21/5, banning price walking and requiring fair value assessments. Pricing teams were building new models to comply.
- 2022–2023: Consumer Duty designed and consulted on. FCA explicitly stated fair value outcomes — not just fair process — would be required.
- 31 July 2023: Consumer Duty live for open products.
- 2024: FCA TR24/2 found inadequate fair value evidence across the market.

A pricing team that built a new model in 2022 to comply with PS21/5, deployed it in 2023, and did not run demographic fairness testing at deployment is now carrying undocumented model history during the Consumer Duty implementation period. The model is live, the testing was not done, and the records do not exist.

If the FCA were to use s404 — or equivalent Solvency II supervisory tools — to mandate a fairness review of pricing models operated during 2021–2024, firms without contemporaneous documentation would be in the same position as motor lenders without DCA disclosure records.

---

## What "contemporaneous documentation" means in practice

The motor finance redress scheme calculates compensation based on what lenders can evidence. Where documentation is absent, firms are presumed to have paid commissions that were not adequately disclosed. The evidential burden runs against the firm.

For insurance pricing, contemporaneous documentation means:

**Model deployment records with fairness testing at version level.** Not a general annual Consumer Duty assessment, but a version-tagged record: "Model v2.3 deployed November 2022. Demographic parity audit run at deployment: pass. Protected characteristics tested: age, gender proxy, postcode deprivation decile. Audit lead: [named individual under SMCR]. Next scheduled retest: Q1 2023."

**Segmented A/E ratios by demographic group for the deployment period.** Not aggregate calibration. Segmented. The FCA's concern, documented repeatedly since EP25/2 (July 2025), is that models can pass aggregate calibration while systematically mispricing a protected sub-group. An A/E ratio at portfolio level does not evidence fair outcomes at segment level.

**Consumer Duty fair value assessment documentation covering the model's rating factors.** PROD 4 (product governance) requires this for product manufacturers. The assessment needs to cover whether rating factors produce outcomes — not just inputs — that are fair across the target market. If postcode is a rating factor and postcode correlates with race, the fair value assessment needs to address that correlation and document the actuarial justification (or removal of the factor).

**Drift monitoring logs showing the model was being watched after deployment.** A model that passed a fairness test at deployment but drifted into discriminatory outcomes over 18 months, with no monitoring evidence showing the drift was caught, has the same problem as a lender that stopped monitoring commission disclosure after 2014.

The `insurance-governance` library covers model inventory and version-tagged deployment records. The `insurance-fairness` library covers demographic parity testing, including the `FairnessAudit` class for multi-characteristic testing and `DoubleFairnessAudit` for both aggregate and individual-level screening. The `insurance-monitoring` library covers ongoing calibration drift monitoring — A/E ratio tracking, PSI/CSI, and `PITMonitor` for distributional stability. If you are running models without all three in continuous operation, the documentation gap exists.

---

## Consumer Duty enforcement is already running

As of March 2026, the FCA has multiple reported active Consumer Duty investigations open. These are not disclosed investigations — the FCA does not publish individual firm names at investigation stage. Consumer Duty enforcement is running. It is not hypothetical, and it is not future tense.

PS26/3 is not a Consumer Duty instrument. It uses Consumer Credit Act and FSMA powers. But the FCA cited Consumer Duty norms as the standard against which the motor finance communications must be measured. The regulatory direction of travel is clear: Consumer Duty outcomes are the baseline, the evidential bar is rising, and retrospective application is confirmed precedent.

---

## What to do now

Three things, in order of urgency:

**First: audit your model inventory for the 2021–2024 period.** For each pricing model (frequency, severity, demand, retention) that was in production during that period: does a version-tagged record exist showing what fairness testing was done at deployment? If the record does not exist, create a reconstruction now — using git history, validation reports, model sign-off documentation — and note that it is a reconstruction. A documented reconstruction is substantially better than no record. Courts and regulators treat absence of evidence as evidence of absence.

**Second: run the fairness tests you should have run at deployment.** Use the current model version on historical data. Document the results, note the limitations (using current model on past data is imperfect), and establish a baseline. If the tests show an issue, you have a decision to make about disclosure. If they show clean results, you have evidence — imperfect but better than nothing.

**Third: confirm ongoing monitoring is running and logged.** Not just model performance monitoring. Demographic split monitoring. PSI/CSI by protected group proxy. If a model has been live since 2022 without a logged fairness retest, the retest is overdue regardless of what the motor finance redress scheme does or does not prompt.

The motor finance scheme covers seventeen years of activity. The FCA has demonstrated that it will reach back that far when the evidence base supports a finding of structural unfairness. The question for insurance pricing governance is not whether an insurance redress scheme is coming. The question is whether, if scrutiny arrives, you can produce the documentation that shows your models were operated fairly and monitored appropriately.

Most firms currently cannot. That is the exposure.

---

## References

- FCA PS26/3: Motor Finance Consumer Redress Scheme, 30 March 2026 — [fca.org.uk/publications/policy-statements/ps26-3-motor-finance-consumer-redress-scheme](https://www.fca.org.uk/publications/policy-statements/ps26-3-motor-finance-consumer-redress-scheme)
- FCA TR24/2: Multi-firm review of product governance, August 2024
- FCA EP25/2: Evaluation of General Insurance Pricing Practices remedies, July 2025
- Consumer Duty (PRIN 2A), effective 31 July 2023
- FCA PS21/5: General Insurance Pricing Practices, effective 1 January 2022
- Consumer Credit Act 1974, section 140A (unfair relationships)
- Financial Services and Markets Act 2000, section 404 (industry redress schemes)

Related on Burning Cost:

- [FCA 2026 Insurance Priorities: A Pricing Actuary Checklist](/2026/04/01/fca-2026-insurance-priorities-pricing-actuary-checklist/) — including reported active Consumer Duty investigations
- [Consumer Duty Fair Value Evidencing: A 12-Step Technical Checklist](/2026/03/25/consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries/)
- [FCA SMCR Rollback: What It Actually Means for Pricing Governance](/2026/03/26/fca-smcr-rollback-consumer-duty-disapply-pricing-governance/)
- [Mean Parity Is Not Tail Parity](/2026/04/04/demographic-parity-tails-regression-insurance-fairness/) — why aggregate fairness checks miss the exposure
- [Actuarial Model Validation in Python: Automated Reports for UK Insurance Pricing](/2026/04/04/actuarial-model-validation-python/) — the automated audit trail that provides the documentation the FCA is looking for
