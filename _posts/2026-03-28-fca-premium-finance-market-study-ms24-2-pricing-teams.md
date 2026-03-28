---
layout: post
title: "FCA Premium Finance Market Study MS24/2 — What Pricing Teams Need to Know"
description: "The FCA published its final premium finance market study on 3 February 2026. No APR cap was imposed. That does not mean your book is clean. Here is what changed, what double dipping actually means for your rating structure, and how to build a cost-plus APR model that can survive a supervisory challenge."
date: 2026-03-28
categories: [pricing, regulation]
tags: [premium-finance, FCA, MS24-2, Consumer-Duty, APR, fair-value, double-dipping, GIPP, PS21-5, UK-motor, UK-home]
---

The FCA published its Premium Finance Market Study final report (MS24/2) on 3 February 2026. The headline — no APR cap, no commission ban — was received as a relief by most of the market. It should not be. The report established that insurers averaging 53% margins on premium finance cannot justify those margins from credit risk alone, flagged "double dipping" as a live supervisory concern, and confirmed that Consumer Duty fair value assessments now apply to the finance product independently of the insurance product. The FCA's 2026 regulatory priorities document, published three weeks later on 24 February, listed premium finance as an active enforcement priority.

This post covers: what the FCA found, what double dipping means for your GLM, the gap in the actuarial literature that leaves pricing teams flying blind, a cost-plus APR framework with real UK numbers, and what you should be doing right now.

---

## What the FCA found

Premium finance is larger than most pricing teams appreciate. Roughly 23 million motor and home policies were financed via instalments in 2023. About 48% of UK motor and home policies are paid monthly. Sixty percent of motor insurance customers paid monthly in 2024 — primarily because they cannot afford the annual payment. The outstanding loan balance across the market exceeds £5 billion; annual revenue is approximately £1.2 billion.

The average APR fell from 23.3% in 2022 to 19.2% in 2026. That 4.1 percentage point reduction happened under Consumer Duty pressure before the final report landed. When the FCA directly challenged individual firms, APRs fell by up to 8 percentage points. The market moved because it was told to — not because firms reworked their cost models. That matters for what happens next.

The distribution is the problem, not the average. Around 60% of customers face APRs of 20-30%. Around 20% face APRs above 30%. And 79% of adults in financial difficulty use premium finance — this product disproportionately reaches the customers Consumer Duty is most concerned about.

Market margins ranged from 14% to 62%. Insurers running self-funded schemes averaged 53% margin. Specialist premium finance providers (SPFPs) operating via brokers averaged 24%. The FCA's conclusion: "revenues materially exceed costs for some providers."

The FCA declined to impose a market-wide APR cap. Its reasoning was that a cap would reduce access to insurance and force base premiums up. This is a legitimate concern. But the absence of a cap is not the absence of a standard — it is the absence of a bright line, which is in some ways harder to manage.

---

## Why this matters for your rating structure

The GIPP rules (PS21/5, effective January 2022) prohibited price walking at renewal for the insurance product itself. Premium finance APR is a separate price dimension. GIPP did not touch it. Many firms treated GIPP compliance as closing the book on Consumer Duty pricing concerns. It did not. MS24/2 is the first regulatory intervention targeting the finance component specifically, and it opens a second front.

A monthly-paying customer now faces two separate fair value tests under Consumer Duty:

1. Is the underlying insurance premium fair? (PROD 4 / GIPP / Consumer Duty — covered by your existing FVA)
2. Is the APR on the finance component fair? (CONC + Consumer Duty PRIN 2A — requires a separate FVA)

You can pass test one and fail test two. Most firms have not done test two properly.

---

## The double dipping problem

Double dipping is charging a customer who pays monthly both a finance charge (the APR) and a higher underlying insurance premium, without being able to show that the premium differential reflects genuine claims risk differences.

The FCA defined it as "using the customer's decision to pay monthly to increase the price of their premium without making it clear." The final report stated that the FCA found "no evidence of credit risk being double counted and priced into both the premium and premium finance" at a market level. Read that carefully: it is a market-aggregate finding. It does not exonerate individual firms.

The regulatory exposure is specific. If your GLM or rating structure contains a factor that correlates with payment method — or if your pricing implicitly loads monthly payers — you need to be able to show that loading reflects genuine claims risk, not finance economics.

Some insurers argued that payment method correlates with claims risk: customers who cannot afford the annual payment may represent higher underlying risk. The FCA neither accepted nor rejected this. It will assess it firm-by-firm.

The practical test is: can you demonstrate from your claims data that monthly payers have materially different frequency or severity after controlling for all other rating factors? If yes, document it thoroughly. If no, the loading is double dipping and it needs to come out. A factor in a GLM that exists primarily because it helped calibrate your margin arithmetic rather than because it explains claims is not a defensible rating factor under Consumer Duty.

---

## The actuarial literature gap

Here is something no one in the market has said clearly: there is no published actuarial or academic literature on APR calibration for insurance premium finance. None. No ASTIN paper. No CAS paper. No IFoA working party. As of March 2026, pricing teams setting monthly payment APRs have no peer-reviewed methodological foundation to draw on.

The closest adjacent literature is consumer credit default modelling under the IRB Basel framework — well-developed but not insurance-specific. The other adjacent literature is insurance lapse modelling — actuarial, but concerned with coverage continuation, not the credit risk of instalment non-payment.

Two structural features of premium finance make direct application of consumer credit frameworks wrong:

First, LGD (loss given default) is structurally lower than in unsecured consumer credit. When a policyholder defaults on their instalment plan, the insurer cancels the policy and recovers the unearned premium pro-rata. For a 12-month policy cancelled at month three, the unearned premium is nine-twelfths of the annual premium — available to offset the outstanding instalment debt. This recovery mechanism substantially reduces LGD relative to an equivalent unsecured loan. Nobody has modelled this in the published literature.

Second, PD (probability of default) varies sharply by distribution channel in the FCA's data: bad debt as a percentage of outstanding balance runs at 0.6% for SPFPs, 1.0% for intermediary lenders, and 3.0% for broker-distributed business. A pricing framework that applies a uniform PD assumption across channels is miscalibrated. But there is no published guidance on what the right segmentation looks like.

This is a genuine gap with commercial consequences. Firms setting APRs without a bottom-up cost model are exposed not because the APR happens to be too high, but because they cannot document why it is what it is.

---

## A cost-plus APR framework

The cost-plus approach is straightforward in structure. The difficulty is populating it with your own numbers rather than market estimates.

```
Fair-value APR = Funding Cost + Expected Loss + Servicing Cost + Target Margin
```

**Funding cost.** For insurers using policyholder float: the foregone investment return on premiums held. In 2023 this was approximately the BoE base rate (5.25%) plus a liquidity spread — call it 5.5-6.5%. For externally funded SPFPs: cost of warehouse or credit facility, typically 7-9% in 2023 conditions. With base rate now lower, these figures need updating against your current WACC.

**Expected loss.** EL = PD × LGD × EAD.

- PD: use your own bad debt rates by segment. The FCA's market figures (0.6% for SPFPs, 3.0% for broker-distributed) are starting points only. If you have payment history data, build a logistic regression on it.
- LGD: this is where standard consumer credit models will mislead you. Your LGD is a function of when default occurs in the policy year and what unearned premium you can recover. For a policy with uniform default timing, expected LGD is around 30-50% — materially below the 60-70% typical in unsecured consumer credit.
- EAD: outstanding instalment balance at default. If defaults are uniformly distributed through the policy year, average EAD is roughly 50% of the annual premium.

**Servicing costs.** Industry estimates run £5-15 per policy per year for administration. Collections costs are separate and material if you have a high-arrears book.

What this produces with 2023-era UK numbers for an SPFP:

| Component | APR-equivalent |
|-----------|---------------|
| Funding cost | 7.5% |
| Expected loss | 0.6% |
| Servicing | 2.0% |
| Minimum cost base | ~10.1% |
| Observed market APR | 20-30% |
| Implied margin | ~10-20 pp |

For an insurer self-funding from float, the minimum cost base is lower — roughly 7-8% once you account for money market yields and a properly calibrated EL. The market average of 19.2% already represents a substantial margin over cost for these firms. The 53% average margin figure is consistent with an effective APR of roughly 15-16% being defensible on a cost-plus basis, against the 19.2% market average.

None of this tells you what the right target margin is. The FCA has not stated a number. The SPFP-level average margin of 24% implies APRs around 12-13% might represent reasonable returns for that channel; insurer margins of 53% imply considerably more compression is expected. The honest answer is that the FCA will tell you it is wrong after the fact if it finds the margin indefensible — there is no safe harbour number published.

---

## How firms currently set APRs (and why it is not enough)

The FCA identified three dominant approaches in the market, none of which it considered adequate on its own:

**Market benchmarking.** Setting APR relative to credit cards, overdrafts, or peer insurers. The FCA's objection: benchmarking against a "favourable subset" of products does not constitute a fair value assessment. It also creates collective action problems — if everyone benchmarks against everyone else, the market never corrects.

**Cost-plus with thin documentation.** Acknowledging funding and bad debt costs but without a granular bottom-up calculation. The FCA view: incomplete methodology. A spreadsheet that says "our cost of funds is X and our bad debt is Y" without supporting data or sensitivity analysis does not pass.

**Relying on the SPFP's own assessment.** Insurer or broker treating the finance provider's FVA as sufficient. The FCA was clear: Consumer Duty accountability cannot be outsourced. If you distribute a premium finance product, you own the fair value question regardless of who actually provides the credit.

Good practice — per the FCA's broader Consumer Duty review — looks like: a documented, board-approved FVA for the finance product; a bottom-up cost decomposition with supporting data; segment-level monitoring including vulnerable customers; an annual review cadence; and an escalation process for when the FVA is borderline or fails.

---

## What to do now

Four things pricing teams should action immediately.

**Audit your rating factors.** Pull out any factor in your rating structure that correlates with payment method. For each one, ask: is there a causal story that runs from this factor to claims frequency or severity, or did this factor get into the model because it helped calibrate the pricing towards a margin target? Payment method itself, if used as a rating factor, requires documented claims-data justification. Credit-proxying factors (household income bands, area deprivation indices used without claims-based validation) are higher risk.

**Run a cost-plus APR calculation.** Use your own management accounts. What is your actual cost of funds? What is your bad debt rate by segment and channel? What are your servicing costs per policy? Build the table above with your own numbers. If your implied margin is above 30%, you are in the territory where the FCA's language about revenues "materially exceeding costs" applies to you. If it is above 50%, you should be talking to your legal team now.

**Produce a separate FVA for the finance product.** Not the insurance product — the finance product. It needs to decompose the total cost of credit into its components. It needs to cover your target market, including any vulnerability indicators. It needs board sign-off. If you already have one and it was written before MS24/2, revisit it — the FCA's final report changed the standard.

**Identify your APR by segment.** Direct versus broker-distributed. Motor versus home. If broker-distributed APRs are materially higher than direct, can you show that reflects genuinely higher costs (bad debt, servicing)? Or does it reflect commission extraction? The FCA found that around 55% of the total cost of credit in the broker-SPFP channel goes to broker commission. That is a structural misalignment the FCA is watching.

---

## The strategic position

The FCA's decision not to impose an APR cap was conditional: it will work through Consumer Duty supervision, engage directly with APR outliers, and reserve structural intervention for 2027-2028 if supervision does not deliver. The regulatory clock is running.

Firms that move APRs now — toward a defensible cost-plus level, with documentation — are in a substantially better position than those that wait for a supervisory challenge. A reactive reduction under FCA scrutiny carries conduct risk implications that a proactive reduction does not.

The deeper issue is that the actuarial toolkit for this product does not yet exist. No published LGD framework accounts for pro-rata cancellation recovery. No peer-reviewed PD model incorporates insurance-specific features. Pricing teams building cost-plus APR models are, at present, working without a map. That is a tractable problem — the data exists, the methods are well-understood from consumer credit, and the structural modifications required are clear. But it is work that needs to happen, and it has not happened yet.

The FCA noticed. The market needs to catch up.
