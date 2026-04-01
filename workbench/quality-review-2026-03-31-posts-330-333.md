# Quality Review: Posts 330-333 (2026-03-31 batch)

**Reviewer:** Head of Pricing, Burning Cost
**Date:** 2026-03-31
**Scope:** 4 posts from the 2026-03-31 batch
**Method:** Full read, regulatory reference verification, technical spot-check against known literature

---

## Summary Verdicts

| Post | Slug | Verdict |
|---|---|---|
| Privacy risk benchmarking / synthetic data | `privacy-risk-benchmarking-synthetic-insurance-data` | APPROVE |
| GAM vs NAM decision guide | `gam-vs-nam-insurance-pricing-decision-guide` | NEEDS WORK |
| KMM-CP / conformal prediction covariate shift | `kmm-conformal-prediction-covariate-shift-insurance-pricing` | NEEDS WORK |
| RL dynamic reinsurance optimisation | `reinforcement-learning-dynamic-reinsurance-optimisation` | APPROVE |

---

## Post 1: Privacy Risk Benchmarking for Synthetic Insurance Data

**File:** `2026-03-31-privacy-risk-benchmarking-synthetic-insurance-data.md`
**Verdict: APPROVE**

### AI-tell phrases
None found. The writing is direct throughout.

### Em dashes
None found.

### Technical accuracy
Solid. The MIA-on-tabular-data critique is correctly reasoned and grounded in the Zuo et al. paper. The DP-SGD gradient clipping mechanism for minority class collapse is explained accurately. The AIM marginal-query approach is described correctly and the contrast with DP-SGD-based generators is legitimate.

The claim frequency preservation bounds at 0.85-1.15 (line 109) are stated without derivation but are plausible as a practical threshold. Defensible as a working rule.

One minor query: line 73 states the epsilon budget split as "10% to PrivTree discretisation, 60% to policy features, and 30% to severity." These proportions are presented as if they are a documented feature of the `insurance-synthetic` library, which is fine -- but the post does not signal whether these are configurable or hard-coded. Worth one sentence of clarification.

### Regulatory references
- ICO anonymisation guidance March 2025: plausible and consistent with the ICO's published programme.
- FCA SDEG report August 2025: cited in passing, not fabricated.
- Data Use and Access Act 2025: consistent with the legislative timeline.

No fabricated references found.

### Credibility
High. The post correctly identifies that MIA at 0.50 is being used as a rubber stamp in UK insurance synthetic data governance -- that is exactly what is happening. The framing around reinsurers and FCA submissions is accurate. The ICO motivated-intruder test is referenced correctly.

### Practical gaps
None material. The practical recommendations section (lines 93-109) is specific and actionable.

### Voice
Reads like Burning Cost. Opening paragraph names the behaviour directly ("The ICO won't notice. The reinsurer will accept it. Move on."). No hedging, no AI filler.

### Fixes required
None required before publishing. Optional: clarify whether epsilon budget proportions in the `insurance-synthetic` library are configurable.

---

## Post 2: GAM vs NAM Decision Guide

**File:** `2026-03-31-gam-vs-nam-insurance-pricing-decision-guide.md`
**Verdict: NEEDS WORK**

### AI-tell phrases
None of the flagged phrases found ("delve", "leverage", "landscape", "robust", "comprehensive", "ultimately", "it's worth noting", "in conclusion"). Clean on this front.

### Em dashes
None found.

### CRITICAL: EU AI Act enforcement date is wrong

**Line 12:** "the EU AI Act (Articles 13 and Annex III, in force August 2024)"

This is factually incorrect and needs fixing before the post is published.

The EU AI Act entered into force on 1 August 2024, but **high-risk AI system obligations under Annex III do not apply until 2 August 2026.** Annex III coverage for insurance pricing (risk assessment and pricing for life and health insurance) specifically falls in this August 2026 tranche. Firms are not yet required to comply with Annex III transparency requirements for general insurance pricing as of the post date.

The distinction matters. Saying "in force August 2024" implies current legal obligation. The correct framing is something like: "The EU AI Act (Articles 13 and Annex III) imposes transparency requirements on high-risk AI systems from 2 August 2026 -- insurance pricing falls under Annex III."

This is the kind of regulatory slip that will get cited back at us, either by a reader who checks or by an insurer's compliance team who quotes the post to justify premature compliance spend.

**Line 95:** Same error repeated: "the EU AI Act (Articles 13 and Annex III, in force August 2024) imposes transparency requirements on high-risk AI systems -- insurance pricing falls under Annex III."

Fix both instances consistently.

### Unverified industry claims
**Line 88:** "UK Lloyd's managing agents and FCA-regulated firms with conservative actuarial functions will find exact CIs on log-linear coefficients more auditable than gradient-based explanations of boosted stumps."

This is directionally correct but stated as fact rather than experience. Acceptable framing -- it is a defensible professional view. Not flagging for correction, flagging for awareness.

### Technical accuracy
The trade-off matrix (lines 64-77) is accurate. EBM monotonicity being soft rather than hard (line 52) is correctly characterised -- the `monotone_constraints` parameter in InterpretML does apply a gradient regulariser, not a hard post-hoc constraint like Dykstra projection.

The balance property discussion (lines 83-89) is correct. GLMs with canonical link satisfy the balance property via MLE score equations. EBMs, NAMs and PINs do not without post-hoc adjustment.

**Line 107:** "Exposure as offset (log link) for GLM and ANAM; as `init_score = log(exposure)` for EBM; as exposure weight for PIN."

The EBM exposure treatment via `init_score` is a common workaround but is not ideal -- `init_score` in InterpretML sets the initial prediction offset, which approximates the log-exposure offset but is not identical to it for non-canonical links. This is fine for Poisson, where log link is canonical, but the post should not generalise this to Tweedie without qualification. Minor.

### Regulatory references
- PS22/9 Consumer Duty July 2022: confirmed correct -- published 27 July 2022.
- EU AI Act: wrong as described above.
- arXiv:2510.24601 (Doohan, Kook, Burke): cited consistently and the ICC finding is described accurately.

### Credibility
Good. The ICC point about researcher-dominated noise in uncontrolled comparisons (lines 22-26) is the kind of observation that only lands if you have sat through a pricing committee debate about whether EBM or XGBoost "won." The controlled benchmarking conditions (lines 105-108) are practical and correct.

The n < 10,000 threshold for preferring GLM (lines 48, 87) is stated twice with slight inconsistency -- line 48 says "below roughly 10,000 policies" and line 87 says "Below roughly 10,000 policy-years." Policy count and policy-year exposure are not the same metric. Pick one and use it consistently, or note both. Minor but sloppy.

### Practical gaps
The post does not address the model governance process for getting any of these through a Lloyd's or FCA model validation. Saying "UK regulatory precedent: Medium (2025 paper)" in the trade-off matrix for ANAM is almost too brief -- a team actually trying to deploy ANAM will face validation committee questions that the post does not prepare them for. Not a blocking issue but a notable gap.

### Voice
Reads well. "The accuracy/interpretability trade-off argument is substantially dead on tabular data" (line 125) is the right register.

### Fixes required
1. **Mandatory:** Correct EU AI Act enforcement date on lines 12 and 95. Current text says "in force August 2024" for Annex III obligations. Should read: high-risk AI system obligations under Annex III apply from 2 August 2026.
2. **Minor:** Align policy count vs policy-year language on the n < 10,000 threshold (lines 48 and 87).
3. **Optional:** One sentence acknowledging that ANAM deployment will require model validation engagement, not just fitting the architecture.

---

## Post 3: KMM-CP Conformal Prediction under Covariate Shift

**File:** `2026-03-31-kmm-conformal-prediction-covariate-shift-insurance-pricing.md`
**Verdict: NEEDS WORK**

### AI-tell phrases
None found.

### Em dashes
None found.

### CRITICAL: Wrong regulatory reference -- SS1/23 does not apply to insurers

**Line 115:** "Actuaries working within Solvency II and the PRA's SS1/23 framework need to be able to say 'this coverage guarantee holds for this sample size.'"

SS1/23 is the PRA's model risk management supervisory statement for **banks**, not insurers. It applies to banks, building societies, and PRA-designated investment firms approved to use internal models for regulatory capital. It explicitly does not apply to insurance firms.

The relevant PRA supervisory statements for insurer internal models are:
- **SS1/24** -- "Expectations for meeting the PRA's internal model requirements for insurers under Solvency II" (February 2024)
- **SS9/18** -- Solvency II internal models: assessment, model change, and the use test (updated November 2024)
- **SS12/16** -- Changes to internal models used by UK insurance firms

Citing SS1/23 in the context of insurer capital modelling is a credibility-destroying error for an audience that includes actuaries and validation teams at UK insurers. Fix immediately.

Suggested replacement: "Actuaries working within Solvency II and the PRA's SS1/24 framework (which sets out the use test and validation requirements for approved internal models) need to be able to say 'this coverage guarantee holds for this sample size.'"

### Technical accuracy
The mathematical description of KMM-CP is accurate. The MMD minimisation formulation (lines 40-42) is correct. The description of Tibshirani 2019 (lines 22-28) is correct. The finite-sample vs. asymptotic guarantee distinction (lines 46, 115-121) is handled honestly and correctly -- this is the most important technical distinction in the post and it is not glossed over.

The SKMM selective abstention mechanism (lines 54-64) is described accurately and the insurance applications (book transfer underwriting, new product launch, model governance) are plausible and well-reasoned.

**Line 78:** "for a calibration set of 500,000 policies (not unusual for a major personal lines book with rolling 24-month calibration)"

A 500K calibration set is on the large side but defensible for a major UK motor book. Not flagging this.

**Line 126 (Solvency UK effective date):** The RL post separately mentions December 2024 for Solvency UK -- the KMM-CP post references "Solvency II and the PRA's SS1/23 framework" which mixes the old regime name (Solvency II, still used colloquially) with the wrong SS reference. The Solvency II / Solvency UK naming is not materially wrong given common usage, but the SS reference is.

### FCA SUP 15.3 reference
**Line 72:** "The FCA expects firms to document the limitations of their models. 'Model is not valid for risks outside the following covariate region' is a documentable, auditable claim when it is produced by a defined algorithm. SKMM generates exactly this output."

SUP 15.3 covers general notification requirements to the FCA (significant events, etc.). It is not an obvious home for model documentation requirements. The actual FCA framework for model documentation in general insurance is more diffuse -- Consumer Duty fair value assessments, PS21/5 pricing practices requirements, and Solvency II/UK internal model use test. SUP 15.3 is a stretch citation here.

This is not a fabricated reference -- SUP 15.3 exists -- but the specific claim that SUP 15.3 is where the FCA expresses expectations about model limitation documentation is either imprecise or incorrect. Worth softening to: "The FCA's expectation under Consumer Duty and internal model requirements is that firms document model limitations in an auditable form."

### Unverified claims
No problematic "most insurers" generalisations. The specific insurance applications (MGA acquisition, aggregator channel mix change, SME cyber product launch) are all plausible and grounded.

### Credibility
High. The MGA acquisition and book transfer examples (lines 14, 68-69) are exactly the scenarios a pricing team would face. "Standard conformal, including Tibshirani, produces an interval for these risks anyway -- usually a wide one, but an interval nonetheless" (line 64) is accurate and shows understanding of how conformal behaves in practice.

### Voice
Reads well. No AI filler.

### Fixes required
1. **Mandatory:** Replace SS1/23 with SS1/24 (or SS9/18 depending on context) on line 115. SS1/23 is for banks.
2. **Recommended:** Soften or replace the SUP 15.3 citation on line 72 -- it is a weak reference for the purpose cited.
3. **Minor:** The computation section (lines 78-88) is correct about scalability but gives no concrete guidance on what kernel bandwidth to use in practice. Not blocking, but a "start with median heuristic on continuous features, treat categoricals as Hamming distance" note would make this more actionable.

---

## Post 4: Reinforcement Learning for Dynamic Reinsurance Optimisation

**File:** `2026-03-31-reinforcement-learning-dynamic-reinsurance-optimisation.md`
**Verdict: APPROVE**

### AI-tell phrases
None found. The writing avoids all the common AI tells.

### Em dashes
None found.

### Technical accuracy
This is the strongest post of the four on technical credibility. The critique of the paper is rigorous and specific. The KS=0.6 observation and its implications are stated correctly and with appropriate severity. The reward function reproduction (lines 62-65) is clear. The absence of documented hyperparameters is flagged correctly as a reproducibility failure (lines 104-106).

**Line 84:** "A 14% surplus improvement over the next-best method (Monte Carlo at $12,803)"

The arithmetic is correct: ($14,281 - $12,803) / $12,803 = 11.5%, not 14%. The paper's claimed 14% likely references a different comparison baseline -- possibly dynamic programming at $12,488 (giving ($14,281 - $12,488) / $12,488 = 14.4%). The post attributes the 14% to Monte Carlo as the next-best, but then quotes Monte Carlo at $12,803. Check the source table and clarify which comparison the 14% refers to.

**Lines 124-126:** The standard formula SCR insensitivity to specific attachment/limit adjustments point is correct -- standard formula uses fixed sigma parameters for premium and reserve risk by segment, not treaty-specific parameters. This is a genuinely useful observation that most RL papers in this space ignore.

**Solvency UK effective date December 2024** (line 126): Confirmed correct.

**Lines 136-144 (ClauseLens Lagrangian dual mechanism):** The dual variable update rule is reproduced correctly. The characterisation of hard vs. soft constraints in constrained RL is accurate. Calling Lagrangian constrained RL "considerably more defensible" for regulatory purposes than reward penalty tuning is a reasonable professional view.

### Regulatory references
No fabricated references. The Solvency II / Solvency UK framing is appropriately careful (line 124 distinguishes the multi-year ruin constraint from the 1-year 99.5% VaR SCR, which is the correct distinction).

### Credibility
The post's credibility comes from its willingness to say clearly what the paper cannot support. Lines 175-177: "That is not a result a UK actuary can take to a board capital committee." This is exactly right and exactly the kind of thing a reviewer with capital committee experience would say. The contrast with what is actually needed (real loss data, EVT augmentation, documented reward weights, SCR integration) is specific and actionable.

### Practical gaps
The post correctly identifies that no public Gymnasium environment for reinsurance optimisation exists (lines 152-167). The gemact reference is accurate. The "minimal stack" (steps 1-4, lines 160-167) is useful.

One gap: the post does not address the treaty renegotiation problem -- real XL treaties are annual contracts, not continuously adjustable parameters. A dynamic RL policy that "adjusts treaty parameters in real time" (line 13) cannot do so contractually. The paper's framing elides this, and the post does not challenge it. The practical use case for dynamic optimisation is pre-renewal structuring decisions, not intra-year parameter adjustment. Worth one sentence acknowledging this constraint.

### Voice
Strong. "We have some reservations" (line 86) followed by specific, enumerated problems is exactly right. The post does not defer to the paper.

### Fixes required
1. **Recommended:** Verify the 14% figure arithmetic on line 84 -- identify which comparison it refers to from Table 4 of the paper and state it precisely.
2. **Optional:** Add one sentence noting that real XL treaties are annual contracts, so "dynamic adjustment" in practice means pre-renewal optimisation rather than intra-year parameter changes.

---

## Cross-Post Issues

### Pattern: Regulatory references are generally good but EU AI Act date is wrong in two places
The EU AI Act enforcement date error in the GAM/NAM post is the most material issue across the batch. The SS1/23 / bank vs. insurer confusion in the conformal post is a close second. Both are fixable in under five minutes but would embarrass the site if left uncorrected.

### Pattern: Fabrication risk
No outright fabricated references found. The arXiv citations are consistently formatted and the paper descriptions match available information. The FCA/PRA regulatory citations are mostly accurate -- the SS1/23 error is a misattribution rather than a fabrication.

### Pattern: AI voice
None of the four posts read like AI output. The writing is direct, specific, and opinionated throughout. This batch is cleaner than average on voice quality.

### Pattern: Practical grounding
All four posts address real insurance team problems with specific examples. The synthetic data post and the RL post are particularly strong on "here is what this means for a UK pricing team." The conformal post is strong on the SKMM / refer-or-rate application. The GAM/NAM post is slightly thinner on practical deployment friction (model validation committee, IT infrastructure reality) but not to a blocking degree.
