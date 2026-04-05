# Quality Review: Posts 181–190 by Filename Sort, March 2026 Batch

**Reviewer:** Head of Pricing, Burning Cost
**Date:** 2026-04-05
**Scope:** 10 posts dated 2026-03-25, positions 181–190 in the `_posts/` directory by lexicographic sort order
**Method:** Full read of each post; verification of paper references, regulatory document names, code API calls, and internal numerical consistency against public sources

---

## Summary Verdicts

| # | File | Verdict |
|---|------|---------|
| 1 | fca-ep25-2-gipp-market-outcomes-cost-push-not-regulation | APPROVE |
| 2 | fca-pure-protection-market-study-what-pricing-teams-should-prepare-for | NEEDS WORK |
| 3 | gamlss-vs-conformal-head-to-head | NEEDS WORK |
| 4 | gini-drift-testing-statistical-power | NEEDS WORK |
| 5 | heterogeneous-price-elasticity-causal-forests-insurance-pricing | NEEDS WORK |
| 6 | how-to-build-double-lift-chart-python | NEEDS WORK |
| 7 | how-to-model-bi-claims-trajectory-under-whiplash-reform-uncertainty | NEEDS WORK |
| 8 | how-to-quantify-model-improvement-value-pounds | APPROVE |
| 9 | market-based-ratemaking-no-claims-history | APPROVE |
| 10 | measure-rate-change-impact-python-did-its | APPROVE |

Six passes, four need fixes. None are reject-grade. The problems are concentrated in citation accuracy and one substantive code issue. Summary below, detail per post after.

---

## Post 1 — `fca-ep25-2-gipp-market-outcomes-cost-push-not-regulation`

**Verdict: APPROVE**

This is good work. The numerical claims (49% ECC rise, £6.63 per policy motor causal estimate, £1.6bn central with £163m–£3.0bn range, home renewal differential £95 to £49) all check out as credible reads of EP25/2. The inflation decomposition framing — cost push, structural GIPP repricing, Consumer Duty — is clear and correct. The board narrative paragraph is publishable as written.

The post refers to "PS21/5" in one place (line 14, in a cross-link label) using a slash format that is correct per FCA document naming convention. No error there.

One minor thing: the `ExternalIndex.from_ons("HPTH")` call (lines 82) — HPTH is identified as the ONS motor repair cost series. This is plausible but unverifiable without access to the ONS API; the footnote-style comment saying "SPPI G4520" in the same line is slightly confusing because HPTH and SPPI G4520 are different series families. Not wrong, but could be clearer. Does not warrant a hold.

---

## Post 2 — `fca-pure-protection-market-study-what-pricing-teams-should-prepare-for`

**Verdict: NEEDS WORK — one factual error, one minor citation issue**

### Issue 1 (FACTUAL ERROR): Wrong MS document number in footnote

Line 268: `*The FCA's interim report (MS24/1.3) is available at...*`

The interim report is **MS24/1.4**, not MS24/1.3. MS24/1.3 is a separate market insights piece ("Structure of the UK pure protection market for retail customers"). MS24/1.4 is the interim report published 29 January 2026. The FCA's own URL confirms: `ms24-1-4-market-study-distribution-pure-protection-products-retail-customers-interim-report.pdf`.

Fix: change `MS24/1.3` to `MS24/1.4` in the footnote.

### Issue 2 (MINOR): Consumer Duty implementation date

Line 36: "Consumer Duty lens (PRIN 2A, live July 2023)" — the Consumer Duty came into force 31 July 2023 for open products. This is stated correctly. Fine.

### Issue 3 (MINOR): The 0.67 threshold framing

Lines 228–232: the post codes a `value_ratio < 0.67` threshold and says "The 0.67 threshold is our inference from the FCA finding that at least some over-50s products pay out less than 67p per £1 of premiums collected." The interim report's wording on over-50s fair value concerns is broadly consistent with this framing, but the 0.67 specific threshold is genuinely inferred, not stated. The post does say this clearly ("It is not a formal regulatory threshold — yet"), so the transparency is there. Fine as written.

---

## Post 3 — `gamlss-vs-conformal-head-to-head`

**Verdict: NEEDS WORK — one material regulatory misattribution**

### Issue 1 (MATERIAL): SS1/23 scope

Lines 242 and 230: the post refers to "PRA SS1/23 model risk management requirements" twice in the context of insurance pricing model validation.

PRA SS1/23 — "Model risk management principles for banks" — applies to banks, building societies, and PRA-designated investment firms with internal model approval for regulatory capital. It does **not** apply to general insurers. Insurance firms are subject to Solvency II/UK Solvency II model validation requirements (Article 120 FSMA 2023 regime, PRA SS17/16 for internal models). The post acknowledges this parenthetically at line 242: "(SS1/23 applies to banks; insurers follow equivalent Solvency II Article 120 standards, but the validation tests are the same in practice)" — but this parenthetical is buried mid-sentence and easily skimmed past.

The problem: the post uses "SS1/23" twice without qualification as a general shorthand for model risk governance, which will mislead a reader who does not catch the parenthetical. A UK motor pricing actuary will know their firm is not subject to SS1/23. An external reader might take the reference at face value.

Fix: move the qualification to the first mention, or just use "Solvency II internal model validation requirements" throughout and drop the SS1/23 shorthand. The current structure implies equivalence that is close but not exact.

### Issue 2 (MINOR): Polars DataFrame construction

Lines 54–58: the `pl.DataFrame` constructor is given a dict of numpy arrays without explicit dtype specification. This is valid Polars syntax. Not an error.

Lines 70–73: `df[idx_train]` is used to subset a Polars DataFrame using numpy integer index arrays. In current Polars (0.20+), row selection by integer array requires `df[idx_train.tolist()]` or `df.row(idx_train)` — direct numpy integer array indexing does not work the same way as in pandas. This may fail at runtime. However, since `X_train` is constructed this way and then passed to `model_gamlss.fit(X_train, y_train)`, where `y_train = y[idx_train]` correctly uses numpy indexing, the inconsistency could cause a silent shape mismatch. Worth checking.

Fix: confirm Polars integer array indexing behaviour and either add `.tolist()` or use `df.filter(pl.arange(0, len(df)).is_in(idx_train))`.

---

## Post 4 — `gini-drift-testing-statistical-power`

**Verdict: NEEDS WORK — wrong author attribution on the core paper**

### Issue 1 (FACTUAL ERROR): Wrong authors for arXiv:2510.04556

Line 14: "Wüthrich, Merz and Noll (2025), arXiv:2510.04556"
Line 38: "Wüthrich, Merz and Noll (2025)"

The actual authors of arXiv:2510.04556 are **Brauer, Menzel and Wüthrich** (Alexej Brauer, Paul Menzel, Mario V. Wüthrich). The paper is titled "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing". There is no Merz and no Noll on this paper.

This is a straightforward factual error that will be immediately visible to anyone who clicks the arXiv link. Fix both mentions.

### Issue 2 (MINOR): Paper title

Line 38 text also refers only to "arXiv:2510.04556" without naming the paper. The v1 title was "Gini-based Model Monitoring: A General Framework..." which changed in later versions. Either title is fine; just do not attribute to the wrong people.

### Issue 3 (TECHNICALLY FINE): One-sigma threshold

The post's recommendation to use `alpha=0.32` for routine monitoring and `alpha=0.05` for governance escalation (line 224) is a reasonable and defensible position that follows the paper's logic. No issue.

---

## Post 5 — `heterogeneous-price-elasticity-causal-forests-insurance-pricing`

**Verdict: NEEDS WORK — citation year imprecision creates ambiguity**

### Issue 1 (CITATION AMBIGUITY): Chernozhukov et al. year

Line 87: "GATES (Sorted Group Average Treatment Effects, Chernozhukov et al. 2020/2025 *Econometrica*)"

The hedged "2020/2025" date is awkward and potentially misleading. The paper circulated as a working paper from 2018; the Econometrica publication is 2025 (Vol. 93(4), pp. 1121–1164). The correct citation for published work is Chernozhukov et al. (2025) Econometrica. If you want to acknowledge the long working paper life, put the arXiv year in a note. Using "2020/2025" suggests co-authorship in two separate papers, which is not what happened.

Fix: "Chernozhukov, Demirer, Duflo, and Fernández-Val (2025), *Econometrica*"

### Issue 2 (TECHNICALLY FINE): RATE paper citation

Line 151: "Yadlowsky et al. 2025 *JASA* 120(549)" — confirmed correct. The paper "Evaluating Treatment Prioritization Rules via Rank-Weighted Average Treatment Effects" was published in JASA Vol. 120(549), 2025. Correct.

### Issue 3 (MINOR): EconML dependency

Line 259: "The causal forest dependencies (econml, catboost) are declared as optional extras; `uv add insurance-causal[causal_forest]`" — this is consistent with standard optional-extra packaging. Fine, but note that EconML has its own dependency footprint (torch in some versions) which may cause pain on non-x86 architectures. Not a post error, just worth flagging for readers on ARM (Raspberry Pi, Apple Silicon).

### Issue 4 (MINOR): CLAN output framing

The CLAN table (lines 134–141) shows `ncd_years` as a continuous variable (mean 0.4 vs 4.7) which implies the model was fitted treating NCD as a scalar, not categorical. The text says "NCD band 0–1" in interpretation but `ncd_years` in the code. This is consistent — NCD years is a common proxy for NCD band in synthetic data — but the labelling mismatch between code column name and the prose interpretation could confuse readers trying to replicate.

---

## Post 6 — `how-to-build-double-lift-chart-python`

**Verdict: NEEDS WORK — material bug in expected claims computation**

### Issue 1 (MATERIAL BUG): Double application of exposure in double_lift() call

Line 127–132: the `double_lift()` function is called with:
```python
chart = double_lift(
    actual=actual_claims.astype(float),
    model_a=glm_pred * exposure,   # expected claims for model A
    model_b=gbm_pred * exposure,   # expected claims for model B
    exposure=exposure,
    ...
)
```

Inside `double_lift()` at lines 109–110, the A/E computation multiplies the model predictions by exposure again:
```python
exp_a = (model_a[orig_idx] * w).sum()   # exposure-weighted expected
exp_b = (model_b[orig_idx] * w).sum()
```

So if `model_a` is passed as `glm_pred * exposure` (already in claim-count units), and inside the function it is multiplied by `w` (exposure) again, the denominator of the A/E ratio is `glm_pred × exposure²`. This overcounts by one factor of exposure and produces A/E ratios that are systematically too small.

The two fixes are either: (a) pass `model_a=glm_pred` and `model_b=gbm_pred` (rate predictions) to the function, and let the function multiply by exposure internally to get expected counts; or (b) pass `model_a=glm_pred * exposure` but change lines 109–110 to `exp_a = model_a[orig_idx].sum()` without the exposure multiplication.

Option (a) is cleaner and consistent with how `actual_claims` is passed (raw count, not rate). The function comment says "expected claims for model A" but the function then re-applies exposure. Either the comment or the code is wrong; both cannot be right.

Note: the A/E values in the output table (lines 138–149) do not look obviously broken (the GLM A/E of 0.894 in decile 1 and 1.184 in decile 10 are plausible for a 10% interaction misspecification), which suggests the two bugs may partially cancel in this particular example — but that is a coincidence of the DGP, not correctness. Fix before publishing.

### Issue 2 (MINOR): Sorting bug risk

Line 101–103: `orig_idx = order[mask_sorted]` where `mask_sorted = decile_idx == d` and `mask_sorted` is indexed on the sorted order but `order` is the sorting permutation. The indexing `order[mask_sorted]` applies `mask_sorted` (which has length n, indexed in sorted order) to `order` (also length n, the permutation). This is correct as written — `order[mask_sorted]` selects elements of the original permutation array where the mask is True, which maps back to original indices. Fine.

---

## Post 7 — `how-to-model-bi-claims-trajectory-under-whiplash-reform-uncertainty`

**Verdict: NEEDS WORK — wrong case name for Supreme Court mixed injury ruling**

### Issue 1 (FACTUAL ERROR): Wrong case name

Line 33: "The Supreme Court's mixed-injury ruling in *ABI v Aviva* (2024)"

The Supreme Court mixed injury ruling in 2024 is ***Hassam and another v Rabot and another* [2024] UKSC 11**, decided 26 March 2024. This involved two conjoined test cases — Hassam v Rabot, and Briggs v Laditan. There is no "ABI v Aviva" case at the Supreme Court. The ABI had sought to clarify the law, and the case was tested by insurers' side, but the parties are Hassam/Rabot and Briggs/Laditan. The case is commonly referred to as "Hassam v Rabot" in practitioner commentary.

Fix: Replace "ABI v Aviva (2024)" with "*Hassam v Rabot* [2024] UKSC 11".

### Issue 2 (TECHNICALLY ADEQUATE): Scenario-weighted quantile averaging

Lines 97–102: the code computes a "scenario-weighted severity curve" by averaging scenario quantiles at each quantile level. The post correctly notes (lines 104–105) that "this is not a mixture distribution in the formal sense — it is a probability-weighted average of scenario outcomes at each quantile." This is a known approximation that does not preserve the mixture distribution's quantile function exactly (Jensen's inequality means the weighted average of quantiles overestimates the mixture distribution's quantiles for convex distributions). The post's framing as appropriate for reserving purposes is defensible. Fine.

### Issue 3 (MINOR): ABI data claim

Line 10: "Bodily injury as a share of UK motor spend fell from roughly 16% in 2021 to 9% by 2025 (ABI data)." This is a plausible ballpark but the specific numbers are hard to verify precisely from public ABI publications. The claim is presented with appropriate hedging ("roughly"). Fine as written, but the ABI data reference would benefit from a specific year of report rather than a generic attribution.

### Issue 4 (MINOR): `SeverityTrendFitter.superimposed_inflation()` call

Line 130: `fitter.superimposed_inflation()` is called on the `SeverityTrendFitter` instance but line 122–128 shows the result is stored as `result = fitter.fit()`. The `superimposed_inflation()` call is on `fitter` (the instance), not on `result` (the fit result). This pattern suggests `superimposed_inflation()` is a method on the fitted instance, not on the result dataclass. This is consistent with the API shown in other posts. Not an error but slightly inconsistent with the `result.trend_rate` access on line 128 (which suggests trend properties live on the result). Flag to check against actual library source — but not a blocking issue.

---

## Post 8 — `how-to-quantify-model-improvement-value-pounds`

**Verdict: APPROVE**

The LRE framework is correctly sourced to arXiv:2512.03242 (Hedges, December 2025), and the paper exists and contains the formula as described. The formula implementation at lines 77–101 is mathematically correct. The numerical examples check out: for ρ=0.93, CV=2.0, η=1.2, target LR=0.65:

E_LR = ((1 + 0.93² × 0.25) / (0.93² × 1.25))^(0.7) − 1
     = ((1.2162) / (1.0812))^0.7 − 1
     ≈ (1.1249)^0.7 − 1 ≈ 0.0859
Expected LR = 1.0859 × 0.65 = 70.58% ✓

The sensitivity table for elasticity and the diminishing returns table are arithmetically consistent with the formula. The ρ vs Gini distinction section (lines 47–60) is correct and important — this is a genuine practical trap and the post addresses it well.

The warning about assumptions (lines 315–323) is honest and appropriate. No issues.

---

## Post 9 — `market-based-ratemaking-no-claims-history`

**Verdict: APPROVE**

Paper reference checks out: Goffard, Piette, and Peters (2025), ASTIN Bulletin Vol. 55, Issue 2, pp. 263–286, arXiv:2502.04082. The authors, journal, and year are all correct.

The description of the PMC-ABC algorithm is consistent with the paper's methodology. The MAP vs Mode estimator comparison, the 1,080-quote dataset, 12 risk classes, J=1,000 particles, G=9 generations, the loss ratio corridor (40–70%) and the Australian Shepherd MAP estimate (λ=0.31, μ=6.14, expected claim €239–245) — all match the paper's reported results.

One note: `scipy.optimize.isotonic_regression` was introduced in SciPy 1.12 (line 88 cites "scipy 1.12+"), which is correct per the SciPy release notes. Not an error.

The R package name `IsoPriceR` and the GitHub repository `market_based_insurance_ratemaking` are cited without hyperlink but no false claims made about their content.

The practical scenarios section (pet insurance, embedded cover, new geographic market) are credible and well-reasoned. The connection to Bühlmann-Straub as a downstream step is correctly positioned.

---

## Post 10 — `measure-rate-change-impact-python-did-its`

**Verdict: APPROVE**

The DiD TWFE estimator output, ITS segmented regression model, and Newey-West HAC approach are all correctly described. The `floor(sqrt(n_periods))` HAC lag formula (lines 280) is the standard Newey-West automatic bandwidth selector. Correct.

The Callaway and Sant'Anna (2021) and Goodman-Bacon (2021) staggered DiD references (line 290) — both published in the Journal of Econometrics Vol. 225 — are correctly cited. Fine.

The shock proximity check is a clever practical addition. The UK_INSURANCE_SHOCKS dictionary entries (Ogden changes at 2017-Q1 and 2019-Q3, COVID at 2020-Q1/Q2, whiplash reform at 2021-Q2, GIPP at 2022-Q1, claims inflation peak at 2022-Q3/2023-Q1) are consistent with known UK insurance market history. The GIPP effective date of 2022-Q1 (January 2022 for pricing and auto-renewal rules) is correct.

The minimum pre-periods guidance (hard floor at 4, warning below 8) is consistent with standard ITS practice. The switch from cluster-robust SE to HC3 below 20 clusters is the right econometric response. No issues.

---

## Cross-cutting observations

**Voice.** No serious AI-tells in this batch. The best posts — the LRE quantification piece and the market-based ratemaking coverage — read like someone who has sat in the meeting where a CFO asks "what does Gini mean in money?" The regulatory posts are factual and direct. The weakest voice is in the pure protection post, which has a couple of list-of-actions sections that read slightly like a compliance checklist generator, but they are redeemed by the practical calendar framing.

**Code quality.** The double-lift chart bug is the most serious technical issue in this batch. Everything else is either correct or a minor API style point. The GAMLSS post's Polars indexing issue is worth checking but is lower stakes.

**Regulatory accuracy.** Two errors found: wrong MS document number in the pure protection post (MS24/1.3 for MS24/1.4), and the wrong Supreme Court case name in the BI whiplash post (ABI v Aviva for Hassam v Rabot). Both are the kind of thing that, if spotted by a practitioner, undermines credibility on the broader analysis. Fix them.

**Author attribution.** The Brauer/Menzel/Wüthrich mis-attribution as "Wüthrich, Merz and Noll" in the Gini drift post is a clear error. The arXiv paper link is in the post; a reader who clicks it will immediately see the wrong authors. Fix before publish.

---

## Required fixes before publishing

| Priority | Post | Fix |
|---|---|---|
| P1 | double-lift-chart | Fix double-exposure multiplication in `double_lift()` call — model_a and model_b should be passed as rates, not claim counts, OR the internal aggregation should not re-multiply by exposure |
| P1 | gini-drift | Change "Wüthrich, Merz and Noll" to "Brauer, Menzel and Wüthrich" — both occurrences (lines 14 and 38) |
| P1 | bi-claims-whiplash | Change "ABI v Aviva (2024)" to "*Hassam v Rabot* [2024] UKSC 11" (line 33) |
| P2 | pure-protection | Change "MS24/1.3" to "MS24/1.4" in footnote (line 268) |
| P2 | gamlss-vs-conformal | Elevate the SS1/23 scope caveat — make it clear up front that SS1/23 is a banking supervisory statement and that insurance equivalents are Solvency II Article 120 / SS17/16 |
| P3 | causal-forests | Change "Chernozhukov et al. 2020/2025" to "Chernozhukov et al. (2025)" with full author list on first mention |
| P3 | gamlss-vs-conformal | Check Polars integer array indexing in train/cal/test split code (lines 70–73) |
