# Quality Audit: March 25 2026 Post Batch — Spot Check

**Reviewer:** Head of Pricing, Burning Cost
**Date:** 2026-03-25
**Scope:** 5 posts from the 60 published today, spanning the five content categories
**Methodology:** Read full post, cross-reference API calls against actual library source in `~/burning-cost/repos/`, verify factual claims against public sources

---

## Overall Verdict

Three posts are publishable as-is or with minor corrections. Two have material problems that need fixing before they go out — one has wrong API calls that will embarrass us the moment someone runs the code, and another has an outright wrong parameter name in a code example.

| Post | Grade | Status |
|---|---|---|
| EP25/2 regulatory analysis | A | Publish |
| ANAM arXiv coverage | A- | Publish (one minor concern) |
| Double-lift how-to | B+ | Publish with small fix |
| PS21/5 end-to-end | B- | Fix API before publishing |
| UK Motor loss-making market intel | C | Two API errors, one regulatory slip — fix before publishing |

---

## Post 1: EP25/2 Regulatory Analysis

**File:** `2026-03-25-fca-ep25-2-gipp-market-outcomes-cost-push-not-regulation.md`
**Grade: A**

### Factual accuracy

EP25/2 is real, published July 2025. The specific figures are confirmed against the actual document:
- ECC home pre-GIPP £92.26, post-GIPP £137.51 (49% rise): **correct**
- Total price inception home £248.52 → £260.92 (5%): **correct**
- Motor ECC £312.26 → £349.38 (12%): **correct**
- Motor price £445.46 → £497.90: **correct**
- Motor causal estimate £6.63 per policy saving: **consistent with published figures**
- Ten-year central consumer savings estimate £1.6bn (range £163m–£3.0bn): **correct**
- 16 home insurers, 13 motor insurers, ~80% home market GWP: **plausible, consistent with EP25/2 scope description**
- Home new/renewal differential: £95.38 to £49.17: **plausible for this paper**
- Motor repair cost inflation: paint +16%, spare parts +11% (Q3 2022 to Q3 2023 per EP25/2): **consistent with EP25/2 citations**
- CDiD methodology description is correct
- Claims ratio fall 64% to 56% in motor: **plausible from paper context**

The final disclaimer is precise and honest: "EP25/2 does not address proxy discrimination or demographic disparity in pricing. That is covered by Consumer Duty (PRIN 2A) and TR24/2, not this paper. Anyone citing EP25/2 as authority for fair value monitoring across protected characteristics is misattributing the source." That is exactly the kind of sourcing discipline we want.

### API correctness

The code calls `insurance_trend.LossCostTrendFitter` and `ExternalIndex.from_ons("HPTH")`. We do not have the `insurance-trend` library source available in the repos to verify. However, the ONS series HPTH is a plausible identifier for motor repair costs (it is close to SPPI indices used for motor). I am flagging this as **unverifiable** rather than wrong — if `insurance-trend` does not have `ExternalIndex.from_ons`, this code is broken. Someone should verify before publishing. For now I am not dropping the grade for it.

The `DriftAttributor` usage in the monitoring section is correct. Parameters `alpha=0.05`, `loss="mse"`, `n_bootstrap=200`, `features`, `model` all match the actual `DriftAttributor.__init__` signature. The `fit_reference(X_ref, y_margin_ref, train_on_ref=False)` call is correct — `train_on_ref` is a valid parameter.

### Credibility

This is the strongest post in the batch. It reads like someone who has actually sat in a board meeting trying to explain why premiums went up without being blamed for it. The "board narrative" section is particularly good — concrete language that a Head of Pricing could drop into a presentation with zero modification. The inflation decomposition into three components (cost push, GIPP structural effect, Consumer Duty back-book repricing) is analytically correct and practically useful. The CDiD methodology critique in the limitations section is accurate and appropriately caveated.

Minor quibble: The post says the FCA's compliance cost data was voluntary and "nine of 16 home firms responded." I cannot verify this specific breakdown from what I have available, though it is plausible. If wrong, it's embarrassing. Low priority to check.

### Tone

Reads like a pricing actuary. No hedging, no AI-isms. The sentence "If you want to tell a story about insurer profiteering, that table does not support it" is good. The board narrative box is correctly framed as "defensible position because it is sourced directly from the FCA's own policy-level dataset."

### Verdict

Publish. One action: verify `insurance-trend` API if possible before it goes live.

---

## Post 2: ANAM arXiv Coverage

**File:** `2026-03-25-actuarial-neural-additive-model-anam-arxiv-2509-08467.md`
**Grade: A-**

### Factual accuracy

arXiv:2509.08467 exists. Authors Laub, Pho and Wong (UNSW) confirmed. September 2025 submission confirmed.

The technical content is substantively correct:

- The additive log-space structure is correct
- The Whittaker-Henderson penalty description is accurate: second-difference penalty with discrete analogue interpretation. The Whittaker (1922) / Henderson (1924) citations are appropriate — actuaries do use this tradition for mortality smoothing, and making this connection explicit is genuinely useful
- The Dykstra algorithm explanation is technically correct: non-negative weight constraint, projection at each gradient step, composition property of monotone functions through ReLU networks. This is the right mechanism and explains it more clearly than most coverage I have seen
- The CANN vs LocalGLMnet vs ANAM distinction is accurate. The characterisation of LocalGLMnet as not additively separable (the weight-generating network sees all features) is correct
- The beMTPL97 benchmark dataset is real (Belgian MTPL frequency data, widely used in actuarial ML literature)
- The multi-task and GAMLSS extensions are accurately described

The phrase "The paper's specific claim is not that ANAM dominates XGBoost" is accurate and important — many coverage posts on this kind of paper misrepresent what the benchmark shows.

One concern: the post references a link to `/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/` dated December 2026. That post does not exist yet — it is in the future relative to today's date (2026-03-25). This is a dead internal link. Low risk if the site handles 404 gracefully, but it signals that internal linking is being generated without checking whether the target actually exists. We need a process to catch forward references.

### API correctness

The `insurance-gam` API — `ANAM.fit(X, y, sample_weight=exposure)`, `ANAM.predict(X, exposure=exposure)`, `model.shape_functions()`, `GLMComparison(model, glm_params).plot_overlay("driver_age")` — I cannot verify against source because `insurance-gam` does not have full source visible (tests and notebooks only). However, the API is consistent with sklearn conventions and the description in the `insurance-gam` notebooks. I am flagging as **likely correct but unverified**.

The training time estimates (5–20 minutes for ANAM on 200k policies, 30–90 seconds for EBM) are plausible based on architecture differences, but are empirical claims without a citation. These will get quoted and will create problems if they are wrong. Consider adding "on our test hardware" or removing specific numbers.

### Credibility

Strong. The "We would phrase this more directly: if your model can be a GBM, use a GBM. ANAM is for the cases where it cannot" is exactly the right practical takeaway. The paper coverage distinguishes itself from generic AI summaries by focusing on the Dykstra mechanism and the additive separability property rather than the benchmark numbers.

### Tone

Good. No buzzwords. The "The Whittaker (1922) and Henderson (1924) citations in the paper are not ornamental" line is the kind of observation that signals the author understands actuarial history, not just ML literature.

### Verdict

Publish. Fix the dead forward link to `/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/` — either remove the internal link or replace with a description of what that post will cover. Do not publish with a dead link dated 9 months in the future.

---

## Post 3: Double-Lift Chart How-To

**File:** `2026-03-25-how-to-build-double-lift-chart-python.md`
**Grade: B+**

### Factual accuracy

The double-lift chart methodology is correct. Exposure-weighted decile construction is the right approach (unweighted is wrong in insurance because short-term policies overweight the tail). The interpretation of the chart — sloped champion line with flat challenger indicates genuine discrimination improvement — is standard.

### API correctness

**Main implementation (manual, no library).** The `double_lift()` function is original code and it mostly works correctly.

One potential issue: the expected claims calculation in Step 2. The post passes `model_a=glm_pred * exposure` and `model_b=gbm_pred * exposure` into the function, so `exp_a = (model_a[orig_idx] * w).sum()` becomes `(glm_pred * exposure * exposure).sum()` — double-weighting by exposure. The frequencies are already scaled by exposure at the call site, but then multiplied by `w` (which is exposure) again inside the loop. This is a bug. If `glm_pred` is a frequency (claims per unit exposure) and `exposure` is car-years, the expected claims should be `glm_pred * exposure`, not `glm_pred * exposure * exposure`. The post is passing already-scaled expected claims into a function that then re-scales them. The synthetic data is constructed such that `glm_pred` and `gbm_pred` are frequencies, so the pass-in of `glm_pred * exposure` as `model_a` and then the loop doing `(model_a[orig_idx] * w).sum()` double-counts exposure.

The displayed output numbers look plausible (A/E near 1.0 for GBM, slope in GLM) which suggests the random seed makes the numbers work out approximately, but the implementation is technically wrong for any deployment where the user follows the pattern literally. **Fix this before publishing.**

Corrected call should be:
```python
chart = double_lift(
    actual=actual_claims.astype(float),
    model_a=glm_pred,      # frequency, not frequency * exposure
    model_b=gbm_pred,
    exposure=exposure,
    n_deciles=10,
)
```

And inside `double_lift`, `exp_a = (model_a[orig_idx] * w).sum()` correctly computes expected claims as frequency × exposure.

**insurance-distill `double_lift_chart()` API.** The post calls:
```python
from insurance_distill import double_lift_chart
chart_dl = double_lift_chart(
    pseudo=gbm_pred,
    glm_pred=glm_pred,
    exposure=exposure,
    n_deciles=10,
)
print(chart_dl)
# Columns: decile, avg_gbm, avg_glm, ratio_gbm_to_glm, exposure_share
```

Verified against source (`_validation.py`). The function signature matches: `double_lift_chart(pseudo, glm_pred, exposure=None, n_deciles=10)`. The returned column `ratio_gbm_to_glm` matches the source (it is `ratio_gbm_to_glm` in the actual code). **API is correct.**

### Model validation context

The reference to "PRA SS1/23 model validation pack" is appropriate and correct — SS1/23 is PRA's model risk management supervisory statement which does require validation evidence including champion/challenger analysis. However, the post does not mention that the holdout data needs to be truly out-of-time for the double-lift to be regulatory evidence rather than just an in-sample fit check. This is a gap — a pricing committee will ask "what period is the holdout?" and the post should say "it must be out-of-time" explicitly. Currently it mentions this only in passing in the "reading the output" section.

### Credibility

The scenario (GLM missing young-driver/high-group interaction) is exactly the right motivating example — this is the single most common interaction that GLMs miss in UK motor, and every actuary in the room will recognise it. The observation that a GBM can win on Gini while miscalibrated in specific cells is important and often missed.

### Tone

Good. The "model documentation" section is practical and directly answers what a pricing committee will ask. No filler.

### Verdict

Fix the double-exposure bug in the pass-in convention. The bug is in the call site, not the function itself — the function is correct, but the example shows `model_a=glm_pred * exposure` which pre-multiplies before passing into a function that multiplies by `w` again. Add a one-sentence note about out-of-time requirements in the documentation section. Then publish.

---

## Post 4: PS21/5 End-to-End Library Post

**File:** `2026-03-25-ps21-5-renewal-pricing-end-to-end-python.md`
**Grade: B-**

### Factual accuracy

The regulatory framing is accurate. The ENBP concept is correctly described — equivalent new business price, the ceiling above which a renewal cannot be priced. The critique of "bolt-on ENBP check" as suboptimal versus joint optimisation is a legitimate and underappreciated point. The Consumer Duty framing (action fairness vs outcome fairness distinction) is correct and is rarely made this clearly in practitioner writing.

### API correctness

**insurance-causal — HeterogeneousElasticityEstimator.** The post uses:
```python
est = HeterogeneousElasticityEstimator(n_estimators=200, catboost_iterations=300)
est.fit(df, outcome="renewed", treatment="log_price_change", confounders=confounders)
cates = est.cate(df)
inf = HeterogeneousInference(n_splits=100, k_groups=5)
hte_result = inf.run(df, estimator=est, cate_proxy=cates)
```

Verified against source. Constructor signature: `HeterogeneousElasticityEstimator(binary_outcome=True, n_estimators=200, catboost_iterations=500)`. The post uses `catboost_iterations=300` — valid, just different from default. The `fit()` method takes `(df, outcome, treatment, confounders)` — matches. `cate()` takes `df` — matches. `HeterogeneousInference(n_splits=100, k_groups=5)` — `n_splits` defaults to some constant (source shows `_N_SPLITS`), `k_groups` defaults to `_DEFAULT_K_GROUPS`. Both are valid named parameters. `run(df, estimator=est, cate_proxy=cates)` matches the source signature `run(df, estimator, cate_proxy, confounders=None)`. **Causal forest API is correct.**

**insurance-optimise — PortfolioOptimiser.** The post uses:
```python
opt = PortfolioOptimiser(
    technical_price=...,
    expected_loss_cost=...,
    p_demand=...,
    elasticity=...,
    renewal_flag=...,
    enbp=...,
    prior_multiplier=...,
    constraints=config,
)
```

Verified against source. Constructor: `__init__(self, technical_price, expected_loss_cost, p_demand, elasticity, renewal_flag=None, enbp=None, prior_multiplier=None, constraints=None)`. All parameters match. **Optimiser API is correct.**

**OptimisationResult attributes.** Post uses `result.summary_df["enbp_binding"]`, `result.expected_loss_ratio`, `result.expected_retention`, `result.expected_profit`, `result.new_premiums`, `result.audit_trail`. Verified against `result.py`: `new_premiums`, `expected_profit`, `expected_loss_ratio`, `expected_retention`, `audit_trail`, `summary_df` are all real attributes. The `summary_df` column `enbp_binding` is described in the docstring as one of the columns in `summary_df`. **Result API appears correct.**

**install command.** The post opens with:
```bash
pip install insurance-causal insurance-optimise insurance-fairness insurance-governance
```

But the rest of the site uses `uv add`. This is inconsistent — the EP25/2 post uses backtick references to `uv add insurance-monitoring`, the double-lift post suggests `uv add insurance-distill`. The PS21/5 post is the only one using `pip install`. This is not a bug but it looks sloppy. Pick one and be consistent.

**insurance-fairness API.** The post uses `detect_proxies` and `FairnessAudit`. I cannot verify these against source (insurance-fairness is in the repos but not in the standard path I have access to). The API looks plausible but is unverified.

**insurance-governance API.** `MRMModelCard`, `Assumption`, `Limitation`, `RiskTierScorer`, `ModelInventory`, `GovernanceReport` — I cannot verify. The usage pattern looks sensible but the `Assumption(description=..., risk=..., mitigation=...)` and `Limitation(description=...)` keyword signatures are unverified.

### Content gap: ENBP estimation quality

The post rightly says "ENBP quality is the weak point of most PS21/5 implementations" and notes that quarterly back-tests are "non-negotiable." But it does not say what the back-test looks like in practice. How do you estimate ENBP for the 40% of your book that renewed direct when you also sell through PCW? How do you handle cases where the new business quote engine re-rates continuously and the ENBP timestamp is ambiguous? These are the questions that every implementation team hits in week two, and the post would be more valuable if it addressed them, even briefly. This is a credibility gap — a post written by someone who has actually done a PS21/5 implementation would include at least one of these cases.

### Tone

Mostly good. The distinction between "legally sufficient" and "commercially optimal" in the opening is sharp. Some of the governance section is slightly formulaic — the `Assumption(risk="HIGH")` example reads like it was generated to fill a template.

### Verdict

Fix the `pip install` to `uv add` for consistency. Verify insurance-fairness and insurance-governance APIs against source — if they are wrong, this becomes a C. Add at minimum a paragraph on the ENBP estimation problem in practice (channel-specific ENBP, re-rating timing risk). Then publish.

---

## Post 5: UK Motor Loss-Making Market Intelligence

**File:** `2026-03-25-uk-motor-is-going-loss-making-again.md`
**Grade: C**

### Factual accuracy

The market data is strong and sourced. EY December 2025 Motor Insurance Results Analysis: 97% NCR for 2024, 101% forecast 2025, 111% forecast 2026. WTW January 2026 Motor Insurance Premium Tracker: 13% annual fall, average £726, from £995 peak December 2023. ABI Q3 2025: £551 average. Repair costs: paint +16%, parts +11%. These are specific, cited, plausible. ABI 2024 claims: £11.7bn (17% above 2023). All sourced inline.

The analysis of why (claims inflation, PCW cycle amplification, model error compound) is correct and well-structured. The Ogden discount rate commentary is accurate — the -0.25% rate was set in 2019 and is under review. BI bodily injury share falling from 16% to 9% of spend is attributed to whiplash reforms, which is the right causal story.

**Regulatory slip:** Line 120 and 129 both reference "PS21/11" as the source of the ENBP constraint:
> "the ENBP constraint from PS21/11"
> `enbp_buffer=0.01,       # PS21/11 margin`

The ENBP obligation originates from PS21/5 (the main market study policy statement). PS21/11 is the amendments paper. While technically PS21/11 amends the rules, no pricing team in the UK identifies the ENBP cap as "from PS21/11" — they say PS21/5. Any regulator or compliance officer reading this will notice it immediately and it undermines confidence in the rest of the regulatory commentary. Fix to PS21/5 (as amended by PS21/11) or just PS21/5.

### API correctness — two confirmed errors

**Error 1 — MonitoringReport constructor.** The post shows:
```python
from insurance_monitoring import MonitoringReport

report = MonitoringReport(
    y_true=current_claims,
    y_pred=model_expected,
    features_reference=reference_features,
    features_current=current_features,
    exposure=current_exposure,
)
```

The actual `MonitoringReport` constructor (verified against `report.py`) takes:
- `reference_actual` (not `y_true`)
- `reference_predicted` (not `y_pred` — and this parameter is missing entirely from the post's call)
- `current_actual` (not `y_true`)
- `current_predicted` (not `y_pred`)
- `feature_df_reference` (not `features_reference`)
- `feature_df_current` (not `features_current`)

The parameters `y_true`, `y_pred`, `features_reference`, `features_current` do not exist on `MonitoringReport`. This code will raise a `TypeError` immediately. This is a hard error. It cannot be published as-is.

**Error 2 — RateChangeEvaluator constructor.** The post shows:
```python
evaluator = RateChangeEvaluator(
    outcome_col="loss_ratio",
    treatment_period=9,
    unit_col="segment",
    weight_col="earned_exposure",
)
```

The actual constructor (verified against `_evaluator.py`) takes:
- `change_period` (not `treatment_period`) — and `change_period` is **required** (raises `ValueError` if None)
- `exposure_col` (not `weight_col`)
- No `treatment_period` parameter exists

Both wrong parameter names. Again, `TypeError` on construction. Cannot publish.

**ae_ratio and ae_ratio_ci usage.** The post uses:
```python
from insurance_monitoring import ae_ratio, ae_ratio_ci
ratio = ae_ratio(actual=paid_claims, expected=model_expected)
lower, upper = ae_ratio_ci(actual=paid_claims, expected=model_expected, alpha=0.05)
```

The actual `ae_ratio` signature is `ae_ratio(actual, predicted, exposure=None, segments=None)` — the keyword argument is `predicted`, not `expected`. The call `ae_ratio(actual=paid_claims, expected=model_expected)` will raise `TypeError: unexpected keyword argument 'expected'`.

The `ae_ratio_ci` signature is `ae_ratio_ci(actual, predicted, exposure=None, alpha=0.05, method='poisson')` and it returns a **dict** (`{"ae": ..., "lower": ..., "upper": ..., "n_claims": ..., "n_expected": ...}`), not a tuple. The post unpacks it as `lower, upper = ae_ratio_ci(...)` which will fail — a dict cannot be unpacked into two variables this way. Should be `result = ae_ratio_ci(...); lower, upper = result["lower"], result["upper"]`.

**That is three confirmed API errors in a single post, all of which produce immediate runtime failures.**

The `psi()` call in the same post does look correct: `psi(reference=..., current=..., n_bins=10, exposure_weights=..., reference_exposure=...)` matches the actual signature perfectly.

The `PortfolioOptimiser` call in this post also has `result.profit` (using the alias) and `result.volume_retention`. The `profit` alias exists (verified). But `volume_retention` does not exist — the actual attribute is `expected_retention`. Another error.

### Credibility

The market analysis content is good — specific, sourced, opinionated. The observation that "your model's expected loss cost reflects 2023 repair prices" is exactly the practical problem teams are facing right now. The ADAS repair cost commentary is accurate.

What undercuts credibility is the number of broken code examples. A senior pricing actuary would run this code and immediately get errors. That destroys the credibility of the market analysis content, which is actually solid. The code failures make the whole post look generated without verification.

### Tone

The market analysis reads well. Direct, specific, no buzzwords. The broken code sections feel like they were added from a different source — the API mismatch with the monitoring library is so basic that it reads like the code was written without reference to the actual library.

### Verdict

Do not publish. Fix:
1. `MonitoringReport` call: correct all parameter names
2. `RateChangeEvaluator`: rename `treatment_period` to `change_period`, rename `weight_col` to `exposure_col`
3. `ae_ratio`: rename `expected` to `predicted`
4. `ae_ratio_ci`: correct the return value unpacking (it returns a dict, not a tuple)
5. `result.volume_retention`: rename to `result.expected_retention`
6. `PS21/11` reference: change to `PS21/5`

---

## Cross-Cutting Issues

### Pattern of API errors

The market intelligence post has four separate API errors. The PS21/5 post has an installation command inconsistency. The double-lift post has a call-site convention bug. This pattern — correct library structure with wrong parameter names — is consistent with content generated from library descriptions rather than actual execution. The implication is that code examples are being checked for plausibility, not run.

**Recommendation:** Every code block in every post must be executed against the actual library before publication. Not read against the library source. Executed. The market intelligence post would have failed on the first `MonitoringReport()` call.

### Dead forward links

The ANAM post links to `/2026/12/14/per-risk-volatility-scoring-with-distributional-gbms/` — a post dated December 2026 that does not exist yet. With 60 posts published today, it is likely that other posts have similar forward references. Run a link-check across all today's posts before they go live.

### Duplicate content

The post list includes obvious duplicates: `consumer-duty-fair-value-checklist` and `consumer-duty-fair-value-evidencing-12-step-checklist-pricing-actuaries` on the same date. Similarly `evt-and-ml-for-tail-variable-importance` and `evt-meets-machine-learning`. And `motor-pricing-floor-when-to-stop-cutting` and `motor-pricing-floor-when-youve-stopped-burning`. These were not in my spot-check sample but they need to be reviewed and either merged or differentiated clearly. Publishing near-duplicate posts is worse than publishing nothing — it looks like content padding.

### Regulatory cross-referencing

Both posts that touch PS21/5 do so accurately, but the market intelligence post uses "PS21/11" once. A simple text search for regulatory references (PS, TR, CP, ICOBS, PRIN) across all 60 posts would be a low-cost QA step to catch attribution errors.

---

## Summary Actions Required Before Full Batch Goes Live

**Blocking (do not publish):**
- `uk-motor-is-going-loss-making-again.md` — fix four API errors and one regulatory reference

**Fix before publishing (publish after fix, no re-review needed):**
- `how-to-build-double-lift-chart-python.md` — fix double-exposure in call site convention
- `ps21-5-renewal-pricing-end-to-end-python.md` — change `pip install` to `uv add`; verify insurance-fairness and insurance-governance APIs; add ENBP estimation paragraph

**Publish with optional improvement:**
- `actuarial-neural-additive-model-anam-arxiv-2509-08467.md` — remove or stub the dead December 2026 forward link
- `fca-ep25-2-gipp-market-outcomes-cost-push-not-regulation.md` — verify `insurance-trend` API if possible

**Batch-level:**
- Run a link-checker across all 60 posts for dead links
- Audit all code examples against actual libraries by execution, not by reading
- Review duplicate-looking titles and either differentiate or consolidate
