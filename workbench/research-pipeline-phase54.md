---
title: "Strategist Scan #34: Phase 54 Candidates"
artifact_type: research_brief
created_by: strategist
tags: [phase54, pipeline, scan34, strategy]
---

# Strategist Scan #34: Phase 54 Candidates

**Date:** 2026-04-02  
**Follows:** Scan #33 (KB 5490) / Phase 53  
**Context:** 34 published libraries, 434 blog posts, ~26K downloads/month. No new repos.

---

## Phase 53 Delivery Audit

Phase 53 is complete. Final state:

| Item | Status |
|------|--------|
| ConditionalCoverageAssessor (insurance-conformal) | BUILT — CVI decomposition in conditional_coverage.py (v0.7.x) |
| FairMultiStateTransitionFitter (insurance-fair-longterm v0.2.0) | BUILT — 2026-04-02 |
| BAWSMonitor notebook | DONE — KB 5495 |
| RobustReinsuranceOptimiser notebook | DONE — KB 5495 |
| LifetimeBoundsCalculator notebook | DONE — KB 5495 |
| ConditionalCoverageAssessor blog | DONE — 2026-04-02 post (KB 5494) |
| FairMultiStateTransitionFitter blog | DONE — 2026-03-31 + 2026-04-01 posts |
| RobustReinsuranceOptimiser blog | DONE — 2026-04-02 post |
| DynamicReinsurancePolicyOptimiser | BLOG ONLY — 2026-03-31 RL reinsurance post. NOT BUILT (correct decision: RL training instability, torch dependency). |

**Phase 53 verdict: COMPLETE. No open items.**

---

## arXiv Coverage Window: March-April 2026

April 2026 arXiv is effectively empty (scan date is 2026-04-02, only ~3 papers exist with 2604.xxxxx IDs, none insurance-relevant). The productive window is late March 2026 papers not yet assessed, plus retrospective check on papers assessed but not yet actioned.

### Papers Confirmed Already Done (deduplication)

- arXiv:2603.27189 CVI/ConditionalCoverageAssessor: BUILT + blog (KB 5493, 5494)
- arXiv:2603.16317 Multicalibration: BUILT (KB 4684) + blog
- arXiv:2603.16720 Discrimination-insensitive pricing: BUILT (KB 4963) + blog
- arXiv:2603.25224 Localized demographic parity: BUILT (KB 5451) + blog
- arXiv:2603.17106 Proxy race fairness: BLOG (KB 4011)
- arXiv:2603.02418 Flood geolocated data: In KB (KB 5199), blog MISSING
- arXiv:2504.06984 Weak signals EVT-ML: BLOG (KB 3730)
- arXiv:2603.15839 Telematics risk index: In KB (KB 3828)
- arXiv:2603.03789 Shapley mortality ensemble: BLOG (KB 5417)
- arXiv:2603.20518 MDMx Tucker mortality: In KB (KB 3993), blog MISSING
- arXiv:2603.11660 One-shot reserving: BLOG (KB 3661)
- arXiv:2504.09396 CVaR-RL reserving: In KB (KB 4774), blog MISSING
- arXiv:2603.10674 Conformal functional mortality: In KB (KB 3983), blog-only verdict
- arXiv:2601.07675 Tab-TRM: BLOG (KB 3706)
- arXiv:2504.11775 Privatized sensitive attributes: In KB (KB 4851), NOT YET BUILT
- arXiv:2603.00973 DMP cause-specific mortality: BUILT + blog (KB 5173)
- arXiv:2603.29530 Community-based insurance ruin: NEW — no prior KB entry

---

## Phase 54 Candidates — Scored

### CANDIDATE 1 — TOP BUILD: PrivatizedFairPricer (insurance-fairness v0.9.0)

**Paper:** Zhang, Liu & Shi (April 2025) arXiv:2504.11775  
**Title:** "Discrimination-free Insurance Pricing with Privatized Sensitive Attributes"

**What it builds:** `PrivatizedFairPricer` — fits a pricing GLM under regulatory constraints where sensitive attributes are only available in privatized (differentially-private) form. Two-step procedure: (1) privatization layer applying Laplace mechanism (continuous attributes) or Gaussian mechanism (binned ordinal attributes) to sensitive features; (2) constrained GLM training enforcing demographic parity conditional on privatized proxies with statistical guarantees. Adapts to varying regulatory transparency levels.

**Why now:** arXiv:2504.11775 has been in KB since Phase 14 (KB 915) with attribution corrected in KB 4843/4851. No build record exists anywhere. The DiscriminationInsensitiveReweighter (v0.6.3) covers KL-divergence reweighting (arXiv:2603.16720) — a different mechanism. After insurance-fairness v0.9.0, the library will cover the full 2024-2026 actuarial fairness literature.

**Gap the build fills:** Colorado SB21-169 testing framework and NY Insurance Circular Letter No. 7 both contemplate proxy/privatized attribute fairness testing. Insurers must demonstrate fairness without holding protected data. No open-source Python tool implements this.

**Scoring:**
| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty | 4/5 | Differential privacy + insurance fairness is new territory for the library |
| Insurance Demand | 4/5 | Direct regulatory driver in CO/NY/EU AI Act |
| Feasibility | 3/5 | DP noise mechanics add complexity; scipy for noise generation, sklearn for constrained GLM |
| Regulatory Relevance | 5/5 | Precisely addresses the regulator use-case: audit fairness without holding protected data |
| Differentiation | 4/5 | No competing Python actuarial fairness library has this |
| **Total** | **20/25** | |

---

### CANDIDATE 2 — BUILD: LinearRiskSharingPool (insurance-optimise v0.6.0)

**Paper:** arXiv:2603.29530 (March 31, 2026)  
**Title:** "Linear Risk Sharing in Community-Based Insurance: Ruin Reduction in the Compound Poisson Model"

**What it builds:** `LinearRiskSharingPool` — a mutual/community insurance pool simulator. Given N participants each with their own Cramér-Lundberg surplus process (arrival rate λ_i, claim severity F_i), applies a linear allocation matrix A to redistribute claims at occurrence time and computes: (a) theoretical sufficient conditions check (scale family test, full allocation check, bounded transfer check), (b) infinite-time ruin probability improvement vs standalone (Monte Carlo or Panjer recursion), (c) ruin probability as a function of pool size and allocation rule parameters.

**Why it matters:** P2P/community insurance is growing in UK (Laka bicycle, Dinghy boat, Neos home), EU (Friendsurance), and Lloyd's syndicate pool structures. No open-source Python tool covers the actuarial mathematics of when risk pooling provably improves solvency. The paper's result is clean: under the three conditions, ruin probability decreases for every participant. The sufficient conditions check is immediately auditable.

**Target library:** insurance-optimise (natural fit alongside RobustReinsuranceOptimiser; both are "structure of risk transfer" tools).

**Scoring:**
| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty | 4/5 | No existing KB build covers mutual pool ruin reduction |
| Insurance Demand | 3/5 | Niche but growing; mutuals + P2P + Lloyd's syndicates |
| Feasibility | 4/5 | Cramér-Lundberg ruin probability has well-established numerical methods |
| Regulatory Relevance | 3/5 | PRA supervises mutuals; Solvency II ruin probability is a standard input |
| Differentiation | 4/5 | Zero competition in open-source actuarial Python |
| **Total** | **18/25** | |

---

### CANDIDATE 3 — BUILD: MonotonicStochasticCountModel (insurance-glm-tools)

**Paper:** arXiv:2602.02398 (February 2026)  
**Title:** "Counting models with excessive zeros ensuring stochastic monotonicity"

**What it builds:** `MonotonicStochasticCountModel` — a zero-inflated or hurdle Poisson/NB frequency model with random effects structured to preserve stochastic monotonicity. The class wraps the paper's constrained random-effect parameterization so that policyholders with worse claim history always receive a weakly higher posterior credibility premium. Exposes: `fit(X, y, exposure)`, `predict_credibility_premium(X, claim_history)`, `check_monotonicity(claim_history_range)`.

**Why it matters:** Standard Boucher/Denuit zero-inflated mixed Poisson models — the workhorse of motor/property frequency modeling — can violate stochastic monotonicity when combined with experience rating. This produces the counterintuitive (and indefensible) result that a policyholder with 3 prior claims could receive a lower renewal premium than one with 2. The fix is tractable. This is the kind of bug that fails a PRA model validation.

**Target library:** insurance-glm-tools (sits alongside existing ZI GLM tools in the library).

**Risk note:** Paper is purely theoretical with no reference code. Implementation requires deriving the constrained random-effect structure from the propositions, which adds engineering risk. Recommend engineer reviews Propositions 2-4 before committing to a sprint.

**Scoring:**
| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Novelty | 4/5 | Identifies and fixes a genuine, rarely-discussed inconsistency in standard freq models |
| Insurance Demand | 4/5 | Claims frequency + credibility = core to motor/property pricing |
| Feasibility | 3/5 | No reference code; requires deriving from propositions |
| Regulatory Relevance | 3/5 | FCA/PRA model validation; stochastic monotonicity is a defensible coherence requirement |
| Differentiation | 3/5 | No competing library has this; Emblem/Earnix don't address this |
| **Total** | **17/25** | |

---

### CANDIDATE 4 — BLOG: CVaR-RL Reserving Commentary

**Paper:** arXiv:2504.09396 (Dong & Finlay, 2025)  
**KB:** 4774/4775 (detailed honest assessment already done)

**Recommended title:** "CVaR-Constrained RL for Insurance Reserving: What It Is, What It Isn't, and Why That Matters for Solvency II"

**Angle:** The gap analysis in KB 4775 is honest and rigorous — this paper is a reserve buffer management tool, not a triangle completion method. The blog should make that distinction clearly while giving the paper credit for what it does do (tail-risk-aware dynamic reserve adjustment under regime change). Key comparison: ODP bootstrap + post-hoc CVaR extraction vs RL approach.

**No library build.** Blog-only.

---

### CANDIDATE 5 — BLOG: Stochastic Monotonicity and Zero-Inflated Frequency Models

**Paper:** arXiv:2602.02398 (February 2026)

**Recommended title:** "Your Zero-Inflated Frequency Model Fails a Basic Coherence Test"

**Angle:** Accessible explanation of stochastic monotonicity in the experience rating context. Why the standard zero-inflated mixed Poisson model can give a lower renewal premium to a policyholder with 3 claims vs 2. What the sufficient conditions for monotonicity are. What the fix looks like. Pairs with BUILD Candidate 3 if the build goes ahead; can also run ahead as a blog teaser.

---

### CANDIDATE 6 — BLOG: Community Insurance Pool Design and Ruin Reduction

**Paper:** arXiv:2603.29530 (March 31, 2026)

**Recommended title:** "When Does Sharing Risk Actually Reduce Ruin? The Mathematics of Mutual Insurance Pools"

**Angle:** Cover the Cramér-Lundberg framework, what the three conditions mean in plain terms, and where risk pooling breaks down (non-scale-family severities, unbounded transfer obligations). Market context: Laka, Dinghy, Neos in UK; COBA mutual sector; FCA's peer-to-peer regulatory developments. Pairs with BUILD Candidate 2.

---

### CANDIDATE 7 — BLOG: MDMx Coherent Mortality Forecasting

**Paper:** arXiv:2603.20518 (Clark, March 2026) — in KB at ID 3993, no blog yet.

**Recommended title:** "Tucker Tensors for Coherent Mortality Forecasting: What Actuaries Should Know"

**Angle:** Cover MDMx's Tucker tensor decomposition, why sex-age coherence matters for annuity reserving, and the Kalman filtering forecasting component. Insurance application: annuity portfolio reserves require consistent male/female mortality projections; MDMx provides this by construction. Contrast with Lee-Carter's independence of sex-specific models.

---

### CANDIDATE 8 — BLOG: Flood Geolocated Data for Insurance Risk

**Paper:** arXiv:2603.02418 (Moriah et al., March 2026) — in KB at ID 5199, no blog yet.

**Recommended title:** "Building-Level Flood Risk: Why Postcode Isn't Enough for UK Property Insurance"

**Angle:** Sequential variable introduction methodology (baseline → climate indicators → rainfall metrics → building attributes). Marginal lift from fine-grained geolocation. UK application context: Flood Re 2039 transition from subsidized to market-based flood pricing means insurers must rebuild flood rating from scratch — building-level granularity is the competitive edge. Pairs with the existing Flood Re 2039 blog (2026-04-02 post).

---

## Recommended Phase 54 Sequence

### Build Order (priority)

1. **PrivatizedFairPricer** — insurance-fairness v0.9.0 (Score 20/25, highest regulatory demand)
2. **LinearRiskSharingPool** — insurance-optimise v0.6.0 (Score 18/25, clean implementation, growing market)
3. **MonotonicStochasticCountModel** — insurance-glm-tools (Score 17/25, technical depth, validate engineering feasibility first)

### Blog Order (can run in parallel with builds)

4. CVaR-RL Reserving Commentary (standalone, no build dependency)
5. Stochastic Monotonicity + ZI Count Models (pairs with Build 3; can precede it)
6. Community Insurance Pool Design (pairs with Build 2)
7. MDMx Coherent Mortality (standalone)
8. Flood Geolocated Data (standalone)

### Missing Coverage Notebooks (carry-forward)

No notebooks are identified as missing from Phase 53. All three Phase 52 notebooks (BAWSMonitor, RobustReinsuranceOptimiser, LifetimeBoundsCalculator) were delivered in Phase 53 (KB 5495).

---

## Library Targeting Summary

| Library | Phase 54 Action | Target Version |
|---------|-----------------|---------------|
| insurance-fairness | BUILD: PrivatizedFairPricer | v0.9.0 |
| insurance-optimise | BUILD: LinearRiskSharingPool | v0.6.0 |
| insurance-glm-tools | BUILD: MonotonicStochasticCountModel | next minor |
| insurance-survival | Blog only (MDMx) | no version change |
| insurance-conformal | No action | at v0.7.1 |
| insurance-monitoring | No action | at v1.1.0 |

---

## Strategic Context

The fairness library (insurance-fairness) has been the dominant build target across Phases 49-53. PrivatizedFairPricer is the last major gap. After v0.9.0, insurance-fairness will cover:

- Pre-processing: reweighting, adversarial debiasing
- In-processing: fairness-constrained GLM
- Post-processing: multicalibration (v0.7.x), localized DP correction (v0.8.0)
- Audit: marginal fairness premium, KL discrimination-insensitive measure
- Regulatory privacy: privatized attribute pricing (v0.9.0)

That is a complete picture of the 2024-2026 actuarial fairness literature. After Phase 54, the fairness cluster reaches saturation — future fairness work will be incremental enhancement, not gap-filling.

The community insurance and monotonic count model builds extend into less-crowded territory (mutual pools, coherent credibility) where there is zero open-source competition.

**April 2026 arXiv note:** Only ~3 papers exist with 2604.xxxxx IDs as of 2026-04-02, none insurance-relevant. Recommend supplemental scan in mid-April when 3-4 weeks of April papers have accumulated.
