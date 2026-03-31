---
layout: post
title: "When Your Calibration Set No Longer Matches Your Book — KMM-CP for Covariate Shift"
seo_title: "KMM-CP: Conformal Prediction under Covariate Shift via Kernel Mean Matching for Insurance Pricing"
date: 2026-03-31
categories: [techniques]
tags: [conformal-prediction, covariate-shift, kernel-methods, RKHS, MMD, selective-abstention, book-transfer, GBM, insurance-pricing, python, Solvency-II, FCA, arXiv-2603-26415, Tibshirani-2019, uncertainty-quantification]
description: "Laghuvarapu, Deb and Sun (arXiv:2603.26415, March 2026) replace per-test-point density ratio estimation with bounded QP weights on the calibration set. Here is what that buys you, what it costs, and which parts are actionable for UK pricing teams right now."
author: burning-cost
---

Conformal prediction gives you a finite-sample coverage guarantee. But the guarantee rests on exchangeability: calibration data and test data drawn from the same distribution. Covariate shift breaks that assumption. And for insurance pricing, covariate shift is not an edge case — it is the normal deployment condition.

When a motor insurer acquires an MGA and applies their own GBM to the acquired book, the vehicle age and occupation mix will differ from the acquirer's training data. When a product is repriced after a channel mix change, the risks flowing through a new aggregator have different covariate distributions from the panel business that calibrated the model. When you launch a new SME cyber product, your calibration set is from legacy technology sector clients; the first year's new business will be overwhelmingly retail and hospitality. Standard conformal intervals — even the Tibshirani (2019) weighted version — can fail badly in these conditions.

KMM-CP (arXiv:2603.26415, Laghuvarapu, Deb and Sun, submitted 27 March 2026) addresses this. It is not a complete solution and it has real limitations. But the selective abstention piece is the most actionable development in this space we have seen this year.

---

## The problem with Tibshirani 2019

The standard approach to conformal prediction under covariate shift — Tibshirani, Barber, Candès and Ramdas (NeurIPS 2019, arXiv:1904.06019) — works by importance-weighting the empirical quantile. Each calibration point receives weight proportional to the density ratio:

$$w(x_i) = \frac{p_{\text{target}}(x_i)}{p_{\text{source}}(x_i)}$$

The weighted quantile replaces the standard unweighted one, restoring marginal coverage on the target distribution. The coverage guarantee is finite-sample and exact, given correctly estimated weights.

That "given" is the problem. You need per-test-point density ratio estimates. In practice this means training a classifier to distinguish source from target, then reading off a probability ratio. On moderate-dimensional data (15–30 pricing features, which is typical for a UK motor model), the classifier is sensitive to its own calibration. Small systematic biases in logistic regression probability estimates propagate directly into weight instability. When the shift is severe — the target distribution has regions with near-zero source density — the estimated weights blow up, and the weighted quantile collapses to the maximum calibration score or worse.

Our `insurance-covariate-shift` library implements Tibshirani 2019 in `ShiftRobustConformal('weighted')`. The docstring is honest about this: the current implementation uses mean calibration weights as a proxy, which is approximately valid but does not carry the full finite-sample guarantee when test weights are heterogeneous. That approximation exists precisely because the per-test-point weights are unstable in practice.

KMM-CP takes a different route.

---

## What KMM-CP does differently

Instead of estimating density ratios at individual points, KMM-CP matches *moment embeddings* in a reproducing kernel Hilbert space (RKHS). The method minimises the Maximum Mean Discrepancy (MMD) between the re-weighted source distribution and the target:

$$\min_{\beta \in \mathcal{B}} \left\| \frac{1}{n_s} \sum_{i=1}^{n_s} \beta_i \phi(x_i) - \frac{1}{n_t} \sum_{j=1}^{n_t} \phi(x_j) \right\|_{\mathcal{H}}^2$$

where $\phi: \mathcal{X} \to \mathcal{H}$ is the feature map induced by a kernel $k(x, x')$, and $\mathcal{B}$ is a convex constraint set bounding the weights: $0 \leq \beta_i \leq B$ with $\sum_i \beta_i = n_s$. The constraint set prevents the weight explosion that plagues density-ratio methods.

This is a quadratic programme in $n_s$ variables (the calibration set size). The bounded solution is obtained via standard QP solvers, and the weights are used directly in the conformal quantile calculation — the same weighted quantile structure as Tibshirani, but with weights derived from distributional matching rather than pointwise density estimation.

The tradeoff is the guarantee. Tibshirani's method gives a **finite-sample** coverage guarantee — valid for all $n$, provided the density ratios are correct. KMM-CP's guarantee is **asymptotic only**: coverage converges to $1 - \alpha$ as $n \to \infty$, but there is no finite-sample bound. For Solvency II and SCR modelling, where you want citable finite-sample guarantees, this is a meaningful weakening. We will be direct: do not use KMM-CP as your primary conformal method for capital modelling. Tibshirani (or the h-transformation approach in `insurance-conformal`) is the right choice when the exchangeability assumption approximately holds.

KMM-CP's advantage is stability under severe shift. When the density ratio has extreme values at specific covariate combinations — which is common in book transfers and new product launches — the bounded weights are more reliable than unbounded ratio estimates. For *interval validity*, you are trading the finite-sample guarantee for robustness. For *decision-making* under shift, that may be the right trade.

---

## Selective abstention: the actionable piece

The paper's most valuable contribution is not the KMM-CP weights themselves. It is SKMM — selective KMM — which jointly optimises the weights and binary selection variables:

$$\min_{\beta, s} \text{MMD}^2(\text{re-weighted source}, \text{selected target})$$

subject to $s_j \in \{0, 1\}$ for each test point $j$, with a constraint on the fraction of points that must be selected (a coverage budget $\rho$).

The selection variables $s_j$ identify which target points lie within the support overlap between source and target distributions. A test point with $s_j = 0$ is flagged as out-of-support: the model should not produce a prediction interval for it, because no amount of re-weighting can recover a valid coverage guarantee there.

In underwriting terms, this is a formal implementation of "refer to manual."

When you apply your GBM to a new book, some risks will be genuinely out-of-distribution — not just from a different part of the same covariate space, but from a region your calibration data cannot cover. Standard conformal, including Tibshirani, produces an interval for these risks anyway — usually a wide one, but an interval nonetheless. SKMM refuses to produce an interval, instead flagging the risk for individual assessment.

This has three immediate applications in UK insurance:

**Book transfer underwriting.** On an acquired portfolio, SKMM identifies which policies fall outside the overlap between the acquired book and your calibration data. Policies with $s_j = 0$ should not be auto-rated; they need individual underwriter review. This is not a new concept — underwriters have always had a gut feel for "this doesn't look like our book" — but SKMM gives a decision-boundary that is documentable and reproducible.

**New product launch pricing.** When your calibration set is from legacy business and new business has materially different covariate distribution, SKMM can identify the subset of new risks where coverage guarantees cannot be established. Rather than setting a uniform wide interval for all new business, you identify the specific risks where the interval is uninformative and escalate them.

**FCA SUP 15.3 documentation.** The FCA expects firms to document the limitations of their models. "Model is not valid for risks outside the following covariate region" is a documentable, auditable claim when it is produced by a defined algorithm. SKMM generates exactly this output. This is stronger than "we reviewed the model documentation and believe it is appropriate" — which is what most current model governance produces.

---

## Computational reality

The O($n_s^2$) QP is the method's practical ceiling. The kernel matrix is $n_s \times n_s$; for a calibration set of 500,000 policies (not unusual for a major personal lines book with rolling 24-month calibration), that is 2.5 × 10¹¹ kernel evaluations. Even with sparse approximations, this is not tractable on standard hardware.

The paper's benchmarks — molecular property prediction datasets with hundreds to a few thousand samples — are a very long way from UK insurance calibration set sizes. The authors do not address this. KMM-CP as published is a method for small to medium calibration sets.

In practice, this means either:

1. Subsample the calibration set (e.g., 5,000–10,000 points via stratified sampling) and accept reduced power in the weights.
2. Use the Nyström approximation to the kernel matrix (rank-$r$ approximation with $r \ll n_s$), which reduces the QP to $r$ variables. This is standard in the kernel methods literature but not implemented in the paper's codebase.
3. Apply KMM-CP at the segment level rather than across the full calibration set — separately per major product line or risk tier, where calibration sets are smaller.

Option 3 is probably the most natural for insurance: your shift problem is usually not uniform across the portfolio. The book transfer case concentrates the shift in specific risk segments. Fitting separate KMM-CP instances per segment, with calibration sets of 10,000–50,000 records, is computationally feasible.

---

## How this relates to what we already have

The `insurance-covariate-shift` library currently offers two approaches for conformal prediction under shift:

1. `ShiftRobustConformal('weighted')` — Tibshirani 2019 weighted quantile, finite-sample guarantee. Uses mean calibration weights as proxy for per-test-point weights.
2. `ShiftRobustConformal('lr_qr')` — the LR-QR method (arXiv:2502.13030), which learns a covariate-dependent threshold via likelihood-ratio regularisation. No per-test-point weight requirement.

KMM-CP sits between these. It is more robust than the standard Tibshirani implementation when shift is severe, but less so than LR-QR for high-dimensional problems (LR-QR's regularised likelihood ratio is better behaved in high dimensions). KMM-CP's selective abstention capability — SKMM — has no equivalent in either of the existing methods; that is its unique contribution.

The four gaps we have identified for implementation work, in order of value vs. effort:

**Gap 2 (high value, low effort): `SelectiveAbstainer` wrapper.** Wrap SKMM's selection step as a post-processing layer on any `ShiftRobustConformal` instance. Input: calibration set, test set. Output: `(selected_mask, intervals_for_selected)`. The underlying weights can come from any method; the selection optimisation is independent. Estimated 2–3 days of implementation work.

**Gap 4 (high value, medium effort): per-test-point weights.** The current `ShiftRobustConformal('weighted')` uses mean calibration weights as the proxy for per-test-point weights. Replace this with the proper per-test-point KMM weights — one weight per test observation — to recover the full Tibshirani coverage guarantee on heterogeneous test batches. This requires the QP at each prediction call, so it is computationally expensive; it makes sense only on small test batches (individual underwriting decisions, not portfolio repricing runs).

**Gap 1 (medium value, medium effort): `KMMConformal` class.** A standalone class exposing the full KMM-CP method for users who want the RKHS matching approach directly, with Nyström approximation for calibration sets above 10,000 records.

**Gap 3 (medium value, high effort): benchmark on insurance data.** The paper has no insurance experiments. Running KMM-CP, Tibshirani, and LR-QR on a synthetic book transfer scenario (two distributions matching the shift profile of a real UK motor acquisition) would tell us under what severity of shift KMM-CP's stability advantage outweighs the asymptotic vs. finite-sample guarantee trade-off.

---

## The guarantee question, honestly

The finite-sample vs. asymptotic distinction matters more in insurance than in the molecular biology applications the paper benchmarks against. Actuaries working within Solvency II and the PRA's SS1/23 framework need to be able to say "this coverage guarantee holds for this sample size." Asymptotic guarantees do not provide that.

If you are using conformal intervals for SCR estimation — as we described in the distribution-free conformal post — you need the finite-sample guarantee from Tibshirani or from the h-transformation approach. KMM-CP's asymptotic guarantee is insufficient for capital modelling sign-off.

For decision-making under operational covariate shift — pricing a new book, launching a new product, monitoring post-acquisition — the asymptotic guarantee is often acceptable. The question becomes empirical: does the method achieve nominal coverage at your actual calibration set size? On a 10,000-record calibration set, the asymptotic approximation may be reasonable. On 500 records, it is not.

We think the right use of KMM-CP in 2026 is: SKMM selection as a refer-or-rate trigger, with Tibshirani or LR-QR providing the actual intervals for the selected risks. Use SKMM's output to identify which risks need interval production at all; use a method with a stronger guarantee to produce the intervals.

---

## What the paper does not address

Beyond the computational scalability issue, two gaps stand out.

**Kernel choice.** The paper uses a Gaussian RBF kernel with bandwidth set via the median heuristic. For mixed continuous/categorical insurance features — vehicle type (categorical), driver age (continuous), no-claims discount class (ordinal) — there is no guidance on kernel construction. The median heuristic is reasonable for continuous features but has no obvious generalisation to one-hot encoded categoricals. This is not a dealbreaker, but it requires thought before applying KMM-CP to standard UK insurance feature spaces.

**No insurance experiments.** All benchmarks are from molecular property prediction (QM9, PCBA, MolPCBA datasets). The shift profiles there — different molecular substructures in training vs. test — are qualitatively different from insurance shift (different portfolio composition within the same feature space, but with similar marginal feature distributions). Whether KMM-CP's stability advantage holds for insurance-type shift, where the marginal distributions overlap substantially but joint distributions differ in high-dimensional ways, is an open question.

---

## References

1. Laghuvarapu, S., Deb, R. & Sun, J. "KMM-CP: Practical Conformal Prediction under Covariate Shift via Selective Kernel Mean Matching." arXiv:2603.26415, March 2026.
2. Tibshirani, R.J., Barber, R.F., Candès, E.J. & Ramdas, A. "Conformal Prediction Under Covariate Shift." NeurIPS 32, 2019. arXiv:1904.06019.
3. Marandon, A. et al. "Conformal Inference under High-Dimensional Covariate Shifts via Likelihood-Ratio Regularization." arXiv:2502.13030, 2025.
4. Huang, J., Smola, A., Gretton, A., Borgwardt, K.M. & Schölkopf, B. "Correcting Sample Selection Bias by Unlabeled Data." NeurIPS 2006. (Original KMM paper.)
5. Gretton, A., Borgwardt, K.M., Rasch, M.J., Schölkopf, B. & Smola, A. "A Kernel Two-Sample Test." *JMLR* 13: 723–773, 2012.

The `insurance-covariate-shift` library is at [/insurance-covariate-shift/](/insurance-covariate-shift/). `SelectiveAbstainer` is on the roadmap for v0.3.0.
