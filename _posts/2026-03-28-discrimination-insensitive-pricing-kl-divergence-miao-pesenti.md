---
layout: post
title: "The Nearest Fair Price: KL-Divergence and Discrimination-Insensitive Pricing"
date: 2026-03-28
categories: [fairness, regulation, research]
tags: [fairness, proxy-discrimination, kl-divergence, measure-theory, fca, ep25-2, lindholm, multicalibration, python, insurance-fairness]
description: "Miao & Pesenti (arXiv:2603.16720) derive discrimination-insensitive premiums by finding the probability measure nearest to the real-world measure in KL-divergence, subject to zero sensitivity to protected attributes. Theoretically the cleanest result in this space. Harder to operationalise than it looks."
---

The standard complaint about fairness corrections in insurance pricing is that they are unprincipled: there is no criterion for how much correction to apply, and no proof that the corrected price is unique. Different methods give different answers, and there is no theoretical basis for choosing between them.

A paper published in March 2026 — Kathleen Miao and Silvana Pesenti, "Discrimination-Insensitive Pricing" ([arXiv:2603.16720](https://arxiv.org/abs/2603.16720)) — attacks this directly. The result is mathematically the cleanest derivation of a fair premium that we have seen in the actuarial literature. The operational question is what it costs to get there.

---

## What the paper does

The setup: a pricing model uses covariates X (legitimate risk factors) and protected attributes D (gender, ethnicity, etc.). You want a premium that does not respond to D, while staying as close as possible to your original pricing basis.

Miao and Pesenti formalise this as an optimisation problem. The original pricing model corresponds to a probability measure P over outcomes. A "discrimination-insensitive" pricing measure Q is one where the Gâteaux derivative of the risk premium with respect to each protected attribute D_i is zero. The Gâteaux derivative is the directional rate of change — it asks: if I nudge the protected attribute by a small amount ε, how much does the premium move? Fairness requires that answer to be zero for every direction of nudge.

The optimisation is:

```
minimise  D_KL(Q || P)
subject to  d_{D_i} rho^Q(Y|x) = 0   for each protected attribute i
            E^Q[Y] = E^P[Y]           (expectation preservation)
```

KL divergence (D_KL) measures the information lost when you approximate P with Q. Minimising it means finding the fair pricing distribution that changes P as little as possible. The second constraint says the discrimination-insensitive measure must preserve expected claims — it is not a device for under-reserving.

The solution (Theorem 3.1) is an exponential tilting of P:

```
dQ*/dP  ∝  exp{ -η* · Φ(X, D, U_{Y|x}) - η_{m+1} · Y }
```

The multipliers η* are found numerically by solving the sensitivity constraints. Existence and uniqueness of Q* are proved, not assumed.

This is the key theoretical contribution: **the discrimination-insensitive premium exists, is unique, and is the minimum-information-loss way to achieve fairness**. Previous approaches could not say that.

---

## The multi-attribute extension

With one protected characteristic, the optimisation above solves the problem. With multiple protected characteristics, you cannot in general satisfy all sensitivity constraints simultaneously with a single exponential tilt. Section 4 handles this with a two-step procedure.

**Step 1.** Solve for the individual discrimination-insensitive measure Q_i for each protected attribute separately — the single-attribute problem run M times.

**Step 2.** Combine via a KL barycentre with weights π_i (user-specified, summing to 1):

```
minimise  Σ_i π_i · D_KL(Q || Q_i)
subject to  E^Q[Y] = E^P[Y]
```

The closed form (Theorem 4.2) is a geometric average of the individual Radon-Nikodym derivatives:

```
dQ̄/dP  ∝  ∏_i (dQ_i/dP)^π_i
```

Equal weights give the geometric mean of the individual corrections. The constrained version (Theorem 4.3) adds a scalar λ_x that enforces the expectation constraint on top of the geometric average.

The theoretical work — proving existence and uniqueness of KL barycentres for general probability measures — is claimed as an independent contribution. Within the actuarial fairness context it is new. The paper's one numerical example (linear regression, binary gender, LogNormal losses) shows the discrimination-insensitive premium interpolating between the gender-specific best estimates, deviating toward the riskier group as risk increases. That behaviour is what you would want.

---

## How it compares to existing approaches

**Lindholm-Richman causal decomposition.** The dominant actuarial approach (Lindholm, Richman, Tsanakas, Wüthrich, ASTIN Bulletin 52(1), 2022). Decomposes the price into direct and indirect effects of the protected attribute, then removes the direct effect by marginalising over the protected characteristic's distribution conditional on legitimate factors. Our [`insurance-fairness`](/insurance-fairness/) library implements this as `LindholmCorrector`. The hard part is specifying the causal graph — which paths through the model are legitimate.

Miao-Pesenti sidesteps the causal graph entirely. It does not ask which variables mediate the discrimination; it asks what is the nearest distribution under which the premium is insensitive to D. Cleaner problem statement, harder to compute.

**Denuit-Michaelides-Trufin multicalibration.** [Covered here last week.](/2026/03/27/multicalibration-fair-insurance-pricing) Requires autocalibration to hold within every protected group at every premium decile. Detects discrimination as a calibration failure. The test requires only predictions and outcomes — no distributional model, no causal graph, no root-finding. Operationally the most tractable of the three approaches.

The two are complementary. Multicalibration tells you where in your premium distribution discrimination is showing up. Miao-Pesenti tells you the theoretically nearest fair distribution. You could use multicalibration as a diagnostic and KL-minimisation as the correction — though nobody has validated this combination in practice yet.

**Wasserstein optimal transport.** EquiPy and our `WassersteinCorrector` operate in W2 distance rather than KL. Wasserstein respects the ordering of outcomes and has better geometry for continuous distributions. KL has better computational tractability via exponential tilting but can assign infinite penalty to regions where Q has zero mass and P does not. The choice of metric is a genuine theoretical question the literature has not settled.

---

## What we implement, and what we do not

`insurance-fairness` v0.6.3 includes `DiscriminationInsensitiveReweighter` — propensity-based sample reweighting that achieves approximate independence between the model and protected attributes. Weights are w_i = P(A=a_i) / P(A=a_i|X_i), which is the empirical analogue of exponential tilting the measure. The implementation is ~290 LOC and integrates with any scikit-learn `sample_weight` API.

What we do not implement: the full KL barycentre. The constrained barycentre requires estimating full conditional distributions F_{Y|X} per policyholder, solving Lagrange multiplier systems via root-finding, and re-solving for λ_x at the barycentre step. One numerical example on synthetic data is not enough to validate the approach at operational scale. Our assessment is WATCH: if the authors or others publish empirical validation on real insurance portfolios, or a reference implementation appears, we will build `KLBarycentreCorrector`. Pesenti is also co-author on the Huang-Pesenti (2025) marginal sensitivity framework (arXiv:2505.18895), which suggests applied follow-up work may come.

---

## What this means for UK pricing teams

Consumer Duty (FCA, July 2022, PS22/9) raised the bar on demonstrating absence of proxy discrimination — not asserting it. The FCA has not specified which methodology firms must use, which creates space for evidence-based approaches but also regulatory risk if the chosen method is challenged.

The Miao-Pesenti framework has one property that helps in this context: it produces the **nearest** fair price, not an arbitrary fair price. If your Lindholm-corrected and KL-minimised premiums diverge substantially, that divergence is diagnostic — it tells you something about the discrimination structure in your model that the correction alone cannot reveal.

The practical combination we would recommend: multicalibration audit to find where the model fails; Lindholm correction where you have a clear causal story; KL reweighting where the causal structure is ambiguous; post-correction multicalibration to confirm the corrections held. The barycentre is a theoretical tool for now, but knowing the nearest fair price exists and is unique is a better basis for attestation than hoping your method is reasonable.

---

**Reference:** Miao, K. and Pesenti, S. (2026). Discrimination-Insensitive Pricing. [arXiv:2603.16720](https://arxiv.org/abs/2603.16720).

**See also:**
- Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2022). Discrimination-Free Insurance Pricing. ASTIN Bulletin 52(1), 55–89.
- Denuit, M., Michaelides, M. and Trufin, J. (2026). Balance and Fairness through Multicalibration in Nonlife Insurance Pricing. [arXiv:2603.16317](https://arxiv.org/abs/2603.16317).
- [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) — `DiscriminationInsensitiveReweighter`, `LindholmCorrector`, `WassersteinCorrector`, `MulticalibrationAudit`.
