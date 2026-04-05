---
layout: post
title: "Minimum-Distortion Fair Pricing: What Miao & Pesenti Actually Prove (and What We Already Build)"
date: 2026-04-03
categories: [fairness, libraries]
tags: [fairness, kl-divergence, discrimination-insensitive, consumer-duty, equality-act, insurance-fairness, measure-change, radon-nikodym, marginal-fairness, fca-ep25-2, proxy-discrimination, arXiv-2603.16720, miao, pesenti, uk-personal-lines]
description: "Miao & Pesenti (2026) derive the nearest fair probability measure in KL divergence. Their existence theorem is useful for FCA Consumer Duty attestation. Our insurance-fairness library already implements the propensity approximation. Here is where we are, what the full theorem gives you, and whether to build it."
author: burning-cost
---

There is a paper that belongs in every UK actuarial fairness toolkit: Miao & Pesenti (arXiv:2603.16720, March 2026). Not because it produces a method you need to implement today — we will explain why the full version is premature — but because it provides the cleanest theoretical foundation for work we are already doing, and because its existence theorem gives pricing teams something concrete to say to the FCA.

---

## The problem it addresses

You have a GBM or GLM trained on UK motor or home data. You have dropped gender from the feature matrix. The model still implicitly uses gender because postcode, vehicle group, and occupation are all correlated with it. Dropping the column does not remove the information; it just obscures the path.

The standard response is either Lindholm-style marginalisation (average out the protected attribute given the other features) or propensity reweighting (adjust sample weights so that features are independent of the protected attribute at training time). Both are reasonable. Neither has a clear answer to the question: "how much have you distorted the pricing?"

Miao & Pesenti answer that question precisely. They look for the *nearest* probability measure Q to your actual measure P — where nearness is KL divergence — under which the pricing principle is insensitive to the protected attribute. Because KL divergence has an information-theoretic interpretation as the minimum additional bits of description needed to encode P using Q as a codebook, the result is provably the least-distorting correction that satisfies the fairness constraint. You are not over-correcting. You are not artificially compressing risk differentials. You are applying the minimum necessary adjustment.

---

## What "insensitive" means technically

The paper's fairness criterion is a Gateaux derivative. For protected attribute D_i and distortion risk measure rho_gamma, the sensitivity at rating cell x is:

```
∂_{D_i} rho(Y|x) = E_x[ D_i · ∂_i h(X,D) · gamma(U_{Y|x}) ]
```

where h(X,D) is your pricing model and gamma is the distortion weight function (gamma = 1 for expected value pricing, gamma = 1/(1-alpha) · 1_{u > alpha} for Expected Shortfall at level alpha).

This is a *local* notion of fairness. It asks: at this specific rating cell, does a marginal change in D_i move the price? It is not asking whether the distribution of prices differs across groups (that is demographic parity — a different and weaker criterion). It is asking whether the pricing function *responds to* the protected attribute.

For expected-value pricing, this is equivalent to asking whether the protected attribute contributes to the conditional mean prediction. For distortion risk measures like ES, it is more sensitive — the distortion weight gamma(u) upweights the tail, so if D_i affects the tail of the loss distribution more than the mean, the sensitivity measure catches that even when the expected-value sensitivity is zero.

---

## The optimisation result

The discrimination-insensitive measure Q* solves:

```
Q* = argmin_{Q << P} D_KL(Q || P)
subject to:
    ∂_{D_i} rho^Q(Y|x) = 0   for all protected attributes i
    E_x^Q[Y] = E_x^P[Y]                          (actuarial balance)
```

**Theorem 3.1** establishes that Q* has a Radon-Nikodym derivative:

```
dQ_x*/dP = (1/C_x) · exp( -eta_x*^T · Phi(X,D,U_{Y|x}) - eta_{m+1}* · Y )
```

This is exponential tilting of P in two directions: the first term downweights scenarios where the model responds strongly to protected attributes; the second term preserves actuarial balance by re-centering the expected loss. The tilting is done at the level of (observation, conditional quantile) pairs, not at the level of groups.

**Theorem 3.2** proves uniqueness: the Lagrange multiplier system has a unique solution when the cumulant generating function of the sensitivity function Phi and the loss Y exists in a neighbourhood of the solution. For UK motor frequency (Poisson counts) and severity (log-normal, gamma), this condition holds. For catastrophe risks in very sparse cells, it may not — use a bounded distortion gamma that assigns zero weight to extreme quantiles.

---

## Multiple protected attributes: the KL barycentre

When you are handling gender, age band, and proxied ethnicity simultaneously, the constraints for each attribute may not be jointly satisfiable. Theorem 4.3 resolves this via a KL barycentre: compute per-attribute measures Q_i, then find the measure that minimises a weighted sum of KL distances to each Q_i, subject to actuarial balance:

```
dQ_bar/dP  proportional to  exp(-lambda_x · Y) · ∏_i (dQ_i/dP)^pi_i
```

The weights pi_i are a governance decision: how much relative priority does each attribute's fairness constraint get? Equal weighting (pi_i = 1/m) is the neutral default but has no theoretical justification — the paper is honest about this. For a UK motor team handling gender, age as a protected attribute (post-discrimination concerns rather than rating use), and ONS-proxied ethnicity, the weight choice is a conversation between the actuarial team, the risk function, and the compliance team. It should be documented and owned.

---

## Why this matters for FCA Consumer Duty and the Equality Act

**Consumer Duty (PS22/9, PRIN 2A)** requires firms to demonstrate that pricing models do not disadvantage protected groups disproportionately. Most firms currently answer this with SHAP importance plots and demographic parity checks. These are defensible but imprecise: they show that the protected attribute is not highly ranked as a feature, not that the model does not respond to it.

The Gateaux sensitivity criterion is more direct. If you can demonstrate that `∂_{D_i} rho(Y|x) = 0` at every rating cell — i.e., that Q* has been applied — you have shown the pricing function does not respond to marginal changes in the protected attribute at any point in the portfolio. This is a stronger statement than demographic parity and more aligned with what the **Equality Act 2010** actually prohibits: differential treatment on the basis of a protected characteristic, rather than differential outcomes.

The existence/uniqueness theorem (Theorem 3.2) is particularly useful for attestation. It lets you write: "We have applied the minimum-distortion correction; Q* is provably the nearest fair measure to our actual pricing measure P. There is no less-distorting correction that also achieves sensitivity-zero." That is a meaningful statement in a supervisory submission. It shows you are not compressing legitimate risk differentials gratuitously — you are applying the minimum necessary adjustment.

**The actuarial balance constraint** is also directly relevant to Solvency II: the correction preserves `E^Q_x[Y] = E^P_x[Y]`, so total fair premiums equal total expected claims under Q*. You are not underpricing the portfolio to achieve a fairness target.

**On the Equality Act specifically:** post-Test Achats (2012, retained in UK law), gender cannot be a rating factor. Age can be used for motor insurance where risk differentials are empirically demonstrable. Disability can be used where it is a genuine risk factor, not applied as a blanket exclusion. The KL sensitivity approach does not require you to specify a causal graph to determine which variables are "proxies" for which protected characteristics. It computes sensitivity of the pricing function to each D_i directly. This is more defensible under proportionality arguments: you are demonstrating the price does not respond to the protected characteristic, without needing to claim that no correlation exists.

---

## What we already implement

`insurance-fairness` v0.6.3 has two relevant classes.

**`DiscriminationInsensitiveReweighter`** implements the propensity approximation:

```python
from insurance_fairness import DiscriminationInsensitiveReweighter

rw = DiscriminationInsensitiveReweighter(protected_col="gender")
weights = rw.fit_transform(X_train)

model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)
```

The weights are `P(A = a_i) / P(A = a_i | X_i)` — inverse propensity weighting where the propensity score is estimated by logistic regression or random forest. This is the correct solution to the Theorem 3.1 system when h is linear in D and gamma = 1 (expected-value pricing). It acts at training time, preventing the model from learning discriminatory patterns.

The reweighter's diagnostics flag when the propensity model is too confident — i.e., when the group is highly predictable from X, meaning proxy discrimination risk is elevated. Low mean propensity scores by group are your early warning of serious proxy exposure.

**`MarginalFairnessPremium`** implements Huang & Pesenti (arXiv:2505.18895), which derives the same sensitivity-zero criterion as Miao & Pesenti but as a direct closed-form correction to the premium rather than a measure change:

```python
from insurance_fairness import MarginalFairnessPremium

mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
mfp.fit(Y_train, D_train, X_train, model=glm, protected_indices=[0])
rho_fair = mfp.transform(Y_test, D_test, X_test)

report = mfp.sensitivity_report()
# report.sensitivities: how much each protected attribute was moving the premium
# report.corrections: what was removed
```

This operates at scoring time and is model-agnostic. It does not require retraining. For GLMs with a clear model equation, `mfp.sensitivity_report()` shows you the raw sensitivity (how much the premium was moving per unit change in the protected attribute) before and after correction. For GBMs, it uses numerical Jacobians.

The two classes are complementary, not redundant. The reweighter prevents the model from learning discriminatory patterns during training. `MarginalFairnessPremium` corrects residual sensitivity at scoring time, particularly relevant for distortion risk measures where the expected-value correction is insufficient.

---

## The full Theorem 3.1 implementation: why we have not built it yet

The full RN derivative solution requires:

1. **Conditional CDF estimation** `F_{Y|X=x}` per rating cell, to compute the conditional rank `U_{Y|x} = F_{Y|X}(y)`. The infrastructure exists in `insurance-fairness` already — `MarginalFairnessPremium._estimate_cdf()` handles this.

2. **Per-cell root-finding** for the Lagrange multipliers `eta_x*`. This is a numerical root-finding problem per (protected attribute, rating cell) pair. For a UK motor portfolio with 500 distinct segments, this is manageable. For a portfolio with millions of individually rated cells, it is not — you need segment aggregation.

3. **GBM Jacobians**: for gradient boosted models, `∂_i h(X,D)` requires either numerical finite differences or SHAP-based attribution, neither of which is cheap.

This is roughly 400-500 lines of code. The blocker is not the implementation — it is the absence of an empirical comparison showing the full solution performs meaningfully better than the propensity approximation we already have.

The paper's validation is entirely on synthetic data with a linear model and binary gender. We do not know whether the KL correction gives materially better-calibrated premiums than `DiscriminationInsensitiveReweighter` on a real UK motor or home portfolio with GBM predictions. Until that comparison exists, building the full solver is premature engineering.

The trigger for building it would be: (a) an empirical comparison on real insurance data showing material improvement over the propensity approximation, (b) open-source reference code from Pesenti's group to benchmark against, or (c) FCA guidance endorsing the sensitivity-based measure change as a specific compliance pathway.

---

## The multi-attribute case: a word on OT

Some teams have reached for Wasserstein / optimal transport as a fairness correction when handling multiple protected attributes simultaneously. We should be direct: OT achieves demographic parity — the same distribution of premiums across groups regardless of risk. That is not the standard the Equality Act imposes, and it is not what the FCA requires. It will over-correct wherever risk genuinely differs by protected group, which for age in UK motor it demonstrably does.

The KL barycentre approach (Theorem 4.3) is the correct multi-attribute reconciliation under sensitivity-zero fairness. OT is appropriate only as a secondary adjustment within a sensitivity-zero framework — not as the primary fairness correction.

---

## Our recommended workflow

For a UK personal lines pricing team addressing EP25/2 or preparing Consumer Duty model governance evidence:

1. **Multicalibration audit** to identify which groups and quantile bands are over- or under-priced relative to actual experience. `insurance-fairness` `MulticalibrationAudit` does this.

2. **`LindholmCorrector`** (Lindholm marginalisation) where you have a clear causal story: this postcode variable is proxying ethnicity via the mechanism ONS Census 2021 -> LSOA -> ethnicity distribution. Document the causal path.

3. **`DiscriminationInsensitiveReweighter`** where the causal story is unclear or contested. This is the propensity approximation to Theorem 3.1 — grounded in Miao & Pesenti even if it is not the full solution.

4. **`MarginalFairnessPremium`** if you use distortion risk measures (ES-based pricing, capital allocation, or reserving contexts). The expected-value correction from step 3 is not sufficient for ES.

5. **Post-correction multicalibration** to verify the correction has not introduced new disparities in subgroups you were not targeting.

6. **Document the correction as minimal.** The Theorem 3.2 uniqueness result lets you attestthe correction is the minimum necessary. That belongs in your EP25/2 evidence file.

```python
from insurance_fairness import (
    DiscriminationInsensitiveReweighter,
    MarginalFairnessPremium,
    MulticalibrationAudit,
)

# Step 1: audit
audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
report = audit.audit(y_train, y_pred_train, protected_train, exposure=exposure_train)
print(f"Is multicalibrated: {report.is_multicalibrated}")
print(f"Overall calibration p-value: {report.overall_calibration_pvalue:.4f}")
for g, pval in report.group_calibration.items():
    print(f"  {g}: p={pval:.4f}")  # identifies which groups x quantile bands need correction

# Step 2/3: training-time reweighting
rw = DiscriminationInsensitiveReweighter(protected_col="gender")
weights = rw.fit_transform(X_train)
# inspect diagnostics before training
diag = rw.diagnostics(X_train)
print(diag.mean_propensity_by_group)

# Fit model with reweighted data
model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)

# Step 4: scoring-time correction for ES
mfp = MarginalFairnessPremium(distortion='es_alpha', alpha=0.75)
mfp.fit(Y_train, D_train, X_train, model=model, protected_indices=[0])
rho_fair = mfp.transform(Y_test, D_test, X_test)

# Step 5: post-correction audit
post_audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
post_report = post_audit.audit(y_test, rho_fair, protected_test, exposure=exposure_test)
print(f"Is multicalibrated: {post_report.is_multicalibrated}")
print(f"Overall calibration p-value: {post_report.overall_calibration_pvalue:.4f}")
for g, pval in post_report.group_calibration.items():
    print(f"  {g}: p={pval:.4f}")
```

---

## What the paper gets you that practice does not

Practitioners have been doing propensity reweighting for fairness for several years. What Miao & Pesenti (2026) adds is not a new technique but a rigorous proof that the propensity approach is optimal in the KL sense under the expected-value, linear-model special case — and that the natural extension to distortion risk measures and nonlinear models has a unique well-defined solution.

That theoretical grounding matters for two reasons. First, it closes the gap between what practitioners are doing and what they can claim to be doing. The reweighter is not just "we adjusted the training data somehow." It is the empirical approximation to the minimum-distortion fair pricing measure. Second, the existence/uniqueness result — which has no equivalent in the propensity literature — gives a specific statement for supervisory attestation that was not available before.

The paper is a preprint as of March 2026 and has not been peer-reviewed. Its numerical examples are entirely synthetic. Treat it as the theoretical foundation, not the empirical benchmark.

---

*Miao, K. E. & Pesenti, S. M. (2026). Discrimination-Insensitive Pricing. arXiv:2603.16720.*

*`insurance-fairness` is at [pypi.org/project/insurance-fairness](https://pypi.org/project/insurance-fairness/). `uv add insurance-fairness` to get started.*
