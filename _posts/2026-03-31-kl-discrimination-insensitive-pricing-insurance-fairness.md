---
layout: post
title: "KL Discrimination-Insensitive Pricing: The Information-Theoretic Foundation"
date: 2026-03-31
categories: [fairness]
tags: [discrimination-insensitive, KL-divergence, measure-change, Gateaux-sensitivity, Miao-Pesenti, arXiv-2603-16720, Radon-Nikodym, exponential-tilting, actuarial-balance, insurance-fairness, discrimination-free, Lindholm, Consumer-Duty, FCA, EP25-2, python, distortion-risk-measure]
description: "Miao & Pesenti (arXiv:2603.16720) derive the discrimination-insensitive pricing measure from first principles: find the nearest probability measure Q to the real-world P, in KL divergence, such that the pricing principle has zero Gâteaux sensitivity to every protected covariate. We explain the mathematics, compare it to Lindholm marginalisation, and show what we have and have not implemented in insurance-fairness."
math: true
---

We already had a working implementation before the theoretical foundation arrived. `DiscriminationInsensitiveReweighter` in `insurance-fairness` v0.6.3 has been in production workflows since before Miao and Pesenti's paper appeared on arXiv in March 2026. The paper gives us the rigorous grounding for what the reweighter is actually doing, shows precisely where the approximation is exact and where it is not, and establishes an existence and uniqueness result that has genuine value for regulatory attestation.

The paper is Miao, K.E. and Pesenti, S.M., *Discrimination-Insensitive Pricing*, arXiv:2603.16720, submitted 17 March 2026, revised 27 March 2026. Pesenti is also the co-author of the Huang & Pesenti (2025) marginal fairness paper that underpins `MarginalFairnessPremium`. The two papers are complementary approaches to the same sensitivity-zero criterion, and we explain the relationship below.

---

## The problem with unawareness

The standard response to proxy discrimination is unawareness: remove the protected attribute from the feature matrix. This does not work. A model trained on postcode, vehicle group, and occupation will reconstruct gender or ethnicity from the correlations embedded in the training data. The unawareness price

$$\mu(x) = \mathbb{E}[Y \mid X = x]$$

is a weighted average of group-specific best estimates

$$\mu(x) = \sum_{d \in \mathcal{D}} \mu(x, d) \cdot \mathbb{P}(D = d \mid X = x)$$

with weights that vary by covariate profile x. Those x-varying weights are the mechanism of proxy discrimination. The model has never seen D; the weights are implicit in the joint distribution of X and D in the training data.

Miao and Pesenti frame the correction differently. Rather than modifying the prediction function, they modify the probability measure under which the premium is computed. Fix the pricing functional. Change the probability measure. The result is a premium that satisfies a precise notion of discrimination-insensitivity at every rating cell.

---

## What discrimination-insensitive means precisely

Let:
- **X** be the vector of legitimate rating factors (vehicle age, NCD, occupation, etc.)
- **D** = (D₁, ..., D_m) be the vector of protected covariates (gender, ethnicity, disability)
- **Y** be the actual loss random variable
- **h(X, D)** be the pricing aggregation function — your GLM or GBM prediction
- **ρ_γ(Y\|x)** be a distortion risk measure: E[Y · γ(U_{Y\|x})] where U_{Y\|x} is the conditional uniform random variable comonotonic to Y\|x and γ is the distortion weight function

Distortion risk measures subsume expected value (γ = 1), Value-at-Risk, and Expected Shortfall as special cases.

**Definition 2.2 of the paper:** The sensitivity of the pricing principle to protected covariate D_i at rating cell x is the Gâteaux derivative

$$\partial_{D_i} \rho(Y \mid x) = \lim_{\varepsilon \to 0} \frac{\rho(Y \mid x, D_i(1+\varepsilon)) - \rho(Y \mid x)}{\varepsilon}$$

where D_{i,ε} = D_i(1 + ε) is a proportional perturbation. Note this is not an additive shift: it asks what happens to the premium if D_i scales up slightly.

**Lemma 2.4** gives the closed form:

$$\partial_{D_i} \rho(Y \mid x) = \mathbb{E}_x\!\left[\Phi_i(X, D, U_{Y\mid x})\right]$$

with the sensitivity weight

$$\Phi_i(x, d, u) = d_i \cdot \partial_i h(x, d) \cdot \gamma(u)$$

Three factors multiply: the value of the protected attribute, the pricing model's partial derivative with respect to D_i, and the distortion weight at that quantile of the conditional loss distribution. Sensitivity is zero at a rating cell only when the model does not respond to marginal changes in D_i there. This is a local, per-cell notion — not a portfolio-level aggregate.

**Definition 2.7 of the paper:** The discrimination-insensitive measure is the solution to

$$Q^* = \arg\min_{Q \ll P} D_{\mathrm{KL}}(Q \| P)$$

subject to:

$$\partial_{D_i} \rho^Q(Y \mid x) = 0 \quad \text{for all } i \in M \qquad \text{(sensitivity constraints)}$$

$$\mathbb{E}_x^Q[Y] = \mathbb{E}_x^P[Y] \qquad \text{(actuarial balance)}$$

The second constraint is critical for insurance. Without it, the trivial solution is to collapse the measure onto scenarios where D has no influence — which is actuarially meaningless and would destroy the premium-to-risk relationship. The actuarial balance constraint forces Q* to preserve expected claims at every rating cell. Total fair premiums equal total expected claims under Q* without any separate correction step.

---

## The solution: exponential tilting

**Theorem 3.1:** The unique solution Q* has Radon-Nikodym derivative

$$\frac{dQ_x^*}{dP} = \frac{1}{C_x} \exp\!\left(-{\eta_x^*}^\top \Phi(X, D, U_{Y\mid x}) - \eta_{m+1}^* \cdot Y\right)$$

where:
- η_x* is the vector of Lagrange multipliers for the sensitivity constraints
- η_{m+1}* is the Lagrange multiplier for the actuarial balance constraint
- C_x is the normalising constant ensuring Q* is a valid probability measure

This is exponential tilting of P by two components simultaneously. The measure Q* downweights observations where (a) sensitivity Φ_i is large — regions where the model responds strongly to the protected attribute — and (b) loss Y is above its conditional mean, which re-centres the measure to preserve actuarial balance. These two tilts are entangled: the Lagrange multipliers that govern them are solved jointly.

The Lagrange multipliers are found by solving a system of cumulant-generating-function equations at each rating cell x. **Theorem 3.2** establishes that this system has a unique solution if and only if the CGF of the joint vector (Φ, Y − E[Y]\|x) exists in a neighbourhood of −η*. For UK personal lines with Poisson frequency and gamma or log-normal severity, this condition is satisfied.

The computation requires a numerical root-finder per protected attribute per rating cell. At UK motor portfolio scale — 10–20 rating factors, potentially millions of distinct cell combinations — this is expensive without aggregation into a manageable segment structure first.

---

## Multi-attribute case: the KL barycentre

When there are multiple protected attributes (gender and ethnicity, say), the joint sensitivity-zero constraints may be contradictory. You cannot necessarily find a single Q that has zero sensitivity to both D₁ and D₂ simultaneously. The paper handles this in two steps.

**Step 1 (Lemma 4.1):** For each D_i separately, solve the single-attribute problem:

$$\frac{dQ_i}{dP} = \frac{\exp(-\eta_i^* \cdot \Phi_i)}{\mathbb{E}_x\!\left[\exp(-\eta_i^* \cdot \Phi_i) \mid U_{Y\mid x}\right]}$$

**Step 2 (Theorem 4.3):** Reconcile the per-attribute measures by minimising a weighted sum of KL distances from each Q_i, subject to actuarial balance:

$$\bar{Q} = \arg\min_{Q} \sum_i \pi_i \cdot D_{\mathrm{KL}}(Q \| Q_i) \quad \text{subject to} \quad \mathbb{E}_x^Q[Y] = \mathbb{E}_x^P[Y]$$

The solution is the geometric average of the per-attribute RN derivatives with a budget adjustment:

$$\frac{d\bar{Q}}{dP} \propto \exp(-\lambda_x \cdot Y) \cdot \prod_i \left(\frac{dQ_i}{dP}\right)^{\pi_i}$$

The weights π_i are user-specified. They represent a governance decision about the relative priority of each protected attribute's fairness constraint, and the paper provides no data-driven way to choose them. Equal weighting (π_i = 1/m) is a reasonable default but has no theoretical justification as the uniquely correct choice.

Theorem 4.2 is a standalone contribution: the unconstrained KL barycentre of {Q_i} with weights {π_i} is the geometric average of the individual RN derivatives. This holds independently of the insurance application and is a general result about KL barycentres.

---

## How this compares to Lindholm marginalisation

The discrimination-free pricing approach from Lindholm, Richman, Tsanakas and Wüthrich (2022) operates in prediction space. It fixes the probability measure P and modifies the pricing functional by marginalising over the protected attribute:

$$\pi_{\text{DF}}(x) = \sum_s \mathbb{P}^*(S = s) \cdot h(x, s)$$

where P*(S = s) is a reference distribution over the protected attribute. The price changes; the probability measure does not.

Miao and Pesenti operate in measure space. The pricing functional stays the same; the underlying probability distribution changes from P to Q*. These look mathematically equivalent from a distance but have materially different properties in practice.

**Causal structure.** Lindholm implicitly requires knowing which variables are protected (S) and which are legitimate (X). In practice, decomposing indirect discrimination from legitimate risk transmission requires a causal DAG. A wrong DAG — misclassifying a justified mediator as a proxy — produces the wrong correction. Miao and Pesenti requires no causal graph. Sensitivity Φ_i is computed directly from the pricing function h. UK insurers who cannot credibly specify the causal path from occupation and postcode to ethnicity — which is most of them — have a defensibility problem with Lindholm that the KL approach does not share. The limitation is symmetric: without a causal graph, the KL method cannot distinguish D acting as a legitimate risk predictor from D acting as an indirect discriminator, and will try to zero out both.

**Actuarial balance.** Lindholm marginalisation introduces portfolio bias: the sum of fair premiums no longer equals the sum of expected claims. A separate correction step — proportional uplift or a KL-optimal correction — is required. The Miao-Pesenti formulation builds the actuarial balance constraint directly into the optimisation. Q* preserves E^Q_x[Y] = E^P_x[Y] by construction.

**Data requirements for protected attributes.** Lindholm requires P(S\|X) for each rating cell. For UK ethnicity (not collectible as a rating factor), this must be estimated from ONS Census 2021 postcode-to-LSOA ethnicity proportions — a significant practical overhead with real estimation error. The KL approach requires D to be observed in the training data to compute Φ_i. For historical data where gender was collected but can no longer be used post-*Test Achats* (2012), the KL approach is directly applicable.

**The relationship to Huang & Pesenti 2025.** Huang and Pesenti (arXiv:2505.18895), which underlies `MarginalFairnessPremium`, derives a closed-form correction to the pricing function that achieves zero Gâteaux sensitivity of the distortion risk measure. Miao and Pesenti derives the same sensitivity-zero criterion but as a measure change rather than a direct functional correction. For linear h and expected-value pricing (γ = 1), the two are equivalent. For nonlinear h or ES-type distortions, the KL measure change is the more principled formulation at higher computational cost. The 2026 paper likely represents the measure-theoretic foundation that the 2025 paper's closed-form approximates.

---

## What we have implemented

`DiscriminationInsensitiveReweighter` in `insurance-fairness` v0.6.3 implements the propensity-reweighting approximation:

$$\frac{dQ}{dP} \propto \frac{P(A = a_i)}{P(A = a_i \mid X_i)}$$

where P(A\|X) is estimated by logistic regression or random forest. This is the marginal independence special case of Theorem 3.1 — it is exact when h is linear in D and γ = 1 (expected-value pricing). For standard frequency-severity GLMs, this is where most UK motor and home portfolios sit. For GBMs with nonlinear feature interactions or ES-based distortions, it is an approximation whose error is second-order in the nonlinearity of D's contribution to h.

```python
import pandas as pd
from insurance_fairness import DiscriminationInsensitiveReweighter

# X_train includes "gender" as a column alongside legitimate rating factors
rw = DiscriminationInsensitiveReweighter(
    protected_col="gender",
    method="logistic",       # use "forest" for nonlinear proxy relationships
    clip_quantile=0.99,      # clip top 1% of weights to prevent dominance
)
weights = rw.fit_transform(X_train)

# weights integrates directly with any sklearn sample_weight parameter
model.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)
```

The diagnostics are important. A propensity model with high accuracy means your features are strong proxies for the protected attribute — the reweighting will be substantial and the effective sample size will fall significantly.

```python
diag = rw.diagnostics(X_train)

print(f"Propensity model accuracy: {diag.propensity_model_score:.3f}")
print(f"Effective sample size: {diag.weight_stats['effective_n']:.0f} / {len(X_train)}")
print(f"Max weight: {diag.weight_stats['max']:.2f}")
print(f"Mean propensity by group: {dict(zip(diag.group_labels, diag.mean_propensity_by_group))}")
```

A propensity accuracy near the chance level (0.5 for binary gender) means your rating factors are not strong proxies and the reweighting barely moves. An accuracy of 0.85 means the portfolio is segmented along gender lines and the correction is material. The effective sample size tells you how much weight you are effectively losing: if 100,000 training policies shrink to an effective N of 60,000, the reweighting is doing something large and you should interrogate which cells are driving it.

The full Theorem 3.1 implementation — root-finding for η* per rating cell, incorporating the conditional loss distribution and distortion weight γ — is not yet implemented. The reason is honest: the paper contains only a single numerical example, on synthetic data with a linear model. We do not yet have evidence that the additional complexity of the full formulation materially improves fairness outcomes on real UK portfolios over the propensity approximation. When either the paper's authors publish an open-source reference implementation or empirical validation on real insurance data appears, we will revisit. That implementation would sit alongside the reweighter as a scoring-time correction rather than a training-time one — the two approaches are complementary.

---

## The regulatory angle

The paper's theoretical contribution has practical value for FCA attestation under Consumer Duty (PS22/9, PRIN 2A) and EP25/2.

**The existence and uniqueness result (Theorem 3.2)** establishes that Q* is the *nearest* fair measure to P under KL divergence. In a supervisory submission, this translates cleanly: we have applied the minimum necessary correction; we are not over-correcting or artificially compressing risk differentials. The correction is uniquely determined by the data and the fairness criterion — there is no degree of freedom that would allow us to have applied a lighter touch while still satisfying sensitivity-zero.

**The actuarial balance property** means the correction does not undermine the solvency position. Total fair premiums equal total expected claims under Q*, satisfying Solvency II technical provision requirements. This is not a trivially satisfied property: Lindholm marginalisation violates it by construction, requiring explicit re-normalisation.

**The "nearest fair measure" framing** matters for the Equality Act 2010 section 19 proportionality test. An insurer asserting that its pricing is not indirectly discriminatory needs to demonstrate not just that it has made some correction, but that the correction is the least restrictive means available. The KL barycentre result provides a formal proof that Q* achieves sensitivity-zero with the minimum divergence from the actuarially correct measure P. You cannot do less and still satisfy the criterion.

EP25/2 (FCA, July 2025) specifically asks for evidence that pricing models do not disadvantage protected groups disproportionately. The sensitivity-zero criterion — that the premium does not *respond to* marginal changes in the protected attribute at any rating cell — is more defensible than a portfolio-level group mean comparison. It is a statement about the pricing mechanism, not just about outcomes.

---

## Where OT / Wasserstein fits and where it does not

We should be direct about this because it causes confusion in practice.

Wasserstein barycentre methods (EquiPy, Charpentier et al. 2023) achieve demographic parity — the distribution of premiums is identical across groups unconditionally. This is the wrong criterion for UK insurance. The Equality Act 2010 permits price differences where they reflect genuine risk differences. Demographic parity erases legitimate risk differentials along with discriminatory ones.

The KL approach achieves sensitivity-zero of the distortion risk measure to protected attribute perturbations. This is closer to conditional fairness — equal price for equal risk profile — than to demographic parity. Sensitivity-zero and demographic parity are different conditions. For binary protected attributes with linear pricing models they come apart immediately.

Wasserstein is appropriate as a secondary adjustment tool — for instance, to reconcile the per-attribute corrections in the KL barycentre step, or as a post-processing check that premium distributions look reasonable across groups. It should not be the primary fairness criterion for UK motor or home insurance. Using it as such will systematically under-price high-risk groups and over-price low-risk groups within each protected attribute category, violating Solvency II and potentially Consumer Duty simultaneously.

---

## Honest limitations

**Rating cell sparsity at UK portfolio scale.** The full Theorem 3.1 root-finding must be performed per rating cell x in support(X). For a UK motor portfolio with 15 rating factors, the cell space is vast. In practice, this requires either aggregating to a manageable number of segments (predicted premium band, say) or treating the sensitivity function as approximately constant within cells. The paper does not address this.

**Continuous protected attributes.** The paper handles continuous D_i — age as a continuous variable — but is silent on the granularity at which sensitivity should be zeroed. Should η_x* be estimated pooling over age bands or at individual level? The answer has material implications for computational feasibility and for how the correction interacts with NCD and vehicle age.

**Choice of distortion function γ.** The framework is general across distortion risk measures, but UK personal lines pricing is predominantly expected-value based. Specifying a non-trivial γ (ES at α = 0.75, say) adds considerable complexity — estimating the conditional quantile function U_{Y\|x} at portfolio scale requires kernel or empirical CDF estimation per cell. The simpler expected-value case (γ = 1) reduces to the propensity reweighting we already implement.

**The weights π_i in the barycentre.** There is no principled basis for choosing how to weight gender versus ethnicity versus disability in the multi-attribute KL barycentre. The paper notes this is a governance decision. Equal weighting is not obviously correct and the paper provides no sensitivity analysis showing how sensitive the correction is to the π_i choice.

**Only synthetic validation.** The paper's numerical example (Example 3.6) uses a linear model with binary gender, log-normal risk factor, and ES₀.₉ distortion, with 1,000,000 Monte Carlo samples. There is no empirical validation on real insurance data. On a real UK motor portfolio with GBM predictions, residual gender correlation from post-*Test Achats* historical data, and ONS-imputed ethnicity, the premium differences between the propensity approximation and the full Theorem 3.1 solution are not yet characterised.

---

## Recommended workflow for UK personal lines

The method is a training-time correction. It belongs early in the pricing cycle.

```python
import pandas as pd
from insurance_fairness import DiscriminationInsensitiveReweighter, MulticalibrationAudit

# Step 1: fit the reweighter on training data
# X_train includes gender in historical data (collected pre-2012) alongside
# current rating factors: age, vehicle_group, ncd, occupation, postcode_district
rw = DiscriminationInsensitiveReweighter(
    protected_col="gender",
    method="forest",        # forest captures nonlinear proxy relationships
    clip_quantile=0.99,
    random_state=42,
)
weights = rw.fit_transform(X_train)

# Inspect before training
diag = rw.diagnostics(X_train)
if diag.propensity_model_score > 0.65:
    print(
        f"WARNING: propensity accuracy {diag.propensity_model_score:.2f} — "
        "your rating factors are strong proxies for gender. "
        f"Effective N drops from {len(X_train)} to {diag.weight_stats['effective_n']:.0f}."
    )

# Step 2: train your model with the reweighting
# Drop gender from features — it cannot be a rating factor post-Test Achats
feature_cols = [c for c in X_train.columns if c != "gender"]
model.fit(X_train[feature_cols], y_train, sample_weight=weights)

# Step 3: post-correction multicalibration audit
# Check that the correction has not introduced new disparities at premium level
# Use age or another observable protected attribute for ongoing monitoring
audit = MulticalibrationAudit(n_bins=10, alpha=0.05)
report = audit.audit(y_true_val, y_pred_val, protected_val, exposure=exposure_val)
print(report.summary())
```

The reweighter does the heavy lifting at training time. The multicalibration audit at step 3 catches anything that slips through — a model can pass the reweighting step and still be miscalibrated within specific age-band or occupation cells. The two corrections address different things: the reweighter prevents the model from learning discriminatory patterns; multicalibration corrects residual within-cell bias after the fact.

For gender specifically — where historical data exists but the attribute cannot be used — this is our recommended approach. The reweighter removes the residual gender signal embedded in the proxy features. The resulting model prices on the risk factors alone, without the gender information that those factors carry.

---

## Reference

Miao, K.E. and Pesenti, S.M. (2026). *Discrimination-Insensitive Pricing*. arXiv:2603.16720. Submitted 17 March 2026, revised 27 March 2026.

Related reading: Huang, F. and Pesenti, S.M. (2025). *Marginal Fairness: Fair Decision-Making under Risk Measures*. arXiv:2505.18895.

Lindholm, M., Richman, R., Tsanakas, A. and Wüthrich, M.V. (2022). Discrimination-Free Insurance Pricing. *ASTIN Bulletin* 52(1), 55–89.

The implementation is in [`insurance-fairness`](/insurance-fairness/) v0.6.3. `DiscriminationInsensitiveReweighter` is the training-time approximation. The full Theorem 3.1 barycentre corrector is WATCH pending empirical validation.
