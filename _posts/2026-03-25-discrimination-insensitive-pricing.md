---
layout: post
title: "Discrimination-Insensitive Pricing: Beyond Removing the Protected Variable"
date: 2026-03-25
published: false
categories: [fairness, techniques, tutorials]
tags: [proxy-discrimination, fairness, reweighting, kl-divergence, propensity-scores, insurance-fairness, fca, consumer-duty, equality-act, uk-motor, tutorial]
description: "insurance-fairness v0.6.3 ships DiscriminationInsensitiveReweighter. Here's why dropping the protected column doesn't work, how propensity-based reweighting does, and what the API looks like."
---

Every pricing team we speak to has done the same thing. They train a model, someone raises a fairness concern, and the fix is to drop gender (or ethnicity, or disability status) from the feature matrix. The model now technically doesn't know about the protected attribute. Job done.

It isn't done. It hasn't even started.

This post covers why dropping the column fails, what propensity-based reweighting does instead, and how to use `DiscriminationInsensitiveReweighter` in [`insurance-fairness`](https://github.com/burning-cost/insurance-fairness) v0.6.3 — which implements the approach from Miao & Pesenti (2026, arXiv:2603.16720).

```bash
uv add insurance-fairness
```

---

## Why dropping the column doesn't work

Suppose you're pricing UK motor and you remove gender from your feature set. Your remaining features include: postcode, occupation, vehicle group, annual mileage, and age. Every one of these is correlated with gender to some degree. Occupation is highly so — the ONS Labour Force Survey shows significant gender stratification across vehicle-heavy occupations like driving and delivery. Vehicle group correlates through both occupational income effects and direct purchasing patterns. Postcode correlates with local demographic composition.

Your model, trained on these remaining features, will reconstruct the removed column implicitly. It doesn't need the label "gender" to price as if it had gender. The proxy discrimination problem is that removing the protected attribute from the feature matrix does not remove the protected attribute's influence from the model's predictions.

This isn't a theoretical concern. [Three methods for detecting proxy discrimination in your pricing model](/2026/03/22/three-methods-proxy-discrimination-detection-compared/) benchmarks the detection approaches available in the insurance-fairness library — how much of each factor's premium effect is mediated through a protected attribute, versus reflecting genuine risk difference. In practice, postcode and occupation routinely show substantial mediated effects once you look for them.

The Lindholm–Richman–Tsanakas–Wüthrich framework (which underlies most of the `insurance-fairness` library) formalises this distinction as the difference between **proxy discrimination** — pricing signals that operate through the protected attribute — and **demographic disparities** — pricing differences that reflect legitimate risk heterogeneity between groups. These are not the same thing, and treating one as the other produces compliance failures in either direction.

---

## What reweighting does differently

The Miao & Pesenti approach doesn't touch the features at all. Instead, it reweights the training data so that, under the reweighted distribution, the features X are **statistically independent** of the protected attribute A.

The intuition: if gender explains none of the variation in X under the training distribution, then no model trained on X can use gender (directly or indirectly) to improve its predictions. You haven't removed any information from the features — you've changed which data points the model learns from.

The mathematical problem is: find the probability measure Q, closest to the empirical measure P in KL divergence, such that X ⊥ A under Q. "Closest in KL divergence" is the constraint that prevents the reweighted distribution from being arbitrarily different from the original data.

The solution is tractable and closed-form. The optimal sample weights are:

```
w_i = P(A = a_i) / P(A = a_i | X_i)
```

The numerator `P(A = a_i)` is just the marginal proportion of group `a_i` in the training data — straightforward to compute. The denominator `P(A = a_i | X_i)` is the **propensity score**: the probability that observation `i` belongs to its group given its features. This is estimated from data, typically by a logistic regression or random forest on the feature matrix.

The ratio behaves as follows: observations that are "predictable" from their features — whose features strongly indicate their group membership, i.e., pure proxy observations — get weight below 1. The features already encode their group, so the model is nudged away from over-relying on those feature combinations. Observations that are "surprising" — whose features don't strongly predict their group — get weight close to 1. The numerator and denominator are approximately equal, so those observations are neither up- nor downweighted.

After reweighting, any model trained with `sample_weight=weights` cannot achieve a loss reduction by exploiting A, either directly or via proxy features. The mathematical guarantee is X ⊥ A under the reweighted measure.

One subtlety worth flagging: the propensity score itself is a diagnostic. If your logistic regression achieves 85% accuracy predicting gender from your feature matrix, that is telling you your features are strong gender proxies. That's information. The `diagnostics()` method in the reweighter surfaces exactly this.

---

## The API

`DiscriminationInsensitiveReweighter` takes the training DataFrame — including the protected column — and returns sample weights. Everything else in your training pipeline stays the same.

```python
from insurance_fairness import DiscriminationInsensitiveReweighter

rw = DiscriminationInsensitiveReweighter(
    protected_col="gender",   # column name for protected attribute
    method="logistic",        # 'logistic' (default) or 'forest'
    clip_quantile=0.99,       # clip top 1% of weights (recommended for production)
    random_state=42,
)
weights = rw.fit_transform(X_train)
```

The `method` parameter controls propensity estimation. `'logistic'` is faster, interpretable, and appropriate when the relationship between features and the protected attribute is roughly linear. `'forest'` (random forest) captures non-linear proxy relationships and is worth trying if you have categorical features with complex interactions — occupation × region type combinations, for instance.

`clip_quantile` is important in practice. Extreme weights arise when a few observations are very "surprising" to the propensity model — they sit in parts of feature space almost exclusively occupied by the other group. Without clipping, these observations can dominate the reweighted training distribution. `clip_quantile=0.99` truncates the top 1%, which is a reasonable default. The effect on fairness is small; the effect on numerical stability can be significant.

Weights are normalised to sum to n by default, so they integrate with sklearn's `sample_weight` parameter without changing the effective learning rate.

---

## Worked example: UK motor book

Here's the full workflow. We build a synthetic motor pricing dataset with occupation as a gender proxy, train a Poisson GLM without reweighting (the naive approach), then train with reweighting, and compare the demographic parity gap.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from insurance_fairness import DiscriminationInsensitiveReweighter
from insurance_fairness.bias_metrics import demographic_parity_ratio

rng = np.random.default_rng(42)
n = 10_000

# Synthetic UK motor features
gender = rng.integers(0, 2, size=n)               # 0=female, 1=male
age = 20 + rng.integers(0, 45, size=n)
# Occupation: correlated with gender (proxy discrimination scenario)
occupation_risk = 0.6 * gender + rng.standard_normal(n)
vehicle_value = 8_000 + 500 * rng.standard_normal(n) + 1_000 * gender
annual_mileage = 8_000 + 2_000 * rng.standard_normal(n)

# True claim frequency: depends on age and mileage, NOT on gender
log_mu = (
    -3.5
    + 0.01 * (age - 35) ** 2 / 100
    + 0.0001 * annual_mileage
    + 0.3 * rng.standard_normal(n)  # noise
)
claims = rng.poisson(np.exp(log_mu))

df = pd.DataFrame({
    "gender": gender,
    "age": age,
    "occupation_risk": occupation_risk,
    "vehicle_value": vehicle_value,
    "annual_mileage": annual_mileage,
})
y = claims

# ── Step 1: fit the reweighter ──────────────────────────────────────────────
rw = DiscriminationInsensitiveReweighter(
    protected_col="gender",
    method="logistic",
    clip_quantile=0.99,
    random_state=42,
)
weights = rw.fit_transform(df)

# ── Step 2: inspect diagnostics ─────────────────────────────────────────────
diag = rw.diagnostics(df)
print(f"Propensity model accuracy: {diag.propensity_model_score:.3f}")
# High accuracy (e.g. 0.73) means features are strong gender proxies.
# That's the problem we're solving.

print(f"Effective sample size: {diag.weight_stats['effective_n']:.0f} / {n}")
# Substantial reduction from n confirms some observations carry high weight.

print(f"Weight range: [{diag.weight_stats['min']:.3f}, {diag.weight_stats['max']:.3f}]")

# ── Step 3: train without reweighting (naive baseline) ──────────────────────
X_model = df.drop(columns=["gender"])
glm_base = PoissonRegressor(max_iter=300)
glm_base.fit(X_model, y)
pred_base = glm_base.predict(X_model)

# ── Step 4: train with reweighting ──────────────────────────────────────────
glm_fair = PoissonRegressor(max_iter=300)
glm_fair.fit(X_model, y, sample_weight=weights)
pred_fair = glm_fair.predict(X_model)

# ── Step 5: compare demographic parity ──────────────────────────────────────
# demographic_parity_ratio: mean(pred | group=1) / mean(pred | group=0)
# A value far from 1.0 indicates pricing differences not explained by risk.
dpr_base = demographic_parity_ratio(pred_base, df["gender"])
dpr_fair = demographic_parity_ratio(pred_fair, df["gender"])

print(f"Demographic parity ratio — baseline: {dpr_base:.4f}")
print(f"Demographic parity ratio — reweighted: {dpr_fair:.4f}")
# Reweighted ratio should be substantially closer to 1.0.
```

In this synthetic example with moderate proxy strength (occupation_risk coefficient 0.6), the baseline model's demographic parity ratio sits around 1.08–1.12 — pricing males 8–12% higher on average despite gender having no causal role in the true claim frequency. After reweighting, the ratio typically falls to 1.01–1.03. The model has stopped exploiting the proxy.

Note that we never remove `occupation_risk` from the model. It remains a feature — and if it genuinely predicts claims independently of the proxy path, that legitimate signal is preserved. That's the key advantage over simply dropping correlated variables: you don't lose information that has legitimate actuarial content.

---

## Reading the diagnostics

The `diagnostics()` method returns a `ReweighterDiagnostics` dataclass with five fields that matter:

**`propensity_model_score`** — accuracy of the propensity model on training data. Chance level for binary protected attribute is 0.5. A score of 0.75 means your features explain 50% of the above-chance variation in group membership. That's a substantial proxy problem.

**`mean_propensity_by_group`** — the average P(A=a | X) within each group. For a well-balanced binary attribute with marginal proportion 0.5, a group mean propensity of 0.75 means the model can identify group members with 75% confidence from features alone — strong proxy signal. The closer this is to the marginal proportion, the weaker the proxy structure.

**`weight_stats['effective_n']`** — the effective sample size after reweighting, computed as (Σwᵢ)² / Σwᵢ². If effective_n is 60% of n, roughly 40% of your data's statistical power has been redistributed to handle the proxy problem. Severe reduction (below 50%) is a signal to either clip more aggressively or reconsider which features remain in scope.

**`weight_stats['max']`** — maximum weight assigned to any observation. Single observations with weight > 10 deserve inspection. They are observations in feature-space regions where the propensity model assigns very low probability to the observed group — surprising observations that the reweighter must up-correct.

---

## FCA context

The regulatory backdrop has two relevant layers.

**FCA Consumer Duty (PRIN 2A, live July 2023)** requires firms to demonstrate they are not causing foreseeable harm and that their products deliver fair value. Foreseeable harm includes systematic mispricing by protected characteristic where that mispricing is driven by proxy discrimination rather than genuine risk. The duty is outcomes-based, which means "we didn't put gender in the model" is not an adequate compliance argument if the outputs are effectively gender-discriminatory.

**Equality Act 2010, Section 19** defines indirect discrimination: applying a provision, criterion, or practice that puts persons sharing a protected characteristic at a particular disadvantage, where that cannot be justified as a proportionate means of achieving a legitimate aim. Actuarial legitimacy — demonstrably risk-based pricing — is the standard justification for differential premiums. If your occupation factor is substantially driven by its gender correlation rather than genuine occupational risk, that justification weakens.

What `DiscriminationInsensitiveReweighter` provides is a documented, reproducible method for reducing proxy discrimination at training time. When an FCA supervisor asks how you ensure your model doesn't discriminate via proxies, "we reweight training data to enforce X ⊥ A, here is the methodology (Miao & Pesenti 2026) and here are the diagnostics" is a substantially better answer than "we dropped the column."

---

## Relation to other approaches in insurance-fairness

The library now has several fairness interventions at different points in the modelling pipeline:

- **Training-time** (this post): `DiscriminationInsensitiveReweighter` — change the distribution the model learns from.
- **Post-processing**: `optimal_transport.DiscriminationFreePrice` — adjust predictions after training using Lindholm marginalisation or causal path decomposition.
- **Constraint-based**: `MarginalFairnessPremium` (v0.5.0) — closed-form fairness correction for distortion risk measures including Expected Shortfall.
- **Audit only**: `FairnessAudit`, `diagnostics.ProxyDiscriminationAudit`, `DoubleFairnessAudit` — measure and report, without intervention.

We think training-time reweighting is the right default for new model builds. The guarantee is clean (X ⊥ A under the training measure) and the intervention is transparent. Post-processing corrections are useful when you cannot retrain — legacy systems, regulatory freeze periods, staged deployments — but they are harder to explain and audit.

The combination we'd recommend: use `DiscriminationInsensitiveReweighter` during model development, and use `diagnostics.ProxyDiscriminationAudit` to verify the reduction post-training. If the audit still shows residual proxy effects above your threshold, consider whether additional proxy-correlated features should be removed from scope entirely — not as the primary fairness intervention, but as a belt-and-braces check.

---

## Limitations

Reweighting achieves statistical independence X ⊥ A under the training measure. This is a marginal independence condition — it does not guarantee conditional independence (X ⊥ A | Z for other covariates Z), and it does not eliminate all causal paths from A to predictions. If A has a direct causal effect on Y (which is the case for some protected attributes in some insurance lines), reweighting will over-correct: it will suppress genuine risk signal alongside the discriminatory signal.

For pure proxy discrimination — where A affects Y only through X — the approach is exact. For mixed cases, the choice of approach depends on your causal graph. The `optimal_transport.CausalGraph` module in insurance-fairness implements the Lindholm causal decomposition if you need finer-grained control over which paths to block.

The propensity model is estimated from training data. If the propensity model is misspecified — if the true relationship between X and A is non-linear and you're using logistic regression — the weights will be approximate. `method='forest'` is more flexible here at the cost of interpretability. In either case, `diagnostics()` tells you whether the propensity model is well-calibrated, which is a reasonable proxy for whether the weights are reliable.

---

## Further reading

- [insurance-fairness on GitHub](https://github.com/burning-cost/insurance-fairness)
- [Your Pricing Model Might Be Discriminating](/2026/03/03/your-pricing-model-might-be-discriminating/) — introduction to proxy discrimination in insurance pricing
- [Discrimination-Free Insurance Pricing via Optimal Transport](/2026/03/10/insurance-fairness-ot/) — post-training correction using causal path decomposition
- [Three Ways to Detect Proxy Discrimination in Your Pricing Model](/2026/03/22/three-methods-proxy-discrimination-detection-compared/) — comparing detection approaches
- [Fairness Auditing When You Don't Have Sensitive Attributes](/2026/03/20/fairness-auditing-without-sensitive-attributes/) — when you cannot hold the protected attribute at all
- Miao, K. E. & Pesenti, S. M. (2026). Discrimination-Insensitive Pricing. arXiv:2603.16720.
- Lindholm, Richman, Tsanakas & Wüthrich (2022). Discrimination-Free Insurance Pricing. ASTIN Bulletin 52(1), 55–89.
