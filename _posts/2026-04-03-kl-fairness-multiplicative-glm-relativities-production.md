---
layout: post
title: "KL Fairness Corrections and Multiplicative GLMs: The Production Deployment Problem"
date: 2026-04-03
categories: [fairness, techniques]
tags: [fairness, kl-divergence, discrimination-insensitive, Miao-Pesenti, arXiv-2603.16720, GLM, multiplicative-structure, relativities, insurance-fairness, DiscriminationInsensitiveReweighter, production-deployment, UK-personal-lines, actuarial-balance, barycentre-weights]
description: "Miao & Pesenti's KL discrimination-insensitive result is theoretically clean. Deploying it in a production GLM-based pricing system is not. The paper is silent on how to extract multiplicative relativities from a corrected probability measure. We work through what you actually have to do."
author: burning-cost
math: true
---

We have written about the Miao and Pesenti (2026) KL discrimination-insensitive pricing result three times now — the [mathematical foundation](/2026/03/31/kl-discrimination-insensitive-pricing-insurance-fairness/), its [value for FCA Consumer Duty attestation](/2026/04/03/kl-discrimination-insensitive-pricing-fca-consumer-duty/), and [how to write a defensible fairness attestation](/2026/04/03/kl-fairness-attestation-fca-ep25-2-model-validation/) using the existence and uniqueness theorem. We have not written about the thing that will actually consume most of the implementation time when you try to deploy this in a real pricing system: how you extract multiplicative GLM relativities from a KL-corrected probability measure.

The paper does not answer this question. The open question is acknowledged in the research but not resolved. This post is our working-through of it.

---

## Why GLM structure matters

UK non-life pricing is built on multiplicative GLMs. The standard form is:

$$\hat{\mu}(x) = \exp\!\left(\beta_0 + \sum_j \beta_j \phi_j(x)\right)$$

where $\phi_j(x)$ is the $j$-th rating factor — vehicle group, NCD band, postcode district, occupation, driver age band — and each $\exp(\beta_j)$ is a relativity: how much more (or less) a policyholder in that category pays relative to the base.

Relativities are not just model outputs. They are underwriting inputs. A motor pricing system at a UK insurer does not score individual policyholders through a single model call. It loads a table of relativities into a rating engine — sometimes a legacy system from 2004, sometimes ICE, sometimes a bespoke one — and multiplies them together. The model development workflow produces a set of GLM coefficients; those coefficients are reviewed, signed off, loaded, and then they run the book for 12 months.

This is the production reality that the KL correction has to fit into. The correction produces a reweighted probability measure Q*. The question is what GLM you fit under Q* and whether the relativities from that GLM are usable in the rating engine.

---

## What the reweighter actually gives you

`DiscriminationInsensitiveReweighter` produces a vector of sample weights. After calling `rw.fit_transform(X_train)`, each training observation has a weight $w_i = P(A = a_i) / P(A = a_i \mid X_i)$ that reweights the training distribution to approximate the KL-optimal fair measure Q*. The workflow then fits the GLM with those weights:

```python
rw = DiscriminationInsensitiveReweighter(protected_col="gender", method="forest")
weights = rw.fit_transform(X_train)

glm = TweedieRegressor(power=1.0, link='log', max_iter=500)
glm.fit(X_train.drop("gender", axis=1), y_train, sample_weight=weights)
```

The output `glm.coef_` gives you a set of GLM coefficients trained under the reweighted measure. For a log-link GLM, these are exactly $\beta_j$ values, and `exp(beta_j)` are relativities in the standard multiplicative sense. These can be loaded into a rating engine.

So far this is straightforward. The complication is in what those relativities mean relative to the uncorrected model, and whether they remain actuarially justifiable at the factor level.

---

## The relativity shift problem

The KL reweighting changes the implicit weight given to each training observation based on its propensity to belong to the protected group. Observations that are hard to distinguish from the protected group by X (high propensity, low weight) have their influence on the coefficients reduced. Observations that are easy to distinguish (low propensity, high weight) have their influence increased.

This has a specific consequence for the multiplicative structure: **the relativities for factors that are highly correlated with the protected attribute will change, and the relativities for factors that are uncorrelated will not**.

For a UK motor book where gender is the protected attribute, the factors most correlated with gender are typically: vehicle group (women skew toward smaller vehicles), occupation band (certain bands are heavily gendered), and annual mileage (men drive more on average). Post-correction, the relativities for these factors will shift. The relativities for, say, the driver age band, will shift less.

This is the correct behaviour. The reweighting is removing the gender signal encoded in those factors. But it creates an underwriting governance problem: the relativity for "occupation: professional driver" may move from 1.42 to 1.31. The underwriting team needs to understand why it moved, and the answer — "the sample reweighting changed the effective composition of training data for that cell" — is technically correct but not immediately intuitive in a review meeting.

---

## Extracting GLM relativities: the practical steps

Here is how to manage this in practice.

**Step 1: Fit both models and tabulate the change.**

```python
# Uncorrected model
glm_base = TweedieRegressor(power=1.0, link='log', max_iter=500)
glm_base.fit(X_train_noprot, y_train)

# Corrected model
rw = DiscriminationInsensitiveReweighter(protected_col="gender", method="forest")
weights = rw.fit_transform(X_train)
glm_fair = TweedieRegressor(power=1.0, link='log', max_iter=500)
glm_fair.fit(X_train_noprot, y_train, sample_weight=weights)

# Relativity comparison
feature_names = X_train_noprot.columns.tolist()
base_rels = pd.Series(np.exp(glm_base.coef_), index=feature_names, name="base")
fair_rels = pd.Series(np.exp(glm_fair.coef_), index=feature_names, name="fair")
rel_change = fair_rels / base_rels - 1

print(rel_change.sort_values(key=abs, ascending=False).head(20))
```

The output is the proportional change in each relativity. Factors with changes above ±3% need an explanatory note in the underwriting sign-off. Factors with changes above ±10% need a business case.

**Step 2: Check actuarial balance at factor level.**

The KL measure Q* preserves total expected claims: $E^Q[Y] = E^P[Y]$. But it does not preserve expected claims within each factor category. After reweighting, the A/E ratios by factor level may shift because the effective composition of each cell has changed.

```python
# Check A/E by vehicle group before and after
val = X_val_noprot.copy()
val["y_actual"] = y_val
val["y_base"] = glm_base.predict(X_val_noprot)
val["y_fair"] = glm_fair.predict(X_val_noprot)

ae = (
    val
    .groupby("vehicle_group")[["y_actual", "y_base", "y_fair"]]
    .mean()
    .assign(ae_base=lambda df: df.y_actual / df.y_base,
            ae_fair=lambda df: df.y_actual / df.y_fair)
)
print(ae[["ae_base", "ae_fair"]])
```

In almost all cases the corrected model will have better A/E ratios for the vehicle groups most correlated with gender, because the model was previously partially pricing gender through those proxies. The A/E improvement is the evidence that the correction is working, not just theoretical.

**Step 3: Segment-level residual sensitivity check.**

After fitting the corrected GLM, verify that the sensitivity of predictions to the protected attribute is reduced. Both models are trained without the gender column, so any remaining gender signal comes through proxy factors. Check whether predictions still differ systematically by gender group, and whether the gap is smaller in the corrected model:

```python
preds_base = glm_base.predict(X_val_noprot)
preds_fair = glm_fair.predict(X_val_noprot)

check = pd.DataFrame({
    "gender": X_val["gender"],
    "occupation": X_val["occupation"],
    "pred_base": preds_base,
    "pred_fair": preds_fair,
})

# Gender gap in mean predictions, by occupation segment
# Segments where gender proxying is strongest will show the largest pre/post change
gender_gap = (
    check
    .groupby(["occupation", "gender"])[["pred_base", "pred_fair"]]
    .mean()
    .unstack("gender")
    .assign(
        gap_base=lambda df: df[("pred_base", 1)] / df[("pred_base", 0)] - 1,
        gap_fair=lambda df: df[("pred_fair", 1)] / df[("pred_fair", 0)] - 1,
    )[["gap_base", "gap_fair"]]
    .sort_values("gap_base", key=abs, ascending=False)
)
print(gender_gap.head(10))
# Occupations where |gap_base| is large and |gap_fair| is smaller confirm
# the correction is removing the gender proxy signal from those cells.
```

Occupations where the gap does not narrow are either (a) genuine risk differences between genders in that occupation, (b) insufficient effective N for the propensity model to have corrected accurately. Distinguish these by checking whether the pre-correction A/E ratios by gender differ within that occupation.

---

## The multiplicative constraint problem

Here is the deeper issue. The KL-corrected GLM coefficients are optimal under the reweighted distribution. But are they identifiable in the same sense as the base GLM coefficients?

In a multiplicative GLM, identifiability requires that the design matrix is full rank — i.e., that the factors are not perfectly collinear. Under the original training distribution, this is almost always satisfied. Under the reweighted distribution, it might not be: if a factor is nearly perfectly predictive of the protected attribute, the reweighter will nearly zero-weight all training observations that belong to the protected group in that factor level. The effective sample size for those cells shrinks to near zero, and the coefficient becomes poorly identified.

The diagnostic to run:

```python
diag = rw.diagnostics(X_train)

# Check effective sample size by factor level for high-propensity factors
eff_n_by_occ = (
    pd.DataFrame({"occupation": X_train["occupation"], "weight": weights})
    .groupby("occupation")
    .agg(
        n=("weight", "count"),
        eff_n=("weight", lambda w: w.sum()**2 / (w**2).sum())
    )
)
print(eff_n_by_occ.sort_values("eff_n"))
```

Any factor level with effective N below 30 should be flagged. GLM coefficients for those levels are unstable. Options: (a) bin that level with an adjacent category, (b) apply a credibility shrinkage to pull the relativity toward the broader factor group mean, (c) accept the instability and widen the confidence interval in the underwriting sign-off.

This is not a failure of the KL approach — it is the correct signal. If a factor level is nearly perfectly predictive of gender, the model should not have that factor at that granularity in the first place. The reweighting is surfacing a proxy discrimination risk that was hidden in the uncorrected model.

---

## The barycentre weight problem for multi-attribute UK books

When you are correcting for both gender and proxied ethnicity simultaneously, Miao and Pesenti's multi-attribute solution (Theorem 4.3) requires assigning weights $\pi_i$ to each attribute. The weights govern how much the barycentre favours each attribute's individual fairness constraint. The paper provides no guidance on choosing them.

For a UK personal lines book, the natural candidates are:

**Equal weighting** ($\pi_{\text{gender}} = \pi_{\text{ethnicity}} = 0.5$). Treats both constraints with equal importance. Simple, auditable. The problem: it implies that a small residual gender sensitivity matters as much as a large ethnicity sensitivity. If your propensity model shows weak gender proxying (accuracy 0.53) and strong ethnicity proxying (ONS-matched postcode gives accuracy 0.71), equal weighting is suboptimal.

**Sensitivity-proportional weighting**. Set $\pi_i$ proportional to the pre-correction sensitivity magnitude: $\pi_i \propto \|\partial_{D_i} \rho(Y|x)\|$ averaged across the portfolio. This weights the correction effort toward the attribute where the original model's sensitivity is largest. More principled but requires estimating the pre-correction sensitivities, which brings you back to needing the full Phi computation.

**Regulatory-priority weighting**. Under FCA EP25/2, ethnicity (race under the Equality Act) is one of the characteristics for which there is most regulatory concern. An insurer might argue for $\pi_{\text{ethnicity}} > \pi_{\text{gender}}$ on the grounds that ethnicity cannot be collected directly (so the proxy mechanism is harder to detect and correct), and that the FCA's December 2024 machine learning research note specifically highlighted race-related bias.

In `DiscriminationInsensitiveReweighter`, the multi-attribute barycentre is currently approximated as a product of per-attribute reweightings with equal weight. This is the equal-weighting default. The choice of weights is documented in the model governance file, and should be. There is no neutral default.

---

## What is actually missing from the paper

Three things, from a production deployment perspective:

**1. GLM-to-relativities mapping.** The paper proves the corrected premium is $E^{Q^*}[Y \mid x]$. It does not specify how to extract this as a set of multiplicative relativities for a rating engine. The propensity reweighting approach plus GLM refit is one method. Fitting the GLM directly under Q* is another. A post-hoc one-way relativities extraction from the corrected predictions is a third. The three will give different numbers, and there is no guidance on which is correct.

**2. Relativity stability bounds.** The paper proves uniqueness of Q* but says nothing about how sensitive the resulting GLM relativities are to estimation error in the propensity model. For factors with low effective N, the relativities from two independent runs of the reweighter could differ by 15-20%. There is no sandwich variance estimator or bootstrap confidence interval derivation for the KL-corrected GLM coefficients. This gap matters for underwriting sign-off.

**3. Interaction terms.** Most UK pricing GLMs include interaction terms — age-by-vehicle-group, NCD-by-age, etc. The additive GLM structure assumed in the paper's sensitivity formula $\Phi_i = D_i \cdot \partial_i h(X,D) \cdot \gamma(U_{Y|x})$ does not account for X–D interactions. For a model where D participated in interaction terms (possible for historical data where gender was included), the partial derivative $\partial_i h$ involves cross-terms and the propensity approximation understates the true sensitivity. The full Lagrange multiplier solution handles this correctly; the propensity approximation does not.

---

## Our recommendation for production deployment

For a team deploying the KL correction into a multiplicative GLM pricing system:

1. Fit the corrected GLM with sample weights as described above. The `TweedieRegressor` with `sample_weight` handles this correctly for both frequency and severity GLMs.

2. Run the relativity comparison and flag any change above ±5% as requiring a narrative explanation. Expect the largest moves in factors with the highest propensity model contribution.

3. Check effective N by factor level. For any level with effective N below 50, apply L2 regularisation (`glm_fair = TweedieRegressor(..., alpha=0.01)`) to stabilise the coefficient.

4. Verify A/E ratios by factor level on the hold-out set. The corrected model should show A/E closer to 1.0 for the factor levels most associated with the protected attribute. If A/E moves away from 1.0, the correction has been too aggressive for that cell and you need to inspect the weight distribution.

5. Document the barycentre weights (even if they are the default 0.5 / 0.5 equal split) and their justification. This is a governance decision, not a statistical one.

6. Do not confuse the corrected model's predictions with a certified zero-sensitivity result. The propensity reweighting is the correct solution for linear h and expected value pricing. For a GBM, it is an approximation whose error depends on the nonlinearity of the model's proxy use. The GLM refit under sample weights is exact for the GLM.

---

## On the question of whether this is worth the complexity

The GLM refit with KL weights is not particularly complex — it is standard weighted least squares, and any GLM implementation handles it. The complexity sits in the governance: explaining to an underwriting committee why the vehicle group relativities have moved, and justifying the barycentre weight choice for multi-attribute correction.

That governance conversation is not a reason to avoid the correction. It is the correction working: the relativities moving away from a proxied-discrimination structure and toward a genuinely risk-based structure. The A/E evidence that the correction improves calibration within high-proxy-risk cells is the argument. "The relativities for commercial van operators moved because they were partly encoding a gender signal, not just a van-related risk signal" is a defensible claim that an underwriter can work with.

The alternative — filing a Consumer Duty attestation that says "we reviewed for indirect discrimination and found no material concerns" without having done the reweighting — is not defensible. Not because of what regulators require today, but because of what they will ask next time they look at the model.

---

*Miao, K.E. and Pesenti, S.M. (2026). Discrimination-Insensitive Pricing. arXiv:2603.16720.*

*Related: [The Information-Theoretic Foundation](/2026/03/31/kl-discrimination-insensitive-pricing-insurance-fairness/) — the full mathematical derivation. [KL Fairness and FCA Consumer Duty](/2026/04/03/kl-discrimination-insensitive-pricing-fca-consumer-duty/) — the regulatory framing. [Writing a Defensible Fairness Attestation](/2026/04/03/kl-fairness-attestation-fca-ep25-2-model-validation/) — the attestation language.*

*`insurance-fairness` is at [pypi.org/project/insurance-fairness](https://pypi.org/project/insurance-fairness/). `uv add insurance-fairness` to get started.*
