---
layout: post
title: "Balance and Fairness Are the Same Problem — Multicalibration in Insurance Pricing"
seo_title: "Multicalibration for Insurance Pricing: Balance and Group Fairness Are the Same Mathematical Condition"
date: 2026-03-31
categories: [fairness, calibration]
tags: [multicalibration, autocalibration, fairness, GBM, balance-property, Consumer-Duty, Equality-Act, insurance-fairness, python, arXiv-2603-16317, conditional-mean-independence, isotonic-regression, credibility, UK-regulation, proxy-sufficiency, Denuit]
description: "Denuit, Michaelides & Trufin (arXiv:2603.16317) prove that autocalibration and group fairness are mathematically equivalent. A GBM that is well-calibrated overall but miscalibrated within subgroups has a fairness problem and an accuracy problem simultaneously. We explain the duality, the three correction algorithms, and what this means for Consumer Duty compliance."
author: burning-cost
---

Actuarial teams and fair-lending teams have been solving the same problem from opposite ends for years without knowing it. One asks: do our predicted premiums equal expected claims at every level of the rate? The other asks: do policyholders with the same risk profile pay the same price regardless of which group they belong to?

A paper published in March 2026 by Denuit, Michaelides and Trufin (arXiv:2603.16317) proves these are formally equivalent. Not related. Not compatible. The same condition.

This matters practically because GBMs — which now underpin most UK personal lines pricing models — fail the weaker single-group calibration condition. They will also fail the joint group-calibration condition. The paper gives three algorithms for fixing this, quantifies the accuracy cost (lower than you would expect), and provides the statistical test that answers the FCA's implicit question about indirect discrimination in models that exclude protected characteristics.

---

## The two conditions, and why GBMs fail both

**Autocalibration** (Denuit, Charpentier & Trufin 2021, Definition 3.1) requires:

```
E[Y | π(X) = p] = p   for all p
```

At every premium level, the expected claims of policies charged that premium should equal the premium itself. GLMs with canonical links satisfy this on training data by construction — it falls out of the score equations. GBMs do not. They minimise average Poisson deviance, which does not enforce the conditional mean constraint at each prediction level. The result: a GBM may show an A/E of 1.00 at the portfolio level while charging 20% too little in the £400–600 band and 15% too much in the £1,200–1,600 band. The errors cancel in the aggregate.

**Conditional mean independence** (Definition 4.4 of the paper) requires:

```
E[Y | π(X) = p, S = s] = E[Y | π(X) = p, S = s']   for all p, s, s'
```

Policyholders paying the same premium should have the same expected claims, regardless of which group they belong to. This is the sufficiency criterion from the fairness literature — the actuarially coherent version of non-discrimination. It does not require equal premiums across groups. It requires that at any given premium level, no group is being systematically over- or under-charged relative to its risk.

**Multicalibration** (Definition 5.1) combines both:

```
E[Y | π(X) = p, S = s] = p   for all p and all s
```

Property 5.2 makes the duality explicit: multicalibration implies autocalibration; autocalibration plus conditional mean independence implies multicalibration. The two conditions are complementary halves of a single constraint. A model that passes the combined test is both accurately calibrated per premium level and fair across groups, simultaneously.

The key implication: if your GBM is miscalibrated within premium bands (which it almost certainly is), it is also unfair in the multicalibration sense. These are not separate problems requiring separate fixes. They are one problem.

---

## Why this framing settles the fairness/accuracy trade-off debate

There is a persistent belief in pricing teams that fairness comes at the cost of accuracy — that constraining a model to produce equitable outcomes necessarily degrades its predictive power. The Bregman loss ordering in Proposition 6.3 of the paper shows this is wrong, at least for the multicalibration criterion.

For any Bregman loss — Poisson deviance, squared error, Tweedie deviance — the hierarchy is:

```
E[L(Y, π)]          baseline GBM
  ≥ E[L(Y, π_bc)]   autocalibrated
    ≥ E[L(Y, π_mbc)] multicalibrated
      ≥ E[L(Y, μ(X))] oracle (true conditional mean)
```

Each step strictly improves expected loss. The fairness correction (moving from autocalibrated to multicalibrated) does not degrade accuracy — it fills in a calibration gap that was already there. The GBM leaves a gap. Multicalibration fills it.

That said, there *is* a Gini cost. The paper reports approximately 5% in the French MTPL case study (677,991 policies, vehicle age as the sensitive feature). Specifically: the autocalibrated model achieves Gini 0.7237; the multicalibrated model achieves 0.6876. That 0.036 gap represents a real reduction in discriminatory power — the model is slightly less able to separate high-risk from low-risk policies after the group correction is applied.

We think this is the honest framing: multicalibration costs roughly 5% Gini. You are buying something real with that cost — the guarantee that no group is systematically mis-priced within any premium band. Whether the trade-off is worth it is a pricing committee decision. But the paper shows the cost is bounded and quantifiable. The previous assumption — that fairness costs are large and unpredictable — was never tested.

---

## Three algorithms, not one

The paper describes three distinct correction methods. Most implementations (including ours in `insurance-fairness`) currently cover only one.

### Algorithm A: Groupwise isotonic regression

One-shot. For each group `l`, fit a monotone non-decreasing regression of observed claims `Y` on the current prediction `π(X)`, restricted to policies in that group:

```python
# Conceptual — for each group l:
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(y_min=0, increasing=True)
ir.fit(y_pred[mask_l], y_true[mask_l], sample_weight=exposure[mask_l])
y_pred_corrected[mask_l] = ir.predict(y_pred[mask_l])
```

Proposition 6.2 proves this achieves exact multibalance. The weakness: no borrowing of strength across groups. A group with 1,000 exposures gets an isotonic fit estimated entirely on its own data. If that group has unusual composition at the tails of the premium distribution, the isotonic correction extrapolates as a step function — the last known value. For small groups this is unstable. Algorithm A is appropriate when all groups are large and well-distributed across the premium range.

### Algorithm B: Iterative bias correction with credibility shrinkage

The main algorithm — and what `IterativeMulticalibrationCorrector` in `insurance-fairness` implements. Iterative, convergent, handles sparse groups.

The pseudocode (Algorithm 2 in the paper, Section 7.1.2):

```
Initialise π⁽⁰⁾ = π

For j = 0, 1, 2, ...:
  1. Bin π⁽ʲ⁾ into K equal-exposure bins {B₁,...,B_K}
  2. Residuals: R_i = Y_i - π⁽ʲ⁾(X_i)
  3. For each (bin k, group l) cell:
       b̂_kl = (1/w_kl) · Σ_{i∈I_kl} w_i · R_i     [cell mean residual]
       b̂_k  = (1/w_k)  · Σ_l Σ_{i∈I_kl} w_i · R_i [bin mean residual]
  4. Credibility blend:
       z_kl = w_kl / (w_kl + c)
       b̃_kl = z_kl · b̂_kl + (1 - z_kl) · b̂_k
  5. Update: π⁽ʲ⁺¹⁾(X_i) = π⁽ʲ⁾(X_i) + η · b̃_kl   for i in cell (k,l)
  6. Stop if: max_{k,l} |η · b̃_kl| / π̄_kl ≤ δ
```

Parameters from the case study: K=10, η=0.2, δ=0.01.

The credibility formula `z_kl = w_kl / (w_kl + c)` uses exposure-weighted cell mass, not policy count. A fleet policy covering 10 car-years counts as 10 units of exposure. This means a cell with 50 fleet policies (500 car-years) receives higher credibility than 500 annual personal-lines policies. The parameter `c` sets the threshold for full credibility — roughly, the exposure required to get Z = 0.5.

The shrinkage target matters: small groups shrink toward the bin-level pooled estimate `b̂_k`, not toward zero. Shrinking toward zero would leave small groups uncorrected and destroy marginal autocalibration. Shrinking toward the bin mean gives them approximately the same correction as the overall portfolio in that premium band, which is consistent with the autocalibration requirement.

Using `IterativeMulticalibrationCorrector` from `insurance-fairness`:

```python
from insurance_fairness import IterativeMulticalibrationCorrector

corrector = IterativeMulticalibrationCorrector(
    n_bins=10,
    eta=0.2,
    delta=0.01,
    c=200.0,      # ~200 car-years for full credibility; tune to your portfolio
    max_iter=50,
)
corrector.fit(y_true_train, y_pred_train, protected_train, exposure=exposure_train)
y_pred_fair = corrector.transform(y_pred_new, protected_new, exposure=exposure_new)
```

Note that correction is applied to the response scale — to the premium itself, not its log. If your downstream system uses log-premiums, convert before applying.

One correctness note on the current implementation: the stopping criterion in our code compares the raw A/E deviation `|b̂_kl - 1|` against `delta`. The paper's criterion normalises by the current mean prediction `π̄_kl` in each cell — a scale-invariant measure. A £10 residual in a £100-premium cell is 10%; the same £10 residual in a £1,000-premium cell is 1%. Our criterion conflates these. This will be corrected in the next release.

### Algorithm C: Local GLM for continuous sensitive attributes

The most technically demanding case: when the sensitive feature is continuous rather than categorical. Vehicle age as a float, not banded into ≤3/3–9/>9 years. Driver age. Property construction year. Income decile.

Algorithm C replaces the discrete binning of Algorithm B with a bivariate local Poisson GLM estimated over the (π, S) surface:

```
b̂⁽ʲ⁾(p, s) = E[Y - π⁽ʲ⁾(X) | π⁽ʲ⁾(X) = p, S = s]
```

estimated via a 2D local regression with neighbourhood parameter α (fraction of nearest neighbours). The critical addition is a centering step absent from Algorithm B:

```
δ⁽ʲ⁾(p,s) = z(p,s) · [b̂⁽ʲ⁾(p,s) - b̂⁽ʲ⁾(p)]
b̃⁽ʲ⁾(p,s) = b̂⁽ʲ⁾(p) + δ⁽ʲ⁾(p,s) - E[δ⁽ʲ⁾(π⁽ʲ⁾(X),S) | π⁽ʲ⁾(X) = p]
```

The centering subtracts the group-averaged deviation at each premium level. Without it, the correction would shift the marginal distribution — you would fix the group-level calibration but break the portfolio-level balance. The credibility weights `z(p,s)` are computed once before the loop (using k-NN local density) rather than recalculated each iteration, to prevent feedback between correction magnitude and shrinkage intensity.

We have not yet implemented Algorithm C. The paper relies on R's `locfit` for the local Poisson GLM; Python has no direct equivalent. `statsmodels.nonparametric.kernel_regression.KernelReg` is the closest option but assumes Gaussian response. This is on the roadmap.

---

## The proxy sufficiency test

Proposition 6.5 is the most commercially valuable piece of this paper for UK pricing teams.

Post-Test-Achats, gender cannot be used in EU-regulated motor pricing. In the UK, disability is generally excluded as a rating factor. When S is excluded, multicalibration without using S in the model is achievable only if:

```
E[Y | π(X), S] = E[Y | π(X)]
```

That is: the model's allowed proxies (age, vehicle type, occupation, geography) must capture all the risk information that S contains. If they do not, the gender-blind or disability-blind model is mis-pricing one group relative to another at specific premium levels — it is indirectly discriminating, in the Equality Act s.19 sense.

The proposition forces an honest question that most technical pricing sign-offs do not ask: does our age-based motor model, operating without gender, produce the same expected claims for male and female drivers at each premium level? If the answer is no — if young male drivers in the £1,500–2,000 band are under-priced and young female drivers in the same band are over-priced — then the model fails conditional mean independence. It passes the A/E check only because the errors cancel.

This is not currently implemented in `insurance-fairness`. The class would look like:

```python
from insurance_fairness import ProxySufficiencyTest

test = ProxySufficiencyTest(n_bins=10, alpha=0.05, min_bin_size=30)
report = test.test(
    y_true=claims,
    y_pred=premiums,           # from gender-blind model
    excluded_attr=gender,      # the excluded characteristic
    exposure=exposure,
)
print(report.passes)           # bool
print(report.failing_cells)    # DataFrame: (bin, group, bias, p_value)
print(report.interpretation)   # regulatory framing
```

The test is structurally identical to `MulticalibrationAudit` — it checks calibration within (bin, group) cells — but the regulatory framing is different. You are not asking "is our model multicalibrated?" You are asking "given that we excluded gender, does the model still satisfy conditional mean independence for gender?" If it does not, you need either to find better proxies or to acknowledge the limitation in your Consumer Duty fair value assessment.

---

## Where this sits in the pricing cycle

Multicalibration is a post-processing step. It does not change how you train the GBM.

The order matters (from KB entry 4700, and from the Bregman hierarchy):

1. **Train the GBM** normally, minimising Poisson deviance. No changes here.
2. **Autocalibrate** the GBM output — a local GLM or isotonic step that corrects the marginal miscalibration (`E[Y | π(X) = p] = p`). If you are already applying a GLM overlay or credibility loading, this may already be handled.
3. **Run `MulticalibrationAudit`** on the holdout set. This is the governance step — the output is your Consumer Duty evidence that the product delivers fair outcomes across demographic groups.
4. **Apply `IterativeMulticalibrationCorrector`** if the audit finds material failures. Apply to training data; use `transform()` on live predictions.
5. **Re-run the audit** after correction. Confirm: multicalibration passes, Gini loss is within the expected ~5%, total premium income is preserved, and no group is now over-charged to compensate.
6. **Deploy `MulticalibrationMonitor`** (from `insurance-monitoring`) in your quarterly claims feed. Re-trigger the correction cycle if monitoring raises an alert.

The autocalibration step (stage 2) is a prerequisite. The Bregman hierarchy (Proposition 6.3) requires autocalibration as the starting point — multicalibration corrects the residual group-level gaps on top of the marginal correction.

---

## The regulatory context — and what EP25/2 does not cover

The regulatory hook for multicalibration in the UK is **Consumer Duty (PS22/9, PRIN 2A)** and **Equality Act 2010 section 19** — not EP25/2.

EP25/2 (FCA, July 2025) is the evaluation of GIPP price-walking remedies. It concerns renewal pricing and what happened to personal lines customers after the January 2022 intervention. It has nothing to say about whether a model systematically under-prices one demographic group in specific premium bands.

Consumer Duty requires firms to demonstrate good outcomes for all retail customers. Multicalibration provides the statistical evidence at the granularity the FCA is asking for — not just "our A/E is 1.00 for young drivers overall" but "our A/E is within tolerance for young drivers at every premium level." FCA TR24/2 (multi-firm Consumer Duty review, 2024) specifically found that firms needed to go further in embedding outcomes monitoring at the segment level. A `MulticalibrationAudit` report is exactly what that review was asking for.

Equality Act s.19 is the indirect discrimination provision. If a gender-blind model inadvertently produces a pattern where female policyholders are consistently overcharged at the £800–1,200 price point, that is potentially indirect discrimination against women as a class. The proxy sufficiency test from Proposition 6.5 is the rigorous test for this — not a correlation check between gender and premium, but a conditional mean independence test at each premium level.

The demographic parity / sufficiency incompatibility is worth stating explicitly for legal and compliance colleagues: the paper proves (Section 4.2, building on Charpentier 2024) that demographic parity — equal premium distributions across groups — is incompatible with sufficiency whenever S has genuine actuarial content. You cannot have a risk-based pricing model that also produces identical premium distributions for all groups. Multicalibration adopts sufficiency — the right criterion for insurance — not demographic parity.

As of March 2026, the FCA has six active Consumer Duty investigations, two involving insurers. "Inability to demonstrate fair value" is being framed as a governance and culture failure, not merely a technical one. Chief Actuaries who cannot show a regulator that their model is not systematically miscalibrated within demographic groups are in a weaker position than those who can.

---

## What the paper does not solve

We will not pretend the framework is complete.

**Multiple protected characteristics simultaneously.** The paper handles one sensitive feature S at a time. In practice, age, disability proxy, and geography all need checking. The cell count for joint multicalibration grows as the Cartesian product — with 3 age bands, 2 disability categories, and 10 geographic regions, you have 60 cells per premium bin. This creates severe sparsity problems and is an open research question.

**The right value of c.** The credibility parameter `c` determines how quickly small groups get their own group-level correction rather than borrowing from the bin pooled estimate. The paper's case study does not provide guidance on calibrating `c` against portfolio structure. We are using c=200 as a default (roughly 200 car-years for Z = 0.5), which is reasonable for standard UK motor portfolios, but sensitivity analysis on your own data is warranted.

**Algorithm C in Python.** The local GLM approach for continuous sensitive features relies on R's `locfit`. We do not yet have a satisfactory Python implementation. Until that is available, the practical recommendation for genuinely continuous sensitive features (vehicle age as a float, driver age) is to band them into 3–5 categories and apply Algorithm B. You lose some precision at band boundaries; you gain a working implementation.

---

## The paper

Denuit, M., Michaelides, M. & Trufin, J. (2026). *Balance and Fairness through Multicalibration in Nonlife Insurance Pricing*. arXiv:2603.16317.

The direct precursor, and essential reading for the mathematical machinery: Denuit, M., Charpentier, A. & Trufin, J. (2021). *Autocalibration and Tweedie-Dominance for Insurance Pricing with Machine Learning*. Insurance: Mathematics and Economics 101: 485–497.

The CS multicalibration paper this builds on: Hébert-Johnson, U., Kim, M., Reingold, O., Rothblum, G. & Valiant, P. (2018). *Multicalibration: Calibration for the (Computationally-Identifiable) Masses*. ICML 2018. arXiv:1711.08513.

The implementation is in [`insurance-fairness`](/insurance-fairness/). The `IterativeMulticalibrationCorrector` covers Algorithm B. Algorithm A, Algorithm C, and the proxy sufficiency test are on the roadmap for v0.7.0.
