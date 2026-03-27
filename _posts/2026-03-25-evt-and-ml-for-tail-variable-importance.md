---
layout: post
title: "EVT and ML for Tail Variable Importance: Which Covariates Drive Your Largest Claims?"
date: 2026-03-25
categories: [pricing, techniques, large-loss]
tags: [evt, extreme-value-theory, gpd, variable-importance, shap, tail-risk, bodily-injury, severity, reserving, insurance-severity, insurance-quantile, eqrn, large-loss, ibnr]
description: "A covariate that predicts mean severity well may tell you almost nothing about your 99th percentile claims. Here is how to identify which rating factors actually drive large losses."
---

Ask a pricing actuary which factors drive motor bodily injury severity and you will get a sensible list: claimant age, legal representation, injury type, vehicle speed. Run your SHAP values on a CatBoost severity model and the same factors come out near the top. That analysis is correct — for the average claim. It may be almost entirely wrong for the claims that actually threaten your solvency.

This is the central problem with applying standard variable importance to severity modelling. A CatBoost SHAP value measures the average contribution of a feature across all predictions. A £2,000 soft tissue whiplash claim and a £2,000,000 catastrophic spinal injury claim contribute symmetrically to mean SHAP — but they sit in completely different statistical regimes. The factors that explain why a claim exceeds £500k may be entirely different from the factors that explain why it reaches £8k rather than £5k.

Clémençon and Sabourin (arXiv:2504.06984, revised June 2025) formalise this. They combine multivariate extreme value theory with statistical learning to provide non-asymptotic guarantees for learning from tail observations — concentration inequalities adapted to low-probability regions, high-dimensional lasso for extreme-value contexts, and cross-validation theory for model selection when your "test set" is the top 2% of your loss distribution. The practical takeaway for pricing actuaries is a framework for doing something most teams do not do: computing variable importance separately for the tail.

---

## Why tail importance differs from average importance

Start with a simple example. You have a motor BI portfolio with two covariates that matter: injury severity code (soft tissue vs. fracture vs. neurological) and whether the claimant has legal representation. Both matter for average claim cost. Soft tissue claims average £4k; neurological average £85k. Legal representation adds 30% across the board.

Now look at only claims above £200k. In that subset, injury severity code explains nearly everything: catastrophic neurological injuries make up 85% of the tail, fractures 14%, soft tissue almost nothing. Legal representation is nearly universal in the tail (virtually all claims above £200k are represented), so it has almost zero discriminatory power there. Its apparent importance in the tail is suppressed by near-constant value.

The mean importance ranking says: injury type first, legal representation second. The tail importance ranking says: injury type first, legal representation irrelevant. These are different models of the world. For pricing, either might be acceptable depending on your use case. For reserving, the tail model is what matters: your IBNR on catastrophic claims is driven by injury type, not legal representation.

The formal expression of this comes from extreme value theory. Under peaks-over-threshold, exceedances above a threshold u follow a Generalised Pareto Distribution:

$$P(X - u > y \mid X > u) = \left(1 + \frac{\xi \cdot y}{\sigma}\right)^{-1/\xi}$$

The shape parameter xi governs how heavy the tail is. The crucial point is that xi can vary by covariate. If xi for neurological claims is 0.4 and xi for fracture claims is 0.1, the neurological tail is much heavier and the covariate becomes more important as you move deeper into the distribution. Standard GLM severity modelling estimates a single conditional mean — it is entirely insensitive to this.

---

## EVT basics for insurance: what the parameters tell you

For UK motor BI, typical tail indices xi run between 0.2 and 0.5, depending on book composition and how aggressively you have applied outwards limits. For employers' liability and public liability, you may see xi approaching 0.7. Property damage tends to be lower — around 0.1 to 0.2 — reflecting physical limits on maximum claim size.

The threshold u is chosen by the mean excess plot: plot the sample mean of (X - u) against u. Under GPD, this should be approximately linear above the correct threshold. In practice for motor BI, thresholds of £50k to £100k are common for splitting attritional from large loss.

The `TruncatedGPD` class in [`insurance-severity`](/insurance-distributional/) handles the most common complication in UK insurance data: per-policy limits create upper truncation in the claim distribution. Standard GPD MLE ignores this and underestimates xi. When you have heterogeneous limits — common in commercial lines and in any EL/PL book — you should be using the truncation-corrected estimator:

```python
from insurance_severity import TruncatedGPD

# exceedances: claim amounts above threshold, shape (n,)
# limits: per-policy limits, shape (n,) — use np.inf for uncapped
gpd = TruncatedGPD(threshold=100_000)
gpd.fit(exceedances, limits)
print(gpd.summary())
# {'xi': 0.38, 'sigma': 62400, 'threshold': 100000,
#  'se_xi': 0.041, 'se_sigma': 4800}
```

For IBNR-heavy books (EL, long-tail liability), use `CensoredHillEstimator` instead. IBNR claims are right-censored: you see current development but not ultimate settlement. The Albrecher et al. (2025) correction divides the Hill numerator by the fraction of uncensored claims in the top-k, preventing the standard Hill estimator from underestimating the tail index:

```python
from insurance_severity import CensoredHillEstimator

hill = CensoredHillEstimator()
hill.fit(claims, censored=ibnr_flag)
print(f"xi = {hill.xi:.3f}, 95% CI = {hill.ci}")
# xi = 0.42, 95% CI = (0.31, 0.54)
```

---

## The paper's approach: ML importance on tail observations only

Clémençon and Sabourin's contribution is to give this intuition rigorous theoretical foundations. The practical algorithm is:

1. Set a tail threshold (GPD u, or simply the 85th or 90th percentile of the loss distribution).
2. Train your ML model — any flexible model, CatBoost or otherwise — on the full dataset as normal.
3. Compute SHAP values or permutation importance **restricting to observations above the tail threshold**.
4. Compare the tail importance ranking to the full-dataset ranking.

The concentration inequalities in the paper show that this restricted importance estimate converges to the true tail importance at a controlled rate, despite the drastically reduced sample size. The key insight is that extreme observations are not a random subsample — they are drawn from the regularly varying part of the distribution, which has structure that the theory exploits. You are not simply discarding 90% of your data and hoping for the best; the EVT framework licenses the inference.

For insurance practitioners, the practical constraint is that the tail is sparse. If your BI portfolio has 50,000 claims per year and you set the threshold at the 95th percentile, you have 2,500 tail observations. That is enough for permutation importance on ten or fifteen covariates, but not enough for SHAP on a high-dimensional model without regularisation. The paper's high-dimensional lasso extensions address this directly — enforcing sparsity to make selection feasible in the tail.

---

## Which factors drive large BI claims vs attritional?

For UK motor third-party bodily injury, the practical comparison is instructive. Our experience is that the importance rankings diverge materially above roughly the 90th percentile (approximately £25k for a typical personal lines book as at 2025).

**Factors that matter more in the tail than in the mean:**

- Injury severity grade (neurological and spinal injuries concentrate almost entirely in the tail)
- Claimant age at injury (younger claimants have longer earnings streams; this matters far more for catastrophic care costs than for soft tissue)
- Rehabilitation pathway (admission to specialist neurological rehabilitation; near-irrelevant for attritional, almost deterministic for the largest claims)
- Liability dispute (contested claims that proceed to trial tend to settle at the top of the Judicial College Guidelines range)

**Factors that matter less in the tail than in the mean:**

- Vehicle type (strongly predictive of soft tissue claim frequency and mean cost; much weaker for catastrophic injury, where the biomechanics matter more than the vehicle)
- Legal representation (as noted above: universal in the tail, so carries little discriminatory power there)
- Region (postcode-level variation in soft tissue claim propensity is well-documented; the regional signal weakens substantially in the catastrophic injury tail)

Region is particularly interesting for pricing. If you have a postcode-level severity relativities that you apply across the full distribution, you are likely misallocating capital: the attritional variation is real but the catastrophic tail is roughly region-agnostic. Your large loss loading should not be differentiating by postcode to the same degree as your attritional loading.

---

## Fitting a covariate-dependent GPD tail

The `EQRNModel` in `insurance-quantile` implements exactly this — a neural network that learns covariate-dependent GPD parameters (xi and sigma both varying by covariate profile). This is the direct operationalisation of asking "which covariates matter for the tail":

```python
from insurance_quantile.eqrn import EQRNModel

# Two-step: Step 1 fits intermediate quantile via CatBoost K-fold OOF
# Step 2 trains GPD neural net on exceedances only
model = EQRNModel(
    tau_0=0.85,           # train GPD on top 15% of claims
    hidden_sizes=(32, 16, 8),
    shape_fixed=True,     # start with scalar xi as regularised baseline
    n_epochs=300,
)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Covariate-dependent 99.5th percentile — what your Solvency II SCR needs
var_995 = model.predict_quantile(X_test, q=0.995)

# TVaR for XL reinsurance layer pricing
tvar_99 = model.predict_tvar(X_test, q=0.99)

# XL layer expected loss: £1m xs £500k
xl_expected = model.predict_xl_layer(X_test, attachment=500_000, limit=1_000_000)
```

The `shape_fixed=True` mode is important as a diagnostic first step. If a scalar xi model (standard GPD with covariate-dependent sigma only) fits nearly as well as the full model where xi also varies, your tail shape is relatively homogeneous and the headline rating factors are mainly affecting the GPD scale rather than the heaviness of the tail. That is a different and less severe problem than having covariates that change xi — changing sigma is correctable by a standard severity multiplier, but changing xi means the entire percentile structure shifts.

Once you have a fitted EQRN, you can compute permutation importance restricted to tail predictions:

```python
import numpy as np

# Permutation importance on TVaR predictions for tail observations
# (X_test_tail, y_test_tail) are the test rows where y > threshold
baseline_tvar = model.predict_tvar(X_test_tail, q=0.99)
baseline_loss = np.mean(np.abs(baseline_tvar - y_test_tail))

feature_names = ["claimant_age", "injury_grade", "legal_rep", "region", "vehicle_type"]
tail_importance = {}

for i, feat in enumerate(feature_names):
    X_perm = X_test_tail.copy()
    rng = np.random.default_rng(42)
    X_perm[:, i] = rng.permutation(X_perm[:, i])
    perm_tvar = model.predict_tvar(X_perm, q=0.99)
    perm_loss = np.mean(np.abs(perm_tvar - y_test_tail))
    tail_importance[feat] = perm_loss - baseline_loss

for feat, imp in sorted(tail_importance.items(), key=lambda x: -x[1]):
    print(f"{feat:20s}  {imp:+.1f}")
```

Run this on the full test set for average importance, then on `X_test_tail` for tail importance. The divergence between the two rankings is the actionable finding.

---

## Connection to reserving: IBNR segmentation

The reserving implication is direct. Your IBNR development factors are typically applied at a relatively coarse level of segmentation: accident year, channel, broad injury class. If you have identified that tail-important variables (injury severity grade, claimant age) drive the largest claims, and those variables are not in your IBNR segmentation, your large loss IBNR is poorly specified.

Consider a simple check: take your open large losses above £100k. Fit a GPD to the exceedances. Now fit separate GPDs by injury severity grade. If the xi values differ materially — say 0.45 for neurological vs 0.15 for fracture — your aggregate IBNR is a mixture of two very different tail behaviours. The mixed-population xi estimate might be 0.25, which underestimates the catastrophic neurological tail and overestimates the fracture tail. That asymmetry is unlikely to cancel in a reserve.

The `CensoredHillEstimator` is the right tool for this analysis. IBNR claims are right-censored by definition (current development is not final settlement), and the standard Hill estimator is biased in the presence of censoring:

```python
from insurance_severity import CensoredHillEstimator

# Split by injury grade
for grade in ["neurological", "fracture", "soft_tissue"]:
    mask = injury_grade == grade
    hill = CensoredHillEstimator()
    hill.fit(
        claims=open_large_loss_development[mask],
        censored=ibnr_flag[mask],
    )
    print(f"{grade:15s}  xi={hill.xi:.3f}  CI={hill.ci}  k_opt={hill.k_opt}")
```

If neurological xi is materially above fracture xi, that is evidence for separate tail models in your IBNR segmentation, not a blended tail factor. This is not a radical methodological claim — it is the same logic as running separate development triangles by claim type, but applied to the tail index rather than the link ratios.

---

## What to do with this

The practical workflow for a UK pricing or reserving actuary:

1. **Fit your standard severity model** (CatBoost, whatever you use). Compute full-sample SHAP.

2. **Set a tail threshold** using the mean excess plot. For motor BI, £50k–£100k is a reasonable starting point. Check that you have at least 300–500 observations above it for stable tail importance estimates.

3. **Compute SHAP or permutation importance restricted to tail observations.** Compare the ranking to full-sample SHAP. The factors that rise in the tail ranking are your large-loss drivers; the factors that fall are attritional-only.

4. **Fit a covariate-dependent GPD** (`TruncatedGPD` for cross-sectional large loss; `EQRNModel` for a full conditional tail model) to verify that xi genuinely varies by covariate. If `shape_fixed=True` performs comparably to the full EQRN, the tail shape is homogeneous and you only need to worry about scale variation.

5. **Check your IBNR segmentation** against the tail importance ranking. Variables that are important for the tail but absent from your IBNR segmentation are a reserving risk.

The Clémençon-Sabourin framework provides the theoretical backing for why this approach is statistically valid — the concentration inequalities ensure that tail importance estimation on sparse exceedance datasets does not simply amplify noise. But you do not need to read the paper to implement the practice. You need a threshold, a tail importance calculation, and the willingness to act on a ranking that differs from your full-sample SHAP.

Most UK pricing teams have sophisticated mean models and rudimentary tail models. The mean model tells you where the average claim comes from. The tail model tells you what to worry about. They are not the same question.

---

Source: Clémençon, S. & Sabourin, A. (2025). "Weak Signals and Heavy Tails." arXiv:2504.06984.

- [insurance-severity](https://github.com/burning-cost/insurance-severity) — `TruncatedGPD`, `CensoredHillEstimator`, `WeibullTemperedPareto`
- [insurance-quantile](https://github.com/burning-cost/insurance-quantile) — `EQRNModel` for covariate-dependent GPD tails
- [Large Loss Loading for Home Insurance](/2026/03/04/large-loss-loading-for-home-insurance/) — applying these ideas to property
- [Distributional GBMs for Insurance: Pricing Variance, Not Just the Mean](/2026/03/05/insurance-distributional/) — a complementary approach to variance-aware pricing
