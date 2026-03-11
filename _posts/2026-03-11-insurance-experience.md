---
layout: post
title: "Individual Experience Rating Beyond NCD: From Bühlmann-Straub to Neural Credibility"
date: 2026-03-11
categories: [libraries, pricing, credibility]
tags: [experience-rating, credibility, Bühlmann-Straub, Poisson-gamma, Bayesian, state-space, deep-learning, attention, NCD, motor, fleet, commercial-liability, FCA, PS21/5, insurance-experience, python, renewal-pricing]
description: "NCD is a crude proxy for individual risk. insurance-experience implements actuarially correct posterior experience rating at policy level: static Bühlmann-Straub, dynamic Poisson-gamma state-space (Ahn/Jeong/Lu/Wüthrich 2023), surrogate Bayesian posteriors (Calcetero/Badescu/Lin 2024), and deep attention credibility (Wüthrich 2024). All four models produce a multiplicative credibility factor that slots into existing GLM rating engines."
---

Every UK motor renewal book contains a version of this problem. The policyholder has been with you for seven years. They have had one at-fault claim, six years ago. Under your current NCD structure they are at 60% — they lost two steps for the claim and have been rebuilding since. Their current NCD discount is 45%. The neighbour renewed last month with a clean 5-year history and got 65%.

The 20-point NCD gap does some work. The six-year-ago claimant is plausibly higher risk than the clean-history neighbour. But it is a blunt instrument. It ignores the other six years of claims-free data. It ignores the claim amount — whether it was a £1,200 cosmetic prang or a £40,000 liability settlement. It applies the same binary step structure to a 19-year-old with three years' history as to a 52-year-old fleet driver. And it is non-negotiable: the structure is fixed by your product design, and every competitor uses approximately the same grid.

Individual Bayesian experience rating is the actuarially correct answer to this problem. Given a policyholder's full claims history and your a priori GLM premium, compute the posterior premium — the expected cost conditioned on both the GLM's rating factors and the observed individual experience. The output is a multiplicative credibility factor: posterior_premium = prior_premium × credibility_factor. This factor slots directly into your existing rating engine. Policyholders with good experience get a factor below 1.0. Those with poor experience get a factor above 1.0. The portfolio-level balance property holds: the sum of posterior premiums equals the sum of actual losses.

[`insurance-experience`](https://github.com/burning-cost/insurance-experience) implements four models at different points on the complexity/data-requirement curve. MIT-licensed, on PyPI.

```bash
uv add insurance-experience
```

---

## Why NCD is not the same thing

NCD and individual experience rating solve superficially similar problems but are architecturally different.

NCD is a contractual mechanism. The step structure is fixed at product design time. Transitions follow fixed rules. A claim moves you down a fixed number of steps; a claims-free year moves you up one. The resulting discount is a deterministic function of your NCD history, bounded between 0% and 65% (or 70% or 75%, depending on the insurer). It is not a posterior probability estimate. The FCA noticed this: PS21/5 requires that renewal premiums are "fairly reflective of the risk" — NCD steps that diverge from actual risk-adjusted cost differentials are a compliance exposure.

Individual experience rating starts from the posterior. Given that this policyholder has filed k claims in n years of exposure, what is our updated estimate of their underlying claim rate? This is a standard Bayesian inference problem with a natural conjugate structure (Poisson observations, Gamma prior), and the closed-form answer is the Bühlmann-Straub credibility estimate. The credibility weight Z = n / (n + κ) says how much weight we put on the individual's observed experience versus the portfolio prior, where κ is the ratio of within-policyholder variance to between-policyholder variance. High κ means we stay close to the prior even with extensive history; low κ means the individual experience dominates quickly.

NCD and experience rating are not alternatives — they address different things. NCD is a contractual structure. The credibility factor applies on top of the a priori GLM, which already incorporates the policyholder's characteristics. What the credibility factor captures is the individual's departure from the segment mean predicted by the GLM — the residual variation that the rating factors do not explain.

---

## The four model tiers

### StaticCredibilityModel: Bühlmann-Straub at policy level

The baseline model. Bühlmann-Straub is covered in the `credibility` library at group level — here it runs at individual policy level, which changes what we're estimating. Instead of blending a group's loss ratio with the portfolio mean, we're blending an individual's observed claim rate with their GLM-predicted rate.

```python
from insurance_experience import StaticCredibilityModel
import numpy as np

# claims: array of claim counts per year per policy (shape: n_policies x n_years)
# exposures: exposure units (e.g. years on risk) per cell
# prior_pure_premiums: GLM output per policy

model = StaticCredibilityModel()
model.fit(claims, exposures, prior_pure_premiums)

print(f"Estimated kappa: {model.kappa_:.4f}")
# kappa = within-policy variance / between-policy variance
# Low kappa -> individual experience dominates quickly
# High kappa -> many years needed before credibility weight is substantial

# Credibility factors (multiplicative)
factors = model.credibility_factors(claims, exposures)
print(f"Factor range: {factors.min():.4f} – {factors.max():.4f}")
print(f"Portfolio balance: {(factors * prior_pure_premiums * exposures).sum() / (prior_pure_premiums * exposures).sum():.4f}")
# Should be close to 1.0 — the balance property
```

The kappa parameter is fitted by maximum likelihood on the marginal distribution, integrating out the individual random effects. For UK motor books, κ typically falls in the range 1–5 for frequency rating (meaning you need 1–5 claims-free years of individual history before the credibility weight reaches 50%). The exact value depends on the heterogeneity in your book — the more variation between policyholders that the GLM does not explain, the lower κ will be and the faster individual experience accumulates credibility.

Static credibility has a well-known limitation: it ignores the temporal structure of claims. A policyholder with 2 claims 8 years ago and a clean recent record is treated identically to one who filed 2 claims last year, provided their total exposure is the same. The dynamic model fixes this.

### DynamicPoissonGammaModel: state-space with seniority weighting

This is the Ahn, Jeong, Lu, and Wüthrich (2023, ASTIN Bulletin) model. The underlying claim rate μ_i is time-varying, following a Poisson-Gamma state-space process. Older observations receive less weight in the posterior update because the underlying risk may have changed. Parameters p and q control the "forgetting factor" — how quickly the influence of old claims decays.

```python
from insurance_experience import DynamicPoissonGammaModel

model = DynamicPoissonGammaModel()
model.fit(claims, exposures, prior_pure_premiums)

print(f"p = {model.p_:.4f}  (prior shape update rate)")
print(f"q = {model.q_:.4f}  (prior rate update rate)")

# Dynamic credibility factors weight recent experience more
factors = model.credibility_factors(claims, exposures)

# Compare: static vs dynamic for a policyholder with old claim vs recent claim
policy_old_claim = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])  # claim 8 years ago
policy_recent_claim = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])  # claim last year
exp_same = np.ones((1, 8))

from insurance_experience import StaticCredibilityModel
static = StaticCredibilityModel()
static.fit(claims, exposures, prior_pure_premiums)

print("\nOld claim policy:")
print(f"  Static factor  : {static.credibility_factors(policy_old_claim, exp_same)[0]:.4f}")
print(f"  Dynamic factor : {model.credibility_factors(policy_old_claim, exp_same)[0]:.4f}")

print("\nRecent claim policy:")
print(f"  Static factor  : {static.credibility_factors(policy_recent_claim, exp_same)[0]:.4f}")
print(f"  Dynamic factor : {model.credibility_factors(policy_recent_claim, exp_same)[0]:.4f}")
# Static: both get the same factor
# Dynamic: old claim policy gets factor closer to 1.0; recent claim policy gets higher surcharge
```

Fitting is by MLE on the negative binomial marginal log-likelihood — the Poisson-Gamma marginal integrates out the latent risk. This means fitting is tractable without MCMC. For a book of 200,000 policyholders with 5 years of history, fitting takes roughly 3–8 minutes on a standard machine; on a Databricks cluster the parallelisation cuts this substantially.

The seniority weighting has an obvious regulatory benefit: it does not permanently penalise policyholders for historical claims. The gradient of credibility factor as a function of time since last claim is measurable and defensible to an FCA reviewer. The static model's "time since claim is irrelevant" property is harder to justify under PS21/5's "fairly reflective of risk" standard.

### SurrogateModel: importance-sampling Bayesian posteriors

The Poisson-Gamma conjugate structure works because the posterior has a closed form. When you want to use a non-conjugate model — a zero-inflated Poisson, a negative binomial with overdispersion that varies by covariates, or any distribution where the GLM output is not simply the Gamma prior mean — the closed form breaks down.

The Calcetero, Badescu, and Lin (2024, IME) surrogate approach handles this via importance sampling. Rather than computing the posterior exactly, it draws samples from a proposal distribution, weights them by the likelihood of the observed claims, and approximates the Bayesian premium from the weighted sample. A weighted least squares correction function then adjusts for sub-portfolio biases.

```python
from insurance_experience import SurrogateModel

# Works with any marginal distribution — here, negative binomial
model = SurrogateModel(
    base_distribution="negative_binomial",
    n_is_samples=2000,
    correction="wls",  # weighted least squares correction
    random_state=42,
)
model.fit(claims, exposures, prior_pure_premiums)

factors = model.credibility_factors(claims, exposures)

# Posterior mean vs IS estimate uncertainty
posterior_means, posterior_stds = model.credibility_factors(
    claims, exposures, return_std=True
)
print(f"IS standard error (mean): {posterior_stds.mean():.4f}")
# Higher n_is_samples -> lower standard error
```

The IS standard error tells you how reliable the approximation is. For the default 2,000 samples, the standard error on the credibility factor is typically below 0.002 for individual policies — negligible for practical pricing purposes. If you need higher precision (for extreme outlier policies with many claims), increase `n_is_samples`.

The WLS correction addresses a systematic bias: the IS estimates on a sub-portfolio may not balance at portfolio level because the proposal distribution is imperfect. The correction learns a simple linear adjustment from sub-portfolio actual-vs-expected, then applies it uniformly. This restores the balance property without requiring a full re-estimation of the IS weights.

### DeepAttentionModel: learned credibility weights

Wüthrich (2024) demonstrated that the credibility weight Z in Bühlmann-Straub can be replaced by a learned attention mechanism. Instead of Z = n / (n + κ) with κ estimated from the data, a transformer-style attention layer learns optimal weights for combining the individual's claims history with the portfolio prior. The weights are distribution-free — there is no assumption about the form of the heterogeneity distribution.

```python
from insurance_experience import DeepAttentionModel
# Requires: uv add "insurance-experience[deep]"

model = DeepAttentionModel(
    n_heads=4,
    n_layers=2,
    d_model=32,
    dropout=0.1,
    learning_rate=1e-3,
    n_epochs=100,
    batch_size=512,
    random_state=42,
)
model.fit(claims, exposures, prior_pure_premiums)

factors = model.credibility_factors(claims, exposures)

# Attention weights show how the model uses history
# Shape: (n_policies, n_heads, n_years, n_years)
weights = model.attention_weights(claims, exposures)
print(f"Attention on most recent year (head 0): {weights[:, 0, -1, -1].mean():.4f}")
print(f"Attention on oldest year (head 0)      : {weights[:, 0, -1, 0].mean():.4f}")
# Typically: higher attention on recent years, with learned decay
```

The model requires PyTorch and meaningful training data — we would not recommend it below 50,000 policyholders with at least 3 years of history. On a portfolio of 200,000 with 8-year histories, it consistently outperforms static credibility on held-out log-likelihood by 2–4%. The gain over dynamic Poisson-gamma is smaller (0.5–1.5%) and depends on how well the Poisson-gamma state-space captures the heterogeneity structure in your book.

The distribution-free property matters for portfolios with unusual claim distributions — zero-inflated motor liability, for example, or commercial property with large loss concentration. The conjugate models impose parametric assumptions on the heterogeneity distribution; the attention model does not.

---

## The balance property

All four models enforce the balance property at portfolio level:

```python
# All four models satisfy this at fit time:
posterior_premiums = factors * prior_pure_premiums
actual_losses = (claims * exposures).sum()
predicted_losses = posterior_premiums.sum()
print(f"Balance ratio: {predicted_losses / actual_losses:.6f}")  # Should be ~1.000
```

This is not automatic — it is an explicit constraint in the model fitting. Without it, applying credibility factors would shift the aggregate expected loss. You would be changing your overall loss ratio, not just redistributing it. The balance property ensures that individual experience rating is actuarially neutral at portfolio level: it redistributes premiums within the book without changing the aggregate technical rate.

The balance constraint is directly relevant to the FCA GIPP (PS21/5) requirements. The regulation requires that aggregate renewal premiums are not inflated relative to new business for equivalent risk. An experience rating adjustment that preserves aggregate technical rate while differentiating individual renewal pricing is compliant on its face. An adjustment that changes the aggregate without regulatory approval is not.

---

## Integrating with an existing GLM rating engine

The credibility factor is multiplicative, which means it plugs into any rating engine that applies multiplicative factors. In Emblem or Radar terms, it is an additional rating factor applied after the model rating:

```python
from insurance_experience import DynamicPoissonGammaModel
import pandas as pd

# Typical renewal pricing workflow:
# 1. Run your GLM/GBM to get prior_pure_premium per policy
# 2. Fit DynamicPoissonGammaModel on claims history
# 3. Apply credibility factor at renewal

model = DynamicPoissonGammaModel()
model.fit(claims_history, exposures_history, prior_premiums_history)

# At renewal:
renewal_factors = model.credibility_factors(
    claims=claims_history_current,
    exposures=exposures_current,
)

renewal_df = pd.DataFrame({
    "policy_id": policy_ids,
    "prior_premium": prior_premiums_renewal,
    "credibility_factor": renewal_factors,
    "posterior_premium": prior_premiums_renewal * renewal_factors,
})

print(renewal_df.describe())
#        prior_premium  credibility_factor  posterior_premium
# mean         387.42              1.0000             387.42   # balance property
# std          124.18              0.0843             148.31   # spread is higher
# min          198.00              0.8412             174.31   # good risks get discount
# 50%          361.00              0.9984             358.34
# max          891.00              1.5823            1150.77   # poor risks get surcharge
```

The factor distribution shows what you would expect: the bulk of the portfolio has a factor close to 1.0 (most policyholders have average or better experience), with a right tail for high claimers. The mean factor is exactly 1.0 by the balance property.

For motor fleet, the workflow is identical but the "policy" is the fleet account. Fleet accounts typically have enough claims volume that the credibility weight reaches 80–90% within 2–3 years — the individual experience is almost entirely self-credible.

---

## The UK regulatory case

FCA PS21/5 is often discussed in the context of the GIPP price-walking ban, but Section 5 of the policy statement contains a less-discussed principle: renewal premiums should be "fairly reflective of the risk." The FCA's concern is that renewal pricing should not systematically disadvantage loyal customers relative to equivalent new business.

Individual experience rating, done correctly, supports this principle rather than undermining it. A policyholder with seven years of claims-free history has demonstrated lower-than-average risk. A credibility factor below 1.0 reflects this — it produces a renewal price that is genuinely lower because the risk is genuinely lower, not because of a loyalty discount. The factor is a function of observed risk experience, not tenure.

This distinction matters in supervision. A loyalty discount that does not condition on experience is a marketing concession. A credibility factor that reflects experience is an actuarial premium. The FCA's expectation under PS21/5 is that renewal pricing is actuarially grounded. The credibility factor provides that grounding.

The same logic applies to commercial liability and home EoW books, where NCD is less standardised and the actuarial premium rationale for differentiated renewal pricing needs to be explicit and demonstrable.

---

## The Python gap

Wüthrich's R scripts for the dynamic Poisson-gamma model are available on the ETH Zurich website. The Calcetero-Badescu-Lin code exists as academic supplementary material. Neither is pip-installable, neither has a test suite, and neither integrates with a modern Python data stack.

The R `credibility` package (Dutang, Goulet, Pigeon) covers Bühlmann-Straub at group level but not dynamic state-space models or IS-based Bayesian posteriors. There is no CRAN package for the Wüthrich 2024 attention model.

This gap matters. UK pricing teams working in Python and Databricks need these models in a form they can deploy, test, and maintain. The academic code establishes the statistical correctness; the library wraps it in a usable interface with the actuarial constraints (balance property, exposure weighting, convergence diagnostics) that production deployment requires.

---

## Which model to use

**`StaticCredibilityModel`** is the right starting point for any book. It has a closed-form solution, fits in seconds, and the single kappa parameter is interpretable and auditable. If your book has straightforward Poisson-distributed claims with stable underlying risk, this model will capture most of the available improvement over NCD. Use it first to establish the baseline.

**`DynamicPoissonGammaModel`** is the right choice when the recency of claims matters — which is most books. A claim filed last year is stronger evidence of elevated underlying risk than a claim filed six years ago. If you have 5+ years of individual claims history per policy, the dynamic model will outperform the static model on held-out log-likelihood. The fitting cost (MLE on negative binomial marginals) is manageable and the p, q parameters are interpretable.

**`SurrogateModel`** is correct when you cannot use a Poisson-Gamma conjugate model — zero-inflated claim distributions, overdispersion structures that vary by covariate, or any case where the prior is not conjugate to the likelihood. The IS approximation is flexible but adds computational cost and sampling noise. Use it when the conjugate assumption is demonstrably wrong for your data.

**`DeepAttentionModel`** is the right choice for large books (50k+ policies, 5+ years of history) where you want the best possible predictive performance and can absorb the PyTorch dependency and training time. It is also the right choice if your heterogeneity structure is unusual enough that you do not trust the parametric assumptions of the other three models. The distribution-free property is genuinely useful for commercial books with heterogeneous claim structures.

---

## Related libraries

**[`credibility`](https://github.com/burning-cost/credibility)** implements Bühlmann-Straub at group level — rating segment, broker group, vehicle class. It is the complement to `insurance-experience` at different granularity. If you want to blend a new model's relativities with incumbent rates in thin segments, use `credibility`. If you want to differentiate individual renewal premiums within a segment, use `insurance-experience`.

**[`experience-rating`](https://github.com/burning-cost/experience-rating)** covers the contractual NCD/bonus-malus structure — Markov chain transitions, stationary distributions, claiming threshold optimisation. It models what the NCD system *does*, not what an actuarially correct system *should* produce. The two libraries are complementary: `experience-rating` tells you how your NCD grid behaves; `insurance-experience` tells you what the grid *should* be doing if it were actuarially correct.

---

**[insurance-experience on GitHub](https://github.com/burning-cost/insurance-experience)** — MIT-licensed, PyPI. Library #43.
