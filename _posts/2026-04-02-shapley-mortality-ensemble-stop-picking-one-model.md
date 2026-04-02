---
layout: post
title: "Stop Picking One Mortality Model: Shapley Values Tell You How Much Each Actually Contributes"
date: 2026-04-02
categories: [techniques, mortality, ensemble-methods]
tags: [mortality, shapley-values, ensemble, lee-carter, cbd, renshaw-haberman, longevity, annuity, ltc, critical-illness, cmi, hmd, oecd, insurance-survival, arXiv-2603-03789, Bimonte]
description: "Bimonte et al. show that age-specific Shapley weights across 15 mortality models outperform any single model at 10-20 year horizons — exactly the range that matters for annuity and LTC pricing. Here is what it means for UK protection and longevity teams."
math: true
author: burning-cost
---

The question mortality modellers ask most often is: Lee-Carter or CBD? The answer is: neither, exclusively. The right question is how much weight to give each model at each age, and the right tool for computing that weight is Shapley values.

Bimonte, Haberman, Russolillo, and Zanini published exactly this argument in March 2026 (arXiv:2603.03789). They ran 15 mortality models — Lee-Carter variants, Cairns-Blake-Dowd variants, Renshaw-Haberman, and others — across 24 OECD countries using Human Mortality Database data, then combined model forecasts using Shapley-value-derived weights computed separately per age group. The headline result: the ensemble outperforms any individual model at 10-20 year horizons, and the performance gap is largest at those horizons. That is the range that drives annuity pricing, long-term care product design, and reserve adequacy for protection books with significant lifetime or whole-of-life exposure.

---

## What Shapley values are doing here

In cooperative game theory, the Shapley value answers: given a coalition of players producing a collective outcome, how much did each player individually contribute? Bimonte et al. map "players" to mortality models and "outcome" to forecast accuracy. Each model's Shapley value is its average marginal contribution to forecast accuracy across all possible subsets of the 15-model ensemble.

The insight that separates this from naïve model averaging is the per-age computation. Shapley weights are not computed once and applied uniformly. A model that performs well at ages 65-75 gets high weight in that band; its weight at ages 25-35 is computed independently and may be low. The ensemble self-organises to match structural reality: Lee-Carter is strong at working ages, where mortality is low and trend-driven; CBD was designed specifically for 50+ mortality, where age-period-cohort interactions are more complex and the force of selection from survivors matters more.

The practical output is a weight vector $w_a^m$ for each model $m$ at each age $a$. The ensemble forecast is then:

$$\hat{\mu}_{a,t} = \sum_m w_a^m \cdot \hat{\mu}_{a,t}^{(m)}$$

where $\hat{\mu}_{a,t}^{(m)}$ is model $m$'s central mortality rate forecast for age $a$ at time $t$.

---

## Why CMI scenarios are not the same thing

The CMI Mortality Projections Model, which most UK life insurers use as their starting point for longevity assumptions, is scenario-based. The Core projection gives one central path; High/Low variants give plausible range. You pick one — typically Core — and run your pricing or reserving off it.

This is model selection dressed up as sensitivity testing. When you price an annuity book off the CMI Core projection, you are implicitly assigning weight 1.0 to that scenario and 0.0 to everything else. The High and Low scenarios do not contribute to your central estimate; they appear in a stress test table and are then set aside.

The Shapley ensemble approach is different in kind, not just degree. It does not ask "which scenario?" It asks "what is each model's actual contribution to predictive accuracy?" and folds that directly into the central estimate. The result is a forecast that is explicitly a weighted combination of competing model structures, where the weights are grounded in historical performance rather than actuarial judgement calls about plausibility.

This matters most when models disagree — and models disagree most at long horizons. CMI's post-COVID recalibration (the 2024 and 2025 parameter updates) highlighted exactly this: the Core projection path shifted materially across consecutive model years, not because mortality fundamentally changed but because the dampening parameter calibration was sensitive to the COVID spike and subsequent bounce-back. An ensemble approach would have partially insulated the central estimate from that sensitivity by distributing weight across models with different structural assumptions about mean-reversion.

Model selection uncertainty is real and it is quantifiable. Picking one model and calling it "Core" does not make that uncertainty go away. It just stops you measuring it.

---

## UK CI and LTC: the age-band alignment

For critical illness and long-term care pricing, the age-specific weighting is not just a statistical nicety — it maps directly onto how you structure your product.

A standard UK CI product has meaningful exposure across ages 25-65. The risk composition shifts substantially across that range:

- **Ages 35-44:** trauma, multiple sclerosis, and early-onset cancer drive the majority of claims. Mortality improvement in this band is trend-driven, and Lee-Carter-type models — which model the age-period interaction through a bilinear term — tend to fit it well.
- **Ages 55-64:** cancer and cardiovascular disease dominate. Mortality in this band is more sensitive to cohort effects and to lifestyle/treatment changes that operate with a lag. CBD-class models, which model the age-period-cohort structure more explicitly, have an edge here.
- **Ages 65+:** relevant for whole-of-life, LTC, and lifetime protection products. CBD was designed for this age range and tends to outperform in it.

When you price a CI product as a single age band with a single mortality improvement assumption, you are implicitly using a uniform model weight across an age range where the optimal weight vector is not uniform. Age-specific Shapley weights let you be explicit about which model structure you are trusting for which part of the age distribution, with the weights empirically grounded rather than asserted.

For LTC specifically, where the projection horizon can be 25-30 years and the benefit trigger is correlated with the mortality improvement path (people who survive longer into old age are also those accumulating care needs), getting the long-horizon mortality trajectory right matters in both directions. The Shapley ensemble's largest performance gain is precisely at 10-20 year horizons.

---

## Connection to cause-specific decomposition

Bimonte et al.'s ensemble operates on all-cause mortality. This is one layer of a fuller picture.

Our `insurance-survival` library's `CauseSpecificMortality` class decomposes all-cause mortality into competing causes — cancer, cardiovascular, respiratory, and so on — and models each cause separately before re-aggregating. These are complementary approaches: Shapley ensemble weighting addresses model selection uncertainty at the all-cause level; cause-specific decomposition addresses compositional uncertainty about which causes drive the aggregate trend.

In principle you could apply Shapley ensemble weighting within each cause-specific model separately, then aggregate — a two-level ensemble. That is not what Bimonte et al. do, and the computational cost would be substantial, but the logic holds.

For CI pricing in particular, the cause-specific layer is more directly actionable because CI products pay on diagnosis of specific conditions. Mortality improvement in cancer survival directly affects CI incidence rates (better survival means more people live long enough to be diagnosed, and also more people survive a CI event to claim again on a different condition). The all-cause ensemble provides the outer constraint; the cause-specific decomposition provides the product-level structure.

---

## Limitations we would not skip over

**No UK-specific validation.** The 24 OECD countries in Bimonte et al. include some CMI-adjacent populations (Germany, France, the Netherlands) but the UK is not separately reported. UK mortality has idiosyncrasies — the CMI English and Welsh dataset, the NHS treatment pathway effects, the post-2012 austerity mortality trend — that may affect which models perform best and at what ages. The Shapley weights derived on US or Japanese data are not directly portable to a UK annuity book.

**Computational cost is not trivial.** Shapley values for 15 models require evaluating all $2^{15} = 32,768$ subsets, or a sampling approximation such as SHAP's Kernel SHAP. For a bootstrap-resampled ensemble run at each age group over multiple projection horizons, this is a material compute requirement. Bimonte et al. do not report runtime; in practice this is likely feasible for model development cycles but not real-time pricing engines.

**The CMI infrastructure question.** UK life insurers do not typically run their own Lee-Carter or CBD implementations. They consume CMI model outputs and may have in-house adjustments on top. Implementing a 15-model Shapley ensemble for a UK book would require building — or acquiring — the underlying model infrastructure, which is a non-trivial project even before the Shapley weighting layer.

**Historical window dependence.** Shapley weights are computed on historical forecast accuracy. The weights are therefore sensitive to which historical window you use and to structural breaks in that window. COVID is a structural break. A weight derived on 1970-2019 data may not represent the post-COVID mortality regime. This is not a reason to reject the approach, but it is a calibration question that the paper does not fully resolve.

---

## What to do with this

If you are on a longevity or protection pricing team and your mortality improvement assumption is "CMI Core, maybe with a small upward adjustment for uncertainty," the Bimonte et al. paper is worth reading as a prompt to ask harder questions about what that assumption actually encodes.

You do not need to implement a 15-model Shapley ensemble to get value from the paper. The first step is simpler: take three or four models — Lee-Carter, CBD, and Renshaw-Haberman are the obvious starting point — implement them on HMD or CMI data for England and Wales, and compare their 15-year projections across age bands. The disagreement between models at ages 65-75 will tell you more about the uncertainty in your longevity assumptions than any CMI scenario table.

The Shapley weighting layer is then a principled way to combine those projections rather than picking the one whose output you find most comfortable.

The paper is at arXiv:2603.03789.
